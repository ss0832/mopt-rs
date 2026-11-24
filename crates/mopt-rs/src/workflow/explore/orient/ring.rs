//! The `ring` placement: every part turned uniformly at random, the centres on a regular polygon
//! in a random plane (the parts in a random order), the polygon shrunk until the closest pair
//! of atoms of two parts is at `s (R_i + R_j)`, and then every part that touches no other one
//! brought towards the centroid of the others until it does.
//!
//! For two parts this is the `contact` placement. For more, every part starts at the same
//! distance from its neighbours, as in MultiOptPy's polygon, but at van der Waals contact
//! instead of a fixed distance between centres; the measurements in running.md
//! (`orientsearch`) show the cyclic water trimer reached from it as often as from MultiOptPy's
//! polygon and more often than from parts added one by one.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py:make_random_orientation_xyz_file` (the
//!  polygon); the shrinking to contact is new.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::{PI, TAU};

use super::contact::{add, centroid, first_contact, group_reach, moved, norm, scale, Placed};
use super::placement::{apply, Sampler};
use crate::core_infra::rng::Rng;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;

/// Step of the scale of the polygon, as a share of the scan step along a line (Bohr).
const SCAN_STEP_BOHR: f64 = 0.25;
/// Bisections of the scale at contact.
const BISECTIONS: usize = 40;

impl Sampler {
    /// Smallest contact ratio between the parts `groups` (about their centres) put at
    /// `lambda` times `corners`.
    fn ring_contact(&self, groups: &[Vec<[f64; 3]>], corners: &[[f64; 3]], lambda: f64) -> f64 {
        let at: Vec<Vec<[f64; 3]>> = groups
            .iter()
            .zip(corners)
            .map(|(g, c)| moved(g, scale(*c, lambda)))
            .collect();
        let mut best = f64::INFINITY;
        for (a, group) in at.iter().enumerate() {
            let mut placed = Placed::new(None);
            placed.extend(group, &self.radii[a]);
            for (b, other) in at.iter().enumerate().skip(a + 1) {
                best = best.min(placed.contact_ratio(other, &self.radii[b]));
            }
        }
        best
    }

    /// One random complex of the `ring` placement (isolated, every part moving).
    pub(super) fn ring(&self, rng: &mut Rng) -> MoptResult<MolecularSystem> {
        let n = self.parts.len();
        let s = rng.uniform(self.settings.contact[0], self.settings.contact[1]);
        let locals: Vec<Vec<[f64; 3]>> = (0..n)
            .map(|k| {
                let (local, _) = self.centred(&self.template, k);
                let r = rng.rotation_matrix_3d();
                local.iter().map(|p| apply(&r, *p)).collect()
            })
            .collect();
        let plane = rng.rotation_matrix_3d();
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);
        let mut corners = vec![[0.0; 3]; n];
        for (slot, &k) in order.iter().enumerate() {
            let t = TAU * slot as f64 / n as f64;
            corners[k] = apply(&plane, [t.cos(), t.sin(), 0.0]);
        }
        // The side of the polygon of circumradius 1, and a scale at which nothing touches.
        let side = 2.0 * (PI / n as f64).sin();
        let reach = (0..n)
            .map(|k| group_reach(&locals[k], &self.radii[k]))
            .fold(0.0, f64::max);
        let gap = |lambda: f64| self.ring_contact(&locals, &corners, lambda) - s;
        let (mut outer, mut inner) = ((2.0 * reach * (1.0 + s) + 2.0) / side, 0.0);
        let step = SCAN_STEP_BOHR / side;
        let mut lambda = outer;
        while lambda > 0.0 {
            let next = (lambda - step).max(0.0);
            if gap(next) <= 0.0 {
                (outer, inner) = (lambda, next);
                break;
            }
            lambda = next;
            outer = next;
        }
        for _ in 0..BISECTIONS {
            let mid = 0.5 * (inner + outer);
            if gap(mid) <= 0.0 {
                inner = mid;
            } else {
                outer = mid;
            }
        }
        let mut groups: Vec<Vec<[f64; 3]>> = locals
            .iter()
            .zip(&corners)
            .map(|(g, c)| moved(g, scale(*c, outer)))
            .collect();
        // Every part that touches no other one comes in towards the centroid of the others.
        for k in 0..n {
            let mut others = Placed::new(None);
            for (m, g) in groups.iter().enumerate() {
                if m != k {
                    others.extend(g, &self.radii[m]);
                }
            }
            if others.contact_ratio(&groups[k], &self.radii[k]) <= self.settings.contact[1] {
                continue;
            }
            let start = others.centroid();
            let here = centroid(&groups[k]);
            let d = add(here, scale(start, -1.0));
            let length = norm(d);
            if length <= 1e-9 {
                continue;
            }
            let direction = scale(d, 1.0 / length);
            if let Some(t) = first_contact(
                &others,
                &locals[k],
                &self.radii[k],
                start,
                direction,
                (0.0, length),
                s,
            ) {
                groups[k] = moved(&locals[k], add(start, scale(direction, t)));
            }
        }
        self.assemble(&groups.into_iter().map(Some).collect::<Vec<_>>())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::parts::ComplexParts;
    use super::super::placement::{PlacementSettings, Scheme};
    use super::*;
    use crate::domain::{positions, Angstrom, AtomId, Element, SpinCharge};
    use crate::workflow::explore::identity::DEFAULT_BOND_SCALE;

    #[test]
    fn a_ring_of_waters_touches_at_the_contact_ratio() {
        let w = [
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ];
        for n in [2, 3, 4] {
            let mut pos = Vec::new();
            for k in 0..n {
                pos.extend(w.iter().map(|p| [p[0] + 4.0 * k as f64, p[1], p[2]]));
            }
            let sys = MolecularSystem::new(
                [Element::O, Element::H, Element::H].repeat(n),
                &positions::<Angstrom>(&pos),
                SpinCharge::default(),
                None,
            )
            .unwrap();
            let parts = ComplexParts::molecules(&sys, &[], DEFAULT_BOND_SCALE).unwrap();
            let settings = PlacementSettings {
                scheme: Scheme::Ring,
                ..PlacementSettings::default()
            };
            let sampler = Sampler::new(&sys, &parts, settings, DEFAULT_BOND_SCALE).unwrap();
            let [lo, hi] = settings.contact;
            let mut rng = Rng::from_seed(n as u64);
            for _ in 0..10 {
                let (x, _) = sampler.sample(&mut rng).unwrap();
                let closest = sampler.closest_contact(&x);
                assert!(
                    closest >= lo - 1e-6 && closest <= hi + 1e-6,
                    "{n}: {closest}"
                );
                // Every part touches another one, and the molecules keep their shape.
                for k in 0..n {
                    let mut others = Placed::new(None);
                    let group = |m: usize| -> Vec<[f64; 3]> {
                        (3 * m..3 * m + 3)
                            .map(|i| x.position(AtomId::new(i)).value())
                            .collect()
                    };
                    for m in (0..n).filter(|&m| m != k) {
                        others.extend(&group(m), &sampler.radii[m]);
                    }
                    let r = others.contact_ratio(&group(k), &sampler.radii[k]);
                    assert!(r <= hi + 1e-6, "{n}: part {k} at {r}");
                    let d = x
                        .distance(AtomId::new(3 * k), AtomId::new(3 * k + 1))
                        .value();
                    assert!(
                        (d - sys.distance(AtomId::new(0), AtomId::new(1)).value()).abs() < 1e-9
                    );
                }
            }
        }
    }
}
