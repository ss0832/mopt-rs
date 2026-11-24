//! Random relative orientations and placements of the parts of a complex.
//!
//! Schemes ([`Scheme`]):
//!
//! - `multioptpy`: MultiOptPy's sampling. Every part is rotated about its geometric centre by
//!   three angles drawn uniformly from `[0, 2 pi)` about x, y, and z in turn (which is not a
//!   uniform distribution of rotations), and the centres are put on a regular polygon in the
//!   xy plane in a random order, neighbours a distance drawn from `[d, 1.1 d]` apart (`d` is
//!   `-dist`). MultiOptPy's errors are not repeated: its polygon radius
//!   `0.5 d tan((n - 2) pi / n)` is replaced by the circumradius `d / (2 sin(pi / n))` (its
//!   radius puts four parts at about 1e16 Angstrom and five or more at wrong distances), and
//!   the distance it draws for two parts is used (it draws it and keeps `d`).
//! - `uniform`: rotations uniform over SO(3) (Shoemake's quaternions), every part's centre at
//!   the drawn distance from the centre of a part placed before, in a direction uniform on the
//!   sphere, no nearer than that to any other centre.
//! - `ring` ([`super::ring`], the default): uniform rotations; the centres on a regular polygon
//!   in a random plane (in a random order), the polygon shrunk until the closest pair of atoms of
//!   two parts is at `s (R_i + R_j)`, then every part that touches no other brought towards the
//!   centroid until it does. In a cell or with parts that stay it places as `contact`.
//! - `contact`: uniform rotations; the parts are added one by one in a random order, each
//!   coming from a random direction towards the centroid of those placed before until its
//!   closest pair of atoms reaches `s (R_i + R_j)` with `s` drawn from the contact range and the
//!   van der Waals radii `R` ([`super::contact`]). Parts that stay (fixed, or bonded to their
//!   periodic images) are placed first, as they are. In a cell with a free side (a slab) the
//!   others come down onto its surface at a random point; in a cell without one (a box) they
//!   are put at random points where they clash with nothing.
//!
//! A sample of `multioptpy` or `uniform` whose closest pair of atoms of two parts is below the
//! `clash` ratio is drawn again, up to `attempts` times.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py:make_random_orientation_xyz_file` and
//!  `Utils/calc_tools.py:rotate_molecule`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::contact::{
    add, centroid, first_contact, group_reach, moved, norm, scale, slab_of, vdw_radius_bohr,
    Placed, Slab,
};
use super::parts::{whole_positions, ComplexParts};
use crate::core_infra::rng::Rng;
use crate::domain::{AtomId, Bohr, MolecularSystem, Position, Quantity, Units};
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::identity::BondGraph;

/// How the parts are placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// MultiOptPy's Euler angles and polygon (with its errors corrected).
    MultiOptPy,
    /// Uniform rotations, centres at the drawn distance.
    Uniform,
    /// Uniform rotations, parts at van der Waals contact.
    Contact,
    /// Uniform rotations, centres on a regular polygon in a random plane, shrunk to contact.
    Ring,
}

impl Scheme {
    /// The scheme of `name` (`multioptpy`, `uniform`, `contact`).
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "multioptpy" => Ok(Self::MultiOptPy),
            "uniform" => Ok(Self::Uniform),
            "contact" => Ok(Self::Contact),
            "ring" => Ok(Self::Ring),
            other => Err(MoptError::parse(format!(
                "unknown sampling scheme '{other}' (multioptpy, uniform, contact, ring)"
            ))),
        }
    }

    /// Its name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::MultiOptPy => "multioptpy",
            Self::Uniform => "uniform",
            Self::Contact => "contact",
            Self::Ring => "ring",
        }
    }
}

/// Settings of the placements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacementSettings {
    /// The scheme.
    pub scheme: Scheme,
    /// `-dist`: the distance between the centres of neighbouring parts (`multioptpy`,
    /// `uniform`); drawn from `[d, 1.1 d]` for every sample.
    pub distance: Quantity<Bohr>,
    /// Range of the contact ratio `s` of the `contact` scheme.
    pub contact: [f64; 2],
    /// Samples of `multioptpy` and `uniform` closer than this contact ratio are drawn again
    /// (0: never).
    pub clash: f64,
    /// Draws of one sample (or of one position) before giving up.
    pub attempts: usize,
}

impl Default for PlacementSettings {
    /// The settings measured in running.md (`orientsearch`).
    fn default() -> Self {
        Self {
            scheme: Scheme::Ring,
            distance: Quantity::new(5.0 * Units::ANGSTROM_TO_BOHR),
            contact: [1.0, 1.2],
            clash: 0.5,
            attempts: 100,
        }
    }
}

/// A random rotation of a part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Perturbation {
    /// Largest angle (radians) of the turn about the part's centre.
    pub max_angle: f64,
    /// Largest shift (Bohr) of the part.
    pub max_shift: f64,
}

impl Default for Perturbation {
    fn default() -> Self {
        Self {
            max_angle: std::f64::consts::FRAC_PI_2,
            max_shift: 1.0 * Units::ANGSTROM_TO_BOHR,
        }
    }
}

type Rot = [[f64; 3]; 3];

pub(super) fn apply(r: &Rot, p: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2])
}

/// MultiOptPy's rotation `geom @ Rx(a) @ Ry(b) @ Rz(c)` of row vectors as a matrix acting on
/// column vectors: `Rz(-c) Ry(-b) Rx(-a)`.
pub fn multioptpy_rotation(a: f64, b: f64, c: f64) -> Rot {
    let rx = |t: f64| {
        [
            [1.0, 0.0, 0.0],
            [0.0, t.cos(), -t.sin()],
            [0.0, t.sin(), t.cos()],
        ]
    };
    let ry = |t: f64| {
        [
            [t.cos(), 0.0, t.sin()],
            [0.0, 1.0, 0.0],
            [-t.sin(), 0.0, t.cos()],
        ]
    };
    let rz = |t: f64| {
        [
            [t.cos(), -t.sin(), 0.0],
            [t.sin(), t.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ]
    };
    let mul = |x: Rot, y: Rot| -> Rot {
        std::array::from_fn(|r| std::array::from_fn(|k| (0..3).map(|m| x[r][m] * y[m][k]).sum()))
    };
    mul(mul(rz(-c), ry(-b)), rx(-a))
}

/// Rotation by `angle` about the unit `axis` (Rodrigues).
fn axis_angle(axis: [f64; 3], angle: f64) -> Rot {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    let [x, y, z] = axis;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// Draws positions of the parts of one complex.
#[derive(Debug, Clone)]
pub struct Sampler {
    pub(super) template: MolecularSystem,
    pub(super) parts: ComplexParts,
    pub(super) graph: BondGraph,
    pub(super) settings: PlacementSettings,
    pub(super) radii: Vec<Vec<f64>>,
    slab: Option<Slab>,
}

impl Sampler {
    /// A sampler for `parts` of `template`; refuses schemes that cannot place them.
    pub fn new(
        template: &MolecularSystem,
        parts: &ComplexParts,
        settings: PlacementSettings,
        bond_scale: f64,
    ) -> MoptResult<Self> {
        if matches!(settings.scheme, Scheme::MultiOptPy | Scheme::Uniform) {
            if template.has_pbc() {
                return Err(MoptError::unsupported(format!(
                    "the {} scheme puts the centres of the parts at a distance along directions \
                     of its own, which in a periodic cell depends on the image; the contact \
                     scheme places parts in a cell (-orient_opt scheme=contact)",
                    settings.scheme.name()
                )));
            }
            if parts.fixed.iter().any(|f| *f) {
                return Err(MoptError::unsupported(format!(
                    "the {} scheme moves every part; parts that stay need the contact scheme \
                     (-orient_opt scheme=contact)",
                    settings.scheme.name()
                )));
            }
        }
        if !(settings.contact[0] > 0.0 && settings.contact[0] <= settings.contact[1]) {
            return Err(MoptError::config(format!(
                "the contact range must be 0 < low <= high, got {:?}",
                settings.contact
            )));
        }
        if !(settings.distance.value().is_finite() && settings.distance.value() > 0.0) {
            return Err(MoptError::config("-dist must be positive"));
        }
        let graph = BondGraph::of(template, bond_scale);
        let radii = parts
            .atoms
            .iter()
            .map(|a| {
                a.iter()
                    .map(|&i| vdw_radius_bohr(template.elements[i]))
                    .collect()
            })
            .collect();
        let slab = match template.active_cell() {
            Some(cell) => {
                let staying: Vec<[f64; 3]> = parts
                    .staying()
                    .iter()
                    .flat_map(|&k| whole_positions(template, &graph, &parts.atoms[k]))
                    .collect();
                slab_of(cell, &staying)
            }
            None => None,
        };
        Ok(Self {
            template: template.clone(),
            parts: parts.clone(),
            graph,
            settings,
            radii,
            slab,
        })
    }

    /// The free side of the parts that stay in a cell, when they form a slab.
    pub fn slab(&self) -> Option<Slab> {
        self.slab
    }

    /// The settings.
    pub fn settings(&self) -> &PlacementSettings {
        &self.settings
    }

    /// Positions of part `k` of `sys` made whole, about their geometric centre, and the centre.
    pub(super) fn centred(&self, sys: &MolecularSystem, k: usize) -> (Vec<[f64; 3]>, [f64; 3]) {
        let whole = whole_positions(sys, &self.graph, &self.parts.atoms[k]);
        let c = centroid(&whole);
        (whole.iter().map(|p| add(*p, scale(c, -1.0))).collect(), c)
    }

    pub(super) fn assemble(&self, groups: &[Option<Vec<[f64; 3]>>]) -> MoptResult<MolecularSystem> {
        let mut out = self.template.clone();
        for (k, group) in groups.iter().enumerate() {
            let Some(group) = group else { continue };
            for (&i, p) in self.parts.atoms[k].iter().zip(group) {
                out.set_position(AtomId::new(i), Position::new(*p));
            }
        }
        out.comment = String::new();
        Ok(out)
    }

    /// Smallest contact ratio between atoms of different parts of `groups`.
    fn closest(&self, groups: &[Option<Vec<[f64; 3]>>]) -> f64 {
        let mut best = f64::INFINITY;
        for a in 0..groups.len() {
            for b in (a + 1)..groups.len() {
                let (Some(ga), Some(gb)) = (&groups[a], &groups[b]) else {
                    continue;
                };
                let mut placed = Placed::new(self.template.active_cell());
                placed.extend(ga, &self.radii[a]);
                best = best.min(placed.contact_ratio(gb, &self.radii[b]));
            }
        }
        best
    }

    /// The parts.
    pub fn parts(&self) -> &ComplexParts {
        &self.parts
    }

    /// The closest contact between atoms of different parts of `sys` (the smallest ratio of a
    /// distance to the sum of the van der Waals radii; minimum-image distances in a cell).
    pub fn closest_contact(&self, sys: &MolecularSystem) -> f64 {
        let groups: Vec<Option<Vec<[f64; 3]>>> = (0..self.parts.len())
            .map(|k| Some(whole_positions(sys, &self.graph, &self.parts.atoms[k])))
            .collect();
        self.closest(&groups)
    }

    /// One random complex and how many draws were rejected for clashes.
    pub fn sample(&self, rng: &mut Rng) -> MoptResult<(MolecularSystem, usize)> {
        let mut redraws = 0;
        loop {
            let groups = match self.settings.scheme {
                Scheme::MultiOptPy => self.multioptpy(rng),
                Scheme::Uniform => self.uniform(rng),
                Scheme::Contact => return Ok((self.contact(rng)?, 0)),
                // In a cell or with parts that stay the ring has no centre: contact.
                Scheme::Ring if self.template.has_pbc() || self.parts.fixed.iter().any(|f| *f) => {
                    return Ok((self.contact(rng)?, 0))
                }
                Scheme::Ring => return Ok((self.ring(rng)?, 0)),
            };
            let clashing = self.settings.clash > 0.0 && self.closest(&groups) < self.settings.clash;
            if !clashing || redraws + 1 >= self.settings.attempts.max(1) {
                return Ok((self.assemble(&groups)?, redraws));
            }
            redraws += 1;
        }
    }

    fn multioptpy(&self, rng: &mut Rng) -> Vec<Option<Vec<[f64; 3]>>> {
        let n = self.parts.len();
        let mut groups: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n);
        for k in 0..n {
            let (local, _) = self.centred(&self.template, k);
            let (a, b, c) = (
                rng.uniform(0.0, std::f64::consts::TAU),
                rng.uniform(0.0, std::f64::consts::TAU),
                rng.uniform(0.0, std::f64::consts::TAU),
            );
            let r = multioptpy_rotation(a, b, c);
            groups.push(local.iter().map(|p| apply(&r, *p)).collect());
        }
        let d = self.settings.distance.value();
        let drawn = rng.uniform(d, 1.1 * d);
        let mut order: Vec<usize> = (0..n).collect();
        if n > 2 {
            rng.shuffle(&mut order);
        }
        let radius = drawn / (2.0 * (std::f64::consts::PI / n as f64).sin());
        groups
            .iter()
            .enumerate()
            .map(|(k, g)| {
                let slot = order.iter().position(|&o| o == k).unwrap_or(k);
                let angle = std::f64::consts::TAU * slot as f64 / n as f64;
                // Two parts lie on the x axis at -d/2 and +d/2, as in MultiOptPy.
                let angle = if n == 2 {
                    std::f64::consts::PI * (1.0 - slot as f64)
                } else {
                    angle
                };
                Some(moved(g, [radius * angle.cos(), radius * angle.sin(), 0.0]))
            })
            .collect()
    }

    fn uniform(&self, rng: &mut Rng) -> Vec<Option<Vec<[f64; 3]>>> {
        let n = self.parts.len();
        let locals: Vec<Vec<[f64; 3]>> = (0..n)
            .map(|k| {
                let (local, _) = self.centred(&self.template, k);
                let r = rng.rotation_matrix_3d();
                local.iter().map(|p| apply(&r, *p)).collect()
            })
            .collect();
        let d = self.settings.distance.value();
        let drawn = rng.uniform(d, 1.1 * d);
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);
        let mut centres: Vec<[f64; 3]> = vec![[0.0; 3]];
        for _ in 1..n {
            let mut candidate = [0.0; 3];
            for _ in 0..self.settings.attempts.max(1) {
                let from = centres[rng.gen_range_usize(centres.len())];
                candidate = add(from, scale(rng.unit_vector_3d(), drawn));
                if centres.iter().all(|c| {
                    norm([
                        c[0] - candidate[0],
                        c[1] - candidate[1],
                        c[2] - candidate[2],
                    ]) >= drawn * (1.0 - 1e-9)
                }) {
                    break;
                }
            }
            centres.push(candidate);
        }
        let mut groups = vec![None; n];
        for (slot, &k) in order.iter().enumerate() {
            groups[k] = Some(moved(&locals[k], centres[slot]));
        }
        groups
    }

    fn contact(&self, rng: &mut Rng) -> MoptResult<MolecularSystem> {
        let n = self.parts.len();
        let cell = self.template.active_cell();
        let mut groups: Vec<Option<Vec<[f64; 3]>>> = vec![None; n];
        let mut placed = Placed::new(cell);
        // The parts that stay keep their positions as they are (`None`: not moved).
        for k in self.parts.staying() {
            let whole = whole_positions(&self.template, &self.graph, &self.parts.atoms[k]);
            placed.extend(&whole, &self.radii[k]);
        }
        let mut order = self.parts.movable();
        rng.shuffle(&mut order);
        for k in order {
            let (local, _) = self.centred(&self.template, k);
            let r = rng.rotation_matrix_3d();
            let local: Vec<[f64; 3]> = local.iter().map(|p| apply(&r, *p)).collect();
            let s = rng.uniform(self.settings.contact[0], self.settings.contact[1]);
            let at = self.contact_point(&placed, &local, &self.radii[k], s, rng)?;
            let group = moved(&local, at);
            placed.extend(&group, &self.radii[k]);
            groups[k] = Some(group);
        }
        self.assemble(&groups)
    }

    /// Where `local` (about its centre) goes to touch the placed atoms at the ratio `s`.
    fn contact_point(
        &self,
        placed: &Placed<'_>,
        local: &[[f64; 3]],
        radii: &[f64],
        s: f64,
        rng: &mut Rng,
    ) -> MoptResult<[f64; 3]> {
        if placed.is_empty() {
            return Ok(match self.template.active_cell() {
                Some(cell) => cell.frac_to_cart([rng.next_f64(), rng.next_f64(), rng.next_f64()]),
                None => [0.0; 3],
            });
        }
        let reach = group_reach(local, radii);
        let Some(cell) = self.template.active_cell() else {
            let start = placed.centroid();
            let t_max = placed.reach(start) + reach * (1.0 + s) + 1.0;
            for _ in 0..self.settings.attempts.max(1) {
                let u = rng.unit_vector_3d();
                if let Some(t) = first_contact(placed, local, radii, start, u, (0.0, t_max), s) {
                    return Ok(add(start, scale(u, t)));
                }
            }
            return Err(MoptError::calc("no direction leads a part to a contact"));
        };
        if let Some(slab) = self.slab {
            // From the middle of the empty layer (or high enough above a surface without a
            // periodic image) down towards the surface below it.
            let t_max = if slab.gap.is_finite() {
                0.5 * slab.gap
            } else {
                reach * (1.0 + s) + 10.0
            };
            for _ in 0..self.settings.attempts.max(1) {
                let mut f = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
                f[slab.axis] = 0.0;
                let lateral = cell.frac_to_cart(f);
                let lift = slab.top - super::contact::dot(lateral, slab.normal);
                let surface = add(lateral, scale(slab.normal, lift));
                let range = (-t_max, t_max);
                if let Some(t) = first_contact(placed, local, radii, surface, slab.normal, range, s)
                {
                    return Ok(add(surface, scale(slab.normal, t)));
                }
            }
            return Err(MoptError::calc(format!(
                "no point above the surface of the slab is free for a part (the empty layer of \
                 {:.1} Angstrom may be too thin for it)",
                slab.gap * Units::BOHR_TO_ANGSTROM
            )));
        }
        for _ in 0..self.settings.attempts.max(1) {
            let at = cell.frac_to_cart([rng.next_f64(), rng.next_f64(), rng.next_f64()]);
            if placed.contact_ratio(&moved(local, at), radii) >= s {
                return Ok(at);
            }
        }
        Err(MoptError::calc(format!(
            "no free place in the cell for a part after {} draws (the cell is too full for \
             the contact ratio {s:.2})",
            self.settings.attempts
        )))
    }

    /// A known minimum `minimum` with one movable part turned and shifted at random, moved
    /// out of contact closer than the lower end of the contact range.
    pub fn perturb(
        &self,
        minimum: &MolecularSystem,
        perturbation: &Perturbation,
        rng: &mut Rng,
    ) -> MoptResult<MolecularSystem> {
        let movable = self.parts.movable();
        if movable.is_empty() {
            return Err(MoptError::config("no part can move"));
        }
        let k = movable[rng.gen_range_usize(movable.len())];
        let (local, centre) = self.centred(minimum, k);
        let turn = axis_angle(
            rng.unit_vector_3d(),
            rng.uniform(0.0, perturbation.max_angle),
        );
        let local: Vec<[f64; 3]> = local.iter().map(|p| apply(&turn, *p)).collect();
        let shift = scale(
            rng.unit_vector_3d(),
            rng.uniform(0.0, perturbation.max_shift),
        );
        let mut at = add(centre, shift);
        let mut placed = Placed::new(minimum.active_cell());
        for other in 0..self.parts.len() {
            if other != k {
                let whole = whole_positions(minimum, &self.graph, &self.parts.atoms[other]);
                placed.extend(&whole, &self.radii[other]);
            }
        }
        let away = match (self.slab, minimum.active_cell()) {
            (Some(slab), _) => slab.normal,
            (None, Some(_)) => rng.unit_vector_3d(),
            (None, None) => {
                let d = add(at, scale(placed.centroid(), -1.0));
                let length = norm(d);
                if length > 1e-6 {
                    scale(d, 1.0 / length)
                } else {
                    rng.unit_vector_3d()
                }
            }
        };
        let low = self.settings.contact[0];
        for _ in 0..400 {
            if placed.contact_ratio(&moved(&local, at), &self.radii[k]) >= low {
                break;
            }
            at = add(at, scale(away, 0.25));
        }
        let mut groups: Vec<Option<Vec<[f64; 3]>>> = vec![None; self.parts.len()];
        groups[k] = Some(moved(&local, at));
        let mut out = minimum.clone();
        for (&i, p) in self.parts.atoms[k].iter().zip(groups[k].iter().flatten()) {
            out.set_position(AtomId::new(i), Position::new(*p));
        }
        out.comment = String::new();
        Ok(out)
    }
}

#[cfg(test)]
#[path = "placement_tests.rs"]
mod tests;
