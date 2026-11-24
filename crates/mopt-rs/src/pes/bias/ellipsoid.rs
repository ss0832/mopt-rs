//! Asymmetric ellipsoidal repulsive potentials (MultiOptPy `-aerp` with the GNB parameters of
//! J. Chem. Theory Comput., DOI 10.1021/acs.jctc.4c01435; `-aerpv2` with UFF parameters).
//!
//! Each ellipsoid sits at distance `L` from a root atom along the bond to a second atom, with
//! semi-axes `(sigma_x+, sigma_x-, sigma_y+, sigma_y-, sigma_z+, sigma_z-)` in a frame whose `z`
//! axis is that bond. Its rotation `phi` about the bond is free and minimized at every
//! evaluation (one auxiliary variable per ellipsoid, relaxed by [`super::MicroBias`] with the
//! ONIOM microiteration machinery). Every atom that is not one of the ellipsoid's own atoms
//! or off-targets feels `eps [r_ell^-12 - 2 r_ell^-6]` with the ellipsoidal distance
//! `r_ell = sqrt((x/s_x)^2 + (y/s_y)^2 + (z/s_z)^2)`; the semi-axis on each side depends on the
//! sign of the coordinate. Pairs of ellipsoids repel each other in the same way.
//!
//! Intentional differences from MultiOptPy: the direction-dependent well depth is evaluated
//! per atom along the unit vector of the atom (MultiOptPy normalizes by the norm of `r_ell` over
//! all atoms, so one atom's well depth depended on every other atom); the ellipsoid-ellipsoid
//! term rotates about the bond (`z`) axis like the atom term (MultiOptPy used the default axis
//! of its rotation helper there). As all three well depths are equal, the first change leaves
//! `eps_dir = sqrt(eps eps_t)`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::auxiliary::AuxiliaryBias;
use super::engine::{norm, Clock, Geom};
use crate::domain::{AtomId, Element};
use crate::math::ad::Scalar;
use crate::parameters::forcefield::{
    gnb_vdw_distance_bohr, gnb_vdw_well_depth_hartree, uff_vdw_distance_bohr,
    uff_vdw_well_depth_hartree,
};
use crate::pes::term::Frame;

/// Parameter set and combination rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EllipsoidVariant {
    /// GNB parameters, `s = sqrt(2^(14/6) sigma_e sigma_t)` (MultiOptPy `-aerp`).
    Gnb,
    /// UFF parameters, `s = (sigma_e + sigma_t)^(7/6)` (MultiOptPy `-aerpv2`).
    Uff,
}

/// One ellipsoid.
#[derive(Debug, Clone, PartialEq)]
pub struct Ellipsoid {
    /// Root atom.
    pub root: AtomId,
    /// Atom defining the axis with the root.
    pub axis_atom: AtomId,
    /// Well depth (Hartree).
    pub eps: f64,
    /// Semi-axes `x+, x-, y+, y-, z+, z-` (Bohr).
    pub sigma: [f64; 6],
    /// Distance of the center from the root (Bohr).
    pub distance: f64,
    /// Atoms that do not feel this ellipsoid (besides its own two).
    pub off_targets: Vec<AtomId>,
}

/// Target atom of an ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Target {
    atom: AtomId,
    eps: f64,
    sigma: f64,
}

/// Asymmetric ellipsoidal repulsive potential with one or more ellipsoids.
#[derive(Debug, Clone)]
pub struct AsymmetricEllipsoids {
    /// Ellipsoids.
    pub ellipsoids: Vec<Ellipsoid>,
    /// Parameter set.
    pub variant: EllipsoidVariant,
    targets: Vec<Vec<Target>>,
    all: Vec<AtomId>,
}

impl AsymmetricEllipsoids {
    /// Potential for a system with `elements`.
    pub fn new(
        elements: &[Element],
        ellipsoids: Vec<Ellipsoid>,
        variant: EllipsoidVariant,
    ) -> Self {
        let n = elements.len();
        let targets = ellipsoids
            .iter()
            .map(|e| {
                (0..n)
                    .map(AtomId::new)
                    .filter(|a| *a != e.root && *a != e.axis_atom && !e.off_targets.contains(a))
                    .map(|a| {
                        let el = elements[a.index()];
                        let (eps, sigma) = match variant {
                            EllipsoidVariant::Gnb => (
                                gnb_vdw_well_depth_hartree(el),
                                gnb_vdw_distance_bohr(el) / 2.0,
                            ),
                            EllipsoidVariant::Uff => (
                                uff_vdw_well_depth_hartree(el),
                                uff_vdw_distance_bohr(el) / 2.0,
                            ),
                        };
                        Target {
                            atom: a,
                            eps,
                            sigma,
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            ellipsoids,
            variant,
            targets,
            all: (0..n).map(AtomId::new).collect(),
        }
    }

    fn combined_sigma(&self, axis: f64, target: f64) -> f64 {
        match self.variant {
            EllipsoidVariant::Gnb => (2f64.powf(14.0 / 6.0) * axis * target).sqrt(),
            EllipsoidVariant::Uff => (axis + target).powf(7.0 / 6.0),
        }
    }
}

/// Orthonormal frame `(e1, e2, e3)` with `e3` along `u`; `e1` comes from the Cartesian axis
/// least aligned with `u` (a piecewise-constant choice; the energy after minimizing over the
/// rotation angle does not depend on it).
fn frame<T: Scalar>(u: [T; 3]) -> [[T; 3]; 3] {
    let k = (0..3)
        .min_by(|&i, &j| u[i].val().abs().total_cmp(&u[j].val().abs()))
        .unwrap_or(0);
    let r: [T; 3] = std::array::from_fn(|i| if i == k { T::one() } else { T::zero() });
    let p = T::dot3(r, u);
    let e1: [T; 3] = std::array::from_fn(|i| r[i] - p * u[i]);
    let n1 = norm(e1);
    let e1 = e1.map(|x| x / n1);
    let e2 = T::cross3(u, e1);
    [e1, e2, u]
}

/// Coordinates of `p` in `frame`, rotated by `phi` about `e3`.
fn local<T: Scalar>(f: &[[T; 3]; 3], phi: T, p: [T; 3]) -> [T; 3] {
    let (x, y, z) = (T::dot3(p, f[0]), T::dot3(p, f[1]), T::dot3(p, f[2]));
    let (s, c) = (phi.sin(), phi.cos());
    [c * x - s * y, s * x + c * y, z]
}

/// `sqrt(sum (x_k / s_k)^2)` with the semi-axis chosen by the sign of each coordinate.
fn ellipsoidal_distance<T: Scalar>(x: [T; 3], s: impl Fn(usize, bool) -> f64) -> T {
    let mut acc = T::zero();
    for (k, xk) in x.iter().enumerate() {
        let q = *xk / T::from_f64(s(k, xk.val() > 0.0));
        acc += q * q;
    }
    acc.sqrt()
}

fn lj_reduced<T: Scalar>(r: T, eps: T) -> T {
    let i6 = (T::one() / r).powi(6);
    eps * (i6 * i6 - T::from_f64(2.0) * i6)
}

impl AuxiliaryBias for AsymmetricEllipsoids {
    fn name(&self) -> &str {
        match self.variant {
            EllipsoidVariant::Gnb => "asymmetric_ellipsoidal_repulsive_potential",
            EllipsoidVariant::Uff => "asymmetric_ellipsoidal_repulsive_potential_v2",
        }
    }

    fn atoms(&self) -> Vec<AtomId> {
        self.all.clone()
    }

    fn n_aux(&self) -> usize {
        self.ellipsoids.len()
    }

    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, phi: &[T], _clock: Clock) -> T {
        // Frames and centers (relative to the roots).
        let geo: Vec<([[T; 3]; 3], [T; 3])> = self
            .ellipsoids
            .iter()
            .map(|e| {
                let b = g.d(e.axis_atom, e.root);
                let nb = norm(b);
                let u = b.map(|x| x / nb);
                (frame(u), u.map(|x| x * T::from_f64(e.distance)))
            })
            .collect();
        let mut energy = T::zero();
        for (i, e) in self.ellipsoids.iter().enumerate() {
            let (f, c) = &geo[i];
            for t in &self.targets[i] {
                let d = g.d(t.atom, e.root);
                let p = local(f, phi[i], std::array::from_fn(|k| d[k] - c[k]));
                let r = ellipsoidal_distance(p, |k, pos| {
                    self.combined_sigma(e.sigma[2 * k + usize::from(!pos)], t.sigma)
                });
                let eps = ((e.eps * t.eps).sqrt() * t.eps).sqrt();
                energy += lj_reduced(r, T::from_f64(eps));
            }
        }
        // Ellipsoid pairs: each center seen from the other ellipsoid.
        let s76 = 2f64.powf(7.0 / 6.0);
        for i in 0..self.ellipsoids.len() {
            for j in i + 1..self.ellipsoids.len() {
                let (ei, ej) = (&self.ellipsoids[i], &self.ellipsoids[j]);
                // Center j relative to center i, and the reverse.
                let rij = g.d(ej.root, ei.root);
                let (ci, cj) = (geo[i].1, geo[j].1);
                let dji: [T; 3] = std::array::from_fn(|k| rij[k] + cj[k] - ci[k]);
                let dij = dji.map(|x| T::zero() - x);
                let pj = local(&geo[i].0, phi[i], dji);
                let pi = local(&geo[j].0, phi[j], dij);
                let r_i =
                    ellipsoidal_distance(pj, |k, pos| s76 * ei.sigma[2 * k + usize::from(!pos)]);
                let r_j =
                    ellipsoidal_distance(pi, |k, pos| s76 * ej.sigma[2 * k + usize::from(!pos)]);
                let eps = (ei.eps * ej.eps).sqrt();
                energy += lj_reduced((r_i * r_j).sqrt(), T::from_f64(eps));
            }
        }
        energy
    }

    /// Invariant: the ellipsoids sit on bonds, and the rotation of each about its bond is
    /// minimized, so the energy does not depend on the Cartesian axis that starts its frame.
    fn frame(&self) -> Frame {
        Frame::Invariant
    }

    fn duplicate(&self) -> Option<Self> {
        Some(self.clone())
    }

    fn initial_aux(&self, g: &Geom<'_, f64>, clock: Clock) -> Vec<f64> {
        // Coordinate-wise grid search (MultiOptPy starts from the best of random angles).
        let n = self.ellipsoids.len();
        let mut phi = vec![0.0; n];
        for _ in 0..2 {
            for i in 0..n {
                let mut best = (f64::INFINITY, phi[i]);
                for k in 0..72 {
                    let mut trial = phi.clone();
                    trial[i] = 2.0 * PI * k as f64 / 72.0;
                    let e = self.energy(g, &trial, clock);
                    if e < best.0 {
                        best = (e, trial[i]);
                    }
                }
                phi[i] = best.1;
            }
        }
        phi
    }
}
