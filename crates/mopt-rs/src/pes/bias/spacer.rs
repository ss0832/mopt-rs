//! Spacer model (MultiOptPy `-smp`): solvent particles around a solute.
//!
//! `n` particles (auxiliary variables, relaxed at every evaluation by [`super::MicroBias`] with
//! the ONIOM microiteration machinery) interact with every atom through
//! `eps_ap [(s_ap/r)^12 - 2 (s_ap/r)^6]` (`s_ap = s_p + s_a`, `eps_ap = sqrt(eps_p eps_a)`, UFF atom
//! parameters) and with each other through the same form with `2 s_p` and `eps_p`. A cavity
//! term keeps each particle near the solute: with `q = d / (c s_t)`, `d` the distance to the
//! nearest target atom `t` and `c` the cavity scaling, it is zero for `q < 0.9`, rises smoothly
//! (`0.5 (10 u^3 - 15 u^4 + 6 u^5)`, `u = (q - 0.9) / 0.1`) to 0.5 at `q = 1`, and is `0.5 q`
//! beyond.
//!
//! Intentional difference from MultiOptPy: the nearest target and its distance are taken
//! between the particle and the atoms (MultiOptPy compares the distances of the particle and
//! of the atoms from the coordinate origin, so the cavity depended on where the molecule
//! sits). Starting particles are spread evenly over a sphere around the targets instead of
//! MultiOptPy's random normal guesses.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::auxiliary::AuxiliaryBias;
use super::engine::{norm, Clock, Geom};
use crate::domain::{AtomId, Element};
use crate::math::ad::Scalar;
use crate::parameters::forcefield::{uff_vdw_distance_bohr, uff_vdw_well_depth_hartree};
use crate::pes::term::Frame;

/// Spacer-model solvent particles.
#[derive(Debug, Clone)]
pub struct SpacerModel {
    /// Particle well depth (Hartree).
    pub particle_eps: f64,
    /// Particle distance parameter `s_p` (Bohr).
    pub particle_sigma: f64,
    /// Cavity scaling `c`.
    pub cavity_scaling: f64,
    /// Number of particles.
    pub n_particles: usize,
    /// `(atom, cavity radius c s_t)` per target.
    targets: Vec<(AtomId, f64)>,
    /// `(atom, eps_a, s_a)` for every atom.
    atoms: Vec<(AtomId, f64, f64)>,
}

/// Height of the cavity wall (MultiOptPy `const`).
const CAVITY_WALL: f64 = 0.5;

impl SpacerModel {
    /// Model for a system with `elements`.
    pub fn new(
        elements: &[Element],
        targets: &[AtomId],
        particle_eps: f64,
        particle_sigma: f64,
        cavity_scaling: f64,
        n_particles: usize,
    ) -> Self {
        let atoms = elements
            .iter()
            .enumerate()
            .map(|(i, &el)| {
                (
                    AtomId::new(i),
                    uff_vdw_well_depth_hartree(el),
                    uff_vdw_distance_bohr(el),
                )
            })
            .collect();
        let targets = targets
            .iter()
            .map(|&t| {
                let el = elements.get(t.index()).copied().unwrap_or(Element::H);
                (t, cavity_scaling * uff_vdw_distance_bohr(el))
            })
            .collect();
        Self {
            particle_eps,
            particle_sigma,
            cavity_scaling,
            n_particles,
            targets,
            atoms,
        }
    }

    fn particle<T: Scalar>(y: &[T], k: usize) -> [T; 3] {
        [y[3 * k], y[3 * k + 1], y[3 * k + 2]]
    }
}

fn lj<T: Scalar>(r: T, eps: f64, sigma: f64) -> T {
    let s6 = (T::from_f64(sigma) / r).powi(6);
    T::from_f64(eps) * (s6 * s6 - T::from_f64(2.0) * s6)
}

fn cavity<T: Scalar>(q: T) -> T {
    let qv = q.val();
    if qv < 0.9 {
        T::zero()
    } else if qv < 1.0 {
        let u = (q - T::from_f64(0.9)) / T::from_f64(0.1);
        let u3 = u * u * u;
        T::from_f64(CAVITY_WALL)
            * (T::from_f64(10.0) * u3 - T::from_f64(15.0) * u3 * u + T::from_f64(6.0) * u3 * u * u)
    } else {
        T::from_f64(CAVITY_WALL) * q
    }
}

impl AuxiliaryBias for SpacerModel {
    fn name(&self) -> &str {
        "spacer_model_potential"
    }

    fn atoms(&self) -> Vec<AtomId> {
        self.atoms.iter().map(|a| a.0).collect()
    }

    fn n_aux(&self) -> usize {
        3 * self.n_particles
    }

    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, y: &[T], _clock: Clock) -> T {
        let mut e = T::zero();
        for k in 0..self.n_particles {
            let p = Self::particle(y, k);
            for &(a, eps_a, s_a) in &self.atoms {
                let r = norm(g.d_to(a, p));
                e += lj(
                    r,
                    (self.particle_eps * eps_a).sqrt(),
                    self.particle_sigma + s_a,
                );
            }
            for l in k + 1..self.n_particles {
                let q = Self::particle(y, l);
                let r = norm(g.mic(std::array::from_fn(|c| p[c] - q[c])));
                e += lj(r, self.particle_eps, 2.0 * self.particle_sigma);
            }
            let nearest = self
                .targets
                .iter()
                .map(|&(t, radius)| norm(g.d_to(t, p)) / T::from_f64(radius))
                .min_by(|a, b| a.val().total_cmp(&b.val()));
            if let Some(q) = nearest {
                e += cavity(q);
            }
        }
        e
    }

    /// Invariant: the particles are minimized together with their positions, so a rigid
    /// motion of the solute carries the relaxed particles with it (the cavity is measured
    /// from the atoms, not from the origin).
    fn frame(&self) -> Frame {
        Frame::Invariant
    }

    fn duplicate(&self) -> Option<Self> {
        Some(self.clone())
    }

    fn initial_aux(&self, g: &Geom<'_, f64>, _clock: Clock) -> Vec<f64> {
        let ids: Vec<AtomId> = self.targets.iter().map(|t| t.0).collect();
        if ids.is_empty() {
            return vec![0.0; self.n_aux()];
        }
        let c = g.centroid(&ids);
        let extent = ids
            .iter()
            .map(|&t| norm(g.d_point(t, c)))
            .fold(0.0_f64, f64::max);
        let s_max = self.atoms.iter().map(|a| a.2).fold(0.0_f64, f64::max);
        let radius = extent + 0.5 * (self.particle_sigma + s_max);
        // Fibonacci sphere.
        let n = self.n_particles.max(1) as f64;
        let golden = PI * (3.0 - 5f64.sqrt());
        (0..self.n_particles)
            .flat_map(|k| {
                let z = 1.0 - 2.0 * (k as f64 + 0.5) / n;
                let rho = (1.0 - z * z).sqrt();
                let a = golden * k as f64;
                [
                    c[0] + radius * rho * a.cos(),
                    c[1] + radius * rho * a.sin(),
                    c[2] + radius * z,
                ]
            })
            .collect()
    }
}
