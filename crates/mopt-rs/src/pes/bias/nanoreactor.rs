//! Ab initio nanoreactor boundary potential (Wang, Titov, McGibbon, Liu, Pande, Martinez,
//! Nat. Chem. 2014, 6, 1044; MultiOptPy `-nrp`).
//!
//! `V(r, t) = f(t) U(r, r1, k1) + (1 - f(t)) U(r, r2, k2)` per atom with
//! `U(r, r0, k) = m k / 2 (r - r0)^2 theta(r - r0)` and the rectangular wave
//! `f(t) = theta(floor(t/T) - t/T + tau/T)`: the wide sphere (`r1`, `k1`) holds for `tau`, the
//! narrow one (`r2`, `k2`) for `T - tau`, pushing the molecules together (paper: `r1` = 14 A,
//! `k1` = 1.0, `r2` = 8 A, `k2` = 0.5 kcal/mol/A^2, `tau` = 1.5 ps, `T` = 2.0 ps).
//!
//! Intentional differences from MultiOptPy: the walls confine from outside, as in the paper
//! (MultiOptPy's inner wall pushes atoms out of the inner sphere, and its outer wall uses the
//! contraction force constant in both phases); its `[inner] [outer] [contraction time]
//! [expansion time] [contraction k] [expansion k]` arguments are read so that the recommended
//! values `8 14 1.5 0.5 1.0 0.5` give the paper's settings (the first time and force constant
//! belong to the outer sphere). Outside molecular dynamics the time is the iteration in atomic
//! units, as in MultiOptPy.
//!
//! The centre of the spheres is a fixed point in space, [`NanoReactor::center`]. MultiOptPy
//! measures `r` as the norm of each position (`torch.linalg.norm(geom_num_list, dim=1)` in
//! `NanoReactorPotential.calc_energy`), so `-nrp` uses the origin in every command; the
//! nanoreactor runs of `mapper` place it at the centroid of their starting structure, once.
//! Either way the term depends on where the molecules are ([`Frame::Absolute`]), so
//! optimizers do not recentre or rotate a structure held by it.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::engine::{norm, Clock, Geom, GeometricBias};
use crate::domain::AtomId;
use crate::math::ad::Scalar;
use crate::pes::term::{Frame, History};

/// The centre of the spheres of `-nrp` (Bohr): the origin, as in MultiOptPy.
pub const MULTIOPTPY_CENTER: [f64; 3] = [0.0; 3];

/// Nanoreactor boundary potential.
#[derive(Debug, Clone)]
pub struct NanoReactor {
    /// `(atom, mass in amu)`.
    pub atoms: Vec<(AtomId, f64)>,
    /// Center of the spheres (Bohr).
    pub center: [f64; 3],
    /// Wide-sphere radius `r1` (Bohr).
    pub r1: f64,
    /// Wide-sphere force constant `k1` (Hartree / Bohr^2 / amu).
    pub k1: f64,
    /// Narrow-sphere radius `r2` (Bohr).
    pub r2: f64,
    /// Narrow-sphere force constant `k2` (Hartree / Bohr^2 / amu).
    pub k2: f64,
    /// Duration of the wide phase `tau` (atomic time units).
    pub tau: f64,
    /// Period `T` (atomic time units).
    pub period: f64,
}

impl NanoReactor {
    /// `f(t)`: 1 in the wide phase, 0 in the narrow phase.
    pub fn phase(&self, time_au: f64) -> f64 {
        let x = time_au / self.period;
        if x.floor() - x + self.tau / self.period > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

fn wall<T: Scalar>(r: T, r0: f64, k: f64, m: f64) -> T {
    if r.val() > r0 {
        let d = r - T::from_f64(r0);
        T::from_f64(0.5 * m * k) * d * d
    } else {
        T::zero()
    }
}

impl GeometricBias for NanoReactor {
    fn name(&self) -> &str {
        "nano_reactor_potential"
    }
    fn atoms(&self) -> Vec<AtomId> {
        self.atoms.iter().map(|a| a.0).collect()
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, clock: Clock) -> T {
        let t = clock.time_au.unwrap_or(clock.iteration as f64);
        let f = self.phase(t);
        self.atoms.iter().fold(T::zero(), |e, &(a, m)| {
            let r = norm(g.d_point(a, self.center));
            e + T::from_f64(f) * wall(r, self.r1, self.k1, m)
                + T::from_f64(1.0 - f) * wall(r, self.r2, self.k2, m)
        })
    }
    /// The spheres sit at a fixed point.
    fn frame(&self) -> Frame {
        Frame::Absolute
    }
    /// The walls switch with the simulated time (the iteration outside dynamics).
    fn history(&self) -> History {
        History::TimeDependent
    }
}
