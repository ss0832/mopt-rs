//! First-order barostats: Berendsen's weak coupling and the stochastic cell rescaling of
//! Bernetti and Bussi. Both compute a scaling of the cell from the pressure at a barostat
//! stage and apply it with the next drift, before the forces at the new positions are
//! evaluated (as GROMACS scales after the update of the positions).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{scale_positions_and_cell, Barostat, BarostatContext};
use crate::domain::PeriodicCell;
use crate::error::MoptResult;
use crate::md::thermostat::positive;

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn product(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// Berendsen's barostat: `mu = 1 - beta dt / (3 tau) (P0 - P)` with the scalar pressure
/// (isotropic) or the symmetric pressure tensor (anisotropic: the cell deforms).
#[derive(Debug, Clone)]
pub struct BerendsenBarostat {
    tau: f64,
    compressibility: f64,
    anisotropic: bool,
    pending: [[f64; 3]; 3],
}

impl BerendsenBarostat {
    /// Coupling time `tau` (atomic units) and compressibility (Bohr^3 / Hartree).
    pub fn new(tau: f64, compressibility: f64, anisotropic: bool) -> MoptResult<Self> {
        Ok(Self {
            tau: positive(tau, "the Berendsen barostat coupling time")?,
            compressibility: positive(compressibility, "the compressibility")?,
            anisotropic,
            pending: IDENTITY,
        })
    }
}

impl Barostat for BerendsenBarostat {
    fn name(&self) -> &str {
        if self.anisotropic {
            "berendsen_aniso"
        } else {
            "berendsen"
        }
    }

    fn apply(
        &mut self,
        dt: f64,
        _v: &mut Col<f64>,
        ctx: &mut BarostatContext<'_>,
    ) -> MoptResult<()> {
        let factor = self.compressibility * dt / (3.0 * self.tau);
        let p = ctx.pressure_tensor();
        let scalar = (p[0][0] + p[1][1] + p[2][2]) / 3.0;
        let mu: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let delta = if a == b { 1.0 } else { 0.0 };
                let own = if self.anisotropic {
                    0.5 * (p[a][b] + p[b][a])
                } else {
                    delta * scalar
                };
                delta - factor * (delta * ctx.pressure - own)
            })
        });
        self.pending = product(&mu, &self.pending);
        Ok(())
    }

    fn drift(
        &mut self,
        h: f64,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        cell: &mut PeriodicCell,
    ) -> MoptResult<()> {
        for (p, w) in x.iter_mut().zip(v.iter()) {
            *p += h * w;
        }
        if self.pending != IDENTITY {
            scale_positions_and_cell(&self.pending, x, cell)?;
            self.pending = IDENTITY;
        }
        Ok(())
    }

    fn reference(&self) -> Option<&'static str> {
        Some(
            "H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola, J. R. Haak, \
             J. Chem. Phys. 1984, 81, 3684",
        )
    }
}

/// Stochastic cell rescaling (isotropic): `d eps = -(beta / tau) (P0 - P) dt +
/// sqrt(2 k_B T beta dt / (V tau)) dW` for `eps = ln V` (the Euler form of the paper; the
/// `k_B T / V` of the equation for `V` cancels in the Ito form for `ln V`); positions and
/// cell are scaled by `exp(d eps / 3)` and the velocities by its inverse (the scaled momenta
/// `p V^(1/3)` are kept), with the instantaneous pressure `2 K / (3 V) - tr(sigma) / 3`.
#[derive(Debug, Clone)]
pub struct CellRescale {
    tau: f64,
    compressibility: f64,
    pending: f64,
}

impl CellRescale {
    /// Coupling time `tau` (atomic units) and compressibility (Bohr^3 / Hartree).
    pub fn new(tau: f64, compressibility: f64) -> MoptResult<Self> {
        Ok(Self {
            tau: positive(tau, "the cell-rescaling coupling time")?,
            compressibility: positive(compressibility, "the compressibility")?,
            pending: 0.0,
        })
    }
}

impl Barostat for CellRescale {
    fn name(&self) -> &str {
        "crescale"
    }

    fn apply(
        &mut self,
        dt: f64,
        _v: &mut Col<f64>,
        ctx: &mut BarostatContext<'_>,
    ) -> MoptResult<()> {
        let kt = ctx.kt();
        let internal = ctx.instantaneous_pressure();
        let drift = -self.compressibility / self.tau * (ctx.pressure - internal) * dt;
        let noise = libm::sqrt(2.0 * kt * self.compressibility * dt / (ctx.volume * self.tau))
            * ctx.rng.standard_normal();
        self.pending += drift + noise;
        Ok(())
    }

    fn drift(
        &mut self,
        h: f64,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        cell: &mut PeriodicCell,
    ) -> MoptResult<()> {
        if self.pending != 0.0 {
            let mu = libm::exp(self.pending / 3.0);
            let matrix = [[mu, 0.0, 0.0], [0.0, mu, 0.0], [0.0, 0.0, mu]];
            scale_positions_and_cell(&matrix, x, cell)?;
            for w in v.iter_mut() {
                *w /= mu;
            }
            self.pending = 0.0;
        }
        for (p, w) in x.iter_mut().zip(v.iter()) {
            *p += h * w;
        }
        Ok(())
    }

    fn reference(&self) -> Option<&'static str> {
        Some("M. Bernetti and G. Bussi, J. Chem. Phys. 2020, 153, 114107")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core_infra::rng::Rng;
    use crate::domain::{Bohr, Quantity};
    use crate::md::space::MdSpace;

    /// An ideal gas of `dof` degrees of freedom at constant pressure: the volume samples
    /// `V^(dof/3) exp(-P0 V / kT)`, whose mean is `(dof/3 + 1) kT / P0`.
    #[test]
    fn the_cell_rescaling_of_an_ideal_gas_has_the_mean_volume_of_the_ensemble() {
        let space = MdSpace::variables(3).unwrap();
        let kt = 1.0e-3;
        let temperature = kt / crate::domain::Units::BOLTZMANN_HARTREE_PER_K;
        let (dof, p0) = (30.0, 1.0e-5);
        let expected = (dof / 3.0 + 1.0) * kt / p0;
        let mut barostat = CellRescale::new(50.0, 1.0 / (p0 * 10.0)).unwrap();
        let edge = libm::cbrt(expected);
        let mut cell = PeriodicCell::orthorhombic(
            Quantity::<Bohr>::new(edge),
            Quantity::new(edge),
            Quantity::new(edge),
        )
        .unwrap();
        let mut rng = Rng::from_seed(4);
        let (mut x, mut v) = (Col::zeros(3), Col::zeros(3));
        let mut sum = 0.0;
        let steps = 400_000;
        for step in 0..steps {
            let mut ctx = BarostatContext {
                space: &space,
                dof,
                temperature,
                pressure: p0,
                volume: cell.volume_bohr3(),
                // A thermostat that keeps the kinetic energy at its mean.
                kinetic: std::array::from_fn(|a| {
                    std::array::from_fn(|b| if a == b { dof * kt / 3.0 } else { 0.0 })
                }),
                stress: [[0.0; 3]; 3],
                rng: &mut rng,
            };
            barostat.apply(1.0, &mut v, &mut ctx).unwrap();
            barostat.drift(1.0, &mut x, &mut v, &mut cell).unwrap();
            if step >= steps / 10 {
                sum += cell.volume_bohr3();
            }
        }
        let mean = sum / (steps - steps / 10) as f64;
        assert!(
            (mean / expected - 1.0).abs() < 0.02,
            "{mean} against {expected}"
        );
    }
}
