//! The isotropic barostat of Martyna, Tobias, and Klein with the measure-preserving
//! integration of Tuckerman, Alejandre, Lopez-Rendon, Jochim, and Martyna.
//!
//! Equations of motion (`v_e = p_e / W`, `alpha = 1 + 3 / N_f`):
//! `dr/dt = v + v_e r`, `dv/dt = F / m - alpha v_e v`, `dV/dt = 3 V v_e`,
//! `dp_e/dt = 3 V (P - P0) + (3 / N_f) 2 K`, with a Nose-Hoover chain of its own on `p_e`.
//! The mass of the cell is `W = (N_f + 3) k_B T tau^2`. The conserved quantity is
//! `K + U + p_e^2 / (2 W) + P0 V` and the energies of the chains.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{scale_positions_and_cell, sinhc, Barostat, BarostatContext};
use crate::domain::{PeriodicCell, Units};
use crate::error::MoptResult;
use crate::md::thermostat::{positive, ThermostatChain};

/// Isotropic Martyna-Tobias-Klein barostat.
#[derive(Debug, Clone)]
pub struct Mtk {
    tau: f64,
    /// Velocity of the logarithm of the cell length, `p_e / W` (inverse atomic time).
    velocity: f64,
    /// `W` at the last action.
    mass: f64,
    /// `alpha = 1 + 3 / N_f` at the last action.
    alpha: f64,
    chain: ThermostatChain,
}

impl Mtk {
    /// Period `tau` (atomic units) of the motion of the cell; `links` of its chain.
    pub fn new(tau: f64, links: usize) -> MoptResult<Self> {
        let tau = positive(tau, "the Martyna-Tobias-Klein period")?;
        Ok(Self {
            tau,
            velocity: 0.0,
            mass: 0.0,
            alpha: 1.0,
            chain: ThermostatChain::new(links, tau, 3, 1)?,
        })
    }

    fn mass_of(&self, dof: f64, kt: f64) -> f64 {
        (dof + 3.0) * kt * self.tau * self.tau
    }
}

impl Barostat for Mtk {
    fn name(&self) -> &str {
        "mtk"
    }

    fn apply(
        &mut self,
        dt: f64,
        _v: &mut Col<f64>,
        ctx: &mut BarostatContext<'_>,
    ) -> MoptResult<()> {
        let kt = ctx.kt();
        self.mass = self.mass_of(ctx.dof, kt);
        self.alpha = 1.0 + 3.0 / ctx.dof.max(1.0);
        let k2 = ctx.kinetic[0][0] + ctx.kinetic[1][1] + ctx.kinetic[2][2];
        let trace_sigma = ctx.stress[0][0] + ctx.stress[1][1] + ctx.stress[2][2];
        let push = self.alpha * k2 - ctx.volume * trace_sigma - 3.0 * ctx.volume * ctx.pressure;
        // Chain, force of the pressure, chain: symmetric within the stage.
        let w = self.mass;
        let s = self
            .chain
            .propagate(0.5 * dt, w * self.velocity * self.velocity, 1.0, kt);
        self.velocity *= s;
        self.velocity += dt * push / w;
        let s = self
            .chain
            .propagate(0.5 * dt, w * self.velocity * self.velocity, 1.0, kt);
        self.velocity *= s;
        Ok(())
    }

    fn kick(&self, h: f64, v: &mut Col<f64>, a: &Col<f64>) {
        let z = self.alpha * self.velocity * h;
        let damp = libm::exp(-z);
        let along = h * libm::exp(-0.5 * z) * sinhc(0.5 * z);
        for (x, acc) in v.iter_mut().zip(a.iter()) {
            *x = *x * damp + along * acc;
        }
    }

    fn drift(
        &mut self,
        h: f64,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        cell: &mut PeriodicCell,
    ) -> MoptResult<()> {
        let z = self.velocity * h;
        let grow = libm::exp(z);
        let along = h * libm::exp(0.5 * z) * sinhc(0.5 * z);
        // Positions: x e^z + v h e^{z/2} sinhc(z/2); the cell scales by e^z.
        let mut moved = x.clone();
        for (p, w) in moved.iter_mut().zip(v.iter()) {
            *p *= grow;
            *p += along * w;
        }
        let mu = [[grow, 0.0, 0.0], [0.0, grow, 0.0], [0.0, 0.0, grow]];
        let mut scratch = Col::zeros(0);
        scale_positions_and_cell(&mu, &mut scratch, cell)?;
        *x = moved;
        Ok(())
    }

    fn energy(&self, volume: f64, pressure: f64, temperature: f64) -> f64 {
        let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature;
        0.5 * self.mass * self.velocity * self.velocity
            + pressure * volume
            + self.chain.energy(1.0, kt)
    }

    fn has_conserved_quantity(&self) -> bool {
        true
    }

    fn notes(&self) -> Vec<String> {
        vec![format!(
            "Martyna-Tobias-Klein: velocity of ln(cell length) {:.6e} 1/a.u.",
            self.velocity
        )]
    }

    fn reference(&self) -> Option<&'static str> {
        Some(
            "G. J. Martyna, D. J. Tobias, M. L. Klein, J. Chem. Phys. 1994, 101, 4177; M. E. \
             Tuckerman, J. Alejandre, R. Lopez-Rendon, A. L. Jochim, G. J. Martyna, J. Phys. A \
             2006, 39, 5629",
        )
    }
}
