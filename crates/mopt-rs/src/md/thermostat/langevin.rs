//! Thermostats that act on every particle with random numbers: Langevin dynamics (the
//! Ornstein-Uhlenbeck part of a BAOAB or OBABO step) and Andersen's collisions.
//!
//! MultiOptPy: `MD/thermostat.py:Thermostat.Langevin_thermostat` (the O part).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{positive, Thermostat, ThermostatContext};
use crate::error::MoptResult;

/// Langevin dynamics with friction `gamma`: over a time `dt` the velocity of every variable
/// becomes `c v + sqrt((1 - c^2) k_B T / m) xi` with `c = exp(-gamma dt)`.
#[derive(Debug, Clone)]
pub struct Langevin {
    gamma: f64,
}

impl Langevin {
    /// Friction `gamma` (inverse atomic time).
    pub fn new(gamma: f64) -> MoptResult<Self> {
        Ok(Self {
            gamma: positive(gamma, "the Langevin friction")?,
        })
    }

    /// The friction (inverse atomic time).
    pub fn friction(&self) -> f64 {
        self.gamma
    }
}

impl Thermostat for Langevin {
    fn name(&self) -> &str {
        "langevin"
    }

    fn apply(
        &mut self,
        dt: f64,
        v: &mut Col<f64>,
        ctx: &mut ThermostatContext<'_>,
    ) -> MoptResult<()> {
        let c = libm::exp(-self.gamma * dt);
        // 1 - c^2 without cancellation for small gamma dt.
        let spread = -libm::expm1(-2.0 * self.gamma * dt);
        let kt = ctx.kt();
        let inverse = ctx.space.inverse_masses();
        for (k, x) in v.iter_mut().enumerate() {
            let w = inverse[k];
            if w > 0.0 {
                *x = c * *x + libm::sqrt(spread * kt * w) * ctx.rng.standard_normal();
            }
        }
        Ok(())
    }

    fn conserves_momentum(&self) -> bool {
        false
    }

    fn is_stochastic(&self) -> bool {
        true
    }

    fn reference(&self) -> Option<&'static str> {
        Some("B. Leimkuhler and C. Matthews, J. Chem. Phys. 2013, 138, 174102")
    }
}

/// Andersen's thermostat: in a time `dt` each particle collides with probability
/// `1 - exp(-nu dt)` and takes a velocity from the Maxwell-Boltzmann distribution.
#[derive(Debug, Clone)]
pub struct Andersen {
    nu: f64,
}

impl Andersen {
    /// Collision frequency `nu` per particle (inverse atomic time).
    pub fn new(nu: f64) -> MoptResult<Self> {
        Ok(Self {
            nu: positive(nu, "the Andersen collision frequency")?,
        })
    }
}

impl Thermostat for Andersen {
    fn name(&self) -> &str {
        "andersen"
    }

    fn apply(
        &mut self,
        dt: f64,
        v: &mut Col<f64>,
        ctx: &mut ThermostatContext<'_>,
    ) -> MoptResult<()> {
        let chance = -libm::expm1(-self.nu * dt);
        let group = ctx.space.group();
        let kt = ctx.kt();
        let inverse = ctx.space.inverse_masses();
        for particle in 0..ctx.space.particles() {
            if ctx.rng.next_f64() >= chance {
                continue;
            }
            for k in group * particle..group * (particle + 1) {
                let w = inverse[k];
                if w > 0.0 {
                    v[k] = libm::sqrt(kt * w) * ctx.rng.standard_normal();
                }
            }
        }
        Ok(())
    }

    fn conserves_momentum(&self) -> bool {
        false
    }

    fn is_stochastic(&self) -> bool {
        true
    }

    fn reference(&self) -> Option<&'static str> {
        Some("H. C. Andersen, J. Chem. Phys. 1980, 72, 2384")
    }
}
