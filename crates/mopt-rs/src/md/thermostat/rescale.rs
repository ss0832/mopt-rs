//! Thermostats that rescale all velocities by one factor: Berendsen's weak coupling and the
//! canonical sampling through velocity rescaling of Bussi, Donadio, and Parrinello.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{positive, Thermostat, ThermostatContext};
use crate::core_infra::rng::Rng;
use crate::error::MoptResult;
use crate::md::space::temperature_of;

/// Berendsen's weak coupling: `lambda^2 = 1 + (dt / tau) (T0 / T - 1)`, limited to
/// `0.8 <= lambda <= 1.25` as in GROMACS. It does not sample the canonical ensemble (the
/// fluctuations of the kinetic energy are too small).
#[derive(Debug, Clone)]
pub struct Berendsen {
    tau: f64,
}

impl Berendsen {
    /// Coupling time `tau` (atomic units).
    pub fn new(tau: f64) -> MoptResult<Self> {
        Ok(Self {
            tau: positive(tau, "the Berendsen coupling time")?,
        })
    }
}

impl Thermostat for Berendsen {
    fn name(&self) -> &str {
        "berendsen"
    }

    fn apply(
        &mut self,
        dt: f64,
        v: &mut Col<f64>,
        ctx: &mut ThermostatContext<'_>,
    ) -> MoptResult<()> {
        let kinetic = ctx.space.kinetic_energy(v);
        let t = temperature_of(kinetic, ctx.dof);
        if t <= 0.0 {
            return Ok(());
        }
        let square = 1.0 + dt / self.tau * (ctx.temperature / t - 1.0);
        let lambda = libm::sqrt(square.max(0.0)).clamp(0.8, 1.25);
        for x in v.iter_mut() {
            *x *= lambda;
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

/// Canonical sampling through velocity rescaling: the kinetic energy follows a stochastic
/// relaxation with time `tau` towards the canonical distribution.
#[derive(Debug, Clone)]
pub struct Bussi {
    tau: f64,
}

impl Bussi {
    /// Relaxation time `tau` (atomic units).
    pub fn new(tau: f64) -> MoptResult<Self> {
        Ok(Self {
            tau: positive(tau, "the relaxation time of the velocity rescaling")?,
        })
    }
}

/// A gamma deviate of shape `a >= 1` and scale 1 (Marsaglia and Tsang, ACM Trans. Math.
/// Softw. 2000, 26, 363).
pub fn gamma_deviate(a: f64, rng: &mut Rng) -> f64 {
    let d = a - 1.0 / 3.0;
    let c = 1.0 / libm::sqrt(9.0 * d);
    loop {
        let (x, v) = loop {
            let x = rng.standard_normal();
            let v = 1.0 + c * x;
            if v > 0.0 {
                break (x, v * v * v);
            }
        };
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x * x * x * x
            || (u > 0.0 && libm::log(u) < 0.5 * x * x + d * (1.0 - v + libm::log(v)))
        {
            return d * v;
        }
    }
}

/// The sum of the squares of `n` standard normal deviates.
pub fn sum_of_squared_normals(n: usize, rng: &mut Rng) -> f64 {
    match n {
        0 => 0.0,
        1 => {
            let g = rng.standard_normal();
            g * g
        }
        _ if n.is_multiple_of(2) => 2.0 * gamma_deviate(n as f64 / 2.0, rng),
        _ => {
            let g = rng.standard_normal();
            2.0 * gamma_deviate((n - 1) as f64 / 2.0, rng) + g * g
        }
    }
}

impl Thermostat for Bussi {
    fn name(&self) -> &str {
        "bussi"
    }

    fn apply(
        &mut self,
        dt: f64,
        v: &mut Col<f64>,
        ctx: &mut ThermostatContext<'_>,
    ) -> MoptResult<()> {
        let kinetic = ctx.space.kinetic_energy(v);
        let dof = ctx.dof.round().max(0.0);
        if kinetic <= 0.0 || dof < 1.0 {
            return Ok(());
        }
        let target = 0.5 * dof * ctx.kt();
        // c = exp(-dt / tau), 1 - c without cancellation.
        let one_minus_c = -libm::expm1(-dt / self.tau);
        let c = 1.0 - one_minus_c;
        let r1 = ctx.rng.standard_normal();
        let rest = sum_of_squared_normals(dof as usize - 1, ctx.rng);
        let new = kinetic
            + one_minus_c * (target * (rest + r1 * r1) / dof - kinetic)
            + 2.0 * r1 * libm::sqrt(kinetic * target / dof * one_minus_c * c);
        let alpha = libm::sqrt((new / kinetic).max(0.0));
        for x in v.iter_mut() {
            *x *= alpha;
        }
        Ok(())
    }

    fn is_stochastic(&self) -> bool {
        true
    }

    fn reference(&self) -> Option<&'static str> {
        Some("G. Bussi, D. Donadio, M. Parrinello, J. Chem. Phys. 2007, 126, 014101")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_sums_of_squares_have_the_mean_and_variance_of_chi_squared() {
        let mut rng = Rng::from_seed(11);
        for n in [1_usize, 2, 7, 40] {
            let samples: Vec<f64> = (0..40_000)
                .map(|_| sum_of_squared_normals(n, &mut rng))
                .collect();
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let var =
                samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / samples.len() as f64;
            let n = n as f64;
            assert!((mean - n).abs() < 0.05 * n.max(1.0), "{n}: {mean}");
            assert!((var - 2.0 * n).abs() < 0.1 * 2.0 * n, "{n}: {var}");
        }
    }
}
