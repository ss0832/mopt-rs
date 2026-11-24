//! Recrossing of the dividing surface at a saddle point: transmission coefficients `kappa <= 1`
//! that multiply the flux of TST (both directions alike: a property of the dividing surface,
//! so detailed balance is kept).
//!
//! - `kramers`: Kramers' spatial-diffusion result for a parabolic barrier top with the
//!   Markovian friction `gamma` on the reaction coordinate (Kramers, Physica 1940, 7, 284;
//!   Hanggi, Talkner, Borkovec, Rev. Mod. Phys. 1990, 62, 251, eq. 4.33):
//!   `kappa = sqrt(1 + (gamma / 2 omega_b)^2) - gamma / (2 omega_b)`, written as
//!   `kappa = exp(-asinh(gamma / 2 omega_b))` (the same number without cancellation). It
//!   tends to 1 without friction and to `omega_b / gamma` for a strong one.
//! - `grote_hynes`: the friction with memory, `kappa = lambda_r / omega_b` with the reactive
//!   frequency `lambda_r` the positive root of `lambda^2 + lambda zeta_hat(lambda) =
//!   omega_b^2` (Grote, Hynes, J. Chem. Phys. 1980, 73, 2715, eq. 2.13), `zeta_hat` the Laplace
//!   transform of the friction kernel per unit mass. The kernel is exponential,
//!   `zeta(t) = (gamma / tau_c) exp(-t / tau_c)` (`zeta_hat = gamma / (1 + lambda tau_c)`;
//!   Kramers for `tau_c = 0`), or tabulated (e.g. the autocorrelation of the force on the
//!   reaction coordinate in dynamics at the barrier top with that coordinate held, divided by
//!   `k_B T`), whose transform is taken by the trapezoidal rule over the table.
//! - `reactive_flux`: the transmission coefficient measured by trajectories from the dividing
//!   surface at the saddle point (Bennett, ACS Symp. Ser. 1977, 46, 63; Chandler, J. Chem.
//!   Phys. 1978, 68, 2959), as `workflow::kinetics_dynamics` estimates it.
//!
//! `omega_b` is the angular frequency of the imaginary mode, `2 pi c |nu|`; the friction is
//! that of the mass-weighted reaction coordinate (mass 1). A saddle point without the data is
//! left as it is (the factor does not apply to it).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::{Diagnostic, RateCorrection, StepContext};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::conditions::Conditions;
use crate::kinetics::network::{Bottleneck, BottleneckKind, Friction, KineticNetwork};

/// The angular frequency (1/s) of the imaginary mode of `b`.
fn barrier_frequency(b: &Bottleneck) -> Option<f64> {
    b.imaginary_wavenumber
        .filter(|w| w.is_finite() && *w != 0.0)
        .map(|w| 2.0 * PI * Units::SPEED_OF_LIGHT_CM_S * w.abs())
}

/// `ln kappa` of Kramers for the friction `gamma` (1/s) at the barrier frequency `omega_b`
/// (1/s).
pub fn kramers_ln_kappa(gamma: f64, omega_b: f64) -> f64 {
    -(gamma / (2.0 * omega_b)).asinh()
}

/// The Laplace transform `zeta_hat(lambda)` (1/s) of a friction kernel.
fn kernel_transform(f: &Friction, lambda: f64) -> MoptResult<f64> {
    if let Some((t, z)) = &f.kernel {
        if t.len() != z.len() || t.len() < 2 {
            return Err(MoptError::config(format!(
                "{}: a friction kernel needs at least two times with one value each",
                f.source
            )));
        }
        let mut sum = 0.0;
        for k in 0..t.len() - 1 {
            let h = t[k + 1] - t[k];
            if h.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                return Err(MoptError::config(format!(
                    "{}: the times of a friction kernel must increase",
                    f.source
                )));
            }
            sum +=
                0.5 * h * ((-lambda * t[k]).exp() * z[k] + (-lambda * t[k + 1]).exp() * z[k + 1]);
        }
        return Ok(sum);
    }
    match (f.gamma, f.memory) {
        (Some(g), Some(tau)) => Ok(g / (1.0 + lambda * tau)),
        (Some(g), None) => Ok(g),
        _ => Err(MoptError::config(format!(
            "{}: grote_hynes needs a friction kernel (a table, or the friction and its memory \
             time)",
            f.source
        ))),
    }
}

/// The reactive frequency `lambda_r` (1/s) of Grote and Hynes: the root of
/// `lambda^2 + lambda zeta_hat(lambda) - omega_b^2` in `(0, omega_b]` (bisection).
pub fn grote_hynes_frequency(f: &Friction, omega_b: f64) -> MoptResult<f64> {
    let g =
        |l: f64| -> MoptResult<f64> { Ok(l * l + l * kernel_transform(f, l)? - omega_b * omega_b) };
    let (mut lo, mut hi) = (0.0, omega_b);
    let mut ghi = g(hi)?;
    let mut widen = 0;
    while ghi < 0.0 {
        // A kernel whose transform is negative there (a table that dips below zero).
        hi *= 2.0;
        ghi = g(hi)?;
        widen += 1;
        if widen > 60 {
            return Err(MoptError::calc(format!(
                "{}: the Grote-Hynes equation has no root (the friction kernel is negative)",
                f.source
            )));
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if g(mid)? < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-15 * hi {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Kramers' transmission coefficient.
#[derive(Debug, Clone, Copy, Default)]
pub struct Kramers {
    /// The friction (1/s) of every saddle point without its own.
    pub friction: Option<f64>,
}

impl RateCorrection for Kramers {
    fn name(&self) -> &str {
        "kramers"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        _c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        if b.kind != BottleneckKind::Saddle {
            return Ok(None);
        }
        let Some(omega) = barrier_frequency(b) else {
            return Ok(None);
        };
        let gamma = b.friction.as_ref().and_then(|f| f.gamma).or(self.friction);
        let Some(gamma) = gamma else {
            return Ok(None);
        };
        if !(gamma >= 0.0 && gamma.is_finite()) {
            return Err(MoptError::config(format!(
                "{}: the friction must be finite and not negative (got {gamma})",
                b.label
            )));
        }
        let ln_k = kramers_ln_kappa(gamma, omega);
        Ok(Some((
            ln_k,
            vec![
                Diagnostic::new("gamma_over_omega_b", gamma / omega),
                Diagnostic::new("kramers_kappa", ln_k.exp()),
            ],
        )))
    }
}

/// Grote and Hynes' transmission coefficient.
#[derive(Debug, Clone, Copy, Default)]
pub struct GroteHynes {
    /// The friction (1/s) of every saddle point without its own (Markovian: Kramers).
    pub friction: Option<f64>,
}

impl RateCorrection for GroteHynes {
    fn name(&self) -> &str {
        "grote_hynes"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        _c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        if b.kind != BottleneckKind::Saddle {
            return Ok(None);
        }
        let Some(omega) = barrier_frequency(b) else {
            return Ok(None);
        };
        let friction = match (&b.friction, self.friction) {
            (Some(f), _) => f.clone(),
            (None, Some(g)) => Friction {
                gamma: Some(g),
                source: format!("{} (--friction)", b.label),
                ..Friction::default()
            },
            (None, None) => return Ok(None),
        };
        let lambda = grote_hynes_frequency(&friction, omega)?;
        let markov = kernel_transform(&friction, 0.0)?;
        let kappa = lambda / omega;
        Ok(Some((
            kappa.ln(),
            vec![
                Diagnostic::new("grote_hynes_kappa", kappa),
                Diagnostic::new("zeta0_over_omega_b", markov / omega),
                Diagnostic::new(
                    "kramers_kappa",
                    kramers_ln_kappa(markov.max(0.0), omega).exp(),
                ),
            ],
        )))
    }
}

/// The transmission coefficient measured by trajectories.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReactiveFlux;

impl RateCorrection for ReactiveFlux {
    fn name(&self) -> &str {
        "reactive_flux"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        _c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        let Some((kappa, error)) = b.reactive_flux else {
            return Ok(None);
        };
        if !(kappa > 0.0 && kappa.is_finite()) {
            return Err(MoptError::config(format!(
                "{}: the transmission coefficient from trajectories must be positive (got \
                 {kappa}; more trajectories, or a dividing surface that is crossed)",
                b.label
            )));
        }
        Ok(Some((
            kappa.ln(),
            vec![
                Diagnostic::new("reactive_flux_kappa", kappa),
                Diagnostic::new("reactive_flux_error", error),
            ],
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kramers_has_its_two_limits() {
        let omega = 2.0 * PI * Units::SPEED_OF_LIGHT_CM_S * 1000.0;
        // No friction: 1; weak: 1 - gamma / (2 omega_b); strong: omega_b / gamma.
        assert_eq!(kramers_ln_kappa(0.0, omega), 0.0);
        let weak = kramers_ln_kappa(1e-6 * omega, omega).exp();
        assert!((weak - (1.0 - 0.5e-6)).abs() < 1e-12, "{weak}");
        let strong = kramers_ln_kappa(1e6 * omega, omega).exp();
        assert!((strong / 1e-6 - 1.0).abs() < 1e-12, "{strong}");
        // The literal form where it has no cancellation.
        let x: f64 = 0.7;
        let literal = (1.0 + x * x).sqrt() - x;
        assert!((kramers_ln_kappa(2.0 * x * omega, omega).exp() - literal).abs() < 1e-15);
    }

    #[test]
    fn grote_hynes_is_kramers_without_memory_and_one_with_a_slow_kernel() {
        let omega = 3.0e13;
        let gamma = 4.0e13;
        let markov = Friction {
            gamma: Some(gamma),
            source: "test".into(),
            ..Friction::default()
        };
        let l = grote_hynes_frequency(&markov, omega).unwrap();
        let kramers = kramers_ln_kappa(gamma, omega).exp();
        assert!(
            (l / omega - kramers).abs() < 1e-12,
            "{} {kramers}",
            l / omega
        );
        // An exponential kernel: the root of l^2 + l gamma / (1 + l tau) = omega^2.
        let tau = 1e-13;
        let memory = Friction {
            memory: Some(tau),
            ..markov.clone()
        };
        let l = grote_hynes_frequency(&memory, omega).unwrap();
        let residual = l * l + l * gamma / (1.0 + l * tau) - omega * omega;
        assert!(residual.abs() < 1e-10 * omega * omega);
        assert!(
            l / omega > kramers,
            "memory lowers the friction a fast barrier feels"
        );
        // A very slow kernel (frozen solvent): the barrier does not feel it, kappa -> 1.
        let frozen = Friction {
            memory: Some(1e-6),
            ..markov.clone()
        };
        let l = grote_hynes_frequency(&frozen, omega).unwrap();
        assert!((l / omega - 1.0).abs() < 1e-6, "{}", l / omega);
        // The same exponential kernel as a table: the transform by the trapezoidal rule.
        let times: Vec<f64> = (0..4001).map(|k| k as f64 * 2.5e-16).collect();
        let values: Vec<f64> = times
            .iter()
            .map(|t| gamma / tau * (-t / tau).exp())
            .collect();
        let table = Friction {
            kernel: Some((times, values)),
            source: "table".into(),
            ..Friction::default()
        };
        let lt = grote_hynes_frequency(&table, omega).unwrap();
        let le = grote_hynes_frequency(&memory, omega).unwrap();
        assert!((lt / le - 1.0).abs() < 1e-4, "{lt} {le}");
    }
}
