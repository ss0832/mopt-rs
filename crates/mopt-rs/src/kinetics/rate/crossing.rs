//! The probability of a step through a crossing point of two electronic states (MECP).
//!
//! By default a crossing is a bottleneck like a saddle point with the probability 1 (the
//! adiabatic limit): `k = (k_B T / h) exp(-(G_MECP - G_min) / RT)`. `landau_zener` multiplies
//! the flux by the thermal average of the double-passage Landau-Zener probability (Harvey,
//! Phys. Chem. Chem. Phys. 2007, 9, 331; Lykhin et al., Int. J. Quantum Chem. 2016, 116, 750),
//!
//! ```text
//! <p_sh> = int_0^inf (1 - P_LZ(E)^2) exp(-E / k_B T) dE / k_B T,
//! P_LZ(E) = exp(-2 pi H12^2 / (hbar dF v)),   v = sqrt(2 E / mu),
//! ```
//!
//! with the coupling `H12` of the two states, the norm `dF` of the difference of their
//! gradients, and the reduced mass `mu` along the hopping coordinate; the integral is taken in
//! `s = sqrt(E / k_B T)` by Gauss-Legendre panels in logarithms. It tends to 1 for a strong
//! coupling and to `4 pi^(3/2) H12^2 / (hbar dF) sqrt(mu / (2 k_B T))` for a weak one. The
//! probability at a given energy is the same from both sides, so the factor belongs in the
//! flux. A crossing without a coupling keeps the probability 1; a crossing with a coupling but
//! without the gradient difference or the reduced mass is refused with what is missing.
//!
//! `1 - P_LZ^2` is the double passage of a peaked crossing (the slopes of the two states of
//! opposite signs, the minima on the two sides): the step is made unless both passages keep
//! the diabatic state. At a sloped crossing (the slopes of the same sign, both minima on one
//! side) the step needs exactly one of the two passages to change the diabatic state:
//! `2 P_LZ (1 - P_LZ)` with the phase averaged, which is at most 1/2. The sloped form is used
//! when the product of the slopes (`slope_product`) is given and positive; without it the
//! crossing is taken as peaked.
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
use crate::kinetics::network::{Bottleneck, BottleneckKind, KineticNetwork};
use crate::kinetics::quadrature::ln_integral;

/// `ln <p_sh>` of a peaked crossing for the coupling `h12` (Hartree), the gradient
/// difference `df` (Hartree/Bohr), the reduced mass `mu` (amu), and `kt` (Hartree).
pub fn landau_zener_ln_average(h12: f64, df: f64, mu: f64, kt: f64) -> f64 {
    let a = landau_zener_exponent(h12, df, mu, kt);
    // int_0^inf 2 s exp(-s^2) (1 - exp(-2 a / s)) ds; the tail beyond s = 9 is below 1e-35.
    ln_integral(
        |s| (2.0 * s).ln() - s * s + (-(-2.0 * a / s).exp_m1()).ln(),
        0.0,
        9.0,
        200,
        16,
    )
}

/// `ln <2 P_LZ (1 - P_LZ)>` of a sloped crossing (the arguments of
/// [`landau_zener_ln_average`]).
pub fn landau_zener_ln_average_sloped(h12: f64, df: f64, mu: f64, kt: f64) -> f64 {
    let a = landau_zener_exponent(h12, df, mu, kt);
    ln_integral(
        |s| (4.0 * s).ln() - s * s - a / s + (-(-a / s).exp_m1()).ln(),
        0.0,
        9.0,
        200,
        16,
    )
}

/// `a` of `P_LZ = exp(-a / s)` with `E = kT s^2` (atomic units, `hbar = 1`).
fn landau_zener_exponent(h12: f64, df: f64, mu: f64, kt: f64) -> f64 {
    let mass = mu * Units::AMU_TO_AU_MASS;
    2.0 * PI * h12 * h12 / (df * (2.0 * kt / mass).sqrt())
}

/// The Landau-Zener probability at crossings with a coupling.
#[derive(Debug, Clone, Copy, Default)]
pub struct LandauZener;

impl RateCorrection for LandauZener {
    fn name(&self) -> &str {
        "landau_zener"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        if b.kind != BottleneckKind::Crossing {
            return Ok(None);
        }
        let Some(h12) = b.coupling else {
            return Ok(None);
        };
        let (Some(df), Some(mu)) = (b.gradient_difference, b.reduced_mass) else {
            return Err(MoptError::config(format!(
                "{}: the Landau-Zener probability needs the gradient difference and the reduced \
                 mass along the hopping coordinate besides the coupling (annotations: \
                 gradient_difference, reduced_mass_amu)",
                b.label
            )));
        };
        if !(h12.is_finite() && df > 0.0 && mu > 0.0 && df.is_finite() && mu.is_finite()) {
            return Err(MoptError::config(format!(
                "{}: the coupling must be finite and the gradient difference and the reduced \
                 mass positive",
                b.label
            )));
        }
        let sloped = b.slope_product.is_some_and(|p| p > 0.0);
        let ln_p = if sloped {
            landau_zener_ln_average_sloped(h12.abs(), df, mu, c.kt())
        } else {
            landau_zener_ln_average(h12.abs(), df, mu, c.kt())
        };
        Ok(Some((
            ln_p,
            vec![
                Diagnostic::new("p_sh", ln_p.exp()),
                Diagnostic::new("sloped", if sloped { 1.0 } else { 0.0 }),
            ],
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_landau_zener_average_has_its_two_limits() {
        let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
        let (df, mu) = (0.02, 10.0);
        // Weak coupling: 4 pi^(3/2) H12^2 / dF sqrt(mu / (2 kT)) (atomic units); the next
        // term is of the order of a ln(1 / a) with the exponent a of P_LZ at k_B T.
        let h12 = 1e-7;
        let weak = landau_zener_ln_average(h12, df, mu, kt).exp();
        let mass = mu * Units::AMU_TO_AU_MASS;
        let want = 4.0 * PI.powf(1.5) * h12 * h12 / df * (mass / (2.0 * kt)).sqrt();
        assert!((weak / want - 1.0).abs() < 1e-5, "{weak} {want}");
        // Strong coupling: 1.
        let strong = landau_zener_ln_average(0.05, df, mu, kt).exp();
        assert!((strong - 1.0).abs() < 1e-8, "{strong}");
        // A sloped crossing: twice the weak limit of one passage (either passage may change
        // the state), and 0 for a strong coupling (both follow the adiabatic state).
        let weak = landau_zener_ln_average_sloped(h12, df, mu, kt).exp();
        assert!((weak / want - 1.0).abs() < 1e-5, "{weak} {want}");
        let strong = landau_zener_ln_average_sloped(0.05, df, mu, kt).exp();
        assert!(strong < 1e-6, "{strong}");
        let middle = landau_zener_ln_average_sloped(2e-4, df, mu, kt).exp();
        assert!(middle > 0.0 && middle <= 0.5, "{middle}");
    }
}
