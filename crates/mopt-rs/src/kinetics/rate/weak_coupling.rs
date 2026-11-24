//! `weak_coupling`: the probability of a step through a crossing point of two electronic
//! states in the weak-coupling limit, with tunnelling below the crossing.
//!
//! For two straight diabatic states with the slopes `F1`, `F2` along the hopping coordinate,
//! the coupling `H12`, and the reduced mass `mu`, the probability of a transition at the energy
//! `E` (the double passage, its phase averaged far above the crossing) is the Airy form of
//! Delos's weak-coupling result as quoted for spin-forbidden reactions (Harvey, Phys. Chem.
//! Chem. Phys. 2007, 9, 331; Lykhin, Kaliakin, dePolo, Kuzubov, Varganov, Int. J. Quantum
//! Chem. 2016, 116, 750; atomic units, `hbar = 1`):
//!
//! ```text
//! P(E) = 4 pi^2 H12^2 (2 mu / (F dF))^(2/3) Ai^2(-(E - E_X) (2 mu dF^2 / F^4)^(1/3))
//! F = sqrt(|F1 F2|),   dF = |F1 - F2|
//! ```
//!
//! Far above the crossing it is the weak limit of Landau-Zener's double passage,
//! `4 pi H12^2 / (dF v)` (checked in the tests); below it `Ai` decays (tunnelling). The
//! thermal factor referred to the crossing, `kappa = (1 / k_B T) int P(E) exp(-(E - E_X) / k_B T)
//! dE` over the energies above the higher of the two minima (the lower barrier, at most the
//! crossing itself: the same factor from both sides), is closed when taken over all energies,
//! with `t = -(E - E_X) gamma`, `gamma = (2 mu dF^2 / F^4)^(1/3)`, `p = 1 / (gamma k_B T)`:
//!
//! ```text
//! int_-inf^inf exp(p t) Ai^2(t) dt = exp(p^3 / 12) / (2 sqrt(pi p))
//! kappa_all = (C / 2) sqrt(p / pi) exp(p^3 / 12),   C = 4 pi^2 H12^2 (2 mu / (F dF))^(2/3)
//! ```
//!
//! which is `4 pi^(3/2) H12^2 / dF sqrt(mu / (2 k_B T))` (the thermal weak-coupling limit of
//! `kinetics_beyond_tst.md`, section 2.5) times the tunnelling factor `exp(p^3 / 12)`. The
//! energies below the higher minimum are taken out by quadrature (`Ai^2` has no oscillation
//! there); for `p >= 1` the whole range is integrated directly instead.
//!
//! The weak-coupling form needs `P << 1`; where the Landau-Zener exponent at `k_B T` is not
//! small (`lz_exponent` in the diagnostics) the strong-coupling forms (`landau_zener`,
//! `zhu_nakamura`) apply. It is the same for peaked and sloped crossings (in the weak limit
//! both double passages are `2 (1 - P_LZ)`).
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
use crate::math::special::{airy_ai, ln_airy_ai_squared_positive, ln_laplace_airy_squared};
use crate::math::stable::{log_diff_exp, log_sum_exp};

/// The weak-coupling factor and its parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeakCoupling {
    /// `ln kappa`.
    pub ln_kappa: f64,
    /// `p = 1 / (gamma k_B T)`: small at high temperature (the classical limit), large where
    /// tunnelling dominates.
    pub p: f64,
    /// The thermal weak-coupling limit `4 pi^(3/2) H12^2 / dF sqrt(mu / (2 k_B T))`.
    pub classical: f64,
    /// The Landau-Zener exponent `2 pi H12^2 / (dF sqrt(2 k_B T / mu))` at `k_B T`.
    pub lz_exponent: f64,
}

/// `ln int_-inf^lo exp(p t) Ai^2(t) dt`.
fn ln_lower_integral(p: f64, lo: f64) -> f64 {
    if p >= 1.0 {
        // Directly: exp(p t) damps the oscillations below 0; above 0 the integrand peaks at
        // t = p^2 / 4 and falls off as exp(p t - 4/3 t^(3/2)).
        let start = -45.0 / p;
        let top = lo.min(p * p + 30.0);
        if top <= start {
            return f64::NEG_INFINITY;
        }
        let mut parts = Vec::new();
        // Below 0: oscillating, about 16 panels per oscillation.
        let neg_end = top.min(0.0);
        if neg_end > start {
            let periods = (2.0 / 3.0 * (-start).powf(1.5)) / PI;
            let panels = (16.0 * periods).ceil().clamp(16.0, 20000.0) as usize;
            // `Ai^2` has zeros there: the integrand is summed plainly (it does not underflow
            // within 45 / p of 0).
            let (x, w) = crate::kinetics::quadrature::gauss_legendre(16);
            let h = (neg_end - start) / panels as f64;
            let mut sum = 0.0;
            for k in 0..panels {
                let mid = start + (k as f64 + 0.5) * h;
                for (xi, wi) in x.iter().zip(&w) {
                    let t = mid + 0.5 * h * xi;
                    let a = airy_ai(t);
                    sum += 0.5 * h * wi * (p * t).exp() * a * a;
                }
            }
            if sum > 0.0 {
                parts.push(sum.ln());
            }
        }
        if top > 0.0 {
            parts.push(ln_integral(
                |t| p * t + ln_airy_ai_squared_positive(t),
                0.0,
                top,
                400,
                16,
            ));
        }
        log_sum_exp(&parts)
    } else {
        // The whole line minus what lies above `lo` (no oscillation there for lo >= 0).
        let all = ln_laplace_airy_squared(p);
        let lo = lo.max(0.0);
        let cut = ln_integral(
            |t| p * t + ln_airy_ai_squared_positive(t),
            lo,
            lo + 40.0,
            400,
            16,
        );
        if cut == f64::NEG_INFINITY {
            all
        } else {
            log_diff_exp(all, cut)
        }
    }
}

/// The weak-coupling factor for the coupling `h12` (Hartree), the difference `df` and the
/// product `slope_product` of the slopes (Hartree/Bohr, Hartree^2/Bohr^2), the reduced mass
/// `mu` (amu), `kt` (Hartree), and the barrier from the higher minimum to the crossing
/// (Hartree; energies below that minimum are not counted, and none above the crossing).
pub fn weak_coupling(
    h12: f64,
    df: f64,
    slope_product: f64,
    mu: f64,
    kt: f64,
    barrier: f64,
) -> MoptResult<WeakCoupling> {
    let f = slope_product.abs().sqrt();
    if !(h12.is_finite() && df > 0.0 && f > 0.0 && mu > 0.0 && kt > 0.0 && df.is_finite()) {
        return Err(MoptError::config(
            "weak_coupling: the coupling must be finite, the gradient difference, the product of \
             the slopes, and the reduced mass positive",
        ));
    }
    let mass = mu * Units::AMU_TO_AU_MASS;
    let gamma = (2.0 * mass * df * df / f.powi(4)).cbrt();
    let ln_c = (4.0 * PI * PI * h12 * h12).ln() + 2.0 / 3.0 * (2.0 * mass / (f * df)).ln();
    let p = 1.0 / (gamma * kt);
    let lo = gamma * barrier.max(0.0);
    let ln_kappa = ln_c + p.ln() + ln_lower_integral(p, lo);
    let classical = 4.0 * PI.powf(1.5) * h12 * h12 / df * (mass / (2.0 * kt)).sqrt();
    let lz_exponent = 2.0 * PI * h12 * h12 / (df * (2.0 * kt / mass).sqrt());
    Ok(WeakCoupling {
        ln_kappa,
        p,
        classical,
        lz_exponent,
    })
}

/// The weak-coupling probability at crossing points with a coupling.
#[derive(Debug, Clone, Copy, Default)]
pub struct WeakCouplingCorrection;

impl RateCorrection for WeakCouplingCorrection {
    fn name(&self) -> &str {
        "weak_coupling"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        if b.kind != BottleneckKind::Crossing {
            return Ok(None);
        }
        let Some(h12) = b.coupling else {
            return Ok(None);
        };
        let (Some(df), Some(product), Some(mu)) =
            (b.gradient_difference, b.slope_product, b.reduced_mass)
        else {
            return Err(MoptError::config(format!(
                "{}: the weak-coupling probability needs the gradient difference, the product of \
                 the slopes, and the reduced mass along the hopping coordinate besides the \
                 coupling (annotations: gradient_difference, slope_product, reduced_mass_amu, or \
                 diabatic_crossings)",
                b.label
            )));
        };
        let barrier = step.barriers[0].min(step.barriers[1]);
        let w = weak_coupling(h12.abs(), df, product, mu, c.kt(), barrier)
            .map_err(|e| e.context(&b.label))?;
        Ok(Some((
            w.ln_kappa,
            vec![
                Diagnostic::new("p_wc", w.ln_kappa.exp()),
                Diagnostic::new("airy_p", w.p),
                Diagnostic::new("p_wc_classical", w.classical),
                Diagnostic::new("lz_exponent", w.lz_exponent),
            ],
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinetics::rate::crossing::landau_zener_ln_average;

    #[test]
    fn the_weak_coupling_factor_has_its_limits() {
        let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
        // A heavy hopping mode with one nearly flat state: the Airy scale 1 / gamma is far
        // below k_B T (p -> 0). With a high barrier the factor is the closed thermal form, with
        // the tunnelling factor exp(p^3 / 12) as the only correction.
        let (df, product, mu, h12) = (0.05, 2.5e-5, 100.0, 1e-6);
        let w = weak_coupling(h12, df, product, mu, kt, 0.1).unwrap();
        assert!(w.p < 0.1, "{}", w.p);
        let want = w.classical * (w.p.powi(3) / 12.0).exp();
        assert!(
            (w.ln_kappa.exp() / want - 1.0).abs() < 1e-8,
            "{} {want}",
            w.ln_kappa.exp()
        );
        // It is the weak limit of Landau-Zener's thermal average to the order of the
        // tunnelling.
        let lz = landau_zener_ln_average(h12, df, mu, kt).exp();
        assert!(
            (w.ln_kappa.exp() / lz - 1.0).abs() < 1e-3,
            "{} {lz}",
            w.ln_kappa.exp()
        );
        // With the higher minimum at the crossing, the energies below it are lost.
        let at = weak_coupling(h12, df, product, mu, kt, 0.0).unwrap();
        assert!(at.ln_kappa < w.ln_kappa);
        // A lighter, steeper crossing at 847 K (p = 2): the direct quadrature, equal to the
        // closed form when the higher minimum lies far below the weight of the tunnelling.
        let (df, product, mu) = (0.02, 0.0015, 10.0);
        let mass = mu * Units::AMU_TO_AU_MASS;
        let gamma = (2.0 * mass * df * df / product * 1.0 / product).cbrt();
        let warm = 1.0 / (2.0 * gamma);
        let w = weak_coupling(h12, df, product, mu, warm, 0.2).unwrap();
        assert!((w.p - 2.0).abs() < 1e-12, "{}", w.p);
        let ln_c = (4.0 * PI * PI * h12 * h12).ln()
            + 2.0 / 3.0 * (2.0 * mass / (product.sqrt() * df)).ln();
        let closed = ln_c + w.p.ln() + ln_laplace_airy_squared(w.p);
        assert!(
            (w.ln_kappa - closed).abs() < 1e-7,
            "{} {closed}",
            w.ln_kappa
        );
        assert!(weak_coupling(h12, 0.0, product, mu, warm, 0.2).is_err());
    }
}
