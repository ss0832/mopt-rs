//! Tunnelling corrections of a saddle point from its imaginary frequency `nu` and the barriers
//! `V1`, `V2` from its two minima (electronic, with the zero-point energies where the
//! frequencies of the three stationary points are known), and a given transmission
//! coefficient. With `u = h nu / k_B T` and the crossover temperature `T_c = h nu / (2 pi k_B)`:
//!
//! - `wigner`: `kappa = 1 + u^2 / 24` (Wigner, Z. Phys. Chem. B 1932, 19, 203), valid for
//!   `u` well below `2 pi`;
//! - `eckart`: the thermal average of the transmission probability of the unsymmetrical
//!   Eckart barrier fitted to `V1`, `V2`, and `nu` (Eckart, Phys. Rev. 1930, 35, 1303; Johnston,
//!   Heicklen, J. Phys. Chem. 1962, 66, 532), referred to the top of the barrier:
//!   `kappa = (1 / k_B T) int_{-min(V1, V2)}^inf P(e) exp(-e / k_B T) de`, the probability
//!   evaluated as `2 sinh(2 pi a) sinh(2 pi b) / (cosh(2 pi (a + b)) + cosh(2 pi d))` with every
//!   term divided by `exp(2 pi (a + b))` (`d` imaginary: `cos`), the integral in logarithms by
//!   Gauss-Legendre panels;
//! - `skodje_truhlar`: the closed form of Skodje and Truhlar (J. Phys. Chem. 1981, 85, 624)
//!   with `alpha = 2 pi / h nu`, `beta = 1 / k_B T` and the barrier from the higher minimum
//!   (`V - V0` with the endoergicity `V0`), so that the factor is the same from both sides;
//!   the cancelling terms at `beta = alpha` are replaced by their expansion;
//! - `bell`: Bell's parabolic barrier (R. P. Bell, Trans. Faraday Soc. 1959, 55, 1; The Tunnel
//!   Effect in Chemistry, 1980; neither was checked here, so the formulas are derived in the
//!   code and checked against the parabolic limit of the Eckart barrier and against
//!   Skodje-Truhlar). The infinite parabola, the transmission `P(E) = 1 / (1 + exp(-2 pi E /
//!   h nu))` (`E` from the top) averaged over all energies: `kappa = (u / 2) / sin(u / 2)` for
//!   `u < 2 pi` (`1 + u^2/24 + 7 u^4/5760 + ...`, Wigner to `O(u^4)`), divergent at `u = 2 pi`.
//!   The truncated parabola, the same average cut at the barrier `V` (by default the barrier
//!   from the higher minimum, as Skodje-Truhlar, so the factor is the same from both sides; the
//!   forward or reverse barrier on request): finite for every `u`, computed in logarithms by
//!   Gauss-Legendre panels. Its expansion below `2 pi`,
//!   `kappa = pi r / sin(pi r) - r sum_n (-1)^(n-1) e^{-(n - r) a} / (n - r)` (`r = u / 2 pi`,
//!   `a = 2 pi V / h nu`), has Skodje-Truhlar's closed form as its first two terms; above
//!   `2 pi` Skodje-Truhlar is its leading behaviour. `--bell-truncation auto` (the default)
//!   takes the infinite parabola where the truncation changes `kappa` by less than 1e-8 of it
//!   and the truncated one otherwise (near and past `2 pi`, low barriers);
//! - `kappa`: a given transmission coefficient (per step from the annotations, or `--kappa`
//!   for every step).
//!
//! Each factor is the same in both directions (the transmission probability at a total energy
//! does not depend on the side), so it multiplies the flux of the bottleneck. Without an
//! imaginary frequency, or with a barrier that is not positive (Eckart), a correction does
//! not apply to the step.
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
use crate::math::stable::{log_sum_exp, softplus};

/// `h nu` (Hartree) of a wavenumber (cm^-1; the sign is ignored).
pub fn quantum_of(wavenumber: f64) -> f64 {
    wavenumber.abs() * Units::PLANCK_J_S * Units::SPEED_OF_LIGHT_CM_S / Units::HARTREE_TO_JOULE
}

fn saddle_quantum(b: &Bottleneck) -> Option<f64> {
    if b.kind != BottleneckKind::Saddle {
        return None;
    }
    b.imaginary_wavenumber
        .filter(|w| w.is_finite() && *w != 0.0)
        .map(quantum_of)
}

fn diagnostics(h_nu: f64, kt: f64) -> Vec<Diagnostic> {
    let u = h_nu / kt;
    vec![
        Diagnostic::new("u", u),
        Diagnostic::new("T/Tc", 2.0 * PI / u),
    ]
}

/// Wigner's correction.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wigner;

impl RateCorrection for Wigner {
    fn name(&self) -> &str {
        "wigner"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        let Some(h_nu) = saddle_quantum(b) else {
            return Ok(None);
        };
        let u = h_nu / c.kt();
        Ok(Some(((u * u / 24.0).ln_1p(), diagnostics(h_nu, c.kt()))))
    }
}

/// When `bell` uses the truncated parabola.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BellTruncation {
    /// The infinite parabola where the truncation changes `kappa` by less than 1e-8 of it
    /// (`u` well below `2 pi` and a barrier many `h nu` high), the truncated one otherwise
    /// (near and past `u = 2 pi`, low barriers).
    #[default]
    Auto,
    /// Always the truncated parabola.
    On,
    /// Always the infinite parabola (refused at `u >= 2 pi`, where it diverges).
    Off,
}

impl BellTruncation {
    /// `auto`, `on`, or `off`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "truncated" => Ok(Self::On),
            "off" | "infinite" => Ok(Self::Off),
            other => Err(MoptError::parse(format!(
                "unknown truncation of bell '{other}' (auto, on, off)"
            ))),
        }
    }
}

/// Which barrier cuts the truncated parabola of `bell`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BellBarrier {
    /// The barrier from the higher minimum (both barriers: no energy below either minimum),
    /// as Skodje and Truhlar.
    #[default]
    Higher,
    /// The barrier from the first state of the step (the forward barrier).
    Forward,
    /// The barrier from the second state of the step (the reverse barrier).
    Reverse,
}

impl BellBarrier {
    /// `higher` (or `both`), `forward`, or `reverse`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "higher" | "both" => Ok(Self::Higher),
            "forward" => Ok(Self::Forward),
            "reverse" => Ok(Self::Reverse),
            other => Err(MoptError::parse(format!(
                "unknown barrier of bell '{other}' (higher, forward, reverse)"
            ))),
        }
    }

    fn of(self, barriers: [f64; 2]) -> f64 {
        match self {
            Self::Higher => barriers[0].min(barriers[1]),
            Self::Forward => barriers[0],
            Self::Reverse => barriers[1],
        }
    }
}

/// The settings of `bell`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BellSettings {
    /// When the truncated parabola is used.
    pub truncation: BellTruncation,
    /// Which barrier cuts it.
    pub barrier: BellBarrier,
}

/// `ln kappa` of Bell's infinite parabola, `ln((u / 2) / sin(u / 2))`, for `0 <= u < 2 pi`
/// (`None` beyond, where it diverges).
pub fn bell_ln_kappa(u: f64) -> Option<f64> {
    if u.is_nan() || !(0.0..2.0 * PI).contains(&u) {
        return None;
    }
    let x = 0.5 * u;
    if x == 0.0 {
        return Some(0.0);
    }
    Some((x / x.sin()).ln())
}

/// `ln kappa` of the truncated parabola: `(1 / k_B T) int_{-V}^inf P(E) exp(-E / k_B T) dE`
/// with `P(E) = 1 / (1 + exp(-2 pi E / h nu))` (`E` from the top of the barrier) cut at the
/// barrier `v` (all energies in the same unit). With `z = 2 pi E / h nu` and
/// `r = h nu / (2 pi k_B T)`:
///
/// ```text
/// kappa = r int_{-a}^0 e^{(1 - r) z} / (1 + e^z) dz + 1 - r int_0^inf e^{-(1 + r) z} / (1 + e^{-z}) dz
/// ```
///
/// (`a = 2 pi V / h nu`), both integrals in logarithms by Gauss-Legendre panels, so that it
/// stays finite for any `u`.
pub fn bell_truncated_ln_kappa(v: f64, h_nu: f64, kt: f64) -> f64 {
    let r = h_nu / (2.0 * PI * kt);
    let a = 2.0 * PI * v / h_nu;
    // Below the top: e^{(1 - r) z} decays toward -a for r < 1, grows for r > 1.
    let rate = (1.0 - r).abs().max(1.0);
    let start = if r < 1.0 - 1e-3 {
        (-a).max(-60.0 / (1.0 - r))
    } else {
        -a
    };
    let lower = if start < 0.0 {
        let panels = ((-start) * rate * 2.0).ceil().clamp(32.0, 400_000.0) as usize;
        ln_integral(|z| (1.0 - r) * z - softplus(z), start, 0.0, panels, 16)
    } else {
        f64::NEG_INFINITY
    };
    // Above the top: r int_0^inf e^{-r z} / (1 + e^{-z}) dz = 1 - r I2.
    let end = 60.0 / (1.0 + r);
    let i2 = ln_integral(|z| -(1.0 + r) * z - softplus(-z), 0.0, end, 64, 16).exp();
    let upper = (-(r * i2)).ln_1p();
    log_sum_exp(&[r.ln() + lower, upper])
}

/// Bell's parabolic-barrier correction.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bell {
    /// Its settings.
    pub settings: BellSettings,
}

impl RateCorrection for Bell {
    fn name(&self) -> &str {
        "bell"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        let Some(h_nu) = saddle_quantum(b) else {
            return Ok(None);
        };
        let kt = c.kt();
        let u = h_nu / kt;
        let v = self.settings.barrier.of(step.barriers);
        let mut d = diagnostics(h_nu, kt);
        let infinite = bell_ln_kappa(u);
        let truncated = match self.settings.truncation {
            BellTruncation::Off => false,
            BellTruncation::On => true,
            BellTruncation::Auto => match infinite {
                None => true,
                // The leading term of the truncation (that of Skodje and Truhlar).
                Some(_) if v.is_nan() || v <= 0.0 => false,
                Some(ln_k) => {
                    let r = u / (2.0 * PI);
                    let a = 2.0 * PI * v / h_nu;
                    let tail = (r / (1.0 - r)).ln() - (1.0 - r) * a;
                    tail - ln_k > (1e-8_f64).ln()
                }
            },
        };
        if truncated {
            if v.is_nan() || v <= 0.0 {
                return Ok(None);
            }
            d.push(Diagnostic::new("bell_truncated", 1.0));
            d.push(Diagnostic::new("bell_barrier_kT", v / kt));
            return Ok(Some((bell_truncated_ln_kappa(v, h_nu, kt), d)));
        }
        match infinite {
            Some(ln_k) => {
                d.push(Diagnostic::new("bell_truncated", 0.0));
                Ok(Some((ln_k, d)))
            }
            None => Err(MoptError::config(format!(
                "{}: Bell's infinite parabola diverges at u = h nu / k_B T = {u:.3} >= 2 pi; \
                 use the truncated parabola (--bell-truncation auto or on)",
                b.label
            ))),
        }
    }
}

/// `ln P(e)` of the Eckart barrier with the barriers `v1`, `v2` (from the two sides) and the
/// quantum `h_nu` of its imaginary frequency, at the energy `e` above its top (all in the
/// same unit); `-inf` at and below `-min(v1, v2)`.
pub fn eckart_ln_probability(e: f64, v1: f64, v2: f64, h_nu: f64) -> f64 {
    let (k1, k2) = (e + v1, e + v2);
    if !(k1 > 0.0 && k2 > 0.0) {
        return f64::NEG_INFINITY;
    }
    let width = h_nu * (1.0 / v1.sqrt() + 1.0 / v2.sqrt());
    let a = 4.0 * PI * k1.sqrt() / width;
    let b = 4.0 * PI * k2.sqrt() / width;
    let d2 = 16.0 * PI * PI * v1 * v2 / (h_nu * h_nu) - PI * PI;
    let s = a + b;
    let numerator = (-(-2.0 * a).exp_m1()).ln() + (-(-2.0 * b).exp_m1()).ln();
    let denominator = if d2 >= 0.0 {
        let d = d2.sqrt();
        log_sum_exp(&[0.0, -2.0 * s, d - s, -d - s])
    } else {
        // cosh(i d) = cos(d): 1 + e^-2s + 2 cos(d) e^-s = (1 - e^-s)^2 + 4 cos^2(d / 2) e^-s.
        let d = (-d2).sqrt();
        let one_minus = -(-s).exp_m1();
        (one_minus * one_minus + 4.0 * (0.5 * d).cos().powi(2) * (-s).exp()).ln()
    };
    numerator - denominator
}

/// `ln kappa` of the Eckart barrier at `kt` (all energies in the same unit).
pub fn eckart_ln_kappa(v1: f64, v2: f64, h_nu: f64, kt: f64) -> f64 {
    let lowest = v1.min(v2) / kt;
    let upper = 40.0 + 40.0 * (h_nu / kt).max(1.0);
    let panels = ((lowest + upper) * 4.0).ceil().clamp(200.0, 40_000.0) as usize;
    // (1 / kT) int P(e) exp(-e / kT) de with e = kT u.
    let body = ln_integral(
        |u| eckart_ln_probability(kt * u, v1, v2, h_nu) - u,
        -lowest,
        upper,
        panels,
        16,
    );
    // Beyond `upper` the probability is 1 to double precision.
    log_sum_exp(&[body, -upper])
}

/// The Eckart correction.
#[derive(Debug, Clone, Copy, Default)]
pub struct Eckart;

impl RateCorrection for Eckart {
    fn name(&self) -> &str {
        "eckart"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        let Some(h_nu) = saddle_quantum(b) else {
            return Ok(None);
        };
        let [v1, v2] = step.barriers;
        if !(v1 > 0.0 && v2 > 0.0) {
            return Ok(None);
        }
        let kt = c.kt();
        let mut d = diagnostics(h_nu, kt);
        d.push(Diagnostic::new(
            "eckart_zero_point",
            if step.zero_point { 1.0 } else { 0.0 },
        ));
        Ok(Some((eckart_ln_kappa(v1, v2, h_nu, kt), d)))
    }
}

/// `ln kappa` of Skodje and Truhlar with the barrier `v` from the higher minimum, the quantum
/// `h_nu`, and `kt` (all in the same unit).
pub fn skodje_truhlar_ln_kappa(v: f64, h_nu: f64, kt: f64) -> f64 {
    let alpha = 2.0 * PI / h_nu;
    let beta = 1.0 / kt;
    let x = beta - alpha;
    if x.abs() < 1e-7 * alpha {
        // Expansion about beta = alpha (the two branches agree there).
        let base = alpha * v;
        let slope = if x < 0.0 {
            alpha * v * v / 2.0 + v - PI * PI / (6.0 * alpha)
        } else {
            alpha * v * v / 2.0 + v
        };
        return (base + x * slope).ln();
    }
    if x < 0.0 {
        let r = beta * PI / alpha;
        let first = r / r.sin();
        let second = beta / (alpha - beta) * (x * v).exp();
        (first - second).ln()
    } else {
        // beta / (beta - alpha) (exp(x v) - 1), in logarithms: exp(x v) may overflow.
        let z = x * v;
        let ln_expm1 = if z > 30.0 {
            z + (-(-z).exp()).ln_1p()
        } else {
            z.exp_m1().ln()
        };
        (beta / x).ln() + ln_expm1
    }
}

/// The Skodje-Truhlar correction.
#[derive(Debug, Clone, Copy, Default)]
pub struct SkodjeTruhlar;

impl RateCorrection for SkodjeTruhlar {
    fn name(&self) -> &str {
        "skodje_truhlar"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        let Some(h_nu) = saddle_quantum(b) else {
            return Ok(None);
        };
        let v = step.barriers[0].min(step.barriers[1]);
        if v.is_nan() || v <= 0.0 {
            return Ok(None);
        }
        Ok(Some((
            skodje_truhlar_ln_kappa(v, h_nu, c.kt()),
            diagnostics(h_nu, c.kt()),
        )))
    }
}

/// A given transmission coefficient: the one of the step, or `value` for every step.
#[derive(Debug, Clone, Copy, Default)]
pub struct Kappa {
    /// The coefficient of the steps without their own.
    pub value: Option<f64>,
}

impl RateCorrection for Kappa {
    fn name(&self) -> &str {
        "kappa"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        _c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        match b.kappa.or(self.value) {
            None => Ok(None),
            Some(k) if k > 0.0 && k.is_finite() => Ok(Some((k.ln(), Vec::new()))),
            Some(k) => Err(MoptError::config(format!(
                "{}: the transmission coefficient must be positive (got {k})",
                b.label
            ))),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn eckart_has_the_wigner_and_parabolic_limits_and_is_symmetric() {
        let kt = 1.0;
        // High barriers: the parabolic barrier, kappa = (u / 2) / sin(u / 2) for u < 2 pi.
        for u in [0.2_f64, 1.0, 2.0, 4.0] {
            // The deviation from the parabola falls as 1 / V (1.7e-4 at V = 5000 kT, u = 4).
            let k = eckart_ln_kappa(20000.0, 20800.0, u * kt, kt).exp();
            let want = (u / 2.0) / (u / 2.0).sin();
            assert!((k / want - 1.0).abs() < 1e-4, "u = {u}: {k} against {want}");
        }
        // Small u and a high barrier: Wigner (at 50 kT the finite barrier adds 2e-6).
        let u = 0.05_f64;
        let k = eckart_ln_kappa(5000.0, 5000.0, u * kt, kt).exp();
        assert!((k - (1.0 + u * u / 24.0)).abs() < 1e-6, "{k}");
        // The same from both sides (the factor is referred to the top).
        for (v1, v2, h) in [(5.0, 12.0, 3.0), (20.0, 2.0, 7.5), (8.0, 8.0, 0.5)] {
            let a = eckart_ln_kappa(v1, v2, h, kt);
            let b = eckart_ln_kappa(v2, v1, h, kt);
            assert!((a - b).abs() < 1e-10, "{a} {b}");
        }
        // Deep tunnelling stays finite in logarithms.
        let deep = eckart_ln_kappa(40.0, 60.0, 30.0, kt);
        assert!(deep.is_finite() && deep > 10.0, "{deep}");
    }

    /// Transmission probability of `V(x) = A y / (1 + y) + B y / (1 + y)^2`, `y = exp(2 pi x
    /// / L)`, at the energy `e` above the reactant side, by Numerov's method (hbar = m = 1).
    fn numerov_transmission(v1: f64, v2: f64, l: f64, e: f64) -> f64 {
        let (a, b) = (v1 - v2, (v1.sqrt() + v2.sqrt()).powi(2));
        let pot = |x: f64| {
            let y = (2.0 * PI * x / l).exp();
            a * y / (1.0 + y) + b * y / ((1.0 + y) * (1.0 + y))
        };
        let (k_r, k_p) = ((2.0 * e).sqrt(), (2.0 * (e - a)).sqrt());
        let (x_lo, x_hi, h) = (-40.0_f64, 40.0_f64, 2e-4_f64);
        let n = ((x_hi - x_lo) / h).round() as usize;
        let x = |k: usize| x_lo + k as f64 * h;
        let q = |k: usize| 2.0 * (e - pot(x(k)));
        let f = |k: usize| 1.0 + h * h * q(k) / 12.0;
        // Complex numbers as (re, im); the transmitted wave exp(i k_p x) on the right.
        let wave = |k: f64, x: f64| (((k * x).cos()), (k * x).sin());
        let mut next = wave(k_p, x(n));
        let mut here = wave(k_p, x(n - 1));
        for k in (1..n).rev() {
            let c = 2.0 * (1.0 - 5.0 * h * h * q(k) / 12.0);
            let prev = (
                (c * here.0 - f(k + 1) * next.0) / f(k - 1),
                (c * here.1 - f(k + 1) * next.1) / f(k - 1),
            );
            next = here;
            here = prev;
        }
        // psi = I exp(i k x) + R exp(-i k x) at x(0) (here) and x(1) (next).
        let mul = |p: (f64, f64), q: (f64, f64)| (p.0 * q.0 - p.1 * q.1, p.0 * q.1 + p.1 * q.0);
        let (x0, x1) = (x(0), x(1));
        let num_a = mul(here, wave(k_r, -x1));
        let num_b = mul(next, wave(k_r, -x0));
        let num = (num_a.0 - num_b.0, num_a.1 - num_b.1);
        let det = (0.0, -2.0 * (k_r * (x1 - x0)).sin());
        let det2 = det.0 * det.0 + det.1 * det.1;
        let inc2 = (num.0 * num.0 + num.1 * num.1) / det2;
        (k_p / k_r) / inc2
    }

    #[test]
    fn the_eckart_probability_is_that_of_the_schroedinger_equation() {
        let (v1, v2, l) = (2.0_f64, 1.0_f64, 4.0_f64);
        let (a, b) = (v1 - v2, (v1.sqrt() + v2.sqrt()).powi(2));
        let omega = (2.0 * PI / l) * (b * b - a * a) / (8.0 * b * b * b).sqrt();
        for e in [1.2, 1.6, 2.0, 2.4, 3.0] {
            let exact = numerov_transmission(v1, v2, l, e);
            let formula = eckart_ln_probability(e - v1, v1, v2, omega).exp();
            assert!(
                (formula - exact).abs() < 1e-6,
                "E = {e}: {formula} against {exact}"
            );
            // The same barrier seen from the other side.
            let back = eckart_ln_probability(e - v1, v2, v1, omega).exp();
            assert!((back - formula).abs() < 1e-14);
        }
    }

    #[test]
    fn skodje_truhlar_is_continuous_and_tends_to_wigner() {
        let (v, h) = (30.0, 1.0);
        let alpha = 2.0 * PI / h;
        let at = |beta: f64| skodje_truhlar_ln_kappa(v, h, 1.0 / beta).exp();
        let mid = at(alpha);
        for s in [1e-9, 1e-8, 2e-7] {
            for beta in [alpha * (1.0 - s), alpha * (1.0 + s)] {
                assert!(
                    (at(beta) / mid - 1.0).abs() < 1e-4,
                    "{s}: {} {mid}",
                    at(beta)
                );
            }
        }
        // u = h nu / kT small: 1 + u^2 / 24.
        let u = 0.1;
        let k = skodje_truhlar_ln_kappa(40.0, u, 1.0).exp();
        assert!((k - (1.0 + u * u / 24.0)).abs() < 1e-5, "{k}");
    }

    #[test]
    fn bell_has_its_limits_and_agrees_with_skodje_truhlar() {
        // High temperature: 1, for both forms.
        assert!(bell_ln_kappa(1e-3).unwrap().abs() < 1e-7);
        assert!(bell_truncated_ln_kappa(50.0, 1e-3, 1.0).abs() < 1e-6);
        // Small u: (u/2) / sin(u/2) = 1 + u^2/24 + 7 u^4/5760 + O(u^6), Wigner to O(u^4).
        for u in [0.1_f64, 0.5, 1.0] {
            let k = bell_ln_kappa(u).unwrap().exp();
            let series = 1.0 + u * u / 24.0 + 7.0 * u.powi(4) / 5760.0;
            assert!((k - series).abs() < 1e-4 * u.powi(6), "{u}: {k} {series}");
            let wigner = 1.0 + u * u / 24.0;
            assert!((k - wigner - 7.0 * u.powi(4) / 5760.0).abs() < 1e-4 * u.powi(6));
        }
        // Tall barriers: the truncated parabola is the infinite one.
        for u in [0.5_f64, 2.0, 5.0] {
            let t = bell_truncated_ln_kappa(200.0, u, 1.0);
            let i = bell_ln_kappa(u).unwrap();
            assert!((t - i).abs() < 1e-10, "{u}: {t} {i}");
        }
        // Finite and smooth across u = 2 pi, and growing with u.
        let at = |u: f64| bell_truncated_ln_kappa(30.0, u, 1.0);
        let c = 2.0 * PI;
        let (lo, mid, hi) = (at(c * (1.0 - 1e-4)), at(c), at(c * (1.0 + 1e-4)));
        assert!(lo.is_finite() && mid.is_finite() && hi.is_finite());
        assert!(lo < mid && mid < hi, "{lo} {mid} {hi}");
        assert!(((hi - mid) - (mid - lo)).abs() < 1e-3 * (hi - lo).abs());
        // At u = 2 pi exactly (r = 1): kappa = int_{-a}^0 1 / (1 + e^z) dz + 1 - I2 with the
        // closed forms a - ln 2 + ln(1 + e^-a) and 1 - (1 - ln 2) = ln 2.
        let a = 2.0 * PI * 30.0 / c;
        let want = a - 2f64.ln() + (-a).exp().ln_1p() + 2f64.ln();
        assert!(
            (mid.exp() / want - 1.0).abs() < 1e-12,
            "{} {want}",
            mid.exp()
        );
        // Skodje-Truhlar is the leading term of the truncation: below 2 pi the difference is
        // the next term of the series, r e^{-(2 - r) a} / (2 - r).
        let (r, a) = (0.8_f64, 5.0_f64);
        let h_nu = 2.0 * PI * r;
        let v = a * r;
        let bell = bell_truncated_ln_kappa(v, h_nu, 1.0).exp();
        let st = skodje_truhlar_ln_kappa(v, h_nu, 1.0).exp();
        let next = r * (-(2.0 - r) * a).exp() / (2.0 - r);
        let third = r * (-(3.0 - r) * a).exp() / (3.0 - r);
        assert!((bell - st - next).abs() < 2.0 * third, "{bell} {st} {next}");
        // Deep tunnelling (u > 2 pi): the same leading behaviour.
        for (r, a) in [(1.5_f64, 30.0_f64), (3.0, 20.0)] {
            let h_nu = 2.0 * PI * r;
            let v = a * r;
            let bell = bell_truncated_ln_kappa(v, h_nu, 1.0);
            let st = skodje_truhlar_ln_kappa(v, h_nu, 1.0);
            assert!((bell - st).abs() < 1e-5, "{r}: {bell} {st}");
        }
    }

    #[test]
    fn bell_is_the_same_from_both_sides_and_switches_near_two_pi() {
        let mut b = Bottleneck::saddle(0, "TS0", [0, 1], 0.1);
        b.imaginary_wavenumber = Some(-1500.0);
        let net = KineticNetwork::default();
        let cold = Conditions {
            temperature: 200.0,
            ..Conditions::default()
        };
        let step = |barriers: [f64; 2]| StepContext {
            barriers,
            zero_point: false,
        };
        let bell = Bell::default();
        let f = |s: &StepContext, c: &Conditions| bell.ln_factor(&b, s, &net, c).unwrap().unwrap();
        // The barrier from the higher minimum: the same with the sides swapped.
        let (x, dx) = f(&step([0.03, 0.02]), &cold);
        let (y, _) = f(&step([0.02, 0.03]), &cold);
        assert!((x - y).abs() < 1e-14);
        // u = h nu / kT = 10.8 at 1500 cm^-1 and 200 K: past 2 pi, truncated.
        let flag = |d: &[Diagnostic]| {
            d.iter()
                .find(|x| x.name == "bell_truncated")
                .map(|x| x.value)
        };
        assert_eq!(flag(&dx), Some(1.0));
        let u = quantum_of(1500.0) / cold.kt();
        assert!((x - bell_truncated_ln_kappa(0.02, quantum_of(1500.0), cold.kt())).abs() < 1e-14);
        assert!(u > 2.0 * PI);
        // The infinite parabola is refused there.
        let off = Bell {
            settings: BellSettings {
                truncation: BellTruncation::Off,
                ..BellSettings::default()
            },
        };
        assert!(off.ln_factor(&b, &step([0.03, 0.02]), &net, &cold).is_err());
        // At 600 K (u = 3.6) with tall barriers the infinite parabola.
        let warm = Conditions {
            temperature: 600.0,
            ..cold
        };
        let (z, dz) = f(&step([0.09, 0.08]), &warm);
        assert_eq!(flag(&dz), Some(0.0));
        let want = bell_ln_kappa(quantum_of(1500.0) / warm.kt()).unwrap();
        assert!((z - want).abs() < 1e-14);
        // ... and the truncated one with a low barrier, near the infinite one.
        let (w, dw) = f(&step([0.02, 0.025]), &warm);
        assert_eq!(flag(&dw), Some(1.0));
        assert!(w < want && w > want - 0.1, "{w} {want}");
        // The forward and the reverse barrier as the cut.
        for (choice, v) in [(BellBarrier::Forward, 0.03), (BellBarrier::Reverse, 0.02)] {
            let one = Bell {
                settings: BellSettings {
                    truncation: BellTruncation::On,
                    barrier: choice,
                },
            };
            let (k, _) = one
                .ln_factor(&b, &step([0.03, 0.02]), &net, &cold)
                .unwrap()
                .unwrap();
            let want = bell_truncated_ln_kappa(v, quantum_of(1500.0), cold.kt());
            assert!((k - want).abs() < 1e-14);
        }
        assert_eq!(BellTruncation::parse("auto").unwrap(), BellTruncation::Auto);
        assert_eq!(BellBarrier::parse("both").unwrap(), BellBarrier::Higher);
        assert!(BellBarrier::parse("side").is_err());
    }
}
