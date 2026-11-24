//! Special functions that the kinetics needs and `libm` does not have: the Airy function
//! `Ai(x)` (the weak-coupling probability of a crossing) and the Laplace transform of its
//! square.
//!
//! `Ai(x)` is evaluated in three ranges (DLMF, chapter 9), to about 1e-11 relative away from
//! `x = -7` and to about 5e-9 next to it, where the cancellation of the series and the
//! truncation of the asymptotic expansion meet (measured against the values of Cephes):
//!
//! - `-7 <= x <= 2`: the Maclaurin series `Ai(x) = c1 f(x) - c2 g(x)` (DLMF 9.4.1), where the
//!   terms cancel by less than a factor `exp(2/3 |x|^(3/2))`;
//! - `x > 2`: `Ai(x) = exp(-zeta) / pi * int_0^inf exp(-sqrt(x) t^2) cos(t^3 / 3) dt`,
//!   `zeta = 2/3 x^(3/2)`, the integral of DLMF 9.5.1 with the path moved to
//!   `t + i sqrt(x)`; its integrand falls off as a Gaussian, so Gauss-Legendre panels converge
//!   fast, and the factor `exp(-zeta)` is kept apart ([`airy_ai_scaled`]) so that nothing
//!   underflows;
//! - `x < -7`: the asymptotic expansion of DLMF 9.7.9 (its smallest term is about
//!   `exp(-2 zeta)`, below 1e-10 there).
//!
//! The Laplace transform of `Ai^2` over the whole line is closed:
//! `int_-inf^inf exp(p t) Ai(t)^2 dt = exp(p^3 / 12) / (2 sqrt(pi p))` for `p > 0` (Vallee and
//! Soares, Airy Functions and Applications to Physics, 2004, section 3.5; checked here against
//! quadrature).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

/// `Ai(0)`.
const AI0: f64 = 0.355_028_053_887_817_2;
/// `-Ai'(0)`.
const AIP0: f64 = 0.258_819_403_792_806_8;

/// Gauss-Legendre nodes and weights on `[-1, 1]` (Newton on the Legendre polynomial).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

/// The Maclaurin series of DLMF 9.4.1.
fn maclaurin(x: f64) -> f64 {
    let x3 = x * x * x;
    let (mut f, mut g) = (1.0, x);
    let (mut tf, mut tg) = (1.0, x);
    for k in 0..200 {
        let k3 = 3.0 * k as f64;
        tf *= x3 / ((k3 + 2.0) * (k3 + 3.0));
        tg *= x3 / ((k3 + 3.0) * (k3 + 4.0));
        f += tf;
        g += tg;
        if tf.abs() < 1e-18 * f.abs().max(1e-300) && tg.abs() < 1e-18 * g.abs().max(1e-300) {
            break;
        }
    }
    AI0 * f - AIP0 * g
}

/// `int_0^inf exp(-sqrt(x) t^2) cos(t^3 / 3) dt / pi` for `x > 0`.
fn moved_integral(x: f64) -> f64 {
    let a = x.sqrt();
    // exp(-a t^2) < 1e-18 beyond t_max.
    let t_max = (41.5 / a).sqrt();
    // About four panels per period of cos(t^3 / 3) at the end, at least 8.
    let periods = t_max.powi(3) / 3.0 / (2.0 * PI);
    let panels = (4.0 * periods).ceil().clamp(8.0, 4000.0) as usize;
    let (nodes, weights) = gauss_legendre(16);
    let h = t_max / panels as f64;
    let mut sum = 0.0;
    for k in 0..panels {
        let mid = (k as f64 + 0.5) * h;
        for (xi, wi) in nodes.iter().zip(&weights) {
            let t = mid + 0.5 * h * xi;
            sum += 0.5 * h * wi * (-a * t * t).exp() * (t * t * t / 3.0).cos();
        }
    }
    sum / PI
}

/// The asymptotic expansion of DLMF 9.7.9 for `Ai(-z)`, `z > 0` large.
fn oscillating(z: f64) -> f64 {
    let zeta = 2.0 / 3.0 * z.powf(1.5);
    // u_k = u_{k-1} (6k-5)(6k-3)(6k-1) / ((2k-1) 216 k).
    let mut u = vec![1.0];
    for k in 1..24 {
        let kf = k as f64;
        let prev: f64 = u[k - 1];
        u.push(
            prev * (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0)
                / ((2.0 * kf - 1.0) * 216.0 * kf),
        );
    }
    let (mut even, mut odd) = (0.0, 0.0);
    let mut sign = 1.0;
    let mut last = f64::INFINITY;
    for k in 0..12 {
        let te = u[2 * k] / zeta.powi(2 * k as i32);
        let to = u[2 * k + 1] / zeta.powi(2 * k as i32 + 1);
        // Stop where the asymptotic terms start to grow.
        if te.abs() > last {
            break;
        }
        last = to.abs();
        even += sign * te;
        odd += sign * to;
        sign = -sign;
    }
    let phase = zeta - PI / 4.0;
    (phase.cos() * even + phase.sin() * odd) / (PI.sqrt() * z.powf(0.25))
}

/// The Airy function `Ai(x)`.
pub fn airy_ai(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 2.0 {
        airy_ai_scaled(x) * (-2.0 / 3.0 * x.powf(1.5)).exp()
    } else if x >= -7.0 {
        maclaurin(x)
    } else if x.is_finite() {
        oscillating(-x)
    } else {
        0.0
    }
}

/// `Ai(x) exp(2/3 x^(3/2))` for `x >= 0` (finite where `Ai` underflows), `Ai(x)` for `x < 0`.
pub fn airy_ai_scaled(x: f64) -> f64 {
    if x > 2.0 {
        moved_integral(x)
    } else if x >= 0.0 {
        maclaurin(x) * (2.0 / 3.0 * x.powf(1.5)).exp()
    } else {
        airy_ai(x)
    }
}

/// `ln(Ai(x)^2)` for `x >= 0` without underflow (`-inf` where `Ai` is 0).
pub fn ln_airy_ai_squared_positive(x: f64) -> f64 {
    let s = airy_ai_scaled(x.max(0.0));
    if s > 0.0 {
        2.0 * s.ln() - 4.0 / 3.0 * x.max(0.0).powf(1.5)
    } else {
        f64::NEG_INFINITY
    }
}

/// `ln int_-inf^inf exp(p t) Ai(t)^2 dt = p^3 / 12 - ln(2 sqrt(pi p))` for `p > 0`.
pub fn ln_laplace_airy_squared(p: f64) -> f64 {
    p * p * p / 12.0 - (2.0 * (PI * p).sqrt()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_airy_function_has_its_known_values() {
        // Reference values of scipy.special.airy (Cephes).
        for (x, want) in [
            (0.0_f64, 0.355_028_053_887_817_2_f64),
            (1.0, 0.135_292_416_312_881_4),
            (2.0, 0.034_924_130_423_274_38),
            (2.5, 0.015_725_923_380_470_43),
            (5.0, 1.083_444_281_360_744e-4),
            (10.0, 1.104_753_255_289_865e-10),
            (-1.0, 0.535_560_883_292_352_1),
            (-2.0, 0.227_407_428_201_685_6),
            (-3.0, -0.378_814_293_677_658),
            (-5.0, 0.350_761_009_024_114_2),
            (-7.0, 0.184_280_835_250_506_2),
            (-8.0, -0.052_705_050_356_386_43),
            (-10.0, 0.040_241_238_486_441_95),
        ] {
            let got = airy_ai(x);
            let tol = if (-8.5..=-6.5).contains(&x) {
                5e-9
            } else {
                1e-10
            } * want.abs();
            assert!((got - want).abs() < tol, "Ai({x}) = {got}, want {want}");
        }
        // The two methods of each switch agree there.
        let x: f64 = 2.0;
        let moved = moved_integral(x) * (-2.0 / 3.0 * x.powf(1.5)).exp();
        assert!(
            (maclaurin(x) - moved).abs() < 1e-12,
            "{} {moved}",
            maclaurin(x)
        );
        let (a, b) = (maclaurin(-7.0), oscillating(7.0));
        assert!((a - b).abs() < 2e-9, "{a} {b}");
    }

    #[test]
    fn the_laplace_transform_of_the_square_is_closed() {
        // Direct quadrature of exp(p t) Ai(t)^2 for moderate p (the oscillations below 0 are
        // damped by exp(p t)).
        for p in [0.5_f64, 1.0, 2.0] {
            let (x, w) = gauss_legendre(32);
            let (lo, hi) = (-60.0 / p, 25.0);
            let panels = 4000;
            let h = (hi - lo) / panels as f64;
            let mut sum = 0.0;
            for k in 0..panels {
                let mid = lo + (k as f64 + 0.5) * h;
                for (xi, wi) in x.iter().zip(&w) {
                    let t = mid + 0.5 * h * xi;
                    let a = airy_ai(t);
                    sum += 0.5 * h * wi * (p * t).exp() * a * a;
                }
            }
            let want = ln_laplace_airy_squared(p).exp();
            assert!((sum / want - 1.0).abs() < 1e-8, "p = {p}: {sum} {want}");
        }
        let x = 12.0;
        let a = airy_ai(x);
        assert!((ln_airy_ai_squared_positive(x) - (a * a).ln()).abs() < 1e-10);
    }
}
