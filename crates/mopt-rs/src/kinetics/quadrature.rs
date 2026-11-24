//! Gauss quadrature rules for the thermal averages of the rate corrections: Gauss-Legendre on
//! an interval and Gauss-Laguerre on `[0, inf)` with the weight `exp(-x)`. The nodes are the
//! roots of the orthogonal polynomials found by Newton iterations from the usual asymptotic
//! first guesses (Press et al., Numerical Recipes, section 4.6).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

/// Nodes and weights of the `n`-point Gauss-Legendre rule on `[-1, 1]`.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    for i in 1..=n.div_ceil(2) {
        let mut z = (PI * (i as f64 - 0.25) / (nf + 0.5)).cos();
        let mut pp = 1.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf - 1.0) * z * p2 - (jf - 1.0) * p3) / jf;
            }
            pp = nf * (z * p1 - p2) / (z * z - 1.0);
            let step = p1 / pp;
            z -= step;
            if step.abs() <= 1e-15 {
                break;
            }
        }
        x[i - 1] = -z;
        x[n - i] = z;
        let weight = 2.0 / ((1.0 - z * z) * pp * pp);
        w[i - 1] = weight;
        w[n - i] = weight;
    }
    (x, w)
}

/// Nodes and weights of the `n`-point Gauss-Laguerre rule: `int_0^inf f(x) exp(-x) dx`.
pub fn gauss_laguerre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    let mut z = 0.0_f64;
    for i in 1..=n {
        if i == 1 {
            z = 3.0 / (1.0 + 2.4 * nf);
        } else if i == 2 {
            z += 15.0 / (1.0 + 2.5 * nf);
        } else {
            let ai = (i - 2) as f64;
            z += (1.0 + 2.55 * ai) / (1.9 * ai) * (z - x[i - 3]);
        }
        let (mut pp, mut p2) = (1.0, 0.0);
        for _ in 0..200 {
            let mut p1 = 1.0;
            p2 = 0.0;
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf - 1.0 - z) * p2 - (jf - 1.0) * p3) / jf;
            }
            pp = (nf * p1 - nf * p2) / z;
            let step = p1 / pp;
            z -= step;
            if step.abs() <= 1e-14 * z.abs().max(1.0) {
                break;
            }
        }
        x[i - 1] = z;
        w[i - 1] = -1.0 / (pp * nf * p2);
    }
    (x, w)
}

/// `int_a^b f(x) dx` by `panels` panels of the `n`-point Gauss-Legendre rule, with the
/// logarithm of the integrand given (`ln_f`), summed in log space: the result is the
/// logarithm of the integral (`-inf` for an empty interval).
pub fn ln_integral(ln_f: impl Fn(f64) -> f64, a: f64, b: f64, panels: usize, n: usize) -> f64 {
    if a.is_nan() || b.is_nan() || b <= a || panels == 0 || n == 0 {
        return f64::NEG_INFINITY;
    }
    let (x, w) = gauss_legendre(n);
    let h = (b - a) / panels as f64;
    let mut terms = Vec::with_capacity(panels * n);
    for k in 0..panels {
        let mid = a + (k as f64 + 0.5) * h;
        for (xi, wi) in x.iter().zip(&w) {
            terms.push(ln_f(mid + 0.5 * h * xi) + (0.5 * h * wi).ln());
        }
    }
    crate::math::stable::log_sum_exp(&terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rules_integrate_polynomials_exactly() {
        for n in [4, 16, 64] {
            let (x, w) = gauss_legendre(n);
            assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-13);
            let x4: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(4)).sum();
            assert!((x4 - 0.4).abs() < 1e-13, "{n}: {x4}");
            let (x, w) = gauss_laguerre(n);
            // int x^k exp(-x) dx = k!
            for (k, fact) in [(0, 1.0), (1, 1.0), (2, 2.0), (3, 6.0), (5, 120.0)] {
                let s: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(k)).sum();
                assert!((s - fact).abs() < 1e-11 * fact, "{n} {k}: {s}");
            }
        }
        // ln of int_0^2 exp(3 x) dx = ln((e^6 - 1) / 3).
        let got = ln_integral(|x| 3.0 * x, 0.0, 2.0, 4, 16);
        assert!((got - ((6.0_f64.exp() - 1.0) / 3.0).ln()).abs() < 1e-13);
    }
}
