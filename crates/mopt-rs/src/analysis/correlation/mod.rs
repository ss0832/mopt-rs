//! Time correlation functions, their spectra, and the transport coefficients that come
//! from them.
//!
//! The autocorrelation `C(m dt) = <x(t) x(t + m dt)>` of a series is averaged over all
//! `n - m` time origins (the unbiased estimator). [`autocorrelation`] computes it with the
//! fast Fourier transform of `math::fft` (the series padded with zeros to a power of two of
//! at least twice its length, so that the circular correlation is the linear one) and
//! [`autocorrelation_direct`] with the double sum, for checking. Vector series (velocities,
//! dipoles, fluxes) are correlated component by component and summed (`a(0) . a(t)`).
//! [`orientational`] gives the Legendre correlation `<P_l(u(0) . u(t))>` of unit vectors
//! (bonds), averaged over the vectors and the origins.
//!
//! [`spectrum`] has the cosine transforms of correlations (vibrational density of states,
//! infrared spectra with quantum corrections, power spectra of any series), [`transport`]
//! the Green-Kubo integrals with their errors and plateaus (diffusion, thermal conductivity,
//! shear viscosity) and the reverse non-equilibrium thermal conductivity of Muller-Plathe.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod spectrum;
pub mod transport;

use crate::error::{MoptError, MoptResult};
use crate::math::fft::autocorrelation_sums;

/// The autocorrelation of `x` at lags `0..lags` (at most `x.len()`), averaged over the
/// time origins, by the fast Fourier transform.
pub fn autocorrelation(x: &[f64], lags: usize) -> Vec<f64> {
    let n = x.len();
    let lags = lags.min(n);
    autocorrelation_sums(x)
        .into_iter()
        .take(lags)
        .enumerate()
        .map(|(m, c)| c / (n - m) as f64)
        .collect()
}

/// [`autocorrelation`] by the double sum (`O(n lags)`), for checking.
pub fn autocorrelation_direct(x: &[f64], lags: usize) -> Vec<f64> {
    let n = x.len();
    (0..lags.min(n))
        .map(|m| (0..n - m).map(|k| x[k] * x[k + m]).sum::<f64>() / (n - m) as f64)
        .collect()
}

/// `x` without its mean.
pub fn centered(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| v - mean).collect()
}

/// The autocorrelation `<a(0) . a(t)>` of a series of vectors (`series[t]`, all of the same
/// length), summed over the components.
pub fn vector_autocorrelation(series: &[Vec<f64>], lags: usize) -> MoptResult<Vec<f64>> {
    let Some(first) = series.first() else {
        return Ok(Vec::new());
    };
    let d = first.len();
    if series.iter().any(|v| v.len() != d) {
        return Err(MoptError::config(
            "the vectors of a series must have the same length",
        ));
    }
    let lags = lags.min(series.len());
    let mut out = vec![0.0; lags];
    for c in 0..d {
        let x: Vec<f64> = series.iter().map(|v| v[c]).collect();
        for (o, v) in out.iter_mut().zip(autocorrelation(&x, lags)) {
            *o += v;
        }
    }
    Ok(out)
}

/// Legendre polynomial `P_l(x)` for `l` = 1 or 2.
fn legendre(l: usize, x: f64) -> f64 {
    if l == 1 {
        x
    } else {
        1.5 * x * x - 0.5
    }
}

/// `<P_l(u(0) . u(t))>` of unit vectors (`vectors[t][k]`: vector `k` at time `t`; they are
/// normalized here), averaged over the vectors and the origins, `l` = 1 or 2. `P_2` uses
/// `(u . u')^2 = sum_ab u_a u_b u'_a u'_b`, a sum of autocorrelations of the products.
pub fn orientational(vectors: &[Vec<[f64; 3]>], l: usize, lags: usize) -> MoptResult<Vec<f64>> {
    if l != 1 && l != 2 {
        return Err(MoptError::config(
            "orientational correlations of order 1 or 2",
        ));
    }
    let Some(first) = vectors.first() else {
        return Ok(Vec::new());
    };
    let k = first.len();
    if k == 0 || vectors.iter().any(|v| v.len() != k) {
        return Err(MoptError::config(
            "every frame needs the same vectors, at least one",
        ));
    }
    let unit = |u: [f64; 3]| -> MoptResult<[f64; 3]> {
        let n = libm::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        if n > 0.0 {
            Ok(u.map(|c| c / n))
        } else {
            Err(MoptError::config(
                "a vector of zero length has no orientation",
            ))
        }
    };
    let lags = lags.min(vectors.len());
    let mut out = vec![0.0; lags];
    for j in 0..k {
        let u: Vec<[f64; 3]> = vectors
            .iter()
            .map(|v| unit(v[j]))
            .collect::<MoptResult<_>>()?;
        let mut dot = vec![0.0; lags];
        if l == 1 {
            for c in 0..3 {
                let x: Vec<f64> = u.iter().map(|v| v[c]).collect();
                for (d, v) in dot.iter_mut().zip(autocorrelation(&x, lags)) {
                    *d += v;
                }
            }
            for (o, d) in out.iter_mut().zip(&dot) {
                *o += legendre(1, *d) / k as f64;
            }
        } else {
            for a in 0..3 {
                for b in 0..3 {
                    let x: Vec<f64> = u.iter().map(|v| v[a] * v[b]).collect();
                    for (d, v) in dot.iter_mut().zip(autocorrelation(&x, lags)) {
                        *d += v;
                    }
                }
            }
            // <P_2> = 3/2 <(u . u')^2> - 1/2.
            for (o, d) in out.iter_mut().zip(&dot) {
                *o += (1.5 * d - 0.5) / k as f64;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_estimator_is_the_direct_one() {
        let x: Vec<f64> = (0..300)
            .map(|k| libm::sin(0.13 * k as f64) + 0.3 * libm::cos(0.71 * k as f64 + 0.2))
            .collect();
        let fast = autocorrelation(&x, 120);
        let direct = autocorrelation_direct(&x, 120);
        for (a, b) in fast.iter().zip(&direct) {
            assert!((a - b).abs() < 1e-10, "{a} {b}");
        }
        let v: Vec<Vec<f64>> = x.iter().map(|a| vec![*a, 2.0 * a]).collect();
        let c = match vector_autocorrelation(&v, 5) {
            Ok(c) => c,
            Err(e) => panic!("{e}"),
        };
        assert!((c[3] - 5.0 * direct[3]).abs() < 1e-9);
    }

    #[test]
    fn a_vector_turning_at_a_constant_rate_has_cosine_correlations() {
        // u(t) = (cos wt, sin wt, 0): P1 = cos(w tau), P2 = (3 cos^2 - 1) / 2.
        let w = 0.05;
        let frames: Vec<Vec<[f64; 3]>> = (0..400)
            .map(|t| {
                let a = w * t as f64;
                vec![[libm::cos(a), libm::sin(a), 0.0]]
            })
            .collect();
        let p1 = match orientational(&frames, 1, 50) {
            Ok(c) => c,
            Err(e) => panic!("{e}"),
        };
        let p2 = match orientational(&frames, 2, 50) {
            Ok(c) => c,
            Err(e) => panic!("{e}"),
        };
        for tau in [0usize, 10, 33] {
            let c = libm::cos(w * tau as f64);
            assert!((p1[tau] - c).abs() < 1e-9);
            assert!((p2[tau] - (1.5 * c * c - 0.5)).abs() < 1e-9);
        }
    }
}
