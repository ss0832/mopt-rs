//! The fast Fourier transform (iterative radix-2 Cooley-Tukey) and the time correlation
//! functions computed with it: the autocorrelation of a series summed over all time origins
//! in `O(n log n)` instead of `O(n^2)` (Wiener-Khinchin: the inverse transform of the power
//! spectrum of the series padded with zeros to at least twice its length).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

/// Transform `(re, im)` in place: `X_k = sum_j x_j exp(-2 pi i j k / n)`, or with `inverse`
/// `x_j = (1/n) sum_k X_k exp(+2 pi i j k / n)`. The length must be a power of two (a
/// length of 0 or 1 is left as it is).
fn transform(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n < 2 || im.len() != n || !n.is_power_of_two() {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * 2.0 * PI / len as f64;
        let half = len / 2;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                // Twiddle factors from the angle directly (no accumulated rounding).
                let (s, c) = libm::sincos(angle * k as f64);
                let (a, b) = (start + k, start + k + half);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / n as f64;
        for (r, i) in re.iter_mut().zip(im.iter_mut()) {
            *r *= scale;
            *i *= scale;
        }
    }
}

/// The discrete Fourier transform of `(re, im)` (both of the same power-of-two length).
pub fn fft(re: &mut [f64], im: &mut [f64]) {
    transform(re, im, false);
}

/// The inverse of [`fft`].
pub fn inverse_fft(re: &mut [f64], im: &mut [f64]) {
    transform(re, im, true);
}

/// The autocorrelation of `x` summed over the time origins: `c_m = sum_{k=0}^{n-1-m}
/// x_k x_{k+m}` for `m = 0..n` (divide by `n - m` for the average over the origins).
pub fn autocorrelation_sums(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let size = (2 * n).next_power_of_two();
    let mut re = vec![0.0; size];
    let mut im = vec![0.0; size];
    re[..n].copy_from_slice(x);
    fft(&mut re, &mut im);
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r = *r * *r + *i * *i;
        *i = 0.0;
    }
    inverse_fft(&mut re, &mut im);
    re.truncate(n);
    re
}

/// The autocorrelation of `x` averaged over the time origins: `c_m / (n - m)`.
pub fn autocorrelation(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    autocorrelation_sums(x)
        .into_iter()
        .enumerate()
        .map(|(m, c)| c / (n - m) as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transform_matches_the_sum_and_inverts() {
        let x: Vec<f64> = (0..16)
            .map(|k| libm::sin(0.3 * k as f64) + 0.1 * k as f64)
            .collect();
        let mut re = x.clone();
        let mut im = vec![0.0; 16];
        fft(&mut re, &mut im);
        for k in 0..16 {
            let (mut sr, mut si) = (0.0, 0.0);
            for (j, xj) in x.iter().enumerate() {
                let a = -2.0 * PI * (j * k) as f64 / 16.0;
                sr += xj * libm::cos(a);
                si += xj * libm::sin(a);
            }
            assert!((re[k] - sr).abs() < 1e-10 && (im[k] - si).abs() < 1e-10);
        }
        inverse_fft(&mut re, &mut im);
        for (a, b) in re.iter().zip(&x) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn the_autocorrelation_matches_the_direct_sum() {
        let x: Vec<f64> = (0..37)
            .map(|k| libm::cos(0.7 * k as f64) * (k as f64).sqrt())
            .collect();
        let c = autocorrelation_sums(&x);
        for (m, cm) in c.iter().enumerate() {
            let direct: f64 = (0..x.len() - m).map(|k| x[k] * x[k + m]).sum();
            assert!(
                (cm - direct).abs() < 1e-9 * direct.abs().max(1.0),
                "lag {m}"
            );
        }
        assert!(autocorrelation(&[]).is_empty());
    }
}
