//! Spectra of time correlation functions: the one-sided cosine transform of a correlation
//! with a window and zero padding, by the fast Fourier transform (and by the direct sum for
//! checking); the vibrational density of states, the infrared spectrum from the dipole
//! autocorrelation with a quantum correction factor, and the power spectrum of any series.
//!
//! For a correlation `C(k dt)`, `k = 0..L`, windowed by `w` (one-sided, `w(0) = 1`,
//! `w(L dt) = 0`):
//!
//! ```text
//! S(nu) = 4 c int_0^tau_max C(tau) w(tau) cos(2 pi c nu tau) d tau      (nu in cm^-1)
//!       = 2 c dt Re FFT[a]_m,   a = (w C)_0, (w C)_1, ..., 0, ..., (w C)_1   (length P)
//! ```
//!
//! at `nu_m = m / (P dt c)`, with `P` the power of two at least `padding` times `2 L`
//! (padding only refines the grid; the resolution is that of `tau_max`). `S` integrates to
//! `C(0)` over `nu >= 0`. Windows: Hann, `cos^2(pi x / 2)`, and Blackman,
//! `0.42 + 0.5 cos(pi x) + 0.08 cos(2 pi x)`, of `x = tau / tau_max` (the halves of the
//! symmetric windows); Blackman has lower side lobes and a wider peak.
//!
//! The infrared absorption from the classical dipole autocorrelation (`S_mu`) is
//! `alpha(omega) ~ omega (1 - exp(-beta hbar omega)) Q(omega) S_mu(omega)` with a quantum
//! correction `Q` (Ramirez, Lopez-Ciudad, Kumar, and Marx, J. Chem. Phys. 2004, 121, 3973):
//! harmonic `Q = beta hbar omega / (1 - exp(-beta hbar omega))` (`alpha ~ omega^2 S_mu`,
//! also the classical limit), standard `Q = 2 / (1 + exp(-beta hbar omega))`, and Schofield
//! `Q = exp(beta hbar omega / 2)`. The spectra are given in arbitrary units, largest value 1.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::{autocorrelation, centered};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::math::fft::fft;

/// Speed of light in cm per fs.
pub const LIGHT_CM_PER_FS: f64 = Units::SPEED_OF_LIGHT_CM_S * 1e-15;

/// Default zero padding (factor on the length of the transform).
pub const DEFAULT_PADDING: usize = 4;

/// A window of a correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// None (a rectangle).
    None,
    /// Hann.
    Hann,
    /// Blackman.
    Blackman,
}

impl Window {
    /// The window called `name`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "none" | "rectangle" => Ok(Self::None),
            "hann" | "hanning" => Ok(Self::Hann),
            "blackman" => Ok(Self::Blackman),
            other => Err(MoptError::config(format!(
                "unknown window '{other}' (hann, blackman, none)"
            ))),
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Hann => "hann",
            Self::Blackman => "blackman",
        }
    }

    /// The window at `x = tau / tau_max` in `[0, 1]`.
    pub fn at(self, x: f64) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Hann => libm::cos(0.5 * PI * x).powi(2),
            Self::Blackman => 0.42 + 0.5 * libm::cos(PI * x) + 0.08 * libm::cos(2.0 * PI * x),
        }
    }
}

/// A spectrum on a grid of wavenumbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    /// Wavenumbers (cm^-1).
    pub wavenumbers: Vec<f64>,
    /// Values.
    pub values: Vec<f64>,
    /// The resolution, about the inverse of the longest lag (cm^-1).
    pub resolution: f64,
    /// Nyquist wavenumber of the series (cm^-1).
    pub nyquist: f64,
}

fn windowed(c: &[f64], window: Window) -> Vec<f64> {
    let last = c.len().saturating_sub(1).max(1) as f64;
    c.iter()
        .enumerate()
        .map(|(k, v)| v * window.at(k as f64 / last))
        .collect()
}

/// The cosine transform of the correlation `c` (lags `dt_fs` apart) up to `max_wavenumber`
/// (the Nyquist wavenumber if `None`), by the fast Fourier transform.
pub fn cosine_transform(
    c: &[f64],
    dt_fs: f64,
    window: Window,
    padding: usize,
    max_wavenumber: Option<f64>,
) -> MoptResult<Spectrum> {
    let l = c.len();
    if l < 2 || dt_fs.is_nan() || dt_fs <= 0.0 {
        return Err(MoptError::config(
            "a spectrum needs at least two lags a positive time apart",
        ));
    }
    let a = windowed(c, window);
    let size = (2 * l * padding.max(1)).next_power_of_two();
    let mut re = vec![0.0; size];
    let mut im = vec![0.0; size];
    re[0] = a[0];
    for k in 1..l {
        re[k] = a[k];
        re[size - k] = a[k];
    }
    fft(&mut re, &mut im);
    let step = 1.0 / (size as f64 * dt_fs * LIGHT_CM_PER_FS);
    let nyquist = 1.0 / (2.0 * dt_fs * LIGHT_CM_PER_FS);
    let top = max_wavenumber.unwrap_or(nyquist).min(nyquist);
    let count = ((top / step).floor() as usize + 1).min(size / 2 + 1);
    Ok(Spectrum {
        wavenumbers: (0..count).map(|m| m as f64 * step).collect(),
        values: (0..count)
            .map(|m| 2.0 * LIGHT_CM_PER_FS * dt_fs * re[m])
            .collect(),
        resolution: 1.0 / (LIGHT_CM_PER_FS * dt_fs * (l - 1) as f64),
        nyquist,
    })
}

/// [`cosine_transform`] by the direct sum at the given wavenumbers, for checking.
pub fn cosine_transform_direct(
    c: &[f64],
    dt_fs: f64,
    window: Window,
    wavenumbers: &[f64],
) -> Vec<f64> {
    let a = windowed(c, window);
    wavenumbers
        .iter()
        .map(|nu| {
            let w = 2.0 * PI * LIGHT_CM_PER_FS * nu * dt_fs;
            let sum: f64 = a
                .iter()
                .enumerate()
                .map(|(k, v)| {
                    if k == 0 {
                        0.5 * v
                    } else {
                        v * libm::cos(w * k as f64)
                    }
                })
                .sum();
            4.0 * LIGHT_CM_PER_FS * dt_fs * sum
        })
        .collect()
}

/// The power spectrum of the series `x` (samples `dt_fs` apart, mean removed): the cosine
/// transform of its autocorrelation up to `share` of its length (Wiener-Khinchin).
pub fn power_spectrum(
    x: &[f64],
    dt_fs: f64,
    share: f64,
    window: Window,
    padding: usize,
) -> MoptResult<Spectrum> {
    let lags = ((x.len() as f64 * share.clamp(0.0, 1.0)) as usize).max(2);
    let c = autocorrelation(&centered(x), lags);
    cosine_transform(&c, dt_fs, window, padding, None)
}

/// A quantum correction of a classical infrared spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumCorrection {
    /// The dipole spectrum as it is (no prefactor).
    None,
    /// Harmonic (`alpha ~ omega^2 S`).
    Harmonic,
    /// Standard.
    Standard,
    /// Schofield.
    Schofield,
}

impl QuantumCorrection {
    /// The correction called `name`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "harmonic" => Ok(Self::Harmonic),
            "standard" => Ok(Self::Standard),
            "schofield" => Ok(Self::Schofield),
            other => Err(MoptError::config(format!(
                "unknown quantum correction '{other}' (harmonic, standard, schofield, none)"
            ))),
        }
    }

    /// The factor that turns the classical dipole spectrum at `nu` (cm^-1) and
    /// `temperature` (K) into an absorption (arbitrary units: `omega` in cm^-1).
    pub fn factor(self, nu: f64, temperature: f64) -> f64 {
        // beta hbar omega = h c nu / k T = 1.4388 cm K * nu / T.
        let x = Units::HC_OVER_KB_CM_K * nu / temperature.max(1e-300);
        match self {
            Self::None => 1.0,
            Self::Harmonic => nu * nu,
            // omega (1 - e^-x) Q / (beta hbar) in units of cm^-1: nu / x = T / 1.4388.
            Self::Standard => 2.0 * nu * libm::tanh(0.5 * x) * nu / x.max(1e-300),
            Self::Schofield => 2.0 * nu * libm::sinh(0.5 * x.min(700.0)) * nu / x.max(1e-300),
        }
    }
}

/// The infrared spectrum from the autocorrelation of the dipole (`c`, mean removed) at
/// `temperature`, with `correction`, scaled to a largest value of 1.
pub fn infrared(
    c: &[f64],
    dt_fs: f64,
    temperature: f64,
    correction: QuantumCorrection,
    window: Window,
    padding: usize,
    max_wavenumber: Option<f64>,
) -> MoptResult<Spectrum> {
    let mut s = cosine_transform(c, dt_fs, window, padding, max_wavenumber)?;
    for (v, nu) in s.values.iter_mut().zip(&s.wavenumbers) {
        *v *= correction.factor(*nu, temperature);
    }
    let top = s.values.iter().copied().fold(0.0_f64, f64::max);
    if top > 0.0 {
        for v in &mut s.values {
            *v /= top;
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_transform_is_the_direct_sum_and_integrates_to_c0() {
        let nu0 = 1200.0;
        let dt = 0.5;
        let c: Vec<f64> = (0..2000)
            .map(|k| 2.0 * libm::cos(2.0 * PI * LIGHT_CM_PER_FS * nu0 * dt * k as f64))
            .collect();
        for window in [Window::Hann, Window::Blackman] {
            let s = match cosine_transform(&c, dt, window, 4, Some(4000.0)) {
                Ok(s) => s,
                Err(e) => panic!("{e}"),
            };
            let direct = cosine_transform_direct(&c, dt, window, &s.wavenumbers[..400]);
            for (a, b) in s.values.iter().zip(&direct) {
                assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{a} {b}");
            }
            let k = s
                .values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(k, _)| k);
            assert!((s.wavenumbers[k] - nu0).abs() <= 2.0 * (s.wavenumbers[1] - s.wavenumbers[0]));
            let full = match cosine_transform(&c, dt, window, 4, None) {
                Ok(s) => s,
                Err(e) => panic!("{e}"),
            };
            let step = full.wavenumbers[1] - full.wavenumbers[0];
            let area: f64 = full.values.iter().sum::<f64>() * step;
            assert!((area / 2.0 - 1.0).abs() < 0.01, "{area}");
        }
    }

    #[test]
    fn the_corrections_agree_at_low_frequency_and_part_at_high() {
        // Below k T / h c (208 cm^-1 at 300 K) all corrections tend to omega^2.
        let t = 300.0;
        for q in [QuantumCorrection::Standard, QuantumCorrection::Schofield] {
            assert!(
                (q.factor(1.0, t) / QuantumCorrection::Harmonic.factor(1.0, t) - 1.0).abs() < 1e-2
            );
        }
        let h = QuantumCorrection::Harmonic.factor(3000.0, t);
        assert!(QuantumCorrection::Standard.factor(3000.0, t) < h);
        assert!(QuantumCorrection::Schofield.factor(3000.0, t) > h);
    }
}
