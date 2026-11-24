//! Time correlations of a trajectory: the mean squared displacement with the diffusion
//! coefficient, and the velocity autocorrelation function with the vibrational density of
//! states.
//!
//! The mean squared displacement `MSD(tau) = <|r_i(t + tau) - r_i(t)|^2>` is averaged over
//! all time origins `t` and over the atoms of each element, from the positions with the
//! jumps through the faces of the cell taken out, by the fast Fourier transform (Kneller,
//! Keiner, Kneller, and Schiller, Comput. Phys. Commun. 1995, 91, 191; nMoldyn).
//! The drift of the centre of mass of all atoms is subtracted by default (a Langevin or
//! Andersen thermostat lets it wander). The diffusion coefficient is `D = slope / 6` of a
//! straight line fitted to the MSD from 10 % to 90 % of the longest lag, the defaults of
//! GROMACS (`gmx msd -beginfit -endfit`).
//!
//! The velocity autocorrelation `C(tau) = <sum_i m_i v_i(t) . v_i(t + tau)>` (mass-weighted,
//! so that `C(0) = 2 <K>`) is averaged over the origins up to half the length of the run by
//! default (each lag then has at least half of the origins). The vibrational density of
//! states is its cosine transform with a window (Hann `w(tau) = cos^2(pi tau / (2 tau_max))`
//! by default, or Blackman) and zero padding (4 times by default, which only refines the
//! grid), `g(nu) = 3 N 4 c int_0^tau_max C(tau) / C(0) w(tau) cos(2 pi c nu tau) d tau`
//! (`nu` in cm^-1, `c` the speed of light; `analysis::correlation::spectrum`), which
//! integrates to `3 N` over `nu >= 0`; its peaks are the frequencies of the vibrations.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use rayon::prelude::*;

use super::{MdTrajectory, B2A};
use crate::analysis::correlation::spectrum::{cosine_transform, Spectrum, Window, DEFAULT_PADDING};
use crate::domain::{Element, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::fft::autocorrelation;

/// Default share of the longest lag where the fit of the diffusion coefficient starts.
pub const DEFAULT_FIT_START: f64 = 0.1;
/// Default share of the longest lag where it ends.
pub const DEFAULT_FIT_END: f64 = 0.9;
/// Default share of the run that the velocity autocorrelation reaches.
pub const DEFAULT_CORRELATION_SHARE: f64 = 0.5;

/// Settings of [`mean_squared_displacement`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsdSettings {
    /// Shares of the longest lag between which the straight line is fitted.
    pub fit: (f64, f64),
    /// Whether the drift of the centre of mass is subtracted.
    pub remove_drift: bool,
}

impl Default for MsdSettings {
    fn default() -> Self {
        Self {
            fit: (DEFAULT_FIT_START, DEFAULT_FIT_END),
            remove_drift: true,
        }
    }
}

/// A diffusion coefficient from the slope of a mean squared displacement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diffusion {
    /// `D` in Angstrom^2 per ps.
    pub coefficient: f64,
    /// The lags of the fit (fs).
    pub window: (f64, f64),
    /// Intercept of the line (Angstrom^2).
    pub intercept: f64,
    /// Coefficient of determination of the fit.
    pub r_squared: f64,
}

impl Diffusion {
    /// `D` in cm^2 per s.
    pub fn cm2_per_s(&self) -> f64 {
        self.coefficient * 1e-4
    }
}

/// Mean squared displacements.
#[derive(Debug, Clone, PartialEq)]
pub struct Msd {
    /// Lags (fs).
    pub lags: Vec<f64>,
    /// The elements, in the order of first appearance.
    pub species: Vec<Element>,
    /// MSD of each element (Angstrom^2), `[element][lag]`.
    pub by_species: Vec<Vec<f64>>,
    /// MSD of all atoms (Angstrom^2).
    pub total: Vec<f64>,
    /// Diffusion coefficient of each element.
    pub diffusion: Vec<Diffusion>,
    /// Diffusion coefficient of all atoms.
    pub total_diffusion: Diffusion,
}

/// MSD of one coordinate series at every lag, by the fast Fourier transform:
/// `MSD(m) = S1(m) - 2 S2(m)`.
fn msd_series(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let s2 = autocorrelation(x);
    let d: Vec<f64> = x.iter().map(|v| v * v).collect();
    let mut q = 2.0 * d.iter().sum::<f64>();
    let mut out = Vec::with_capacity(n);
    for m in 0..n {
        if m > 0 {
            q -= d[m - 1] + d[n - m];
        }
        out.push((q / (n - m) as f64 - 2.0 * s2[m]).max(0.0));
    }
    out
}

/// A straight line through `(xs, ys)` from `x >= lo` to `x <= hi`.
fn fit_line(xs: &[f64], ys: &[f64], (lo, hi): (f64, f64)) -> MoptResult<(f64, f64, f64)> {
    let points: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .filter(|(x, _)| **x >= lo - 1e-9 && **x <= hi + 1e-9)
        .map(|(x, y)| (*x, *y))
        .collect();
    if points.len() < 2 {
        return Err(MoptError::config(
            "the fit of the mean squared displacement needs at least two lags",
        ));
    }
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let syy: f64 = points.iter().map(|p| (p.1 - my).powi(2)).sum();
    if sxx <= 0.0 {
        return Err(MoptError::config("the lags of the fit are all the same"));
    }
    let slope = sxy / sxx;
    let r2 = if syy > 0.0 {
        (sxy * sxy) / (sxx * syy)
    } else {
        1.0
    };
    Ok((slope, my - slope * mx, r2))
}

/// The mean squared displacements of `traj` (evenly spaced frames, at least three).
pub fn mean_squared_displacement(traj: &MdTrajectory, s: &MsdSettings) -> MoptResult<Msd> {
    let dt = traj.interval()?;
    let frames = traj.len();
    if frames < 3 {
        return Err(MoptError::config(
            "a mean squared displacement needs at least three frames",
        ));
    }
    let (a, b) = s.fit;
    if !(0.0..1.0).contains(&a) || !(a < b && b <= 1.0) {
        return Err(MoptError::config(format!(
            "the fit must lie within the lags: 0 <= start < end <= 1, not {a} and {b}"
        )));
    }
    let u = if s.remove_drift {
        traj.unwrapped_without_drift()
    } else {
        traj.unwrapped()
    };
    let n = traj.n_atoms();
    let per_atom: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut sum = vec![0.0; frames];
            for c in 0..3 {
                let x: Vec<f64> = u.iter().map(|row| row[i][c] * B2A).collect();
                for (s, v) in sum.iter_mut().zip(msd_series(&x)) {
                    *s += v;
                }
            }
            sum
        })
        .collect();
    let lags: Vec<f64> = (0..frames).map(|m| m as f64 * dt).collect();
    let longest = lags[frames - 1];
    let window = (a * longest, b * longest);
    let diffusion_of = |msd: &[f64]| -> MoptResult<Diffusion> {
        let (slope, intercept, r_squared) = fit_line(&lags, msd, window)?;
        Ok(Diffusion {
            coefficient: slope / 6.0 * 1000.0,
            window,
            intercept,
            r_squared,
        })
    };
    let species = traj.species();
    let mut by_species = Vec::new();
    let mut diffusion = Vec::new();
    for (_, atoms) in &species {
        let mean: Vec<f64> = (0..frames)
            .map(|m| atoms.iter().map(|&i| per_atom[i][m]).sum::<f64>() / atoms.len() as f64)
            .collect();
        diffusion.push(diffusion_of(&mean)?);
        by_species.push(mean);
    }
    let total: Vec<f64> = (0..frames)
        .map(|m| per_atom.iter().map(|v| v[m]).sum::<f64>() / n as f64)
        .collect();
    Ok(Msd {
        total_diffusion: diffusion_of(&total)?,
        lags,
        species: species.iter().map(|(e, _)| *e).collect(),
        by_species,
        total,
        diffusion,
    })
}

/// Velocity autocorrelation functions.
#[derive(Debug, Clone, PartialEq)]
pub struct Vacf {
    /// Lags (fs).
    pub lags: Vec<f64>,
    /// The elements, in the order of first appearance.
    pub species: Vec<Element>,
    /// The part of each element, divided by the `C(0)` of all atoms, `[element][lag]`.
    pub by_species: Vec<Vec<f64>>,
    /// `C(tau) / C(0)` of all atoms.
    pub total: Vec<f64>,
    /// `C(0) = <sum_i m_i v_i^2>` (hartree, twice the mean kinetic energy).
    pub c0: f64,
    /// Number of atoms.
    pub atoms: usize,
}

/// The mass-weighted velocity autocorrelation of `traj` up to `share` of its length
/// (velocities needed; evenly spaced frames).
pub fn velocity_autocorrelation(traj: &MdTrajectory, share: f64) -> MoptResult<Vacf> {
    let dt = traj.interval()?;
    let v = traj.velocities().ok_or_else(|| {
        MoptError::config("the velocity autocorrelation needs the velocities of the frames")
    })?;
    if !(share > 0.0 && share <= 1.0) {
        return Err(MoptError::config(format!(
            "the share of the run must be in (0, 1], not {share}"
        )));
    }
    let frames = traj.len();
    let lags = ((frames as f64 * share).floor() as usize).clamp(2, frames);
    let masses: Vec<f64> = traj
        .masses()
        .iter()
        .map(|m| m * Units::AMU_TO_AU_MASS)
        .collect();
    let n = traj.n_atoms();
    let per_atom: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut sum = vec![0.0; lags];
            for c in 0..3 {
                let x: Vec<f64> = v.iter().map(|row| row[i][c]).collect();
                for (s, value) in sum.iter_mut().zip(autocorrelation(&x)) {
                    *s += masses[i] * value;
                }
            }
            sum
        })
        .collect();
    let total: Vec<f64> = (0..lags)
        .map(|m| per_atom.iter().map(|c| c[m]).sum())
        .collect();
    let c0 = total[0];
    if !c0.is_finite() || c0 <= 0.0 {
        return Err(MoptError::config(
            "the atoms do not move: all velocities are zero",
        ));
    }
    let species = traj.species();
    let by_species = species
        .iter()
        .map(|(_, atoms)| {
            (0..lags)
                .map(|m| atoms.iter().map(|&i| per_atom[i][m]).sum::<f64>() / c0)
                .collect()
        })
        .collect();
    Ok(Vacf {
        lags: (0..lags).map(|m| m as f64 * dt).collect(),
        species: species.iter().map(|(e, _)| *e).collect(),
        by_species,
        total: total.iter().map(|c| c / c0).collect(),
        c0,
        atoms: n,
    })
}

/// Settings of [`vibrational_density_of_states`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VdosSettings {
    /// Largest wavenumber (cm^-1); `None`: the Nyquist wavenumber of the frames.
    pub max: Option<f64>,
    /// Window of the autocorrelation.
    pub window: Window,
    /// Zero padding of the transform (factor on its length; the grid is `1 / (P dt c)`).
    pub padding: usize,
}

impl Default for VdosSettings {
    fn default() -> Self {
        Self {
            max: None,
            window: Window::Hann,
            padding: DEFAULT_PADDING,
        }
    }
}

/// Vibrational densities of states.
#[derive(Debug, Clone, PartialEq)]
pub struct Vdos {
    /// Wavenumbers (cm^-1).
    pub wavenumbers: Vec<f64>,
    /// The elements.
    pub species: Vec<Element>,
    /// The part of each element (states per cm^-1), `[element][wavenumber]`.
    pub by_species: Vec<Vec<f64>>,
    /// All atoms (states per cm^-1; integrates to `3 N`).
    pub total: Vec<f64>,
    /// The Nyquist wavenumber of the frames (cm^-1).
    pub nyquist: f64,
    /// The resolution, about the inverse of the longest lag (cm^-1).
    pub resolution: f64,
}

/// The vibrational density of states of a velocity autocorrelation: its cosine transform
/// with the window and zero padding of `s` (`analysis::correlation::spectrum`), scaled to
/// `3 N` states.
pub fn vibrational_density_of_states(vacf: &Vacf, s: &VdosSettings) -> MoptResult<Vdos> {
    let dt = match vacf.lags.get(1) {
        Some(dt) if *dt > 0.0 => *dt,
        _ => {
            return Err(MoptError::config(
                "the autocorrelation has fewer than two lags",
            ))
        }
    };
    let scale = 3.0 * vacf.atoms as f64;
    let spectrum = |c: &[f64]| -> MoptResult<Spectrum> {
        let mut sp = cosine_transform(c, dt, s.window, s.padding, s.max)?;
        for v in &mut sp.values {
            *v *= scale;
        }
        Ok(sp)
    };
    let total = spectrum(&vacf.total)?;
    let by_species = vacf
        .by_species
        .iter()
        .map(|c| spectrum(c).map(|sp| sp.values))
        .collect::<MoptResult<_>>()?;
    Ok(Vdos {
        by_species,
        species: vacf.species.clone(),
        nyquist: total.nyquist,
        resolution: total.resolution,
        wavenumbers: total.wavenumbers,
        total: total.values,
    })
}

/// The local maxima of `ys` over `xs` that reach `share` of the largest value, highest
/// first.
pub fn peaks(xs: &[f64], ys: &[f64], share: f64) -> Vec<(f64, f64)> {
    let top = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<(f64, f64)> = (1..ys.len().saturating_sub(1))
        .filter(|&k| ys[k] > ys[k - 1] && ys[k] >= ys[k + 1] && ys[k] >= share * top)
        .map(|k| (xs[k], ys[k]))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}
