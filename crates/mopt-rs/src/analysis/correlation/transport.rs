//! Transport coefficients: Green-Kubo integrals of equilibrium fluctuations (diffusion from
//! the velocity autocorrelation, thermal conductivity from the heat current, shear viscosity
//! from the off-diagonal pressure) with block errors and a documented plateau, and the
//! thermal conductivity of a reverse non-equilibrium run (Muller-Plathe, J. Chem. Phys.
//! 1997, 106, 6082).
//!
//! A Green-Kubo coefficient is `L = A int_0^inf <X(0) X(t)> dt`, the correlation averaged
//! over the given components (the three Cartesian components of a flux, the three
//! off-diagonal elements of the pressure, the `3N` velocity components):
//!
//! | coefficient | `X` | `A` |
//! |---|---|---|
//! | self-diffusion `D` | `v_i,a` | 1 |
//! | thermal conductivity `kappa` | `J_a = sum_i (e_i v_i + W_i v_i)_a` | `1 / (V k_B T^2)` |
//! | shear viscosity `eta` | `P_ab`, `a != b` | `V / (k_B T)` |
//!
//! (`e_i` the energy of atom `i`, kinetic and site energy, `W_i = sum_j r_ij (x) dU_j/dr_ji`
//! its virial: Fan et al., Phys. Rev. B 2015, 92, 094301, eq. 13.) The correlation and the
//! running integral are those of the whole series; the error of the integral at every lag is
//! the standard error over `blocks` consecutive parts of the series (10 by default). The
//! integral is read at the first lag where the correlation crosses zero (the end of the
//! first decay, after which the integral of the noise only adds noise); the mean of the
//! running integral from there to twice that lag is given as a check of the plateau. When the
//! correlation never crosses zero within the lags, the last lag is taken and the result is
//! marked so.
//!
//! In the reverse method, every `every` steps the atom of the cold slab (the first along the
//! axis) with the largest kinetic energy and the atom of the hot slab (the middle one) with
//! the smallest exchange their velocities (atoms of the same mass: energy and momentum are
//! conserved). The heat flux is the exchanged energy over `2 A t` (it flows both ways around
//! the periodic cell), the conductivity that flux over the gradient of the temperature
//! profile, fitted in both halves between the two slabs (the slabs themselves left out).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::autocorrelation;
use crate::analysis::trajectory::MdTrajectory;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};

/// Default number of blocks of the errors.
pub const DEFAULT_BLOCKS: usize = 10;

/// Watts per metre and kelvin in one atomic unit of thermal conductivity
/// (Hartree / (atomic time Bohr K)).
pub fn watts_per_metre_kelvin() -> f64 {
    Units::HARTREE_TO_JOULE / (Units::AU_TIME_TO_SEC * Units::BOHR_TO_METER)
}

/// Pascal seconds in one atomic unit of viscosity (Hartree atomic time / Bohr^3).
pub fn pascal_seconds() -> f64 {
    Units::HARTREE_TO_JOULE * Units::AU_TIME_TO_SEC / Units::BOHR_TO_METER.powi(3)
}

/// A Green-Kubo integral.
#[derive(Debug, Clone, PartialEq)]
pub struct GreenKubo {
    /// Lags (fs).
    pub lags: Vec<f64>,
    /// The correlation averaged over the components (units of `X^2`).
    pub correlation: Vec<f64>,
    /// `A int_0^t C`, at every lag.
    pub integral: Vec<f64>,
    /// Its standard error over the blocks.
    pub error: Vec<f64>,
    /// Index of the lag the coefficient is read at.
    pub plateau: usize,
    /// The coefficient.
    pub value: f64,
    /// Its standard error.
    pub value_error: f64,
    /// Mean of the integral from the plateau lag to twice it.
    pub plateau_mean: f64,
    /// Whether the correlation crossed zero within the lags.
    pub crossed: bool,
}

/// The correlation averaged over `components` and its running integral (`dt` in atomic
/// time), `lags` long.
fn correlate(components: &[Vec<f64>], lags: usize, dt: f64) -> (Vec<f64>, Vec<f64>) {
    let mut c = vec![0.0; lags];
    for x in components {
        for (o, v) in c.iter_mut().zip(autocorrelation(x, lags)) {
            *o += v / components.len() as f64;
        }
    }
    let mut integral = Vec::with_capacity(lags);
    let mut sum = 0.0;
    for k in 0..lags {
        if k > 0 {
            sum += 0.5 * dt * (c[k - 1] + c[k]);
        }
        integral.push(sum);
    }
    (c, integral)
}

/// `A int_0^inf <X(0) X(t)> dt` of the series `components` (each one sample every `dt_fs`)
/// up to `lags` lags, with the errors of `blocks` blocks.
pub fn green_kubo(
    components: &[Vec<f64>],
    dt_fs: f64,
    prefactor: f64,
    lags: usize,
    blocks: usize,
) -> MoptResult<GreenKubo> {
    let n = components.first().map_or(0, Vec::len);
    if n < 4 || components.iter().any(|x| x.len() != n) {
        return Err(MoptError::config(
            "a Green-Kubo integral needs series of the same length, at least four samples",
        ));
    }
    let blocks = blocks.max(2);
    let block = n / blocks;
    let lags = lags.min(block / 2).max(2);
    if block < 4 {
        return Err(MoptError::config(format!(
            "{n} samples are too few for {blocks} blocks"
        )));
    }
    let dt = dt_fs * Units::FS_TO_AU_TIME;
    let (c, integral) = correlate(components, lags, dt);
    let per_block: Vec<Vec<f64>> = (0..blocks)
        .map(|b| {
            let part: Vec<Vec<f64>> = components
                .iter()
                .map(|x| x[b * block..(b + 1) * block].to_vec())
                .collect();
            correlate(&part, lags, dt).1
        })
        .collect();
    let error: Vec<f64> = (0..lags)
        .map(|k| {
            let mean = per_block.iter().map(|i| i[k]).sum::<f64>() / blocks as f64;
            let var =
                per_block.iter().map(|i| (i[k] - mean).powi(2)).sum::<f64>() / (blocks - 1) as f64;
            prefactor.abs() * libm::sqrt(var / blocks as f64)
        })
        .collect();
    let integral: Vec<f64> = integral.iter().map(|i| i * prefactor).collect();
    let first_zero = (1..lags).find(|&k| c[k] <= 0.0);
    let plateau = first_zero.unwrap_or(lags - 1);
    let end = (2 * plateau).min(lags - 1);
    let window = &integral[plateau..=end];
    Ok(GreenKubo {
        lags: (0..lags).map(|k| k as f64 * dt_fs).collect(),
        correlation: c,
        value: integral[plateau],
        value_error: error[plateau],
        plateau_mean: window.iter().sum::<f64>() / window.len() as f64,
        crossed: first_zero.is_some(),
        plateau,
        integral,
        error,
    })
}

/// The self-diffusion coefficient (Angstrom^2 / ps) of the atoms `atoms` of `traj` from
/// their velocity autocorrelation, `lags` lags.
pub fn diffusion_green_kubo(
    traj: &MdTrajectory,
    atoms: &[usize],
    lags: usize,
    blocks: usize,
) -> MoptResult<GreenKubo> {
    let dt = traj.interval()?;
    let v = traj.velocities().ok_or_else(|| {
        MoptError::config("the Green-Kubo diffusion needs the velocities of the frames")
    })?;
    let components: Vec<Vec<f64>> = atoms
        .iter()
        .flat_map(|&i| (0..3).map(move |c| (i, c)))
        .map(|(i, c)| v.iter().map(|row| row[i][c]).collect())
        .collect();
    // Bohr^2 / atomic time to Angstrom^2 / ps.
    let scale = Units::BOHR_TO_ANGSTROM.powi(2) / (Units::AU_TIME_TO_FS * 1e-3);
    green_kubo(&components, dt, scale, lags, blocks)
}

/// The result of a reverse non-equilibrium run.
#[derive(Debug, Clone, PartialEq)]
pub struct ReverseFlux {
    /// Centres of the slabs along the axis (Angstrom).
    pub centres: Vec<f64>,
    /// Temperature of each slab (K).
    pub temperatures: Vec<f64>,
    /// Its standard error over the blocks.
    pub temperature_errors: Vec<f64>,
    /// Gradient of the temperature (K / Angstrom), the mean of the two halves.
    pub gradient: f64,
    /// Heat flux (W / m^2).
    pub flux: f64,
    /// Thermal conductivity (W / (m K)).
    pub conductivity: f64,
    /// Its standard error over the blocks.
    pub conductivity_error: f64,
}

/// Slab of `x` (fractional coordinate along the axis) among `slabs`.
fn slab_of(s: f64, slabs: usize) -> usize {
    ((s - libm::floor(s)) * slabs as f64).floor() as usize % slabs
}

/// Temperatures of the slabs of the frames `range` of `traj`.
fn profile(
    traj: &MdTrajectory,
    range: std::ops::Range<usize>,
    axis: usize,
    slabs: usize,
) -> MoptResult<Vec<f64>> {
    let v = traj
        .velocities()
        .ok_or_else(|| MoptError::config("the temperature profile needs the velocities"))?;
    let masses: Vec<f64> = traj
        .masses()
        .iter()
        .map(|m| m * Units::AMU_TO_AU_MASS)
        .collect();
    let mut kinetic = vec![0.0; slabs];
    let mut count = vec![0.0; slabs];
    for t in range {
        let sys = &traj.systems()[t];
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("a reverse non-equilibrium run needs a cell"))?;
        for i in 0..traj.n_atoms() {
            let s = cell.cart_to_frac(sys.position(crate::domain::AtomId::new(i)).value())[axis];
            let k = slab_of(s, slabs);
            kinetic[k] += masses[i] * v[t][i].iter().map(|c| c * c).sum::<f64>();
            count[k] += 3.0;
        }
    }
    Ok(kinetic
        .iter()
        .zip(&count)
        .map(|(k, n)| {
            if *n > 0.0 {
                k / (n * Units::BOLTZMANN_HARTREE_PER_K)
            } else {
                f64::NAN
            }
        })
        .collect())
}

/// Slope of a straight line through `(x, y)`.
fn slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    sxy / sxx
}

/// The mean gradient (K per slab width) of both halves of the profile, the exchange slabs
/// left out.
fn gradient_of(t: &[f64]) -> f64 {
    let n = t.len();
    let half = n / 2;
    let left: Vec<usize> = (1..half).collect();
    let right: Vec<usize> = (half + 1..n).collect();
    let fit = |idx: &[usize]| {
        let x: Vec<f64> = idx.iter().map(|&k| k as f64).collect();
        let y: Vec<f64> = idx.iter().map(|&k| t[k]).collect();
        slope(&x, &y)
    };
    0.5 * (fit(&left) - fit(&right))
}

/// The thermal conductivity of a reverse non-equilibrium run: the frames of `traj` (with
/// velocities, in a cell), the energy exchanged from the cold to the hot slab over the
/// frames (Hartree) and their duration (fs), `slabs` slabs along `axis`.
pub fn reverse_conductivity(
    traj: &MdTrajectory,
    exchanged: f64,
    duration_fs: f64,
    axis: usize,
    slabs: usize,
    blocks: usize,
) -> MoptResult<ReverseFlux> {
    if slabs < 6 || !slabs.is_multiple_of(2) {
        return Err(MoptError::config(
            "a reverse non-equilibrium run needs an even number of slabs, at least six",
        ));
    }
    let cell = traj.systems()[0]
        .active_cell()
        .cloned()
        .ok_or_else(|| MoptError::config("a reverse non-equilibrium run needs a cell"))?;
    let widths = cell.perpendicular_widths_bohr();
    let length = widths[axis];
    let area = cell.volume_bohr3() / length;
    let width_ang = length / slabs as f64 * Units::BOHR_TO_ANGSTROM;
    let frames = traj.len();
    let blocks = blocks.clamp(2, frames.max(2));
    let mean = profile(traj, 0..frames, axis, slabs)?;
    let per_block: Vec<f64> = (0..blocks)
        .map(|b| {
            let range = b * frames / blocks..(b + 1) * frames / blocks;
            profile(traj, range, axis, slabs).map(|p| gradient_of(&p))
        })
        .collect::<MoptResult<_>>()?;
    let temperature_errors = (0..slabs)
        .map(|k| {
            let values: Vec<f64> = (0..blocks)
                .map(|b| {
                    profile(
                        traj,
                        b * frames / blocks..(b + 1) * frames / blocks,
                        axis,
                        slabs,
                    )
                    .map_or(f64::NAN, |p| p[k])
                })
                .collect();
            let m = values.iter().sum::<f64>() / blocks as f64;
            libm::sqrt(
                values.iter().map(|v| (v - m).powi(2)).sum::<f64>()
                    / ((blocks - 1) * blocks) as f64,
            )
        })
        .collect();
    let gradient_per_slab = gradient_of(&mean);
    let t_au = duration_fs * Units::FS_TO_AU_TIME;
    let flux_au = exchanged / (2.0 * area * t_au);
    let gradient_au = gradient_per_slab / (length / slabs as f64);
    let kappa = flux_au / gradient_au * watts_per_metre_kelvin();
    let m = per_block.iter().sum::<f64>() / blocks as f64;
    let sd =
        libm::sqrt(per_block.iter().map(|g| (g - m).powi(2)).sum::<f64>() / (blocks - 1) as f64);
    Ok(ReverseFlux {
        centres: (0..slabs).map(|k| (k as f64 + 0.5) * width_ang).collect(),
        temperatures: mean,
        temperature_errors,
        gradient: gradient_per_slab / width_ang,
        flux: flux_au * Units::HARTREE_TO_JOULE
            / (Units::AU_TIME_TO_SEC * Units::BOHR_TO_METER.powi(2)),
        conductivity: kappa,
        conductivity_error: kappa * sd / (m.abs() * libm::sqrt(blocks as f64)).max(1e-300),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_infra::rng::Rng;

    #[test]
    fn an_exponential_correlation_integrates_to_its_area() {
        // Ornstein-Uhlenbeck: <x(0) x(t)> = s^2 exp(-t / tau): the integral is s^2 tau.
        let mut rng = Rng::from_seed(4);
        let (tau, dt) = (20.0, 1.0);
        let a = libm::exp(-dt / tau);
        let mut x = 0.0;
        let series: Vec<f64> = (0..200_000)
            .map(|_| {
                x = a * x + libm::sqrt(1.0 - a * a) * rng.standard_normal();
                x
            })
            .collect();
        let gk = match green_kubo(&[series], dt, 1.0, 400, 10) {
            Ok(g) => g,
            Err(e) => panic!("{e}"),
        };
        // The coefficient in atomic time of fs units: the integral in fs is tau.
        let area_fs = gk.plateau_mean * Units::AU_TIME_TO_FS;
        assert!((area_fs / tau - 1.0).abs() < 0.15, "{area_fs}");
        assert!(gk.value_error > 0.0);
    }
}
