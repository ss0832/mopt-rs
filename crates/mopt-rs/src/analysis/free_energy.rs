//! Free-energy profiles from umbrella sampling: the weighted histogram analysis method
//! (WHAM; Kumar, Rosenberg, Bouzida, Swendsen, Kollman, J. Comput. Chem. 1992, 13, 1011)
//! and the multistate Bennett acceptance ratio (MBAR; Shirts and Chodera, J. Chem. Phys.
//! 2008, 129, 124105), both solved self-consistently in logarithms (log-sum-exp), so that
//! no exponential of a reduced energy is ever formed.
//!
//! The windows are harmonic restraints `k / 2 (xi - c)^2` on one collective variable
//! (periodic variables take the difference wrapped into half a period). The profile is
//! `F(xi) = -k_B T ln p(xi)` on bins, shifted to zero at its minimum (Hartree).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// One window of umbrella sampling.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// Centre `c` of the restraint.
    pub center: f64,
    /// Force constant `k` (Hartree per unit of the variable squared).
    pub k: f64,
    /// Values of the variable sampled in the window.
    pub samples: Vec<f64>,
}

/// Settings of the analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileSettings {
    /// `k_B T` (Hartree).
    pub kt: f64,
    /// Number of bins.
    pub bins: usize,
    /// Range of the bins (`None`: that of the samples).
    pub range: Option<(f64, f64)>,
    /// Period of the variable.
    pub period: Option<f64>,
    /// Largest change of the dimensionless free energies of the windows at convergence.
    pub tolerance: f64,
    /// Largest number of iterations.
    pub max_iterations: usize,
}

impl ProfileSettings {
    /// 50 bins over the samples, tolerance 1e-9, 100000 iterations.
    pub fn at(kt: f64) -> Self {
        Self {
            kt,
            bins: 50,
            range: None,
            period: None,
            tolerance: 1e-9,
            max_iterations: 100_000,
        }
    }
}

/// A free-energy profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    /// Centres of the bins.
    pub centers: Vec<f64>,
    /// Free energy of each bin (Hartree, 0 at the minimum; infinite where nothing was
    /// sampled).
    pub free_energy: Vec<f64>,
    /// Free energies of the windows (Hartree, the first at 0).
    pub window_free_energies: Vec<f64>,
    /// Iterations of the self-consistent solution.
    pub iterations: usize,
}

/// `ln sum exp(values)` (`-inf` for nothing).
fn lse(values: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = values.collect();
    let top = v.iter().fold(f64::NEG_INFINITY, |m, x| m.max(*x));
    if !top.is_finite() {
        return top;
    }
    v.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + top
}

fn bias(w: &Window, xi: f64, period: Option<f64>) -> f64 {
    let mut d = xi - w.center;
    if let Some(p) = period {
        d -= p * (d / p).round();
    }
    0.5 * w.k * d * d
}

struct Bins {
    lo: f64,
    width: f64,
    count: usize,
}

impl Bins {
    fn of(windows: &[Window], s: &ProfileSettings) -> MoptResult<Self> {
        if windows.is_empty() || windows.iter().any(|w| w.samples.is_empty()) {
            return Err(MoptError::config("every window needs samples"));
        }
        if !(s.kt.is_finite() && s.kt > 0.0) || s.bins == 0 {
            return Err(MoptError::config("the analysis needs k_B T > 0 and bins"));
        }
        let (lo, hi) = s.range.unwrap_or_else(|| {
            let all = windows.iter().flat_map(|w| w.samples.iter().copied());
            all.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| {
                (a.min(x), b.max(x))
            })
        });
        if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
            return Err(MoptError::config("the samples span no range"));
        }
        Ok(Self {
            lo,
            width: (hi - lo) / s.bins as f64,
            count: s.bins,
        })
    }

    fn index(&self, xi: f64) -> Option<usize> {
        let k = ((xi - self.lo) / self.width).floor();
        (k >= 0.0 && (k as usize) < self.count)
            .then_some(k as usize)
            .or_else(|| {
                ((xi - self.lo - self.width * self.count as f64).abs()
                    < 1e-12 * self.width.abs().max(1.0))
                .then_some(self.count - 1)
            })
    }

    fn centers(&self) -> Vec<f64> {
        (0..self.count)
            .map(|b| self.lo + (b as f64 + 0.5) * self.width)
            .collect()
    }
}

/// The profile from the logarithms of the unnormalized probabilities of the bins.
fn profile(ln_p: &[f64], bins: &Bins, f: &[f64], kt: f64, iterations: usize) -> Profile {
    let top = ln_p.iter().fold(f64::NEG_INFINITY, |m, x| m.max(*x));
    let free_energy = ln_p
        .iter()
        .map(|x| {
            if x.is_finite() {
                kt * (top - x)
            } else {
                f64::INFINITY
            }
        })
        .collect();
    Profile {
        centers: bins.centers(),
        free_energy,
        window_free_energies: f.iter().map(|v| kt * (v - f[0])).collect(),
        iterations,
    }
}

/// WHAM on histograms of the windows.
pub fn wham(windows: &[Window], s: &ProfileSettings) -> MoptResult<Profile> {
    let bins = Bins::of(windows, s)?;
    let beta = 1.0 / s.kt;
    let centers = bins.centers();
    let mut counts = vec![0.0_f64; bins.count];
    // The samples of each window inside the bins (those outside are not in a histogram and
    // must not weigh in the denominator either).
    let mut inside = vec![0.0_f64; windows.len()];
    for (k, w) in windows.iter().enumerate() {
        for &xi in &w.samples {
            if let Some(b) = bins.index(xi) {
                counts[b] += 1.0;
                inside[k] += 1.0;
            }
        }
    }
    let ln_n: Vec<f64> = inside.iter().map(|n| n.ln()).collect();
    let u: Vec<Vec<f64>> = windows
        .iter()
        .map(|w| {
            centers
                .iter()
                .map(|c| beta * bias(w, *c, s.period))
                .collect()
        })
        .collect();
    let mut f = vec![0.0_f64; windows.len()];
    let mut ln_p = vec![f64::NEG_INFINITY; bins.count];
    for iteration in 1..=s.max_iterations {
        for b in 0..bins.count {
            ln_p[b] = if counts[b] > 0.0 {
                counts[b].ln() - lse((0..windows.len()).map(|k| ln_n[k] + f[k] - u[k][b]))
            } else {
                f64::NEG_INFINITY
            };
        }
        let mut next: Vec<f64> = (0..windows.len())
            .map(|k| -lse((0..bins.count).map(|b| ln_p[b] - u[k][b])))
            .collect();
        let shift = next[0];
        next.iter_mut().for_each(|v| *v -= shift);
        let change = next
            .iter()
            .zip(&f)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        f = next;
        if change < s.tolerance {
            return Ok(profile(&ln_p, &bins, &f, s.kt, iteration));
        }
    }
    Err(MoptError::conv(format!(
        "WHAM did not converge in {} iterations",
        s.max_iterations
    )))
}

/// MBAR on the samples of all windows (the unbiased state is the surface without
/// restraints).
pub fn mbar(windows: &[Window], s: &ProfileSettings) -> MoptResult<Profile> {
    let bins = Bins::of(windows, s)?;
    let beta = 1.0 / s.kt;
    let samples: Vec<f64> = windows
        .iter()
        .flat_map(|w| w.samples.iter().copied())
        .collect();
    let ln_n: Vec<f64> = windows
        .iter()
        .map(|w| (w.samples.len() as f64).ln())
        .collect();
    // u[k][n]: reduced bias of window k at sample n.
    let u: Vec<Vec<f64>> = windows
        .iter()
        .map(|w| {
            samples
                .iter()
                .map(|x| beta * bias(w, *x, s.period))
                .collect()
        })
        .collect();
    let mut f = vec![0.0_f64; windows.len()];
    let mut denominator = vec![0.0_f64; samples.len()];
    for iteration in 1..=s.max_iterations {
        for (n, d) in denominator.iter_mut().enumerate() {
            *d = lse((0..windows.len()).map(|k| ln_n[k] + f[k] - u[k][n]));
        }
        let mut next: Vec<f64> = (0..windows.len())
            .map(|k| -lse((0..samples.len()).map(|n| -u[k][n] - denominator[n])))
            .collect();
        let shift = next[0];
        next.iter_mut().for_each(|v| *v -= shift);
        let change = next
            .iter()
            .zip(&f)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        f = next;
        if change < s.tolerance {
            let mut ln_p = vec![f64::NEG_INFINITY; bins.count];
            for (n, x) in samples.iter().enumerate() {
                if let Some(b) = bins.index(*x) {
                    ln_p[b] = lse([ln_p[b], -denominator[n]].into_iter());
                }
            }
            return Ok(profile(&ln_p, &bins, &f, s.kt, iteration));
        }
    }
    Err(MoptError::conv(format!(
        "MBAR did not converge in {} iterations",
        s.max_iterations
    )))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core_infra::rng::Rng;

    /// A double well `U = h (x^2 - 1)^2` (Hartree).
    fn well(x: f64) -> f64 {
        4.0e-3 * (x * x - 1.0).powi(2)
    }

    /// Metropolis samples of the biased distribution of a window.
    fn sample(center: f64, k: f64, kt: f64, rng: &mut Rng) -> Vec<f64> {
        let energy = |x: f64| well(x) + 0.5 * k * (x - center).powi(2);
        let mut x = center;
        let mut out = Vec::new();
        for step in 0..60_000 {
            let trial = x + rng.uniform(-0.1, 0.1);
            let de = energy(trial) - energy(x);
            if de <= 0.0 || rng.next_f64() < (-de / kt).exp() {
                x = trial;
            }
            if step >= 2000 && step % 5 == 0 {
                out.push(x);
            }
        }
        out
    }

    #[test]
    fn wham_and_mbar_recover_a_double_well() {
        let kt = 1.0e-3;
        let k = 0.02;
        let mut rng = Rng::from_seed(17);
        let windows: Vec<Window> = (0..16)
            .map(|i| {
                let center = -1.5 + 0.2 * i as f64;
                Window {
                    center,
                    k,
                    samples: sample(center, k, kt, &mut rng),
                }
            })
            .collect();
        let settings = ProfileSettings {
            range: Some((-1.4, 1.4)),
            bins: 28,
            ..ProfileSettings::at(kt)
        };
        for profile in [
            wham(&windows, &settings).unwrap(),
            mbar(&windows, &settings).unwrap(),
        ] {
            let reference: Vec<f64> = profile.centers.iter().map(|x| well(*x)).collect();
            let lowest = reference.iter().fold(f64::INFINITY, |m, v| m.min(*v));
            // The barrier of 4e-3 Hartree is recovered within a tenth of k_B T on average.
            let errors: Vec<f64> = profile
                .free_energy
                .iter()
                .zip(&reference)
                .map(|(f, r)| f - (r - lowest))
                .collect();
            let mean = errors.iter().sum::<f64>() / errors.len() as f64;
            let rms = (errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>()
                / errors.len() as f64)
                .sqrt();
            assert!(rms < 0.1 * kt, "rms {rms}");
            assert!(profile.free_energy.iter().all(|f| f.is_finite()));
        }
        assert!(wham(&[], &settings).is_err());
    }
}
