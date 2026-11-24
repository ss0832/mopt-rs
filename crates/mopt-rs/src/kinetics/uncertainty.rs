//! Propagation of energy errors: `N` draws of independent normal errors of standard deviation
//! `sigma` on the energy of every state and bottleneck (the electronic and the free energy of a
//! stationary point move together), each draw solved like the model itself; the mean, the 5 %,
//! 50 % and 95 % quantiles of every population, and the probability that a state is the most
//! populated one. The draws are made in parallel from seeds `seed + k`, so the result does not
//! depend on the number of threads.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use rayon::prelude::*;

use super::network::KineticNetwork;
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};

/// The statistics of the populations over the draws.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationStatistics {
    /// Draws.
    pub samples: usize,
    /// Mean of every population.
    pub mean: Vec<f64>,
    /// 5 % quantile.
    pub q05: Vec<f64>,
    /// Median.
    pub q50: Vec<f64>,
    /// 95 % quantile.
    pub q95: Vec<f64>,
    /// Share of the draws in which the state is the most populated.
    pub top: Vec<f64>,
}

/// `net` with every energy shifted by a normal error of standard deviation `sigma` (Hartree).
pub fn perturbed(net: &KineticNetwork, sigma: f64, rng: &mut Rng) -> KineticNetwork {
    let mut out = net.clone();
    for s in &mut out.states {
        let d = rng.normal(0.0, sigma);
        s.energy += d;
        if let Some(g) = s.free_energy.as_mut() {
            *g += d;
        }
    }
    for b in &mut out.bottlenecks {
        let d = rng.normal(0.0, sigma);
        b.energy += d;
        if let Some(g) = b.free_energy.as_mut() {
            *g += d;
        }
    }
    out
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let x = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo as f64)
}

/// The statistics of `solve(perturbed network)` over `samples` draws with errors `sigma`
/// (Hartree) from `seed`; `solve` returns the populations of every state of the model.
pub fn sample<F>(
    net: &KineticNetwork,
    sigma: f64,
    samples: usize,
    seed: u64,
    solve: F,
) -> MoptResult<PopulationStatistics>
where
    F: Fn(&KineticNetwork) -> MoptResult<Vec<f64>> + Sync,
{
    if !(sigma >= 0.0 && sigma.is_finite()) || samples == 0 {
        return Err(MoptError::config(
            "uncertainty: the standard deviation must not be negative and the draws at least 1",
        ));
    }
    let draws: Vec<MoptResult<Vec<f64>>> = (0..samples)
        .into_par_iter()
        .map(|k| {
            let mut rng = Rng::from_seed(seed.wrapping_add(k as u64));
            solve(&perturbed(net, sigma, &mut rng))
        })
        .collect();
    let draws: Vec<Vec<f64>> = draws.into_iter().collect::<MoptResult<_>>()?;
    let n = draws.first().map_or(0, Vec::len);
    if draws.iter().any(|d| d.len() != n) {
        return Err(MoptError::config(
            "uncertainty: the draws have different numbers of states",
        ));
    }
    let mut mean = vec![0.0; n];
    let mut top = vec![0.0; n];
    let (mut q05, mut q50, mut q95) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for d in &draws {
        for i in 0..n {
            mean[i] += d[i] / samples as f64;
        }
        let best = (0..n).fold(0, |b, i| if d[i] > d[b] { i } else { b });
        if n > 0 {
            top[best] += 1.0 / samples as f64;
        }
    }
    for i in 0..n {
        let mut v: Vec<f64> = draws.iter().map(|d| d[i]).collect();
        v.sort_by(f64::total_cmp);
        q05[i] = quantile(&v, 0.05);
        q50[i] = quantile(&v, 0.5);
        q95[i] = quantile(&v, 0.95);
    }
    Ok(PopulationStatistics {
        samples,
        mean,
        q05,
        q50,
        q95,
        top,
    })
}
