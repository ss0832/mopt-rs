//! Observables of a kinetic solution that rank the states of a network for its exploration
//! (`kinetics_beyond_tst.md`, section 3.11), with the time horizon `t_max`:
//!
//! - `population`: `p_i(t_max)` (MultiOptPy's `RCMCQueue`);
//! - `max_population`: the largest `p_i(t)` for `0 <= t <= t_max` on a logarithmic grid of
//!   eight points per decade from `0.01 / k_max` (`k_max` the largest rate out of a state)
//!   to `t_max`, with `t = 0`: it reaches short-lived intermediates that are passed early;
//! - `residence`: `tau_i = int_0^t_max p_i(t) dt`, the expected time spent in `i`;
//! - `discovery_flux`: `tau_i k_guess / (1 + n_i)` with the visits `n_i` and the largest rate
//!   constant of the network as the optimistic rate of an exit not found yet (a proposal of
//!   the design: the expected number of reactive events through unknown channels).
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop` (the population at the reaction time).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::solve::{Evolution, KineticSolution, KineticSolver};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};

/// The score of a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Score {
    /// `p_i(t_max)`.
    Population,
    /// `max_{t <= t_max} p_i(t)`.
    MaxPopulation,
    /// `int_0^t_max p_i dt`.
    Residence,
    /// `tau_i k_guess / (1 + n_i)`.
    DiscoveryFlux,
}

impl Score {
    /// `population`, `max_population`, `residence`, `discovery_flux`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "population" => Ok(Self::Population),
            "max_population" => Ok(Self::MaxPopulation),
            "residence" => Ok(Self::Residence),
            "discovery_flux" => Ok(Self::DiscoveryFlux),
            other => Err(MoptError::parse(format!(
                "unknown kinetic score '{other}' (population, max_population, residence, \
                 discovery_flux)"
            ))),
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Population => "population",
            Self::MaxPopulation => "max_population",
            Self::Residence => "residence",
            Self::DiscoveryFlux => "discovery_flux",
        }
    }
}

/// The grid of `max_population`: eight points per decade from `0.01 / k_max` (at most 30
/// decades below `t_max`) to `t_max`.
pub fn time_grid(k_max: f64, t_max: f64) -> Vec<f64> {
    let lo = if k_max > 0.0 {
        (0.01 / k_max).max(t_max * 1e-30)
    } else {
        t_max
    };
    if lo.is_nan() || lo >= t_max {
        return vec![t_max];
    }
    let decades = (t_max / lo).log10();
    let points = (decades * 8.0).ceil().max(1.0) as usize;
    let mut out: Vec<f64> = (0..points)
        .map(|k| lo * 10f64.powf(decades * k as f64 / points as f64))
        .collect();
    out.push(t_max);
    out
}

/// The scores of every state of `ev` from `p0` at `t_max`; `visits` are needed by
/// `discovery_flux`.
pub fn scores(
    solver: &mut dyn KineticSolver,
    ev: &Evolution,
    p0: &[f64],
    t_max: f64,
    score: Score,
    visits: Option<&[usize]>,
    rng: &mut Rng,
) -> MoptResult<Vec<f64>> {
    scores_observed(solver, ev, p0, t_max, score, visits, rng, None)
}

/// Observe the solution at the score horizon, including its RCMC superstates.
/// Integral scores use quadrature nodes inside the interval and need a separate endpoint.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scores_observed(
    solver: &mut dyn KineticSolver,
    ev: &Evolution,
    p0: &[f64],
    t_max: f64,
    score: Score,
    visits: Option<&[usize]>,
    rng: &mut Rng,
    mut observe: Option<&mut dyn FnMut(&KineticSolution)>,
) -> MoptResult<Vec<f64>> {
    if !(t_max > 0.0 && t_max.is_finite()) {
        return Err(MoptError::config(format!(
            "the time of the kinetic score must be positive (got {t_max})"
        )));
    }
    if matches!(score, Score::Residence | Score::DiscoveryFlux) {
        if let Some(observe) = observe.as_mut() {
            observe(&solver.solve(ev, p0, &[t_max], &mut rng.clone())?);
        }
    }
    match score {
        Score::Population => {
            let sol = solver.solve(ev, p0, &[t_max], rng)?;
            if let Some(observe) = observe {
                observe(&sol);
            }
            Ok(sol.populations.into_iter().next().unwrap_or_default())
        }
        Score::MaxPopulation => {
            let k_max = ev.generator()?.rates().largest_out();
            let grid = time_grid(k_max, t_max);
            let sol = solver.solve(ev, p0, &grid, rng)?;
            if let Some(observe) = observe {
                observe(&sol);
            }
            let mut best = p0.to_vec();
            for p in &sol.populations {
                for (b, v) in best.iter_mut().zip(p) {
                    *b = b.max(*v);
                }
            }
            Ok(best)
        }
        Score::Residence => solver.residence(ev, p0, t_max, rng),
        Score::DiscoveryFlux => {
            let Some(visits) = visits else {
                return Err(MoptError::config(
                    "the score discovery_flux needs the visits of the states",
                ));
            };
            let tau = solver.residence(ev, p0, t_max, rng)?;
            let g = ev.generator()?;
            let k_guess = g
                .entries
                .iter()
                .map(|e| e.ln_k)
                .fold(f64::NEG_INFINITY, f64::max)
                .exp();
            Ok(tau
                .iter()
                .zip(visits)
                .map(|(t, n)| t * k_guess / (1.0 + *n as f64))
                .collect())
        }
    }
}
