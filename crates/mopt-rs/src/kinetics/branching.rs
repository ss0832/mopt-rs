//! Branching fractions and the transmission coefficient from trajectories started at the
//! dividing surface of a saddle point (`kinetics_beyond_tst.md`, sections 2.2, 2.3, 3.3).
//!
//! Every sample is a point on the dividing surface (no displacement along the imaginary mode)
//! with thermal velocities, run forwards and with all velocities reversed: the two runs are the
//! future and the past of one trajectory that crosses the surface with the speed `v > 0` along
//! the imaginary mode (the pair). The ends are the minima the runs commit to (`e+` forwards,
//! `e-` backwards). With the flux of the reactive trajectories through the surface (Bennett,
//! ACS Symp. Ser. 1977, 46, 63; Chandler, J. Chem. Phys. 1978, 68, 2959),
//!
//! ```text
//! J_AB = < v delta(x) h_A(past) h_B(future) >  over both members of every pair
//!      = (1 / 2N) sum_i w_i ([e-_i = A][e+_i = B] - [e+_i = A][e-_i = B])
//! F_TST = (1 / 2N) sum_i w_i                       (the crossings in the positive direction)
//! ```
//!
//! (`w_i = |v_i|`, the flux weighting; `w_i = 1` counts trajectories, as most studies of
//! bifurcations do). `J_AB` is the net flux of the trajectories from `A` into `B` through the
//! surface: positive for `A` behind and `B` ahead of it, and zero on average for two states on one
//! side. The transmission coefficient is `kappa = sum |J_AB| / F_TST` over the pairs of states;
//! the branching fractions are `phi_AB = |J_AB| / sum |J|`, which are those of the thermal
//! (equilibrium) flux and so enter the generator symmetrically (section 3.3). Their standard
//! errors are those of a ratio of sample means (the delta method over the pairs); for counted
//! trajectories they are close to `sqrt(phi (1 - phi) / N)` over the `N` reactive pairs. Pairs that do not commit within the
//! time are counted as undecided and left out (the report says how many).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use crate::error::{MoptError, MoptResult};

/// One pair of runs from the dividing surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryPair {
    /// The speed along the imaginary mode of the member that moves in its positive direction
    /// (any consistent unit; only ratios are used).
    pub speed: f64,
    /// The state its forward run (in the positive direction) commits to.
    pub forward: Option<usize>,
    /// The state its time-reversed run commits to.
    pub backward: Option<usize>,
}

/// How the pairs are weighted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weighting {
    /// By the flux through the surface, `|v|` (Bennett and Chandler).
    Flux,
    /// One per pair.
    Count,
}

impl Weighting {
    /// `flux` or `count`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "flux" => Ok(Self::Flux),
            "count" => Ok(Self::Count),
            other => Err(MoptError::parse(format!(
                "unknown weighting '{other}' (flux, count)"
            ))),
        }
    }
}

/// The reactive flux between two states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outcome {
    /// The state behind the surface (where the net flux comes from).
    pub from: usize,
    /// The state ahead of it.
    pub to: usize,
    /// `|J_AB| / F_TST`: its share of the transmission coefficient.
    pub kappa: f64,
    /// Its branching fraction `|J_AB| / sum |J|`.
    pub fraction: f64,
    /// The standard error of the fraction.
    pub error: f64,
    /// The pairs that connect the two states (either way).
    pub pairs: usize,
}

/// The statistics of the pairs of a saddle point.
#[derive(Debug, Clone, PartialEq)]
pub struct Branching {
    /// Pairs.
    pub pairs: usize,
    /// Pairs with an end that did not commit.
    pub undecided: usize,
    /// Pairs whose two ends are the same state (recrossing).
    pub recrossing: usize,
    /// The reactive fluxes, largest first.
    pub outcomes: Vec<Outcome>,
    /// The transmission coefficient `sum |J| / F_TST`.
    pub kappa: f64,
    /// Its standard error.
    pub kappa_error: f64,
}

/// The standard error of the ratio of the sample sums `sum a / sum b` over `n` samples.
fn ratio_error(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (sa, sb): (f64, f64) = (a.iter().sum(), b.iter().sum());
    if sb.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || n < 2.0 {
        return f64::NAN;
    }
    let r = sa / sb;
    let mean_b = sb / n;
    let var: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - r * y).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    (var / n).sqrt() / mean_b
}

/// The statistics of `pairs`.
pub fn branching(pairs: &[TrajectoryPair], weighting: Weighting) -> MoptResult<Branching> {
    if pairs.is_empty() {
        return Err(MoptError::config("branching: no trajectory pairs"));
    }
    let weight = |p: &TrajectoryPair| match weighting {
        Weighting::Flux => p.speed.abs(),
        Weighting::Count => 1.0,
    };
    let decided: Vec<(f64, usize, usize)> = pairs
        .iter()
        .filter_map(|p| match (p.backward, p.forward) {
            (Some(b), Some(f)) => Some((weight(p), b, f)),
            _ => None,
        })
        .collect();
    let undecided = pairs.len() - decided.len();
    let total: Vec<f64> = pairs.iter().map(weight).collect();
    // The unordered pairs of states that some pair connects.
    let mut keys: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut recrossing = 0;
    for &(_, b, f) in &decided {
        if b == f {
            recrossing += 1;
        } else {
            *keys.entry((b.min(f), b.max(f))).or_default() += 1;
        }
    }
    // Per pair i: its signed contribution to J between the states of every key, the states
    // ordered (lower, higher).
    let mut fluxes: Vec<((usize, usize), Vec<f64>, usize)> = Vec::new();
    for (&(lo, hi), &count) in &keys {
        let a: Vec<f64> = pairs
            .iter()
            .map(|p| match (p.backward, p.forward) {
                (Some(b), Some(f)) if b == lo && f == hi => weight(p),
                (Some(b), Some(f)) if b == hi && f == lo => -weight(p),
                _ => 0.0,
            })
            .collect();
        fluxes.push(((lo, hi), a, count));
    }
    let sum_t: f64 = total.iter().sum();
    if sum_t.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(MoptError::config(
            "branching: the pairs have no speed along the imaginary mode",
        ));
    }
    // |J| per key, oriented from the side it comes from.
    let mut abs: Vec<Vec<f64>> = Vec::new();
    let mut outcomes = Vec::new();
    for ((lo, hi), a, count) in &fluxes {
        let j: f64 = a.iter().sum();
        let (from, to, sign) = if j >= 0.0 {
            (*lo, *hi, 1.0)
        } else {
            (*hi, *lo, -1.0)
        };
        let oriented: Vec<f64> = a.iter().map(|x| sign * x).collect();
        outcomes.push(Outcome {
            from,
            to,
            kappa: sign * j / sum_t,
            fraction: 0.0,
            error: ratio_error(&oriented, &total),
            pairs: *count,
        });
        abs.push(oriented);
    }
    let all: Vec<f64> = (0..pairs.len())
        .map(|i| abs.iter().map(|a| a[i]).sum())
        .collect();
    let sum_all: f64 = all.iter().sum();
    let kappa = sum_all / sum_t;
    let kappa_error = ratio_error(&all, &total);
    for (o, a) in outcomes.iter_mut().zip(&abs) {
        if sum_all > 0.0 {
            o.fraction = a.iter().sum::<f64>() / sum_all;
            o.error = ratio_error(a, &all);
        } else {
            o.fraction = f64::NAN;
            o.error = f64::NAN;
        }
    }
    outcomes.sort_by(|a, b| b.fraction.total_cmp(&a.fraction).then(a.to.cmp(&b.to)));
    Ok(Branching {
        pairs: pairs.len(),
        undecided,
        recrossing,
        outcomes,
        kappa,
        kappa_error,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn pair(speed: f64, backward: usize, forward: usize) -> TrajectoryPair {
        TrajectoryPair {
            speed,
            forward: Some(forward),
            backward: Some(backward),
        }
    }

    #[test]
    fn fractions_kappa_and_errors_of_counted_pairs() {
        // 60 pairs R -> P1, 30 R -> P2, 10 recrossing back to R.
        let mut pairs = Vec::new();
        pairs.extend((0..60).map(|_| pair(1.0, 0, 1)));
        pairs.extend((0..30).map(|_| pair(1.0, 0, 2)));
        pairs.extend((0..10).map(|_| pair(1.0, 0, 0)));
        let b = branching(&pairs, Weighting::Count).unwrap();
        assert_eq!((b.pairs, b.recrossing, b.undecided), (100, 10, 0));
        assert!((b.kappa - 0.9).abs() < 1e-12);
        let p1 = b.outcomes.iter().find(|o| o.to == 1).unwrap();
        assert_eq!(p1.from, 0);
        assert!((p1.fraction - 2.0 / 3.0).abs() < 1e-12);
        // The delta method over the 100 pairs: sum (a - phi b)^2 / 99 = (60/9 + 30 4/9) / 99,
        // over the mean of b (0.9); close to sqrt(phi (1 - phi) / 90).
        let want = ((60.0 / 9.0 + 30.0 * 4.0 / 9.0) / 99.0 / 100.0_f64).sqrt() / 0.9;
        assert!((p1.error - want).abs() < 1e-12, "{} {want}", p1.error);
        assert!((p1.error / (2.0_f64 / 9.0 / 90.0).sqrt() - 1.0).abs() < 0.01);
        // A trajectory that crosses back the wrong way counts against the flux.
        pairs.push(pair(1.0, 1, 0));
        let b = branching(&pairs, Weighting::Count).unwrap();
        let p1 = b.outcomes.iter().find(|o| o.to == 1).unwrap();
        assert!((p1.kappa - 59.0 / 101.0).abs() < 1e-12);
        // Flux weighting: a fast reactive pair weighs more.
        let pairs = vec![pair(3.0, 0, 1), pair(1.0, 0, 2)];
        let b = branching(&pairs, Weighting::Flux).unwrap();
        assert!((b.outcomes[0].fraction - 0.75).abs() < 1e-12);
        assert!((b.kappa - 1.0).abs() < 1e-12);
    }
}
