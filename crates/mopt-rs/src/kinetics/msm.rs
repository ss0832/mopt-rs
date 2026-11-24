//! Rate matrices from molecular dynamics: Markov state models and the statistics of a
//! semi-Markov chain from a trajectory assigned to the states of a network.
//!
//! A discrete trajectory is the state of every frame (`None` where the frame is in no state);
//! frames in no state keep the last state visited (core-set assignment: a transition is counted
//! when the next state is entered, Buchete and Hummer, J. Phys. Chem. B 2008, 112, 6057).
//!
//! - [`count_matrix`]: transitions at a lag of `tau` frames (sliding windows).
//! - [`reversible_transition_matrix`]: the reversible maximum-likelihood transition matrix by
//!   the fixed-point iteration of Prinz et al. (J. Chem. Phys. 2011, 134, 174105, eq. 25):
//!   `x_ij = (c_ij + c_ji) / (c_i / x_i + c_j / x_j)`, `T_ij = x_ij / x_i`.
//! - [`implied_timescales`]: `t_k = -tau / ln |lambda_k|`; a Markov model is valid at the lag
//!   where they no longer change with it (Prinz et al.; Husic and Pande, J. Am. Chem. Soc. 2018,
//!   140, 2386).
//! - [`rate_matrix`]: the generator `K = ln(T) / tau` from the eigen-decomposition of the
//!   symmetrized reversible `T` (a generator only where its off-diagonal entries are not
//!   negative, which is checked).
//! - [`dwell_statistics`]: for the semi-Markov model, every visit of a state with its duration
//!   and the state entered next: mean residence times, their second moments, and jump
//!   probabilities (`kinetics_beyond_tst.md`, section 3.6).
//!
//! Matrices here are in the convention of the kinetics, `[to][from]`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};
use crate::kinetics::generator::scaled_symmetric_eigen;

/// The states of the frames with the frames outside every state given the last state visited
/// (frames before the first state are dropped).
pub fn core_set(dtraj: &[Option<usize>]) -> Vec<usize> {
    let mut out = Vec::with_capacity(dtraj.len());
    let mut last = None;
    for s in dtraj {
        if s.is_some() {
            last = *s;
        }
        if let Some(l) = last {
            out.push(l);
        }
    }
    out
}

/// The transition counts `C[to][from]` at the lag `lag` (frames) of the trajectories, sliding
/// windows.
pub fn count_matrix(dtrajs: &[Vec<usize>], n: usize, lag: usize) -> MoptResult<Vec<Vec<f64>>> {
    if lag == 0 {
        return Err(MoptError::config(
            "a Markov model needs a lag of at least one frame",
        ));
    }
    let mut c = vec![vec![0.0; n]; n];
    for d in dtrajs {
        for k in lag..d.len() {
            let (from, to) = (d[k - lag], d[k]);
            if from >= n || to >= n {
                return Err(MoptError::config(format!(
                    "a trajectory names the state {} of {n}",
                    from.max(to)
                )));
            }
            c[to][from] += 1.0;
        }
    }
    Ok(c)
}

/// The reversible maximum-likelihood transition matrix `T[to][from]` of the counts `c[to][from]`
/// and its stationary distribution (states without counts keep `T = 1` on the diagonal and a
/// weight 0).
pub fn reversible_transition_matrix(c: &[Vec<f64>]) -> MoptResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let n = c.len();
    // Rows of Prinz et al.: c_ij from i to j = c[j][i].
    let cij = |i: usize, j: usize| c[j][i];
    let ci: Vec<f64> = (0..n).map(|i| (0..n).map(|j| cij(i, j)).sum()).collect();
    let mut x: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| cij(i, j) + cij(j, i)).collect())
        .collect();
    for _ in 0..10_000 {
        let xi: Vec<f64> = x.iter().map(|r| r.iter().sum()).collect();
        let mut change: f64 = 0.0;
        let mut next = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let s = cij(i, j) + cij(j, i);
                if s == 0.0 {
                    continue;
                }
                let den = if xi[i] > 0.0 { ci[i] / xi[i] } else { 0.0 }
                    + if xi[j] > 0.0 { ci[j] / xi[j] } else { 0.0 };
                next[i][j] = if den > 0.0 { s / den } else { 0.0 };
                change = change.max((next[i][j] - x[i][j]).abs() / x[i][j].max(1e-300));
            }
        }
        x = next;
        if change < 1e-12 {
            break;
        }
    }
    let xi: Vec<f64> = x.iter().map(|r| r.iter().sum()).collect();
    let total: f64 = xi.iter().sum();
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(MoptError::config(
            "a Markov model needs at least one transition",
        ));
    }
    let mut t = vec![vec![0.0; n]; n];
    for i in 0..n {
        if xi[i] > 0.0 {
            for j in 0..n {
                t[j][i] = x[i][j] / xi[i];
            }
        } else {
            t[i][i] = 1.0;
        }
    }
    let pi = xi.iter().map(|v| v / total).collect();
    Ok((t, pi))
}

/// The eigen-decomposition of the symmetrized reversible transition matrix (states of weight 0
/// left out): eigenvalues (descending) and eigenvectors, and the states used.
fn symmetrized(t: &[Vec<f64>], pi: &[f64]) -> MoptResult<(Vec<f64>, Mat<f64>, Vec<usize>)> {
    let used: Vec<usize> = (0..pi.len()).filter(|&i| pi[i] > 0.0).collect();
    let m = used.len();
    let s = Mat::from_fn(m, m, |a, b| {
        let (i, j) = (used[a], used[b]);
        // S_ij = sqrt(pi_i / pi_j) T[i][j] (symmetric for a reversible T).
        0.5 * ((pi[j] / pi[i]).sqrt() * t[i][j] + (pi[i] / pi[j]).sqrt() * t[j][i])
    });
    let (values, vectors) = scaled_symmetric_eigen(&s)?;
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let vals = order.iter().map(|&k| values[k]).collect();
    let vecs = Mat::from_fn(m, m, |i, k| vectors[(i, order[k])]);
    Ok((vals, vecs, used))
}

/// The implied time scales `-lag_time / ln |lambda_k|` of the reversible transition matrix
/// (the stationary eigenvalue left out), slowest first.
pub fn implied_timescales(t: &[Vec<f64>], pi: &[f64], lag_time: f64) -> MoptResult<Vec<f64>> {
    let (values, _, _) = symmetrized(t, pi)?;
    Ok(values
        .iter()
        .skip(1)
        .map(|l| {
            let a = l.abs();
            if a >= 1.0 {
                f64::INFINITY
            } else if a > 0.0 {
                -lag_time / a.ln()
            } else {
                0.0
            }
        })
        .collect())
}

/// The rate matrix `K[to][from] = ln(T) / lag_time` (1/s for a lag in s) of a reversible
/// transition matrix, and the most negative off-diagonal entry over the largest rate out of a
/// state (`-inf` when an eigenvalue of `T` is at or below zero and `ln(T)` does not exist): a
/// generator has none, and a small negative value is the statistical error of the counts.
pub fn rate_matrix(t: &[Vec<f64>], pi: &[f64], lag_time: f64) -> MoptResult<(Vec<Vec<f64>>, f64)> {
    let n = pi.len();
    let (values, vectors, used) = symmetrized(t, pi)?;
    let mut worst = if values.iter().all(|l| *l > 0.0) {
        0.0
    } else {
        f64::NEG_INFINITY
    };
    let m = used.len();
    let mut k = vec![vec![0.0; n]; n];
    for a in 0..m {
        for b in 0..m {
            let (i, j) = (used[a], used[b]);
            let s: f64 = (0..m)
                .map(|x| {
                    let ln = if values[x] > 0.0 {
                        values[x].ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                    vectors[(a, x)] * ln * vectors[(b, x)]
                })
                .filter(|v| v.is_finite())
                .sum::<f64>()
                / lag_time;
            // K = Pi^(1/2) S_log Pi^(-1/2) in the [to][from] convention.
            k[i][j] = (pi[i] / pi[j]).sqrt() * s;
        }
    }
    let scale = (0..n)
        .map(|j| k[j][j].abs())
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    for (i, row) in k.iter().enumerate() {
        for (j, x) in row.iter().enumerate() {
            if i != j {
                worst = worst.min(x / scale);
            }
        }
    }
    Ok((k, worst))
}

/// The visits of the states of a core-set trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct DwellStatistics {
    /// The completed visits of every state.
    pub visits: Vec<usize>,
    /// The mean residence time of every state (s; NaN without a completed visit).
    pub mean: Vec<f64>,
    /// The second moment of the residence time (s^2).
    pub second_moment: Vec<f64>,
    /// `jumps[to][from]`: the visits of `from` that ended in `to`.
    pub jumps: Vec<Vec<usize>>,
}

impl DwellStatistics {
    /// The squared coefficient of variation `<t^2> / tau^2 - 1` of the residence time of `j`.
    pub fn cv2(&self, j: usize) -> f64 {
        self.second_moment[j] / (self.mean[j] * self.mean[j]) - 1.0
    }

    /// The Markov limit `K[to][from] = P[to][from] / tau_from` (1/s).
    pub fn markov_limit(&self) -> Vec<Vec<f64>> {
        let n = self.visits.len();
        let mut k = vec![vec![0.0; n]; n];
        for j in 0..n {
            if self.visits[j] == 0
                || self.mean[j].partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            {
                continue;
            }
            for (i, row) in k.iter_mut().enumerate() {
                if i != j {
                    row[j] = self.jumps[i][j] as f64 / self.visits[j] as f64 / self.mean[j];
                }
            }
        }
        k
    }
}

/// The residence times and jumps of core-set trajectories with the time `dt` (s) between
/// frames (a visit ends when another state is entered; the first and the last visit of every
/// trajectory are not complete and are not counted).
pub fn dwell_statistics(dtrajs: &[Vec<usize>], n: usize, dt: f64) -> MoptResult<DwellStatistics> {
    if !(dt > 0.0 && dt.is_finite()) {
        return Err(MoptError::config(
            "the time between frames must be positive",
        ));
    }
    let mut visits = vec![0usize; n];
    let mut sum = vec![0.0; n];
    let mut sum2 = vec![0.0; n];
    let mut jumps = vec![vec![0usize; n]; n];
    for d in dtrajs {
        if d.iter().any(|&s| s >= n) {
            return Err(MoptError::config(format!(
                "a trajectory names a state beyond {n}"
            )));
        }
        // A visit runs from the transition into it to the transition out of it (both taken
        // midway between their frames); the first visit of a trajectory started before it and
        // the last is not finished: neither is counted.
        let mut start = 0;
        for k in 1..d.len() {
            if d[k] != d[k - 1] {
                if start > 0 {
                    let from = d[k - 1];
                    let t = (k - start) as f64 * dt;
                    visits[from] += 1;
                    sum[from] += t;
                    sum2[from] += t * t;
                    jumps[d[k]][from] += 1;
                }
                start = k;
            }
        }
    }
    let mean = (0..n)
        .map(|j| {
            if visits[j] > 0 {
                sum[j] / visits[j] as f64
            } else {
                f64::NAN
            }
        })
        .collect();
    let second_moment = (0..n)
        .map(|j| {
            if visits[j] > 0 {
                sum2[j] / visits[j] as f64
            } else {
                f64::NAN
            }
        })
        .collect();
    Ok(DwellStatistics {
        visits,
        mean,
        second_moment,
        jumps,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core_infra::rng::Rng;

    /// A discrete Markov chain with the transition matrix `t[to][from]`.
    fn chain(t: &[Vec<f64>], steps: usize, rng: &mut Rng) -> Vec<usize> {
        let mut s = 0;
        let mut out = vec![s];
        for _ in 0..steps {
            let u = rng.next_f64();
            let mut cum = 0.0;
            let mut next = t.len() - 1;
            for (i, row) in t.iter().enumerate() {
                cum += row[s];
                if u < cum {
                    next = i;
                    break;
                }
            }
            s = next;
            out.push(s);
        }
        out
    }

    #[test]
    fn a_markov_model_recovers_the_rates_of_a_chain() {
        // The exact transition matrix of a known generator at the lag of one frame.
        let k = [[-1.0, 2.0, 0.0], [1.0, -3.0, 0.5], [0.0, 1.0, -0.5]];
        let dt = 0.05;
        let km = Mat::from_fn(3, 3, |i, j| k[i][j]);
        let t = crate::kinetics::expm::expm_metzler(&km, dt).unwrap();
        let t: Vec<Vec<f64>> = (0..3)
            .map(|i| (0..3).map(|j| t[(i, j)]).collect())
            .collect();
        let mut rng = Rng::from_seed(5);
        let d = chain(&t, 400_000, &mut rng);
        let c = count_matrix(std::slice::from_ref(&d), 3, 1).unwrap();
        let (te, pi) = reversible_transition_matrix(&c).unwrap();
        // The stationary distribution of K: (2, 1, 2) / 5.
        for (p, want) in pi.iter().zip([0.4, 0.2, 0.4]) {
            assert!((p - want).abs() < 0.01, "{p} {want}");
        }
        let (ke, worst) = rate_matrix(&te, &pi, dt).unwrap();
        assert!(worst > -0.02, "{worst}");
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (ke[i][j] - k[i][j]).abs() < 0.08 * k[i][j].abs().max(0.5),
                    "{i} {j}: {} {}",
                    ke[i][j],
                    k[i][j]
                );
            }
        }
        // The implied time scales: -1/eigenvalues of K, the same at every lag of a Markov chain.
        let ts1 = implied_timescales(&te, &pi, dt).unwrap();
        let c5 = count_matrix(&[d], 3, 5).unwrap();
        let (t5, pi5) = reversible_transition_matrix(&c5).unwrap();
        let ts5 = implied_timescales(&t5, &pi5, 5.0 * dt).unwrap();
        assert!((ts1[0] / ts5[0] - 1.0).abs() < 0.05, "{ts1:?} {ts5:?}");
    }

    #[test]
    fn dwell_statistics_of_a_core_set_trajectory() {
        let d = core_set(&[
            None,
            Some(0),
            Some(0),
            None,
            Some(1),
            Some(1),
            Some(1),
            Some(0),
        ]);
        assert_eq!(d, vec![0, 0, 0, 1, 1, 1, 0]);
        let s = dwell_statistics(&[d], 2, 1.0).unwrap();
        // The visit of 0 at the start and the last one are not counted: one visit of 1.
        assert_eq!(s.visits, vec![0, 1]);
        assert_eq!(s.jumps[1][0], 0);
        assert_eq!(s.jumps[0][1], 1);
        assert!(s.mean[0].is_nan());
        assert!((s.mean[1] - 3.0).abs() < 1e-12, "{}", s.mean[1]);
        let k = s.markov_limit();
        assert!((k[0][1] - 1.0 / 3.0).abs() < 1e-12);
    }
}
