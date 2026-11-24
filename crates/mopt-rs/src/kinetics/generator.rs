//! The generator `K` of `dp/dt = K p` (`K[i][j]`, `i != j`: the rate constant from `j` to `i`;
//! the columns sum to zero), kept as logarithms of its rate constants, with the reference
//! weights `ln pi` when it has detailed balance.
//!
//! In flux form every bottleneck is one symmetric weight `F`: `ln K[i][j] = ln F - ln pi[j]`
//! and `ln K[j][i] = ln F - ln pi[i]`, so detailed balance cannot be broken by rounding.
//! Parallel bottlenecks between the same pair add up (log-sum-exp). Rate constants are
//! exponentiated only for a solver ([`Generator::rates`]); entries more than `exp(-700)` below
//! the largest rate are dropped there and counted.
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop` (the rate matrix `K`, built directly from the
//! exponentials).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use faer::Mat;

use crate::error::{MoptError, MoptResult};
use crate::math::stable::log_sum_exp;

/// Entries more than this far below the largest rate (in `ln`) are dropped.
pub const LN_FLOOR: f64 = -700.0;

/// One directed rate constant `from -> to`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateEntry {
    /// The state it leaves.
    pub from: usize,
    /// The state it enters.
    pub to: usize,
    /// `ln k` (k in 1/s).
    pub ln_k: f64,
}

/// A generator in log form.
#[derive(Debug, Clone, PartialEq)]
pub struct Generator {
    /// Number of states.
    pub n: usize,
    /// Labels of the states.
    pub labels: Vec<String>,
    /// `ln pi` of the states (shifted so that the largest is 0) when the generator has
    /// detailed balance with respect to them.
    pub ln_pi: Option<Vec<f64>>,
    /// The rate constants, merged: one entry per ordered pair.
    pub entries: Vec<RateEntry>,
}

/// The rate constants as a dense matrix (`w[i * n + j]`: from `j` to `i`; the diagonal is 0).
#[derive(Debug, Clone, PartialEq)]
pub struct Rates {
    /// Number of states.
    pub n: usize,
    /// The off-diagonal rates.
    pub w: Vec<f64>,
    /// Entries below the floor that were dropped.
    pub dropped: usize,
}

impl Rates {
    /// The rate from `j` to `i`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.w[i * self.n + j]
    }

    /// The total rate out of `j` (a sum of non-negative numbers).
    pub fn out(&self, j: usize) -> f64 {
        (0..self.n)
            .filter(|&i| i != j)
            .map(|i| self.w[i * self.n + j])
            .sum()
    }

    /// The matrix `K` with its diagonal.
    pub fn matrix(&self) -> Mat<f64> {
        let n = self.n;
        let outs: Vec<f64> = (0..n).map(|j| self.out(j)).collect();
        Mat::from_fn(n, n, |i, j| if i == j { -outs[j] } else { self.get(i, j) })
    }

    /// The largest rate out of a state.
    pub fn largest_out(&self) -> f64 {
        (0..self.n).map(|j| self.out(j)).fold(0.0, f64::max)
    }
}

impl Generator {
    /// The generator of the symmetric fluxes `(a, b, ln F)` between states with `ln_pi`:
    /// `ln K[b][a] = ln F - ln pi[a]`, `ln K[a][b] = ln F - ln pi[b]`.
    pub fn from_fluxes(
        labels: Vec<String>,
        ln_pi: Vec<f64>,
        fluxes: &[(usize, usize, f64)],
    ) -> MoptResult<Self> {
        let n = ln_pi.len();
        if labels.len() != n {
            return Err(MoptError::config("generator: one label per state"));
        }
        let top = ln_pi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            return Err(MoptError::config(
                "generator: the reference weights are not finite",
            ));
        }
        let ln_pi: Vec<f64> = ln_pi.iter().map(|v| v - top).collect();
        let mut directed = Vec::with_capacity(2 * fluxes.len());
        for &(a, b, ln_f) in fluxes {
            if a >= n || b >= n || a == b {
                return Err(MoptError::config(format!(
                    "generator: a flux joins the states {a} and {b} of {n}"
                )));
            }
            let ln_f = ln_f - top;
            directed.push(RateEntry {
                from: a,
                to: b,
                ln_k: ln_f - ln_pi[a],
            });
            directed.push(RateEntry {
                from: b,
                to: a,
                ln_k: ln_f - ln_pi[b],
            });
        }
        Ok(Self {
            n,
            labels,
            ln_pi: Some(ln_pi),
            entries: merge(&directed)?,
        })
    }

    /// A generator of directed rate constants without a claim of detailed balance.
    pub fn from_directed(labels: Vec<String>, entries: &[RateEntry]) -> MoptResult<Self> {
        let n = labels.len();
        if entries
            .iter()
            .any(|e| e.from >= n || e.to >= n || e.from == e.to)
        {
            return Err(MoptError::config(
                "generator: a rate constant joins states that are not in it",
            ));
        }
        Ok(Self {
            n,
            labels,
            ln_pi: None,
            entries: merge(entries)?,
        })
    }

    /// The generator of a matrix `k[i][j]` (rate from `j` to `i`; the diagonal is ignored).
    pub fn from_matrix(k: &[Vec<f64>]) -> MoptResult<Self> {
        let n = k.len();
        let mut entries = Vec::new();
        for (i, row) in k.iter().enumerate() {
            if row.len() != n {
                return Err(MoptError::config("generator: the matrix must be square"));
            }
            for (j, &v) in row.iter().enumerate() {
                if i == j {
                    continue;
                }
                if !(v >= 0.0 && v.is_finite()) {
                    return Err(MoptError::config(format!(
                        "generator: the rate K[{i}][{j}] = {v} must be finite and not negative"
                    )));
                }
                if v > 0.0 {
                    entries.push(RateEntry {
                        from: j,
                        to: i,
                        ln_k: v.ln(),
                    });
                }
            }
        }
        Self::from_directed((0..n).map(|i| format!("S{i}")).collect(), &entries)
    }

    /// The same generator with reference weights when it has detailed balance: the
    /// stationary distribution of every closed class (from the subtraction-free elimination,
    /// with one unit of population per state) is taken as `pi` and kept when every step
    /// balances its reverse to `tol` in `ln`; otherwise the generator is returned as it is.
    pub fn with_detected_balance(self, tol: f64) -> Self {
        if self.ln_pi.is_some() || self.n == 0 {
            return self;
        }
        let Ok(pi) = super::contract::stationary(&self.rates(), &vec![1.0; self.n]) else {
            return self;
        };
        if pi.iter().any(|p| !(*p > 0.0 && p.is_finite())) {
            return self;
        }
        let ln: Vec<f64> = pi.iter().map(|p| p.ln()).collect();
        let top = ln.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let candidate = Self {
            ln_pi: Some(ln.iter().map(|v| v - top).collect()),
            ..self.clone()
        };
        if candidate
            .detailed_balance_residual()
            .is_some_and(|r| r <= tol)
        {
            candidate
        } else {
            self
        }
    }

    /// `ln k` from `j` to `i` (`-inf` if there is no such step).
    pub fn ln_rate(&self, i: usize, j: usize) -> f64 {
        self.entries
            .iter()
            .find(|e| e.from == j && e.to == i)
            .map_or(f64::NEG_INFINITY, |e| e.ln_k)
    }

    /// The dense rate constants, with the entries below the floor dropped.
    pub fn rates(&self) -> Rates {
        let n = self.n;
        let top = self
            .entries
            .iter()
            .map(|e| e.ln_k)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut w = vec![0.0; n * n];
        let mut dropped = 0;
        for e in &self.entries {
            if e.ln_k - top < LN_FLOOR || e.ln_k < -745.0 {
                dropped += 1;
                continue;
            }
            w[e.to * n + e.from] = e.ln_k.exp();
        }
        Rates { n, w, dropped }
    }

    /// The largest `|ln(K[i][j] pi[j]) - ln(K[j][i] pi[i])|` over the pairs with a step, or
    /// `None` without reference weights. A step without its reverse counts as infinite.
    pub fn detailed_balance_residual(&self) -> Option<f64> {
        let ln_pi = self.ln_pi.as_ref()?;
        let mut worst: f64 = 0.0;
        for e in &self.entries {
            let back = self.ln_rate(e.from, e.to);
            let r = (e.ln_k + ln_pi[e.from]) - (back + ln_pi[e.to]);
            worst = worst.max(if r.is_finite() {
                r.abs()
            } else {
                f64::INFINITY
            });
        }
        Some(worst)
    }

    /// The generator restricted to `keep` (in that order).
    pub fn subset(&self, keep: &[usize]) -> Self {
        let mut index = vec![usize::MAX; self.n];
        for (k, &s) in keep.iter().enumerate() {
            index[s] = k;
        }
        let entries = self
            .entries
            .iter()
            .filter(|e| index[e.from] != usize::MAX && index[e.to] != usize::MAX)
            .map(|e| RateEntry {
                from: index[e.from],
                to: index[e.to],
                ln_k: e.ln_k,
            })
            .collect();
        Self {
            n: keep.len(),
            labels: keep.iter().map(|&s| self.labels[s].clone()).collect(),
            ln_pi: self
                .ln_pi
                .as_ref()
                .map(|p| keep.iter().map(|&s| p[s]).collect()),
            entries,
        }
    }
}

/// The eigen-decomposition of the symmetric matrix `a` (eigenvalues ascending), computed on `a`
/// scaled by a power of two near its largest entry and scaled back. The scaling is exact; it
/// keeps the decomposition of faer accurate for the matrices of rate constants, whose norms reach
/// 1e14 1/s and beyond (unscaled, eigenpairs with residuals of the order of the norm were found
/// for the symmetrized generator of an energy-grained master equation).
pub fn scaled_symmetric_eigen(a: &Mat<f64>) -> MoptResult<(faer::Col<f64>, Mat<f64>)> {
    let (n, m) = (a.nrows(), a.ncols());
    let mut big: f64 = 0.0;
    for j in 0..m {
        for i in 0..n {
            big = big.max(a[(i, j)].abs());
        }
    }
    if !(big > 0.0 && big.is_finite()) {
        return crate::math::linalg::symmetric_eigen(a);
    }
    let s = 2f64.powi(-(big.log2().round() as i32));
    let scaled = Mat::from_fn(n, m, |i, j| a[(i, j)] * s);
    let (values, vectors) = crate::math::linalg::symmetric_eigen(&scaled)?;
    Ok((
        faer::Col::from_fn(values.nrows(), |k| values[k] / s),
        vectors,
    ))
}

/// One entry per ordered pair: parallel steps add (log-sum-exp).
fn merge(entries: &[RateEntry]) -> MoptResult<Vec<RateEntry>> {
    let mut by_pair: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
    for e in entries {
        if e.ln_k.is_nan() || e.ln_k == f64::INFINITY {
            return Err(MoptError::config(format!(
                "generator: the rate constant from {} to {} is not finite",
                e.from, e.to
            )));
        }
        by_pair.entry((e.from, e.to)).or_default().push(e.ln_k);
    }
    Ok(by_pair
        .into_iter()
        .filter_map(|((from, to), ln)| {
            let ln_k = log_sum_exp(&ln);
            (ln_k > f64::NEG_INFINITY).then_some(RateEntry { from, to, ln_k })
        })
        .collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn fluxes_give_detailed_balance_and_parallel_steps_add() {
        let ln_pi = vec![0.0, -3.0, -1.0];
        let labels = vec!["A".into(), "B".into(), "C".into()];
        let g =
            Generator::from_fluxes(labels, ln_pi, &[(0, 1, -10.0), (1, 0, -12.0), (1, 2, -5.0)])
                .unwrap();
        // A -> B: (e^-10 + e^-12) / pi_A.
        let want = ((-10.0_f64).exp() + (-12.0_f64).exp()).ln();
        assert!((g.ln_rate(1, 0) - want).abs() < 1e-14);
        assert!((g.ln_rate(0, 1) - (want + 3.0)).abs() < 1e-14);
        assert!(g.detailed_balance_residual().unwrap() < 1e-13);
        let r = g.rates();
        let k = r.matrix();
        for j in 0..3 {
            assert!((0..3).map(|i| k[(i, j)]).sum::<f64>().abs() < 1e-15);
        }
        // K pi = 0.
        let pi: Vec<f64> = [0.0_f64, -3.0, -1.0].iter().map(|v| v.exp()).collect();
        for i in 0..3 {
            let s: f64 = (0..3).map(|j| k[(i, j)] * pi[j]).sum();
            assert!(s.abs() < 1e-17, "{s}");
        }
        assert!(Generator::from_matrix(&[vec![0.0, -1.0], vec![1.0, 0.0]]).is_err());
    }
}
