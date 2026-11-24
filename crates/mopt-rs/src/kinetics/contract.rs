//! Contraction of a generator by the elimination of states, written so that no two numbers of
//! the same sign are subtracted (the Grassmann-Taksar-Heyman scheme; Oper. Res. 1985, 33,
//! 1107), with the quantities of RCMC (Iwata, Oki, Sakaue, SIAM J. Sci. Comput. 2026, 48,
//! A261; `kinetics_beyond_tst.md`, section 3.4):
//!
//! ```text
//! B = -K_TS K_SS^-1   (|T| x |S|): where the population of an eliminated state enters T first
//! C = -K_SS^-1 K_ST   (|S| x |T|): quasi-steady population of an eliminated state per kept one
//! D = K_TT + K_TS C   (Schur complement), M = I + B C, 1^T M = 1 + (1^T B) C
//! ```
//!
//! One step (state `j` out of the kept set, the other kept states `i`, `k`):
//!
//! ```text
//! out      = sum_{i != j} D[i][j] + loss[j]          (rebuilt from the column, never accumulated)
//! D[i][k] += D[i][j] D[j][k] / out                    (non-negative terms only)
//! C[s][k] += C[s][j] D[j][k] / out, C[j][k] = D[j][k] / out
//! B[k][s] += D[k][j] / out B[j][s], B[k][j] = D[k][j] / out
//! ```
//!
//! The diagonal of `D` is never stored: it is `-(sum of the column) - loss`. The row and the
//! column of every pivot are kept; they are the triangular factors of `-K_SS = L U` in the
//! order of elimination, so `(-K_SS)^-1 1` and `(-K_SS)^-T 1` (the bounds of ref. \[4\] on
//! `sigma(K_SS)`) need two triangular solves with additions only, for every prefix of the
//! order. `K_SS` is never factorized by a general LU.
//!
//! The greedy order eliminates the state with the largest `|D[j][j]|` (the shortest lifetime)
//! first, ties broken by the lower index so that the result does not depend on the storage
//! order; it stops when every kept state has `out = 0` (one state per closed class).
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop` (Schur-complement updates by subtraction with the
//! diagonal rebuilt from the column sums, and an LU factorization of `K_SS` after every step).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::generator::Rates;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::generator::scaled_symmetric_eigen;
use crate::math::ad::Scalar;

/// The reference time of a step of RCMC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeRule {
    /// `ln 2 / sqrt(sigma_hat(K_SS) rho_hat(D))` with the Gershgorin-type bounds of ref. \[4\]
    /// (MultiOptPy's rule).
    Gershgorin,
    /// `ln 2 / sqrt(sigma(K_SS) rho(D))` with the eigenvalues (detailed balance needed).
    Eigen,
    /// `1 / |D[j][j]|` of the eliminated state (the original papers).
    Diag,
}

impl TimeRule {
    /// `gershgorin`, `eigen`, or `diag`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "gershgorin" => Ok(Self::Gershgorin),
            "eigen" => Ok(Self::Eigen),
            "diag" => Ok(Self::Diag),
            other => Err(MoptError::parse(format!(
                "unknown time rule '{other}' (gershgorin, eigen, diag)"
            ))),
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gershgorin => "gershgorin",
            Self::Eigen => "eigen",
            Self::Diag => "diag",
        }
    }
}

/// The incremental elimination, generic over the scalar (`f64`, or dual numbers for the
/// derivatives: the choices of the pivots use the real parts only).
#[derive(Debug, Clone)]
pub struct Eliminator<T = f64> {
    n: usize,
    alive: Vec<bool>,
    order: Vec<usize>,
    d: Vec<T>,
    loss: Vec<T>,
    c: Vec<T>,
    b: Vec<T>,
    rows: Vec<Vec<T>>,
    cols: Vec<Vec<T>>,
    outs: Vec<T>,
}

impl Eliminator<f64> {
    /// Start from the rates of a generator (no state eliminated).
    pub fn new(rates: &Rates) -> Self {
        Self::from_weights(rates.n, rates.w.clone())
    }

    /// `(-K_SS)^-1 1` and `(-K_SS)^-T 1` of the eliminated states (in the order of
    /// elimination), from the triangular factors, by additions only.
    #[allow(clippy::needless_range_loop)] // triangular solves read best with their indices
    pub fn inverse_sums(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.order.len();
        let s = &self.order;
        let mut y = vec![0.0; m];
        for l in 0..m {
            let mut v = 1.0;
            for k in 0..l {
                v += self.cols[k][s[l]] / self.outs[k] * y[k];
            }
            y[l] = v;
        }
        let mut x = vec![0.0; m];
        for k in (0..m).rev() {
            let mut v = y[k];
            for l in (k + 1)..m {
                v += self.rows[k][s[l]] * x[l];
            }
            x[k] = v / self.outs[k];
        }
        let mut w = vec![0.0; m];
        for k in 0..m {
            let mut v = 1.0;
            for l in 0..k {
                v += self.rows[l][s[k]] * w[l];
            }
            w[k] = v / self.outs[k];
        }
        let mut z = vec![0.0; m];
        for k in (0..m).rev() {
            let mut v = w[k];
            for l in (k + 1)..m {
                v += self.cols[k][s[l]] / self.outs[k] * z[l];
            }
            z[k] = v;
        }
        (x, z)
    }

    /// `sigma_hat(K_SS) = 1 / min(max (-K_SS)^-1 1, max (-K_SS)^-T 1)` (infinite without an
    /// eliminated state).
    pub fn sigma_hat(&self) -> f64 {
        if self.order.is_empty() {
            return f64::INFINITY;
        }
        let (x, z) = self.inverse_sums();
        let mx = x.iter().copied().fold(0.0, f64::max);
        let mz = z.iter().copied().fold(0.0, f64::max);
        let rho = mx.min(mz);
        if rho > 0.0 {
            1.0 / rho
        } else {
            0.0
        }
    }

    /// `rho_hat(D) = min(largest absolute row sum, largest absolute column sum)` of the
    /// Schur complement on the kept states.
    pub fn rho_hat(&self) -> f64 {
        let n = self.n;
        let kept = self.kept();
        let outs: Vec<f64> = kept.iter().map(|&j| self.out(j)).collect();
        let mut max_row: f64 = 0.0;
        let mut max_col: f64 = 0.0;
        for (a, &i) in kept.iter().enumerate() {
            let mut row = outs[a];
            let mut col = outs[a];
            for &k in &kept {
                if k != i {
                    row += self.d[i * n + k];
                    col += self.d[k * n + i];
                }
            }
            max_row = max_row.max(row);
            max_col = max_col.max(col);
        }
        max_row.min(max_col)
    }
}

impl<T: Scalar> Eliminator<T> {
    /// Start from the rates `w` (`w[i * n + j]`: from `j` to `i`; the diagonal is ignored).
    pub fn from_weights(n: usize, w: Vec<T>) -> Self {
        Self {
            n,
            alive: vec![true; n],
            order: Vec::new(),
            d: w,
            loss: vec![T::zero(); n],
            c: vec![T::zero(); n * n],
            b: vec![T::zero(); n * n],
            rows: Vec::new(),
            cols: Vec::new(),
            outs: Vec::new(),
        }
    }

    /// The states eliminated so far, in order.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// The kept states, ascending.
    pub fn kept(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.alive[i]).collect()
    }

    /// `-D[j][j]` of a kept state.
    pub fn out(&self, j: usize) -> T {
        let n = self.n;
        let mut s = self.loss[j];
        for i in 0..n {
            if i != j && self.alive[i] {
                s += self.d[i * n + j];
            }
        }
        s
    }

    /// The kept state with the largest `out` (the lower index on ties), or `None` when at most
    /// one state is kept or every kept state has `out = 0`.
    pub fn pivot(&self) -> Option<usize> {
        if self.alive.iter().filter(|a| **a).count() <= 1 {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for j in 0..self.n {
            if !self.alive[j] {
                continue;
            }
            let o = self.out(j).val();
            if best.is_none_or(|(_, b)| o > b) {
                best = Some((j, o));
            }
        }
        best.filter(|(_, o)| *o > 0.0).map(|(j, _)| j)
    }

    /// Eliminate the kept state `j`.
    pub fn eliminate(&mut self, j: usize) -> MoptResult<()> {
        let n = self.n;
        if j >= n || !self.alive[j] {
            return Err(MoptError::config(format!(
                "contraction: state {j} is not a kept state"
            )));
        }
        let out = self.out(j);
        if out.val().is_nan() || out.val() <= 0.0 {
            return Err(MoptError::config(format!(
                "contraction: state {j} has no way out and cannot be eliminated"
            )));
        }
        let zero = T::zero();
        let others: Vec<usize> = (0..n).filter(|&i| i != j && self.alive[i]).collect();
        let mut row = vec![zero; n];
        let mut col = vec![zero; n];
        for &i in &others {
            row[i] = self.d[j * n + i];
            col[i] = self.d[i * n + j];
        }
        // Schur complement.
        for &i in &others {
            let ci = col[i] / out;
            if ci == zero {
                continue;
            }
            for &k in &others {
                if k != i && row[k] != zero {
                    self.d[i * n + k] += ci * row[k];
                }
            }
        }
        let lj = self.loss[j];
        if lj != zero {
            for &k in &others {
                self.loss[k] += lj * row[k] / out;
            }
        }
        // C of the states eliminated before, and of j.
        for &s in &self.order {
            let csj = self.c[s * n + j];
            if csj != zero {
                for &k in &others {
                    self.c[s * n + k] += csj * row[k] / out;
                }
            }
            self.c[s * n + j] = zero;
        }
        for &k in &others {
            self.c[j * n + k] = row[k] / out;
        }
        // B of the states eliminated before, and of j.
        for &s in &self.order {
            let bjs = self.b[j * n + s];
            if bjs != zero {
                for &k in &others {
                    self.b[k * n + s] += col[k] / out * bjs;
                }
            }
            self.b[j * n + s] = zero;
        }
        for &k in &others {
            self.b[k * n + j] = col[k] / out;
        }
        self.alive[j] = false;
        self.order.push(j);
        self.rows.push(row);
        self.cols.push(col);
        self.outs.push(out);
        Ok(())
    }

    /// The contraction as it stands.
    pub fn snapshot(&self) -> Contraction<T> {
        let n = self.n;
        let zero = T::zero();
        let kept = self.kept();
        let elim = self.order.clone();
        let (nt, ns) = (kept.len(), elim.len());
        let mut d = vec![zero; nt * nt];
        for (a, &i) in kept.iter().enumerate() {
            for (c, &k) in kept.iter().enumerate() {
                if a != c {
                    d[a * nt + c] = self.d[i * n + k];
                }
            }
        }
        for (c, &k) in kept.iter().enumerate() {
            d[c * nt + c] = -self.out(k);
        }
        let mut b = vec![zero; nt * ns];
        let mut cm = vec![zero; ns * nt];
        for (a, &t) in kept.iter().enumerate() {
            for (p, &s) in elim.iter().enumerate() {
                b[a * ns + p] = self.b[t * n + s];
                cm[p * nt + a] = self.c[s * n + t];
            }
        }
        let b_sums: Vec<T> = (0..ns)
            .map(|p| (0..nt).fold(zero, |acc, a| acc + b[a * ns + p]))
            .collect();
        let weights = (0..nt)
            .map(|a| (0..ns).fold(T::one(), |acc, p| acc + b_sums[p] * cm[p * nt + a]))
            .collect();
        Contraction {
            kept,
            eliminated: elim,
            d,
            b,
            c: cm,
            weights,
        }
    }
}

/// A contraction: the kept and the eliminated states and the blocks of RCMC.
#[derive(Debug, Clone, PartialEq)]
pub struct Contraction<T = f64> {
    /// The kept states (indices of the generator), ascending.
    pub kept: Vec<usize>,
    /// The eliminated states, in the order of elimination.
    pub eliminated: Vec<usize>,
    /// `D` (`|T| x |T|`, row-major, with its diagonal).
    pub d: Vec<T>,
    /// `B` (`|T| x |S|`, row-major).
    pub b: Vec<T>,
    /// `C` (`|S| x |T|`, row-major).
    pub c: Vec<T>,
    /// `1^T M` (`|T|`): `pi_t (1^T M)_t` is the equilibrium population of superstate `t`.
    pub weights: Vec<T>,
}

impl<T: Scalar> Contraction<T> {
    /// `q = p_T + B p_S` (the lumped populations, exact at every time).
    pub fn lump(&self, p: &[T]) -> Vec<T> {
        let (nt, ns) = (self.kept.len(), self.eliminated.len());
        (0..nt)
            .map(|a| {
                (0..ns).fold(p[self.kept[a]], |acc, q| {
                    acc + self.b[a * ns + q] * p[self.eliminated[q]]
                })
            })
            .collect()
    }

    /// The populations of all `n` states from the populations `p_t` of the kept states:
    /// `p_S = C p_T`.
    pub fn unfold_kept(&self, p_t: &[T], n: usize) -> Vec<T> {
        let (nt, ns) = (self.kept.len(), self.eliminated.len());
        let mut p = vec![T::zero(); n];
        for a in 0..nt {
            p[self.kept[a]] = p_t[a];
        }
        for q in 0..ns {
            p[self.eliminated[q]] =
                (0..nt).fold(T::zero(), |acc, a| acc + self.c[q * nt + a] * p_t[a]);
        }
        p
    }

    /// Closure A: `p_T = diag(1^T M)^-1 q`, `p_S = C p_T`.
    pub fn unfold_a(&self, q: &[T], n: usize) -> Vec<T> {
        let p_t: Vec<T> = q.iter().zip(&self.weights).map(|(q, w)| *q / *w).collect();
        self.unfold_kept(&p_t, n)
    }

    /// The superstate generator of closure A, `G = D diag(1^T M)^-1`, row-major.
    pub fn generator_a_dense(&self) -> Vec<T> {
        let nt = self.kept.len();
        (0..nt * nt)
            .map(|x| self.d[x] / self.weights[x % nt])
            .collect()
    }
}

impl Contraction<f64> {
    /// `D` as a matrix.
    pub fn d_matrix(&self) -> Mat<f64> {
        let nt = self.kept.len();
        Mat::from_fn(nt, nt, |i, k| self.d[i * nt + k])
    }

    /// The superstate generator of closure A, `G = D diag(1^T M)^-1`.
    pub fn generator_a(&self) -> Mat<f64> {
        let nt = self.kept.len();
        Mat::from_fn(nt, nt, |i, k| self.d[i * nt + k] / self.weights[k])
    }

    /// `M = I + B C`.
    pub fn m_matrix(&self) -> Mat<f64> {
        let (nt, ns) = (self.kept.len(), self.eliminated.len());
        Mat::from_fn(nt, nt, |i, k| {
            let bc: f64 = (0..ns)
                .map(|q| self.b[i * ns + q] * self.c[q * nt + k])
                .sum();
            bc + if i == k { 1.0 } else { 0.0 }
        })
    }

    /// The superstate generator of closure B, `G = D M^-1`, and `M^-1`.
    pub fn generator_b(&self) -> (Mat<f64>, Mat<f64>) {
        let nt = self.kept.len();
        let m = self.m_matrix();
        let lu = m.partial_piv_lu();
        let inv = faer::linalg::solvers::Solve::solve(&lu, Mat::<f64>::identity(nt, nt).as_ref());
        (self.d_matrix() * &inv, inv)
    }
}

/// One step of the greedy elimination with its reference times and bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    /// The eliminated state.
    pub state: usize,
    /// Its `|D[j][j]|` when it was eliminated.
    pub out: f64,
    /// `sigma_hat(K_SS)` after the step (Gershgorin-type bound).
    pub sigma_hat: f64,
    /// `rho_hat(D)` after the step.
    pub rho_hat: f64,
    /// The eigenvalues `sigma(K_SS)`, `rho(D)` (rule `eigen` only).
    pub eigen: Option<(f64, f64)>,
    /// The reference time of the rule (infinite when `rho(D)` is 0).
    pub t_ref: f64,
    /// `eps_hat = rho_hat(D) / sigma_hat(K_SS)`.
    pub eps_hat: f64,
}

/// The greedy elimination of a generator down to one state per closed class.
#[derive(Debug, Clone, PartialEq)]
pub struct GreedyOrder {
    /// The steps.
    pub steps: Vec<Step>,
    /// The rule of the reference times.
    pub rule: TimeRule,
    /// The states left at the end.
    pub kept: Vec<usize>,
}

impl GreedyOrder {
    /// The eliminated states of the first `k` steps.
    pub fn prefix(&self, k: usize) -> Vec<usize> {
        self.steps.iter().take(k).map(|s| s.state).collect()
    }

    /// The number of leading steps whose reference times do not exceed `t` (the steps of the
    /// published RCMC output at `t`: the loop stops at the first step beyond it).
    pub fn steps_until(&self, t: f64) -> usize {
        self.steps.iter().take_while(|s| s.t_ref <= t).count()
    }
}

/// Smallest eigenvalue of `-A_SS` and largest of `-D`, from the symmetrized matrices
/// `sqrt(A[i][j] A[j][i])` (similar to them under detailed balance).
fn eigen_bounds(rates: &Rates, e: &Eliminator) -> MoptResult<(f64, f64)> {
    let s = e.order();
    let m = s.len();
    let a = Mat::from_fn(m, m, |i, j| {
        if i == j {
            rates.out(s[i])
        } else {
            -(rates.get(s[i], s[j]) * rates.get(s[j], s[i])).sqrt()
        }
    });
    let (va, _) = scaled_symmetric_eigen(&a)?;
    let sigma = if m > 0 { va[0] } else { f64::INFINITY };
    let snap = e.snapshot();
    let nt = snap.kept.len();
    let dm = Mat::from_fn(nt, nt, |i, k| {
        if i == k {
            -snap.d[i * nt + k]
        } else {
            -(snap.d[i * nt + k] * snap.d[k * nt + i]).sqrt()
        }
    });
    let (vd, _) = scaled_symmetric_eigen(&dm)?;
    let rho = if nt > 0 {
        vd[nt - 1].max(-vd[0]).max(0.0)
    } else {
        0.0
    };
    Ok((sigma, rho))
}

fn reference_time(sigma: f64, rho: f64) -> f64 {
    if rho > 0.0 && sigma > 0.0 {
        std::f64::consts::LN_2 / (sigma * rho).sqrt()
    } else if sigma > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

/// The greedy order of `rates` with the reference times of `rule`; `detailed_balance` must
/// hold for the rule `eigen`.
pub fn greedy_order(
    rates: &Rates,
    rule: TimeRule,
    detailed_balance: bool,
) -> MoptResult<GreedyOrder> {
    if rule == TimeRule::Eigen && !detailed_balance {
        return Err(MoptError::config(
            "the time rule eigen needs a generator with detailed balance (the eigenvalues are \
             taken from the symmetrized matrices); use gershgorin or diag",
        ));
    }
    let mut e = Eliminator::new(rates);
    let mut steps = Vec::new();
    while let Some(j) = e.pivot() {
        let out = e.out(j);
        e.eliminate(j)?;
        let sigma_hat = e.sigma_hat();
        let rho_hat = e.rho_hat();
        let eigen = if rule == TimeRule::Eigen {
            Some(eigen_bounds(rates, &e)?)
        } else {
            None
        };
        let t_ref = match (rule, eigen) {
            (TimeRule::Diag, _) => 1.0 / out,
            (TimeRule::Eigen, Some((s, r))) => reference_time(s, r),
            _ => reference_time(sigma_hat, rho_hat),
        };
        steps.push(Step {
            state: j,
            out,
            sigma_hat,
            rho_hat,
            eigen,
            t_ref,
            eps_hat: if sigma_hat > 0.0 {
                rho_hat / sigma_hat
            } else {
                f64::INFINITY
            },
        });
    }
    Ok(GreedyOrder {
        steps,
        rule,
        kept: e.kept(),
    })
}

/// The contraction of `rates` with `eliminated` (in that order).
pub fn contract(rates: &Rates, eliminated: &[usize]) -> MoptResult<Contraction> {
    let mut e = Eliminator::new(rates);
    for &j in eliminated {
        e.eliminate(j)?;
    }
    Ok(e.snapshot())
}

/// The stationary distribution reached from `p0` (`t -> inf`): every closed class relaxes to
/// its equilibrium with the population that `p0` puts into it, from the full elimination.
pub fn stationary(rates: &Rates, p0: &[f64]) -> MoptResult<Vec<f64>> {
    let mut e = Eliminator::new(rates);
    while let Some(j) = e.pivot() {
        e.eliminate(j)?;
    }
    let snap = e.snapshot();
    let q = snap.lump(p0);
    Ok(snap.unfold_a(&q, rates.n))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core_infra::rng::Rng;
    use crate::kinetics::generator::Generator;

    fn random_rates(n: usize, seed: u64, balanced: bool) -> Rates {
        let mut rng = Rng::from_seed(seed);
        let mut k = vec![vec![0.0; n]; n];
        let ln_pi: Vec<f64> = (0..n).map(|_| rng.uniform(-8.0, 0.0)).collect();
        for i in 0..n {
            for j in 0..i {
                if rng.next_f64() < 0.6 || j + 1 == i {
                    let ln_f = rng.uniform(-14.0, 2.0);
                    if balanced {
                        k[i][j] = (ln_f - ln_pi[j]).exp();
                        k[j][i] = (ln_f - ln_pi[i]).exp();
                    } else {
                        k[i][j] = rng.uniform(-10.0, 4.0).exp();
                        k[j][i] = rng.uniform(-10.0, 4.0).exp();
                    }
                }
            }
        }
        Generator::from_matrix(&k).unwrap().rates()
    }

    #[test]
    fn the_blocks_satisfy_the_identities_of_the_contraction() {
        for (seed, balanced) in [(1, true), (2, false), (3, true), (4, false)] {
            let n = 9;
            let rates = random_rates(n, seed, balanced);
            let k = rates.matrix();
            let order = greedy_order(&rates, TimeRule::Gershgorin, balanced).unwrap();
            for m in [1, 3, 5, n - 1] {
                let c = contract(&rates, &order.prefix(m)).unwrap();
                let (t, s) = (&c.kept, &c.eliminated);
                let (nt, ns) = (t.len(), s.len());
                // L K = [D 0] with L = [I B].
                for (a, &ti) in t.iter().enumerate() {
                    for col in 0..n {
                        let lk = k[(ti, col)]
                            + (0..ns)
                                .map(|q| c.b[a * ns + q] * k[(s[q], col)])
                                .sum::<f64>();
                        let want = match t.iter().position(|&x| x == col) {
                            Some(b) => c.d[a * nt + b],
                            None => 0.0,
                        };
                        let scale = k[(col, col)].abs().max(1e-300);
                        assert!(
                            (lk - want).abs() <= 1e-12 * scale,
                            "{seed} {m}: {lk} {want}"
                        );
                    }
                }
                // K_SS C + K_ST = 0.
                for (p, &si) in s.iter().enumerate() {
                    for (a, &ti) in t.iter().enumerate() {
                        let r = k[(si, ti)]
                            + (0..ns)
                                .map(|q| k[(si, s[q])] * c.c[q * nt + a])
                                .sum::<f64>();
                        let scale = k[(si, si)].abs() * (1.0 + c.c[p * nt + a]);
                        assert!(r.abs() <= 1e-12 * scale, "{seed} {m}: {r}");
                    }
                }
                // 1^T G = 0 and the unfolded populations keep the total.
                let g = c.generator_a();
                for col in 0..nt {
                    let sum: f64 = (0..nt).map(|i| g[(i, col)]).sum();
                    assert!(sum.abs() <= 1e-12 * g[(col, col)].abs().max(1e-300));
                }
                let p0: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
                let total: f64 = p0.iter().sum();
                let q = c.lump(&p0);
                assert!((q.iter().sum::<f64>() - total).abs() < 1e-12 * total);
                let p = c.unfold_a(&q, n);
                assert!((p.iter().sum::<f64>() - total).abs() < 1e-12 * total);
                // B, C, 1^T M are not negative.
                assert!(c.b.iter().chain(&c.c).all(|v| *v >= 0.0));
                assert!(c.weights.iter().all(|w| *w >= 1.0));
            }
            // The factors give (-K_SS)^-1 1 as a dense solve does.
            let mut e = Eliminator::new(&rates);
            for &j in &order.prefix(5) {
                e.eliminate(j).unwrap();
            }
            let s = e.order().to_vec();
            let a = Mat::from_fn(5, 5, |i, j| -k[(s[i], s[j])]);
            let lu = a.partial_piv_lu();
            let ones = Mat::from_fn(5, 1, |_, _| 1.0);
            let x = faer::linalg::solvers::Solve::solve(&lu, ones.as_ref());
            let (got, _) = e.inverse_sums();
            for i in 0..5 {
                assert!(
                    (got[i] / x[(i, 0)] - 1.0).abs() < 1e-10,
                    "{seed}: {} {}",
                    got[i],
                    x[(i, 0)]
                );
            }
        }
    }

    #[test]
    fn the_full_elimination_gives_the_boltzmann_distribution_over_500_rt() {
        // Energies (in RT) spread over 500, saddle points above both ends, a sparse network.
        let mut rng = Rng::from_seed(11);
        let n = 30;
        let g: Vec<f64> = (0..n).map(|i| 500.0 * i as f64 / (n - 1) as f64).collect();
        let mut fluxes = Vec::new();
        for i in 1..n {
            let j = rng.gen_range_usize(i);
            let ts = g[i].max(g[j]) + rng.uniform(1.0, 30.0);
            fluxes.push((i, j, -ts));
            if i > 2 && rng.next_f64() < 0.5 {
                let k = rng.gen_range_usize(i - 1);
                let ts = g[i].max(g[k]) + rng.uniform(1.0, 30.0);
                fluxes.push((i, k, -ts));
            }
        }
        let ln_pi: Vec<f64> = g.iter().map(|x| -x).collect();
        let labels = (0..n).map(|i| format!("S{i}")).collect();
        let gen = Generator::from_fluxes(labels, ln_pi.clone(), &fluxes).unwrap();
        let rates = gen.rates();
        assert_eq!(rates.dropped, 0);
        let mut p0 = vec![0.0; n];
        p0[n - 1] = 1.0;
        let p = stationary(&rates, &p0).unwrap();
        let z = crate::math::stable::log_sum_exp(&ln_pi);
        for i in 0..n {
            let want = (ln_pi[i] - z).exp();
            assert!((p[i] / want - 1.0).abs() < 1e-12, "{i}: {} {want}", p[i]);
        }
    }
}
