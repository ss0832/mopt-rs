//! `legacy_rcmc`: MultiOptPy's `rcmc.py` as it is, for comparison with its numbers. Its
//! differences from the published method are reproduced on purpose (`kinetics_beyond_tst.md`,
//! section 1.4, and `migration.md`):
//!
//! 1. the stop test comes after the contraction, so the step whose reference time exceeds the
//!    requested time is kept;
//! 2. with one state left (`rho(D) = 0`) the reference time is not updated, so the last step
//!    is never tested;
//! 3. the contracted equation `dp_T/dt = D p_T` is integrated (RK4, clipped and renormalized at
//!    every step) from the superstate population without the factor `diag(1^T M)^-1`;
//! 4. the fallback when RK4 would need more than 10^7 steps weights with
//!    `1 - K_TS K_SS^-1 1_S`;
//! 5. (the clamp of the rate constants is that of `--clamp legacy`, outside the solver);
//! 6. without a known start node the contracted equation starts from a uniform vector;
//! 8. `rcmc_K_contracted.csv` holds the Schur complement `D`;
//! 9. `K_SS` is factorized by LU with partial pivoting after every step.
//!
//! Item 7 (the mapper of MultiOptPy always uses electronic energies) belongs to the settings of
//! the mapper. The Schur complement is updated by subtraction with the diagonal rebuilt from
//! the column sums, and the superstate members are merged into the kept state that is most
//! strongly coupled to the eliminated one, as MultiOptPy does.
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop`, `RCMCQueue._solve_ode_rk4`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::{
    check_input, conservation, Evolution, KineticSolution, KineticSolver, SolveDiagnostics,
    Superstates,
};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};

/// RK4 needs at most this many steps (MultiOptPy's `_MAX_STEPS`).
const MAX_STEPS: f64 = 10_000_000.0;

/// MultiOptPy's RCMC.
#[derive(Debug, Clone, Copy)]
pub struct LegacyRcmc {
    start_known: bool,
}

impl LegacyRcmc {
    /// `start_known = false` reproduces MultiOptPy's mapper without a start node.
    pub fn new(start_known: bool) -> Self {
        Self { start_known }
    }
}

/// A dense LU factorization with partial pivoting (the first largest pivot of a column, as
/// LAPACK's `getrf`).
struct Lu {
    n: usize,
    a: Vec<f64>,
    piv: Vec<usize>,
}

impl Lu {
    fn new(m: &[f64], n: usize) -> Self {
        let mut a = m.to_vec();
        let mut piv = vec![0; n];
        for k in 0..n {
            let mut p = k;
            for i in (k + 1)..n {
                if a[i * n + k].abs() > a[p * n + k].abs() {
                    p = i;
                }
            }
            piv[k] = p;
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            let d = a[k * n + k];
            if d != 0.0 {
                for i in (k + 1)..n {
                    a[i * n + k] /= d;
                }
            }
            for i in (k + 1)..n {
                let l = a[i * n + k];
                if l != 0.0 {
                    for j in (k + 1)..n {
                        a[i * n + j] -= l * a[k * n + j];
                    }
                }
            }
        }
        Self { n, a, piv }
    }

    /// `A x = b`.
    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = b.to_vec();
        for k in 0..n {
            x.swap(k, self.piv[k]);
        }
        for i in 0..n {
            for k in 0..i {
                x[i] -= self.a[i * n + k] * x[k];
            }
        }
        for i in (0..n).rev() {
            for k in (i + 1)..n {
                x[i] -= self.a[i * n + k] * x[k];
            }
            x[i] /= self.a[i * n + i];
        }
        x
    }

    /// `A^T x = b`.
    fn solve_transposed(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = b.to_vec();
        // U^T y = b, L^T z = y, then undo the row exchanges.
        for i in 0..n {
            for k in 0..i {
                x[i] -= self.a[k * n + i] * x[k];
            }
            x[i] /= self.a[i * n + i];
        }
        for i in (0..n).rev() {
            for k in (i + 1)..n {
                x[i] -= self.a[k * n + i] * x[k];
            }
        }
        for k in (0..n).rev() {
            x.swap(k, self.piv[k]);
        }
        x
    }
}

/// `min(largest absolute row sum, largest absolute column sum)`.
fn gershgorin(d: &[f64], n: usize) -> f64 {
    let rows = (0..n)
        .map(|i| (0..n).map(|k| d[i * n + k].abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let cols = (0..n)
        .map(|k| (0..n).map(|i| d[i * n + k].abs()).sum::<f64>())
        .fold(0.0, f64::max);
    rows.min(cols)
}

/// `_solve_ode_rk4`: the population at `t_end` and whether RK4 was within its step limit.
fn rk4(d: &[f64], n: usize, p0: &[f64], t_end: f64) -> (Vec<f64>, bool) {
    if t_end < 1e-16 {
        return (p0.to_vec(), true);
    }
    let rho = gershgorin(d, n);
    if rho < 1e-16 {
        return (p0.to_vec(), true);
    }
    let dt_max = 2.5 / rho;
    let steps = (t_end / dt_max).ceil();
    if steps > MAX_STEPS {
        return (p0.to_vec(), false);
    }
    let steps = steps as usize;
    let dt = t_end / steps as f64;
    let apply = |y: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| (0..n).map(|k| d[i * n + k] * y[k]).sum())
            .collect()
    };
    let mut y = p0.to_vec();
    for _ in 0..steps {
        let k1 = apply(&y);
        let y2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * dt * k1[i]).collect();
        let k2 = apply(&y2);
        let y3: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * dt * k2[i]).collect();
        let k3 = apply(&y3);
        let y4: Vec<f64> = (0..n).map(|i| y[i] + dt * k3[i]).collect();
        let k4 = apply(&y4);
        for i in 0..n {
            y[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            y[i] = y[i].max(0.0);
        }
        let total: f64 = y.iter().sum();
        if total > 0.0 {
            for v in &mut y {
                *v /= total;
            }
        }
    }
    (y, true)
}

/// What one run of MultiOptPy's `pop()` gives.
#[derive(Debug, Clone)]
pub struct LegacyResult {
    /// The populations.
    pub q: Vec<f64>,
    /// The kept states `T` at termination, in MultiOptPy's order.
    pub kept: Vec<usize>,
    /// The eliminated states `S`, in order.
    pub eliminated: Vec<usize>,
    /// `D` at termination (`|T| x |T|`).
    pub d: Vec<f64>,
    /// The members of every kept state, in MultiOptPy's order.
    pub members: Vec<Vec<usize>>,
    /// The last reference time.
    pub time: f64,
    /// Whether the RK4 fallback was used.
    pub fallback: bool,
}

/// MultiOptPy's `pop()` on the matrix `k` (dense, `k[i * n + j]` rate from `j` to `i`, with
/// its diagonal) from the node `start` (`None`: not known, the vector `p` starts at
/// `lowest`).
pub fn legacy_pop(
    k: &[f64],
    n: usize,
    start: Option<usize>,
    lowest: usize,
    reaction_time: f64,
) -> LegacyResult {
    let mut p = vec![0.0; n];
    p[start.unwrap_or(lowest)] = 1.0;
    let mut s: Vec<usize> = Vec::new();
    let mut t: Vec<usize> = (0..n).collect();
    let mut d = k.to_vec();
    let mut nt = n;
    let mut time_current = 0.0;
    let mut members: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    while t.len() > 1 {
        let mut j_local = 0;
        for i in 1..nt {
            if d[i * nt + i].abs() > d[j_local * nt + j_local].abs() {
                j_local = i;
            }
        }
        let j_global = t[j_local];
        let d_jj = d[j_local * nt + j_local];
        if d_jj.abs() < 1e-30 {
            break;
        }
        let keep: Vec<usize> = (0..nt).filter(|&i| i != j_local).collect();
        let m = keep.len();
        let mut new = vec![0.0; m * m];
        for (a, &i) in keep.iter().enumerate() {
            for (b, &kk) in keep.iter().enumerate() {
                new[a * m + b] = d[i * nt + kk] - d[i * nt + j_local] * d[j_local * nt + kk] / d_jj;
            }
        }
        for b in 0..m {
            let col: f64 = (0..m).map(|a| new[a * m + b]).sum::<f64>() - new[b * m + b];
            new[b * m + b] = -col;
        }
        // The kept state most strongly coupled to j absorbs its members.
        let coupling: Vec<f64> = keep.iter().map(|&i| d[i * nt + j_local].abs()).collect();
        let max = coupling.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut absorb = 0;
        if max > 0.0 {
            for (a, c) in coupling.iter().enumerate() {
                if *c > coupling[absorb] {
                    absorb = a;
                }
            }
        }
        let remaining: Vec<usize> = keep.iter().map(|&i| t[i]).collect();
        let absorb_global = remaining[absorb];
        let moved = members[j_global].take().unwrap_or_default();
        if let Some(list) = members[absorb_global].as_mut() {
            list.extend(moved);
        }
        s.push(j_global);
        t.remove(j_local);
        d = new;
        nt = m;
        // sigma(K_SS) from one LU of -K_SS.
        let ns = s.len();
        let neg: Vec<f64> = (0..ns * ns)
            .map(|x| -k[s[x / ns] * n + s[x % ns]])
            .collect();
        let lu = Lu::new(&neg, ns);
        let ones = vec![1.0; ns];
        let a = lu.solve(&ones);
        let b = lu.solve_transposed(&ones);
        let max_a = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max_b = b.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let rho_inv = if max_b < max_a { max_b } else { max_a };
        let sigma = if rho_inv > 0.0 { 1.0 / rho_inv } else { 1e-30 };
        let rho_d = gershgorin(&d, nt);
        if sigma > 0.0 && rho_d > 0.0 {
            time_current = std::f64::consts::LN_2 / (sigma * rho_d).sqrt();
        }
        if time_current > reaction_time {
            break;
        }
    }
    let mut q;
    let mut fallback = false;
    if !s.is_empty() && !t.is_empty() {
        let (ns, nt) = (s.len(), t.len());
        let k_ss: Vec<f64> = (0..ns * ns).map(|x| k[s[x / ns] * n + s[x % ns]]).collect();
        let k_st = |a: usize, b: usize| k[s[a] * n + t[b]];
        let k_ts = |a: usize, b: usize| k[t[a] * n + s[b]];
        let lu = Lu::new(&k_ss, ns);
        let mut p_init = vec![0.0; nt];
        match start.map(|g| {
            (
                t.iter().position(|&x| x == g),
                s.iter().position(|&x| x == g),
            )
        }) {
            Some((Some(a), _)) => p_init[a] = 1.0,
            Some((None, Some(pos))) => {
                let mut e = vec![0.0; ns];
                e[pos] = 1.0;
                let x = lu.solve(&e);
                for (a, v) in p_init.iter_mut().enumerate() {
                    *v = (-(0..ns).map(|b| k_ts(a, b) * x[b]).sum::<f64>()).max(0.0);
                }
                let total: f64 = p_init.iter().sum();
                if total > 0.0 {
                    for v in &mut p_init {
                        *v /= total;
                    }
                } else {
                    p_init = vec![1.0 / nt as f64; nt];
                }
            }
            _ => p_init = vec![1.0 / nt as f64; nt],
        }
        // X_ST = K_SS^-1 K_ST.
        let mut x_st = vec![0.0; ns * nt];
        for b in 0..nt {
            let col: Vec<f64> = (0..ns).map(|a| k_st(a, b)).collect();
            let x = lu.solve(&col);
            for a in 0..ns {
                x_st[a * nt + b] = x[a];
            }
        }
        let (mut q_t, reliable) = rk4(&d, nt, &p_init, reaction_time);
        if !reliable {
            fallback = true;
            let p_s: Vec<f64> = s.iter().map(|&g| p[g]).collect();
            let x_ps = lu.solve(&p_s);
            let x_ones = lu.solve(&vec![1.0; ns]);
            q_t = (0..nt)
                .map(|a| {
                    let v = 1.0 - (0..ns).map(|b| k_ts(a, b) * x_ones[b]).sum::<f64>();
                    let v = if v.abs() > 1e-16 { v } else { 1e-16 };
                    (p_init[a] - (0..ns).map(|b| k_ts(a, b) * x_ps[b]).sum::<f64>()) / v
                })
                .collect();
        }
        let q_s: Vec<f64> = (0..ns)
            .map(|a| -(0..nt).map(|b| x_st[a * nt + b] * q_t[b]).sum::<f64>())
            .collect();
        q = vec![0.0; n];
        for (b, &g) in t.iter().enumerate() {
            q[g] = q_t[b].max(0.0);
        }
        for (a, &g) in s.iter().enumerate() {
            q[g] = q_s[a].max(0.0);
        }
        let total: f64 = q.iter().sum();
        if total > 0.0 {
            for v in &mut q {
                *v /= total;
            }
        }
        if q.iter().any(|v| !v.is_finite()) {
            q = p.clone();
        }
    } else {
        q = p.clone();
    }
    let kept_members = t
        .iter()
        .map(|&g| members[g].clone().unwrap_or_else(|| vec![g]))
        .collect();
    LegacyResult {
        q,
        kept: t,
        eliminated: s,
        d,
        members: kept_members,
        time: time_current,
        fallback,
    }
}

impl KineticSolver for LegacyRcmc {
    fn name(&self) -> &str {
        "legacy_rcmc"
    }

    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        _rng: &mut Rng,
    ) -> MoptResult<KineticSolution> {
        let g = ev.generator()?;
        check_input(g.n, p0, times)?;
        let n = g.n;
        let ones: Vec<usize> = (0..n).filter(|&i| p0[i] > 0.0).collect();
        let start =
            match ones[..] {
                [s] if (p0[s] - 1.0).abs() < 1e-12 => s,
                _ => return Err(MoptError::config(
                    "legacy_rcmc reproduces MultiOptPy, which starts from one node with all of \
                     the population; give one start node",
                )),
            };
        let rates = g.rates();
        let km = rates.matrix();
        let k: Vec<f64> = (0..n * n).map(|x| km[(x / n, x % n)]).collect();
        let mut populations = Vec::new();
        let mut diagnostics = Vec::new();
        let mut superstates = Vec::new();
        for &t in times {
            let r = legacy_pop(&k, n, self.start_known.then_some(start), start, t);
            let (cons, neg) = conservation(&r.q, p0);
            let mut notes = vec![format!(
                "MultiOptPy's RCMC: {} state(s) eliminated, last reference time {:.6e} s",
                r.eliminated.len(),
                r.time
            )];
            if r.fallback {
                notes.push(
                    "RK4 would have needed more than 1e7 steps: MultiOptPy's algebraic fallback \
                     was used"
                        .to_string(),
                );
            }
            diagnostics.push(SolveDiagnostics {
                time: t,
                eliminated: r.eliminated.len(),
                conservation: cons,
                most_negative: neg,
                notes,
                ..SolveDiagnostics::default()
            });
            let nt = r.kept.len();
            superstates.push(Some(Superstates {
                time: t,
                kept: r.kept.clone(),
                members: r.members.clone(),
                matrix: Mat::from_fn(nt, nt, |i, j| r.d[i * nt + j]),
                matrix_name: "RCMC contracted super-state rate matrix D [s^-1]".to_string(),
                contraction: None,
            }));
            populations.push(r.q);
        }
        Ok(KineticSolution {
            solver: "legacy_rcmc".to_string(),
            times: times.to_vec(),
            populations,
            diagnostics,
            superstates,
            notes: Vec::new(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_lu_solves_both_systems() {
        let a = [4.0, -1.0, 0.5, 2.0, -3.0, 1.0, 0.1, 0.7, 5.0];
        let lu = Lu::new(&a, 3);
        let b = [1.0, 2.0, 3.0];
        let x = lu.solve(&b);
        let y = lu.solve_transposed(&b);
        for i in 0..3 {
            let ax: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            let aty: f64 = (0..3).map(|j| a[j * 3 + i] * y[j]).sum();
            assert!((ax - b[i]).abs() < 1e-14 && (aty - b[i]).abs() < 1e-14);
        }
    }
}
