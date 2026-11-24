//! `propagate`: contract, then propagate the superstates exactly
//! (`kinetics_beyond_tst.md`, section 3.4):
//!
//! ```text
//! p_T(t) = N exp(G t) L p(0),   p_S(t) = C p_T(t),
//! closure A: G = D diag(1^T M)^-1, N = diag(1^T M)^-1;   closure B: G = D M^-1, N = M^-1
//! ```
//!
//! The cut is chosen per output time after the greedy order is known: among the prefixes
//! whose eliminated states have relaxed (`sigma_hat(K_SS) t >= ln(1 / tol)`), the one with the
//! smallest separation ratio `eps_hat = rho_hat(D) / sigma_hat(K_SS)` (the longer one on ties);
//! if that ratio exceeds `tol`, no state is eliminated and the solution is `exp(t K) p0`. The
//! error is of the order of `eps + exp(-sigma t)`, so the default `tol = 1e-8` keeps it below
//! the accuracy of the matrix exponential that `--validate` compares with. The superstate
//! generator is small and not stiff on the scale of `t`.
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue._solve_ode_rk4` (integrates `D` instead of `G`, from the
//! superstate population without the factor `N`; see `migration.md`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::rcmc::snapshots;
use super::{
    check_input, conservation, mat_vec, members_of, superstate_balance, Closure, Evolution,
    KineticSolution, KineticSolver, SolveDiagnostics, Superstates,
};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::contract::{greedy_order, GreedyOrder, TimeRule};
use crate::kinetics::expm::{exp_and_integral, expm, expm_metzler};

/// Contract and propagate.
#[derive(Debug, Clone, Copy)]
pub struct Propagate {
    tolerance: f64,
    closure: Closure,
}

impl Propagate {
    /// With the tolerance `tol` of the cut and the closure.
    pub fn new(tol: f64, closure: Closure) -> MoptResult<Self> {
        if !(tol > 0.0 && tol < 1.0) {
            return Err(MoptError::config(format!(
                "the tolerance of propagate must lie between 0 and 1 (got {tol})"
            )));
        }
        Ok(Self {
            tolerance: tol,
            closure,
        })
    }

    /// The number of eliminated states at `t`.
    pub fn cut(&self, order: &GreedyOrder, t: f64) -> usize {
        let need = (1.0 / self.tolerance).ln();
        let mut best: Option<(usize, f64)> = None;
        for (k, s) in order.steps.iter().enumerate() {
            let relaxed = s.sigma_hat * t;
            if relaxed.is_nan() || relaxed < need {
                continue;
            }
            if best.is_none_or(|(_, e)| s.eps_hat <= e) {
                best = Some((k + 1, s.eps_hat));
            }
        }
        match best {
            Some((k, e)) if e <= self.tolerance => k,
            _ => 0,
        }
    }
}

impl KineticSolver for Propagate {
    fn name(&self) -> &str {
        "propagate"
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
        let rates = g.rates();
        let order = greedy_order(&rates, TimeRule::Gershgorin, g.ln_pi.is_some())?;
        let cuts: Vec<usize> = times.iter().map(|&t| self.cut(&order, t)).collect();
        let snaps = snapshots(&rates, &order, &cuts)?;
        let full = rates.matrix();
        let mut populations = Vec::with_capacity(times.len());
        let mut diagnostics = Vec::with_capacity(times.len());
        let mut superstates = Vec::with_capacity(times.len());
        for ((&t, &k), c) in times.iter().zip(&cuts).zip(&snaps) {
            if k == 0 {
                let e = expm_metzler(&full, t)?;
                let p = mat_vec(&e, p0);
                let (cons, neg) = conservation(&p, p0);
                diagnostics.push(SolveDiagnostics {
                    time: t,
                    conservation: cons,
                    most_negative: neg,
                    notes: vec![
                        "no cut within the tolerance: the matrix exponential of the whole \
                         generator"
                            .to_string(),
                    ],
                    ..SolveDiagnostics::default()
                });
                superstates.push(None);
                populations.push(p);
                continue;
            }
            let step = order.steps[k - 1];
            let q0 = c.lump(p0);
            let nt = c.kept.len();
            let (gen, p_t, balance) = match self.closure {
                Closure::A => {
                    let gen = c.generator_a();
                    let e = expm_metzler(&gen, t)?;
                    let q = mat_vec(&e, &q0);
                    let p_t: Vec<f64> = q.iter().zip(&c.weights).map(|(q, w)| q / w).collect();
                    let balance = g.ln_pi.as_ref().map(|ln_pi| {
                        let kept: Vec<f64> = c.kept.iter().map(|&i| ln_pi[i]).collect();
                        superstate_balance(&gen, &kept, &c.weights)
                    });
                    (gen, p_t, balance)
                }
                Closure::B => {
                    let (gen, inv) = c.generator_b();
                    let e = match expm_metzler(&gen, t) {
                        Ok(e) => e,
                        Err(_) => expm(&Mat::from_fn(nt, nt, |i, j| gen[(i, j)] * t))?,
                    };
                    let q = mat_vec(&e, &q0);
                    (gen, mat_vec(&inv, &q), None)
                }
            };
            let p = c.unfold_kept(&p_t, n);
            let (cons, neg) = conservation(&p, p0);
            diagnostics.push(SolveDiagnostics {
                time: t,
                eliminated: k,
                eps_hat: Some(step.eps_hat),
                sigma_t: Some(step.sigma_hat * t),
                rho_t: Some(step.rho_hat * t),
                conservation: cons,
                detailed_balance: balance,
                most_negative: neg,
                notes: Vec::new(),
            });
            superstates.push(Some(Superstates {
                time: t,
                kept: c.kept.clone(),
                members: members_of(c),
                matrix: gen,
                matrix_name: match self.closure {
                    Closure::A => "superstate rate matrix G = D diag(1^T M)^-1 [s^-1] (closure A)",
                    Closure::B => "superstate rate matrix G = D M^-1 [s^-1] (closure B)",
                }
                .to_string(),
                contraction: Some(c.clone()),
            }));
            populations.push(p);
        }
        Ok(KineticSolution {
            solver: "propagate".to_string(),
            times: times.to_vec(),
            populations,
            diagnostics,
            superstates,
            notes: if rates.dropped > 0 {
                vec![format!(
                    "{} rate constant(s) more than exp(-700) below the largest were dropped",
                    rates.dropped
                )]
            } else {
                Vec::new()
            },
        })
    }

    fn residence(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        t: f64,
        _rng: &mut Rng,
    ) -> MoptResult<Vec<f64>> {
        // The contraction is not valid at the early times of the integral; the integral of the
        // whole generator is exact.
        let g = ev.generator()?;
        check_input(g.n, p0, &[t])?;
        Ok(exp_and_integral(&g.rates().matrix(), t, p0)?.1)
    }
}
