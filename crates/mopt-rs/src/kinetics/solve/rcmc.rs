//! `rcmc`: the published rate constant matrix contraction, Type A (Sumiya et al., J. Phys.
//! Chem. A 2015, 119, 11641; Iwata, Oki, Sakaue, SIAM J. Sci. Comput. 2026, 48, A261).
//!
//! States are eliminated in the greedy order; step `k` has the reference time `t(k)` of the
//! rule (`gershgorin`: `ln 2 / sqrt(sigma_hat(K_SS) rho_hat(D))`, `eigen`: the same with the
//! eigenvalues, `diag`: `1 / |D[j][j]|`). The output at the time `t` is the snapshot of the
//! last step before the first one whose reference time exceeds `t` (none: `p0`):
//!
//! ```text
//! q_T = diag(1^T M)^-1 (p_T + B p_S),    q_S = C q_T
//! ```
//!
//! No differential equation is integrated: the eliminated states are equilibrated with their
//! neighbourhood, the kept ones have not reacted. A step after which nothing is left to react
//! (`rho(D) = 0`, e.g. the last step of a connected network) has an infinite reference time
//! under `gershgorin` and `eigen`, so its equilibrium is never output by those rules.
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop` (see `legacy_rcmc` for its differences).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::{
    check_input, conservation, members_of, superstate_balance, Evolution, KineticSolution,
    KineticSolver, SolveDiagnostics, Superstates,
};
use crate::core_infra::rng::Rng;
use crate::error::MoptResult;
use crate::kinetics::contract::{greedy_order, Contraction, Eliminator, GreedyOrder, TimeRule};
use crate::kinetics::generator::Rates;

/// The published RCMC of Type A.
#[derive(Debug, Clone, Copy)]
pub struct Rcmc {
    rule: TimeRule,
}

impl Rcmc {
    /// With the reference times of `rule`.
    pub fn new(rule: TimeRule) -> Self {
        Self { rule }
    }
}

/// Snapshots of the contraction after `prefixes[k]` steps of `order` (one pass).
pub fn snapshots(
    rates: &Rates,
    order: &GreedyOrder,
    prefixes: &[usize],
) -> MoptResult<Vec<Contraction>> {
    let mut wanted: Vec<usize> = prefixes.to_vec();
    wanted.sort_unstable();
    wanted.dedup();
    let mut e = Eliminator::new(rates);
    let mut done = Vec::with_capacity(wanted.len());
    for &k in &wanted {
        while e.order().len() < k {
            let j = order.steps[e.order().len()].state;
            e.eliminate(j)?;
        }
        done.push((k, e.snapshot()));
    }
    Ok(prefixes
        .iter()
        .filter_map(|k| done.iter().find(|(m, _)| m == k).map(|(_, c)| c.clone()))
        .collect())
}

impl KineticSolver for Rcmc {
    fn name(&self) -> &str {
        "rcmc"
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
        let rates = g.rates();
        let order = greedy_order(&rates, self.rule, g.ln_pi.is_some())?;
        let prefixes: Vec<usize> = times.iter().map(|&t| order.steps_until(t)).collect();
        let snaps = snapshots(&rates, &order, &prefixes)?;
        let mut populations = Vec::with_capacity(times.len());
        let mut diagnostics = Vec::with_capacity(times.len());
        let mut superstates = Vec::with_capacity(times.len());
        for ((&t, &k), c) in times.iter().zip(&prefixes).zip(&snaps) {
            let q = c.lump(p0);
            let p = c.unfold_a(&q, g.n);
            let (cons, neg) = conservation(&p, p0);
            let step = (k > 0).then(|| order.steps[k - 1]);
            let gen = c.generator_a();
            let balance = g.ln_pi.as_ref().map(|ln_pi| {
                let kept: Vec<f64> = c.kept.iter().map(|&i| ln_pi[i]).collect();
                superstate_balance(&gen, &kept, &c.weights)
            });
            diagnostics.push(SolveDiagnostics {
                time: t,
                eliminated: k,
                eps_hat: step.map(|s| s.eps_hat),
                sigma_t: step.map(|s| s.sigma_hat * t),
                rho_t: step.map(|s| s.rho_hat * t),
                conservation: cons,
                detailed_balance: balance,
                most_negative: neg,
                notes: vec![format!(
                    "snapshot after {k} step(s) of {} (time rule {}{})",
                    order.steps.len(),
                    self.rule.name(),
                    step.map_or(String::new(), |s| format!(
                        ", reference time {:.6e} s",
                        s.t_ref
                    ))
                )],
            });
            superstates.push(Some(Superstates {
                time: t,
                kept: c.kept.clone(),
                members: members_of(c),
                matrix: gen,
                matrix_name: "superstate rate matrix G = D diag(1^T M)^-1 [s^-1] (closure A, \
                              the RCMC superstates)"
                    .to_string(),
                contraction: Some(c.clone()),
            }));
            populations.push(p);
        }
        Ok(KineticSolution {
            solver: "rcmc".to_string(),
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
}
