//! The order of every step judged automatically: whether a bimolecular step `a + S <-> b` may
//! be taken as pseudo-first order at the concentration of `S` (the linear model `canonical`,
//! whose solvers are RCMC and its contraction) or is of second order because `S` changes over
//! the reaction (the model `mass_action`; `kinetics_beyond_tst.md`, section 2.6: "any step whose
//! rate depends on a concentration that changes by more than a few per cent").
//!
//! With the states together at the concentration `c_tot`, a species `S` held by `n_S(g)` free
//! molecules relative to its component in state `g` (the composition of the rate assembly)
//! changes by at most
//!
//! ```text
//! |delta [S]| <= c_tot (max_g n_S(g) - min_g n_S(g))
//! ```
//!
//! whatever the kinetics does, so its relative change is bounded by that over `[S]0`. A step
//! whose partner stays within `tolerance` (1e-2 by default, `--order-tolerance`) is pseudo-first
//! order; one whose partner may change more, starts at zero (a product released), or is turned
//! over by a cycle of steps (no composition: a catalytic cycle) is of second order. A network
//! whose bimolecular steps are all pseudo-first order is linear and goes to `canonical`
//! (`--model auto`); otherwise to `mass_action`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::rate::RateSet;

/// The default largest relative change of a partner of a pseudo-first-order step.
pub const DEFAULT_ORDER_TOLERANCE: f64 = 1e-2;

/// The order of one step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepOrder {
    /// A unimolecular step.
    First,
    /// A bimolecular step whose partner stays within the tolerance.
    PseudoFirst {
        /// The partner (index into the species of the rate set).
        species: usize,
        /// The bound of its relative change.
        change: f64,
    },
    /// A bimolecular step whose partner changes (or is not bounded).
    Second {
        /// The partner.
        species: usize,
        /// The bound of its relative change (infinite where none exists).
        change: f64,
        /// Why.
        reason: String,
    },
}

/// The orders of the steps of a rate set.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderJudgment {
    /// The order of every step (in the order of the rate set).
    pub steps: Vec<StepOrder>,
    /// The tolerance of the judgment.
    pub tolerance: f64,
}

impl OrderJudgment {
    /// Whether every step is of first or pseudo-first order (the model is linear).
    pub fn linear(&self) -> bool {
        self.steps
            .iter()
            .all(|s| !matches!(s, StepOrder::Second { .. }))
    }

    /// One line per bimolecular step.
    pub fn lines(&self, rates: &RateSet) -> Vec<String> {
        let names = &rates.species.names;
        self.steps
            .iter()
            .zip(&rates.steps)
            .filter_map(|(o, st)| {
                let [a, b] = st.states;
                let step = format!("{} <-> {}", rates.labels[a], rates.labels[b]);
                match o {
                    StepOrder::First => None,
                    StepOrder::PseudoFirst { species, change } => Some(format!(
                        "{step}: pseudo-first order in {} (it changes by at most {change:.1e})",
                        names[*species]
                    )),
                    StepOrder::Second {
                        species, reason, ..
                    } => Some(format!(
                        "{step}: second order in {} ({reason})",
                        names[*species]
                    )),
                }
            })
            .collect()
    }
}

/// The orders of the steps of `rates` with the states together at `total` (mol/L).
pub fn judge_orders(rates: &RateSet, total: f64, tolerance: f64) -> OrderJudgment {
    let sp = &rates.species;
    let steps = rates
        .steps
        .iter()
        .map(|st| {
            let Some((s, _)) = st.partner else {
                return StepOrder::First;
            };
            let Some(comp) = &sp.composition else {
                return StepOrder::Second {
                    species: s,
                    change: f64::INFINITY,
                    reason: "a cycle of steps turns it over".to_string(),
                };
            };
            let c0 = sp.concentrations[s].unwrap_or(0.0);
            let (lo, hi) = comp
                .iter()
                .map(|row| row[s])
                .fold((i32::MAX, i32::MIN), |(l, h), x| (l.min(x), h.max(x)));
            let spread = f64::from((hi - lo).max(0));
            if c0.is_nan() || c0 <= 0.0 {
                return StepOrder::Second {
                    species: s,
                    change: f64::INFINITY,
                    reason: "it starts at zero".to_string(),
                };
            }
            let change = total * spread / c0;
            if change <= tolerance {
                StepOrder::PseudoFirst { species: s, change }
            } else {
                StepOrder::Second {
                    species: s,
                    change,
                    reason: format!("it may change by {change:.1e} of its concentration"),
                }
            }
        })
        .collect();
    OrderJudgment { steps, tolerance }
}
