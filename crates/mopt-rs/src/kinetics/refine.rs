//! Refinement tasks ranked by sensitivity times uncertainty (`kinetics_beyond_tst.md`,
//! section 3.11): the first-order change of the logarithm of an observable `y` (a population
//! at an output time, or a final yield) that a better value of a parameter could cause,
//!
//! ```text
//! score = |d ln y / d theta| sigma_theta
//! ```
//!
//! on the scale of `ln y` for every kind of task, so that the tasks of different kinds are
//! ranked together (and with exploration scores of the same observable):
//!
//! | Task | Parameter | `sigma` |
//! |---|---|---|
//! | `refine_energy` | the energy of a minimum or bottleneck | its own (`uncertainty_kcal_mol` of the annotations), else the default (`--uncertainty`, kcal/mol) |
//! | `branching` | the share `phi_j` of a step of an N-furcation, the others changed in proportion so that the shares sum to 1: `dy/dphi_j = d_j / phi_j - sum_{l != j} d_l / (1 - phi_j)` with `d = dy / d ln k` | the standard error of a measured share, else that of a share anywhere in `[0, 1]`, `1 / sqrt(12)` |
//! | `coupling` | `ln` of the probability of a crossing point that has no coupling (taken as 1) | `ln 1e3` (the factors of spin-forbidden steps span about three decades, section 2.5) |
//!
//! The derivatives are those of [`crate::kinetics::derivative`] (dual numbers through the
//! solver). A task whose score is 0 is left out.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use super::derivative::{Derivatives, Parameter};
use super::network::{BottleneckKind, KineticNetwork};
use crate::error::{MoptError, MoptResult};

/// The standard deviation of a share that is assumed (uniform on `[0, 1]`).
pub const ASSUMED_SHARE_SIGMA: f64 = 0.288_675_134_594_812_9;
/// The standard deviation of `ln` of the probability of a crossing without a coupling.
pub const UNKNOWN_COUPLING_SIGMA: f64 = 6.907_755_278_982_137;

/// What a refinement task is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    /// A better energy of a minimum or a bottleneck (a higher level, a conformer search of the
    /// saddle point).
    RefineEnergy,
    /// The branching of an N-furcation (trajectories from the saddle point).
    Branching,
    /// The coupling of a crossing point.
    Coupling,
}

impl TaskKind {
    /// `refine_energy`, `branching`, or `coupling`.
    pub fn name(self) -> &'static str {
        match self {
            Self::RefineEnergy => "refine_energy",
            Self::Branching => "branching",
            Self::Coupling => "coupling",
        }
    }
}

/// One refinement task.
#[derive(Debug, Clone, PartialEq)]
pub struct RefinementTask {
    /// Its kind.
    pub kind: TaskKind,
    /// The node or bottleneck (label).
    pub target: String,
    /// The parameter (`E`, `phi -> EQ5`, `ln p_cross`).
    pub parameter: String,
    /// `d ln y / d theta` (per Hartree for energies).
    pub sensitivity: f64,
    /// `sigma_theta` (Hartree for energies).
    pub sigma: f64,
    /// `|sensitivity| sigma`.
    pub score: f64,
}

/// The observable of the ranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observable {
    /// The state of the model.
    pub state: usize,
    /// The output time (index), or `None` for the final yield.
    pub time: Option<usize>,
}

/// The default standard deviations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uncertainties {
    /// Of an energy without its own (Hartree).
    pub energy: f64,
}

/// The tasks for `observable`, highest score first.
pub fn refinement_tasks(
    d: &Derivatives,
    net: &KineticNetwork,
    observable: Observable,
    u: &Uncertainties,
) -> MoptResult<Vec<RefinementTask>> {
    let i = observable.state;
    let (y, dy): (f64, Vec<f64>) = match observable.time {
        Some(k) => {
            let Some(p) = d.populations.get(k).and_then(|p| p.get(i)) else {
                return Err(MoptError::config(
                    "refinement: no such output time or state",
                ));
            };
            (*p, d.d_populations.iter().map(|m| m[k][i]).collect())
        }
        None => {
            let Some(p) = d.final_populations.get(i) else {
                return Err(MoptError::config("refinement: no such state"));
            };
            (*p, d.d_final.iter().map(|m| m[i]).collect())
        }
    };
    if y.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(MoptError::config(format!(
            "refinement: the observable (the population of {}) is zero",
            d.labels.get(i).map_or("?", String::as_str)
        )));
    }
    let mut tasks = Vec::new();
    // The derivatives with respect to ln k, by bottleneck.
    let mut ln_k: BTreeMap<usize, f64> = BTreeMap::new();
    for (m, info) in d.parameters.iter().enumerate() {
        match info.parameter {
            Parameter::StateEnergy(k) => {
                let s = &net.states[k];
                let sigma = s.extra.uncertainty.unwrap_or(u.energy);
                tasks.push(RefinementTask {
                    kind: TaskKind::RefineEnergy,
                    target: s.label.clone(),
                    parameter: "E".to_string(),
                    sensitivity: dy[m] / y,
                    sigma,
                    score: (dy[m] / y).abs() * sigma,
                });
            }
            Parameter::BottleneckEnergy(k) => {
                let b = &net.bottlenecks[k];
                let sigma = b.uncertainty.unwrap_or(u.energy);
                // The steps of an N-furcation share their saddle point: one task.
                if tasks.iter().any(|t: &RefinementTask| {
                    t.kind == TaskKind::RefineEnergy && t.target == b.label && t.parameter == "E"
                }) {
                    if let Some(t) = tasks
                        .iter_mut()
                        .find(|t| t.kind == TaskKind::RefineEnergy && t.target == b.label)
                    {
                        t.sensitivity += dy[m] / y;
                        t.score = t.sensitivity.abs() * t.sigma;
                    }
                    continue;
                }
                tasks.push(RefinementTask {
                    kind: TaskKind::RefineEnergy,
                    target: b.label.clone(),
                    parameter: "E".to_string(),
                    sensitivity: dy[m] / y,
                    sigma,
                    score: (dy[m] / y).abs() * sigma,
                });
            }
            Parameter::LnK(k) => {
                ln_k.insert(k, dy[m]);
            }
            _ => {}
        }
    }
    // The shares of the N-furcations.
    let mut groups: BTreeMap<(usize, String), Vec<usize>> = BTreeMap::new();
    for (k, b) in net.bottlenecks.iter().enumerate() {
        if b.share.is_some() && ln_k.contains_key(&k) {
            groups.entry((b.id, b.label.clone())).or_default().push(k);
        }
    }
    for ((_, label), steps) in &groups {
        if steps.len() < 2 {
            continue;
        }
        let total: f64 = steps.iter().map(|k| ln_k[k]).sum();
        for &k in steps {
            let b = &net.bottlenecks[k];
            let phi = b.share.unwrap_or(1.0);
            if !(phi > 0.0 && phi < 1.0) {
                continue;
            }
            let d_phi = ln_k[&k] / phi - (total - ln_k[&k]) / (1.0 - phi);
            let sigma = b.share_error.unwrap_or(ASSUMED_SHARE_SIGMA);
            let other = net.states[b.states[1]].label.clone();
            tasks.push(RefinementTask {
                kind: TaskKind::Branching,
                target: label.clone(),
                parameter: format!("phi {} -> {}", net.states[b.states[0]].label, other),
                sensitivity: d_phi / y,
                sigma,
                score: (d_phi / y).abs() * sigma,
            });
        }
    }
    // Crossing points without a coupling.
    for (&k, &dk) in &ln_k {
        let b = &net.bottlenecks[k];
        if b.kind == BottleneckKind::Crossing && b.coupling.is_none() {
            tasks.push(RefinementTask {
                kind: TaskKind::Coupling,
                target: b.label.clone(),
                parameter: "ln p_cross".to_string(),
                sensitivity: dk / y,
                sigma: UNKNOWN_COUPLING_SIGMA,
                score: (dk / y).abs() * UNKNOWN_COUPLING_SIGMA,
            });
        }
    }
    tasks.retain(|t| t.score > 0.0 && t.score.is_finite());
    tasks.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.target.cmp(&b.target))
            .then_with(|| a.parameter.cmp(&b.parameter))
    });
    Ok(tasks)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::kinetics::conditions::Conditions;
    use crate::kinetics::derivative::derivatives;
    use crate::kinetics::network::{Bottleneck, KineticState};
    use crate::kinetics::{KineticsParts, KineticsSettings};

    #[test]
    fn the_controlling_saddle_point_and_an_assumed_split_rank_first() {
        // A -> B over a barrier of 25 mHartree (about 20 1/s at 300 K), then B splits into C and
        // D (an N-furcation with equal assumed shares) over a low saddle point; C and D do not
        // come back within a second (their barrier back is 50 mHartree).
        let mut net = KineticNetwork::default();
        for (k, e) in [0.0, -0.01, -0.05, -0.05].iter().enumerate() {
            net.states
                .push(KineticState::with_energy(k, &format!("EQ{k}"), *e));
        }
        net.bottlenecks
            .push(Bottleneck::saddle(0, "TS0", [0, 1], 0.025));
        for to in [2usize, 3] {
            let mut b = Bottleneck::saddle(1, "TS1", [1, to], 0.0);
            b.share = Some(0.5);
            net.bottlenecks.push(b);
        }
        let s = KineticsSettings {
            conditions: Conditions {
                temperature: 300.0,
                ..Conditions::default()
            },
            solver: "expm".to_string(),
            ..KineticsSettings::default()
        };
        let parts = KineticsParts::default();
        let d = derivatives(&net, &s, &parts, &[(0, 1.0)], &[1e-3, 1.0]).unwrap();
        let u = Uncertainties {
            energy: crate::domain::Units::KCAL_MOL_TO_HARTREE,
        };
        // At 1 ms A -> B has hardly started: its saddle point (and the energy of A) control
        // the yield of C.
        let early = refinement_tasks(
            &d,
            &net,
            Observable {
                state: 2,
                time: Some(0),
            },
            &u,
        )
        .unwrap();
        assert!(early[..2].iter().any(|t| t.target == "TS0"), "{early:?}");
        assert_eq!(early[0].kind, TaskKind::RefineEnergy);
        // At 1 s A has gone and the split decides the yield of C: y = phi_C, d ln y / d phi = 2.
        let late = refinement_tasks(
            &d,
            &net,
            Observable {
                state: 2,
                time: Some(1),
            },
            &u,
        )
        .unwrap();
        assert_eq!(late[0].kind, TaskKind::Branching, "{late:?}");
        assert_eq!(late[0].target, "TS1");
        let c = late
            .iter()
            .find(|t| t.kind == TaskKind::Branching && t.parameter.ends_with("EQ2"))
            .unwrap();
        assert!((c.sensitivity - 2.0).abs() < 1e-4, "{}", c.sensitivity);
        assert!((c.score - 2.0 * ASSUMED_SHARE_SIGMA).abs() < 1e-4);
    }
}
