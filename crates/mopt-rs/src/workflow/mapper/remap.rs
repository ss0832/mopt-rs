//! A network drawn again on another surface (`-remap RUN_DIR`): the minima of the first
//! network are the inputs of the run, and every transition-state edge, crossing edge and
//! saddle point of higher order of the first network is one task of the minimum that runs it.
//!
//! - A transition-state edge: the first chain is the minimum of its first end on the surface
//!   of the run, the structures of its path in the first network (minimum to minimum), and the
//!   minimum of its second end; a two-point method of [`crate::workflow::pair_search`]
//!   (`method_path`, `neb`) relaxes it, and its maxima are followed to transition states and
//!   their IRCs as for the other methods. With `method_path=none` there is no chain: the
//!   transition state of the first network is the guess.
//! - A crossing edge: the crossing point of the first network is refined on the gradient
//!   projection of its two states and followed down on each ([`refine_crossing`]).
//! - A saddle point of order 2 or more: optimized at its order from its structure and
//!   descended from ([`follow_saddle_point`]; `higher_saddles`).
//!
//! What a job finds is matched and joined as in any run, so the second network is a network
//! of its own; [`RemapSource`] is what was read of the first, and the table of what each of
//! its items became is made from the jobs of the second (`cli::mapper_remap`). The rules are
//! in `docs/source/mapper_design.md`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use super::configuration::Configuration;
use super::crossing::{refine_crossing, surface_name};
use super::driver::LEFT_OUT;
use super::follow::{follow_guesses, follow_saddle_point, Guess};
use super::method::{ExplorationMethod, Job, JobContext, JobOutcome, PairSearchRequest, Task};
use super::network::{ElectronicState, Network, Node, NodeKind};
use crate::core_infra::json::JsonValue;
use crate::error::{MoptError, MoptResult};
use crate::workflow::pair_search::PairMethodRegistry;

/// The prefix of the key of a task of the method.
pub const REMAP_PREFIX: &str = "remap:";

/// The value of `method_path` that leaves the chain out.
pub const NO_PATH: &str = "none";

/// The two-point method of the chain when none is given.
pub const DEFAULT_PATH: &str = "neb";

/// The most structures of the path of the first network in a first chain (the first, the
/// last, and structures evenly spread between; the two-point method lays its own images along
/// them).
pub const CHAIN_FRAMES: usize = 60;

/// What an item of the first network is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemapKind {
    /// A transition-state edge.
    Transition,
    /// A crossing edge between two states of the run (their indices among its states).
    Crossing([usize; 2]),
    /// A saddle point of this order (2 or more).
    Saddle(usize),
}

/// An item of the first network that is drawn again.
#[derive(Debug, Clone)]
pub struct RemapItem {
    /// Its name among the items, stable between runs: `ts:EDGE`, `x:EDGE`, `sp:NODE` (the
    /// numbers of the first network).
    pub key: String,
    /// What it is in the first network, for the log (`TS3 between EQ0 and EQ5`).
    pub label: String,
    /// Its kind.
    pub kind: RemapKind,
    /// The input (a minimum of the first network) whose node runs its job.
    pub runner: usize,
    /// The inputs that are the minima at its two ends (`None`: the end is no minimum).
    pub ends: [Option<usize>; 2],
    /// Its point in the first network: the transition state, the crossing point, the saddle
    /// point.
    pub point: Configuration,
    /// The structures of its path in the first network from its first end to its second
    /// (empty: none was kept).
    pub chain: Vec<Configuration>,
}

/// What was read of the first network.
#[derive(Debug, Clone, Default)]
pub struct RemapSource {
    /// The node of the first network that each input of the run is (its minima, in the order
    /// of the inputs).
    pub minima: Vec<usize>,
    /// The items, in the order of their jobs.
    pub items: Vec<RemapItem>,
}

/// At most `most` of `frames`: the first, the last, and frames evenly spread between.
pub fn thinned(frames: Vec<Configuration>, most: usize) -> Vec<Configuration> {
    let count = frames.len();
    if count <= most || most < 2 {
        return frames;
    }
    let mut keep: Vec<usize> = (0..most).map(|k| k * (count - 1) / (most - 1)).collect();
    keep.dedup();
    keep.into_iter()
        .filter_map(|k| frames.get(k).cloned())
        .collect()
}

/// The settings of the method (`method_KEY=VALUE`).
#[derive(Debug, Clone, PartialEq)]
pub struct RemapSettings {
    /// The two-point method that relaxes the first chain (`method_path`), or [`NO_PATH`].
    pub path: String,
    /// Its settings (`method_path_KEY`).
    pub path_settings: Vec<(String, String)>,
}

impl Default for RemapSettings {
    fn default() -> Self {
        Self {
            path: DEFAULT_PATH.to_string(),
            path_settings: Vec::new(),
        }
    }
}

impl RemapSettings {
    /// From the `method_*` settings (without the prefix); what cannot run is refused.
    pub fn parse(extra: &BTreeMap<String, String>) -> MoptResult<Self> {
        let mut s = Self::default();
        for (k, v) in extra {
            match k.as_str() {
                "path" => s.path = v.trim().to_string(),
                other => match other.strip_prefix("path_") {
                    Some(key) if !key.is_empty() => {
                        s.path_settings.push((key.to_string(), v.clone()));
                    }
                    _ => {
                        return Err(MoptError::parse(format!(
                            "remap has no setting method_{other} (path, path_KEY)"
                        )))
                    }
                },
            }
        }
        if !s.path.eq_ignore_ascii_case(NO_PATH) {
            // The method must be one of the registry, and one that takes a first path.
            PairMethodRegistry::default().get(&s.path)?;
        } else if let Some((key, _)) = s.path_settings.first() {
            return Err(MoptError::parse(format!(
                "method_path_{key} is a setting of the two-point method, and method_path=none \
                 runs none"
            )));
        }
        Ok(s)
    }
}

/// The method `remap`.
pub struct Remap {
    source: RemapSource,
    settings: RemapSettings,
    states: Vec<ElectronicState>,
}

impl Remap {
    /// Its name.
    pub const NAME: &'static str = "remap";

    /// The method for `source` with `settings`; `states` are the states of the run (empty
    /// for a run of one state).
    pub fn new(
        source: RemapSource,
        settings: RemapSettings,
        states: Vec<ElectronicState>,
    ) -> MoptResult<Self> {
        for item in &source.items {
            if item.runner >= source.minima.len()
                || item
                    .ends
                    .iter()
                    .flatten()
                    .any(|&k| k >= source.minima.len())
            {
                return Err(MoptError::config(format!(
                    "remap: {} names an input the run does not have",
                    item.label
                )));
            }
            if let RemapKind::Crossing(pair) = &item.kind {
                if pair.iter().any(|&s| s >= states.len()) {
                    return Err(MoptError::config(format!(
                        "remap: {} is between states the run does not have (states)",
                        item.label
                    )));
                }
            }
        }
        Ok(Self {
            source,
            settings,
            states,
        })
    }

    /// What was read of the first network.
    pub fn source(&self) -> &RemapSource {
        &self.source
    }

    /// The number of its tasks.
    pub fn task_count(&self) -> usize {
        self.source.items.len()
    }

    /// The item of `job`.
    fn item(&self, job: &Job) -> MoptResult<&RemapItem> {
        job.task
            .parameters
            .get("item")
            .and_then(JsonValue::as_usize)
            .and_then(|k| self.source.items.get(k))
            .filter(|item| job.task.key == format!("{REMAP_PREFIX}{}", item.key))
            .ok_or_else(|| {
                MoptError::parse(format!(
                    "remap: the task '{}' is no item of the first network",
                    job.task.key
                ))
            })
    }

    /// The node that runs the job of `item` in `network`: the minimum of one of its ends,
    /// then that of the input it was given, then the first input that became a node (an end
    /// that was left out has none); for a crossing point only a minimum on one of its two
    /// states. `None`: no input can run it.
    fn runner(&self, item: &RemapItem, network: &Network) -> Option<usize> {
        let fits = |id: usize| match &item.kind {
            RemapKind::Crossing(pair) => network.node(id).is_ok_and(|n| {
                pair.iter()
                    .any(|&s| self.states.get(s).is_some_and(|state| *state == n.state))
            }),
            _ => true,
        };
        let node_of = |k: usize| network.inputs.get(k).and_then(|i| i.node);
        item.ends
            .iter()
            .flatten()
            .copied()
            .chain([item.runner])
            .filter_map(node_of)
            .find(|&id| fits(id))
            .or_else(|| {
                network
                    .inputs
                    .iter()
                    .filter_map(|i| i.node)
                    .find(|&id| fits(id))
            })
    }

    /// The warning of an item whose job found nothing of its kind.
    fn left_out(item: &RemapItem, what: &str) -> String {
        format!(
            "{} of the first network {LEFT_OUT}: its job gave no {what} on the surface of this \
             run",
            item.label
        )
    }

    /// The structure of the minimum that input `k` became.
    fn minimum_of(network: &Network, k: usize) -> Option<Configuration> {
        let node = network.inputs.get(k)?.node?;
        network.node(node).ok().map(|n| n.system.clone())
    }

    fn run_transition(
        &self,
        job: &Job,
        ctx: &mut JobContext<'_>,
        item: &RemapItem,
    ) -> MoptResult<JobOutcome> {
        let mut outcome = JobOutcome::default();
        let first = Guess {
            system: item.point.clone(),
            origin: format!(
                "the transition state of {} in the first network",
                item.label
            ),
        };
        let mut guesses = Vec::new();
        // The path of the chain on this surface, for its peak (`path_peaks`).
        let mut peak_path: Option<(Vec<super::method::PathPoint>, bool)> = None;
        if !self.settings.path.eq_ignore_ascii_case(NO_PATH) {
            // The minima of the two ends on this surface round the path of the first network.
            let mut chain: Vec<Configuration> = Vec::new();
            chain.extend(item.ends[0].and_then(|k| Self::minimum_of(ctx.network, k)));
            if item.chain.is_empty() {
                chain.push(item.point.clone());
            } else {
                chain.extend(item.chain.iter().cloned());
            }
            chain.extend(item.ends[1].and_then(|k| Self::minimum_of(ctx.network, k)));
            if chain.len() < 2 {
                outcome.warnings.push(format!(
                    "{} has no path and no minimum at its ends in the first network: its \
                     transition state is the guess",
                    item.label
                ));
            } else {
                let search = ctx.jobs.pair_search(
                    &PairSearchRequest {
                        label: format!("{} of {}", self.settings.path, item.label),
                        job: job.id,
                        method: self.settings.path.clone(),
                        structures: chain,
                        settings: self.settings.path_settings.clone(),
                        dissociation_scale: None,
                        dissociation_atoms: None,
                    },
                    ctx.frames,
                )?;
                if search.interrupted {
                    return Ok(JobOutcome::stopped());
                }
                peak_path = Some((search.path, search.converged));
                let mut estimates = search.estimates;
                estimates.sort_by(|a, b| b.energy.total_cmp(&a.energy));
                if estimates.is_empty() {
                    outcome.warnings.push(format!(
                        "{} gave no transition-state estimate for {} ({}): its transition \
                         state in the first network is the guess",
                        self.settings.path, item.label, search.end
                    ));
                }
                guesses.extend(estimates.into_iter().enumerate().map(|(m, e)| Guess {
                    system: e.system,
                    origin: format!("{} maximum {} of {}", self.settings.path, m + 1, item.label),
                }));
            }
        }
        if guesses.is_empty() {
            guesses.push(first);
        }
        let followed = match &peak_path {
            // The peak of the path of the remap, where no transition state comes of it.
            Some((path, converged)) => super::peak::follow_with_peak(
                job,
                ctx,
                &guesses,
                path,
                &format!("{} of {}", self.settings.path, item.label),
                *converged,
                &mut outcome,
            )?,
            None => follow_guesses(job, ctx, &guesses, &mut outcome)?,
        };
        if !followed {
            return Ok(JobOutcome::stopped());
        }
        if outcome.reactions.is_empty() && outcome.known.is_empty() {
            outcome
                .warnings
                .push(Self::left_out(item, "transition state"));
        }
        Ok(outcome)
    }
}

impl ExplorationMethod for Remap {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        if node.kind != NodeKind::Minimum {
            return Ok(Vec::new());
        }
        // The items whose runner is this node (an end of the item that came through).
        Ok(self
            .source
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| self.runner(item, network) == Some(node.id))
            .map(|(k, item)| {
                let mut p = BTreeMap::new();
                p.insert("item".to_string(), JsonValue::Number(k as f64));
                Task {
                    key: format!("{REMAP_PREFIX}{}", item.key),
                    description: format!("{} of the first network", item.label),
                    parameters: JsonValue::Object(p),
                }
            })
            .collect())
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let item = self.item(job)?;
        match &item.kind {
            RemapKind::Transition => self.run_transition(job, ctx, item),
            RemapKind::Saddle(order) => {
                let mut outcome = JobOutcome::default();
                if ctx.jobs.higher_saddles().is_none() {
                    outcome.warnings.push(format!(
                        "{} is not drawn again: the saddle points of higher order are followed \
                         with higher_saddles=on",
                        item.label
                    ));
                    return Ok(outcome);
                }
                if !follow_saddle_point(
                    job,
                    ctx,
                    &item.point,
                    *order,
                    &[],
                    &format!("{} of the first network", item.label),
                    &mut outcome,
                )? {
                    return Ok(JobOutcome::stopped());
                }
                if outcome.saddles.is_empty() && outcome.known.is_empty() {
                    outcome
                        .warnings
                        .push(Self::left_out(item, "saddle point of its order"));
                }
                Ok(outcome)
            }
            RemapKind::Crossing(pair) => {
                let mut outcome = JobOutcome::default();
                let Some(other) = pair
                    .iter()
                    .copied()
                    .find(|&s| s != job.state)
                    .filter(|_| pair.contains(&job.state))
                else {
                    outcome.warnings.push(format!(
                        "{} is between {} and {}, and its job runs on {}: it is not drawn \
                         again",
                        item.label,
                        surface_name(pair[0]),
                        surface_name(pair[1]),
                        surface_name(job.state)
                    ));
                    return Ok(outcome);
                };
                let what = format!("{} of the first network", item.label);
                if !refine_crossing(
                    &self.states,
                    job,
                    ctx,
                    other,
                    &item.point,
                    &what,
                    &mut outcome,
                )? {
                    return Ok(JobOutcome::stopped());
                }
                if outcome.crossings.is_empty() {
                    outcome
                        .warnings
                        .push(Self::left_out(item, "crossing point"));
                }
                Ok(outcome)
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_settings_are_checked() {
        let mut o = BTreeMap::new();
        assert_eq!(RemapSettings::parse(&o).unwrap(), RemapSettings::default());
        o.insert("path_images".to_string(), "12".to_string());
        let s = RemapSettings::parse(&o).unwrap();
        assert_eq!(s.path, "neb");
        assert_eq!(
            s.path_settings,
            vec![("images".to_string(), "12".to_string())]
        );
        // Without a chain there is nothing for the settings of its method.
        o.insert("path".to_string(), "none".to_string());
        assert!(RemapSettings::parse(&o).is_err());
        o.clear();
        o.insert("path".to_string(), "none".to_string());
        assert_eq!(RemapSettings::parse(&o).unwrap().path, "none");
        o.insert("path".to_string(), "no_such_method".to_string());
        assert!(RemapSettings::parse(&o).is_err());
        o.clear();
        o.insert("nope".to_string(), "1".to_string());
        assert!(RemapSettings::parse(&o).is_err());
    }

    #[test]
    fn a_path_is_thinned_to_its_ends_and_structures_between() {
        let frames: Vec<Configuration> = (0..10)
            .map(|k| {
                Configuration::Variables(super::super::configuration::VariablePoint {
                    values: faer::Col::from_fn(1, |_| f64::from(k)),
                    names: std::sync::Arc::new(vec!["x".to_string()]),
                })
            })
            .collect();
        let value = |c: &Configuration| match c {
            Configuration::Variables(p) => p.values[0],
            Configuration::Structure(_) => f64::NAN,
        };
        let kept = thinned(frames.clone(), 4);
        assert_eq!(
            kept.iter().map(value).collect::<Vec<_>>(),
            vec![0.0, 3.0, 6.0, 9.0]
        );
        assert_eq!(thinned(frames.clone(), 20).len(), 10);
        assert_eq!(thinned(frames, 1).len(), 10);
    }
}
