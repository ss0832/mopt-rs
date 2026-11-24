//! The conformers of the transition states of a network (`ts_conformers=on`): beside the
//! tasks of the method, one task for every transition state, in which its conformers are
//! searched with its saddle direction held
//! ([`super::method::MapperJobs::ts_conformers`], the search of `confsearch -tsconf`,
//! [`crate::workflow::confsearch::ts`]) and every saddle point found is
//! followed as any transition-state guess: optimized, its IRC run, the ends minimized. The
//! conformers of a transition state are thus transition states of their own between the
//! conformers of the minima, and the lowest of them is the barrier between two species.
//!
//! A transition state that such a task found is not searched again: its conformers are those
//! of the search that found it. The rules are in `docs/source/mapper_design.md`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

use super::follow::{follow_guesses, Guess};
use super::method::{ExplorationMethod, Job, JobContext, JobOutcome, Task};
use super::network::{Level, Network, Node, NodeKind};
use crate::core_infra::json::JsonValue;
use crate::error::{MoptError, MoptResult};

/// The prefix of the key of a task of the conformer search of a transition state.
pub const TS_CONFORMER_PREFIX: &str = "tsconf:";

/// A method with the conformer searches of the transition states before its own tasks.
pub struct WithTsConformers {
    base: Box<dyn ExplorationMethod>,
    name: String,
}

impl WithTsConformers {
    /// `base` with the conformer searches.
    pub fn new(base: Box<dyn ExplorationMethod>) -> Self {
        let name = base.name().to_string();
        Self { base, name }
    }

    /// Whether the transition state `ts` of `network` was found by a conformer search.
    fn from_a_search(network: &Network, ts: &Node) -> bool {
        ts.origin
            .job
            .and_then(|job| network.jobs.iter().find(|r| r.id == job))
            .is_some_and(|r| r.task.starts_with(TS_CONFORMER_PREFIX))
    }

    /// The conformer searches that `node` runs: those of the transition states of its edges
    /// whose other minimum (where there is one) has a higher number.
    fn searches(network: &Network, node: &Node) -> Vec<Task> {
        if node.kind != NodeKind::Minimum {
            return Vec::new();
        }
        let minimum = |id: usize| network.node(id).is_ok_and(|n| n.kind == NodeKind::Minimum);
        let mut seen = BTreeSet::new();
        let mut tasks = Vec::new();
        for e in network
            .ts_edges
            .iter()
            .filter(|e| e.level == Level::Exploration)
        {
            let runner = e.ends.iter().copied().filter(|&id| minimum(id)).min();
            if runner != Some(node.id) || !seen.insert(e.ts) {
                continue;
            }
            let Ok(ts) = network.node(e.ts) else {
                continue;
            };
            if Self::from_a_search(network, ts) {
                continue;
            }
            let mut p = BTreeMap::new();
            p.insert("ts".to_string(), JsonValue::Number(e.ts as f64));
            tasks.push(Task {
                key: format!("{TS_CONFORMER_PREFIX}{}", e.ts),
                description: format!("the conformers of {}", ts.label()),
                parameters: JsonValue::Object(p),
            });
        }
        tasks
    }

    fn run_search(job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let ts = job
            .task
            .parameters
            .get("ts")
            .and_then(JsonValue::as_usize)
            .and_then(|id| ctx.network.node(id).ok())
            .filter(|n| n.kind == NodeKind::TransitionState)
            .ok_or_else(|| {
                MoptError::parse(format!(
                    "the task '{}' names no transition state of the network",
                    job.task.key
                ))
            })?;
        let label = ts.label();
        let mut outcome = JobOutcome::default();
        let search =
            ctx.jobs
                .ts_conformers(&ts.system, job.id, &format!("the conformers of {label}"))?;
        if search.interrupted {
            return Ok(JobOutcome::stopped());
        }
        outcome.warnings.extend(search.warnings);
        if search.conformers.is_empty() {
            ctx.jobs.note(&format!(
                "Job {}: the search gave no other conformer of {label}.",
                job.id
            ));
            return Ok(outcome);
        }
        let guesses: Vec<Guess> = search
            .conformers
            .into_iter()
            .enumerate()
            .map(|(k, s)| Guess {
                system: s.system,
                origin: format!("conformer {} of {label}", k + 1),
            })
            .collect();
        if !follow_guesses(job, ctx, &guesses, &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

impl ExplorationMethod for WithTsConformers {
    fn name(&self) -> &str {
        &self.name
    }

    fn tasks_follow_network(&self) -> bool {
        // The transition states of a minimum come with the jobs.
        true
    }

    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        let mut tasks = Self::searches(network, node);
        tasks.extend(self.base.tasks(node, network)?);
        Ok(tasks)
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        if job.task.key.starts_with(TS_CONFORMER_PREFIX) {
            Self::run_search(job, ctx)
        } else {
            self.base.run(job, ctx)
        }
    }
}
