//! Exploration methods of one's own that only propose ([`Proposer`]): the tasks of a minimum,
//! and for a job transition-state guesses and first paths ([`Proposal`]). [`ProposalMethod`]
//! makes such a proposer an [`ExplorationMethod`]: every first path is relaxed by a chain
//! method of the registry of two-point methods (`neb` by default, as the AFIR paths of
//! `afir_ts`), whose highest maxima join the guesses, and every guess is followed as those of
//! the built-in methods (the optimization to a transition state, its IRC, the ends optimized
//! to minima; [`super::follow`]).
//!
//! This is the plug-in point of methods written outside mopt-rs: the Python package makes a
//! proposer of a Python class (`method=python:MODULE:CLASS`), and a prefix of the method
//! registry ([`super::method::MethodRegistry::register_prefix`]) resolves such names. A
//! proposer computes what it needs itself; the calculations of the run are the mapper's.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::configuration::Configuration;
use super::follow::{follow_guesses, Guess};
use super::method::{ExplorationMethod, Job, JobContext, JobOutcome, PairSearchRequest, Task};
use super::network::{Network, Node, NodeKind};
use crate::error::MoptResult;

/// What a proposer gives for a job.
#[derive(Debug, Clone, Default)]
pub struct Proposal {
    /// Transition-state guesses, each optimized to a transition state.
    pub guesses: Vec<Configuration>,
    /// First paths (from the minimum of the job, or anywhere), each relaxed by the chain
    /// method of the [`ProposalMethod`]; their highest maxima are further guesses.
    pub paths: Vec<Vec<Configuration>>,
    /// What the proposer wants said in the log of the job.
    pub notes: Vec<String>,
}

/// A way of exploring a minimum that proposes guesses and paths, the rest done by the mapper.
pub trait Proposer: Send + Sync {
    /// Its name (that of the method).
    fn name(&self) -> &str;

    /// The tasks of `node` (a minimum), the same list in every run with the same settings: a
    /// restart makes them again and skips those that were done.
    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>>;

    /// What job `job` (one of the tasks of its node) proposes.
    fn propose(&self, job: &Job) -> MoptResult<Proposal>;
}

/// The chain method that relaxes the first paths of a proposal by default.
pub const DEFAULT_PATH_METHOD: &str = "neb";

/// A [`Proposer`] as an exploration method of the mapper.
pub struct ProposalMethod {
    proposer: Box<dyn Proposer>,
    /// The chain method of the paths and its settings.
    path_method: String,
    path_settings: Vec<(String, String)>,
    /// Maxima of a relaxed path that become guesses, the highest first.
    top_n: usize,
}

impl ProposalMethod {
    /// `proposer` with the chain method `path_method` (`KEY=VALUE` settings `path_settings`)
    /// for its paths, of which the `top_n` highest maxima are guesses.
    pub fn new(
        proposer: Box<dyn Proposer>,
        path_method: &str,
        path_settings: Vec<(String, String)>,
        top_n: usize,
    ) -> Self {
        Self {
            proposer,
            path_method: path_method.to_string(),
            path_settings,
            top_n: top_n.max(1),
        }
    }
}

impl ExplorationMethod for ProposalMethod {
    fn name(&self) -> &str {
        self.proposer.name()
    }

    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        if node.kind != NodeKind::Minimum {
            return Ok(Vec::new());
        }
        self.proposer.tasks(node, network)
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let proposal = self.proposer.propose(job)?;
        for note in &proposal.notes {
            ctx.jobs
                .note(&format!("Job {}: {}: {note}", job.id, self.proposer.name()));
        }
        let mut guesses: Vec<Guess> = proposal
            .guesses
            .into_iter()
            .enumerate()
            .map(|(k, system)| Guess {
                system,
                origin: format!("guess {} of {}", k + 1, self.proposer.name()),
            })
            .collect();
        for (p, path) in proposal.paths.into_iter().enumerate() {
            if path.len() < 2 {
                continue;
            }
            let label = format!(
                "{} of path {} of {}",
                self.path_method,
                p + 1,
                self.proposer.name()
            );
            let search = ctx.jobs.pair_search(
                &PairSearchRequest {
                    label: label.clone(),
                    job: job.id,
                    method: self.path_method.clone(),
                    structures: path,
                    settings: self.path_settings.clone(),
                    dissociation_scale: None,
                    dissociation_atoms: None,
                },
                ctx.frames,
            )?;
            if search.interrupted {
                return Ok(JobOutcome::stopped());
            }
            let mut estimates = search.estimates;
            estimates.sort_by(|a, b| b.energy.total_cmp(&a.energy));
            guesses.extend(
                estimates
                    .into_iter()
                    .take(self.top_n)
                    .enumerate()
                    .map(|(m, e)| Guess {
                        system: e.system,
                        origin: format!("maximum {} of the {label}", m + 1),
                    }),
            );
        }
        let mut outcome = JobOutcome::default();
        if guesses.is_empty() {
            ctx.jobs.note(&format!(
                "Job {}: {} proposed no guess and no path with a maximum.",
                job.id,
                self.proposer.name()
            ));
            return Ok(outcome);
        }
        if !follow_guesses(job, ctx, &guesses, &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}
