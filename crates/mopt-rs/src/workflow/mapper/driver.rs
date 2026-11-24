//! The exploration loop of the mapper as an ask/tell state machine: [`MapperDriver::ask`]
//! chooses the node to explore next ([`super::priority::ExplorationPriority`]) and the first of
//! its tasks that has not been run ([`super::method::ExplorationMethod::tasks`]), and returns
//! the job; whoever runs the job (the command, a test, a user's own loop) hands what it found to
//! [`MapperDriver::tell`], which compares it with the known nodes, adds what is new, records
//! the job, and asks the stop rules whether to go on.
//!
//! Everything the driver decides follows from the network and the settings: the order of the
//! tasks of a node depends on the node alone, and the random choice of job `k` uses a generator
//! seeded by the seed of the run and `k`. A network read back from its file (with its finished
//! jobs) therefore goes on with the same jobs as the run that wrote it, and the stop rules are
//! brought to the same state by replaying the finished jobs.
//!
//! Several jobs can be asked for before any of them is told (jobs that run at the same time,
//! [`super::schedule`]): [`MapperDriver::ask_batch`] chooses them one after the other from the
//! network as it is, each job seeing the ones chosen before it as in flight (their tasks are
//! not offered again, and they count as visits of their nodes), and
//! [`MapperDriver::tell_job`] takes their outcomes in any order.
//!
//! MultiOptPy: `Wrapper/mapper.py:ReactionNetworkMapper` (`run`, `_run_sequential`,
//!  `_run_parallel_rolling` (`_try_submit`), `_find_or_register_node`, `_process_profile`,
//!  `_enqueue_perturbations`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use super::afir::mix;
use super::configuration::Configuration;
use super::method::{
    DescentEnd, ExplorationMethod, FrameSink, Job, JobContext, JobOutcome, MapperJobs, SaddleEnd,
    Stationary, Task,
};
use super::network::{
    IrcPath, JobRecord, JobStatus, Level, Network, NetworkWarning, Node, NodeKind, OpenTsEdge,
    Origin, SaddleEdge, TsEdge,
};
use super::priority::{Candidate, ExplorationPriority};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::identity::StructureMatcher;
use crate::workflow::explore::{StopRule, StopRules, TrialSummary};

/// The settings of the driver.
#[derive(Debug, Clone)]
pub struct DriverSettings {
    /// Seed of the random choices.
    pub seed: u64,
    /// When two structures are the same.
    pub matcher: StructureMatcher,
    /// Nodes that are never explored (MultiOptPy's `--exclude_nodes`).
    pub excluded: Vec<usize>,
    /// New minima whose bond graph is not that of the input they come from are not explored
    /// (MultiOptPy's `--exclude_bond_rearrangement`).
    pub exclude_bond_rearrangement: bool,
    /// A new minimum is not explored when the bonds changed more than this many times on the
    /// way from its input (`max_bond_changes`; `None`: no limit). A step of the way (from the
    /// minimum a job started at to a minimum it found) counts when the two bond graphs are
    /// not the same graph.
    pub max_bond_changes: Option<usize>,
    /// The same for the chemical species, by their SMILES strings (`max_species_changes`).
    pub max_species_changes: Option<usize>,
    /// The inputs that are the structures of an entrance channel (`entrance_structures`;
    /// their indices among the inputs, empty: none).
    pub entrance_inputs: Vec<usize>,
    /// The excited state of the surface (for the states of the nodes).
    pub excited_state: usize,
    /// The exploration ends when the goal of a goal-directed search (an input marked `goal`)
    /// is joined to a start (`goal_stop`).
    pub goal_stop: bool,
    /// The goal counts as reached when a minimum with its bond graph is joined to a start,
    /// not only the goal structure itself (`goal_match=bonds`).
    pub goal_bonds: bool,
    /// A new minimum is explored only with the probability `min(1, exp(-(E_PT - E_X) / kT))`
    /// at this temperature (K), `E_PT` the highest transition state on the way of its job
    /// from the minimum `X` it started at (`accept_temperature`; `None`: every minimum).
    pub accept_temperature: Option<f64>,
    /// The state of every input where it is known beforehand (its index among the states of
    /// the run): the minima of another network, each on its own state (`-remap`). An input
    /// without one (or beyond the list) lies on the first state of its charge and
    /// multiplicity.
    pub input_states: Vec<Option<usize>>,
    /// An input that does not converge to a minimum, or whose minimization fails, is left
    /// out with a warning and the run goes on (the minima of another network on another
    /// surface, `-remap`: some do not come through). Without it such an input ends the run.
    pub skip_failed_inputs: bool,
    /// The structures of the network are cores of a larger system (an ONIOM surface): a
    /// minimum or a transition state found again with another environment of its core is
    /// listed on the node ([`Node::environments`]), and the node takes the one of the lowest
    /// energy (`docs/source/mapper_design.md`, when two nodes are the same).
    pub environments: bool,
    /// Energies of two environments of one core within this (Hartree) are the same
    /// environment: the energy tolerance of the identity (`energy_tolerance_kcalmol`), since
    /// two minimizations of one environment, converged to the force thresholds, end a little
    /// apart (6e-8 Hartree for vinyl alcohol with two waters on GFN1-xTB:GFN-FF).
    pub environment_tolerance: f64,
    /// In a periodic cell, a minimum whose molecules in contact no longer span the periodic
    /// directions of its cell (pulled apart into layers, chains, or molecules with vacuum
    /// between them) is kept as a dissociation channel, with the margin of the contacts
    /// (Bohr; `channel_margin`); `None` (`channels=false`): it stays a minimum.
    pub cell_contact_margin: Option<f64>,
    /// Every minimum is verified by its Hessian (`verify_minima`): a new minimum of a
    /// structure that comes without the wavenumbers of that check (its Hessian could not be
    /// had to convergence) is kept with a warning that its stability was not judged.
    pub verify_minima: bool,
}

/// An end of the IRC of a transition state whose energy differs from that of its node (another
/// environment of the same core) by more than this (Hartree; 1 kcal/mol, a judgement not yet
/// measured) is reported as environment heterogeneity.
pub const ENVIRONMENT_HETEROGENEITY: f64 = 1.0 / crate::domain::Units::HARTREE_TO_KCAL_MOL;

/// What the warning of an input or of an item of another network says when it does not come
/// through and is not in the network of the run.
pub const LEFT_OUT: &str = "is left out";

/// The start of the reason of a minimum the acceptance filter refused (it is drawn again when a
/// later job reaches it).
pub const REFUSED_REASON: &str = "not accepted by the acceptance filter";

/// The reason of a goal node: it is kept and never explored.
pub const GOAL_REASON: &str =
    "the goal of the goal-directed search (goal=FILE): kept, not explored";

/// What the driver wants next.
#[derive(Debug, Clone)]
pub enum MapperAction {
    /// Run this job and hand its outcome to [`MapperDriver::tell`].
    Run(Box<Job>),
    /// The exploration has ended, for this reason.
    Finished(String),
}

/// The jobs chosen together ([`MapperDriver::ask_batch`]).
#[derive(Debug, Clone)]
pub enum MapperBatch {
    /// Run these jobs (in the order of their numbers) and tell each of them.
    Run(Vec<Job>),
    /// The exploration has ended, for this reason.
    Finished(String),
}

/// What the driver wants next while jobs may be in flight ([`MapperDriver::ask_next`]).
#[derive(Debug, Clone)]
pub enum NextJob {
    /// Run this job and hand its outcome to [`MapperDriver::tell_job`].
    Run(Box<Job>),
    /// Nothing to start before a job in flight is told (no task is left that is not in
    /// flight, or the limit of jobs is reached with jobs in flight).
    Wait,
    /// The exploration has ended, for this reason (jobs in flight may still be told).
    Finished(String),
}

/// What [`MapperDriver::tell`] did with a job.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Told {
    /// Minima that were not known.
    pub new_minima: Vec<usize>,
    /// Transition-state edges that were added, with the index of their reaction in the outcome.
    pub new_edges: Vec<(usize, usize)>,
    /// Transition states with an end of their IRC that is not known that were added
    /// ([`Network::open_ts_edges`]), with the index of their reaction in the outcome.
    pub new_open_edges: Vec<(usize, usize)>,
    /// Crossing edges that were added, with the index of their crossing point in the outcome.
    pub new_crossings: Vec<(usize, usize)>,
    /// Path peaks that were recorded, with the index of each in the outcome and its number
    /// among the path peaks of the network.
    pub new_peaks: Vec<(usize, usize)>,
    /// Nodes whose structure the job replaced by another environment of the same core of a
    /// lower energy ([`DriverSettings::environments`]): their files are written again.
    pub lowered: Vec<usize>,
    /// Reactions whose transition state was known.
    pub known: usize,
    /// Why the exploration ends after this job, if it does.
    pub stop: Option<String>,
}

/// The exploration of a network.
pub struct MapperDriver {
    network: Network,
    method: Arc<dyn ExplorationMethod>,
    priority: Box<dyn ExplorationPriority>,
    stops: StopRules,
    settings: DriverSettings,
    tasks: Vec<Option<Vec<Task>>>,
    /// The jobs asked for and not told yet, in the order they were asked for.
    pending: Vec<Job>,
    /// The number of the next job.
    next_id: usize,
    /// No job is started once this many are finished or in flight (0: no limit).
    job_limit: usize,
    finished: Option<String>,
    /// The nodes the job being told replaced by a lower environment of their core, with the
    /// structure each took ([`DriverSettings::environments`]).
    lowered: Vec<(usize, Configuration)>,
    /// Why the priority chose the node of a job, by job ([`ExplorationPriority::reason`]).
    reasons: BTreeMap<usize, String>,
}

/// The number after the largest number of a finished job of `network`.
fn after_last(network: &Network) -> usize {
    network.jobs.iter().map(|j| j.id + 1).max().unwrap_or(0)
}

/// The energies of the minima known after job `job` (all of them for `None`).
fn energies_after(network: &Network, job: Option<usize>) -> Vec<f64> {
    network
        .minima()
        .filter(|n| match (n.origin.job, job) {
            (Some(found), Some(k)) => found <= k,
            _ => true,
        })
        .map(|n| n.energy)
        .collect()
}

impl MapperDriver {
    /// The exploration of `network` by `method` with `priority` and the stop rules `rules`;
    /// the finished jobs of `network` are replayed to the rules.
    pub fn new(
        network: Network,
        method: Box<dyn ExplorationMethod>,
        priority: Box<dyn ExplorationPriority>,
        rules: Vec<Box<dyn StopRule>>,
        settings: DriverSettings,
    ) -> Self {
        let mut stops = StopRules::new();
        for rule in rules {
            stops = stops.with(rule);
        }
        let mut finished = None;
        // The rules count the jobs that were told (their numbers have gaps where jobs that
        // ran at the same time as others were stopped in the rolling schedule).
        for (told, job) in network.jobs.iter().enumerate() {
            let energies = energies_after(&network, Some(job.id));
            let reason = stops.after_trial(&TrialSummary {
                trial: told,
                energies: &energies,
                found_new: job.found_new,
                found_lower: job.found_lower,
            });
            if finished.is_none() {
                finished = reason;
            }
        }
        let next_id = after_last(&network);
        Self {
            network,
            method: Arc::from(method),
            priority,
            stops,
            settings,
            tasks: Vec::new(),
            pending: Vec::new(),
            next_id,
            job_limit: 0,
            finished,
            lowered: Vec::new(),
            reasons: BTreeMap::new(),
        }
    }

    /// Start no job once `max` jobs are finished or in flight (0: no limit), so that jobs
    /// asked for together do not go beyond the limit of the run (`--max_iter`, which the
    /// stop rule `max_jobs` enforces as well).
    pub fn set_job_limit(&mut self, max: usize) {
        self.job_limit = max;
    }

    /// The network.
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// The network, to set what the driver does not decide (file names, evaluations).
    pub fn network_mut(&mut self) -> &mut Network {
        &mut self.network
    }

    /// The method of exploration.
    pub fn method(&self) -> &dyn ExplorationMethod {
        self.method.as_ref()
    }

    /// The method of exploration, to run jobs while the driver goes on (jobs whose results
    /// are taken as they come).
    pub fn shared_method(&self) -> Arc<dyn ExplorationMethod> {
        Arc::clone(&self.method)
    }

    /// The settings of the driver.
    pub fn settings(&self) -> &DriverSettings {
        &self.settings
    }

    /// The settings, to change in a test.
    #[cfg(test)]
    pub(crate) fn settings_mut(&mut self) -> &mut DriverSettings {
        &mut self.settings
    }

    /// The jobs asked for and not told yet.
    pub fn in_flight(&self) -> &[Job] {
        &self.pending
    }

    /// Why the exploration has ended, once it has.
    pub fn finished(&self) -> Option<&str> {
        self.finished.as_deref()
    }

    /// The name of the priority.
    pub fn priority_name(&self) -> &str {
        self.priority.name()
    }

    /// When two structures are the same.
    pub fn matcher(&self) -> &StructureMatcher {
        &self.settings.matcher
    }

    /// Leave the input `input` out of the network (`why`: what happened to it): a warning of
    /// the network, once, and no node.
    pub fn leave_out_input(&mut self, input: usize, why: &str) -> MoptResult<String> {
        let record = self
            .network
            .inputs
            .get(input)
            .ok_or_else(|| MoptError::config(format!("the network has no input {input}")))?;
        let message = format!("input {} ({}) {LEFT_OUT}: {why}", input + 1, record.file);
        if !self.network.warnings.iter().any(|w| w.message == message) {
            self.network.warnings.push(NetworkWarning {
                job: None,
                message: message.clone(),
            });
        }
        Ok(message)
    }

    /// Add the input structure `input` of the network, optimized to `system` with `energy`:
    /// a new minimum, or the known one it is the same as. Returns the node.
    pub fn add_input(
        &mut self,
        input: usize,
        system: Configuration,
        energy: f64,
    ) -> MoptResult<usize> {
        let record = self
            .network
            .inputs
            .get(input)
            .cloned()
            .ok_or_else(|| MoptError::config(format!("the network has no input {input}")))?;
        if let Some(node) = record.node {
            return Ok(node);
        }
        let state = self.input_state(input, &system)?;
        let on = (!self.network.states.is_empty()).then_some(state);
        let known = self.network.find_same_in(
            &NodeKind::Minimum,
            Level::Exploration,
            &system,
            energy,
            &self.settings.matcher,
            on.as_ref(),
        )?;
        let node = match known {
            Some(id) => id,
            None => {
                let mut node = Node::new(
                    NodeKind::Minimum,
                    Level::Exploration,
                    system,
                    energy,
                    state,
                    Origin {
                        input: Some(input),
                        job: None,
                        from_node: None,
                    },
                );
                if self.settings.excluded.contains(&self.network.nodes.len()) {
                    node.excluded = Some("excluded by the user (--exclude_nodes)".to_string());
                } else if record.goal {
                    node.excluded = Some(GOAL_REASON.to_string());
                }
                self.network.add_node(node)?
            }
        };
        if let Some(record) = self.network.inputs.get_mut(input) {
            record.node = Some(node);
        }
        Ok(node)
    }

    fn tasks_of(&mut self, id: usize) -> MoptResult<&[Task]> {
        if self.tasks.len() <= id {
            self.tasks.resize(id + 1, None);
        }
        if self.tasks[id].is_none() || self.method.tasks_follow_network() {
            let node = self.network.node(id)?;
            self.tasks[id] = Some(self.method.tasks(node, &self.network)?);
        }
        Ok(self.tasks[id].as_deref().unwrap_or_default())
    }

    fn explorable(&self, node: &Node) -> bool {
        node.kind == NodeKind::Minimum
            && node.excluded.is_none()
            && !self.settings.excluded.contains(&node.id)
    }

    /// The next job, or the end of the exploration (one job at a time: the job asked for
    /// before must have been told).
    pub fn ask(&mut self) -> MoptResult<MapperAction> {
        if let Some(reason) = &self.finished {
            return Ok(MapperAction::Finished(reason.clone()));
        }
        if !self.pending.is_empty() {
            return Err(MoptError::config(
                "the mapper was asked for a job before the last one was told",
            ));
        }
        match self.ask_next()? {
            NextJob::Run(job) => Ok(MapperAction::Run(job)),
            NextJob::Finished(reason) => Ok(MapperAction::Finished(reason)),
            NextJob::Wait => Err(MoptError::config(
                "the mapper has nothing to wait for without a job in flight",
            )),
        }
    }

    /// Up to `n` jobs chosen together from the network as it is (job after job, each with
    /// the ones before it in flight), or the end of the exploration when none is left and
    /// none is in flight. Fewer than `n` come when the tasks or the limit of jobs run out.
    pub fn ask_batch(&mut self, n: usize) -> MoptResult<MapperBatch> {
        let mut jobs = Vec::new();
        while jobs.len() < n.max(1) {
            match self.ask_next()? {
                NextJob::Run(job) => jobs.push(*job),
                NextJob::Wait => break,
                NextJob::Finished(reason) => {
                    if jobs.is_empty() && self.pending.is_empty() {
                        return Ok(MapperBatch::Finished(reason));
                    }
                    break;
                }
            }
        }
        if jobs.is_empty() {
            return Err(MoptError::config(
                "the mapper chose no job while others are in flight; tell them first",
            ));
        }
        Ok(MapperBatch::Run(jobs))
    }

    /// The tasks of node `id` that are neither finished nor in flight.
    fn open_tasks(&mut self, id: usize) -> MoptResult<Vec<Task>> {
        let mut taken: Vec<String> = self
            .network
            .done_tasks(id)
            .into_iter()
            .map(str::to_string)
            .collect();
        taken.extend(
            self.pending
                .iter()
                .filter(|j| j.node == id)
                .map(|j| j.task.key.clone()),
        );
        Ok(self
            .tasks_of(id)?
            .iter()
            .filter(|t| !taken.contains(&t.key))
            .cloned()
            .collect())
    }

    /// The next job with the jobs in flight left out (their tasks are not offered again, and
    /// they count as visits of their nodes), a wait for them, or the end of the exploration.
    pub fn ask_next(&mut self) -> MoptResult<NextJob> {
        if self.finished.is_none() {
            self.finished = self.goal_reached();
        }
        if let Some(reason) = &self.finished {
            return Ok(NextJob::Finished(reason.clone()));
        }
        let started = self.network.jobs.len() + self.pending.len();
        // (A limit the user gave while the run went on, steer.txt, comes before the settings.)
        let job_limit = self.network.steering.max_jobs.unwrap_or(self.job_limit);
        if job_limit > 0 && started >= job_limit {
            if !self.pending.is_empty() {
                return Ok(NextJob::Wait);
            }
            let reason = format!("The limit of {job_limit} jobs is reached.");
            self.finished = Some(reason.clone());
            return Ok(NextJob::Finished(reason));
        }
        let job = self.next_id;
        let ids: Vec<usize> = self
            .network
            .nodes
            .iter()
            .filter(|n| self.explorable(n))
            .map(|n| n.id)
            .collect();
        let mut candidates = Vec::new();
        for id in ids {
            let untried = self.open_tasks(id)?.len();
            if untried > 0 {
                let node = self.network.node(id)?;
                let flying = self.pending.iter().filter(|j| j.node == id).count();
                candidates.push(Candidate {
                    node: id,
                    energy: node.free_energy.unwrap_or(node.energy),
                    visits: self.network.visits(id) + flying,
                    untried,
                });
            }
        }
        // The minima the user asked for first (steer.txt: explore), in their order, while
        // they have a task left; one without a task left (or not explored) leaves the list.
        while let Some(&first) = self.network.steering.first.first() {
            if candidates.iter().any(|c| c.node == first) {
                candidates.retain(|c| c.node == first);
                break;
            }
            let waits = self.pending.iter().any(|j| j.node == first);
            if waits {
                break;
            }
            self.network.steering.first.remove(0);
        }
        if candidates.is_empty() {
            if !self.pending.is_empty() {
                return Ok(NextJob::Wait);
            }
            // The acceptance filter orders the search, it does not end it: with nothing else
            // left, the refused minimum most likely to pass (the lowest barrier) is explored.
            if let Some(id) = self.most_likely_refused() {
                let label = self.network.node(id)?.label();
                self.network.nodes[id].excluded = None;
                self.network.warnings.push(NetworkWarning {
                    job: None,
                    message: format!(
                        "{label}, which the acceptance filter refused, is explored: no accepted \
                         minimum has a task left, and of the refused ones it has the lowest \
                         barrier"
                    ),
                });
                return self.ask_next();
            }
            let reason = "Every task of every minimum that is explored has been run.".to_string();
            self.finished = Some(reason.clone());
            return Ok(NextJob::Finished(reason));
        }
        let mut rng = Rng::from_seed(mix(self.settings.seed, job as u64 + 1));
        let Some(pick) = self
            .priority
            .select(&self.network, &candidates, job, &mut rng)?
        else {
            let reason = format!(
                "The priority {} ended the exploration.",
                self.priority.name()
            );
            self.finished = Some(reason.clone());
            return Ok(NextJob::Finished(reason));
        };
        let chosen = candidates.get(pick).ok_or_else(|| {
            MoptError::config(format!(
                "the priority {} chose candidate {pick} of {}",
                self.priority.name(),
                candidates.len()
            ))
        })?;
        if let Some(reason) = self.priority.reason() {
            self.reasons.insert(job, reason);
        }
        let id = chosen.node;
        let task = self
            .open_tasks(id)?
            .into_iter()
            .next()
            .ok_or_else(|| MoptError::config(format!("node {id} has no task left")))?;
        let next = Job {
            id: job,
            node: id,
            task,
            start: self.network.node(id)?.system.clone(),
            state: self.network.state_of(id)?,
            roots: self.network.node(id)?.roots.clone(),
        };
        self.pending.push(next.clone());
        self.next_id = job + 1;
        Ok(NextJob::Run(Box::new(next)))
    }

    /// Run the pending `job` with the method of the driver.
    pub fn run_job(
        &self,
        job: &Job,
        jobs: &dyn MapperJobs,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<JobOutcome> {
        let mut context = JobContext {
            jobs,
            network: &self.network,
            matcher: &self.settings.matcher,
            frames,
            state: self.network.states.get(job.state).copied(),
        };
        let mut outcome = self.method.run(job, &mut context)?;
        outcome.followed = jobs.take_followed();
        Ok(outcome)
    }

    /// Give the node `id` the followed electronic states of its structure among `followed`
    /// (the ends of optimizations with their states, by their coordinates); nothing when its
    /// structure is not among them.
    pub fn set_followed(
        &mut self,
        id: usize,
        followed: &[(Vec<f64>, super::method::FollowedRoots)],
    ) -> MoptResult<()> {
        if followed.is_empty() {
            return Ok(());
        }
        let Some(x) = self
            .network
            .node(id)?
            .system
            .structure()
            .map(|s| s.coords_flat_bohr().iter().copied().collect::<Vec<f64>>())
        else {
            return Ok(());
        };
        if let Some((_, roots)) = followed.iter().rev().find(|(at, _)| *at == x) {
            if let Some(node) = self.network.nodes.get_mut(id) {
                node.roots.clone_from(roots);
            }
        }
        Ok(())
    }

    fn after(&mut self, record: JobRecord) -> Option<String> {
        let energies = energies_after(&self.network, None);
        let summary = TrialSummary {
            trial: self.network.jobs.len(),
            energies: &energies,
            found_new: record.found_new,
            found_lower: record.found_lower,
        };
        let reason = self.stops.after_trial(&summary);
        self.network.jobs.push(record);
        let reason = reason.or_else(|| self.goal_reached());
        if self.finished.is_none() {
            self.finished.clone_from(&reason);
        }
        reason
    }

    /// The minimum the acceptance filter refused with the highest probability `p` (the lowest
    /// barrier; the lowest number among equal ones).
    fn most_likely_refused(&self) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for n in &self.network.nodes {
            let Some(reason) = n.excluded.as_deref() else {
                continue;
            };
            let Some(rest) = reason.strip_prefix(REFUSED_REASON) else {
                continue;
            };
            let p = rest
                .strip_prefix(": p = ")
                .and_then(|r| r.split_whitespace().next())
                .and_then(|t| t.parse::<f64>().ok())
                .unwrap_or(0.0);
            if best.is_none_or(|(_, q)| p > q) {
                best = Some((n.id, p));
            }
        }
        best.map(|(id, _)| id)
    }

    /// Why the exploration ends at the goal, when the goal is joined to a start.
    fn goal_reached(&self) -> Option<String> {
        if !self.settings.goal_stop {
            return None;
        }
        let goal = self.network.goal_node()?;
        let label = self.network.node(goal).ok()?.label();
        if self.network.goal_joined() == Some(true) {
            return Some(format!(
                "The goal {label} is joined to a start (goal_stop=off goes on)."
            ));
        }
        if self.settings.goal_bonds {
            let id = self.network.goal_bonds_joined(&self.settings.matcher)?;
            let found = self.network.node(id).ok()?.label();
            return Some(format!(
                "{found}, with the bond graph of the goal {label}, is joined to a start \
                 (goal_match=bonds; goal_stop=off goes on)."
            ));
        }
        None
    }

    /// The acceptance filter of the new minima of `job` at `temperature` (K): each is
    /// explored only with the probability `min(1, exp(-(E_PT - E_X) / kT))`, `E_PT` the
    /// highest transition state on the lowest way of the job from its start `X` (through the
    /// transition-state edges it added); a minimum the job did not reach from `X` takes the
    /// lowest transition state of the job that ends at it. The draw uses a generator seeded by
    /// the seed, the job, and the node, so a restart draws the same.
    fn accept(
        &mut self,
        job: &Job,
        told: &Told,
        known_ts: &[usize],
        temperature: f64,
    ) -> MoptResult<()> {
        let e_from = self.network.node(job.node)?.energy;
        // The edges of the transition states the job added or reached again.
        let edges: Vec<(f64, [usize; 2])> = told
            .new_edges
            .iter()
            .filter_map(|&(_, e)| self.network.ts_edges.get(e))
            .chain(
                self.network
                    .ts_edges
                    .iter()
                    .filter(|e| known_ts.contains(&e.ts)),
            )
            .filter_map(|e| self.network.nodes.get(e.ts).map(|t| (t.energy, e.ends)))
            .collect();
        // The lowest highest energy on a way from the start (a few edges: relaxed until
        // nothing changes).
        let mut best: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
        best.insert(job.node, e_from);
        loop {
            let mut changed = false;
            for (e_ts, ends) in &edges {
                for (a, b) in [(ends[0], ends[1]), (ends[1], ends[0])] {
                    if let Some(&ea) = best.get(&a) {
                        let via = ea.max(*e_ts);
                        if best.get(&b).is_none_or(|&eb| via < eb) {
                            best.insert(b, via);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let kt = crate::domain::Units::BOLTZMANN_HARTREE_PER_K * temperature;
        let from = self.network.node(job.node)?.label();
        // The new minima, and the known ones the filter refused before that this job reached
        // again (in the paper a refused node is discarded, and found again it is drawn again).
        let mut drawn: Vec<usize> = told.new_minima.clone();
        for (_, ends) in &edges {
            for &id in ends {
                let refused = self.network.nodes.get(id).is_some_and(|n| {
                    n.excluded
                        .as_deref()
                        .is_some_and(|r| r.starts_with(REFUSED_REASON))
                });
                if refused && !drawn.contains(&id) {
                    drawn.push(id);
                }
            }
        }
        for &id in &drawn {
            let Some(node) = self.network.nodes.get(id) else {
                continue;
            };
            let refused = node
                .excluded
                .as_deref()
                .is_some_and(|r| r.starts_with(REFUSED_REASON));
            if node.kind != NodeKind::Minimum || (node.excluded.is_some() && !refused) {
                continue;
            }
            let e_pt = match best.get(&id) {
                Some(&e) => e,
                None => {
                    let lowest = edges
                        .iter()
                        .filter(|(_, ends)| ends.contains(&id))
                        .map(|(e, _)| *e)
                        .fold(f64::INFINITY, f64::min);
                    if !lowest.is_finite() {
                        continue;
                    }
                    lowest
                }
            };
            let barrier = (e_pt - e_from).max(0.0);
            // The exponent is at most 0: no overflow.
            let p = (-barrier / kt).exp();
            let mut rng = Rng::from_seed(mix(
                mix(self.settings.seed, job.id as u64 + 1),
                id as u64 + 1,
            ));
            self.network.nodes[id].excluded = (rng.next_f64() >= p).then(|| {
                format!(
                    "{REFUSED_REASON}: p = {p:e} (the highest transition state on the way from \
                     {from} is {:.2} kcal/mol above it; accept_temperature={temperature} K; job \
                     {})",
                    barrier * crate::domain::Units::HARTREE_TO_KCAL_MOL,
                    job.id
                )
            });
        }
        Ok(())
    }

    /// The job in flight with the number `id`, taken out of the jobs in flight.
    fn take_pending(&mut self, id: usize) -> MoptResult<Job> {
        let k = self
            .pending
            .iter()
            .position(|j| j.id == id)
            .ok_or_else(|| {
                MoptError::config(format!(
                    "the mapper was told of job {id}, which it did not ask for"
                ))
            })?;
        Ok(self.pending.remove(k))
    }

    /// The only job in flight (for the calls of one job at a time).
    fn only_pending(&self) -> MoptResult<usize> {
        match self.pending.as_slice() {
            [job] => Ok(job.id),
            [] => Err(MoptError::config(
                "the mapper was told of a job it did not ask for",
            )),
            _ => Err(MoptError::config(
                "several jobs are in flight; tell each by its number (tell_job, fail_job)",
            )),
        }
    }

    /// Drop the job in flight `id` without recording it: it is offered again.
    pub fn drop_job(&mut self, id: usize) {
        self.pending.retain(|j| j.id != id);
        if self.pending.is_empty() {
            // The numbers of the jobs that were not kept are given again.
            self.next_id = after_last(&self.network);
        }
    }

    /// The jobs that were asked for were stopped: nothing of them is kept, and the next
    /// [`Self::ask`] (or [`Self::ask_batch`]) returns them again.
    pub fn abandon(&mut self) {
        self.pending.clear();
        self.next_id = after_last(&self.network);
    }

    /// The job that was asked for failed with `error`; it is recorded and not run again.
    pub fn fail(&mut self, error: &str, trajectory: Option<String>) -> MoptResult<Told> {
        let id = self.only_pending()?;
        self.fail_job(id, error, trajectory)
    }

    /// The job in flight `id` failed with `error`; it is recorded and not run again.
    pub fn fail_job(
        &mut self,
        id: usize,
        error: &str,
        trajectory: Option<String>,
    ) -> MoptResult<Told> {
        let job = self.take_pending(id)?;
        let record = JobRecord {
            id: job.id,
            node: job.node,
            method: self.method.name().to_string(),
            task: job.task.key.clone(),
            description: job.task.description.clone(),
            status: JobStatus::Failed(error.to_string()),
            new_nodes: Vec::new(),
            new_edges: Vec::new(),
            found_new: false,
            found_lower: false,
            trajectory,
        };
        self.network.warnings.push(NetworkWarning {
            job: Some(job.id),
            message: format!("the job failed: {error}"),
        });
        Ok(Told {
            stop: self.after(record),
            ..Told::default()
        })
    }

    /// Record `wavenumbers` (cm^-1, from the Hessian that verified a minimum, or from the start
    /// check of the IRC of a transition state) on `node` where it has none yet.
    pub fn set_frequencies(&mut self, node: usize, wavenumbers: &[f64]) -> MoptResult<()> {
        self.set_spectra(node, wavenumbers, None)
    }

    /// Store the atomic vibrations and the separately measured unstable modes.
    pub(super) fn set_spectra(
        &mut self,
        node: usize,
        wavenumbers: &[f64],
        index: Option<&[f64]>,
    ) -> MoptResult<()> {
        let n = self
            .network
            .nodes
            .get_mut(node)
            .ok_or_else(|| MoptError::config(format!("the network has no node {node}")))?;
        if n.index_wavenumbers.is_none() {
            n.index_wavenumbers = index.map(<[f64]>::to_vec);
        }
        if n.frequencies.is_none() {
            n.frequencies = Some(wavenumbers.to_vec());
        }
        Ok(())
    }

    /// The minimum `system` found by `job`: the known one, or a new node; its wavenumbers,
    /// where they are known, go onto the node.
    pub(super) fn minimum(
        &mut self,
        job: &Job,
        end: &Stationary,
        new: &mut Vec<usize>,
    ) -> MoptResult<usize> {
        self.minimum_on(job, end, job.state, new)
    }

    /// The node `id` found again by `job` as `found`: with
    /// [`DriverSettings::environments`], another environment of its core is listed on it, and
    /// one of a lower energy becomes the node (its old one listed instead); returns `Some(id)`
    /// when the node took the new structure. Nothing otherwise.
    pub(super) fn another_environment(
        &mut self,
        id: usize,
        job: &Job,
        found: &Stationary,
    ) -> Option<usize> {
        if !self.settings.environments {
            return None;
        }
        let node = self.network.nodes.get_mut(id)?;
        let same = self.settings.environment_tolerance;
        if (found.energy - node.energy).abs() <= same {
            return None;
        }
        if found.energy < node.energy {
            node.environments
                .push(super::network::EnvironmentConformer {
                    energy: node.energy,
                    job: node.origin.job,
                });
            node.system = found.system.clone();
            node.energy = found.energy;
            node.frequencies.clone_from(&found.wavenumbers);
            node.index_wavenumbers.clone_from(&found.index_wavenumbers);
            node.free_energy = None;
            node.whole = None;
            self.lowered.push((id, found.system.clone()));
            Some(id)
        } else {
            if !node
                .environments
                .iter()
                .any(|e| (e.energy - found.energy).abs() <= same)
            {
                node.environments
                    .push(super::network::EnvironmentConformer {
                        energy: found.energy,
                        job: Some(job.id),
                    });
            }
            None
        }
    }

    /// Why the priority chose the node of job `job`, where it said
    /// ([`ExplorationPriority::reason`]).
    pub fn reason_of(&self, job: usize) -> Option<&str> {
        self.reasons.get(&job).map(String::as_str)
    }

    /// The whole structure of node `id` where the network holds its core and the node has
    /// none yet (an input).
    pub fn set_whole(&mut self, id: usize, whole: Configuration) {
        if let Some(node) = self.network.nodes.get_mut(id) {
            if node.whole.is_none() {
                node.whole = Some(whole);
            }
        }
    }

    /// [`Self::minimum`] on the state of index `on` (a descent of a crossing point to the
    /// other state).
    pub(super) fn minimum_on(
        &mut self,
        job: &Job,
        end: &Stationary,
        on: usize,
        new: &mut Vec<usize>,
    ) -> MoptResult<usize> {
        let (system, energy) = (&end.system, end.energy);
        let matcher = self.settings.matcher;
        if let Some(why) = self.apart_in_cell(system) {
            return self.periodic_channel(job, end, on, &why);
        }
        if let Some(id) = self.network.find_same_in(
            &NodeKind::Minimum,
            Level::Exploration,
            system,
            energy,
            &matcher,
            self.network.states.get(on),
        )? {
            self.another_environment(id, job, end);
            if !self.settings.environments
                || (end.energy - self.network.node(id)?.energy).abs()
                    <= self.settings.environment_tolerance
            {
                if let Some(w) = &end.wavenumbers {
                    self.set_spectra(id, w, end.index_wavenumbers.as_deref())?;
                }
            }
            return Ok(id);
        }
        let from = self.network.node(job.node)?;
        let input = from.origin.input;
        let state = self.state_for(on, system);
        let node = Node::new(
            NodeKind::Minimum,
            Level::Exploration,
            system.clone(),
            energy,
            state,
            Origin {
                input,
                job: Some(job.id),
                from_node: Some(job.node),
            },
        );
        let id = self.network.add_node(node)?;
        new.push(id);
        if let Some(w) = &end.wavenumbers {
            self.set_spectra(id, w, end.index_wavenumbers.as_deref())?;
        } else if self.settings.verify_minima
            && system
                .structure()
                .is_some_and(|s| s.n_atoms() > 1 || s.has_pbc())
        {
            let label = self.network.node(id)?.label();
            self.network.warnings.push(super::network::NetworkWarning {
                job: Some(job.id),
                message: format!(
                    "{label}: its stability was not judged (the Hessian of its check could \
                         not be had to convergence); kept as a minimum without the check"
                ),
            });
        }
        if self.settings.excluded.contains(&id) {
            self.network.nodes[id].excluded =
                Some("excluded by the user (--exclude_nodes)".to_string());
        } else if self.settings.exclude_bond_rearrangement {
            let seed = input.and_then(|i| self.network.inputs.get(i).and_then(|r| r.node));
            if let Some(seed) = seed {
                if !self.network.same_bonds(id, seed, &matcher)? {
                    let label = self.network.node(seed)?.label();
                    self.network.nodes[id].excluded = Some(format!(
                        "its bonds are not those of {label} (--exclude_bond_rearrangement)"
                    ));
                }
            }
        }
        if self.network.nodes[id].excluded.is_none() {
            self.network.nodes[id].excluded = self.beyond_changes(id, &matcher)?;
        }
        Ok(id)
    }

    /// Why the minimum `system` is a dissociation channel of its periodic cell, if it is: its
    /// molecules in contact do not span the periodic directions
    /// ([`super::channels::cell_contacts`]).
    fn apart_in_cell(&self, system: &Configuration) -> Option<String> {
        let margin = self.settings.cell_contact_margin?;
        let sys = system.structure()?;
        let contacts =
            super::channels::cell_contacts(sys, self.settings.matcher.bond_scale, margin)?;
        contacts.apart().then(|| {
            format!(
                "it came apart in its cell: {}; kept as a dissociation channel, not explored",
                contacts.describe()
            )
        })
    }

    /// The dissociation channel of a minimum `end` of `job` that came apart in its periodic
    /// cell (`why`): a minimum all the same, with its own structure and energy (the parts are
    /// periodic themselves and are not optimized apart), the known node when the structure
    /// matcher finds it again.
    fn periodic_channel(
        &mut self,
        job: &Job,
        end: &Stationary,
        on: usize,
        why: &str,
    ) -> MoptResult<usize> {
        let (system, energy) = (&end.system, end.energy);
        let matcher = self.settings.matcher;
        if let Some(id) = self.network.find_same_in(
            &NodeKind::Dissociated,
            Level::Exploration,
            system,
            energy,
            &matcher,
            self.network.states.get(on),
        )? {
            return Ok(id);
        }
        let from = self.network.node(job.node)?;
        let input = from.origin.input;
        let state = self.state_for(on, system);
        let mut node = Node::new(
            NodeKind::Dissociated,
            Level::Exploration,
            system.clone(),
            energy,
            state,
            Origin {
                input,
                job: Some(job.id),
                from_node: Some(job.node),
            },
        );
        node.excluded = Some(why.to_string());
        let id = self.network.add_node(node)?;
        if let Some(w) = &end.wavenumbers {
            self.set_spectra(id, w, end.index_wavenumbers.as_deref())?;
        }
        let label = self.network.node(id)?.label();
        self.network.warnings.push(super::network::NetworkWarning {
            job: Some(job.id),
            message: format!("{label}: {why}"),
        });
        Ok(id)
    }

    /// Why the new minimum `id` is not explored by the limits of the changes on the way from
    /// its input (`max_bond_changes`, `max_species_changes`), if it is not: the steps of the
    /// chain of the minima the jobs started from at which the bond graph (the SMILES string)
    /// changed are counted.
    fn beyond_changes(&self, id: usize, matcher: &StructureMatcher) -> MoptResult<Option<String>> {
        let (bonds, species) = (
            self.settings.max_bond_changes,
            self.settings.max_species_changes,
        );
        if bonds.is_none() && species.is_none() {
            return Ok(None);
        }
        let (mut bond_changes, mut species_changes) = (0_usize, 0_usize);
        let mut at = id;
        let mut steps = 0;
        // (The chain ends at an input; a network has no more steps than nodes.)
        while let Some(from) = self.network.node(at)?.origin.from_node {
            if steps > self.network.nodes.len() {
                break;
            }
            steps += 1;
            if !self.network.same_bonds(at, from, matcher)? {
                bond_changes += 1;
            }
            let scale = matcher.bond_scale;
            let (a, b) = (self.network.node(at)?, self.network.node(from)?);
            if a.smiles(scale).is_some() && a.smiles(scale) != b.smiles(scale) {
                species_changes += 1;
            }
            at = from;
        }
        let start = self.network.node(at)?.label();
        Ok(match (bonds, species) {
            (Some(most), _) if bond_changes > most => Some(format!(
                "its bonds changed {bond_changes} time(s) on the way from {start}, more than \
                 max_bond_changes={most}"
            )),
            (_, Some(most)) if species_changes > most => Some(format!(
                "its chemical species changed {species_changes} time(s) on the way from \
                 {start}, more than max_species_changes={most}"
            )),
            _ => None,
        })
    }

    /// Apply the line `line` of `steer.txt` ([`super::steer`]); returns what it did, or why
    /// it was not applied (a line that cannot be read changes nothing and does not end the
    /// run).
    pub fn steer(&mut self, line: &str) -> String {
        use super::steer::{parse_line, AppliedLine, SteerCommand, STEERED_REASON};
        let command = match parse_line(line, &self.network) {
            Ok(Some(c)) => c,
            Ok(None) => return "nothing (an empty line or a comment)".to_string(),
            Err(e) => return format!("not applied ({e})"),
        };
        let label = |id: usize| {
            self.network
                .nodes
                .get(id)
                .map_or_else(|| id.to_string(), Node::label)
        };
        let outcome = match command {
            SteerCommand::Explore(id) => {
                let name = label(id);
                if !self.network.steering.first.contains(&id) {
                    self.network.steering.first.push(id);
                }
                format!("the open tasks of {name} run next")
            }
            SteerCommand::Skip(id) => {
                let name = label(id);
                self.network.steering.first.retain(|&n| n != id);
                match &self.network.nodes[id].excluded {
                    Some(why) => format!("{name} is not explored already ({why})"),
                    None => {
                        self.network.nodes[id].excluded = Some(STEERED_REASON.to_string());
                        format!("no more jobs start from {name}")
                    }
                }
            }
            SteerCommand::Unskip(id) => {
                let name = label(id);
                if self.network.nodes[id].excluded.as_deref() == Some(STEERED_REASON) {
                    self.network.nodes[id].excluded = None;
                    format!("{name} is explored again")
                } else {
                    format!("{name} was not skipped by a line of steer.txt; nothing changes")
                }
            }
            SteerCommand::MaxJobs(n) => {
                self.network.steering.max_jobs = Some(n);
                // A run that had ended on its limit of jobs goes on.
                if self
                    .finished
                    .as_deref()
                    .is_some_and(|r| r.starts_with("The limit of"))
                {
                    self.finished = None;
                }
                if n == 0 {
                    "no limit of jobs from now on".to_string()
                } else {
                    format!("the limit of jobs is {n} from now on")
                }
            }
        };
        self.network.steering.applied.push(AppliedLine {
            after_jobs: self.network.jobs.len(),
            line: line.trim().to_string(),
        });
        outcome
    }

    /// What the job that was asked for found.
    pub fn tell(&mut self, outcome: &JobOutcome, trajectory: Option<String>) -> MoptResult<Told> {
        let id = self.only_pending()?;
        self.tell_job(id, outcome, trajectory)
    }

    /// What the job in flight `id` found (an outcome that was interrupted is not kept, and
    /// the job is offered again).
    pub fn tell_job(
        &mut self,
        id: usize,
        outcome: &JobOutcome,
        trajectory: Option<String>,
    ) -> MoptResult<Told> {
        if outcome.interrupted {
            self.take_pending(id)?;
            self.drop_job(id);
            return Ok(Told::default());
        }
        let job = self.take_pending(id)?;
        let matcher = self.settings.matcher;
        let lowest = self.network.lowest_minimum();
        let mut told = Told::default();
        let mut new_nodes = Vec::new();
        // The saddle points of higher order of the job first: the ends of reactions and of
        // descents refer to them.
        let mut saddle_of: Vec<usize> = Vec::with_capacity(outcome.saddles.len());
        for f in &outcome.saddles {
            let kind = NodeKind::of_order(f.order)?;
            let id = match self.network.find_same_in(
                &kind,
                Level::Exploration,
                &f.point.system,
                f.point.energy,
                &matcher,
                self.network.states.get(job.state),
            )? {
                Some(id) => id,
                None => {
                    let input = self.network.node(job.node)?.origin.input;
                    let id = self.network.add_node(Node::new(
                        kind,
                        Level::Exploration,
                        f.point.system.clone(),
                        f.point.energy,
                        self.state_for(job.state, &f.point.system),
                        Origin {
                            input,
                            job: Some(job.id),
                            from_node: Some(job.node),
                        },
                    ))?;
                    new_nodes.push(id);
                    if let Some(w) = &f.point.wavenumbers {
                        self.set_spectra(id, w, f.point.index_wavenumbers.as_deref())?;
                    }
                    id
                }
            };
            saddle_of.push(id);
        }
        // The node of the transition state of every reaction, for the ends of later reactions
        // at it (TS–TS links).
        let mut ts_of: Vec<Option<usize>> = Vec::with_capacity(outcome.reactions.len());
        // The known transition states the job reached (for the acceptance filter).
        let mut known_ts: Vec<usize> = Vec::new();
        for (k, r) in outcome.reactions.iter().enumerate() {
            if let Some(known) = self.network.find_same_in(
                &NodeKind::TransitionState,
                Level::Exploration,
                &r.ts.system,
                r.ts.energy,
                &matcher,
                self.network.states.get(job.state),
            )? {
                told.known += 1;
                ts_of.push(Some(known));
                known_ts.push(known);
                self.another_environment(known, &job, &r.ts);
                // Its ends, minima of the network, may have been reached in other
                // environments of their cores too.
                if self.settings.environments {
                    for end in r.ends.iter().flatten() {
                        if let Some(m) = self.network.find_same_in(
                            &NodeKind::Minimum,
                            Level::Exploration,
                            &end.system,
                            end.energy,
                            &matcher,
                            self.network.states.get(job.state),
                        )? {
                            self.another_environment(m, &job, end);
                        }
                    }
                }
                continue;
            }
            // The nodes of the ends; `None` where the end of a side is unknown (its
            // minimization did not converge).
            let mut ends: [Option<usize>; 2] = [None, None];
            let mut settled = true;
            for (s, end) in r.ends.iter().enumerate() {
                ends[s] = match r.saddle_ends[s] {
                    Some(SaddleEnd::Reaction(j)) => match ts_of.get(j).copied().flatten() {
                        Some(id) => Some(id),
                        None => {
                            settled = false;
                            None
                        }
                    },
                    Some(SaddleEnd::Known(id)) => {
                        if !self.network.node(id)?.kind.is_saddle() {
                            return Err(MoptError::config(format!(
                                "job {}: the end of a reaction at node {id}, which is not a \
                                 saddle point",
                                job.id
                            )));
                        }
                        Some(id)
                    }
                    Some(SaddleEnd::Higher(i)) => match saddle_of.get(i) {
                        Some(&id) => Some(id),
                        None => {
                            settled = false;
                            None
                        }
                    },
                    None => match end {
                        Some(end) => {
                            Some(self.end_node(&job, end, &mut told.new_minima, &mut new_nodes)?)
                        }
                        None => None,
                    },
                };
            }
            if !settled {
                // The transition state its end reached gave no reaction of this job.
                self.network.warnings.push(NetworkWarning {
                    job: Some(job.id),
                    message: format!(
                        "a reaction whose IRC ends at a transition state that was not added is \
                         left out (reaction {} of the job)",
                        k + 1
                    ),
                });
                ts_of.push(None);
                continue;
            }
            // A channel has the energy of its fragments as the network knows them; a transition
            // state at the end (a TS–TS link) its own.
            let end_energy = |id: usize, end: &Stationary| match self.network.nodes.get(id) {
                Some(n) if n.kind == NodeKind::Dissociated || n.kind.is_saddle() => n.energy,
                _ => end.energy,
            };
            let barriers: [Option<f64>; 2] = std::array::from_fn(|s| {
                let (id, end) = (ends[s]?, r.ends[s].as_ref()?);
                Some(r.ts.energy - end_energy(id, end))
            });
            let input = self.network.node(job.node)?.origin.input;
            let ts = self.network.add_node(Node::new(
                NodeKind::TransitionState,
                Level::Exploration,
                r.ts.system.clone(),
                r.ts.energy,
                self.state_for(job.state, &r.ts.system),
                Origin {
                    input,
                    job: Some(job.id),
                    from_node: Some(job.node),
                },
            ))?;
            new_nodes.push(ts);
            if let Some(w) = &r.ts.wavenumbers {
                self.set_spectra(ts, w, r.ts.index_wavenumbers.as_deref())?;
            }
            let number = self.network.node(ts)?.number;
            let mut irc = IrcPath {
                sides: r.branches.iter().map(|b| b.side.clone()).collect(),
                file: Some(super::observers::irc_file(
                    number,
                    self.network.variables.is_some(),
                )),
                possible_bifurcations: r
                    .branches
                    .iter()
                    .filter_map(|b| b.possible_bifurcation.clone())
                    .collect(),
            };
            ts_of.push(Some(ts));
            // The minima a confirmed bifurcation found, beside the end of their side.
            for (side, found) in &r.bifurcation_ends {
                let mut ids = Vec::new();
                for end in found {
                    let id = self.end_node(&job, end, &mut told.new_minima, &mut new_nodes)?;
                    if Some(id) != ends[*side] && !ids.contains(&id) {
                        ids.push(id);
                    }
                }
                let name = r.branches.get(*side).map_or("", |b| b.side.name.as_str());
                let label = self.network.node(ts)?.label();
                let others: Vec<String> = ids
                    .iter()
                    .filter_map(|&id| self.network.node(id).ok().map(Node::label))
                    .collect();
                if let Some(b) = irc
                    .possible_bifurcations
                    .iter_mut()
                    .find(|b| b.side == name)
                {
                    b.confirmed_ends.clone_from(&ids);
                }
                self.network.warnings.push(NetworkWarning {
                    job: Some(job.id),
                    message: if others.is_empty() {
                        format!(
                            "{label}: the possible bifurcation of the {name} side of its IRC was \
                             tried (both ways across the path, minimized): nothing beside the \
                             end of the side"
                        )
                    } else {
                        format!(
                            "{label}: the possible bifurcation of the {name} side of its IRC is \
                             confirmed: across the path the side also leads to {}",
                            others.join(", ")
                        )
                    },
                });
            }
            for (s, which) in r.saddle_ends.iter().enumerate() {
                if let (Some(_), Some(end)) = (which, ends[s]) {
                    let side = r.branches.get(s).map_or("", |b| b.side.name.as_str());
                    let (from, to) = (
                        self.network.node(ts)?.label(),
                        self.network.node(end)?.label(),
                    );
                    self.network.warnings.push(NetworkWarning {
                        job: Some(job.id),
                        message: format!(
                            "{from}: the {side} side of its IRC ends at the transition state \
                             {to} (a TS–TS link: past a valley-ridge inflection the path runs \
                             down a ridge into {to}, so the products {to} leads to are reached \
                             from {from} without a minimum between)"
                        ),
                    });
                }
            }
            // An end in another environment of its core than its node (ONIOM): the edge
            // keeps its own barrier, and the difference is reported.
            if self.settings.environments {
                for (s, end) in r.ends.iter().enumerate() {
                    let (Some(id), Some(end)) = (ends[s], end.as_ref()) else {
                        continue;
                    };
                    let node = self.network.node(id)?;
                    if node.kind != NodeKind::Minimum {
                        continue;
                    }
                    let difference = end.energy - node.energy;
                    if difference.abs() > ENVIRONMENT_HETEROGENEITY {
                        let (from, to) = (self.network.node(ts)?.label(), node.label());
                        let side = r.branches.get(s).map_or("", |b| b.side.name.as_str());
                        self.network.warnings.push(NetworkWarning {
                            job: Some(job.id),
                            message: format!(
                                "{from}: the {side} end of its IRC is {to} in another \
                                 environment of its core, {:.2} kcal/mol from that of the node \
                                 (environment heterogeneity: the edge keeps the barrier of its \
                                 own end)",
                                difference * crate::domain::Units::HARTREE_TO_KCAL_MOL
                            ),
                        });
                    }
                }
            }
            match (ends, barriers) {
                ([Some(first), Some(second)], [Some(b_first), Some(b_second)]) => {
                    let edge = self.network.add_ts_edge(TsEdge {
                        id: 0,
                        level: Level::Exploration,
                        ts,
                        ends: [first, second],
                        barriers: [b_first, b_second],
                        imaginary_wavenumber: r.imaginary_wavenumber,
                        irc,
                        job: Some(job.id),
                    })?;
                    told.new_edges.push((k, edge));
                }
                _ => {
                    // An end that is not known: the transition state is kept with its IRC and
                    // the end that is known, apart from the edges of the kinetics.
                    let unknown: Vec<String> = irc
                        .sides
                        .iter()
                        .zip(&ends)
                        .filter(|(_, end)| end.is_none())
                        .map(|(side, _)| side.name.clone())
                        .collect();
                    let label = self.network.node(ts)?.label();
                    let open = self.network.add_open_ts_edge(OpenTsEdge {
                        id: 0,
                        level: Level::Exploration,
                        ts,
                        ends,
                        barriers,
                        imaginary_wavenumber: r.imaginary_wavenumber,
                        irc,
                        job: Some(job.id),
                    })?;
                    told.new_open_edges.push((k, open));
                    self.network.warnings.push(NetworkWarning {
                        job: Some(job.id),
                        message: format!(
                            "{label}: the end of the {} side of its IRC is unknown; the transition \
                             state is kept with its known connections and is left out of the kinetics",
                            unknown.join(" and the ")
                        ),
                    });
                }
            }
        }
        // The descents of the saddle points of higher order.
        for (f, &saddle) in outcome.saddles.iter().zip(&saddle_of) {
            for d in &f.descents {
                let mut ends = [None, None];
                for (k, end) in d.ends.iter().enumerate() {
                    ends[k] = match end {
                        DescentEnd::Unsettled => None,
                        DescentEnd::Minimum(st) => {
                            Some(self.end_node(&job, st, &mut told.new_minima, &mut new_nodes)?)
                        }
                        DescentEnd::Reaction(r) => ts_of.get(*r).copied().flatten(),
                        DescentEnd::Known(id) => Some(*id),
                        DescentEnd::Higher(i) => saddle_of.get(*i).copied(),
                    };
                }
                self.network.add_saddle_edge(SaddleEdge {
                    id: 0,
                    level: Level::Exploration,
                    saddle,
                    mode: d.mode,
                    wavenumber: d.wavenumber,
                    ends,
                    job: Some(job.id),
                })?;
            }
        }
        // The path peaks: records beside the network. Their ends are minima like any other.
        for (k, p) in outcome.peaks.iter().enumerate() {
            let mut known = false;
            for q in &self.network.path_peaks {
                if q.level == Level::Exploration
                    && q.system
                        .same_as(q.energy, &p.point.system, p.point.energy, &matcher)?
                {
                    known = true;
                    break;
                }
            }
            if known {
                continue;
            }
            let mut ends = [None, None];
            for (s, end) in p.ends.iter().enumerate() {
                if let Some(end) = end {
                    ends[s] =
                        Some(self.end_node(&job, end, &mut told.new_minima, &mut new_nodes)?);
                }
            }
            if ends[0].is_some() && ends[0] == ends[1] {
                // Both sides at one node of the network: no barrier between two.
                continue;
            }
            let mut barriers = [None, None];
            for (s, end) in ends.iter().enumerate() {
                if let Some(id) = end {
                    barriers[s] = Some(p.point.energy - self.network.node(*id)?.energy);
                }
            }
            let id = self.network.path_peaks.len();
            self.network.path_peaks.push(super::peak::PathPeak {
                id,
                level: Level::Exploration,
                system: p.point.system.clone(),
                energy: p.point.energy,
                ends,
                barriers,
                basis: p.basis.clone(),
                job: Some(job.id),
            });
            told.new_peaks.push((k, id));
        }
        // The crossing points between two states and the minima of their descents.
        self.tell_crossings(&job, outcome, &mut told, &mut new_nodes)?;
        for w in &outcome.warnings {
            self.network.warnings.push(NetworkWarning {
                job: Some(job.id),
                message: w.clone(),
            });
        }
        if let Some(t) = self.settings.accept_temperature {
            self.accept(&job, &told, &known_ts, t)?;
        }
        new_nodes.extend(told.new_minima.iter().copied());
        new_nodes.sort_unstable();
        // The followed electronic states at the structures of the new nodes.
        for &id in &new_nodes {
            self.set_followed(id, &outcome.followed)?;
        }
        // The whole structures of the new nodes and of those that took a lower environment,
        // where the network holds the cores of a larger system.
        let lowered = std::mem::take(&mut self.lowered);
        if !outcome.wholes.is_empty() {
            for &id in &new_nodes {
                if let Some(node) = self.network.nodes.get_mut(id) {
                    if node.whole.is_none() {
                        node.whole = outcome.whole_of(&node.system).cloned();
                    }
                }
            }
            for (id, system) in &lowered {
                if let Some(node) = self.network.nodes.get_mut(*id) {
                    node.whole = outcome.whole_of(system).cloned();
                }
            }
        }
        told.lowered = lowered.iter().map(|(id, _)| *id).collect();
        told.lowered.sort_unstable();
        told.lowered.dedup();
        let found_lower = match lowest {
            Some(low) => told
                .new_minima
                .iter()
                .any(|&id| self.network.nodes[id].energy < low),
            None => !told.new_minima.is_empty(),
        };
        let record = JobRecord {
            id: job.id,
            node: job.node,
            method: self.method.name().to_string(),
            task: job.task.key.clone(),
            description: job.task.description.clone(),
            status: JobStatus::Done,
            new_nodes,
            new_edges: told.new_edges.iter().map(|(_, e)| *e).collect(),
            found_new: !told.new_minima.is_empty(),
            found_lower,
            trajectory,
        };
        told.stop = self.after(record);
        Ok(told)
    }
}

#[path = "driver_channels.rs"]
mod channels;

#[path = "driver_states.rs"]
mod states;

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;
