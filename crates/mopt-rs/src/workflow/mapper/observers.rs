//! What the exploration loop tells the outside: a [`MapperObserver`] sees every structure of
//! every job as it is computed, the network after every finished job, after every batch of
//! jobs that is kept, and at the end. The files of a run are observers ([`TrajectoryFiles`],
//! [`PathFiles`], [`StoreObserver`]); the command adds its report, and users add their own (a
//! progress display, a check that stops the run).
//!
//! When several jobs run at the same time ([`super::schedule`]), every job is started
//! ([`MapperObserver::on_start`]) before the structures of any of them come, the structures of
//! the jobs come mixed (each with its job, and those of one job in their order), and the jobs
//! are told one after the other in the order of their numbers when their batch is kept. All
//! calls come from the thread of the exploration loop.
//!
//! File names are fixed, so that a client finds the results of a run: `jobs/input_000_traj.xyz`
//! (the optimization of an input), `jobs/job_000003/trajectories.csv` (the ordered stage files
//! of a job), `jobs/job_000003.log` (the log of a job, written by the command),
//! `jobs/job_000003/` (the files and the full log of the searches of a job: ADDF, the
//! two-point methods),
//! `paths/TS000007_irc.xyz` (the IRC of a transition state: the backward side reversed, the
//! transition state, the forward side; before and after them the minimizations of the two
//! ends, so that the file runs from minimum to minimum), and the files of [`super::store`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's mapper writes the directories of
//!  AutoTS for every job, `runs/<iteration>/autots_workspace/`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::configuration::{frame_extension, Configuration};
use super::driver::Told;
use super::method::{Job, JobOutcome};
use super::network::{JobStatus, Network};
use super::store::{write_whole, NetworkStore};
use crate::core_infra::json::JsonValue;
use crate::error::MoptResult;
use crate::workflow::trajectories::GrowingTrajectory;

/// Publish the completion history from the same job records as the restart checkpoint.
pub(crate) fn write_exploration_history(dir: &Path, network: &Network) -> MoptResult<()> {
    use std::fmt::Write;
    let mut text = String::new();
    for job in &network.jobs {
        let origin = network.node(job.node)?;
        let task = job.description.replace(['\r', '\n'], " ");
        let status = match &job.status {
            JobStatus::Done => "DONE".to_string(),
            JobStatus::Failed(reason) => format!("FAILED ({})", reason.replace(['\r', '\n'], " ")),
        };
        let _ = writeln!(text, "Iter: {:06} | Node: {}{:06} | Method: {} | Task: {} | Pair/parameters: {} | Status: {} | New nodes: {} | New TS connections: {}",
            job.id, origin.kind.tag(), origin.number, job.method, task, job.task, status, job.new_nodes.len(), job.new_edges.len());
    }
    write_whole(&dir.join("exploration_history.log"), text.as_bytes())
}

/// The directory of the trajectories of the jobs.
pub const JOBS_DIR: &str = "jobs";
/// The directory of the IRC paths.
pub const PATHS_DIR: &str = "paths";

/// What a structure belongs to.
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    /// The optimization of input structure `k`.
    Input(usize),
    /// A job.
    Job(&'a Job),
}

/// The trajectory of `source`, relative to the run directory.
pub fn trajectory_file(source: &Source<'_>, variables: bool) -> String {
    let ext = frame_extension(variables);
    match source {
        Source::Input(k) => format!("{JOBS_DIR}/input_{k:03}_traj.{ext}"),
        Source::Job(job) => format!("{}/trajectories.csv", job_files_dir(job.id)),
    }
}

/// The log of job `id`, relative to the run directory.
pub fn job_log_file(id: usize) -> String {
    format!("{JOBS_DIR}/job_{id:06}.log")
}

/// The directory of the files of the searches of job `id` (ADDF, the two-point methods),
/// relative to the run directory.
pub fn job_files_dir(id: usize) -> String {
    format!("{JOBS_DIR}/job_{id:06}")
}

/// The IRC path of the transition state `TS<number>` (numbered among the transition states),
/// relative to the run directory.
pub fn irc_file(number: usize, variables: bool) -> String {
    format!(
        "{PATHS_DIR}/TS{number:06}_irc.{}",
        frame_extension(variables)
    )
}

/// How an exploration ended.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExplorationEnd {
    /// Why it ended (`None` when it was interrupted).
    pub reason: Option<String>,
    /// Whether it was stopped (a `STOP` file): `--restart RUN_DIR` continues it.
    pub interrupted: bool,
}

/// Sees the exploration.
pub trait MapperObserver: Send {
    /// The optimization of an input or a job starts.
    fn on_start(&mut self, source: &Source<'_>) -> MoptResult<()> {
        let _ = source;
        Ok(())
    }

    /// A point of `source` was computed.
    fn on_frame(
        &mut self,
        source: &Source<'_>,
        system: &Configuration,
        energy: f64,
        label: &str,
    ) -> MoptResult<()> {
        let _ = (source, system, energy, label);
        Ok(())
    }

    /// The trajectory this observer keeps of `source`, relative to the run directory.
    fn trajectory(&self, source: &Source<'_>) -> Option<String> {
        let _ = source;
        None
    }

    /// The network after the inputs were optimized.
    fn after_inputs(&mut self, network: &Network) -> MoptResult<()> {
        let _ = network;
        Ok(())
    }

    /// The network after `job` was told (`outcome` is what it found, `told` what was new).
    fn after_job(
        &mut self,
        network: &Network,
        job: &Job,
        outcome: &JobOutcome,
        told: &Told,
    ) -> MoptResult<()> {
        let _ = (network, job, outcome, told);
        Ok(())
    }

    /// The network after the jobs `jobs` (numbers), chosen together, were told: a state that
    /// a restart goes on from (one job at a time: after every job).
    fn after_batch(&mut self, network: &Network, jobs: &[usize]) -> MoptResult<()> {
        let _ = (network, jobs);
        Ok(())
    }

    /// The exploration ended (also when it was interrupted).
    fn on_finish(&mut self, network: &Network, end: &ExplorationEnd) -> MoptResult<()> {
        let _ = (network, end);
        Ok(())
    }
}

/// The stage name from an optimization iteration, search evaluation or IRC point label.
pub(crate) fn frame_stage(label: &str) -> &str {
    let event = label
        .split_once(" states=")
        .map_or(label, |(event, _)| event);
    let Some((prefix, number)) = event.rsplit_once(' ') else {
        return label;
    };
    if number.parse::<usize>().is_err() {
        return label;
    }
    [" point", " iteration", " step"]
        .iter()
        .find_map(|suffix| prefix.strip_suffix(suffix))
        .unwrap_or(label)
}

/// A safe filename within the job directory, numbered in the order stages are reached.
pub(crate) fn stage_file(number: usize, stage: &str, variables: bool) -> String {
    let stem: String = stage
        .chars()
        .take(80)
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("{number:04}_{stem}.{}", frame_extension(variables))
}

struct StageTrajectory {
    stage: String,
    number: usize,
    output: GrowingTrajectory,
}

/// Input trajectories and ordered, stage-specific job trajectories.
pub struct TrajectoryFiles {
    dir: PathBuf,
    open: BTreeMap<String, GrowingTrajectory>,
    atoms: Option<Vec<crate::domain::AtomId>>,
    jobs: BTreeMap<usize, StageTrajectory>,
    save_jobs: bool,
    variables: bool,
}

impl TrajectoryFiles {
    /// Trajectories under the run directory `dir` (frames of plain variables, `.txt`, with
    /// `variables`; xyz frames otherwise).
    pub fn new(dir: &Path, variables: bool) -> Self {
        Self {
            dir: dir.to_path_buf(),
            open: BTreeMap::new(),
            atoms: None,
            jobs: BTreeMap::new(),
            save_jobs: true,
            variables,
        }
    }

    /// Enable job trajectories independently of the input optimization.
    pub fn with_job_steps(mut self, enabled: bool) -> Self {
        self.save_jobs = enabled;
        self
    }

    /// Select the atoms written to job and input XYZ trajectories.
    pub fn with_atoms(mut self, atoms: Option<Vec<crate::domain::AtomId>>) -> Self {
        self.atoms = atoms;
        self
    }
}

impl MapperObserver for TrajectoryFiles {
    fn on_start(&mut self, source: &Source<'_>) -> MoptResult<()> {
        if let Source::Job(job) = source {
            if self.save_jobs {
                self.jobs.remove(&job.id);
                write_whole(
                    &self.dir.join(trajectory_file(source, self.variables)),
                    b"stage,file\n",
                )?;
            }
            return Ok(());
        }
        let file = trajectory_file(source, self.variables);
        let path = self.dir.join(&file);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if matches!(source, Source::Input(_)) {
            // The inputs are optimized before any job, one after the other.
            self.open.clear();
        }
        self.open.insert(file, GrowingTrajectory::start(path)?);
        Ok(())
    }

    fn on_frame(
        &mut self,
        source: &Source<'_>,
        system: &Configuration,
        energy: f64,
        label: &str,
    ) -> MoptResult<()> {
        let target = match source {
            Source::Input(_) => self.open.get_mut(&trajectory_file(source, self.variables)),
            Source::Job(job) if self.save_jobs => {
                use std::io::Write;
                let stage = frame_stage(label);
                let current = self.jobs.get(&job.id);
                if current.is_none_or(|t| t.stage != stage) {
                    let number = current.map_or(1, |t| t.number + 1);
                    let file = stage_file(number, stage, self.variables);
                    let dir = self.dir.join(job_files_dir(job.id));
                    let output = GrowingTrajectory::start(dir.join(&file))?;
                    let mut list = std::fs::OpenOptions::new()
                        .append(true)
                        .open(dir.join("trajectories.csv"))?;
                    writeln!(list, "\"{}\",{file}", stage.replace('"', "\"\""))?;
                    self.jobs.insert(
                        job.id,
                        StageTrajectory {
                            stage: stage.to_string(),
                            number,
                            output,
                        },
                    );
                }
                self.jobs.get_mut(&job.id).map(|s| &mut s.output)
            }
            Source::Job(_) => None,
        };
        if let Some(t) = target {
            let comment = format!("{label} energy={energy:.12}");
            if let (Some(atoms), Some(s)) = (&self.atoms, system.structure()) {
                let selected = Configuration::Structure(s.trajectory_atoms(atoms)?);
                t.push(&selected.frame(&format!("{comment} oniom_layer=high")))?;
            } else {
                t.push(&system.frame(&comment))?;
            }
        }
        Ok(())
    }

    fn trajectory(&self, source: &Source<'_>) -> Option<String> {
        (self.save_jobs || matches!(source, Source::Input(_)))
            .then(|| trajectory_file(source, self.variables))
    }

    fn after_job(
        &mut self,
        _network: &Network,
        job: &Job,
        _outcome: &JobOutcome,
        _told: &Told,
    ) -> MoptResult<()> {
        self.jobs.remove(&job.id);
        Ok(())
    }
}

/// The IRC path of every new transition-state edge.
pub struct PathFiles {
    dir: PathBuf,
}

impl PathFiles {
    /// Paths under the run directory `dir`.
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
        }
    }
}

impl MapperObserver for PathFiles {
    fn after_job(
        &mut self,
        network: &Network,
        _job: &Job,
        outcome: &JobOutcome,
        told: &Told,
    ) -> MoptResult<()> {
        // The nodes that took a lower environment of their core: their files again.
        for &id in &told.lowered {
            if let Some(node) = network.nodes.get(id) {
                write_whole(
                    &self.dir.join(super::store::structure_file(node)),
                    super::store::node_file_text(node).as_bytes(),
                )?;
            }
        }
        let files = told
            .new_edges
            .iter()
            .map(|&(reaction, edge)| (reaction, network.ts_edges.get(edge).map(|e| &e.irc)))
            .chain(told.new_open_edges.iter().map(|&(reaction, edge)| {
                (reaction, network.open_ts_edges.get(edge).map(|e| &e.irc))
            }));
        for (reaction, irc) in files {
            let (Some(r), Some(irc)) = (outcome.reactions.get(reaction), irc) else {
                continue;
            };
            let Some(file) = &irc.file else {
                continue;
            };
            let mut text = String::new();
            // The whole structure where the job has it (the core of an ONIOM surface).
            let mut frame = |point: &Configuration, energy: f64, label: &str| {
                let point = outcome.whole_of(point).unwrap_or(point);
                text.push_str(&point.frame(&format!("{label} energy={energy:.12}")));
            };
            // From the minimum of the backward side (the minimization of the end of the side,
            // reversed) along the IRC to the minimum of the forward side: an IRC stops before
            // its minimum, and the minimization is the rest of the way.
            if let Some(back) = r.branches.first() {
                for p in back.settle.iter().rev() {
                    frame(&p.system, p.energy, "backward minimization");
                }
                for (s, e) in back.frames.iter().zip(&back.side.energies).rev() {
                    frame(s, *e, "backward");
                }
            }
            frame(&r.ts.system, r.ts.energy, "TS");
            for branch in r.branches.iter().skip(1) {
                for (s, e) in branch.frames.iter().zip(&branch.side.energies) {
                    frame(s, *e, &branch.side.name);
                }
                for p in &branch.settle {
                    frame(
                        &p.system,
                        p.energy,
                        &format!("{} minimization", branch.side.name),
                    );
                }
            }
            write_whole(&self.dir.join(file), text.as_bytes())?;
        }
        // The path of every new path peak between its two sides: the minimization of the
        // first side backwards, the points of the path through the peak, the minimization of
        // the second side.
        let variables = network.variables.is_some();
        for &(peak, id) in &told.new_peaks {
            let Some(p) = outcome.peaks.get(peak) else {
                continue;
            };
            let mut text = String::new();
            let mut frame = |point: &Configuration, energy: f64, label: &str| {
                text.push_str(&point.frame(&format!("{label} energy={energy:.12}")));
            };
            for q in p.settles[0].iter().rev() {
                frame(&q.system, q.energy, "backward minimization");
            }
            for q in &p.path {
                frame(&q.system, q.energy, "path");
            }
            for q in &p.settles[1] {
                frame(&q.system, q.energy, "forward minimization");
            }
            write_whole(
                &self.dir.join(super::peak::peak_path_file(id, variables)),
                text.as_bytes(),
            )?;
        }
        // The path of every new crossing edge between its two minima: from the minimum of the
        // first state (the minimization of the end of its meta-IRC backwards, then the
        // meta-IRC backwards) up to the crossing point, then down the second state to its
        // minimum. Every frame has the energy of the state it lies on.
        for &(crossing, edge) in &told.new_crossings {
            let (Some(c), Some(e)) = (
                outcome.crossings.get(crossing),
                network.crossing_edges.get(edge),
            ) else {
                continue;
            };
            let Some(file) = &e.irc.file else {
                continue;
            };
            let mut text = String::new();
            let mut frame = |point: &Configuration, energy: f64, label: &str| {
                text.push_str(&point.frame(&format!("{label} energy={energy:.12}")));
            };
            let surface = |side: usize| super::surface_name(c.states[side]);
            for p in c.settles[0].iter().rev() {
                frame(&p.system, p.energy, &format!("{} minimization", surface(0)));
            }
            if let Some(first) = &c.paths[0] {
                for (s, e) in first.frames.iter().zip(&first.side.energies).rev() {
                    frame(s, *e, &first.side.name);
                }
            }
            frame(&c.point.system, c.point.energy, c.kind.tag());
            if let Some(second) = &c.paths[1] {
                for (s, e) in second.frames.iter().zip(&second.side.energies) {
                    frame(s, *e, &second.side.name);
                }
            }
            for p in &c.settles[1] {
                frame(&p.system, p.energy, &format!("{} minimization", surface(1)));
            }
            write_whole(&self.dir.join(file), text.as_bytes())?;
        }
        Ok(())
    }
}

/// Writes the network to its stores after the inputs, after every batch of jobs that is kept
/// (after every job when one job runs at a time), and at the end; `run` gives the entry `run`
/// of the network file for the network at that moment. The network of a store is therefore
/// always one that a restart goes on from to the network of a run that was not stopped.
pub struct StoreObserver<F: Fn(&Network, &ExplorationEnd) -> JsonValue + Send> {
    dir: PathBuf,
    stores: Vec<Box<dyn NetworkStore>>,
    run: F,
}

impl<F: Fn(&Network, &ExplorationEnd) -> JsonValue + Send> StoreObserver<F> {
    /// Write to `stores` in the run directory `dir`.
    pub fn new(dir: &Path, stores: Vec<Box<dyn NetworkStore>>, run: F) -> Self {
        Self {
            dir: dir.to_path_buf(),
            stores,
            run,
        }
    }

    fn write(&self, network: &Network, end: &ExplorationEnd) -> MoptResult<()> {
        let run = (self.run)(network, end);
        for store in &self.stores {
            store.write(network, &run, &self.dir)?;
        }
        write_exploration_history(&self.dir, network)
    }
}

impl<F: Fn(&Network, &ExplorationEnd) -> JsonValue + Send> MapperObserver for StoreObserver<F> {
    fn after_inputs(&mut self, network: &Network) -> MoptResult<()> {
        self.write(network, &ExplorationEnd::default())
    }

    fn after_batch(&mut self, network: &Network, _jobs: &[usize]) -> MoptResult<()> {
        self.write(network, &ExplorationEnd::default())
    }

    fn on_finish(&mut self, network: &Network, end: &ExplorationEnd) -> MoptResult<()> {
        self.write(network, end)
    }
}
