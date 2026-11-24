//! The loop of an exploration: the input structures optimized to minima, then the jobs of the
//! driver ([`super::driver::MapperDriver`] asks, [`MapperJobs`] computes, the driver is told),
//! with the observers told of every structure and of the network after every job and every
//! batch of jobs that is kept.
//!
//! [`explore`] runs one job after the other with one [`MapperJobs`]. [`explore_with`] runs the
//! jobs as a [`Schedule`] says, every job with the calculations of [`MapperWorkers::job`]:
//!
//! - in batches (the default): the jobs chosen together run (one after the other with one
//!   worker, in threads of their own with several), and when all of them have ended they are
//!   told in the order of their numbers, so that the network does not depend on which job
//!   ends first;
//! - rolling: a job starts whenever a worker is free and is told when it ends.
//!
//! With several workers every worker has a thread pool of its own with its share of the
//! threads of the pool of the loop, for what a job computes in parallel: jobs that each woke
//! every thread of a pool for the small linear algebra of their optimizations spent more time
//! waking threads than computing (the measurements in `docs/source/mapper.md`).
//!
//! What a job says (its optimizations, transition-state searches, IRCs, and what the
//! calculator says) is kept by its own calculations ([`MapperWorkers::job`],
//! [`MapperJobs::take_lines`]) and goes to the log of the run as one block, headed
//! `Job N (from EQk, task ...)`, when the job is kept: in the order of the numbers of the jobs,
//! as one job after the other would write them, whatever order the jobs end in (in the rolling
//! schedule a block waits for those of the jobs before it), and only for jobs that are kept, so
//! that a run stopped and restarted writes the same blocks in the same order. The screen
//! ([`MapperJobs::progress`]) gets a short line when a job starts and when it ends. The lines of
//! jobs that run at the same time are never mixed, and nothing of a job is written from its
//! thread to the screen or the log of the run.
//!
//! A `STOP` (a stop within jobs) ends the loop with [`ExplorationEnd::interrupted`]: the jobs
//! that run are stopped, and nothing of their batch is kept (in the rolling mode: of the jobs
//! that had not ended), so that a restart from the network written after the last batch that
//! was kept chooses the same jobs again and gets their evaluations back from the journal of the
//! run. A stop between jobs (MultiOptPy's `stop.txt`) lets the jobs that run end, keeps them,
//! and ends the loop. A job that runs longer than the time limit of the schedule is stopped and
//! recorded as failed. A job whose calculation fails (an error of the calculator, of the
//! convergence, of the linear algebra, of the coordinates) is recorded as failed and the loop
//! goes on; any other error (settings, files) ends it.
//!
//! MultiOptPy: `Wrapper/mapper.py:ReactionNetworkMapper.run`, `_run_sequential`,
//!  `_run_parallel_rolling`, `_register_seed_structure`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use super::configuration::Configuration;
use super::dissociation::{dissociation_of, Discard};
use super::driver::{MapperBatch, MapperDriver, NextJob, Told};
use super::method::{
    ExplorationMethod, FrameSink, Job, JobContext, JobOutcome, MapperJobs, OptimizeRequest, Stage,
};
use super::network::{Network, NetworkWarning, Origin};
use super::observers::{ExplorationEnd, MapperObserver, Source};
use super::schedule::{MapperWorkers, Schedule, ScheduleMode};
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::identity::StructureMatcher;

/// The structures of a source go to every observer.
struct FanOut<'a, 'b, 'c> {
    observers: &'a mut [&'b mut dyn MapperObserver],
    source: Source<'c>,
}

impl FrameSink for FanOut<'_, '_, '_> {
    fn push(&mut self, system: &Configuration, energy: f64, label: &str) -> MoptResult<()> {
        for o in self.observers.iter_mut() {
            o.on_frame(&self.source, system, energy, label)?;
        }
        Ok(())
    }
}

/// Whether `error` is a failure of a calculation (the job is recorded as failed) rather than
/// of the settings or the files (the run ends).
pub fn is_calculation_error(error: &MoptError) -> bool {
    matches!(
        error,
        MoptError::Calculator(_)
            | MoptError::Convergence(_)
            | MoptError::Linalg(_)
            | MoptError::Coordinate(_)
    )
}

fn finish(
    driver: &mut MapperDriver,
    jobs: &dyn MapperJobs,
    observers: &mut [&mut dyn MapperObserver],
    end: ExplorationEnd,
) -> MoptResult<ExplorationEnd> {
    // The evaluations of a job that was stopped count as well (its next run takes them from
    // the journal).
    if let Some(totals) = jobs.evaluations() {
        driver.network_mut().evaluations = totals;
    }
    for o in observers.iter_mut() {
        o.on_finish(driver.network(), &end)?;
    }
    Ok(end)
}

fn stopped(
    driver: &mut MapperDriver,
    jobs: &dyn MapperJobs,
    observers: &mut [&mut dyn MapperObserver],
) -> MoptResult<ExplorationEnd> {
    finish(
        driver,
        jobs,
        observers,
        ExplorationEnd {
            reason: None,
            interrupted: true,
        },
    )
}

fn ended(
    driver: &mut MapperDriver,
    jobs: &dyn MapperJobs,
    observers: &mut [&mut dyn MapperObserver],
    reason: String,
) -> MoptResult<ExplorationEnd> {
    finish(
        driver,
        jobs,
        observers,
        ExplorationEnd {
            reason: Some(reason),
            interrupted: false,
        },
    )
}

/// The channel of the molecules of the input `k` (the node `node`, optimized to `system`)
/// when it has two or more, with the barrierless edge from the input to it; `true` when the
/// run was stopped meanwhile.
fn input_channel(
    driver: &mut MapperDriver,
    jobs: &dyn MapperJobs,
    k: usize,
    node: usize,
    system: &Configuration,
) -> MoptResult<bool> {
    let mut warnings = Vec::new();
    let matcher = *driver.matcher();
    let molecules = {
        let mut sink = Discard;
        let mut ctx = JobContext {
            jobs,
            network: driver.network(),
            matcher: &matcher,
            frames: &mut sink,
            state: None,
        };
        dissociation_of(&mut ctx, system, false, &mut warnings)?
    };
    for message in warnings {
        driver
            .network_mut()
            .warnings
            .push(NetworkWarning { job: None, message });
    }
    if jobs.stop_requested() {
        return Ok(true);
    }
    if let Some(d) = molecules {
        let origin = Origin {
            input: Some(k),
            job: None,
            from_node: None,
        };
        let channel = driver.channel(origin, system, &d, &mut Vec::new())?;
        driver.join_channel(node, channel, None, d.scan.as_ref())?;
        jobs.note(&format!(
            "Input {} is a complex of {} molecules; its dissociation channel is {} at {:.10} \
             Hartree.",
            k + 1,
            d.parts.len(),
            driver.network().node(channel)?.label(),
            d.energy
        ));
    }
    Ok(false)
}

/// Optimize the inputs that were not optimized before; `Some` when the run was stopped.
fn optimize_inputs(
    driver: &mut MapperDriver,
    jobs: &dyn MapperJobs,
    inputs: &[Configuration],
    observers: &mut [&mut dyn MapperObserver],
) -> MoptResult<Option<ExplorationEnd>> {
    if inputs.len() != driver.network().inputs.len() {
        return Err(MoptError::config(format!(
            "the network has {} input structure(s), {} were given",
            driver.network().inputs.len(),
            inputs.len()
        )));
    }
    for (k, system) in inputs.iter().enumerate() {
        if driver.network().inputs[k].node.is_some() {
            continue;
        }
        if jobs.stop_between_jobs() {
            return stopped(driver, jobs, observers).map(Some);
        }
        let source = Source::Input(k);
        for o in observers.iter_mut() {
            o.on_start(&source)?;
        }
        // In a run of several states an input is minimized on its state (the node it becomes
        // lies on that state): on the ground state of its spin the minimum of an excited
        // state would be that of another surface.
        let on = driver.input_state_index(k, system)?;
        let attempt = {
            let mut sink = FanOut {
                observers: &mut *observers,
                source,
            };
            let label = format!("input {}", k + 1);
            match on {
                Some(state) => jobs.minimize_in(system, state, &label, &mut sink),
                None => jobs.optimize(
                    system,
                    &OptimizeRequest {
                        label,
                        stage: Stage::Minimum,
                        terms: Vec::new(),
                    },
                    &mut sink,
                ),
            }
        };
        // Where the run says so (the minima of another network), an input that fails is left
        // out and the others go on.
        let skip = driver.settings().skip_failed_inputs;
        let optimized = match attempt {
            Ok(optimized) => optimized,
            Err(e) if skip && !jobs.stop_requested() => {
                let message =
                    driver.leave_out_input(k, &format!("its minimization failed ({e})"))?;
                jobs.note(&format!("Warning: {message}"));
                continue;
            }
            Err(e) => return Err(e),
        };
        if optimized.interrupted {
            return stopped(driver, jobs, observers).map(Some);
        }
        if !optimized.converged && skip {
            let message = driver.leave_out_input(k, "it did not converge to a minimum")?;
            jobs.note(&format!("Warning: {message}"));
            continue;
        }
        if !optimized.converged {
            return Err(MoptError::conv(format!(
                "input structure {} ({}, frame {}) did not converge to a minimum; the network \
                 starts from minima (raise NSTEP of step4_settings or start from another \
                 structure)",
                k + 1,
                driver.network().inputs[k].file,
                driver.network().inputs[k].frame + 1
            )));
        }
        let wavenumbers = optimized.wavenumbers.clone();
        let structure = optimized.system.clone();
        let node = driver.add_input(k, optimized.system, optimized.energy)?;
        // Its whole structure where the network holds its core (ONIOM).
        if let Some(whole) = jobs.whole_of(&structure)? {
            driver.set_whole(node, whole);
        }
        // The followed electronic states at the minimum of the input.
        driver.set_followed(node, &jobs.take_followed())?;
        if let Some(w) = &wavenumbers {
            driver.set_spectra(node, w, optimized.index_wavenumbers.as_deref())?;
        }
        jobs.note(&format!(
            "Input {} is {} at {:.10} Hartree.",
            k + 1,
            driver.network().node(node)?.label(),
            optimized.energy
        ));
        if input_channel(driver, jobs, k, node, &structure)? {
            return stopped(driver, jobs, observers).map(Some);
        }
    }
    // The entrance channel of the run: the channel of the complexes that its structures
    // became (`entrance_structures`).
    let entrance = driver.settings().entrance_inputs.clone();
    if !entrance.is_empty() && driver.network().entrance.is_none() {
        let net = driver.network();
        let channel = entrance
            .iter()
            .filter_map(|&k| net.inputs.get(k).and_then(|i| i.node))
            .find_map(|node| {
                net.dissociation_edges
                    .iter()
                    .find(|e| e.minimum == node)
                    .map(|e| e.channel)
            });
        match channel {
            Some(channel) => {
                let label = net.node(channel)?.label();
                let minima: std::collections::BTreeSet<usize> = entrance
                    .iter()
                    .filter_map(|&k| net.inputs.get(k).and_then(|i| i.node))
                    .collect();
                jobs.note(&format!(
                    "Entrance channel: {label}; its {} structure(s) became {} minimum(s). The \
                     kinetics of this network starts with the whole population on {label}.",
                    entrance.len(),
                    minima.len()
                ));
                driver.network_mut().entrance = Some(super::entrance::Entrance {
                    channel,
                    structures: entrance.len(),
                    inputs: entrance,
                });
            }
            None => driver.network_mut().warnings.push(NetworkWarning {
                job: None,
                message: "none of the structures of the entrance channel became a complex \
                          with a dissociation channel (channels=off, or every one reacted): \
                          the network has no entrance"
                    .to_string(),
            }),
        }
    }
    if let Some(totals) = jobs.evaluations() {
        driver.network_mut().evaluations = totals;
    }
    for o in observers.iter_mut() {
        o.after_inputs(driver.network())?;
    }
    Ok(None)
}

/// Apply the lines the user left in `steer.txt` of the run directory ([`super::steer`]): each
/// is applied by the driver, told in the log, and written to `steer.log`.
fn steer(driver: &mut MapperDriver, main: &dyn MapperJobs) -> MoptResult<()> {
    for line in main.take_steering()? {
        let outcome = driver.steer(&line);
        main.note(&format!(
            "{}: '{}': {outcome}.",
            super::steer::STEER_FILE,
            line.trim()
        ));
        main.log_steering(driver.network().jobs.len(), &line, &outcome)?;
    }
    Ok(())
}

/// Explore with `driver` from the structures `inputs` (in the order of the inputs of its
/// network; those that were optimized before are skipped) on the surface of `jobs`, one job
/// after the other.
pub fn explore(
    driver: &mut MapperDriver,
    jobs: &dyn MapperJobs,
    inputs: &[Configuration],
    observers: &mut [&mut dyn MapperObserver],
) -> MoptResult<ExplorationEnd> {
    if let Some(end) = optimize_inputs(driver, jobs, inputs, observers)? {
        return Ok(end);
    }
    batches(
        driver,
        jobs,
        &Jobs::Same(jobs),
        &Schedule::default(),
        &[],
        observers,
    )
}

/// The thread pools of jobs that run at the same time, one per worker, which share out the
/// threads of the pool of the loop: what a job computes in parallel (the linear algebra of
/// its optimizations, a calculator that uses threads) stays within its share, instead of
/// every job waking all the threads of the machine.
fn job_pools(workers: usize) -> MoptResult<Vec<rayon::ThreadPool>> {
    let each = (rayon::current_num_threads() / workers).max(1);
    (0..workers)
        .map(|k| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(each)
                .thread_name(move |i| format!("mapper-job-{k}-{i}"))
                .build()
                .map_err(|e| MoptError::config(format!("the thread pool of a job: {e}")))
        })
        .collect()
}

/// Explore with `driver` from the structures `inputs` as [`explore`] does, with the jobs run
/// as `schedule` says and the calculations of every job from `workers`.
pub fn explore_with(
    driver: &mut MapperDriver,
    workers: &dyn MapperWorkers,
    schedule: &Schedule,
    inputs: &[Configuration],
    observers: &mut [&mut dyn MapperObserver],
) -> MoptResult<ExplorationEnd> {
    schedule.check()?;
    let main = workers.run();
    if let Some(end) = optimize_inputs(driver, main, inputs, observers)? {
        return Ok(end);
    }
    let pools = if schedule.workers > 1 {
        job_pools(schedule.workers)?
    } else {
        Vec::new()
    };
    match schedule.mode {
        ScheduleMode::Rolling if schedule.workers > 1 => {
            rolling(driver, workers, schedule, &pools, observers)
        }
        _ => batches(
            driver,
            main,
            &Jobs::Each(workers),
            schedule,
            &pools,
            observers,
        ),
    }
}

/// Where the calculations of a job come from.
enum Jobs<'a> {
    /// The calculations of the run for every job (one job at a time).
    Same(&'a dyn MapperJobs),
    /// Calculations of its own for every job.
    Each(&'a dyn MapperWorkers),
}

/// The calculations of one job.
enum JobJobs<'a> {
    Same(&'a dyn MapperJobs),
    Own(Box<dyn MapperJobs + 'a>),
}

impl JobJobs<'_> {
    fn get(&self) -> &dyn MapperJobs {
        match self {
            Self::Same(j) => *j,
            Self::Own(j) => j.as_ref(),
        }
    }

    /// A line for the log of the job, where it has one of its own.
    fn note(&self, message: &str) {
        if let Self::Own(j) = self {
            j.note(message);
        }
    }

    /// What the job said, where it has a log of its own.
    fn take_lines(&self) -> Vec<String> {
        match self {
            Self::Same(_) => Vec::new(),
            Self::Own(j) => j.take_lines(),
        }
    }
}

impl<'a> Jobs<'a> {
    fn for_job(&self, job: &Job, deadline: Option<Instant>) -> MoptResult<JobJobs<'a>> {
        match self {
            Self::Same(j) => Ok(JobJobs::Same(*j)),
            Self::Each(w) => w.job(job, deadline).map(JobJobs::Own),
        }
    }
}

/// A job that has ended (or was not run because the run was stopped).
struct Ran {
    result: MoptResult<JobOutcome>,
    seconds: f64,
    /// Whether it was stopped at its time limit.
    timed_out: bool,
    /// What it said, for its block in the log of the run.
    lines: Vec<String>,
}

impl Ran {
    fn not_run() -> Self {
        Self {
            result: Ok(JobOutcome::stopped()),
            seconds: 0.0,
            timed_out: false,
            lines: Vec::new(),
        }
    }

    /// Whether it was stopped by the stop of the run (not by its time limit).
    fn stopped(&self) -> bool {
        matches!(&self.result, Ok(o) if o.interrupted) && !self.timed_out
    }
}

/// The node a job starts from, by its label.
fn from_label(network: &Network, job: &Job) -> String {
    network
        .nodes
        .get(job.node)
        .map_or_else(|| format!("node {}", job.node), |n| n.label())
}

/// The first line of the block of a job in the log of the run (and of its own log).
fn header_line(network: &Network, job: &Job) -> String {
    format!(
        "Job {} (from {}, {}; {} job(s) from it before):",
        job.id,
        from_label(network, job),
        job.task.description,
        network.visits(job.node)
    )
}

/// The line on the screen of a job that starts.
fn start_line(network: &Network, job: &Job) -> String {
    format!(
        "Job {} started: from {}, {}.",
        job.id,
        from_label(network, job),
        job.task.description
    )
}

/// The line of a job that has ended, before it is told.
fn end_line(job: &Job, ran: &Ran, limit: Option<f64>) -> String {
    let t = ran.seconds;
    match &ran.result {
        Ok(_) if ran.timed_out => format!(
            "Job {} was stopped after {t:.1} s: it exceeded worker_timeout_s ({} s).",
            job.id,
            limit.unwrap_or(0.0)
        ),
        Ok(o) if o.interrupted => format!("Job {} was stopped after {t:.1} s.", job.id),
        Ok(o) => format!(
            "Job {} done after {t:.1} s: {} reaction(s), {} warning(s).",
            job.id,
            o.reactions.len(),
            o.warnings.len()
        ),
        Err(e) => format!("Job {} failed after {t:.1} s: {e}", job.id),
    }
}

/// Run `job` with the calculations of `jobs`, its structures to `frames`.
fn run_one(
    method: &dyn ExplorationMethod,
    network: &Network,
    matcher: &StructureMatcher,
    job: &Job,
    jobs: &dyn MapperJobs,
    frames: &mut dyn FrameSink,
) -> MoptResult<JobOutcome> {
    let mut context = JobContext {
        jobs,
        network,
        matcher,
        frames,
        state: network.states.get(job.state).copied(),
    };
    if let Ok(node) = network.node(job.node) {
        jobs.start_from(node);
    }
    let mut outcome = method.run(job, &mut context)?;
    outcome.followed = jobs.take_followed();
    if !outcome.interrupted {
        jobs.complete(&mut outcome)?;
    }
    Ok(outcome)
}

/// Run `job` from `source` with its time limit; the lines of its start and end go to its own
/// log.
fn run_timed(
    source: &Jobs<'_>,
    schedule: &Schedule,
    method: &dyn ExplorationMethod,
    network: &Network,
    matcher: &StructureMatcher,
    job: &Job,
    frames: &mut dyn FrameSink,
) -> Ran {
    let start = Instant::now();
    let deadline = schedule.deadline(start);
    let ran = |result: MoptResult<JobOutcome>| Ran {
        // Stopped at its own time limit: the run itself was not stopped (a stop of the run
        // stops the batch, whatever the clock says).
        timed_out: matches!(&result, Ok(o) if o.interrupted)
            && deadline.is_some_and(|d| Instant::now() >= d),
        result,
        seconds: start.elapsed().as_secs_f64(),
        lines: Vec::new(),
    };
    // A job that panics (in its calculations or in making them) fails alone: the other jobs,
    // their threads, and the run go on.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match source.for_job(job, deadline) {
            Err(e) => ran(Err(e)),
            Ok(jobs) => {
                jobs.note(&header_line(network, job));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_one(method, network, matcher, job, jobs.get(), frames)
                }))
                .unwrap_or_else(|panic| Err(panicked(panic.as_ref())));
                let mut done = ran(result);
                jobs.note(&end_line(job, &done, limit_seconds(schedule)));
                done.lines = jobs.take_lines();
                done
            }
        }
    }))
    .unwrap_or_else(|panic| ran(Err(panicked(panic.as_ref()))))
}

/// The block of a job in the log of the run: what it said, the warnings of its outcome, then
/// what was kept of it.
fn block_of(lines: &[String], warnings: &[String], kept: &str) -> Option<String> {
    if lines.is_empty() {
        // Calculations that say everything at once (one job after the other) have no block.
        return None;
    }
    let mut block = lines.join("\n");
    for w in warnings {
        block.push_str("\nWarning: ");
        block.push_str(w);
    }
    block.push('\n');
    block.push_str(kept);
    Some(block)
}

/// The warnings of job `id` for the screen: those it said and those of its outcome.
fn screen_warnings(main: &dyn MapperJobs, id: usize, lines: &[String], warnings: &[String]) {
    let said = lines.iter().filter_map(|l| l.strip_prefix("Warning: "));
    for w in said.chain(warnings.iter().map(String::as_str)) {
        main.progress(&format!("Job {id}: warning: {w}"));
    }
}

/// The failure of a job that panicked with `panic`.
fn panicked(panic: &(dyn std::any::Any + Send)) -> MoptError {
    let what = panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".to_string());
    MoptError::calc(format!("the job panicked: {what}"))
}

fn limit_seconds(schedule: &Schedule) -> Option<f64> {
    schedule.time_limit.map(|t| t.as_secs_f64())
}

/// What a job thread tells the loop.
enum Event {
    /// A structure of the job with this index.
    Frame {
        index: usize,
        system: Box<Configuration>,
        energy: f64,
        label: String,
    },
    /// The job with this index has ended.
    Done { index: usize, ran: Ran },
}

/// The structures of a job in a thread of its own, to the loop.
struct Channel {
    index: usize,
    to: mpsc::Sender<Event>,
}

impl FrameSink for Channel {
    fn push(&mut self, system: &Configuration, energy: f64, label: &str) -> MoptResult<()> {
        self.to
            .send(Event::Frame {
                index: self.index,
                system: Box::new(system.clone()),
                energy,
                label: label.to_string(),
            })
            .map_err(|_| MoptError::config("the exploration loop no longer takes structures"))
    }
}

/// Run the jobs of `batch` in threads, up to `schedule.workers` at the same time (worker `k`
/// in `pools[k]`); the structures go to the observers as they come.
fn run_threads(
    driver: &MapperDriver,
    main: &dyn MapperJobs,
    workers: &dyn MapperWorkers,
    schedule: &Schedule,
    pools: &[rayon::ThreadPool],
    batch: &[Job],
    observers: &mut [&mut dyn MapperObserver],
) -> MoptResult<Vec<Ran>> {
    let network = driver.network();
    let method = driver.method();
    let matcher = driver.settings().matcher;
    let next = AtomicUsize::new(0);
    let mut ran: Vec<Option<Ran>> = (0..batch.len()).map(|_| None).collect();
    let mut failure: Option<MoptError> = None;
    std::thread::scope(|scope| {
        let (to, from) = mpsc::channel::<Event>();
        let mut threads = Vec::new();
        for pool in pools.iter().take(batch.len()) {
            let to = to.clone();
            let next = &next;
            threads.push(scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::SeqCst);
                let Some(job) = batch.get(index) else {
                    break;
                };
                let mut frames = Channel {
                    index,
                    to: to.clone(),
                };
                let done = pool.install(|| {
                    run_timed(
                        &Jobs::Each(workers),
                        schedule,
                        method,
                        network,
                        &matcher,
                        job,
                        &mut frames,
                    )
                });
                if to.send(Event::Done { index, ran: done }).is_err() {
                    break;
                }
            }));
        }
        drop(to);
        for event in from {
            match event {
                Event::Frame {
                    index,
                    system,
                    energy,
                    label,
                } => {
                    if failure.is_some() {
                        continue;
                    }
                    let Some(job) = batch.get(index) else {
                        continue;
                    };
                    for o in observers.iter_mut() {
                        if let Err(e) = o.on_frame(&Source::Job(job), &system, energy, &label) {
                            failure = Some(e);
                            break;
                        }
                    }
                }
                Event::Done { index, ran: done } => {
                    if let Some(job) = batch.get(index) {
                        main.progress(&end_line(job, &done, limit_seconds(schedule)));
                    }
                    if let Some(slot) = ran.get_mut(index) {
                        *slot = Some(done);
                    }
                }
            }
        }
        // Joined here, so that a thread that ended by a panic (the jobs catch theirs) is an
        // error of the run rather than a panic of the loop.
        for thread in threads {
            if thread.join().is_err() && failure.is_none() {
                failure = Some(MoptError::config("a job thread ended by a panic"));
            }
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    Ok(ran
        .into_iter()
        .map(|r| {
            r.unwrap_or_else(|| Ran {
                result: Err(MoptError::config("a job thread ended without its result")),
                seconds: 0.0,
                timed_out: false,
                lines: Vec::new(),
            })
        })
        .collect())
}

/// Run the jobs of `batch` one after the other in this thread.
fn run_inline(
    driver: &MapperDriver,
    main: &dyn MapperJobs,
    source: &Jobs<'_>,
    schedule: &Schedule,
    batch: &[Job],
    observers: &mut [&mut dyn MapperObserver],
) -> MoptResult<Vec<Ran>> {
    let mut out = Vec::with_capacity(batch.len());
    for (k, job) in batch.iter().enumerate() {
        if k > 0 && main.stop_requested() {
            out.push(Ran::not_run());
            continue;
        }
        let source_of = Source::Job(job);
        let done = {
            let mut sink = FanOut {
                observers: &mut *observers,
                source: source_of,
            };
            run_timed(
                source,
                schedule,
                driver.method(),
                driver.network(),
                &driver.settings().matcher,
                job,
                &mut sink,
            )
        };
        if !schedule.one_at_a_time() {
            main.progress(&end_line(job, &done, limit_seconds(schedule)));
        }
        out.push(done);
    }
    Ok(out)
}

/// The labels of the nodes `ids`.
fn labels(network: &Network, ids: &[usize]) -> String {
    ids.iter()
        .filter_map(|&id| network.nodes.get(id).map(|n| n.label()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The transition states of the new edges of `told`.
fn new_ts(network: &Network, told: &Told) -> Vec<usize> {
    told.new_edges
        .iter()
        .filter_map(|(_, e)| network.ts_edges.get(*e).map(|x| x.ts))
        .collect()
}

/// What a job that was told added, in a few words.
fn added(network: &Network, told: &Told, failed: Option<&str>) -> String {
    if let Some(error) = failed {
        return format!("failed ({error})");
    }
    let ts = new_ts(network, told);
    let mut parts = Vec::new();
    if !told.new_minima.is_empty() {
        parts.push(format!("new {}", labels(network, &told.new_minima)));
    }
    if !ts.is_empty() {
        parts.push(format!("new {}", labels(network, &ts)));
    }
    if told.known > 0 {
        parts.push(format!("{} known", told.known));
    }
    if parts.is_empty() {
        "nothing new".to_string()
    } else {
        parts.join(", ")
    }
}

/// The line of a job that was told, one job at a time.
fn told_line(network: &Network, job: &Job, told: &Told, seconds: f64) -> String {
    let ts = new_ts(network, told);
    format!(
        "Job {} done after {seconds:.1} s: {} new minimum(s){}, {} new transition state(s){}{}; \
         the network has {} minima and {} transition states.",
        job.id,
        told.new_minima.len(),
        if told.new_minima.is_empty() {
            String::new()
        } else {
            format!(" ({})", labels(network, &told.new_minima))
        },
        ts.len(),
        if ts.is_empty() {
            String::new()
        } else {
            format!(" ({})", labels(network, &ts))
        },
        if told.known > 0 {
            format!(", {} known", told.known)
        } else {
            String::new()
        },
        network.minima().count(),
        network.ts_edges.len()
    )
}

/// Tell the driver of `job`, which ran as `ran`; `None` when it was stopped by the run and
/// is dropped. Errors that are not of a calculation are returned.
fn tell_one(
    driver: &mut MapperDriver,
    schedule: &Schedule,
    job: &Job,
    ran: Ran,
    trajectory: Option<String>,
) -> MoptResult<Option<(JobOutcome, Told, Option<String>)>> {
    match ran.result {
        Ok(_) if ran.timed_out => {
            let error = format!(
                "it was stopped after it exceeded worker_timeout_s ({} s)",
                limit_seconds(schedule).unwrap_or(0.0)
            );
            let told = driver.fail_job(job.id, &error, trajectory)?;
            Ok(Some((JobOutcome::default(), told, Some(error))))
        }
        Ok(outcome) if outcome.interrupted => {
            driver.drop_job(job.id);
            Ok(None)
        }
        Ok(outcome) => {
            let told = driver.tell_job(job.id, &outcome, trajectory)?;
            Ok(Some((outcome, told, None)))
        }
        Err(e) if is_calculation_error(&e) => {
            let error = e.to_string();
            let told = driver.fail_job(job.id, &error, trajectory)?;
            Ok(Some((JobOutcome::default(), told, Some(error))))
        }
        Err(e) => Err(e.context(format!("job {}", job.id))),
    }
}

/// The trajectory the observers keep of `job`.
fn trajectory_of(observers: &[&mut dyn MapperObserver], job: &Job) -> Option<String> {
    observers
        .iter()
        .find_map(|o| o.trajectory(&Source::Job(job)))
}

/// The jobs in batches of `schedule` (one job at a time by default).
fn batches(
    driver: &mut MapperDriver,
    main: &dyn MapperJobs,
    source: &Jobs<'_>,
    schedule: &Schedule,
    pools: &[rayon::ThreadPool],
    observers: &mut [&mut dyn MapperObserver],
) -> MoptResult<ExplorationEnd> {
    let single = schedule.one_at_a_time();
    loop {
        if main.stop_between_jobs() {
            return stopped(driver, main, observers);
        }
        steer(driver, main)?;
        let batch = match driver.ask_batch(schedule.chosen_together())? {
            MapperBatch::Finished(reason) => return ended(driver, main, observers, reason),
            MapperBatch::Run(jobs) => jobs,
        };
        if let (Some(first), Some(last)) = (batch.first(), batch.last()) {
            if batch.len() > 1 {
                main.note(&format!(
                    "Jobs {} to {}, chosen together, run {} at a time.",
                    first.id,
                    last.id,
                    schedule.workers.min(batch.len())
                ));
            }
        }
        for job in &batch {
            main.progress(&start_line(driver.network(), job));
            if let Some(reason) = driver.reason_of(job.id) {
                main.progress(&format!("  chosen by {reason}"));
                main.record(&format!("Job {}: chosen by {reason}", job.id));
            }
            for o in observers.iter_mut() {
                o.on_start(&Source::Job(job))?;
            }
        }
        let results = match source {
            Jobs::Each(workers) if pools.len() > 1 && batch.len() > 1 => {
                run_threads(driver, main, *workers, schedule, pools, &batch, observers)
            }
            _ => run_inline(driver, main, source, schedule, &batch, observers),
        };
        let results = match results {
            Ok(r) => r,
            Err(e) => {
                driver.abandon();
                return Err(e);
            }
        };
        // A stop within the batch: nothing of it is kept, and a restart runs it again.
        if results.iter().any(Ran::stopped) {
            let flying: Vec<String> = batch
                .iter()
                .zip(&results)
                .filter(|(_, r)| r.stopped())
                .map(|(j, _)| j.id.to_string())
                .collect();
            if batch.len() > 1 {
                main.note(&format!(
                    "The run was stopped with job(s) {} unfinished: jobs {} to {} are not kept \
                     (a restart runs them again, with their evaluations from the journal).",
                    flying.join(", "),
                    batch.first().map_or(0, |j| j.id),
                    batch.last().map_or(0, |j| j.id)
                ));
            }
            driver.abandon();
            return stopped(driver, main, observers);
        }
        // An error that is not of a calculation ends the run before anything of the batch is
        // kept.
        if let Some((job, e)) = batch
            .iter()
            .zip(&results)
            .find_map(|(j, r)| match &r.result {
                Err(e) if !is_calculation_error(e) => Some((j, e.clone())),
                _ => None,
            })
        {
            driver.abandon();
            return Err(e.context(format!("job {}", job.id)));
        }
        let mut kept = Vec::new();
        let mut summary = Vec::new();
        let mut discarded = Vec::new();
        let mut blocks = Vec::new();
        let mut stop: Option<String> = None;
        for (job, mut ran) in batch.iter().zip(results) {
            if stop.is_some() {
                driver.drop_job(job.id);
                discarded.push(job.id.to_string());
                continue;
            }
            let seconds = ran.seconds;
            let lines = std::mem::take(&mut ran.lines);
            if single && (ran.result.is_err() || ran.timed_out) {
                main.progress(&end_line(job, &ran, limit_seconds(schedule)));
            }
            let trajectory = trajectory_of(observers, job);
            let Some((outcome, told, failed)) = tell_one(driver, schedule, job, ran, trajectory)?
            else {
                continue;
            };
            // What the job said, as one block of the log of the run, in the order of the jobs
            // (written once the batch is stored, so that a run that ends before that writes
            // nothing of the batch, which its restart runs again).
            let result = told_line(driver.network(), job, &told, seconds);
            blocks.extend(block_of(&lines, &outcome.warnings, &result));
            screen_warnings(main, job.id, &lines, &outcome.warnings);
            if single {
                main.progress(&result);
            } else {
                summary.push(format!(
                    "job {}: {}",
                    job.id,
                    added(driver.network(), &told, failed.as_deref())
                ));
            }
            for o in observers.iter_mut() {
                o.after_job(driver.network(), job, &outcome, &told)?;
            }
            kept.push(job.id);
            if told.stop.is_some() {
                stop.clone_from(&told.stop);
            }
        }
        if let Some(totals) = main.evaluations() {
            driver.network_mut().evaluations = totals;
        }
        for o in observers.iter_mut() {
            o.after_batch(driver.network(), &kept)?;
        }
        for block in &blocks {
            main.record(block);
        }
        if !single {
            let net = driver.network();
            main.note(&format!(
                "Jobs {} kept ({}); the network has {} minima and {} transition states.",
                kept.iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
                summary.join("; "),
                net.minima().count(),
                net.ts_edges.len()
            ));
        }
        if !discarded.is_empty() {
            main.note(&format!(
                "Job(s) {} of the batch are not kept: the exploration ended with the job before \
                 them.",
                discarded.join(", ")
            ));
        }
        if let Some(reason) = stop {
            return ended(driver, main, observers, reason);
        }
    }
}

/// The blocks of jobs that end out of order, written to the log of the run in the order the
/// jobs started: a block waits for the blocks of the jobs before it.
#[derive(Default)]
struct InOrder {
    /// The jobs that started and whose blocks are not written yet, in order.
    order: std::collections::VecDeque<usize>,
    /// The blocks of jobs that ended (`None`: a job that is not kept, with nothing to write).
    ended: BTreeMap<usize, Option<String>>,
}

impl InOrder {
    fn started(&mut self, id: usize) {
        self.order.push_back(id);
    }

    fn ended(&mut self, id: usize, block: Option<String>, main: &dyn MapperJobs) {
        self.ended.insert(id, block);
        while let Some(&first) = self.order.front() {
            let Some(block) = self.ended.remove(&first) else {
                break;
            };
            self.order.pop_front();
            if let Some(block) = block {
                main.record(&block);
            }
        }
    }
}

/// The jobs as the workers get free: every job is told when it ends.
fn rolling(
    driver: &mut MapperDriver,
    workers: &dyn MapperWorkers,
    schedule: &Schedule,
    pools: &[rayon::ThreadPool],
    observers: &mut [&mut dyn MapperObserver],
) -> MoptResult<ExplorationEnd> {
    let main = workers.run();
    let method = driver.shared_method();
    let matcher = driver.settings().matcher;
    // The jobs in flight with the pool each runs in, and the pools that are free.
    let mut flying: BTreeMap<usize, (Job, usize)> = BTreeMap::new();
    let mut free: Vec<usize> = (0..pools.len()).rev().collect();
    let mut failure: Option<MoptError> = None;
    let mut reason: Option<String> = None;
    let mut interrupted = false;
    let mut snapshot: Option<Arc<Network>> = None;
    let mut blocks = InOrder::default();
    std::thread::scope(|scope| -> MoptResult<()> {
        let (to, from) = mpsc::channel::<Event>();
        loop {
            // Start jobs while workers are free and the exploration goes on.
            while failure.is_none() && reason.is_none() && !interrupted && !free.is_empty() {
                if main.stop_between_jobs() {
                    interrupted = true;
                    break;
                }
                steer(driver, main)?;
                let job = match driver.ask_next()? {
                    NextJob::Run(job) => *job,
                    NextJob::Wait => break,
                    NextJob::Finished(r) => {
                        // The jobs in flight end and are kept before the loop ends.
                        reason = Some(r);
                        break;
                    }
                };
                main.progress(&start_line(driver.network(), &job));
                if let Some(reason) = driver.reason_of(job.id) {
                    main.progress(&format!("  chosen by {reason}"));
                    main.record(&format!("Job {}: chosen by {reason}", job.id));
                }
                for o in observers.iter_mut() {
                    o.on_start(&Source::Job(&job))?;
                }
                blocks.started(job.id);
                let network = match &snapshot {
                    Some(n) => Arc::clone(n),
                    None => {
                        let n = Arc::new(driver.network().clone());
                        snapshot = Some(Arc::clone(&n));
                        n
                    }
                };
                let Some(slot) = free.pop() else {
                    break;
                };
                let Some(pool) = pools.get(slot) else {
                    return Err(MoptError::config("a worker without its thread pool"));
                };
                let (to, method, own) = (to.clone(), Arc::clone(&method), job.clone());
                scope.spawn(move || {
                    let index = own.id;
                    let mut frames = Channel {
                        index,
                        to: to.clone(),
                    };
                    let done = pool.install(|| {
                        run_timed(
                            &Jobs::Each(workers),
                            schedule,
                            method.as_ref(),
                            &network,
                            &matcher,
                            &own,
                            &mut frames,
                        )
                    });
                    let _ = to.send(Event::Done { index, ran: done });
                });
                flying.insert(job.id, (job, slot));
            }
            if flying.is_empty() {
                return Ok(());
            }
            let Ok(event) = from.recv() else {
                return Err(MoptError::config(
                    "the job threads ended without their results",
                ));
            };
            match event {
                Event::Frame {
                    index,
                    system,
                    energy,
                    label,
                } => {
                    if failure.is_some() {
                        continue;
                    }
                    let Some((job, _)) = flying.get(&index) else {
                        continue;
                    };
                    for o in observers.iter_mut() {
                        if let Err(e) = o.on_frame(&Source::Job(job), &system, energy, &label) {
                            failure = Some(e);
                            break;
                        }
                    }
                }
                Event::Done { index, mut ran } => {
                    let Some((job, slot)) = flying.remove(&index) else {
                        continue;
                    };
                    free.push(slot);
                    if failure.is_some() {
                        driver.drop_job(job.id);
                        blocks.ended(job.id, None, main);
                        continue;
                    }
                    if ran.stopped() {
                        main.progress(&end_line(&job, &ran, limit_seconds(schedule)));
                        driver.drop_job(job.id);
                        blocks.ended(job.id, None, main);
                        interrupted = true;
                        continue;
                    }
                    if let Err(e) = &ran.result {
                        if !is_calculation_error(e) {
                            failure = Some(e.clone().context(format!("job {}", job.id)));
                            driver.drop_job(job.id);
                            blocks.ended(job.id, None, main);
                            continue;
                        }
                    }
                    let seconds = ran.seconds;
                    let lines = std::mem::take(&mut ran.lines);
                    if ran.result.is_err() || ran.timed_out {
                        main.progress(&end_line(&job, &ran, limit_seconds(schedule)));
                    }
                    let trajectory = trajectory_of(observers, &job);
                    let Some((outcome, told, _)) =
                        tell_one(driver, schedule, &job, ran, trajectory)?
                    else {
                        blocks.ended(job.id, None, main);
                        continue;
                    };
                    snapshot = None;
                    let result = told_line(driver.network(), &job, &told, seconds);
                    screen_warnings(main, job.id, &lines, &outcome.warnings);
                    main.progress(&result);
                    for o in observers.iter_mut() {
                        o.after_job(driver.network(), &job, &outcome, &told)?;
                    }
                    if let Some(totals) = main.evaluations() {
                        driver.network_mut().evaluations = totals;
                    }
                    for o in observers.iter_mut() {
                        o.after_batch(driver.network(), &[job.id])?;
                    }
                    blocks.ended(job.id, block_of(&lines, &outcome.warnings, &result), main);
                    if reason.is_none() {
                        reason.clone_from(&told.stop);
                    }
                }
            }
        }
    })?;
    if let Some(e) = failure {
        return Err(e);
    }
    match reason {
        Some(r) if !interrupted => ended(driver, main, observers, r),
        _ => stopped(driver, main, observers),
    }
}
