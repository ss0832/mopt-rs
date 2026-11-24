//! The calculations of `run_mapper` ([`MapperJobs`]): the optimizations of the steps and the
//! IRC on the one surface of the run, with the calculator of the registry (through the
//! evaluation journal of the run directory) and the stop signal of the run.
//!
//! The jobs ([`CommandWorkers`], [`MapperWorkers`]) each have calculations of their own: a
//! stop signal that also stops the job at its time limit (`worker_timeout_s`), a log of its own
//! (`jobs/job_000003.log`), and the calculator of the run, which jobs that run at the same time
//! share, or, where the calculator evaluates in order (it follows excited states, or relaxes an
//! environment from structure to structure), a calculator of the job's own made by the
//! registry of the run, so that what one job evaluated never changes what another gets.
//!
//! On an objective of plain variables (`-os mb2d`, ...) the same calculations run on the
//! objective: the optimizer of plain variables for the steps, the IRC with unit masses, the
//! two-point searches on the variables. The objective is shared by every job (objectives are
//! cheap and give the same values every time, so a restart computes them again instead of
//! reading them from the journal).
//!
//! A `STOP` file in the run directory stops a run at once (it is removed, so that
//! `--restart RUN_DIR` goes on); MultiOptPy's `stop.txt` stops it between two jobs, as in
//! MultiOptPy (it is left in place): the jobs that run end first.
//!
//! MultiOptPy: the calls of `OptimizationJob` and AutoTS in `Wrapper/mapper.py`
//!  (`_run_initial_optimization`, `_autots_worker`, the worker processes of `n_parallel` and
//!  their `worker_timeout_s`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::irc::irc_states;
use super::mapper_settings::StepConfigs;
use super::optmain::state_surface;
use crate::core_infra::log::{Level, Logger, MemorySink};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::irc::types::Branch;
use crate::irc::{BranchEnd, Irc, IrcEvent, IrcSettings, IrcSpace, StartKind};
use crate::opt::state_machine::{GeometryOptimizer, IterationReport, Termination};
use crate::opt::vector::VectorOptimizer;
use crate::pes::calculator::Calculator;
use crate::pes::counting::EvaluationCounts;
use crate::pes::objective::{Objective, ObjectiveRegistry};
use crate::pes::registry::CalculatorRegistry;
use crate::workflow::irc::{run_irc, IrcObserver, ObjectiveSurface};
use crate::workflow::journal::RunJournal;
use crate::workflow::mapper::configuration::VariablePoint;
use crate::workflow::mapper::network::{EvaluationTotals, IrcSide, PossibleBifurcation};
use crate::workflow::mapper::observers::job_log_file;
use crate::workflow::mapper::{
    Configuration, DynamicsOutcome, DynamicsRequest, FollowedRoots, FrameSink, IrcBranch,
    IrcOutcome, Job, MapperJobs, MapperWorkers, OptimizeRequest, Optimized, PairSearchOutcome,
    PairSearchRequest, PathPoint, Stage, TsConformerSearch,
};
use crate::workflow::observers::STOP_FILE;
use crate::workflow::{run_optimization, run_vector_optimization_until, Observer};

mod cell;

/// MultiOptPy's file that stops the mapper between two jobs.
pub const MULTIOPTPY_STOP_FILE: &str = "stop.txt";

/// The stop signal of a run, or of one of its jobs (which also stops at its time limit).
#[derive(Debug, Clone)]
pub struct MapperStop {
    dir: PathBuf,
    /// A `STOP` file was seen: everything stops.
    flag: Arc<AtomicBool>,
    /// MultiOptPy's `stop.txt` was seen: no job starts any more.
    between: Arc<AtomicBool>,
    log: Logger,
    /// The time limit of a job.
    deadline: Option<Instant>,
    /// The job has passed its time limit.
    expired: Arc<AtomicBool>,
}

impl MapperStop {
    /// Watch the run directory `dir`.
    pub fn new(dir: &Path, log: Logger) -> Self {
        Self {
            dir: dir.to_path_buf(),
            flag: Arc::new(AtomicBool::new(false)),
            between: Arc::new(AtomicBool::new(false)),
            log,
            deadline: None,
            expired: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The stop signal of a job: the stop of the run, and the time limit `deadline`. It says
    /// nothing when it finds the `STOP` file (the thread of the job writes nowhere but to the
    /// log of the job; the run says that it was stopped).
    #[must_use]
    pub fn for_job(&self, deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            expired: Arc::new(AtomicBool::new(false)),
            log: Logger::silent(),
            ..self.clone()
        }
    }

    /// Whether the run (or this job) is to stop now: a `STOP` file (removed, and reported
    /// once), or the time limit of the job.
    pub fn check(&self) -> bool {
        if self.flag.load(Ordering::SeqCst) {
            return true;
        }
        let stop = self.dir.join(STOP_FILE);
        if stop.exists() {
            let _ = std::fs::remove_file(&stop);
            if !self.flag.swap(true, Ordering::SeqCst) {
                self.log.info(&format!(
                    "{} found: the run is interrupted; --restart {} continues it.",
                    stop.display(),
                    self.dir.display()
                ));
            }
            return true;
        }
        if self.deadline.is_some_and(|d| Instant::now() >= d) {
            self.expired.store(true, Ordering::SeqCst);
        }
        self.expired.load(Ordering::SeqCst)
    }

    /// Whether the run is to stop before the next job (also MultiOptPy's `stop.txt`, which
    /// lets the jobs that run end).
    pub fn between_jobs(&self) -> bool {
        if self.check() {
            return true;
        }
        if self.between.load(Ordering::SeqCst) {
            return true;
        }
        let file = self.dir.join(MULTIOPTPY_STOP_FILE);
        if file.exists() {
            if !self.between.swap(true, Ordering::SeqCst) {
                self.log.info(&format!(
                    "{} found: the run stops before the next job (remove it before --restart).",
                    file.display()
                ));
            }
            return true;
        }
        false
    }

    /// Whether a stop was seen (of the run, or the time limit of this job).
    pub fn stopped(&self) -> bool {
        self.flag.load(Ordering::SeqCst) || self.expired.load(Ordering::SeqCst)
    }

    /// The run directory it watches.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Observer for MapperStop {
    fn on_iteration(&mut self, _report: &IterationReport) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self, _current: &MolecularSystem) -> bool {
        self.check()
    }
}

/// Every evaluated structure of an optimization, to the path and the trajectory of the job,
/// and what the calculator warned of, to the log of the job.
struct Recorder<'a> {
    label: &'a str,
    path: Vec<PathPoint>,
    /// The energies of `State1` and `State2` at every point, where the surface has them.
    states: Vec<Option<[f64; 2]>>,
    frames: &'a mut dyn FrameSink,
    log: &'a Logger,
    /// Ends a crossing search where its two states run parallel.
    watch: Option<ParallelWatch>,
}

/// Counts the iterations in a row of a crossing search with a closed gap and a small
/// difference of the two gradients (`crossing_parallel_stop`).
struct ParallelWatch {
    rule: super::mapper_settings::ParallelStop,
    /// The gap that counts as closed (Hartree).
    gap: f64,
    /// Such iterations in a row so far.
    run: usize,
}

impl ParallelWatch {
    /// Whether the states have run parallel for the iterations of the rule.
    fn ended(&self) -> bool {
        self.run >= self.rule.steps
    }
}

impl Observer for Recorder<'_> {
    fn on_warning(&mut self, message: &str) -> MoptResult<()> {
        self.log.warn(&format!("{}: {message}", self.label));
        Ok(())
    }

    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        let point = Configuration::Structure(r.system.clone());
        // The first entries of these names are the energies of the states themselves.
        let state = |name: &str| {
            r.components
                .iter()
                .find(|(label, _)| label == name)
                .map(|(_, e)| e.value())
        };
        let states = match (state("State1"), state("State2")) {
            (Some(a), Some(b)) => Some([a, b]),
            _ => None,
        };
        // On a surface of two states the frame of the trajectory has the energies of both
        // (`states=E1,E2`), for the reports that draw them along the search.
        let both = states.map_or_else(String::new, |[a, b]| format!(" states={a:.10},{b:.10}"));
        self.frames.push(
            &point,
            r.energy.value(),
            &format!("{} iteration {}{both}", self.label, r.iteration),
        )?;
        self.path.push(PathPoint {
            system: point,
            energy: r.energy.value(),
            biased_energy: r.biased_energy.value(),
        });
        self.states.push(states);
        if let Some(watch) = &mut self.watch {
            match r.crossing {
                Some(c)
                    if c.gap.abs() <= watch.gap && c.gradient_difference < watch.rule.gradient =>
                {
                    watch.run += 1;
                }
                _ => watch.run = 0,
            }
        }
        Ok(())
    }

    fn should_stop(&mut self, _current: &MolecularSystem) -> bool {
        self.watch.as_ref().is_some_and(ParallelWatch::ended)
    }
}

fn end_name(end: BranchEnd) -> &'static str {
    match end {
        BranchEnd::Converged => "converged",
        BranchEnd::Oscillation => "oscillation",
        BranchEnd::MaxSteps => "max_steps",
        BranchEnd::EnergyRise => "energy_rise",
        BranchEnd::ModesRemoved => "modes_removed",
        BranchEnd::NothingToRemove => "nothing_to_remove",
    }
}

/// The arc length at which the curvature across the path changes sign before the last point
/// of `transverse` (arc lengths and wavenumbers, the last one negative): linear in the
/// curvature, which goes as the square of the wavenumber with its sign, between the last point
/// whose curvature is not negative and the point after it. `None` where no point before the
/// last has a curvature that is not negative.
fn inflection_of(transverse: &[[f64; 2]]) -> Option<f64> {
    let curvature = |w: f64| w * w.abs();
    let i = transverse.iter().rposition(|p| p[1] >= 0.0)?;
    let (a, b) = (transverse.get(i)?, transverse.get(i + 1)?);
    let (ca, cb) = (curvature(a[1]), curvature(b[1]));
    if cb >= 0.0 || ca - cb <= 0.0 {
        return None;
    }
    Some(a[0] + (b[0] - a[0]) * ca / (ca - cb))
}

/// The points of the IRC as they come, and whether its start is a transition state.
struct IrcCollector<'a> {
    start: &'a Configuration,
    frames: &'a mut dyn FrameSink,
    sides: Vec<(Branch, IrcBranch)>,
    kind: Option<StartKind>,
    imaginary: usize,
    wavenumber: Option<f64>,
    /// Whether the vibrations leave out the rigid motions.
    rigid: bool,
    /// The vibrations of a transition state at the start.
    wavenumbers: Option<Vec<f64>>,
    /// How negative a curvature across the path must be for a possible bifurcation (cm^-1).
    bifurcation_threshold_cm1: f64,
    /// The arc length of the last point of every side with an exact Hessian and no negative
    /// curvature across the path.
    clear: Vec<(Branch, f64)>,
    refused: bool,
    /// A start that is no transition state is followed as a meta-IRC instead of refused.
    meta: bool,
    /// The largest component and the RMS of the gradient at the start.
    start_gradient: Option<(f64, f64)>,
    /// The variables of a variable cell, whose points are structures with their cells.
    cell: Option<&'a crate::opt::cell::VariableCell>,
}

impl IrcCollector<'_> {
    /// The point of the path at `x`.
    fn point(&self, x: &faer::Col<f64>) -> MoptResult<Configuration> {
        match self.cell {
            Some(map) => Ok(Configuration::Structure(map.system(x)?)),
            None => self.start.with_flat(x),
        }
    }

    fn side(&mut self, branch: Branch) -> &mut IrcBranch {
        if let Some(k) = self.sides.iter().position(|(b, _)| *b == branch) {
            return &mut self.sides[k].1;
        }
        self.sides.push((
            branch,
            IrcBranch {
                side: IrcSide {
                    name: branch.label().to_string(),
                    ..IrcSide::default()
                },
                frames: Vec::new(),
                last: self.start.clone(),
                settle: Vec::new(),
                possible_bifurcation: None,
            },
        ));
        let last = self.sides.len() - 1;
        &mut self.sides[last].1
    }
}

impl IrcObserver for IrcCollector<'_> {
    fn on_event(&mut self, event: &IrcEvent) -> MoptResult<()> {
        match event {
            IrcEvent::Start(report) => {
                self.kind = Some(report.kind);
                self.imaginary = report.imaginary;
                self.wavenumber = report.transition_wavenumber;
                self.start_gradient = Some((report.max_gradient, report.rms_gradient));
                self.refused = !self.meta && report.kind != StartKind::TransitionState;
                if let (StartKind::TransitionState, Some(sys)) =
                    (report.kind, self.start.structure())
                {
                    // On a variable cell: the atoms in the cell of the transition state.
                    let n = sys.dof();
                    let atoms = match self.cell {
                        Some(_) => faer::Mat::from_fn(n, n, |i, j| report.hessian[(i, j)]),
                        None => report.hessian.clone(),
                    };
                    self.wavenumbers =
                        Some(super::mapper_minima::vibrations(sys, &atoms, self.rigid)?);
                } else if report.kind == StartKind::TransitionState {
                    // Plain variables: the wavenumbers of unit masses.
                    self.wavenumbers =
                        Some(super::mapper_minima::variable_wavenumbers(&report.hessian)?);
                }
            }
            IrcEvent::Step { branch, record } => {
                if let Some(c) = record.curvature_across {
                    let wavenumber = crate::irc::eigenvalue_to_wavenumber(c);
                    let after = self
                        .clear
                        .iter()
                        .find(|(b, _)| b == branch)
                        .map(|(_, a)| *a);
                    let threshold = self.bifurcation_threshold_cm1;
                    let side = self.side(*branch);
                    // The curvature across the path at every exact Hessian of the side.
                    side.side.transverse.push([record.arc_length, wavenumber]);
                    if side.possible_bifurcation.is_none() {
                        if wavenumber < -threshold {
                            side.possible_bifurcation = Some(PossibleBifurcation {
                                side: branch.label().to_string(),
                                after,
                                at: record.arc_length,
                                wavenumber,
                                confirmed_ends: Vec::new(),
                                inflection: inflection_of(&side.side.transverse),
                            });
                        } else {
                            self.clear.retain(|(b, _)| b != branch);
                            self.clear.push((*branch, record.arc_length));
                        }
                    }
                }
                let system = self.point(&record.x)?;
                self.frames.push(
                    &system,
                    record.energy,
                    &format!("IRC {} point {}", branch.label(), record.step),
                )?;
                let side = self.side(*branch);
                side.side.arc_lengths.push(record.arc_length);
                side.side.energies.push(record.energy);
                side.frames.push(system.clone());
                side.last = system;
            }
            IrcEvent::BranchFinished(summary) => {
                let last = self.point(&summary.x)?;
                let side = self.side(summary.branch);
                side.side.end = end_name(summary.end).to_string();
                side.last = last;
            }
            IrcEvent::FirstStep(_) => {}
        }
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        self.refused
    }
}

/// The calculator of a job.
enum JobCalc<'a> {
    /// The (journaled) calculator of the run: the calculations of the run itself (its
    /// inputs), recorded under [`JOURNAL_LABEL`].
    Shared(&'a dyn Calculator),
    /// The calculator of the run for a job, recorded in the journal of the run under the
    /// label of the job.
    Job(&'a dyn Calculator, &'a RunJournal, String),
    /// A calculator of the job's own, recorded in the journal of the run under the label of
    /// the job.
    Own(Box<dyn Calculator>, &'a RunJournal, String),
    /// The objective of plain variables of the run.
    Objective(&'a dyn Objective),
}

/// The label of the calculator of the run in its journal.
pub const JOURNAL_LABEL: &str = "main";

/// The label of the evaluations of job `id` in the journal of the run. Every job has its own:
/// a record answers only the job that made it. With one label for all, a restarted run could
/// answer an evaluation of a later job from the record of another job whose structure lay
/// within the tolerance of the journal without being the same (two searches that converged
/// to one stationary point by Newton steps end within 1e-10 Bohr of each other); the
/// energy and gradient of the other structure then ended a search a step earlier or later
/// than in a run that was not stopped, and on a flat surface the next search went elsewhere
/// (found on the argon cluster of the restart tests, 2026-11-19).
pub fn job_label(id: usize) -> String {
    format!("{JOURNAL_LABEL}:job{id}")
}

/// Where the calculations of the run write: the screen and the log of the run, the screen
/// alone, the log of the run alone.
#[derive(Debug, Clone, Default)]
pub struct RunOutput {
    /// The screen and `mopt.log` (the lines of the run).
    pub both: Logger,
    /// The screen alone (a job starts or ends).
    pub screen: Logger,
    /// `mopt.log` alone (the blocks of the jobs).
    pub file: Option<BlockLog>,
}

/// Appends whole blocks of lines to a log file, each with one write (the lines of a block are
/// never split by other lines, and a run killed while it writes one leaves at most that block
/// torn).
#[derive(Debug, Clone)]
pub struct BlockLog {
    path: PathBuf,
}

impl BlockLog {
    /// Blocks appended to `path`.
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    /// Append `block` and a line end (a failing disk does not stop the run, as for the lines
    /// of a log).
    pub fn append(&self, block: &str) {
        use std::io::Write as _;
        let mut text = String::with_capacity(block.len() + 1);
        text.push_str(block);
        text.push('\n');
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = file.write_all(text.as_bytes());
        }
    }
}

/// The calculations of a run (or of one of its jobs).
pub struct CommandJobs<'a> {
    /// The steps of these calculations (on the state of the job).
    steps: &'a StepConfigs,
    /// The steps of the run, with those on each of its states.
    run_steps: &'a StepConfigs,
    /// The index of the state of the job among the states of the run (0 for one state).
    state: usize,
    calc: JobCalc<'a>,
    stop: MapperStop,
    /// The log of the run, or of the job.
    log: Logger,
    /// The screen alone and the log of the run alone (none for a job).
    screen: Logger,
    file: Option<BlockLog>,
    /// What a job said, for its block in the log of the run.
    said: Option<MemorySink>,
    registry: &'a CalculatorRegistry,
    /// The registry of the objective, when the surface is one (its evaluations are counted
    /// there).
    objectives: Option<&'a ObjectiveRegistry>,
    since: EvaluationCounts,
    before: EvaluationTotals,
    /// Whether these are the calculations of the run (which count the evaluations).
    of_run: bool,
    /// The followed electronic states at the ends of the optimizations so far
    /// ([`MapperJobs::take_followed`]).
    followed: std::sync::Mutex<Vec<(Vec<f64>, FollowedRoots)>>,
    /// The whole system where the structures are the cores of an ONIOM surface
    /// ([`super::mapper_oniom`]).
    whole: Option<std::sync::Arc<super::mapper_oniom::WholeSystem>>,
}

/// A line of a log as its file has it.
fn as_written(level: Level, line: &str) -> String {
    match level {
        Level::Error => format!("Error: {line}"),
        Level::Warn => format!("Warning: {line}"),
        Level::Info | Level::Debug => line.to_string(),
    }
}

impl<'a> CommandJobs<'a> {
    /// The calculations with the steps `steps` and the (journaled) calculator `calc`, writing
    /// to `output`; the evaluations of `registry` from now on are added to `before`, those of
    /// the earlier sessions of the run.
    pub fn new(
        steps: &'a StepConfigs,
        calc: &'a dyn Calculator,
        stop: MapperStop,
        output: RunOutput,
        registry: &'a CalculatorRegistry,
        before: EvaluationTotals,
    ) -> Self {
        Self {
            steps,
            run_steps: steps,
            state: 0,
            calc: JobCalc::Shared(calc),
            stop,
            log: output.both,
            screen: output.screen,
            file: output.file,
            said: None,
            since: registry.counts(),
            registry,
            objectives: None,
            before,
            of_run: true,
            followed: std::sync::Mutex::new(Vec::new()),
            whole: None,
        }
    }

    /// These calculations on the cores of the ONIOM surface of the whole system `whole`.
    #[must_use]
    pub fn with_whole(mut self, whole: std::sync::Arc<super::mapper_oniom::WholeSystem>) -> Self {
        self.whole = Some(whole);
        self
    }

    /// The calculations of a run on the objective of plain variables `objective` (made by
    /// `objectives`, which counts its evaluations).
    pub fn of_objective(
        steps: &'a StepConfigs,
        objective: &'a dyn Objective,
        objectives: &'a ObjectiveRegistry,
        stop: MapperStop,
        output: RunOutput,
        registry: &'a CalculatorRegistry,
        before: EvaluationTotals,
    ) -> Self {
        Self {
            steps,
            run_steps: steps,
            state: 0,
            calc: JobCalc::Objective(objective),
            stop,
            log: output.both,
            screen: output.screen,
            file: output.file,
            said: None,
            since: objectives.counts(),
            registry,
            objectives: Some(objectives),
            before,
            of_run: true,
            followed: std::sync::Mutex::new(Vec::new()),
            whole: None,
        }
    }

    /// Run `f` with the calculator of these calculations (an error on an objective of plain
    /// variables).
    fn with_calc<R>(&self, f: impl FnOnce(&dyn Calculator) -> MoptResult<R>) -> MoptResult<R> {
        match &self.calc {
            JobCalc::Shared(c) => f(*c),
            JobCalc::Job(c, journal, label) => f(&journal.wrap(label, *c)),
            JobCalc::Own(c, journal, label) => f(&journal.wrap(label, c.as_ref())),
            JobCalc::Objective(o) => Err(MoptError::config(format!(
                "the surface of the run is the objective '{}' of plain variables, not a \
                 calculator of structures",
                o.name()
            ))),
        }
    }

    /// The objective of these calculations, or the error that the surface is a calculator.
    fn objective(&self) -> MoptResult<&dyn Objective> {
        match &self.calc {
            JobCalc::Objective(o) => Ok(*o),
            _ => Err(MoptError::config(
                "a point of plain variables on a calculator of structures",
            )),
        }
    }

    fn config(&self, stage: Stage) -> &super::optmain::OptmainConfig {
        self.steps.config(stage)
    }

    /// The step settings of the run.
    pub(super) fn steps(&self) -> &StepConfigs {
        self.steps
    }

    /// The trajectory selection in the coordinate numbering of this job.
    pub(super) fn trajectory_atoms(&self) -> Option<&[crate::domain::AtomId]> {
        self.whole.as_ref()?.trajectory_atoms.as_deref()
    }

    /// The stop signal of these calculations.
    pub(super) fn stop_signal(&self) -> MapperStop {
        self.stop.clone()
    }

    /// The log of the run, or of the job.
    pub(super) fn job_log(&self) -> &Logger {
        &self.log
    }

    /// The settings of the dissociation channels (`None` on an ONIOM surface: a core sits in
    /// its environment, no fragment of one is apart from the rest).
    pub(super) fn channel_settings(&self) -> Option<crate::workflow::mapper::ChannelSettings> {
        if self.whole.is_some() {
            return None;
        }
        Some(self.steps.channels)
    }

    /// What `sys` came apart into in its periodic cell, where it did and channels are on: such
    /// a structure is a dissociation channel and is kept with whatever imaginary modes it has
    /// (`crate::workflow::mapper::channels::cell_contacts`).
    pub(super) fn apart_in_cell(&self, sys: &MolecularSystem) -> Option<String> {
        let c = self.channel_settings().filter(|c| c.enabled)?;
        let contacts =
            crate::workflow::mapper::channels::cell_contacts(sys, c.bond_scale, c.margin)?;
        contacts.apart().then(|| contacts.describe())
    }

    fn irc_settings(steps: &StepConfigs) -> IrcSettings {
        let c = &steps.saddle.settings.criteria;
        let mut irc = steps
            .irc
            .clone()
            .accepting_converged(c.max_force.value(), c.rms_force.value());
        if irc.exact_hessian_interval.is_none() && steps.irc_hessians {
            irc.exact_hessian_interval = Some(steps.irc_hessian_interval);
        }
        irc
    }
}

/// The method of the optimizations of saddle points that follow given modes (the descents from a
/// saddle point of higher order): RS-I-RFO with mode following and the Bofill update.
const FOLLOWING_METHOD: &str = "mf_rsirfo_block_bofill";

impl MapperJobs for CommandJobs<'_> {
    fn optimize(
        &self,
        start: &Configuration,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        if self.stop.check() {
            return Ok(Optimized {
                index_wavenumbers: None,
                system: start.clone(),
                energy: f64::NAN,
                converged: false,
                interrupted: true,
                path: Vec::new(),
                wavenumbers: None,
            });
        }
        match start {
            Configuration::Structure(sys) => {
                self.with_calc(|calc| self.optimize_with(calc, sys, request, frames))
            }
            Configuration::Variables(point) => {
                self.optimize_variables(self.objective()?, point, request, frames)
            }
        }
    }

    fn irc(&self, ts: &Configuration, frames: &mut dyn FrameSink) -> MoptResult<IrcOutcome> {
        match ts {
            Configuration::Structure(sys) => {
                self.with_calc(|calc| self.irc_with(calc, sys, frames))
            }
            Configuration::Variables(_) => self.irc_variables(self.objective()?, ts, frames),
        }
    }

    fn pair_search(
        &self,
        request: &PairSearchRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<PairSearchOutcome> {
        if let JobCalc::Objective(o) = &self.calc {
            return self.pair_search_variables(*o, request, frames);
        }
        if let Some(cell) = cell::variable_cell(self.steps) {
            return self.with_calc(|calc| self.pair_search_cell(calc, request, frames, cell));
        }
        self.with_calc(|calc| self.pair_search_with(calc, request, frames))
    }

    fn dynamics(
        &self,
        start: &Configuration,
        request: &DynamicsRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<DynamicsOutcome> {
        let sys = start.require_structure("the molecular dynamics of the mapper")?;
        self.with_calc(|calc| self.dynamics_with(calc, sys, request, frames))
    }

    fn negative_curvatures(&self, point: &Configuration) -> MoptResult<Option<usize>> {
        let system = match point {
            Configuration::Structure(sys) => sys,
            Configuration::Variables(p) => {
                let o = self.objective()?;
                let v = o.evaluate(&p.values, true)?;
                let hessian = v.hessian.ok_or_else(|| {
                    MoptError::calc(format!("objective '{}' gave no Hessian", o.name()))
                })?;
                let mode_kill = super::mapper_minima::mapper_mode_kill(
                    self.config(Stage::Minimum).settings.mode_kill,
                );
                return Ok(Some(
                    match crate::opt::mode_kill::examine(
                        &hessian,
                        &v.gradient,
                        0,
                        &mode_kill,
                        None,
                        None,
                    )? {
                        crate::opt::mode_kill::ModeKillVerdict::Accept { negative }
                        | crate::opt::mode_kill::ModeKillVerdict::Displace { negative, .. } => {
                            negative
                        }
                    },
                ));
            }
        };
        if let Some(cell) = cell::variable_cell(self.steps) {
            return self
                .with_calc(|calc| Ok(Some(self.cell_modes(calc, system, false, cell)?.negative)));
        }
        self.with_calc(|calc| {
            // On the surface of the minimizations, with their projection and the threshold of
            // their mode kill; the Hessian is that a minimization from here asks for first, so
            // the journal answers it again where the end is a minimum after all.
            let o = self.config(Stage::Minimum);
            let mut settings = o.settings.clone();
            settings.rigid_body_invariant = calc.is_rigid_body_invariant();
            let mode_kill = super::mapper_minima::mapper_mode_kill(settings.mode_kill);
            let mut opt = GeometryOptimizer::new(settings, system.clone())?;
            if let Some(surface) = state_surface(&o.model_function, o.excited_state)? {
                opt.set_surface(surface)?;
            }
            // A count that cannot be judged (no converged Hessian) is unknown.
            Ok(
                match super::mapper_minima::check_minimum(&mut opt, calc, system, &mode_kill)? {
                    super::mapper_minima::MinimumCheck::Minimum(_) => Some(0),
                    super::mapper_minima::MinimumCheck::Saddle { negative, .. } => Some(negative),
                    super::mapper_minima::MinimumCheck::Unjudged => None,
                },
            )
        })
    }

    fn irc_end_threshold_cm1(&self) -> f64 {
        self.steps.irc_end_threshold_cm1
    }

    fn higher_saddles(&self) -> Option<crate::workflow::mapper::HigherSaddleSettings> {
        self.steps.higher_saddles
    }

    fn confirm_bifurcations(&self) -> bool {
        self.steps.confirm_bifurcations
    }

    fn saddle(
        &self,
        start: &Configuration,
        order: usize,
        follow: &[faer::Col<f64>],
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<crate::workflow::mapper::OrderedSaddle> {
        if self.stop.check() {
            return Ok(crate::workflow::mapper::OrderedSaddle {
                optimized: Optimized {
                    index_wavenumbers: None,
                    system: start.clone(),
                    energy: f64::NAN,
                    converged: false,
                    interrupted: true,
                    path: Vec::new(),
                    wavenumbers: None,
                },
                modes: None,
            });
        }
        match start {
            Configuration::Structure(sys) => self.with_calc(|calc| {
                self.saddle_with(calc, sys, order, follow, FOLLOWING_METHOD, label, frames)
            }),
            Configuration::Variables(point) => self.saddle_variables(
                self.objective()?,
                point,
                order,
                follow,
                FOLLOWING_METHOD,
                label,
                frames,
            ),
        }
    }

    fn mode_kill_saddle(
        &self,
        start: &Configuration,
        modes: &crate::workflow::mapper::PointModes,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Option<crate::workflow::mapper::OrderedSaddle>> {
        if let (Some(cell), Configuration::Structure(sys)) =
            (cell::variable_cell(self.steps), start)
        {
            return self.with_calc(|calc| self.escape_cell_saddle(calc, sys, cell, label, frames));
        }
        let Some(unstable) = modes.unstable.get(..modes.negative).filter(|u| u.len() > 1) else {
            return Ok(None);
        };
        let o = self.config(Stage::Saddle);
        let settings = super::mapper_minima::mapper_mode_kill(o.settings.mode_kill);
        let origin = start.flat();
        let constraints = match start {
            Configuration::Structure(sys) if !o.settings.constraints.is_empty() => Some(
                crate::pes::constraint::ConstraintSet::new(&o.settings.constraints, sys, None)?,
            ),
            _ => None,
        };
        for attempt in 0..settings.max_rounds {
            if self.stop.check() {
                break;
            }
            let keep = (attempt / 2) % unstable.len();
            let scale = settings.displacement
                * if attempt % 2 == 0 { 1.0 } else { -1.0 }
                * 2.0_f64.powi((attempt / (2 * unstable.len())) as i32);
            let mut x = origin.clone();
            for (i, mode) in unstable.iter().enumerate() {
                if i != keep {
                    x += &mode.displacement * scale;
                }
            }
            let seed = start.with_flat(&x)?;
            let follow = std::slice::from_ref(&unstable[keep].displacement);
            let tag = format!("{label} HiSD attempt {}", attempt + 1);
            let dynamics = match &seed {
                Configuration::Structure(sys) => {
                    let seed = match &constraints {
                        Some(c) => c.restore(sys)?,
                        None => sys.clone(),
                    };
                    self.with_calc(|calc| {
                        self.saddle_with(calc, &seed, 1, follow, "hisd", &tag, frames)
                    })?
                }
                Configuration::Variables(p) => {
                    self.saddle_variables(self.objective()?, p, 1, follow, "hisd", &tag, frames)?
                }
            };
            if dynamics.optimized.interrupted {
                break;
            }
            if !dynamics.optimized.converged {
                continue;
            }
            let Some(m) = dynamics.modes.filter(|m| m.negative == 1) else {
                continue;
            };
            let refined = self.saddle(
                &dynamics.optimized.system,
                1,
                std::slice::from_ref(&m.unstable[0].displacement),
                label,
                frames,
            )?;
            let delta = refined.optimized.system.flat() - &origin;
            if refined.optimized.converged
                && refined.modes.as_ref().is_some_and(|m| m.negative == 1)
                && (crate::math::linalg::col_max_abs(&delta)
                    > o.settings.criteria.max_displacement.value()
                    || crate::math::linalg::col_norm(&delta) / (delta.nrows() as f64).sqrt()
                        > o.settings.criteria.rms_displacement.value())
            {
                return Ok(Some(refined));
            }
        }
        Ok(None)
    }

    fn hisd(
        &self,
        start: &Configuration,
        order: usize,
        directions: &[faer::Col<f64>],
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<crate::workflow::mapper::HisdOutcome> {
        if self.stop.check() {
            return Ok(crate::workflow::mapper::HisdOutcome {
                point: start.clone(),
                energy: f64::NAN,
                settled: false,
                end: "stopped".to_string(),
                directions: Vec::new(),
                interrupted: true,
            });
        }
        let outcome = match &self.calc {
            JobCalc::Objective(o) => {
                let surface = crate::workflow::pair_search::surface::VariableSurface(*o);
                super::mapper_searches::run_hisd(
                    &surface,
                    start,
                    order,
                    directions,
                    label,
                    frames,
                    &self.stop_signal(),
                    &self.config(Stage::Saddle).settings.criteria,
                    crate::path::hisd::SearchFrame::plain(),
                )
            }
            _ => self.with_calc(|calc| {
                let sys = start.require_structure(label)?;
                if let Some(cell) = cell::variable_cell(self.steps) {
                    return self.cell_hisd(calc, sys, cell, order, directions, label, frames);
                }
                let o = &self.steps.minimum;
                let images = vec![crate::workflow::pair_search::surface::ImageState {
                    template: sys.clone(),
                    excited_state: o.excited_state,
                    constraints: None,
                }];
                let mut surface =
                    crate::workflow::pair_search::surface::StructureSurface::new(calc, images)?;
                if let Some(expr) = state_surface(&o.model_function, o.excited_state)? {
                    surface = surface.with_states(expr);
                }
                super::mapper_searches::run_hisd(
                    &surface,
                    start,
                    order,
                    directions,
                    label,
                    frames,
                    &self.stop_signal(),
                    &self.config(Stage::Saddle).settings.criteria,
                    crate::path::hisd::SearchFrame::structure(
                        sys.clone(),
                        calc.is_rigid_body_invariant(),
                        true,
                    ),
                )
            }),
        }?;
        self.log.debug(&format!(
            "{label}: {}, E = {:.10} Hartree",
            outcome.end, outcome.energy
        ));
        Ok(outcome)
    }

    fn crossing(
        &self,
        start: &Configuration,
        other: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<crate::workflow::mapper::CrossingSearch> {
        if self.stop.check() {
            return Ok(crate::workflow::mapper::CrossingSearch {
                point: start.clone(),
                energy: f64::NAN,
                gap: f64::NAN,
                gradient_difference: None,
                converged: false,
                stationary: false,
                gap_limit: self.run_steps.crossing_models.gap_limit,
                parallel: None,
                interrupted: true,
                seam_check_failed: false,
                seam: None,
            });
        }
        let sys = start.require_structure(label)?;
        self.with_calc(|calc| self.crossing_with(calc, sys, other, label, frames))
    }

    fn meta_irc_in(
        &self,
        start: &Configuration,
        state: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<crate::workflow::mapper::MetaIrc> {
        if self.stop.check() {
            return Ok(crate::workflow::mapper::MetaIrc {
                interrupted: true,
                branch: None,
            });
        }
        let sys = start.require_structure(label)?;
        self.with_calc(|calc| self.meta_irc_with(calc, sys, state, label, frames))
    }

    fn crossing_path(
        &self,
        start: &Configuration,
        other: usize,
        path: crate::workflow::mapper::CrossingPath,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        if self.stop.check() {
            return Ok(Optimized {
                index_wavenumbers: None,
                system: start.clone(),
                energy: f64::NAN,
                converged: false,
                interrupted: true,
                path: Vec::new(),
                wavenumbers: None,
            });
        }
        let sys = start.require_structure(&request.label)?;
        self.with_calc(|calc| self.crossing_path_with(calc, sys, other, path, request, frames))
    }

    fn minimize_in(
        &self,
        start: &Configuration,
        state: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        let sys = start.require_structure(label)?;
        self.with_calc(|calc| self.minimize_with(calc, sys, state, label, frames))
    }

    fn displace_mode(
        &self,
        point: &Configuration,
        direction: &faer::Col<f64>,
        sign: f64,
        step: f64,
    ) -> MoptResult<Configuration> {
        if let (Configuration::Structure(system), Some(cell)) =
            (point, cell::variable_cell(self.steps))
        {
            let map = cell.coordinates(system)?;
            if direction.nrows() == map.dimension() {
                let x = map.variables(system)? + direction * (sign * step);
                return Ok(Configuration::Structure(map.system(&x)?));
            }
        }
        point.displaced_along(direction, sign * step)
    }

    fn modes(
        &self,
        point: &Configuration,
        across: bool,
    ) -> MoptResult<Option<crate::workflow::mapper::PointModes>> {
        let o = self.config(Stage::Minimum);
        let mode_kill = super::mapper_minima::mapper_mode_kill(o.settings.mode_kill);
        match point {
            Configuration::Variables(p) => {
                let o = self.objective()?;
                let v = o.evaluate(&p.values, true)?;
                let hessian = v.hessian.ok_or_else(|| {
                    MoptError::calc(format!("objective '{}' gave no Hessian", o.name()))
                })?;
                Ok(Some(super::mapper_minima::variable_modes(
                    &hessian,
                    &v.gradient,
                    across,
                    &mode_kill,
                )?))
            }
            Configuration::Structure(system) => self.with_calc(|calc| {
                if let Some(cell) = cell::variable_cell(self.steps) {
                    return Ok(Some(self.cell_modes(calc, system, across, cell)?));
                }
                let mut settings = o.settings.clone();
                settings.rigid_body_invariant = calc.is_rigid_body_invariant();
                let mut opt = GeometryOptimizer::new(settings, system.clone())?;
                if let Some(surface) = state_surface(&o.model_function, o.excited_state)? {
                    opt.set_surface(surface)?;
                }
                Ok(Some(super::mapper_minima::point_modes(
                    &mut opt, calc, system, across, &mode_kill,
                )?))
            }),
        }
    }

    fn energy(
        &self,
        point: &Configuration,
        on: Option<usize>,
        label: &str,
    ) -> MoptResult<Option<f64>> {
        if self.stop.check() {
            return Ok(None);
        }
        let energy = match point {
            Configuration::Variables(p) => self.objective()?.evaluate(&p.values, false)?.value,
            Configuration::Structure(sys) => match cell::variable_cell(self.steps) {
                Some(cell) => self.with_calc(|calc| self.cell_enthalpy(calc, sys, cell))?,
                None => self.with_calc(|calc| self.energy_with(calc, sys, on))?,
            },
        };
        self.log.info(&format!(
            "{label}: {} = {energy:.10} Hartree (the energy alone)",
            if cell::variable_cell(self.steps).is_some() {
                "H"
            } else {
                "E"
            }
        ));
        Ok(Some(energy))
    }

    fn path_peaks(&self) -> crate::workflow::mapper::PeakMode {
        self.steps.path_peaks
    }

    fn held(
        &self,
        start: &Configuration,
        constraints: &[crate::pes::constraint::Constraint],
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        if self.stop.check() {
            return Ok(Optimized {
                index_wavenumbers: None,
                system: start.clone(),
                energy: f64::NAN,
                converged: false,
                interrupted: true,
                path: Vec::new(),
                wavenumbers: None,
            });
        }
        let sys = start.require_structure(label)?;
        let request = OptimizeRequest {
            label: label.to_string(),
            stage: Stage::Biased,
            terms: Vec::new(),
        };
        self.with_calc(|calc| {
            self.optimize_held(calc, sys, &request, frames, self.steps, constraints)
        })
    }

    fn dissociation_scan(&self) -> Option<crate::workflow::mapper::ScanSettings> {
        self.steps.dissociation_scan
    }

    fn take_steering(&self) -> MoptResult<Vec<String>> {
        let path = self
            .stop
            .dir()
            .join(crate::workflow::mapper::steer::STEER_FILE);
        if !path.is_file() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| MoptError::from(e).context(path.display()))?;
        std::fs::remove_file(&path).map_err(|e| MoptError::from(e).context(path.display()))?;
        Ok(text.lines().map(str::to_string).collect())
    }

    fn log_steering(&self, after_jobs: usize, line: &str, outcome: &str) -> MoptResult<()> {
        use std::io::Write as _;
        let path = self
            .stop
            .dir()
            .join(crate::workflow::mapper::steer::STEER_LOG);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| MoptError::from(e).context(path.display()))?;
        writeln!(
            file,
            "after {after_jobs} job(s): {}: {outcome}",
            line.trim()
        )
        .map_err(|e| MoptError::from(e).context(path.display()))
    }

    fn ts_conformers(
        &self,
        ts: &Configuration,
        job: usize,
        label: &str,
    ) -> MoptResult<TsConformerSearch> {
        let sys = ts.require_structure(label)?;
        self.with_calc(|calc| self.ts_conformers_with(calc, sys, job, label))
    }

    fn stop_requested(&self) -> bool {
        self.stop.check()
    }

    fn stop_between_jobs(&self) -> bool {
        self.stop.between_jobs()
    }

    fn take_followed(&self) -> Vec<(Vec<f64>, FollowedRoots)> {
        self.followed
            .lock()
            .map(|mut kept| std::mem::take(&mut *kept))
            .unwrap_or_default()
    }

    fn channels(&self) -> Option<crate::workflow::mapper::ChannelSettings> {
        self.channel_settings()
    }

    fn whole_of(&self, point: &Configuration) -> MoptResult<Option<Configuration>> {
        match &self.whole {
            Some(whole) => self.with_calc(|calc| whole.whole_at(calc, point)),
            None => Ok(None),
        }
    }

    fn evaluations(&self) -> Option<EvaluationTotals> {
        if !self.of_run {
            return None;
        }
        let now = match self.objectives {
            Some(o) => o.counts(),
            None => self.registry.counts(),
        }
        .since(&self.since);
        Some(EvaluationTotals {
            energies: self.before.energies + now.energies,
            gradients: self.before.gradients + now.gradients,
            hessians: self.before.hessians + now.hessians,
            micro_relaxations: self.before.micro_relaxations + now.micro_relaxations,
            micro_iterations: self.before.micro_iterations + now.micro_iterations,
        })
    }

    fn start_from(&self, node: &crate::workflow::mapper::Node) {
        if let Some(whole) = &self.whole {
            // Nothing to give on an objective of plain variables (an error there).
            let _ = self.with_calc(|calc| {
                whole.start_from(calc, node);
                Ok(())
            });
        }
    }

    fn complete(&self, outcome: &mut crate::workflow::mapper::JobOutcome) -> MoptResult<()> {
        match &self.whole {
            Some(whole) => self.with_calc(|calc| whole.complete(calc, outcome)),
            None => Ok(()),
        }
    }

    fn note(&self, message: &str) {
        self.log.info(message);
    }

    fn progress(&self, message: &str) {
        self.screen.info(message);
    }

    fn record(&self, block: &str) {
        if let Some(file) = &self.file {
            file.append(block);
        }
    }

    fn take_lines(&self) -> Vec<String> {
        self.said.as_ref().map_or_else(Vec::new, |s| {
            s.lines()
                .into_iter()
                .map(|(level, line)| as_written(level, &line))
                .collect()
        })
    }
}

/// The hard constraints of the run as the paths and the IRC hold them: the set at a structure
/// (for the projections of the gradients), the fixed coordinates of the atoms held by `-fix`,
/// and the other constraints as a manifold of the points (put back onto them after every step).
pub(super) struct PathConstraints {
    /// The set at the template (`None`: the run has no constraints).
    pub(super) set: Option<crate::pes::constraint::ConstraintSet>,
    /// The coordinates of the fixed atoms (`3 i + k`).
    pub(super) frozen: Vec<usize>,
    /// The constraints beyond fixed atoms as a manifold of the points.
    pub(super) manifold: Option<Arc<crate::path::manifold::PointManifold>>,
    /// Whether the constraints pin the frame (fixed atoms, coordinates, orientations).
    pub(super) pins_frame: bool,
}

impl CommandJobs<'_> {
    /// [`PathConstraints`] of the constraints of the steps `steps` at `template` (those of the
    /// minimizations, which every stage shares).
    pub(super) fn path_constraints(
        steps: &StepConfigs,
        template: &MolecularSystem,
    ) -> MoptResult<PathConstraints> {
        use crate::pes::constraint::ConstraintKind;
        let constraints = &steps.minimum.settings.constraints;
        if constraints.is_empty() {
            return Ok(PathConstraints {
                set: None,
                frozen: Vec::new(),
                manifold: None,
                pins_frame: false,
            });
        }
        let set = crate::pes::constraint::ConstraintSet::new(constraints, template, None)?;
        let mut frozen = Vec::new();
        let mut others = false;
        for c in constraints {
            match &c.kind {
                ConstraintKind::Frozen(atoms) => frozen.extend(
                    atoms
                        .iter()
                        .flat_map(|a| (0..3).map(move |k| 3 * a.index() + k)),
                ),
                _ => others = true,
            }
        }
        frozen.sort_unstable();
        frozen.dedup();
        let manifold = if others {
            Some(Arc::new(crate::path::manifold::PointManifold::new(
                set.clone(),
                template.clone(),
            )?))
        } else {
            None
        };
        Ok(PathConstraints {
            pins_frame: set.breaks_rigid_body(),
            set: Some(set),
            frozen,
            manifold,
        })
    }
}

/// How an optimization ended, in a word.
pub(super) fn ended(termination: Option<Termination>) -> &'static str {
    match termination {
        Some(Termination::Converged) => "converged",
        Some(Termination::MaxIterations | Termination::SeamCheckFailed) => "not converged",
        Some(Termination::Dissociated) => "dissociated",
        None => "stopped",
    }
}

impl CommandJobs<'_> {
    /// Run `opt` with `calc` to its end: every point is recorded and sent to `frames`, what
    /// the optimizer warns of goes to the log under `label`, and a stop of the run ends it.
    /// Every optimization of structures of the mapper runs through here.
    pub(super) fn run_recorded(
        &self,
        opt: &mut GeometryOptimizer,
        calc: &dyn Calculator,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<(crate::workflow::driver::Outcome, Vec<PathPoint>)> {
        let (outcome, path, _) = self.run_recorded_states(opt, calc, label, frames)?;
        Ok((outcome, path))
    }

    /// [`Self::run_recorded`] with the energies of the two states of the surface at every
    /// point of the path (`None` where the surface is not one of two states).
    #[allow(clippy::type_complexity)]
    pub(super) fn run_recorded_states(
        &self,
        opt: &mut GeometryOptimizer,
        calc: &dyn Calculator,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<(
        crate::workflow::driver::Outcome,
        Vec<PathPoint>,
        Vec<Option<[f64; 2]>>,
    )> {
        let mut stop = self.stop.clone();
        let mut recorder = Recorder {
            label,
            path: Vec::new(),
            states: Vec::new(),
            frames,
            log: &self.log,
            watch: None,
        };
        let outcome = {
            let mut observers: Vec<&mut dyn Observer> = vec![&mut stop, &mut recorder];
            run_optimization(opt, calc, &mut observers)?
        };
        self.remember_followed(calc, &outcome.system);
        Ok((outcome, recorder.path, recorder.states))
    }

    /// [`Self::run_recorded`] of a crossing search that `parallel` (the rule and the gap that
    /// counts as closed) ends where its two states run parallel; with the outcome, the
    /// number of iterations in a row that ended it (`None`: it ended otherwise).
    pub(super) fn run_watched(
        &self,
        opt: &mut GeometryOptimizer,
        calc: &dyn Calculator,
        label: &str,
        frames: &mut dyn FrameSink,
        parallel: Option<(super::mapper_settings::ParallelStop, f64)>,
    ) -> MoptResult<(crate::workflow::driver::Outcome, Option<usize>)> {
        let mut stop = self.stop.clone();
        let mut recorder = Recorder {
            label,
            path: Vec::new(),
            states: Vec::new(),
            frames,
            log: &self.log,
            watch: parallel.map(|(rule, gap)| ParallelWatch { rule, gap, run: 0 }),
        };
        let outcome = {
            let mut observers: Vec<&mut dyn Observer> = vec![&mut stop, &mut recorder];
            run_optimization(opt, calc, &mut observers)?
        };
        self.remember_followed(calc, &outcome.system);
        let ended = recorder
            .watch
            .as_ref()
            .filter(|w| w.ended() && outcome.termination.is_none())
            .map(|w| w.run);
        Ok((outcome, ended))
    }

    /// Keep the followed electronic states at `sys` as `calc` knows them (the excited states
    /// of the run that it has followed to this structure), for the node that `sys` becomes.
    fn remember_followed(&self, calc: &dyn Calculator, sys: &MolecularSystem) {
        let mut labels: Vec<usize> = self
            .run_steps
            .states
            .iter()
            .map(|s| s.excited_state)
            .filter(|&k| k > 0)
            .collect();
        labels.sort_unstable();
        labels.dedup();
        let roots: FollowedRoots = labels
            .into_iter()
            .filter_map(|label| calc.followed_state(sys, label).map(|root| (label, root)))
            .collect();
        if roots.is_empty() {
            return;
        }
        if let Ok(mut kept) = self.followed.lock() {
            kept.push((sys.coords_flat_bohr().iter().copied().collect(), roots));
        }
    }

    fn optimize_with(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        self.optimize_in(calc, start, request, frames, self.steps)
    }

    /// [`Self::optimize_with`] with the steps `steps` (those of another state).
    fn optimize_in(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
        steps: &StepConfigs,
    ) -> MoptResult<Optimized> {
        self.optimize_held(calc, start, request, frames, steps, &[])
    }

    /// [`Self::optimize_in`] with `constraints` held beside those of the settings (a point of
    /// a relaxed scan).
    fn optimize_held(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
        steps: &StepConfigs,
        constraints: &[crate::pes::constraint::Constraint],
    ) -> MoptResult<Optimized> {
        if let Some(cell) = cell::variable_cell(steps) {
            if !constraints.is_empty() {
                return Err(MoptError::unsupported(format!(
                    "{}: constraints on a variable cell (the optimization of the atoms and the \
                     cell holds none)",
                    request.label
                )));
            }
            return self.optimize_cell(calc, start, request, frames, steps, cell);
        }
        let o = steps.config(request.stage);
        let mut settings = o.settings.clone();
        settings.constraints.extend(constraints.iter().cloned());
        settings.rigid_body_invariant = crate::pes::term::frame_invariant(
            calc.is_rigid_body_invariant(),
            request.terms.iter().map(|(_, t)| t.as_ref()),
        );
        let verify = matches!(request.stage, Stage::Minimum | Stage::IrcEnd) && steps.verify_minima;
        // The threshold of the mode kill scaled to the surface.
        settings.mode_kill = settings
            .mode_kill
            .map(|m| super::mapper_minima::mapper_mode_kill(Some(m)));
        let mode_kill = super::mapper_minima::mapper_mode_kill(settings.mode_kill);
        let mut opt = GeometryOptimizer::new(settings, start.clone())?;
        if let Some(surface) = state_surface(&o.model_function, o.excited_state)? {
            opt.set_surface(surface)?;
        }
        for (label, term) in &request.terms {
            opt.add_term(label, Arc::clone(term))?;
        }
        let (outcome, path) = self.run_recorded(&mut opt, calc, &request.label, frames)?;
        let mut converged = outcome.termination == Some(Termination::Converged);
        let mut wavenumbers = None;
        let mut index_wavenumbers = None;
        if verify && converged {
            match super::mapper_minima::check_minimum(&mut opt, calc, &outcome.system, &mode_kill)?
            {
                super::mapper_minima::MinimumCheck::Minimum(w) => {
                    index_wavenumbers = Some(Vec::new());
                    wavenumbers = Some(w);
                }
                super::mapper_minima::MinimumCheck::Unjudged if request.stage == Stage::IrcEnd => {
                    converged = false;
                    self.log
                        .warn(&format!("{}: no converged endpoint Hessian", request.label));
                }
                super::mapper_minima::MinimumCheck::Saddle {
                    wavenumbers: w,
                    imaginary,
                    ..
                } if request.stage == Stage::IrcEnd => {
                    index_wavenumbers = Some(imaginary);
                    wavenumbers = Some(w)
                }
                super::mapper_minima::MinimumCheck::Unjudged => self.log.warn(&format!(
                    "{}: the stability of the converged structure cannot be judged (its Hessian \
                     could not be had to convergence); kept as a minimum without the check",
                    request.label
                )),
                super::mapper_minima::MinimumCheck::Saddle {
                    negative,
                    lowest,
                    imaginary,
                    wavenumbers: w,
                } if self.apart_in_cell(&outcome.system).is_some() => {
                    // Apart in its cell: a dissociation channel, kept with its imaginary modes.
                    self.log.info(&format!(
                        "{}: {negative} imaginary mode(s) remain (lowest {lowest:.1} cm^-1), but \
                         {}: kept as a dissociation channel",
                        request.label,
                        self.apart_in_cell(&outcome.system).unwrap_or_default()
                    ));
                    index_wavenumbers = Some(imaginary);
                    wavenumbers = Some(w);
                }
                super::mapper_minima::MinimumCheck::Saddle {
                    negative, lowest, ..
                } => {
                    converged = false;
                    self.log.warn(&format!(
                        "{}: the converged structure is not a minimum: {negative} imaginary \
                         mode(s) remain (lowest {lowest:.1} cm^-1)",
                        request.label
                    ));
                }
            }
        }
        self.log.debug(&format!(
            "{}: {} iterations, E = {:.10} Hartree, {}",
            request.label,
            outcome.iterations,
            outcome.energy.value(),
            ended(outcome.termination)
        ));
        Ok(Optimized {
            index_wavenumbers,
            energy: outcome.energy.value(),
            system: Configuration::Structure(outcome.system),
            converged,
            interrupted: outcome.termination.is_none() && self.stop.stopped(),
            path,
            wavenumbers,
        })
    }

    fn irc_with(
        &self,
        calc: &dyn Calculator,
        ts: &MolecularSystem,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<IrcOutcome> {
        self.irc_in(calc, ts, frames, self.steps, false)
    }

    /// The IRC from `ts` with the steps `steps` (those of a state), run by the machine of
    /// `ircmain` ([`super::irc::run_structure_observed`]). With `meta` a start that is no
    /// transition state is followed down as a meta-IRC (one side) instead of being refused.
    pub(super) fn irc_in(
        &self,
        calc: &dyn Calculator,
        ts: &MolecularSystem,
        frames: &mut dyn FrameSink,
        steps: &StepConfigs,
        meta: bool,
    ) -> MoptResult<IrcOutcome> {
        if let Some(cell) = cell::variable_cell(steps) {
            return self.irc_cell(calc, ts, frames, steps, meta, cell);
        }
        let settings = Self::irc_settings(steps);
        let o = &steps.minimum;
        // The IRC on the surface of the constraints of the run (`irc::constraints`).
        let held = Self::path_constraints(steps, ts)?;
        let start = Configuration::Structure(ts.clone());
        let mut collector = IrcCollector {
            start: &start,
            frames,
            sides: Vec::new(),
            kind: None,
            imaginary: 0,
            wavenumber: None,
            // In a cell: the translations along its periodic directions.
            rigid: calc.is_rigid_body_invariant() && !held.pins_frame,
            wavenumbers: None,
            bifurcation_threshold_cm1: steps.bifurcation_threshold_cm1,
            clear: Vec::new(),
            refused: false,
            meta,
            start_gradient: None,
            cell: None,
        };
        let mut watch = StopWatch(self.stop.clone());
        let end = super::irc::run_structure_observed(
            &settings,
            calc,
            super::irc::IrcSurfaceParts {
                terms: Vec::new(),
                excited_state: o.excited_state,
                states: irc_states(&o.model_function, o.excited_state)?,
                constraints: held.set,
                iteration: 0,
            },
            ts,
            &self.log,
            &mut [&mut collector, &mut watch],
        )?;
        say_not_stationary(&collector, &settings, &self.log);
        Ok(irc_outcome(collector, end.is_none()))
    }

    /// [`MapperJobs::optimize`] of a point of plain variables on `objective`.
    fn optimize_variables(
        &self,
        objective: &dyn Objective,
        start: &VariablePoint,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        if !request.terms.is_empty() {
            return Err(MoptError::unsupported(format!(
                "{}: bias potentials need atoms; the points of this network are plain variables",
                request.label
            )));
        }
        let o = self.config(request.stage);
        let values: Vec<f64> = start.values.iter().copied().collect();
        let mut settings = super::variables::vector_settings(o, &values)?;
        let verify =
            matches!(request.stage, Stage::Minimum | Stage::IrcEnd) && self.steps.verify_minima;
        // The threshold of the mode kill scaled to the surface.
        settings.mode_kill = settings
            .mode_kill
            .map(|m| super::mapper_minima::mapper_mode_kill(Some(m)));
        let mode_kill = super::mapper_minima::mapper_mode_kill(settings.mode_kill);
        let mut opt = VectorOptimizer::new(settings, start.values.clone())?;
        let template = Configuration::Variables(start.clone());
        let mut path = Vec::new();
        let stop = self.stop.clone();
        let outcome = run_vector_optimization_until(
            &mut opt,
            objective,
            &mut |r| {
                let point = template.with_flat(&r.x)?;
                frames.push(
                    &point,
                    r.energy,
                    &format!("{} iteration {}", request.label, r.iteration),
                )?;
                path.push(PathPoint {
                    system: point,
                    energy: r.energy,
                    biased_energy: r.energy,
                });
                Ok(())
            },
            &mut || stop.check(),
        )?;
        let mut converged = outcome.termination == Some(Termination::Converged);
        let mut wavenumbers = None;
        let mut index_wavenumbers = None;
        if verify && converged {
            let v = objective.evaluate(&outcome.x, true)?;
            if let Some(h) = &v.hessian {
                let negative = match crate::opt::mode_kill::examine(
                    h,
                    &v.gradient,
                    0,
                    &mode_kill,
                    None,
                    None,
                )? {
                    crate::opt::mode_kill::ModeKillVerdict::Accept { negative }
                    | crate::opt::mode_kill::ModeKillVerdict::Displace { negative, .. } => negative,
                };
                let waves = super::mapper_minima::variable_wavenumbers(h)?;
                index_wavenumbers = Some(waves.iter().take(negative).copied().collect());
                wavenumbers = Some(waves);
                if negative > 0 && request.stage != Stage::IrcEnd {
                    converged = false;
                    self.log.warn(&format!(
                        "{}: the converged point is not a minimum: {negative} negative \
                         curvature(s) remain",
                        request.label
                    ));
                }
            } else if request.stage == Stage::IrcEnd {
                converged = false;
                self.log
                    .warn(&format!("{}: no endpoint Hessian", request.label));
            }
        }
        self.log.debug(&format!(
            "{}: {} iterations, value {:.10}, {}",
            request.label,
            outcome.iterations,
            outcome.value,
            ended(outcome.termination)
        ));
        Ok(Optimized {
            index_wavenumbers,
            energy: outcome.value,
            system: template.with_flat(&outcome.x)?,
            converged,
            interrupted: outcome.termination.is_none() && self.stop.stopped(),
            path,
            wavenumbers,
        })
    }

    /// [`MapperJobs::saddle`] of a structure: the settings of the transition-state search at
    /// `order`, with the method that follows modes where `follow` is given, and the modes of
    /// the Hessian at a converged end.
    #[allow(clippy::too_many_arguments)]
    fn saddle_with(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        order: usize,
        follow: &[faer::Col<f64>],
        method: &str,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<crate::workflow::mapper::OrderedSaddle> {
        if let Some(cell) = cell::variable_cell(self.steps) {
            return self.saddle_cell(calc, start, order, follow, label, frames, cell);
        }
        let o = self.config(Stage::Saddle);
        let mut settings = o.settings.clone();
        settings.saddle_order = order;
        settings.rigid_body_invariant = calc.is_rigid_body_invariant();
        settings.mode_kill = None;
        if !follow.is_empty() {
            settings.methods = vec![method.to_string()];
        }
        let mode_kill =
            super::mapper_minima::mapper_mode_kill(self.config(Stage::Minimum).settings.mode_kill);
        let mut opt = GeometryOptimizer::new(settings, start.clone())?;
        if let Some(surface) = state_surface(&o.model_function, o.excited_state)? {
            opt.set_surface(surface)?;
        }
        if !follow.is_empty() {
            opt.follow_directions(follow)?;
        }
        let (outcome, path) = self.run_recorded(&mut opt, calc, label, frames)?;
        let converged = outcome.termination == Some(Termination::Converged);
        let modes = if converged {
            Some(super::mapper_minima::point_modes(
                &mut opt,
                calc,
                &outcome.system,
                false,
                &mode_kill,
            )?)
        } else {
            None
        };
        self.log.debug(&format!(
            "{label}: order {order}, {} iterations, E = {:.10} Hartree, {}",
            outcome.iterations,
            outcome.energy.value(),
            match (&modes, outcome.termination) {
                (Some(m), _) => format!("converged, {} negative curvature(s)", m.negative),
                (None, Some(Termination::MaxIterations)) => "not converged".to_string(),
                (None, _) => "stopped".to_string(),
            }
        ));
        Ok(crate::workflow::mapper::OrderedSaddle {
            optimized: Optimized {
                index_wavenumbers: None,
                energy: outcome.energy.value(),
                system: Configuration::Structure(outcome.system),
                converged,
                interrupted: outcome.termination.is_none() && self.stop.stopped(),
                path,
                wavenumbers: None,
            },
            modes,
        })
    }

    /// [`MapperJobs::saddle`] of a point of plain variables on `objective`.
    #[allow(clippy::too_many_arguments)]
    fn saddle_variables(
        &self,
        objective: &dyn Objective,
        start: &VariablePoint,
        order: usize,
        follow: &[faer::Col<f64>],
        method: &str,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<crate::workflow::mapper::OrderedSaddle> {
        let o = self.config(Stage::Saddle);
        let values: Vec<f64> = start.values.iter().copied().collect();
        let mut settings = super::variables::vector_settings(o, &values)?;
        settings.saddle_order = order;
        settings.mode_kill = None;
        if !follow.is_empty() {
            settings.methods = vec![method.to_string()];
        }
        let mode_kill =
            super::mapper_minima::mapper_mode_kill(self.config(Stage::Minimum).settings.mode_kill);
        let mut opt = VectorOptimizer::new(settings, start.values.clone())?;
        if !follow.is_empty() {
            opt.follow_directions(follow)?;
        }
        let template = Configuration::Variables(start.clone());
        let mut path = Vec::new();
        let stop = self.stop.clone();
        let outcome = run_vector_optimization_until(
            &mut opt,
            objective,
            &mut |r| {
                let point = template.with_flat(&r.x)?;
                frames.push(
                    &point,
                    r.energy,
                    &format!("{label} iteration {}", r.iteration),
                )?;
                path.push(PathPoint {
                    system: point,
                    energy: r.energy,
                    biased_energy: r.energy,
                });
                Ok(())
            },
            &mut || stop.check(),
        )?;
        let converged = outcome.termination == Some(Termination::Converged);
        let modes = if converged {
            let v = objective.evaluate(&outcome.x, true)?;
            match &v.hessian {
                Some(h) => Some(super::mapper_minima::variable_modes(
                    h,
                    &v.gradient,
                    false,
                    &mode_kill,
                )?),
                None => None,
            }
        } else {
            None
        };
        Ok(crate::workflow::mapper::OrderedSaddle {
            optimized: Optimized {
                index_wavenumbers: None,
                energy: outcome.value,
                system: template.with_flat(&outcome.x)?,
                converged,
                interrupted: outcome.termination.is_none() && self.stop.stopped(),
                path,
                wavenumbers: None,
            },
            modes,
        })
    }

    /// [`MapperJobs::irc`] from a point of plain variables on `objective` (unit masses).
    fn irc_variables(
        &self,
        objective: &dyn Objective,
        ts: &Configuration,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<IrcOutcome> {
        let settings = Self::irc_settings(self.steps);
        let x = ts.flat();
        let mut irc = Irc::new(settings.clone(), IrcSpace::variables(x.nrows()), x)?;
        let mut collector = IrcCollector {
            start: ts,
            frames,
            sides: Vec::new(),
            kind: None,
            imaginary: 0,
            wavenumber: None,
            rigid: false,
            wavenumbers: None,
            bifurcation_threshold_cm1: self.steps.bifurcation_threshold_cm1,
            clear: Vec::new(),
            refused: false,
            meta: false,
            start_gradient: None,
            cell: None,
        };
        let mut watch = StopWatch(self.stop.clone());
        let end = run_irc(
            &mut irc,
            &ObjectiveSurface(objective),
            &mut [&mut collector, &mut watch],
        )?;
        say_not_stationary(&collector, &settings, &self.log);
        Ok(irc_outcome(collector, end.is_none()))
    }
}

/// Say in `log` that the start of an IRC had one imaginary mode but a gradient above the
/// thresholds of `settings` (so it was not taken as a transition state), with the numbers.
fn say_not_stationary(collector: &IrcCollector<'_>, settings: &IrcSettings, log: &Logger) {
    if collector.meta
        || collector.imaginary != 1
        || collector.kind == Some(StartKind::TransitionState)
    {
        return;
    }
    let Some((max, rms)) = collector.start_gradient else {
        return;
    };
    let (max_allowed, rms_allowed) = settings
        .start_forces
        .unwrap_or((settings.max_force, settings.rms_force));
    log.warn(&format!(
        "IRC: the start has one imaginary mode, but its gradient (largest component {max:.3e}, \
         RMS {rms:.3e} Hartree/Bohr) is above the thresholds of a transition state \
         ({max_allowed:.3e}, {rms_allowed:.3e})"
    ));
}

/// What an IRC whose points `collector` kept ended with (`stopped`: it ended without an end).
fn irc_outcome(collector: IrcCollector<'_>, stopped: bool) -> IrcOutcome {
    let transition_state = collector.kind == Some(StartKind::TransitionState);
    let interrupted = stopped && !collector.refused;
    let collector_meta = collector.meta;
    let mut branches: Vec<(Branch, IrcBranch)> = collector.sides;
    branches.sort_by_key(|(b, _)| match b {
        Branch::Backward => 0,
        Branch::Forward => 1,
        Branch::Single => 2,
    });
    IrcOutcome {
        interrupted,
        transition_state,
        imaginary_modes: collector.imaginary,
        imaginary_wavenumber: collector.wavenumber,
        wavenumbers: if transition_state {
            collector.wavenumbers
        } else {
            None
        },
        branches: if transition_state || collector_meta {
            branches.into_iter().map(|(_, b)| b).collect()
        } else {
            Vec::new()
        },
    }
}

/// Makes a calculator of the surface of the run (for a job that cannot share the one of the
/// run).
pub type MakeCalculator<'a> = dyn Fn() -> MoptResult<Box<dyn Calculator>> + Sync + 'a;

/// The calculations of the jobs of a run, which may run at the same time.
pub struct CommandWorkers<'a> {
    run: CommandJobs<'a>,
    dir: PathBuf,
    journal: &'a RunJournal,
    /// The calculator of the run (not journaled: every job records under its own label),
    /// shared by the jobs; `None` when it evaluates in order and every job makes its own with
    /// `make`.
    shared: Option<&'a dyn Calculator>,
    make: &'a MakeCalculator<'a>,
}

impl<'a> CommandWorkers<'a> {
    /// The jobs of the run of `run` in the run directory `dir`; they share the calculator
    /// `shared` of the run (as it is, without the journal) unless it evaluates in order, in
    /// which case every job takes one of its own from `make`; either way the evaluations of a
    /// job are recorded in `journal` under the label of the job ([`job_label`]). `None`: the
    /// run is on an objective of plain variables, which every job shares.
    pub fn new(
        run: CommandJobs<'a>,
        dir: &Path,
        journal: &'a RunJournal,
        shared: Option<&'a dyn Calculator>,
        make: &'a MakeCalculator<'a>,
    ) -> Self {
        let shared = shared.filter(|c| !c.evaluates_in_order());
        Self {
            run,
            dir: dir.to_path_buf(),
            journal,
            shared,
            make,
        }
    }

    /// Whether the jobs share the calculator (or the objective) of the run.
    pub fn share_calculator(&self) -> bool {
        self.shared.is_some() || matches!(self.run.calc, JobCalc::Objective(_))
    }
}

impl MapperWorkers for CommandWorkers<'_> {
    fn run(&self) -> &dyn MapperJobs {
        &self.run
    }

    fn job(&self, job: &Job, deadline: Option<Instant>) -> MoptResult<Box<dyn MapperJobs + '_>> {
        // The log of the job, written anew when a restart runs it again.
        let save_steps = self.run.run_steps.save_job_steps;
        // What the job says goes to its own file (to follow it while it runs) and is kept for
        // its block in the log of the run; nothing of it goes to the screen from its thread.
        let log = Logger::silent();
        if save_steps {
            let path = self.dir.join(job_log_file(job.id));
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
            log.add_file(&path, Level::Debug)?;
        }
        let said = MemorySink::new();
        log.add_sink(
            if save_steps {
                Level::Debug
            } else {
                Level::Warn
            },
            Box::new(said.clone()),
        );
        let calc = match (&self.run.calc, self.shared) {
            (JobCalc::Objective(o), _) => JobCalc::Objective(*o),
            (_, Some(c)) => JobCalc::Job(c, self.journal, job_label(job.id)),
            (_, None) => {
                let own = (self.make)()?;
                // A calculator that follows its excited states starts from those of the node:
                // a state of the run is in the job the state it is at the node, whatever its
                // place in the order of the energies there.
                if let Some(start) = job.start.structure() {
                    for &(label, root) in &job.roots {
                        own.recall_state(start, label, root);
                    }
                }
                JobCalc::Own(own, self.journal, job_label(job.id))
            }
        };
        // The steps of the state of the job.
        let steps = self
            .run
            .run_steps
            .on_states
            .get(job.state)
            .unwrap_or(self.run.run_steps);
        Ok(Box::new(CommandJobs {
            steps,
            run_steps: self.run.run_steps,
            state: job.state,
            calc,
            stop: self.run.stop.for_job(deadline),
            log,
            screen: Logger::silent(),
            file: None,
            said: Some(said),
            registry: self.run.registry,
            objectives: self.run.objectives,
            since: self.run.since,
            before: self.run.before,
            of_run: false,
            followed: std::sync::Mutex::new(Vec::new()),
            whole: self.run.whole.clone(),
        }))
    }
}

#[path = "mapper_states.rs"]
mod states;

#[path = "mapper_conformers.rs"]
mod conformers;

/// The stop signal as an observer of an IRC.
struct StopWatch(MapperStop);

impl IrcObserver for StopWatch {
    fn on_event(&mut self, _event: &IrcEvent) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        self.0.check()
    }
}
