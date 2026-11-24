//! How a node of the network is explored (extension point): an [`ExplorationMethod`] lists
//! the tasks of a node (for AFIR, the pushes between pairs of fragments) and runs one of them
//! as a job, with the optimizations and IRCs of the run ([`MapperJobs`]), into what the job
//! found ([`JobOutcome`]). Methods are registered by name in a [`MethodRegistry`]; users add
//! their own from Rust with [`MethodRegistry::register`].
//!
//! A method does not touch the network: the driver ([`super::driver`]) compares what a job
//! found with the known nodes and adds what is new, so that every method gets the same
//! duplicate detection, bookkeeping, and restart.
//!
//! MultiOptPy: `Wrapper/mapper.py:PerturbationGenerator` and `ReactionNetworkMapper._run_autots`
//!  (one AutoTS workflow per AFIR pair, run in a subprocess).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use super::channels::ChannelSettings;
use super::configuration::Configuration;
use super::dissociation::Dissociation;
use super::network::{IrcSide, Network, Node, PossibleBifurcation};
use crate::core_infra::json::JsonValue;
use crate::error::{MoptError, MoptResult};
use crate::pes::term::PesTerm;
use crate::workflow::explore::identity::StructureMatcher;

/// One task of a node: what a job started from the node does.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    /// Unique among the tasks of the node and stable between runs (the restart and the
    /// record of what was done go by it), e.g. `afir:+100:1-5`.
    pub key: String,
    /// What it does, for the log and the report.
    pub description: String,
    /// Its parameters, for the method that made it.
    pub parameters: JsonValue,
}

/// A job: a task started from a node.
#[derive(Debug, Clone)]
pub struct Job {
    /// Its number (from 0, in the order of the run).
    pub id: usize,
    /// The node it starts from.
    pub node: usize,
    /// The task.
    pub task: Task,
    /// The structure of the node.
    pub start: Configuration,
    /// The electronic state of the node, its index among the states of the run (`states`;
    /// 0 for a run of one state): every calculation of the job is on that surface.
    pub state: usize,
    /// The followed electronic states at the structure of the node ([`FollowedRoots`] of the
    /// node): the calculator of the job starts from them, so that a state of the run is the
    /// same state in the job as at the node.
    pub roots: FollowedRoots,
}

/// The followed electronic states at a structure, where the calculator follows its excited
/// states from structure to structure (`follow_states` of the steps, `-es_follow`): for the
/// number of an excited state as the run has it (`ROOT` of `states`, from 1), its place in
/// the order of the energies at this structure (0: the lowest excited state). Empty where
/// nothing is followed.
pub type FollowedRoots = Vec<(usize, usize)>;

/// Default minimum magnitude of an imaginary vibrational wavenumber at an optimized IRC endpoint.
pub const DEFAULT_IRC_END_THRESHOLD_CM1: f64 = 50.0;

/// Which optimization a method asks for; the run takes the settings of the matching step
/// (`step1_settings`, `step3_settings`, `step4_settings` of the settings file).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// A minimization with bias potentials that drive a reaction (step 1).
    Biased,
    /// A transition-state search (step 3, saddle order 1).
    Saddle,
    /// A minimization of a structure (step 4 settings).
    Minimum,
    /// Minimize an IRC endpoint and return its signed vibrational wavenumbers.
    IrcEnd,
}

/// An optimization a method asks for.
pub struct OptimizeRequest {
    /// What it is, for the log and the trajectory (`AFIR path`, `TS guess 1`, ...).
    pub label: String,
    /// Which one.
    pub stage: Stage,
    /// Bias potentials on top of the surface of the run, with their labels.
    pub terms: Vec<(String, Arc<dyn PesTerm>)>,
}

/// A point of an optimization path.
#[derive(Debug, Clone)]
pub struct PathPoint {
    /// The structure.
    pub system: Configuration,
    /// Energy of the surface without the bias potentials (Hartree).
    pub energy: f64,
    /// Energy with them.
    pub biased_energy: f64,
}

/// What an optimization ended with.
#[derive(Debug, Clone)]
pub struct Optimized {
    /// Signed wavenumbers from the modes counted in the saddle index (cm^-1).
    pub index_wavenumbers: Option<Vec<f64>>,
    /// The last structure.
    pub system: Configuration,
    /// Its energy without the bias potentials (Hartree).
    pub energy: f64,
    /// Whether it converged.
    pub converged: bool,
    /// Whether the run was stopped during it.
    pub interrupted: bool,
    /// Every evaluated structure in order.
    pub path: Vec<PathPoint>,
    /// The vibrational wavenumbers of a minimum from the Hessian that verified it (cm^-1,
    /// imaginary ones negative; `verify_minima`); `None` when it was not verified.
    pub wavenumbers: Option<Vec<f64>>,
}

/// One side of an IRC with its structures.
#[derive(Debug, Clone)]
pub struct IrcBranch {
    /// The side as data (names, arc lengths, energies, end).
    pub side: IrcSide,
    /// The structures of its points, in the order of [`IrcSide::energies`].
    pub frames: Vec<Configuration>,
    /// Its last structure.
    pub last: Configuration,
    /// The minimization of its last structure to the minimum at its end, every evaluated
    /// structure in order (empty where the end is not a minimum reached that way: a
    /// dissociation channel, another saddle point). The path file of the reaction goes on
    /// through it, so that it runs from minimum to minimum.
    pub settle: Vec<PathPoint>,
    /// Where it may bifurcate (negative curvature across the path at an exact Hessian).
    pub possible_bifurcation: Option<PossibleBifurcation>,
}

/// What an IRC from a structure ended with.
#[derive(Debug, Clone)]
pub struct IrcOutcome {
    /// Whether the run was stopped during it.
    pub interrupted: bool,
    /// Whether the structure is a first-order saddle point (one imaginary mode); when it is
    /// not, no path was run.
    pub transition_state: bool,
    /// The number of imaginary modes found at it.
    pub imaginary_modes: usize,
    /// The imaginary wavenumber (cm^-1, negative).
    pub imaginary_wavenumber: Option<f64>,
    /// The wavenumbers of the vibrations of a transition state from the Hessian of the
    /// start check (cm^-1, the imaginary one negative); `None` when it is not one.
    pub wavenumbers: Option<Vec<f64>>,
    /// The sides, backward first.
    pub branches: Vec<IrcBranch>,
}

/// Takes the structures of a job as they come (the trajectory of the job).
pub trait FrameSink: Send {
    /// One structure with its energy and what it is.
    fn push(&mut self, system: &Configuration, energy: f64, label: &str) -> MoptResult<()>;
}

/// A search with a method of the registry of two-point methods
/// ([`crate::workflow::pair_search::PairMethodRegistry`]) on the surface of the run.
#[derive(Debug, Clone)]
pub struct PairSearchRequest {
    /// What it is, for the log and the trajectory (`ADDF`, `dsafir of reaction 1`).
    pub label: String,
    /// The job it belongs to (its files go to the directory of the job).
    pub job: usize,
    /// The name of the method in the registry (`addf`, `dsafir`, `dmf`, ...).
    pub method: String,
    /// The structures it starts from (one or two, as the method takes them).
    pub structures: Vec<Configuration>,
    /// Its `KEY=VALUE` settings.
    pub settings: Vec<(String, String)>,
    /// A point whose structure has more fragments than the first structure, with bonds up to
    /// this times the sums of the covalent radii, has fallen apart (ADDF ends such a path as a
    /// dissociation channel); `None`: the scale of `ieipmain`.
    pub dissociation_scale: Option<f64>,
    /// Only fragments that hold one of these atoms (from 0) count for that test; `None`: all.
    pub dissociation_atoms: Option<Vec<usize>>,
}

/// What a search of the registry ended with.
#[derive(Debug, Clone, Default)]
pub struct PairSearchOutcome {
    /// Whether the run was stopped during it.
    pub interrupted: bool,
    /// Whether the method ended on its criteria.
    pub converged: bool,
    /// Why it ended, in words.
    pub end: String,
    /// The estimates of transition states, the lowest first (energies in Hartree).
    pub estimates: Vec<Stationary>,
    /// The path, where the method makes one.
    pub path: Vec<PathPoint>,
}

impl PairSearchOutcome {
    /// The outcome of a search that was stopped.
    pub fn stopped() -> Self {
        Self {
            interrupted: true,
            end: "stopped".to_string(),
            ..Self::default()
        }
    }
}

/// Molecular dynamics from a structure on the surface of the run with bias potentials.
pub struct DynamicsRequest {
    /// What it is, for the log and the trajectory.
    pub label: String,
    /// Bias potentials on top of the surface of the run, with their labels.
    pub terms: Vec<(String, Arc<dyn PesTerm>)>,
    /// Temperature of the thermostat and of the starting velocities (K).
    pub temperature: f64,
    /// Time step (atomic units).
    pub time_step: f64,
    /// Steps.
    pub steps: usize,
    /// Seed of the starting velocities and of the thermostat (the same for a restart).
    pub seed: u64,
    /// Every how many steps a frame is kept (the start is step 0).
    pub frame_every: usize,
    /// The starting velocities (Bohr per atomic unit of time; `None`: drawn at the
    /// temperature): a trajectory that goes on from where another ended.
    pub velocities: Option<faer::Col<f64>>,
}

/// Where molecular dynamics ended: the last structure, its velocities, and its energies
/// without and with the bias potentials.
#[derive(Debug, Clone)]
pub struct DynamicsEnd {
    /// The structure.
    pub point: Configuration,
    /// The velocities (Bohr per atomic unit of time).
    pub velocities: faer::Col<f64>,
    /// The energy of the surface (Hartree).
    pub energy: f64,
    /// The same with the bias potentials.
    pub biased_energy: f64,
}

/// What molecular dynamics ended with.
#[derive(Debug, Clone, Default)]
pub struct DynamicsOutcome {
    /// Whether the run was stopped during it.
    pub interrupted: bool,
    /// The kept frames, the start first, with their energies without and with the bias
    /// potentials.
    pub frames: Vec<PathPoint>,
    /// The step at which the dynamics blew up (an atom jumped farther in one step than any
    /// thermal motion goes); it ended there, and the frames are those before it.
    pub blew_up: Option<usize>,
    /// Where it ended (`None` where it blew up or ran no step).
    pub end: Option<DynamicsEnd>,
}

/// The calculations a method can ask for, on the surface of the run (implemented by the
/// command, with its calculator, settings, journal, and stop signal).
pub trait MapperJobs {
    /// An optimization from `start`; every evaluated structure goes to `frames`.
    fn optimize(
        &self,
        start: &Configuration,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized>;

    /// The IRC from `ts` (both sides) after its Hessian was checked; every point goes to
    /// `frames`.
    fn irc(&self, ts: &Configuration, frames: &mut dyn FrameSink) -> MoptResult<IrcOutcome>;

    /// A search with a method of the registry of two-point methods; every evaluated
    /// structure goes to `frames`. Not available by default.
    fn pair_search(
        &self,
        request: &PairSearchRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<PairSearchOutcome> {
        let _ = frames;
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper do not run the two-point method '{}' ({})",
            request.method, request.label
        )))
    }

    /// Molecular dynamics from `start`; the kept frames go to `frames`. Not available by
    /// default.
    fn dynamics(
        &self,
        start: &Configuration,
        request: &DynamicsRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<DynamicsOutcome> {
        let _ = (start, frames);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper do not run molecular dynamics ({})",
            request.label
        )))
    }

    /// The number of negative curvatures of the Hessian at `system`, the last point of a side
    /// of an IRC that ended with a small gradient, with the projection and the threshold of the
    /// check of a minimum: one means a transition state (a TS–TS link). `None` (the default):
    /// these calculations do not check, and the end is minimized as it is.
    fn negative_curvatures(&self, system: &Configuration) -> MoptResult<Option<usize>> {
        let _ = system;
        Ok(None)
    }

    /// Minimum magnitude (cm^-1) of an imaginary mode that triggers a linked IRC at an optimized endpoint.
    fn irc_end_threshold_cm1(&self) -> f64 {
        DEFAULT_IRC_END_THRESHOLD_CM1
    }

    /// Whether saddle points of order 2 or more are optimized and descended from, and how
    /// (`higher_saddles`); `None` (the default): they are not.
    fn higher_saddles(&self) -> Option<HigherSaddleSettings> {
        None
    }

    /// Whether a possible bifurcation of an IRC is confirmed by minimizations from both sides
    /// of the transverse mode (`confirm_bifurcations`; off by default).
    fn confirm_bifurcations(&self) -> bool {
        false
    }

    /// An optimization from `start` to a stationary point of order `order` (the settings of
    /// the transition-state search at that order; a minimization for order 0), following the
    /// modes `follow` from the start (displacements in the coordinates of the point, one per
    /// order; empty: the lowest modes), and the modes of the Hessian at its end. Not available
    /// by default.
    fn saddle(
        &self,
        start: &Configuration,
        order: usize,
        follow: &[faer::Col<f64>],
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<OrderedSaddle> {
        let _ = (start, follow, frames);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper do not optimize saddle points of order {order} \
             ({label})"
        )))
    }

    /// Leave the surplus unstable modes of a higher saddle and refine an order-one point.
    fn mode_kill_saddle(
        &self,
        point: &Configuration,
        modes: &PointModes,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Option<OrderedSaddle>> {
        modes.escape(self, point, label, frames)
    }

    /// A search of a saddle point of order `order` by high-index saddle dynamics (the
    /// shrinking-dimer form, gradients only) from `start` with the directions `directions`
    /// (displacements in the coordinates of the point; the first `order` reflected), handed
    /// over at a loose threshold to the optimization at that order. Not available by default.
    fn hisd(
        &self,
        start: &Configuration,
        order: usize,
        directions: &[faer::Col<f64>],
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<HisdOutcome> {
        let _ = (start, directions, frames);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper do not run high-index saddle dynamics of order \
             {order} ({label})"
        )))
    }

    /// Displace a point along a mode in the optimization coordinates.
    fn displace_mode(
        &self,
        start: &Configuration,
        direction: &faer::Col<f64>,
        sign: f64,
        step: f64,
    ) -> MoptResult<Configuration> {
        start.displaced_along(direction, sign * step)
    }

    /// The modes of the Hessian at `point` (the rigid motions taken out, and with `across` the
    /// direction of the gradient as well, as across a reaction path). `None` (the default):
    /// these calculations give none.
    fn modes(&self, point: &Configuration, across: bool) -> MoptResult<Option<PointModes>> {
        let _ = (point, across);
        Ok(None)
    }

    /// The minimum-energy crossing point of the state of the job and the state `other` (its
    /// index among the states of the run) from `start`: MECP for states of different spin or
    /// charge, MECI for two roots of one spin. Not available by default.
    fn crossing(
        &self,
        start: &Configuration,
        other: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<CrossingSearch> {
        let _ = (start, other, frames);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper do not optimize crossing points ({label})"
        )))
    }

    /// An optimization from `start` with the bias potentials of `request` (the settings of its
    /// stage) towards the crossing points of the state of the job and the state `other`
    /// (`crossing_afir`): with [`CrossingPath::Gp`] on their gradient projection, the bias
    /// potentials in the place of the energies of both states, so that the force acts within
    /// the seam space; with [`CrossingPath::Model`] on the model function whose minima lie
    /// near their crossing points (the seam model function for states of different spin or
    /// charge, the Avoiding model function for two roots of one spin). Not available by
    /// default.
    fn crossing_path(
        &self,
        start: &Configuration,
        other: usize,
        path: CrossingPath,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        let _ = (start, other, path, frames);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper do not optimize on two states ({})",
            request.label
        )))
    }

    /// The meta-IRC from `start` on the state `state` (its index among the states of the run):
    /// the path of steepest descent in mass-weighted coordinates from a point that is no
    /// stationary point of that state (a crossing point), with the settings of the IRC of the
    /// run. Not available by default.
    fn meta_irc_in(
        &self,
        start: &Configuration,
        state: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<MetaIrc> {
        let _ = (start, state, frames);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper do not run a meta-IRC on another state ({label})"
        )))
    }

    /// A minimization from `start` on the state `state` (its index among the states of the
    /// run; the settings of the minimizations, verified as they are). Not available by
    /// default.
    fn minimize_in(
        &self,
        start: &Configuration,
        state: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        let _ = (start, state, frames);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper do not minimize on another state ({label})"
        )))
    }

    /// The energy of `point` alone (no gradient): on the state `on` of the run (its index
    /// among the states; a structure gets the charge and multiplicity of that state), or as
    /// the point is, on the surface of these calculations (`None`). `Ok(None)`: the run was
    /// stopped. What the single points of a whole network are made of (`-single_points`).
    fn energy(
        &self,
        point: &Configuration,
        on: Option<usize>,
        label: &str,
    ) -> MoptResult<Option<f64>> {
        let _ = (point, on);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper give no energy of a point alone ({label})"
        )))
    }

    /// Which path peaks the run saves (`path_peaks`).
    fn path_peaks(&self) -> super::peak::PeakMode {
        super::peak::PeakMode::Off
    }

    /// A minimization from `start` with `constraints` held, with the settings of the biased
    /// optimizations (a point of a relaxed scan); every evaluated structure goes to `frames`.
    /// Not available by default.
    fn held(
        &self,
        start: &Configuration,
        constraints: &[crate::pes::constraint::Constraint],
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        let _ = (start, constraints, frames);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper hold no constraints ({label})"
        )))
    }

    /// The settings of the scans that confirm barrierless dissociations
    /// (`confirm_dissociations`); `None`: no scan.
    fn dissociation_scan(&self) -> Option<super::scan::ScanSettings> {
        None
    }

    /// The lines of the user that wait in `steer.txt` of the run directory (the file is
    /// removed when it is read; [`super::steer`]). None by default.
    fn take_steering(&self) -> MoptResult<Vec<String>> {
        Ok(Vec::new())
    }

    /// What the run was told while it went on is written down (`steer.log`).
    fn log_steering(&self, after_jobs: usize, line: &str, outcome: &str) -> MoptResult<()> {
        let _ = (after_jobs, line, outcome);
        Ok(())
    }

    /// The conformers of the transition state `ts`: the saddle points of order 1 that a
    /// conformer search with its saddle direction held finds, without `ts` itself, the lowest
    /// first (`ts_conformers=on`). `job` is the number of the job (its files go to the
    /// directory of the job). Not available by default.
    fn ts_conformers(
        &self,
        ts: &Configuration,
        job: usize,
        label: &str,
    ) -> MoptResult<TsConformerSearch> {
        let _ = (ts, job);
        Err(MoptError::unsupported(format!(
            "these calculations of the mapper search no conformers of a transition state \
             ({label})"
        )))
    }

    /// Whether the run is to stop now (asked during a job).
    fn stop_requested(&self) -> bool;

    /// Whether the run is to stop before the next job (by default as during a job; a command
    /// may watch further files there, such as MultiOptPy's `stop.txt`).
    fn stop_between_jobs(&self) -> bool {
        self.stop_requested()
    }

    /// The followed electronic states at the ends of the optimizations that ran since the
    /// last call, each with the coordinates of its structure; nothing where the calculator
    /// does not follow its states.
    fn take_followed(&self) -> Vec<(Vec<f64>, FollowedRoots)> {
        Vec::new()
    }

    /// The evaluations of the run so far over all its sessions, where they are counted.
    fn evaluations(&self) -> Option<super::network::EvaluationTotals> {
        None
    }

    /// When the molecules of a structure are a dissociation channel; `None` (the default):
    /// these calculations recognize no channels.
    fn channels(&self) -> Option<ChannelSettings> {
        None
    }

    /// A line for the log of the run.
    fn note(&self, message: &str);

    /// Told before the job runs: it starts from `node` (a calculator that relaxes a part of
    /// the structure itself starts from the whole structure of the node, [`Node::whole`]).
    /// Nothing by default.
    fn start_from(&self, node: &Node) {
        let _ = node;
    }

    /// The whole structure of `point` where the network holds a part of it (the core of an
    /// ONIOM surface, its environment as the calculator relaxed it there); `None` by default.
    fn whole_of(&self, point: &Configuration) -> MoptResult<Option<Configuration>> {
        let _ = point;
        Ok(None)
    }

    /// Fill what the job leaves to the calculations once it has run: the whole structures of
    /// its structures where the network holds a part of them ([`JobOutcome::wholes`]).
    /// Nothing by default.
    fn complete(&self, outcome: &mut JobOutcome) -> MoptResult<()> {
        let _ = outcome;
        Ok(())
    }

    /// A short line for the screen alone (a job starts or ends); by default a line of the log.
    fn progress(&self, message: &str) {
        self.note(message);
    }

    /// A block of lines for the log of the run alone (what a job said), written whole; by
    /// default a line of the log.
    fn record(&self, block: &str) {
        self.note(block);
    }

    /// What these calculations have said so far, line by line, for the block of the job in
    /// the log of the run (nothing where they say it at once).
    fn take_lines(&self) -> Vec<String> {
        Vec::new()
    }
}

/// What a job gives a method besides its own settings.
pub struct JobContext<'a> {
    /// The calculations.
    pub jobs: &'a dyn MapperJobs,
    /// The network as it was when the job started (to skip what is known).
    pub network: &'a Network,
    /// When two structures are the same.
    pub matcher: &'a StructureMatcher,
    /// The trajectory of the job.
    pub frames: &'a mut dyn FrameSink,
    /// The electronic state of the job where the run has several (`states`): what it finds is
    /// matched among the nodes of that state only. `None` for a run of one state.
    pub state: Option<super::network::ElectronicState>,
}

/// A stationary point a job found.
#[derive(Debug, Clone)]
pub struct Stationary {
    /// Signed wavenumbers from the modes counted in the saddle index (cm^-1).
    pub index_wavenumbers: Option<Vec<f64>>,
    /// The structure.
    pub system: Configuration,
    /// Its energy (Hartree).
    pub energy: f64,
    /// Its vibrational wavenumbers, where a Hessian gave them (cm^-1, imaginary ones
    /// negative).
    pub wavenumbers: Option<Vec<f64>>,
    /// Its molecules, for an end of an IRC of two or more: a dissociation channel (the
    /// structure is then the channel) or a complex with the channel of its molecules.
    pub dissociation: Option<Dissociation>,
}

/// What the conformer search of a transition state gave ([`MapperJobs::ts_conformers`]).
#[derive(Debug, Clone, Default)]
pub struct TsConformerSearch {
    /// The run was stopped during the search.
    pub interrupted: bool,
    /// The other saddle points of order 1 it found, the lowest first.
    pub conformers: Vec<Stationary>,
    /// What the job is to warn about (a search that could not run).
    pub warnings: Vec<String>,
}

/// The transition state a side of an IRC ended at instead of a minimum (a TS–TS link: the
/// path ran down a ridge into it, past a valley-ridge inflection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaddleEnd {
    /// The transition state of an earlier reaction of the same outcome (its index in
    /// [`JobOutcome::reactions`]).
    Reaction(usize),
    /// A transition state, or a saddle point of higher order, the network knew when the job
    /// looked (its node).
    Known(usize),
    /// A saddle point of order 2 or more that the job found (its index in
    /// [`JobOutcome::saddles`]; `higher_saddles`).
    Higher(usize),
}

/// Where a descent from a saddle point of higher order along one of its modes ended.
#[derive(Debug, Clone)]
pub enum DescentEnd {
    /// The descent did not converge to a stationary point of the order below.
    Unsettled,
    /// A minimum (from a saddle point of order 2 down to order 1 and on to a minimum, or from
    /// an order 1 point).
    Minimum(Box<Stationary>),
    /// The transition state of a reaction of the same outcome (its index in
    /// [`JobOutcome::reactions`]).
    Reaction(usize),
    /// A transition state or saddle point of higher order the network knew (its node).
    Known(usize),
    /// A saddle point of order 2 or more the job found (its index in [`JobOutcome::saddles`]).
    Higher(usize),
}

/// The descent from a saddle point of order `k` along one of its unstable modes: the ends
/// from both signs of the displacement, each optimized at order `k - 1`.
#[derive(Debug, Clone)]
pub struct Descent {
    /// The mode (0: the lowest curvature).
    pub mode: usize,
    /// Its imaginary wavenumber (cm^-1, negative).
    pub wavenumber: Option<f64>,
    /// The ends from `+` and `-`.
    pub ends: [DescentEnd; 2],
}

/// A saddle point of order 2 or more a job found and descended from (`higher_saddles`).
#[derive(Debug, Clone)]
pub struct FoundSaddle {
    /// The point, optimized at its order.
    pub point: Stationary,
    /// Its order: the negative curvatures of its Hessian.
    pub order: usize,
    /// The descents along its unstable modes (empty where the descent was not run, a saddle
    /// point deeper than `max_saddle_order` allows).
    pub descents: Vec<Descent>,
}

/// The settings of the saddle points of higher order (`higher_saddles`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HigherSaddleSettings {
    /// The highest order that is optimized and descended from (`max_saddle_order`).
    pub max_order: usize,
    /// The displacement along a mode at the start of a descent (amu^(1/2) Bohr in
    /// mass-weighted coordinates; the units of the variables for plain variables).
    pub step: f64,
}

/// The displacement at the start of a descent: that of the first step of the IRC from a
/// transition state.
pub const DEFAULT_DESCENT_STEP: f64 = 0.1;

/// The highest order of a saddle point that the mapper follows by default.
pub const DEFAULT_MAX_SADDLE_ORDER: usize = 2;

/// A mode of the Hessian at a point.
#[derive(Debug, Clone, PartialEq)]
pub struct PointMode {
    /// The curvature (Hartree / (amu Bohr^2), mass-weighted; for plain variables the
    /// eigenvalue of the Hessian).
    pub curvature: f64,
    /// As a wavenumber (cm^-1, negative for a negative curvature; `None` for plain variables).
    pub wavenumber: Option<f64>,
    /// The displacement of a step of length 1 along the mode in mass-weighted coordinates
    /// (Bohr per amu^(1/2) Bohr; the unit eigenvector for plain variables), in the
    /// coordinates of the point.
    pub displacement: faer::Col<f64>,
}

/// The modes of the Hessian at a point, the lowest curvature first, and how many of them are
/// negative (by the threshold of the mode kill).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PointModes {
    /// Unstable modes of the index Hessian, in the optimization coordinates.
    pub unstable: Vec<PointMode>,
    /// The modes (the rigid motions, and the direction of the gradient where it was asked to
    /// be taken out, are not among them).
    pub modes: Vec<PointMode>,
    /// The negative curvatures.
    pub negative: usize,
    /// The wavenumbers of every vibration (cm^-1), where the point is a structure.
    pub wavenumbers: Option<Vec<f64>>,
}

impl PointModes {
    /// Displace the point in its Cartesian or plain coordinates, keeping the lowest mode.
    pub(crate) fn escape<J: MapperJobs + ?Sized>(
        &self,
        jobs: &J,
        point: &Configuration,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Option<OrderedSaddle>> {
        let Some(unstable) = self.unstable.get(..self.negative).filter(|m| m.len() > 1) else {
            return Ok(None);
        };
        let mut x = point.flat();
        if unstable.iter().any(|m| m.displacement.nrows() != x.nrows()) {
            return Err(MoptError::config(
                "mode-kill displacement dimension differs from the point",
            ));
        }
        for mode in &unstable[1..] {
            x += &mode.displacement * DEFAULT_DESCENT_STEP;
        }
        let seed = point.with_flat(&x)?;
        jobs.saddle(
            &seed,
            1,
            std::slice::from_ref(&unstable[0].displacement),
            label,
            frames,
        )
        .map(Some)
    }
}

/// What a search by high-index saddle dynamics ended with ([`MapperJobs::hisd`]).
#[derive(Debug, Clone)]
pub struct HisdOutcome {
    /// The last point.
    pub point: Configuration,
    /// Its energy (Hartree).
    pub energy: f64,
    /// Whether it ended at a saddle point of the order sought, or close enough to hand it to
    /// the optimization at that order (`Converged`, `HandOff`).
    pub settled: bool,
    /// Why it ended, in words.
    pub end: String,
    /// The tracked unstable directions there (displacements in the coordinates of the point,
    /// one per order), to follow in the optimization that refines it.
    pub directions: Vec<faer::Col<f64>>,
    /// Whether the run was stopped during it.
    pub interrupted: bool,
}

/// A stationary point of a given order: the optimization and the modes of its Hessian.
#[derive(Debug, Clone)]
pub struct OrderedSaddle {
    /// The optimization.
    pub optimized: Optimized,
    /// The modes at its end (`None` when it did not converge or was stopped).
    pub modes: Option<PointModes>,
}

/// A reaction a job found: a transition state and the two ends of its IRC.
#[derive(Debug, Clone)]
pub struct FoundReaction {
    /// The transition state.
    pub ts: Stationary,
    /// Its imaginary wavenumber (cm^-1, negative).
    pub imaginary_wavenumber: Option<f64>,
    /// The ends of the IRC, backward end first: minima (optimized), dissociation channels, or
    /// the transition states of [`Self::saddle_ends`]; `None` where the minimization of the
    /// end of that side did not converge (the end is unknown, and the transition state is kept
    /// with the other end: `docs/source/mapper_design.md`, edges).
    pub ends: [Option<Stationary>; 2],
    /// The two sides of the IRC.
    pub branches: Vec<IrcBranch>,
    /// Which ends are transition states, and which (`None`: a minimum or a channel).
    pub saddle_ends: [Option<SaddleEnd>; 2],
    /// The minima found by confirming a possible bifurcation of a side (`confirm_bifurcations`):
    /// the side (`0`: the first end) and the minima from both signs of the displacement.
    pub bifurcation_ends: Vec<(usize, Vec<Stationary>)>,
}

/// What a job ended with.
#[derive(Debug, Clone, Default)]
pub struct JobOutcome {
    /// The run was stopped during the job: nothing of it is kept, and a restart runs it again
    /// (its evaluations come back from the journal).
    pub interrupted: bool,
    /// The reactions it found.
    pub reactions: Vec<FoundReaction>,
    /// The saddle points of order 2 or more it found (`higher_saddles`).
    pub saddles: Vec<FoundSaddle>,
    /// The crossing points between two states it found (`states`).
    pub crossings: Vec<FoundCrossing>,
    /// The path peaks it keeps (`path_peaks`): records beside the network.
    pub peaks: Vec<super::peak::FoundPeak>,
    /// What went wrong without ending the job (for the warnings of the network).
    pub warnings: Vec<String>,
    /// The transition states and saddle points of higher order of the network that the job
    /// converged to again (their nodes): nothing new, but found.
    pub known: Vec<usize>,
    /// The followed electronic states at the ends of the optimizations of the job
    /// ([`MapperJobs::take_followed`]): the coordinates of each (flat, as the structure has
    /// them) and its states. A node of the job gets those of its structure.
    pub followed: Vec<(Vec<f64>, FollowedRoots)>,
    /// The whole structures of the structures of the job where the network holds a part of
    /// them (the core of an ONIOM surface; [`MapperJobs::complete`]): each structure with its
    /// whole structure.
    pub wholes: Vec<(Configuration, Configuration)>,
}

impl JobOutcome {
    /// The whole structure of the structure `point` of the job ([`Self::wholes`]).
    pub fn whole_of(&self, point: &Configuration) -> Option<&Configuration> {
        let x: Vec<f64> = point.flat().iter().copied().collect();
        self.wholes
            .iter()
            .find(|(p, _)| p.flat().iter().copied().eq(x.iter().copied()))
            .map(|(_, w)| w)
    }
}

/// The surface of an AFIR path towards the crossing points of two states
/// ([`MapperJobs::crossing_path`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingPath {
    /// The gradient projection of the two states with the force in the place of both energies.
    Gp,
    /// The model function of the two states (seam or Avoiding).
    Model,
}

/// A meta-IRC ([`MapperJobs::meta_irc_in`]).
#[derive(Debug, Clone)]
pub struct MetaIrc {
    /// Whether the run was stopped during it.
    pub interrupted: bool,
    /// Its path (`None` where the start gave none, e.g. a stationary point).
    pub branch: Option<IrcBranch>,
}

/// The search of a crossing point ([`MapperJobs::crossing`]).
#[derive(Debug, Clone)]
pub struct CrossingSearch {
    /// The last point (with the charge and multiplicity of the state of the job).
    pub point: Configuration,
    /// The mean of the energies of the two states there (Hartree).
    pub energy: f64,
    /// The energy of the first state minus that of the second there (Hartree).
    pub gap: f64,
    /// The norm of the difference of the gradients of the two states there (Hartree/Bohr).
    pub gradient_difference: Option<f64>,
    /// Whether it converged (the forces of the projected gradient and the gap).
    pub converged: bool,
    /// Whether the optimization itself converged (its forces): with `converged` false, its
    /// end is a stationary point of the search whose two states are further apart than a
    /// crossing point allows.
    pub stationary: bool,
    /// The largest gap of a crossing point (Hartree) that `gap` was held against.
    pub gap_limit: f64,
    /// The search was ended where the two states run parallel (`crossing_parallel_stop`):
    /// for this many iterations the gap was closed and the difference of the two gradients
    /// below the threshold. The end is a region where the states are degenerate, no point.
    pub parallel: Option<usize>,
    /// Whether the run was stopped during it.
    pub interrupted: bool,
    /// A conical intersection whose check of the branching plane still found it not stationary
    /// on the seam after its last correction (`Termination::SeamCheckFailed`): no crossing
    /// point.
    pub seam_check_failed: bool,
    /// The check to second order within the seam of an accepted point (`None` where the
    /// point is not accepted).
    pub seam: Option<super::network::SeamCheck>,
}

/// A crossing point a job found, with the minima its descents reached.
#[derive(Debug, Clone)]
pub struct FoundCrossing {
    /// The crossing point (its energy the mean of the two states).
    pub point: Stationary,
    /// MECP or MECI.
    pub kind: super::network::CrossingKind,
    /// The two states, their indices among the states of the run, in that order.
    pub states: [usize; 2],
    /// The energy of the first state of `states` minus that of the second at the point
    /// (Hartree).
    pub gap: f64,
    /// The norm of the difference of the two gradients there (Hartree/Bohr).
    pub gradient_difference: Option<f64>,
    /// The check to second order within the seam.
    pub seam: Option<super::network::SeamCheck>,
    /// The minima of the descents on each of the two states (`None` where it did not
    /// converge), in the order of `states`; each structure has the charge and multiplicity of
    /// its state.
    pub ends: [Option<Stationary>; 2],
    /// The meta-IRC from the crossing point on each of the two states, in the order of
    /// `states` (`None` where there is none).
    pub paths: [Option<IrcBranch>; 2],
    /// The points of the minimization of the end of each meta-IRC, in the order of `states`
    /// (from the end of the path to the minimum; empty where it did not converge).
    pub settles: [Vec<PathPoint>; 2],
}

impl JobOutcome {
    /// The outcome of a job that was stopped.
    pub fn stopped() -> Self {
        Self {
            interrupted: true,
            ..Self::default()
        }
    }
}

/// A way of exploring a node.
pub trait ExplorationMethod: Send + Sync {
    /// Its name in the registry.
    fn name(&self) -> &str;

    /// The tasks of `node`, in the order they are run. The list must depend only on the node
    /// and the settings (a restart makes it again and skips the tasks that were done), unless
    /// [`Self::tasks_follow_network`] says otherwise.
    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>>;

    /// Whether the tasks of a node follow the network as it grows (the gaps of the missing
    /// connections): the driver then asks for them before every choice instead of keeping
    /// the first list. They must still follow from the network alone, so that a restart
    /// gives the same jobs.
    fn tasks_follow_network(&self) -> bool {
        false
    }

    /// Run `job` (one of the tasks of its node).
    fn run(&self, job: &Job, context: &mut JobContext<'_>) -> MoptResult<JobOutcome>;
}

/// Settings a method is made with: the AFIR settings of the run and the `method_KEY=VALUE`
/// settings of `-map_opt` (without the prefix), for methods registered by users.
#[derive(Debug, Clone, Default)]
pub struct MethodSettings {
    /// The AFIR pushes of the built-in method.
    pub afir: super::afir::AfirSettings,
    /// Transition-state guesses of a job that are optimized (`top_n_candidates`).
    pub top_n: usize,
    /// Further settings, by key.
    pub extra: BTreeMap<String, String>,
    /// The `step2_settings` of the settings file as compact JSON (`nebmain`'s settings: the
    /// relaxation of the AFIR paths of the built-in method).
    pub step2: Option<String>,
}

/// Makes a method from the settings of a run.
pub type MethodFactory =
    Arc<dyn Fn(&MethodSettings) -> MoptResult<Box<dyn ExplorationMethod>> + Send + Sync>;

/// Makes the method of a name of a prefix (the whole name, as it was given) from the
/// settings of a run.
pub type PrefixFactory =
    Arc<dyn Fn(&str, &MethodSettings) -> MoptResult<Box<dyn ExplorationMethod>> + Send + Sync>;

/// Methods by name, and by the prefix of a name (`python:` for the methods of Python classes).
pub struct MethodRegistry {
    entries: Vec<(String, MethodFactory)>,
    prefixes: Vec<(String, PrefixFactory)>,
}

impl MethodRegistry {
    /// The built-in methods: `afir_ts` ([`super::afir::AfirTs`]), `addf`
    /// ([`super::addf::AddfMap`]), `nanoreactor` ([`super::nanoreactor::Nanoreactor`]), and
    /// `connect` ([`super::connect::Connect`], the missing connections to a goal).
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
            prefixes: Vec::new(),
        };
        r.register(
            super::afir::AfirTs::NAME,
            Arc::new(|s: &MethodSettings| {
                let relax = super::afir::PathRelaxation::parse(&s.extra, s.step2.as_deref())?;
                let step1 = super::afir::Step1::parse(&s.extra)?;
                Ok(Box::new(
                    super::afir::AfirTs::new(s.afir.clone(), s.top_n)?
                        .with_relaxation(relax)
                        .with_step1(step1),
                ) as Box<dyn ExplorationMethod>)
            }),
        );
        r.register(
            super::addf::AddfMap::NAME,
            Arc::new(|s: &MethodSettings| {
                Ok(Box::new(super::addf::AddfMap::new(&s.extra)?) as Box<dyn ExplorationMethod>)
            }),
        );
        r.register(
            super::connect::Connect::NAME,
            Arc::new(|s: &MethodSettings| {
                let settings = super::connect::ConnectSettings::parse(&s.extra)?;
                Ok(Box::new(super::connect::Connect::new(settings)?) as Box<dyn ExplorationMethod>)
            }),
        );
        r.register(
            super::upward::Upward::NAME,
            Arc::new(|s: &MethodSettings| {
                Ok(Box::new(super::upward::Upward::new(
                    super::upward::UpwardSettings::parse(&s.extra)?,
                )) as Box<dyn ExplorationMethod>)
            }),
        );
        r.register(
            super::nanoreactor::Nanoreactor::NAME,
            Arc::new(|s: &MethodSettings| {
                let settings =
                    super::nanoreactor::NanoreactorSettings::parse(&s.extra, s.afir.seed)?;
                Ok(Box::new(super::nanoreactor::Nanoreactor::new(settings)?)
                    as Box<dyn ExplorationMethod>)
            }),
        );
        r
    }

    /// Register (or replace) a method.
    pub fn register(&mut self, name: &str, factory: MethodFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// Register (or replace) the methods of every name that starts with `prefix` (as
    /// given, case and all): `factory` makes the method of the whole name. The names of
    /// [`Self::register`] go first.
    pub fn register_prefix(&mut self, prefix: &str, factory: PrefixFactory) {
        self.prefixes.retain(|(p, _)| p != prefix);
        self.prefixes.push((prefix.to_string(), factory));
    }

    /// The registered names, and the prefixes as `PREFIX...`.
    pub fn names(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|(k, _)| k.clone())
            .chain(self.prefixes.iter().map(|(p, _)| format!("{p}...")))
            .collect()
    }

    /// The method `name` with `settings`.
    pub fn create(
        &self,
        name: &str,
        settings: &MethodSettings,
    ) -> MoptResult<Box<dyn ExplorationMethod>> {
        let key = name.to_ascii_lowercase();
        if let Some((_, factory)) = self.entries.iter().find(|(k, _)| *k == key) {
            return factory(settings);
        }
        if let Some((_, factory)) = self
            .prefixes
            .iter()
            .find(|(p, _)| name.starts_with(p.as_str()))
        {
            return factory(name, settings);
        }
        let python = if name.starts_with("python:") {
            "; a method of a Python class (python:MODULE:CLASS) runs from the Python package, \
             whose run_mapper knows them"
        } else {
            ""
        };
        Err(MoptError::parse(format!(
            "unknown exploration method '{name}' of the mapper (available: {}){python}",
            self.names().join(", ")
        )))
    }
}

impl Default for MethodRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}
