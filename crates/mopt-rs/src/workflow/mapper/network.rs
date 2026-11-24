//! The reaction network of the mapper: its nodes (minima, transition states, crossing points
//! of two electronic states) and its edges (a transition state between the two minima of its
//! IRC, a crossing point between the minima on its two sides), with what each was computed
//! from.
//!
//! Every node records the level it was computed at ([`Level`]: exploration or refinement,
//! never mixed), its energy, its frequencies where they are known, its electronic state, and
//! where it came from (the input structure its lineage starts at, the job that found it, the
//! node that job started from). Edges keep the raw quantities of a step (barrier heights from
//! both ends, the imaginary frequency, the energies along the IRC, the gradient difference at a
//! crossing), not rate constants: rates are computed afterwards by a chosen kinetic model.
//!
//! Duplicates are found with [`crate::workflow::explore::identity::StructureMatcher`]: equal
//! elements and bond graphs, an energy within the tolerance, and an RMSD after the best matching
//! of equivalent atoms within the tolerance (minimum-image distances in a periodic cell).
//!
//! MultiOptPy: `Wrapper/mapper.py:EQNode`, `TSEdge`, `NetworkGraph` (minima and edges with the
//!  transition-state structure, without levels, states, or IRC data).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::configuration::Configuration;
use crate::domain::{Element, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::identity::{BondGraph, StructureMatcher};

/// Version of the layout of the network file (`reaction_network.json`).
pub const NETWORK_FORMAT_VERSION: u64 = 1;

/// The level a node or an edge was computed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Level 1: the exploration (the surface of the search).
    Exploration,
    /// Level 2: a refinement of what level 1 found on another surface.
    Refinement,
}

impl Level {
    /// 1 or 2.
    pub fn number(self) -> u64 {
        match self {
            Self::Exploration => 1,
            Self::Refinement => 2,
        }
    }

    /// The level of the number `n`.
    pub fn from_number(n: u64) -> MoptResult<Self> {
        match n {
            1 => Ok(Self::Exploration),
            2 => Ok(Self::Refinement),
            other => Err(MoptError::parse(format!(
                "a level of the network is 1 or 2, not {other}"
            ))),
        }
    }
}

/// An electronic state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElectronicState {
    /// Total charge.
    pub charge: i32,
    /// Spin multiplicity `2S + 1`.
    pub multiplicity: u32,
    /// Excited state (0: the ground state of the multiplicity).
    pub excited_state: usize,
}

impl ElectronicState {
    /// The state of `system` with the excited state `excited_state`.
    pub fn of(system: &MolecularSystem, excited_state: usize) -> Self {
        Self {
            charge: system.spin_charge.charge,
            multiplicity: system.spin_charge.multiplicity,
            excited_state,
        }
    }

    /// The state of the point `point`: that of its structure, or for plain variables the one
    /// surface of the objective (charge 0, multiplicity 1, `excited_state` 0).
    pub fn of_point(point: &Configuration, excited_state: usize) -> Self {
        match point.structure() {
            Some(s) => Self::of(s, excited_state),
            None => Self {
                charge: 0,
                multiplicity: 1,
                excited_state: 0,
            },
        }
    }
}

/// What kind of crossing point a node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingKind {
    /// Minimum-energy crossing point of two states of different spin.
    Mecp,
    /// Minimum-energy conical intersection of two states of the same spin.
    Meci,
}

impl CrossingKind {
    /// `MECP` or `MECI`.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Mecp => "MECP",
            Self::Meci => "MECI",
        }
    }
}

/// What a node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// A local minimum (EQ).
    Minimum,
    /// A first-order saddle point (TS).
    TransitionState,
    /// A crossing point of two electronic states.
    Crossing {
        /// MECP or MECI.
        kind: CrossingKind,
        /// The two states, in the order of the states of the run (`states`).
        states: [ElectronicState; 2],
    },
    /// A dissociation channel (DC): fragments separated beyond the interaction range
    /// ([`super::channels`]).
    Dissociated,
    /// A saddle point of order 2 or more (`2SP`, `3SP`, ...; `higher_saddles`).
    HigherSaddle {
        /// Its order (2 to [`MAX_TAGGED_ORDER`]).
        order: u8,
    },
}

/// The highest order of a saddle point with a tag of its own (`9SP`).
pub const MAX_TAGGED_ORDER: u8 = 9;

/// The tags of the saddle points of higher order, by order.
const SADDLE_TAGS: [&str; 10] = [
    "EQ", "TS", "2SP", "3SP", "4SP", "5SP", "6SP", "7SP", "8SP", "9SP",
];

impl NodeKind {
    /// `EQ`, `TS`, `MECP`, `MECI`, `DC`, or `2SP` to `9SP`.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Minimum => "EQ",
            Self::TransitionState => "TS",
            Self::Crossing { kind, .. } => kind.tag(),
            Self::Dissociated => "DC",
            Self::HigherSaddle { order } => {
                SADDLE_TAGS[usize::from((*order).min(MAX_TAGGED_ORDER))]
            }
        }
    }

    /// The kind of a saddle point of `order` (0: a minimum, 1: a transition state; orders above
    /// [`MAX_TAGGED_ORDER`] are refused).
    pub fn of_order(order: usize) -> MoptResult<Self> {
        match order {
            0 => Ok(Self::Minimum),
            1 => Ok(Self::TransitionState),
            k if k <= usize::from(MAX_TAGGED_ORDER) => Ok(Self::HigherSaddle { order: k as u8 }),
            k => Err(MoptError::unsupported(format!(
                "a saddle point of order {k}: the network keeps orders up to {MAX_TAGGED_ORDER}"
            ))),
        }
    }

    /// The kind of the tag `2SP` to `9SP`, `None` for other tags.
    pub fn of_saddle_tag(tag: &str) -> Option<Self> {
        (2..=usize::from(MAX_TAGGED_ORDER))
            .find(|&k| SADDLE_TAGS[k] == tag)
            .map(|k| Self::HigherSaddle { order: k as u8 })
    }

    /// Whether it is a saddle point of order 1 or more (a transition state, or `kSP`).
    pub fn is_saddle(&self) -> bool {
        matches!(self, Self::TransitionState | Self::HigherSaddle { .. })
    }

    /// Whether `other` is the same kind (crossings of the same kind and states).
    pub fn same_kind(&self, other: &Self) -> bool {
        self == other
    }
}

/// Where a node came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Origin {
    /// The input structure its lineage starts at (index into [`Network::inputs`]).
    pub input: Option<usize>,
    /// The job that found it (`None`: an input structure).
    pub job: Option<usize>,
    /// The node that job started from.
    pub from_node: Option<usize>,
}

/// Another environment of the core of a node (an ONIOM network): its energy and the job that
/// found it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentConformer {
    /// Its energy (Hartree) on the surface of the node.
    pub energy: f64,
    /// The job that found it.
    pub job: Option<usize>,
}

/// A node of the network.
#[derive(Debug, Clone)]
pub struct Node {
    /// Its number (index into [`Network::nodes`]).
    pub id: usize,
    /// Its number among the nodes of its kind, from 0 (EQ0, EQ1, ... and TS0, TS1, ...
    /// separately), set when it is added to a network.
    pub number: usize,
    /// What it is.
    pub kind: NodeKind,
    /// The level it was computed at.
    pub level: Level,
    /// The point: a structure (Bohr), or the values of the variables of an objective.
    pub system: Configuration,
    /// Its energy (Hartree) on the surface of its level (with the static bias potentials of
    /// the run, where it has any).
    pub energy: f64,
    /// The energy of the calculator alone at its structure (Hartree), where the surface of the
    /// run has static bias potentials ([`Network::bare`]); `None` otherwise.
    pub bare_energy: Option<f64>,
    /// Its Gibbs free energy (Hartree), where a vibrational analysis gave one.
    pub free_energy: Option<f64>,
    /// Its vibrational wavenumbers (cm^-1, imaginary ones negative), where they are known.
    pub frequencies: Option<Vec<f64>>,
    /// Imaginary modes of the Hessian used for classification. Cell modes use effective masses.
    pub index_wavenumbers: Option<Vec<f64>>,
    /// The electronic state of its surface.
    pub state: ElectronicState,
    /// Where it came from.
    pub origin: Origin,
    /// Why it is not explored (`None`: it is), e.g. given by the user.
    pub excluded: Option<String>,
    /// The fragments of a dissociation channel (empty for the other kinds): which atoms of
    /// `system` make each, and its record in [`Network::fragments`].
    pub parts: Vec<super::channels::ChannelPart>,
    /// The followed electronic states at its structure
    /// ([`super::method::FollowedRoots`]): the jobs from it start with them. Empty where the
    /// calculator does not follow its states, or did not tell them for this structure (the
    /// states are then counted by their energies here).
    pub roots: super::method::FollowedRoots,
    /// The whole structure where `system` is a part of it (the core of an ONIOM surface, with
    /// the environment relaxed at it, in the atoms of [`Network::whole_elements`]); `None`
    /// where `system` is the whole structure.
    pub whole: Option<Configuration>,
    /// The other environments of the same `system` that later jobs found (an ONIOM network:
    /// the nodes are told apart by their cores, `docs/source/mapper_design.md`), with their
    /// energies; the node itself is the one of the lowest energy.
    pub environments: Vec<EnvironmentConformer>,
    /// Its bond graph, computed when it is first compared.
    graph: OnceLock<BondGraph>,
    /// Its SMILES string, written when it is first asked for.
    smiles: OnceLock<String>,
}

impl Node {
    /// A node of `kind` at `level`.
    pub fn new(
        kind: NodeKind,
        level: Level,
        system: Configuration,
        energy: f64,
        state: ElectronicState,
        origin: Origin,
    ) -> Self {
        Self {
            id: 0,
            number: 0,
            kind,
            level,
            system,
            energy,
            bare_energy: None,
            free_energy: None,
            frequencies: None,
            index_wavenumbers: None,
            state,
            origin,
            excluded: None,
            parts: Vec::new(),
            roots: Vec::new(),
            whole: None,
            environments: Vec::new(),
            graph: OnceLock::new(),
            smiles: OnceLock::new(),
        }
    }

    /// The SMILES string of a minimum or a dissociation channel
    /// ([`crate::workflow::explore::smiles`], with the bonds of `bond_scale`): the name of
    /// its chemical species, which its conformers share. `None` for the other kinds (the
    /// bonds of a transition state or a crossing point are between two species) and for a
    /// point of plain variables.
    pub fn smiles(&self, bond_scale: f64) -> Option<&str> {
        if !matches!(self.kind, NodeKind::Minimum | NodeKind::Dissociated) {
            return None;
        }
        let sys = self.system.structure()?;
        Some(
            self.smiles
                .get_or_init(|| crate::workflow::explore::smiles::smiles(sys, bond_scale)),
        )
    }

    /// `EQ3`, `TS12`, ... (numbered within its kind).
    pub fn label(&self) -> String {
        format!("{}{}", self.kind.tag(), self.number)
    }

    /// The bond graph of the structure at the bond scale of `matcher` (`None` for a point of
    /// plain variables).
    pub fn graph(&self, matcher: &StructureMatcher) -> Option<&BondGraph> {
        let sys = self.system.structure()?;
        Some(
            self.graph
                .get_or_init(|| BondGraph::of(sys, matcher.bond_scale)),
        )
    }
}

/// One side of an IRC as data: the arc length and the energy of every point, in the order of
/// the integration (from the transition state outwards).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IrcSide {
    /// `forward` or `backward` (MultiOptPy's names of the sides).
    pub name: String,
    /// Mass-weighted arc length of every point from the first point of the side
    /// (amu^(1/2) Bohr).
    pub arc_lengths: Vec<f64>,
    /// Energy of every point (Hartree).
    pub energies: Vec<f64>,
    /// Why the side ended (`converged`, `max_steps`, ...).
    pub end: String,
    /// The lowest curvature across the path at the points where the IRC had an exact Hessian,
    /// in the order of the path: the arc length and the curvature as a wavenumber (cm^-1,
    /// negative for a negative curvature): every `irc_hessian_interval`-th point by default.
    pub transverse: Vec<[f64; 2]>,
}

/// How negative the curvature across the path at an exact Hessian of an IRC must be before
/// the side is said to possibly bifurcate (cm^-1, `bifurcation_threshold_cm1`). Points of an
/// IRC are not stationary across the path, so soft modes can have a small negative
/// curvature there without any branching; see `docs/source/mapper.md` for the measurement.
pub const DEFAULT_BIFURCATION_THRESHOLD_CM1: f64 = 50.0;

/// A place along a side of an IRC where the curvature across the path was negative at an
/// exact Hessian: the path may bifurcate there, a valley-ridge inflection lying between the
/// two arc lengths. A possibility, not a branch the network has confirmed.
#[derive(Debug, Clone, PartialEq)]
pub struct PossibleBifurcation {
    /// The side (`forward`, `backward`).
    pub side: String,
    /// Arc length of the last point before it with an exact Hessian and no negative curvature
    /// across the path (amu^(1/2) Bohr; `None`: no such point on the side, the inflection may
    /// lie anywhere from the transition state on).
    pub after: Option<f64>,
    /// Arc length of the first point with negative curvature across the path.
    pub at: f64,
    /// That curvature as a wavenumber (cm^-1, negative).
    pub wavenumber: f64,
    /// The minima reached from both sides of the transverse mode there, other than the end of
    /// the side (`confirm_bifurcations`; empty: not confirmed, or nothing else was found).
    pub confirmed_ends: Vec<usize>,
    /// The arc length at which the curvature across the path changes sign: linear in the
    /// curvature between the last point with an exact Hessian where it was not negative and
    /// the first where it was (`None`: no point of the side before it had a curvature that
    /// was not negative). An estimate of where the valley turns into a ridge.
    pub inflection: Option<f64>,
}

/// The descent from a saddle point of order k ≥ 2 along one of its unstable modes: the two
/// stationary points of order k - 1 (or minima, from k = 1 below) that both signs of the
/// displacement led to (`higher_saddles`).
#[derive(Debug, Clone, PartialEq)]
pub struct SaddleEdge {
    /// Its number (index into [`Network::saddle_edges`]).
    pub id: usize,
    /// The level it was computed at.
    pub level: Level,
    /// The saddle point of higher order.
    pub saddle: usize,
    /// The mode (0: the lowest curvature).
    pub mode: usize,
    /// Its imaginary wavenumber (cm^-1, negative).
    pub wavenumber: Option<f64>,
    /// The ends from `+` and `-` (`None`: the descent did not settle).
    pub ends: [Option<usize>; 2],
    /// The job that found it.
    pub job: Option<usize>,
}

/// The IRC of a transition state as data: both sides, and the file with the structures of the
/// whole path (the backward side reversed, the transition state, the forward side), which a
/// refinement takes as its first chain.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IrcPath {
    /// The sides, `backward` first.
    pub sides: Vec<IrcSide>,
    /// The structures of the path, relative to the run directory.
    pub file: Option<String>,
    /// Where a side may bifurcate (from the exact Hessians along it; possibilities only).
    pub possible_bifurcations: Vec<PossibleBifurcation>,
}

/// A transition state between the two minima its IRC ends at.
#[derive(Debug, Clone)]
pub struct TsEdge {
    /// Its number (index into [`Network::ts_edges`]).
    pub id: usize,
    /// The level it was computed at.
    pub level: Level,
    /// The transition-state node.
    pub ts: usize,
    /// The minima at the backward and the forward end of the IRC (the same twice for a
    /// degenerate rearrangement).
    pub ends: [usize; 2],
    /// `E(TS) - E(end)` for both ends (Hartree).
    pub barriers: [f64; 2],
    /// The imaginary wavenumber of the transition state (cm^-1, negative).
    pub imaginary_wavenumber: Option<f64>,
    /// The IRC.
    pub irc: IrcPath,
    /// The job that found it.
    pub job: Option<usize>,
}

/// A transition state whose IRC has an unresolved endpoint (`docs/source/mapper_design.md`,
/// edges). The transition state is a
/// node of the network; the edge is kept apart from [`Network::ts_edges`], so that the
/// kinetics and everything that needs both ends leave it out.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenTsEdge {
    /// Its number (index into [`Network::open_ts_edges`]).
    pub id: usize,
    /// The level it was computed at.
    pub level: Level,
    /// The transition-state node.
    pub ts: usize,
    /// The ends at the backward and the forward side of the IRC, where they are known.
    pub ends: [Option<usize>; 2],
    /// `E(TS) - E(end)` where the end is known (Hartree).
    pub barriers: [Option<f64>; 2],
    /// The imaginary wavenumber of the transition state (cm^-1, negative).
    pub imaginary_wavenumber: Option<f64>,
    /// The IRC.
    pub irc: IrcPath,
    /// The job that found it.
    pub job: Option<usize>,
}

/// A crossing point between the minima on its two sides (each where it is known).
#[derive(Debug, Clone, PartialEq)]
pub struct CrossingEdge {
    /// Its number (index into [`Network::crossing_edges`]).
    pub id: usize,
    /// The level it was computed at.
    pub level: Level,
    /// The crossing-point node.
    pub crossing: usize,
    /// The minima on the sides of the two states.
    pub ends: [Option<usize>; 2],
    /// `E(crossing) - E(end)` for both ends where they are known (Hartree).
    pub barriers: [Option<f64>; 2],
    /// Norm of the difference of the gradients of the two states at the crossing
    /// (Hartree/Bohr), as a Landau-Zener model needs it.
    pub gradient_difference: Option<f64>,
    /// The energy of the first state of the crossing point minus that of the second, in the
    /// order of the states of the run, at the end of its search (Hartree; `None` in a network
    /// written before it was kept).
    pub gap: Option<f64>,
    /// The check to second order within the seam ([`SeamCheck`]; `None` in a network written
    /// before it was made).
    pub seam: Option<SeamCheck>,
    /// The meta-IRC from the crossing point down each of its two states: a side per state
    /// that has one, named after its surface (`PES1`, ...), each on its own state, and the
    /// file of their structures.
    pub irc: IrcPath,
    /// The job that found it.
    pub job: Option<usize>,
}

/// The check to second order within the seam of a crossing point (`crossing_seam_check`):
/// the vibrational analysis within the seam (`vib::seam_analysis`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeamCheck {
    /// The number of imaginary modes within the seam by the one criterion
    /// (`opt::mode_kill::IMAGINARY_CURVATURE`): 0, a minimum on the seam (a minimum-energy
    /// crossing point); more, a saddle point on the seam.
    Checked(usize),
    /// Not checked, and why (the check switched off, a variable cell, an error of the
    /// analysis).
    NotChecked(String),
}

impl SeamCheck {
    /// Whether the point is checked to be a saddle point on the seam.
    pub fn is_saddle(&self) -> bool {
        matches!(self, Self::Checked(n) if *n > 0)
    }

    /// What the check says, for tables and pages: `minimum on the seam`, `saddle point on the
    /// seam (N imaginary mode(s))`, or `not checked (WHY)`.
    pub fn describe(&self) -> String {
        match self {
            Self::Checked(0) => "minimum on the seam".to_string(),
            Self::Checked(n) => format!("saddle point on the seam ({n} imaginary mode(s))"),
            Self::NotChecked(why) => format!("not checked ({why})"),
        }
    }
}

/// What the check on the seam of the crossing edges says of a crossing point (`None`: no edge
/// with a check): a saddle point on the seam if any of its edges says so.
pub fn seam_of(network: &Network, crossing: usize) -> Option<&SeamCheck> {
    let mut found = None;
    for e in network
        .crossing_edges
        .iter()
        .filter(|e| e.crossing == crossing)
    {
        match &e.seam {
            Some(s) if s.is_saddle() => return Some(s),
            Some(s @ SeamCheck::Checked(_)) => found = Some(s),
            Some(s) if found.is_none() => found = Some(s),
            _ => {}
        }
    }
    found
}

/// An input structure of the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputRecord {
    /// The file as it was given.
    pub file: String,
    /// The frame of the file (from 0).
    pub frame: usize,
    /// The node it was optimized to (`None` while that has not been done).
    pub node: Option<usize>,
    /// It is the goal of a goal-directed search (`goal=FILE`): kept, never explored, and the
    /// run ends when it is joined to a start.
    pub goal: bool,
}

/// How a job ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// It ran to its end.
    Done,
    /// It failed with this error; it is not tried again.
    Failed(String),
}

/// A finished job.
#[derive(Debug, Clone, PartialEq)]
pub struct JobRecord {
    /// Its number (from 0, in the order the jobs ran).
    pub id: usize,
    /// The node it started from.
    pub node: usize,
    /// The method of exploration.
    pub method: String,
    /// The task within the node (unique among the tasks of the node).
    pub task: String,
    /// What the task does, for the log and the report.
    pub description: String,
    /// How it ended.
    pub status: JobStatus,
    /// The nodes it added.
    pub new_nodes: Vec<usize>,
    /// The transition-state edges it added.
    pub new_edges: Vec<usize>,
    /// Whether it found a minimum that was not known.
    pub found_new: bool,
    /// Whether it found one below the lowest known before it.
    pub found_lower: bool,
    /// The trajectory of the job, relative to the run directory.
    pub trajectory: Option<String>,
}

/// Something the network warns about (a transition-state search that did not converge, a
/// stationary point with the wrong number of imaginary frequencies, an IRC end that did not
/// become a minimum).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkWarning {
    /// The job it is from.
    pub job: Option<usize>,
    /// What happened.
    pub message: String,
}

/// Evaluations of the run over all its sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EvaluationTotals {
    /// Evaluations of the energy alone.
    pub energies: usize,
    /// Evaluations with the gradient.
    pub gradients: usize,
    /// Evaluations with the exact Hessian.
    pub hessians: usize,
    /// Relaxations of an ONIOM environment (microiterations).
    pub micro_relaxations: usize,
    /// Their inner iterations, all together.
    pub micro_iterations: usize,
}

/// The reaction network.
#[derive(Debug, Clone, Default)]
pub struct Network {
    /// The elements of every structure, in atom order (empty for plain variables).
    pub elements: Vec<Element>,
    /// The elements of the whole system where the structures of the network are a part of it
    /// (the core of an ONIOM surface; [`Node::whole`]).
    pub whole_elements: Option<Vec<Element>>,
    /// The names of the variables when the points are those of an objective of plain
    /// variables (`None`: structures).
    pub variables: Option<Vec<String>>,
    /// The electronic states of the run, PES1, PES2, ... in this order (`states`; empty for a
    /// run of one state): every minimum, transition state and saddle point lies on one of
    /// them, a crossing point on two.
    pub states: Vec<ElectronicState>,
    /// The input structures.
    pub inputs: Vec<InputRecord>,
    /// The nodes.
    pub nodes: Vec<Node>,
    /// The transition-state edges (both ends known).
    pub ts_edges: Vec<TsEdge>,
    /// The transition states with an end of their IRC that is not known.
    pub open_ts_edges: Vec<OpenTsEdge>,
    /// The crossing edges.
    pub crossing_edges: Vec<CrossingEdge>,
    /// The barrierless dissociations of complexes into their channels.
    pub dissociation_edges: Vec<super::channels::DissociationEdge>,
    /// The descents from the saddle points of higher order.
    pub saddle_edges: Vec<SaddleEdge>,
    /// The path peaks (`path_peaks`): records beside the network, no nodes or edges of it.
    pub path_peaks: Vec<super::peak::PathPeak>,
    /// The entrance channel of the run (`entrance_structures`).
    pub entrance: Option<super::entrance::Entrance>,
    /// What the user changed while the run went on (`steer.txt`).
    pub steering: super::steer::Steering,
    /// The fragments of the dissociation channels, each optimized on its own once (the
    /// cache of their energies).
    pub fragments: Vec<super::channels::FragmentRecord>,
    /// The finished jobs, in their order.
    pub jobs: Vec<JobRecord>,
    /// What the network warns about.
    pub warnings: Vec<NetworkWarning>,
    /// Evaluations of the earlier sessions of the run.
    pub evaluations: EvaluationTotals,
    /// The energy of the static bias potentials of the surface at a point, where the run has
    /// any: every node added gets its energy without them ([`Node::bare_energy`]).
    pub bare: Option<BiasEnergy>,
    /// The atoms that hard constraints of the run name, which the identity test keeps in their
    /// places (`docs/source/mapper_design.md`).
    pub held: HeldAtoms,
    /// The constraints and static bias potentials of the surface of the run in words (the
    /// network file, the report); empty where it has none.
    pub surface: Vec<String>,
    /// How many nodes of each kind (by tag) there are: the number of the next one.
    numbers: BTreeMap<&'static str, usize>,
    /// The jobs by the node they started from (a cache of [`Self::jobs`]).
    job_index: SharedJobIndex,
}

/// The jobs of a network by the node they started from: the indices of the jobs of every node,
/// for the first `counted` jobs, whose last job (its number and node) was `last`. Brought up to
/// date with the jobs appended since, and made again when the jobs are not those it counted
/// (fewer, or another last one).
#[derive(Debug, Clone, Default)]
struct JobIndex {
    counted: usize,
    last: Option<(usize, usize)>,
    by_node: BTreeMap<usize, Vec<usize>>,
}

impl JobIndex {
    fn update(&mut self, jobs: &[JobRecord]) {
        let same = self.counted <= jobs.len()
            && self.last
                == self
                    .counted
                    .checked_sub(1)
                    .and_then(|k| jobs.get(k))
                    .map(|j| (j.id, j.node));
        if !same {
            *self = Self::default();
        }
        for (k, j) in jobs.iter().enumerate().skip(self.counted) {
            self.by_node.entry(j.node).or_default().push(k);
        }
        self.counted = jobs.len();
        self.last = jobs.last().map(|j| (j.id, j.node));
    }
}

/// [`JobIndex`] behind a lock (the network is read by `&self`); a copy of the network gets a
/// copy of the index.
#[derive(Debug, Default)]
struct SharedJobIndex(std::sync::Mutex<JobIndex>);

impl Clone for SharedJobIndex {
    fn clone(&self) -> Self {
        Self(std::sync::Mutex::new(
            self.0.lock().map(|i| i.clone()).unwrap_or_default(),
        ))
    }
}

/// The atoms of the hard constraints of a run, for the identity of its nodes: every one is
/// matched only with itself, and with `fixed_frame` (fixed atoms, which pin the frame) the
/// structures are compared where they are, without superposition or mirror images.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HeldAtoms {
    /// The atoms (from 0), ascending.
    pub atoms: Vec<usize>,
    /// Whether the structures share a frame.
    pub fixed_frame: bool,
}

/// The energy of the static bias potentials of a surface at a point (Hartree).
#[derive(Clone)]
pub struct BiasEnergy(pub std::sync::Arc<dyn Fn(&Configuration) -> MoptResult<f64> + Send + Sync>);

impl std::fmt::Debug for BiasEnergy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BiasEnergy")
    }
}

impl Network {
    /// An empty network of structures with `elements`.
    pub fn new(elements: Vec<Element>) -> Self {
        Self {
            elements,
            ..Self::default()
        }
    }

    /// An empty network of points of the variables `names` of an objective.
    pub fn of_variables(names: Vec<String>) -> Self {
        Self {
            variables: Some(names),
            ..Self::default()
        }
    }

    /// The node `id`.
    pub fn node(&self, id: usize) -> MoptResult<&Node> {
        self.nodes
            .get(id)
            .ok_or_else(|| MoptError::config(format!("the network has no node {id}")))
    }

    /// Add `node` (its number is set) and return its number.
    pub fn add_node(&mut self, mut node: Node) -> MoptResult<usize> {
        match (&node.system, &self.variables) {
            (Configuration::Structure(s), None) if s.elements == self.elements => {}
            (Configuration::Variables(p), Some(names)) if p.values.nrows() == names.len() => {}
            (Configuration::Structure(_), None) => {
                return Err(MoptError::config(format!(
                    "{}: the structure has other atoms than the network",
                    node.label()
                )))
            }
            _ => {
                return Err(MoptError::config(format!(
                    "{}: the point is not one of the {} of the network",
                    node.label(),
                    match &self.variables {
                        Some(names) => format!("{} variables", names.len()),
                        None => "structures".to_string(),
                    }
                )))
            }
        }
        if let (Some(bias), None) = (&self.bare, node.bare_energy) {
            node.bare_energy = Some(node.energy - (bias.0)(&node.system)?);
        }
        node.id = self.nodes.len();
        node.number = self.next_number(&node.kind);
        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// The number of the next node of `kind` (and count it).
    fn next_number(&mut self, kind: &NodeKind) -> usize {
        let count = self.numbers.entry(kind.tag()).or_insert(0);
        *count += 1;
        *count - 1
    }

    /// Put back `node` as it was saved (its id and number are kept; a node saved without its
    /// number, by an earlier version, gets the next one of its kind).
    pub fn restore_node(&mut self, mut node: Node, number: Option<usize>) {
        node.id = self.nodes.len();
        match number {
            Some(n) => {
                let count = self.numbers.entry(node.kind.tag()).or_insert(0);
                *count = (*count).max(n + 1);
                node.number = n;
            }
            None => node.number = self.next_number(&node.kind),
        }
        self.nodes.push(node);
    }

    /// Add `edge` (its number is set) and return its number.
    pub fn add_ts_edge(&mut self, mut edge: TsEdge) -> MoptResult<usize> {
        for id in [edge.ts, edge.ends[0], edge.ends[1]] {
            self.node(id)?;
        }
        edge.id = self.ts_edges.len();
        self.ts_edges.push(edge);
        Ok(self.ts_edges.len() - 1)
    }

    /// Add `edge` (its number is set) and return its number.
    pub fn add_open_ts_edge(&mut self, mut edge: OpenTsEdge) -> MoptResult<usize> {
        self.node(edge.ts)?;
        for id in edge.ends.iter().flatten() {
            self.node(*id)?;
        }
        edge.id = self.open_ts_edges.len();
        self.open_ts_edges.push(edge);
        Ok(self.open_ts_edges.len() - 1)
    }

    /// Add `edge` (its number is set) and return its number.
    pub fn add_crossing_edge(&mut self, mut edge: CrossingEdge) -> MoptResult<usize> {
        self.node(edge.crossing)?;
        for id in edge.ends.iter().flatten() {
            self.node(*id)?;
        }
        edge.id = self.crossing_edges.len();
        self.crossing_edges.push(edge);
        Ok(self.crossing_edges.len() - 1)
    }

    /// The number of crossing points (nodes of either kind of crossing): the count that the
    /// console, the pages and the summaries give (a node can have several crossing edges).
    pub fn crossing_points(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Crossing { .. }))
            .count()
    }

    /// The nodes of `kind`.
    pub fn of_kind<'a>(&'a self, kind: &'a NodeKind) -> impl Iterator<Item = &'a Node> + 'a {
        self.nodes.iter().filter(move |n| n.kind.same_kind(kind))
    }

    /// Add a descent edge of a saddle point of higher order; returns its number.
    pub fn add_saddle_edge(&mut self, mut edge: SaddleEdge) -> MoptResult<usize> {
        self.node(edge.saddle)?;
        for end in edge.ends.iter().flatten() {
            self.node(*end)?;
        }
        edge.id = self.saddle_edges.len();
        self.saddle_edges.push(edge);
        Ok(self.saddle_edges.len() - 1)
    }

    /// The nodes other than saddle points (minima, channels) that the saddle point `ts` leads
    /// to through its own edges of `level`: the ends of its IRC (a transition state), the ends
    /// of its descents (a saddle point of higher order), following further saddle points at
    /// the ends (TS–TS links, a side that ends at a `kSP`, the transition states below a
    /// `kSP`; at most eight deep, each saddle point once), in the order of their numbers.
    pub fn beyond_link(&self, ts: usize, level: Level) -> Vec<usize> {
        let mut out = Vec::new();
        let mut seen = vec![ts];
        let mut todo = vec![(ts, 0_usize)];
        while let Some((t, depth)) = todo.pop() {
            let ends: Vec<usize> = self
                .ts_edges
                .iter()
                .filter(|e| e.level == level && e.ts == t)
                .flat_map(|e| e.ends)
                .chain(
                    self.saddle_edges
                        .iter()
                        .filter(|e| e.level == level && e.saddle == t)
                        .flat_map(|e| e.ends.into_iter().flatten()),
                )
                .collect();
            for end in ends {
                let linked = self.nodes.get(end).is_some_and(|n| n.kind.is_saddle());
                if linked {
                    if depth < 8 && !seen.contains(&end) {
                        seen.push(end);
                        todo.push((end, depth + 1));
                    }
                } else if !out.contains(&end) {
                    out.push(end);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// The pairs of nodes other than transition states that the edge `e` joins: its two ends,
    /// or, where an end is a transition state (a TS–TS link), the other end with every node
    /// beyond it ([`Self::beyond_link`]). Empty where both ends are transition states.
    pub fn joined_by(&self, e: &TsEdge) -> Vec<[usize; 2]> {
        let linked = |id: usize| self.nodes.get(id).is_some_and(|n| n.kind.is_saddle());
        let [a, b] = e.ends;
        match (linked(a), linked(b)) {
            (false, false) => vec![[a, b]],
            (true, true) => Vec::new(),
            (true, false) => self
                .beyond_link(a, e.level)
                .into_iter()
                .filter(|&p| p != b)
                .map(|p| [p, b])
                .collect(),
            (false, true) => self
                .beyond_link(b, e.level)
                .into_iter()
                .filter(|&p| p != a)
                .map(|p| [a, p])
                .collect(),
        }
    }

    /// The minima.
    pub fn minima(&self) -> impl Iterator<Item = &Node> {
        self.of_kind(&NodeKind::Minimum)
    }

    /// The lowest energy of the minima.
    pub fn lowest_minimum(&self) -> Option<f64> {
        self.minima().map(|n| n.energy).reduce(f64::min)
    }

    /// How many jobs started from node `id`.
    pub fn visits(&self, id: usize) -> usize {
        self.jobs_of(id).len()
    }

    /// How many jobs started from every node, by its id (one pass over the jobs).
    pub fn visit_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.nodes.len()];
        for j in &self.jobs {
            if let Some(c) = counts.get_mut(j.node) {
                *c += 1;
            }
        }
        counts
    }

    /// The tasks of node `id` that were run (finished or failed), in the order of the jobs.
    pub fn done_tasks(&self, id: usize) -> Vec<&str> {
        self.jobs_of(id)
            .into_iter()
            .filter_map(|k| self.jobs.get(k).map(|j| j.task.as_str()))
            .collect()
    }

    /// The indices of the jobs (into [`Self::jobs`]) that started from node `id`, ascending:
    /// from the index of the jobs by node, brought up to date with the jobs appended since
    /// ([`JobIndex`]), so that the choice of the next job does not scan every job for every
    /// node.
    fn jobs_of(&self, id: usize) -> Vec<usize> {
        let Ok(mut index) = self.job_index.0.lock() else {
            return self
                .jobs
                .iter()
                .enumerate()
                .filter(|(_, j)| j.node == id)
                .map(|(k, _)| k)
                .collect();
        };
        index.update(&self.jobs);
        index.by_node.get(&id).cloned().unwrap_or_default()
    }

    /// The node of `kind` and `level` that `system` with `energy` is the same as, by `matcher`:
    /// of the nodes within the energy tolerance, nearest energy first, the first one whose bond
    /// graph is the same and whose RMSD after matching is within the tolerance.
    pub fn find_same(
        &self,
        kind: &NodeKind,
        level: Level,
        point: &Configuration,
        energy: f64,
        matcher: &StructureMatcher,
    ) -> MoptResult<Option<usize>> {
        self.find_same_in(kind, level, point, energy, matcher, None)
    }

    /// [`Self::find_same`] among the nodes of the electronic state `state` only (`None`: of any
    /// state, as for a run of one state): with several states two nodes are the same only on
    /// the same surface.
    pub fn find_same_in(
        &self,
        kind: &NodeKind,
        level: Level,
        point: &Configuration,
        energy: f64,
        matcher: &StructureMatcher,
        state: Option<&ElectronicState>,
    ) -> MoptResult<Option<usize>> {
        let on_state = |n: &&Node| state.is_none_or(|s| n.state == *s);
        let tolerance = matcher.energy_tolerance.value();
        let Some(system) = point.structure() else {
            // Plain variables: the nearest node of the kind within the distance and the energy.
            let mut best: Option<(f64, usize)> = None;
            for n in self
                .nodes
                .iter()
                .filter(|n| n.kind.same_kind(kind) && n.level == level)
                .filter(on_state)
                .filter(|n| (n.energy - energy).abs() <= tolerance)
            {
                if let Some(d) = n.system.variable_distance(point) {
                    if d <= matcher.rmsd_tolerance.value() && best.is_none_or(|(b, _)| d < b) {
                        best = Some((d, n.id));
                    }
                }
            }
            return Ok(best.map(|(_, id)| id));
        };
        let graph = BondGraph::of(system, matcher.bond_scale);
        let mut near: Vec<&Node> = self
            .nodes
            .iter()
            .filter(|n| n.kind.same_kind(kind) && n.level == level)
            .filter(on_state)
            .filter(|n| (n.energy - energy).abs() <= tolerance)
            .collect();
        near.sort_by(|a, b| {
            (a.energy - energy)
                .abs()
                .total_cmp(&(b.energy - energy).abs())
        });
        for node in near {
            let (Some(theirs), Some(their_graph)) = (node.system.structure(), node.graph(matcher))
            else {
                continue;
            };
            let found = if self.held.atoms.is_empty() {
                matcher.best_match_with(theirs, their_graph, system, &graph)?
            } else {
                matcher.best_match_holding(
                    theirs,
                    their_graph,
                    system,
                    &graph,
                    &self.held.atoms,
                    self.held.fixed_frame,
                )?
            };
            if found.is_some_and(|m| m.rmsd.value() <= matcher.rmsd_tolerance.value()) {
                return Ok(Some(node.id));
            }
        }
        Ok(None)
    }

    /// Whether the bond graph of node `id` is that of node `reference` up to a renumbering.
    pub fn same_bonds(
        &self,
        id: usize,
        reference: usize,
        matcher: &StructureMatcher,
    ) -> MoptResult<bool> {
        // Points of plain variables have no bonds: their bonds never change.
        Ok(
            match (
                self.node(id)?.graph(matcher),
                self.node(reference)?.graph(matcher),
            ) {
                (Some(a), Some(b)) => a.same_as(b),
                _ => true,
            },
        )
    }

    /// The transition-state edges that end at node `id`.
    pub fn edges_of(&self, id: usize) -> impl Iterator<Item = &TsEdge> {
        self.ts_edges.iter().filter(move |e| e.ends.contains(&id))
    }

    /// The index of the state of node `id` among [`Self::states`] (0 for a run of one state).
    pub fn state_of(&self, id: usize) -> MoptResult<usize> {
        let node = self.node(id)?;
        if self.states.is_empty() {
            return Ok(0);
        }
        self.states
            .iter()
            .position(|s| *s == node.state)
            .ok_or_else(|| {
                MoptError::config(format!(
                    "{} is on a state (charge {}, multiplicity {}, root {}) that is not one of \
                     the states of the run",
                    node.label(),
                    node.state.charge,
                    node.state.multiplicity,
                    node.state.excited_state
                ))
            })
    }

    /// The node of the goal of a goal-directed search (`goal=FILE`), once it is optimized.
    pub fn goal_node(&self) -> Option<usize> {
        self.inputs.iter().find(|i| i.goal).and_then(|i| i.node)
    }

    /// The nodes of the inputs that are not the goal (the starts).
    pub fn start_nodes(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .inputs
            .iter()
            .filter(|i| !i.goal)
            .filter_map(|i| i.node)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The nodes joined to `from` by edges of the network at `level`: transition-state edges
    /// (both ends, a transition state at an end included), crossing edges, and barrierless
    /// dissociations; `from` among them.
    pub fn joined_to(&self, from: usize, level: Level) -> Vec<bool> {
        let n = self.nodes.len();
        let mut links: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut add = |a: usize, b: usize| {
            if a < n && b < n {
                links[a].push(b);
                links[b].push(a);
            }
        };
        for e in self.ts_edges.iter().filter(|e| e.level == level) {
            add(e.ts, e.ends[0]);
            add(e.ts, e.ends[1]);
        }
        for e in self.crossing_edges.iter().filter(|e| e.level == level) {
            for end in e.ends.iter().flatten() {
                add(e.crossing, *end);
            }
        }
        for e in self.dissociation_edges.iter().filter(|e| e.level == level) {
            add(e.minimum, e.channel);
        }
        for e in self.saddle_edges.iter().filter(|e| e.level == level) {
            for end in e.ends.iter().flatten() {
                add(e.saddle, *end);
            }
        }
        let mut seen = vec![false; n];
        if from >= n {
            return seen;
        }
        let mut stack = vec![from];
        seen[from] = true;
        while let Some(a) = stack.pop() {
            for &b in &links[a] {
                if !seen[b] {
                    seen[b] = true;
                    stack.push(b);
                }
            }
        }
        seen
    }

    /// A minimum with the bond graph of the goal (by `matcher`) that is joined to a start by
    /// edges of level 1: the goal itself when it is joined, or another conformer of it
    /// (`None`: none, no goal, or not optimized yet).
    pub fn goal_bonds_joined(&self, matcher: &StructureMatcher) -> Option<usize> {
        let goal = self.goal_node()?;
        if self.goal_joined() == Some(true) {
            return Some(goal);
        }
        let graph = self.node(goal).ok()?.graph(matcher)?;
        let mut reached = vec![false; self.nodes.len()];
        for s in self.start_nodes() {
            for (k, r) in self
                .joined_to(s, Level::Exploration)
                .into_iter()
                .enumerate()
            {
                reached[k] |= r;
            }
            if let Some(r) = reached.get_mut(s) {
                *r = true;
            }
        }
        self.minima()
            .filter(|n| reached.get(n.id).copied().unwrap_or(false))
            .find(|n| n.graph(matcher).is_some_and(|g| g.same_as(graph)))
            .map(|n| n.id)
    }

    /// Whether the goal is joined to a start by edges of level 1 (`None`: no goal, or not
    /// optimized yet).
    pub fn goal_joined(&self) -> Option<bool> {
        let goal = self.goal_node()?;
        let starts = self.start_nodes();
        if starts.is_empty() {
            return None;
        }
        Some(starts.iter().any(|&s| {
            s == goal
                || self
                    .joined_to(s, Level::Exploration)
                    .get(goal)
                    .copied()
                    .unwrap_or(false)
        }))
    }
}

/// The matcher of the mapper: `rmsd` (Angstrom) and `energy` (kcal/mol) as MultiOptPy gives
/// them, mirror images distinct (MultiOptPy's `StructureChecker` never takes an enantiomer for
/// the same structure, and in a network a racemization is a reaction).
pub fn mapper_matcher(rmsd_angstrom: f64, energy_kcal_mol: f64, mirror: bool) -> StructureMatcher {
    StructureMatcher {
        rmsd_tolerance: Quantity::new(rmsd_angstrom * crate::domain::Units::ANGSTROM_TO_BOHR),
        energy_tolerance: Quantity::new(
            energy_kcal_mol * crate::domain::Units::KCAL_MOL_TO_HARTREE,
        ),
        mirror_images: mirror,
        ..StructureMatcher::default()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, SpinCharge};

    fn argon(x: f64) -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::from_atomic_number(18).unwrap(); 3],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [x, 3.3, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    fn minimum(x: f64, energy: f64) -> Node {
        Node::new(
            NodeKind::Minimum,
            Level::Exploration,
            argon(x).into(),
            energy,
            ElectronicState::of(&argon(x), 0),
            Origin::default(),
        )
    }

    #[test]
    fn duplicates_are_found_by_energy_and_structure() {
        let matcher = mapper_matcher(0.3, 1.0, false);
        let mut net = Network::new(vec![Element::from_atomic_number(18).unwrap(); 3]);
        let a = net.add_node(minimum(1.9, -0.001)).unwrap();
        let b = net.add_node(minimum(6.0, -0.0005)).unwrap();
        assert_eq!((a, b), (0, 1));
        // The same triangle with the atoms in another order.
        let swapped = MolecularSystem::new(
            vec![Element::from_atomic_number(18).unwrap(); 3],
            &positions::<Angstrom>(&[[3.8, 0.0, 0.0], [1.9, 3.3, 0.0], [0.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let found = net
            .find_same(
                &NodeKind::Minimum,
                Level::Exploration,
                &swapped.clone().into(),
                -0.001,
                &matcher,
            )
            .unwrap();
        assert_eq!(found, Some(0));
        // Not among the transition states, nor at the other level.
        assert_eq!(
            net.find_same(
                &NodeKind::TransitionState,
                Level::Exploration,
                &swapped.clone().into(),
                -0.001,
                &matcher
            )
            .unwrap(),
            None
        );
        assert_eq!(
            net.find_same(
                &NodeKind::Minimum,
                Level::Refinement,
                &swapped.clone().into(),
                -0.001,
                &matcher
            )
            .unwrap(),
            None
        );
        assert_eq!(net.lowest_minimum(), Some(-0.001));
        assert_eq!(net.node(1).unwrap().label(), "EQ1");
        assert!(net.node(2).is_err());
        let mut other = minimum(1.9, 0.0);
        if let Configuration::Structure(s) = &mut other.system {
            s.elements[0] = Element::from_atomic_number(36).unwrap();
        }
        assert!(net.add_node(other).is_err());
    }
}
