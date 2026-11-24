//! Conformation search: the conformers of a structure that keep its bond graph (or, in the
//! mode that collects minima, every minimum), without duplicates, with rotamers counted, and
//! with Boltzmann populations.
//!
//! The search runs its optimizations and dynamics through [`SearchJobs`] (the command builds
//! them with its calculator, its surface, and the settings of `optmain`), keeps the minima in
//! a [`ConformerEnsemble`] of the exploration core, chooses where to go on with a
//! [`crate::workflow::explore::SamplingPriority`], ends by [`crate::workflow::explore::StopRules`],
//! and reports everything to a [`SearchObserver`], which writes the files. Strategies:
//!
//! - [`afir`]: MultiOptPy's sampling, an AFIR push or pull of one atom pair per trial followed
//!   by an optimization without it, from a minimum chosen by the priority;
//! - [`imtd`]: the iMTD-GC workflow of CREST, implemented from its papers: RMSD metadynamics,
//!   multi-level optimization with energy windows, iterations from every lower conformer, the
//!   genetic crossing of conformers, and short dynamics from the lowest conformers;
//! - [`ts`]: conformers of a transition state, sampled with the saddle direction held and
//!   optimized to saddle points of the same reaction.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py:run_confsearch`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod afir;
pub mod imtd;
pub mod moves;
mod search;
pub mod ts;
mod ts_coordinate;

use std::sync::Arc;

use faer::Mat;

use crate::coord::graph::{BondChanges, MolecularGraph};
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;
use crate::md::FreezeSettings;
use crate::pes::constraint::Constraint;
use crate::pes::term::PesTerm;
use crate::workflow::explore::{CompareSettings, ConformerEnsemble};

/// How tightly a minimization converges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// The loose criteria of the first stage (`optmain -lcc`).
    Loose,
    /// The criteria of the command (`-tcc`, `-lcc`, or the default).
    Final,
}

/// A minimization (or saddle-point search) the search asks for.
#[derive(Clone)]
pub struct RelaxJob {
    /// What it is, for the log.
    pub label: String,
    /// Its criteria.
    pub level: Level,
    /// Terms added to the surface of the command (the AFIR force of a trial).
    pub terms: Vec<(String, Arc<dyn PesTerm>)>,
    /// Constraints added to those of the command (the saddle direction of a transition state).
    pub constraints: Vec<Constraint>,
    /// Order of the stationary point (0: a minimum).
    pub saddle_order: usize,
}

impl RelaxJob {
    /// A minimization with the criteria `level` on the surface of the command.
    pub fn minimize(label: impl Into<String>, level: Level) -> Self {
        Self {
            label: label.into(),
            level,
            terms: Vec::new(),
            constraints: Vec::new(),
            saddle_order: 0,
        }
    }
}

/// The end of a [`RelaxJob`].
#[derive(Debug, Clone)]
pub struct Relaxed {
    /// The last structure.
    pub system: MolecularSystem,
    /// Its energy on the surface of the command, without the terms of the job (Hartree): with
    /// the bias potentials of the command when the job adds none (the energy the conformers
    /// are ranked by).
    pub energy: f64,
    /// Its energy without any bias potential (the bare surface of the calculator, Hartree).
    pub bare_energy: f64,
    /// Whether the criteria were met.
    pub converged: bool,
    /// Whether fragments separated (MultiOptPy's `-dc`).
    pub dissociated: bool,
    /// Iterations taken.
    pub iterations: usize,
    /// Whether a `STOP` file ended it.
    pub interrupted: bool,
}

/// A dynamics the search asks for.
#[derive(Clone)]
pub struct DynamicsJob {
    /// What it is, for the log.
    pub label: String,
    /// Temperature of the thermostat (K).
    pub temperature: f64,
    /// Time step (atomic units).
    pub time_step: f64,
    /// Steps.
    pub steps: usize,
    /// A frame every this many steps.
    pub frame_every: usize,
    /// Seed of the starting velocities and of the thermostat.
    pub seed: u64,
    /// Terms added to the surface (the RMSD metadynamics bias).
    pub terms: Vec<(String, Arc<dyn PesTerm>)>,
    /// Constraints held by SHAKE and RATTLE (bond lengths, the saddle direction).
    pub constraints: Vec<Constraint>,
    /// Mass of hydrogen atoms (amu; `None`: their mass).
    pub hydrogen_mass: Option<f64>,
    /// Modes of the Hessian the dynamics does not move along (the saddle direction).
    pub freeze: Option<FreezeSettings>,
}

/// The frames of a [`DynamicsJob`].
#[derive(Debug, Clone, Default)]
pub struct Sampled {
    /// The frames, the first at the start.
    pub frames: Vec<MolecularSystem>,
    /// Whether a `STOP` file ended it.
    pub interrupted: bool,
}

/// What the search needs from the command: optimizations, dynamics, and, for transition
/// states, Hessians and the ends of an IRC.
pub trait SearchJobs: Sync {
    /// Minimize (or search a saddle point) from `start`.
    fn relax(&self, start: &MolecularSystem, job: &RelaxJob) -> MoptResult<Relaxed>;

    /// Run a dynamics from `start`.
    fn dynamics(&self, start: &MolecularSystem, job: &DynamicsJob) -> MoptResult<Sampled>;

    /// Cartesian Hessian of the surface at `sys` (Hartree / Bohr^2).
    fn hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>>;

    /// The normal modes of `sys` whose imaginary ones the search counts (by the one criterion
    /// of the crate, `NormalModes::imaginary_modes`): the zero modes projected as the
    /// frequency analysis of `optmain -freq` projects them (the rigid-body motions; with
    /// constraints the modes on the constraint surface). The default projects the rigid-body
    /// motions of [`Self::hessian`].
    fn modes(&self, sys: &MolecularSystem) -> MoptResult<crate::vib::NormalModes> {
        crate::vib::normal_modes(sys, &self.hessian(sys)?, true)
    }

    /// The two ends of the IRC of the transition state `ts`, optimized to minima, with their
    /// energies (`None` when the IRC could not be followed to two minima).
    fn reaction_ends(
        &self,
        ts: &MolecularSystem,
        label: &str,
    ) -> MoptResult<Option<[(MolecularSystem, f64); 2]>>;

    /// Whether the run was asked to stop (a `STOP` file).
    fn stop_requested(&self) -> bool;

    /// The atoms (from 0) that the constraints of the command freeze: moves leave them where
    /// they are (none by default).
    fn frozen_atoms(&self) -> Vec<usize> {
        Vec::new()
    }
}

/// The start of the message of a [`SearchJobs::relax`] whose candidate could not be restored
/// onto the constraints of the command: such a candidate is discarded and counted
/// ([`CandidateFate::OffConstraints`]).
pub const OFF_CONSTRAINTS: &str = "the candidate could not be restored onto the constraints";

/// What became of a candidate.
#[derive(Debug, Clone)]
pub enum CandidateFate {
    /// The optimization did not converge.
    NotConverged,
    /// Fragments separated.
    Dissociated,
    /// The calculation failed (the message).
    Failed(String),
    /// The candidate could not be restored onto the constraints of the command (the message).
    OffConstraints(String),
    /// Outside the energy window (energy above the lowest, Hartree).
    OutsideWindow(f64),
    /// Its bond graph differs from that of the input.
    Rejected {
        /// The structure.
        system: Box<MolecularSystem>,
        /// Its energy.
        energy: f64,
        /// The bonds formed and broken (in the numbering of the input).
        changes: BondChanges,
    },
    /// The same structure as conformer `of`.
    Duplicate {
        /// The conformer (its number of discovery).
        of: usize,
    },
    /// A rotamer of conformer `of`.
    Rotamer {
        /// The conformer.
        of: usize,
    },
    /// A new conformer (its number of discovery) and its energy.
    New {
        /// Its number.
        index: usize,
        /// Its energy.
        energy: f64,
    },
}

/// What the search tells its observer.
#[derive(Debug, Clone)]
pub enum SearchEvent<'a> {
    /// A line for the log.
    Note(String),
    /// A trial of MultiOptPy's sampling starts from conformer `from`.
    Sample {
        /// The trial (from 0).
        trial: usize,
        /// The conformer it starts from.
        from: usize,
        /// What it does.
        what: String,
    },
    /// A candidate was judged.
    Candidate {
        /// How it was made.
        origin: &'a str,
        /// What became of it.
        fate: &'a CandidateFate,
        /// The optimized structure, when the optimization ended.
        system: Option<&'a MolecularSystem>,
    },
    /// The next trial starts from conformer `to` (MultiOptPy's `switch_conformer.log`).
    Switch {
        /// The trial after which it was chosen.
        trial: usize,
        /// The conformer.
        to: usize,
        /// Trials started from it so far.
        visits: usize,
        /// Probabilities of the choice, for rules that have them.
        probabilities: Option<Vec<f64>>,
    },
    /// The counter of MultiOptPy's termination after a trial (`no_update_count.log`).
    Counter {
        /// The trial.
        trial: usize,
        /// Trials without a change of the lowest conformers.
        count: usize,
    },
    /// The atom pairs of MultiOptPy's sampling (`search_atom_pairs.log`).
    Pairs(&'a [([usize; 2], f64)]),
}

/// Receives the progress of a search; it writes the files and the log.
pub trait SearchObserver {
    /// An event.
    fn on_event(&mut self, event: &SearchEvent<'_>) -> MoptResult<()>;
}

/// An observer that keeps nothing.
pub struct NoObserver;

impl SearchObserver for NoObserver {
    fn on_event(&mut self, _event: &SearchEvent<'_>) -> MoptResult<()> {
        Ok(())
    }
}

/// Which minima the search keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Conformers: minima with the bond graph of the optimized input (others are rejected and
    /// kept apart; MultiOptPy's `-pbc`).
    Conformers,
    /// Every distinct minimum, whatever its bonds (MultiOptPy without `-pbc`).
    Minima,
}

/// How two bond graphs are compared in [`Mode::Conformers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphCheck {
    /// The same graph up to a renumbering of the atoms (a hydrogen moved between equivalent
    /// atoms keeps the molecule).
    Isomorphic,
    /// The same bonds in the same numbering (MultiOptPy's `-pbc`).
    SameNumbering,
}

/// Settings of every strategy.
#[derive(Debug, Clone)]
pub struct CommonSettings {
    /// Which minima are kept.
    pub mode: Mode,
    /// How the graphs are compared.
    pub graph_check: GraphCheck,
    /// Scale of the covalent radii of the bonds.
    pub bond_scale: f64,
    /// Duplicates and rotamers.
    pub compare: CompareSettings,
    /// Atoms compared (from 0; empty: all).
    pub compare_atoms: Vec<usize>,
    /// Seed of the random numbers.
    pub seed: u64,
    /// Energy window of the first, loose optimizations (Hartree above the lowest).
    pub loose_window: f64,
    /// Snapshots whose loose optimization lands within the RMSD threshold of a known
    /// conformer are not optimized further.
    pub skip_known: bool,
    /// Every new minimum (and the optimized input) is checked by its Hessian; from a saddle
    /// point the optimization goes on along the imaginary mode to a minimum.
    pub verify: bool,
}

impl Default for CommonSettings {
    fn default() -> Self {
        Self {
            mode: Mode::Conformers,
            graph_check: GraphCheck::Isomorphic,
            bond_scale: crate::coord::graph::MULTIOPTPY_BOND_SCALE,
            compare: CompareSettings::default(),
            compare_atoms: Vec::new(),
            seed: 1,
            loose_window: 12.0 * Units::KCAL_MOL_TO_HARTREE,
            skip_known: true,
            verify: true,
        }
    }
}

/// A minimum rejected for its bonds.
#[derive(Debug, Clone)]
pub struct Rejected {
    /// The structure.
    pub system: MolecularSystem,
    /// Its energy.
    pub energy: f64,
    /// How it was made.
    pub origin: String,
    /// The bonds formed and broken.
    pub changes: BondChanges,
}

/// The end of a search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The conformers (members in the order they were found: EQ0, EQ1, ...).
    pub ensemble: ConformerEnsemble,
    /// The bond graph of the optimized input.
    pub reference: MolecularGraph,
    /// Minima rejected for their bonds.
    pub rejected: Vec<Rejected>,
    /// Why the search ended.
    pub reason: String,
    /// Whether a `STOP` file ended it.
    pub interrupted: bool,
    /// Candidates judged, and how many became new conformers, duplicates, rotamers.
    pub tally: Tally,
    /// The energy of every member without the bias potentials of the command (Hartree), in
    /// the order of the members; the members' own energies, which rank them, include them.
    pub bare_energies: Vec<f64>,
}

impl SearchResult {
    /// The energy of member `i` without the bias potentials (its ranking energy when the
    /// search has no record of it).
    pub fn bare_energy(&self, i: usize) -> f64 {
        self.bare_energies
            .get(i)
            .copied()
            .unwrap_or_else(|| self.ensemble.members()[i].energy)
    }
}

/// Counts of the candidates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    /// Candidates judged.
    pub candidates: usize,
    /// New conformers.
    pub new: usize,
    /// Duplicates.
    pub duplicates: usize,
    /// Rotamers.
    pub rotamers: usize,
    /// Rejected for their bonds.
    pub rejected: usize,
    /// Not converged, dissociated, or failed.
    pub failed: usize,
    /// Discarded because they could not be restored onto the constraints.
    pub off_constraints: usize,
    /// Outside the energy window.
    pub outside: usize,
}
