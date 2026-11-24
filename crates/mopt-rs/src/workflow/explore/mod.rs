//! Shared building blocks of exploration workflows (conformation search, orientation search,
//! reaction-network mapping): choosing which local minimum to sample next, when to stop,
//! telling whether two structures are the same minimum, the pool of distinct minima, the
//! ensemble of conformers with duplicates and rotamers, and new candidates by crossing
//! conformers.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py` (Boltzmann choice of the next minimum, the
//!  termination, the duplicate check).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod addf;
pub mod identity;
pub mod orient;
pub mod pool;
// Candidates from internal-coordinate differences of conformers.
pub mod crossover;
// Duplicates, conformers and rotamers, populations.
pub mod ensemble;
pub mod priority;
// MultiOptPy's heated selection and the adaptive rising water.
pub mod priority_rules;
// A SMILES string of a structure, for grouping by chemical species.
pub mod smiles;
// When a search ends.
pub mod stop;

pub use crossover::{crossover_candidates, CrossoverSettings};
pub use ensemble::{CompareSettings, ConformerEnsemble, EnsembleThermo, Member, Verdict};
pub use identity::{BondGraph, Match, StructureMatcher};
pub use pool::{boltzmann_weights, Insertion, MinimaPool, Minimum};
pub use priority::{
    Boltzmann, EqRecord, PenalizedBoltzmann, PriorityRegistry, SamplingPriority, WaterFilling,
};
pub use priority_rules::{AdaptiveWaterFilling, HeatedBoltzmann};
pub use stop::{
    MaxTrials, NoLowerMinimum, NothingNew, RankUnchanged, StopRule, StopRules, TrialSummary,
};
