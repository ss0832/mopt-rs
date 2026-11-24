//! Workflows: drivers that connect optimizers to calculators, and observers that handle output.
//!
//! MultiOptPy: `optimization.py:Optimize`, `RunLogger`, `ResultPaths`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod chain;
pub mod chain_figures;
pub mod chain_files;
pub mod chain_input;
pub mod chain_internal;
pub mod chain_surrogate;
// The conformation search (confsearch).
pub mod confsearch;
// The crossings of freqmain -diabatize for the kinetics.
pub mod diabatic_crossings;
pub mod dmf;
pub mod driver;
pub mod electron_transfer;
pub mod explore;
// The hop of a charge between two molecules (etmain -et_method hopping).
pub mod hopping;
pub mod irc;
pub mod irc_cell;
mod irc_combined;
pub mod irc_output;
pub mod journal;
// The kinetics of a network file (run_kinetics, run_rcmc).
pub mod kinetics;
// The models, analyses, and outputs of run_kinetics beyond the canonical model.
pub mod kinetics_beyond;
// Dynamics at a saddle point for the kinetics (branching, recrossing, friction).
pub mod kinetics_dynamics;
// The reaction-network mapper (run_mapper).
pub mod mapper;
// Molecular dynamics (mdmain).
pub mod md;
pub mod md_analysis;
pub mod md_analysis_output;
pub mod md_analysis_transport;
pub mod md_coordinates;
pub mod md_cvs;
pub mod md_files;
pub mod md_flux;
// Figures and tables of surface hopping.
pub mod md_hopping_output;
pub mod md_log;
pub mod narrated;
pub mod observers;
pub mod orbital_analysis;
pub mod path_analysis;
// The searches of `ieipmain`.
pub mod pair_search;
// Scans of coordinates (`relaxedscan`).
pub mod scan;
pub mod state_path;
pub mod trajectories;
pub mod umbrella;

pub use driver::{
    run_optimization, run_optimization_with, run_vector_optimization,
    run_vector_optimization_until, CalculatorSet, Observer, Outcome, VectorOutcome,
};
