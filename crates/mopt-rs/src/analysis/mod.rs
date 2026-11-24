//! Analysis of computed results: rate constants and energy decompositions, maps of a
//! series of structures in few dimensions, dynamic modes of an optimization, free-energy
//! profiles of umbrella sampling, and the analysis of molecular dynamics trajectories.
//!
//! MultiOptPy: `optimization.py:EDEELHandler` (energy decompositions) and `PESAnalyzer/`
//! (`cmds_analysis.py`, `pca_analysis.py`, `koopman_analysis.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

// Time correlation functions, spectra, transport coefficients.
pub mod correlation;
pub mod diabatic;
// Disconnectivity graphs and lowest-barrier paths of networks of minima.
pub mod disconnectivity;
// The rate formulas of electron transfer live in the kinetics (they are rate theories there).
pub use crate::kinetics::rate::electron_transfer;
// Free-energy profiles of umbrella sampling.
pub mod free_energy;
// Force-directed places of the nodes of a graph for drawings.
pub mod graph_layout;
pub mod koopman;
pub mod orbitals;
pub mod path_map;
// ReSPer coordinates of new structures (Gower's projection) and their pre-images.
pub mod resper;
// Analysis of molecular dynamics trajectories.
pub mod trajectory;
