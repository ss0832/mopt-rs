//! Geometry optimization: step generators, Hessian updaters, step enhancers, trust radius,
//! convergence criteria, and the ask/tell optimizer state machine.
//!
//! Sign conventions: coordinates in Bohr, energies in Hartree, gradient `g = +dE/dx`, and steps
//! `dx = x_{k+1} - x_k`.
//!
//! MultiOptPy: `optimizer.py:CalculateMoveVector` and `Optimizer/`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod cell;
pub mod cell_constraints;
pub mod constrained;
pub mod convergence;
pub mod crossing_check;
pub mod enhance;
pub mod micro;
pub mod mode_kill;
pub mod registry;
pub mod state_machine;
pub mod stationary;
pub mod step;
pub mod stepper;
pub mod trust;
pub mod update;
pub mod vector;
pub mod vector_constraints;
