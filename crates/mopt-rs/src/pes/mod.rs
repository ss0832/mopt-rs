//! Potential energy surfaces: the `Calculator` trait, in-process terms (bias potentials),
//! combined surfaces, and native calculators.
//!
//! MultiOptPy: `Calculator/`, `Potential/`, `ModelFunction/`, `Constraint/`, `optimization.py`
//!  (handlers).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod bias;
pub mod bias_response;
pub mod biased;
pub mod calculator;
pub mod combine;
pub mod constraint;
pub mod counting;
pub mod dispersion;
pub mod fd_hessian;
pub mod followed;
pub mod given_hessian;
pub mod microiter;
pub mod model_objectives;
pub mod native;
pub mod o1numhess;
pub mod objective;
pub mod oniom;
pub mod point_charges;
pub mod registry;
pub mod replay;
pub mod states;
pub mod stress;
pub mod term;
pub mod term_check;

pub use calculator::{evaluate_checked, Calculator};
pub use combine::{EvalTask, PesExpr, SurfaceEvaluation, TaskSpec};
pub use term::{CalculatorTerm, Frame, History, PesTerm, TermContext};
