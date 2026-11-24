//! `mopt-rs` — core engine of the successor implementation of MultiOptPy: extensible geometry
//! optimization and reaction path exploration.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later
//!
//! Built strictly under `#![forbid(unsafe_code)]` with zero PyTorch dependency, providing:
//! - Pure-Rust forward-mode `Dual` and `HyperDual` automatic differentiation
//! - Generic N-layer `OniomCalculator` with analytical link-atom Jacobians and second derivatives
//! - First-class Periodic Boundary Conditions (`PeriodicCell`), including PBC-AFIR
//! - Pluggable optimizer (`OptimizerRegistry`) and reaction path (`PathMethodRegistry`) engines
//! - Variational Direct MaxFlux (`DMF`), `DE-GSM` / `SE-GSM`, `FSM`, `DS-AFIR`, `BITSS`, `iEIP`, `NEB`
//! - Enhanced `IRC` with hypersphere constrained optimization and high-precision `LQA`
//! - Automated TS search (`AutoTS` v1/v2) and reaction network exploration (`ReactionNetworkMapper`)

#![forbid(unsafe_code)]
#![allow(dead_code)]

pub mod analysis;
pub mod cli;
pub mod coord;
pub mod core_infra;
pub mod domain;
pub mod error;
pub mod io;
pub mod irc;
pub mod kinetics;
pub mod math;
pub mod mcp;
pub mod md;
pub mod model_hessian;
pub mod opt;
pub mod parameters;
pub mod path;
pub mod pes;
pub mod vib;
pub mod workflow;

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
