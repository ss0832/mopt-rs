//! Calculators implemented in pure Rust (force fields and model surfaces).
//!
//! MultiOptPy: `Calculator/lj_calculation_tools.py`, `emt_calculation_tools.py`,
//!  `tersoff_calculation_tools.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod eam;
pub mod embedding;
pub mod emt;
/// The central-difference Hessians of [crate::pes::fd_hessian], at their former path.
pub use crate::pes::fd_hessian;
pub mod gfn0;
pub mod gfn1;
pub mod gfnff;
pub mod lennard_jones;
pub mod meam;
pub mod muller_brown;
pub mod neighbors;
pub mod parallel;
pub mod scf_diis;
pub mod switching;
pub mod tb_response;
pub(crate) mod tda_gradient;
pub mod tersoff;
pub mod unrestricted_tda;

pub use eam::Eam;
pub use emt::Emt;
pub use gfn0::Gfn0Xtb;
pub use gfn1::Gfn1Xtb;
pub use gfnff::Gfnff;
pub use lennard_jones::{LennardJones, TruncatedLennardJones};
pub use meam::Meam;
pub use muller_brown::MullerBrown;
pub use tersoff::Tersoff;
