//! Step enhancers (extension point).
//!
//! An enhancer transforms a proposed step without additional energy evaluations (DIIS family,
//! LARS, TRIM, component-wise scaling, ...). Methods that need the energy at trial points (the
//! line search) take over the step for several iterations instead: each trial point is an
//! ordinary iteration of the ask/tell optimizer.
//!
//! MultiOptPy: the step modifiers of `optimizer.py:CalculateMoveVector` (`Optimizer/gdiis.py`,
//!  `lars.py`, `trim.py`, `component_wise_scaling.py`, ...).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod adiis;
pub mod c2diis;
pub mod component_wise_scaling;
pub mod coordinate_locking;
pub mod diis_core;
pub mod ediis;
pub mod gan_step;
pub mod gdiis;
pub mod gediis;
pub mod geodesic_step;
pub mod gpr_step;
pub mod kdiis;
pub mod lars;
pub mod linesearch;
pub mod lookahead;
pub mod perturbation;
pub mod rl_step;
pub mod trim;

use faer::Col;

use crate::error::MoptResult;
use crate::opt::step::StepInput;

/// Transforms a proposed step.
pub trait StepEnhancer: Send {
    /// Canonical name.
    fn name(&self) -> &str;

    /// Return the transformed step.
    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>>;

    /// Forget internal history.
    fn reset(&mut self) {}

    /// Whether [`Self::enhance`] reads the exact Hessian of the bias potentials
    /// ([`StepInput::bias_hessian`]); see
    /// [`crate::opt::step::StepGenerator::uses_bias_hessian`].
    fn uses_bias_hessian(&self) -> bool {
        false
    }
}
