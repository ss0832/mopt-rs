//! Step generators (extension point).
//!
//! A step generator proposes a step `dq` in the optimizer's coordinate system from the current
//! point, the previous point, and the shared quasi-Newton Hessian. Generators never evaluate the
//! energy themselves; everything that needs another evaluation is expressed through the ask/tell
//! state machine.
//!
//! MultiOptPy: `Optimizer/` (one step generator per MultiOptPy optimizer).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod abc_fire;
pub mod adabelief;
pub mod adam;
pub mod ccqn;
pub mod cone;
pub mod conjugate_gradient;
pub mod eve;
pub mod fire;
pub mod fire2;
pub mod gpmin;
pub mod gradient_descent;
pub mod hisd;
pub mod lbfgs;
pub mod mf_rsirfo;
pub mod mode_following;
pub mod newton;
pub mod prfo;
pub mod prodigy;
pub mod radam;
pub mod rfo_core;
pub mod rsirfo;
pub mod rsprfo;
pub mod tr_lbfgs;

use crate::coord::system::CoordinateFrame;
use crate::domain::{Hartree, MolecularSystem, Quantity};
use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};

/// Data of the previous optimizer iteration (all vectors in the optimizer's coordinates).
#[derive(Debug, Clone)]
pub struct PreviousPoint {
    /// Bias-including energy.
    pub energy: Quantity<Hartree>,
    /// Bias-including gradient.
    pub gradient: Col<f64>,
    /// Unbiased gradient.
    pub raw_gradient: Col<f64>,
    /// Step taken from the previous point to the current one.
    pub step: Col<f64>,
}

/// Input of one step proposal.
#[derive(Debug, Clone, Copy)]
pub struct StepInput<'a> {
    /// Iteration index (0-based).
    pub iteration: usize,
    /// Cartesian coordinates `x` (Bohr, length `3N`).
    pub cartesian: &'a Col<f64>,
    /// Bias-including energy.
    pub energy: Quantity<Hartree>,
    /// Bias-including gradient in the optimizer's coordinates.
    pub gradient: &'a Col<f64>,
    /// Unbiased gradient in the optimizer's coordinates.
    pub raw_gradient: &'a Col<f64>,
    /// Previous iteration, if any.
    pub previous: Option<&'a PreviousPoint>,
    /// Quasi-Newton (model / exact / updated) Hessian of the unbiased surface.
    pub hessian: &'a Mat<f64>,
    /// Exact Hessian of the bias potentials, if any.
    pub bias_hessian: Option<&'a Mat<f64>>,
    /// Orthonormal rigid-body modes to remove (Cartesian coordinates only).
    pub rigid_body_basis: Option<&'a Mat<f64>>,
    /// Whether a generator that keeps a Hessian of its own takes `rigid_body_basis` out of it
    /// after every update (`-update_proj`; see [`crate::opt::update::project_out_rigid`]).
    pub project_updates: bool,
    /// Number of negative-curvature directions to maximize along (0 = minimization).
    pub saddle_order: usize,
    /// Structure being optimized (elements, positions, cell), when available.
    pub system: Option<&'a MolecularSystem>,
    /// Whether the optimizer's coordinates are the Cartesian coordinates of `system`.
    pub cartesian_coordinates: bool,
    /// Linear map between Cartesian and optimizer coordinates at `system`, when available.
    pub frame: Option<&'a CoordinateFrame>,
    /// Bias-including Cartesian gradient (Hartree/Bohr), when available.
    pub cartesian_gradient: Option<&'a Col<f64>>,
}

/// Refuse a step with a component that is not finite (`methods` names its source).
///
/// MultiOptPy: no counterpart (new in mopt-rs). MultiOptPy checks only inside RS-I-RFO; a step
/// that is not finite from any other method moves the structure to NaN and the failure shows up
/// later, in the calculator.
pub fn ensure_finite_step(step: &Col<f64>, methods: &[String]) -> MoptResult<()> {
    if step.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(MoptError::conv(format!(
            "the step of '{}' is not finite",
            methods.join(" + ")
        )))
    }
}

/// Proposes optimization steps.
pub trait StepGenerator: Send {
    /// Canonical name (for logging).
    fn name(&self) -> &str;

    /// Whether the generator uses the quasi-Newton Hessian (MultiOptPy's `newton_tag`).
    fn uses_hessian(&self) -> bool {
        false
    }

    /// Whether [`Self::propose`] reads the exact Hessian of the bias potentials
    /// ([`StepInput::bias_hessian`]); the optimizer computes the Hessians of its terms only
    /// when a method or an enhancer reads them, or when the iteration asks for a Hessian
    /// anyway. The default: the methods that use the quasi-Newton Hessian.
    fn uses_bias_hessian(&self) -> bool {
        self.uses_hessian()
    }

    /// Whether the Hessian update must also learn from pairs with `s . y <= 0` (methods that
    /// rely on the quasi-Newton Hessian turning indefinite, such as CCQN). Saddle searches
    /// (`saddle_order > 0`) learn from them with every method.
    fn learns_negative_curvature(&self) -> bool {
        false
    }

    /// Scale `c` of the initial Hessian `c I` the method prefers (MultiOptPy uses `I`).
    fn initial_hessian_scale(&self) -> Option<f64> {
        None
    }

    /// Whether the method controls the step length itself, so that the optimizer's outer trust
    /// radius only applies when the user sets one (`-tr`).
    fn controls_step_length(&self) -> bool {
        false
    }

    /// Set the step factor of the method (`-d`, the `DELTA` of MultiOptPy's optimizers: the
    /// learning rate of the first-order methods, the factor on the Newton and L-BFGS
    /// steps); `false` if the method has none.
    fn set_step_factor(&mut self, _delta: f64) -> bool {
        false
    }

    /// What the method did inside its last step (the lines MultiOptPy's optimizers print
    /// while they work), for the debug level of the run log.
    fn take_debug_notes(&mut self) -> Vec<String> {
        Vec::new()
    }

    /// Propose a step `dq = q_{k+1} - q_k`.
    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>>;

    /// Forget internal history.
    fn reset(&mut self) {}

    /// Follow these directions (one per mode of the saddle order, in the coordinates of the
    /// optimization) from the start, instead of the lowest modes of the first Hessian; `false`
    /// for a method that does not follow modes.
    fn seed_followed_modes(&mut self, directions: &[Col<f64>]) -> bool {
        let _ = directions;
        false
    }
}
