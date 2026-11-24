//! Intrinsic reaction coordinate (IRC): the steepest-descent path in mass-weighted
//! coordinates from a transition state to the minima it connects.
//!
//! [`Irc`] is an ask/tell state machine like the optimizers of [`crate::opt`]: it asks for
//! the surface at one point at a time and is told the result, so it works with any
//! calculator and from Python. It examines the starting structure ([`StartKind`]: a
//! transition state, a structure for a meta-IRC, or ModeKill), optionally tests and corrects
//! the first point of each side ([`first_step`]), and runs one integrator per side:
//!
//! | name | method | evaluations per step |
//! |---|---|---|
//! | `lqa` | local quadratic approximation ([`lqa`]) | 1 gradient |
//! | `hpc` | Hessian-based predictor-corrector ([`hpc`], [`dwi`]) | 1 gradient |
//! | `dvv` | damped velocity Verlet ([`dvv`]) | 1 gradient |
//! | `euler` | Euler ([`euler`]) | 1 gradient |
//! | `rk4` | fourth-order Runge-Kutta ([`rk4`]) | 4 gradients |
//! | `gs2` | Gonzalez-Schlegel second order ([`gs2`], [`sphere`]) | a few gradients |
//! | `modekill` | removal of imaginary modes ([`modekill`]) | 1 gradient per cycle |
//!
//! The variables are the Cartesian coordinates of a structure with its atomic masses, or
//! any `n` variables with unit masses ([`IrcSpace`]). Every part can be used on its own:
//! [`lqa_step`], [`DwiSurface`], [`SphereMinimizer`], [`test_first_step`],
//! [`convergence_check`], and the integrators through [`IntegratorRegistry`].
//!
//! MultiOptPy: `irc.py:IRC` and `IRC/` (`lqa.py`, `hpc.py`, `dvv.py`, `euler.py`, `rk4.py`,
//! `modekill.py`, `converge_criteria.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod constraints;
pub mod control;
pub mod convergence;
pub mod dvv;
pub mod dwi;
pub mod euler;
pub mod first_step;
pub mod gs2;
pub mod history;
pub mod hpc;
pub mod integrator;
pub mod lqa;
pub mod machine;
pub mod modekill;
pub mod registry;
pub mod rk4;
pub mod settings;
pub mod space;
pub mod sphere;
pub mod start;
#[cfg(test)]
mod tests;
pub mod types;

pub use constraints::IrcConstraints;
pub use convergence::convergence_check;
pub use dwi::{DwiSurface, TaylorPoint};
pub use first_step::{test_first_step, FirstStepReport, FirstStepTest};
pub use history::{angle_deg, HessianUpdate, PathHistory, SurfaceModel};
pub use integrator::{BranchAction, BranchStart, Integrator};
pub use lqa::{lqa_step, lqa_step_in_modes, LqaStep};
pub use machine::{BranchSummary, Irc, IrcAction, IrcEnd, IrcEvent, IrcRequest};
pub use modekill::eigenvalue_to_wavenumber;
pub use registry::{IntegratorFactory, IntegratorRegistry, INTEGRATORS};
pub use settings::{FirstStepSettings, IrcSettings, Masses, Sides, SphereSettings};
pub use space::{most_abundant_isotope_masses, IrcSpace};
pub use sphere::{SphereMinimizer, SphereStatus};
pub use start::{StartKind, StartReport};
pub use types::{
    Branch, BranchEnd, PathCurvature, PointRequest, PointRole, PointValue, StepCounts, StepRecord,
};
