//! Pure safe-Rust forward-mode automatic differentiation (`Scalar`, `Dual`, `HyperDual`).
//!
//! Replaces the `torch.func` (`grad`, `hessian`, `jacrev`) usage of MultiOptPy. Potentials,
//! internal coordinates, and constraints are written once, generic over [`Scalar`], and
//! differentiated per term: each term touches a handful of atoms, so the drivers in
//! [`drivers`] are evaluated over 6-12 local variables and the result is scattered into the
//! global `3N` gradient / Hessian (see [`local`]). Whole-system Hessians through
//! hyper-dual numbers cost `O(n^2)` passes and are reserved for verification tests.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); it stands in for the `torch.func` calls of
//!  `Potential/*.py`, `ModelFunction/*.py`, and
//!  `Coordinate/redundant_coordinate.py:TorchDerivatives`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod drivers;
pub mod dual;
pub mod hyperdual;
pub mod local;
pub mod mic;
pub mod scalar;

pub use drivers::{eval_value_and_gradient, eval_value_grad_hessian, eval_vector_and_jacobian};
pub use dual::Dual;
pub use hyperdual::HyperDual;
pub use local::{
    local_value_grad, local_value_grad_hess, scatter_gradient, scatter_hessian, LocalDerivatives,
};
pub use mic::mic_delta_ad;
pub use scalar::{angle_between, Scalar};

#[cfg(test)]
mod tests;
