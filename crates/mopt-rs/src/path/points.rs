//! The requests of the searches that evaluate points one or a few at a time (the dimer
//! method, the spring pair method, the two-point scaled hypersphere search, the growing
//! Newton trajectory, the anharmonic downward distortion following): a point of `n`
//! variables with what is wanted there, and the value of the surface that answers it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's methods call the calculator
//! themselves).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

pub use super::bitss::PointValue;
use crate::error::{MoptError, MoptResult};

/// A point to evaluate.
#[derive(Debug, Clone, PartialEq)]
pub struct PointRequest {
    /// The point.
    pub x: Col<f64>,
    /// Whether its gradient is wanted (the energy always is).
    pub gradient: bool,
    /// Whether its Hessian is wanted.
    pub hessian: bool,
}

impl PointRequest {
    /// Energy and gradient at `x`.
    pub fn gradient(x: Col<f64>) -> Self {
        Self {
            x,
            gradient: true,
            hessian: false,
        }
    }

    /// Energy alone at `x`.
    pub fn energy(x: Col<f64>) -> Self {
        Self {
            x,
            gradient: false,
            hessian: false,
        }
    }

    /// Energy, gradient, and Hessian at `x`.
    pub fn hessian(x: Col<f64>) -> Self {
        Self {
            x,
            gradient: true,
            hessian: true,
        }
    }
}

/// Takes a step of a search from `x` to `y` through other coordinates and returns the point
/// reached instead of `y` (`-coord` with the methods whose distances stay Cartesian).
pub type StepMap =
    std::sync::Arc<dyn Fn(&Col<f64>, &Col<f64>) -> MoptResult<Col<f64>> + Send + Sync>;

/// Next action of such a search.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchAction {
    /// Evaluate these points (in this order; they are independent of each other).
    Evaluate(Vec<PointRequest>),
    /// The search has ended (the search says why).
    Finished,
}

/// Check that `values` answer `asked` (same number, finite, the gradients and Hessians that
/// were asked for, of `n` variables); `what` names the search in the messages.
pub fn check_values(
    what: &str,
    asked: &[PointRequest],
    values: &[PointValue],
    n: usize,
) -> MoptResult<()> {
    if values.len() != asked.len() {
        return Err(MoptError::config(format!(
            "{what}: {} values for {} points",
            values.len(),
            asked.len()
        )));
    }
    for (point, value) in asked.iter().zip(values) {
        if !value.energy.is_finite() {
            return Err(MoptError::calc(format!("{what}: an energy is not finite")));
        }
        let gradient_fits = value
            .gradient
            .as_ref()
            .is_some_and(|g| g.nrows() == n && g.iter().all(|v| v.is_finite()));
        let hessian_fits = value
            .hessian
            .as_ref()
            .is_some_and(|h| h.nrows() == n && h.ncols() == n);
        if (point.gradient && !gradient_fits) || (point.hessian && !hessian_fits) {
            return Err(MoptError::calc(format!(
                "{what}: a gradient or Hessian that was asked for is missing, of the wrong \
                 size, or not finite"
            )));
        }
    }
    Ok(())
}

/// The gradient of a value that was asked for with one.
pub fn gradient_of(what: &str, value: &PointValue) -> MoptResult<Col<f64>> {
    value
        .gradient
        .clone()
        .ok_or_else(|| MoptError::calc(format!("{what}: a gradient is missing")))
}

/// The Hessian of a value that was asked for with one.
pub fn hessian_of(what: &str, value: &PointValue) -> MoptResult<Mat<f64>> {
    value
        .hessian
        .clone()
        .ok_or_else(|| MoptError::calc(format!("{what}: a Hessian is missing")))
}
