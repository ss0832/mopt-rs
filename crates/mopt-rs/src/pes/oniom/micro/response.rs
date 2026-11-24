//! The predicted response of the relaxed environment to a step of the core.
//!
//! At a relaxed environment `y*(x)`, a step `dx` of the core moves the minimum of the
//! environment by `dy = R dx` to first order, with `R = -H_yy^-1 H_yx`. The columns of `R`
//! are those the effective Hessian of the core solves for (one per core coordinate); between
//! Hessians, and before the first, `R` is corrected after every relaxation by Broyden's update
//! with the step of the core and the change of the relaxed environment that followed
//! (C. G. Broyden, Math. Comput. 1965, 19, 577). The next relaxation starts from the
//! prediction `y*(x_last) + R (x - x_last)` when its energy is lower than that of the last
//! relaxed environment (two low-level evaluations), else from the last relaxed environment.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::math::linalg::col_dot;

/// Entries of the response matrix at most (32 MB); larger environments go without the
/// prediction.
pub(super) const RESPONSE_ENTRIES: usize = 1 << 22;

/// The linear model `dy = R dx` of the response of the relaxed environment.
#[derive(Debug, Clone)]
pub(super) struct Response {
    /// `m x n`: environment coordinates by core coordinates.
    r: Mat<f64>,
    /// Whether `r` holds anything yet (a zero model predicts no change).
    known: bool,
}

impl Response {
    /// A model that knows nothing yet, for `m` environment and `n` core coordinates.
    pub(super) fn new(m: usize, n: usize) -> Self {
        Self {
            r: Mat::zeros(m, n),
            known: false,
        }
    }

    /// Whether the model predicts anything.
    pub(super) fn is_known(&self) -> bool {
        self.known
    }

    /// The predicted change of the relaxed environment for the step `dx` of the core.
    pub(super) fn predict(&self, dx: &Col<f64>) -> Col<f64> {
        &self.r * dx
    }

    /// Column `k` of `R` (the response to core coordinate `k`).
    pub(super) fn set_column(&mut self, k: usize, column: &Col<f64>) {
        for j in 0..self.r.nrows() {
            self.r[(j, k)] = column[j];
        }
        self.known = true;
    }

    /// Broyden's update with the step `dx` of the core and the change `dy` of the relaxed
    /// environment: `R += (dy - R dx) dx^T / dx^T dx`.
    pub(super) fn update(&mut self, dx: &Col<f64>, dy: &Col<f64>) {
        let dd = col_dot(dx, dx);
        if dd < 1e-24 {
            return;
        }
        let residual = dy - &self.r * dx;
        for j in 0..self.r.nrows() {
            let rj = residual[j] / dd;
            for k in 0..self.r.ncols() {
                self.r[(j, k)] += rj * dx[k];
            }
        }
        self.known = true;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn broyden_learns_a_linear_response_along_the_steps() {
        // dy = A dx with A of 3 x 2; a step reproduces its own response, and after
        // orthogonal steps along both core coordinates the model is A.
        let a = Mat::from_fn(3, 2, |j, k| (j + 2 * k) as f64 - 1.5);
        let mut model = Response::new(3, 2);
        assert!(!model.is_known());
        for dx in [
            Col::from_fn(2, |k| [0.6, 0.8][k]),
            Col::from_fn(2, |k| [-0.8, 0.6][k]),
        ] {
            let dy = &a * &dx;
            model.update(&dx, &dy);
            let error = crate::math::linalg::col_norm(&(model.predict(&dx) - &dy));
            assert!(error < 1e-12, "{error}");
        }
        let probe = Col::from_fn(2, |k| [-0.4, 1.1][k]);
        let error = crate::math::linalg::col_norm(&(model.predict(&probe) - &a * &probe));
        assert!(error < 1e-12, "{error}");
    }
}
