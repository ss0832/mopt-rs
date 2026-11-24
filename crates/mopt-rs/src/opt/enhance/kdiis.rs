//! Krylov-subspace DIIS: the Krylov-subspace accelerated inexact Newton method (KAIN; Harrison,
//! J. Comput. Chem. 2004, 25, 328), which Rohwedder and Schneider (J. Math. Chem. 2011, 49,
//! 1889) relate to DIIS as a projected (quasi-)Newton method on the subspace of the history.
//!
//! With the differences to the newest point `n`, `d_i = x_i - x_n` and `y_i = g_i - g_n`, the
//! Hessian is known on the span of the `d_i` through the secant relations `H d_i = y_i`. The
//! Newton equation is solved in that subspace by the Galerkin condition
//! `sum_i c_i d_j.y_i = -d_j.g_n` for all `j`,
//! and the gradient left over, `g_n + sum_i c_i y_i`, is relaxed by the base method:
//! `dx = sum_i c_i d_i + r_n + sum_i c_i (r_i - r_n)`,
//! which is the relaxed combination of [`super::diis_core`] with the coefficients
//! `(c_1, ..., c_{n-1}, 1 - sum_i c_i)`. GDIIS instead minimizes the norm of the interpolated
//! gradient; the two agree on a quadratic surface once the history spans the space, where both
//! are exact. The acceptance rules are those of the family.
//!
//! MultiOptPy's `KrylovDIIS` builds its "Krylov basis" from one gradient difference rescaled as
//! a Hessian-vector product, solves a projected system from it, and blends the result with a
//! DIIS geometry and the base step by alignment heuristics, with the sign and relaxation
//! problems of its other DIIS classes; see `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/kdiis.py:KrylovDIIS`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::diis_core::{first_acceptable, DiisHistory, DiisPoint};
use super::StepEnhancer;
use crate::error::MoptResult;
use crate::math::linalg::col_dot;
use crate::math::simplex_qp::least_squares;
use crate::opt::step::StepInput;

/// KAIN coefficients of `points` (sum 1; the last one belongs to the newest point).
pub fn kain_coefficients(points: &[DiisPoint]) -> Option<Vec<f64>> {
    let last = points.last()?;
    let m = points.len() - 1;
    let d: Vec<Col<f64>> = points[..m]
        .iter()
        .map(|p| &p.position - &last.position)
        .collect();
    let y: Vec<Col<f64>> = points[..m]
        .iter()
        .map(|p| &p.gradient - &last.gradient)
        .collect();
    let a = Mat::from_fn(m, m, |j, i| col_dot(&d[j], &y[i]));
    let b: Vec<f64> = d.iter().map(|dj| -col_dot(dj, &last.gradient)).collect();
    let mut c = least_squares(&a, &b).ok()?;
    let sum: f64 = c.iter().sum();
    c.push(1.0 - sum);
    Some(c)
}

/// KAIN step from `points`, or `None` when no window gives an acceptable step.
pub fn kdiis_step(points: &[DiisPoint], reference: &Col<f64>) -> Option<Col<f64>> {
    first_acceptable(points, reference, kain_coefficients)
}

/// Krylov-DIIS (KAIN) step enhancer.
#[derive(Debug, Clone)]
pub struct Kdiis {
    history: DiisHistory,
    /// Number of points needed before extrapolating.
    pub min_points: usize,
}

impl Kdiis {
    /// KAIN over at most `max_points` points, starting with `min_points`.
    pub fn new(max_points: usize, min_points: usize) -> Self {
        Self {
            history: DiisHistory::new(max_points),
            min_points: min_points.max(2),
        }
    }
}

impl Default for Kdiis {
    fn default() -> Self {
        Self::new(6, 3)
    }
}

impl StepEnhancer for Kdiis {
    fn name(&self) -> &str {
        "kdiis"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        self.history.record(input, &step);
        let points = self.history.points();
        if points.len() < self.min_points {
            return Ok(step);
        }
        Ok(kdiis_step(points, &step).unwrap_or(step))
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::diis_core::testing::{quadratic_case, quadratic_points};
    use super::super::diis_core::{pulay_coefficients, relaxed_step};
    use super::*;
    use crate::math::linalg::col_norm;

    #[test]
    fn subspace_newton_step_is_exact_on_a_quadratic_and_agrees_with_gdiis() {
        let (a, minimum, positions) = quadratic_case();
        let points = quadratic_points(&a, &minimum, &positions, 0.3);
        let c = kain_coefficients(&points).unwrap();
        let reached = &points[3].position + &relaxed_step(&points, &c);
        assert!(col_norm(&(&reached - &minimum)) < 1e-9);
        let errors: Vec<&Col<f64>> = points.iter().map(|p| &p.gradient).collect();
        let pulay = pulay_coefficients(&errors).unwrap();
        for (x, y) in c.iter().zip(&pulay) {
            assert!((x - y).abs() < 1e-8);
        }
        // With a subspace smaller than the space the two differ: KAIN makes the remaining
        // gradient orthogonal to the subspace, GDIIS makes it shortest.
        let two = &points[1..];
        let c = kain_coefficients(two).unwrap();
        let mut g: Col<f64> = Col::zeros(3);
        for (p, ci) in two.iter().zip(&c) {
            g += &p.gradient * *ci;
        }
        for p in &two[..2] {
            let d = &p.position - &two[2].position;
            assert!(col_dot(&d, &g).abs() < 1e-10);
        }
        assert!(kdiis_step(&points, &points[3].step.clone()).is_some());
    }
}
