//! ADIIS: augmented-energy DIIS (Hu and Yang, J. Chem. Phys. 2010, 132, 054109), applied to
//! geometries.
//!
//! ADIIS minimizes the quadratic (augmented Roothaan-Hall) energy model built at the newest
//! point `n` from the differences to the earlier points. For geometries, with
//! `d_i = x_i - x_n` and `y_i = g_i - g_n`:
//! `E(c) = E_n + sum_i c_i g_n.d_i + (1/2) sum_ij c_i c_j d_i.y_j`,
//! the second-order expansion about `x_n` with the Hessian acting through the secant pairs,
//! minimized over the simplex `c_i >= 0`, `sum_i c_i = 1`. Unlike EDIIS it uses no energies of
//! the earlier points, only the gradients. The step leads to the relaxed interpolated point
//! `sum_i c_i (x_i + r_i)` and is refused when it is longer than ten times the step of the base
//! method. Minimizations only: saddle searches pass unchanged.
//!
//! MultiOptPy's `ADIIS` minimizes a sum of pairwise first-order energy errors and gradient
//! differences that is not the ADIIS functional, with SLSQP, and has the relaxation and sign
//! problems of its other DIIS classes; see `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/adiis.py:ADIIS`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::diis_core::{relaxed_step, DiisHistory, DiisPoint};
use super::StepEnhancer;
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm};
use crate::math::simplex_qp::simplex_minimum;
use crate::opt::step::StepInput;

/// ADIIS coefficients of `points` (the last one is the expansion point).
pub fn adiis_coefficients(points: &[DiisPoint]) -> Option<Vec<f64>> {
    let n = points.len();
    let last = points.last()?;
    let d: Vec<Col<f64>> = points
        .iter()
        .map(|p| &p.position - &last.position)
        .collect();
    let y: Vec<Col<f64>> = points
        .iter()
        .map(|p| &p.gradient - &last.gradient)
        .collect();
    let l: Vec<f64> = d.iter().map(|di| col_dot(&last.gradient, di)).collect();
    // c.M c with M_ij = d_i.y_j only depends on the symmetric part of M.
    let q = Mat::from_fn(n, n, |i, j| {
        0.5 * (col_dot(&d[i], &y[j]) + col_dot(&d[j], &y[i]))
    });
    simplex_minimum(&q, &l).ok()
}

/// ADIIS step from `points`, or `None` when it is not usable.
pub fn adiis_step(points: &[DiisPoint], reference: &Col<f64>) -> Option<Col<f64>> {
    let c = adiis_coefficients(points)?;
    let step = relaxed_step(points, &c);
    let len = col_norm(&step);
    (len.is_finite() && len > 0.0 && len <= 10.0 * col_norm(reference)).then_some(step)
}

/// ADIIS step enhancer.
#[derive(Debug, Clone)]
pub struct Adiis {
    history: DiisHistory,
}

impl Adiis {
    /// ADIIS over at most `max_points` points.
    pub fn new(max_points: usize) -> Self {
        Self {
            history: DiisHistory::new(max_points),
        }
    }
}

impl Default for Adiis {
    fn default() -> Self {
        Self::new(5)
    }
}

impl StepEnhancer for Adiis {
    fn name(&self) -> &str {
        "adiis"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        if input.saddle_order > 0 {
            return Ok(step);
        }
        self.history.record(input, &step);
        let points = self.history.points();
        if points.len() < 2 {
            return Ok(step);
        }
        Ok(adiis_step(points, &step).unwrap_or(step))
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::ediis::ediis_coefficients;
    use super::super::ediis::tests::points_around_the_minimum;
    use super::*;

    #[test]
    fn model_minimum_is_the_minimum_of_a_quadratic_inside_the_hull() {
        let (points, minimum) = points_around_the_minimum(0.3);
        let c = adiis_coefficients(&points).unwrap();
        assert!(c.iter().all(|v| *v >= 0.0) && (c.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let step = adiis_step(&points, &points[3].step.clone()).unwrap();
        let reached = &points[3].position + &step;
        assert!(col_norm(&(&reached - &minimum)) < 1e-8);
        // On a quadratic the ADIIS model and the EDIIS energy have the same minimizer.
        let e = ediis_coefficients(&points).unwrap();
        for (a, b) in c.iter().zip(&e) {
            assert!((a - b).abs() < 1e-8);
        }
    }
}
