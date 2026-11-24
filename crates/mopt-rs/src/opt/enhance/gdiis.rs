//! GDIIS: geometry optimization by direct inversion in the iterative subspace (Csaszar and
//! Pulay, J. Mol. Struct. 1984, 114, 31), with the safeguards of Farkas and Schlegel (Phys.
//! Chem. Chem. Phys. 2002, 4, 11).
//!
//! The coefficients minimize the norm of the interpolated gradient, `|sum_i c_i g_i|` with
//! `sum_i c_i = 1`, over the latest points, and the step leads to the relaxed interpolated
//! point `sum_i c_i (x_i + r_i)` (`r_i`: the step of the base method at point `i`); see
//! [`super::diis_core`] for the acceptance rules. On a quadratic surface the extrapolation is
//! exact once the points span the space.
//!
//! MultiOptPy's `GDIIS` extrapolates the geometry only (no relaxation step, so the result stays
//! in the span of the earlier geometries), clips the coefficients to `[-0.7, 0.7]`, and blends
//! the displacement toward that geometry with the base step under opposite sign conventions
//! (the base step enters as `x_k - x_{k+1}`, the GDIIS step as `x_GDIIS - x_k`); see
//! `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/gdiis.py:GDIIS`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::diis_core::{first_acceptable, pulay_coefficients, DiisHistory, DiisPoint};
use super::StepEnhancer;
use crate::error::MoptResult;
use crate::opt::step::StepInput;

/// GDIIS step from `points` (the last one is the current point), or `None` when no window of
/// points gives an acceptable step.
pub fn gdiis_step(points: &[DiisPoint], reference: &Col<f64>) -> Option<Col<f64>> {
    first_acceptable(points, reference, |window| {
        let errors: Vec<&Col<f64>> = window.iter().map(|p| &p.gradient).collect();
        pulay_coefficients(&errors).ok()
    })
}

/// GDIIS step enhancer.
#[derive(Debug, Clone)]
pub struct Gdiis {
    history: DiisHistory,
    /// Number of points needed before extrapolating.
    pub min_points: usize,
}

impl Gdiis {
    /// GDIIS over at most `max_points` points, starting with `min_points`.
    pub fn new(max_points: usize, min_points: usize) -> Self {
        Self {
            history: DiisHistory::new(max_points),
            min_points: min_points.max(2),
        }
    }
}

impl Default for Gdiis {
    fn default() -> Self {
        Self::new(5, 3)
    }
}

impl StepEnhancer for Gdiis {
    fn name(&self) -> &str {
        "gdiis"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        self.history.record(input, &step);
        let points = self.history.points();
        if points.len() < self.min_points {
            return Ok(step);
        }
        Ok(gdiis_step(points, &step).unwrap_or(step))
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::diis_core::testing::{quadratic_case, quadratic_points};
    use super::*;
    use crate::math::linalg::col_norm;

    #[test]
    fn accepted_step_reaches_the_minimum_of_a_quadratic() {
        let (a, minimum, positions) = quadratic_case();
        // Base steps of a preconditioner close to the inverse Hessian scale: the exact step is
        // within the acceptance rules.
        let points = quadratic_points(&a, &minimum, &positions, 0.3);
        let reference = points[3].step.clone();
        let step = gdiis_step(&points, &reference).unwrap();
        let reached = &points[3].position + &step;
        assert!(col_norm(&(&reached - &minimum)) < 1e-9);
        // Tiny base steps: the exact step is more than ten times longer and is refused for
        // every window.
        let points = quadratic_points(&a, &minimum, &positions, 1e-4);
        assert!(gdiis_step(&points, &points[3].step.clone()).is_none());
    }
}
