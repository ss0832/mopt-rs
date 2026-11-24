//! Energy-DIIS for geometry optimization (Kudin, Scuseria, and Cances, J. Chem. Phys. 2002,
//! 116, 8255; for geometries: Li and Frisch, J. Chem. Theory Comput. 2006, 2, 835).
//!
//! The energy at the interpolated point `sum_i c_i x_i` is estimated from the energies and
//! gradients of the points,
//! `E(c) = sum_i c_i E_i - (1/4) sum_ij c_i c_j (g_i - g_j).(x_i - x_j)`,
//! which is exact for a quadratic surface, and minimized over the simplex `c_i >= 0`,
//! `sum_i c_i = 1` (an interpolation, never an extrapolation). The step leads to the relaxed
//! interpolated point `sum_i c_i (x_i + r_i)` and is refused when it is longer than ten times
//! the step of the base method. Minimizations only: saddle searches pass unchanged.
//!
//! MultiOptPy's `EDIIS` minimizes `sum_ij c_i c_j [E_j - E_i - g_i.(x_j - x_i)] / 2` (not the
//! EDIIS energy) with SLSQP, moves toward the interpolated geometry without a relaxation step,
//! and blends that displacement with the base step under opposite sign conventions; see
//! `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/ediis.py:EDIIS`.
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

/// EDIIS coefficients of `points`: the minimizer of the interpolated energy on the simplex.
pub fn ediis_coefficients(points: &[DiisPoint]) -> Option<Vec<f64>> {
    let n = points.len();
    let lowest = points
        .iter()
        .map(|p| p.energy)
        .fold(f64::INFINITY, f64::min);
    let l: Vec<f64> = points.iter().map(|p| p.energy - lowest).collect();
    // E(c) = l.c + c.Q c / 2 with Q_ij = -(g_i - g_j).(x_i - x_j) / 2.
    let q = Mat::from_fn(n, n, |i, j| {
        let dg = &points[i].gradient - &points[j].gradient;
        let dx = &points[i].position - &points[j].position;
        -0.5 * col_dot(&dg, &dx)
    });
    simplex_minimum(&q, &l).ok()
}

/// EDIIS step from `points`, or `None` when it is not usable (non-finite, or longer than ten
/// times the step of the base method).
pub fn ediis_step(points: &[DiisPoint], reference: &Col<f64>) -> Option<Col<f64>> {
    let c = ediis_coefficients(points)?;
    let step = relaxed_step(points, &c);
    let len = col_norm(&step);
    (len.is_finite() && len > 0.0 && len <= 10.0 * col_norm(reference)).then_some(step)
}

/// EDIIS step enhancer.
#[derive(Debug, Clone)]
pub struct Ediis {
    history: DiisHistory,
}

impl Ediis {
    /// EDIIS over at most `max_points` points.
    pub fn new(max_points: usize) -> Self {
        Self {
            history: DiisHistory::new(max_points),
        }
    }
}

impl Default for Ediis {
    fn default() -> Self {
        Self::new(5)
    }
}

impl StepEnhancer for Ediis {
    fn name(&self) -> &str {
        "ediis"
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
        Ok(ediis_step(points, &step).unwrap_or(step))
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::diis_core::testing::{quadratic_case, quadratic_points};
    use super::*;

    /// Four points whose convex hull contains the minimum of the quadratic test case.
    pub(crate) fn points_around_the_minimum(p: f64) -> (Vec<DiisPoint>, Col<f64>) {
        let (a, minimum, _) = quadratic_case();
        let offsets = [
            [0.3, 0.1, -0.1],
            [-0.2, 0.3, 0.1],
            [0.1, -0.3, 0.2],
            [-0.2, -0.1, -0.2],
        ];
        let positions: Vec<Vec<f64>> = offsets
            .iter()
            .map(|o| (0..3).map(|i| minimum[i] + o[i]).collect())
            .collect();
        (quadratic_points(&a, &minimum, &positions, p), minimum)
    }

    #[test]
    fn interpolated_energy_minimum_is_the_minimum_of_a_quadratic_inside_the_hull() {
        let (points, minimum) = points_around_the_minimum(0.3);
        let c = ediis_coefficients(&points).unwrap();
        assert!(c.iter().all(|v| *v >= 0.0) && (c.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // The minimum is the centroid of the four points.
        for ci in &c {
            assert!((ci - 0.25).abs() < 1e-8, "{c:?}");
        }
        let step = ediis_step(&points, &points[3].step.clone()).unwrap();
        let reached = &points[3].position + &step;
        assert!(col_norm(&(&reached - &minimum)) < 1e-8);
    }

    #[test]
    fn the_lowest_point_wins_when_the_minimum_is_outside_and_saddles_pass_unchanged() {
        let (a, minimum, positions) = quadratic_case();
        // Two points on a line beyond the minimum on the same side: no interpolation helps.
        let far: Vec<Vec<f64>> = vec![
            (0..3)
                .map(|i| minimum[i] + 2.0 * (positions[1][i] - minimum[i]))
                .collect(),
            positions[1].clone(),
        ];
        let points = quadratic_points(&a, &minimum, &far, 0.3);
        let c = ediis_coefficients(&points).unwrap();
        assert!(c[0].abs() < 1e-10 && (c[1] - 1.0).abs() < 1e-10, "{c:?}");
        let x = Col::zeros(3);
        let g = Col::from_fn(3, |i| [0.1, 0.2, -0.1][i]);
        let h = Mat::<f64>::identity(3, 3);
        let input = crate::opt::stepper::plain_input(0, &x, 0.0, &g, &h, None, 1);
        let s = Col::from_fn(3, |i| [0.01, 0.0, 0.0][i]);
        assert_eq!(Ediis::default().enhance(&input, s.clone()).unwrap(), s);
    }
}
