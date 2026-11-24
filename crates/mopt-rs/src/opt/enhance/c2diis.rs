//! C2-DIIS (Sellers, Int. J. Quantum Chem. 1993, 45, 31).
//!
//! Instead of minimizing `c.B c` (`B_ij = g_i . g_j`) under `sum_i c_i = 1`, which requires the
//! inverse of the nearly singular `B`, the norm constraint `c.c = 1` is used: the candidates are
//! the eigenvectors of `B`, each rescaled to `sum_i c_i = 1`, and the one with the smallest
//! residual `|sum_i c_i g_i|` whose rescaling is well defined is taken. No linear system is
//! solved, so linear dependence among the gradients is harmless. The step leads to the relaxed
//! interpolated point as in GDIIS ([`super::diis_core`]).
//!
//! MultiOptPy's `C2DIIS` projects each gradient on its own direction (which returns the
//! gradient itself) and then solves the ordinary DIIS equations with clipped coefficients; it
//! is GDIIS under another name, with the same sign and relaxation problems (see
//! `docs/source/migration_optimizers.md`).
//!
//! MultiOptPy: `Optimizer/c2diis.py:C2DIIS`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::diis_core::{first_acceptable, DiisHistory, DiisPoint};
use super::StepEnhancer;
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, symmetric_eigen};
use crate::opt::step::StepInput;

/// C2-DIIS coefficients of the error vectors: the eigenvector of `B` (rescaled to sum 1) with
/// the smallest residual; eigenvectors whose components sum to less than `1e-6` of their
/// largest component are skipped.
pub fn c2_coefficients(errors: &[&Col<f64>]) -> Option<Vec<f64>> {
    let n = errors.len();
    if n == 0 {
        return None;
    }
    let b = Mat::from_fn(n, n, |i, j| col_dot(errors[i], errors[j]));
    let (vals, vecs) = symmetric_eigen(&b).ok()?;
    let mut best: Option<(f64, Vec<f64>)> = None;
    for k in 0..n {
        let sum: f64 = (0..n).map(|i| vecs[(i, k)]).sum();
        let largest = (0..n).fold(0.0_f64, |m, i| m.max(vecs[(i, k)].abs()));
        if sum.abs() < 1e-6 * largest || sum == 0.0 {
            continue;
        }
        // |sum_i c_i g_i|^2 = lambda / sum^2 for c = v / sum.
        let residual = vals[k].max(0.0) / (sum * sum);
        if residual.is_finite() && best.as_ref().is_none_or(|(r, _)| residual < *r) {
            best = Some((residual, (0..n).map(|i| vecs[(i, k)] / sum).collect()));
        }
    }
    best.map(|(_, c)| c)
}

/// C2-DIIS step from `points`, or `None` when no window gives an acceptable step.
pub fn c2diis_step(points: &[DiisPoint], reference: &Col<f64>) -> Option<Col<f64>> {
    first_acceptable(points, reference, |window| {
        let errors: Vec<&Col<f64>> = window.iter().map(|p| &p.gradient).collect();
        c2_coefficients(&errors)
    })
}

/// C2-DIIS step enhancer.
#[derive(Debug, Clone)]
pub struct C2Diis {
    history: DiisHistory,
    /// Number of points needed before extrapolating.
    pub min_points: usize,
}

impl C2Diis {
    /// C2-DIIS over at most `max_points` points, starting with `min_points`.
    pub fn new(max_points: usize, min_points: usize) -> Self {
        Self {
            history: DiisHistory::new(max_points),
            min_points: min_points.max(2),
        }
    }
}

impl Default for C2Diis {
    fn default() -> Self {
        Self::new(5, 2)
    }
}

impl StepEnhancer for C2Diis {
    fn name(&self) -> &str {
        "c2diis"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        self.history.record(input, &step);
        let points = self.history.points();
        if points.len() < self.min_points {
            return Ok(step);
        }
        Ok(c2diis_step(points, &step).unwrap_or(step))
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
    fn null_eigenvector_gives_the_exact_extrapolation_on_a_quadratic() {
        let (a, minimum, positions) = quadratic_case();
        let points = quadratic_points(&a, &minimum, &positions, 0.3);
        let errors: Vec<&Col<f64>> = points.iter().map(|p| &p.gradient).collect();
        let c = c2_coefficients(&errors).unwrap();
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        // Four gradients in three dimensions: B has a null vector, the same coefficients as
        // the constrained least-squares solution.
        let pulay = pulay_coefficients(&errors).unwrap();
        for (x, y) in c.iter().zip(&pulay) {
            assert!((x - y).abs() < 1e-7, "{c:?} vs {pulay:?}");
        }
        let reached = &points[3].position + &relaxed_step(&points, &c);
        assert!(col_norm(&(&reached - &minimum)) < 1e-7);
        // Fewer points than dimensions: a residual remains; the coefficients still sum to one.
        let three: Vec<&Col<f64>> = errors[1..].to_vec();
        let c = c2_coefficients(&three).unwrap();
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        let mut g: Col<f64> = Col::zeros(3);
        for (e, ci) in three.iter().zip(&c) {
            g += *e * *ci;
        }
        assert!(col_norm(&g) > 1e-6 && col_norm(&g).is_finite());
    }
}
