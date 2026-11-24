//! What the DIIS-type enhancers share: the history of points, the Pulay coefficients, the
//! relaxed extrapolated point, and the acceptance rules of Farkas and Schlegel.
//!
//! Every enhancer of the family records, per iteration, the position (relative to the first
//! point, accumulated from the steps actually taken), the energy, the gradient, and the step
//! `r` proposed by the base method at that point. For coefficients `c` (`sum_i c_i = 1`) the
//! new point is the combination of the relaxed points,
//! `x' = sum_i c_i (x_i + r_i)`,
//! i.e. the interpolated geometry plus the interpolated step (Csaszar and Pulay, J. Mol. Struct.
//! 1984, 114, 31, with the base method as the relaxation). The error vectors are the gradients,
//! which vanish at every stationary point whatever the base method is.
//!
//! On a quadratic surface the gradient is an affine function of the position. With `n + 1`
//! affinely independent points in `n` dimensions the coefficients that annihilate the
//! interpolated gradient exist and give the stationary point exactly, provided the base steps
//! are linear in the gradient.
//!
//! Acceptance (Farkas and Schlegel, Phys. Chem. Chem. Phys. 2002, 4, 11): the extrapolated step
//! must not be longer than ten times the step of the base method, the angle between the two
//! must stay below a bound that widens with the number of points (cosines 0.97, 0.84, 0.71,
//! 0.67, 0.62, 0.56, 0.49, 0.41 for 2 to 9 points), and the sum of the positive coefficients
//! must not exceed 15; otherwise the oldest point is left out and the test repeated, down to
//! two points, and then the step of the base method is taken.
//!
//! MultiOptPy: the shared parts of `Optimizer/gdiis.py`, `ediis.py`, `adiis.py`, `c2diis.py`,
//! `kdiis.py` (history handling and the `B` matrix).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm};
use crate::math::simplex_qp::least_squares;
use crate::opt::step::StepInput;

/// One recorded iteration.
#[derive(Debug, Clone)]
pub struct DiisPoint {
    /// Position relative to the first recorded point.
    pub position: Col<f64>,
    /// Energy.
    pub energy: f64,
    /// Gradient (the error vector).
    pub gradient: Col<f64>,
    /// Step proposed by the base method at this point.
    pub step: Col<f64>,
}

/// History of the latest points.
#[derive(Debug, Clone)]
pub struct DiisHistory {
    points: Vec<DiisPoint>,
    max_points: usize,
    position: Option<Col<f64>>,
}

impl DiisHistory {
    /// History of at most `max_points` points (at least 2).
    pub fn new(max_points: usize) -> Self {
        Self {
            points: Vec::new(),
            max_points: max_points.max(2),
            position: None,
        }
    }

    /// Record the current point with the step the base method proposes there.
    pub fn record(&mut self, input: &StepInput<'_>, step: &Col<f64>) {
        let n = step.nrows();
        let position = match (self.position.take(), input.previous) {
            (Some(p), Some(prev)) if p.nrows() == n => p + &prev.step,
            _ => {
                self.points.clear();
                Col::zeros(n)
            }
        };
        self.points.push(DiisPoint {
            position: position.clone(),
            energy: input.energy.value(),
            gradient: input.gradient.clone(),
            step: step.clone(),
        });
        if self.points.len() > self.max_points {
            self.points.remove(0);
        }
        self.position = Some(position);
    }

    /// Recorded points, oldest first; the last one is the current point.
    pub fn points(&self) -> &[DiisPoint] {
        &self.points
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.points.clear();
        self.position = None;
    }
}

/// Coefficients `c` (`sum c = 1`) minimizing `|sum_i c_i e_i|` (Pulay, Chem. Phys. Lett. 1980,
/// 73, 393). The constraint is eliminated with the last vector,
/// `min |e_n + sum_{i<n} c_i (e_i - e_n)|`, and the least-squares problem is solved with a
/// rank-revealing decomposition, so that linearly dependent vectors do no harm.
pub fn pulay_coefficients(errors: &[&Col<f64>]) -> MoptResult<Vec<f64>> {
    let n = errors.len();
    let Some(last) = errors.last() else {
        return Ok(Vec::new());
    };
    let dim = last.nrows();
    let d = Mat::from_fn(dim, n - 1, |r, c| errors[c][r] - last[r]);
    let rhs: Vec<f64> = (0..dim).map(|r| -last[r]).collect();
    let mut c = least_squares(&d, &rhs)?;
    let sum: f64 = c.iter().sum();
    c.push(1.0 - sum);
    Ok(c)
}

/// Step from the current (last) point to `sum_i c_i (x_i + r_i)`.
pub fn relaxed_step(points: &[DiisPoint], c: &[f64]) -> Col<f64> {
    let Some(last) = points.last() else {
        return Col::zeros(0);
    };
    let mut target: Col<f64> = Col::zeros(last.position.nrows());
    for (p, ci) in points.iter().zip(c) {
        target += (&p.position + &p.step) * *ci;
    }
    target - &last.position
}

/// The acceptance rules of Farkas and Schlegel for a step from `n_points` points with
/// coefficients `c`, against the step of the base method.
pub fn acceptable(step: &Col<f64>, reference: &Col<f64>, c: &[f64], n_points: usize) -> bool {
    const COSINES: [f64; 8] = [0.97, 0.84, 0.71, 0.67, 0.62, 0.56, 0.49, 0.41];
    let (len, ref_len) = (col_norm(step), col_norm(reference));
    if !len.is_finite() || len == 0.0 || ref_len == 0.0 {
        return false;
    }
    let cosine = col_dot(step, reference) / (len * ref_len);
    let bound = COSINES[n_points.clamp(2, 9) - 2];
    let positive: f64 = c.iter().filter(|v| **v > 0.0).sum();
    len <= 10.0 * ref_len && cosine >= bound && positive <= 15.0
}

/// Try `coefficients` on the latest `m` points for `m` from all points down to two, and return
/// the first acceptable relaxed step (`None`: take the step of the base method).
pub fn first_acceptable(
    points: &[DiisPoint],
    reference: &Col<f64>,
    mut coefficients: impl FnMut(&[DiisPoint]) -> Option<Vec<f64>>,
) -> Option<Col<f64>> {
    for start in 0..points.len().saturating_sub(1) {
        let window = &points[start..];
        if let Some(c) = coefficients(window) {
            if c.len() == window.len() && c.iter().all(|v| v.is_finite()) {
                let step = relaxed_step(window, &c);
                if acceptable(&step, reference, &c, window.len()) {
                    return Some(step);
                }
            }
        }
    }
    None
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// Points of the quadratic `E = (x - m).A (x - m) / 2` with base steps `-p g`.
    pub(crate) fn quadratic_points(
        a: &Mat<f64>,
        minimum: &Col<f64>,
        positions: &[Vec<f64>],
        p: f64,
    ) -> Vec<DiisPoint> {
        positions
            .iter()
            .map(|x| {
                let x = Col::from_fn(x.len(), |i| x[i]);
                let d = &x - minimum;
                let g = a * &d;
                DiisPoint {
                    energy: 0.5 * col_dot(&d, &g),
                    step: &g * (-p),
                    gradient: g,
                    position: x,
                }
            })
            .collect()
    }

    /// A positive-definite matrix, its minimum, and `n + 1` affinely independent points.
    pub(crate) fn quadratic_case() -> (Mat<f64>, Col<f64>, Vec<Vec<f64>>) {
        let a = Mat::from_fn(3, 3, |i, j| {
            [[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 4.0]][i][j]
        });
        let minimum = Col::from_fn(3, |i| [0.4, -0.7, 0.2][i]);
        let positions = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.5, 0.1, -0.2],
            vec![-0.3, 0.6, 0.1],
            vec![0.2, -0.4, 0.7],
        ];
        (a, minimum, positions)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn extrapolation_is_exact_on_a_quadratic() {
        let (a, minimum, positions) = quadratic_case();
        let points = quadratic_points(&a, &minimum, &positions, 0.1);
        let errors: Vec<&Col<f64>> = points.iter().map(|p| &p.gradient).collect();
        let c = pulay_coefficients(&errors).unwrap();
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // The interpolated gradient vanishes ...
        let mut g: Col<f64> = Col::zeros(3);
        for (p, ci) in points.iter().zip(&c) {
            g += &p.gradient * *ci;
        }
        assert!(col_norm(&g) < 1e-10);
        // ... and the relaxed point is the minimum.
        let step = relaxed_step(&points, &c);
        let reached = &points[3].position + &step;
        assert!(col_norm(&(&reached - &minimum)) < 1e-10);
    }

    #[test]
    fn dependent_error_vectors_and_the_acceptance_rules() {
        let e = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let twice = &e * 2.0;
        // Parallel vectors: e + c (2e - e) = 0 at c = -1, i.e. coefficients (-1, 2).
        let c = pulay_coefficients(&[&twice, &e]).unwrap();
        assert!((c[0] + 1.0).abs() < 1e-12 && (c[1] - 2.0).abs() < 1e-12);
        // Identical vectors: any coefficients; the minimum-norm choice keeps the last point.
        let c = pulay_coefficients(&[&e, &e]).unwrap();
        assert!(c[0].abs() < 1e-12 && (c[1] - 1.0).abs() < 1e-12);
        assert_eq!(pulay_coefficients(&[&e]).unwrap(), vec![1.0]);
        let r = Col::from_fn(2, |i| [0.1, 0.0][i]);
        let along = Col::from_fn(2, |i| [0.5, 0.05][i]);
        assert!(acceptable(&along, &r, &[0.5, 0.5], 2));
        // Too long, too far off the base step, or wild coefficients.
        assert!(!acceptable(&(&along * 3.0), &r, &[0.5, 0.5], 2));
        let off = Col::from_fn(2, |i| [0.1, 0.1][i]);
        assert!(!acceptable(&off, &r, &[0.5, 0.5], 2));
        assert!(acceptable(&off, &r, &[0.5, 0.5], 5));
        assert!(!acceptable(&along, &r, &[16.0, -15.0], 2));
        assert!(!acceptable(&Col::zeros(2), &r, &[1.0], 2));
    }
}
