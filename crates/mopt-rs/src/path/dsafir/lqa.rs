//! The step of an end of DS-AFIR by the local quadratic approximation (LQA), and what an end
//! knows of the curvature of the energy for it.
//!
//! The paper integrates the AFIR path, the steepest-descent path of the AFIR function `F`
//! (eq. 6), "by the LQA method" (M. Page and J. W. McIver, J. Chem. Phys. 1988, 88, 922).
//! With the gradient `g_F = g + X u` at a point (`X` and `Y` held) and a Hessian `H` there,
//! the gradient of the quadratic model decays along the model's own steepest-descent path
//! as `g_i exp(-lambda_i t)` in the eigenvectors `v_i` of `H`; the displacement after the
//! parameter `t` is `sum_i v_i g_i (exp(-lambda_i t) - 1) / lambda_i`, and `t` is the one at
//! which the length of that path, `integral |g(t)| dt`, is the step length. The stiff modes
//! across the drive relax within the step, where a step of fixed length straight down
//! `grad F` overshoots them and zigzags.
//!
//! `H` is the Hessian of the energy without the rigid-body motions and the fixed variables
//! (the second derivatives of the two distance terms of `F` are added if asked for). Unlike
//! [`crate::irc::lqa`] no mode is left out for a small eigenvalue (the drive may lie in a
//! flat mode; its factor is the limit `-t`), and the coordinates are not mass-weighted (the
//! paper: "the AFIR and LUP paths were calculated in the non-mass-weighted coordinate").
//! Where the model has a minimum closer than the step length, the step ends there.
//!
//! The Hessian of the energy at an end is exact where one was asked for and is updated in
//! between by Bofill's formula from the differences of the gradients of the energy (not of
//! `F`, which changes with `X` and `Y`) of consecutive points of that end.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::MoptResult;
use crate::math::linalg::{col_norm, symmetric_eigen};
use crate::math::stable::EXP_MAX;
use crate::opt::update::formulas::bofill;

/// Parts in which the length of the model's path is summed up to the step length.
const ARC_PARTS: usize = 2000;
/// Parts at most (the path of a model with a minimum nearby never reaches the step length).
const ARC_LIMIT: usize = 20 * ARC_PARTS;
/// The model's path has ended where its gradient is this share of the one at the start.
const ENDED: f64 = 1.0e-8;

/// The displacement of one LQA step of the arc length `length` for the gradient `gradient`
/// and the Hessian `hessian` (`None` for a zero gradient).
pub fn lqa_displacement(
    hessian: &Mat<f64>,
    gradient: &Col<f64>,
    length: f64,
) -> MoptResult<Option<Col<f64>>> {
    let n = gradient.nrows();
    let start = col_norm(gradient);
    if !(start.is_finite() && start > 0.0) {
        return Ok(None);
    }
    let (values, vectors) = symmetric_eigen(hessian)?;
    let parts: Vec<f64> = (0..n)
        .map(|k| (0..n).map(|i| vectors[(i, k)] * gradient[i]).sum())
        .collect();
    let speed = |t: f64| -> f64 {
        libm::sqrt(
            parts
                .iter()
                .enumerate()
                .map(|(k, p)| p * p * libm::exp((-2.0 * values[k] * t).min(EXP_MAX)))
                .sum::<f64>(),
        )
    };
    // The parameter at which the path is `length` long: each part of the sum is about
    // `length / ARC_PARTS` long.
    let part = length / ARC_PARTS as f64;
    let (mut t, mut walked) = (0.0, 0.0);
    for _ in 0..ARC_LIMIT {
        let v = speed(t);
        if v < ENDED * start {
            break;
        }
        let dt = part / v;
        // The midpoint rule for the length of this part.
        walked += speed(t + 0.5 * dt) * dt;
        t += dt;
        if walked >= length {
            // Back by what the last part went too far.
            let v = speed(t);
            if v > 0.0 {
                t -= ((walked - length) / v).min(dt);
            }
            break;
        }
    }
    let mut step = Col::<f64>::zeros(n);
    for (k, p) in parts.iter().enumerate() {
        if *p == 0.0 {
            continue;
        }
        let x = (-values[k] * t).min(EXP_MAX);
        // (exp(x) - 1) / lambda, with its limit -t for x -> 0.
        let factor = if x.abs() < 1.0e-8 {
            -t
        } else {
            libm::expm1(x) / values[k]
        };
        for i in 0..n {
            step[i] += factor * p * vectors[(i, k)];
        }
    }
    Ok(Some(step))
}

/// The Hessian of the energy at the points of one end.
#[derive(Debug, Clone, Default)]
pub struct EndCurvature {
    hessian: Option<Mat<f64>>,
    at: Option<(Col<f64>, Col<f64>)>,
    updates: usize,
}

impl EndCurvature {
    /// Whether the next point of the end is to be evaluated with its exact Hessian: the
    /// first one, and every `every`-th after an exact one (0: the first one only).
    pub fn needs_exact(&self, every: usize) -> bool {
        self.hessian.is_none() || (every > 0 && self.updates + 1 >= every)
    }

    /// Take the point `x` with the gradient `g` of the energy: its exact Hessian if there is
    /// one, the update of the Hessian of the point before otherwise.
    pub fn take(&mut self, x: &Col<f64>, g: &Col<f64>, exact: Option<&Mat<f64>>) {
        match exact {
            Some(h) => {
                let n = h.nrows();
                self.hessian = Some(Mat::from_fn(n, n, |i, j| 0.5 * (h[(i, j)] + h[(j, i)])));
                self.updates = 0;
            }
            None => {
                if let (Some(h), Some((x0, g0))) = (self.hessian.as_mut(), self.at.as_ref()) {
                    let (s, y) = (x - x0, g - g0);
                    if col_norm(&s) > 1.0e-10 {
                        let correction = bofill(h, &s, &y, 1.0);
                        *h += correction;
                    }
                    self.updates += 1;
                }
            }
        }
        self.at = Some((x.clone(), g.clone()));
    }

    /// The Hessian at the point taken last.
    pub fn hessian(&self) -> Option<&Mat<f64>> {
        self.hessian.as_ref()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn diagonal(d: &[f64]) -> Mat<f64> {
        Mat::from_fn(d.len(), d.len(), |i, j| if i == j { d[i] } else { 0.0 })
    }

    #[test]
    fn the_step_follows_the_path_of_the_quadratic_model() {
        // A stiff and a soft mode: the path first runs down the stiff one and then turns
        // along the soft one; its length is the step length.
        let h = diagonal(&[4.0, 0.1]);
        let g = Col::from_fn(2, |k| [1.0, 0.2][k]);
        let length = 0.05;
        let step = lqa_displacement(&h, &g, length).unwrap().unwrap();
        // The same path by small steps of steepest descent on the model.
        let (mut x, mut walked) = ([0.0_f64; 2], 0.0);
        while walked < length {
            let grad = [g[0] + 4.0 * x[0], g[1] + 0.1 * x[1]];
            let norm = (grad[0] * grad[0] + grad[1] * grad[1]).sqrt();
            let ds = 1.0e-6;
            x[0] -= ds * grad[0] / norm;
            x[1] -= ds * grad[1] / norm;
            walked += ds;
        }
        assert!((step[0] - x[0]).abs() < 1.0e-4, "{} {}", step[0], x[0]);
        assert!((step[1] - x[1]).abs() < 1.0e-4, "{} {}", step[1], x[1]);
        // Straight down the gradient the stiff mode would be overshot.
        assert!(step[0].abs() < length * 1.0 / (1.04_f64).sqrt());
    }

    #[test]
    fn flat_and_downhill_modes_and_the_minimum_of_the_model() {
        // A flat mode takes the limit of its factor: a straight step of the full length.
        let h = diagonal(&[0.0, 0.0]);
        let g = Col::from_fn(2, |k| [0.3, 0.4][k]);
        let step = lqa_displacement(&h, &g, 0.1).unwrap().unwrap();
        assert!((step[0] + 0.06).abs() < 1.0e-5 && (step[1] + 0.08).abs() < 1.0e-5);
        // Negative curvature: the step is as long as asked for and goes downhill.
        let h = diagonal(&[-0.5, 1.0]);
        let step = lqa_displacement(&h, &g, 0.1).unwrap().unwrap();
        assert!(step[0] < 0.0 && col_norm(&step) <= 0.1 + 1.0e-6);
        // A minimum of the model nearer than the step length: the step ends there.
        let h = diagonal(&[10.0, 10.0]);
        let step = lqa_displacement(&h, &g, 1.0).unwrap().unwrap();
        assert!((step[0] + 0.03).abs() < 1.0e-6 && (step[1] + 0.04).abs() < 1.0e-6);
        // No gradient, no step.
        assert!(lqa_displacement(&h, &Col::zeros(2), 0.1).unwrap().is_none());
    }

    #[test]
    fn the_hessian_is_exact_where_asked_and_updated_in_between() {
        let exact = diagonal(&[2.0, 0.5]);
        let gradient = |x: &Col<f64>| Col::from_fn(2, |k| [2.0, 0.5][k] * x[k]);
        let mut c = EndCurvature::default();
        assert!(c.needs_exact(10) && c.needs_exact(0));
        let x0 = Col::from_fn(2, |k| [1.0, 1.0][k]);
        // A first Hessian that is wrong in the first mode.
        c.take(&x0, &gradient(&x0), Some(&diagonal(&[1.0, 0.5])));
        assert!(!c.needs_exact(3) && !c.needs_exact(0));
        let x1 = Col::from_fn(2, |k| [0.5, 1.0][k]);
        c.take(&x1, &gradient(&x1), None);
        // The update has learnt the curvature along the step.
        let h = c.hessian().unwrap();
        assert!((h[(0, 0)] - 2.0).abs() < 1.0e-12 && (h[(1, 1)] - 0.5).abs() < 1.0e-12);
        assert!(!c.needs_exact(3));
        let x2 = Col::from_fn(2, |k| [0.5, 0.8][k]);
        c.take(&x2, &gradient(&x2), None);
        assert!(c.needs_exact(3) && !c.needs_exact(0));
        c.take(&x2, &gradient(&x2), Some(&exact));
        assert!(!c.needs_exact(3));
    }
}
