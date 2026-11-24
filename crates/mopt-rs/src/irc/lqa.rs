//! Local quadratic approximation (LQA): the steepest-descent path of the quadratic model of
//! the surface at a point is followed analytically for one step length.
//!
//! In the eigenvectors of the mass-weighted Hessian the gradient decays as
//! `g_i(t) = g_i exp(-lambda_i t)`; the displacement after the parameter `t` is
//! `sum_i v_i g_i (exp(-lambda_i t) - 1) / lambda_i`, and `t` is the one at which the length
//! of the path, `integral |g(t)| dt`, equals the step size. As in MultiOptPy the length is
//! summed in 20000 steps of `t`, modes with `|lambda| < 1e-8` (the rigid-body motion) are
//! left out, and the factor is taken with `expm1` and its limit `-t`.
//!
//! References: M. Page and J. W. McIver, J. Chem. Phys. 1988, 88, 922; M. Page, C. Doubleday,
//! and J. W. McIver, J. Chem. Phys. 1990, 93, 5634.
//!
//! MultiOptPy: `IRC/lqa.py:LQA` (`step`, `run`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::control::{Approach, Arrived};
use super::history::HessianUpdate;
use super::integrator::{BranchAction, BranchStart, Integrator, Walk};
use super::types::{norm, PointValue, StepCounts, StepRecord};
use crate::error::MoptResult;
use crate::math::linalg::symmetric_eigen;
use crate::math::stable::EXP_MAX;

/// MultiOptPy's number of steps of the parameter `t` in which the length is summed.
pub const N_EULER: usize = 20_000;
/// Modes with eigenvalues smaller than this in magnitude are left out.
pub const SMALL_EIGENVALUE: f64 = 1e-8;
/// The part of the gradient in a mode, relative to its norm, below which it is rounding and
/// left out (new in mopt-rs; MultiOptPy keeps it).
pub const ROUNDING: f64 = 1e-12;
/// MultiOptPy's share of the step size of the first step of a side, taken straight down the
/// gradient.
pub const FIRST_STEP_SHARE: f64 = 0.05;

/// One LQA step.
#[derive(Debug, Clone)]
pub struct LqaStep {
    /// The displacement in mass-weighted coordinates.
    pub step: Col<f64>,
    /// The parameter `t` at which the step ends.
    pub time: f64,
    /// The gradient of the quadratic model at the end of the step.
    pub gradient: Col<f64>,
}

/// The LQA displacement in mass-weighted coordinates for the Hessian `mw_hessian` and the
/// gradient `mw_gradient` there; `gradient_floor` is MultiOptPy's lower limit of the
/// gradient norm in the step of `t` (`1e-8` in LQA, `1e-6` in the predictor of HPC).
pub fn lqa_step(
    mw_hessian: &Mat<f64>,
    mw_gradient: &Col<f64>,
    step_size: f64,
    gradient_floor: f64,
) -> MoptResult<Col<f64>> {
    let (values, vectors) = symmetric_eigen(mw_hessian)?;
    Ok(lqa_step_in_modes(&values, &vectors, mw_gradient, step_size, gradient_floor).step)
}

/// [`lqa_step`] with the eigenvalues `values` and eigenvectors `vectors` (columns) of the
/// Hessian.
pub fn lqa_step_in_modes(
    values: &Col<f64>,
    vectors: &Mat<f64>,
    mw_gradient: &Col<f64>,
    step_size: f64,
    gradient_floor: f64,
) -> LqaStep {
    let n = mw_gradient.nrows();
    let kept: Vec<usize> = (0..n)
        .filter(|&k| values[k].abs() >= SMALL_EIGENVALUE)
        .collect();
    // A part of the gradient in a mode that is rounding (below ROUNDING of the whole) is left
    // out: along a mode of negative curvature it would grow by exp(|lambda| t), and near a
    // stationary point, where t is large, it would carry a path that runs down a symmetric
    // ridge (into the transition state at its end) off the ridge. Steepest descent stays on it.
    let whole = norm(mw_gradient);
    let projected: Vec<f64> = kept
        .iter()
        .map(|&k| {
            let p: f64 = (0..n).map(|i| vectors[(i, k)] * mw_gradient[i]).sum();
            if p.abs() < ROUNDING * whole {
                0.0
            } else {
                p
            }
        })
        .collect();
    let dt = step_size / N_EULER as f64 / norm(mw_gradient).max(gradient_floor);
    let mut t = dt;
    let mut length = 0.0;
    for _ in 0..N_EULER {
        let speed_squared: f64 = kept
            .iter()
            .zip(&projected)
            .map(|(&k, p)| p * p * libm::exp((-2.0 * values[k] * t).min(EXP_MAX)))
            .sum();
        length += libm::sqrt(speed_squared) * dt;
        if length > step_size {
            break;
        }
        t += dt;
    }
    let mut step = Col::<f64>::zeros(n);
    // What is left of the gradient: its part in the modes that were left out as it is.
    let mut gradient = mw_gradient.clone();
    for (&k, p) in kept.iter().zip(&projected) {
        if *p == 0.0 {
            continue;
        }
        let x = (-values[k] * t).min(EXP_MAX);
        // (exp(x) - 1) / lambda, with its limit -t for x -> 0.
        let alpha = if x.abs() < 1e-8 {
            -t
        } else {
            libm::expm1(x) / values[k]
        };
        let decay = libm::expm1(x);
        for i in 0..n {
            step[i] += alpha * p * vectors[(i, k)];
            gradient[i] += decay * p * vectors[(i, k)];
        }
    }
    LqaStep {
        step,
        time: t,
        gradient,
    }
}

/// The LQA integrator.
pub struct Lqa {
    walk: Walk,
}

impl Lqa {
    /// A side from `start`.
    pub fn new(start: BranchStart) -> Self {
        // MultiOptPy: `for iter in range(1, max_step)`.
        let last = start.settings.max_steps.saturating_sub(1);
        Self {
            walk: Walk::new(start, last, HessianUpdate::Fsb).with_final_approach(),
        }
    }
}

impl Integrator for Lqa {
    fn name(&self) -> &'static str {
        "lqa"
    }

    fn ask(&self) -> BranchAction {
        self.walk.next.clone()
    }

    fn tell(&mut self, value: &PointValue) -> MoptResult<Option<StepRecord>> {
        let w = &mut self.walk;
        // A point of higher energy is dropped and the step taken again, shorter.
        let (arrival, new) = match w.arrive_or_retry(value)? {
            Arrived::Point(arrival) => (arrival, true),
            Arrived::Again(arrival) => (arrival, false),
            Arrived::Wait => return Ok(None),
        };
        let record = new.then(|| arrival.record.clone());
        if arrival.end.is_some() {
            return Ok(record);
        }
        let next = match w.approach(&arrival)? {
            Approach::Arrived => return Ok(record),
            Approach::Step(x) => x,
            Approach::Path => {
                let s = w.step_length();
                if w.step == 1 {
                    w.downhill(&arrival.g, FIRST_STEP_SHARE * s)
                } else {
                    let hessian = w.model.total(&w.space, &arrival.value);
                    let dq = lqa_step(&hessian, &arrival.g, s, 1e-8)?;
                    w.space.displaced(&w.x, &dq)
                }
            }
        };
        w.advance(next);
        Ok(record)
    }

    fn current(&self) -> &Col<f64> {
        &self.walk.x
    }

    fn counts(&self) -> StepCounts {
        self.walk.counts
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::irc::types::dot;

    #[test]
    fn the_step_has_the_length_asked_for_and_follows_the_model() {
        // A valley along x that is steep across: the path bends towards the floor.
        let h = Mat::from_fn(2, 2, |i, j| [[0.2, 0.0], [0.0, 3.0]][i][j]);
        let g = Col::from_fn(2, |i| [0.3, 0.4][i]);
        let step = lqa_step(&h, &g, 0.1, 1e-8).unwrap();
        // The chord is shorter than the arc of 0.1 (the path bends sharply).
        let chord = norm(&step);
        assert!(chord < 0.1 && chord > 0.08, "{chord}");
        assert!(dot(&step, &g) < 0.0);
        // The component across the valley decays faster than the one along it.
        assert!(step[1].abs() / step[0].abs() < 0.4 / 0.3);
        // The same displacement from the closed form at the parameter t of the step.
        let t = libm::log(1.0 - 0.2 * step[0] / -0.3) / -0.2;
        let across = 0.4 * libm::expm1(-3.0 * t) / 3.0;
        assert!((step[1] - across).abs() < 1e-12);
    }

    #[test]
    fn flat_and_uphill_modes() {
        // A mode without curvature is left out (the rigid-body motion); along negative
        // curvature the step grows, still of the length asked for.
        let h = Mat::from_fn(3, 3, |i, j| if i == j { [0.0, -0.5, 1.0][i] } else { 0.0 });
        let g = Col::from_fn(3, |i| [0.05, -0.2, 0.0][i]);
        let step = lqa_step(&h, &g, 0.3, 1e-8).unwrap();
        assert_eq!(step[0], 0.0);
        assert!(step[1] > 0.29 && step[1] <= 0.3 + 1e-4, "{}", step[1]);
        assert_eq!(step[2], 0.0);
        // No gradient at all: no step, and no NaN.
        let none = lqa_step(&h, &Col::zeros(3), 0.3, 1e-8).unwrap();
        assert_eq!(norm(&none), 0.0);
    }
}
