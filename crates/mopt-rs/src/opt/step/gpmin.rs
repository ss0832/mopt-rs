//! GPMin: minimization on a Gaussian-process model of the surface (Garijo del Rio, Mortensen,
//! and Jacobsen, Phys. Rev. B 2019, 100, 104103; the `GPMin` optimizer of ASE).
//!
//! Every evaluated point (energy and gradient) is added to the training set of a Gaussian
//! process with a squared-exponential kernel ([`crate::math::gp::GradientGp`]). The prior mean
//! is the highest energy seen, so that the model rises away from the data. The next point is
//! the minimum of the model reached from the lowest-energy point seen so far; when a step
//! raises the energy, the new point improves the model and the model is minimized again from
//! the same best point, until the model yields a descent step. The model minimization costs no
//! evaluations of the surface.
//!
//! Defaults: length scale 0.4 Angstrom and relative noise 0.002 (ASE's `scale = 0.4`,
//! `noise = 0.005`, `weight = 1`, without hyperparameter updates); the training set keeps the
//! latest 25 points and the best one.
//!
//! Saddle points of any order (`saddle_order > 0`; not part of the published method, which
//! minimizes). A prior mean above all data would put a minimum where a saddle is sought, and a
//! Gaussian process of a few points knows curvature only along the directions sampled. The
//! prior mean is therefore the quadratic model at the current point, with the optimizer's
//! Hessian (exact, from a model, or O1NumHess: `-fc`, `-modelhess`), and the Gaussian process
//! is fitted to what the evaluated points show beyond it. On this surface (energy, gradient
//! and Hessian in closed form) the next point is found by restricted-step image-function
//! iterations as in RS-I-RFO: at every iteration the lowest `saddle_order` modes of the
//! surface's Hessian are reflected, so the walk maximizes along them and minimizes along the
//! others. It starts at the current point and ends at a stationary point of the surface or
//! one length scale away, where the data say nothing any more. The first step is thus the
//! RS-I-RFO step of the given Hessian, and later steps are corrected by every gradient seen;
//! the surface is searched without evaluations. Fitting a surface and searching it with a
//! partitioned RFO method is the approach of Denzel and Kastner (J. Chem. Theory Comput. 2018,
//! 14, 5777), whose prior, step control, and hyperparameters differ. The training set keeps
//! the latest points and the one with the smallest gradient.
//!
//! The fit has a noise term and does not pass through the data exactly, so the gradient of
//! the Gaussian process at the current point is subtracted: the surface then has the gradient
//! of the function there. Without this the search ends near, not at, the saddle point and
//! took 19 to 33 iterations in the measurement below instead of 5 to 9.
//!
//! Defaults of the saddle search, measured on a saddle of order 2 of an anharmonic function
//! of three variables from three starting points (iterations to a gradient of 1e-5; `tests`
//! below), with one Hessian at the start / a Hessian at every iteration: relative noise
//! `1e-4` (5, 8, 6 / 5, 6, 5; with the 0.002 of the minimization 6, 9, 6 / 5, 7, 5; `1e-6`
//! as `1e-4`), reach of one length scale (half and twice give the same within one
//! iteration), walk step of a quarter length scale (0.1 to 1 give the same within one
//! iteration; a full length scale with twice the reach ended at another stationary point
//! once). In the optimizer the search is on a par with RS-I-RFO with Bofill updates (see
//! `docs/source/migration_optimizers.md` for the comparison, which includes block Bofill).
//!
//! MultiOptPy's `GPmin` fits the energies only (in inverse interatomic distances), enforces a
//! minimum step of 0.1 Bohr and a minimum trust radius (2.0) above its initial radius, replaces
//! the model step by a move of 80% of the trust radius along the force whenever the model does
//! not predict a decrease, and blends in such moves for forces above 0.1: it cannot converge.
//! See `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/gpmin.py:GPmin`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::rfo_core::{rs_step, RfoParams};
use super::{StepGenerator, StepInput};
use crate::domain::Units;
use crate::error::MoptResult;
use crate::math::gp::{GradientGp, Observation};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen, symmetrize};
use crate::math::minimize::{minimize, MinimizeOptions};
use crate::math::projection::{project_out_basis_mat, project_out_basis_vec};

/// Modes with a smaller absolute eigenvalue are left alone in the saddle walk (rigid-body
/// modes and directions without curvature).
const FLAT_MODE: f64 = 1e-8;

/// GPMin parameters.
#[derive(Debug, Clone, Copy)]
pub struct GpMinParams {
    /// Kernel length scale (Bohr, or the unit of the variables).
    pub length_scale: f64,
    /// Relative noise of the observations.
    pub noise: f64,
    /// Number of most recent points kept for training (the best point is always kept).
    pub max_points: usize,
    /// Iteration limit of the model minimization.
    pub model_iterations: usize,
    /// Gradient tolerance of the model minimization.
    pub model_tolerance: f64,
    /// Saddle searches: the walk on the surface ends this many length scales from the current
    /// point.
    pub saddle_reach: f64,
    /// Saddle searches: longest step of one iteration of the walk, in length scales.
    pub saddle_walk_step: f64,
    /// Saddle searches: relative noise of the fit to what the points show beyond the
    /// quadratic model.
    pub saddle_noise: f64,
}

impl Default for GpMinParams {
    fn default() -> Self {
        Self {
            length_scale: 0.4 * Units::ANGSTROM_TO_BOHR,
            noise: 0.002,
            max_points: 25,
            model_iterations: 200,
            model_tolerance: 1e-9,
            saddle_reach: 1.0,
            saddle_walk_step: 0.25,
            saddle_noise: 1e-4,
        }
    }
}

/// GPMin step generator.
#[derive(Debug, Clone)]
pub struct GpMin {
    params: GpMinParams,
    samples: Vec<Observation>,
    /// Current position relative to the first point.
    position: Option<Col<f64>>,
}

impl GpMin {
    /// New GPMin generator.
    pub fn new(params: GpMinParams) -> Self {
        Self {
            params,
            samples: Vec::new(),
            position: None,
        }
    }

    /// Number of points in the training set.
    pub fn training_points(&self) -> usize {
        self.samples.len()
    }

    /// The point that is always kept: the lowest energy in a minimization, the smallest
    /// gradient in a saddle search.
    fn best(&self, saddle_order: usize) -> usize {
        let score = |a: usize| {
            if saddle_order == 0 {
                self.samples[a].value
            } else {
                col_norm(&self.samples[a].gradient)
            }
        };
        (0..self.samples.len())
            .min_by(|&a, &b| score(a).total_cmp(&score(b)))
            .unwrap_or(0)
    }

    /// The step to the stationary point of order `input.saddle_order` of the surface "quadratic
    /// model at `position` plus Gaussian process of the remainder" (see the module
    /// documentation).
    fn saddle_step(&self, input: &StepInput<'_>, position: &Col<f64>) -> MoptResult<Col<f64>> {
        let n = position.nrows();
        let mut h0 = match input.bias_hessian {
            Some(b) => input.hessian + b,
            None => input.hessian.clone(),
        };
        symmetrize(&mut h0);
        let (e0, g0) = (input.energy.value(), input.gradient);
        // What the points show beyond the quadratic model.
        let remainder: Vec<Observation> = self
            .samples
            .iter()
            .map(|s| {
                let d = &s.x - position;
                let hd = &h0 * &d;
                Observation {
                    x: s.x.clone(),
                    value: s.value - e0 - col_dot(g0, &d) - 0.5 * col_dot(&d, &hd),
                    gradient: &s.gradient - g0 - &hd,
                }
            })
            .collect();
        let gp = GradientGp::fit(
            &remainder,
            self.params.length_scale,
            self.params.saddle_noise,
            0.0,
        )?;
        // With noise the fit does not pass through the data exactly. The gradient at the
        // current point is known, so the surface is made to reproduce it: otherwise its
        // stationary point is not that of the function, however close the search has come.
        let offset = gp.predict(position).1;
        let surface = |x: &Col<f64>| -> (Col<f64>, Mat<f64>) {
            let d = x - position;
            let mut g = g0 + &h0 * &d + gp.predict(x).1 - &offset;
            let mut h = &h0 + gp.hessian(x);
            if let Some(basis) = input.rigid_body_basis {
                g = project_out_basis_vec(basis, &g);
                h = project_out_basis_mat(basis, &h);
            }
            symmetrize(&mut h);
            (g, h)
        };
        let reach = self.params.saddle_reach * self.params.length_scale;
        let walk_step = self.params.saddle_walk_step * self.params.length_scale;
        let rfo = RfoParams::default();
        let mut x = position.clone();
        for _ in 0..self.params.model_iterations {
            let (g, h) = surface(&x);
            if col_norm(&g) < self.params.model_tolerance {
                break;
            }
            let Ok((values, vectors)) = symmetric_eigen(&h) else {
                break;
            };
            let live: Vec<usize> = (0..n).filter(|&k| values[k].abs() >= FLAT_MODE).collect();
            // Image function: the lowest `saddle_order` modes change sign.
            let sign = |rank: usize| if rank < input.saddle_order { -1.0 } else { 1.0 };
            let curvature: Vec<f64> = live
                .iter()
                .enumerate()
                .map(|(rank, &k)| sign(rank) * values[k])
                .collect();
            let slope: Vec<f64> = live
                .iter()
                .enumerate()
                .map(|(rank, &k)| sign(rank) * (0..n).map(|r| vectors[(r, k)] * g[r]).sum::<f64>())
                .collect();
            let along = rs_step(&curvature, &slope, walk_step, &rfo);
            if !along.iter().all(|v| v.is_finite()) {
                break;
            }
            for (j, &k) in live.iter().enumerate() {
                for r in 0..n {
                    x[r] += vectors[(r, k)] * along[j];
                }
            }
            let gone = col_norm(&(&x - position));
            if gone > reach {
                let scale = reach / gone;
                x = Col::from_fn(n, |r| position[r] + (x[r] - position[r]) * scale);
                break;
            }
            if col_norm(&along) < 1e-12 {
                break;
            }
        }
        Ok(&x - position)
    }
}

impl Default for GpMin {
    fn default() -> Self {
        Self::new(GpMinParams::default())
    }
}

impl StepGenerator for GpMin {
    fn name(&self) -> &str {
        "gpmin"
    }

    /// The prior of the surrogate takes the curvature of the bias potentials.
    fn uses_bias_hessian(&self) -> bool {
        true
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let n = input.gradient.nrows();
        let position = match (self.position.take(), input.previous) {
            (Some(p), Some(prev)) if p.nrows() == n => p + &prev.step,
            _ => {
                self.samples.clear();
                Col::zeros(n)
            }
        };
        self.samples.push(Observation {
            x: position.clone(),
            value: input.energy.value(),
            gradient: input.gradient.clone(),
        });
        // Keep the newest points and the best one.
        while self.samples.len() > self.params.max_points.max(2) {
            let best = self.best(input.saddle_order);
            self.samples.remove(usize::from(best == 0));
        }
        if input.saddle_order > 0 {
            let step = self.saddle_step(input, &position)?;
            self.position = Some(position);
            return Ok(step);
        }
        let best = self.best(0);
        let prior = self
            .samples
            .iter()
            .map(|s| s.value)
            .fold(f64::NEG_INFINITY, f64::max);
        let gp = GradientGp::fit(
            &self.samples,
            self.params.length_scale,
            self.params.noise,
            prior,
        )?;
        let options = MinimizeOptions {
            gradient_tolerance: self.params.model_tolerance,
            max_iterations: self.params.model_iterations,
            ..MinimizeOptions::default()
        };
        let found = minimize(|x| gp.predict(x), &self.samples[best].x, &options, None);
        let mut step = &found.x - &position;
        if col_norm(&step) == 0.0 {
            // The model has its minimum at the current point although the gradient does not
            // vanish (only possible through the noise term): a short move along the force.
            let g = input.gradient;
            let len = col_norm(g);
            step = if len > 0.0 {
                g * (-self.params.length_scale.min(1.0) / len * 1e-2)
            } else {
                Col::zeros(n)
            };
        }
        self.position = Some(position);
        Ok(step)
    }

    fn reset(&mut self) {
        self.samples.clear();
        self.position = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::step::PreviousPoint;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn model_minimum_is_approached_and_bad_steps_return_to_the_best_point() {
        // Anharmonic bowl: f = sum_i k_i (x_i - c_i)^2 / 2 + 0.2 (x_0 - c_0)^4.
        let k = [1.0, 3.0, 0.5];
        let c = [0.6, -0.4, 0.3];
        let f = |x: &Col<f64>| {
            let d = Col::from_fn(3, |i| x[i] - c[i]);
            let e = (0..3).map(|i| 0.5 * k[i] * d[i] * d[i]).sum::<f64>() + 0.2 * d[0].powi(4);
            let g = Col::from_fn(3, |i| {
                k[i] * d[i] + if i == 0 { 0.8 * d[0].powi(3) } else { 0.0 }
            });
            (e, g)
        };
        let h = Mat::<f64>::identity(3, 3);
        let mut opt = GpMin::default();
        let mut x: Col<f64> = Col::zeros(3);
        let mut prev: Option<PreviousPoint> = None;
        let mut best = f64::INFINITY;
        let mut returned_to_best = true;
        let mut best_x = x.clone();
        for it in 0..25 {
            let (e, g) = f(&x);
            let s = opt
                .propose(&plain_input(it, &x, e, &g, &h, prev.as_ref(), 0))
                .unwrap();
            if e < best {
                best = e;
                best_x = x.clone();
            } else {
                // After an uphill step the next proposal starts from the best point: it is
                // closer to the best point than the rejected point was.
                let next = &x + &s;
                returned_to_best &=
                    col_norm(&(&next - &best_x)) < col_norm(&(&x - &best_x)) + 1e-12;
            }
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::new(e),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
        }
        let (_, g) = f(&x);
        assert!(col_norm(&g) < 1e-4, "|g| = {}", col_norm(&g));
        assert!(returned_to_best);
        assert_eq!(opt.training_points(), 25);
        // The first step is downhill and about one length scale long at most.
        let mut opt = GpMin::default();
        let (e, g) = f(&Col::zeros(3));
        let s = opt
            .propose(&plain_input(0, &Col::zeros(3), e, &g, &h, None, 0))
            .unwrap();
        assert!(col_dot(&s, &g) < 0.0 && col_norm(&s) < 2.0 * GpMinParams::default().length_scale);
    }

    /// Saddle of order 2 at `c`: curvatures -1, -0.6, 2 there, with cubic and quartic terms.
    fn second_order_saddle(x: &Col<f64>) -> (f64, Col<f64>, Mat<f64>) {
        let c = [0.5, -0.3, 0.4];
        let d = [x[0] - c[0], x[1] - c[1], x[2] - c[2]];
        let e = -0.5 * d[0] * d[0] - 0.3 * d[1] * d[1]
            + d[2] * d[2]
            + 0.3 * d[0] * d[0] * d[2]
            + 0.2 * d[1].powi(3)
            + 0.1 * d[0].powi(4);
        let g = Col::from_fn(3, |i| match i {
            0 => -d[0] + 0.6 * d[0] * d[2] + 0.4 * d[0].powi(3),
            1 => -0.6 * d[1] + 0.6 * d[1] * d[1],
            _ => 2.0 * d[2] + 0.3 * d[0] * d[0],
        });
        let mut h = Mat::<f64>::zeros(3, 3);
        h[(0, 0)] = -1.0 + 0.6 * d[2] + 1.2 * d[0] * d[0];
        h[(1, 1)] = -0.6 + 1.2 * d[1];
        h[(2, 2)] = 2.0;
        h[(0, 2)] = 0.6 * d[0];
        h[(2, 0)] = 0.6 * d[0];
        (e, g, h)
    }

    #[test]
    fn saddles_of_higher_order_from_one_hessian_at_the_start() {
        let start = Col::from_fn(3, |i| [0.15, -0.05, 0.75][i]);
        // The Hessian is taken once, at the start; the surface learns the rest.
        let h = second_order_saddle(&start).2;
        let mut opt = GpMin::default();
        let mut x = start.clone();
        let mut prev: Option<PreviousPoint> = None;
        let mut iterations = 0;
        for it in 0..40 {
            let (e, g, _) = second_order_saddle(&x);
            if col_norm(&g) < 1e-5 {
                break;
            }
            let s = opt
                .propose(&plain_input(it, &x, e, &g, &h, prev.as_ref(), 2))
                .unwrap();
            // No step is longer than the reach of the surface.
            assert!(col_norm(&s) <= GpMinParams::default().length_scale * (1.0 + 1e-9));
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::new(e),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
            iterations = it + 1;
        }
        let (_, g, exact) = second_order_saddle(&x);
        assert!(
            col_norm(&g) < 1e-5,
            "|g| = {} after {iterations}",
            col_norm(&g)
        );
        // Five iterations when measured; RS-I-RFO with this one Hessian does not get there.
        assert!(iterations <= 8, "{iterations}");
        let (values, _) = symmetric_eigen(&exact).unwrap();
        assert!(values[1] < 0.0 && values[2] > 0.0);
        for (i, c) in [0.5, -0.3, 0.4].iter().enumerate() {
            assert!((x[i] - c).abs() < 1e-4, "{i}: {}", x[i]);
        }
    }
}
