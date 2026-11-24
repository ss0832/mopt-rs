//! Step refinement on a Gaussian-process model of the surface (Denzel and Kastner, J. Chem.
//! Phys. 2018, 148, 094114; Rasmussen and Williams, Gaussian Processes for Machine Learning,
//! 2006).
//!
//! The energies and gradients of the latest points train a Gaussian process
//! ([`crate::math::gp::GradientGp`]: squared-exponential kernel, prior mean at the highest
//! energy seen). The proposed step `s` defines a trust ball of radius `2 |s|` around the
//! current point; the model is minimized inside this ball, starting from `x + s`, and the
//! minimizer replaces the step if the model predicts an energy below the current one and below
//! that of `x + s`. The refinement therefore never moves farther than twice the step of the
//! base method, uses no additional evaluations, and is deterministic. Saddle searches pass
//! unchanged, and nothing happens before three points are known.
//!
//! MultiOptPy's `GPRStep` fits the energies only (no gradients), which says little about a
//! surface of `3N` dimensions from 25 points, picks the best of 25 random candidate steps by a
//! lower-confidence bound (different in every run), and blends it into the base step under
//! opposite sign conventions. See `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/gpr_step.py:GPRStep`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::StepEnhancer;
use crate::domain::Units;
use crate::error::MoptResult;
use crate::math::gp::{GradientGp, Observation};
use crate::math::linalg::col_norm;
use crate::math::minimize::{minimize, MinimizeOptions};
use crate::opt::step::StepInput;

/// Parameters of the Gaussian-process step refinement.
#[derive(Debug, Clone, Copy)]
pub struct GprStepParams {
    /// Kernel length scale (Bohr, or the unit of the variables).
    pub length_scale: f64,
    /// Relative noise of the observations.
    pub noise: f64,
    /// Points needed before the model is used.
    pub min_points: usize,
    /// Number of most recent points kept for training.
    pub max_points: usize,
    /// Iteration limit of the model minimization.
    pub model_iterations: usize,
    /// Radius of the trust ball in units of the length of the proposed step.
    pub radius_factor: f64,
}

impl Default for GprStepParams {
    fn default() -> Self {
        Self {
            length_scale: 0.4 * Units::ANGSTROM_TO_BOHR,
            noise: 0.002,
            min_points: 3,
            max_points: 12,
            model_iterations: 100,
            radius_factor: 2.0,
        }
    }
}

/// Gaussian-process step enhancer.
#[derive(Debug, Clone)]
pub struct GprStep {
    params: GprStepParams,
    samples: Vec<Observation>,
    position: Option<Col<f64>>,
    refined: usize,
}

impl GprStep {
    /// New enhancer.
    pub fn new(params: GprStepParams) -> Self {
        Self {
            params,
            samples: Vec::new(),
            position: None,
            refined: 0,
        }
    }

    /// Number of steps replaced by the model minimizer so far.
    pub fn refined_steps(&self) -> usize {
        self.refined
    }
}

impl Default for GprStep {
    fn default() -> Self {
        Self::new(GprStepParams::default())
    }
}

impl StepEnhancer for GprStep {
    fn name(&self) -> &str {
        "gpr_step"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let n = step.nrows();
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
        if self.samples.len() > self.params.max_points.max(1) {
            self.samples.remove(0);
        }
        self.position = Some(position.clone());
        let radius = self.params.radius_factor * col_norm(&step);
        if input.saddle_order > 0 || self.samples.len() < self.params.min_points || radius == 0.0 {
            return Ok(step);
        }
        let prior = self
            .samples
            .iter()
            .map(|s| s.value)
            .fold(f64::NEG_INFINITY, f64::max);
        // A model that cannot be built leaves the base step.
        let Ok(gp) = GradientGp::fit(
            &self.samples,
            self.params.length_scale,
            self.params.noise,
            prior,
        ) else {
            return Ok(step);
        };
        let start = &position + &step;
        let options = MinimizeOptions {
            max_iterations: self.params.model_iterations,
            ..MinimizeOptions::default()
        };
        let found = minimize(
            |x| gp.predict(x),
            &start,
            &options,
            Some((&position, radius)),
        );
        let refined = &found.x - &position;
        let at_base = gp.predict(&start).0;
        let usable = refined.iter().all(|v| v.is_finite())
            && found.value.is_finite()
            && found.value < input.energy.value()
            && found.value < at_base
            && col_norm(&refined) <= radius * (1.0 + 1e-9);
        if usable {
            self.refined += 1;
            Ok(refined)
        } else {
            Ok(step)
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::linalg::col_dot;
    use crate::opt::step::PreviousPoint;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn refined_steps_stay_in_the_ball_and_speed_up_steepest_descent() {
        // Ill-conditioned bowl: steepest descent with a safe factor is slow along x.
        let k = [0.2, 4.0];
        let f = |x: &Col<f64>| {
            let e = 0.5 * (k[0] * x[0] * x[0] + k[1] * x[1] * x[1]);
            (e, Col::from_fn(2, |i| k[i] * x[i]))
        };
        let h = Mat::<f64>::identity(2, 2);
        let run = |with_gpr: bool| -> (f64, usize) {
            let mut gpr = GprStep::default();
            let mut x = Col::from_fn(2, |i| [1.5, 0.4][i]);
            let mut prev: Option<PreviousPoint> = None;
            for it in 0..40 {
                let (e, g) = f(&x);
                let base = &g * (-0.2);
                let s = if with_gpr {
                    let s = gpr
                        .enhance(
                            &plain_input(it, &x, e, &g, &h, prev.as_ref(), 0),
                            base.clone(),
                        )
                        .unwrap();
                    assert!(col_norm(&s) <= 2.0 * col_norm(&base) * (1.0 + 1e-8));
                    s
                } else {
                    base
                };
                prev = Some(PreviousPoint {
                    energy: crate::domain::Quantity::new(e),
                    gradient: g.clone(),
                    raw_gradient: g,
                    step: s.clone(),
                });
                x += &s;
            }
            (f(&x).0, gpr.refined_steps())
        };
        let (plain, _) = run(false);
        let (refined, count) = run(true);
        assert!(count > 5, "{count} refined steps");
        assert!(refined < 0.5 * plain, "{refined} vs {plain}");
    }

    #[test]
    fn the_base_step_is_kept_without_enough_points_and_for_saddle_searches() {
        let h = Mat::<f64>::identity(2, 2);
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [0.3, -0.1][i]);
        let s = &g * (-0.1);
        let mut gpr = GprStep::default();
        assert_eq!(
            gpr.enhance(&plain_input(0, &x, 0.0, &g, &h, None, 0), s.clone())
                .unwrap(),
            s
        );
        let mut gpr = GprStep::new(GprStepParams {
            min_points: 1,
            ..GprStepParams::default()
        });
        assert_eq!(
            gpr.enhance(&plain_input(0, &x, 0.0, &g, &h, None, 1), s.clone())
                .unwrap(),
            s
        );
        // One point is enough for a model with gradients: the refined step is downhill.
        let out = gpr
            .enhance(&plain_input(0, &x, 0.0, &g, &h, None, 0), s.clone())
            .unwrap();
        assert!(col_dot(&out, &g) < 0.0);
    }
}
