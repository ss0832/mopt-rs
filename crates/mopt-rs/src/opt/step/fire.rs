//! Fast inertial relaxation engine (Bitzek et al., Phys. Rev. Lett. 2006, 97, 170201), as
//! implemented in MultiOptPy's `Optimizer/fire.py`.
//!
//! MultiOptPy: `Optimizer/fire.py:FIRE`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm};

/// FIRE parameters.
#[derive(Debug, Clone, Copy)]
pub struct FireParams {
    /// Number of downhill steps before the time step may grow.
    pub n_acc: usize,
    /// Time-step growth factor.
    pub f_inc: f64,
    /// Mixing decay factor.
    pub f_acc: f64,
    /// Time-step reduction factor on uphill motion.
    pub f_dec: f64,
    /// Maximum time step.
    pub dt_max: f64,
    /// Initial mixing parameter.
    pub alpha_start: f64,
    /// Initial time step.
    pub dt_start: f64,
}

impl Default for FireParams {
    fn default() -> Self {
        Self {
            n_acc: 5,
            f_inc: 1.10,
            f_acc: 0.99,
            f_dec: 0.50,
            dt_max: 0.8,
            alpha_start: 0.1,
            dt_start: 0.1,
        }
    }
}

/// FIRE step generator.
#[derive(Debug, Clone)]
pub struct Fire {
    params: FireParams,
    dt: f64,
    alpha: f64,
    n_reset: usize,
    velocity: Option<Col<f64>>,
}

impl Fire {
    /// New FIRE generator.
    pub fn new(params: FireParams) -> Self {
        Self {
            params,
            dt: params.dt_start,
            alpha: params.alpha_start,
            n_reset: 0,
            velocity: None,
        }
    }
}

impl Default for Fire {
    fn default() -> Self {
        Self::new(FireParams::default())
    }
}

impl StepGenerator for Fire {
    fn name(&self) -> &str {
        "fire"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let g = input.gradient;
        let n = g.nrows();
        let pre_v = self.velocity.take().unwrap_or_else(|| Col::zeros(n));
        let g_norm = col_norm(g);
        let ratio = if g_norm > 0.0 {
            col_norm(&pre_v) / g_norm
        } else {
            0.0
        };
        // MultiOptPy's "velocity" is parallel to the gradient; the step is its negative.
        let mut v = &pre_v * (1.0 - self.alpha) + g * (self.alpha * ratio);
        if col_dot(&pre_v, g) > 0.0 {
            if self.n_reset > self.params.n_acc {
                self.dt = (self.dt * self.params.f_inc).min(self.params.dt_max);
                self.alpha *= self.params.f_acc;
            }
            self.n_reset += 1;
        } else {
            v = Col::zeros(n);
            self.alpha = self.params.alpha_start;
            self.dt *= self.params.f_dec;
            self.n_reset = 0;
        }
        v += g * self.dt;
        let step = &v * (-self.dt);
        self.velocity = Some(v);
        Ok(step)
    }

    fn reset(&mut self) {
        *self = Self::new(self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn first_steps_follow_reference_recurrence() {
        let x = Col::zeros(2);
        let h = Mat::identity(2, 2);
        let g = Col::from_fn(2, |i| [1.0, -2.0][i]);
        let input = StepInput {
            iteration: 0,
            cartesian: &x,
            energy: crate::domain::Quantity::zero(),
            gradient: &g,
            raw_gradient: &g,
            previous: None,
            hessian: &h,
            bias_hessian: None,
            rigid_body_basis: None,
            project_updates: true,
            saddle_order: 0,
            system: None,
            cartesian_coordinates: true,
            frame: None,
            cartesian_gradient: None,
        };
        let mut fire = Fire::default();
        // Iteration 0: dt -> 0.05, v = dt g, step = -dt^2 g.
        let s0 = fire.propose(&input).unwrap();
        assert!((s0[0] + 0.0025).abs() < 1e-15 && (s0[1] - 0.005).abs() < 1e-15);
        // Iteration 1: v.g > 0 -> keep dt; v = 0.9 v + 0.1 (|v|/|g|) g + dt g.
        let s1 = fire.propose(&input).unwrap();
        let v1 = 0.9 * 0.05 + 0.1 * 0.05 + 0.05;
        assert!((s1[0] + 0.05 * v1).abs() < 1e-15);
    }
}
