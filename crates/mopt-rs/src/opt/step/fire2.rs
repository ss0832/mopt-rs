//! FIRE 2.0 with the semi-implicit Euler integrator (Guenole et al., Comput. Mater. Sci. 2020,
//! 175, 109584).
//!
//! With the velocity `u`, the force `F = -g`, and the power `P = F . u`:
//! - `P > 0`: after `n_min` such steps the time step grows (`dt <- min(dt f_inc, dt_max)`) and
//!   the mixing decays (`a <- a f_alpha`);
//! - `P <= 0`: `dt <- max(dt f_dec, dt_min)`, `a <- a_start`, the position goes back by half a
//!   step (`-dt u / 2`, when `P < 0`), and the velocity is set to zero.
//!
//! Then `u <- u + dt F`, mixed as `u <- (1 - a) u + a |u| F / |F|` when `P > 0`, and
//! `dx = dt u`, shortened to `max_step`.
//!
//! MultiOptPy applies the half step back with the sign of its own convention reversed (it moves
//! half a step further uphill); the port goes back along the previous displacement as the
//! method defines.
//!
//! MultiOptPy: `Optimizer/fire2.py:FIRE2`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm};

/// FIRE 2.0 parameters.
#[derive(Debug, Clone, Copy)]
pub struct Fire2Params {
    /// Number of downhill steps before the time step may grow.
    pub n_min: usize,
    /// Time-step growth factor.
    pub f_inc: f64,
    /// Mixing decay factor.
    pub f_alpha: f64,
    /// Time-step reduction factor.
    pub f_dec: f64,
    /// Maximum time step.
    pub dt_max: f64,
    /// Minimum time step.
    pub dt_min: f64,
    /// Initial time step.
    pub dt_start: f64,
    /// Initial mixing parameter.
    pub alpha_start: f64,
    /// Maximum length of the integration step.
    pub max_step: f64,
    /// Go back half a step when the power turns negative.
    pub half_step_back: bool,
    /// Stop moving (zero steps) after more than this many consecutive steps with `P <= 0`;
    /// `0` disables the limit.
    pub max_uphill_steps: usize,
}

impl Default for Fire2Params {
    fn default() -> Self {
        Self {
            n_min: 5,
            f_inc: 1.10,
            f_alpha: 0.99,
            f_dec: 0.50,
            dt_max: 1.0,
            dt_min: 0.01,
            dt_start: 0.1,
            alpha_start: 0.25,
            max_step: 0.2,
            half_step_back: true,
            max_uphill_steps: 0,
        }
    }
}

/// FIRE 2.0 step generator.
#[derive(Debug, Clone)]
pub struct Fire2 {
    params: Fire2Params,
    dt: f64,
    alpha: f64,
    n_downhill: usize,
    n_uphill: usize,
    velocity: Option<Col<f64>>,
}

impl Fire2 {
    /// New FIRE 2.0 generator.
    pub fn new(params: Fire2Params) -> Self {
        Self {
            params,
            dt: params.dt_start,
            alpha: params.alpha_start,
            n_downhill: 0,
            n_uphill: 0,
            velocity: None,
        }
    }

    /// Current time step.
    pub fn time_step(&self) -> f64 {
        self.dt
    }
}

impl Default for Fire2 {
    fn default() -> Self {
        Self::new(Fire2Params::default())
    }
}

impl StepGenerator for Fire2 {
    fn name(&self) -> &str {
        "fire2"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let p = self.params;
        let force = -input.gradient;
        let n = force.nrows();
        let mut u = self.velocity.take().unwrap_or_else(|| Col::zeros(n));
        let power = col_dot(&force, &u);
        let mut back: Col<f64> = Col::zeros(n);
        if power > 0.0 {
            self.n_downhill += 1;
            self.n_uphill = 0;
            if self.n_downhill > p.n_min {
                self.dt = (self.dt * p.f_inc).min(p.dt_max);
                self.alpha *= p.f_alpha;
            }
        } else {
            self.n_downhill = 0;
            self.dt = (self.dt * p.f_dec).max(p.dt_min);
            self.alpha = p.alpha_start;
            self.n_uphill += 1;
            if p.max_uphill_steps > 0 && self.n_uphill > p.max_uphill_steps {
                self.velocity = Some(u);
                return Ok(Col::zeros(n));
            }
            if p.half_step_back && power < 0.0 {
                back = &u * (-0.5 * self.dt);
            }
            u = Col::zeros(n);
        }
        u += &force * self.dt;
        if power > 0.0 {
            let (un, fnorm) = (col_norm(&u), col_norm(&force));
            if un > 1e-10 && fnorm > 1e-10 {
                u = &u * (1.0 - self.alpha) + &force * (self.alpha * un / fnorm);
            }
        }
        let mut step = &u * self.dt;
        let len = col_norm(&step);
        if len > p.max_step {
            step *= p.max_step / len;
        }
        self.velocity = Some(u);
        Ok(back + step)
    }

    fn reset(&mut self) {
        *self = Self::new(self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn first_steps_and_half_step_back() {
        let h = Mat::identity(2, 2);
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [1.0, -2.0][i]);
        let mut fire = Fire2::default();
        // Start: P = 0 -> dt = 0.05, u = dt F, dx = dt^2 F.
        let s0 = fire
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 0))
            .unwrap();
        assert!((s0[0] + 0.0025).abs() < 1e-15 && (s0[1] - 0.005).abs() < 1e-15);
        // Downhill: u = 2 dt F is parallel to F, so the mixing leaves it unchanged.
        let s1 = fire
            .propose(&plain_input(1, &x, 0.0, &g, &h, None, 0))
            .unwrap();
        assert!((s1[0] + 2.0 * 0.0025).abs() < 1e-15);
        // The force reverses: P < 0 -> dt = 0.025, back by dt u / 2, restart along the force.
        let up = Col::from_fn(2, |i| [-1.0, 2.0][i]);
        let s2 = fire
            .propose(&plain_input(2, &x, 0.0, &up, &h, None, 0))
            .unwrap();
        assert!((fire.time_step() - 0.025).abs() < 1e-15);
        // u before = 2 * 0.05 * F_old = (-0.1, 0.2); back = -0.5 * 0.025 * u = (0.00125, -0.0025).
        let want0 = 0.00125 + 0.025 * 0.025 * 1.0;
        assert!((s2[0] - want0).abs() < 1e-15, "{}", s2[0]);
        // The back step opposes the previous displacement.
        assert!((s2[0] - 0.025 * 0.025) * s1[0] < 0.0);
    }

    #[test]
    fn long_steps_are_capped_and_uphill_limit_stops_the_motion() {
        let h = Mat::identity(1, 1);
        let x = Col::zeros(1);
        let g = Col::from_fn(1, |_| -1e4);
        let mut fire = Fire2::default();
        let s = fire
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 0))
            .unwrap();
        assert!((s[0] - 0.2).abs() < 1e-15);
        let mut fire = Fire2::new(Fire2Params {
            max_uphill_steps: 1,
            ..Fire2Params::default()
        });
        let zero = Col::zeros(1);
        assert!(fire
            .propose(&plain_input(0, &x, 0.0, &zero, &h, None, 0))
            .is_ok());
        let s = fire
            .propose(&plain_input(1, &x, 0.0, &zero, &h, None, 0))
            .unwrap();
        assert_eq!(s[0], 0.0);
    }
}
