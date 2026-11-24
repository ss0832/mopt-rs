//! ABC-FIRE: accelerated bias-corrected FIRE (Echeverri Restrepo and Andric, Comput. Mater.
//! Sci. 2023, 218, 111978).
//!
//! FIRE 2.0 (see [`super::fire2`]) whose mixed velocity is bias-corrected like the moments of
//! Adam: with `i` consecutive downhill steps since the last restart,
//! `u <- [(1 - a) u + a |u| F / |F|] / (1 - (1 - a)^(i + 1))`,
//! and each component of the displacement `dt u` is limited to `max_step` separately.
//!
//! MultiOptPy applies the correction to the old velocity only, with the exponent starting at
//! zero, where the factor is `1 / (0 + 1e-10)`; whenever the counter is reset while the velocity
//! is kept, the velocity is multiplied by `1e10`. The port follows the published algorithm, with
//! MultiOptPy's parameters (`n_acc = 5`, `a_start = 0.1`, `dt_max = 0.8`).
//!
//! MultiOptPy: `Optimizer/abc_fire.py:ABC_FIRE`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm};

/// ABC-FIRE parameters.
#[derive(Debug, Clone, Copy)]
pub struct AbcFireParams {
    /// Number of downhill steps before the time step may grow.
    pub n_acc: usize,
    /// Time-step growth factor.
    pub f_inc: f64,
    /// Mixing decay factor.
    pub f_acc: f64,
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
    /// Maximum displacement of one component per step.
    pub max_step: f64,
}

impl Default for AbcFireParams {
    fn default() -> Self {
        Self {
            n_acc: 5,
            f_inc: 1.10,
            f_acc: 0.99,
            f_dec: 0.50,
            dt_max: 0.8,
            dt_min: 0.01,
            dt_start: 0.1,
            alpha_start: 0.1,
            max_step: 0.2,
        }
    }
}

/// Bias correction `1 / (1 - (1 - a)^k)` for `0 < a <= 1`, `k >= 1`, without cancellation for
/// small `a`.
pub fn bias_correction(alpha: f64, k: usize) -> f64 {
    let a = alpha.clamp(1e-10, 1.0);
    // 1 - (1 - a)^k = -expm1(k ln(1 - a)); (1 - a)^k = 0 for a = 1.
    let denom = if a >= 1.0 {
        1.0
    } else {
        -libm::expm1(k as f64 * libm::log1p(-a))
    };
    1.0 / denom
}

/// ABC-FIRE step generator.
#[derive(Debug, Clone)]
pub struct AbcFire {
    params: AbcFireParams,
    dt: f64,
    alpha: f64,
    n_downhill: usize,
    velocity: Option<Col<f64>>,
}

impl AbcFire {
    /// New ABC-FIRE generator.
    pub fn new(params: AbcFireParams) -> Self {
        Self {
            params,
            dt: params.dt_start,
            alpha: params.alpha_start,
            n_downhill: 0,
            velocity: None,
        }
    }
}

impl Default for AbcFire {
    fn default() -> Self {
        Self::new(AbcFireParams::default())
    }
}

impl StepGenerator for AbcFire {
    fn name(&self) -> &str {
        "abcfire"
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
            if self.n_downhill > p.n_acc {
                self.dt = (self.dt * p.f_inc).min(p.dt_max);
                self.alpha *= p.f_acc;
            }
        } else {
            self.n_downhill = 0;
            self.dt = (self.dt * p.f_dec).max(p.dt_min);
            self.alpha = p.alpha_start;
            if power < 0.0 {
                back = &u * (-0.5 * self.dt);
            }
            u = Col::zeros(n);
        }
        u += &force * self.dt;
        let fnorm = col_norm(&force);
        if fnorm > 0.0 {
            let mixed = &u * (1.0 - self.alpha) + &force * (self.alpha * col_norm(&u) / fnorm);
            u = mixed * bias_correction(self.alpha, self.n_downhill + 1);
        }
        let limit = p.max_step / self.dt;
        for i in 0..n {
            u[i] = u[i].clamp(-limit, limit);
        }
        let step = &u * self.dt;
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
    fn bias_correction_is_finite_and_tends_to_one() {
        assert!((bias_correction(0.1, 1) - 10.0).abs() < 1e-12);
        assert!((bias_correction(0.25, 2) - 1.0 / (1.0 - 0.5625)).abs() < 1e-12);
        assert!((bias_correction(0.1, 500) - 1.0).abs() < 1e-12);
        // Tiny mixing parameters: 1 / (k a) without cancellation.
        assert!((bias_correction(1e-12, 3) * 3e-10 - 1.0).abs() < 1e-6);
        assert_eq!(bias_correction(1.0, 4), 1.0);
    }

    #[test]
    fn first_step_is_bias_corrected_and_components_are_limited() {
        let h = Mat::identity(2, 2);
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [1.0, -2.0][i]);
        let mut fire = AbcFire::default();
        // dt = 0.05, u = dt F, corrected by 1 / (1 - 0.9) = 10: dx = 10 dt^2 F.
        let s0 = fire
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 0))
            .unwrap();
        assert!((s0[0] + 0.025).abs() < 1e-14 && (s0[1] - 0.05).abs() < 1e-14);
        let big = Col::from_fn(2, |i| [1e3, -1e-3][i]);
        let mut fire = AbcFire::default();
        let s = fire
            .propose(&plain_input(0, &x, 0.0, &big, &h, None, 0))
            .unwrap();
        assert!((s[0] + 0.2).abs() < 1e-15 && s[1].abs() < 0.2);
    }
}
