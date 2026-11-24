//! Adam: adaptive moment estimation (Kingma and Ba, arXiv:1412.6980).
//!
//! With the moments `m <- b1 m + (1 - b1) g`, `v <- b2 v + (1 - b2) g^2` and their bias
//! corrections `m^ = m / (1 - b1^t)`, `v^ = (v + eps) / (1 - b2^t)`, the step is
//! `dx = -delta m^ / sqrt(v^ + eps)`. MultiOptPy places `eps = 1e-8` inside the square root
//! (twice) where the paper adds it to the root; the port keeps MultiOptPy's form, in which the
//! denominator never falls below `1e-4` and the step shrinks with the gradient near a minimum.
//!
//! MultiOptPy: `Optimizer/adam.py:Adam`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;

/// Adam parameters.
#[derive(Debug, Clone, Copy)]
pub struct AdamParams {
    /// Step size `delta`.
    pub delta: f64,
    /// Decay of the first moment.
    pub beta_m: f64,
    /// Decay of the second moment.
    pub beta_v: f64,
    /// Regularization `eps`.
    pub epsilon: f64,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            delta: 0.03,
            beta_m: 0.9,
            beta_v: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Adam step generator.
#[derive(Debug, Clone)]
pub struct Adam {
    params: AdamParams,
    m: Option<Col<f64>>,
    v: Option<Col<f64>>,
    count: i32,
}

impl Adam {
    /// New Adam generator.
    pub fn new(params: AdamParams) -> Self {
        Self {
            params,
            m: None,
            v: None,
            count: 1,
        }
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new(AdamParams::default())
    }
}

impl StepGenerator for Adam {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.params.delta = delta;
        true
    }

    fn name(&self) -> &str {
        "adam"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let p = self.params;
        let g = input.gradient;
        let n = g.nrows();
        let m0 = self.m.take().unwrap_or_else(|| Col::zeros(n));
        let v0 = self.v.take().unwrap_or_else(|| Col::zeros(n));
        let m = Col::from_fn(n, |i| p.beta_m * m0[i] + (1.0 - p.beta_m) * g[i]);
        let v = Col::from_fn(n, |i| p.beta_v * v0[i] + (1.0 - p.beta_v) * g[i] * g[i]);
        // 1 - beta^t > 0 for t >= 1.
        let cm = 1.0 - p.beta_m.powi(self.count);
        let cv = 1.0 - p.beta_v.powi(self.count);
        let step = Col::from_fn(n, |i| {
            let v_hat = (v[i] + p.epsilon) / cv;
            -p.delta * (m[i] / cm) / libm::sqrt(v_hat + p.epsilon)
        });
        self.m = Some(m);
        self.v = Some(v);
        self.count = self.count.saturating_add(1);
        Ok(step)
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
    fn first_step_has_length_delta_per_component_for_large_gradients() {
        let h = Mat::identity(2, 2);
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [3.0, -0.5][i]);
        let mut adam = Adam::default();
        let s = adam
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 0))
            .unwrap();
        // m^ = g, v^ = g^2 + eps / (1 - b2): |dx| = delta up to the regularization.
        assert!((s[0] + 0.03).abs() < 1e-6 && (s[1] - 0.03).abs() < 1e-5);
        // A vanishing gradient gives a vanishing step (no division by zero).
        let zero = Col::zeros(2);
        let mut adam = Adam::default();
        let s = adam
            .propose(&plain_input(0, &x, 0.0, &zero, &h, None, 0))
            .unwrap();
        assert!(s[0] == 0.0 && s[1] == 0.0);
    }
}
