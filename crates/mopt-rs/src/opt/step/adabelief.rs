//! AdaBelief (Zhuang et al., arXiv:2010.07468).
//!
//! The step is scaled by the "belief" in the gradient, the variance of the gradient about its
//! running mean: `m <- b1 m + (1 - b1) g`, `s <- b2 s + (1 - b2)(g - m)^2`,
//! `dx = -delta m / sqrt(s + eps)`. As in MultiOptPy, no bias corrections are applied, so the
//! first steps are about `(1 - b1) / ((1 - b2) b1^2)^(1/2) = 3.5` times `delta` per component.
//!
//! MultiOptPy: `Optimizer/adabelief.py:Adabelief`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;

/// AdaBelief parameters.
#[derive(Debug, Clone, Copy)]
pub struct AdaBeliefParams {
    /// Step size `delta`.
    pub delta: f64,
    /// Decay of the first moment.
    pub beta_m: f64,
    /// Decay of the variance.
    pub beta_v: f64,
    /// Regularization `eps` inside the square root.
    pub epsilon: f64,
}

impl Default for AdaBeliefParams {
    fn default() -> Self {
        Self {
            delta: 0.03,
            beta_m: 0.9,
            beta_v: 0.999,
            epsilon: 1e-15,
        }
    }
}

/// AdaBelief step generator.
#[derive(Debug, Clone)]
pub struct AdaBelief {
    params: AdaBeliefParams,
    m: Option<Col<f64>>,
    s: Option<Col<f64>>,
}

impl AdaBelief {
    /// New AdaBelief generator.
    pub fn new(params: AdaBeliefParams) -> Self {
        Self {
            params,
            m: None,
            s: None,
        }
    }
}

impl Default for AdaBelief {
    fn default() -> Self {
        Self::new(AdaBeliefParams::default())
    }
}

impl StepGenerator for AdaBelief {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.params.delta = delta;
        true
    }

    fn name(&self) -> &str {
        "adabelief"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let p = self.params;
        let g = input.gradient;
        let n = g.nrows();
        let m0 = self.m.take().unwrap_or_else(|| Col::zeros(n));
        let s0 = self.s.take().unwrap_or_else(|| Col::zeros(n));
        let m = Col::from_fn(n, |i| p.beta_m * m0[i] + (1.0 - p.beta_m) * g[i]);
        let s = Col::from_fn(n, |i| {
            p.beta_v * s0[i] + (1.0 - p.beta_v) * (g[i] - m[i]).powi(2)
        });
        // eps > 0 keeps the denominator positive; m = 0 gives a zero step.
        let step = Col::from_fn(n, |i| -p.delta * m[i] / libm::sqrt(s[i] + p.epsilon));
        self.m = Some(m);
        self.s = Some(s);
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
    fn first_step_follows_the_recurrence() {
        let h = Mat::identity(2, 2);
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [2.0, -0.01][i]);
        let s = AdaBelief::default()
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 0))
            .unwrap();
        // m = 0.1 g, s = 0.001 (0.9 g)^2: dx = -delta 0.1 / (0.9 sqrt(0.001)) sign(g).
        let want = 0.03 * 0.1 / (0.9 * 0.001_f64.sqrt());
        assert!((s[0] + want).abs() < 1e-9 && (s[1] - want).abs() < 1e-6);
        let zero = Col::zeros(2);
        let s = AdaBelief::default()
            .propose(&plain_input(0, &x, 0.0, &zero, &h, None, 0))
            .unwrap();
        assert!(s[0] == 0.0);
    }
}
