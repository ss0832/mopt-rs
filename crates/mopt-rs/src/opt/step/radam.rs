//! Rectified Adam (Liu et al., arXiv:1908.03265).
//!
//! The adaptive step is used only when the variance of the adaptive learning rate is tractable:
//! with `rho_inf = 2 / (1 - b2) - 1` and `rho_t = rho_inf - 2 t b2^t / (1 - b2^t)`, the step is
//! `dx = -delta r_t m^ sqrt((1 - b2^t) / v)` with the rectification
//! `r_t = sqrt((rho_t - 4)(rho_t - 2) rho_inf / ((rho_inf - 4)(rho_inf - 2) rho_t))` when
//! `rho_t > 4`, and the momentum step `dx = -delta m^` otherwise (the first four iterations).
//!
//! As in MultiOptPy, the second moment is that of the deviation from the first moment,
//! `v <- b2 v + (1 - b2)(g - m)^2 + eps` (the variance estimate of AdaBelief, arXiv:2010.07468),
//! not of the gradient itself.
//!
//! MultiOptPy: `Optimizer/radam.py:RADAM`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;

/// RAdam parameters.
#[derive(Debug, Clone, Copy)]
pub struct RadamParams {
    /// Step size `delta`.
    pub delta: f64,
    /// Decay of the first moment.
    pub beta_m: f64,
    /// Decay of the second moment.
    pub beta_v: f64,
    /// Regularization added to the second moment in every iteration.
    pub epsilon: f64,
}

impl Default for RadamParams {
    fn default() -> Self {
        Self {
            delta: 0.03,
            beta_m: 0.9,
            beta_v: 0.999,
            epsilon: 1e-12,
        }
    }
}

/// RAdam step generator.
#[derive(Debug, Clone)]
pub struct Radam {
    params: RadamParams,
    m: Option<Col<f64>>,
    v: Option<Col<f64>>,
    count: i32,
}

impl Radam {
    /// New RAdam generator.
    pub fn new(params: RadamParams) -> Self {
        Self {
            params,
            m: None,
            v: None,
            count: 1,
        }
    }

    /// Rectification factor `r_t` for iteration `t` (`None` while `rho_t <= 4`).
    pub fn rectification(beta_v: f64, t: i32) -> Option<f64> {
        let rho_inf = 2.0 / (1.0 - beta_v) - 1.0;
        let bt = beta_v.powi(t);
        let rho = rho_inf - 2.0 * f64::from(t) * bt / (1.0 - bt);
        (rho > 4.0).then(|| {
            libm::sqrt(
                (rho - 4.0) * (rho - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho),
            )
        })
    }
}

impl Default for Radam {
    fn default() -> Self {
        Self::new(RadamParams::default())
    }
}

impl StepGenerator for Radam {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.params.delta = delta;
        true
    }

    fn name(&self) -> &str {
        "radam"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let p = self.params;
        let g = input.gradient;
        let n = g.nrows();
        let m0 = self.m.take().unwrap_or_else(|| Col::zeros(n));
        let v0 = self.v.take().unwrap_or_else(|| Col::zeros(n));
        let m = Col::from_fn(n, |i| p.beta_m * m0[i] + (1.0 - p.beta_m) * g[i]);
        let v = Col::from_fn(n, |i| {
            p.beta_v * v0[i] + (1.0 - p.beta_v) * (g[i] - m[i]).powi(2) + p.epsilon
        });
        let cm = 1.0 - p.beta_m.powi(self.count);
        let cv = 1.0 - p.beta_v.powi(self.count);
        let step = match Self::rectification(p.beta_v, self.count) {
            // v >= eps > 0, so the root is finite.
            Some(r) => Col::from_fn(n, |i| -p.delta * r * (m[i] / cm) * libm::sqrt(cv / v[i])),
            None => Col::from_fn(n, |i| -p.delta * m[i] / cm),
        };
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
    fn rectification_starts_at_the_fifth_iteration_and_tends_to_one() {
        for t in 1..=4 {
            assert!(Radam::rectification(0.999, t).is_none(), "t = {t}");
        }
        let r5 = Radam::rectification(0.999, 5).unwrap();
        assert!(r5 > 0.0 && r5 < 0.1, "{r5}");
        let late = Radam::rectification(0.999, 100_000).unwrap();
        assert!((late - 1.0).abs() < 1e-6, "{late}");
        // Unrectified iterations take the momentum step -delta m^ = -delta g.
        let h = Mat::identity(1, 1);
        let x = Col::zeros(1);
        let g = Col::from_fn(1, |_| 2.0);
        let s = Radam::default()
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 0))
            .unwrap();
        assert!((s[0] + 0.06).abs() < 1e-15);
    }
}
