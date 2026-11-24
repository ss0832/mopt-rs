//! Prodigy: Adam with a learned distance to the solution (Mishchenko and Defazio,
//! arXiv:2306.06101, Algorithm 4).
//!
//! The estimate `d` of the distance from the start to the solution scales the step and only
//! grows:
//! ```text
//! m <- b1 m + (1 - b1) d g                 v <- b2 v + (1 - b2) d^2 g^2
//! r <- sqrt(b2) r + (1 - sqrt(b2)) gamma d^2 <g, x0 - x>
//! s <- sqrt(b2) s + (1 - sqrt(b2)) gamma d^2 g
//! dx = -gamma d m / (sqrt(v) + d eps)       d <- max(d, r / ||s||_1)
//! ```
//! `gamma = 0.1` and the initial `d = 0.03` are MultiOptPy's values. MultiOptPy never stores the
//! updated `s`, so its `||s||_1` holds one term instead of the running sum; the estimate then
//! is no longer the lower bound of the distance that the paper derives (it exceeded the true
//! distance in a test, see `docs/source/migration_optimizers.md`). The port accumulates `s` as the
//! algorithm defines.
//!
//! MultiOptPy: `Optimizer/prodigy.py:Prodigy`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;

/// Prodigy parameters.
#[derive(Debug, Clone, Copy)]
pub struct ProdigyParams {
    /// Step size `gamma`.
    pub gamma: f64,
    /// Initial distance estimate `d0`.
    pub d0: f64,
    /// Decay of the first moment.
    pub beta_m: f64,
    /// Decay of the second moment.
    pub beta_v: f64,
    /// Regularization `eps`.
    pub epsilon: f64,
}

impl Default for ProdigyParams {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            d0: 0.03,
            beta_m: 0.9,
            beta_v: 0.999,
            epsilon: 1e-12,
        }
    }
}

struct State {
    m: Col<f64>,
    v: Col<f64>,
    s: Col<f64>,
    r: f64,
    /// `x - x0`, accumulated from the steps actually taken.
    travelled: Col<f64>,
}

/// Prodigy step generator.
pub struct Prodigy {
    params: ProdigyParams,
    d: f64,
    state: Option<State>,
}

impl Prodigy {
    /// New Prodigy generator.
    pub fn new(params: ProdigyParams) -> Self {
        Self {
            params,
            d: params.d0,
            state: None,
        }
    }

    /// Current estimate of the distance to the solution.
    pub fn distance_estimate(&self) -> f64 {
        self.d
    }
}

impl Default for Prodigy {
    fn default() -> Self {
        Self::new(ProdigyParams::default())
    }
}

impl StepGenerator for Prodigy {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.params.gamma = delta;
        true
    }

    fn name(&self) -> &str {
        "prodigy"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let p = self.params;
        let g = input.gradient;
        let n = g.nrows();
        let mut st = self.state.take().unwrap_or_else(|| State {
            m: Col::zeros(n),
            v: Col::zeros(n),
            s: Col::zeros(n),
            r: 0.0,
            travelled: Col::zeros(n),
        });
        if let Some(prev) = input.previous {
            st.travelled += &prev.step;
        }
        let d = self.d;
        let root = libm::sqrt(p.beta_v);
        let w = (1.0 - root) * p.gamma * d * d;
        let mut g_dot_back = 0.0;
        for i in 0..n {
            st.m[i] = p.beta_m * st.m[i] + (1.0 - p.beta_m) * d * g[i];
            st.v[i] = p.beta_v * st.v[i] + (1.0 - p.beta_v) * d * d * g[i] * g[i];
            st.s[i] = root * st.s[i] + w * g[i];
            g_dot_back -= g[i] * st.travelled[i];
        }
        st.r = root * st.r + w * g_dot_back;
        // m_i = 0 whenever v_i = 0, so the quotient is 0 there (eps > 0).
        let step = Col::from_fn(n, |i| {
            -p.gamma * d * st.m[i] / (libm::sqrt(st.v[i]) + d * p.epsilon)
        });
        let s_norm: f64 = st.s.iter().map(|v| v.abs()).sum();
        if s_norm > 0.0 {
            let estimate = st.r / s_norm;
            if estimate.is_finite() && estimate > self.d {
                self.d = estimate;
            }
        }
        self.state = Some(st);
        Ok(step)
    }

    fn reset(&mut self) {
        *self = Self::new(self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::step::PreviousPoint;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn distance_estimate_grows_toward_the_true_distance_and_never_shrinks() {
        // f = (x - 5)^2 / 2 from x = 0: the solution is at distance 5.
        let h = Mat::identity(1, 1);
        let mut x = Col::from_fn(1, |_| 0.0);
        let mut opt = Prodigy::default();
        let mut prev: Option<PreviousPoint> = None;
        let mut last_d = opt.distance_estimate();
        for it in 0..4000 {
            let g = Col::from_fn(1, |_| x[0] - 5.0);
            let s = opt
                .propose(&plain_input(it, &x, 0.0, &g, &h, prev.as_ref(), 0))
                .unwrap();
            assert!(opt.distance_estimate() >= last_d);
            last_d = opt.distance_estimate();
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::zero(),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
        }
        // The estimate is a lower bound of the true distance.
        assert!((last_d - 2.46).abs() < 0.05, "d = {last_d}");
        assert!((x[0] - 5.0).abs() < 0.2, "x = {}", x[0]);
        // A vanishing gradient changes nothing and divides by nothing.
        let zero = Col::zeros(1);
        let s = Prodigy::default()
            .propose(&plain_input(0, &x, 0.0, &zero, &h, None, 0))
            .unwrap();
        assert!(s[0] == 0.0);
    }
}
