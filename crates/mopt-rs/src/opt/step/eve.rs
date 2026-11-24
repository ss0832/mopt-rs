//! Eve: Adam with feedback from the objective function (Hayashi, Koushik, and Neubig,
//! arXiv:1611.01505).
//!
//! Adam's step is divided by a smoothed relative change of the energy,
//! `d_t = |E_t - E_{t-1}| / |min(E_t, E_{t-1}) - E*|`, clipped to `[1/c, c]` and averaged as
//! `d~ <- b3 d~ + (1 - b3) d_t` (`d~ = 1` at the first iteration):
//! `dx = -(delta / d~) m^ / (sqrt(v^) + eps)`. Large relative changes shorten the steps, small
//! ones lengthen them, by at most the factor `c`.
//!
//! The paper assumes a positive objective with minimum `E*`. MultiOptPy divides by
//! `min(E_t, E_{t-1})` itself, which is negative for electronic energies, so `d_t` is negative
//! and always clipped to `1/c`. The port divides by the magnitude `|min(E_t, E_{t-1}) - E*|`
//! (`E* = 0` unless set), with `d_t = c` when the denominator vanishes.
//!
//! MultiOptPy: `Optimizer/eve.py:EVE`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;

/// Eve parameters.
#[derive(Debug, Clone, Copy)]
pub struct EveParams {
    /// Step size `delta`.
    pub delta: f64,
    /// Decay of the first moment.
    pub beta_m: f64,
    /// Decay of the second moment.
    pub beta_v: f64,
    /// Decay of the smoothed relative change.
    pub beta_d: f64,
    /// Clipping bound `c` of the relative change.
    pub c: f64,
    /// Regularization `eps` added to the root of the second moment.
    pub epsilon: f64,
    /// Reference energy `E*` of the relative change.
    pub reference_energy: f64,
}

impl Default for EveParams {
    fn default() -> Self {
        Self {
            delta: 0.03,
            beta_m: 0.9,
            beta_v: 0.999,
            beta_d: 0.999,
            c: 10.0,
            epsilon: 1e-12,
            reference_energy: 0.0,
        }
    }
}

/// Eve step generator.
#[derive(Debug, Clone)]
pub struct Eve {
    params: EveParams,
    m: Option<Col<f64>>,
    v: Option<Col<f64>>,
    count: i32,
    d_tilde: f64,
    prev_energy: Option<f64>,
}

impl Eve {
    /// New Eve generator.
    pub fn new(params: EveParams) -> Self {
        Self {
            params,
            m: None,
            v: None,
            count: 1,
            d_tilde: 1.0,
            prev_energy: None,
        }
    }

    /// Smoothed relative change `d~` dividing the step size.
    pub fn feedback(&self) -> f64 {
        self.d_tilde
    }

    /// Clipped relative change between two energies.
    pub fn relative_change(&self, energy: f64, prev_energy: f64) -> f64 {
        let c = self.params.c;
        let denom = (energy.min(prev_energy) - self.params.reference_energy).abs();
        let num = (energy - prev_energy).abs();
        // num / denom >= c  <=>  num >= c denom (no division by a vanishing denominator).
        if num >= c * denom {
            if num == 0.0 {
                1.0 / c
            } else {
                c
            }
        } else {
            (num / denom).max(1.0 / c)
        }
    }
}

impl Default for Eve {
    fn default() -> Self {
        Self::new(EveParams::default())
    }
}

impl StepGenerator for Eve {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.params.delta = delta;
        true
    }

    fn name(&self) -> &str {
        "eve"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let p = self.params;
        let g = input.gradient;
        let n = g.nrows();
        let e = input.energy.value();
        let m0 = self.m.take().unwrap_or_else(|| Col::zeros(n));
        let v0 = self.v.take().unwrap_or_else(|| Col::zeros(n));
        let m = Col::from_fn(n, |i| p.beta_m * m0[i] + (1.0 - p.beta_m) * g[i]);
        let v = Col::from_fn(n, |i| p.beta_v * v0[i] + (1.0 - p.beta_v) * g[i] * g[i]);
        let cm = 1.0 - p.beta_m.powi(self.count);
        let cv = 1.0 - p.beta_v.powi(self.count);
        if let Some(prev) = self.prev_energy {
            let d = self.relative_change(e, prev);
            self.d_tilde = p.beta_d * self.d_tilde + (1.0 - p.beta_d) * d;
        }
        let scale = p.delta / self.d_tilde;
        let step = Col::from_fn(n, |i| {
            -scale * (m[i] / cm) / (libm::sqrt(v[i] / cv) + p.epsilon)
        });
        self.m = Some(m);
        self.v = Some(v);
        self.prev_energy = Some(e);
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
    fn relative_change_is_clipped_and_defined_for_any_energies() {
        let eve = Eve::default();
        // Negative energies: the magnitude of the lower one is the scale.
        assert!((eve.relative_change(-10.0, -8.0) - 0.2).abs() < 1e-15);
        assert_eq!(eve.relative_change(-10.0, -10.0), 0.1);
        assert_eq!(eve.relative_change(-100.0, -99.99), 0.1);
        assert_eq!(eve.relative_change(5.0, 0.0), 10.0);
        assert_eq!(eve.relative_change(0.0, 0.0), 0.1);
        assert_eq!(eve.relative_change(1.0, 30.0), 10.0);
    }

    #[test]
    fn feedback_rescales_the_adam_step() {
        let h = Mat::identity(1, 1);
        let x = Col::zeros(1);
        let g = Col::from_fn(1, |_| 1.0);
        let mut eve = Eve::default();
        let s0 = eve
            .propose(&plain_input(0, &x, 4.0, &g, &h, None, 0))
            .unwrap();
        assert!((s0[0] + 0.03).abs() < 1e-12);
        // Energy drops by half: d = 2 / 2 = 1, d~ stays 1.
        let s1 = eve
            .propose(&plain_input(1, &x, 2.0, &g, &h, None, 0))
            .unwrap();
        assert!((eve.feedback() - 1.0).abs() < 1e-15 && (s1[0] + 0.03).abs() < 1e-12);
        // No change: d = 1/c, d~ decreases, the step grows.
        let s2 = eve
            .propose(&plain_input(2, &x, 2.0, &g, &h, None, 0))
            .unwrap();
        assert!((eve.feedback() - (0.999 + 0.001 * 0.1)).abs() < 1e-15);
        assert!(s2[0] < s1[0]);
    }
}
