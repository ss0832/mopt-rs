//! Limited-memory BFGS (MultiOptPy `Optimizer/lbfgs.py`, normal mode, with two corrections).
//!
//! The first step is `-delta * g`; later steps use the two-loop recursion with secant pairs built
//! from the unbiased gradient. Differences from the reference (see `docs/source/migration_optimizers.md`):
//! - pairs violating the curvature condition `s . y > eps ||s|| ||y||` are skipped (the
//!   reference stores them with `rho = 1000` when `|s . y| < 1e-10`, an absolute threshold that
//!   stalls the method on weakly bound systems);
//! - the initial inverse Hessian is scaled by `gamma = s . y / y . y` of the newest pair
//!   (Nocedal and Wright, eq. 7.20) instead of the constant `1 / alpha`, which is used only until
//!   the first pair is accepted.
//!
//! MultiOptPy: `Optimizer/lbfgs.py:LBFGS`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm};

/// L-BFGS parameters.
#[derive(Debug, Clone, Copy)]
pub struct LbfgsParams {
    /// Step scaling `delta`.
    pub delta: f64,
    /// Number of stored secant pairs.
    pub memory: usize,
    /// Initial inverse-Hessian scale before any pair is accepted (`H0 = 1 / alpha`).
    pub alpha: f64,
    /// Relative curvature threshold for accepting a pair.
    pub curvature_eps: f64,
    /// Maximum per-atom displacement (Cartesian only); `None` disables the cap.
    pub max_atom_step: Option<f64>,
}

impl Default for LbfgsParams {
    fn default() -> Self {
        Self {
            delta: 1.0,
            memory: 30,
            alpha: 10.0,
            curvature_eps: 1e-8,
            max_atom_step: None,
        }
    }
}

/// L-BFGS step generator.
#[derive(Debug, Clone)]
pub struct Lbfgs {
    params: LbfgsParams,
    history: VecDeque<(Col<f64>, Col<f64>, f64)>,
    started: bool,
}

impl Lbfgs {
    /// New L-BFGS generator.
    pub fn new(params: LbfgsParams) -> Self {
        Self {
            params,
            history: VecDeque::new(),
            started: false,
        }
    }

    /// Number of stored secant pairs.
    pub fn stored_pairs(&self) -> usize {
        self.history.len()
    }

    fn push_pair(&mut self, s: Col<f64>, y: Col<f64>) {
        let sy = col_dot(&s, &y);
        let rho = 1.0 / sy;
        if sy <= self.params.curvature_eps * col_norm(&s) * col_norm(&y) || !rho.is_finite() {
            return;
        }
        self.history.push_back((s, y, rho));
        while self.history.len() > self.params.memory {
            self.history.pop_front();
        }
    }

    fn two_loop(&self, g: &Col<f64>) -> Col<f64> {
        let mut q = g.clone();
        let mut a = vec![0.0; self.history.len()];
        for (i, (s, y, rho)) in self.history.iter().enumerate().rev() {
            a[i] = rho * col_dot(s, &q);
            q -= y * a[i];
        }
        let gamma = self
            .history
            .back()
            .map(|(s, y, _)| col_dot(s, y) / col_dot(y, y));
        let h0 = match gamma {
            Some(g) if g.is_finite() && g > 0.0 => g,
            _ => 1.0 / self.params.alpha,
        };
        let mut z = q * h0;
        for (i, (s, y, rho)) in self.history.iter().enumerate() {
            let b = rho * col_dot(y, &z);
            z += s * (a[i] - b);
        }
        z
    }
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self::new(LbfgsParams::default())
    }
}

impl StepGenerator for Lbfgs {
    fn set_step_factor(&mut self, delta: f64) -> bool {
        self.params.delta = delta;
        true
    }

    fn name(&self) -> &str {
        "lbfgs"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        if !self.started {
            self.started = true;
            return Ok(input.gradient * (-self.params.delta));
        }
        if let Some(prev) = input.previous {
            self.push_pair(prev.step.clone(), input.raw_gradient - &prev.raw_gradient);
        }
        let mut step = self.two_loop(input.gradient) * (-self.params.delta);
        if let Some(max) = self.params.max_atom_step {
            // The longest atom displacement; for plain variables, the largest component.
            let longest = if input.system.is_some() && step.nrows().is_multiple_of(3) {
                (0..step.nrows() / 3)
                    .map(|a| libm::sqrt((0..3).map(|c| step[3 * a + c].powi(2)).sum::<f64>()))
                    .fold(0.0, f64::max)
            } else {
                step.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
            };
            if longest > max {
                step *= max / longest;
            }
        }
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
    use faer::Mat;

    #[test]
    fn negative_curvature_pairs_are_skipped_and_quadratic_is_solved() {
        // f = x^T A x / 2 with tiny curvatures (argon-like scale).
        let a = Mat::from_fn(2, 2, |i, j| if i == j { [4e-4, 1e-4][i] } else { 0.0 });
        let h = Mat::identity(2, 2);
        let mut x = Col::from_fn(2, |i| [1.0, -2.0][i]);
        let mut opt = Lbfgs::default();
        let mut prev: Option<PreviousPoint> = None;
        for it in 0..12 {
            let g = &a * &x;
            let input = StepInput {
                iteration: it,
                cartesian: &x,
                energy: crate::domain::Quantity::zero(),
                gradient: &g,
                raw_gradient: &g,
                previous: prev.as_ref(),
                hessian: &h,
                bias_hessian: None,
                rigid_body_basis: None,
                saddle_order: 0,
                system: None,
                cartesian_coordinates: true,
                frame: None,
                cartesian_gradient: None,
            };
            let s = opt.propose(&input).unwrap();
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::zero(),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
        }
        assert!(
            col_norm(&(&a * &x)) < 1e-12,
            "|g| = {}",
            col_norm(&(&a * &x))
        );
    }
}
