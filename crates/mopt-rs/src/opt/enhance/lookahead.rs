//! Lookahead: `k` steps forward, one step back (Zhang, Lucas, Hinton, and Ba,
//! arXiv:1907.08610).
//!
//! The inner optimizer moves the fast point for `k` steps; then the slow point is moved toward
//! the fast one, `phi <- phi + alpha (theta - phi)`, and the fast point restarts from it. As an
//! enhancer: on every `k`-th iteration the proposed step `theta_k -> theta_{k+1}` is replaced by
//! the step to `phi + alpha (theta_{k+1} - phi)`. No evaluation is added, because the fast point
//! `theta_{k+1}` itself is never evaluated.
//!
//! MultiOptPy overwrites its slow point with the current geometry in every iteration and then
//! averages that with the lowest-energy geometry since the last synchronization, which is not
//! the method of the paper; see `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/lookahead.py:LookAhead`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::StepEnhancer;
use crate::error::MoptResult;
use crate::opt::step::StepInput;

/// Lookahead step enhancer.
#[derive(Debug, Clone)]
pub struct Lookahead {
    /// Number of fast steps between synchronizations.
    pub k: usize,
    /// Interpolation factor of the slow point.
    pub alpha: f64,
    /// Fast point relative to the slow point.
    offset: Option<Col<f64>>,
    count: usize,
}

impl Lookahead {
    /// Lookahead with `k` fast steps and slow step size `alpha`.
    pub fn new(k: usize, alpha: f64) -> Self {
        Self {
            k: k.max(1),
            alpha,
            offset: None,
            count: 0,
        }
    }
}

impl Default for Lookahead {
    fn default() -> Self {
        Self::new(10, 0.5)
    }
}

impl StepEnhancer for Lookahead {
    fn name(&self) -> &str {
        "lookahead"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let n = step.nrows();
        let mut offset = match (self.offset.take(), input.previous) {
            (Some(o), Some(prev)) => o + &prev.step,
            _ => Col::zeros(n),
        };
        if self.count > 0 && self.count.is_multiple_of(self.k) {
            // The slow point is wherever the last synchronization actually arrived.
            offset = Col::zeros(n);
        }
        self.count += 1;
        let out = if self.count.is_multiple_of(self.k) {
            // theta_{k+1} - phi = offset + step; go to phi + alpha (theta_{k+1} - phi).
            (&offset + &step) * self.alpha - &offset
        } else {
            step
        };
        self.offset = Some(offset);
        Ok(out)
    }

    fn reset(&mut self) {
        self.offset = None;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::step::PreviousPoint;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn every_kth_step_lands_between_the_slow_and_the_fast_point() {
        let h = Mat::identity(1, 1);
        let g = Col::from_fn(1, |_| 1.0);
        let mut la = Lookahead::new(3, 0.5);
        let mut x = Col::from_fn(1, |_| 0.0);
        let mut prev: Option<PreviousPoint> = None;
        let mut path = vec![0.0];
        for it in 0..6 {
            let input = plain_input(it, &x, 0.0, &g, &h, prev.as_ref(), 0);
            // The inner optimizer always proposes a unit step.
            let s = la.enhance(&input, Col::from_fn(1, |_| 1.0)).unwrap();
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::zero(),
                gradient: g.clone(),
                raw_gradient: g.clone(),
                step: s.clone(),
            });
            x += &s;
            path.push(x[0]);
        }
        // Fast points 1, 2, (3): slow point 0 -> 1.5; then 2.5, 3.5, (4.5): 1.5 -> 3.0.
        let want = [0.0, 1.0, 2.0, 1.5, 2.5, 3.5, 3.0];
        for (a, b) in path.iter().zip(want) {
            assert!((a - b).abs() < 1e-15, "{path:?}");
        }
    }
}
