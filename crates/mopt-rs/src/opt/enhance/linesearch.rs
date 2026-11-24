//! Line search along the proposed step, for the strong Wolfe conditions.
//!
//! The step `p` of the base method becomes a search direction: the full step is the first
//! trial, and the following iterations evaluate further trial points `x + alpha p` chosen by
//! [`crate::math::line_search::WolfeSearch`] (Nocedal and Wright, Numerical Optimization, 2006,
//! Algorithms 3.5 and 3.6; `c1 = 1e-4`, `c2 = 0.9`) until one satisfies
//! `E(x + alpha p) <= E(x) + c1 alpha g.p` and `|g(x + alpha p).p| <= c2 |g.p|`. From the
//! accepted point the base method's step there starts the next line. The optimizer is an
//! ask/tell state machine: each trial is one iteration, no evaluation happens inside the
//! enhancer, and the base method sees (and may learn from) every trial point. If the first
//! trial already satisfies the conditions, nothing changes, so a well-scaled quasi-Newton
//! method is not slowed down.
//!
//! Steps that are not descent directions and all steps of saddle-point searches pass
//! unchanged (the energy is no merit function there).
//!
//! MultiOptPy's `LineSearch` takes the previous energy and gradient as the start of the line
//! but the current geometry as its origin, extrapolates by a fixed factor of 1.5 while the
//! energy falls and the gradient is not orthogonal to the direction, and otherwise steps back to
//! the best point; no sufficient-decrease or curvature condition is tested. See
//! `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/linesearch.py:LineSearch`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::StepEnhancer;
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm};
use crate::math::line_search::{WolfeAdvice, WolfeParams, WolfeSearch};
use crate::opt::step::StepInput;

struct Line {
    direction: Col<f64>,
    search: WolfeSearch,
    /// Current position relative to the start of the line.
    offset: Col<f64>,
}

/// Line-search step enhancer.
pub struct LineSearch {
    params: WolfeParams,
    line: Option<Line>,
    /// Scale of the first trial (reduced after a search that found no lower point).
    first_scale: f64,
    searches: usize,
    trials: usize,
}

impl LineSearch {
    /// Line search with the given Wolfe constants.
    pub fn new(params: WolfeParams) -> Self {
        Self {
            params,
            line: None,
            first_scale: 1.0,
            searches: 0,
            trials: 0,
        }
    }

    /// Number of lines started.
    pub fn searches(&self) -> usize {
        self.searches
    }

    /// Number of trial points beyond the first of each line.
    pub fn extra_trials(&self) -> usize {
        self.trials
    }

    /// Whether a line search is in progress.
    pub fn is_searching(&self) -> bool {
        self.line.is_some()
    }

    fn start(&mut self, input: &StepInput<'_>, step: Col<f64>) -> Col<f64> {
        self.line = None;
        if input.saddle_order > 0 {
            return step;
        }
        let slope = col_dot(input.gradient, &step);
        let Ok(search) = WolfeSearch::new(input.energy.value(), slope, self.params) else {
            // Not a descent direction (or a stationary point): nothing to search along.
            return step;
        };
        let first = &step * self.first_scale;
        self.line = Some(Line {
            direction: step,
            search,
            offset: Col::zeros(first.nrows()),
        });
        self.searches += 1;
        first
    }
}

impl Default for LineSearch {
    fn default() -> Self {
        Self::new(WolfeParams::default())
    }
}

impl StepEnhancer for LineSearch {
    fn name(&self) -> &str {
        "linesearch"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let (Some(mut line), Some(prev)) = (self.line.take(), input.previous) else {
            return Ok(self.start(input, step));
        };
        let offset = &line.offset + &prev.step;
        let dd = col_dot(&line.direction, &line.direction);
        let alpha = col_dot(&offset, &line.direction) / dd;
        let off_line = col_norm(&(&offset - &line.direction * alpha));
        if !alpha.is_finite() || alpha <= 0.0 || off_line > 0.2 * col_norm(&offset) {
            // The optimizer has left the line (constraints, alignment): start again here.
            return Ok(self.start(input, step));
        }
        let slope = col_dot(input.gradient, &line.direction);
        match line.search.advise(alpha, input.energy.value(), slope) {
            WolfeAdvice::Accept => {
                self.first_scale = 1.0;
                Ok(self.start(input, step))
            }
            WolfeAdvice::Try(next) => {
                self.trials += 1;
                let out = &line.direction * (next - alpha);
                line.offset = offset;
                self.line = Some(line);
                Ok(out)
            }
            WolfeAdvice::Failed { best_alpha } => {
                if (best_alpha - alpha).abs() <= 1e-12 * alpha {
                    return Ok(self.start(input, step));
                }
                if best_alpha == 0.0 {
                    self.first_scale *= 0.1;
                }
                // Back to the best point seen; the next call starts a new line there.
                Ok(&line.direction * (best_alpha - alpha))
            }
        }
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

    /// Steepest descent with a fixed factor on `E = x.A x / 2`, with the line search.
    fn descend(factor: f64, iterations: usize) -> (f64, LineSearch, bool) {
        let a = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, 10.0][i] } else { 0.0 });
        let h = Mat::<f64>::identity(2, 2);
        let mut ls = LineSearch::default();
        let mut x = Col::from_fn(2, |i| [1.0, 1.0][i]);
        let mut prev: Option<PreviousPoint> = None;
        let mut monotone = true;
        let mut accepted_energy = f64::INFINITY;
        for it in 0..iterations {
            let g = &a * &x;
            let e = 0.5 * col_dot(&x, &g);
            let input = plain_input(it, &x, e, &g, &h, prev.as_ref(), 0);
            let before = ls.searches();
            let s = ls.enhance(&input, &g * (-factor)).unwrap();
            if ls.searches() > before {
                // A new line starts at an accepted point: its energy never rises.
                monotone &= e <= accepted_energy;
                accepted_energy = e;
            }
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::new(e),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
        }
        let g = &a * &x;
        (col_norm(&g), ls, monotone)
    }

    #[test]
    fn unstable_step_lengths_are_corrected_and_good_ones_left_alone() {
        // Factor 0.5 on curvature 10: the plain iteration multiplies the error by -4 per step.
        let (g, ls, monotone) = descend(0.5, 300);
        assert!(g < 1e-6, "|g| = {g}");
        assert!(monotone && ls.extra_trials() > 0);
        // Factor 0.12: every full step satisfies the Wolfe conditions, no extra trial.
        let (g, ls, monotone) = descend(0.12, 400);
        assert!(g < 1e-6, "|g| = {g}");
        assert_eq!(ls.extra_trials(), 0);
        assert!(monotone);
        // Far too short steps are extended (the curvature condition fails at the first trial).
        let (_, ls, _) = descend(1e-4, 30);
        assert!(ls.extra_trials() > 0);
    }

    #[test]
    fn uphill_steps_and_saddle_searches_pass_unchanged() {
        let h = Mat::<f64>::identity(1, 1);
        let x = Col::zeros(1);
        let g = Col::from_fn(1, |_| 1.0);
        let up = Col::from_fn(1, |_| 0.3);
        let mut ls = LineSearch::default();
        let input = plain_input(0, &x, 0.0, &g, &h, None, 0);
        assert_eq!(ls.enhance(&input, up.clone()).unwrap(), up);
        assert!(!ls.is_searching());
        let down = Col::from_fn(1, |_| -0.3);
        let saddle = plain_input(0, &x, 0.0, &g, &h, None, 1);
        assert_eq!(ls.enhance(&saddle, down.clone()).unwrap(), down);
        assert!(!ls.is_searching());
        assert_eq!(ls.enhance(&input, down.clone()).unwrap(), down);
        assert!(ls.is_searching());
    }
}
