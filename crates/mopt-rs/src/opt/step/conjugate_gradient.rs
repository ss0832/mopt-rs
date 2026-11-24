//! Nonlinear conjugate gradients: Fletcher-Reeves, Polak-Ribiere(+), Hestenes-Stiefel(+), and
//! Dai-Yuan, with a strong Wolfe line search.
//!
//! Fletcher and Reeves, Comput. J. 1964, 7, 149; Polak and Ribiere, Rev. Fr. Inform. Rech. Oper.
//! 1969, 16, 35; Hestenes and Stiefel, J. Res. Natl. Bur. Stand. 1952, 49, 409; Dai and Yuan,
//! SIAM J. Optim. 1999, 10, 177; Gilbert and Nocedal, SIAM J. Optim. 1992, 2, 21 (the
//! non-negative `beta` of PR+ and HS+).
//!
//! Search directions are `d_{k+1} = -g_{k+1} + beta d_k` with, for `y = g_{k+1} - g_k`,
//! `beta_FR = g_{k+1}.g_{k+1} / g_k.g_k`, `beta_PR = max(0, g_{k+1}.y / g_k.g_k)`,
//! `beta_HS = max(0, g_{k+1}.y / d_k.y)`, `beta_DY = g_{k+1}.g_{k+1} / d_k.y`. The direction is
//! reset to `-g` when it is not a descent direction and every `n` directions.
//!
//! Conjugate gradients need the minimum along each direction. The optimizer is an ask/tell
//! state machine, so the line search ([`crate::math::line_search::WolfeSearch`], `c2 = 0.1`) is
//! spread over the iterations: every iteration evaluates one trial point of the current line,
//! and the next direction is built when a trial satisfies the strong Wolfe conditions. On a
//! quadratic the second trial is the exact line minimum, the directions are conjugate, and the
//! minimum is reached after `n` directions.
//!
//! MultiOptPy's version has no line search: it moves by the projection of the current gradient
//! on a direction built from the previous gradient (its second iteration does not move at all),
//! takes `|beta|`, and for the plain `cg` name squares the denominator of the Polak-Ribiere
//! formula. See `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/conjugate_gradient.py:ConjgateGradient`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm};
use crate::math::line_search::{WolfeAdvice, WolfeParams, WolfeSearch};

/// Formula of `beta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgVariant {
    /// Polak-Ribiere, restricted to `beta >= 0` (`cg`, `cg_pr`).
    PolakRibiere,
    /// Fletcher-Reeves (`cg_fr`).
    FletcherReeves,
    /// Hestenes-Stiefel, restricted to `beta >= 0` (`cg_hs`).
    HestenesStiefel,
    /// Dai-Yuan (`cg_dy`).
    DaiYuan,
}

impl CgVariant {
    /// Variant selected by a method string (`cg_fr`, `cg_hs`, `cg_dy`; otherwise Polak-Ribiere).
    pub fn from_method(method: &str) -> Self {
        let m = method.to_ascii_lowercase();
        if m.contains("cg_fr") {
            Self::FletcherReeves
        } else if m.contains("cg_hs") {
            Self::HestenesStiefel
        } else if m.contains("cg_dy") {
            Self::DaiYuan
        } else {
            Self::PolakRibiere
        }
    }

    /// `beta` from the new gradient, the old gradient, and the old direction (`0` when the
    /// denominator vanishes).
    pub fn beta(self, g_new: &Col<f64>, g_old: &Col<f64>, d_old: &Col<f64>) -> f64 {
        let y = g_new - g_old;
        let (num, den) = match self {
            Self::FletcherReeves => (col_dot(g_new, g_new), col_dot(g_old, g_old)),
            Self::PolakRibiere => (col_dot(g_new, &y), col_dot(g_old, g_old)),
            Self::HestenesStiefel => (col_dot(g_new, &y), col_dot(d_old, &y)),
            Self::DaiYuan => (col_dot(g_new, g_new), col_dot(d_old, &y)),
        };
        let beta = num / den;
        if den == 0.0 || !beta.is_finite() {
            0.0
        } else if matches!(self, Self::PolakRibiere | Self::HestenesStiefel) {
            beta.max(0.0)
        } else {
            beta
        }
    }
}

/// Parameters of the conjugate-gradient generator.
#[derive(Debug, Clone, Copy)]
pub struct CgParams {
    /// Formula of `beta`.
    pub variant: CgVariant,
    /// Line-search constants.
    pub wolfe: WolfeParams,
    /// Longest trial move along a line in one iteration.
    pub max_step: f64,
    /// Directions between resets to steepest descent; `None` uses the number of variables.
    pub restart_interval: Option<usize>,
}

impl CgParams {
    /// Defaults for a variant: `c1 = 1e-4`, `c2 = 0.1`, moves of at most 0.5.
    pub fn new(variant: CgVariant) -> Self {
        Self {
            variant,
            wolfe: WolfeParams {
                c2: 0.1,
                ..WolfeParams::default()
            },
            max_step: 0.5,
            restart_interval: None,
        }
    }
}

struct Line {
    direction: Col<f64>,
    search: WolfeSearch,
    /// Gradient at the start of the line.
    gradient: Col<f64>,
    /// Current position relative to the start of the line.
    offset: Col<f64>,
}

/// Conjugate-gradient step generator.
pub struct ConjugateGradient {
    params: CgParams,
    line: Option<Line>,
    since_restart: usize,
    lines: usize,
    /// Scale of the first trial of a steepest-descent line (reduced after a failed search).
    first_scale: f64,
}

impl ConjugateGradient {
    /// New generator.
    pub fn new(params: CgParams) -> Self {
        Self {
            params,
            line: None,
            since_restart: 0,
            lines: 0,
            first_scale: 1.0,
        }
    }

    /// Current search direction.
    pub fn direction(&self) -> Option<&Col<f64>> {
        self.line.as_ref().map(|l| &l.direction)
    }

    /// Number of line searches started so far.
    pub fn lines(&self) -> usize {
        self.lines
    }

    fn capped(&self, step: Col<f64>) -> Col<f64> {
        let len = col_norm(&step);
        if len > self.params.max_step {
            step * (self.params.max_step / len)
        } else {
            step
        }
    }

    /// Start a line at the current point and return its first trial step.
    fn begin(&mut self, direction: Col<f64>, input: &StepInput<'_>, guess: f64) -> Col<f64> {
        let g = input.gradient;
        let n = g.nrows();
        let slope = col_dot(g, &direction);
        let Ok(search) = WolfeSearch::new(input.energy.value(), slope, self.params.wolfe) else {
            // Stationary point (or no descent direction left): stay.
            self.line = None;
            return Col::zeros(n);
        };
        let first = if guess.is_finite() && guess > 0.0 {
            guess
        } else {
            1.0
        };
        let step = self.capped(&direction * first);
        self.line = Some(Line {
            direction,
            search,
            gradient: g.clone(),
            offset: Col::zeros(n),
        });
        self.lines += 1;
        step
    }

    fn steepest(&mut self, input: &StepInput<'_>) -> Col<f64> {
        self.since_restart = 0;
        let scale = self.first_scale;
        self.begin(-input.gradient, input, scale)
    }
}

impl StepGenerator for ConjugateGradient {
    fn name(&self) -> &str {
        match self.params.variant {
            CgVariant::PolakRibiere => "cg_pr",
            CgVariant::FletcherReeves => "cg_fr",
            CgVariant::HestenesStiefel => "cg_hs",
            CgVariant::DaiYuan => "cg_dy",
        }
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let g = input.gradient;
        let (Some(mut line), Some(prev)) = (self.line.take(), input.previous) else {
            return Ok(self.steepest(input));
        };
        // Where on the line the step actually taken has led.
        let offset = &line.offset + &prev.step;
        let dd = col_dot(&line.direction, &line.direction);
        let alpha = col_dot(&offset, &line.direction) / dd;
        let off_line = col_norm(&(&offset - &line.direction * alpha));
        if !alpha.is_finite() || alpha <= 0.0 || off_line > 0.2 * col_norm(&offset) {
            return Ok(self.steepest(input));
        }
        let slope = col_dot(g, &line.direction);
        match line.search.advise(alpha, input.energy.value(), slope) {
            WolfeAdvice::Accept => {
                self.first_scale = 1.0;
                let beta = self.params.variant.beta(g, &line.gradient, &line.direction);
                let mut direction = -g + &line.direction * beta;
                self.since_restart += 1;
                let interval = self.params.restart_interval.unwrap_or(g.nrows()).max(1);
                let slope = col_dot(g, &direction);
                if slope.is_nan() || slope >= 0.0 || self.since_restart >= interval {
                    direction = -g;
                    self.since_restart = 0;
                }
                // First trial from the decrease of the previous line (Nocedal and Wright,
                // eq. 3.60): the same first-order change of the function.
                let new_slope = col_dot(g, &direction);
                let guess =
                    (alpha * line.search.dphi0() / new_slope).clamp(0.1 * alpha, 10.0 * alpha);
                Ok(self.begin(direction, input, guess))
            }
            WolfeAdvice::Try(next) => {
                let step = self.capped(&line.direction * (next - alpha));
                line.offset = offset;
                self.line = Some(line);
                Ok(step)
            }
            WolfeAdvice::Failed { best_alpha } => {
                if (best_alpha - alpha).abs() <= 1e-12 * alpha {
                    // The current point is the best one seen: go on from here.
                    return Ok(self.steepest(input));
                }
                if best_alpha == 0.0 {
                    self.first_scale *= 0.1;
                }
                // Return to the best point; the next call restarts there.
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

    fn quadratic() -> Mat<f64> {
        Mat::from_fn(5, 5, |i, j| {
            if i == j {
                1.0 + 2.0 * i as f64
            } else {
                0.3 / (1.0 + (i + j) as f64)
            }
        })
    }

    #[test]
    fn directions_are_conjugate_and_the_quadratic_is_solved_in_n_lines() {
        let a = quadratic();
        let h = Mat::identity(5, 5);
        for variant in [
            CgVariant::PolakRibiere,
            CgVariant::FletcherReeves,
            CgVariant::HestenesStiefel,
            CgVariant::DaiYuan,
        ] {
            let mut params = CgParams::new(variant);
            // Exact line minima.
            params.wolfe.c1 = 1e-12;
            params.wolfe.c2 = 1e-9;
            params.max_step = 1e3;
            let mut cg = ConjugateGradient::new(params);
            let mut x = Col::from_fn(5, |i| [1.0, -2.0, 0.5, 3.0, -1.0][i]);
            let mut prev: Option<PreviousPoint> = None;
            let mut dirs: Vec<Col<f64>> = Vec::new();
            let mut evaluations = 0;
            for it in 0..40 {
                let g = &a * &x;
                if col_norm(&g) < 1e-9 {
                    break;
                }
                let e = 0.5 * col_dot(&x, &g);
                let before = cg.lines();
                let s = cg
                    .propose(&plain_input(it, &x, e, &g, &h, prev.as_ref(), 0))
                    .unwrap();
                evaluations += 1;
                if cg.lines() > before {
                    dirs.push(cg.direction().unwrap().clone());
                }
                prev = Some(PreviousPoint {
                    energy: crate::domain::Quantity::new(e),
                    gradient: g.clone(),
                    raw_gradient: g,
                    step: s.clone(),
                });
                x += &s;
            }
            assert!(col_norm(&(&a * &x)) < 1e-9, "{variant:?}");
            // n directions (the last line search is started at the solution's neighbor).
            assert!(dirs.len() <= 6, "{variant:?}: {} lines", dirs.len());
            assert!(evaluations <= 16, "{variant:?}: {evaluations} evaluations");
            for i in 0..5.min(dirs.len()) {
                for j in 0..i {
                    let c = col_dot(&dirs[i], &(&a * &dirs[j]));
                    let scale = col_dot(&dirs[i], &(&a * &dirs[i])).sqrt()
                        * col_dot(&dirs[j], &(&a * &dirs[j])).sqrt();
                    assert!(c.abs() < 1e-6 * scale, "{variant:?}: d{i}.A.d{j} = {c}");
                }
            }
        }
    }

    #[test]
    fn beta_formulas_and_their_guards() {
        let g1 = Col::from_fn(2, |i| [1.0, 2.0][i]);
        let g0 = Col::from_fn(2, |i| [2.0, -1.0][i]);
        let d0 = Col::from_fn(2, |i| [-2.0, 1.0][i]);
        // y = (-1, 3): g1.y = 5, g0.g0 = 5, d0.y = 5, g1.g1 = 5.
        for v in [
            CgVariant::PolakRibiere,
            CgVariant::FletcherReeves,
            CgVariant::HestenesStiefel,
            CgVariant::DaiYuan,
        ] {
            assert!((v.beta(&g1, &g0, &d0) - 1.0).abs() < 1e-15, "{v:?}");
        }
        // Negative Polak-Ribiere values are replaced by a restart (beta = 0).
        let shrunk = &g0 * 0.5;
        assert_eq!(CgVariant::PolakRibiere.beta(&shrunk, &g0, &d0), 0.0);
        let zero = Col::zeros(2);
        assert_eq!(CgVariant::FletcherReeves.beta(&g1, &zero, &d0), 0.0);
        assert_eq!(CgVariant::DaiYuan.beta(&g1, &g1, &d0), 0.0);
        assert_eq!(
            CgVariant::from_method("CG_FR_lars"),
            CgVariant::FletcherReeves
        );
        assert_eq!(CgVariant::from_method("cg"), CgVariant::PolakRibiere);
    }
}
