//! One-dimensional line search for the strong Wolfe conditions, as an ask/tell state machine.
//!
//! Nocedal and Wright, Numerical Optimization (2nd ed., 2006), Algorithms 3.5 (bracketing) and
//! 3.6 (zoom) with the cubic interpolant of eq. (3.59). The caller evaluates
//! `phi(alpha) = f(x + alpha p)` and `phi'(alpha) = grad f(x + alpha p) . p` and reports them
//! with [`WolfeSearch::advise`], which answers with the next trial length or with acceptance:
//! - sufficient decrease: `phi(alpha) <= phi(0) + c1 alpha phi'(0)`,
//! - curvature: `|phi'(alpha)| <= c2 |phi'(0)|`.
//!
//! The search never evaluates anything itself, and the reported `alpha` need not be the advised
//! one (a caller may have shortened the trial step). On a quadratic `phi` the interpolation is
//! exact, so the second trial is the line minimum.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; `Optimizer/linesearch.py` extrapolates by a fixed
//! factor without these conditions).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// Parameters of the strong Wolfe search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WolfeParams {
    /// Sufficient-decrease constant `c1` (`0 < c1 < c2`).
    pub c1: f64,
    /// Curvature constant `c2` (`c1 < c2 < 1`; 0.9 for quasi-Newton, 0.1 for conjugate gradients).
    pub c2: f64,
    /// Maximum number of trial points.
    pub max_trials: usize,
    /// Largest factor by which a trial length grows while bracketing.
    pub max_growth: f64,
    /// Upper bound of `alpha`.
    pub alpha_max: f64,
}

impl Default for WolfeParams {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            max_trials: 12,
            max_growth: 4.0,
            alpha_max: 1e10,
        }
    }
}

/// Answer of the search to a reported trial point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WolfeAdvice {
    /// The reported point satisfies the strong Wolfe conditions.
    Accept,
    /// Evaluate at this `alpha` next.
    Try(f64),
    /// No acceptable point within the trial limit; `best_alpha` has the lowest `phi` seen
    /// (`0` when no trial improved on the start).
    Failed {
        /// Trial length with the lowest value.
        best_alpha: f64,
    },
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    alpha: f64,
    phi: f64,
    dphi: f64,
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Bracket { prev: Sample },
    Zoom { lo: Sample, hi: Sample },
}

/// Strong Wolfe line search along one direction.
#[derive(Debug, Clone)]
pub struct WolfeSearch {
    params: WolfeParams,
    start: Sample,
    phase: Phase,
    trials: usize,
    best: Sample,
}

/// Minimizer of the cubic through `(a, phi, phi')` and `(b, phi, phi')` (eq. 3.59), or the
/// midpoint when the cubic has no minimizer between them; kept away from the ends by 2% of the
/// interval.
fn interpolate(a: Sample, b: Sample) -> f64 {
    let (lo, hi) = (a.alpha.min(b.alpha), a.alpha.max(b.alpha));
    let width = hi - lo;
    let mid = 0.5 * (lo + hi);
    if !(b.phi.is_finite() && b.dphi.is_finite() && a.phi.is_finite() && a.dphi.is_finite()) {
        return mid;
    }
    let d1 = a.dphi + b.dphi - 3.0 * (a.phi - b.phi) / (a.alpha - b.alpha);
    let disc = d1 * d1 - a.dphi * b.dphi;
    if disc < 0.0 {
        return mid;
    }
    let d2 = (b.alpha - a.alpha).signum() * libm::sqrt(disc);
    let denom = b.dphi - a.dphi + 2.0 * d2;
    if denom == 0.0 {
        return mid;
    }
    let x = b.alpha - (b.alpha - a.alpha) * (b.dphi + d2 - d1) / denom;
    if x.is_finite() {
        x.clamp(lo + 0.02 * width, hi - 0.02 * width)
    } else {
        mid
    }
}

impl WolfeSearch {
    /// Search from `phi(0) = phi0` with slope `phi'(0) = dphi0 < 0`.
    pub fn new(phi0: f64, dphi0: f64, params: WolfeParams) -> MoptResult<Self> {
        if !(0.0 < params.c1 && params.c1 < params.c2 && params.c2 < 1.0) {
            return Err(MoptError::config(format!(
                "line search needs 0 < c1 < c2 < 1, got c1 = {}, c2 = {}",
                params.c1, params.c2
            )));
        }
        if dphi0.is_nan() || dphi0 >= 0.0 || !phi0.is_finite() {
            return Err(MoptError::config(
                "line search needs a descent direction (negative initial slope)",
            ));
        }
        let start = Sample {
            alpha: 0.0,
            phi: phi0,
            dphi: dphi0,
        };
        Ok(Self {
            params,
            start,
            phase: Phase::Bracket { prev: start },
            trials: 0,
            best: start,
        })
    }

    /// Value at the start of the search.
    pub fn phi0(&self) -> f64 {
        self.start.phi
    }

    /// Slope at the start of the search.
    pub fn dphi0(&self) -> f64 {
        self.start.dphi
    }

    /// Number of trial points reported so far.
    pub fn trials(&self) -> usize {
        self.trials
    }

    /// Whether `(alpha, phi, dphi)` satisfies both strong Wolfe conditions.
    pub fn satisfied(&self, alpha: f64, phi: f64, dphi: f64) -> bool {
        phi <= self.start.phi + self.params.c1 * alpha * self.start.dphi
            && dphi.abs() <= -self.params.c2 * self.start.dphi
    }

    /// Report the trial point `alpha > 0` and get the next action.
    pub fn advise(&mut self, alpha: f64, phi: f64, dphi: f64) -> WolfeAdvice {
        self.trials += 1;
        let finite = phi.is_finite() && dphi.is_finite();
        let s = Sample {
            alpha,
            phi: if finite { phi } else { f64::INFINITY },
            dphi,
        };
        if finite && s.phi < self.best.phi {
            self.best = s;
        }
        let armijo = finite && s.phi <= self.start.phi + self.params.c1 * alpha * self.start.dphi;
        if armijo && dphi.abs() <= -self.params.c2 * self.start.dphi {
            return WolfeAdvice::Accept;
        }
        if self.trials >= self.params.max_trials {
            return WolfeAdvice::Failed {
                best_alpha: self.best.alpha,
            };
        }
        let (lo, hi) = match self.phase {
            Phase::Bracket { prev } => {
                if !armijo || (s.phi >= prev.phi && prev.alpha > 0.0) {
                    (prev, s)
                } else if dphi >= 0.0 {
                    (s, prev)
                } else {
                    if alpha >= self.params.alpha_max {
                        return WolfeAdvice::Accept;
                    }
                    // Still descending: extend, by the secant of the slope when it is rising.
                    let rise = dphi - prev.dphi;
                    let secant = if rise > 0.0 {
                        alpha - dphi * (alpha - prev.alpha) / rise
                    } else {
                        f64::INFINITY
                    };
                    let next = secant
                        .clamp(1.1 * alpha, self.params.max_growth * alpha)
                        .min(self.params.alpha_max);
                    self.phase = Phase::Bracket { prev: s };
                    return WolfeAdvice::Try(next);
                }
            }
            Phase::Zoom { lo, hi } => {
                if !armijo || s.phi >= lo.phi {
                    (lo, s)
                } else if dphi * (hi.alpha - lo.alpha) >= 0.0 {
                    (s, lo)
                } else {
                    (s, hi)
                }
            }
        };
        if (hi.alpha - lo.alpha).abs() <= 1e-14 * lo.alpha.abs().max(hi.alpha.abs()) {
            return WolfeAdvice::Failed {
                best_alpha: self.best.alpha,
            };
        }
        self.phase = Phase::Zoom { lo, hi };
        WolfeAdvice::Try(interpolate(lo, hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: impl Fn(f64) -> (f64, f64), first: f64, params: WolfeParams) -> (Option<f64>, usize) {
        let (phi0, dphi0) = f(0.0);
        let mut s = WolfeSearch::new(phi0, dphi0, params).unwrap();
        let mut alpha = first;
        loop {
            let (p, d) = f(alpha);
            match s.advise(alpha, p, d) {
                WolfeAdvice::Accept => {
                    assert!(s.satisfied(alpha, p, d));
                    return (Some(alpha), s.trials());
                }
                WolfeAdvice::Try(a) => alpha = a,
                WolfeAdvice::Failed { .. } => return (None, s.trials()),
            }
        }
    }

    #[test]
    fn quadratic_line_minimum_is_found_on_the_second_trial() {
        // phi = (alpha - 3)^2: minimum at 3.
        let f = |a: f64| ((a - 3.0).powi(2), 2.0 * (a - 3.0));
        let tight = WolfeParams {
            c1: 1e-12,
            c2: 1e-8,
            ..WolfeParams::default()
        };
        for first in [0.5, 1.0, 5.0, 40.0] {
            let (alpha, trials) = run(f, first, tight);
            let alpha = alpha.unwrap();
            assert!((alpha - 3.0).abs() < 1e-7, "{first}: {alpha}");
            assert!(trials <= 3, "{first}: {trials} trials");
        }
    }

    #[test]
    fn strong_wolfe_point_of_a_nonquadratic_function() {
        // phi = exp(alpha) - 4 alpha: minimum at ln 4.
        let f = |a: f64| (a.exp() - 4.0 * a, a.exp() - 4.0);
        for (first, c2) in [(0.01, 0.1), (1.0, 0.1), (30.0, 0.1), (1.0, 0.9)] {
            let (alpha, _) = run(
                f,
                first,
                WolfeParams {
                    c2,
                    max_trials: 30,
                    ..WolfeParams::default()
                },
            );
            let alpha = alpha.unwrap();
            assert!((alpha.exp() - 4.0).abs() <= c2 * 3.0, "{first}: {alpha}");
        }
    }

    #[test]
    fn invalid_input_and_failure_are_reported() {
        assert!(WolfeSearch::new(0.0, 1.0, WolfeParams::default()).is_err());
        let bad = WolfeParams {
            c1: 0.5,
            c2: 0.1,
            ..WolfeParams::default()
        };
        assert!(WolfeSearch::new(0.0, -1.0, bad).is_err());
        // A function that only rises although the slope claims descent.
        let mut s = WolfeSearch::new(0.0, -1.0, WolfeParams::default()).unwrap();
        let mut alpha = 1.0;
        let mut failed = false;
        for _ in 0..20 {
            match s.advise(alpha, alpha, 1.0) {
                WolfeAdvice::Try(a) => alpha = a,
                WolfeAdvice::Failed { best_alpha } => {
                    assert_eq!(best_alpha, 0.0);
                    failed = true;
                    break;
                }
                WolfeAdvice::Accept => break,
            }
        }
        assert!(failed);
        // Non-finite values shrink the interval instead of propagating.
        let mut s = WolfeSearch::new(0.0, -1.0, WolfeParams::default()).unwrap();
        assert!(matches!(s.advise(8.0, f64::NAN, f64::NAN), WolfeAdvice::Try(a) if a < 8.0));
    }
}
