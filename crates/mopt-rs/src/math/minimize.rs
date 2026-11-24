//! Minimization of a cheap function of `n` variables given by a closure (surrogate models).
//!
//! L-BFGS (Nocedal, Math. Comput. 1980, 35, 773; Liu and Nocedal, Math. Program. 1989, 45, 503)
//! with a backtracking line search for the sufficient-decrease condition; secant pairs without
//! positive curvature are skipped, so every direction is a descent direction. Optionally the
//! iterates stay inside a ball (trial points are projected onto it).
//!
//! This is for functions that are evaluated inside one optimizer iteration, such as the mean of
//! a Gaussian process; potential-energy surfaces are minimized by the ask/tell optimizers.
//!
//! MultiOptPy: no counterpart (MultiOptPy calls `scipy.optimize.minimize`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::Col;

use crate::math::linalg::{col_dot, col_norm};

/// Settings of [`minimize`].
#[derive(Debug, Clone, Copy)]
pub struct MinimizeOptions {
    /// Stop when the gradient norm (inside the ball: of the projected gradient) is below this.
    pub gradient_tolerance: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Number of stored secant pairs.
    pub memory: usize,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self {
            gradient_tolerance: 1e-8,
            max_iterations: 200,
            memory: 10,
        }
    }
}

/// Result of [`minimize`].
#[derive(Debug, Clone)]
pub struct Minimum {
    /// Point with the lowest value found.
    pub x: Col<f64>,
    /// Value there.
    pub value: f64,
    /// Whether the gradient tolerance was reached.
    pub converged: bool,
    /// Iterations used.
    pub iterations: usize,
}

/// Minimize `f` (returning value and gradient) from `x0`, optionally inside the ball
/// `(center, radius)` (the start is projected onto it). Non-finite values are treated as
/// rejected trial points; the start must be finite.
pub fn minimize(
    mut f: impl FnMut(&Col<f64>) -> (f64, Col<f64>),
    x0: &Col<f64>,
    options: &MinimizeOptions,
    ball: Option<(&Col<f64>, f64)>,
) -> Minimum {
    let project = |x: Col<f64>| -> Col<f64> {
        match ball {
            Some((center, r)) => {
                let d = &x - center;
                let len = col_norm(&d);
                if len > r {
                    center + d * (r / len)
                } else {
                    x
                }
            }
            None => x,
        }
    };
    let mut x = project(x0.clone());
    let (mut value, mut g) = f(&x);
    let mut pairs: VecDeque<(Col<f64>, Col<f64>, f64)> = VecDeque::new();
    let mut iterations = 0;
    let mut converged = false;
    while iterations < options.max_iterations {
        if !value.is_finite() || g.iter().any(|v| !v.is_finite()) {
            break;
        }
        // Two-loop recursion.
        let mut q = g.clone();
        let mut a = vec![0.0; pairs.len()];
        for (i, (s, y, rho)) in pairs.iter().enumerate().rev() {
            a[i] = rho * col_dot(s, &q);
            q -= y * a[i];
        }
        let scale = pairs
            .back()
            .map_or(1.0, |(s, y, _)| col_dot(s, y) / col_dot(y, y));
        let mut z = q * scale;
        for (i, (s, y, rho)) in pairs.iter().enumerate() {
            let b = rho * col_dot(y, &z);
            z += s * (a[i] - b);
        }
        let mut direction = -z;
        if col_dot(&direction, &g) >= 0.0 {
            direction = -&g;
            pairs.clear();
        }
        // Backtracking from the unit step (limited to a unit move on the first iteration).
        let mut t = if pairs.is_empty() {
            1.0_f64.min(1.0 / col_norm(&g))
        } else {
            1.0
        };
        let mut accepted = None;
        for _ in 0..40 {
            let trial = project(&x + &direction * t);
            let moved = &trial - &x;
            let (tv, tg) = f(&trial);
            // Sufficient decrease along the move actually made.
            if tv.is_finite() && tv <= value + 1e-4 * col_dot(&g, &moved) {
                accepted = Some((trial, tv, tg, moved));
                break;
            }
            t *= 0.5;
        }
        iterations += 1;
        let Some((nx, nv, ng, s)) = accepted else {
            converged = col_norm(&g) < options.gradient_tolerance;
            break;
        };
        let y = &ng - &g;
        let sy = col_dot(&s, &y);
        if sy > 1e-12 * col_norm(&s) * col_norm(&y) {
            pairs.push_back((s.clone(), y, 1.0 / sy));
            if pairs.len() > options.memory.max(1) {
                pairs.pop_front();
            }
        }
        let small_move = col_norm(&s) <= 1e-15 * col_norm(&nx).max(1.0);
        x = nx;
        value = nv;
        g = ng;
        if col_norm(&g) < options.gradient_tolerance || small_move {
            converged = col_norm(&g) < options.gradient_tolerance;
            break;
        }
    }
    Minimum {
        x,
        value,
        converged,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_and_a_ball_constraint() {
        let rosen = |x: &Col<f64>| {
            let (a, b) = (x[0], x[1]);
            let v = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
            let g = Col::from_fn(2, |i| {
                if i == 0 {
                    -2.0 * (1.0 - a) - 400.0 * a * (b - a * a)
                } else {
                    200.0 * (b - a * a)
                }
            });
            (v, g)
        };
        let x0 = Col::from_fn(2, |i| [-1.2, 1.0][i]);
        let m = minimize(
            rosen,
            &x0,
            &MinimizeOptions {
                max_iterations: 500,
                ..MinimizeOptions::default()
            },
            None,
        );
        assert!(m.converged, "{m:?}");
        assert!((m.x[0] - 1.0).abs() < 1e-6 && (m.x[1] - 1.0).abs() < 1e-6);
        // Inside a ball of radius 0.5 the minimizer of |x - c|^2 lies on the boundary toward c.
        let c = Col::from_fn(2, |i| [3.0, 4.0][i]);
        let quad = |x: &Col<f64>| {
            let d = x - &c;
            (col_dot(&d, &d), &d * 2.0)
        };
        let origin = Col::zeros(2);
        let m = minimize(
            quad,
            &origin,
            &MinimizeOptions::default(),
            Some((&origin, 0.5)),
        );
        assert!(
            (m.x[0] - 0.3).abs() < 1e-6 && (m.x[1] - 0.4).abs() < 1e-6,
            "{:?}",
            m.x
        );
        // A function that is not finite away from the start leaves the start in place.
        let wall = |x: &Col<f64>| {
            if x[0] == 0.0 {
                (0.0, Col::from_fn(1, |_| 1.0))
            } else {
                (f64::NAN, Col::from_fn(1, |_| f64::NAN))
            }
        };
        let m = minimize(wall, &Col::zeros(1), &MinimizeOptions::default(), None);
        assert_eq!(m.x[0], 0.0);
    }
}
