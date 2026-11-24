//! The mode-maximized surface: the energy maximized along a fixed direction `e`,
//! `E~(y) = E(y + t*(y) e)` with `t*` the local maximum of `t -> E(y + t e)` continued from
//! the last one, on the hyperplane `e . (y - y_0) = 0` (the "fixed-mode reduced surface").
//!
//! Where the maximum is non-degenerate (`e^T H e < 0`) and continues smoothly:
//!
//! - its gradient is `(I - e e^T) g(y + t* e)` (the envelope theorem, exact at the exact
//!   maximum, where `e . g = 0` too), so stationary points of `E~` are stationary points of
//!   `E`, and stationary points of `E` with `e^T H e < 0` are stationary points of `E~`;
//! - its Hessian is the Schur complement `S = H_yy - H_ye H_ee^(-1) H_ey`, and by the
//!   inertia additivity of Haynsworth (Linear Algebra Appl. 1968, 1, 73) with `H_ee < 0`,
//!   `index E = 1 + index E~`: a transition state whose imaginary mode has a component along
//!   `e` is a minimum of `E~`, and an index-2 saddle is a transition state of `E~`.
//!
//! These hold locally. The surface is a function only where one maximum is followed: a
//! point keeps its branch (the `t*` it continues from), reports a jump of `t*` beyond a
//! threshold as a change of branch, and reports when there is no maximum along `e` (the
//! curvature along `e` is not negative: the surface ends there). `e` is kept fixed, in the
//! orientation of the point it was taken at; it is not projected anew at other points,
//! which would change the surface.
//!
//! [`LineMaximum`] is one point of the surface as an ask/tell search along its line
//! (Newton steps on `e . g` with the curvature from secants, the first from a given value);
//! the band of [`crate::path::band::ridge_fixed`] and the exploration on the surface of a
//! transition state use it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_dot;

/// Settings of the line maximizations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSettings {
    /// At the maximum `|e . g|` is below this (in the metric of the search).
    pub tolerance: f64,
    /// Largest change of `t` in one step.
    pub max_step: f64,
    /// Steps at most.
    pub max_iterations: usize,
    /// The curvature along `e` must be below `-min_curvature` for a maximum.
    pub min_curvature: f64,
    /// A maximum farther than this from where the line started is a change of branch.
    pub branch_jump: f64,
}

impl Default for LineSettings {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            max_step: 0.2,
            max_iterations: 30,
            min_curvature: 0.0,
            branch_jump: 0.5,
        }
    }
}

/// State of a line maximization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineStatus {
    /// Steps go on.
    Running,
    /// The maximum is found.
    Converged,
    /// The curvature along the line is not negative: no maximum (the surface ends here).
    NoMaximum,
    /// The steps are used up.
    Steps,
}

/// One point `y` of the mode-maximized surface, maximized along `e`.
#[derive(Debug, Clone)]
pub struct LineMaximum {
    settings: LineSettings,
    y: Col<f64>,
    e: Col<f64>,
    t: f64,
    start: f64,
    last: Option<(f64, f64)>,
    curvature: f64,
    status: LineStatus,
    iterations: usize,
    energy: f64,
    gradient: Col<f64>,
}

impl LineMaximum {
    /// The point `y` with the unit direction `e` (both in the metric of the search), the
    /// line started at `t` with the curvature estimate `curvature` (negative; the
    /// curvature along `e` at the transition state, for example).
    pub fn new(
        settings: LineSettings,
        y: Col<f64>,
        e: Col<f64>,
        t: f64,
        curvature: f64,
    ) -> MoptResult<Self> {
        if y.nrows() != e.nrows() || !(t.is_finite() && curvature.is_finite()) {
            return Err(MoptError::config(
                "line maximum: the point and the direction differ in size, or a start value is \
                 not finite",
            ));
        }
        if !(settings.tolerance > 0.0 && settings.max_step > 0.0 && settings.max_iterations > 0) {
            return Err(MoptError::config(
                "line maximum: the tolerance, the step and the iterations must be positive",
            ));
        }
        let n = y.nrows();
        Ok(Self {
            settings,
            y,
            e,
            t,
            start: t,
            last: None,
            curvature,
            status: LineStatus::Running,
            iterations: 0,
            energy: f64::NAN,
            gradient: Col::zeros(n),
        })
    }

    /// The point to evaluate next, `y + t e` (metric of the search).
    pub fn point(&self) -> Col<f64> {
        &self.y + &self.e * self.t
    }

    /// The point on the hyperplane.
    pub fn base(&self) -> &Col<f64> {
        &self.y
    }

    /// The energy and gradient (metric of the search, without the motions left out) at
    /// [`Self::point`].
    pub fn tell(&mut self, energy: f64, gradient: Col<f64>) -> MoptResult<LineStatus> {
        if self.status != LineStatus::Running {
            return Err(MoptError::config("line maximum: tell() after the end"));
        }
        if !energy.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(MoptError::calc("line maximum: a value is not finite"));
        }
        let slope = col_dot(&self.e, &gradient);
        if let Some((t0, s0)) = self.last {
            if (self.t - t0).abs() > 1e-14 {
                self.curvature = (slope - s0) / (self.t - t0);
            }
        }
        self.last = Some((self.t, slope));
        self.energy = energy;
        self.gradient = gradient;
        self.iterations += 1;
        let maximum = self.curvature < -self.settings.min_curvature;
        if slope.abs() < self.settings.tolerance && maximum {
            self.status = LineStatus::Converged;
            return Ok(self.status);
        }
        if !maximum && self.iterations >= 3 {
            self.status = LineStatus::NoMaximum;
            return Ok(self.status);
        }
        if self.iterations >= self.settings.max_iterations {
            self.status = LineStatus::Steps;
            return Ok(self.status);
        }
        let step = if maximum {
            -slope / self.curvature
        } else {
            // Uphill until the curvature turns negative.
            slope.signum() * self.settings.max_step
        };
        self.t += step.clamp(-self.settings.max_step, self.settings.max_step);
        Ok(self.status)
    }

    /// The state.
    pub fn status(&self) -> LineStatus {
        self.status
    }

    /// The displacement along `e` of the last point evaluated.
    pub fn t(&self) -> f64 {
        self.last.map_or(self.t, |(t, _)| t)
    }

    /// The curvature along `e` (from the last secant).
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Whether the maximum lies farther than [`LineSettings::branch_jump`] from where the
    /// line started (another branch of maxima).
    pub fn jumped(&self) -> bool {
        (self.t() - self.start).abs() > self.settings.branch_jump
    }

    /// The energy of the surface (at the last point evaluated).
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// The gradient of the surface, `(I - e e^T) g`.
    pub fn reduced_gradient(&self) -> Col<f64> {
        &self.gradient - &self.e * col_dot(&self.e, &self.gradient)
    }

    /// Steps so far (gradients evaluated).
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_maximum_along_a_line_and_its_reduced_gradient() {
        // E = -(t - 0.3 y)^2 + y^2 along e = (1, 0): the maximum at t = 0.3 y.
        let value = |p: &Col<f64>| {
            let (t, y) = (p[0], p[1]);
            let e = -(t - 0.3 * y).powi(2) + y * y;
            let g = Col::from_fn(2, |i| {
                [-2.0 * (t - 0.3 * y), 0.6 * (t - 0.3 * y) + 2.0 * y][i]
            });
            (e, g)
        };
        let y = Col::from_fn(2, |i| [0.0, 0.5][i]);
        let e = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let mut line = LineMaximum::new(LineSettings::default(), y, e, 0.0, -1.0).unwrap();
        while line.status() == LineStatus::Running {
            let (energy, g) = value(&line.point());
            line.tell(energy, g).unwrap();
        }
        assert_eq!(line.status(), LineStatus::Converged);
        assert!((line.t() - 0.15).abs() < 1e-6);
        assert!((line.energy() - 0.25).abs() < 1e-9);
        // The envelope theorem: dE~/dy = 2 y.
        assert!((line.reduced_gradient()[1] - 1.0).abs() < 1e-5);
        assert!(!line.jumped());
        // Along a direction of positive curvature there is no maximum.
        let mut line = LineMaximum::new(
            LineSettings::default(),
            Col::from_fn(2, |i| [0.0, 0.5][i]),
            Col::from_fn(2, |i| [0.0, 1.0][i]),
            0.0,
            -1.0,
        )
        .unwrap();
        while line.status() == LineStatus::Running {
            let (energy, g) = value(&line.point());
            line.tell(energy, g).unwrap();
        }
        assert_eq!(line.status(), LineStatus::NoMaximum);
    }
}
