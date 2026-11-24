//! Damped velocity Verlet (DVV): a classical trajectory in mass-weighted coordinates whose
//! velocity is rescaled to a constant magnitude `v0` after every step, so that the
//! trajectory creeps down the valley instead of oscillating in it. The time step is adapted
//! to an estimate of the error: a point is compared with the one a single step over the
//! last two time steps would have reached,
//! `q' = q_(i-2) + v_(i-2) (dt_(i-2) + dt_(i-1)) + a_(i-2) (dt_(i-2) + dt_(i-1))^2 / 2`, and
//! `dt_i = dt_(i-1) (tol / |q_i - q'|)^(1/3)`, kept between 0.025 and 3 fs.
//!
//! The step size of the command is not used (as in MultiOptPy): the length of a step is
//! `v0 dt`. Options of the method: `v0` (0.04 amu^(1/2) Bohr/fs), `dt0` (first time step,
//! 0.5 fs), `tol` (error tolerance, 0.003 Bohr).
//!
//! MultiOptPy moves every point with the acceleration of the point before it and evaluates
//! its first point twice; here the velocity Verlet equations are used as published
//! (`docs/source/migration_paths.md`).
//!
//! Reference: H. P. Hratchian and H. B. Schlegel, J. Phys. Chem. A 2002, 106, 165.
//!
//! MultiOptPy: `IRC/dvv.py:DVV` (`mw_grad_to_acc`, `damp_velocity`, `estimate_error`,
//! `step`, `run`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::history::HessianUpdate;
use super::integrator::{Arrival, BranchAction, BranchStart, Integrator, Walk};
use super::settings::MethodOptions;
use super::types::{norm, PointValue, StepRecord};
use crate::core_infra::log::python_float;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};

/// MultiOptPy's magnitude of the velocity (amu^(1/2) Bohr/fs).
pub const DEFAULT_V0: f64 = 0.04;
/// MultiOptPy's first time step (fs).
pub const DEFAULT_DT0: f64 = 0.5;
/// MultiOptPy's error tolerance of the time step (Bohr).
pub const DEFAULT_ERROR_TOLERANCE: f64 = 0.003;
/// Smallest time step (fs).
pub const MIN_TIME_STEP: f64 = 0.025;
/// Largest time step (fs).
pub const MAX_TIME_STEP: f64 = 3.0;
/// Hartree/(Bohr amu^(1/2)) to amu^(1/2) Bohr/fs^2 (`E_h / u / a_0^2 / 1e30`, the formula of
/// MultiOptPy's `AU2J / AMU2KG / BOHR2M^2 / 1e30` with the CODATA 2018 constants).
pub const GRADIENT_TO_ACCELERATION: f64 = Units::HARTREE_TO_JOULE
    / Units::AMU_TO_KG
    / (Units::BOHR_TO_METER * Units::BOHR_TO_METER)
    / 1e30;

/// A path point of the trajectory.
struct Motion {
    q: Col<f64>,
    /// Damped velocity at the point.
    v: Col<f64>,
    a: Col<f64>,
    /// The time step taken from the point.
    dt: f64,
}

/// The DVV integrator.
pub struct Dvv {
    walk: Walk,
    v0: f64,
    dt0: f64,
    tolerance: f64,
    last: Option<Motion>,
    before: Option<Motion>,
}

impl Dvv {
    /// A side from `start`.
    pub fn new(start: BranchStart) -> MoptResult<Self> {
        let options = MethodOptions::parse(&start.settings.method_options);
        options.only("dvv", &["v0", "dt0", "tol"], false)?;
        let v0 = options.number("dvv", "v0", DEFAULT_V0)?;
        let dt0 = options.number("dvv", "dt0", DEFAULT_DT0)?;
        let tolerance = options.number("dvv", "tol", DEFAULT_ERROR_TOLERANCE)?;
        if !(v0 > 0.0 && dt0 > 0.0 && tolerance > 0.0) {
            return Err(MoptError::config(
                "IRC method dvv: v0, dt0, and tol must be positive",
            ));
        }
        // MultiOptPy: `for iter in range(1, max_step)`.
        let last = start.settings.max_steps.saturating_sub(1);
        Ok(Self {
            walk: Walk::new(start, last, HessianUpdate::GuardedBfgs),
            v0,
            dt0,
            tolerance,
            last: None,
            before: None,
        })
    }

    /// The velocity scaled to the magnitude `v0`, and the factor (1 for a velocity below
    /// `1e-10`, as in MultiOptPy).
    fn damped(&self, v: &Col<f64>) -> (Col<f64>, f64) {
        let size = norm(v);
        let factor = if size < 1e-10 { 1.0 } else { self.v0 / size };
        (v * factor, factor)
    }

    /// The time step after the one of `last` that led to `q`.
    fn next_time_step(&self, q: &Col<f64>, last: &Motion) -> f64 {
        let Some(before) = &self.before else {
            return last.dt;
        };
        let t = before.dt + last.dt;
        let n = q.nrows();
        let reference = Col::from_fn(n, |i| {
            before.q[i] + before.v[i] * t + 0.5 * before.a[i] * t * t
        });
        // MultiOptPy measures the difference in Bohr.
        let error = norm(&self.walk.space.from_mw(&(q - &reference)));
        if error == 0.0 {
            return MAX_TIME_STEP;
        }
        (last.dt * libm::cbrt(self.tolerance / error)).clamp(MIN_TIME_STEP, MAX_TIME_STEP)
    }
}

impl Integrator for Dvv {
    fn name(&self) -> &'static str {
        "dvv"
    }

    fn ask(&self) -> BranchAction {
        self.walk.next.clone()
    }

    fn tell(&mut self, value: &PointValue) -> MoptResult<Option<StepRecord>> {
        let Arrival {
            mut record,
            end,
            q,
            g,
            ..
        } = self.walk.arrive(value)?;
        let a = &g * (-GRADIENT_TO_ACCELERATION);
        let (v, dt, taken, factor) = match &self.last {
            None => {
                let (v, _) = self.damped(&a);
                (v, self.dt0, self.dt0, None)
            }
            Some(last) => {
                let raw = Col::from_fn(q.nrows(), |i| {
                    last.v[i] + 0.5 * (last.a[i] + a[i]) * last.dt
                });
                let (v, factor) = self.damped(&raw);
                (v, self.next_time_step(&q, last), last.dt, Some(factor))
            }
        };
        record.extras.push(("Time Step (fs)", python_float(taken)));
        record.extras.push((
            "Damping Factor",
            factor.map_or_else(String::new, python_float),
        ));
        if end.is_some() {
            return Ok(Some(record));
        }
        let moved = Col::from_fn(q.nrows(), |i| v[i] * dt + 0.5 * a[i] * dt * dt);
        let next = self.walk.space.displaced(&self.walk.x, &moved);
        self.before = self.last.take();
        self.last = Some(Motion { q, v, a, dt });
        self.walk.advance(next);
        Ok(Some(record))
    }

    fn current(&self) -> &Col<f64> {
        &self.walk.x
    }
}
