//! Euler integration: every step goes straight down the mass-weighted gradient by the step
//! size. The simplest integrator, for comparison; it zigzags across a curved valley unless
//! the step is small.
//!
//! MultiOptPy's version stops with an error at its second point (it reads a Hessian it
//! never set, for the curvature it prints); here the Hessian is kept as in RK4 and the run
//! goes on (`docs/source/migration_paths.md`).
//!
//! MultiOptPy: `IRC/euler.py:Euler`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::control::{Approach, Arrived};
use super::history::HessianUpdate;
use super::integrator::{BranchAction, BranchStart, Integrator, Walk};
use super::types::{PointValue, StepCounts, StepRecord};
use crate::core_infra::log::python_float;
use crate::error::MoptResult;

/// The Euler integrator.
pub struct Euler {
    walk: Walk,
    path_length: f64,
    /// Length of the last step, which does not count when the step is taken again.
    last_length: f64,
}

impl Euler {
    /// A side from `start`.
    pub fn new(start: BranchStart) -> Self {
        // MultiOptPy: `for iter in range(1, max_step + 1)`.
        let last = start.settings.max_steps;
        Self {
            walk: Walk::new(start, last, HessianUpdate::GuardedBfgs)
                .with_curvature()
                .with_final_approach(),
            path_length: 0.0,
            last_length: 0.0,
        }
    }
}

impl Integrator for Euler {
    fn name(&self) -> &'static str {
        "euler"
    }

    fn ask(&self) -> BranchAction {
        self.walk.next.clone()
    }

    fn tell(&mut self, value: &PointValue) -> MoptResult<Option<StepRecord>> {
        let w = &mut self.walk;
        // A point of higher energy is dropped and the step taken again, shorter.
        let (arrival, new) = match w.arrive_or_retry(value)? {
            Arrived::Point(arrival) => (arrival, true),
            Arrived::Again(arrival) => {
                // The length of the step that is taken back does not count.
                self.path_length -= self.last_length;
                (arrival, false)
            }
            Arrived::Wait => return Ok(None),
        };
        let record = new.then(|| {
            let mut record = arrival.record.clone();
            // MultiOptPy's column: the sum of the step sizes taken before this point.
            record
                .extras
                .push(("Path Length", python_float(self.path_length)));
            record
        });
        if arrival.end.is_some() {
            return Ok(record);
        }
        let next = match w.approach(&arrival)? {
            Approach::Arrived => return Ok(record),
            Approach::Step(x) => {
                self.last_length = 0.0;
                x
            }
            Approach::Path => {
                let s = w.step_length();
                self.last_length = s;
                self.path_length += s;
                w.downhill(&arrival.g, s)
            }
        };
        w.advance(next);
        Ok(record)
    }

    fn current(&self) -> &Col<f64> {
        &self.walk.x
    }

    fn counts(&self) -> StepCounts {
        self.walk.counts
    }
}
