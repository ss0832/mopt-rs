//! Fourth-order Runge-Kutta integration of `dq/ds = -g/|g|` in mass-weighted coordinates:
//! four gradients for every step (three of them at points between two path points).
//!
//! As in MultiOptPy the first step of a side is a short one straight down the gradient
//! (0.05 of the step size), and the Hessian is only kept for the curvature that is reported
//! (BFGS updates with `s.y > 1e-10`).
//!
//! References: K. K. Baldridge, M. S. Gordon, R. Steckler, and D. G. Truhlar, J. Phys. Chem.
//! 1989, 93, 5107; the references MultiOptPy gives are J. Chem. Phys. 1991, 95, 6758 and
//! Chem. Phys. Lett. 2007, 437, 120.
//!
//! MultiOptPy: `IRC/rk4.py:RK4` (`get_k`, `step`, `run`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::control::Approach;
use super::history::HessianUpdate;
use super::integrator::{offset, Arrival, BranchAction, BranchStart, Integrator, Walk};
use super::lqa::FIRST_STEP_SHARE;
use super::types::{norm, PointValue, StepCounts, StepRecord};
use crate::error::MoptResult;

/// The RK4 integrator.
pub struct Rk4 {
    walk: Walk,
    /// The path point the stages start from, mass-weighted.
    q: Col<f64>,
    /// `k1`, `k2`, ... as far as they are known.
    stages: Vec<Col<f64>>,
    /// The record of the path point, held back until its step is complete.
    in_stage: bool,
}

impl Rk4 {
    /// A side from `start`.
    pub fn new(start: BranchStart) -> Self {
        // MultiOptPy: `for iter in range(1, max_step)`.
        let last = start.settings.max_steps.saturating_sub(1);
        let n = start.x.nrows();
        Self {
            walk: Walk::new(start, last, HessianUpdate::GuardedBfgs)
                .with_curvature()
                .with_final_approach(),
            q: Col::zeros(n),
            stages: Vec::new(),
            in_stage: false,
        }
    }

    /// MultiOptPy's `get_k`: the step size along the downhill direction (zero where the
    /// gradient norm is below `1e-12`).
    fn k(&self, g: &Col<f64>) -> Col<f64> {
        let size = norm(g);
        if size < 1e-12 {
            Col::zeros(g.nrows())
        } else {
            g * (-self.walk.settings.step_size / size)
        }
    }

    /// Ask for the gradient at `q + f * k` for the last stage known.
    fn next_stage(&mut self, f: f64) {
        let Some(k) = self.stages.last() else {
            return;
        };
        let x = offset(&self.walk.space, &self.q, f, k);
        self.walk.auxiliary(x, false);
    }
}

impl Integrator for Rk4 {
    fn name(&self) -> &'static str {
        "rk4"
    }

    fn ask(&self) -> BranchAction {
        self.walk.next.clone()
    }

    fn tell(&mut self, value: &PointValue) -> MoptResult<Option<StepRecord>> {
        if self.in_stage {
            self.walk.check(value)?;
            let g = self.walk.space.gradient_to_mw(&value.gradient);
            let k = self.k(&g);
            self.stages.push(k);
            match self.stages.len() {
                2 => self.next_stage(0.5),
                3 => self.next_stage(1.0),
                _ => {
                    // (k1 + 2 k2 + 2 k3 + k4) / 6
                    let n = self.q.nrows();
                    let dq = Col::from_fn(n, |i| {
                        (self.stages[0][i]
                            + 2.0 * self.stages[1][i]
                            + 2.0 * self.stages[2][i]
                            + self.stages[3][i])
                            / 6.0
                    });
                    let next = self.walk.space.displaced(&self.walk.x, &dq);
                    self.in_stage = false;
                    self.stages.clear();
                    self.walk.advance(next);
                }
            }
            return Ok(None);
        }
        let arrival = self.walk.arrive(value)?;
        let Arrival {
            record, end, q, g, ..
        } = arrival.clone();
        if end.is_some() {
            return Ok(Some(record));
        }
        // Near the minimum: steps to the minimum of the quadratic model.
        match self.walk.approach(&arrival)? {
            Approach::Arrived => return Ok(Some(record)),
            Approach::Step(next) => {
                self.walk.advance(next);
                return Ok(Some(record));
            }
            Approach::Path => {}
        }
        if self.walk.step == 1 {
            let s = self.walk.settings.step_size;
            let next = self.walk.downhill(&g, FIRST_STEP_SHARE * s);
            self.walk.advance(next);
        } else {
            self.q = q;
            self.stages = vec![self.k(&g)];
            self.in_stage = true;
            self.next_stage(0.5);
        }
        Ok(Some(record))
    }

    fn current(&self) -> &Col<f64> {
        &self.walk.x
    }

    fn counts(&self) -> StepCounts {
        self.walk.counts
    }
}
