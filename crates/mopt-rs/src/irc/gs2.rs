//! Gonzalez-Schlegel second-order integrator (GS2): from the path point `q_k` with the
//! gradient `g_k` the pivot is `q* = q_k - (s/2) g_k/|g_k|`, and the next path point is the
//! minimum of the energy on the hypersphere of radius `s/2` around the pivot
//! ([`super::sphere`]), found from the first guess `q_k - s g_k/|g_k|`. The two half steps
//! are tangent to the path at both ends, so the path is an arc of a circle between two
//! points.
//!
//! The centre of the hypersphere is the pivot, not the path point: this is the difference
//! to the correction of the first step ([`super::first_step`]), whose sphere is centred at
//! the transition state.
//!
//! A step of fixed length cannot end in a minimum: it passes it. When the energy of the new
//! point is above that of the last path point the point is not taken; the step is halved
//! and repeated from the last path point, down to 1/64 of the step size, where the side
//! ends ([`BranchEnd::EnergyRise`]) unless the gradient fell below the thresholds before.
//!
//! Every minimization costs a few gradients; with `-fc N` the exact Hessian is asked for
//! at every N-th path point (one more evaluation of that point).
//!
//! References: C. Gonzalez and H. B. Schlegel, J. Chem. Phys. 1989, 90, 2154 and J. Phys.
//! Chem. 1990, 94, 5523.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::history::HessianUpdate;
use super::integrator::{Arrival, BranchAction, BranchStart, Integrator, Walk};
use super::sphere::{SphereMinimizer, SphereStatus};
use super::types::{axpy, norm, BranchEnd, PointValue, StepRecord};
use crate::error::MoptResult;

/// The step is halved at most down to the step size divided by this.
pub const SMALLEST_STEP_DIVISOR: f64 = 64.0;

/// The last path point, from which a step can be repeated.
struct Anchor {
    q: Col<f64>,
    g: Col<f64>,
    energy: f64,
    hessian: Mat<f64>,
}

/// The GS2 integrator.
pub struct Gs2 {
    walk: Walk,
    /// The minimization that leads to the next path point, while it runs.
    minimizer: Option<SphereMinimizer>,
    anchor: Option<Anchor>,
    /// The step length in use (halved where the energy rises).
    step: f64,
    /// What the last minimization took, for the record of the point it found.
    note: Option<String>,
}

impl Gs2 {
    /// A side from `start`.
    pub fn new(start: BranchStart) -> Self {
        let last = start.settings.max_steps;
        let step = start.settings.step_size;
        Self {
            walk: Walk::new(start, last, HessianUpdate::Bofill),
            minimizer: None,
            anchor: None,
            step,
            note: None,
        }
    }

    /// Start the minimization of the next step from the last path point.
    fn begin_step(&mut self) -> MoptResult<()> {
        let Some(anchor) = &self.anchor else {
            return Ok(());
        };
        let size = norm(&anchor.g);
        let pivot = axpy(&anchor.q, -0.5 * self.step / size, &anchor.g);
        let guess = axpy(&anchor.q, -self.step / size, &anchor.g);
        let minimizer = SphereMinimizer::new(
            pivot,
            0.5 * self.step,
            &guess,
            anchor.hessian.clone(),
            self.walk.settings.sphere,
        )?;
        let x = self
            .walk
            .space
            .kept(&self.walk.space.from_mw(minimizer.point()));
        self.minimizer = Some(minimizer);
        self.walk.auxiliary(x, false);
        Ok(())
    }

    /// The surface at a path point: its record, and the sphere of the next step.
    fn at_path_point(&mut self, value: &PointValue) -> MoptResult<StepRecord> {
        let Arrival {
            mut record,
            end,
            q,
            g,
            ..
        } = self.walk.arrive(value)?;
        record.notes.extend(self.note.take());
        if end.is_some() {
            return Ok(record);
        }
        if self.walk.is_last() {
            self.walk.stop(BranchEnd::MaxSteps);
            return Ok(record);
        }
        if norm(&g) == 0.0 {
            // No direction to go on.
            self.walk.stop(BranchEnd::Converged);
            return Ok(record);
        }
        self.anchor = Some(Anchor {
            hessian: self.walk.model.total(&self.walk.space, value),
            energy: value.energy,
            q,
            g,
        });
        self.begin_step()?;
        Ok(record)
    }
}

impl Integrator for Gs2 {
    fn name(&self) -> &'static str {
        "gs2"
    }

    fn ask(&self) -> BranchAction {
        self.walk.next.clone()
    }

    fn tell(&mut self, value: &PointValue) -> MoptResult<Option<StepRecord>> {
        let Some(mut minimizer) = self.minimizer.take() else {
            return self.at_path_point(value).map(Some);
        };
        self.walk.check(value)?;
        let g = self.walk.space.gradient_to_mw(&value.gradient);
        let status = minimizer.tell(value.energy, &g)?;
        let x = self
            .walk
            .space
            .kept(&self.walk.space.from_mw(minimizer.point()));
        if status == SphereStatus::Continue {
            self.minimizer = Some(minimizer);
            self.walk.auxiliary(x, false);
            return Ok(None);
        }
        let rises = self
            .anchor
            .as_ref()
            .is_some_and(|a| value.energy > a.energy);
        if rises {
            // The step passed a minimum: repeat it with half the length, or end.
            let smallest = self.walk.settings.step_size / SMALLEST_STEP_DIVISOR;
            if 0.5 * self.step < smallest {
                self.walk.stop(BranchEnd::EnergyRise);
            } else {
                self.step *= 0.5;
                self.begin_step()?;
            }
            return Ok(None);
        }
        self.note = Some(format!(
            "Hypersphere minimization: {} evaluation(s), RMS gradient within the sphere \
             {:.3e}, step {:.6}{}",
            minimizer.iterations(),
            minimizer.tangential_rms(),
            self.step,
            if status == SphereStatus::Exhausted {
                " (not converged)"
            } else {
                ""
            }
        ));
        // The point just evaluated is the next path point.
        self.walk.advance(x);
        match &self.walk.next {
            BranchAction::Evaluate(request) if !request.need_hessian => {
                self.at_path_point(value).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn current(&self) -> &Col<f64> {
        &self.walk.x
    }
}
