//! Hessian-based predictor-corrector (HPC): an LQA step predicts the next point, the surface
//! is evaluated there, and the step is taken again on the distance-weighted interpolant of
//! the surface between the last point and the predicted one ([`super::dwi`]). One
//! evaluation per step: the energy and the gradient at the corrected point come from the
//! interpolant, the Hessian from the predicted point.
//!
//! Everything is done in mass-weighted coordinates, as the method is published. MultiOptPy
//! builds the interpolant and integrates the corrector in Cartesian coordinates with
//! gradients that are not mass-weighted, and mass-weights its starting Hessian three times
//! (`docs/source/migration_paths.md`). As in MultiOptPy the corrector is integrated in 100 Euler
//! steps, the table shows the values of the surface at the predicted point, and the
//! structures are the corrected points.
//!
//! References: H. P. Hratchian and H. B. Schlegel, J. Chem. Phys. 2004, 120, 9918 and
//! J. Chem. Theory Comput. 2005, 1, 61.
//!
//! MultiOptPy: `IRC/hpc.py:HPC` (`step`, `run`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::Col;

use super::convergence::convergence_check;
use super::dwi::{DwiSurface, TaylorPoint};
use super::history::{HessianUpdate, PathHistory, SurfaceModel};
use super::integrator::{BranchAction, BranchStart, Integrator};
use super::lqa::lqa_step;
use super::settings::IrcSettings;
use super::space::IrcSpace;
use super::types::{rms, BranchEnd, PointRequest, PointRole, PointValue, StepRecord};
use crate::error::{MoptError, MoptResult};

/// MultiOptPy's number of Euler steps of the corrector.
pub const N_CORRECTOR: usize = 100;

/// The HPC integrator.
pub struct Hpc {
    space: Arc<IrcSpace>,
    settings: IrcSettings,
    /// Number of the point under evaluation (0: the start).
    step: usize,
    /// The point under evaluation: the start, then the predicted points.
    evaluated: Col<f64>,
    /// The last point of the path (corrected).
    x: Col<f64>,
    /// The last point of the path with the surface expanded about it.
    anchor: Option<TaylorPoint>,
    history: PathHistory,
    model: SurfaceModel,
    next: BranchAction,
    last_step: usize,
}

impl Hpc {
    /// A side from `start`.
    pub fn new(start: BranchStart) -> Self {
        // MultiOptPy: point 0, then `for iter in range(1, max_step)`.
        let last_step = start.settings.max_steps.saturating_sub(1);
        let next = BranchAction::Evaluate(PointRequest {
            x: start.x.clone(),
            need_hessian: start.settings.exact_hessian_at(0),
            role: PointRole::Path,
            step: 0,
        });
        Self {
            space: start.space,
            step: 0,
            evaluated: start.x.clone(),
            x: start.x,
            anchor: None,
            history: PathHistory::new(),
            model: SurfaceModel::new(start.hessian, HessianUpdate::GuardedBfgs),
            next,
            last_step,
            settings: start.settings,
        }
    }

    /// The corrected point on the interpolant between the last point and `predicted`.
    fn correct(&self, anchor: &TaylorPoint, predicted: TaylorPoint) -> (Col<f64>, TaylorPoint) {
        let hessian = predicted.hessian.clone();
        let surface = DwiSurface::new(anchor.clone(), predicted);
        let reached = surface.descend(&anchor.x, self.settings.step_size, N_CORRECTOR);
        let x = self.space.placed(&self.space.from_mw(&reached));
        let q = self.space.to_mw(&x);
        let point = TaylorPoint {
            energy: surface.energy(&q),
            gradient: surface.gradient(&q),
            hessian,
            x: q,
        };
        (x, point)
    }
}

impl Integrator for Hpc {
    fn name(&self) -> &'static str {
        "hpc"
    }

    fn ask(&self) -> BranchAction {
        self.next.clone()
    }

    fn tell(&mut self, value: &PointValue) -> MoptResult<Option<StepRecord>> {
        if !value.is_finite() || value.gradient.nrows() != self.space.dimension() {
            return Err(MoptError::calc(format!(
                "IRC step {}: the surface returned a value that is not finite or of another size",
                self.step
            )));
        }
        self.model.observe(&self.space, &self.evaluated, value)?;
        let here = TaylorPoint {
            x: self.space.to_mw(&self.evaluated),
            energy: value.energy,
            gradient: self.space.gradient_to_mw(&value.gradient),
            hessian: self.model.total(&self.space, value),
        };
        let mut notes = Vec::new();
        let point = match self.anchor.take() {
            None => here,
            Some(anchor) => {
                let (x, point) = self.correct(&anchor, here);
                self.x = x;
                notes.push(format!(
                    "Bias Energy (corr) :  {}",
                    crate::core_infra::log::python_float(point.energy)
                ));
                point
            }
        };
        let bending_angle = self.history.push(point.x.clone(), point.energy);
        let record = StepRecord {
            step: self.step,
            x: self.x.clone(),
            energy: value.raw_energy,
            biased_energy: value.energy,
            rms_gradient: rms(&value.raw_gradient),
            rms_biased_gradient: rms(&value.gradient),
            arc_length: self.history.arc_length(),
            bending_angle,
            extras: Vec::new(),
            curvature: None,
            in_table: true,
            notes,
            curvature_across: super::integrator::across_path(&self.space, &self.evaluated, value)?,
        };
        let s = &self.settings;
        let arrived = self.step > s.min_points
            && convergence_check(
                &self.space.gradient_from_mw(&point.gradient),
                s.max_force,
                s.rms_force,
            );
        self.next = if self.history.reversals() >= s.oscillation_limit {
            BranchAction::Finished(BranchEnd::Oscillation)
        } else if s.stop_on_energy_rise && self.history.rose() {
            BranchAction::Finished(BranchEnd::EnergyRise)
        } else if arrived {
            BranchAction::Finished(BranchEnd::Converged)
        } else if self.step >= self.last_step {
            BranchAction::Finished(BranchEnd::MaxSteps)
        } else {
            // The predictor: an LQA step from the last point of the path.
            let dq = lqa_step(&point.hessian, &point.gradient, s.step_size, 1e-6)?;
            self.evaluated = self.space.displaced(&self.x, &dq);
            self.step += 1;
            BranchAction::Evaluate(PointRequest {
                x: self.evaluated.clone(),
                need_hessian: s.exact_hessian_at(self.step),
                role: PointRole::Path,
                step: self.step,
            })
        };
        self.anchor = Some(point);
        Ok(Some(record))
    }

    fn current(&self) -> &Col<f64> {
        &self.x
    }
}
