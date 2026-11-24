//! The interface of the integrators of a side of the path, and what they share.
//!
//! An integrator asks for one point at a time and is told the surface there; it never
//! evaluates anything and writes nothing. `Walk` holds what MultiOptPy's integrators repeat
//! in their loops: the step counter, the last points, the Hessian, the row of the table for
//! every point, and the two tests that end a side (oscillation of the energy, arrival at a
//! stationary point).
//!
//! The step control of the walks that take failed steps again and go on to the minimum is
//! in [`super::control`].
//!
//! MultiOptPy: the common parts of `run` in `IRC/lqa.py`, `IRC/rk4.py`, `IRC/euler.py`, and
//! `IRC/dvv.py`; `PESAnalyzer/calc_irc_curvature.py` for the curvature that the Euler and RK4
//! integrators report.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::history::{HessianUpdate, PathHistory, SurfaceModel};
use super::settings::IrcSettings;
use super::space::IrcSpace;
use super::types::{
    axpy, dot, norm, rms, BranchEnd, PathCurvature, PointRequest, PointRole, PointValue,
    StepCounts, StepRecord,
};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// The lowest curvature across the path at `x` of the surface that is followed, where `value`
/// has an exact Hessian (`IrcSpace::curvature_across`).
pub(crate) fn across_path(
    space: &IrcSpace,
    x: &Col<f64>,
    value: &PointValue,
) -> MoptResult<Option<f64>> {
    let Some(h) = &value.hessian else {
        return Ok(None);
    };
    let total = match &value.bias_hessian {
        Some(b) => h + b,
        None => h.clone(),
    };
    space.curvature_across(&total, x, &value.gradient)
}

/// Where a side starts.
#[derive(Debug, Clone)]
pub struct BranchStart {
    /// The first point of the side (already displaced from the transition state).
    pub x: Col<f64>,
    /// The mass-weighted Hessian of the surface without bias potentials at the transition
    /// state (rigid-body motion projected out).
    pub hessian: Mat<f64>,
    /// The settings of the calculation.
    pub settings: IrcSettings,
    /// The variables and their masses.
    pub space: Arc<IrcSpace>,
    /// Energy of the surface that is followed at the structure the side starts from (the
    /// transition state); `None`: the energy of the first point of the side is taken.
    pub energy: Option<f64>,
}

/// What an integrator wants next.
#[derive(Debug, Clone)]
pub enum BranchAction {
    /// The surface at this point.
    Evaluate(PointRequest),
    /// Nothing: the side ended.
    Finished(BranchEnd),
}

/// An integrator of one side of the path.
pub trait Integrator: Send {
    /// Name of the method.
    fn name(&self) -> &'static str;
    /// The point to evaluate next, or the end of the side.
    fn ask(&self) -> BranchAction;
    /// The surface at the point asked for. Returns the record of a point of the path when
    /// the value completed one.
    fn tell(&mut self, value: &PointValue) -> MoptResult<Option<StepRecord>>;
    /// The last point of the path.
    fn current(&self) -> &Col<f64>;
    /// What the step control did so far (integrators without one: nothing).
    fn counts(&self) -> StepCounts {
        StepCounts::default()
    }
}

/// A path point that was just evaluated.
#[derive(Clone)]
pub(crate) struct Arrival {
    /// Its row for the files.
    pub record: StepRecord,
    /// The end of the side, when this point is its last.
    pub end: Option<BranchEnd>,
    /// The point in mass-weighted coordinates.
    pub q: Col<f64>,
    /// The mass-weighted gradient of the surface that is followed.
    pub g: Col<f64>,
    /// The surface at the point (the one the next step starts from).
    pub value: PointValue,
}

/// Step counter, history, and Hessian of a side whose path points are all evaluated.
pub(crate) struct Walk {
    pub space: Arc<IrcSpace>,
    pub settings: IrcSettings,
    /// Number of the path point `x` (from 1).
    pub step: usize,
    /// The path point that is being evaluated, or the last one.
    pub x: Col<f64>,
    pub history: PathHistory,
    pub model: SurfaceModel,
    pub next: BranchAction,
    pub(super) last_step: usize,
    with_curvature: bool,
    pub(super) tangent: Option<(Col<f64>, Col<f64>)>,
    /// The last point of the path, for a step that is taken again.
    pub(super) accepted: Option<Arrival>,
    /// Factor on the step length (halved after a step that raised the energy).
    pub(super) scale: f64,
    pub(super) halvings: usize,
    /// Waiting for the exact Hessian at the last point of the path.
    pub(super) refreshing: bool,
    pub(super) refreshed: bool,
    /// What the step control did.
    pub counts: StepCounts,
    /// Longest step the bend of the valley allows at the last point of the path.
    pub(super) bend_limit: Option<f64>,
    /// Energy of the surface that is followed at the start of the side (the transition
    /// state), or at its first point when that was not given.
    pub(super) reference_energy: Option<f64>,
    /// Whether the last point of the path lay in the bowl of the minimum
    /// ([`Walk::approach`]).
    pub(super) in_a_bowl: bool,
    /// Whether the integrator calls [`Self::approach`].
    pub(super) approaches: bool,
    /// Whether the side is in its final approach.
    pub(super) approaching: bool,
    /// Whether steps that raise the energy are taken again.
    pub(super) retrying: bool,
}

pub(super) fn request(
    x: &Col<f64>,
    step: usize,
    settings: &IrcSettings,
    role: PointRole,
) -> PointRequest {
    PointRequest {
        x: x.clone(),
        need_hessian: role == PointRole::Path && settings.exact_hessian_at(step),
        role,
        step,
    }
}

impl Walk {
    /// A side from `start` that ends after the point `last_step` at the latest.
    pub fn new(start: BranchStart, last_step: usize, update: HessianUpdate) -> Self {
        let next = if last_step == 0 {
            BranchAction::Finished(BranchEnd::MaxSteps)
        } else {
            BranchAction::Evaluate(request(&start.x, 1, &start.settings, PointRole::Path))
        };
        Self {
            space: start.space,
            step: 1,
            x: start.x,
            history: PathHistory::new(),
            model: SurfaceModel::new(start.hessian, update),
            next,
            last_step,
            with_curvature: false,
            tangent: None,
            accepted: None,
            scale: 1.0,
            halvings: 0,
            refreshing: false,
            refreshed: false,
            counts: StepCounts::default(),
            bend_limit: None,
            reference_energy: start.energy,
            in_a_bowl: false,
            approaches: false,
            approaching: false,
            retrying: false,
            settings: start.settings,
        }
    }

    /// Report the curvature of the path with every point (Euler and RK4).
    pub fn with_curvature(mut self) -> Self {
        self.with_curvature = true;
        self
    }

    /// The integrator calls [`Self::approach`] at every point of the path: the arrival is
    /// decided there, not by the gradient alone.
    pub fn with_final_approach(mut self) -> Self {
        self.approaches = true;
        self
    }

    /// Fail unless `value` fits the space and is finite.
    pub fn check(&self, value: &PointValue) -> MoptResult<()> {
        let n = self.space.dimension();
        if value.gradient.nrows() != n || value.raw_gradient.nrows() != n {
            return Err(MoptError::config(format!(
                "IRC: expected gradients of {n} components"
            )));
        }
        if !value.is_finite() {
            return Err(MoptError::calc(format!(
                "IRC step {}: the surface returned a value that is not finite",
                self.step
            )));
        }
        Ok(())
    }

    /// Take in the surface at the path point `x`: Hessian, history, the row of the table,
    /// and the tests that end the side.
    pub fn arrive(&mut self, value: &PointValue) -> MoptResult<Arrival> {
        self.check(value)?;
        self.reference_energy.get_or_insert(value.energy);
        self.model.observe(&self.space, &self.x, value)?;
        let q = self.space.to_mw(&self.x);
        let g = self.space.gradient_to_mw(&value.gradient);
        let bending_angle = self.history.push(q.clone(), value.energy);
        let curvature = if self.with_curvature {
            self.curvature(&q, &g, value)?
        } else {
            None
        };
        let curvature_across = across_path(&self.space, &self.x, value)?;
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
            curvature,
            in_table: true,
            notes: Vec::new(),
            curvature_across,
        };
        let s = &self.settings;
        // With steps that are taken again when they raise the energy there is nothing
        // left for the count of reversals to catch but rounding at the minimum.
        let end = if !self.retrying && self.history.reversals() >= s.oscillation_limit {
            Some(BranchEnd::Oscillation)
        } else if s.stop_on_energy_rise && self.history.rose() {
            Some(BranchEnd::EnergyRise)
        } else if !(s.final_approach && self.approaches) && self.forces_arrived(value) {
            Some(BranchEnd::Converged)
        } else {
            None
        };
        if let Some(end) = end {
            self.next = BranchAction::Finished(end);
        }
        Ok(Arrival {
            record,
            end,
            q,
            g,
            value: value.clone(),
        })
    }

    /// Change of the unit tangent of the path per length since the last point, with its
    /// components along the modes of positive curvature.
    fn curvature(
        &mut self,
        q: &Col<f64>,
        g: &Col<f64>,
        value: &PointValue,
    ) -> MoptResult<Option<PathCurvature>> {
        let length = norm(g);
        if length == 0.0 {
            return Ok(None);
        }
        let tangent = g * (-1.0 / length);
        let previous = self.tangent.replace((tangent.clone(), q.clone()));
        let Some((t0, q0)) = previous else {
            return Ok(None);
        };
        let ds = norm(&(q - &q0));
        if ds == 0.0 {
            return Ok(None);
        }
        let kappa = (&tangent - &t0) * (1.0 / ds);
        let (values, vectors) = symmetric_eigen(&self.model.total(&self.space, value))?;
        let n = values.nrows();
        let couplings = (0..n)
            .filter(|&k| values[k] > 1e-8)
            .map(|k| (0..n).map(|i| vectors[(i, k)] * kappa[i]).sum())
            .collect();
        Ok(Some(PathCurvature {
            scalar: norm(&kappa),
            couplings,
        }))
    }

    /// Go on to the path point `x` (the side ends when the steps are used up).
    pub fn advance(&mut self, x: Col<f64>) {
        self.x = x;
        self.step += 1;
        self.next = if self.step > self.last_step {
            BranchAction::Finished(BranchEnd::MaxSteps)
        } else {
            BranchAction::Evaluate(request(&self.x, self.step, &self.settings, PointRole::Path))
        };
    }

    /// Whether the path point `x` is the last one the steps allow.
    pub fn is_last(&self) -> bool {
        self.step >= self.last_step
    }

    /// End the side.
    pub fn stop(&mut self, end: BranchEnd) {
        self.next = BranchAction::Finished(end);
    }

    /// Ask for the surface at `x` between two path points.
    pub fn auxiliary(&mut self, x: Col<f64>, need_hessian: bool) {
        self.next = BranchAction::Evaluate(PointRequest {
            x,
            need_hessian,
            role: PointRole::Auxiliary,
            step: self.step,
        });
    }

    /// The point reached from `x` by `length` along the downhill direction of the
    /// mass-weighted gradient `g` (unchanged where the gradient vanishes).
    pub fn downhill(&self, g: &Col<f64>, length: f64) -> Col<f64> {
        let size = norm(g);
        if size == 0.0 {
            return self.x.clone();
        }
        self.space.displaced(&self.x, &(g * (-length / size)))
    }
}

/// `x + f * d` for a direction in mass-weighted coordinates, without recentring (the stages
/// between two path points).
pub(crate) fn offset(space: &IrcSpace, q: &Col<f64>, f: f64, d: &Col<f64>) -> Col<f64> {
    space.from_mw(&axpy(q, f, d))
}

/// `v` scaled to unit length (zero stays zero).
pub(crate) fn unit(v: &Col<f64>) -> Col<f64> {
    let size = norm(v);
    if size == 0.0 {
        v.clone()
    } else {
        v * (1.0 / size)
    }
}

/// The component of `v` along the unit vector `direction`.
pub(crate) fn component(v: &Col<f64>, direction: &Col<f64>) -> f64 {
    dot(v, direction)
}
