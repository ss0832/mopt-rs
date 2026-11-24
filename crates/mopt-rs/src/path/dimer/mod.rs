//! The dimer method: a saddle point of order 1 from one structure without Hessians.
//!
//! A dimer of two points, the centre `x` and the end point `x + dR N`, is rotated about its
//! centre until its axis `N` lies along the direction of lowest curvature ([`rotation`]; the
//! gradient at the rotated end point is extrapolated, and evaluated anew after a rotation by
//! more than [`EXTRAPOLATION_LIMIT`] trial angles; for structures the rigid-body motions are
//! taken out of the axis, [`Dimer::with_rigid_body`]), and
//! the centre is moved with the gradient whose component along the axis is inverted,
//! `g_eff = g - 2 (g . N) N` while the curvature `C` along the axis is negative; while it is
//! positive the centre is moved uphill along the axis by a step of the largest length
//! (`g_eff = -(g . N) N`, the effective force of Henkelman and Jonsson, as a fixed step).
//! The translation is an L-BFGS step on `g_eff` (Kastner and Sherwood) limited to the largest
//! length; its memory is cleared when the curvature changes sign. The search has converged
//! when the gradient at the centre meets the force thresholds and the curvature is negative.
//!
//! G. Henkelman, H. Jonsson, J. Chem. Phys. 1999, 111, 7010; A. Heyden, A. T. Bell, F. J. Keil,
//! J. Chem. Phys. 2005, 123, 224101; J. Kastner, P. Sherwood, J. Chem. Phys. 2008, 128, 014106.
//!
//! [`Dimer`] is an ask/tell state machine over points of `n` variables; the axis to start with
//! comes from the caller (towards a second structure, or any direction).
//!
//! MultiOptPy: `OtherMethod/dimer.py:DimerMethod`. MultiOptPy treats the gradients its
//! calculators return as forces (every sign of the method is inverted), ignores `-dimer_sep`,
//! `-dimer_trial_angle`, and `-dimer_maxiter` (its own constants are used), starts from a random
//! axis, translates by a fixed 0.1 Bohr in every iteration (so it cannot settle on the saddle),
//! and evaluates both end points; here the method follows the papers and the options are used.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod rotation;

use std::collections::VecDeque;

use faer::Col;

use super::points::{check_values, gradient_of, PointRequest, PointValue, SearchAction};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_max_abs, col_norm};
use crate::math::projection::{build_tr_rot_basis, project_out_basis_vec};
use crate::opt::convergence::ConvergenceCriteria;
use rotation::{curvature_and_force, rotate, trial_axis};

/// Settings of a dimer search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimerSettings {
    /// Distance `dR` from the centre to the end point (Bohr; `-dimer_sep`).
    pub separation: f64,
    /// Angle of the trial rotation (rad; `-dimer_trial_angle`).
    pub trial_angle: f64,
    /// Iterations (translations) at most (`-dimer_maxiter`).
    pub max_iterations: usize,
    /// Rotations at most before a translation.
    pub max_rotations: usize,
    /// The rotations stop when the last one turned the axis by less than this (rad).
    pub rotation_tolerance: f64,
    /// No rotation when the rotational force (per length) is below this.
    pub rotational_force_tolerance: f64,
    /// Largest length of a translation (Bohr).
    pub max_step: f64,
    /// Pairs of the L-BFGS memory of the translation.
    pub memory: usize,
    /// The force thresholds at the centre (the displacement thresholds are not used).
    pub criteria: ConvergenceCriteria,
    /// How the centre is translated while the curvature is negative.
    pub translation: DimerTranslation,
    /// Step per unit of the gradient across the axis (`Plain`, `Optimistic`,
    /// `Extragradient`; Bohr^2/Hartree for molecules).
    pub rate_across: f64,
    /// Step per unit of the gradient along the axis (the climb; the two time scales of a
    /// descent-ascent).
    pub rate_along: f64,
}

/// The translation of the centre while the curvature is negative, a descent across the axis
/// and an ascent along it (`-(g - 2 (g . N) N)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimerTranslation {
    /// L-BFGS steps on the effective gradient (Kastner, Sherwood 2008).
    Lbfgs,
    /// Plain descent-ascent steps `-a G` with `G = (g - (g . N) N) rate_across / a - (g . N)
    /// N rate_along / a` (mopt-rs, experimental).
    Plain,
    /// Optimistic steps `-(2 G_k - G_k-1)` (Popov 1980; Daskalakis et al. 2018; no extra
    /// gradient; mopt-rs, experimental).
    Optimistic,
    /// Extragradient steps: `G` at the point a step ahead, one extra gradient per step
    /// (Korpelevich 1976; mopt-rs, experimental).
    Extragradient,
}

impl DimerTranslation {
    /// The translation called `name` (`lbfgs`, `plain`, `optimistic`, `extragradient`).
    pub fn from_name(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "lbfgs" | "l-bfgs" => Ok(Self::Lbfgs),
            "plain" | "gda" => Ok(Self::Plain),
            "optimistic" | "ogda" => Ok(Self::Optimistic),
            "extragradient" | "eg" => Ok(Self::Extragradient),
            other => Err(MoptError::parse(format!(
                "dimer translation '{other}': lbfgs, plain, optimistic, or extragradient"
            ))),
        }
    }
}

impl Default for DimerSettings {
    fn default() -> Self {
        Self {
            separation: 1e-4,
            trial_angle: std::f64::consts::PI / 32.0,
            max_iterations: 1000,
            max_rotations: 8,
            rotation_tolerance: 1.0_f64.to_radians(),
            rotational_force_tolerance: 1e-4,
            max_step: 0.1,
            memory: 10,
            criteria: ConvergenceCriteria::NORMAL,
            translation: DimerTranslation::Lbfgs,
            rate_across: 0.5,
            rate_along: 0.5,
        }
    }
}

/// Why a dimer search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimerEnd {
    /// The centre meets the force thresholds with a negative curvature along the axis.
    Converged,
    /// The iterations are used up.
    IterationLimit,
}

/// After a rotation by more than this multiple of the trial angle the gradient at the new
/// end point is evaluated instead of extrapolated: the extrapolation multiplies the error of
/// the one-sided difference by up to `sin(phi) / sin(phi_t)` (measured on HCN with GFN0-xTB:
/// three rotations of about 85 degrees each turned a curvature of 0.10 into -1.6).
pub const EXTRAPOLATION_LIMIT: f64 = 3.0;

#[derive(Debug, Clone)]
enum Phase {
    /// Centre and end point.
    Pair,
    /// The end point after a large rotation.
    EndPoint,
    /// The point a step ahead of the centre (extragradient).
    LookAhead(Col<f64>),
    /// A trial rotation; `direction` is the plane of the rotation.
    Trial {
        direction: Col<f64>,
        force: f64,
        trial: Col<f64>,
    },
    Done(DimerEnd),
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct DimerReport {
    /// Iteration (translations so far).
    pub iteration: usize,
    /// Energy at the centre.
    pub energy: f64,
    /// Curvature along the axis.
    pub curvature: f64,
    /// Largest component of the gradient at the centre.
    pub max_force: f64,
    /// RMS of the gradient at the centre.
    pub rms_force: f64,
    /// Angle of the rotation made (rad), if one was.
    pub rotation: Option<f64>,
    /// Length of the translation made, if one was.
    pub step: Option<f64>,
    /// The centre that was evaluated.
    pub center: Col<f64>,
    /// The axis after the rotations.
    pub axis: Col<f64>,
    /// Why the search ended, if it did.
    pub end: Option<DimerEnd>,
}

/// Ask/tell state machine of the dimer method.
pub struct Dimer {
    settings: DimerSettings,
    x: Col<f64>,
    axis: Col<f64>,
    frozen: Vec<usize>,
    energy: f64,
    g0: Col<f64>,
    g1: Col<f64>,
    curvature: f64,
    rotations: usize,
    iteration: usize,
    memory: VecDeque<(Col<f64>, Col<f64>)>,
    last: Option<(Col<f64>, Col<f64>, bool)>,
    phase: Phase,
    rigid_body: Option<MolecularSystem>,
    /// The scaled effective gradient of the last translation (optimistic steps).
    previous: Option<Col<f64>>,
    step_map: Option<super::points::StepMap>,
}

impl Dimer {
    /// Take the translations of the centre through `map`, which turns the step from `x` to
    /// `y` into the point it reaches (the point put back onto constraints).
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs).
    pub fn with_step_map(mut self, map: super::points::StepMap) -> Self {
        self.step_map = Some(map);
        self
    }

    /// The centre a translation by `step` reaches.
    fn translated(&self, step: &Col<f64>) -> MoptResult<Col<f64>> {
        let to = &self.x + step;
        match &self.step_map {
            Some(map) => map(&self.x, &to),
            None => Ok(to),
        }
    }

    /// For structures on surfaces without a preferred orientation (their atoms): the
    /// rigid-body motions at the centre are taken out of the axis, the gradients, and the
    /// steps. Without this a dimer at a minimum turns into a rotation of the whole
    /// structure, the direction of zero curvature.
    pub fn with_rigid_body(mut self, template: MolecularSystem) -> MoptResult<Self> {
        if template.dof() != self.x.nrows() {
            return Err(MoptError::config(
                "dimer: the structure and the start differ in size",
            ));
        }
        self.rigid_body = Some(template);
        let axis = self.masked(&self.axis);
        let size = col_norm(&axis);
        if size == 0.0 {
            return Err(MoptError::config(
                "dimer: the axis is a rigid-body motion only",
            ));
        }
        self.axis = axis * (1.0 / size);
        Ok(self)
    }
    /// A search from `start` with the axis `axis` (any length but zero).
    pub fn new(settings: DimerSettings, start: &Col<f64>, axis: &Col<f64>) -> MoptResult<Self> {
        let n = start.nrows();
        if n == 0 || axis.nrows() != n {
            return Err(MoptError::config(
                "dimer: the axis and the start differ in size",
            ));
        }
        let size = col_norm(axis);
        if !(size.is_finite() && size > 0.0) {
            return Err(MoptError::config("dimer: the axis must not be zero"));
        }
        let positive = [settings.separation, settings.trial_angle, settings.max_step];
        if positive.iter().any(|v| !(v.is_finite() && *v > 0.0)) || settings.max_iterations == 0 {
            return Err(MoptError::config(
                "dimer: the separation, the trial angle, the step, and the iterations must be \
                 positive",
            ));
        }
        Ok(Self {
            settings,
            x: start.clone(),
            axis: axis * (1.0 / size),
            frozen: Vec::new(),
            energy: f64::NAN,
            g0: Col::zeros(n),
            g1: Col::zeros(n),
            curvature: f64::NAN,
            rotations: 0,
            iteration: 0,
            memory: VecDeque::new(),
            last: None,
            phase: Phase::Pair,
            rigid_body: None,
            previous: None,
            step_map: None,
        })
    }

    /// Keep these variables where they are (fixed atoms).
    pub fn with_frozen(mut self, variables: Vec<usize>) -> MoptResult<Self> {
        if variables.iter().any(|&k| k >= self.x.nrows()) {
            return Err(MoptError::config(
                "dimer: a frozen variable is out of range",
            ));
        }
        self.frozen = variables;
        let axis = self.masked(&self.axis);
        let size = col_norm(&axis);
        if size == 0.0 {
            return Err(MoptError::config("dimer: the axis moves fixed atoms only"));
        }
        self.axis = axis * (1.0 / size);
        Ok(self)
    }

    fn masked(&self, v: &Col<f64>) -> Col<f64> {
        let mut out = v.clone();
        if let Some(sys) = self
            .rigid_body
            .as_ref()
            .and_then(|t| t.with_flat_coords_bohr(&self.x).ok())
        {
            out = project_out_basis_vec(&build_tr_rot_basis(&sys, false), &out);
        }
        for &k in &self.frozen {
            out[k] = 0.0;
        }
        out
    }

    /// The centre (after the end: the converged or last one).
    pub fn center(&self) -> &Col<f64> {
        &self.x
    }

    /// The axis.
    pub fn axis(&self) -> &Col<f64> {
        &self.axis
    }

    /// The curvature along the axis at the last centre.
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Iterations (translations) so far.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    /// Why the search ended, once it has.
    pub fn end(&self) -> Option<DimerEnd> {
        match self.phase {
            Phase::Done(end) => Some(end),
            _ => None,
        }
    }

    fn end_point(&self, axis: &Col<f64>) -> Col<f64> {
        &self.x + axis * self.settings.separation
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        match &self.phase {
            Phase::Pair => SearchAction::Evaluate(vec![
                PointRequest::gradient(self.x.clone()),
                PointRequest::gradient(self.end_point(&self.axis)),
            ]),
            Phase::Trial { trial, .. } => {
                SearchAction::Evaluate(vec![PointRequest::gradient(self.end_point(trial))])
            }
            Phase::EndPoint => {
                SearchAction::Evaluate(vec![PointRequest::gradient(self.end_point(&self.axis))])
            }
            Phase::LookAhead(x) => SearchAction::Evaluate(vec![PointRequest::gradient(x.clone())]),
            Phase::Done(_) => SearchAction::Finished,
        }
    }

    /// The values of the points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<DimerReport> {
        let asked = match self.ask() {
            SearchAction::Evaluate(points) => points,
            SearchAction::Finished => {
                return Err(MoptError::config(
                    "tell() called after the dimer search ended",
                ))
            }
        };
        check_values("dimer", &asked, &values, self.x.nrows())?;
        let mut rotation = None;
        match std::mem::replace(&mut self.phase, Phase::Pair) {
            Phase::Pair => {
                self.energy = values[0].energy;
                self.g0 = self.masked(&gradient_of("dimer", &values[0])?);
                self.g1 = self.masked(&gradient_of("dimer", &values[1])?);
                self.rotations = 0;
                let (c, _) =
                    curvature_and_force(&self.g0, &self.g1, &self.axis, self.settings.separation);
                self.curvature = c;
            }
            Phase::Trial {
                direction,
                force,
                trial,
            } => {
                let g_trial = self.masked(&gradient_of("dimer", &values[0])?);
                let (c_trial, _) =
                    curvature_and_force(&self.g0, &g_trial, &trial, self.settings.separation);
                let r = rotate(
                    &self.axis,
                    &direction,
                    &self.g0,
                    self.curvature,
                    &self.g1,
                    force,
                    c_trial,
                    &g_trial,
                    self.settings.trial_angle,
                );
                self.axis = r.axis;
                self.g1 = r.end_gradient;
                self.rotations += 1;
                rotation = Some(r.angle);
                if r.angle.abs() > EXTRAPOLATION_LIMIT * self.settings.trial_angle {
                    self.curvature = r.curvature;
                    self.phase = Phase::EndPoint;
                    return Ok(self.report(rotation, None));
                }
                // The curvature of the extrapolated end point (the fit's for a quadratic
                // surface), so that the next rotation starts from consistent values.
                self.curvature =
                    curvature_and_force(&self.g0, &self.g1, &self.axis, self.settings.separation).0;
            }
            Phase::LookAhead(_) => {
                // The extragradient step: the scaled effective gradient a step ahead, with the
                // axis of the centre.
                let g = self.masked(&gradient_of("dimer", &values[0])?);
                let step = self.capped(&(self.scaled(&g) * -1.0));
                self.x = self.translated(&step)?;
                self.iteration += 1;
                self.phase = if self.iteration >= self.settings.max_iterations {
                    Phase::Done(DimerEnd::IterationLimit)
                } else {
                    Phase::Pair
                };
                return Ok(self.report(None, Some(col_norm(&step))));
            }
            Phase::EndPoint => {
                self.g1 = self.masked(&gradient_of("dimer", &values[0])?);
                self.curvature =
                    curvature_and_force(&self.g0, &self.g1, &self.axis, self.settings.separation).0;
            }
            Phase::Done(end) => {
                self.phase = Phase::Done(end);
                return Err(MoptError::config(
                    "tell() called after the dimer search ended",
                ));
            }
        }
        let rotated_enough = rotation.is_some_and(|a| a.abs() < self.settings.rotation_tolerance);
        if !rotated_enough && self.rotations < self.settings.max_rotations {
            let (_, force) =
                curvature_and_force(&self.g0, &self.g1, &self.axis, self.settings.separation);
            let size = col_norm(&force);
            if size > self.settings.rotational_force_tolerance {
                let direction = &force * (1.0 / size);
                let trial = trial_axis(&self.axis, &direction, self.settings.trial_angle);
                self.phase = Phase::Trial {
                    direction,
                    force: size,
                    trial,
                };
                return Ok(self.report(rotation, None));
            }
        }
        // The rotations are done: test the centre, then translate.
        let c = &self.settings.criteria;
        let max = col_max_abs(&self.g0);
        let free = (self.x.nrows() - self.frozen.len()).max(1);
        let rms = libm::sqrt(col_dot(&self.g0, &self.g0) / free as f64);
        if self.curvature < 0.0 && max < c.max_force.value() && rms < c.rms_force.value() {
            self.phase = Phase::Done(DimerEnd::Converged);
            return Ok(self.report(rotation, None));
        }
        if self.curvature < 0.0 && self.settings.translation == DimerTranslation::Extragradient {
            let ahead = &self.x + &self.capped(&(self.scaled(&self.g0) * -1.0));
            self.phase = Phase::LookAhead(ahead);
            return Ok(self.report(rotation, None));
        }
        let step = self.translation();
        self.x = self.translated(&step)?;
        self.iteration += 1;
        self.phase = if self.iteration >= self.settings.max_iterations {
            Phase::Done(DimerEnd::IterationLimit)
        } else {
            Phase::Pair
        };
        Ok(self.report(rotation, Some(col_norm(&step))))
    }

    fn report(&self, rotation: Option<f64>, step: Option<f64>) -> DimerReport {
        let free = (self.x.nrows() - self.frozen.len()).max(1);
        DimerReport {
            iteration: self.iteration,
            energy: self.energy,
            curvature: self.curvature,
            max_force: col_max_abs(&self.g0),
            rms_force: libm::sqrt(col_dot(&self.g0, &self.g0) / free as f64),
            rotation,
            step,
            center: self.x.clone(),
            axis: self.axis.clone(),
            end: self.end(),
        }
    }

    /// The step of the centre.
    fn translation(&mut self) -> Col<f64> {
        let along = col_dot(&self.g0, &self.axis);
        let negative = self.curvature < 0.0;
        if !negative {
            // Uphill along the axis by the largest step.
            self.memory.clear();
            self.last = None;
            self.previous = None;
            let sign = if along < 0.0 { -1.0 } else { 1.0 };
            return self.masked(&(&self.axis * (sign * self.settings.max_step)));
        }
        match self.settings.translation {
            DimerTranslation::Plain | DimerTranslation::Extragradient => {
                self.previous = None;
                return self.capped(&(self.scaled(&self.g0) * -1.0));
            }
            DimerTranslation::Optimistic => {
                let now = self.scaled(&self.g0);
                let step = match self.previous.take() {
                    Some(before) => (&now * 2.0 - before) * -1.0,
                    None => &now * -1.0,
                };
                self.previous = Some(now);
                return self.capped(&step);
            }
            DimerTranslation::Lbfgs => {}
        }
        let effective = &self.g0 - &self.axis * (2.0 * along);
        if let Some((x_old, g_old, was_negative)) = self.last.take() {
            if was_negative {
                let s = &self.x - &x_old;
                let y = &effective - &g_old;
                if col_dot(&s, &y) > 1e-12 {
                    self.memory.push_back((s, y));
                    while self.memory.len() > self.settings.memory {
                        self.memory.pop_front();
                    }
                }
            } else {
                self.memory.clear();
            }
        }
        let mut step = lbfgs(&self.memory, &effective);
        if col_dot(&step, &effective) >= 0.0 {
            self.memory.clear();
            step = &effective * -1.0;
        }
        let step = self.masked(&step);
        let length = col_norm(&step);
        let step = if length > self.settings.max_step {
            step * (self.settings.max_step / length)
        } else {
            step
        };
        self.last = Some((self.x.clone(), effective, true));
        step
    }
}

impl Dimer {
    /// The scaled effective gradient of a descent-ascent step at the gradient `g`:
    /// `rate_across (g - (g . N) N) - rate_along (g . N) N`.
    fn scaled(&self, g: &Col<f64>) -> Col<f64> {
        let along = col_dot(g, &self.axis);
        let across = g - &self.axis * along;
        self.masked(
            &(across * self.settings.rate_across - &self.axis * (along * self.settings.rate_along)),
        )
    }

    /// `step` without fixed atoms and rigid-body motions, at most `max_step` long.
    fn capped(&self, step: &Col<f64>) -> Col<f64> {
        let step = self.masked(step);
        let length = col_norm(&step);
        if length > self.settings.max_step {
            step * (self.settings.max_step / length)
        } else {
            step
        }
    }
}

/// The L-BFGS step `-H g` (two-loop recursion; the unit matrix without pairs).
fn lbfgs(memory: &VecDeque<(Col<f64>, Col<f64>)>, g: &Col<f64>) -> Col<f64> {
    let Some((s_last, y_last)) = memory.back() else {
        return g * -1.0;
    };
    let k = memory.len();
    let rho: Vec<f64> = memory.iter().map(|(s, y)| 1.0 / col_dot(y, s)).collect();
    let mut alpha = vec![0.0; k];
    let mut q = g.clone();
    for i in (0..k).rev() {
        alpha[i] = rho[i] * col_dot(&memory[i].0, &q);
        q -= &memory[i].1 * alpha[i];
    }
    let mut r = q * (col_dot(s_last, y_last) / col_dot(y_last, y_last));
    for i in 0..k {
        let b = rho[i] * col_dot(&memory[i].1, &r);
        r += &memory[i].0 * (alpha[i] - b);
    }
    r * -1.0
}

#[cfg(test)]
mod tests;
