//! Test of the first point away from the transition state, and its correction on a
//! hypersphere around the transition state.
//!
//! Integrators without a constrained optimization (LQA, HPC, DVV, Euler, RK4) leave the
//! transition state by a straight displacement `q_TS +- L v` along the transition vector
//! `v`. That point is on the path only as far as the quadratic model of the transition
//! state holds over the length `L`. Two numbers measure it at the trial point (all in
//! mass-weighted coordinates):
//!
//! - the angle between the downhill direction `-g` there and the displacement: on the path
//!   of the quadratic model the gradient at `q_TS + L v` is `lambda L v`, parallel to the
//!   displacement, so the angle is zero;
//! - the ratio `rho = dE_actual / dE_predicted` of the energy change to the one of the
//!   quadratic model, `dE_predicted = g_TS . dq + lambda L^2 / 2` (the trust ratio of
//!   trust-region methods; the first term vanishes at a converged transition state).
//!
//! When the angle is above its limit or `rho` outside `[rho_min, rho_max]`, the first point
//! is replaced by the minimum of the energy on the hypersphere `|q - q_TS| = L`, found from
//! the trial point with [`super::sphere::SphereMinimizer`]. The sphere is centred at the
//! transition state; the spheres of GS2 ([`super::gs2`]) are centred at pivot points half a
//! step away from the last path point. The numbers, whether the correction ran, and its
//! evaluations are reported.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::history::angle_deg;
use super::settings::FirstStepSettings;
use super::space::IrcSpace;
use super::sphere::{SphereMinimizer, SphereStatus};
use super::types::{dot, norm, Branch, PointValue};
use crate::error::{MoptError, MoptResult};

/// The two numbers of the test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstStepTest {
    /// Angle (degrees) between the downhill direction at the trial point and the
    /// displacement from the transition state.
    pub angle_deg: f64,
    /// Actual over predicted energy change.
    pub rho: f64,
    /// Energy change of the quadratic model of the transition state.
    pub predicted_change: f64,
    /// Energy change found.
    pub actual_change: f64,
}

impl FirstStepTest {
    /// Whether both numbers are within the limits of `settings`.
    pub fn passes(&self, settings: &FirstStepSettings) -> bool {
        self.angle_deg <= settings.max_angle_deg
            && self.rho >= settings.rho_min
            && self.rho <= settings.rho_max
    }
}

/// Test a trial point at the mass-weighted displacement `displacement` from the transition
/// state, where the energy is `energy_ts`, the mass-weighted gradient `gradient_ts`, and
/// the curvature along the displacement `eigenvalue`.
pub fn test_first_step(
    displacement: &Col<f64>,
    eigenvalue: f64,
    energy_ts: f64,
    gradient_ts: &Col<f64>,
    energy_trial: f64,
    gradient_trial: &Col<f64>,
) -> FirstStepTest {
    let length = norm(displacement);
    let predicted_change = dot(gradient_ts, displacement) + 0.5 * eigenvalue * length * length;
    let actual_change = energy_trial - energy_ts;
    let downhill = gradient_trial * -1.0;
    FirstStepTest {
        angle_deg: angle_deg(&downhill, displacement).unwrap_or(0.0),
        rho: actual_change / predicted_change,
        predicted_change,
        actual_change,
    }
}

/// The first step of a side as it was tested and, where needed, corrected.
#[derive(Debug, Clone)]
pub struct FirstStepReport {
    /// The side.
    pub branch: Branch,
    /// Mass-weighted length of the displacement from the transition state.
    pub length: f64,
    /// The numbers of the test at the trial point.
    pub test: FirstStepTest,
    /// Whether the point was replaced by the minimum on the hypersphere.
    pub corrected: bool,
    /// Evaluations of the minimization (0 without it).
    pub iterations: usize,
    /// Whether the minimization converged (true without it).
    pub converged: bool,
    /// Mass-weighted distance between the trial point and the corrected one.
    pub moved: f64,
    /// Energy change from the trial point to the corrected one.
    pub energy_gain: f64,
}

/// What the test needs of the transition state.
pub(crate) struct Origin {
    pub q: Col<f64>,
    pub energy: f64,
    pub gradient: Col<f64>,
    pub eigenvalue: f64,
    pub total_hessian: Mat<f64>,
}

/// The test and correction of one side, one evaluation at a time.
pub(crate) struct FirstStep {
    space: Arc<IrcSpace>,
    settings: FirstStepSettings,
    branch: Branch,
    origin: Arc<Origin>,
    trial: Col<f64>,
    x: Col<f64>,
    minimizer: Option<SphereMinimizer>,
    test: Option<FirstStepTest>,
    trial_energy: f64,
}

/// The end of a first step: where the side starts, the surface there, and the report.
pub(crate) struct FirstStepResult {
    pub x: Col<f64>,
    pub value: PointValue,
    pub report: FirstStepReport,
}

impl FirstStep {
    /// Test the trial point `trial` (Cartesian) of `branch`.
    pub fn new(
        space: Arc<IrcSpace>,
        settings: FirstStepSettings,
        branch: Branch,
        origin: Arc<Origin>,
        trial: Col<f64>,
    ) -> Self {
        Self {
            space,
            settings,
            branch,
            origin,
            x: trial.clone(),
            trial,
            minimizer: None,
            test: None,
            trial_energy: 0.0,
        }
    }

    /// The point to evaluate next.
    pub fn point(&self) -> &Col<f64> {
        &self.x
    }

    fn report(
        &self,
        test: FirstStepTest,
        value: &PointValue,
        status: Option<SphereStatus>,
    ) -> FirstStepReport {
        let moved = norm(&self.space.to_mw(&(&self.x - &self.trial)));
        FirstStepReport {
            branch: self.branch,
            length: norm(&(self.space.to_mw(&self.trial) - &self.origin.q)),
            test,
            corrected: status.is_some(),
            iterations: self
                .minimizer
                .as_ref()
                .map_or(0, SphereMinimizer::iterations),
            converged: status != Some(SphereStatus::Exhausted),
            moved,
            energy_gain: if status.is_some() {
                value.energy - self.trial_energy
            } else {
                0.0
            },
        }
    }

    /// The surface at [`Self::point`]; the result once the first point is settled.
    pub fn tell(&mut self, value: PointValue) -> MoptResult<Option<FirstStepResult>> {
        if !value.is_finite() || value.gradient.nrows() != self.space.dimension() {
            return Err(MoptError::calc(
                "IRC first step: the surface returned a value that is not finite or of another size",
            ));
        }
        let g = self.space.gradient_to_mw(&value.gradient);
        let Some(test) = self.test else {
            // The trial point.
            let q = self.space.to_mw(&self.trial);
            let displacement = &q - &self.origin.q;
            let test = test_first_step(
                &displacement,
                self.origin.eigenvalue,
                self.origin.energy,
                &self.origin.gradient,
                value.energy,
                &g,
            );
            self.test = Some(test);
            self.trial_energy = value.energy;
            if test.passes(&self.settings) {
                let report = self.report(test, &value, None);
                return Ok(Some(FirstStepResult {
                    x: self.x.clone(),
                    value,
                    report,
                }));
            }
            // Centre: the transition state (GS2 centres its spheres at pivot points).
            let mut minimizer = SphereMinimizer::new(
                self.origin.q.clone(),
                norm(&displacement),
                &q,
                self.origin.total_hessian.clone(),
                self.settings.sphere,
            )?
            .with_rigid_motions(&self.space);
            let status = minimizer.tell(value.energy, &g)?;
            return self.after(minimizer, status, test, value);
        };
        let Some(mut minimizer) = self.minimizer.take() else {
            return Err(MoptError::config("IRC first step: nothing was asked for"));
        };
        let status = minimizer.tell(value.energy, &g)?;
        self.after(minimizer, status, test, value)
    }

    fn after(
        &mut self,
        minimizer: SphereMinimizer,
        status: SphereStatus,
        test: FirstStepTest,
        value: PointValue,
    ) -> MoptResult<Option<FirstStepResult>> {
        if status == SphereStatus::Continue {
            self.x = self.space.kept(&self.space.from_mw(minimizer.point()));
            self.minimizer = Some(minimizer);
            return Ok(None);
        }
        self.minimizer = Some(minimizer);
        let report = self.report(test, &value, Some(status));
        Ok(Some(FirstStepResult {
            x: self.x.clone(),
            value,
            report,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_quadratic_model_passes_and_a_bent_valley_fails() {
        let settings = FirstStepSettings::default();
        let d = Col::from_fn(2, |i| [0.2, 0.0][i]);
        let zero = Col::<f64>::zeros(2);
        // Exactly quadratic: E = -0.5 * 0.3 * x^2, gradient -0.3 x along the displacement.
        let g = Col::from_fn(2, |i| [-0.06, 0.0][i]);
        let t = test_first_step(&d, -0.3, 1.0, &zero, 1.0 - 0.006, &g);
        assert!(t.angle_deg.abs() < 1e-9 && (t.rho - 1.0).abs() < 1e-12);
        assert!(t.passes(&settings));
        // The valley bends away: the gradient points 30 degrees off.
        let g = Col::from_fn(2, |i| [-0.06 * 0.866, 0.06 * 0.5][i]);
        let t = test_first_step(&d, -0.3, 1.0, &zero, 1.0 - 0.006, &g);
        assert!((t.angle_deg - 30.0).abs() < 0.01);
        assert!(!t.passes(&settings));
        // The energy falls half as much as the model says.
        let g = Col::from_fn(2, |i| [-0.06, 0.0][i]);
        let t = test_first_step(&d, -0.3, 1.0, &zero, 1.0 - 0.003, &g);
        assert!((t.rho - 0.5).abs() < 1e-12 && !t.passes(&settings));
        // A gradient at the start enters the prediction.
        let g0 = Col::from_fn(2, |i| [-0.01, 0.0][i]);
        let t = test_first_step(&d, -0.3, 1.0, &g0, 1.0 - 0.008, &g);
        assert!((t.predicted_change + 0.008).abs() < 1e-15 && (t.rho - 1.0).abs() < 1e-12);
    }
}
