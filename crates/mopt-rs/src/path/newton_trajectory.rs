//! The growing Newton trajectory (GNT): a curve of points whose gradient is parallel to a
//! fixed direction `r`, grown from a minimum one point at a time until it has passed over a
//! transition state.
//!
//! Every Newton trajectory of a direction `r` connects stationary points, and every point of
//! it has `P g = 0` with `P = 1 - r r^T`. The trajectory grows by a predictor step of a fixed
//! length along `r` from its last point and a corrector: the energy is minimized in the
//! hyperplane through the predicted point perpendicular to `r` (with the projected gradient
//! `P g`) until the RMS of `P g` is below a threshold. When the energies of the last three
//! points have a maximum, the trajectory has passed a transition state, which the middle point
//! approximates; the search ends there. After a minimum has been passed the direction is
//! taken anew (towards the final structure, or along the given bonds) if it has turned away
//! from the first one (overlap at most 0.5).
//!
//! W. Quapp, J. Chem. Phys. 2005, 122, 174106 (the growing string of a Newton trajectory);
//! W. Quapp, M. Hirsch, O. Imig, D. Heidrich, J. Comput. Chem. 1998, 19, 1087 (reduced gradient
//! following). The predictor, the threshold 1.7e-3 of the RMS of the projected gradient, and the
//! update of the direction follow pysisyphus (`cos/GrowingNT.py`), whose procedure MultiOptPy
//! ports.
//!
//! Hard constraints `c(x) = t` ([`GrowingNewtonTrajectory::with_manifold`]; new in mopt-rs,
//! not in the papers): the trajectory is the Newton trajectory of the energy on their
//! manifold `M`, the curve where the gradient on `M` (`P_M g`, which the surfaces give) is
//! parallel to `P_M r`. The direction is taken into `T_x M` where it is set (at the start,
//! and when it is taken anew after a minimum), the predictor step goes along it, and the
//! corrector minimizes the energy on `M` intersected with the plane `r . x = r . x_pred`:
//! an optimizer with both held exactly ([`crate::opt::vector::VectorOptimizer::
//! set_constraints`], the restore after every step), which starts from the predicted point
//! moved onto both. Its projected gradient is `g` without its part along the normals of the
//! constraints and of the plane at `x` together; where it vanishes, `P_M g` is parallel to
//! `P_M r` (the Lagrange condition of the minimum on `M` and the plane), the condition of a
//! Newton trajectory on `M`. The points of the trajectory are the points of the corrector:
//! each holds the constraints.
//!
//! [`GrowingNewtonTrajectory`] is an ask/tell state machine over points of `n` variables; the
//! corrector is an optimizer of [`crate::opt::vector`] (any `-opt` minimizer) told the projected
//! gradient.
//!
//! MultiOptPy: `OtherMethod/newton_traj.py:NewtonTrajectory`. MultiOptPy takes a new direction
//! from the current structure alone after a minimum and stops with an error when it has no
//! bonds (`-gnt_vec`), the usual case of a final structure; its corrector is a BFGS step on the
//! projected gradient within 0.02 Bohr. Here the direction is taken anew towards the final
//! structure as in pysisyphus.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::ieip::aligned_pair;
use super::manifold::PointManifold;
use super::points::{check_values, gradient_of, PointRequest, PointValue, SearchAction};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};
use crate::math::projection::{build_tr_rot_basis, project_out_basis_vec};
use crate::opt::state_machine::Termination;
use crate::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};
use crate::opt::vector_constraints::{VectorConstraint, VectorConstraintSet};

/// Where the direction of the trajectory comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum NtDirection {
    /// Towards this final point (laid onto the current one for structures).
    Toward(Col<f64>),
    /// The bonds between these pairs of atoms (indices from 0) are shortened: atom `i` moves
    /// towards `j` and `j` towards `i` (MultiOptPy `-gnt_vec`).
    Bonds(Vec<(usize, usize)>),
    /// A fixed direction.
    Fixed(Col<f64>),
}

/// Settings of a growing Newton trajectory.
#[derive(Debug, Clone)]
pub struct NtSettings {
    /// Length of a predictor step (Bohr; `-gnt_step`).
    pub step_length: f64,
    /// The corrector ends when the RMS of the projected gradient is below this.
    pub rms_tolerance: f64,
    /// Corrector iterations per round (`-gnt_mi`); a round that does not converge counts as
    /// one iteration of the run and the corrector goes on.
    pub corrector_iterations: usize,
    /// Iterations of the run at most (growths and corrector rounds; `-ns`).
    pub max_iterations: usize,
    /// Overlap with the first direction below which a new direction is taken after a
    /// minimum (0.5).
    pub update_overlap: f64,
    /// The minimizer of the corrector.
    pub minimizer: VectorSettings,
    /// Lay the final structure onto the current one (structures without a preferred
    /// orientation).
    pub align: bool,
    /// For structures on surfaces without a preferred orientation: their atoms. The
    /// rigid-body motions at the current point are then taken out of the direction and of the
    /// projected gradient, so that the corrector cannot meet the hyperplane by turning the
    /// whole structure.
    pub rigid_body: Option<MolecularSystem>,
}

impl Default for NtSettings {
    fn default() -> Self {
        Self {
            step_length: 0.5,
            rms_tolerance: 1.7e-3,
            corrector_iterations: 25,
            max_iterations: 999,
            update_overlap: 0.5,
            minimizer: VectorSettings::default(),
            align: true,
            rigid_body: None,
        }
    }
}

/// Why a Newton trajectory ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NtEnd {
    /// It passed a transition state.
    PassedTransitionState,
    /// The iterations are used up.
    IterationLimit,
}

/// A point of the trajectory.
#[derive(Debug, Clone)]
pub struct NtPoint {
    /// The point.
    pub x: Col<f64>,
    /// Its energy.
    pub energy: f64,
    /// Its gradient.
    pub gradient: Col<f64>,
    /// RMS of the projected gradient.
    pub projected_rms: f64,
    /// Corrector iterations it took.
    pub corrector_iterations: usize,
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct NtReport {
    /// Iteration of the run.
    pub iteration: usize,
    /// The point that was evaluated.
    pub x: Col<f64>,
    /// Its energy.
    pub energy: f64,
    /// RMS of the projected gradient there.
    pub projected_rms: f64,
    /// The point added to the trajectory, if one was.
    pub added: Option<NtPoint>,
    /// Whether a minimum or a transition state was passed (by the energies of the last three
    /// points).
    pub passed: Option<&'static str>,
    /// Whether the direction was taken anew.
    pub direction_updated: bool,
    /// Why the run ended, if it did.
    pub end: Option<NtEnd>,
}

/// Ask/tell state machine of the growing Newton trajectory.
pub struct GrowingNewtonTrajectory {
    settings: NtSettings,
    source: NtDirection,
    frozen: Vec<usize>,
    r: Col<f64>,
    r_first: Col<f64>,
    points: Vec<NtPoint>,
    corrector: Option<VectorOptimizer>,
    in_round: usize,
    on_point: usize,
    iteration: usize,
    start: Col<f64>,
    transition_state: Option<usize>,
    end: Option<NtEnd>,
    manifold: Option<Arc<PointManifold>>,
    /// The constraints of the corrector in use (the hard constraints and its plane).
    plane: Option<VectorConstraintSet>,
}

/// The plane of a corrector: `r . x` held at the value of the predicted point.
struct Plane {
    r: Col<f64>,
}

impl VectorConstraint for Plane {
    fn name(&self) -> String {
        "the plane of the corrector".to_string()
    }

    fn evaluate(&self, x: &Col<f64>) -> MoptResult<(f64, Col<f64>)> {
        Ok((col_dot(&self.r, x), self.r.clone()))
    }
}

impl GrowingNewtonTrajectory {
    /// A trajectory from the minimum `start`.
    pub fn new(settings: NtSettings, start: &Col<f64>, source: NtDirection) -> MoptResult<Self> {
        if !(settings.step_length.is_finite() && settings.step_length > 0.0)
            || settings.rms_tolerance.is_nan()
            || settings.rms_tolerance <= 0.0
            || settings.corrector_iterations == 0
            || settings.max_iterations == 0
        {
            return Err(MoptError::config(
                "Newton trajectory: the step, the tolerance, and the counts must be positive",
            ));
        }
        let mut nt = Self {
            r: Col::zeros(start.nrows()),
            r_first: Col::zeros(start.nrows()),
            settings,
            source,
            frozen: Vec::new(),
            points: Vec::new(),
            corrector: None,
            in_round: 0,
            on_point: 0,
            iteration: 0,
            start: start.clone(),
            transition_state: None,
            end: None,
            manifold: None,
            plane: None,
        };
        nt.r = nt.direction(start)?;
        nt.r_first = nt.r.clone();
        Ok(nt)
    }

    /// Keep these variables where they are (fixed atoms).
    pub fn with_frozen(mut self, variables: Vec<usize>) -> MoptResult<Self> {
        if variables.iter().any(|&k| k >= self.start.nrows()) {
            return Err(MoptError::config(
                "Newton trajectory: a frozen variable is out of range",
            ));
        }
        self.frozen = variables;
        self.settings.align = false;
        self.r = self.direction(&self.start.clone())?;
        self.r_first = self.r.clone();
        Ok(self)
    }

    /// Hold the points on the hard constraints `manifold` (the start lies on them): the
    /// direction is taken into their tangent space and the corrector minimizes on them and
    /// on its plane together.
    pub fn with_manifold(mut self, manifold: Arc<PointManifold>) -> MoptResult<Self> {
        self.manifold = Some(manifold);
        self.r = self.direction(&self.start.clone())?;
        self.r_first = self.r.clone();
        Ok(self)
    }

    /// The unit direction at the point `x`.
    fn direction(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        let n = x.nrows();
        let mut r = match &self.source {
            NtDirection::Toward(target) => {
                if target.nrows() != n {
                    return Err(MoptError::config(
                        "Newton trajectory: the final structure differs in size",
                    ));
                }
                if self.settings.align && n.is_multiple_of(3) {
                    // The target turned onto x (both centred), shifted back to x's centroid.
                    let (turned, centred) = aligned_pair(target, x);
                    &turned - &centred
                } else {
                    target - x
                }
            }
            NtDirection::Bonds(pairs) => {
                let mut r = Col::<f64>::zeros(n);
                for &(i, j) in pairs {
                    if 3 * i.max(j) + 2 >= n {
                        return Err(MoptError::config(
                            "Newton trajectory: an atom of -gnt_vec is out of range",
                        ));
                    }
                    for c in 0..3 {
                        let d = x[3 * j + c] - x[3 * i + c];
                        r[3 * i + c] = d;
                        r[3 * j + c] = -d;
                    }
                }
                r
            }
            NtDirection::Fixed(r) => r.clone(),
        };
        for &k in &self.frozen {
            r[k] = 0.0;
        }
        if let Some(d) = self.rigid_basis(x) {
            r = project_out_basis_vec(&d, &r);
        }
        if let Some(m) = &self.manifold {
            r = m.project(x, &r)?;
        }
        let size = col_norm(&r);
        if !(size.is_finite() && size > 0.0) {
            return Err(MoptError::config(
                "Newton trajectory: the direction is zero",
            ));
        }
        Ok(r * (1.0 / size))
    }

    /// The rigid-body motions at `x` (orthonormal columns), for structures that have them.
    fn rigid_basis(&self, x: &Col<f64>) -> Option<Mat<f64>> {
        let template = self.settings.rigid_body.as_ref()?;
        let sys = template.with_flat_coords_bohr(x).ok()?;
        Some(build_tr_rot_basis(&sys, false))
    }

    fn projected(&self, g: &Col<f64>, x: &Col<f64>) -> MoptResult<Col<f64>> {
        let mut p = match &self.plane {
            // Without the normals of the constraints and of the plane at x together.
            Some(set) => set.frame(x)?.project(g),
            None => g - &self.r * col_dot(g, &self.r),
        };
        if let Some(d) = self.rigid_basis(x) {
            p = project_out_basis_vec(&d, &p);
        }
        for &k in &self.frozen {
            p[k] = 0.0;
        }
        Ok(p)
    }

    /// The points of the trajectory so far.
    pub fn points(&self) -> &[NtPoint] {
        &self.points
    }

    /// The direction in use.
    pub fn direction_in_use(&self) -> &Col<f64> {
        &self.r
    }

    /// The point that approximates the transition state, once it was passed.
    pub fn transition_state_estimate(&self) -> Option<&NtPoint> {
        self.transition_state.and_then(|k| self.points.get(k))
    }

    /// Why the run ended, once it has.
    pub fn end(&self) -> Option<NtEnd> {
        self.end
    }

    /// Iterations of the run so far.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    fn asked_point(&self) -> Option<Col<f64>> {
        match &self.corrector {
            Some(c) => match c.ask() {
                VectorAction::Evaluate { x, .. } => Some(x),
                VectorAction::Finished(_) => None,
            },
            None => self.points.is_empty().then(|| self.start.clone()),
        }
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        if self.end.is_some() {
            return SearchAction::Finished;
        }
        match self.asked_point() {
            Some(x) => SearchAction::Evaluate(vec![PointRequest::gradient(x)]),
            None => SearchAction::Finished,
        }
    }

    /// The value at the point of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<NtReport> {
        let x = self
            .asked_point()
            .filter(|_| self.end.is_none())
            .ok_or_else(|| MoptError::config("tell() called after the Newton trajectory ended"))?;
        check_values(
            "Newton trajectory",
            &[PointRequest::gradient(x.clone())],
            &values,
            x.nrows(),
        )?;
        let energy = values[0].energy;
        let g = gradient_of("Newton trajectory", &values[0])?;
        let pg = self.projected(&g, &x)?;
        let free = (x.nrows() - self.frozen.len()).max(1);
        let projected_rms = libm::sqrt(col_dot(&pg, &pg) / free as f64);
        let mut report = NtReport {
            iteration: self.iteration,
            x: x.clone(),
            energy,
            projected_rms,
            added: None,
            passed: None,
            direction_updated: false,
            end: None,
        };
        let accept = match self.corrector.as_mut() {
            None => true,
            Some(c) => {
                self.on_point += 1;
                self.in_round += 1;
                if projected_rms <= self.settings.rms_tolerance {
                    true
                } else {
                    c.tell(energy, pg, None)?;
                    // The corrector cannot get further: the point is taken as it is.
                    matches!(
                        c.termination(),
                        Some(Termination::Converged | Termination::MaxIterations)
                    )
                }
            }
        };
        if !accept {
            if self.in_round >= self.settings.corrector_iterations {
                self.in_round = 0;
                self.next_iteration();
            }
            report.end = self.end;
            return Ok(report);
        }
        let point = NtPoint {
            x,
            energy,
            gradient: g,
            projected_rms,
            corrector_iterations: self.on_point,
        };
        self.points.push(point.clone());
        report.added = Some(point);
        report.passed = self.passed();
        if report.passed == Some("transition state") {
            self.transition_state = Some(self.points.len() - 2);
            self.end = Some(NtEnd::PassedTransitionState);
            report.end = self.end;
            return Ok(report);
        }
        if report.passed == Some("minimum") {
            let last = self.points[self.points.len() - 1].x.clone();
            let r_new = self.direction(&last)?;
            if col_dot(&r_new, &self.r_first) <= self.settings.update_overlap {
                self.r = r_new;
                report.direction_updated = true;
            }
        }
        self.grow()?;
        self.next_iteration();
        report.end = self.end;
        Ok(report)
    }

    fn next_iteration(&mut self) {
        self.iteration += 1;
        if self.end.is_none() && self.iteration >= self.settings.max_iterations {
            self.end = Some(NtEnd::IterationLimit);
        }
    }

    /// A minimum or a transition state among the last three points.
    fn passed(&self) -> Option<&'static str> {
        let k = self.points.len();
        if k < 3 {
            return None;
        }
        let e = |i: usize| self.points[k - i].energy;
        if e(3) > e(2) && e(2) < e(1) {
            Some("minimum")
        } else if e(3) < e(2) && e(2) > e(1) {
            Some("transition state")
        } else {
            None
        }
    }

    /// The predictor step from the last point and a new corrector.
    fn grow(&mut self) -> MoptResult<()> {
        let last = self
            .points
            .last()
            .ok_or_else(|| MoptError::calc("Newton trajectory: no point to grow from"))?;
        let predicted = &last.x + &self.r * self.settings.step_length;
        let mut minimizer = self.settings.minimizer.clone();
        minimizer.saddle_order = 0;
        minimizer.max_iterations = minimizer.max_iterations.max(1);
        let mut corrector = VectorOptimizer::new(minimizer, predicted.clone())?;
        if let Some(m) = &self.manifold {
            // The corrector on the constraints and on the plane through the predicted point
            // perpendicular to r: it starts from the predicted point moved onto both.
            let mut set = m.vector_constraints()?;
            set.push(
                Arc::new(Plane { r: self.r.clone() }),
                col_dot(&self.r, &predicted),
            )?;
            corrector.set_constraints(set.clone())?;
            self.plane = Some(set);
        }
        self.corrector = Some(corrector);
        self.on_point = 0;
        self.in_round = 0;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// `E = (x^2 - 1)^2 + 2 (y - 0.2 x)^2`.
    fn surface(p: &Col<f64>) -> PointValue {
        let (x, y) = (p[0], p[1]);
        let w = y - 0.2 * x;
        PointValue {
            energy: (x * x - 1.0).powi(2) + 2.0 * w * w,
            gradient: Some(Col::from_fn(2, |i| {
                [4.0 * x * (x * x - 1.0) - 0.8 * w, 4.0 * w][i]
            })),
            hessian: None,
        }
    }

    #[test]
    fn grows_over_the_saddle() {
        let settings = NtSettings {
            step_length: 0.15,
            align: false,
            ..NtSettings::default()
        };
        let start = Col::from_fn(2, |i| [-1.0, -0.2][i]);
        let end = Col::from_fn(2, |i| [1.0, 0.2][i]);
        let mut nt =
            GrowingNewtonTrajectory::new(settings, &start, NtDirection::Toward(end)).unwrap();
        while let SearchAction::Evaluate(points) = nt.ask() {
            let values = points.iter().map(|p| surface(&p.x)).collect();
            nt.tell(values).unwrap();
        }
        assert_eq!(nt.end(), Some(NtEnd::PassedTransitionState));
        let ts = nt.transition_state_estimate().unwrap();
        assert!(col_norm(&ts.x) < 0.1, "{:?}", ts.x);
        // Every point of the trajectory has its gradient along the direction.
        for p in nt.points() {
            assert!(p.projected_rms <= 1.7e-3 + 1e-12);
        }
    }
}
