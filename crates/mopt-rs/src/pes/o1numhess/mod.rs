//! O1NumHess: a Hessian from a number of gradients that does not grow with the size of the
//! system (B. Wang, S. Luo, Z. Wang, W. Liu, J. Chem. Theory Comput. 2025, 21, 10893,
//! arXiv:2508.07544; reference implementations `ilcpm/O1NumHess` and `ilcpm/O1NumHess_QC`,
//! MIT).
//!
//! The Hessian elements between variables that are far apart are small, so the gradient is
//! differentiated along directions that excite all neighbourhoods at once (`directions.rs`),
//! and the Hessian is fitted to those derivatives (`fit.rs`): elements within the cutoff
//! exactly, elements beyond it damped, and the remainder as a low-rank correction. For a
//! molecule the neighbourhoods come from the distances between the van der Waals spheres, the
//! model Hessian is the modified Swart model, and translations and rotations cost no gradient
//! (`molecule.rs`).
//!
//! The procedure is a state machine that does no calculations itself:
//!
//! ```text
//! let mut job = O1NumHess::new(problem, settings)?;
//! while let O1Action::Evaluate(points) = job.ask() {
//!     job.tell(&gradients_at(points))?;
//! }
//! let hessian = job.hessian();
//! ```
//!
//! The first round asks for the gradients at the displaced points. If the Hessian fitted to
//! them has negative eigenvalues, a second round asks for the gradients along those
//! eigenvectors, so that the curvature of every negative mode is measured rather than fitted;
//! eigenvalues that then lie below the measured curvatures are spurious and are reflected
//! (the `version 2` procedure of `O1NumHess_QC`).
//!
//! [`O1NumHessCalculator`] wraps any calculator and answers Hessian requests this way;
//! `-modelhess o1numhess` uses it every `-mfc` iterations.
//!
//! MultiOptPy: `ModelHessian/o1numhess.py:O1NumHessCalculator` (which departs from the
//!  published method: neighbours from covalent radii, rotations by finite differences, the
//!  eigenvector of the largest absolute eigenvalue, unit-norm steps, a momentum term in the
//!  low-rank loop, no second round; this module follows the publication).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod calculator;
mod directions;
mod fit;
mod molecule;

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen};

pub use calculator::{
    hessian_from_gradients, hessian_from_gradients_parallel, is_o1numhess, wrap_if_requested,
    O1Hessian, O1NumHessCalculator,
};
pub use molecule::{molecular_problem, vdw_distance_matrix, Invariance};

/// Parameters of O1NumHess (the defaults are the published ones).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct O1Settings {
    /// Largest component of a displacement (Bohr for molecules): `0.005`.
    pub delta: f64,
    /// Variables closer than this are neighbours, whose Hessian elements are fitted without
    /// damping: `1.0` (Bohr between van der Waals spheres).
    pub dmax: f64,
    /// Elements up to `dmax + ddmax` are fitted with damping, those beyond come from the
    /// low-rank correction only: `5.0`.
    pub ddmax: f64,
    /// Strength of the damping: `1e-2`.
    pub lambda: f64,
    /// Half the power of the distance in the damping: `1.5`.
    pub beta: f64,
    /// Relative residual at which the local fit stops: `1e-10`.
    pub local_tolerance: f64,
    /// Iterations of the local fit at most: `1000`.
    pub local_iterations: usize,
    /// Residual `|G - H D|` at which the low-rank correction stops: `1e-8`.
    pub low_rank_threshold: f64,
    /// Iterations of the low-rank correction at most: `100`.
    pub low_rank_iterations: usize,
    /// Points on both sides of every direction: twice the gradients, and an error of second
    /// instead of first order in the displacement. Off by default, as published.
    pub central: bool,
    /// Eigenvalues below minus this are negative: `1e-8`.
    pub negative_threshold: f64,
    /// Differentiate the gradient along the eigenvectors of negative eigenvalues in a second
    /// round.
    pub follow_negative_modes: bool,
    /// After the second round, reflect eigenvalues below the measured curvatures.
    pub remove_spurious_negative: bool,
}

impl Default for O1Settings {
    fn default() -> Self {
        Self {
            delta: 0.005,
            dmax: 1.0,
            ddmax: 5.0,
            lambda: 1e-2,
            beta: 1.5,
            local_tolerance: 1e-10,
            local_iterations: 1000,
            low_rank_threshold: 1e-8,
            low_rank_iterations: 100,
            central: false,
            negative_threshold: 1e-8,
            follow_negative_modes: true,
            remove_spurious_negative: true,
        }
    }
}

/// A direction given at the start.
#[derive(Debug, Clone)]
pub struct InitialDirection {
    /// The direction (normalized by [`O1NumHess::new`]).
    pub vector: Col<f64>,
    /// The derivative of the gradient along the unit direction when it is known without a
    /// calculation (zero for translations, from the gradient itself for rotations).
    pub derivative: Option<Col<f64>>,
    /// Differentiate with points on both sides.
    pub double_sided: bool,
}

/// What O1NumHess needs to know about a function.
#[derive(Debug, Clone)]
pub struct O1Problem {
    /// The point.
    pub x: Col<f64>,
    /// The gradient there.
    pub gradient: Col<f64>,
    /// Distances between the variables (symmetric).
    pub distance: Mat<f64>,
    /// A model Hessian that guides the choice of directions.
    pub model: Mat<f64>,
    /// Directions to start from.
    pub initial: Vec<InitialDirection>,
}

impl O1Problem {
    /// Variables along a line: the distance of two variables is the difference of their
    /// positions in `x`, and the model is the unit matrix (the reference implementation's
    /// defaults).
    pub fn banded(x: Col<f64>, gradient: Col<f64>) -> Self {
        let n = x.nrows();
        Self {
            x,
            gradient,
            distance: Mat::from_fn(n, n, |i, j| (i as f64 - j as f64).abs()),
            model: Mat::identity(n, n),
            initial: Vec::new(),
        }
    }
}

/// What to do next.
#[derive(Debug, Clone)]
pub enum O1Action {
    /// Evaluate the gradient at these points and pass the results to [`O1NumHess::tell`] in
    /// the same order.
    Evaluate(Vec<Col<f64>>),
    /// The Hessian is ready.
    Finished,
}

/// A point to evaluate: direction and side.
#[derive(Debug, Clone, Copy)]
struct Request {
    direction: usize,
    forward: bool,
}

#[derive(Debug, Clone)]
enum Stage {
    First,
    /// Second round, with the eigenvectors of the negative eigenvalues of the first Hessian.
    NegativeModes(Mat<f64>),
    Done,
}

/// The O1NumHess state machine (see the module documentation).
#[derive(Debug, Clone)]
pub struct O1NumHess {
    settings: O1Settings,
    x: Col<f64>,
    gradient: Col<f64>,
    distance: Mat<f64>,
    directions: Vec<Col<f64>>,
    derivatives: Vec<Option<Col<f64>>>,
    double_sided: Vec<bool>,
    /// Directions whose derivative was given.
    given: Vec<usize>,
    requests: Vec<Request>,
    stage: Stage,
    hessian: Option<Mat<f64>>,
    residual: f64,
    gradients: usize,
    notes: Vec<String>,
}

impl O1NumHess {
    /// Choose the directions for `problem`.
    pub fn new(problem: O1Problem, settings: O1Settings) -> MoptResult<Self> {
        let n = problem.x.nrows();
        let square = |m: &Mat<f64>| m.nrows() == n && m.ncols() == n;
        if problem.gradient.nrows() != n || !square(&problem.distance) || !square(&problem.model) {
            return Err(MoptError::config(format!(
                "O1NumHess: the gradient, the distances and the model must match the {n} variables"
            )));
        }
        if !(settings.delta > 0.0 && settings.delta.is_finite()) {
            return Err(MoptError::config(
                "O1NumHess: the displacement must be positive",
            ));
        }
        let mut job = Self {
            settings,
            x: problem.x,
            gradient: problem.gradient,
            distance: problem.distance,
            directions: Vec::new(),
            derivatives: Vec::new(),
            double_sided: Vec::new(),
            given: Vec::new(),
            requests: Vec::new(),
            stage: Stage::First,
            hessian: None,
            residual: 0.0,
            gradients: 0,
            notes: Vec::new(),
        };
        for start in problem.initial {
            job.add_initial(start)?;
        }
        let lists = directions::neighbor_lists(&job.distance, settings.dmax);
        directions::generate(&lists, &problem.model, &mut job.directions)?;
        job.derivatives.resize(job.directions.len(), None);
        job.double_sided
            .resize(job.directions.len(), settings.central);
        for k in 0..job.directions.len() {
            job.fix_sign(k);
        }
        job.requests = job.requests_from(0);
        if job.requests.is_empty() {
            job.finish_round()?;
        }
        Ok(job)
    }

    fn add_initial(&mut self, start: InitialDirection) -> MoptResult<()> {
        let n = self.x.nrows();
        if start.vector.nrows() != n || start.derivative.as_ref().is_some_and(|d| d.nrows() != n) {
            return Err(MoptError::config(format!(
                "O1NumHess: a starting direction does not have {n} components"
            )));
        }
        let length = col_norm(&start.vector);
        if !(length > 0.0 && length.is_finite()) {
            return Ok(());
        }
        let mut v = Col::from_fn(n, |i| start.vector[i] / length);
        let left = directions::orthogonalize(&mut v, &self.directions);
        if left < 1e-8 {
            // Already spanned by the directions before it.
            return Ok(());
        }
        if start.derivative.is_some() && (left - 1.0).abs() > 1e-8 {
            return Err(MoptError::config(
                "O1NumHess: starting directions with known derivatives must be orthogonal to \
                 the directions before them",
            ));
        }
        if start.derivative.is_some() {
            self.given.push(self.directions.len());
        }
        self.directions.push(Col::from_fn(n, |i| v[i] / left));
        self.derivatives.push(start.derivative);
        self.double_sided
            .push(start.double_sided || self.settings.central);
        Ok(())
    }

    /// Give direction `k` its conventional sign, with its derivative.
    fn fix_sign(&mut self, k: usize) {
        if directions::fix_sign(&mut self.directions[k]) {
            if let Some(d) = self.derivatives[k].as_mut() {
                for i in 0..d.nrows() {
                    d[i] = -d[i];
                }
            }
        }
    }

    /// The points needed for the directions from `first` on: forward points, then the
    /// backward points of the double-sided directions.
    fn requests_from(&self, first: usize) -> Vec<Request> {
        let open = |k: &usize| self.derivatives[*k].is_none();
        let forward = (first..self.directions.len()).filter(open);
        let backward = (first..self.directions.len())
            .filter(open)
            .filter(|&k| self.double_sided[k]);
        forward
            .map(|direction| Request {
                direction,
                forward: true,
            })
            .chain(backward.map(|direction| Request {
                direction,
                forward: false,
            }))
            .collect()
    }

    /// Largest absolute component of direction `k`.
    fn largest(&self, k: usize) -> f64 {
        self.directions[k]
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    /// Next action.
    pub fn ask(&self) -> O1Action {
        if matches!(self.stage, Stage::Done) {
            return O1Action::Finished;
        }
        let points = self
            .requests
            .iter()
            .map(|r| {
                let d = &self.directions[r.direction];
                let step = self.settings.delta / self.largest(r.direction);
                let step = if r.forward { step } else { -step };
                Col::from_fn(self.x.nrows(), |i| self.x[i] + step * d[i])
            })
            .collect();
        O1Action::Evaluate(points)
    }

    /// Take the gradients at the points of [`Self::ask`].
    pub fn tell(&mut self, gradients: &[Col<f64>]) -> MoptResult<()> {
        if matches!(self.stage, Stage::Done) {
            return Err(MoptError::config(
                "O1NumHess: tell() after the Hessian was finished",
            ));
        }
        let n = self.x.nrows();
        if gradients.len() != self.requests.len() || gradients.iter().any(|g| g.nrows() != n) {
            return Err(MoptError::config(format!(
                "O1NumHess: expected {} gradients of length {n}",
                self.requests.len()
            )));
        }
        if gradients.iter().any(|g| g.iter().any(|v| !v.is_finite())) {
            return Err(MoptError::calc("O1NumHess: a gradient is not finite"));
        }
        let delta = self.settings.delta;
        for (at, request) in self.requests.iter().enumerate().filter(|(_, r)| r.forward) {
            let k = request.direction;
            let backward = self
                .requests
                .iter()
                .position(|r| r.direction == k && !r.forward);
            let scale = self.largest(k);
            let plus = &gradients[at];
            self.derivatives[k] = Some(match backward {
                Some(b) => Col::from_fn(n, |i| (plus[i] - gradients[b][i]) / (2.0 * delta) * scale),
                None => Col::from_fn(n, |i| (plus[i] - self.gradient[i]) / delta * scale),
            });
        }
        self.gradients += gradients.len();
        self.requests.clear();
        self.finish_round()
    }

    /// The directions and derivatives as matrices.
    fn matrices(&self) -> MoptResult<(Mat<f64>, Mat<f64>)> {
        let n = self.x.nrows();
        let m = self.directions.len();
        let d = Mat::from_fn(n, m, |i, k| self.directions[k][i]);
        let mut g = Mat::<f64>::zeros(n, m);
        for (k, derivative) in self.derivatives.iter().enumerate() {
            let derivative = derivative
                .as_ref()
                .ok_or_else(|| MoptError::config("O1NumHess: a derivative is missing"))?;
            for i in 0..n {
                g[(i, k)] = derivative[i];
            }
        }
        Ok((d, g))
    }

    fn finish_round(&mut self) -> MoptResult<()> {
        let (d, g) = self.matrices()?;
        let fitted = fit::fit(&self.distance, &d, &g, &self.settings);
        if !fitted.local_converged {
            self.notes.push(format!(
                "O1NumHess: the local fit stopped before its tolerance ({} iterations).",
                self.settings.local_iterations
            ));
        }
        self.residual = fitted.residual;
        let mut hessian = fitted.hessian;
        match std::mem::replace(&mut self.stage, Stage::Done) {
            Stage::First if self.settings.follow_negative_modes => {
                if let Some(modes) = self.negative_modes(&hessian)? {
                    let first = self.directions.len();
                    for k in 0..modes.ncols() {
                        self.add_initial(InitialDirection {
                            vector: Col::from_fn(modes.nrows(), |i| modes[(i, k)]),
                            derivative: None,
                            double_sided: false,
                        })?;
                    }
                    for k in first..self.directions.len() {
                        self.fix_sign(k);
                    }
                    self.requests = self.requests_from(first);
                    if !self.requests.is_empty() {
                        self.stage = Stage::NegativeModes(modes);
                    }
                }
            }
            Stage::NegativeModes(modes) if self.settings.remove_spurious_negative => {
                let removed = self.reflect_spurious(&mut hessian, &modes, &d, &g)?;
                if removed > 0 {
                    self.notes.push(format!(
                        "O1NumHess: {removed} spurious negative eigenvalue(s) removed."
                    ));
                }
            }
            _ => {}
        }
        self.hessian = Some(hessian);
        Ok(())
    }

    /// Eigenvectors of the negative eigenvalues of `hessian` (most negative first), as many as
    /// there is room for further directions.
    fn negative_modes(&self, hessian: &Mat<f64>) -> MoptResult<Option<Mat<f64>>> {
        let n = hessian.nrows();
        let (values, vectors) = symmetric_eigen(hessian)?;
        let negative = (0..n)
            .filter(|&i| values[i] < -self.settings.negative_threshold)
            .count()
            .min(n - self.directions.len());
        Ok((negative > 0).then(|| Mat::from_fn(n, negative, |i, k| vectors[(i, k)])))
    }

    /// Reflect the eigenvalues of `hessian` (without the directions of known derivative) that
    /// lie below the curvatures measured along `modes`; the number that became positive.
    fn reflect_spurious(
        &self,
        hessian: &mut Mat<f64>,
        modes: &Mat<f64>,
        d: &Mat<f64>,
        g: &Mat<f64>,
    ) -> MoptResult<usize> {
        let n = hessian.nrows();
        let threshold = self.settings.negative_threshold;
        // v^T D G^T v: the curvature along v as measured.
        let mut measured: Vec<f64> = (0..modes.ncols())
            .map(|k| {
                let v = Col::from_fn(n, |i| modes[(i, k)]);
                let dv = d.transpose() * &v;
                let gv = g.transpose() * &v;
                col_dot(&dv, &gv).min(0.0)
            })
            .collect();
        measured.sort_by(f64::total_cmp);
        let projected = if self.given.is_empty() {
            hessian.clone()
        } else {
            let known = Mat::from_fn(n, self.given.len(), |i, k| {
                self.directions[self.given[k]][i]
            });
            crate::math::projection::project_out_basis_mat(&known, hessian)
        };
        let (values, vectors) = symmetric_eigen(&projected)?;
        let mut removed = 0;
        let mut rank = 0;
        for i in 0..n {
            let value = values[i];
            if value.abs() < threshold {
                continue;
            }
            let bound = measured[rank.min(measured.len() - 1)];
            rank += 1;
            if value >= bound {
                continue;
            }
            let shift = 2.0 * (bound - value);
            if value < -threshold && value + shift > threshold {
                removed += 1;
            }
            for a in 0..n {
                for b in 0..n {
                    hessian[(a, b)] += shift * vectors[(a, i)] * vectors[(b, i)];
                }
            }
        }
        Ok(removed)
    }

    /// The Hessian, once [`Self::ask`] returns [`O1Action::Finished`].
    pub fn hessian(&self) -> Option<&Mat<f64>> {
        match self.stage {
            Stage::Done => self.hessian.as_ref(),
            _ => None,
        }
    }

    /// The unit directions along which the gradient is differentiated (columns).
    pub fn directions(&self) -> Mat<f64> {
        Mat::from_fn(self.x.nrows(), self.directions.len(), |i, k| {
            self.directions[k][i]
        })
    }

    /// Gradients taken so far (besides the one at the point itself).
    pub fn gradient_count(&self) -> usize {
        self.gradients
    }

    /// `|G - H D|` of the last fit: how well the Hessian reproduces the derivatives.
    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// Remarks for the log.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }
}
