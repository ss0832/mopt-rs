//! Saddle points of index `k >= 1` from one point and `k` directions, by high-index saddle
//! dynamics.
//!
//! The point follows the gradient reflected across the `k` tracked directions `V`,
//! `dx/dt = -(I - 2 V V^T) g` (HiSD: J. Yin, L. Zhang, P. Zhang, SIAM J. Sci. Comput. 2019,
//! 41, A3576; for `k = 1` the gentlest ascent dynamics of W. E and X. Zhou, Nonlinearity
//! 2011, 24, 1831): it climbs along the directions and descends along the rest. The
//! directions follow the `k` lowest curvatures of the Hessian at the point, from products of
//! the Hessian with them ([`DirectionUpdate`]; differences of gradients, the shrinking dimer
//! of HiOSD, or exact Hessians). A linearly stable fixed point of the coupled dynamics is a
//! saddle point of index `k` with `V` spanning its unstable subspace (the theorem of Yin,
//! Zhang, Zhang); away from it nothing holds the index of the Hessian at the point at `k`,
//! so the index of what is found is checked by the Hessian afterwards.
//!
//! One iteration evaluates the gradient at the point and the products along the
//! directions in one batch, updates the directions there, tests the ends, and takes a
//! Barzilai-Borwein step along the reflected gradient (the steps of the optimization-based
//! shrinking dimer, HiOSD): `beta = s^T y / y^T y` (`s`, `y`: the changes of the point and
//! of the reflected gradient), the first step and every step with `s^T y <= 0` of a fixed
//! length, every step at most a given length.
//!
//! The ends ([`HisdEnd`]): the force criteria with all `k` tracked curvatures negative
//! (converged), or with one of them not (a stationary point of another index); the hand-off
//! to a tighter method (RMS gradient and change of the projector `V V^T` below thresholds,
//! all `k` curvatures negative); the region of index `k` left for good; the iteration limit.
//!
//! The search works in the metric of [`SearchFrame`] (mass-weighted Cartesians for
//! structures by choice), with the rigid-body motions and fixed variables left out of every
//! vector. It is an ask/tell state machine over points of `n` variables, like
//! [`crate::path::dimer::Dimer`]. A search is resumed by a new search from its last point
//! and directions (the step lengths start anew).
//!
//! The upward and downward searches of the solution landscape (J. Yin, B. Yu, L. Zhang, Sci.
//! China Math. 2021, 64, 1801; arXiv:2002.10690) start such searches from a saddle point
//! ([`landscape`]); the lowest curvatures at a point from gradients alone come from
//! [`modes::LowestModes`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod frame;
pub mod landscape;
pub mod mode_max;
pub mod modes;
pub mod settings;
pub mod subspace;

#[cfg(test)]
mod tests;

use faer::{Col, Mat};

use super::points::{
    check_values, gradient_of, hessian_of, PointRequest, PointValue, SearchAction,
};
use crate::core_infra::rng::Rng;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_max_abs, col_norm};
use crate::math::stable::logistic;
pub use frame::SearchFrame;
pub use settings::{Curvature, DirectionUpdate, HandOff, HisdSettings, PointFlow, StepSettings};
use subspace::{
    euler_directions, gram_schmidt, largest_principal_sine, orthonormal_pairs, orthonormal_to,
    rayleigh_ritz, Pair,
};

/// Why a search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HisdEnd {
    /// The force criteria are met and the `k` tracked curvatures are negative (and the
    /// guard curvature positive, with a guard).
    Converged,
    /// The hand-off thresholds are met with the `k` tracked curvatures negative.
    HandOff,
    /// The force criteria are met but not all `k` tracked curvatures are negative (or the
    /// guard curvature is not positive): a stationary point of another index.
    WrongIndex,
    /// After all `k` tracked curvatures were negative, the highest stayed non-negative for
    /// longer than [`HisdSettings::lost_after`].
    LeftIndex,
    /// The iterations are used up.
    IterationLimit,
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct HisdReport {
    /// Iteration (steps of the point so far).
    pub iteration: usize,
    /// The point that was evaluated.
    pub point: Col<f64>,
    /// Energy there.
    pub energy: f64,
    /// Largest component of the gradient there (without the motions left out).
    pub max_force: f64,
    /// RMS of the gradient there.
    pub rms_force: f64,
    /// Curvatures along the tracked directions (Ritz values, in the metric of the search;
    /// the guard last).
    pub curvatures: Vec<f64>,
    /// Sine of the largest principal angle between the reflected subspaces of this and the
    /// last update (1 before the first).
    pub projector_change: f64,
    /// Length of the difference of gradients used.
    pub dimer_length: f64,
    /// Largest displacement of an atom (or the length) of the step taken, if one was.
    pub step: Option<f64>,
    /// Gradients evaluated so far, the differences of gradients included.
    pub gradients: usize,
    /// Hessians evaluated so far.
    pub hessians: usize,
    /// Why the search ended, if it did.
    pub end: Option<HisdEnd>,
}

#[derive(Debug, Clone)]
enum Purpose {
    /// The first direction was turned along the gradient (GGAD).
    Start,
    /// Another update of the directions at the same point.
    Sweep(usize),
    /// The extra directions of a LOBPCG step; the Ritz pairs of the current directions.
    Expand(Vec<Pair>),
}

#[derive(Debug, Clone)]
enum Phase {
    /// The gradient at the point and the products along the directions.
    Point,
    /// Products along `directions` at the same point.
    Products {
        directions: Vec<Col<f64>>,
        purpose: Purpose,
    },
    Done(HisdEnd),
}

/// Ask/tell search for a saddle point of index `k`.
pub struct HighIndexSaddleSearch {
    settings: HisdSettings,
    frame: SearchFrame,
    /// The directions given (in the coordinates of the points).
    given: Vec<Col<f64>>,
    x: Col<f64>,
    /// Tracked directions (metric of the search, orthonormal).
    v: Vec<Col<f64>>,
    /// The directions before the last update and their Euler directions.
    previous: Option<(Vec<Col<f64>>, Vec<Col<f64>>)>,
    curvatures: Vec<f64>,
    energy: f64,
    /// Gradient at the point (points' coordinates, motions left out).
    gradient: Col<f64>,
    /// The same in the metric of the search.
    gradient_q: Col<f64>,
    /// Exact Hessian at the point (metric of the search, projected).
    hessian_q: Option<Mat<f64>>,
    removed: Option<Mat<f64>>,
    dimer: f64,
    /// The last point and reflected gradient (metric of the search) of the step rule.
    last_step: Option<(Col<f64>, Col<f64>)>,
    /// `logit(alpha)` of iHiSD.
    alpha_logit: f64,
    phase: Phase,
    iteration: usize,
    gradients: usize,
    hessians: usize,
    projector_change: f64,
    reached: bool,
    lost: usize,
    started: bool,
    rng: Rng,
}

impl HighIndexSaddleSearch {
    /// A search from `start` with the directions `directions` (in the coordinates of the
    /// points; at least `order` independent ones, the first `order` reflected; with a guard
    /// the next one, or a random one, is its start).
    pub fn new(
        settings: HisdSettings,
        start: &Col<f64>,
        directions: &[Col<f64>],
    ) -> MoptResult<Self> {
        settings.validate()?;
        let n = start.nrows();
        if n == 0 || directions.iter().any(|d| d.nrows() != n) {
            return Err(MoptError::config(
                "high-index saddle search: the start and the directions differ in size",
            ));
        }
        if settings.tracked() > n {
            return Err(MoptError::config(format!(
                "high-index saddle search: {} directions for {n} variables",
                settings.tracked()
            )));
        }
        let alpha_logit = match settings.flow {
            PointFlow::Improved { alpha0, .. } => libm::log(alpha0 / (1.0 - alpha0)),
            PointFlow::Reflected => 0.0,
        };
        let mut s = Self {
            rng: Rng::from_seed(settings.seed),
            dimer: settings.dimer_length,
            settings,
            frame: SearchFrame::plain(),
            given: directions.to_vec(),
            x: start.clone(),
            v: Vec::new(),
            previous: None,
            curvatures: Vec::new(),
            energy: f64::NAN,
            gradient: Col::zeros(n),
            gradient_q: Col::zeros(n),
            hessian_q: None,
            removed: None,
            last_step: None,
            alpha_logit,
            phase: Phase::Point,
            iteration: 0,
            gradients: 0,
            hessians: 0,
            projector_change: 1.0,
            reached: false,
            lost: 0,
            started: false,
        };
        s.prepare()?;
        Ok(s)
    }

    /// The points are the coordinates (Bohr) of `template`: the rigid-body motions are left
    /// out with `rigid`, and the metric is mass-weighted with `mass_weighted`.
    pub fn with_structure(
        mut self,
        template: MolecularSystem,
        rigid: bool,
        mass_weighted: bool,
    ) -> MoptResult<Self> {
        let frozen = std::mem::take(&mut self.frame).frozen_variables();
        self.frame = SearchFrame::structure(template, rigid, mass_weighted).with_frozen(frozen);
        self.prepare()?;
        Ok(self)
    }

    pub(crate) fn with_frame(mut self, frame: SearchFrame) -> MoptResult<Self> {
        self.frame = frame;
        self.prepare()?;
        Ok(self)
    }

    /// Keep these variables where they are.
    pub fn with_frozen(mut self, variables: Vec<usize>) -> MoptResult<Self> {
        self.frame = std::mem::take(&mut self.frame).with_frozen(variables);
        self.prepare()?;
        Ok(self)
    }

    /// The directions of the search from the given ones, in the frame.
    fn prepare(&mut self) -> MoptResult<()> {
        let n = self.x.nrows();
        self.frame.check(n)?;
        self.removed = self.frame.removed_at(&self.x)?;
        let candidates: Vec<Col<f64>> = self
            .given
            .iter()
            .map(|d| {
                self.frame
                    .mask(self.removed.as_ref(), &self.frame.direction_to_q(d))
            })
            .collect();
        let mut v = gram_schmidt(&candidates);
        if v.len() < self.settings.order {
            return Err(MoptError::config(format!(
                "high-index saddle search: {} independent directions (without the motions \
                 left out) for index {}",
                v.len(),
                self.settings.order
            )));
        }
        v.truncate(self.settings.tracked());
        while v.len() < self.settings.tracked() {
            let fill = self.random_direction(&v)?;
            v.push(fill);
        }
        self.v = v;
        Ok(())
    }

    /// A random direction orthogonal to `basis` without the motions left out.
    fn random_direction(&mut self, basis: &[Col<f64>]) -> MoptResult<Col<f64>> {
        let n = self.x.nrows();
        for _ in 0..20 {
            let r = Col::from_fn(n, |_| self.rng.standard_normal());
            let r = self.frame.mask(self.removed.as_ref(), &r);
            if let Some(d) = orthonormal_to(basis, &r) {
                return Ok(d);
            }
        }
        Err(MoptError::config(
            "high-index saddle search: no direction is left to track",
        ))
    }

    /// The settings.
    pub fn settings(&self) -> &HisdSettings {
        &self.settings
    }

    /// The point (after the end: the last one evaluated).
    pub fn point(&self) -> &Col<f64> {
        &self.x
    }

    /// Energy at the last point evaluated.
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// Gradient at the last point evaluated (without the motions left out).
    pub fn gradient(&self) -> &Col<f64> {
        &self.gradient
    }

    /// The tracked directions in the metric of the search (orthonormal; the `k` reflected
    /// ones first, then the guard).
    pub fn directions(&self) -> &[Col<f64>] {
        &self.v
    }

    /// The tracked directions in the coordinates of the points, each of unit length (for
    /// the modes an optimizer is to follow; not orthogonal when the metric is
    /// mass-weighted).
    pub fn point_directions(&self) -> Vec<Col<f64>> {
        self.v
            .iter()
            .map(|v| {
                let d = self.frame.direction_to_x(v);
                let length = col_norm(&d);
                if length > 0.0 {
                    d * (1.0 / length)
                } else {
                    d
                }
            })
            .collect()
    }

    /// Curvatures along the tracked directions (in the metric of the search).
    pub fn curvatures(&self) -> &[f64] {
        &self.curvatures
    }

    /// Steps of the point so far.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    /// Gradients evaluated so far (the differences of gradients included).
    pub fn gradient_evaluations(&self) -> usize {
        self.gradients
    }

    /// Hessians evaluated so far.
    pub fn hessian_evaluations(&self) -> usize {
        self.hessians
    }

    /// The frame of the search.
    pub fn frame(&self) -> &SearchFrame {
        &self.frame
    }

    /// Why the search ended, once it has.
    pub fn end(&self) -> Option<HisdEnd> {
        match self.phase {
            Phase::Done(end) => Some(end),
            _ => None,
        }
    }

    /// The displacement of the difference of gradients along `v` (metric of the search) and
    /// the factor `c` with `x + c M^(-1/2) v` the displaced point.
    fn displacement(&self, v: &Col<f64>) -> (Col<f64>, f64) {
        let d = self.frame.direction_to_x(v);
        let size = self.frame.largest_displacement(&d).max(1e-300);
        let c = self.dimer / size;
        (d * c, c)
    }

    /// The points of the differences of gradients along `directions`.
    fn probes(&self, directions: &[Col<f64>]) -> Vec<PointRequest> {
        let mut out = Vec::new();
        for v in directions {
            let (d, _) = self.displacement(v);
            out.push(PointRequest::gradient(&self.x + &d));
            if self.settings.curvature == Curvature::Central {
                out.push(PointRequest::gradient(&self.x - &d));
            }
        }
        out
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        match &self.phase {
            Phase::Done(_) => SearchAction::Finished,
            Phase::Point => {
                if self.settings.curvature == Curvature::Hessian {
                    return SearchAction::Evaluate(vec![PointRequest::hessian(self.x.clone())]);
                }
                let mut points = vec![PointRequest::gradient(self.x.clone())];
                if !(self.settings.first_along_gradient && !self.started) {
                    points.extend(self.probes(&self.v));
                }
                SearchAction::Evaluate(points)
            }
            Phase::Products { directions, .. } => SearchAction::Evaluate(self.probes(directions)),
        }
    }

    /// The products of the Hessian along `directions` from the values of their probes.
    fn products(
        &self,
        directions: &[Col<f64>],
        values: &[PointValue],
    ) -> MoptResult<Vec<Col<f64>>> {
        if let Some(h) = &self.hessian_q {
            return Ok(directions.iter().map(|v| h * v).collect());
        }
        let per = if self.settings.curvature == Curvature::Central {
            2
        } else {
            1
        };
        let mut out = Vec::with_capacity(directions.len());
        for (j, v) in directions.iter().enumerate() {
            let (_, c) = self.displacement(v);
            let plus = self.frame.mask(
                self.removed.as_ref(),
                &self
                    .frame
                    .gradient_to_q(&gradient_of("high-index saddle search", &values[per * j])?),
            );
            // The projection is linear, so the difference of the projected gradients is
            // the projected difference.
            let difference = if per == 2 {
                let minus = self.frame.mask(
                    self.removed.as_ref(),
                    &self.frame.gradient_to_q(&gradient_of(
                        "high-index saddle search",
                        &values[per * j + 1],
                    )?),
                );
                (plus - minus) * (0.5 / c)
            } else {
                (plus - &self.gradient_q) * (1.0 / c)
            };
            out.push(difference);
        }
        Ok(out)
    }

    /// The directions that were probed, masked at the point, with their products
    /// (orthonormalized together); directions lost on the way are filled with random ones
    /// of zero product, which the next update corrects.
    fn masked_pairs(
        &mut self,
        asked: &[Col<f64>],
        products: Vec<Col<f64>>,
    ) -> MoptResult<Vec<Col<f64>>> {
        let pairs: Vec<Pair> = asked
            .iter()
            .map(|v| self.frame.mask(self.removed.as_ref(), v))
            .zip(products)
            .collect();
        let pairs = orthonormal_pairs(&pairs);
        let (mut v, mut u): (Vec<Col<f64>>, Vec<Col<f64>>) = pairs.into_iter().unzip();
        let n = self.x.nrows();
        while v.len() < self.settings.tracked() {
            let d = self.random_direction(&v)?;
            u.push(match &self.hessian_q {
                Some(h) => h * &d,
                None => Col::zeros(n),
            });
            v.push(d);
        }
        self.v = v;
        Ok(u)
    }

    /// The values of the points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<HisdReport> {
        let asked = match self.ask() {
            SearchAction::Evaluate(points) => points,
            SearchAction::Finished => {
                return Err(MoptError::config(
                    "tell() called after the high-index saddle search ended",
                ))
            }
        };
        check_values("high-index saddle search", &asked, &values, self.x.nrows())?;
        self.gradients += values.iter().filter(|v| v.gradient.is_some()).count();
        match std::mem::replace(&mut self.phase, Phase::Point) {
            Phase::Point => {
                self.removed = self.frame.removed_at(&self.x)?;
                self.energy = values[0].energy;
                let g = gradient_of("high-index saddle search", &values[0])?;
                self.gradient_q = self
                    .frame
                    .mask(self.removed.as_ref(), &self.frame.gradient_to_q(&g));
                self.gradient = self.frame.gradient_to_x(&self.gradient_q);
                self.hessian_q = if self.settings.curvature == Curvature::Hessian {
                    self.hessians += 1;
                    let h = hessian_of("high-index saddle search", &values[0])?;
                    Some(self.frame.hessian_to_q(self.removed.as_ref(), &h))
                } else {
                    None
                };
                if self.settings.first_along_gradient && !self.started {
                    self.started = true;
                    // The rotations move with the point: the directions are masked anew.
                    let mut v: Vec<Col<f64>> =
                        orthonormal_to(&[], &self.gradient_q).into_iter().collect();
                    v.extend(
                        self.v
                            .iter()
                            .map(|d| self.frame.mask(self.removed.as_ref(), d)),
                    );
                    self.v = self.refilled(gram_schmidt(&v))?;
                    if self.hessian_q.is_none() {
                        self.phase = Phase::Products {
                            directions: self.v.clone(),
                            purpose: Purpose::Start,
                        };
                        return Ok(self.report(None, None));
                    }
                }
                let asked_v = self.v.clone();
                let products = self.products(&asked_v, &values[1..])?;
                let products = self.masked_pairs(&asked_v, products)?;
                self.after_products(products, 0)
            }
            Phase::Products {
                directions,
                purpose,
            } => {
                let products = self.products(&directions, &values)?;
                match purpose {
                    Purpose::Start | Purpose::Sweep(_) => {
                        let done = match purpose {
                            Purpose::Sweep(done) => done,
                            _ => 0,
                        };
                        let products = self.masked_pairs(&directions, products)?;
                        self.after_products(products, done)
                    }
                    Purpose::Expand(pairs) => {
                        let mut all = pairs;
                        all.extend(directions.into_iter().zip(products));
                        let ritz = rayleigh_ritz(&orthonormal_pairs(&all))?;
                        let m = self.settings.tracked().min(ritz.vectors.len());
                        let v: Vec<Col<f64>> = ritz.vectors[..m].to_vec();
                        let v = self.refilled(v)?;
                        let curvatures = ritz.values[..m].to_vec();
                        self.finish_update(v, curvatures, None, 0)
                    }
                }
            }
            Phase::Done(end) => {
                self.phase = Phase::Done(end);
                Err(MoptError::config(
                    "tell() called after the high-index saddle search ended",
                ))
            }
        }
    }

    /// `v` with random directions added up to the number tracked.
    fn refilled(&mut self, mut v: Vec<Col<f64>>) -> MoptResult<Vec<Col<f64>>> {
        v.truncate(self.settings.tracked());
        while v.len() < self.settings.tracked() {
            let d = self.random_direction(&v)?;
            v.push(d);
        }
        Ok(v)
    }

    /// Update the directions from their products `u = H v`; `done` updates were made at
    /// this point before.
    fn after_products(&mut self, u: Vec<Col<f64>>, done: usize) -> MoptResult<HisdReport> {
        let update = self.settings.update;
        match update {
            DirectionUpdate::Eigenvectors => {
                let h = self.hessian_q.clone().ok_or_else(|| {
                    MoptError::config("the eigenvector update needs exact Hessians")
                })?;
                let (values, vectors) = self.frame.eigen(self.removed.as_ref(), &h)?;
                let m = self.settings.tracked().min(vectors.len());
                let v = self.refilled(vectors[..m].to_vec())?;
                self.finish_update(v, values[..m].to_vec(), None, done)
            }
            DirectionUpdate::Lobpcg => {
                // The Ritz pairs inside the span of the directions, their residuals, and the
                // directions before the last update span the next Rayleigh-Ritz problem.
                let pairs: Vec<Pair> = self.v.iter().cloned().zip(u).collect();
                let ritz = rayleigh_ritz(&orthonormal_pairs(&pairs))?;
                let inner: Vec<Pair> = ritz
                    .vectors
                    .iter()
                    .cloned()
                    .zip(ritz.products.iter().cloned())
                    .collect();
                let scale = ritz.values.iter().fold(1e-12_f64, |m, v| m.max(v.abs()));
                let mut extra: Vec<Col<f64>> = Vec::new();
                for ((v, hv), theta) in inner.iter().zip(&ritz.values) {
                    let r = self.frame.mask(self.removed.as_ref(), &(hv - v * *theta));
                    if col_norm(&r) > 1e-10 * scale {
                        extra.push(r);
                    }
                }
                if let Some((before, _)) = &self.previous {
                    for b in before {
                        extra.push(self.frame.mask(self.removed.as_ref(), b));
                    }
                }
                let basis: Vec<Col<f64>> = inner.iter().map(|p| p.0.clone()).collect();
                let mut extra_basis = basis.clone();
                let mut directions = Vec::new();
                for e in &extra {
                    if let Some(d) = orthonormal_to(&extra_basis, e) {
                        extra_basis.push(d.clone());
                        directions.push(d);
                    }
                }
                if directions.is_empty() || self.hessian_q.is_some() {
                    // With an exact Hessian the products of the extra directions are known.
                    if let Some(h) = &self.hessian_q {
                        let mut all = inner;
                        all.extend(directions.iter().map(|d| (d.clone(), h * d)));
                        let ritz = rayleigh_ritz(&orthonormal_pairs(&all))?;
                        let m = self.settings.tracked().min(ritz.vectors.len());
                        let v = self.refilled(ritz.vectors[..m].to_vec())?;
                        return self.finish_update(v, ritz.values[..m].to_vec(), None, done);
                    }
                    let m = self.settings.tracked().min(basis.len());
                    let v = self.refilled(basis[..m].to_vec())?;
                    return self.finish_update(v, ritz.values[..m].to_vec(), None, done);
                }
                self.phase = Phase::Products {
                    directions,
                    purpose: Purpose::Expand(inner),
                };
                Ok(self.report(None, None))
            }
            DirectionUpdate::HisdEuler | DirectionUpdate::OjaQr | DirectionUpdate::Ggad => {
                // The Ritz values of the tracked block decide the index (the Rayleigh
                // quotients of single directions can all be negative where only one
                // curvature is).
                let pairs: Vec<Pair> = self.v.iter().cloned().zip(u.iter().cloned()).collect();
                let ritz = rayleigh_ritz(&pairs)?;
                let d = euler_directions(update, &self.v, &u);
                let gammas = self.rotation_steps(&d, &ritz.values);
                let moved: Vec<Col<f64>> = self
                    .v
                    .iter()
                    .zip(&d)
                    .zip(&gammas)
                    .map(|((v, d), g)| self.frame.mask(self.removed.as_ref(), &(v + d * *g)))
                    .collect();
                let mut v = gram_schmidt(&moved);
                if update == DirectionUpdate::OjaQr && v.len() == self.v.len() {
                    // The subspace has no order of its own: its basis is turned as the Ritz
                    // vectors of the block before, so that the reflected directions are the
                    // lowest ones and the guard the highest.
                    v = self.ordered_like(&v, &pairs);
                }
                let v = self.refilled(v)?;
                self.finish_update(v, ritz.values, Some(d), done)
            }
        }
    }

    /// The basis `v` turned by the eigenvectors of the projected Hessian of `pairs` (the
    /// block before the update, with its products), so that its columns go from the lowest
    /// curvature to the highest.
    fn ordered_like(&self, v: &[Col<f64>], pairs: &[Pair]) -> Vec<Col<f64>> {
        let m = pairs.len();
        let a = Mat::<f64>::from_fn(m, m, |i, j| {
            0.5 * (col_dot(&pairs[i].0, &pairs[j].1) + col_dot(&pairs[j].0, &pairs[i].1))
        });
        match crate::math::linalg::symmetric_eigen(&a) {
            Ok((_, y)) => (0..m)
                .map(|j| {
                    let mut out = Col::<f64>::zeros(self.x.nrows());
                    for (i, vi) in v.iter().enumerate() {
                        out += vi * y[(i, j)];
                    }
                    out
                })
                .collect(),
            Err(_) => v.to_vec(),
        }
    }

    /// The steps `gamma_i` of the Euler updates along `d`: Barzilai-Borwein steps from the
    /// last update (one for the whole subspace with the Oja flow), the first a fraction of
    /// the inverse of the largest curvature magnitude, each change at most
    /// [`HisdSettings::max_rotation`] long.
    fn rotation_steps(&self, d: &[Col<f64>], curvatures: &[f64]) -> Vec<f64> {
        let scale = curvatures.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        let first = self.settings.rotation_rate / scale.max(1e-12);
        let mut gammas: Vec<f64> = match &self.previous {
            Some((before, d_before))
                if before.len() == self.v.len() && d_before.len() == d.len() =>
            {
                let pairs: Vec<(f64, f64)> = (0..d.len())
                    .map(|i| {
                        let dv = &self.v[i] - &before[i];
                        let dd = &d[i] - &d_before[i];
                        (col_dot(&dv, &dd).abs(), col_dot(&dd, &dd))
                    })
                    .collect();
                if self.settings.update == DirectionUpdate::OjaQr {
                    let (a, b) = pairs
                        .iter()
                        .fold((0.0, 0.0), |(x, y), (p, q)| (x + p, y + q));
                    vec![crate::path::barzilai_borwein(a, b, first); d.len()]
                } else {
                    pairs
                        .iter()
                        .map(|&(a, b)| crate::path::barzilai_borwein(a, b, first))
                        .collect()
                }
            }
            _ => vec![first; d.len()],
        };
        for (g, di) in gammas.iter_mut().zip(d) {
            let length = col_norm(di);
            if length * *g > self.settings.max_rotation {
                *g = self.settings.max_rotation / length;
            }
        }
        if self.settings.update == DirectionUpdate::OjaQr {
            let g = gammas.iter().fold(f64::INFINITY, |m, g| m.min(*g));
            gammas = vec![g; d.len()];
        }
        gammas
    }

    /// Take the updated directions `v` with the curvatures that go with the point, test the
    /// ends, and step.
    fn finish_update(
        &mut self,
        v: Vec<Col<f64>>,
        curvatures: Vec<f64>,
        d: Option<Vec<Col<f64>>>,
        done: usize,
    ) -> MoptResult<HisdReport> {
        let k = self.settings.order;
        self.projector_change = largest_principal_sine(&self.v[..k], &v[..k]);
        let before = std::mem::replace(&mut self.v, v);
        self.previous = Some((before, d.unwrap_or_default()));
        self.curvatures = curvatures;
        if done + 1 < self.settings.sweeps && self.hessian_q.is_none() {
            self.phase = Phase::Products {
                directions: self.v.clone(),
                purpose: Purpose::Sweep(done + 1),
            };
            return Ok(self.report(None, None));
        }
        self.decide_and_step()
    }

    /// Whether the tracked curvatures say index `k`: the reflected ones negative, the guard
    /// (if any) positive.
    fn index_k(&self) -> bool {
        let k = self.settings.order;
        let negative = self.curvatures.len() >= k && self.curvatures[..k].iter().all(|c| *c < 0.0);
        let guard = !self.settings.guard || self.curvatures.get(k).is_some_and(|c| *c > 0.0);
        negative && guard
    }

    fn decide_and_step(&mut self) -> MoptResult<HisdReport> {
        let index_k = self.index_k();
        if index_k {
            self.reached = true;
            self.lost = 0;
        } else if self.reached {
            self.lost += 1;
        }
        let c = &self.settings.criteria;
        let (max, rms) = self.force_norms();
        if max < c.max_force.value() && rms < c.rms_force.value() {
            let end = if index_k {
                HisdEnd::Converged
            } else {
                HisdEnd::WrongIndex
            };
            self.phase = Phase::Done(end);
            return Ok(self.report(None, None));
        }
        if let Some(h) = self.settings.hand_off {
            if index_k && rms < h.rms_gradient && self.projector_change < h.projector_change {
                self.phase = Phase::Done(HisdEnd::HandOff);
                return Ok(self.report(None, None));
            }
        }
        if self.lost > self.settings.lost_after {
            self.phase = Phase::Done(HisdEnd::LeftIndex);
            return Ok(self.report(None, None));
        }
        if self.iteration + 1 >= self.settings.max_iterations {
            self.phase = Phase::Done(HisdEnd::IterationLimit);
            return Ok(self.report(None, None));
        }
        let evaluated = self.x.clone();
        let step = self.step();
        self.x = &self.x + &step;
        self.iteration += 1;
        if self.settings.shrink > 0.0 {
            self.dimer =
                (self.dimer * libm::exp(-self.settings.shrink)).max(self.settings.min_dimer_length);
        }
        if let PointFlow::Improved { rate, .. } = self.settings.flow {
            self.alpha_logit += 2.0 * rate;
        }
        self.phase = Phase::Point;
        Ok(self.report(
            Some(evaluated),
            Some(self.frame.largest_displacement(&step)),
        ))
    }

    /// Largest component and RMS of the gradient over the free variables.
    fn force_norms(&self) -> (f64, f64) {
        let n = self.x.nrows();
        let free = n.saturating_sub(self.frame.frozen_count()).max(1);
        (
            col_max_abs(&self.gradient),
            libm::sqrt(col_dot(&self.gradient, &self.gradient) / free as f64),
        )
    }

    /// The flow vector `f` (the step is `-beta f`) in the metric of the search.
    fn flow(&self) -> Col<f64> {
        let k = self.settings.order;
        let g = &self.gradient_q;
        let mut along = Col::<f64>::zeros(g.nrows());
        for v in &self.v[..k] {
            along += v * col_dot(v, g);
        }
        match self.settings.flow {
            PointFlow::Reflected => g - along * 2.0,
            PointFlow::Improved { upward, .. } => {
                let alpha = logistic(self.alpha_logit);
                let s = if upward { -1.0 } else { 1.0 };
                g * (alpha + (1.0 - alpha) * s) - along * (2.0 * alpha)
            }
        }
    }

    /// The step of the point (coordinates of the points).
    fn step(&mut self) -> Col<f64> {
        let f = self.flow();
        let q = self.frame.direction_to_q(&self.x);
        let first = {
            let size = self
                .frame
                .largest_displacement(&self.frame.direction_to_x(&f))
                .max(1e-300);
            self.settings.step.first / size
        };
        let beta = match &self.last_step {
            Some((q_before, f_before)) => {
                let s = &q - q_before;
                let y = &f - f_before;
                crate::path::barzilai_borwein(col_dot(&s, &y), col_dot(&y, &y), first)
            }
            None => first,
        };
        self.last_step = Some((q, f.clone()));
        let mut dx = self.frame.direction_to_x(&(f * -beta));
        let size = self.frame.largest_displacement(&dx);
        if size > self.settings.step.max_step {
            dx *= self.settings.step.max_step / size;
        }
        dx
    }

    fn report(&self, evaluated: Option<Col<f64>>, step: Option<f64>) -> HisdReport {
        let (max, rms) = self.force_norms();
        HisdReport {
            iteration: self.iteration,
            point: evaluated.unwrap_or_else(|| self.x.clone()),
            energy: self.energy,
            max_force: max,
            rms_force: rms,
            curvatures: self.curvatures.clone(),
            projector_change: self.projector_change,
            dimer_length: self.dimer,
            step,
            gradients: self.gradients,
            hessians: self.hessians,
            end: self.end(),
        }
    }
}
