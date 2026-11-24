//! A band on the ridge between two transition states whose highest image climbs to the
//! saddle point of index 2 between them (the moving-frame form, after J. B. Maronsson,
//! H. Jonsson, T. Vegge, Phys. Chem. Chem. Phys. 2012, 14, 2884; the same walk along a
//! ridge one index lower is the ridge method of I. V. Ionova and E. A. Carter, J. Chem.
//! Phys. 1993, 98, 6377).
//!
//! The images `x_0 = T1, x_1, ..., x_N, x_(N+1) = T2` join two transition states; the ends
//! stay where they are. At every inner image `i`:
//!
//! - the tangent `tau_i` towards the higher neighbour (the upwind tangent of the band,
//!   [`super::force`]);
//! - the transverse mode `v_i`: the lowest eigenvector of `P_i Q H Q P_i` (`P_i = I -
//!   tau_i tau_i^T`, `Q` without the rigid-body motions), the minimum mode of the Hessian
//!   reduced to the directions across the band, as in the paper (where a dimer rotated
//!   with `e . tau = 0` finds it); or, by choice ([`TransverseMode::Continued`]), of its two
//!   lowest eigenvectors the one that continues the mode of the last iteration. The two
//!   lowest are tracked by an Oja step of their subspace per iteration from two products
//!   of the Hessian (differences of gradients or an exact Hessian), started from the
//!   imaginary modes of the two ends interpolated along the band (the paper starts from
//!   random modes);
//! - the ridge condition: a maximum along `v_i` (`g . v_i = 0` with negative curvature)
//!   and a minimum across the band in the other directions. The effective gradient of an
//!   inner image is `(I - tau tau^T - 2 v v^T) g - F_s` (eqs. 8-11 of the paper), with the
//!   spring force `F_s = k (x_(i+1) - 2 x_i + x_(i-1))` in full (the paper keeps its part
//!   across the band for stability, which cuts corners of a curved ridge,
//!   [`RidgeSpring::Full`]) or only along the tangent, `k (|x_(i+1) - x_i| - |x_i -
//!   x_(i-1)|) tau`, as in a nudged elastic band ([`RidgeSpring::Tangential`]);
//! - the climbing image maximizes along the band as well: `(I - 2 v v^T - 2 t t^T) g` with
//!   `t = unit (I - v v^T) tau` (eq. 13), the reflected gradient of an index-2 search with
//!   the directions `(v, t)`, without springs. As the paper advises, it starts only once
//!   the energy has a maximum along the band: the highest inner image above its
//!   neighbours and both ends, after the band has settled (`climb_rms`) or after
//!   `climb_after` iterations.
//!
//! Why the climbing image ends at an index-2 saddle `S`: on the ridge (the maxima along the
//! transverse mode) the saddle has one negative direction across the ridge, maximized at
//! every image, and one along it; it is a first-order saddle of the energy on the ridge,
//! and the two transition states are minima there. A band with a climbing image is the
//! tool for first-order saddles: the converged band leaves `S` along its unstable direction
//! in the ridge, so `tau -> u_b`, the lowest mode across it is the other unstable one,
//! `v -> u_a`, and `(v, t)` spans the unstable plane of `S`, which is a stable fixed point
//! of the reflected gradient (the stability theorem of high-index saddle dynamics). The
//! point is not a constrained minimum and need not be: one negative direction is maximized
//! across the ridge, the other by the climbing image. Two transition states that share a
//! minimum meet at `S` at a corner of the band (their imaginary modes are about the two
//! unstable directions of `S`, turned by 90 degrees); there `v` turns from one to the other
//! along a few images, which the report lists as switches.
//!
//! When it fails, and what says so ([`RidgeBandEnd`]):
//!
//! - the transverse curvature of an image stays above `-min_curvature` for more than
//!   `lost_after` iterations (`RidgeModeLost`): no maximum across the band there. This is a
//!   numerical state, not a diagnosis: the two transition states may share no basin
//!   boundary, the ridge may fold or end (a shoulder), or the band may lie off the ridge;
//!   it is not the test of a valley-ridge inflection point. Two transition states that
//!   share a minimum are candidates for a band, not a guarantee of one ridge or of an
//!   index-2 saddle between them;
//! - the transition states lie in series (the band leaves `T1` along its imaginary mode):
//!   the ridge picture does not apply (`Series`, at the start);
//! - a second transverse negative mode (a region of index 3): the mode switches, the band
//!   may climb to an index-3 point; the index of what is found is checked by its Hessian;
//! - several maxima along the converged band: each is a candidate (the report gives the
//!   energies); intermediate minima are transition states of their own.
//!
//! The metric is that of [`SearchFrame`] (mass-weighted Cartesians for structures by
//! choice); every vector is projected without the rigid-body motions.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::fire::{ChainFire, ChainFireSettings};
use super::force::UpwindProjection;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_max_abs, col_norm};
use crate::path::hisd::frame::SearchFrame;
use crate::path::hisd::settings::{Curvature, DirectionUpdate, StepSettings};
use crate::path::hisd::subspace::{
    euler_directions, gram_schmidt, orthonormal_pairs, orthonormal_to, rayleigh_ritz, Pair,
};
use crate::path::points::{
    check_values, gradient_of, hessian_of, PointRequest, PointValue, SearchAction,
};

/// The spring force between the images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidgeSpring {
    /// `k (x_(i+1) - 2 x_i + x_(i-1))` in full (Maronsson, Jonsson, Vegge 2012, eq. 10).
    Full,
    /// Its part along the tangent only (as in the nudged elastic band).
    Tangential,
}

/// The step rule of the images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidgeStep {
    /// Barzilai-Borwein steps of each image on its effective gradient.
    BarzilaiBorwein,
    /// The step rule of the bands of this module ([`ChainFire`]: steps along the forces of
    /// the whole band with a common time step, bounded by the curvature observed).
    Fire,
}

/// Which mode across the band is maximized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransverseMode {
    /// The lowest mode of the Hessian reduced to the directions across the band (the
    /// paper).
    Lowest,
    /// Of the two lowest, the one with the larger overlap with the mode of the last
    /// iteration.
    Continued,
}

/// Settings of a [`RidgeBand`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RidgeBandSettings {
    /// The spring force.
    pub spring_force: RidgeSpring,
    /// The mode across the band that is maximized.
    pub transverse: TransverseMode,
    /// The step rule of the images.
    pub step_rule: RidgeStep,
    /// Spring constant along the band (Hartree/Bohr^2 in the metric of the search).
    pub spring: f64,
    /// The highest image starts to climb when the RMS of the effective gradients of the
    /// inner images is below this ...
    pub climb_rms: f64,
    /// ... or after this many iterations.
    pub climb_after: usize,
    /// Converged when the climbing image climbs and the largest component of its
    /// effective gradient is below this (Hartree/Bohr) ...
    pub tolerance: f64,
    /// ... and, if given, the largest component of the effective gradients of the other
    /// inner images below this. `None`: the climbing image alone decides, which is what is
    /// handed on (measured: with the full spring the other images may keep moving near a
    /// corner of the ridge while the climbing image has settled on the saddle).
    pub band_tolerance: Option<f64>,
    /// Iterations at most.
    pub max_iterations: usize,
    /// Steps of each image.
    pub step: StepSettings,
    /// Products of the Hessian: differences of gradients or exact Hessians.
    pub curvature: Curvature,
    /// Length of the differences (the largest displacement of an atom).
    pub dimer_length: f64,
    /// Below this overlap with the last iteration a transverse mode counts as switched.
    pub switch_overlap: f64,
    /// Above this overlap of the first segment of the band with the mode of `T1` the
    /// transition states lie in series.
    pub series_overlap: f64,
    /// A transverse curvature above `-min_curvature` is no ridge ...
    pub min_curvature: f64,
    /// ... for more than this many iterations in a row.
    pub lost_after: usize,
    /// Step of the Oja updates of the tracked subspaces, as a multiple of the inverse of
    /// the largest curvature magnitude in them.
    pub rotation_rate: f64,
    /// Largest change of a tracked direction in one update.
    pub max_rotation: f64,
}

impl Default for RidgeBandSettings {
    fn default() -> Self {
        Self {
            spring_force: RidgeSpring::Full,
            transverse: TransverseMode::Lowest,
            step_rule: RidgeStep::Fire,
            spring: 0.1,
            climb_rms: 3e-3,
            climb_after: 50,
            tolerance: 3e-4,
            band_tolerance: None,
            max_iterations: 500,
            step: StepSettings {
                first: 0.02,
                max_step: 0.1,
            },
            curvature: Curvature::Forward,
            dimer_length: 1e-3,
            switch_overlap: 0.5,
            series_overlap: 0.9,
            min_curvature: 0.0,
            lost_after: 20,
            rotation_rate: 0.5,
            max_rotation: 0.5,
        }
    }
}

/// Why a band ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidgeBandEnd {
    /// The climbing image climbs and the effective gradients are below the tolerance.
    Converged,
    /// The iterations are used up.
    IterationLimit,
    /// The transverse curvature of this image stayed above `-min_curvature`: no maximum
    /// across the band there (a numerical state; see the module documentation).
    RidgeModeLost(usize),
    /// The band leaves the first transition state along its imaginary mode.
    Series,
}

/// What one iteration did.
#[derive(Debug, Clone)]
pub struct RidgeBandReport {
    /// Iteration (from zero).
    pub iteration: usize,
    /// Energies of the images.
    pub energies: Vec<f64>,
    /// Transverse curvatures of the inner images (ends: the curvatures of their modes).
    pub curvatures: Vec<f64>,
    /// Largest component of the effective gradient of the inner images.
    pub largest_force: f64,
    /// Largest component of the effective gradient of the climbing image, once one climbs.
    pub climbing_force: Option<f64>,
    /// The climbing image, once one climbs.
    pub climbing: Option<usize>,
    /// Inner images whose transverse mode switched in this iteration.
    pub switches: Vec<usize>,
    /// Gradients evaluated so far (differences included).
    pub gradients: usize,
    /// Hessians evaluated so far.
    pub hessians: usize,
    /// Why the band ended, if it did.
    pub end: Option<RidgeBandEnd>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Energies and gradients of all images (the start).
    Start,
    /// Gradients of the inner images and the products of their subspaces.
    Iterate,
    Done(RidgeBandEnd),
}

/// Ask/tell band on the ridge between two transition states.
pub struct RidgeBand {
    settings: RidgeBandSettings,
    frame: SearchFrame,
    chain: Vec<Col<f64>>,
    /// The imaginary modes of the ends as given (coordinates of the points).
    end_modes: [Col<f64>; 2],
    /// Transverse modes (metric of the search); the ends keep their imaginary modes.
    modes: Vec<Col<f64>>,
    /// The two tracked directions of each image, orthonormal and across the band.
    subspaces: Vec<Vec<Col<f64>>>,
    energies: Vec<f64>,
    gradients_q: Vec<Col<f64>>,
    tangents: Vec<Col<f64>>,
    curvatures: Vec<f64>,
    /// The last point and effective gradient of each image (steps).
    last: Vec<Option<(Col<f64>, Col<f64>)>>,
    /// The step rule of the whole band, its forces and steps of the last iteration.
    fire: ChainFire,
    fire_last: Option<(Vec<Col<f64>>, Vec<Col<f64>>)>,
    lost: Vec<usize>,
    climbing: Option<usize>,
    climbing_force: Option<f64>,
    phase: Phase,
    iteration: usize,
    gradients: usize,
    hessians: usize,
}

impl RidgeBand {
    /// A band along `chain` (the first and last images are the transition states, at
    /// least one image between them) with the imaginary modes of the two ends.
    pub fn new(
        settings: RidgeBandSettings,
        chain: Vec<Col<f64>>,
        end_modes: [Col<f64>; 2],
    ) -> MoptResult<Self> {
        if chain.len() < 3 {
            return Err(MoptError::config(
                "ridge band: two transition states and at least one image between them",
            ));
        }
        let n = chain[0].nrows();
        if chain.iter().any(|x| x.nrows() != n) || end_modes.iter().any(|m| m.nrows() != n) {
            return Err(MoptError::config(
                "ridge band: the images and modes differ in size",
            ));
        }
        let positive = [
            settings.dimer_length,
            settings.step.first,
            settings.step.max_step,
            settings.tolerance,
            settings.rotation_rate,
            settings.max_rotation,
        ];
        if positive.iter().any(|v| !(v.is_finite() && *v > 0.0)) || settings.max_iterations == 0 {
            return Err(MoptError::config(
                "ridge band: lengths, steps, the tolerance and the iterations must be positive",
            ));
        }
        let images = chain.len();
        let mut band = Self {
            settings,
            frame: SearchFrame::plain(),
            end_modes,
            modes: vec![Col::zeros(n); images],
            subspaces: vec![Vec::new(); images],
            energies: vec![f64::NAN; images],
            gradients_q: vec![Col::zeros(n); images],
            tangents: vec![Col::zeros(n); images],
            curvatures: vec![f64::NAN; images],
            last: vec![None; images],
            fire: ChainFire::new(ChainFireSettings::default()),
            fire_last: None,
            lost: vec![0; images],
            climbing: None,
            climbing_force: None,
            phase: Phase::Start,
            iteration: 0,
            gradients: 0,
            hessians: 0,
            chain,
        };
        band.prepare_modes()?;
        Ok(band)
    }

    /// As [`crate::path::hisd::HighIndexSaddleSearch::with_structure`].
    pub fn with_structure(
        mut self,
        template: MolecularSystem,
        rigid: bool,
        mass_weighted: bool,
    ) -> MoptResult<Self> {
        self.frame = SearchFrame::structure(template, rigid, mass_weighted);
        self.frame.check(self.chain[0].nrows())?;
        self.prepare_modes()?;
        Ok(self)
    }

    /// The modes of the ends in the metric of the search.
    fn end_modes_q(&self) -> MoptResult<[Col<f64>; 2]> {
        let last = self.chain.len() - 1;
        let unit = |m: &Col<f64>, x: &Col<f64>| -> MoptResult<Col<f64>> {
            let removed = self.frame.removed(x);
            let q = self
                .frame
                .mask(removed.as_ref(), &self.frame.direction_to_q(m));
            orthonormal_to(&[], &q).ok_or_else(|| {
                MoptError::config("ridge band: a mode of an end is a rigid-body motion only")
            })
        };
        let first = unit(&self.end_modes[0], &self.chain[0])?;
        let mut second = unit(&self.end_modes[1], &self.chain[last])?;
        if col_dot(&first, &second) < 0.0 {
            second = -second;
        }
        Ok([first, second])
    }

    /// The modes of the inner images interpolated between those of the ends.
    fn prepare_modes(&mut self) -> MoptResult<()> {
        let [a, b] = self.end_modes_q()?;
        let images = self.chain.len();
        let last = images - 1;
        self.modes[0] = a.clone();
        self.modes[last] = b.clone();
        for i in 1..last {
            let s = i as f64 / last as f64;
            let removed = self.frame.removed(&self.chain[i]);
            let v = self
                .frame
                .mask(removed.as_ref(), &(&a * (1.0 - s) + &b * s));
            let w = self.frame.mask(removed.as_ref(), &(&a - &b));
            let mut basis = gram_schmidt(&[v, w]);
            if basis.is_empty() {
                return Err(MoptError::config(
                    "ridge band: the modes of the ends cancel along the band",
                ));
            }
            if basis.len() < 2 {
                // A second direction from the coordinates themselves.
                let n = self.chain[i].nrows();
                for k in 0..n {
                    let e = self.frame.mask(
                        removed.as_ref(),
                        &Col::from_fn(n, |j| if j == k { 1.0 } else { 0.0 }),
                    );
                    if let Some(d) = orthonormal_to(&basis, &e) {
                        basis.push(d);
                        break;
                    }
                }
            }
            self.modes[i] = basis[0].clone();
            self.subspaces[i] = basis;
        }
        Ok(())
    }

    /// The images.
    pub fn chain(&self) -> &[Col<f64>] {
        &self.chain
    }

    /// Energies of the images at the last evaluation.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// The climbing image, once one climbs.
    pub fn climbing(&self) -> Option<usize> {
        self.climbing
    }

    /// The highest inner image (the climbing one once it climbs).
    pub fn highest(&self) -> usize {
        let last = self.chain.len() - 1;
        self.climbing.unwrap_or_else(|| {
            (1..last)
                .max_by(|a, b| self.energies[*a].total_cmp(&self.energies[*b]))
                .unwrap_or(1)
        })
    }

    /// The two directions of the highest image for a search of index 2 (the transverse
    /// mode and the tangent made orthogonal to it), in the coordinates of the points, each
    /// of unit length.
    pub fn saddle_directions(&self) -> Vec<Col<f64>> {
        let c = self.highest();
        let v = &self.modes[c];
        let t = &self.tangents[c] - v * col_dot(v, &self.tangents[c]);
        [v.clone(), t]
            .iter()
            .filter_map(|d| {
                let x = self.frame.direction_to_x(d);
                orthonormal_to(&[], &x)
            })
            .collect()
    }

    /// Why the band ended, once it has.
    pub fn end(&self) -> Option<RidgeBandEnd> {
        match self.phase {
            Phase::Done(end) => Some(end),
            _ => None,
        }
    }

    fn displacement(&self, v: &Col<f64>) -> (Col<f64>, f64) {
        let d = self.frame.direction_to_x(v);
        let size = self.frame.largest_displacement(&d).max(1e-300);
        let c = self.settings.dimer_length / size;
        (d * c, c)
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        let last = self.chain.len() - 1;
        match self.phase {
            Phase::Done(_) => SearchAction::Finished,
            Phase::Start => SearchAction::Evaluate(
                self.chain
                    .iter()
                    .map(|x| PointRequest::gradient(x.clone()))
                    .collect(),
            ),
            Phase::Iterate => {
                let mut out = Vec::new();
                for i in 1..last {
                    if self.settings.curvature == Curvature::Hessian {
                        out.push(PointRequest::hessian(self.chain[i].clone()));
                        continue;
                    }
                    out.push(PointRequest::gradient(self.chain[i].clone()));
                    for w in &self.subspaces[i] {
                        let (d, _) = self.displacement(w);
                        out.push(PointRequest::gradient(&self.chain[i] + &d));
                        if self.settings.curvature == Curvature::Central {
                            out.push(PointRequest::gradient(&self.chain[i] - &d));
                        }
                    }
                }
                SearchAction::Evaluate(out)
            }
        }
    }

    /// The values of the points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<RidgeBandReport> {
        let asked = match self.ask() {
            SearchAction::Evaluate(points) => points,
            SearchAction::Finished => {
                return Err(MoptError::config(
                    "tell() called after the ridge band ended",
                ))
            }
        };
        let n = self.chain[0].nrows();
        check_values("ridge band", &asked, &values, n)?;
        self.gradients += values.iter().filter(|v| v.gradient.is_some()).count();
        let last = self.chain.len() - 1;
        let q_gradient = |frame: &SearchFrame, x: &Col<f64>, value: &PointValue| {
            let removed = frame.removed(x);
            gradient_of("ridge band", value)
                .map(|g| frame.mask(removed.as_ref(), &frame.gradient_to_q(&g)))
        };
        match self.phase {
            Phase::Start => {
                for (i, value) in values.iter().enumerate() {
                    self.energies[i] = value.energy;
                    self.gradients_q[i] = q_gradient(&self.frame, &self.chain[i], value)?;
                }
                self.update_tangents();
                // The transition states in series: the band leaves T1 along its mode.
                let first = self
                    .frame
                    .direction_to_q(&(&self.chain[1] - &self.chain[0]));
                let length = col_norm(&first);
                if length > 0.0
                    && col_dot(&first, &self.modes[0]).abs() / length > self.settings.series_overlap
                {
                    self.phase = Phase::Done(RidgeBandEnd::Series);
                    return Ok(self.report(Vec::new(), f64::NAN));
                }
                for i in 1..last {
                    self.across(i);
                }
                self.phase = Phase::Iterate;
                Ok(self.report(Vec::new(), f64::NAN))
            }
            Phase::Iterate => {
                let per = match self.settings.curvature {
                    Curvature::Hessian => 1,
                    Curvature::Forward => 3,
                    Curvature::Central => 5,
                };
                let mut products: Vec<Vec<Col<f64>>> = vec![Vec::new(); self.chain.len()];
                for (i, image_products) in products.iter_mut().enumerate().take(last).skip(1) {
                    let base = per * (i - 1);
                    let x = self.chain[i].clone();
                    let removed = self.frame.removed(&x);
                    self.energies[i] = values[base].energy;
                    self.gradients_q[i] = q_gradient(&self.frame, &x, &values[base])?;
                    if self.settings.curvature == Curvature::Hessian {
                        self.hessians += 1;
                        let h = self.frame.hessian_to_q(
                            removed.as_ref(),
                            &hessian_of("ridge band", &values[base])?,
                        );
                        *image_products = self.subspaces[i].iter().map(|w| &h * w).collect();
                        continue;
                    }
                    let step = if self.settings.curvature == Curvature::Central {
                        2
                    } else {
                        1
                    };
                    for (j, w) in self.subspaces[i].iter().enumerate() {
                        let (_, c) = self.displacement(w);
                        let plus = q_gradient(&self.frame, &x, &values[base + 1 + step * j])?;
                        let hw = if step == 2 {
                            let minus = q_gradient(&self.frame, &x, &values[base + 2 + step * j])?;
                            (plus - minus) * (0.5 / c)
                        } else {
                            (plus - &self.gradients_q[i]) * (1.0 / c)
                        };
                        image_products.push(hw);
                    }
                }
                self.update_tangents();
                let mut switches = Vec::new();
                for (i, image_products) in products.iter().enumerate().take(last).skip(1) {
                    if self.track(i, image_products)? {
                        switches.push(i);
                    }
                }
                self.decide_and_step(switches)
            }
            Phase::Done(_) => Err(MoptError::config(
                "tell() called after the ridge band ended",
            )),
        }
    }

    /// Tangents of the inner images (metric of the search, unit).
    fn update_tangents(&mut self) {
        let last = self.chain.len() - 1;
        let q: Vec<Col<f64>> = self
            .chain
            .iter()
            .map(|x| self.frame.direction_to_q(x))
            .collect();
        for i in 1..last {
            let removed = self.frame.removed(&self.chain[i]);
            let d = UpwindProjection::direction(
                [&q[i - 1], &q[i], &q[i + 1]],
                [self.energies[i - 1], self.energies[i], self.energies[i + 1]],
            );
            let d = self.frame.mask(removed.as_ref(), &d);
            let length = col_norm(&d);
            if length > 0.0 {
                self.tangents[i] = d * (1.0 / length);
            }
        }
    }

    /// The tracked directions of image `i` made orthogonal to its tangent.
    fn across(&mut self, i: usize) {
        let t = self.tangents[i].clone();
        let removed = self.frame.removed(&self.chain[i]);
        let projected: Vec<Col<f64>> = self.subspaces[i]
            .iter()
            .map(|w| {
                self.frame
                    .mask(removed.as_ref(), &(w - &t * col_dot(&t, w)))
            })
            .collect();
        let mut basis = gram_schmidt(&[vec![t.clone()], projected].concat());
        if !basis.is_empty() {
            basis.remove(0);
        }
        if basis.len() == self.subspaces[i].len() {
            self.subspaces[i] = basis;
        }
    }

    /// The transverse mode of image `i` from the products `hw` of its tracked directions;
    /// then an Oja step of the tracked subspace. Whether the mode switched.
    fn track(&mut self, i: usize, hw: &[Col<f64>]) -> MoptResult<bool> {
        let t = self.tangents[i].clone();
        let across = |v: &Col<f64>| v - &t * col_dot(&t, v);
        let pairs: Vec<Pair> = self.subspaces[i]
            .iter()
            .zip(hw)
            .map(|(w, h)| (across(w), across(h)))
            .collect();
        let ritz = rayleigh_ritz(&orthonormal_pairs(&pairs))?;
        if ritz.vectors.is_empty() {
            return Ok(false);
        }
        let previous = self.modes[i].clone();
        let (best, overlap) = match self.settings.transverse {
            TransverseMode::Lowest => (0, col_dot(&ritz.vectors[0], &previous)),
            TransverseMode::Continued => ritz
                .vectors
                .iter()
                .enumerate()
                .map(|(j, y)| (j, col_dot(y, &previous)))
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap_or((0, 1.0)),
        };
        let mut v = ritz.vectors[best].clone();
        if overlap < 0.0 {
            v = -v;
        }
        let switched = best != 0 || overlap.abs() < self.settings.switch_overlap;
        self.modes[i] = v;
        self.curvatures[i] = ritz.values[best];
        // The subspace follows the two lowest curvatures across the band.
        let d = euler_directions(DirectionUpdate::OjaQr, &ritz.vectors, &ritz.products);
        let scale = ritz.values.iter().fold(1e-12_f64, |m, c| m.max(c.abs()));
        let mut gamma = self.settings.rotation_rate / scale;
        let longest = d.iter().fold(0.0_f64, |m, x| m.max(col_norm(x)));
        if longest * gamma > self.settings.max_rotation {
            gamma = self.settings.max_rotation / longest;
        }
        let moved: Vec<Col<f64>> = ritz
            .vectors
            .iter()
            .zip(&d)
            .map(|(y, d)| y + d * gamma)
            .collect();
        let basis = gram_schmidt(&moved);
        if basis.len() == self.subspaces[i].len() {
            self.subspaces[i] = basis;
        }
        Ok(switched)
    }

    /// The effective gradient of image `i` (metric of the search).
    fn effective(&self, i: usize, climbing: bool) -> Col<f64> {
        let g = &self.gradients_q[i];
        let v = &self.modes[i];
        let tau = &self.tangents[i];
        if climbing {
            let t = tau - v * col_dot(v, tau);
            let t = orthonormal_to(&[], &t).unwrap_or_else(|| tau.clone());
            return g - v * (2.0 * col_dot(v, g)) - &t * (2.0 * col_dot(&t, g));
        }
        let q = |x: &Col<f64>| self.frame.direction_to_q(x);
        let ahead = q(&self.chain[i + 1]) - q(&self.chain[i]);
        let behind = q(&self.chain[i]) - q(&self.chain[i - 1]);
        let removed = self.frame.removed(&self.chain[i]);
        let spring = match self.settings.spring_force {
            RidgeSpring::Full => self.frame.mask(
                removed.as_ref(),
                &((&ahead - &behind) * self.settings.spring),
            ),
            RidgeSpring::Tangential => {
                tau * (self.settings.spring * (col_norm(&ahead) - col_norm(&behind)))
            }
        };
        g - tau * col_dot(tau, g) - v * (2.0 * col_dot(v, g)) - spring
    }

    fn decide_and_step(&mut self, switches: Vec<usize>) -> MoptResult<RidgeBandReport> {
        let last = self.chain.len() - 1;
        for i in 1..last {
            if self.curvatures[i] >= -self.settings.min_curvature {
                self.lost[i] += 1;
            } else {
                self.lost[i] = 0;
            }
            if self.lost[i] > self.settings.lost_after {
                self.phase = Phase::Done(RidgeBandEnd::RidgeModeLost(i));
                return Ok(self.report(switches, f64::NAN));
            }
        }
        let mut effective: Vec<Col<f64>> = (0..=last)
            .map(|i| {
                if i == 0 || i == last {
                    Col::zeros(self.chain[0].nrows())
                } else {
                    self.effective(i, false)
                }
            })
            .collect();
        let size = |g: &Col<f64>| col_max_abs(&self.frame.gradient_to_x(g));
        let count = (last - 1).max(1) as f64;
        let rms = libm::sqrt(
            (1..last)
                .map(|i| col_dot(&effective[i], &effective[i]))
                .sum::<f64>()
                / (count * self.chain[0].nrows() as f64),
        );
        // The highest inner image climbs once the band has settled and the energy has a
        // maximum along it there (above its neighbours and both ends); it may change
        // while the band moves.
        let peak = (1..last)
            .max_by(|a, b| self.energies[*a].total_cmp(&self.energies[*b]))
            .filter(|&c| {
                let e = self.energies[c];
                e > self.energies[c - 1]
                    && e > self.energies[c + 1]
                    && e > self.energies[0]
                    && e > self.energies[last]
            });
        let settled = rms < self.settings.climb_rms || self.iteration >= self.settings.climb_after;
        if self.climbing.is_some() || settled {
            self.climbing = peak.or(self.climbing);
        }
        if let Some(c) = self.climbing {
            effective[c] = self.effective(c, true);
        }
        let largest = (1..last).fold(0.0_f64, |m, i| m.max(size(&effective[i])));
        self.climbing_force = self.climbing.map(|c| size(&effective[c]));
        let others = (1..last)
            .filter(|i| Some(*i) != self.climbing)
            .fold(0.0_f64, |m, i| m.max(size(&effective[i])));
        // A climbing image on a stationary point without a maximum across the band (an
        // index-1 point under the band) is not the end.
        let band_settled = self.settings.band_tolerance.is_none_or(|t| others < t)
            && self
                .climbing
                .is_some_and(|c| self.curvatures[c] < -self.settings.min_curvature);
        if self
            .climbing_force
            .is_some_and(|f| f < self.settings.tolerance)
            && band_settled
        {
            self.phase = Phase::Done(RidgeBandEnd::Converged);
            return Ok(self.report(switches, largest));
        }
        if self.iteration + 1 >= self.settings.max_iterations {
            self.phase = Phase::Done(RidgeBandEnd::IterationLimit);
            return Ok(self.report(switches, largest));
        }
        if self.settings.step_rule == RidgeStep::Fire {
            self.fire_step(&effective);
        } else {
            self.bb_step(&effective);
        }
        // The tracked directions stay across the band at the new tangents.
        for i in 1..last {
            self.across(i);
        }
        self.iteration += 1;
        Ok(self.report(switches, largest))
    }

    /// One step of the band rule along the forces `-effective` (metric of the search).
    fn fire_step(&mut self, effective: &[Col<f64>]) {
        let last = self.chain.len() - 1;
        let forces: Vec<Col<f64>> = effective.iter().map(|g| -g).collect();
        if let Some((before, steps)) = &self.fire_last {
            self.fire.observe_curvature(&forces, before, steps);
        }
        let previous = self.fire_last.as_ref().map(|(f, _)| f.as_slice());
        let mut steps = self.fire.displacements(&forces, previous);
        for (i, step) in steps.iter_mut().enumerate() {
            if i == 0 || i == last {
                *step = Col::zeros(step.nrows());
                continue;
            }
            let mut dx = self.frame.direction_to_x(step);
            let length = self.frame.largest_displacement(&dx);
            if length > self.settings.step.max_step {
                dx *= self.settings.step.max_step / length;
                *step = self.frame.direction_to_q(&dx);
            }
            self.chain[i] += dx;
        }
        self.fire_last = Some((forces, steps));
    }

    /// Barzilai-Borwein steps of each image on its effective gradient.
    fn bb_step(&mut self, effective: &[Col<f64>]) {
        let last = self.chain.len() - 1;
        for (i, g_eff) in effective.iter().enumerate().take(last).skip(1) {
            let q = self.frame.direction_to_q(&self.chain[i]);
            let first = self.settings.step.first
                / self
                    .frame
                    .largest_displacement(&self.frame.direction_to_x(g_eff))
                    .max(1e-300);
            let beta = match &self.last[i] {
                Some((q0, g0)) => {
                    let s = &q - q0;
                    let y = g_eff - g0;
                    crate::path::barzilai_borwein(col_dot(&s, &y), col_dot(&y, &y), first)
                }
                None => first,
            };
            self.last[i] = Some((q, g_eff.clone()));
            let mut dx = self.frame.direction_to_x(&(g_eff * -beta));
            let length = self.frame.largest_displacement(&dx);
            if length > self.settings.step.max_step {
                dx *= self.settings.step.max_step / length;
            }
            self.chain[i] += dx;
        }
    }

    fn report(&self, switches: Vec<usize>, largest: f64) -> RidgeBandReport {
        let mut curvatures = self.curvatures.clone();
        let last = self.chain.len() - 1;
        curvatures[0] = f64::NAN;
        curvatures[last] = f64::NAN;
        RidgeBandReport {
            iteration: self.iteration,
            energies: self.energies.clone(),
            curvatures,
            largest_force: largest,
            climbing_force: self.climbing_force,
            climbing: self.climbing,
            switches,
            gradients: self.gradients,
            hessians: self.hessians,
            end: self.end(),
        }
    }
}

/// The settings of the chain relaxation of mopt-rs ([`super::machine::ChainRelaxation`]) on
/// the ridge between two transition states with the Hessians of the images (the Hessian
/// form of the band on a ridge, [`super::rfo::ChainRidge`]): the ends fixed,
/// energy-weighted springs, half the quasi-Newton step of every image and half the step
/// along the forces (`ratio = 0.5`, the default of `nebmain`), the exact Hessians of the
/// images asked for in every iteration, the highest image climbing after five iterations,
/// no redistribution. `end_modes` are the imaginary modes of the two transition states.
///
/// Measured on two independent double wells with the inner images displaced by 0.1 from the
/// start (`ridge_tests.rs`): with `ratio = 0.5` all 18 bands (three pairs, three spring
/// strengths, inner steps of 0.05 and 0.1) converged, in 73 to 133 iterations; with
/// `ratio = 1` the bands between transition states side by side did not settle unless the
/// springs were 30 times stronger (each image is drawn to the nearest stationary point when
/// its tangent is off), and with `ratio = 0.3` one band diverged.
pub fn hessian_ridge_settings(
    end_modes: [Col<f64>; 2],
    kind: super::ChainKind,
) -> super::machine::ChainSettings {
    super::machine::ChainSettings {
        kind,
        max_iterations: 200,
        ends: super::trust::FixedEnds {
            first: true,
            last: true,
        },
        springs: Some(super::springs::EnergyWeightedSprings::default()),
        climb_after: Some(5),
        rfo: Some(super::rfo::ChainRfoSettings {
            ratio: 0.5,
            ..super::rfo::ChainRfoSettings::default()
        }),
        hessian_every: Some(1),
        force_tolerance: Some(3e-4),
        predicted_maxima_every: None,
        equal_spacing_every: None,
        ridge: Some(super::rfo::ChainRidge::new(end_modes)),
        ..super::machine::ChainSettings::default()
    }
}

/// A band of `inner` images from `first` to `last` over a corner, for two transition
/// states that share a minimum (their imaginary modes `first_mode` and `last_mode` nearly
/// orthogonal): the ridge leaves each of them across its own mode, about along the mode of
/// the other, so the corner is put where the line through `first` along `last_mode` comes
/// closest to the line through `last` along `first_mode` (the midpoint of the two closest
/// points). The images are spread evenly along the two legs. On a surface of two
/// independent double wells the corner is the index-2 saddle itself. A straight band cuts
/// through the basin of the shared minimum, where there is no maximum across the band near
/// the ends.
pub fn cornered_band(
    first: &Col<f64>,
    last: &Col<f64>,
    first_mode: &Col<f64>,
    last_mode: &Col<f64>,
    inner: usize,
) -> MoptResult<Vec<Col<f64>>> {
    let unit = |v: &Col<f64>| {
        orthonormal_to(&[], v).ok_or_else(|| MoptError::config("cornered band: a mode is zero"))
    };
    let (e1, e2) = (unit(first_mode)?, unit(last_mode)?);
    let b = col_dot(&e1, &e2);
    if b.abs() > 0.95 {
        return Err(MoptError::config(
            "cornered band: the two modes are nearly parallel (the transition states lie side \
             by side on one ridge; start from a straight band)",
        ));
    }
    // Closest points of first + s e2 and last + t e1.
    let d = first - last;
    let (e1d, e2d) = (col_dot(&e1, &d), col_dot(&e2, &d));
    let s = (b * e1d - e2d) / (1.0 - b * b);
    let t = b * s + e1d;
    let corner = ((first + &e2 * s) + (last + &e1 * t)) * 0.5;
    let (leg1, leg2) = (col_norm(&(&corner - first)), col_norm(&(last - &corner)));
    let total = leg1 + leg2;
    if total <= 0.0 {
        return Ok(straight_band(first, last, inner));
    }
    Ok((0..inner + 2)
        .map(|i| {
            let along = total * i as f64 / (inner + 1) as f64;
            if along <= leg1 {
                first + (&corner - first) * (along / leg1.max(1e-300))
            } else {
                &corner + (last - &corner) * ((along - leg1) / leg2.max(1e-300))
            }
        })
        .collect())
}

/// A straight band of `inner` images between two points.
pub fn straight_band(first: &Col<f64>, last: &Col<f64>, inner: usize) -> Vec<Col<f64>> {
    (0..inner + 2)
        .map(|i| {
            let s = i as f64 / (inner + 1) as f64;
            first * (1.0 - s) + last * s
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "ridge_tests.rs"]
pub(crate) mod tests;
