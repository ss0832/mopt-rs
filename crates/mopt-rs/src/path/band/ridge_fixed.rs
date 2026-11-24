//! A band on the mode-maximized surface of one fixed direction between two transition
//! states whose imaginary modes are alike: its climbing image converges to an index-2 saddle
//! (the fixed-direction form of the band on a ridge; [`super::ridge`] is the moving-frame
//! form).
//!
//! With the imaginary modes `e1`, `e2` of the transition states `T1`, `T2` (signs aligned,
//! overlap at least [`FixedRidgeSettings::min_overlap`]), the direction `e = unit(e1 + e2)`
//! is kept fixed. The images `y` lie on the hyperplane through `T1` across `e`; the band is
//! the nudged elastic band of [`super::machine::ChainRelaxation`] (springs, climbing image,
//! the ends fixed) on the surface `E~(y) = max_t E(y + t e)` of
//! [`crate::path::hisd::mode_max`], each image maximized along `e` before every iteration of
//! the band (warm-started from its last maximum). The ends are `T1` and the projection of
//! `T2` onto the hyperplane along `e`; both transition states are maxima along `e` where
//! `e^T H e < 0` there, so they lie on the surface, are stationary points of it (the
//! envelope theorem) and minima of it (inertia additivity: `index E = 1 + index E~`).
//!
//! What is maximized: `E` along `e` at every image, to the line tolerance. What is
//! minimized: `E~` across the band. The climbing image maximizes `E~` along the band. Where
//! the band converges, its climbing image is a first-order saddle of `E~` between two of its
//! minima, so a stationary point of `E` of index `1 + 1 = 2`. Nothing tracks a mode and no
//! third derivatives enter. It needs a maximum along `e` at every image: the band ends with
//! [`FixedRidgeEnd::RidgeModeLost`] at an image where the curvature along `e` is not
//! negative, which is why it is for pairs with similar modes (a pair that shares a minimum
//! has nearly orthogonal modes and needs the moving frame).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::machine::{ChainAction, ChainEnd, ChainRelaxation, ChainSettings, ImageValue};
use super::springs::EnergyWeightedSprings;
use super::trust::FixedEnds;
use super::ChainKind;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};
use crate::path::hisd::frame::SearchFrame;
use crate::path::hisd::mode_max::{LineMaximum, LineSettings, LineStatus};
use crate::path::hisd::subspace::orthonormal_to;
use crate::path::points::{check_values, gradient_of, PointRequest, PointValue, SearchAction};

/// Settings of a [`FixedRidgeBand`].
#[derive(Debug, Clone, PartialEq)]
pub struct FixedRidgeSettings {
    /// The band on the surface (plain variables in the metric of the search; the ends are
    /// fixed whatever this says).
    pub chain: ChainSettings,
    /// The maximizations along `e`.
    pub line: LineSettings,
    /// The absolute overlap of the two imaginary modes must be at least this.
    pub min_overlap: f64,
    /// Curvature along `e` the first step of a line assumes (later steps take secants).
    pub initial_curvature: f64,
}

impl Default for FixedRidgeSettings {
    fn default() -> Self {
        Self {
            chain: ChainSettings {
                kind: ChainKind::Variables,
                max_iterations: 300,
                ends: FixedEnds {
                    first: true,
                    last: true,
                },
                align: false,
                springs: Some(EnergyWeightedSprings::default()),
                climb_after: Some(10),
                force_tolerance: Some(3e-4),
                ..ChainSettings::default()
            },
            line: LineSettings::default(),
            min_overlap: 0.7,
            initial_curvature: -1.0,
        }
    }
}

/// Why a band ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedRidgeEnd {
    /// The band met its force tolerance.
    Converged,
    /// The iterations of the band are used up.
    IterationLimit,
    /// No maximum along `e` at this image: the surface ends there.
    RidgeModeLost(usize),
    /// The maximization along `e` of this image did not converge.
    LineSteps(usize),
}

/// What one iteration of the band did.
#[derive(Debug, Clone)]
pub struct FixedRidgeReport {
    /// Iteration of the band.
    pub iteration: usize,
    /// Energies of the surface at the images.
    pub energies: Vec<f64>,
    /// The images that climbed.
    pub climbing: Vec<usize>,
    /// Largest force component of the images that move.
    pub largest_force: f64,
    /// Images whose maximum along `e` lay farther than the branch threshold from where its
    /// line started (another branch of maxima).
    pub jumps: Vec<usize>,
    /// Gradients evaluated so far.
    pub gradients: usize,
    /// Why the band ended, if it did.
    pub end: Option<FixedRidgeEnd>,
}

#[derive(Debug)]
enum Phase {
    /// The two transition states.
    Ends,
    /// The maximizations along `e` of the inner images.
    Lines,
    Done(FixedRidgeEnd),
}

/// Ask/tell band on the mode-maximized surface between two transition states.
pub struct FixedRidgeBand {
    settings: FixedRidgeSettings,
    frame: SearchFrame,
    ends: [Col<f64>; 2],
    end_modes: [Col<f64>; 2],
    inner: usize,
    e: Col<f64>,
    relaxation: Option<ChainRelaxation>,
    images: Vec<Col<f64>>,
    lines: Vec<Option<LineMaximum>>,
    t: Vec<f64>,
    end_values: Vec<(f64, Col<f64>)>,
    energies: Vec<f64>,
    phase: Phase,
    iteration: usize,
    gradients: usize,
}

impl FixedRidgeBand {
    /// A band of `inner` images between the transition states `first` and `last` with
    /// their imaginary modes (coordinates of the points).
    pub fn new(
        settings: FixedRidgeSettings,
        first: &Col<f64>,
        last: &Col<f64>,
        first_mode: &Col<f64>,
        last_mode: &Col<f64>,
        inner: usize,
    ) -> MoptResult<Self> {
        let n = first.nrows();
        if inner == 0 || [last, first_mode, last_mode].iter().any(|v| v.nrows() != n) {
            return Err(MoptError::config(
                "fixed ridge band: at least one inner image, and points and modes of one size",
            ));
        }
        let mut band = Self {
            settings,
            frame: SearchFrame::plain(),
            ends: [first.clone(), last.clone()],
            end_modes: [first_mode.clone(), last_mode.clone()],
            inner,
            e: Col::zeros(n),
            relaxation: None,
            images: Vec::new(),
            lines: Vec::new(),
            t: Vec::new(),
            end_values: Vec::new(),
            energies: Vec::new(),
            phase: Phase::Ends,
            iteration: 0,
            gradients: 0,
        };
        band.prepare()?;
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
        self.frame.check(self.ends[0].nrows())?;
        self.prepare()?;
        Ok(self)
    }

    fn prepare(&mut self) -> MoptResult<()> {
        let unit = |m: &Col<f64>, x: &Col<f64>| -> MoptResult<Col<f64>> {
            let removed = self.frame.removed(x);
            let q = self
                .frame
                .mask(removed.as_ref(), &self.frame.direction_to_q(m));
            orthonormal_to(&[], &q).ok_or_else(|| {
                MoptError::config("fixed ridge band: a mode is a rigid-body motion only")
            })
        };
        let e1 = unit(&self.end_modes[0], &self.ends[0])?;
        let mut e2 = unit(&self.end_modes[1], &self.ends[1])?;
        let overlap = col_dot(&e1, &e2);
        if overlap.abs() < self.settings.min_overlap {
            return Err(MoptError::config(format!(
                "fixed ridge band: the imaginary modes overlap by {:.2}, less than {:.2}; the \
                 moving-frame band is for such pairs",
                overlap.abs(),
                self.settings.min_overlap
            )));
        }
        if overlap < 0.0 {
            e2 = -e2;
        }
        self.e = orthonormal_to(&[], &(&e1 + &e2))
            .ok_or_else(|| MoptError::config("fixed ridge band: the modes cancel"))?;
        let q1 = self.frame.direction_to_q(&self.ends[0]);
        let q2 = self.frame.direction_to_q(&self.ends[1]);
        let t_end = col_dot(&self.e, &(&q2 - &q1));
        let y_end = &q2 - &self.e * t_end;
        let images = self.inner + 2;
        let chain: Vec<Col<f64>> = (0..images)
            .map(|i| {
                let s = i as f64 / (images - 1) as f64;
                &q1 * (1.0 - s) + &y_end * s
            })
            .collect();
        self.t = (0..images)
            .map(|i| t_end * i as f64 / (images - 1) as f64)
            .collect();
        let mut chain_settings = self.settings.chain.clone();
        chain_settings.kind = ChainKind::Variables;
        chain_settings.align = false;
        chain_settings.ends = FixedEnds {
            first: true,
            last: true,
        };
        self.relaxation = Some(ChainRelaxation::new(chain_settings, chain)?);
        self.phase = Phase::Ends;
        Ok(())
    }

    /// A point of the metric of the search back to the coordinates of the points.
    fn to_x(&self, q: &Col<f64>) -> Col<f64> {
        self.frame.direction_to_x(q)
    }

    /// The fixed direction (metric of the search).
    pub fn direction(&self) -> &Col<f64> {
        &self.e
    }

    /// The images on the surface, `y + t* e`, in the coordinates of the points.
    pub fn chain(&self) -> Vec<Col<f64>> {
        self.images
            .iter()
            .zip(&self.t)
            .map(|(y, t)| self.to_x(&(y + &self.e * *t)))
            .collect()
    }

    /// The highest inner image.
    pub fn highest(&self) -> usize {
        let last = self.energies.len().saturating_sub(1);
        (1..last)
            .max_by(|a, b| self.energies[*a].total_cmp(&self.energies[*b]))
            .unwrap_or(1)
    }

    /// The highest inner image in the coordinates of the points.
    pub fn saddle_point(&self) -> Option<Col<f64>> {
        self.chain().get(self.highest()).cloned()
    }

    /// The two directions of the highest image for a search of index 2: `e` and the
    /// tangent of the band there, in the coordinates of the points, each of unit length.
    pub fn saddle_directions(&self) -> Vec<Col<f64>> {
        let c = self.highest();
        let mut out: Vec<Col<f64>> = orthonormal_to(&[], &self.to_x(&self.e))
            .into_iter()
            .collect();
        if c >= 1 && c + 1 < self.images.len() {
            let tangent = &self.images[c + 1] - &self.images[c - 1];
            let tangent = &tangent - &self.e * col_dot(&self.e, &tangent);
            if let Some(t) = orthonormal_to(&[], &self.to_x(&tangent)) {
                out.push(t);
            }
        }
        out
    }

    /// Why the band ended, once it has.
    pub fn end(&self) -> Option<FixedRidgeEnd> {
        match self.phase {
            Phase::Done(end) => Some(end),
            _ => None,
        }
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        match &self.phase {
            Phase::Done(_) => SearchAction::Finished,
            Phase::Ends => SearchAction::Evaluate(vec![
                PointRequest::gradient(self.ends[0].clone()),
                PointRequest::gradient(self.ends[1].clone()),
            ]),
            Phase::Lines => SearchAction::Evaluate(
                self.lines
                    .iter()
                    .flatten()
                    .filter(|l| l.status() == LineStatus::Running)
                    .map(|l| PointRequest::gradient(self.to_x(&l.point())))
                    .collect(),
            ),
        }
    }

    /// The gradient at `x` (points' coordinates) in the metric of the search, without the
    /// motions left out.
    fn q_gradient(&self, x: &Col<f64>, value: &PointValue) -> MoptResult<Col<f64>> {
        let removed = self.frame.removed(x);
        Ok(self.frame.mask(
            removed.as_ref(),
            &self
                .frame
                .gradient_to_q(&gradient_of("fixed ridge band", value)?),
        ))
    }

    /// The values of the points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<Option<FixedRidgeReport>> {
        let asked = match self.ask() {
            SearchAction::Evaluate(points) => points,
            SearchAction::Finished => {
                return Err(MoptError::config(
                    "tell() called after the fixed ridge band ended",
                ))
            }
        };
        check_values("fixed ridge band", &asked, &values, self.ends[0].nrows())?;
        self.gradients += values.len();
        match self.phase {
            Phase::Ends => {
                self.end_values = Vec::new();
                for (k, value) in values.iter().enumerate() {
                    let g = self.q_gradient(&self.ends[k], value)?;
                    let reduced = &g - &self.e * col_dot(&self.e, &g);
                    self.end_values.push((value.energy, reduced));
                }
                self.next_band_iteration()?;
                Ok(None)
            }
            Phase::Lines => {
                let mut k = 0;
                for i in 0..self.lines.len() {
                    let running = self.lines[i]
                        .as_ref()
                        .is_some_and(|l| l.status() == LineStatus::Running);
                    if !running {
                        continue;
                    }
                    let x = asked[k].x.clone();
                    let g = self.q_gradient(&x, &values[k])?;
                    k += 1;
                    let status = match self.lines[i].as_mut() {
                        Some(line) => line.tell(values[k - 1].energy, g)?,
                        None => continue,
                    };
                    match status {
                        LineStatus::NoMaximum => {
                            self.phase = Phase::Done(FixedRidgeEnd::RidgeModeLost(i));
                            return Ok(Some(self.report(Vec::new(), f64::NAN, Vec::new())));
                        }
                        LineStatus::Steps => {
                            self.phase = Phase::Done(FixedRidgeEnd::LineSteps(i));
                            return Ok(Some(self.report(Vec::new(), f64::NAN, Vec::new())));
                        }
                        LineStatus::Running | LineStatus::Converged => {}
                    }
                }
                let all_done = self
                    .lines
                    .iter()
                    .flatten()
                    .all(|l| l.status() == LineStatus::Converged);
                if !all_done {
                    return Ok(None);
                }
                self.finish_band_iteration().map(Some)
            }
            Phase::Done(_) => Err(MoptError::config(
                "tell() called after the fixed ridge band ended",
            )),
        }
    }

    /// The images of the band have their maxima: tell the band, then start the next
    /// iteration.
    fn finish_band_iteration(&mut self) -> MoptResult<FixedRidgeReport> {
        let images = self.images.len();
        let mut values = Vec::with_capacity(images);
        let mut jumps = Vec::new();
        for i in 0..images {
            if i == 0 || i == images - 1 {
                let (energy, g) = &self.end_values[if i == 0 { 0 } else { 1 }];
                values.push(ImageValue::new(*energy, g.clone()));
                continue;
            }
            let line = self.lines[i]
                .as_ref()
                .ok_or_else(|| MoptError::config("fixed ridge band: an image has no line"))?;
            if line.jumped() {
                jumps.push(i);
            }
            self.t[i] = line.t();
            values.push(ImageValue::new(line.energy(), line.reduced_gradient()));
        }
        self.energies = values.iter().map(|v| v.energy).collect();
        let relaxation = self
            .relaxation
            .as_mut()
            .ok_or_else(|| MoptError::config("fixed ridge band: no band"))?;
        let report = relaxation.tell(values)?;
        self.iteration = report.iteration + 1;
        let climbing = report.climbing.clone();
        let largest = report.largest_force;
        self.next_band_iteration()?;
        Ok(self.report(climbing, largest, jumps))
    }

    /// Ask the band for its next images and start their maximizations along `e`.
    fn next_band_iteration(&mut self) -> MoptResult<()> {
        let relaxation = self
            .relaxation
            .as_mut()
            .ok_or_else(|| MoptError::config("fixed ridge band: no band"))?;
        match relaxation.ask() {
            ChainAction::Finished(end) => {
                self.phase = Phase::Done(match end {
                    ChainEnd::Converged => FixedRidgeEnd::Converged,
                    ChainEnd::IterationLimit => FixedRidgeEnd::IterationLimit,
                });
            }
            ChainAction::Evaluate(request) => {
                let images = request.images.len();
                if images != self.t.len() {
                    // The band changed its number of images: lines start on the straight
                    // interpolation of the displacements of the ends.
                    let t_end = self.t.last().copied().unwrap_or(0.0);
                    self.t = (0..images)
                        .map(|i| t_end * i as f64 / (images - 1).max(1) as f64)
                        .collect();
                }
                self.lines = (0..images)
                    .map(|i| {
                        if i == 0 || i == images - 1 {
                            Ok(None)
                        } else {
                            LineMaximum::new(
                                self.settings.line,
                                request.images[i].clone(),
                                self.e.clone(),
                                self.t[i],
                                self.settings.initial_curvature,
                            )
                            .map(Some)
                        }
                    })
                    .collect::<MoptResult<_>>()?;
                self.images = request.images;
                self.phase = Phase::Lines;
            }
        }
        Ok(())
    }

    fn report(&self, climbing: Vec<usize>, largest: f64, jumps: Vec<usize>) -> FixedRidgeReport {
        FixedRidgeReport {
            iteration: self.iteration,
            energies: self.energies.clone(),
            climbing,
            largest_force: largest,
            jumps,
            gradients: self.gradients,
            end: self.end(),
        }
    }

    /// Gradients evaluated so far.
    pub fn gradient_evaluations(&self) -> usize {
        self.gradients
    }

    /// The length of the band (sum of the distances of its images on the hyperplane).
    pub fn length(&self) -> f64 {
        self.images
            .windows(2)
            .map(|w| col_norm(&(&w[1] - &w[0])))
            .sum()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::ridge::tests::{at, mode, quartic};
    use super::*;
    use crate::math::linalg::symmetric_eigen;

    #[test]
    fn the_climbing_image_on_the_mode_maximized_surface_is_the_index_two_saddle() {
        let (f, r) = quartic();
        let saddle = at(&r, [0.0, 0.0, 1.0, 1.0]);
        for (first, second) in [
            ([1.0, 0.0, 1.0, 1.0], [-1.0, 0.0, 1.0, 1.0]),
            ([0.0, 1.0, 1.0, 1.0], [0.0, -1.0, 1.0, 1.0]),
        ] {
            let (t1, t2) = (at(&r, first), at(&r, second));
            let mut settings = FixedRidgeSettings::default();
            settings.chain.force_tolerance = Some(1e-5);
            settings.chain.max_iterations = 3000;
            settings.line.tolerance = 1e-8;
            let mut band =
                FixedRidgeBand::new(settings, &t1, &t2, &mode(&f, &t1), &mode(&f, &t2), 7).unwrap();
            let mut trace = Vec::new();
            while let SearchAction::Evaluate(points) = band.ask() {
                let values = points
                    .iter()
                    .map(|p| {
                        let mut v = f(&p.x);
                        v.hessian = None;
                        v
                    })
                    .collect();
                if let Some(r) = band.tell(values).unwrap() {
                    if r.iteration % 100 == 1 {
                        trace.push(format!(
                            "{} {:.1e} {:?}",
                            r.iteration, r.largest_force, r.climbing
                        ));
                    }
                }
            }
            assert_eq!(
                band.end(),
                Some(FixedRidgeEnd::Converged),
                "{first:?} {trace:?}"
            );
            let top = band.saddle_point().unwrap();
            assert!(col_norm(&(&top - &saddle)) < 1e-4, "{top:?}");
            let h = f(&top).hessian.unwrap();
            let negative = symmetric_eigen(&h)
                .unwrap()
                .0
                .iter()
                .filter(|v| **v < 0.0)
                .count();
            assert_eq!(negative, 2);
            assert_eq!(band.saddle_directions().len(), 2);
        }
        // An adjacent pair: the modes are orthogonal.
        let (t1, t2) = (at(&r, [1.0, 0.0, 1.0, 1.0]), at(&r, [0.0, 1.0, 1.0, 1.0]));
        assert!(FixedRidgeBand::new(
            FixedRidgeSettings::default(),
            &t1,
            &t2,
            &mode(&f, &t1),
            &mode(&f, &t2),
            5
        )
        .is_err());
    }
}
