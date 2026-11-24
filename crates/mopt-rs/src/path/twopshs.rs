//! Two-point scaled hypersphere search (2PSHS): a path from one minimum to another along
//! energy minima on contracting scaled hyperspheres, whose highest point approximates the
//! transition state between them.
//!
//! The hypersphere is centred at the first minimum `A` in its scaled normal coordinates
//! ([`super::shs`]); the second structure `B` lies on the sphere of radius `|q(B)|`. The
//! radius is reduced step by step and on each sphere the energy minimum is located, starting
//! from the minimum of the sphere before (moved onto the smaller sphere along its radius), until
//! the sphere has shrunk onto `A`. The minima on the spheres lead from `B` over the ridge
//! between the basins into the basin of `A`; the highest of them is the estimate of the
//! transition state, to be refined by a search of order 1.
//!
//! S. Maeda, K. Ohno, Chem. Phys. Lett. 2005, 404, 95 (2PSHS); K. Ohno, S. Maeda, Chem. Phys.
//! Lett. 2004, 384, 277 (SHS). The radius goes down by the length `step` along the stiffest
//! mode of `A` per sphere (in the units of the radius, `step * l_max^(1/2)`).
//!
//! Hard constraints `c(x) = t` ([`TwoPointShs::with_manifold`]; new in mopt-rs, not in the
//! papers): both structures lie on the manifold `M` of the constraints (the second put onto
//! the values of the first), and `A` is the minimum of the energy on `M`. The spheres are
//! those of the normal coordinates of the projected Hessian of the Lagrangian at `A`
//! ([`ScaledCoordinates::on_manifold`]), whose modes span `T_A M`; a point of a sphere is the
//! point of `M` with those coordinates, so that the minimization on a sphere keeps the radius
//! and the constraints together, and its gradient is the derivative of the energy along `M`
//! ([`ScaledCoordinates::gradient_at`]). The path is then the 2PSHS path of the energy on
//! `M`, and its highest point the estimate of the constrained transition state.
//!
//! [`TwoPointShs`] is an ask/tell state machine over points of `n` variables. The minimization
//! on a sphere is that of the IRC ([`crate::irc::sphere::SphereMinimizer`]) in the scaled
//! coordinates, whose Hessian at `A` is the unit matrix; the Hessian it updates is carried from
//! sphere to sphere.
//!
//! MultiOptPy: `OtherMethod/twopshs.py:twoPSHSlikeMethod`. MultiOptPy's version does not
//! follow the paper: its radius is the Cartesian distance of the structures, its maps between
//! scaled and Cartesian coordinates are not inverse to each other, it minimizes the ADD of
//! `B` measured against the energy of `A` with steepest-descent steps biased towards `A`, it
//! reads the step size and number of the ADDF options (`-addf_step`, `-addf_num`) instead of
//! `-2pshs_step` and `-2pshs_num`, and it reports the structure of the lowest ADD.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::manifold::PointManifold;
use super::points::{
    check_values, gradient_of, hessian_of, PointRequest, PointValue, SearchAction,
};
use super::shs::ScaledCoordinates;
use crate::error::{MoptError, MoptResult};
use crate::irc::sphere::{SphereMinimizer, SphereStatus};
use crate::irc::SphereSettings;
use crate::math::linalg::col_norm;

/// Settings of a 2PSHS run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPshsSettings {
    /// Length along the stiffest mode of `A` by which the radius shrinks per sphere (Bohr;
    /// the unit of the variables otherwise).
    pub step: f64,
    /// Spheres at most (`-2pshs_num`).
    pub max_spheres: usize,
    /// The minimization on a sphere.
    pub sphere: SphereSettings,
}

impl Default for TwoPshsSettings {
    fn default() -> Self {
        Self {
            step: 0.05 * crate::domain::Units::ANGSTROM_TO_BOHR,
            max_spheres: 300,
            sphere: SphereSettings::default(),
        }
    }
}

/// Why a 2PSHS run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoPshsEnd {
    /// The sphere has shrunk onto the first minimum.
    ReachedCenter,
    /// The spheres are used up.
    SphereLimit,
    /// With hard constraints: no point of the constraints has the coordinates the
    /// minimization on a sphere asks for (they curve away from the tangent space of the
    /// minimum there); the path ends with the points so far.
    LeftConstraints,
}

/// One point of the path: the energy minimum on a sphere.
#[derive(Debug, Clone)]
pub struct WalkPoint {
    /// Sphere (0: the second structure itself).
    pub sphere: usize,
    /// Radius in scaled coordinates.
    pub radius: f64,
    /// The point.
    pub x: Col<f64>,
    /// Its energy.
    pub energy: f64,
    /// Evaluations on its sphere.
    pub evaluations: usize,
    /// RMS of the gradient within the sphere (scaled coordinates) at the end.
    pub tangential_rms: f64,
}

#[derive(Debug, Clone)]
enum Phase {
    /// The Hessian of the centre is asked for, and the energy of `B`.
    Setup,
    /// A sphere is minimized.
    Sphere {
        k: usize,
        radius: f64,
        minimizer: Box<SphereMinimizer>,
    },
    Done(TwoPshsEnd),
}

/// Ask/tell state machine of the two-point scaled hypersphere search.
pub struct TwoPointShs {
    settings: TwoPshsSettings,
    center: Col<f64>,
    other: Col<f64>,
    removed: Option<Mat<f64>>,
    coordinates: Option<ScaledCoordinates>,
    phase: Phase,
    path: Vec<WalkPoint>,
    radius_step: f64,
    evaluations_on_sphere: usize,
    manifold: Option<Arc<PointManifold>>,
    /// The point of the sphere asked for (on the hard constraints, if any).
    asked: Option<Col<f64>>,
}

impl TwoPointShs {
    /// Hold every point on the hard constraints `manifold` (both structures lie on them; the
    /// first is their constrained minimum): the spheres are those of the constrained normal
    /// coordinates ([`ScaledCoordinates::on_manifold`]).
    pub fn with_manifold(mut self, manifold: Arc<PointManifold>) -> Self {
        self.manifold = Some(manifold);
        self
    }

    /// The point of the minimizer of the sphere, placed (the run ends where no point of the
    /// constraints has its coordinates).
    fn place_asked(&mut self) -> MoptResult<()> {
        self.asked = match (&self.phase, &self.coordinates) {
            (Phase::Sphere { minimizer, .. }, Some(c)) => c.try_place(minimizer.point())?,
            _ => None,
        };
        if self.asked.is_none() && matches!(self.phase, Phase::Sphere { .. }) {
            self.phase = Phase::Done(TwoPshsEnd::LeftConstraints);
        }
        Ok(())
    }

    /// A search from `other` to the minimum `center`; the columns of `removed` (orthonormal)
    /// are the directions left out of the scaled coordinates (the rigid-body motions of
    /// structures).
    pub fn new(
        settings: TwoPshsSettings,
        center: &Col<f64>,
        other: &Col<f64>,
        removed: Option<Mat<f64>>,
    ) -> MoptResult<Self> {
        let n = center.nrows();
        if n == 0 || other.nrows() != n {
            return Err(MoptError::config("2PSHS: the structures differ in size"));
        }
        if !(settings.step.is_finite() && settings.step > 0.0) || settings.max_spheres == 0 {
            return Err(MoptError::config(
                "2PSHS: the step and the number of spheres must be positive",
            ));
        }
        Ok(Self {
            settings,
            center: center.clone(),
            other: other.clone(),
            removed,
            coordinates: None,
            phase: Phase::Setup,
            path: Vec::new(),
            radius_step: 0.0,
            evaluations_on_sphere: 0,
            manifold: None,
            asked: None,
        })
    }

    /// The points of the path so far, from the second structure towards the first.
    pub fn path(&self) -> &[WalkPoint] {
        &self.path
    }

    /// The scaled coordinates of the centre, once its Hessian was told.
    pub fn coordinates(&self) -> Option<&ScaledCoordinates> {
        self.coordinates.as_ref()
    }

    /// Why the run ended, once it has.
    pub fn end(&self) -> Option<TwoPshsEnd> {
        match self.phase {
            Phase::Done(end) => Some(end),
            _ => None,
        }
    }

    /// The highest point of the path besides its ends: the estimate of the transition
    /// state (`None` while there is none, or if the path has no maximum within).
    pub fn transition_state_estimate(&self) -> Option<&WalkPoint> {
        let highest = self
            .path
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.energy.total_cmp(&b.1.energy))?;
        (highest.0 > 0 && highest.0 + 1 < self.path.len()).then_some(highest.1)
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        match &self.phase {
            Phase::Setup => SearchAction::Evaluate(vec![
                PointRequest::hessian(self.center.clone()),
                PointRequest::energy(self.other.clone()),
            ]),
            Phase::Sphere { .. } => match &self.asked {
                Some(x) => SearchAction::Evaluate(vec![PointRequest::gradient(x.clone())]),
                None => SearchAction::Finished,
            },
            Phase::Done(_) => SearchAction::Finished,
        }
    }

    /// The values of the points of [`Self::ask`]. Returns the point of the path that was
    /// completed, if one was.
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<Option<WalkPoint>> {
        let asked = match self.ask() {
            SearchAction::Evaluate(points) => points,
            SearchAction::Finished => {
                return Err(MoptError::config("tell() called after the 2PSHS run ended"))
            }
        };
        check_values("2PSHS", &asked, &values, self.center.nrows())?;
        match std::mem::replace(&mut self.phase, Phase::Done(TwoPshsEnd::SphereLimit)) {
            Phase::Setup => self.setup(&values),
            Phase::Sphere {
                k,
                radius,
                mut minimizer,
            } => {
                let value = &values[0];
                let c = self
                    .coordinates
                    .as_ref()
                    .ok_or_else(|| MoptError::calc("2PSHS: no scaled coordinates"))?;
                let g = c.gradient_at(&asked[0].x, &gradient_of("2PSHS", value)?)?;
                self.evaluations_on_sphere += 1;
                match minimizer.tell(value.energy, &g)? {
                    SphereStatus::Continue => {
                        self.phase = Phase::Sphere {
                            k,
                            radius,
                            minimizer,
                        };
                        self.place_asked()?;
                        Ok(None)
                    }
                    SphereStatus::Converged | SphereStatus::Exhausted => {
                        let point = WalkPoint {
                            sphere: k,
                            radius,
                            x: c.place(minimizer.point())?,
                            energy: value.energy,
                            evaluations: self.evaluations_on_sphere,
                            tangential_rms: minimizer.tangential_rms(),
                        };
                        self.path.push(point.clone());
                        let start = minimizer.point().clone();
                        self.next_sphere(k + 1, &start, minimizer.into_hessian())?;
                        Ok(Some(point))
                    }
                }
            }
            Phase::Done(end) => {
                self.phase = Phase::Done(end);
                Err(MoptError::config("tell() called after the 2PSHS run ended"))
            }
        }
    }

    fn setup(&mut self, values: &[PointValue]) -> MoptResult<Option<WalkPoint>> {
        let hessian = hessian_of("2PSHS", &values[0])?;
        let c = match &self.manifold {
            Some(m) => ScaledCoordinates::on_manifold(
                &self.center,
                &hessian,
                self.removed.as_ref(),
                Arc::clone(m),
            )?,
            None => ScaledCoordinates::new(&self.center, &hessian, self.removed.as_ref())?,
        };
        let q = c.coordinates(&self.other);
        // The second structure is the point of the constraints with these coordinates: the
        // walks to the points of the first spheres start from it.
        c.remember(&self.other);
        let radius = col_norm(&q);
        self.radius_step = c.radius_for_length(self.settings.step);
        if radius <= self.radius_step {
            return Err(MoptError::config(
                "2PSHS: the second structure lies within one step of the first",
            ));
        }
        let start = WalkPoint {
            sphere: 0,
            radius,
            x: self.other.clone(),
            energy: values[1].energy,
            evaluations: 0,
            tangential_rms: f64::NAN,
        };
        self.path.push(start.clone());
        let m = c.dimension();
        self.coordinates = Some(c);
        self.next_sphere(1, &q, Mat::identity(m, m))?;
        Ok(Some(start))
    }

    fn next_sphere(&mut self, k: usize, start: &Col<f64>, hessian: Mat<f64>) -> MoptResult<()> {
        let first = self.path.first().map_or(0.0, |p| p.radius);
        let radius = first - k as f64 * self.radius_step;
        self.evaluations_on_sphere = 0;
        if radius <= 0.5 * self.radius_step {
            self.phase = Phase::Done(TwoPshsEnd::ReachedCenter);
            return Ok(());
        }
        if k > self.settings.max_spheres {
            self.phase = Phase::Done(TwoPshsEnd::SphereLimit);
            return Ok(());
        }
        let m = hessian.nrows();
        let minimizer =
            SphereMinimizer::new(Col::zeros(m), radius, start, hessian, self.settings.sphere)?;
        self.phase = Phase::Sphere {
            k,
            radius,
            minimizer: Box::new(minimizer),
        };
        self.place_asked()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// `E = (x^2 - 1)^2 + 2 (y - 0.2 x)^2` with its derivatives.
    fn surface(p: &Col<f64>) -> PointValue {
        let (x, y) = (p[0], p[1]);
        let w = y - 0.2 * x;
        let e = (x * x - 1.0).powi(2) + 2.0 * w * w;
        let g = Col::from_fn(2, |i| {
            if i == 0 {
                4.0 * x * (x * x - 1.0) - 0.8 * w
            } else {
                4.0 * w
            }
        });
        let h = Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 12.0 * x * x - 4.0 + 0.16,
            (1, 1) => 4.0,
            _ => -0.8,
        });
        PointValue {
            energy: e,
            gradient: Some(g),
            hessian: Some(h),
        }
    }

    #[test]
    fn the_highest_point_is_near_the_saddle() {
        let a = Col::from_fn(2, |i| [1.0, 0.2][i]);
        let b = Col::from_fn(2, |i| [-1.0, -0.2][i]);
        let settings = TwoPshsSettings {
            step: 0.05,
            ..TwoPshsSettings::default()
        };
        let mut s = TwoPointShs::new(settings, &a, &b, None).unwrap();
        while let SearchAction::Evaluate(points) = s.ask() {
            let values = points.iter().map(|p| surface(&p.x)).collect();
            s.tell(values).unwrap();
        }
        assert_eq!(s.end(), Some(TwoPshsEnd::ReachedCenter));
        let ts = s.transition_state_estimate().unwrap();
        assert!(col_norm(&ts.x) < 0.1, "{:?}", ts.x);
        assert!((ts.energy - 1.0).abs() < 0.02, "{}", ts.energy);
        // The radii shrink from B towards A.
        assert!(s.path().windows(2).all(|w| w[1].radius < w[0].radius));
    }
}
