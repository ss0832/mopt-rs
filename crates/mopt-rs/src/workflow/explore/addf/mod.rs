//! Anharmonic downward distortion following (ADDF): reaction paths from a minimum, found
//! where the energy falls furthest below its harmonic approximation, and followed outwards on
//! growing scaled hyperspheres until they pass over a transition state.
//!
//! In the scaled normal coordinates of the minimum ([`crate::path::shs`]) the harmonic energy is
//! `|q|^2 / 2`; on a sphere `|q| = R` the energy minima are the maximal anharmonic downward
//! distortions (ADD, `E - E0 - R^2 / 2`). The procedure:
//!
//! 1. the Hessian at the minimum gives the scaled coordinates;
//! 2. on the first sphere (the displacement along the stiffest mode 0.03 Angstrom) the ADD is
//!    evaluated along both directions of every mode, and the largest ADDs (the `paths` most
//!    negative ones; all with `paths = 0`) are followed;
//! 3. on the sphere `p` (radius `p` times that of a displacement of `step_length` along the
//!    stiffest mode) every path is moved onto the sphere along its radius and the energy is
//!    minimized there, the paths in the order of their ADD on the sphere before, each with the
//!    elimination functions (`cos^3`) of the paths already minimized on the sphere added
//!    (iterative optimization-elimination), so that two paths do not end on one ADD; a path that
//!    ends closer to another than a small angle has run into it and is dropped;
//! 4. when the energy along a path falls from one sphere to the next, the path has passed a
//!    transition state, approximated by its highest point; the path ends there; a path whose
//!    structure falls apart into fragments has reached a dissociation channel and ends too
//!    (the test is given by the caller, [`Addf::with_dissociation_test`]);
//! 5. the run ends when no path is left, or after the largest number of spheres.
//!
//! K. Ohno, S. Maeda, Chem. Phys. Lett. 2004, 384, 277; S. Maeda, K. Ohno, J. Phys. Chem. A
//! 2005, 109, 5742; the sizes of the spheres, the predictor (the projection of the maximal ADD
//! points of a sphere onto the next), and the order of the elimination as the reviews and the
//! documentation of the method describe them (S. Maeda, K. Ohno, K. Morokuma, Phys. Chem. Chem.
//! Phys. 2013, 15, 3683; S. Maeda et al., Bull. Chem. Soc. Jpn. 2014, 87, 1315). Following only
//! the largest ADDs of the first sphere is the large-ADD variant of the method.
//!
//! Hard constraints `c(x) = t` ([`Addf::with_manifold`]; new in mopt-rs, not in the papers):
//! the start is the minimum of the energy on the manifold `M` of the constraints (its
//! gradient on `M` vanishes; the surfaces give that gradient), the scaled coordinates are the
//! normal coordinates of the projected Hessian of the Lagrangian there, which span `T_A M`
//! ([`ScaledCoordinates::on_manifold`]), and a point of a sphere is the point of `M` with its
//! coordinates, so that the radius and the constraints are kept together through the scan and
//! every minimization. The ADD `E - E0 - R^2 / 2` is then the distortion of the energy on `M`
//! below its harmonic approximation on `M`, the elimination functions live in the same
//! coordinates, and a path passes the transition state of the energy on `M`.
//!
//! [`Addf`] is an ask/tell state machine over points of `n` variables, in the exploration core
//! so that reaction-network searches can use it; `ieipmain -addf` runs it on one structure.
//! The minimization on a sphere is that of the IRC ([`crate::irc::sphere::SphereMinimizer`]).
//!
//! MultiOptPy: `OtherMethod/addf.py:ADDFlikeMethod`. MultiOptPy's version does not follow the
//! method: its scaled coordinates are not levels of the harmonic energy (see
//! [`crate::path::shs`]), the first sphere is not minimized and the following ones are
//! minimized by a few steepest-descent steps with random kicks, the radius grows in the square
//! root of the energy, the transition state is declared where the gradient along the radius
//! turns negative after the fifth sphere, and the elimination functions come from differences.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#[cfg(test)]
mod tests;
mod types;

use std::sync::Arc;

use faer::{Col, Mat};

pub use types::{AddfEnd, AddfEvent, AddfPath, AddfPoint, AddfSettings, Candidate, PathState};

use crate::error::{MoptError, MoptResult};
use crate::irc::sphere::{SphereMinimizer, SphereStatus};
use crate::math::linalg::{col_max_abs, col_norm};
use crate::path::manifold::PointManifold;
use crate::path::points::{
    check_values, gradient_of, hessian_of, PointRequest, PointValue, SearchAction,
};
use crate::path::shs::{angle, elimination, ScaledCoordinates};

/// The sphere being minimized.
struct OnSphere {
    sphere: usize,
    radius: f64,
    /// The paths of this sphere in their order, and the position of the current one.
    order: Vec<usize>,
    position: usize,
    minimizer: Box<SphereMinimizer>,
    evaluations: usize,
    /// Maximal ADD points found on this sphere: path, scaled coordinates, ADD.
    found: Vec<(usize, Col<f64>, f64)>,
}

enum Phase {
    Setup,
    Scan,
    Sphere(Box<OnSphere>),
    Done(AddfEnd),
}

/// Whether a point has fallen apart into fragments (structures; a dissociation channel).
pub type DissociationTest = Box<dyn Fn(&Col<f64>) -> bool + Send>;

/// Ask/tell state machine of the anharmonic downward distortion following.
pub struct Addf {
    settings: AddfSettings,
    origin: Col<f64>,
    removed: Option<Mat<f64>>,
    coordinates: Option<ScaledCoordinates>,
    energy0: f64,
    first_radius: f64,
    radius_step: f64,
    paths: Vec<AddfPath>,
    phase: Phase,
    dissociated: Option<DissociationTest>,
    manifold: Option<Arc<PointManifold>>,
    /// The points of the scan of the first sphere (placed on the hard constraints; `None`
    /// where none is).
    scan_x: Vec<Option<Col<f64>>>,
    /// The point of the sphere asked for.
    asked: Option<Col<f64>>,
}

impl Addf {
    /// Hold every point on the hard constraints `manifold` (the start is their constrained
    /// minimum): the spheres are those of the constrained normal coordinates
    /// ([`ScaledCoordinates::on_manifold`]), and the ADD is measured on the constraints.
    pub fn with_manifold(mut self, manifold: Arc<PointManifold>) -> Self {
        self.manifold = Some(manifold);
        self
    }

    /// The point of the minimizer of the sphere, placed. Where no point of the hard
    /// constraints has its coordinates, the path ends there (as where the surface is not
    /// defined) and the next path of the sphere goes on; the events of that are returned.
    fn place_asked(&mut self) -> MoptResult<Vec<AddfEvent>> {
        let placed = match (&self.phase, &self.coordinates) {
            (Phase::Sphere(s), Some(c)) => c.try_place(s.minimizer.point())?,
            _ => {
                self.asked = None;
                return Ok(Vec::new());
            }
        };
        if placed.is_some() {
            self.asked = placed;
            return Ok(Vec::new());
        }
        self.asked = None;
        match std::mem::replace(&mut self.phase, Phase::Done(AddfEnd::SphereLimit)) {
            Phase::Sphere(s) => {
                let k = s.order[s.position];
                self.paths[k].state = PathState::Undefined;
                let events = vec![AddfEvent::Undefined {
                    path: self.paths[k].id,
                }];
                self.next_on_sphere(*s, events)
            }
            other => {
                self.phase = other;
                Ok(Vec::new())
            }
        }
    }

    /// A run from the minimum `origin`; the columns of `removed` (orthonormal) are the
    /// directions left out of the scaled coordinates (the rigid-body motions of structures).
    pub fn new(
        settings: AddfSettings,
        origin: &Col<f64>,
        removed: Option<Mat<f64>>,
    ) -> MoptResult<Self> {
        settings.validate()?;
        if origin.nrows() == 0 || origin.iter().any(|v| !v.is_finite()) {
            return Err(MoptError::config(
                "ADDF: the starting point is empty or not finite",
            ));
        }
        Ok(Self {
            settings,
            origin: origin.clone(),
            removed,
            coordinates: None,
            energy0: f64::NAN,
            first_radius: 0.0,
            radius_step: 0.0,
            paths: Vec::new(),
            phase: Phase::Setup,
            dissociated: None,
            manifold: None,
            scan_x: Vec::new(),
            asked: None,
        })
    }

    /// End a path as a dissociation channel when `test` says that its point has fallen
    /// apart (the method follows such a path no further).
    pub fn with_dissociation_test(mut self, test: DissociationTest) -> Self {
        self.dissociated = Some(test);
        self
    }

    /// The paths (after the scan of the first sphere).
    pub fn paths(&self) -> &[AddfPath] {
        &self.paths
    }

    /// The scaled coordinates, once the Hessian was told.
    pub fn coordinates(&self) -> Option<&ScaledCoordinates> {
        self.coordinates.as_ref()
    }

    /// Energy of the minimum.
    pub fn minimum_energy(&self) -> f64 {
        self.energy0
    }

    /// Why the run ended, once it has.
    pub fn end(&self) -> Option<AddfEnd> {
        match self.phase {
            Phase::Done(end) => Some(end),
            _ => None,
        }
    }

    /// The estimates of transition states: path number and highest point.
    pub fn transition_states(&self) -> Vec<(usize, &AddfPoint)> {
        self.paths
            .iter()
            .filter_map(|p| p.transition_state().map(|t| (p.id, t)))
            .collect()
    }

    fn scaled(&self) -> MoptResult<&ScaledCoordinates> {
        self.coordinates
            .as_ref()
            .ok_or_else(|| MoptError::calc("ADDF: the scaled coordinates are not set up"))
    }

    /// The scan points of the first sphere: mode by mode, plus then minus.
    fn scan_points(&self) -> Vec<(usize, f64, Col<f64>)> {
        let Some(c) = &self.coordinates else {
            return Vec::new();
        };
        (0..c.dimension())
            .flat_map(|k| [1.0, -1.0].map(|s| (k, s, c.along_mode(k, s, self.first_radius))))
            .collect()
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        match &self.phase {
            Phase::Setup => {
                SearchAction::Evaluate(vec![PointRequest::hessian(self.origin.clone())])
            }
            Phase::Scan => SearchAction::Evaluate(
                self.scan_x
                    .iter()
                    .flatten()
                    .map(|x| PointRequest::energy(x.clone()))
                    .collect(),
            ),
            Phase::Sphere(_) => match &self.asked {
                Some(x) => SearchAction::Evaluate(vec![PointRequest::gradient(x.clone())]),
                None => SearchAction::Finished,
            },
            Phase::Done(_) => SearchAction::Finished,
        }
    }

    /// The values of the points of [`Self::ask`]; returns what was found out.
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<Vec<AddfEvent>> {
        let asked = match self.ask() {
            SearchAction::Evaluate(points) => points,
            SearchAction::Finished => {
                return Err(MoptError::config("tell() called after the ADDF run ended"))
            }
        };
        // A path that reached a point where the surface is not defined ends there; the other
        // paths go on (the start and the first sphere must be defined).
        let undefined = values.len() == 1
            && (!values[0].energy.is_finite()
                || values[0]
                    .gradient
                    .as_ref()
                    .is_some_and(|g| g.iter().any(|v| !v.is_finite())));
        if undefined && matches!(self.phase, Phase::Sphere(_)) {
            if let Phase::Sphere(s) =
                std::mem::replace(&mut self.phase, Phase::Done(AddfEnd::SphereLimit))
            {
                let k = s.order[s.position];
                self.paths[k].state = PathState::Undefined;
                let events = vec![AddfEvent::Undefined {
                    path: self.paths[k].id,
                }];
                return self.next_on_sphere(*s, events);
            }
        }
        check_values("ADDF", &asked, &values, self.origin.nrows())?;
        match std::mem::replace(&mut self.phase, Phase::Done(AddfEnd::SphereLimit)) {
            Phase::Setup => self.setup(&values[0]),
            Phase::Scan => self.scan(&values),
            Phase::Sphere(s) => self.on_sphere(*s, &values[0]),
            Phase::Done(end) => {
                self.phase = Phase::Done(end);
                Err(MoptError::config("tell() called after the ADDF run ended"))
            }
        }
    }

    fn setup(&mut self, value: &PointValue) -> MoptResult<Vec<AddfEvent>> {
        let g = gradient_of("ADDF", value)?;
        let largest = col_max_abs(&g);
        if largest > self.settings.gradient_tolerance {
            return Err(MoptError::config(format!(
                "ADDF: the starting structure is not a minimum (largest gradient component \
                 {largest:.2e} > {:.1e}); optimize it first",
                self.settings.gradient_tolerance
            )));
        }
        let h = hessian_of("ADDF", value)?;
        let c = match &self.manifold {
            Some(m) => ScaledCoordinates::on_manifold(
                &self.origin,
                &h,
                self.removed.as_ref(),
                Arc::clone(m),
            )?,
            None => ScaledCoordinates::new(&self.origin, &h, self.removed.as_ref())?,
        };
        self.energy0 = value.energy;
        self.first_radius = c.radius_for_length(self.settings.initial_length);
        self.radius_step = c.radius_for_length(self.settings.step_length);
        let event = AddfEvent::Started {
            energy: value.energy,
            eigenvalues: c.eigenvalues().to_vec(),
            first_radius: self.first_radius,
            radius_step: self.radius_step,
        };
        self.coordinates = Some(c);
        // A direction of the first sphere without a point on the hard constraints is left
        // out of the scan (no path starts along it).
        let placed = {
            let c = self.scaled()?;
            self.scan_points()
                .iter()
                .map(|(_, _, q)| c.try_place(q))
                .collect::<MoptResult<Vec<_>>>()?
        };
        if placed.iter().all(Option::is_none) {
            return Err(MoptError::calc(
                "ADDF: no point of the first sphere lies on the constraints",
            ));
        }
        self.scan_x = placed;
        self.phase = Phase::Scan;
        Ok(vec![event])
    }

    fn scan(&mut self, values: &[PointValue]) -> MoptResult<Vec<AddfEvent>> {
        let c = self.scaled()?.clone();
        let harmonic = 0.5 * self.first_radius * self.first_radius;
        let mut candidates: Vec<(Candidate, Col<f64>, f64, Col<f64>)> = self
            .scan_points()
            .into_iter()
            .zip(&self.scan_x)
            .filter_map(|(point, x)| x.as_ref().map(|x| (point, x)))
            .zip(values)
            .map(|(((mode, sign, q), x), v)| {
                let add = v.energy - self.energy0 - harmonic;
                let candidate = Candidate {
                    mode,
                    sign,
                    eigenvalue: c.eigenvalues()[mode],
                    add,
                };
                (candidate, q, v.energy, x.clone())
            })
            .collect();
        candidates.sort_by(|a, b| a.0.add.total_cmp(&b.0.add));
        let followed = match self.settings.paths {
            0 => candidates.len(),
            k => k.min(candidates.len()),
        };
        let m = c.dimension();
        self.paths = candidates
            .iter()
            .take(followed)
            .enumerate()
            .map(|(id, (cand, q, energy, x))| AddfPath {
                id,
                mode: (cand.mode, cand.sign),
                points: vec![AddfPoint {
                    sphere: 0,
                    radius: self.first_radius,
                    q: q.clone(),
                    x: x.clone(),
                    energy: *energy,
                    add: cand.add,
                    add_ioe: cand.add,
                    evaluations: 1,
                }],
                state: PathState::Active,
                hessian: Mat::identity(m, m),
            })
            .collect();
        let event = AddfEvent::Scanned {
            candidates: candidates.iter().map(|(c, _, _, _)| *c).collect(),
            followed,
        };
        let mut events = vec![event];
        events.extend(self.start_sphere(1)?);
        Ok(events)
    }

    /// Start the minimizations of sphere `sphere` (or end the run).
    fn start_sphere(&mut self, sphere: usize) -> MoptResult<Vec<AddfEvent>> {
        let mut order: Vec<usize> = (0..self.paths.len())
            .filter(|&k| self.paths[k].state == PathState::Active)
            .collect();
        if order.is_empty() {
            self.phase = Phase::Done(AddfEnd::AllPathsEnded);
            return Ok(Vec::new());
        }
        if sphere > self.settings.max_spheres {
            for k in order {
                self.paths[k].state = PathState::SphereLimit;
            }
            self.phase = Phase::Done(AddfEnd::SphereLimit);
            return Ok(Vec::new());
        }
        // The largest ADD of the sphere before first.
        let last_add = |k: usize| self.paths[k].last().map_or(0.0, |p| p.add);
        order.sort_by(|&a, &b| last_add(a).total_cmp(&last_add(b)));
        let radius = sphere as f64 * self.radius_step;
        let minimizer = self.minimizer_for(order[0], radius)?;
        self.phase = Phase::Sphere(Box::new(OnSphere {
            sphere,
            radius,
            order,
            position: 0,
            minimizer: Box::new(minimizer),
            evaluations: 0,
            found: Vec::new(),
        }));
        self.place_asked()
    }

    /// The minimizer of path `k` on the sphere of `radius`, from its last point moved onto
    /// it along the radius.
    fn minimizer_for(&self, k: usize, radius: f64) -> MoptResult<SphereMinimizer> {
        let path = &self.paths[k];
        let start = path
            .last()
            .map(|p| p.q.clone())
            .ok_or_else(|| MoptError::calc("ADDF: a path without points"))?;
        let m = start.nrows();
        SphereMinimizer::new(
            Col::zeros(m),
            radius,
            &start,
            path.hessian.clone(),
            self.settings.sphere,
        )
    }

    fn on_sphere(&mut self, mut s: OnSphere, value: &PointValue) -> MoptResult<Vec<AddfEvent>> {
        let c = self.scaled()?.clone();
        let q = s.minimizer.point().clone();
        let at = self
            .asked
            .clone()
            .ok_or_else(|| MoptError::calc("ADDF: no point was asked for"))?;
        let mut f = value.energy;
        let mut g = c.gradient_at(&at, &gradient_of("ADDF", value)?)?;
        for (_, centre, add) in &s.found {
            let (e, ge) = elimination(&q, centre, *add);
            f += e;
            g += &ge;
        }
        s.evaluations += 1;
        match s.minimizer.tell(f, &g)? {
            SphereStatus::Continue => {
                self.phase = Phase::Sphere(Box::new(s));
                self.place_asked()
            }
            SphereStatus::Converged | SphereStatus::Exhausted => {
                self.path_done(s, &c, value.energy, f)
            }
        }
    }

    /// The minimization of the current path on its sphere is done at the point just told.
    fn path_done(
        &mut self,
        mut s: OnSphere,
        c: &ScaledCoordinates,
        energy: f64,
        with_elimination: f64,
    ) -> MoptResult<Vec<AddfEvent>> {
        let k = s.order[s.position];
        let q = s.minimizer.point().clone();
        let harmonic = 0.5 * s.radius * s.radius;
        let add = energy - self.energy0 - harmonic;
        let point = AddfPoint {
            sphere: s.sphere,
            radius: s.radius,
            q: q.clone(),
            x: c.place(&q)?,
            energy,
            add,
            add_ioe: with_elimination - self.energy0 - harmonic,
            evaluations: s.evaluations,
        };
        let mut events = Vec::new();
        let twin = s
            .found
            .iter()
            .find(|(_, other, _)| angle(&q, other) < self.settings.duplicate_angle)
            .map(|(j, _, _)| *j);
        let minimizer = std::mem::replace(
            &mut s.minimizer,
            Box::new(SphereMinimizer::new(
                Col::zeros(q.nrows()),
                s.radius,
                &q,
                Mat::identity(q.nrows(), q.nrows()),
                self.settings.sphere,
            )?),
        );
        let into = twin.map(|j| self.paths[j].id);
        let apart = self.dissociated.as_ref().is_some_and(|test| test(&point.x));
        let path = &mut self.paths[k];
        path.hessian = minimizer.into_hessian();
        match into {
            Some(into) => {
                path.state = PathState::Merged(into);
                events.push(AddfEvent::Merged {
                    path: path.id,
                    into,
                });
            }
            None => {
                path.points.push(point.clone());
                s.found.push((k, q, add));
                events.push(AddfEvent::Point {
                    path: path.id,
                    point,
                });
                if apart {
                    path.state = PathState::Dissociated;
                    events.push(AddfEvent::Dissociated { path: path.id });
                }
            }
        }
        self.next_on_sphere(s, events)
    }

    /// The next path on the sphere of `s` after its current one, or the end of the sphere
    /// (with `events` so far).
    fn next_on_sphere(
        &mut self,
        mut s: OnSphere,
        mut events: Vec<AddfEvent>,
    ) -> MoptResult<Vec<AddfEvent>> {
        s.position += 1;
        s.evaluations = 0;
        if s.position < s.order.len() {
            let next = s.order[s.position];
            s.minimizer = Box::new(self.minimizer_for(next, s.radius)?);
            self.phase = Phase::Sphere(Box::new(s));
            events.extend(self.place_asked()?);
            return Ok(events);
        }
        // The sphere is done: has a path passed a maximum of its energy?
        for path in &mut self.paths {
            if path.state != PathState::Active || path.points.len() < 3 {
                continue;
            }
            let n = path.points.len();
            let (before, last) = (&path.points[n - 2], &path.points[n - 1]);
            if last.sphere == s.sphere && last.energy < before.energy && before.sphere > 0 {
                path.state = PathState::TransitionState(n - 2);
                events.push(AddfEvent::TransitionState {
                    path: path.id,
                    point: before.clone(),
                });
            }
        }
        let active = self
            .paths
            .iter()
            .filter(|p| p.state == PathState::Active)
            .count();
        events.push(AddfEvent::SphereDone {
            sphere: s.sphere,
            active,
        });
        events.extend(self.start_sphere(s.sphere + 1)?);
        Ok(events)
    }
}

/// Length of the displacement of a point of scaled coordinates from the minimum.
pub fn displacement_length(c: &ScaledCoordinates, q: &Col<f64>) -> f64 {
    col_norm(&(c.point(q) - c.origin()))
}
