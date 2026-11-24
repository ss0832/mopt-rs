//! The searches of `ieipmain` that evaluate a few points at a time, from Python: the dimer
//! method, the spring pair method, the two-point scaled hypersphere search, the growing
//! Newton trajectory, and the anharmonic downward distortion following. Each is an ask/tell
//! state machine: `ask()` gives the points to evaluate (dicts with `x`, `gradient`, `hessian`:
//! what is wanted there) or `None` at the end, and `tell()` takes their values in the same
//! order as `(energy, gradient or None, Hessian (flat, row-major) or None)`.
//!
//! MultiOptPy: `OtherMethod/dimer.py`, `spring_pair_method.py`, `twopshs.py`, `newton_traj.py`,
//! `addf.py`.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::Mat;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::irc::SphereSettings;
use mopt_rs::path::dimer::{Dimer, DimerSettings};
use mopt_rs::path::newton_trajectory::{GrowingNewtonTrajectory, NtDirection, NtSettings};
use mopt_rs::path::points::{PointValue, SearchAction};
use mopt_rs::path::spring_pair::{SpringPair, SpringPairSettings};
use mopt_rs::path::twopshs::{TwoPointShs, TwoPshsSettings};
use mopt_rs::workflow::explore::addf::{Addf, AddfEvent, AddfSettings};

use super::err;
use super::ieip::{column, list};
use super::variables::criteria_of;

/// One value of `tell`.
pub(crate) type PointInput = (f64, Option<Vec<f64>>, Option<Vec<f64>>);

pub(crate) fn values(n: usize, input: Vec<PointInput>) -> PyResult<Vec<PointValue>> {
    input
        .into_iter()
        .map(|(energy, gradient, hessian)| {
            let hessian = match hessian {
                Some(h) if h.len() == n * n => Some(Mat::from_fn(n, n, |i, j| h[i * n + j])),
                Some(h) => {
                    return Err(PyValueError::new_err(format!(
                        "Hessian has {} entries, expected {}",
                        h.len(),
                        n * n
                    )))
                }
                None => None,
            };
            Ok(PointValue {
                energy,
                gradient: gradient.as_deref().map(column),
                hessian,
            })
        })
        .collect()
}

pub(crate) fn points<'py>(
    py: Python<'py>,
    action: SearchAction,
) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
    let SearchAction::Evaluate(points) = action else {
        return Ok(None);
    };
    points
        .iter()
        .map(|p| {
            let d = PyDict::new(py);
            d.set_item("x", list(&p.x))?;
            d.set_item("gradient", p.gradient)?;
            d.set_item("hessian", p.hessian)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

fn locked<T>(m: &Mutex<T>) -> PyResult<std::sync::MutexGuard<'_, T>> {
    m.lock()
        .map_err(|_| PyRuntimeError::new_err("the state of the search is poisoned"))
}

/// Ask/tell dimer method from one point along an axis.
#[pyclass(name = "Dimer")]
pub struct PyDimer {
    inner: Mutex<Dimer>,
    n: usize,
}

#[pymethods]
impl PyDimer {
    #[new]
    #[pyo3(signature = (start, axis, *, separation = 1e-4, trial_angle = std::f64::consts::PI / 32.0,
        max_iterations = 1000, max_rotations = 8, rotation_tolerance = 1.0_f64.to_radians(),
        max_step = 0.1, criteria = "normal", frozen = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        start: Vec<f64>,
        axis: Vec<f64>,
        separation: f64,
        trial_angle: f64,
        max_iterations: usize,
        max_rotations: usize,
        rotation_tolerance: f64,
        max_step: f64,
        criteria: &str,
        frozen: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let settings = DimerSettings {
            separation,
            trial_angle,
            max_iterations,
            max_rotations,
            rotation_tolerance,
            max_step,
            criteria: criteria_of(criteria)?,
            ..DimerSettings::default()
        };
        let d = Dimer::new(settings, &column(&start), &column(&axis))
            .and_then(|d| d.with_frozen(frozen.unwrap_or_default()))
            .map_err(err)?;
        Ok(Self {
            inner: Mutex::new(d),
            n: start.len(),
        })
    }

    /// The points to evaluate next (the centre and the end point, or a trial end point), or
    /// `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the points of `ask()`; returns a report.
    fn tell<'py>(&self, py: Python<'py>, input: Vec<PointInput>) -> PyResult<Bound<'py, PyDict>> {
        let r = locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("iteration", r.iteration)?;
        d.set_item("energy", r.energy)?;
        d.set_item("curvature", r.curvature)?;
        d.set_item("max_force", r.max_force)?;
        d.set_item("rms_force", r.rms_force)?;
        d.set_item("rotation", r.rotation)?;
        d.set_item("step", r.step)?;
        d.set_item("center", list(&r.center))?;
        d.set_item("axis", list(&r.axis))?;
        d.set_item("end", r.end.map(|e| format!("{e:?}").to_ascii_lowercase()))?;
        Ok(d)
    }

    /// The centre (after the end: the saddle point estimate).
    #[getter]
    fn center(&self) -> PyResult<Vec<f64>> {
        Ok(list(locked(&self.inner)?.center()))
    }

    /// The axis of the dimer.
    #[getter]
    fn axis(&self) -> PyResult<Vec<f64>> {
        Ok(list(locked(&self.inner)?.axis()))
    }

    /// Curvature along the axis.
    #[getter]
    fn curvature(&self) -> PyResult<f64> {
        Ok(locked(&self.inner)?.curvature())
    }

    /// Whether the search converged (`None` while it runs).
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        Ok(locked(&self.inner)?
            .end()
            .map(|e| e == mopt_rs::path::dimer::DimerEnd::Converged))
    }
}

/// Ask/tell spring pair method from one point along a direction.
#[pyclass(name = "SpringPair")]
pub struct PySpringPair {
    inner: Mutex<SpringPair>,
    n: usize,
}

#[pymethods]
impl PySpringPair {
    #[new]
    #[pyo3(signature = (start, direction, *, spring_length = 0.1, offset = 0.1, drift_rate = 0.5,
        spring_rate = 1.0, climb_rate = 5.0, drift_tolerance = 2e-3, drift_steps = 200,
        max_cycles = 999, max_step = 0.1, criteria = "normal"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        start: Vec<f64>,
        direction: Vec<f64>,
        spring_length: f64,
        offset: f64,
        drift_rate: f64,
        spring_rate: f64,
        climb_rate: f64,
        drift_tolerance: f64,
        drift_steps: usize,
        max_cycles: usize,
        max_step: f64,
        criteria: &str,
    ) -> PyResult<Self> {
        let settings = SpringPairSettings {
            spring_length,
            offset,
            drift_rate,
            spring_rate,
            climb_rate,
            drift_tolerance,
            drift_steps,
            max_cycles,
            max_step,
            criteria: criteria_of(criteria)?,
            align: false,
        };
        let s = SpringPair::new(settings, &column(&start), &column(&direction)).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(s),
            n: start.len(),
        })
    }

    /// The two points to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the two points of `ask()`; returns a report.
    fn tell<'py>(&self, py: Python<'py>, input: Vec<PointInput>) -> PyResult<Bound<'py, PyDict>> {
        let r = locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("cycle", r.cycle)?;
        d.set_item("drift_step", r.drift_step)?;
        d.set_item("energies", r.energies.to_vec())?;
        d.set_item("max_force", r.max_force.to_vec())?;
        d.set_item("length", r.length)?;
        d.set_item("end", r.end.map(|e| format!("{e:?}").to_ascii_lowercase()))?;
        Ok(d)
    }

    /// The point near the saddle at the end (the one with the smaller force).
    #[getter]
    fn saddle_estimate(&self) -> PyResult<Option<Vec<f64>>> {
        Ok(locked(&self.inner)?.saddle_estimate().map(|(_, x)| list(x)))
    }

    /// Whether the search converged (`None` while it runs).
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        Ok(locked(&self.inner)?
            .end()
            .map(|e| e == mopt_rs::path::spring_pair::SpringPairEnd::Converged))
    }
}

fn sphere(iterations: usize, gradient: f64) -> SphereSettings {
    SphereSettings {
        max_iterations: iterations,
        gradient_tolerance: gradient,
        ..SphereSettings::default()
    }
}

/// Ask/tell two-point scaled hypersphere search from `other` to the minimum `center`.
#[pyclass(name = "TwoPointShs")]
pub struct PyTwoPointShs {
    inner: Mutex<TwoPointShs>,
    n: usize,
}

#[pymethods]
impl PyTwoPointShs {
    #[new]
    #[pyo3(signature = (center, other, *, step = 0.05, max_spheres = 300, sphere_iterations = 30,
        sphere_gradient = 1e-4))]
    fn new(
        center: Vec<f64>,
        other: Vec<f64>,
        step: f64,
        max_spheres: usize,
        sphere_iterations: usize,
        sphere_gradient: f64,
    ) -> PyResult<Self> {
        let settings = TwoPshsSettings {
            step,
            max_spheres,
            sphere: sphere(sphere_iterations, sphere_gradient),
        };
        let s = TwoPointShs::new(settings, &column(&center), &column(&other), None).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(s),
            n: center.len(),
        })
    }

    /// The points to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the points of `ask()`; returns the point of the path that was completed
    /// (`(sphere, radius, x, energy)`), if one was.
    fn tell(&self, input: Vec<PointInput>) -> PyResult<Option<(usize, f64, Vec<f64>, f64)>> {
        let point = locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?;
        Ok(point.map(|p| (p.sphere, p.radius, list(&p.x), p.energy)))
    }

    /// The path so far: `(sphere, radius, x, energy)` per point.
    #[getter]
    fn path(&self) -> PyResult<Vec<(usize, f64, Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .path()
            .iter()
            .map(|p| (p.sphere, p.radius, list(&p.x), p.energy))
            .collect())
    }

    /// The highest point of the path within (the estimate of the transition state).
    #[getter]
    fn transition_state_estimate(&self) -> PyResult<Option<(Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .transition_state_estimate()
            .map(|p| (list(&p.x), p.energy)))
    }
}

/// Ask/tell growing Newton trajectory from a minimum towards a final point (or along a fixed
/// direction).
#[pyclass(name = "NewtonTrajectory")]
pub struct PyNewtonTrajectory {
    inner: Mutex<GrowingNewtonTrajectory>,
    n: usize,
}

#[pymethods]
impl PyNewtonTrajectory {
    #[new]
    #[pyo3(signature = (start, *, final_point = None, direction = None, step_length = 0.5,
        rms_tolerance = 1.7e-3, corrector_iterations = 25, max_iterations = 999))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        start: Vec<f64>,
        final_point: Option<Vec<f64>>,
        direction: Option<Vec<f64>>,
        step_length: f64,
        rms_tolerance: f64,
        corrector_iterations: usize,
        max_iterations: usize,
    ) -> PyResult<Self> {
        let source = match (final_point, direction) {
            (Some(f), None) => NtDirection::Toward(column(&f)),
            (None, Some(d)) => NtDirection::Fixed(column(&d)),
            _ => {
                return Err(PyValueError::new_err(
                    "give either final_point or direction",
                ))
            }
        };
        let settings = NtSettings {
            step_length,
            rms_tolerance,
            corrector_iterations,
            max_iterations,
            align: false,
            ..NtSettings::default()
        };
        let nt = GrowingNewtonTrajectory::new(settings, &column(&start), source).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(nt),
            n: start.len(),
        })
    }

    /// The point to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The value at the point of `ask()`; returns a report.
    fn tell<'py>(&self, py: Python<'py>, input: Vec<PointInput>) -> PyResult<Bound<'py, PyDict>> {
        let r = locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("iteration", r.iteration)?;
        d.set_item("energy", r.energy)?;
        d.set_item("projected_rms", r.projected_rms)?;
        d.set_item("added", r.added.is_some())?;
        d.set_item("passed", r.passed)?;
        d.set_item("end", r.end.map(|e| format!("{e:?}").to_ascii_lowercase()))?;
        Ok(d)
    }

    /// The points of the trajectory: `(x, energy)`.
    #[getter]
    fn points(&self) -> PyResult<Vec<(Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .points()
            .iter()
            .map(|p| (list(&p.x), p.energy))
            .collect())
    }

    /// The point next to the transition state, once it was passed.
    #[getter]
    fn transition_state_estimate(&self) -> PyResult<Option<(Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .transition_state_estimate()
            .map(|p| (list(&p.x), p.energy)))
    }
}

/// Ask/tell anharmonic downward distortion following from a minimum.
#[pyclass(name = "Addf")]
pub struct PyAddf {
    inner: Mutex<Addf>,
    n: usize,
}

#[pymethods]
impl PyAddf {
    #[new]
    #[pyo3(signature = (start, *, initial_length = 0.0567, step_length = 0.189, max_spheres = 300,
        paths = 5, sphere_iterations = 30, sphere_gradient = 1e-4, duplicate_angle = 0.01,
        gradient_tolerance = 1e-3))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        start: Vec<f64>,
        initial_length: f64,
        step_length: f64,
        max_spheres: usize,
        paths: usize,
        sphere_iterations: usize,
        sphere_gradient: f64,
        duplicate_angle: f64,
        gradient_tolerance: f64,
    ) -> PyResult<Self> {
        let settings = AddfSettings {
            initial_length,
            step_length,
            max_spheres,
            paths,
            sphere: sphere(sphere_iterations, sphere_gradient),
            duplicate_angle,
            gradient_tolerance,
        };
        let a = Addf::new(settings, &column(&start), None).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(a),
            n: start.len(),
        })
    }

    /// The points to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the points of `ask()`; returns what was found out, as short texts.
    fn tell(&self, input: Vec<PointInput>) -> PyResult<Vec<String>> {
        let events = locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?;
        Ok(events
            .iter()
            .map(|e| match e {
                AddfEvent::Started { .. } => "started".to_string(),
                AddfEvent::Scanned { followed, .. } => format!("scanned: {followed} paths"),
                AddfEvent::Point { path, point } => format!("path {path}: sphere {}", point.sphere),
                AddfEvent::Merged { path, into } => format!("path {path} merged into {into}"),
                AddfEvent::Dissociated { path } => format!("path {path} dissociated"),
                AddfEvent::Undefined { path } => format!("path {path}: the surface is not defined"),
                AddfEvent::TransitionState { path, .. } => format!("path {path}: transition state"),
                AddfEvent::SphereDone { sphere, .. } => format!("sphere {sphere} done"),
            })
            .collect())
    }

    /// The estimates of transition states: `(path, x, energy)`.
    #[getter]
    fn transition_states(&self) -> PyResult<Vec<(usize, Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .transition_states()
            .into_iter()
            .map(|(k, p)| (k, list(&p.x), p.energy))
            .collect())
    }

    /// The paths: per path its points as `(sphere, radius, x, energy, add)`.
    #[getter]
    fn paths(&self) -> PyResult<Vec<Vec<(usize, f64, Vec<f64>, f64, f64)>>> {
        Ok(locked(&self.inner)?
            .paths()
            .iter()
            .map(|p| {
                p.points
                    .iter()
                    .map(|q| (q.sphere, q.radius, list(&q.x), q.energy, q.add))
                    .collect()
            })
            .collect())
    }

    /// Whether the run has ended.
    #[getter]
    fn finished(&self) -> PyResult<bool> {
        Ok(locked(&self.inner)?.end().is_some())
    }
}

/// Add the classes of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDimer>()?;
    m.add_class::<PySpringPair>()?;
    m.add_class::<PyTwoPointShs>()?;
    m.add_class::<PyNewtonTrajectory>()?;
    m.add_class::<PyAddf>()?;
    Ok(())
}
