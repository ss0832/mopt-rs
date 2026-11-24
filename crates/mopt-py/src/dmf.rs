//! The direct MaxFlux method from Python: the ask/tell state machine [`DirectMaxFlux`] over
//! a path of `n` variables, the path itself (a B-spline with its arc length and the MaxFlux
//! objective at given energies) and the energy profile through the points.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `MEP/pathopt_dmf_force.py` is
//! not ported, see `path::dmf`).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::Col;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::error::MoptError;
use mopt_rs::path::dmf::action::{action, trapezoid_weights};
use mopt_rs::path::dmf::arc::ArcLength;
use mopt_rs::path::dmf::profile::EnergyProfile;
use mopt_rs::path::dmf::spline::Basis;
use mopt_rs::path::dmf::{
    DirectMaxFlux, DmfAction, DmfEnd, DmfReport, DmfSettings, PointMoves, SplinePath,
};

use super::err;

fn column(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn list(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn lists(c: &[Col<f64>]) -> Vec<Vec<f64>> {
    c.iter().map(list).collect()
}

fn columns(points: &[Vec<f64>]) -> PyResult<Vec<Col<f64>>> {
    let n = points.first().map_or(0, Vec::len);
    if points.iter().any(|p| p.len() != n) || n == 0 {
        return Err(err(MoptError::config(
            "the points of a path must all have the same number of variables (one at least)",
        )));
    }
    Ok(points.iter().map(|p| column(p)).collect())
}

fn end_name(end: DmfEnd) -> &'static str {
    match end {
        DmfEnd::Converged => "converged",
        DmfEnd::IterationLimit => "iteration_limit",
        DmfEnd::Settled => "settled",
    }
}

/// The energies of a report's profile at `samples` equal shares of the length.
fn profile_samples(profile: &EnergyProfile, reference: f64, samples: usize) -> Vec<(f64, f64)> {
    let samples = samples.max(2);
    (0..samples)
        .map(|k| {
            let share = k as f64 / (samples - 1) as f64;
            (share, profile.at(share) + reference)
        })
        .collect()
}

fn report_dict<'py>(py: Python<'py>, r: &DmfReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("iteration", r.iteration)?;
    d.set_item("objective", r.objective)?;
    d.set_item("largest_gradient", r.largest_gradient)?;
    d.set_item("shares", r.t.clone())?;
    d.set_item("points", lists(&r.points))?;
    d.set_item("energies", r.energies.clone())?;
    d.set_item("reference", r.reference)?;
    d.set_item("top_share", r.t_max)?;
    d.set_item("top_energy", r.e_max)?;
    d.set_item("highest", list(&r.highest))?;
    d.set_item("length", r.length)?;
    d.set_item("unevenness", r.unevenness)?;
    d.set_item("refitted", r.refitted)?;
    d.set_item("taken_back", r.taken_back)?;
    d.set_item("profile", profile_samples(&r.profile, r.reference, 41))?;
    Ok(d)
}

/// How the points move towards the top, from the keys of a dict (defaults for the rest).
fn moves_from(moves: bool, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Option<PointMoves>> {
    if !moves {
        return Ok(None);
    }
    let mut m = PointMoves::default();
    if let Some(d) = settings {
        for (key, value) in d.iter() {
            let key: String = key.extract()?;
            let value: f64 = value.extract()?;
            match key.as_str() {
                "largest_share" => m.largest_share = value,
                "spacing" => m.spacing = value,
                "start" => m.start = value,
                "start_sharpness" => m.start_sharpness = value,
                "settle" => m.settle = value,
                "settle_sharpness" => m.settle_sharpness = value,
                "settle_rate" => m.settle_rate = value,
                other => {
                    return Err(err(MoptError::config(format!(
                        "unknown setting `{other}` of the moves of the points (largest_share, \
                         spacing, start, start_sharpness, settle, settle_sharpness, settle_rate)"
                    ))))
                }
            }
        }
    }
    Ok(Some(m))
}

/// Ask/tell direct MaxFlux method on a path of `n` variables.
#[pyclass(name = "DirectMaxFlux")]
pub struct PyDirectMaxFlux {
    inner: Mutex<DirectMaxFlux>,
}

impl PyDirectMaxFlux {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, DirectMaxFlux>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the state of the method is poisoned"))
    }
}

#[pymethods]
impl PyDirectMaxFlux {
    #[new]
    #[pyo3(signature = (
        chain,
        *,
        points = 3,
        guards = 1,
        segments = 4,
        degree = 3,
        beta = None,
        moves = true,
        move_settings = None,
        tolerance = None,
        max_iterations = 200,
        max_step = 0.3,
        first_curvature = None,
        memory = 6,
        rise = 0.01,
        settle = 1e-3,
        patience = 8,
        refit_unevenness = 3.0,
        fixed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        chain: Vec<Vec<f64>>,
        points: usize,
        guards: usize,
        segments: usize,
        degree: usize,
        beta: Option<f64>,
        moves: bool,
        move_settings: Option<&Bound<'_, PyDict>>,
        tolerance: Option<f64>,
        max_iterations: usize,
        max_step: f64,
        first_curvature: Option<f64>,
        memory: usize,
        rise: f64,
        settle: f64,
        patience: usize,
        refit_unevenness: f64,
        fixed: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let defaults = DmfSettings::default();
        let settings = DmfSettings {
            beta: beta.unwrap_or(defaults.beta),
            degree,
            segments,
            points,
            guards,
            moves: moves_from(moves, move_settings)?,
            refit_unevenness,
            tolerance: tolerance.unwrap_or(defaults.tolerance),
            max_iterations,
            max_step,
            first_curvature: first_curvature.unwrap_or(defaults.first_curvature),
            memory,
            rise,
            settle,
            patience,
            fixed: fixed.unwrap_or_default(),
        };
        let machine = DirectMaxFlux::new(settings, &columns(&chain)?).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(machine),
        })
    }

    /// The points to evaluate next, or `None` when the run has ended. Each is a dict:
    /// `index` (among the points of the path, 0 is the first end), `share` (of the length
    /// of the path) and `x`. Every point needs an energy and a gradient.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        let points = match self.lock()?.ask().map_err(err)? {
            DmfAction::Finished(_) => return Ok(None),
            DmfAction::Evaluate(points) => points,
        };
        points
            .iter()
            .map(|p| {
                let d = PyDict::new(py);
                d.set_item("index", p.index)?;
                d.set_item("share", p.t)?;
                d.set_item("x", list(&p.x))?;
                Ok(d)
            })
            .collect::<PyResult<Vec<_>>>()
            .map(Some)
    }

    /// The energies and gradients at the points of `ask()`, in its order. Returns the
    /// report of the iteration once all points of it are known (the first call, with the
    /// two ends among the points, completes the first iteration too).
    fn tell<'py>(
        &self,
        py: Python<'py>,
        values: Vec<(f64, Vec<f64>)>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let mut machine = self.lock()?;
        let values = values
            .into_iter()
            .map(|(energy, gradient)| (energy, column(&gradient)))
            .collect();
        machine.tell(values).map_err(err)?;
        machine.report().map(|r| report_dict(py, r)).transpose()
    }

    /// The report of the iteration completed last.
    #[getter]
    fn report<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.lock()?
            .report()
            .map(|r| report_dict(py, r))
            .transpose()
    }

    /// What the run leaves: the last iteration when it converged, else the one with the
    /// lowest top of the path.
    #[getter]
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.lock()?
            .result()
            .map(|r| report_dict(py, r))
            .transpose()
    }

    /// Why the run ended (`converged`, `settled`, `iteration_limit`), or `None` while it
    /// runs.
    #[getter]
    fn end(&self) -> PyResult<Option<&'static str>> {
        Ok(self.lock()?.end().map(end_name))
    }

    /// Whether the run ended on one of its criteria (converged or settled).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(matches!(
            self.lock()?.end(),
            Some(DmfEnd::Converged | DmfEnd::Settled)
        ))
    }

    /// The estimate of the transition state: the highest point of the path that the run
    /// leaves.
    #[getter]
    fn transition_state_estimate(&self) -> PyResult<Option<Vec<f64>>> {
        Ok(self.lock()?.result().map(|r| list(&r.highest)))
    }

    /// `count` points of the path as it is now, at equal shares of its length, the ends
    /// among them.
    #[pyo3(signature = (count = 41))]
    fn path_points(&self, count: usize) -> PyResult<Vec<Vec<f64>>> {
        let machine = self.lock()?;
        points_at_shares(machine.path(), count)
    }

    /// The control points of the spline of the path as it is now.
    #[getter]
    fn control_points(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(lists(&self.lock()?.path().control))
    }

    /// Iterations completed.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        Ok(self.lock()?.report().map_or(0, |r| r.iteration + 1))
    }

    /// Number of variables of a point.
    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.lock()?.path().dimension())
    }
}

fn points_at_shares(path: &SplinePath, count: usize) -> PyResult<Vec<Vec<f64>>> {
    let arc = ArcLength::new(path).map_err(err)?;
    let count = count.max(2);
    Ok((0..count)
        .map(|k| list(&path.at(arc.parameter(k as f64 / (count - 1) as f64))))
        .collect())
}

/// A path as a B-spline of a parameter `t` in `[0, 1]` through given control points, with
/// its arc length and the MaxFlux objective.
#[pyclass(name = "DmfPath")]
pub struct PyDmfPath {
    path: SplinePath,
}

impl PyDmfPath {
    fn arc(&self) -> PyResult<ArcLength<'_>> {
        ArcLength::new(&self.path).map_err(err)
    }
}

#[pymethods]
impl PyDmfPath {
    /// The spline of `degree` with `segments` segments that passes through the first and
    /// the last point of `chain` and is fitted to the others by least squares at equal
    /// shares of their chord length.
    #[new]
    #[pyo3(signature = (chain, *, degree = 3, segments = 4))]
    fn new(chain: Vec<Vec<f64>>, degree: usize, segments: usize) -> PyResult<Self> {
        let path = SplinePath::through(&columns(&chain)?, degree, segments).map_err(err)?;
        Ok(Self { path })
    }

    /// The spline of `degree` with the control points `control` (the first and the last
    /// are the ends of the path).
    #[staticmethod]
    #[pyo3(signature = (control, *, degree = 3))]
    fn from_control(control: Vec<Vec<f64>>, degree: usize) -> PyResult<Self> {
        let control = columns(&control)?;
        if control.len() <= degree {
            return Err(err(MoptError::config(format!(
                "a spline of degree {degree} needs {} control points at least",
                degree + 1
            ))));
        }
        let basis = Basis::new(degree, control.len() - degree).map_err(err)?;
        Ok(Self {
            path: SplinePath { basis, control },
        })
    }

    /// The control points.
    #[getter]
    fn control(&self) -> Vec<Vec<f64>> {
        lists(&self.path.control)
    }

    /// The point at the parameter `t`.
    fn at(&self, t: f64) -> Vec<f64> {
        list(&self.path.at(t))
    }

    /// `dx/dt` at the parameter `t`.
    fn velocity(&self, t: f64) -> Vec<f64> {
        list(&self.path.velocity(t))
    }

    /// The length of the path.
    #[getter]
    fn length(&self) -> PyResult<f64> {
        Ok(self.arc()?.length())
    }

    /// The parameter where `share` of the length is reached.
    fn parameter(&self, share: f64) -> PyResult<f64> {
        Ok(self.arc()?.parameter(share))
    }

    /// The points at the shares `shares` of the length.
    fn points(&self, shares: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let arc = self.arc()?;
        Ok(shares
            .iter()
            .map(|s| list(&self.path.at(arc.parameter(*s))))
            .collect())
    }

    /// Largest speed of the parameter over its smallest.
    #[getter]
    fn unevenness(&self) -> PyResult<f64> {
        Ok(self.arc()?.unevenness())
    }

    /// The MaxFlux objective `ln(integral exp(beta E) ds) / beta` from the energies and
    /// gradients at the points that are at the shares `shares` of the length (ascending,
    /// from 0 to 1, trapezoid weights). Returns a dict: `value`, `gradient` (one row per
    /// inner control point), `weights` (the share of every point in the sum), `slopes`
    /// (`dE/dtau`) and `length`.
    fn action<'py>(
        &self,
        py: Python<'py>,
        shares: Vec<f64>,
        energies: Vec<f64>,
        gradients: Vec<Vec<f64>>,
        beta: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let arc = self.arc()?;
        let t: Vec<f64> = shares.iter().map(|s| arc.parameter(*s)).collect();
        let weights = trapezoid_weights(&shares);
        let gradients = columns(&gradients)?;
        let found = action(
            &self.path, &arc, &shares, &t, &weights, &energies, &gradients, beta,
        )
        .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("value", found.value)?;
        d.set_item("gradient", lists(&found.gradient))?;
        d.set_item("weights", found.shares)?;
        d.set_item("slopes", found.slopes)?;
        d.set_item("length", found.length)?;
        Ok(d)
    }

    /// Number of variables of a point.
    #[getter]
    fn dimension(&self) -> usize {
        self.path.dimension()
    }
}

/// The energy along a path through the energies `energies` with the slopes `slopes`
/// (`dE/dt`) at the points `shares` (ascending), as cubic Hermite pieces: a dict with
/// `top_share` and `top_energy` (its highest point, inside a piece or at a point) and
/// `profile` (`samples` pairs of share and energy).
#[pyfunction]
#[pyo3(signature = (shares, energies, slopes, samples = 41))]
fn dmf_profile<'py>(
    py: Python<'py>,
    shares: Vec<f64>,
    energies: Vec<f64>,
    slopes: Vec<f64>,
    samples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let profile = EnergyProfile::new(&shares, &energies, &slopes).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("top_share", profile.t_max)?;
    d.set_item("top_energy", profile.e_max)?;
    d.set_item("profile", profile_samples(&profile, 0.0, samples))?;
    Ok(d)
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDirectMaxFlux>()?;
    m.add_class::<PyDmfPath>()?;
    m.add_function(wrap_pyfunction!(dmf_profile, m)?)?;
    Ok(())
}
