//! The string methods of `ieipmain` and the registry of two-point methods from Python: the
//! growing string (`GrowingString`, double- or single-ended), the freezing string
//! (`FreezingString`), DS-AFIR (`DsAfir`) as ask/tell state machines whose settings are the
//! `KEY=VALUE` keys of the registry (a dict), and `pair_methods()` / `run_pair_method()` for
//! every method of the registry on a built-in objective of plain variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::core_infra::log::Logger;
use mopt_rs::path::dsafir::DoubleSphereAfir;
use mopt_rs::path::fsm::FreezingString;
use mopt_rs::path::gsm::{Driving, DrivingCoordinate, GrowingString};
use mopt_rs::path::string_nodes::PointGeometry;
use mopt_rs::pes::objective::ObjectiveRegistry;
use mopt_rs::pes::registry::CalculatorOptions;
use mopt_rs::workflow::pair_search::registry_strings::{
    dsafir_settings, fsm_settings, gsm_settings,
};
use mopt_rs::workflow::pair_search::{
    PairInput, PairMethodRegistry, PairSettings, SearchRun, VariableSurface,
};

use super::err;
use super::ieip::{column, list};
use super::searches::{points, values, PointInput};

fn settings_of(map: Option<BTreeMap<String, String>>) -> PairSettings {
    PairSettings::new(map.unwrap_or_default().into_iter().collect())
}

fn plain(points: &[Vec<f64>]) -> PairInput {
    PairInput {
        points: points.iter().map(|p| column(p)).collect(),
        ..PairInput::default()
    }
}

fn locked<T>(m: &Mutex<T>) -> PyResult<std::sync::MutexGuard<'_, T>> {
    m.lock()
        .map_err(|_| PyRuntimeError::new_err("the state of the search is poisoned"))
}

fn driving_of(texts: &[String]) -> PyResult<Vec<DrivingCoordinate>> {
    texts
        .iter()
        .map(|t| DrivingCoordinate::parse(t).map_err(err))
        .collect()
}

/// Ask/tell growing string on points of plain variables: double-ended between `first` and
/// `second`, or single-ended from `first` along `driving` (`"VAR k value"`, counted from 1).
#[pyclass(name = "GrowingString")]
pub struct PyGrowingString {
    inner: Mutex<GrowingString>,
    n: usize,
}

#[pymethods]
impl PyGrowingString {
    #[new]
    #[pyo3(signature = (first, second = None, *, driving = None, settings = None))]
    fn new(
        first: Vec<f64>,
        second: Option<Vec<f64>>,
        driving: Option<Vec<String>>,
        settings: Option<BTreeMap<String, String>>,
    ) -> PyResult<Self> {
        let s = settings_of(settings);
        let input = plain(std::slice::from_ref(&first));
        let machine = match (second, driving) {
            (Some(b), None) => {
                let gs = gsm_settings(&s, &input, false).map_err(err)?;
                GrowingString::double_ended(
                    gs,
                    &column(&first),
                    &column(&b),
                    PointGeometry::default(),
                )
            }
            (None, Some(texts)) => {
                let gs = gsm_settings(&s, &input, true).map_err(err)?;
                let d = Driving {
                    coordinates: driving_of(&texts)?,
                    elements: None,
                };
                GrowingString::single_ended(gs, &column(&first), d, PointGeometry::default())
            }
            _ => {
                return Err(PyValueError::new_err(
                    "give either a second point (DE-GSM) or driving coordinates (SE-GSM)",
                ))
            }
        }
        .map_err(err)?;
        s.finish("gsm", &[]).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(machine),
            n: first.len(),
        })
    }

    /// The points to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the points of `ask()`; returns the iteration.
    fn tell(&self, input: Vec<PointInput>) -> PyResult<usize> {
        let r = locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?;
        Ok(r.iteration)
    }

    /// The estimate of the transition state `(x, energy)`, if there is one.
    #[getter]
    fn transition_state_estimate(&self) -> PyResult<Option<(Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .transition_state_estimate()
            .map(|(x, e)| (list(&x), e)))
    }

    /// Whether the climbing node met the thresholds (`None` while it runs).
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        Ok(locked(&self.inner)?
            .end()
            .map(|e| e == mopt_rs::path::gsm::GsmEnd::Converged))
    }
}

/// Ask/tell freezing string between two points of plain variables.
#[pyclass(name = "FreezingString")]
pub struct PyFreezingString {
    inner: Mutex<FreezingString>,
    n: usize,
}

#[pymethods]
impl PyFreezingString {
    #[new]
    #[pyo3(signature = (first, second, *, settings = None))]
    fn new(
        first: Vec<f64>,
        second: Vec<f64>,
        settings: Option<BTreeMap<String, String>>,
    ) -> PyResult<Self> {
        let s = settings_of(settings);
        let fs = fsm_settings(&s, &plain(std::slice::from_ref(&first))).map_err(err)?;
        let m = FreezingString::new(
            fs,
            &column(&first),
            &column(&second),
            PointGeometry::default(),
        )
        .map_err(err)?;
        Ok(Self {
            inner: Mutex::new(m),
            n: first.len(),
        })
    }

    /// The points to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the points of `ask()`; returns whether a node was frozen.
    fn tell(&self, input: Vec<PointInput>) -> PyResult<bool> {
        Ok(locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?
            .frozen)
    }

    /// The highest node `(x, energy)`, if there is one.
    #[getter]
    fn transition_state_estimate(&self) -> PyResult<Option<(Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .transition_state_estimate()
            .map(|(x, e)| (list(&x), e)))
    }

    /// The path `[(x, energy)]`.
    #[getter]
    fn path(&self) -> PyResult<Vec<(Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .path()
            .into_iter()
            .map(|(x, e)| (list(&x), e))
            .collect())
    }
}

/// Ask/tell DS-AFIR between two points of plain variables (`settings`: the keys of the
/// registry, `variant` `paper` or `default` among them).
#[pyclass(name = "DsAfir")]
pub struct PyDsAfir {
    inner: Mutex<DoubleSphereAfir>,
    n: usize,
}

#[pymethods]
impl PyDsAfir {
    #[new]
    #[pyo3(signature = (first, second, *, settings = None))]
    fn new(
        first: Vec<f64>,
        second: Vec<f64>,
        settings: Option<BTreeMap<String, String>>,
    ) -> PyResult<Self> {
        let s = settings_of(settings);
        let ds = dsafir_settings(&s, &plain(std::slice::from_ref(&first))).map_err(err)?;
        let m = DoubleSphereAfir::new(
            ds,
            &column(&first),
            &column(&second),
            PointGeometry::default(),
        )
        .map_err(err)?;
        Ok(Self {
            inner: Mutex::new(m),
            n: first.len(),
        })
    }

    /// The points to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the points of `ask()`; returns the distance of the ends.
    fn tell(&self, input: Vec<PointInput>) -> PyResult<f64> {
        Ok(locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?
            .separation)
    }

    /// The highest point of the (relaxed) path `(x, energy)`, if there is one.
    #[getter]
    fn transition_state_estimate(&self) -> PyResult<Option<(Vec<f64>, f64)>> {
        Ok(locked(&self.inner)?
            .transition_state_estimate()
            .map(|(x, e)| (list(&x), e)))
    }

    /// Whether the ends met (`None` while it runs).
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        Ok(locked(&self.inner)?
            .end()
            .map(|e| e == mopt_rs::path::dsafir::DsAfirEnd::Met))
    }
}

/// The methods of the registry: `(name, ends, summary, [(key, meaning)])`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn pair_methods() -> Vec<(String, String, String, Vec<(String, String)>)> {
    PairMethodRegistry::default()
        .methods()
        .iter()
        .map(|m| {
            (
                m.name().to_string(),
                m.ends().text().to_string(),
                m.summary().to_string(),
                m.keys()
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                    .collect(),
            )
        })
        .collect()
}

/// Run the method `name` of the registry on the built-in objective `objective` from
/// `points` (one or two); returns a dict with `converged`, `end`, `transition_states`
/// (`[(x, energy)]`, the lowest first), `path`, and `gradients`, `hessians`, `energies`.
#[pyfunction]
#[pyo3(signature = (name, objective, points, *, settings = None, driving = None, directory = "."))]
fn run_pair_method<'py>(
    py: Python<'py>,
    name: &str,
    objective: &str,
    points: Vec<Vec<f64>>,
    settings: Option<BTreeMap<String, String>>,
    driving: Option<Vec<String>>,
    directory: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let obj = ObjectiveRegistry::builtin()
        .create(objective, &CalculatorOptions::default())
        .map_err(err)?;
    let surface = VariableSurface(obj.as_ref());
    let mut input = plain(&points);
    input.driving = driving_of(&driving.unwrap_or_default())?;
    let dir = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let log = Logger::silent();
    let run = SearchRun {
        save_steps: true,
        dir: &dir,
        stems: ["first".to_string(), "second".to_string()],
        log: &log,
        figures: false,
        commands: &mopt_rs::cli::pair_settings::COMMAND_PARSERS,
    };
    let r = PairMethodRegistry::default()
        .run(name, &input, &settings_of(settings), &surface, &run)
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("method", r.method)?;
    d.set_item("converged", r.converged)?;
    d.set_item("end", r.end)?;
    let ts: Vec<(Vec<f64>, f64)> = r
        .transition_states
        .iter()
        .map(|(x, e)| (list(x), *e))
        .collect();
    d.set_item("transition_states", ts)?;
    let path: Option<Vec<(Vec<f64>, f64)>> = r
        .path
        .map(|p| p.iter().map(|(x, e)| (list(x), *e)).collect());
    d.set_item("path", path)?;
    d.set_item("gradients", r.counts.gradients)?;
    d.set_item("hessians", r.counts.hessians)?;
    d.set_item("energies", r.counts.energies)?;
    Ok(d)
}

/// Add the classes and functions to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrowingString>()?;
    m.add_class::<PyFreezingString>()?;
    m.add_class::<PyDsAfir>()?;
    m.add_function(wrap_pyfunction!(pair_methods, m)?)?;
    m.add_function(wrap_pyfunction!(run_pair_method, m)?)?;
    Ok(())
}
