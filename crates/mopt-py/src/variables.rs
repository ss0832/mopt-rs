//! Optimization of plain variables from Python: the ask/tell [`VectorOptimizer`] and the
//! built-in objectives (`mb2d`, `expr:...`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy optimizes atomic coordinates only).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::{Col, Mat};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::analysis::resper::ResperVariable;
use mopt_rs::domain::{Bohr, Quantity};
use mopt_rs::error::MoptError;
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::step::ccqn::{CcqnAxis, CcqnSettings};
use mopt_rs::opt::trust::TrustSettings;
use mopt_rs::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};
use mopt_rs::opt::vector_constraints::{FixedVariable, VectorConstraintSet};
use mopt_rs::pes::objective::ObjectiveRegistry;
use mopt_rs::pes::registry::CalculatorOptions;

use super::{err, termination_name};

pub(crate) fn criteria_of(name: &str) -> PyResult<ConvergenceCriteria> {
    match name.to_ascii_lowercase().as_str() {
        "normal" => Ok(ConvergenceCriteria::NORMAL),
        "tight" => Ok(ConvergenceCriteria::TIGHT),
        "loose" => Ok(ConvergenceCriteria::LOOSE),
        other => Err(PyValueError::new_err(format!(
            "criteria must be normal, tight, or loose; got '{other}'"
        ))),
    }
}

/// Ask/tell optimizer of `n` plain variables (no atoms; `n` need not be a multiple of 3).
#[pyclass(name = "VectorOptimizer")]
pub struct PyVectorOptimizer {
    inner: Mutex<VectorOptimizer>,
}

impl PyVectorOptimizer {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, VectorOptimizer>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer state is poisoned"))
    }
}

#[pymethods]
impl PyVectorOptimizer {
    #[new]
    #[pyo3(signature = (
        x0,
        *,
        methods = Vec::new(),
        saddle_order = 0,
        max_iterations = 1000,
        criteria = "normal",
        exact_hessian_interval = None,
        max_trust_radius = None,
        min_trust_radius = 0.01,
        ccqn_direction = None,
        ccqn_cos_phi = None,
        mode_kill = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x0: Vec<f64>,
        methods: Vec<String>,
        saddle_order: usize,
        max_iterations: usize,
        criteria: &str,
        exact_hessian_interval: Option<usize>,
        max_trust_radius: Option<f64>,
        min_trust_radius: f64,
        ccqn_direction: Option<Vec<f64>>,
        ccqn_cos_phi: Option<f64>,
        mode_kill: bool,
    ) -> PyResult<Self> {
        let ccqn = ccqn_direction.map(|d| {
            let mut c = CcqnSettings::new(CcqnAxis::Direction(Col::from_fn(d.len(), |i| d[i])));
            if let Some(cp) = ccqn_cos_phi {
                c.cos_phi = cp;
            }
            c
        });
        let settings = VectorSettings {
            methods,
            saddle_order,
            max_iterations,
            criteria: criteria_of(criteria)?,
            trust: TrustSettings {
                max: max_trust_radius.map(Quantity::<Bohr>::new),
                min: Quantity::new(min_trust_radius),
            },
            exact_hessian_interval: exact_hessian_interval.filter(|&n| n > 0),
            ccqn,
            mode_kill: mode_kill.then(mopt_rs::opt::mode_kill::ModeKillSettings::default),
        };
        let x0 = Col::from_fn(x0.len(), |i| x0[i]);
        Ok(Self {
            inner: Mutex::new(VectorOptimizer::new(settings, x0).map_err(err)?),
        })
    }

    /// Hold constraints exactly from now on (before the first `tell`): `variables` as pairs of
    /// a variable (0-based) and its value, `resper` as triples of a `ResperMap` of points,
    /// a coordinate (0-based), and its value. The point is moved onto them.
    #[pyo3(signature = (variables = Vec::new(), resper = Vec::new()))]
    fn hold(
        &self,
        variables: Vec<(usize, f64)>,
        resper: Vec<(PyRef<'_, super::scan::PyResperMap>, usize, f64)>,
    ) -> PyResult<()> {
        let mut set = VectorConstraintSet::new();
        for (index, value) in variables {
            set.push(Arc::new(FixedVariable { index }), value)
                .map_err(err)?;
        }
        for (map, component, value) in resper {
            let c = ResperVariable::new(Arc::clone(&map.map), component).map_err(err)?;
            set.push(Arc::new(c), value).map_err(err)?;
        }
        self.lock()?.set_constraints(set).map_err(err)
    }

    /// `(x, need_hessian)` to evaluate next, or `None` when finished.
    fn ask(&self) -> PyResult<Option<(Vec<f64>, bool)>> {
        Ok(match self.lock()?.ask() {
            VectorAction::Finished(_) => None,
            VectorAction::Evaluate { x, need_hessian } => {
                Some((x.iter().copied().collect(), need_hessian))
            }
        })
    }

    /// The objective at the point of `ask()`: value, gradient, and the Hessian (flat,
    /// row-major) when it was requested or is available. Returns an iteration report.
    #[pyo3(signature = (value, gradient, hessian = None))]
    fn tell<'py>(
        &self,
        py: Python<'py>,
        value: f64,
        gradient: Vec<f64>,
        hessian: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = gradient.len();
        let g = Col::from_fn(n, |i| gradient[i]);
        let h = match hessian {
            Some(h) if h.len() == n * n => Some(Mat::from_fn(n, n, |i, j| h[i * n + j])),
            Some(h) => {
                return Err(err(MoptError::config(format!(
                    "Hessian has {} entries, expected {}",
                    h.len(),
                    n * n
                ))))
            }
            None => None,
        };
        let r = self.lock()?.tell(value, g, h).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("iteration", r.iteration)?;
        d.set_item("x", r.x.iter().copied().collect::<Vec<f64>>())?;
        d.set_item("value", r.energy)?;
        d.set_item("max_gradient", r.convergence.max_force.value())?;
        d.set_item("rms_gradient", r.convergence.rms_force.value())?;
        d.set_item("max_step", r.convergence.max_displacement.value())?;
        d.set_item("rms_step", r.convergence.rms_displacement.value())?;
        d.set_item("converged", r.convergence.converged)?;
        d.set_item("step_norm", r.step_norm)?;
        d.set_item("trust_radius", r.trust_radius)?;
        d.set_item("notes", r.notes)?;
        Ok(d)
    }

    /// `converged`, `max_iterations`, or `None` while running.
    #[getter]
    fn termination(&self) -> PyResult<Option<&'static str>> {
        Ok(self.lock()?.termination().map(termination_name))
    }

    /// Iterations performed.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        Ok(self.lock()?.iterations())
    }

    /// Current (or final) point.
    #[getter]
    fn x(&self) -> PyResult<Vec<f64>> {
        Ok(self.lock()?.current().iter().copied().collect())
    }

    /// Number of variables.
    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.lock()?.dimension())
    }
}

/// Names of the built-in objectives of plain variables.
#[pyfunction]
fn objective_names() -> Vec<String> {
    ObjectiveRegistry::builtin().names()
}

/// Evaluate a built-in objective (`mb2d`, `expr:...`) at `x`: `(value, gradient, hessian or
/// None)`, the Hessian flat and row-major; `options` are its settings (`energy_scale`, ...).
#[pyfunction]
#[pyo3(signature = (name, x, need_hessian = false, options = None))]
fn evaluate_objective(
    py: Python<'_>,
    name: &str,
    x: Vec<f64>,
    need_hessian: bool,
    options: Option<Vec<(String, String)>>,
) -> PyResult<(f64, Vec<f64>, Option<Vec<f64>>)> {
    let opts = CalculatorOptions {
        extra: options.unwrap_or_default(),
        ..CalculatorOptions::default()
    };
    let obj = ObjectiveRegistry::builtin()
        .create(name, &opts)
        .map_err(err)?;
    let xc = Col::from_fn(x.len(), |i| x[i]);
    let v = py
        .allow_threads(|| obj.evaluate(&xc, need_hessian))
        .map_err(err)?;
    let h = v.hessian.map(|h| {
        let n = h.nrows();
        (0..n * n).map(|k| h[(k / n, k % n)]).collect()
    });
    Ok((v.value, v.gradient.iter().copied().collect(), h))
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorOptimizer>()?;
    m.add_function(wrap_pyfunction!(objective_names, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_objective, m)?)?;
    Ok(())
}
