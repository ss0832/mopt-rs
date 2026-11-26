//! One optimization method used on its own from Python: any `-opt` method string (step
//! generator, enhancers, Hessian updater) as an object that proposes steps for points supplied
//! by the caller.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `Optimizer/` classes are driven by
//! `optimizer.py:CalculateMoveVector` only).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::{Col, Mat};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use mopt_rs::opt::stepper::Stepper;

use super::err;

/// A method (as `-opt`) proposing steps on `n` plain variables.
#[pyclass(name = "StepMethod")]
pub struct PyStepMethod {
    inner: Mutex<Stepper>,
    n: usize,
}

impl PyStepMethod {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Stepper>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("method state is poisoned"))
    }
}

#[pymethods]
impl PyStepMethod {
    #[new]
    #[pyo3(signature = (method, n, *, saddle_order = 0, max_step = None, update_projection = true))]
    fn new(
        method: &str,
        n: usize,
        saddle_order: usize,
        max_step: Option<f64>,
        update_projection: bool,
    ) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("no variables"));
        }
        let mut stepper = Stepper::new(method, n, saddle_order)
            .map_err(err)?
            .with_project_updates(update_projection);
        if let Some(max) = max_step {
            if !(max.is_finite() && max > 0.0) {
                return Err(PyValueError::new_err("max_step must be positive"));
            }
            stepper = stepper.with_max_step(max);
        }
        Ok(Self {
            inner: Mutex::new(stepper),
            n,
        })
    }

    /// Step proposed at `x` with the given value and gradient of the function. `rigid_basis`:
    /// directions along which the function does not change at `x` (the rigid motions of
    /// variables that describe a structure, each of `n` components; orthonormalized here),
    /// taken out of the Hessian after its update (unless `update_projection=False`) and
    /// given to the step generator; none by default.
    #[pyo3(signature = (x, value, gradient, rigid_basis = None))]
    fn propose(
        &self,
        x: Vec<f64>,
        value: f64,
        gradient: Vec<f64>,
        rigid_basis: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        if x.len() != self.n || gradient.len() != self.n {
            return Err(PyValueError::new_err(format!(
                "x and gradient must have {} components",
                self.n
            )));
        }
        let basis = match rigid_basis {
            None => None,
            Some(vectors) => {
                if vectors.iter().any(|v| v.len() != self.n) {
                    return Err(PyValueError::new_err(format!(
                        "every vector of rigid_basis must have {} components",
                        self.n
                    )));
                }
                let cols: Vec<Col<f64>> = vectors
                    .iter()
                    .map(|v| Col::from_fn(self.n, |i| v[i]))
                    .collect();
                Some(mopt_rs::math::linalg::orthonormalize_columns(
                    self.n, &cols, 1e-10,
                ))
            }
        };
        let x = Col::from_fn(self.n, |i| x[i]);
        let g = Col::from_fn(self.n, |i| gradient[i]);
        let mut stepper = self.lock()?;
        stepper.set_rigid_basis(basis).map_err(err)?;
        let step = stepper.propose(&x, value, &g).map_err(err)?;
        Ok(step.iter().copied().collect())
    }

    /// Replace the Hessian (flat row-major `n x n`).
    fn set_hessian(&self, hessian: Vec<f64>) -> PyResult<()> {
        let n = self.n;
        if hessian.len() != n * n {
            return Err(PyValueError::new_err(format!(
                "the Hessian must have {} entries",
                n * n
            )));
        }
        let h = Mat::from_fn(n, n, |i, j| hessian[i * n + j]);
        self.lock()?.set_hessian(h).map_err(err)
    }

    /// Current Hessian (flat row-major `n x n`).
    #[getter]
    fn hessian(&self) -> PyResult<Vec<f64>> {
        let guard = self.lock()?;
        let h = guard.hessian();
        let n = self.n;
        Ok((0..n * n).map(|k| h[(k / n, k % n)]).collect())
    }

    /// Name of the step generator.
    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok(self.lock()?.name().to_string())
    }

    /// Names of the step enhancers in application order.
    #[getter]
    fn enhancers(&self) -> PyResult<Vec<String>> {
        Ok(self.lock()?.enhancer_names())
    }

    /// Number of proposals made.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        Ok(self.lock()?.iterations())
    }

    /// Forget the history (the Hessian is kept).
    fn reset(&self) -> PyResult<()> {
        self.lock()?.reset();
        Ok(())
    }
}

/// Register the classes of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStepMethod>()?;
    Ok(())
}
