//! The binary-image transition state search from Python: the ask/tell state machine
//! [`Bitss`] over two points of `n` variables and the BITSS function alone.
//!
//! MultiOptPy: `ModelFunction/binary_image_ts_search_model_function.py:BITSSModelFunction`.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::{Col, Mat};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::domain::{Bohr, Quantity};
use mopt_rs::error::MoptError;
use mopt_rs::opt::trust::TrustSettings;
use mopt_rs::opt::vector::VectorSettings;
use mopt_rs::path::bitss::{
    penalized, Bitss, BitssAction, BitssEnd, BitssSettings, Coefficients, PointRole, PointValue,
};

use super::err;
use super::variables::criteria_of;

fn column(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn list(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn end_name(end: BitssEnd) -> &'static str {
    match end {
        BitssEnd::MidpointGradient => "midpoint_gradient",
        BitssEnd::Separation => "separation",
        BitssEnd::MidpointShift => "midpoint_shift",
        BitssEnd::NoBarrier => "no_barrier",
        BitssEnd::StageLimit => "stage_limit",
        BitssEnd::IterationLimit => "iteration_limit",
    }
}

/// Ask/tell binary-image transition state search between two points of `n` variables.
#[pyclass(name = "Bitss")]
pub struct PyBitss {
    inner: Mutex<Bitss>,
}

impl PyBitss {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Bitss>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the state of the search is poisoned"))
    }
}

/// One point of `tell`: `(energy, gradient or None, Hessian (flat, row-major) or None)`.
type PointInput = (f64, Option<Vec<f64>>, Option<Vec<f64>>);

#[pymethods]
impl PyBitss {
    #[new]
    #[pyo3(signature = (
        first,
        second,
        *,
        methods = vec!["lbfgs".to_string()],
        max_iterations = 1000,
        criteria = "normal",
        exact_hessian_interval = None,
        max_trust_radius = None,
        min_trust_radius = 0.01,
        reduction = mopt_rs::path::bitss::DEFAULT_REDUCTION,
        alpha = mopt_rs::path::bitss::DEFAULT_ALPHA,
        beta = mopt_rs::path::bitss::DEFAULT_BETA,
        coefficient_interval = mopt_rs::path::bitss::DEFAULT_COEFFICIENT_INTERVAL,
        interpolation_points = mopt_rs::path::bitss::DEFAULT_INTERPOLATION_POINTS,
        min_separation = mopt_rs::path::bitss::DEFAULT_MIN_SEPARATION,
        max_stages = mopt_rs::path::bitss::DEFAULT_MAX_STAGES,
        step_share = mopt_rs::path::bitss::DEFAULT_STEP_SHARE,
        midpoint_shift = None,
        stop_on_midpoint_gradient = true,
        frozen = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        first: Vec<f64>,
        second: Vec<f64>,
        methods: Vec<String>,
        max_iterations: usize,
        criteria: &str,
        exact_hessian_interval: Option<usize>,
        max_trust_radius: Option<f64>,
        min_trust_radius: f64,
        reduction: f64,
        alpha: f64,
        beta: f64,
        coefficient_interval: usize,
        interpolation_points: usize,
        min_separation: f64,
        max_stages: usize,
        step_share: Option<f64>,
        midpoint_shift: Option<f64>,
        stop_on_midpoint_gradient: bool,
        frozen: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let settings = BitssSettings {
            reduction,
            alpha,
            beta,
            coefficient_interval,
            interpolation_points,
            stop_on_midpoint_gradient,
            min_separation,
            midpoint_shift,
            max_stages,
            step_share,
            minimizer: VectorSettings {
                methods,
                saddle_order: 0,
                max_iterations,
                criteria: criteria_of(criteria)?,
                trust: TrustSettings {
                    max: max_trust_radius.map(Quantity::<Bohr>::new),
                    min: Quantity::new(min_trust_radius),
                },
                exact_hessian_interval: exact_hessian_interval.filter(|&n| n > 0),
                ccqn: None,
                mode_kill: None,
                ..VectorSettings::default()
            },
        };
        let bitss = Bitss::new(settings, &column(&first), &column(&second))
            .and_then(|b| b.with_frozen(&frozen.unwrap_or_default()))
            .map_err(err)?;
        Ok(Self {
            inner: Mutex::new(bitss),
        })
    }

    /// The points to evaluate next, or `None` when the search has ended. Each is a dict:
    /// `role` (`first`, `second`, `between`), `share` (of the way from the first image to
    /// the second, for `between`), `x`, `gradient` and `hessian` (whether they are asked
    /// for).
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        let points = match self.lock()?.ask() {
            BitssAction::Finished(_) => return Ok(None),
            BitssAction::Evaluate(points) => points,
        };
        points
            .iter()
            .map(|p| {
                let d = PyDict::new(py);
                let (role, share) = match p.role {
                    PointRole::First => ("first", None),
                    PointRole::Second => ("second", None),
                    PointRole::Between(share) => ("between", Some(share)),
                };
                d.set_item("role", role)?;
                d.set_item("share", share)?;
                d.set_item("x", list(&p.x))?;
                d.set_item("gradient", p.gradient)?;
                d.set_item("hessian", p.hessian)?;
                Ok(d)
            })
            .collect::<PyResult<Vec<_>>>()
            .map(Some)
    }

    /// The surface at the points of `ask()`, in its order: for each `(energy, gradient or
    /// None, Hessian (flat, row-major) or None)`. Returns a report.
    fn tell<'py>(&self, py: Python<'py>, values: Vec<PointInput>) -> PyResult<Bound<'py, PyDict>> {
        let mut bitss = self.lock()?;
        let n = bitss.dimension();
        let values = values
            .into_iter()
            .map(|(energy, gradient, hessian)| {
                let hessian = match hessian {
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
                Ok(PointValue {
                    energy,
                    gradient: gradient.as_deref().map(column),
                    hessian,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let r = bitss.tell(values).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("iteration", r.iteration)?;
        d.set_item("stage", r.stage)?;
        d.set_item("new_stage", r.new_stage)?;
        d.set_item("target_separation", r.target_separation)?;
        d.set_item("separation", r.separation)?;
        d.set_item("first", list(&r.images[0]))?;
        d.set_item("second", list(&r.images[1]))?;
        d.set_item("energies", r.energies.to_vec())?;
        d.set_item("barrier", r.coefficients.barrier)?;
        d.set_item("kappa_e", r.coefficients.kappa_e)?;
        d.set_item("kappa_d", r.coefficients.kappa_d)?;
        d.set_item("value", r.function)?;
        d.set_item("gradient", list(&r.gradient))?;
        d.set_item("end", r.end.map(end_name))?;
        d.set_item("notes", r.notes)?;
        if let Some(step) = &r.step {
            d.set_item("max_gradient", step.convergence.max_force.value())?;
            d.set_item("rms_gradient", step.convergence.rms_force.value())?;
            d.set_item("step_norm", step.step_norm)?;
            d.set_item("trust_radius", step.trust_radius)?;
            d.set_item("stage_converged", step.stage_converged)?;
        }
        if let Some(m) = &r.midpoint {
            d.set_item("midpoint", list(&m.x))?;
            d.set_item("midpoint_energy", m.energy)?;
            d.set_item("midpoint_max_gradient", m.max_force)?;
            d.set_item("midpoint_rms_gradient", m.rms_force)?;
        }
        Ok(d)
    }

    /// Why the search ended (`midpoint_gradient`, `separation`, `midpoint_shift`,
    /// `no_barrier`, `stage_limit`, `iteration_limit`), or `None` while it runs.
    #[getter]
    fn end(&self) -> PyResult<Option<&'static str>> {
        Ok(self.lock()?.end().map(end_name))
    }

    /// Whether the search ended on one of its criteria.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.lock()?.end().is_some_and(BitssEnd::converged))
    }

    /// The two images.
    #[getter]
    fn images(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let (first, second) = self.lock()?.images();
        Ok((list(&first), list(&second)))
    }

    /// The point halfway between the images: the estimate of the transition state.
    #[getter]
    fn transition_state_estimate(&self) -> PyResult<Vec<f64>> {
        Ok(list(&self.lock()?.transition_state_estimate()))
    }

    /// Energy of the midpoint as the last survey of the line between the images found it.
    #[getter]
    fn midpoint_energy(&self) -> PyResult<Option<f64>> {
        Ok(self.lock()?.midpoint().map(|m| m.energy))
    }

    /// Stage (0 before the first reduction of the separation).
    #[getter]
    fn stage(&self) -> PyResult<usize> {
        Ok(self.lock()?.stage())
    }

    /// Separation asked for at this stage.
    #[getter]
    fn target_separation(&self) -> PyResult<f64> {
        Ok(self.lock()?.target_separation())
    }

    /// Iterations of the minimizer so far.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        Ok(self.lock()?.iterations())
    }

    /// Number of variables of one image.
    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.lock()?.dimension())
    }
}

/// The BITSS function `E1 + E2 + kappa_e (E1 - E2)^2 + kappa_d (d - target)^2` of two images
/// and its gradient in their `2n` variables (those of the first image, then those of the
/// second).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bitss_function(
    kappa_e: f64,
    kappa_d: f64,
    target: f64,
    first: Vec<f64>,
    second: Vec<f64>,
    energies: (f64, f64),
    gradient_first: Vec<f64>,
    gradient_second: Vec<f64>,
) -> PyResult<(f64, Vec<f64>)> {
    let c = Coefficients {
        barrier: 0.0,
        kappa_e,
        kappa_d,
    };
    let (value, gradient) = penalized(
        &c,
        target,
        [&column(&first), &column(&second)],
        [energies.0, energies.1],
        [&column(&gradient_first), &column(&gradient_second)],
    )
    .map_err(err)?;
    Ok((value, list(&gradient)))
}

/// The coefficients of the BITSS function (Eqs. (4) and (5) of Avis, Panter, and
/// Kusumaatmaja 2022): `(kappa_e, kappa_d)` for the separation `target` asked for, the
/// barrier estimate, and the sizes of the two gradients along the line between the images.
#[pyfunction]
fn bitss_coefficients(
    alpha: f64,
    beta: f64,
    target: f64,
    barrier: f64,
    projected: (f64, f64),
) -> (f64, f64) {
    let c = Coefficients::new(alpha, beta, target, barrier, [projected.0, projected.1]);
    (c.kappa_e, c.kappa_d)
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBitss>()?;
    m.add_function(wrap_pyfunction!(bitss_function, m)?)?;
    m.add_function(wrap_pyfunction!(bitss_coefficients, m)?)?;
    Ok(())
}
