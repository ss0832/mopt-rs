//! The improved elastic image pair method from Python: the ask/tell state machine
//! [`ElasticImagePair`] over two points of `n` variables (or two structures).
//!
//! MultiOptPy: `OtherMethod/elastic_image_pair.py:ElasticImagePair`.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::Col;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::domain::{positions, Bohr, Element, MolecularSystem, SpinCharge};
use mopt_rs::path::ieip::{
    ElasticImagePair, IeipAction, IeipEnd, IeipSettings, ImageValue, MicroCoordinates,
    MicroSettings,
};

use super::err;

pub(crate) fn column(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

pub(crate) fn list(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

/// A structure of the elements `symbols` at the flat coordinates `x` (Bohr).
pub(crate) fn structure(symbols: &[String], x: &[f64]) -> PyResult<MolecularSystem> {
    if x.len() != 3 * symbols.len() {
        return Err(PyValueError::new_err(format!(
            "{} coordinates for {} atoms",
            x.len(),
            symbols.len()
        )));
    }
    let elements = symbols
        .iter()
        .map(|s| Element::parse_token(s))
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let pos: Vec<[f64; 3]> = x.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    MolecularSystem::new(
        elements,
        &positions::<Bohr>(&pos),
        SpinCharge::default(),
        None,
    )
    .map_err(err)
}

fn end_name(end: IeipEnd) -> &'static str {
    match end {
        IeipEnd::Converged => "converged",
        IeipEnd::IterationLimit => "iteration_limit",
    }
}

/// Ask/tell improved elastic image pair method between two points.
#[pyclass(name = "ElasticImagePair")]
pub struct PyElasticImagePair {
    inner: Mutex<ElasticImagePair>,
}

impl PyElasticImagePair {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, ElasticImagePair>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the state of the search is poisoned"))
    }
}

/// One image of `tell`: `(energy, gradient)` or `(energy, gradient, energy with bias,
/// gradient with bias)`.
type ImageInput = (f64, Vec<f64>, Option<f64>, Option<Vec<f64>>);

#[pymethods]
impl PyElasticImagePair {
    #[new]
    #[pyo3(signature = (
        first,
        second,
        *,
        max_iterations = 999,
        beta = 1.0,
        elements = None,
        microiterations = 0,
        micro_tolerance = 0.01,
        frozen = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        first: Vec<f64>,
        second: Vec<f64>,
        max_iterations: usize,
        beta: f64,
        elements: Option<Vec<String>>,
        microiterations: usize,
        micro_tolerance: f64,
        frozen: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let template = match &elements {
            Some(symbols) => Some(structure(symbols, &first)?),
            None => None,
        };
        let frozen = frozen.unwrap_or_default();
        let settings = IeipSettings {
            max_iterations,
            beta,
            micro: MicroSettings {
                steps: microiterations,
                force_tolerance: micro_tolerance,
                ..MicroSettings::default()
            },
            micro_coordinates: template
                .clone()
                .filter(|_| microiterations > 0)
                .map(MicroCoordinates::polar),
            align: template.is_some() && frozen.is_empty(),
            frozen,
        };
        let pair =
            ElasticImagePair::new(settings, &column(&first), &column(&second)).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(pair),
        })
    }

    /// The two points to evaluate next, or `None` when the run has ended.
    fn ask(&self) -> PyResult<Option<(Vec<f64>, Vec<f64>)>> {
        Ok(match self.lock()?.ask() {
            IeipAction::Evaluate([a, b]) => Some((list(&a), list(&b))),
            IeipAction::Finished(_) => None,
        })
    }

    /// The surface at the two points of `ask()`: for each `(energy, gradient)`, or with the
    /// energy and gradient with bias potentials as well. Returns a report.
    fn tell<'py>(
        &self,
        py: Python<'py>,
        images: (ImageInput, ImageInput),
    ) -> PyResult<Bound<'py, PyDict>> {
        let value = |(e, g, be, bg): ImageInput| ImageValue {
            energy: e,
            biased_energy: be.unwrap_or(e),
            biased_gradient: column(bg.as_deref().unwrap_or(&g)),
            gradient: column(&g),
        };
        let r = self
            .lock()?
            .tell([value(images.0), value(images.1)])
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("iteration", r.iteration)?;
        d.set_item("micro_step", r.micro_step)?;
        d.set_item("first", list(&r.images[0]))?;
        d.set_item("second", list(&r.images[1]))?;
        d.set_item("energies", (r.values[0].energy, r.values[1].energy))?;
        d.set_item("end", r.end.map(end_name))?;
        if let Some(step) = &r.step {
            d.set_item("separation", step.separation)?;
            d.set_item("beta", step.beta)?;
            d.set_item("new_first", list(&step.new_images[0]))?;
            d.set_item("new_second", list(&step.new_images[1]))?;
        }
        Ok(d)
    }

    /// Why the run ended (`converged`, `iteration_limit`), or `None` while it runs.
    #[getter]
    fn end(&self) -> PyResult<Option<&'static str>> {
        Ok(self.lock()?.end().map(end_name))
    }

    /// The images (to evaluate next; after the end, those after the last step).
    #[getter]
    fn images(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let m = self.lock()?;
        let [a, b] = m.images();
        Ok((list(a), list(b)))
    }

    /// The midpoint of the images: the estimate of the transition state.
    #[getter]
    fn transition_state_estimate(&self) -> PyResult<Vec<f64>> {
        Ok(list(&self.lock()?.transition_state_estimate()))
    }

    /// Iterations completed.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        Ok(self.lock()?.iteration())
    }

    /// Separation of the images after the last step.
    #[getter]
    fn separation(&self) -> PyResult<f64> {
        Ok(self.lock()?.separation())
    }

    /// `beta` of the next iteration.
    #[getter]
    fn beta(&self) -> PyResult<f64> {
        Ok(self.lock()?.beta())
    }
}

/// Add the classes of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyElasticImagePair>()?;
    Ok(())
}
