//! The searches for saddle points of index `k` from Python: high-index saddle dynamics
//! (HiOSD, GGAD, iHiSD), the lowest curvatures at a point from gradients, the starts of the
//! upward and downward searches of the solution landscape, and the two bands on the ridge
//! between two transition states. Each search is an ask/tell state machine like `Dimer`:
//! `ask()` gives the points to evaluate (dicts with `x`, `gradient`, `hessian`) or `None` at
//! the end, `tell()` takes `(energy, gradient or None, Hessian (flat, row-major) or None)`
//! per point in the same order.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::Col;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::domain::MolecularSystem;
use mopt_rs::path::band::ridge::{
    RidgeBand, RidgeBandSettings, RidgeSpring, RidgeStep, TransverseMode,
};
use mopt_rs::path::band::ridge_fixed::{FixedRidgeBand, FixedRidgeSettings};
use mopt_rs::path::hisd::landscape::{downward_starts, upward_starts, SearchStart};
use mopt_rs::path::hisd::modes::{LowestModes, ModeSettings};
use mopt_rs::path::hisd::{
    Curvature, DirectionUpdate, HandOff, HighIndexSaddleSearch, HisdSettings, PointFlow,
};

use super::err;
use super::ieip::{column, list, structure};
use super::searches::{points, values, PointInput};
use super::variables::criteria_of;

fn locked<T>(m: &Mutex<T>) -> PyResult<std::sync::MutexGuard<'_, T>> {
    m.lock()
        .map_err(|_| PyRuntimeError::new_err("the state of the search is poisoned"))
}

fn columns(v: &[Vec<f64>]) -> Vec<Col<f64>> {
    v.iter().map(|d| column(d)).collect()
}

fn template(symbols: Option<Vec<String>>, x: &[f64]) -> PyResult<Option<MolecularSystem>> {
    symbols.map(|s| structure(&s, x)).transpose()
}

fn curvature_of(name: &str) -> PyResult<Curvature> {
    Curvature::from_name(name).map_err(err)
}

/// Ask/tell search for a saddle point of index `order`.
#[pyclass(name = "HighIndexSaddleSearch")]
pub struct PyHighIndexSaddle {
    inner: Mutex<HighIndexSaddleSearch>,
    n: usize,
}

#[pymethods]
impl PyHighIndexSaddle {
    /// `method` is `hiosd` or `ggad` (their defaults); `update` overrides the update of the
    /// directions (`euler`, `oja`, `lobpcg`, `ggad`, `eigenvectors`); `flow` is `reflected`,
    /// `ihisd_down` or `ihisd_up`; with `symbols` the points are structures (Bohr).
    #[new]
    #[pyo3(signature = (start, directions, *, order, method = "hiosd", update = None,
        curvature = "forward", guard = false, flow = "reflected", alpha0 = 1e-7,
        alpha_rate = 0.5, dimer_length = 1e-3, shrink = 0.0, min_dimer_length = 1e-4,
        first_step = 0.05, max_step = 0.2, max_iterations = 1000, criteria = "normal",
        hand_off = None, symbols = None, rigid = true, mass_weighted = false, frozen = None,
        seed = 1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        start: Vec<f64>,
        directions: Vec<Vec<f64>>,
        order: usize,
        method: &str,
        update: Option<&str>,
        curvature: &str,
        guard: bool,
        flow: &str,
        alpha0: f64,
        alpha_rate: f64,
        dimer_length: f64,
        shrink: f64,
        min_dimer_length: f64,
        first_step: f64,
        max_step: f64,
        max_iterations: usize,
        criteria: &str,
        hand_off: Option<(f64, f64)>,
        symbols: Option<Vec<String>>,
        rigid: bool,
        mass_weighted: bool,
        frozen: Option<Vec<usize>>,
        seed: u64,
    ) -> PyResult<Self> {
        let mut s = match method.to_ascii_lowercase().as_str() {
            "hiosd" | "hisd" => HisdSettings::hiosd(order),
            "ggad" => HisdSettings::ggad(order),
            other => {
                return Err(PyValueError::new_err(format!(
                    "method must be hiosd or ggad; got '{other}'"
                )))
            }
        };
        if let Some(u) = update {
            s.update = DirectionUpdate::from_name(u).map_err(err)?;
        }
        s.curvature = curvature_of(curvature)?;
        s.guard = guard;
        s.flow = match flow.to_ascii_lowercase().as_str() {
            "reflected" | "hisd" => PointFlow::Reflected,
            "ihisd_down" | "ihisd" => PointFlow::Improved {
                alpha0,
                rate: alpha_rate,
                upward: false,
            },
            "ihisd_up" => PointFlow::Improved {
                alpha0,
                rate: alpha_rate,
                upward: true,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "flow must be reflected, ihisd_down, or ihisd_up; got '{other}'"
                )))
            }
        };
        s.dimer_length = dimer_length;
        s.shrink = shrink;
        s.min_dimer_length = min_dimer_length;
        s.step.first = first_step;
        s.step.max_step = max_step;
        s.max_iterations = max_iterations;
        s.criteria = criteria_of(criteria)?;
        s.hand_off = hand_off.map(|(rms_gradient, projector_change)| HandOff {
            rms_gradient,
            projector_change,
        });
        s.seed = seed;
        let mut search =
            HighIndexSaddleSearch::new(s, &column(&start), &columns(&directions)).map_err(err)?;
        if let Some(t) = template(symbols, &start)? {
            search = search
                .with_structure(t, rigid, mass_weighted)
                .map_err(err)?;
        }
        if let Some(f) = frozen {
            search = search.with_frozen(f).map_err(err)?;
        }
        Ok(Self {
            inner: Mutex::new(search),
            n: start.len(),
        })
    }

    /// The points to evaluate next, or `None` at the end.
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
        d.set_item("point", list(&r.point))?;
        d.set_item("energy", r.energy)?;
        d.set_item("max_force", r.max_force)?;
        d.set_item("rms_force", r.rms_force)?;
        d.set_item("curvatures", r.curvatures)?;
        d.set_item("projector_change", r.projector_change)?;
        d.set_item("dimer_length", r.dimer_length)?;
        d.set_item("step", r.step)?;
        d.set_item("gradients", r.gradients)?;
        d.set_item("hessians", r.hessians)?;
        d.set_item("end", r.end.map(|e| format!("{e:?}").to_ascii_lowercase()))?;
        Ok(d)
    }

    /// The point (after the end: the last one evaluated).
    #[getter]
    fn point(&self) -> PyResult<Vec<f64>> {
        Ok(list(locked(&self.inner)?.point()))
    }

    /// The tracked directions in the coordinates of the points (unit length each).
    #[getter]
    fn directions(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(locked(&self.inner)?
            .point_directions()
            .iter()
            .map(list)
            .collect())
    }

    /// The curvatures along the tracked directions (metric of the search).
    #[getter]
    fn curvatures(&self) -> PyResult<Vec<f64>> {
        Ok(locked(&self.inner)?.curvatures().to_vec())
    }

    /// Why the search ended (`converged`, `handoff`, `wrongindex`, `leftindex`,
    /// `iterationlimit`), or `None` while it runs.
    #[getter]
    fn end(&self) -> PyResult<Option<String>> {
        Ok(locked(&self.inner)?
            .end()
            .map(|e| format!("{e:?}").to_ascii_lowercase()))
    }

    /// Gradients evaluated (the differences included).
    #[getter]
    fn gradients(&self) -> PyResult<usize> {
        Ok(locked(&self.inner)?.gradient_evaluations())
    }

    /// Hessians evaluated.
    #[getter]
    fn hessians(&self) -> PyResult<usize> {
        Ok(locked(&self.inner)?.hessian_evaluations())
    }
}

/// Ask/tell search for the lowest curvatures at a point.
#[pyclass(name = "LowestModes")]
pub struct PyLowestModes {
    inner: Mutex<LowestModes>,
    n: usize,
}

#[pymethods]
impl PyLowestModes {
    #[new]
    #[pyo3(signature = (point, *, count = 2, start = None, curvature = "forward",
        dimer_length = 1e-3, max_iterations = 100, tolerance = 1e-3, symbols = None,
        rigid = true, mass_weighted = false, seed = 1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        point: Vec<f64>,
        count: usize,
        start: Option<Vec<Vec<f64>>>,
        curvature: &str,
        dimer_length: f64,
        max_iterations: usize,
        tolerance: f64,
        symbols: Option<Vec<String>>,
        rigid: bool,
        mass_weighted: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let settings = ModeSettings {
            count,
            curvature: curvature_of(curvature)?,
            dimer_length,
            max_iterations,
            tolerance,
            seed,
        };
        let mut m = LowestModes::new(
            settings,
            &column(&point),
            &columns(&start.unwrap_or_default()),
        )
        .map_err(err)?;
        if let Some(t) = template(symbols, &point)? {
            m = m.with_structure(t, rigid, mass_weighted).map_err(err)?;
        }
        Ok(Self {
            inner: Mutex::new(m),
            n: point.len(),
        })
    }

    /// The points to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the points of `ask()`.
    fn tell(&self, input: Vec<PointInput>) -> PyResult<()> {
        locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)
    }

    /// The curvatures found (ascending).
    #[getter]
    fn values(&self) -> PyResult<Vec<f64>> {
        Ok(locked(&self.inner)?.values().to_vec())
    }

    /// Their directions in the coordinates of the point (unit length each).
    #[getter]
    fn directions(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(locked(&self.inner)?
            .point_directions()
            .iter()
            .map(list)
            .collect())
    }

    /// Whether the residuals met the tolerance.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(locked(&self.inner)?.converged())
    }
}

fn start_dict<'py>(py: Python<'py>, s: &SearchStart) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("point", list(&s.point))?;
    d.set_item(
        "directions",
        s.directions.iter().map(list).collect::<Vec<_>>(),
    )?;
    d.set_item("order", s.order)?;
    d.set_item("along", list(&s.along))?;
    d.set_item("sign", s.sign)?;
    Ok(d)
}

/// The starts of the upward searches from a stationary point (its unstable directions)
/// along each of `stable`, both signs.
#[pyfunction]
#[pyo3(name = "hisd_upward_starts", signature = (point, unstable, stable, displacement = 0.05))]
fn py_upward_starts<'py>(
    py: Python<'py>,
    point: Vec<f64>,
    unstable: Vec<Vec<f64>>,
    stable: Vec<Vec<f64>>,
    displacement: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    upward_starts(
        &column(&point),
        &columns(&unstable),
        &columns(&stable),
        displacement,
    )
    .map_err(err)?
    .iter()
    .map(|s| start_dict(py, s))
    .collect()
}

/// The starts of the downward searches from a saddle point along each unstable direction
/// (or a fan of `fan` directions of the plane of two), both signs.
#[pyfunction]
#[pyo3(name = "hisd_downward_starts", signature = (point, unstable, displacement = 0.05, fan = 0))]
fn py_downward_starts<'py>(
    py: Python<'py>,
    point: Vec<f64>,
    unstable: Vec<Vec<f64>>,
    displacement: f64,
    fan: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    downward_starts(&column(&point), &columns(&unstable), displacement, fan)
        .map_err(err)?
        .iter()
        .map(|s| start_dict(py, s))
        .collect()
}

/// Ask/tell band on the ridge between two transition states (moving frame).
#[pyclass(name = "RidgeBand")]
pub struct PyRidgeBand {
    inner: Mutex<RidgeBand>,
    n: usize,
}

#[pymethods]
impl PyRidgeBand {
    /// `chain` includes the two transition states as its first and last images; `modes`
    /// are their imaginary modes.
    #[new]
    #[pyo3(signature = (chain, modes, *, spring_force = "full", transverse = "lowest",
        step_rule = "fire", spring = 0.1, climb_rms = 3e-3, climb_after = 50, tolerance = 3e-4,
        max_iterations = 500, curvature = "forward", dimer_length = 1e-3, symbols = None,
        rigid = true, mass_weighted = false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        chain: Vec<Vec<f64>>,
        modes: (Vec<f64>, Vec<f64>),
        spring_force: &str,
        transverse: &str,
        step_rule: &str,
        spring: f64,
        climb_rms: f64,
        climb_after: usize,
        tolerance: f64,
        max_iterations: usize,
        curvature: &str,
        dimer_length: f64,
        symbols: Option<Vec<String>>,
        rigid: bool,
        mass_weighted: bool,
    ) -> PyResult<Self> {
        let settings = RidgeBandSettings {
            spring_force: match spring_force {
                "full" => RidgeSpring::Full,
                "tangential" => RidgeSpring::Tangential,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "spring_force must be full or tangential; got '{other}'"
                    )))
                }
            },
            transverse: match transverse {
                "lowest" => TransverseMode::Lowest,
                "continued" => TransverseMode::Continued,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "transverse must be lowest or continued; got '{other}'"
                    )))
                }
            },
            step_rule: match step_rule {
                "bb" => RidgeStep::BarzilaiBorwein,
                "fire" => RidgeStep::Fire,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "step_rule must be bb or fire; got '{other}'"
                    )))
                }
            },
            spring,
            climb_rms,
            climb_after,
            tolerance,
            max_iterations,
            curvature: curvature_of(curvature)?,
            dimer_length,
            ..RidgeBandSettings::default()
        };
        let n = chain.first().map_or(0, Vec::len);
        let first = chain.first().cloned().unwrap_or_default();
        let mut band = RidgeBand::new(
            settings,
            columns(&chain),
            [column(&modes.0), column(&modes.1)],
        )
        .map_err(err)?;
        if let Some(t) = template(symbols, &first)? {
            band = band.with_structure(t, rigid, mass_weighted).map_err(err)?;
        }
        Ok(Self {
            inner: Mutex::new(band),
            n,
        })
    }

    /// The points to evaluate next, or `None` at the end.
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
        d.set_item("energies", r.energies)?;
        d.set_item("curvatures", r.curvatures)?;
        d.set_item("largest_force", r.largest_force)?;
        d.set_item("climbing", r.climbing)?;
        d.set_item("switches", r.switches)?;
        d.set_item("gradients", r.gradients)?;
        d.set_item("end", r.end.map(|e| format!("{e:?}").to_ascii_lowercase()))?;
        Ok(d)
    }

    /// The images.
    #[getter]
    fn chain(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(locked(&self.inner)?.chain().iter().map(list).collect())
    }

    /// The highest inner image.
    #[getter]
    fn highest(&self) -> PyResult<usize> {
        Ok(locked(&self.inner)?.highest())
    }

    /// The two directions of the highest image for a search of index 2.
    #[getter]
    fn saddle_directions(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(locked(&self.inner)?
            .saddle_directions()
            .iter()
            .map(list)
            .collect())
    }
}

/// Ask/tell band on the mode-maximized surface of a fixed direction between two
/// transition states with similar imaginary modes.
#[pyclass(name = "FixedRidgeBand")]
pub struct PyFixedRidgeBand {
    inner: Mutex<FixedRidgeBand>,
    n: usize,
}

#[pymethods]
impl PyFixedRidgeBand {
    #[new]
    #[pyo3(signature = (first, last, first_mode, last_mode, *, images = 7, tolerance = 3e-4,
        max_iterations = 300, line_tolerance = 1e-5, min_overlap = 0.7, symbols = None,
        rigid = true, mass_weighted = false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        first: Vec<f64>,
        last: Vec<f64>,
        first_mode: Vec<f64>,
        last_mode: Vec<f64>,
        images: usize,
        tolerance: f64,
        max_iterations: usize,
        line_tolerance: f64,
        min_overlap: f64,
        symbols: Option<Vec<String>>,
        rigid: bool,
        mass_weighted: bool,
    ) -> PyResult<Self> {
        let mut settings = FixedRidgeSettings::default();
        settings.chain.force_tolerance = Some(tolerance);
        settings.chain.max_iterations = max_iterations;
        settings.line.tolerance = line_tolerance;
        settings.min_overlap = min_overlap;
        let mut band = FixedRidgeBand::new(
            settings,
            &column(&first),
            &column(&last),
            &column(&first_mode),
            &column(&last_mode),
            images,
        )
        .map_err(err)?;
        if let Some(t) = template(symbols, &first)? {
            band = band.with_structure(t, rigid, mass_weighted).map_err(err)?;
        }
        Ok(Self {
            inner: Mutex::new(band),
            n: first.len(),
        })
    }

    /// The points to evaluate next, or `None` at the end.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        points(py, locked(&self.inner)?.ask())
    }

    /// The values at the points of `ask()`; a report after every iteration of the band.
    fn tell<'py>(
        &self,
        py: Python<'py>,
        input: Vec<PointInput>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(r) = locked(&self.inner)?
            .tell(values(self.n, input)?)
            .map_err(err)?
        else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("iteration", r.iteration)?;
        d.set_item("energies", r.energies)?;
        d.set_item("climbing", r.climbing)?;
        d.set_item("largest_force", r.largest_force)?;
        d.set_item("jumps", r.jumps)?;
        d.set_item("gradients", r.gradients)?;
        d.set_item("end", r.end.map(|e| format!("{e:?}").to_ascii_lowercase()))?;
        Ok(Some(d))
    }

    /// The highest inner image (coordinates of the points).
    #[getter]
    fn saddle_point(&self) -> PyResult<Option<Vec<f64>>> {
        Ok(locked(&self.inner)?.saddle_point().as_ref().map(list))
    }

    /// The two directions of the highest image for a search of index 2.
    #[getter]
    fn saddle_directions(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(locked(&self.inner)?
            .saddle_directions()
            .iter()
            .map(list)
            .collect())
    }

    /// Why the band ended, or `None` while it runs.
    #[getter]
    fn end(&self) -> PyResult<Option<String>> {
        Ok(locked(&self.inner)?
            .end()
            .map(|e| format!("{e:?}").to_ascii_lowercase()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHighIndexSaddle>()?;
    m.add_class::<PyLowestModes>()?;
    m.add_class::<PyRidgeBand>()?;
    m.add_class::<PyFixedRidgeBand>()?;
    m.add_function(wrap_pyfunction!(py_upward_starts, m)?)?;
    m.add_function(wrap_pyfunction!(py_downward_starts, m)?)?;
    Ok(())
}
