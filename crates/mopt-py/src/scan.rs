//! Scans from Python (`relaxedscan`): ReSPer coordinates of structures and points with their
//! gradients and pre-images, the points of a scan and their order, the stationary points of a
//! scanned surface, structures of rigid scans, and the command itself.
//!
//! MultiOptPy: `Entrypoints/relaxed_scan.py:run_relaxedscan` (the command); the parts are new
//! in mopt-rs.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::Col;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::analysis::resper::preimage::{aligned_references, preimage, PreimageSettings};
use mopt_rs::analysis::resper::ResperMap;
use mopt_rs::cli::constraints::projection_constraint;
use mopt_rs::domain::{AtomId, MolecularSystem};
use mopt_rs::workflow::scan::rigid::{bond_graph, set_rigidly};
use mopt_rs::workflow::scan::{stationary_points_periodic, GridOrder, ScanAxis, ScanGrid};

use super::{cell_from, err, system};

/// The map of reference structures (or points of plain variables) by classical
/// multidimensional scaling, with the coordinates of any structure on it (Gower's formula)
/// and their gradients.
#[pyclass(name = "ResperMap")]
pub struct PyResperMap {
    pub(crate) map: Arc<ResperMap>,
    aligned: Vec<Col<f64>>,
    template: Option<MolecularSystem>,
}

fn frames(
    symbols: &[String],
    frames_bohr: &[Vec<f64>],
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Vec<MolecularSystem>> {
    let cell = cell_from(cell_bohr)?;
    frames_bohr
        .iter()
        .map(|f| system(symbols, f, 0, 1, cell.clone()))
        .collect()
}

#[pymethods]
impl PyResperMap {
    /// The map of the structures `frames_bohr` (flat positions in Bohr) of the atoms
    /// `symbols`; `atoms` (0-based) are those of the distances (all when `None`);
    /// `mass_weighted` uses the masses (unit masses otherwise).
    #[new]
    #[pyo3(signature = (symbols, frames_bohr, atoms = None, mass_weighted = true, cell_bohr = None))]
    fn new(
        py: Python<'_>,
        symbols: Vec<String>,
        frames_bohr: Vec<Vec<f64>>,
        atoms: Option<Vec<usize>>,
        mass_weighted: bool,
        cell_bohr: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let structures = frames(&symbols, &frames_bohr, cell_bohr)?;
        let template = structures.first().cloned();
        let atoms: Vec<AtomId> = atoms
            .unwrap_or_default()
            .into_iter()
            .map(AtomId::new)
            .collect();
        let map = py
            .allow_threads(|| ResperMap::from_structures(structures, &atoms, mass_weighted))
            .map_err(err)?;
        let aligned = aligned_references(&map).map_err(err)?;
        Ok(Self {
            map: Arc::new(map),
            aligned,
            template,
        })
    }

    /// The map of points of plain variables.
    #[staticmethod]
    fn from_points(points: Vec<Vec<f64>>) -> PyResult<Self> {
        let cols = points
            .iter()
            .map(|p| Col::from_fn(p.len(), |i| p[i]))
            .collect();
        let map = ResperMap::from_points(cols).map_err(err)?;
        let aligned = aligned_references(&map).map_err(err)?;
        Ok(Self {
            map: Arc::new(map),
            aligned,
            template: None,
        })
    }

    /// Number of coordinates (two, or one for references on a line).
    #[getter]
    fn components(&self) -> usize {
        self.map.components()
    }

    /// The coordinates of the references (a row per reference).
    #[getter]
    fn coordinates(&self) -> Vec<Vec<f64>> {
        (0..self.map.len())
            .map(|i| {
                (0..self.map.components())
                    .map(|k| self.map.reference_coordinate(i, k))
                    .collect()
            })
            .collect()
    }

    /// Eigenvalues of the coordinates.
    #[getter]
    fn eigenvalues(&self) -> Vec<f64> {
        self.map.eigenvalues().to_vec()
    }

    /// Share of each coordinate in the sum of the positive eigenvalues.
    #[getter]
    fn contributions(&self) -> Vec<f64> {
        self.map.contributions().to_vec()
    }

    /// How well points of a Euclidean space reproduce the distances (1: exactly).
    #[getter]
    fn reproducibility(&self) -> f64 {
        self.map.reproducibility()
    }

    /// The coordinates of `x` (flat positions in Bohr of a structure like the references, or a
    /// point of plain variables) and their gradients (one list per coordinate).
    fn project(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let xc = Col::from_fn(x.len(), |i| x[i]);
        let (values, gradients) = py
            .allow_threads(|| self.map.project_flat(&xc))
            .map_err(err)?;
        Ok((
            values,
            gradients
                .iter()
                .map(|g| g.iter().copied().collect())
                .collect(),
        ))
    }

    /// A structure for the point `target` of the coordinates `components` (0-based): a dict
    /// with `x` (flat, Bohr or variables), `error` (distance between the point and the
    /// coordinates of the interpolated structure), `remaining` (after the correction),
    /// `correction` (its length), `nearest` (distance to the nearest reference on the map),
    /// `extrapolated`, and `weights` (reference, weight).
    #[pyo3(signature = (components, target, correct = true, outside_factor = 2.0))]
    fn preimage<'py>(
        &self,
        py: Python<'py>,
        components: Vec<usize>,
        target: Vec<f64>,
        correct: bool,
        outside_factor: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let settings = PreimageSettings {
            correct,
            outside_factor,
        };
        let p = preimage(&self.map, &self.aligned, &components, &target, &settings).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("x", p.x.iter().copied().collect::<Vec<f64>>())?;
        d.set_item("error", p.error)?;
        d.set_item("remaining", p.remaining)?;
        d.set_item("correction", p.correction)?;
        d.set_item("nearest", p.nearest)?;
        d.set_item("extrapolated", p.extrapolated)?;
        d.set_item("weights", p.weights)?;
        Ok(d)
    }

    /// The element symbols of the reference structures (empty for points).
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.template
            .as_ref()
            .map(|t| t.elements.iter().map(|e| e.symbol().to_string()).collect())
            .unwrap_or_default()
    }
}

/// The points of a scan of coordinates with `ranges` (start, end) and `counts` points each:
/// all moving together (`grid=False`, MultiOptPy's scan) or every combination (`grid=True`),
/// in `order` (`snake` or `raster`). A dict with `shape`, `order` (the points in the order
/// they are taken), `values` (a row per point), and `start` (the point each one starts from,
/// `None` for the first).
#[pyfunction]
#[pyo3(signature = (ranges, counts, grid = false, order = "snake"))]
fn scan_points<'py>(
    py: Python<'py>,
    ranges: Vec<(f64, f64)>,
    counts: Vec<usize>,
    grid: bool,
    order: &str,
) -> PyResult<Bound<'py, PyDict>> {
    if ranges.len() != counts.len() {
        return Err(PyValueError::new_err("one count per range"));
    }
    let axes = ranges
        .iter()
        .zip(&counts)
        .enumerate()
        .map(|(k, (r, &n))| ScanAxis::linspace(format!("q{}", k + 1), "", r.0, r.1, n))
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let g = if grid {
        ScanGrid::product(axes, GridOrder::parse(order).map_err(err)?)
    } else {
        ScanGrid::joint(axes)
    }
    .map_err(err)?;
    let mut starts = Vec::with_capacity(g.len());
    for (k, &p) in g.order().iter().enumerate() {
        starts.push(g.start_of(p, &g.order()[..k]));
    }
    let d = PyDict::new(py);
    d.set_item("shape", g.shape())?;
    d.set_item("order", g.order().to_vec())?;
    d.set_item(
        "values",
        (0..g.len()).map(|p| g.values(p)).collect::<Vec<_>>(),
    )?;
    d.set_item("start", starts)?;
    Ok(d)
}

/// The stationary points of a scanned surface: `energies` on a grid of `shape` (first
/// dimension fastest; NaN where missing). A list of dicts with `point`, `kind` (`minimum`,
/// `saddle`), `energy`, `offset` (grid steps from the point to the stationary point of the
/// local quadratic model), and `model_energy`. `periodic` marks the dimensions that cover one
/// whole period with the last point equal to the first (a dihedral from -180 to 180).
#[pyfunction]
#[pyo3(signature = (shape, energies, periodic = None))]
fn scan_stationary_points<'py>(
    py: Python<'py>,
    shape: Vec<usize>,
    energies: Vec<f64>,
    periodic: Option<Vec<bool>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let periodic = periodic.unwrap_or_else(|| vec![false; shape.len()]);
    stationary_points_periodic(&shape, &energies, &periodic)
        .map_err(err)?
        .into_iter()
        .map(|c| {
            let d = PyDict::new(py);
            d.set_item("point", c.point)?;
            d.set_item("kind", c.kind.name())?;
            d.set_item("energy", c.energy)?;
            d.set_item("offset", c.offset)?;
            d.set_item("model_energy", c.model_energy)?;
            Ok(d)
        })
        .collect()
}

/// The structure of a rigid scan: `positions_bohr` (flat) with every coordinate of
/// `targets` (`(kind, atoms, value)` as `-pc manual`: Angstrom or degrees) set by moving the
/// atoms on one side of it rigidly.
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, targets, cell_bohr = None))]
fn rigid_scan_structure(
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    targets: Vec<(String, String, f64)>,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let constraints = targets
        .iter()
        .map(|(kind, atoms, value)| projection_constraint(kind, atoms, Some(*value)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let moved = set_rigidly(&sys, &bond_graph(&sys), &constraints).map_err(err)?;
    Ok(moved.coords_flat_bohr().iter().copied().collect())
}

/// Run `relaxedscan` with `args` (as on the command line, without the program name): a dict
/// with `directory`, `interrupted`, `points` (a dict per point: `point`, `energy`,
/// `biased_energy`, `converged`, `iterations`, `file`), and `candidates` (the stationary
/// points). `console`: print the progress.
#[pyfunction]
#[pyo3(signature = (args, console = true))]
fn run_relaxedscan<'py>(
    py: Python<'py>,
    args: Vec<String>,
    console: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use mopt_rs::cli::relaxedscan;
    use mopt_rs::core_infra::log::Level;
    let registry = super::pycalc::python_registry();
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = relaxedscan::config_from_cli(&refs).map_err(err)?;
    let level = console.then_some(Level::Info);
    let run = py
        .allow_threads(|| {
            relaxedscan::run_with(&config, std::path::Path::new("."), level, &registry)
        })
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("directory", run.directory.to_string_lossy().into_owned())?;
    d.set_item("interrupted", run.interrupted)?;
    let points = run
        .points
        .iter()
        .map(|p| {
            let e = PyDict::new(py);
            e.set_item("point", p.point)?;
            e.set_item("energy", p.energy)?;
            e.set_item("biased_energy", p.biased)?;
            e.set_item("converged", p.converged)?;
            e.set_item("iterations", p.iterations)?;
            e.set_item("file", p.file.clone())?;
            Ok(e)
        })
        .collect::<PyResult<Vec<_>>>()?;
    d.set_item("points", points)?;
    let candidates = run
        .candidates
        .iter()
        .map(|c| {
            let e = PyDict::new(py);
            e.set_item("point", c.point)?;
            e.set_item("kind", c.kind.name())?;
            e.set_item("energy", c.energy)?;
            e.set_item("offset", c.offset.clone())?;
            Ok(e)
        })
        .collect::<PyResult<Vec<_>>>()?;
    d.set_item("candidates", candidates)?;
    Ok(d)
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResperMap>()?;
    m.add_function(wrap_pyfunction!(scan_points, m)?)?;
    m.add_function(wrap_pyfunction!(scan_stationary_points, m)?)?;
    m.add_function(wrap_pyfunction!(rigid_scan_structure, m)?)?;
    m.add_function(wrap_pyfunction!(run_relaxedscan, m)?)?;
    Ok(())
}
