//! More Rust components callable one by one from Python: coordinate systems, hyperspherical
//! coordinates, superposition of structures, the bond graph, sampling priorities,
//! electron-transfer rates, and O1NumHess.
//!
//! Positions are flat lists in Bohr, energies in Hartree, matrices flat and row-major. Atoms
//! are numbered from 1, as in the other functions of this module.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy's parts cannot be called one by one.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::{Col, Mat};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::analysis::electron_transfer::{edeel_rate, TransferConditions};
use mopt_rs::coord::alignment::kabsch_align_bohr;
use mopt_rs::coord::connectivity::BondConnectivity;
use mopt_rs::coord::hyperspherical;
use mopt_rs::coord::system::transform::BackTransformOptions;
use mopt_rs::coord::system::{back_transform, CoordinateRegistry, CoordinateSystem};
use mopt_rs::core_infra::rng::Rng;
use mopt_rs::domain::{AtomId, MolecularSystem, Quantity};
use mopt_rs::pes::o1numhess::{molecular_problem, Invariance, O1Action, O1NumHess, O1Settings};
use mopt_rs::workflow::explore::{EqRecord, PriorityRegistry};

use super::{cell_from, err, system};

fn flat(m: &Mat<f64>) -> Vec<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    (0..r * c).map(|k| m[(k / c, k % c)]).collect()
}

fn list(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

struct CoordinateState {
    coords: Box<dyn CoordinateSystem>,
    /// The structure the coordinates were set up at (elements, charge, cell).
    template: MolecularSystem,
}

/// A coordinate system (`cart`, `ric`, `dic`, `tric`, `tric-p`, `auto`, or a registered name)
/// set up at a structure: values, Wilson matrix, displacements, and Cartesian steps.
#[pyclass(name = "Coordinates")]
pub struct PyCoordinates {
    inner: Mutex<CoordinateState>,
}

impl PyCoordinates {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, CoordinateState>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("coordinate state is poisoned"))
    }
}

fn at(state: &CoordinateState, positions_bohr: &[f64]) -> PyResult<MolecularSystem> {
    let x = Col::from_fn(positions_bohr.len(), |i| positions_bohr[i]);
    state.template.with_flat_coords_bohr(&x).map_err(err)
}

#[pymethods]
impl PyCoordinates {
    #[new]
    #[pyo3(signature = (name, symbols, positions_bohr, cell_bohr = None))]
    fn new(
        name: &str,
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
        cell_bohr: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let template = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
        let mut coords = CoordinateRegistry::default().create(name).map_err(err)?;
        coords.initialize(&template).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(CoordinateState { coords, template }),
        })
    }

    /// Name of the system in use (`auto (tric-cart)` for an automatic choice).
    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok(self.lock()?.coords.name().to_string())
    }

    /// Number of coordinates.
    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        let s = self.lock()?;
        Ok(s.coords.dim(&s.template))
    }

    /// Coordinate values at `positions_bohr`.
    fn values(&self, positions_bohr: Vec<f64>) -> PyResult<Vec<f64>> {
        let s = self.lock()?;
        let sys = at(&s, &positions_bohr)?;
        Ok(list(&s.coords.values(&sys).map_err(err)?))
    }

    /// Wilson matrix `dq/dx` at `positions_bohr` (flat, `dimension x 3N`).
    fn wilson_b(&self, positions_bohr: Vec<f64>) -> PyResult<Vec<f64>> {
        let s = self.lock()?;
        let sys = at(&s, &positions_bohr)?;
        Ok(flat(&s.coords.wilson_b(&sys).map_err(err)?))
    }

    /// `q(to) - q(from)`, with periodic coordinates wrapped.
    fn displacement(&self, to_bohr: Vec<f64>, from_bohr: Vec<f64>) -> PyResult<Vec<f64>> {
        let s = self.lock()?;
        let (to, from) = (at(&s, &to_bohr)?, at(&s, &from_bohr)?);
        Ok(list(&s.coords.displacement(&to, &from).map_err(err)?))
    }

    /// Cartesian displacement (Bohr) that realizes the step `dq` from `positions_bohr`
    /// (iterative back-transformation).
    fn cartesian_step(&self, positions_bohr: Vec<f64>, dq: Vec<f64>) -> PyResult<Vec<f64>> {
        let s = self.lock()?;
        let sys = at(&s, &positions_bohr)?;
        let step = Col::from_fn(dq.len(), |i| dq[i]);
        let dx = back_transform(
            s.coords.as_ref(),
            &sys,
            &step,
            &BackTransformOptions::default(),
        )
        .map_err(err)?;
        Ok(list(&dx))
    }

    /// Whether the coordinates should be set up anew at `positions_bohr` (an angle has become
    /// linear, a fragment has turned far from its reference orientation, ...).
    fn needs_reinitialization(&self, positions_bohr: Vec<f64>) -> PyResult<bool> {
        let s = self.lock()?;
        let sys = at(&s, &positions_bohr)?;
        Ok(s.coords.needs_reinitialization(&sys))
    }

    /// Set the coordinates up anew at `positions_bohr`.
    fn reinitialize(&self, positions_bohr: Vec<f64>) -> PyResult<()> {
        let mut s = self.lock()?;
        let sys = at(&s, &positions_bohr)?;
        s.coords.initialize(&sys).map_err(err)?;
        s.template = sys;
        Ok(())
    }
}

/// Names of the coordinate systems.
#[pyfunction]
fn coordinate_names() -> Vec<String> {
    CoordinateRegistry::default()
        .names()
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// The hyperspherical coordinates `(r, q_1, ..., q_(n-1))` of the point `x` (`n >= 2`):
/// `x_0 = r cos q_1`, `x_a = r sin q_1 ... sin q_a cos q_(a+1)`, the last component with the
/// sine of the last angle (MultiOptPy's `cart2polar`).
#[pyfunction]
fn to_hyperspherical(x: Vec<f64>) -> PyResult<Vec<f64>> {
    hyperspherical::to_hyperspherical(&x).map_err(err)
}

/// The point of the hyperspherical coordinates `q` (MultiOptPy's `polar2cart`).
#[pyfunction]
fn from_hyperspherical(q: Vec<f64>) -> PyResult<Vec<f64>> {
    hyperspherical::from_hyperspherical(&q).map_err(err)
}

/// `dx/dq` at the hyperspherical coordinates `q`, flat and row-major (`n x n`).
#[pyfunction]
fn hyperspherical_jacobian(q: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(flat(&hyperspherical::jacobian(&q).map_err(err)?))
}

/// `dq/dx` at the point `x`, flat and row-major (`n x n`).
#[pyfunction]
fn hyperspherical_wilson(x: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(flat(&hyperspherical::wilson(&x).map_err(err)?))
}

/// The gradient with respect to the hyperspherical coordinates of `x` from the one with
/// respect to `x` (MultiOptPy's `cart_grad_2_polar_grad`).
#[pyfunction]
fn gradient_to_hyperspherical(x: Vec<f64>, gradient: Vec<f64>) -> PyResult<Vec<f64>> {
    hyperspherical::gradient_to_hyperspherical(&x, &gradient).map_err(err)
}

/// Superpose `mobile_bohr` on `reference_bohr` (Kabsch): the moved positions and the RMSD
/// (Bohr); `mass_weighted` weights the atoms by their masses.
#[pyfunction]
#[pyo3(signature = (symbols, mobile_bohr, reference_bohr, mass_weighted = false))]
fn align(
    symbols: Vec<String>,
    mobile_bohr: Vec<f64>,
    reference_bohr: Vec<f64>,
    mass_weighted: bool,
) -> PyResult<(Vec<f64>, f64)> {
    let mobile = system(&symbols, &mobile_bohr, 0, 1, None)?;
    let reference = system(&symbols, &reference_bohr, 0, 1, None)?;
    let (moved, rmsd) = kabsch_align_bohr(&mobile, &reference, mass_weighted).map_err(err)?;
    Ok((list(&moved.coords_flat_bohr()), rmsd))
}

fn one_based(atoms: &[AtomId]) -> Vec<usize> {
    atoms.iter().map(|a| a.index() + 1).collect()
}

/// Bond graph of a structure: `bonds`, `angles`, `dihedrals` (atoms numbered from 1) and the
/// bonded `fragments`; two atoms are bonded within `scale` times the sum of their covalent
/// radii.
#[pyfunction]
#[pyo3(signature = (symbols, positions_bohr, scale = 1.3, cell_bohr = None))]
fn connectivity<'py>(
    py: Python<'py>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    scale: f64,
    cell_bohr: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let conn = BondConnectivity::from_system(&sys, scale);
    let d = PyDict::new(py);
    let rows = |v: &[[AtomId; 2]]| v.iter().map(|a| one_based(a)).collect::<Vec<_>>();
    d.set_item("bonds", rows(&conn.bonds))?;
    d.set_item(
        "angles",
        conn.angles.iter().map(|a| one_based(a)).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "dihedrals",
        conn.dihedrals
            .iter()
            .map(|a| one_based(a))
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "fragments",
        conn.fragments
            .iter()
            .map(|a| one_based(a))
            .collect::<Vec<_>>(),
    )?;
    Ok(d)
}

/// Names of the rules that choose the local minimum to sample next.
#[pyfunction]
fn sampling_priority_names() -> Vec<String> {
    PriorityRegistry::default()
        .names()
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// A rule that chooses the local minimum to sample next (`boltzmann`, `tabu`,
/// `water_filling`, ...); it keeps its own state between calls.
#[pyclass(name = "SamplingPriority")]
pub struct PySamplingPriority {
    inner: Mutex<(Box<dyn mopt_rs::workflow::explore::SamplingPriority>, Rng)>,
}

#[pymethods]
impl PySamplingPriority {
    #[new]
    #[pyo3(signature = (name, seed = 0))]
    fn new(name: &str, seed: u64) -> PyResult<Self> {
        let rule = PriorityRegistry::default().create(name).map_err(err)?;
        Ok(Self {
            inner: Mutex::new((rule, Rng::from_seed(seed))),
        })
    }

    /// Index (from 0) of the minimum to sample next among minima with `energies` (Hartree)
    /// that were sampled `visits` times, or `None` when the rule considers the search finished.
    fn select(&self, energies: Vec<f64>, visits: Vec<usize>) -> PyResult<Option<usize>> {
        if energies.is_empty() || energies.len() != visits.len() {
            return Err(err(mopt_rs::error::MoptError::config(
                "energies and visits must be non-empty lists of the same length",
            )));
        }
        let pool: Vec<EqRecord> = energies
            .iter()
            .zip(&visits)
            .map(|(e, v)| EqRecord {
                energy: Quantity::new(*e),
                visits: *v,
            })
            .collect();
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("sampling state is poisoned"))?;
        let (rule, rng) = &mut *guard;
        Ok(rule.select(&pool, rng))
    }
}

/// Electron-transfer rate at a crossing of two diabatic surfaces (Marcus theory with the
/// Landau-Zener transmission coefficient, as in the EDEEL scheme): Gibbs energies (Hartree) of
/// the crossing and of the initial and final states, and the coupling `v12` (Hartree). Returns
/// the reorganization energy, the barriers, the Landau-Zener probability, the transmission
/// coefficient, and the rate constant.
#[pyfunction]
#[pyo3(signature = (
    g_crossing,
    g_initial,
    g_final,
    v12,
    temperature = 298.15,
    intermolecular = true,
    nuclear_frequency = None,
    standard_pressure = 101_325.0,
))]
#[allow(clippy::too_many_arguments)]
fn electron_transfer_rate<'py>(
    py: Python<'py>,
    g_crossing: f64,
    g_initial: f64,
    g_final: f64,
    v12: f64,
    temperature: f64,
    intermolecular: bool,
    nuclear_frequency: Option<f64>,
    standard_pressure: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let cond = TransferConditions {
        temperature,
        standard_pressure,
        intermolecular,
        nuclear_frequency,
    };
    let r = edeel_rate(
        Quantity::new(g_crossing),
        Quantity::new(g_initial),
        Quantity::new(g_final),
        Quantity::new(v12),
        &cond,
    )
    .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("reorganization_energy", r.lambda.value())?;
    d.set_item("nonadiabatic_barrier", r.dg_nonadiabatic.value())?;
    d.set_item("barrier", r.dg_activation.value())?;
    d.set_item("landau_zener_probability", r.p_lz)?;
    d.set_item("transmission_coefficient", r.kappa_el)?;
    d.set_item("rate_constant", r.rate)?;
    Ok(d)
}
/// O1NumHess: the Hessian of a structure from a number of gradients that does not grow with
/// its size. `ask()` returns the displaced positions to evaluate (or `None` when the Hessian
/// is ready) and `tell()` takes the gradients there, in the same order.
#[pyclass(name = "O1NumHess")]
pub struct PyO1NumHess {
    inner: Mutex<O1NumHess>,
}

impl PyO1NumHess {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, O1NumHess>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("O1NumHess state is poisoned"))
    }
}

#[pymethods]
impl PyO1NumHess {
    #[new]
    #[pyo3(signature = (symbols, positions_bohr, gradient, *, cell_bohr = None, translation = true, rotation = None, delta_bohr = 0.005, central = false, follow_negative_modes = true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
        gradient: Vec<f64>,
        cell_bohr: Option<Vec<f64>>,
        translation: bool,
        rotation: Option<bool>,
        delta_bohr: f64,
        central: bool,
        follow_negative_modes: bool,
    ) -> PyResult<Self> {
        if gradient.len() != positions_bohr.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "gradient and positions differ in length",
            ));
        }
        let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
        let invariance = Invariance {
            translation,
            rotation: rotation.unwrap_or(translation && !sys.has_pbc()),
        };
        let g = Col::from_fn(gradient.len(), |i| gradient[i]);
        let settings = O1Settings {
            delta: delta_bohr,
            central,
            follow_negative_modes,
            ..O1Settings::default()
        };
        let problem = molecular_problem(&sys, &g, invariance).map_err(err)?;
        Ok(Self {
            inner: Mutex::new(O1NumHess::new(problem, settings).map_err(err)?),
        })
    }

    /// Positions (flat, Bohr) at which gradients are needed, or `None` when finished.
    fn ask(&self) -> PyResult<Option<Vec<Vec<f64>>>> {
        Ok(match self.lock()?.ask() {
            O1Action::Evaluate(points) => Some(points.iter().map(list).collect()),
            O1Action::Finished => None,
        })
    }

    /// The gradients (flat, Hartree/Bohr) at the positions of `ask()`.
    fn tell(&self, gradients: Vec<Vec<f64>>) -> PyResult<()> {
        let cols: Vec<Col<f64>> = gradients
            .iter()
            .map(|g| Col::from_fn(g.len(), |i| g[i]))
            .collect();
        self.lock()?.tell(&cols).map_err(err)
    }

    /// The Hessian (flat `3N x 3N`, Hartree/Bohr^2) once `ask()` returns `None`.
    fn hessian(&self) -> PyResult<Vec<f64>> {
        self.lock()?
            .hessian()
            .map(flat)
            .ok_or_else(|| PyRuntimeError::new_err("the Hessian is not finished: ask() first"))
    }

    /// The unit directions along which the gradient is differentiated (flat, `3N x count`).
    fn directions(&self) -> PyResult<Vec<f64>> {
        Ok(flat(&self.lock()?.directions()))
    }

    /// Gradients taken so far.
    #[getter]
    fn gradient_count(&self) -> PyResult<usize> {
        Ok(self.lock()?.gradient_count())
    }

    /// How well the Hessian reproduces the gradient derivatives (`|G - H D|`).
    #[getter]
    fn residual(&self) -> PyResult<f64> {
        Ok(self.lock()?.residual())
    }

    /// Remarks for the log.
    #[getter]
    fn notes(&self) -> PyResult<Vec<String>> {
        Ok(self.lock()?.notes().to_vec())
    }
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCoordinates>()?;
    m.add_class::<PySamplingPriority>()?;
    m.add_class::<PyO1NumHess>()?;
    m.add_function(wrap_pyfunction!(coordinate_names, m)?)?;
    m.add_function(wrap_pyfunction!(to_hyperspherical, m)?)?;
    m.add_function(wrap_pyfunction!(from_hyperspherical, m)?)?;
    m.add_function(wrap_pyfunction!(hyperspherical_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(hyperspherical_wilson, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_to_hyperspherical, m)?)?;
    m.add_function(wrap_pyfunction!(align, m)?)?;
    m.add_function(wrap_pyfunction!(connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(sampling_priority_names, m)?)?;
    m.add_function(wrap_pyfunction!(electron_transfer_rate, m)?)?;
    Ok(())
}
