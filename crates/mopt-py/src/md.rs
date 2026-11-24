//! Molecular dynamics from Python: the ask/tell state machine [`MolecularDynamics`] over the
//! Cartesian coordinates of a structure or `n` plain variables, with the thermostats,
//! barostats, constraints, and frozen modes of `mdmain`, and the names of its parts.
//!
//! MultiOptPy: `moleculardynamics.py:MD` and `Wrapper/md_wrapper.py:MDJob`.
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::{Col, Mat};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use mopt_rs::analysis::free_energy::{self, Profile, ProfileSettings, Window};
use mopt_rs::cli::constraints::{parse_projection_constraints, user_constraint};
use mopt_rs::cli::mdmain_coordinates::is_cartesian;
use mopt_rs::cli::mdmain_settings::{apply_md_options, ensemble};
use mopt_rs::coord::system::CoordinateRegistry;
use mopt_rs::domain::{AtomId, Units};
use mopt_rs::md::generalized::GeneralizedDynamics;
use mopt_rs::md::settings::kilopascal_to_atomic;
use mopt_rs::md::{
    BarostatRegistry, InitialVelocities, IntegratorRegistry, MdAction, MdMachine, MdSettings,
    MdSpace, MdStepReport, MdValue, MolecularDynamics, RigidMotion, ThermostatRegistry,
};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind};

use super::{cell_from, err, system};

fn column(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn list(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn report_dict<'py>(py: Python<'py>, r: &MdStepReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("step", r.step)?;
    d.set_item("time", r.time)?;
    d.set_item("time_fs", r.time_fs())?;
    d.set_item("x", list(&r.x))?;
    d.set_item("v", list(&r.v))?;
    let cell = r.cell.as_ref().map(|c| {
        (0..3)
            .flat_map(|k| c.lattice_vector(k).value())
            .collect::<Vec<f64>>()
    });
    d.set_item("cell_bohr", cell)?;
    d.set_item("energy", r.energy)?;
    d.set_item("raw_energy", r.raw_energy)?;
    d.set_item("kinetic", r.kinetic)?;
    d.set_item("total_energy", r.total_energy())?;
    d.set_item("temperature", r.temperature)?;
    d.set_item("target_temperature", r.target_temperature)?;
    d.set_item("dof", r.dof)?;
    d.set_item("conserved", r.conserved)?;
    d.set_item("pressure", r.pressure)?;
    d.set_item("volume", r.volume)?;
    d.set_item("constraint_residual", r.constraint_residual)?;
    match &r.freeze {
        Some(f) => {
            let fd = PyDict::new(py);
            fd.set_item("step", f.step)?;
            fd.set_item("eigenvalues", f.eigenvalues.clone())?;
            fd.set_item("overlaps", f.overlaps.clone())?;
            d.set_item("freeze", fd)?;
        }
        None => d.set_item("freeze", py.None())?,
    }
    d.set_item("notes", r.notes.clone())?;
    d.set_item("dipole", r.dipole.map(|m| m.to_vec()))?;
    d.set_item("heat_exchanged", r.heat_exchanged)?;
    d.set_item("electronic_state", r.electronic_state)?;
    d.set_item("state_energies", r.state_energies.clone())?;
    let hops = PyList::empty(py);
    for h in &r.hops {
        let hd = PyDict::new(py);
        hd.set_item("step", h.step)?;
        hd.set_item("time", h.time)?;
        hd.set_item("decided", h.decided)?;
        hd.set_item("from_state", h.from)?;
        hd.set_item("to_state", h.to)?;
        hd.set_item("gap", h.gap)?;
        hd.set_item("crossing", h.zn.map(|z| z.kind.name()))?;
        hd.set_item("allowed", h.zn.map(|z| z.allowed))?;
        hd.set_item("a2", h.zn.map(|z| z.a2))?;
        hd.set_item("b2", h.zn.map(|z| z.b2))?;
        hd.set_item("probability", h.zn.and_then(|z| z.probability))?;
        hd.set_item("transmission", h.zn.and_then(|z| z.transmission))?;
        hd.set_item("kinetic_along", h.kinetic_along)?;
        hd.set_item("random", h.random)?;
        hd.set_item("outcome", h.outcome.name())?;
        hops.append(hd)?;
    }
    d.set_item("hops", hops)?;
    Ok(d)
}

/// Ask/tell molecular dynamics. With `symbols` the variables are the Cartesian coordinates
/// (Bohr) of a structure with the atomic weights (or `masses` in amu), optionally in the
/// periodic cell `cell_bohr`; otherwise they are plain variables with `masses` (one each by
/// default). `ensemble` takes the names of `-mt` (`nve`, `nosehoover`, `langevin`, `npt`, ...),
/// `options` the `KEY=VALUE` settings of `-md_opt` (thermostat, tau in fs, seed, freeze, ...),
/// `time_step` is in atomic units of time, `pressure` in kPa. Constraints as in the
/// optimizer: `fix_atoms` (numbers from 1), `constraints` (the tokens of `-pc`),
/// `user_constraints` (`-uc`), `constraint_functions` (Python callables). With
/// `coordinates` (a name of `-coord`, such as `dic`) the dynamics of a structure runs in
/// those generalized coordinates (the generalized leapfrog; `invariant`: the surface does
/// not change under rigid motions, which internal coordinates need); positions, velocities,
/// and gradients stay Cartesian.
#[pyclass(name = "MolecularDynamics")]
pub struct PyMd {
    inner: Mutex<Machine>,
}

/// Cartesian dynamics or dynamics in generalized coordinates.
enum Machine {
    Cartesian(Box<MolecularDynamics>),
    Generalized(Box<GeneralizedDynamics>),
}

impl Machine {
    fn get(&self) -> &dyn MdMachine {
        match self {
            Self::Cartesian(m) => m.as_ref(),
            Self::Generalized(m) => m.as_ref(),
        }
    }

    fn get_mut(&mut self) -> &mut dyn MdMachine {
        match self {
            Self::Cartesian(m) => m.as_mut(),
            Self::Generalized(m) => m.as_mut(),
        }
    }

    fn part_names(&self) -> (String, String, Option<String>) {
        match self {
            Self::Cartesian(m) => m.part_names(),
            Self::Generalized(m) => {
                let (i, t) = m.part_names();
                (i, t, None)
            }
        }
    }

    fn references(&self) -> Vec<&'static str> {
        match self {
            Self::Cartesian(m) => m.references(),
            Self::Generalized(m) => m.references(),
        }
    }
}

impl PyMd {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Machine>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("the state of the dynamics is poisoned"))
    }
}

/// The constraints of the keyword arguments.
fn held(
    fix_atoms: Option<Vec<usize>>,
    constraints: Option<Vec<String>>,
    user_constraints: Option<Vec<String>>,
    constraint_functions: Option<Vec<Bound<'_, PyAny>>>,
) -> PyResult<Vec<Constraint>> {
    let mut out = Vec::new();
    if let Some(atoms) = fix_atoms.filter(|a| !a.is_empty()) {
        let ids = atoms
            .into_iter()
            .map(AtomId::from_one_based)
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        out.push(Constraint::at_start(ConstraintKind::Frozen(ids)));
    }
    if let Some(tokens) = constraints {
        out.extend(parse_projection_constraints(&tokens).map_err(err)?);
    }
    for source in user_constraints.unwrap_or_default() {
        out.push(user_constraint(&source).map_err(err)?);
    }
    for (k, item) in constraint_functions.unwrap_or_default().iter().enumerate() {
        out.push(super::constraint_function(k, item)?);
    }
    Ok(out)
}

#[pymethods]
impl PyMd {
    #[new]
    #[pyo3(signature = (
        x,
        *,
        symbols = None,
        masses = None,
        cell_bohr = None,
        charge = 0,
        multiplicity = 1,
        ensemble = "nve",
        time_step = None,
        steps = 1000,
        temperature = 298.15,
        pressure = 101.3,
        options = None,
        velocities = None,
        fix_atoms = None,
        constraints = None,
        user_constraints = None,
        constraint_functions = None,
        rigid = None,
        coordinates = None,
        invariant = true,
        electronic_state = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: Vec<f64>,
        symbols: Option<Vec<String>>,
        masses: Option<Vec<f64>>,
        cell_bohr: Option<Vec<f64>>,
        charge: i64,
        multiplicity: i64,
        ensemble: &str,
        time_step: Option<f64>,
        steps: usize,
        temperature: f64,
        pressure: f64,
        options: Option<Vec<String>>,
        velocities: Option<Vec<f64>>,
        fix_atoms: Option<Vec<usize>>,
        constraints: Option<Vec<String>>,
        user_constraints: Option<Vec<String>>,
        constraint_functions: Option<Vec<Bound<'_, PyAny>>>,
        rigid: Option<&str>,
        coordinates: Option<&str>,
        invariant: bool,
        electronic_state: Option<usize>,
    ) -> PyResult<Self> {
        let parts = mopt_rs::cli::mdmain_settings::ensemble(ensemble).map_err(err)?;
        let atoms = symbols.is_some();
        let default_step = if atoms {
            mopt_rs::md::settings::DEFAULT_TIME_STEP_FS * Units::FS_TO_AU_TIME
        } else {
            mopt_rs::md::settings::DEFAULT_VARIABLE_TIME_STEP
        };
        let mut settings = MdSettings {
            time_step: time_step.unwrap_or(default_step),
            steps,
            temperature,
            pressure: kilopascal_to_atomic(pressure),
            thermostat: parts.thermostat.to_string(),
            integrator: parts.integrator.to_string(),
            barostat: parts.barostat.map(str::to_string),
            constraints: held(
                fix_atoms,
                constraints,
                user_constraints,
                constraint_functions,
            )?,
            ..MdSettings::default()
        };
        let run = apply_md_options(&options.unwrap_or_default(), &mut settings).map_err(err)?;
        // The state surface hopping starts on (the lowest of hopping_states without it).
        if let Some(state) = electronic_state {
            let hopping = settings.hopping.as_mut().ok_or_else(|| {
                PyValueError::new_err(
                    "electronic_state is the starting state of surface hopping; give the states \
                     with options hopping_states=...",
                )
            })?;
            hopping.start = state;
            hopping.validate().map_err(err)?;
        }
        if let Some(v) = velocities {
            settings.velocities = InitialVelocities::Given(column(&v));
        }
        let (space, template) = match (&symbols, &masses) {
            (Some(symbols), _) => {
                let cell = cell_from(cell_bohr)?;
                let sys = system(symbols, &x, charge, multiplicity, cell)?;
                let amu = masses.clone().unwrap_or_else(|| sys.masses_amu());
                (MdSpace::atoms(&amu).map_err(err)?, Some(sys))
            }
            (None, Some(m)) => (MdSpace::variables_with_masses(m).map_err(err)?, None),
            (None, None) => (MdSpace::variables(x.len()).map_err(err)?, None),
        };
        settings.rigid = match rigid {
            Some(text) => RigidMotion::parse(text).map_err(err)?,
            None if !run.rigid_auto => settings.rigid,
            None => match &template {
                Some(t) if t.has_pbc() => RigidMotion::Translation,
                Some(_) => RigidMotion::TranslationRotation,
                None => RigidMotion::None,
            },
        };
        let machine = match coordinates.filter(|c| !is_cartesian(c)) {
            Some(name) => {
                let sys = template.ok_or_else(|| {
                    PyValueError::new_err("coordinates are those of a structure: give symbols")
                })?;
                let coords = CoordinateRegistry::default().create(name).map_err(err)?;
                Machine::Generalized(Box::new(
                    GeneralizedDynamics::new(settings, coords, sys, invariant).map_err(err)?,
                ))
            }
            None => Machine::Cartesian(Box::new(
                MolecularDynamics::new(settings, space, column(&x), template).map_err(err)?,
            )),
        };
        Ok(Self {
            inner: Mutex::new(machine),
        })
    }

    /// The point to evaluate next, or `None` when the run has ended: a dict with `x`,
    /// `cell_bohr`, `hessian` (whether the Hessian is asked for), `stress` (whether the
    /// stress is asked for), `step`, and `time` (atomic units).
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let request = match self.lock()?.get().ask() {
            MdAction::Finished(_) => return Ok(None),
            MdAction::Evaluate(r) => r,
        };
        let d = PyDict::new(py);
        d.set_item("x", list(&request.x))?;
        let cell = request.cell.as_ref().map(|c| {
            (0..3)
                .flat_map(|k| c.lattice_vector(k).value())
                .collect::<Vec<f64>>()
        });
        d.set_item("cell_bohr", cell)?;
        d.set_item("hessian", request.need_hessian)?;
        d.set_item("stress", request.need_stress)?;
        d.set_item("step", request.step)?;
        d.set_item("time", request.time)?;
        match &request.states {
            Some(states) => {
                let sd = PyDict::new(py);
                sd.set_item("active", states.active)?;
                sd.set_item("states", states.states.clone())?;
                sd.set_item("neighbours", states.neighbours())?;
                sd.set_item("gradient_gap", states.gradient_gap)?;
                sd.set_item("expected", states.expected.clone())?;
                d.set_item("states", sd)?;
            }
            None => d.set_item("states", py.None())?,
        }
        Ok(Some(d))
    }

    /// The surface at the point of `ask()`: energy and gradient (with bias potentials), the
    /// Hessian (flat, row-major) when it was asked for, the stress (nine numbers, Hartree /
    /// Bohr^3, ASE's sign) when it was asked for, and the energy without bias potentials.
    /// With surface hopping (`states` of `ask()`), `state_energies` are the energies of all
    /// the states in their order and `state_gradients` their gradients (`None` where not
    /// computed; those of the active state and of the neighbours closer than `gradient_gap`
    /// are needed). Returns the report of the step.
    #[pyo3(signature = (
        energy,
        gradient,
        hessian = None,
        *,
        raw_energy = None,
        stress = None,
        state_energies = None,
        state_gradients = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn tell<'py>(
        &self,
        py: Python<'py>,
        energy: f64,
        gradient: Vec<f64>,
        hessian: Option<Vec<f64>>,
        raw_energy: Option<f64>,
        stress: Option<Vec<f64>>,
        state_energies: Option<Vec<f64>>,
        state_gradients: Option<Vec<Option<Vec<f64>>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = gradient.len();
        let hessian = match hessian {
            Some(h) if h.len() != n * n => {
                return Err(PyValueError::new_err(format!(
                    "the Hessian has {} entries, expected {}",
                    h.len(),
                    n * n
                )))
            }
            Some(h) => Some(Mat::from_fn(n, n, |i, j| h[i * n + j])),
            None => None,
        };
        let stress = match stress {
            Some(s) if s.len() == 9 => Some(std::array::from_fn(|a| {
                std::array::from_fn(|b| s[3 * a + b])
            })),
            Some(_) => return Err(PyValueError::new_err("the stress has nine numbers")),
            None => None,
        };
        let mut value = MdValue::plain(energy, column(&gradient));
        value.raw_energy = raw_energy.unwrap_or(energy);
        value.hessian = hessian;
        value.stress = stress;
        if let Some(energies) = state_energies {
            let asked = match self.lock()?.get().ask() {
                MdAction::Evaluate(r) => r.states,
                MdAction::Finished(_) => None,
            };
            let states = asked
                .ok_or_else(|| PyValueError::new_err("state_energies without surface hopping"))?
                .states;
            let gradients = state_gradients.unwrap_or_else(|| vec![None; states.len()]);
            if energies.len() != states.len() || gradients.len() != states.len() {
                return Err(PyValueError::new_err(format!(
                    "{} states: {} energies and {} gradients",
                    states.len(),
                    energies.len(),
                    gradients.len()
                )));
            }
            value.states = Some(mopt_rs::md::hopping::StateValues {
                states,
                energies,
                gradients: gradients.iter().map(|g| g.as_deref().map(column)).collect(),
            });
        }
        let report = self.lock()?.get_mut().tell(value).map_err(err)?;
        report_dict(py, &report)
    }

    /// Degrees of freedom of the temperature (known after the first `tell`).
    #[getter]
    fn dof(&self) -> PyResult<f64> {
        Ok(self.lock()?.get().dof())
    }

    /// Whether the run has ended.
    #[getter]
    fn finished(&self) -> PyResult<bool> {
        Ok(self.lock()?.get().finished())
    }

    /// Values told so far.
    #[getter]
    fn evaluations(&self) -> PyResult<usize> {
        Ok(self.lock()?.get().evaluations())
    }

    /// Number of frozen modes.
    #[getter]
    fn frozen_modes(&self) -> PyResult<usize> {
        Ok(self.lock()?.get().frozen_modes())
    }

    /// Names of the integrator, the thermostat, and the barostat.
    #[getter]
    fn parts(&self) -> PyResult<(String, String, Option<String>)> {
        Ok(self.lock()?.part_names())
    }

    /// The papers of the parts.
    #[getter]
    fn references(&self) -> PyResult<Vec<String>> {
        Ok(self
            .lock()?
            .references()
            .into_iter()
            .map(str::to_string)
            .collect())
    }
}

/// The names of the integrators, thermostats, barostats, and of the ensembles of `-mt`.
#[pyfunction]
fn md_parts() -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let ensembles = [
        "nve",
        "nosehoover",
        "nosehooverchain",
        "langevin",
        "berendsen",
        "bussi",
        "andersen",
        "npt",
        "npt_berendsen",
        "npt_berendsen_aniso",
        "npt_crescale",
    ];
    (
        IntegratorRegistry::builtin().names(),
        ThermostatRegistry::builtin().names(),
        BarostatRegistry::builtin().names(),
        ensembles
            .iter()
            .filter(|e| ensemble(e).is_ok())
            .map(|e| (*e).to_string())
            .collect(),
    )
}

/// The windows and settings of a free-energy analysis.
fn analysis(
    centers: Vec<f64>,
    k: Vec<f64>,
    samples: Vec<Vec<f64>>,
    kt: f64,
    bins: usize,
    range: Option<(f64, f64)>,
    period: Option<f64>,
) -> PyResult<(Vec<Window>, ProfileSettings)> {
    if centers.len() != samples.len() || (k.len() != 1 && k.len() != centers.len()) {
        return Err(PyValueError::new_err(
            "one centre and one list of samples per window, and one force constant or one per window",
        ));
    }
    let windows = centers
        .iter()
        .zip(samples)
        .enumerate()
        .map(|(i, (c, s))| Window {
            center: *c,
            k: if k.len() == 1 { k[0] } else { k[i] },
            samples: s,
        })
        .collect();
    let settings = ProfileSettings {
        bins,
        range,
        period,
        ..ProfileSettings::at(kt)
    };
    Ok((windows, settings))
}

fn profile_dict<'py>(py: Python<'py>, p: &Profile) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("centers", p.centers.clone())?;
    d.set_item("free_energy", p.free_energy.clone())?;
    d.set_item("window_free_energies", p.window_free_energies.clone())?;
    d.set_item("iterations", p.iterations)?;
    Ok(d)
}

/// The free-energy profile (Hartree, 0 at its minimum) of umbrella windows by WHAM:
/// `centers` and `k` (one or one per window) of the restraints `k/2 (xi - c)^2`, the
/// `samples` of each window, `kt` (Hartree).
#[pyfunction]
#[pyo3(signature = (centers, k, samples, kt, bins = 50, range = None, period = None))]
#[allow(clippy::too_many_arguments)]
fn wham<'py>(
    py: Python<'py>,
    centers: Vec<f64>,
    k: Vec<f64>,
    samples: Vec<Vec<f64>>,
    kt: f64,
    bins: usize,
    range: Option<(f64, f64)>,
    period: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (windows, settings) = analysis(centers, k, samples, kt, bins, range, period)?;
    profile_dict(py, &free_energy::wham(&windows, &settings).map_err(err)?)
}

/// The same profile by MBAR.
#[pyfunction]
#[pyo3(signature = (centers, k, samples, kt, bins = 50, range = None, period = None))]
#[allow(clippy::too_many_arguments)]
fn mbar<'py>(
    py: Python<'py>,
    centers: Vec<f64>,
    k: Vec<f64>,
    samples: Vec<Vec<f64>>,
    kt: f64,
    bins: usize,
    range: Option<(f64, f64)>,
    period: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (windows, settings) = analysis(centers, k, samples, kt, bins, range, period)?;
    profile_dict(py, &free_energy::mbar(&windows, &settings).map_err(err)?)
}

/// Add the classes and functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMd>()?;
    m.add_function(wrap_pyfunction!(md_parts, m)?)?;
    m.add_function(wrap_pyfunction!(wham, m)?)?;
    m.add_function(wrap_pyfunction!(mbar, m)?)?;
    Ok(())
}
