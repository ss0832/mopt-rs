//! `mopt-py` — Python (`PyO3`) bindings for `mopt-rs`.
//!
//! The geometry optimizer is exposed through its ask/tell interface: `ask()` returns the
//! evaluations needed at the current structure, the Python side computes them with any
//! calculator, and `tell()` advances the optimizer. Arrays cross the boundary as flat lists of
//! floats in atomic units (Bohr, Hartree); `python/mopt_rs` wraps them in NumPy arrays. One copy
//! per evaluation is negligible next to an electronic-structure calculation.
//!
//! MultiOptPy: `Wrapper/optimize_wrapper.py` (Python access; here to every component).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod analysis;
mod bitss;
mod callback;
mod chain;
mod components;
mod network;
// The parts of the conformation search.
mod confsearch;
mod dmf;
mod electron_transfer;
mod ieip;
mod irc;
// The kinetics of networks (mopt_rs.kinetics).
mod kinetics;
mod mapper_plugins;
// The ONIOM calculator.
mod oniom;
// Molecular dynamics.
mod md;
// The orientation search of orientsearch.
mod orient;
// Trajectory analysis.
mod parts;
mod pycalc;
mod saddles;
// Scans of coordinates (relaxedscan) and ReSPer coordinates.
mod scan;
mod searches;
mod steps;
mod strings;
mod trajectory;
mod variables;
mod vib;
// Structure files (.xyz, extended .xyz).
mod xyz;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use faer::{Col, Mat};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::cli::constraints::resolve_for_run;
use mopt_rs::cli::constraints::{parse_projection_constraints, user_constraint};
use mopt_rs::cli::oniom_micro::{forces_of, verified_saddle_points, wrap_constrained};
use mopt_rs::cli::optmain::native_calculator_with;
use mopt_rs::domain::{
    AtomId, Bohr, Element, EvalOutput, EvalRequest, MolecularSystem, PeriodicCell, Quantity,
    SpinCharge, Units, Vector3,
};
use mopt_rs::error::MoptError;
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{Action, GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::opt::step::ccqn::{CcqnAxis, CcqnSettings};
use mopt_rs::opt::trust::TrustSettings;
use mopt_rs::pes::bias::Afir;
use mopt_rs::pes::combine::EvalTask;
use mopt_rs::pes::constraint::{Constraint, ConstraintFunction, ConstraintKind};
use mopt_rs::pes::term_check::check_term as check_pes_term;
use mopt_rs::pes::PesTerm;

use callback::PyCallbackTerm;

fn err(e: MoptError) -> PyErr {
    match e {
        MoptError::Config(_) | MoptError::Parse(_) | MoptError::Unsupported(_) => {
            PyValueError::new_err(e.to_string())
        }
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn cell_from(cell_bohr: Option<Vec<f64>>) -> PyResult<Option<PeriodicCell>> {
    cell_with_axes(cell_bohr, None)
}

fn cell_with_axes(
    cell_bohr: Option<Vec<f64>>,
    pbc: Option<[bool; 3]>,
) -> PyResult<Option<PeriodicCell>> {
    let Some(c) = cell_bohr else {
        if pbc.is_some_and(|axes| axes.contains(&true)) {
            return Err(PyValueError::new_err("periodic axes require cell_bohr"));
        }
        return Ok(None);
    };
    if c.len() != 9 {
        return Err(PyValueError::new_err(
            "cell_bohr must have 9 numbers (a, b, c rows)",
        ));
    }
    let v = |k: usize| Vector3::<Bohr>::new([c[3 * k], c[3 * k + 1], c[3 * k + 2]]);
    PeriodicCell::from_lattice_vectors(v(0), v(1), v(2), pbc.unwrap_or([true; 3]))
        .map(Some)
        .map_err(err)
}

/// The multiplicity from `multiplicity` (`2S + 1`) or `unpaired_electrons` (`2S`, the "spin" of
/// PySCF, xTB, and several Rust engines); 1 when neither is given, an error when they disagree.
fn multiplicity_of(multiplicity: Option<i64>, unpaired_electrons: Option<i64>) -> PyResult<i64> {
    match (multiplicity, unpaired_electrons) {
        (_, Some(u)) if u < 0 => Err(PyValueError::new_err(format!(
            "unpaired_electrons cannot be negative ({u})"
        ))),
        (Some(m), Some(u)) if m != u + 1 => Err(PyValueError::new_err(format!(
            "multiplicity {m} and unpaired_electrons {u} disagree (multiplicity = 2S + 1)"
        ))),
        (Some(m), _) => Ok(m),
        (None, Some(u)) => Ok(u + 1),
        (None, None) => Ok(1),
    }
}

fn system(
    symbols: &[String],
    positions_bohr: &[f64],
    charge: i64,
    multiplicity: i64,
    cell: Option<PeriodicCell>,
) -> PyResult<MolecularSystem> {
    let elements = symbols
        .iter()
        .map(|s| Element::parse_token(s))
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let flat = Col::from_fn(positions_bohr.len(), |i| positions_bohr[i]);
    let spin = SpinCharge::checked(charge, multiplicity).map_err(err)?;
    MolecularSystem::from_flat_bohr(elements, &flat, spin, cell).map_err(err)
}

/// `out` with the `<S^2>` a calculator told.
fn with_spin_square(mut out: EvalOutput, spin_square: Option<f64>) -> EvalOutput {
    if let Some(value) = spin_square {
        out.orbitals
            .get_or_insert_with(Default::default)
            .spin_square = Some(value);
    }
    out
}

fn output(energy: f64, gradient: Vec<f64>, hessian: Option<Vec<f64>>) -> PyResult<EvalOutput> {
    let n = gradient.len();
    let hessian = match hessian {
        Some(h) if h.len() != n * n => {
            return Err(PyValueError::new_err(format!(
                "Hessian has {} entries, expected {}",
                h.len(),
                n * n
            )))
        }
        Some(h) => Some(Mat::from_fn(n, n, |i, j| h[i * n + j])),
        None => None,
    };
    Ok(EvalOutput {
        energy: Quantity::new(energy),
        gradient_bohr: Col::from_fn(n, |i| gradient[i]),
        hessian_bohr: hessian,
        orbitals: None,
        stress: None,
        strain_hessian: None,
    })
}

fn task_dict<'py>(py: Python<'py>, t: &EvalTask) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let s = &t.system;
    d.set_item("label", &t.label)?;
    d.set_item("calculator", &t.calculator)?;
    d.set_item(
        "symbols",
        s.elements.iter().map(|e| e.symbol()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "positions_bohr",
        s.coords_flat_bohr().iter().copied().collect::<Vec<f64>>(),
    )?;
    d.set_item("charge", s.spin_charge.charge)?;
    d.set_item("multiplicity", s.spin_charge.multiplicity)?;
    d.set_item("need_hessian", t.request.need_hessian)?;
    d.set_item("excited_state", t.request.excited_state)?;
    let cell = s.cell.as_ref().map(|c| {
        (0..3)
            .flat_map(|k| c.lattice_vector(k).value())
            .collect::<Vec<f64>>()
    });
    d.set_item("cell_bohr", cell)?;
    d.set_item(
        "pbc",
        s.cell
            .as_ref()
            .map_or([false; 3], |c| c.periodic_axes)
            .to_vec(),
    )?;
    Ok(d)
}

fn termination_name(t: Termination) -> &'static str {
    match t {
        Termination::Converged => "converged",
        Termination::MaxIterations => "max_iterations",
        Termination::Dissociated => "dissociated",
        Termination::SeamCheckFailed => "seam_check_failed",
    }
}

/// Ask/tell geometry optimizer.
#[pyclass(name = "Optimizer")]
pub struct PyOptimizer {
    inner: Mutex<GeometryOptimizer>,
    /// Bias terms added so far, per option (for MultiOptPy-style labels).
    bias_counts: Mutex<HashMap<String, usize>>,
    /// `model_hessian` named O1NumHess: the Hessians asked for are meant to come from it.
    o1numhess: bool,
    /// The initial Hessian given as a matrix or a file (`initial_hessian`), handed to the
    /// optimizer as the answer to its first request for a Hessian, as `optmain -ih` does.
    first_hessian: Mutex<FirstHessian>,
    /// The surface the optimizer was made with (`calculator=`, an ONIOM calculator).
    own: Option<OwnSurface>,
}

/// The surface of an optimizer made with an ONIOM calculator: that of the core with the
/// environment relaxed (or the whole ONIOM surface with `micro=off`), as `optmain -oniom`
/// sets it up (`cli::oniom_micro::wrap_constrained`).
struct OwnSurface {
    /// The structure the optimizer moves (the core).
    template: MolecularSystem,
    calc: Box<dyn mopt_rs::pes::Calculator>,
    micro: Option<Arc<mopt_rs::pes::oniom::micro::RelaxedEnvironment>>,
    note: Option<String>,
}

/// The initial Hessian of an optimizer until the optimizer has it: the answer to the first
/// request for a Hessian, as `pes::given_hessian::GivenHessian` gives it to `optmain -ih`
/// (the evaluation is asked for without the Hessian, which is added in `tell`).
#[derive(Default)]
struct FirstHessian {
    /// The matrix (Hartree/Bohr^2), until it is handed over.
    matrix: Option<Mat<f64>>,
    /// The task of the last `ask` that gets it.
    answered: Option<usize>,
}

/// What `initial_hessian` of `Optimizer` names: a model Hessian (the names of `-modelhess`),
/// or a matrix from an array or a file (the files of `-ih`).
enum InitialHessian {
    Model(String),
    Matrix(Mat<f64>),
}

/// `initial_hessian` for the structure `start`: an existing file (read as `optmain -ih` reads
/// it), a model Hessian name, or an array of `3N x 3N` numbers (nested or flat).
fn initial_hessian_of(
    item: &Bound<'_, PyAny>,
    start: &MolecularSystem,
) -> PyResult<InitialHessian> {
    use mopt_rs::pes::given_hessian::{checked_hessian, read_for};
    let path = if let Ok(text) = item.extract::<String>() {
        let path = std::path::PathBuf::from(&text);
        if !path.is_file() {
            let known = mopt_rs::pes::o1numhess::is_o1numhess(&text)
                || mopt_rs::model_hessian::resolve(&text).is_ok();
            if known {
                return Ok(InitialHessian::Model(text));
            }
            return Err(PyValueError::new_err(format!(
                "initial_hessian: '{text}' is neither a file nor a model Hessian (the names of \
                 -modelhess)"
            )));
        }
        Some(path)
    } else if item.hasattr("__fspath__")? {
        Some(std::path::PathBuf::from(
            item.py()
                .import("os")?
                .getattr("fspath")?
                .call1((item,))?
                .extract::<String>()?,
        ))
    } else {
        None
    };
    let matrix = match path {
        Some(path) => read_for(&path, start).map_err(err)?,
        None => {
            let values = if item.hasattr("tolist")? {
                item.call_method0("tolist")?
            } else {
                item.clone()
            };
            let n = start.dof();
            let flat: Vec<f64> = match values.extract::<Vec<Vec<f64>>>() {
                Ok(rows) => {
                    if rows.len() != n || rows.iter().any(|r| r.len() != n) {
                        return Err(PyValueError::new_err(format!(
                            "initial_hessian: a {n} x {n} matrix for {} atoms",
                            start.n_atoms()
                        )));
                    }
                    rows.concat()
                }
                Err(_) => values.extract::<Vec<f64>>()?,
            };
            if flat.len() != n * n {
                return Err(PyValueError::new_err(format!(
                    "initial_hessian: {} numbers for a {n} x {n} matrix ({} atoms)",
                    flat.len(),
                    start.n_atoms()
                )));
            }
            Mat::from_fn(n, n, |i, j| flat[i * n + j])
        }
    };
    checked_hessian(matrix)
        .map(InitialHessian::Matrix)
        .map_err(|e| err(e.context("initial_hessian")))
}

/// The constraints of the keyword arguments `fix_atoms` (atoms from 1, `-fix`),
/// `constraints` (the tokens of `-pc`), `user_constraints` (the expressions of `-uc`), and
/// `constraint_functions` (Python callables): the same for `Optimizer`, `Irc`, and
/// `relax_band`.
pub(crate) fn held_constraints(
    fix_atoms: Option<Vec<usize>>,
    constraints: Option<Vec<String>>,
    user_constraints: Option<Vec<String>>,
    constraint_functions: Option<Vec<Bound<'_, PyAny>>>,
) -> PyResult<Vec<Constraint>> {
    let mut held = Vec::new();
    if let Some(fixed) = fix_atoms.filter(|f| !f.is_empty()) {
        let atoms = fixed
            .into_iter()
            .map(AtomId::from_one_based)
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        held.push(Constraint::at_start(ConstraintKind::Frozen(atoms)));
    }
    if let Some(tokens) = constraints {
        held.extend(parse_projection_constraints(&tokens).map_err(err)?);
    }
    for source in user_constraints.unwrap_or_default() {
        held.push(user_constraint(&source).map_err(err)?);
    }
    for (k, item) in constraint_functions.unwrap_or_default().iter().enumerate() {
        held.push(constraint_function(k, item)?);
    }
    Ok(held)
}

impl PyOptimizer {
    fn push_term(&self, key: &str, prefix: &str, term: Arc<dyn PesTerm>) -> PyResult<()> {
        let mut counts = self
            .bias_counts
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer lock poisoned"))?;
        let k = counts.entry(key.to_string()).or_insert(0);
        self.lock()?
            .add_term(&format!("{prefix}_{k}"), term)
            .map_err(err)?;
        *k += 1;
        Ok(())
    }

    fn lock(&self) -> PyResult<MutexGuard<'_, GeometryOptimizer>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("optimizer state is poisoned"))
    }
}

/// A constraint from a Python callable (same protocol as `add_term`; its "energy" is the
/// constrained value) or a mapping `{"function", "target", "period", "fixes_frame"}`.
pub(crate) fn constraint_function(k: usize, item: &Bound<'_, PyAny>) -> PyResult<Constraint> {
    let label = format!("constraint_function_{k}");
    let (function, target, period, fixes_frame) = if item.is_callable() {
        (item.clone(), None, None, false)
    } else {
        let get = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            match item.get_item(key) {
                Ok(v) if v.is_none() => Ok(None),
                Ok(v) => Ok(Some(v)),
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyKeyError>(item.py()) => Ok(None),
                Err(e) => Err(e),
            }
        };
        let function = get("function")?
            .filter(|f| f.is_callable())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{label}: give a callable or a mapping with a callable 'function'"
                ))
            })?;
        let target = get("target")?.map(|v| v.extract::<f64>()).transpose()?;
        let period = get("period")?.map(|v| v.extract::<f64>()).transpose()?;
        let fixes = get("fixes_frame")?
            .map(|v| v.extract::<bool>())
            .transpose()?
            .unwrap_or(false);
        (function, target, period, fixes)
    };
    Ok(Constraint {
        kind: ConstraintKind::Function(ConstraintFunction {
            term: Arc::new(PyCallbackTerm::new(&label, function.unbind())?),
            period,
            fixes_frame,
            tolerance: None,
            warns_only: false,
        }),
        target,
        anchor: None,
    })
}

#[pymethods]
impl PyOptimizer {
    /// Create an optimizer for `symbols` at `positions_bohr` (flat, 3N).
    #[new]
    #[pyo3(signature = (
        symbols,
        positions_bohr,
        *,
        charge = 0,
        multiplicity = None,
        unpaired_electrons = None,
        methods = Vec::new(),
        saddle_order = 0,
        max_iterations = 1000,
        criteria = "normal",
        coordinates = "cart",
        exact_hessian_interval = None,
        max_trust_radius = None,
        min_trust_radius = 0.01,
        rigid_body_invariant = true,
        dissociation_distance_angstrom = Some(10.0),
        cell_bohr = None,
        ccqn_bonds = None,
        ccqn_product_bohr = None,
        ccqn_cos_phi = None,
        fix_atoms = None,
        constraints = None,
        user_constraints = None,
        constraint_functions = None,
        model_hessian = None,
        model_hessian_interval = 50,
        mode_kill = false,
        crossing_check = true,
        projection_trust = false,
        step_factor = None,
        fragments = None,
        pbc = None,
        model_function = None,
        excited_state = 0,
        initial_hessian = None,
        calculator = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
        charge: i64,
        multiplicity: Option<i64>,
        unpaired_electrons: Option<i64>,
        methods: Vec<String>,
        saddle_order: usize,
        max_iterations: usize,
        criteria: &str,
        coordinates: &str,
        exact_hessian_interval: Option<usize>,
        max_trust_radius: Option<f64>,
        min_trust_radius: f64,
        rigid_body_invariant: bool,
        dissociation_distance_angstrom: Option<f64>,
        cell_bohr: Option<Vec<f64>>,
        ccqn_bonds: Option<Vec<(usize, usize)>>,
        ccqn_product_bohr: Option<Vec<f64>>,
        ccqn_cos_phi: Option<f64>,
        fix_atoms: Option<Vec<usize>>,
        constraints: Option<Vec<String>>,
        user_constraints: Option<Vec<String>>,
        constraint_functions: Option<Vec<Bound<'_, PyAny>>>,
        model_hessian: Option<String>,
        model_hessian_interval: usize,
        mode_kill: bool,
        crossing_check: bool,
        projection_trust: bool,
        step_factor: Option<f64>,
        fragments: Option<Vec<Vec<usize>>>,
        pbc: Option<[bool; 3]>,
        model_function: Option<Vec<String>>,
        excited_state: usize,
        initial_hessian: Option<Bound<'_, PyAny>>,
        calculator: Option<PyRef<'_, oniom::PyOniomCalculator>>,
    ) -> PyResult<Self> {
        let cell = cell_with_axes(cell_bohr, pbc)?;
        let multiplicity = multiplicity_of(multiplicity, unpaired_electrons)?;
        let mut start = system(&symbols, &positions_bohr, charge, multiplicity, cell)?;
        let criteria = match criteria.to_ascii_lowercase().as_str() {
            "normal" => ConvergenceCriteria::NORMAL,
            "tight" => ConvergenceCriteria::TIGHT,
            "loose" => ConvergenceCriteria::LOOSE,
            other => {
                return Err(PyValueError::new_err(format!(
                    "criteria must be normal, tight, or loose; got '{other}'"
                )))
            }
        };
        let ccqn = match (ccqn_bonds, ccqn_product_bohr) {
            (None, None) => None,
            (Some(bonds), None) => {
                let bonds = bonds
                    .into_iter()
                    .map(|(a, b)| Ok((AtomId::from_one_based(a)?, AtomId::from_one_based(b)?)))
                    .collect::<Result<Vec<_>, MoptError>>()
                    .map_err(err)?;
                Some(CcqnSettings::new(CcqnAxis::ReactiveBonds(bonds)))
            }
            (None, Some(p)) => {
                let product = start
                    .with_flat_coords_bohr(&Col::from_fn(p.len(), |i| p[i]))
                    .map_err(err)?;
                Some(CcqnSettings::new(
                    CcqnAxis::linear_midpoint(&start, &product).map_err(err)?,
                ))
            }
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "give either ccqn_bonds or ccqn_product_bohr",
                ))
            }
        };
        let ccqn = ccqn.map(|mut c| {
            if let Some(cp) = ccqn_cos_phi {
                c.cos_phi = cp;
            }
            c
        });
        let mut held = held_constraints(
            fix_atoms,
            constraints,
            user_constraints,
            constraint_functions,
        )?;
        let model_function = model_function.unwrap_or_default();
        // Atom numbers starting at 1, as on the command line (-of).
        let mut fragments = fragments
            .unwrap_or_default()
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(mopt_rs::domain::AtomId::from_one_based)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        // An ONIOM calculator: the surface of optmain -oniom, with the constraints resolved
        // at the input and taken to the core of the microiterations.
        let whole_start = start.clone();
        let mut rigid_body_invariant = rigid_body_invariant;
        let mut own = None;
        if let Some(oniom) = &calculator {
            if ccqn.is_some() {
                return Err(PyValueError::new_err(
                    "ccqn_bonds and ccqn_product_bohr with the ONIOM calculator are not \
                     available from Python (optmain -oniom takes them)",
                ));
            }
            let uncovered = model_function
                .first()
                .is_some_and(|name| name.eq_ignore_ascii_case("bitss"))
                .then_some("-mf bitss (its two images are not set up on the relaxed surface)");
            let resolved = resolve_for_run(&held, &start).map_err(err)?;
            let others: Vec<usize> = fragments.iter().flatten().map(|a| a.index()).collect();
            let registry = pycalc::python_registry();
            let wrapped = wrap_constrained(
                &oniom.options,
                uncovered,
                &resolved,
                &[],
                &others,
                forces_of(&criteria),
                start.clone(),
                &oniom.name,
                &registry,
            )
            .map_err(err)?;
            fragments = fragments
                .iter()
                .map(|group| group.iter().map(|a| wrapped.on_core(*a)).collect())
                .collect::<Result<Vec<Vec<AtomId>>, _>>()
                .map_err(err)?;
            rigid_body_invariant &= wrapped.calc.is_rigid_body_invariant();
            held = wrapped.constraints;
            start = wrapped.start;
            own = Some(OwnSurface {
                template: start.clone(),
                calc: wrapped.calc,
                micro: wrapped.micro,
                note: wrapped.note,
            });
        }
        // -ih: a model Hessian by its name, or the matrix for the first request.
        let (model_hessian, given) = match initial_hessian.as_ref() {
            None => (model_hessian, None),
            Some(item) => {
                if !model_function.is_empty() {
                    return Err(PyValueError::new_err(
                        "initial_hessian gives the Hessian of one state; it is not available \
                         with model_function (as -ih is not with -mf)",
                    ));
                }
                let micro = own.as_ref().and_then(|o| o.micro.as_deref());
                let found = match (micro, initial_hessian_of(item, &whole_start)) {
                    // The Hessian of the whole system, reduced to the core (optmain -ih).
                    (Some(relaxed), Ok(InitialHessian::Matrix(m))) if start.dof() != m.nrows() => {
                        InitialHessian::Matrix(relaxed.core_hessian_from(&m).map_err(err)?)
                    }
                    _ => initial_hessian_of(item, &start)?,
                };
                match found {
                    InitialHessian::Model(name) => match model_hessian {
                        Some(other) if !other.eq_ignore_ascii_case(&name) => {
                            return Err(PyValueError::new_err(format!(
                                "initial_hessian '{name}' and model_hessian '{other}' name two \
                                 model Hessians; give one"
                            )))
                        }
                        _ => (Some(name), None),
                    },
                    InitialHessian::Matrix(matrix) => (model_hessian, Some(matrix)),
                }
            }
        };
        let mut settings = OptimizerSettings {
            methods,
            saddle_order,
            max_iterations,
            criteria,
            trust: TrustSettings {
                max: max_trust_radius.map(Quantity::<Bohr>::new),
                min: Quantity::new(min_trust_radius),
            },
            exact_hessian_interval: exact_hessian_interval.filter(|&n| n > 0),
            coordinates: coordinates.to_string(),
            rigid_body_invariant,
            dissociation_distance: dissociation_distance_angstrom
                .map(|d| Quantity::<Bohr>::new(d * Units::ANGSTROM_TO_BOHR)),
            ccqn,
            constraints: held,
            model_hessian,
            model_hessian_interval,
            mode_kill: mode_kill.then(mopt_rs::opt::mode_kill::ModeKillSettings::default),
            crossing_check: crossing_check
                .then(mopt_rs::opt::crossing_check::CrossingCheckSettings::default),
            projection_trust,
            step_factor,
            fragments,
        };
        if let Some(note) =
            verified_saddle_points(&mut settings, own.as_ref().and_then(|o| o.micro.as_deref()))
        {
            if let Some(o) = own.as_mut() {
                o.note = Some(match o.note.take() {
                    Some(before) => format!("{before}\n{note}"),
                    None => note,
                });
            }
        }
        let o1numhess = settings.exact_hessian_interval.is_none()
            && settings
                .model_hessian
                .as_deref()
                .is_some_and(mopt_rs::pes::o1numhess::is_o1numhess);
        if given.is_some() && settings.exact_hessian_interval.is_none() {
            // As optmain -ih: a Hessian is asked for at the first point only, and it is the
            // one given.
            settings.exact_hessian_interval =
                Some(mopt_rs::cli::optmain::HESSIAN_AT_THE_START_ONLY);
        }
        let mut inner = GeometryOptimizer::new(settings, start).map_err(err)?;
        if let Some(surface) =
            mopt_rs::cli::optmain::state_surface(&model_function, excited_state).map_err(err)?
        {
            inner.set_surface(surface).map_err(err)?;
        }
        Ok(Self {
            inner: Mutex::new(inner),
            bias_counts: Mutex::new(HashMap::new()),
            o1numhess,
            first_hessian: Mutex::new(FirstHessian {
                matrix: given,
                answered: None,
            }),
            own,
        })
    }

    /// Add an AFIR term between 1-based atom lists (`gamma` in kJ/mol).
    #[pyo3(signature = (gamma, fragment_a, fragment_b, gamma_final = None))]
    fn add_afir(
        &mut self,
        gamma: f64,
        fragment_a: Vec<usize>,
        fragment_b: Vec<usize>,
        gamma_final: Option<f64>,
    ) -> PyResult<()> {
        let ids = |v: Vec<usize>| {
            v.into_iter()
                .map(AtomId::from_one_based)
                .collect::<Result<Vec<_>, _>>()
        };
        let mut term = Afir::new(
            gamma,
            ids(fragment_a).map_err(err)?,
            ids(fragment_b).map_err(err)?,
        )
        .map_err(err)?;
        if let Some(g) = gamma_final {
            term = term.ramp_to(g);
        }
        self.push_term("manual_AFIR", "AFIR_pot", Arc::new(term))
    }

    /// Add bias potentials given as a MultiOptPy option and its arguments, e.g.
    /// `add_bias("-kp", ["0.5", "1.2", "1,2"])` (same units and grouping as the command line);
    /// returns the number of terms added.
    fn add_bias(&self, option: &str, args: Vec<String>) -> PyResult<usize> {
        let specs = parse_bias_option(option, &args).map_err(err)?;
        let terms = {
            let opt = self.lock()?;
            specs
                .iter()
                .map(|s| s.build(opt.initial()).map(|t| (s.key.clone(), t)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(err)?
        };
        let n = terms.len();
        for (key, term) in terms {
            let prefix = if key == "manual_AFIR" {
                "AFIR_pot"
            } else {
                key.as_str()
            };
            self.push_term(&key, prefix, term)?;
        }
        Ok(n)
    }

    /// Add a surface term implemented in Python:
    /// `callback(positions_bohr, cell_bohr, iteration, need_hessian)` returns
    /// `(energy, gradient, hessian or None)` in atomic units (flat lists or NumPy arrays). A
    /// missing Hessian is built from central differences of the callback's gradient.
    /// Optional attributes of the callable say what the term depends on: `frame`
    /// (`"invariant"`, `"translation_only"`, or `"absolute"`, the default) and `history`
    /// (`"static"`, `"iteration_ramp"`, `"time_dependent"`, or `"stateful"`, the default). An
    /// optimizer that removes rigid-body motions (`rigid_body_invariant=True`, the default)
    /// takes only terms with `frame = "invariant"`; create it with
    /// `rigid_body_invariant=False` for the others.
    fn add_term(&self, label: &str, callback: Py<PyAny>) -> PyResult<()> {
        let term = PyCallbackTerm::new(label, callback)?;
        self.lock()?.add_term(label, Arc::new(term)).map_err(err)
    }

    /// Evaluations needed now (list of dicts), or `None` when the optimization has finished.
    /// With an initial Hessian (`initial_hessian`, a matrix or a file), the first evaluation
    /// that asks for a Hessian asks for the energy and the gradient alone: `tell` adds the
    /// Hessian given.
    fn ask<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyDict>>>> {
        let action = self.lock()?.ask();
        match action {
            Action::Finished(_) => Ok(None),
            Action::Evaluate(tasks) => {
                let mut first = self
                    .first_hessian
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("optimizer state is poisoned"))?;
                first.answered = first.matrix.as_ref().and_then(|h| {
                    tasks
                        .iter()
                        .position(|t| t.request.need_hessian && t.system.dof() == h.nrows())
                });
                tasks
                    .iter()
                    .enumerate()
                    .map(|(k, t)| {
                        let d = task_dict(py, t)?;
                        if first.answered == Some(k) {
                            d.set_item("need_hessian", false)?;
                        }
                        Ok(d)
                    })
                    .collect::<PyResult<Vec<_>>>()
                    .map(Some)
            }
        }
    }

    /// Results of the evaluations from `ask()`, in order: `(energy, gradient, hessian or None)`
    /// in Hartree, Hartree/Bohr, Hartree/Bohr^2 (flat lists). `spin_squares`: `<S^2>` of
    /// each evaluation where the calculator tells it (`None` otherwise). Returns an
    /// iteration report.
    #[pyo3(signature = (results, spin_squares = None))]
    fn tell<'py>(
        &mut self,
        py: Python<'py>,
        results: Vec<(f64, Vec<f64>, Option<Vec<f64>>)>,
        spin_squares: Option<Vec<Option<f64>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let spins = spin_squares.unwrap_or_else(|| vec![None; results.len()]);
        if spins.len() != results.len() {
            return Err(PyValueError::new_err(format!(
                "{} values of <S^2> for {} results",
                spins.len(),
                results.len()
            )));
        }
        let mut outs = results
            .into_iter()
            .zip(spins)
            .map(|((e, g, h), s)| output(e, g, h).map(|out| with_spin_square(out, s)))
            .collect::<PyResult<Vec<_>>>()?;
        {
            let mut first = self
                .first_hessian
                .lock()
                .map_err(|_| PyRuntimeError::new_err("optimizer state is poisoned"))?;
            if let Some(k) = first.answered.take() {
                if let Some(out) = outs.get_mut(k) {
                    out.hessian_bohr = first.matrix.take();
                }
            }
        }
        let inner = &self.inner;
        let r = py
            .allow_threads(|| {
                inner
                    .lock()
                    .map_err(|_| MoptError::config("optimizer state is poisoned"))
                    .and_then(|mut opt| opt.tell(outs))
            })
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("iteration", r.iteration)?;
        d.set_item("energy", r.energy.value())?;
        d.set_item("biased_energy", r.biased_energy.value())?;
        d.set_item("max_force", r.convergence.max_force.value())?;
        d.set_item("rms_force", r.convergence.rms_force.value())?;
        d.set_item("max_displacement", r.convergence.max_displacement.value())?;
        d.set_item("rms_displacement", r.convergence.rms_displacement.value())?;
        d.set_item("converged", r.convergence.converged)?;
        d.set_item(
            "spin_squares",
            r.spin_squares
                .iter()
                .map(|s| (s.label.clone(), s.value, s.exact))
                .collect::<Vec<_>>(),
        )?;
        d.set_item("step_norm", r.step_norm.value())?;
        d.set_item(
            "positions_bohr",
            r.system
                .coords_flat_bohr()
                .iter()
                .copied()
                .collect::<Vec<f64>>(),
        )?;
        d.set_item(
            "components",
            r.components
                .iter()
                .map(|(k, v)| (k.clone(), v.value()))
                .collect::<Vec<_>>(),
        )?;
        d.set_item("notes", r.notes)?;
        Ok(d)
    }

    /// Whether the optimizer was made with a calculator of its own (`calculator=`), which
    /// `evaluate_own` evaluates and `mopt_rs.optimize.run` uses.
    #[getter]
    fn has_own_calculator(&self) -> bool {
        self.own.is_some()
    }

    /// What the set-up of its own calculator said (the microiterations), or `None`.
    #[getter]
    fn calculator_note(&self) -> Option<String> {
        self.own.as_ref().and_then(|o| o.note.clone())
    }

    /// `(energy, gradient, hessian or None)` of the calculator the optimizer was made with
    /// at `positions_bohr` (the structure the optimizer moves: the core with ONIOM
    /// microiterations), as for a task of `ask()`.
    #[pyo3(signature = (positions_bohr, need_hessian = false, excited_state = 0))]
    fn evaluate_own(
        &self,
        py: Python<'_>,
        positions_bohr: Vec<f64>,
        need_hessian: bool,
        excited_state: usize,
    ) -> PyResult<NativeResult> {
        let own = self
            .own
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the optimizer has no calculator of its own"))?;
        let sys = own
            .template
            .with_flat_coords_bohr(&Col::from_fn(positions_bohr.len(), |i| positions_bohr[i]))
            .map_err(err)?;
        evaluate_flat(py, own.calc.as_ref(), &sys, need_hessian, excited_state)
    }

    /// The current structure as a whole system (with ONIOM microiterations the core with its
    /// relaxed environment), flat, in Bohr.
    #[getter]
    fn whole_positions_bohr(&self) -> PyResult<Vec<f64>> {
        let current = self.lock()?.current().clone();
        let whole = match self.own.as_ref().and_then(|o| o.micro.as_ref()) {
            Some(relaxed) => relaxed.full_system(&current).map_err(err)?,
            None => current,
        };
        Ok(whole.coords_flat_bohr().iter().copied().collect())
    }

    /// `converged`, `max_iterations`, `dissociated`, or `None` while running.
    /// Whether the Hessians this optimizer asks for are meant to come from O1NumHess
    /// (`model_hessian="o1numhess"` without exact Hessians).
    #[getter]
    fn wants_o1numhess(&self) -> bool {
        self.o1numhess
    }

    #[getter]
    fn termination(&self) -> PyResult<Option<&'static str>> {
        Ok(self.lock()?.termination().map(termination_name))
    }

    /// Iterations performed.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        Ok(self.lock()?.iterations())
    }

    /// Current (or final) positions, flat, in Bohr.
    #[getter]
    fn positions_bohr(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .lock()?
            .current()
            .coords_flat_bohr()
            .iter()
            .copied()
            .collect())
    }
}

type NativeResult = (f64, Vec<f64>, Option<Vec<f64>>);

/// Evaluate `calc` at `sys` and flatten the result.
fn evaluate_flat(
    py: Python<'_>,
    calc: &dyn mopt_rs::pes::Calculator,
    sys: &MolecularSystem,
    need_hessian: bool,
    excited_state: usize,
) -> PyResult<NativeResult> {
    py.allow_threads(|| evaluate_native_flat(calc, sys, need_hessian, excited_state))
        .map_err(err)
}

fn evaluate_native_flat(
    calc: &dyn mopt_rs::pes::Calculator,
    sys: &MolecularSystem,
    need_hessian: bool,
    excited_state: usize,
) -> mopt_rs::error::MoptResult<NativeResult> {
    let mut req = if need_hessian {
        EvalRequest::ENERGY_GRAD_HESS
    } else {
        EvalRequest::ENERGY_GRAD
    };
    req.excited_state = excited_state;
    let out = mopt_rs::pes::evaluate_checked(calc, sys, &req)?;
    let h = out.hessian_bohr.map(|h| {
        let n = h.nrows();
        (0..n * n).map(|k| h[(k / n, k % n)]).collect()
    });
    Ok((
        out.energy.value(),
        out.gradient_bohr.iter().copied().collect(),
        h,
    ))
}

/// Evaluate a built-in calculator (`lj`, `emt`, `eam`, `meam`, `tersoff`, `muller_brown`,
/// `gfn0xtb`, `gfn1xtb`, `gfnff`)
/// once, made for this structure: returns `(energy, gradient, hessian or None)` as flat lists
/// in atomic units. [`PyNativeEvaluator`] keeps the calculator between evaluations.
#[pyfunction]
#[pyo3(signature = (name, symbols, positions_bohr, charge = 0, multiplicity = None, need_hessian = false, cell_bohr = None, unpaired_electrons = None, settings = None, *, pbc = None, excited_state = 0))]
#[allow(clippy::too_many_arguments)]
fn evaluate_native(
    py: Python<'_>,
    name: &str,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    charge: i64,
    multiplicity: Option<i64>,
    need_hessian: bool,
    cell_bohr: Option<Vec<f64>>,
    unpaired_electrons: Option<i64>,
    settings: Option<Vec<(String, String)>>,
    pbc: Option<[bool; 3]>,
    excited_state: usize,
) -> PyResult<NativeResult> {
    let multiplicity = multiplicity_of(multiplicity, unpaired_electrons)?;
    let sys = system(
        &symbols,
        &positions_bohr,
        charge,
        multiplicity,
        cell_with_axes(cell_bohr, pbc)?,
    )?;
    let calc = native_calculator_with(
        &name.to_ascii_lowercase(),
        &sys,
        &settings.unwrap_or_default(),
    )
    .map_err(err)?;
    evaluate_flat(py, calc.as_ref(), &sys, need_hessian, excited_state)
}

/// One set of orbitals as flat lists: coefficients (row by row), number of orbitals,
/// energies, occupations.
type FlatOrbitalSet = (Vec<f64>, usize, Vec<f64>, Vec<f64>);

/// The orbitals of a structure as flat lists: the overlap (row by row), its size, the sets, and
/// the atom of every atomic orbital.
type NativeOrbitals = (Vec<f64>, usize, Vec<FlatOrbitalSet>, Option<Vec<usize>>);

/// The molecular orbitals of a structure from a built-in calculator (`gfn0xtb`, `gfn1xtb`): the
/// overlap of the atomic orbitals (row by row) and its size, and for each set of orbitals (one for
/// a closed shell, alpha and beta otherwise) the coefficients (row by row), the number of
/// orbitals, their energies, and their occupations, and the atom (0-based) of every atomic
/// orbital (`None` if the calculator does not say); `None` for a calculator without orbitals.
#[pyfunction]
#[pyo3(signature = (name, symbols, positions_bohr, charge = 0, multiplicity = None, unpaired_electrons = None, settings = None))]
fn native_orbitals(
    py: Python<'_>,
    name: &str,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    charge: i64,
    multiplicity: Option<i64>,
    unpaired_electrons: Option<i64>,
    settings: Option<Vec<(String, String)>>,
) -> PyResult<Option<NativeOrbitals>> {
    let multiplicity = multiplicity_of(multiplicity, unpaired_electrons)?;
    let sys = system(&symbols, &positions_bohr, charge, multiplicity, None)?;
    let calc = native_calculator_with(
        &name.to_ascii_lowercase(),
        &sys,
        &settings.unwrap_or_default(),
    )
    .map_err(err)?;
    let Some(found) = py.allow_threads(|| calc.orbitals(&sys)).map_err(err)? else {
        return Ok(None);
    };
    let flat = |m: &Mat<f64>| -> Vec<f64> {
        (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
            .collect()
    };
    let sets = found
        .sets()
        .iter()
        .map(|(_, set)| {
            (
                flat(&set.coefficients),
                set.len(),
                set.energies.clone(),
                set.occupations.clone(),
            )
        })
        .collect();
    Ok(Some((
        flat(&found.overlap),
        found.overlap.nrows(),
        sets,
        found.atom_of_ao.clone(),
    )))
}

/// Stress of a cell periodic in three directions with a built-in calculator, as the rows of
/// the tensor `sigma = (1 / V) dE / d(strain)` in Hartree/Bohr^3 (negative under
/// compression, the sign of ASE): the calculator's own where it gives one, otherwise from
/// differences of the energy.
#[pyfunction]
#[pyo3(signature = (name, symbols, positions_bohr, cell_bohr, settings = None))]
fn native_stress(
    py: Python<'_>,
    name: &str,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    cell_bohr: Vec<f64>,
    settings: Option<Vec<(String, String)>>,
) -> PyResult<Vec<Vec<f64>>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(Some(cell_bohr))?)?;
    let calc = native_calculator_with(
        &name.to_ascii_lowercase(),
        &sys,
        &settings.unwrap_or_default(),
    )
    .map_err(err)?;
    let stress = py
        .allow_threads(|| mopt_rs::pes::stress::stress_of(calc.as_ref(), &sys))
        .map_err(err)?;
    Ok(stress.iter().map(|row| row.to_vec()).collect())
}

/// The six components of ASE (`xx, yy, zz, yz, xz, xy`, eV/Angstrom^3) of a stress given as
/// the rows of the tensor in Hartree/Bohr^3.
#[pyfunction]
fn stress_to_ase_voigt(stress: [[f64; 3]; 3]) -> [f64; 6] {
    mopt_rs::pes::stress::to_ase_voigt(&stress)
}

/// The rows of the stress tensor in Hartree/Bohr^3 from the six components of ASE.
#[pyfunction]
fn stress_from_ase_voigt(voigt: [f64; 6]) -> Vec<Vec<f64>> {
    mopt_rs::pes::stress::from_ase_voigt(&voigt)
        .iter()
        .map(|row| row.to_vec())
        .collect()
}

/// A built-in calculator kept between evaluations: made at the first structure, so that GFN-FF
/// keeps the topology of that structure (as in a command-line run) and reports bonding changes
/// through `take_warnings()`.
#[pyclass(name = "NativeEvaluator")]
pub struct PyNativeEvaluator {
    name: String,
    follow_states: bool,
    /// Settings of the calculator (`key`, `value` as after `-calc_opt`).
    settings: Vec<(String, String)>,
    /// The calculator and the elements it was made for (a structure with other elements, such
    /// as another subsystem, gets a calculator of its own).
    calc: Mutex<Option<(Vec<Element>, Box<dyn mopt_rs::pes::Calculator>)>>,
}

#[pymethods]
impl PyNativeEvaluator {
    #[new]
    #[pyo3(signature = (name, settings = None, *, follow_states=false))]
    fn new(
        name: &str,
        settings: Option<Vec<(String, String)>>,
        follow_states: bool,
    ) -> PyResult<Self> {
        let name = name.to_ascii_lowercase();
        if !mopt_rs::cli::optmain::NATIVE_CALCULATORS.contains(&name.as_str()) {
            return Err(PyValueError::new_err(format!(
                "no native calculator '{name}' (known: {})",
                mopt_rs::cli::optmain::NATIVE_CALCULATORS.join(", ")
            )));
        }
        Ok(Self {
            name,
            settings: settings.unwrap_or_default(),
            follow_states,
            calc: Mutex::new(None),
        })
    }

    /// The calculator name.
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// `(energy, gradient, hessian or None)` as flat lists in atomic units.
    #[pyo3(signature = (symbols, positions_bohr, charge = 0, multiplicity = None, need_hessian = false, cell_bohr = None, unpaired_electrons = None, *, pbc = None, excited_state = 0))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        py: Python<'_>,
        symbols: Vec<String>,
        positions_bohr: Vec<f64>,
        charge: i64,
        multiplicity: Option<i64>,
        need_hessian: bool,
        cell_bohr: Option<Vec<f64>>,
        unpaired_electrons: Option<i64>,
        pbc: Option<[bool; 3]>,
        excited_state: usize,
    ) -> PyResult<NativeResult> {
        let multiplicity = multiplicity_of(multiplicity, unpaired_electrons)?;
        let sys = system(
            &symbols,
            &positions_bohr,
            charge,
            multiplicity,
            cell_with_axes(cell_bohr, pbc)?,
        )?;
        py.allow_threads(|| {
            let mut guard = self
                .calc
                .lock()
                .map_err(|_| MoptError::calc("native calculator lock poisoned"))?;
            if guard.as_ref().is_none_or(|(el, _)| *el != sys.elements) {
                let calc = native_calculator_with(&self.name, &sys, &self.settings)?;
                let calc: Box<dyn mopt_rs::pes::Calculator> = if self.follow_states {
                    Box::new(mopt_rs::pes::followed::FollowedStates::new(calc))
                } else {
                    calc
                };
                *guard = Some((sys.elements.clone(), calc));
            }
            let (_, calc) = guard
                .as_ref()
                .ok_or_else(|| MoptError::calc("native calculator missing"))?;
            evaluate_native_flat(calc.as_ref(), &sys, need_hessian, excited_state)
        })
        .map_err(err)
    }

    /// Warnings gathered since the last call (for example a changed GFN-FF bonding pattern).
    fn take_warnings(&self) -> Vec<String> {
        self.calc
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|(_, c)| c.take_warnings()))
            .unwrap_or_default()
    }
}

/// Evaluate the bias potentials of one MultiOptPy option (e.g. `"-kp", ["0.5", "1.2", "1,2"]`)
/// at a structure: returns `(energy, gradient, hessian or None)` as flat lists in atomic units.
/// `iteration` drives parameter ramps; `time_au` is the simulated time of time-dependent
/// biases (the nanoreactor).
#[pyfunction]
#[pyo3(signature = (option, args, symbols, positions_bohr, need_hessian = false, cell_bohr = None, iteration = 0, time_au = None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_bias(
    py: Python<'_>,
    option: &str,
    args: Vec<String>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    need_hessian: bool,
    cell_bohr: Option<Vec<f64>>,
    iteration: usize,
    time_au: Option<f64>,
) -> PyResult<(f64, Vec<f64>, Option<Vec<f64>>)> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let specs = parse_bias_option(option, &args).map_err(err)?;
    let terms = specs
        .iter()
        .map(|s| s.build(&sys))
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let n = sys.dof();
    let (mut e, mut g, mut h) = (0.0, vec![0.0; n], need_hessian.then(|| vec![0.0; n * n]));
    py.allow_threads(|| -> Result<(), MoptError> {
        let mut ctx = mopt_rs::pes::TermContext::at_iteration(iteration, &sys);
        ctx.time_au = time_au;
        for t in &terms {
            let out = t.evaluate(&sys, &ctx, need_hessian)?;
            e += out.energy.value();
            for (gi, v) in g.iter_mut().zip(out.gradient_bohr.iter()) {
                *gi += v;
            }
            if let (Some(acc), Some(hm)) = (h.as_mut(), out.hessian_bohr.as_ref()) {
                for (k, a) in acc.iter_mut().enumerate() {
                    *a += hm[(k / n, k % n)];
                }
            }
        }
        Ok(())
    })
    .map_err(err)?;
    Ok((e, g, h))
}

fn check_dict<'py>(
    py: Python<'py>,
    term: &dyn PesTerm,
    sys: &MolecularSystem,
    step: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let c = py
        .allow_threads(|| check_pes_term(term, sys, step))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("gradient_error", c.gradient_error)?;
    d.set_item("hessian_error", c.hessian_error)?;
    d.set_item("asymmetry", c.asymmetry)?;
    d.set_item("lattice_invariance_error", c.lattice_invariance_error)?;
    Ok(d)
}

/// Check a Python term (see `Optimizer.add_term`) like the built-in biases: largest deviations
/// of its gradient and Hessian from central differences of step `step` (Bohr), Hessian
/// asymmetry, and the energy change when single atoms move by lattice vectors.
#[pyfunction]
#[pyo3(signature = (callback, symbols, positions_bohr, cell_bohr = None, step = 1e-5))]
fn check_term<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    cell_bohr: Option<Vec<f64>>,
    step: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    check_dict(py, &PyCallbackTerm::new("check", callback)?, &sys, step)
}

/// The same checks for the biases of one MultiOptPy option (e.g. `"-ubias", ["..."]`), summed.
#[pyfunction]
#[pyo3(signature = (option, args, symbols, positions_bohr, cell_bohr = None, step = 1e-5))]
fn check_bias<'py>(
    py: Python<'py>,
    option: &str,
    args: Vec<String>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    cell_bohr: Option<Vec<f64>>,
    step: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let specs = parse_bias_option(option, &args).map_err(err)?;
    let sum = SumOfTerms(
        specs
            .iter()
            .map(|s| s.build(&sys))
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?,
    );
    check_dict(py, &sum, &sys, step)
}

/// The derivatives in every constant of one bias option at a structure (see `optmain
/// -bias_derivatives`): a list of dicts with `name`, `argument` and `field` (from 1), `value`,
/// `step`, `d_energy` (Hartree per unit of the option), and `d_gradient` (flat). With `hessian`
/// (flat, of the whole surface at a stationary point) and `unbiased_gradient`, also
/// `d_structure` (`dx/dp = -H^+ dg/dp`, flat) and `d_unbiased_energy`; `removed` (flat, columns
/// of `3N` rows) are directions taken out, such as the rigid motions.
#[pyfunction]
#[pyo3(signature = (option, args, symbols, positions_bohr, hessian = None, unbiased_gradient = None, removed = None, cell_bohr = None, iteration = 0))]
#[allow(clippy::too_many_arguments)]
fn bias_constant_derivatives<'py>(
    py: Python<'py>,
    option: &str,
    args: Vec<String>,
    symbols: Vec<String>,
    positions_bohr: Vec<f64>,
    hessian: Option<Vec<f64>>,
    unbiased_gradient: Option<Vec<f64>>,
    removed: Option<Vec<f64>>,
    cell_bohr: Option<Vec<f64>>,
    iteration: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    use mopt_rs::cli::bias::BiasRegistry;
    use mopt_rs::cli::bias_constants::{constant_results, StationaryData};
    let sys = system(&symbols, &positions_bohr, 0, 1, cell_from(cell_bohr)?)?;
    let n = sys.dof();
    let h = match hessian {
        Some(v) if v.len() == n * n => Some(Mat::from_fn(n, n, |i, j| v[i * n + j])),
        Some(v) => {
            return Err(PyValueError::new_err(format!(
                "the Hessian has {} entries, not {}",
                v.len(),
                n * n
            )))
        }
        None => None,
    };
    let g0 = match unbiased_gradient {
        Some(v) if v.len() == n => Some(Col::from_fn(n, |i| v[i])),
        Some(v) => {
            return Err(PyValueError::new_err(format!(
                "the gradient has {} entries, not {n}",
                v.len()
            )))
        }
        None => None,
    };
    let r = match removed {
        Some(v) if !v.is_empty() && v.len() % n == 0 => {
            let m = v.len() / n;
            mopt_rs::pes::bias_response::orthonormal_columns(&Mat::from_fn(n, m, |i, j| {
                v[i * m + j]
            }))
        }
        Some(v) if v.is_empty() => None,
        Some(v) => {
            return Err(PyValueError::new_err(format!(
                "removed has {} entries, not a multiple of {n}",
                v.len()
            )))
        }
        None => None,
    };
    let options = vec![(option.to_string(), args)];
    let results = py
        .allow_threads(|| {
            let ctx = mopt_rs::pes::TermContext::at_iteration(iteration, &sys);
            let data = match (&h, &g0) {
                (Some(h), Some(g0)) => Some(StationaryData {
                    hessian: h,
                    unbiased_gradient: g0,
                    removed: r.as_ref(),
                }),
                _ => None,
            };
            constant_results(&BiasRegistry::new(), &options, &sys, &ctx, data.as_ref())
        })
        .map_err(err)?;
    let mut out = Vec::with_capacity(results.len());
    for res in &results {
        let d = PyDict::new(py);
        d.set_item("name", res.constant.name())?;
        d.set_item("argument", res.constant.argument + 1)?;
        d.set_item("field", res.constant.field + 1)?;
        d.set_item("value", res.constant.value)?;
        d.set_item("step", res.derivative.step)?;
        d.set_item("d_energy", res.derivative.d_energy)?;
        d.set_item(
            "d_gradient",
            res.derivative
                .d_gradient
                .iter()
                .copied()
                .collect::<Vec<f64>>(),
        )?;
        if let Some(s) = &res.response {
            d.set_item(
                "d_structure",
                s.d_structure.iter().copied().collect::<Vec<f64>>(),
            )?;
            d.set_item("d_unbiased_energy", s.d_unbiased_energy)?;
            d.set_item("smallest_curvature", s.smallest_curvature)?;
        }
        out.push(d);
    }
    Ok(out)
}

/// Sum of terms (for `check_bias`).
struct SumOfTerms(Vec<Arc<dyn PesTerm>>);

impl PesTerm for SumOfTerms {
    fn name(&self) -> &str {
        "sum"
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        ctx: &mopt_rs::pes::TermContext<'_>,
        need_hessian: bool,
    ) -> Result<EvalOutput, MoptError> {
        let n = sys.dof();
        let mut acc = EvalOutput {
            energy: Quantity::new(0.0),
            gradient_bohr: Col::zeros(n),
            hessian_bohr: need_hessian.then(|| Mat::zeros(n, n)),
            orbitals: None,
            stress: None,
            strain_hessian: None,
        };
        for t in &self.0 {
            let o = t.evaluate(sys, ctx, need_hessian)?;
            acc.energy = Quantity::new(acc.energy.value() + o.energy.value());
            acc.gradient_bohr += &o.gradient_bohr;
            if let (Some(a), Some(h)) = (acc.hessian_bohr.as_mut(), o.hessian_bohr.as_ref()) {
                *a += h;
            }
        }
        Ok(acc)
    }

    /// The largest frame of the terms (that of a sum).
    fn frame(&self) -> mopt_rs::pes::Frame {
        self.0
            .iter()
            .fold(mopt_rs::pes::Frame::Invariant, |f, t| f.and(t.frame()))
    }

    /// The largest history of the terms.
    fn history(&self) -> mopt_rs::pes::History {
        self.0
            .iter()
            .fold(mopt_rs::pes::History::Static, |h, t| h.and(t.history()))
    }
}

/// Return the `mopt-rs` version string.
#[pyfunction]
fn version() -> &'static str {
    mopt_rs::VERSION
}

/// Run one of the ten commands with `args` (without the program name); returns the exit code.
/// The GIL is released while the command runs.
#[pyfunction]
fn run_cli(py: Python<'_>, command: String, args: Vec<String>) -> i32 {
    let registry = pycalc::python_registry();
    let parts = mapper_plugins::parts();
    py.allow_threads(|| mopt_rs::cli::run_command_with_parts(&command, &args, &registry, &parts))
}

/// [`run_cli`], with the evaluations of the calculators in that run: the exit code, the
/// evaluations of the energy alone, those with the gradient (those with the Hessian are
/// among them), and those with the Hessian. Runs on objectives of plain variables write
/// their counts to the log only.
#[pyfunction]
fn run_cli_counted(
    py: Python<'_>,
    command: String,
    args: Vec<String>,
) -> (i32, usize, usize, usize) {
    let registry = pycalc::python_registry();
    let parts = mapper_plugins::parts();
    let code = py
        .allow_threads(|| mopt_rs::cli::run_command_with_parts(&command, &args, &registry, &parts));
    let counts = registry.counts();
    (code, counts.energies, counts.gradients, counts.hessians)
}

/// Run `optmain` with `args` (as on the command line, without the program name; names with
/// wildcards stand for the files that match) and return what every job ended with, in the
/// order of the inputs: MultiOptPy's `Optimize.all_results`. `console`: print the progress.
#[pyfunction]
#[pyo3(signature = (args, console = true))]
fn run_optmain<'py>(
    py: Python<'py>,
    args: Vec<String>,
    console: bool,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    use mopt_rs::cli::optmain;
    use mopt_rs::core_infra::log::Level;
    let registry = pycalc::python_registry();
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = optmain::config_from_cli(&refs).map_err(err)?;
    let level = console.then_some(if config.debug {
        Level::Debug
    } else {
        Level::Info
    });
    let runs = py
        .allow_threads(|| optmain::run_with(&config, std::path::Path::new("."), level, &registry))
        .map_err(err)?;
    let mut all = Vec::with_capacity(runs.len());
    for run in runs {
        let d = PyDict::new(py);
        let folder = run.directory.to_string_lossy().into_owned();
        d.set_item("folder_directory", &folder)?;
        d.set_item("final_file_directory", &folder)?;
        d.set_item(
            "optimized_struct_file",
            run.optimized.to_string_lossy().into_owned(),
        )?;
        d.set_item("optimized_flag", run.converged)?;
        d.set_item("interrupted", run.interrupted)?;
        if let Some(s) = run.summary {
            let geometry: Vec<Vec<f64>> = s
                .system
                .atoms()
                .map(|a| s.system.position(a).value().to_vec())
                .collect();
            let symbols: Vec<&str> = s.system.elements.iter().map(|e| e.symbol()).collect();
            d.set_item("input_file", s.input.to_string_lossy().into_owned())?;
            d.set_item("traj_file", s.trajectory.to_string_lossy().into_owned())?;
            d.set_item("element_list", symbols)?;
            d.set_item("final_geometry", geometry)?;
            d.set_item("final_energy", s.energy.value())?;
            d.set_item("final_bias_energy", s.biased_energy.value())?;
            d.set_item("symmetry", s.symmetry)?;
            d.set_item("iterations", s.iterations)?;
        }
        all.push(d);
    }
    Ok(all)
}

/// `mopt-py <command> [options]`: `argv` includes the program name.
#[pyfunction]
fn main_from_argv(py: Python<'_>, argv: Vec<String>) -> i32 {
    let registry = pycalc::python_registry();
    py.allow_threads(|| mopt_rs::cli::main_from_argv_with(&argv, &registry))
}

/// Initialize the native `mopt_rs._core` Python module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Before anything computes: results that do not depend on the number of threads.
    mopt_rs::core_infra::parallelism::deterministic_linear_algebra();
    m.add_class::<PyOptimizer>()?;
    m.add_class::<PyNativeEvaluator>()?;
    components::register(m)?;
    chain::register(m)?;
    m.add_class::<network::PyMapperNetwork>()?;
    m.add_class::<pycalc::PyFollowedEvaluator>()?;
    variables::register(m)?;
    bitss::register(m)?;
    dmf::register(m)?;
    irc::register(m)?;
    md::register(m)?;
    trajectory::register(m)?;
    analysis::register(m)?;
    vib::register(m)?;
    steps::register(m)?;
    parts::register(m)?;
    orient::register(m)?;
    confsearch::register(m)?;
    // The searches of ieipmain.
    ieip::register(m)?;
    searches::register(m)?;
    saddles::register(m)?;
    strings::register(m)?;
    scan::register(m)?;
    kinetics::register(m)?;
    electron_transfer::register(m)?;
    xyz::register(m)?;
    oniom::register(m)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_native, m)?)?;
    m.add_function(wrap_pyfunction!(native_stress, m)?)?;
    m.add_function(wrap_pyfunction!(native_orbitals, m)?)?;
    m.add_function(wrap_pyfunction!(stress_to_ase_voigt, m)?)?;
    m.add_function(wrap_pyfunction!(stress_from_ase_voigt, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_bias, m)?)?;
    m.add_function(wrap_pyfunction!(check_term, m)?)?;
    m.add_function(wrap_pyfunction!(check_bias, m)?)?;
    m.add_function(wrap_pyfunction!(bias_constant_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_function(wrap_pyfunction!(run_cli_counted, m)?)?;
    m.add_function(wrap_pyfunction!(run_optmain, m)?)?;
    m.add_function(wrap_pyfunction!(main_from_argv, m)?)?;
    m.add("__version__", mopt_rs::VERSION)?;
    add_units(m)?;
    // The one criterion of an imaginary mode (an eigenvalue of the mass-weighted Hessian
    // below minus this, Hartree / (Bohr^2 amu)), so that a script counts as the mode kill,
    // the mapper, and the start of an IRC do (new).
    m.add(
        "IMAGINARY_CURVATURE",
        mopt_rs::opt::mode_kill::IMAGINARY_CURVATURE,
    )?;
    Ok(())
}

/// The CODATA 2018 conversion factors of `mopt_rs::domain::Units` as module constants, so that
/// the Python side converts with the same numbers as the library (new).
fn add_units(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BOHR_TO_ANGSTROM", Units::BOHR_TO_ANGSTROM)?;
    m.add("ANGSTROM_TO_BOHR", Units::ANGSTROM_TO_BOHR)?;
    m.add("HARTREE_TO_EV", Units::HARTREE_TO_EV)?;
    m.add("HARTREE_TO_KCAL_MOL", Units::HARTREE_TO_KCAL_MOL)?;
    m.add("HARTREE_TO_KJ_MOL", Units::HARTREE_TO_KJ_MOL)?;
    m.add("HARTREE_TO_WAVENUMBER", Units::HARTREE_TO_WAVENUMBER)?;
    m.add("BOLTZMANN_HARTREE_PER_K", Units::BOLTZMANN_HARTREE_PER_K)?;
    m.add("AMU_TO_AU_MASS", Units::AMU_TO_AU_MASS)?;
    m.add("AU_TIME_TO_FS", Units::AU_TIME_TO_FS)?;
    m.add("GPA_PER_AU_PRESSURE", Units::GPA_PER_AU_PRESSURE)?;
    Ok(())
}
