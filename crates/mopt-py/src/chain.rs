//! Python surfaces and observers for chain relaxations and environment microiterations.
//!
//! MultiOptPy: no counterpart (Python access to the Rust workflows).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use faer::Col;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::{cell_with_axes, err, held_constraints, multiplicity_of, output, system, NativeResult};
use mopt_rs::cli::nebmain::{
    chain_surrogate, internal_coordinates, options_from_cli, structure_chain, variable_chain,
    NebmainConfig, StructureChain,
};
use mopt_rs::cli::nebmain_cell::{cell_chain, check_options, CellChain};
use mopt_rs::cli::oniom_micro::{build_terms, wrap_constrained, Wrapped, NORMAL_FORCES};
use mopt_rs::cli::variable_cell::VariableCellSettings;
use mopt_rs::coord::system::CoordinateRegistry;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::path::band::machine::{ChainEnd, ChainRelaxation, ChainReport, ImageValue};
use mopt_rs::path::band::surrogate::SurrogateSettings;
use mopt_rs::path::band::ChainKind;
use mopt_rs::pes::constraint::breaks_rigid_body;
use mopt_rs::pes::microiter::{relax, InnerMethod, MicroEval, MicroProblem};
use mopt_rs::pes::objective::ObjectiveRegistry;
use mopt_rs::pes::term::frame_invariant;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::chain::{
    check_chain_terms, relax_chain, warm_start_label, CalculatorSurface, ChainObserver,
    ChainSurface, Flow, KeepConstraints, ObjectiveSurface,
};
use mopt_rs::workflow::chain_figures::ChainFigures;
use mopt_rs::workflow::chain_files::{ChainFiles, ChainPoints};
use mopt_rs::workflow::chain_input::join_in_the_cell_centred;
use mopt_rs::workflow::chain_internal::InternalSurface;
use mopt_rs::workflow::chain_surrogate::SurrogateSurface;

fn lists(points: &[Col<f64>]) -> Vec<Vec<f64>> {
    points.iter().map(|p| p.iter().copied().collect()).collect()
}

struct PythonSurface {
    function: Py<PyAny>,
    dimension: usize,
    analytic: bool,
}

impl PythonSurface {
    fn value(&self, x: &[f64], index: Option<usize>, hessian: bool) -> MoptResult<MicroEval> {
        Python::with_gil(|py| {
            let f = self.function.bind(py);
            let value = match index {
                Some(i) => f.call1((x, i, hessian))?,
                None => f.call1((x, hessian))?,
            };
            let (energy, gradient, matrix): NativeResult = value.extract()?;
            if gradient.len() != x.len()
                || !energy.is_finite()
                || !gradient
                    .iter()
                    .chain(matrix.iter().flatten())
                    .all(|v| v.is_finite())
            {
                return Err(PyValueError::new_err(
                    "invalid energy or derivatives from Python surface",
                ));
            }
            let out = output(energy, gradient, matrix)?;
            Ok(MicroEval {
                energy,
                gradient: out.gradient_bohr,
                hessian: out.hessian_bohr,
            })
        })
        .map_err(|e: PyErr| MoptError::calc(e.to_string()))
    }
}

impl MicroProblem for PythonSurface {
    fn dim(&self) -> usize {
        self.dimension
    }
    fn has_hessian(&self) -> bool {
        self.analytic
    }
    fn evaluate(&self, y: &[f64], need_hessian: bool) -> MoptResult<MicroEval> {
        self.value(y, None, need_hessian)
    }
}

impl ChainSurface for PythonSurface {
    fn description(&self) -> String {
        "Python chain surface".to_owned()
    }
    fn evaluate(
        &self,
        point: &Col<f64>,
        index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue> {
        let x: Vec<_> = point.iter().copied().collect();
        let v = self.value(&x, Some(index), need_hessian)?;
        Ok(ImageValue {
            energy: v.energy,
            gradient: v.gradient,
            hessian: v.hessian,
        })
    }
}

/// Relax environment variables at fixed outer coordinates supplied by the caller's closure.
#[pyfunction]
#[pyo3(signature = (evaluate, initial, *, method="lbfgs", max_iterations=500, tolerance=1e-5, has_hessian=false, mode_kill=true))]
#[allow(clippy::too_many_arguments)]
fn relax_environment<'py>(
    py: Python<'py>,
    evaluate: Py<PyAny>,
    initial: Vec<f64>,
    method: &str,
    max_iterations: usize,
    tolerance: f64,
    has_hessian: bool,
    mode_kill: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if initial.is_empty()
        || !initial.iter().all(|x| x.is_finite())
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || max_iterations == 0
    {
        return Err(PyValueError::new_err(
            "expected finite initial coordinates, positive tolerance and iterations",
        ));
    }
    let problem = PythonSurface {
        function: evaluate,
        dimension: initial.len(),
        analytic: has_hessian,
    };
    let mut settings = mopt_rs::opt::micro::settings();
    settings.method = InnerMethod::parse(method);
    settings.max_iterations = max_iterations;
    settings.gradient_tolerance = tolerance;
    settings.rms_tolerance = tolerance;
    settings.mode_kill.enabled = mode_kill;
    let result = py
        .allow_threads(|| relax(&problem, &initial, &settings))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("positions", result.y)?;
    d.set_item("energy", result.energy)?;
    d.set_item("converged", result.converged)?;
    d.set_item("iterations", result.iterations)?;
    d.set_item("max_gradient", result.max_gradient)?;
    Ok(d)
}

/// A Python function `f(x, image, need_hessian) -> (energy, gradient, hessian or None)` of
/// the flat Cartesian coordinates of a structure (Bohr) as a calculator, so that a chain of
/// structures on it is relaxed on the surface of `nebmain` (bias potentials, constraints,
/// model Hessians, states). The number of the image is the one [`Indexed`] sets before the
/// evaluation; the images are evaluated in order.
struct FunctionCalculator {
    function: Py<PyAny>,
    image: Arc<AtomicUsize>,
    rigid_body_invariant: bool,
}

impl Calculator for FunctionCalculator {
    fn name(&self) -> &str {
        "python"
    }

    fn description(&self) -> String {
        "Python chain surface".to_owned()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let index = self.image.load(Ordering::SeqCst);
        Python::with_gil(|py| {
            let value = self
                .function
                .bind(py)
                .call1((x.clone(), index, req.need_hessian))?;
            let (energy, gradient, matrix): NativeResult = value.extract()?;
            if gradient.len() != x.len()
                || !energy.is_finite()
                || !gradient
                    .iter()
                    .chain(matrix.iter().flatten())
                    .all(|v| v.is_finite())
            {
                return Err(PyValueError::new_err(
                    "invalid energy or derivatives from Python surface",
                ));
            }
            output(energy, gradient, matrix)
        })
        .map_err(|e: PyErr| MoptError::calc(e.to_string()))
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn is_rigid_body_invariant(&self) -> bool {
        self.rigid_body_invariant
    }

    fn evaluates_in_order(&self) -> bool {
        true
    }
}

/// The surface of `nebmain` over a [`FunctionCalculator`]: tells the calculator the number of
/// the image before every evaluation, one image after the other.
struct Indexed<'a> {
    inner: &'a dyn ChainSurface,
    image: Arc<AtomicUsize>,
}

impl ChainSurface for Indexed<'_> {
    fn description(&self) -> String {
        self.inner.description()
    }
    fn evaluate(
        &self,
        point: &Col<f64>,
        index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue> {
        self.image.store(index, Ordering::SeqCst);
        self.inner.evaluate(point, index, need_hessian)
    }
    fn evaluate_all(
        &self,
        images: &[(usize, &Col<f64>)],
        need_hessian: bool,
    ) -> Vec<MoptResult<ImageValue>> {
        images
            .iter()
            .map(|(index, point)| self.evaluate(point, *index, need_hessian))
            .collect()
    }
    fn begin_iteration(&self, iteration: usize) {
        self.inner.begin_iteration(iteration);
    }
    fn images_dropped(&self, dropped: &[usize]) {
        self.inner.images_dropped(dropped);
    }
    fn images_added(&self) {
        self.inner.images_added();
    }
    fn values_are_calculated(&self) -> bool {
        self.inner.values_are_calculated()
    }
    fn calculate_next(&self, check: bool) {
        self.inner.calculate_next(check);
    }
    fn told(&self, largest_force: f64) {
        self.inner.told(largest_force);
    }
    fn take_notes(&self) -> Vec<String> {
        self.inner.take_notes()
    }
    fn take_bare_energies(&self) -> Vec<(usize, f64)> {
        self.inner.take_bare_energies()
    }
}

/// What `relax_band` is given to evaluate.
enum BandSurface {
    /// A Python function.
    Function(Py<PyAny>),
    /// A calculator of the registry of the commands (built-in engines, `mopt_rs.calculators`)
    /// or an objective of plain variables, by its name.
    Named(String),
}

/// The keyword arguments of `relax_band` that override the options of `nebmain`.
struct BandOverrides {
    coordinates: Option<String>,
    inserted: Option<usize>,
    equal_spacing: Option<bool>,
    spacing: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    surrogate: Option<bool>,
    update_projection: Option<bool>,
}

impl BandOverrides {
    /// `config` with the keyword arguments; `spacing` in Bohr for structures.
    fn apply(&self, config: &mut NebmainConfig, structures: bool) -> PyResult<()> {
        if let Some(c) = &self.coordinates {
            config.coordinates.clone_from(c);
        }
        if let Some(n) = self.inserted {
            config.inserted = n;
        }
        if let Some(e) = self.equal_spacing {
            config.first_equal_spacing = e;
        }
        if let Some(d) = self.spacing {
            if !d.is_finite() || d <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "spacing: the distance of the images must be positive, not {d}"
                )));
            }
            // -nd is in the unit of the input: Angstrom for structures.
            config.spacing = Some(if structures {
                d * Units::BOHR_TO_ANGSTROM
            } else {
                d
            });
        }
        if let Some(n) = self.max_iterations {
            if n == 0 {
                return Err(PyValueError::new_err("max_iterations: at least one"));
            }
            config.chain.max_iterations = n;
        }
        if let Some(t) = self.tolerance {
            if !t.is_finite() || t < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "tolerance: 0 (all iterations) or more, not {t}"
                )));
            }
            config.chain.force_tolerance = (t > 0.0).then_some(t);
        }
        match self.surrogate {
            Some(true) if config.surrogate.is_none() => {
                config.surrogate = Some(SurrogateSettings::default());
            }
            Some(false) => config.surrogate = None,
            _ => {}
        }
        if let (Some(on), Some(rfo)) = (self.update_projection, config.chain.rfo.as_mut()) {
            rfo.project_updates = on;
        }
        if config.dmf.is_some() {
            return Err(PyValueError::new_err(
                "-dmf: the direct MaxFlux method is mopt_rs.dmf_search from Python",
            ));
        }
        if config.follow_up.is_some() {
            return Err(PyValueError::new_err(
                "-neb_opt optimize: the optimizations after the chain are Optimizer runs from \
                 Python (from the images relax_band returns)",
            ));
        }
        Ok(())
    }
}

/// The observers of a Python run: the files of `output_dir`, the constraints, the observer.
fn observe_band(
    report: &ChainReport,
    machine: &mut ChainRelaxation,
    internal: Option<&InternalSurface<'_>>,
    files: &mut Option<(ChainFiles, ChainFigures)>,
    keep: &mut Option<KeepConstraints>,
    observer: Option<&Py<PyAny>>,
) -> MoptResult<Flow> {
    let points = match internal {
        Some(i) => i.structures(&report.chain)?,
        None => report.chain.clone(),
    };
    let mut flow = Flow::Continue;
    if let Some((files, figures)) = files.as_mut() {
        let written = ChainReport {
            chain: points.clone(),
            ..report.clone()
        };
        flow = files.after_iteration(&written, machine)?;
        figures.after_iteration(report, machine)?;
    }
    if let Some(keep) = keep.as_mut() {
        keep.after_iteration(report, machine)?;
    }
    if let Some(callback) = observer {
        let stop: bool = Python::with_gil(|py| {
            callback
                .call1(
                    py,
                    (
                        report.iteration,
                        lists(&points),
                        report.energies.as_slice(),
                        report.largest_force,
                    ),
                )?
                .extract::<Option<bool>>(py)
        })
        .map_err(|e| MoptError::calc(e.to_string()))?
        .unwrap_or(false);
        if stop {
            flow = Flow::Stop;
        }
    }
    Ok(flow)
}

/// Relax a chain of images as `nebmain` does: the same parser of its options (`options`, the
/// options of the command line such as `["-qsm"]`, `["-lup"]`, `["-ewbneb"]`,
/// `["-modelhess", "fischerd3"]`, `["-neb_opt", "tangent=top"]`, `["-fix", "1,2"]`), the same
/// defaults, and the same set-up of the chain (`cli::nebmain::structure_chain`,
/// `variable_chain`). `evaluate` is a Python function `f(x, image, need_hessian)` or the name
/// of a calculator of the commands (a built-in engine, an engine of `mopt_rs.calculators`, or
/// an objective of plain variables such as `mb2d`), or an `OniomCalculator` (its options are
/// those of the run; ONIOM can also be given as `-oniom` in `options`). The
/// keyword arguments override the options: `spacing` (Bohr for structures), `max_iterations`
/// (`-ns`), `tolerance` (`-neb_opt tolerance=`), `inserted` (`-p`), `equal_spacing` (`-ad`),
/// `coordinates` (`-coord`), `surrogate` (`-neb_opt gp=on`), `update_projection`
/// (`-update_proj on|off`: the rigid-body motions taken out of the updated Hessians of the
/// images). Constraints (`fix_atoms`,
/// `constraints`, `user_constraints`, `constraint_functions`, as for `Optimizer`) are added to
/// those of `options`; `cell_bohr` and `pbc` make the structures periodic.
#[pyfunction]
#[pyo3(signature = (
    evaluate,
    images,
    *,
    symbols = None,
    charge = 0,
    multiplicity = None,
    unpaired_electrons = None,
    cell_bohr = None,
    pbc = None,
    options = None,
    coordinates = None,
    inserted = None,
    equal_spacing = None,
    spacing = None,
    max_iterations = None,
    tolerance = None,
    surrogate = None,
    update_projection = None,
    fix_atoms = None,
    constraints = None,
    user_constraints = None,
    constraint_functions = None,
    rigid_body_invariant = true,
    output_dir = None,
    observer = None,
))]
#[allow(clippy::too_many_arguments)]
fn relax_band<'py>(
    py: Python<'py>,
    evaluate: Bound<'py, PyAny>,
    images: Vec<Vec<f64>>,
    symbols: Option<Vec<String>>,
    charge: i64,
    multiplicity: Option<i64>,
    unpaired_electrons: Option<i64>,
    cell_bohr: Option<Vec<f64>>,
    pbc: Option<[bool; 3]>,
    options: Option<Vec<String>>,
    coordinates: Option<String>,
    inserted: Option<usize>,
    equal_spacing: Option<bool>,
    spacing: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    surrogate: Option<bool>,
    update_projection: Option<bool>,
    fix_atoms: Option<Vec<usize>>,
    constraints: Option<Vec<String>>,
    user_constraints: Option<Vec<String>>,
    constraint_functions: Option<Vec<Bound<'py, PyAny>>>,
    rigid_body_invariant: bool,
    output_dir: Option<PathBuf>,
    observer: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let dimension = images.first().map_or(0, Vec::len);
    if dimension == 0
        || images.len() < 2
        || images
            .iter()
            .any(|p| p.len() != dimension || !p.iter().all(|x| x.is_finite()))
    {
        return Err(PyValueError::new_err(
            "expected two or more equally sized images of finite numbers",
        ));
    }
    let mut tokens = Vec::new();
    let surface = if let Ok(name) = evaluate.extract::<String>() {
        BandSurface::Named(name)
    } else if let Ok(oniom) = evaluate.extract::<PyRef<'_, crate::oniom::PyOniomCalculator>>() {
        // The options of the ONIOM calculator are those of the run (-os, -oniom, ...).
        tokens = oniom.options_tokens();
        BandSurface::Named(oniom.name.clone())
    } else if evaluate.is_callable() {
        BandSurface::Function(evaluate.clone().unbind())
    } else {
        return Err(PyValueError::new_err(
            "evaluate: a function f(x, image, need_hessian) or the name of a calculator",
        ));
    };
    tokens.extend(options.unwrap_or_default());
    let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
    let mut config = options_from_cli(&refs).map_err(err)?;
    let overrides = BandOverrides {
        coordinates,
        inserted,
        equal_spacing,
        spacing,
        max_iterations,
        tolerance,
        surrogate,
        update_projection,
    };
    overrides.apply(&mut config, symbols.is_some())?;
    let held = held_constraints(
        fix_atoms,
        constraints,
        user_constraints,
        constraint_functions,
    )?;
    config.constraints.extend(held);
    let raw: Vec<Col<f64>> = images
        .iter()
        .map(|p| Col::from_fn(p.len(), |i| p[i]))
        .collect();
    match symbols {
        Some(symbols) => {
            let multiplicity = multiplicity_of(multiplicity, unpaired_electrons)?;
            let cell = cell_with_axes(cell_bohr, pbc)?;
            let template = system(&symbols, &images[0], charge, multiplicity, cell)?;
            let structures = raw
                .iter()
                .map(|x| template.with_flat_coords_bohr(x))
                .collect::<MoptResult<Vec<_>>>()
                .map_err(err)?;
            let run = StructureBand {
                config: &config,
                surface,
                rigid_body_invariant,
                output_dir,
                observer,
            };
            py.allow_threads(|| run.relax(structures))
                .map_err(err)?
                .into_dict(py)
        }
        None => {
            if cell_bohr.is_some() || pbc.is_some() {
                return Err(PyValueError::new_err(
                    "a cell needs a structure (symbols); plain variables have none",
                ));
            }
            let run = VariableBand {
                config: &config,
                surface,
                output_dir,
                observer,
            };
            py.allow_threads(|| run.relax(raw))
                .map_err(err)?
                .into_dict(py)
        }
    }
}

/// What a relaxation of `relax_band` gives back.
struct BandResult {
    iterations: usize,
    end: Option<ChainEnd>,
    images: Option<Vec<Col<f64>>>,
    energies: Option<Vec<f64>>,
    /// With ONIOM microiterations: the whole structures of the images.
    whole: Option<Vec<Col<f64>>>,
    /// With `-vc`: the lattice vectors of every image (rows, Bohr).
    cells: Option<Vec<Vec<f64>>>,
    predicted: Option<usize>,
    notes: Vec<String>,
}

impl BandResult {
    fn into_dict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("iterations", self.iterations)?;
        d.set_item("termination", self.end.map(|v| format!("{v:?}")))?;
        if let Some(images) = &self.images {
            d.set_item("images", lists(images))?;
        }
        if let Some(energies) = self.energies {
            d.set_item("energies", energies)?;
        }
        if let Some(whole) = &self.whole {
            d.set_item("whole_images", lists(whole))?;
        }
        if let Some(cells) = self.cells {
            d.set_item("cells_bohr", cells)?;
        }
        if let Some(predicted) = self.predicted {
            d.set_item("predicted_images", predicted)?;
        }
        d.set_item("notes", self.notes)?;
        Ok(d)
    }
}

/// A chain of structures, as `nebmain` relaxes one (`cli::nebmain::run_structures`).
struct StructureBand<'a> {
    config: &'a NebmainConfig,
    surface: BandSurface,
    rigid_body_invariant: bool,
    output_dir: Option<PathBuf>,
    observer: Option<Py<PyAny>>,
}

impl StructureBand<'_> {
    fn relax(self, mut structures: Vec<MolecularSystem>) -> MoptResult<BandResult> {
        let config = self.config;
        if config.excited_state > 0 && matches!(self.surface, BandSurface::Function(_)) {
            return Err(MoptError::unsupported(
                "-es: a Python function has no excited states; give the name of a calculator",
            ));
        }
        let first = structures
            .first()
            .cloned()
            .ok_or_else(|| MoptError::config("no images"))?;
        let biases = build_terms(&config.biases, &first)?;
        check_chain_terms(&biases, "nebmain", &warm_start_label)?;
        if let Some(cell) = &config.variable_cell {
            // -vc: the chain of the atoms and the cell of `nebmain -vc`.
            return self.relax_cell(cell, &structures, biases);
        }
        let free_shifts = frame_invariant(true, biases.iter().map(|(_, t)| t.as_ref()))
            && !breaks_rigid_body(&config.constraints);
        let periodic_shifts = join_in_the_cell_centred(&mut structures, free_shifts)?;
        let whole = structures[0].clone();
        let image = Arc::new(AtomicUsize::new(0));
        let registry = crate::pycalc::python_registry();
        let wrapped = match &self.surface {
            BandSurface::Named(name) => wrap_constrained(
                &config.calculator_options,
                None,
                &config.constraints,
                &biases,
                &[],
                match config.chain.force_tolerance {
                    Some(t) => (t, f64::INFINITY),
                    None => NORMAL_FORCES,
                },
                whole.clone(),
                name,
                &registry,
            )?,
            BandSurface::Function(function) => {
                if config.calculator_options.oniom.is_some() {
                    return Err(MoptError::unsupported(
                        "-oniom: the ONIOM surface is made from the calculators of the layers; \
                         give the name of the calculator of the highest level, not a function",
                    ));
                }
                let calc: Box<dyn Calculator> = Box::new(FunctionCalculator {
                    function: Python::with_gil(|py| function.clone_ref(py)),
                    image: Arc::clone(&image),
                    rigid_body_invariant: self.rigid_body_invariant,
                });
                Wrapped {
                    start: whole.clone(),
                    electrons: false,
                    calc,
                    micro: None,
                    note: None,
                    constraints: config.constraints.clone(),
                    terms: biases,
                }
            }
        };
        let mut notes: Vec<String> = wrapped.note.iter().cloned().collect();
        let (template, calc, micro, constraints, terms) = (
            wrapped.start,
            wrapped.calc,
            wrapped.micro,
            wrapped.constraints,
            wrapped.terms,
        );
        if let Some(relaxed) = &micro {
            for structure in &structures {
                relaxed.remember(structure)?;
            }
            structures = structures
                .iter()
                .map(|structure| relaxed.core_of(structure))
                .collect::<MoptResult<_>>()?;
        }
        let calc = mopt_rs::pes::o1numhess::wrap_if_requested(
            calc,
            config.model_hessian.as_deref(),
            config.exact_hessians,
        );
        if calc.uses_electrons() {
            template.spin_charge.check_electrons(&template.elements)?;
        }
        let rigid = frame_invariant(
            calc.is_rigid_body_invariant(),
            terms.iter().map(|(_, t)| t.as_ref()),
        );
        let StructureChain {
            mut settings,
            chain,
            constraints: shared,
            notes: set_up,
        } = structure_chain(
            config,
            &structures,
            &template,
            &constraints,
            rigid,
            periodic_shifts,
            self.output_dir.clone(),
        )?;
        notes.extend(set_up);
        let mut surface = CalculatorSurface::new(calc.as_ref(), template.clone())
            .with_terms(terms)
            .with_excited_state(config.excited_state);
        if let Some(shared) = &shared {
            surface = surface.with_constraints(Arc::clone(shared));
        }
        if !config.model_function.is_empty() {
            let (spec, root) = (config.model_function.clone(), config.excited_state);
            mopt_rs::cli::optmain::state_surface(&spec, root)?;
            surface = surface.with_states(Arc::new(move || {
                mopt_rs::cli::optmain::state_surface(&spec, root)?
                    .ok_or_else(|| MoptError::config("-mf: no surface of states"))
            }));
        }
        if let (false, Some(name)) = (config.exact_hessians, &config.model_hessian) {
            if !mopt_rs::pes::o1numhess::is_o1numhess(name) && settings.rfo.is_some() {
                surface = surface.with_model_hessian(mopt_rs::model_hessian::resolve(name)?);
            }
        }
        let indexed = Indexed {
            inner: &surface,
            image: Arc::clone(&image),
        };
        let mut keep = shared.map(KeepConstraints::new);
        let mut chain = match keep.as_mut() {
            Some(keep) => keep.restore_first(&chain)?,
            None => chain,
        };
        let candidates = internal_coordinates(config, &settings, keep.is_some())?;
        let mut internal = None;
        if !candidates.is_empty() {
            let middle = template.with_flat_coords_bohr(&chain[chain.len() / 2])?;
            for name in &candidates {
                let coordinates = CoordinateRegistry::default().create(name)?;
                match InternalSurface::new(&indexed, coordinates, middle.clone(), &chain) {
                    Ok((made, points)) => {
                        notes.push(format!(
                            "Coordinates (-coord {name}): the chain is relaxed in {} internal \
                             coordinates.",
                            made.dimension()
                        ));
                        settings.kind = ChainKind::Variables;
                        settings.block = None;
                        settings.align = false;
                        settings.periodic_shifts = None;
                        chain = points;
                        internal = Some(made);
                        break;
                    }
                    Err(error) => notes.push(format!(
                        "Warning: coordinates (-coord {name}) cannot be set up for this chain: \
                         {error}"
                    )),
                }
            }
        }
        let behind: &dyn ChainSurface = match &internal {
            Some(internal) => internal,
            None => &indexed,
        };
        let plain = internal.is_none() && template.active_cell().is_none();
        let modelled =
            chain_surrogate(config, true, plain)?.map(|model| SurrogateSurface::new(behind, model));
        let relaxed_on = modelled.as_ref().map_or(behind, |m| m as &dyn ChainSurface);
        let mut machine = ChainRelaxation::new(settings, chain)?;
        if internal.is_none() {
            machine = machine.with_structure(template.clone())?;
        }
        let mut files = match &self.output_dir {
            Some(dir) => Some((
                ChainFiles::new(
                    dir,
                    "band",
                    ChainPoints::Structures(Box::new(template.clone())),
                )?,
                ChainFigures::new(dir)
                    .keep_every_iteration(false)
                    .energy_axis(Units::HARTREE_TO_KCAL_MOL, "Energy (kcal/mol)"),
            )),
            None => None,
        };
        let observer = self.observer;
        let outcome = {
            let mut observe = |report: &ChainReport, machine: &mut ChainRelaxation| {
                observe_band(
                    report,
                    machine,
                    internal.as_ref(),
                    &mut files,
                    &mut keep,
                    observer.as_ref(),
                )
            };
            relax_chain(&mut machine, relaxed_on, &mut [&mut observe])?
        };
        let (images, energies) = match outcome.last {
            Some(last) => {
                let points = match &internal {
                    Some(i) => i.structures(&last.chain)?,
                    None => last.chain,
                };
                (Some(points), Some(last.energies))
            }
            None => (None, None),
        };
        let whole = match (&micro, &images) {
            (Some(relaxed), Some(points)) => Some(
                points
                    .iter()
                    .map(|x| {
                        relaxed
                            .full_system(&template.with_flat_coords_bohr(x)?)
                            .map(|s| s.coords_flat_bohr())
                    })
                    .collect::<MoptResult<Vec<_>>>()?,
            ),
            _ => None,
        };
        Ok(BandResult {
            iterations: outcome.iterations,
            end: outcome.end,
            images,
            energies,
            whole,
            cells: None,
            predicted: modelled.map(|m| m.counts().predicted),
            notes,
        })
    }
}

impl StructureBand<'_> {
    /// `-vc`: the chain of the atoms and the cell, as `nebmain -vc` relaxes one
    /// (`cli::nebmain_cell`); the images come back as Cartesian positions with their cells.
    fn relax_cell(
        &self,
        cell: &VariableCellSettings,
        structures: &[MolecularSystem],
        biases: Vec<(String, Arc<dyn mopt_rs::pes::term::PesTerm>)>,
    ) -> MoptResult<BandResult> {
        let config = self.config;
        check_options(config, cell)?;
        let first = structures
            .first()
            .cloned()
            .ok_or_else(|| MoptError::config("no images"))?;
        let image = Arc::new(AtomicUsize::new(0));
        let registry = crate::pycalc::python_registry();
        let wrapped = match &self.surface {
            BandSurface::Named(name) => wrap_constrained(
                &config.calculator_options,
                Some("a variable cell (-vc; ONIOM has no stress)"),
                &config.constraints,
                &biases,
                &[],
                NORMAL_FORCES,
                first.clone(),
                name,
                &registry,
            )?,
            BandSurface::Function(function) => Wrapped {
                start: first.clone(),
                electrons: false,
                calc: Box::new(FunctionCalculator {
                    function: Python::with_gil(|py| function.clone_ref(py)),
                    image: Arc::clone(&image),
                    rigid_body_invariant: self.rigid_body_invariant,
                }),
                micro: None,
                note: None,
                constraints: config.constraints.clone(),
                terms: biases,
            },
        };
        if wrapped.calc.uses_electrons() {
            first.spin_charge.check_electrons(&first.elements)?;
        }
        let CellChain {
            settings,
            chain,
            surface,
            mut keep,
            notes,
        } = cell_chain(
            config,
            cell,
            structures,
            wrapped.calc.as_ref(),
            wrapped.terms,
            &wrapped.constraints,
            self.output_dir.clone(),
        )?;
        let map = surface.map().clone();
        let indexed = Indexed {
            inner: &surface,
            image: Arc::clone(&image),
        };
        let modelled = chain_surrogate(config, false, false)?
            .map(|model| SurrogateSurface::new(&indexed, model));
        let relaxed_on = modelled
            .as_ref()
            .map_or(&indexed as &dyn ChainSurface, |m| m as &dyn ChainSurface);
        let mut machine = ChainRelaxation::new(settings, chain)?;
        let mut files = match &self.output_dir {
            Some(dir) => Some(ChainFiles::variable_cell(dir, "band", &map)?),
            None => None,
        };
        let observer = self.observer.as_ref();
        let outcome = {
            let mut observe = |report: &ChainReport, machine: &mut ChainRelaxation| {
                let mut flow = Flow::Continue;
                if let Some(files) = files.as_mut() {
                    flow = files.after_iteration(report, machine)?;
                }
                if let Some(keep) = keep.as_mut() {
                    keep.after_iteration(report, machine)?;
                }
                if let Some(callback) = observer {
                    let positions = report
                        .chain
                        .iter()
                        .map(|x| map.system(x).map(|s| s.coords_flat_bohr()))
                        .collect::<MoptResult<Vec<_>>>()?;
                    let stop: bool = Python::with_gil(|py| {
                        callback
                            .call1(
                                py,
                                (
                                    report.iteration,
                                    lists(&positions),
                                    report.energies.as_slice(),
                                    report.largest_force,
                                ),
                            )?
                            .extract::<Option<bool>>(py)
                    })
                    .map_err(|e| MoptError::calc(e.to_string()))?
                    .unwrap_or(false);
                    if stop {
                        flow = Flow::Stop;
                    }
                }
                Ok(flow)
            };
            relax_chain(&mut machine, relaxed_on, &mut [&mut observe])?
        };
        let (images, energies, cells) = match outcome.last {
            Some(last) => {
                let systems = last
                    .chain
                    .iter()
                    .map(|x| map.system(x))
                    .collect::<MoptResult<Vec<_>>>()?;
                let cells = systems
                    .iter()
                    .map(|s| {
                        s.active_cell()
                            .map(|c| c.h_bohr.iter().flatten().copied().collect::<Vec<f64>>())
                            .unwrap_or_default()
                    })
                    .collect();
                (
                    Some(
                        systems
                            .iter()
                            .map(MolecularSystem::coords_flat_bohr)
                            .collect(),
                    ),
                    Some(last.energies),
                    Some(cells),
                )
            }
            None => (None, None, None),
        };
        Ok(BandResult {
            iterations: outcome.iterations,
            end: outcome.end,
            images,
            energies,
            whole: None,
            cells,
            predicted: modelled.map(|m| m.counts().predicted),
            notes,
        })
    }
}

/// A chain of points of plain variables, as `nebmain` relaxes one
/// (`cli::nebmain::run_objective`).
struct VariableBand<'a> {
    config: &'a NebmainConfig,
    surface: BandSurface,
    output_dir: Option<PathBuf>,
    observer: Option<Py<PyAny>>,
}

impl VariableBand<'_> {
    fn relax(self, points: Vec<Col<f64>>) -> MoptResult<BandResult> {
        let config = self.config;
        if config.variable_cell.is_some() {
            return Err(MoptError::unsupported(
                "-vc needs structures in a periodic cell (symbols and cell_bohr); plain variables \
                 have no cell",
            ));
        }
        if !config.biases.is_empty()
            || !config.constraints.is_empty()
            || config.excited_state > 0
            || !config.model_function.is_empty()
        {
            return Err(MoptError::parse(
                "bias potentials, constraints, -es, and -mf need atoms; they cannot be used with \
                 an objective of plain variables",
            ));
        }
        let dimension = points[0].nrows();
        let (settings, chain) = variable_chain(config, &points, self.output_dir.clone())?;
        let objectives = ObjectiveRegistry::builtin();
        let objective = match &self.surface {
            BandSurface::Named(name) => Some(objectives.create(name, &config.calculator_options)?),
            BandSurface::Function(_) => None,
        };
        let function = match &self.surface {
            BandSurface::Function(f) => Some(PythonSurface {
                function: Python::with_gil(|py| f.clone_ref(py)),
                dimension,
                analytic: false,
            }),
            BandSurface::Named(_) => None,
        };
        let named = objective.as_deref().map(ObjectiveSurface);
        let base: &dyn ChainSurface = match (&named, &function) {
            (Some(n), _) => n,
            (None, Some(f)) => f,
            (None, None) => return Err(MoptError::config("relax_band: no surface")),
        };
        let modelled =
            chain_surrogate(config, false, false)?.map(|model| SurrogateSurface::new(base, model));
        let relaxed_on = modelled.as_ref().map_or(base, |m| m as &dyn ChainSurface);
        let mut machine = ChainRelaxation::new(settings, chain)?;
        let names = (1..=dimension).map(|i| format!("x{i}")).collect();
        let mut files = match &self.output_dir {
            Some(dir) => Some((
                ChainFiles::new(dir, "band", ChainPoints::Variables(names))?,
                ChainFigures::new(dir)
                    .keep_every_iteration(false)
                    .energy_axis(1.0, "Value of the objective"),
            )),
            None => None,
        };
        let mut keep = None;
        let observer = self.observer;
        let outcome = {
            let mut observe = |report: &ChainReport, machine: &mut ChainRelaxation| {
                observe_band(
                    report,
                    machine,
                    None,
                    &mut files,
                    &mut keep,
                    observer.as_ref(),
                )
            };
            relax_chain(&mut machine, relaxed_on, &mut [&mut observe])?
        };
        let (images, energies) = match outcome.last {
            Some(last) => (Some(last.chain), Some(last.energies)),
            None => (None, None),
        };
        Ok(BandResult {
            iterations: outcome.iterations,
            end: outcome.end,
            images,
            energies,
            whole: None,
            cells: None,
            predicted: modelled.map(|m| m.counts().predicted),
            notes: Vec::new(),
        })
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(relax_environment, m)?)?;
    m.add_function(wrap_pyfunction!(relax_band, m)?)?;
    Ok(())
}
