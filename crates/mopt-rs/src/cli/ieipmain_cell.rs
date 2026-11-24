//! `ieipmain -vc`: the searches of `ieipmain` in the variables of the atoms and the cell.
//!
//! The points of every method are points of the variables of the variable-cell optimization
//! made from the first structure ([`crate::opt::cell::VariableCell`]: quasi-Cartesian
//! coordinates of the atoms and scaled lattice vectors), on the enthalpy `E + P V` with the
//! bias potentials ([`cell_value`], the surface of `optmain -vc`): the vector and the metric of
//! the two-point searches of the mapper on a variable cell (`run_mapper` with `variable_cell`
//! in the step settings). The structures that are read may have different cells; every atom of
//! the second is taken to the periodic image nearest to it in the first
//! ([`super::nebmain_cell::joined_points`]). The files of the methods have the structures of
//! their points with their cells (extended XYZ), and so have the estimates and the path.
//!
//! Constraints are those of `optmain -vc` ([`crate::opt::cell_constraints`]) at the values of
//! the first structure: frozen atoms keep their fractional coordinates (fixed variables of
//! every method), the others (`-pc`, `-gfix`, `-uc`, `-vc_fix`) are taken out of the gradient
//! and the Hessian and the points put back onto them after every step, for the methods that
//! are made of steps (iEIP, the dimer, the spring pair, DS-AFIR); the methods that hold
//! constraints through a manifold of structures refuse them, with the reason.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `ieipmain` keeps the cell fixed).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faer::Col;

use super::ieipmain::{close, fill_energies, open_run, stem_of, IeipmainResult};
use super::ieipmain_methods::{check_fragments, fragment_test, run_method, Geometry};
use super::ieipmain_settings::{IeipMethod, IeipmainConfig};
use super::variable_cell::VariableCellSettings;
use crate::core_infra::log::Level;
use crate::domain::{MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{read_xyz_file, write_xyz_file};
use crate::math::linalg::col_norm;
use crate::opt::cell::VariableCell;
use crate::opt::cell_constraints::cell_constraints;
use crate::opt::vector_constraints::VectorConstraintSet;
use crate::path::points::{PointRequest, StepMap};
use crate::pes::calculator::Calculator;
use crate::pes::constraint::{Constraint, ConstraintKind};
use crate::pes::registry::CalculatorRegistry;
use crate::pes::term::PesTerm;
use crate::workflow::chain::{check_chain_terms, warm_start_label};
use crate::workflow::irc_cell::{cell_hessian_of, cell_value, CellParts};
use crate::workflow::journal::{RunJournal, JOURNAL_FILE};
use crate::workflow::pair_search::{SearchRun, SearchSurface, SurfaceValue};

/// What of `config` a search in the variables of the atoms and the cell cannot do, refused
/// with the reason before anything is calculated (`constrained`: there are constraints beyond
/// frozen atoms; `frozen`: there are frozen atoms).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn check_options(
    config: &IeipmainConfig,
    cell: &VariableCellSettings,
    constrained: bool,
    frozen: bool,
) -> MoptResult<()> {
    if !cell.forces.is_empty() {
        return Err(MoptError::unsupported(
            "-vc_afir with ieipmain: the artificial forces on the cell push it in an \
             optimization; the searches of ieipmain run on the surface itself",
        ));
    }
    if config.calculator_options.oniom.is_some() {
        return Err(MoptError::unsupported(
            "-oniom with -vc: a variable cell needs the stress of the whole surface; ONIOM has \
             no stress (as optmain -vc)",
        ));
    }
    if !super::ieipmain_coords::is_cartesian(&config.coordinates) {
        return Err(MoptError::unsupported(format!(
            "-coord {} with -vc: the searches run in the variables of the atoms and the cell \
             (those of optmain -vc); internal coordinates of the atoms have no part for the cell",
            config.coordinates
        )));
    }
    if let Some(name) = &config.surface_function {
        return Err(MoptError::unsupported(format!(
            "-mf {name} with {} and -vc: the method on a function of the states of the two \
             structures is set up for a fixed cell; -mf alone runs optmain -vc -mf",
            config.method.title()
        )));
    }
    if config.excited_states[0] != config.excited_states[1] {
        return Err(MoptError::unsupported(
            "-vc with two electronic states (-es of the two structures): the search in the \
             variables of the cell runs on one surface",
        ));
    }
    let made_of_steps = matches!(
        config.method,
        IeipMethod::Ieip | IeipMethod::Dimer | IeipMethod::SpringPair | IeipMethod::DsAfir
    );
    if constrained && !made_of_steps {
        return Err(MoptError::unsupported(format!(
            "the {} with -vc and -pc, -gfix, -uc, or -vc_fix: the method holds constraints \
             through a manifold of structures in one cell; iEIP, the dimer, the spring pair, and \
             DS-AFIR hold them in the variables of the atoms and the cell (-fix is held by every \
             method)",
            config.method.title()
        )));
    }
    if frozen && matches!(config.method, IeipMethod::Addf | IeipMethod::TwoPshs) {
        return Err(MoptError::unsupported(format!(
            "the {} with -vc and -fix: its scaled coordinates leave fixed atoms out only for \
             structures in Cartesian coordinates",
            config.method.title()
        )));
    }
    Ok(())
}

/// The enthalpy of the atoms and the cell as a surface of the searches (the values of
/// [`crate::workflow::irc_cell::CellObjective`]); the structures of its points, with their
/// cells, are what the files of the methods get. The directions of the constraints are taken
/// out of the gradient and the Hessian with the bias potentials.
pub struct CellSearchSurface<'a> {
    calc: &'a dyn Calculator,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
    map: VariableCell,
    start: MolecularSystem,
    excited_state: usize,
    constraints: Option<VectorConstraintSet>,
    iteration: AtomicUsize,
}

impl<'a> CellSearchSurface<'a> {
    /// The enthalpy of `calc` with `terms` in the variables `map` of structures like `start`,
    /// holding `constraints`.
    pub fn new(
        calc: &'a dyn Calculator,
        terms: Vec<(String, Arc<dyn PesTerm>)>,
        map: VariableCell,
        start: MolecularSystem,
        excited_state: usize,
        constraints: Option<VectorConstraintSet>,
    ) -> Self {
        Self {
            calc,
            terms,
            map,
            start,
            excited_state,
            constraints,
            iteration: AtomicUsize::new(0),
        }
    }

    fn parts<'b>(&'b self, terms: &'b [(String, Arc<dyn PesTerm>)]) -> CellParts<'b> {
        CellParts {
            calc: self.calc,
            terms,
            map: &self.map,
            start: &self.start,
            excited_state: self.excited_state,
            states: None,
            log: None,
        }
    }
}

impl SearchSurface for CellSearchSurface<'_> {
    fn description(&self) -> String {
        format!(
            "the enthalpy of the atoms and the cell ({} variables) of {}",
            self.map.dimension(),
            self.calc.description()
        )
    }

    fn structure(&self, x: &Col<f64>, _image: usize) -> Option<MolecularSystem> {
        self.map.system(x).ok()
    }

    fn begin_iteration(&self, iteration: usize) {
        self.iteration.store(iteration, Ordering::Relaxed);
    }

    fn in_order(&self) -> bool {
        self.calc.evaluates_in_order()
    }

    fn evaluate(
        &self,
        x: &Col<f64>,
        _image: usize,
        request: &PointRequest,
    ) -> MoptResult<SurfaceValue> {
        let step = self.iteration.load(Ordering::Relaxed);
        let parts = self.parts(&self.terms);
        let v = cell_value(&parts, x, step, "search point")?;
        // The enthalpy and its gradient without the bias potentials (iEIP uses both).
        let (energy, gradient) = if self.terms.is_empty() {
            (v.biased, v.slope.clone())
        } else {
            let bare = cell_value(&self.parts(&[]), x, step, "search point")?;
            (bare.biased, bare.slope)
        };
        let mut hessian = if request.hessian {
            Some(cell_hessian_of(&parts, x, step, "search point")?)
        } else {
            None
        };
        let mut biased_gradient = v.slope;
        if let Some(set) = &self.constraints {
            let frame = set.frame(x).context("the constraints of the search")?;
            biased_gradient = frame.project(&biased_gradient);
            hessian = hessian.map(|h| frame.step_hessian(&h));
        }
        Ok(SurfaceValue {
            energy,
            gradient,
            biased_energy: v.biased,
            biased_gradient,
            hessian,
        })
    }
}

/// `ieipmain -vc`: the method of `config` in the variables of the atoms and the cell.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub(super) fn run_cell(
    config: &IeipmainConfig,
    cell: &VariableCellSettings,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<IeipmainResult> {
    let frozen_atoms: Vec<usize> = config
        .constraints
        .iter()
        .filter_map(|c| match &c.kind {
            ConstraintKind::Frozen(atoms) => Some(atoms.iter().map(|a| a.index())),
            _ => None,
        })
        .flatten()
        .collect();
    let others: Vec<Constraint> = config
        .constraints
        .iter()
        .filter(|c| !matches!(c.kind, ConstraintKind::Frozen(_)))
        .cloned()
        .collect();
    check_options(config, cell, !others.is_empty(), !frozen_atoms.is_empty())?;
    let mut structures = Vec::with_capacity(config.inputs.len());
    for (k, path) in config.inputs.iter().enumerate() {
        let spin = config.states[k.min(1)].of_file(path)?;
        structures.push(read_xyz_file(path, spin, None).context(path.display())?);
    }
    if let [a, b] = structures.as_slice() {
        if a.elements != b.elements {
            return Err(MoptError::config(
                "the two structures must have the same atoms in the same order",
            ));
        }
        if a.spin_charge != b.spin_charge {
            return Err(MoptError::unsupported(
                "-vc with two structures in different states (charge, multiplicity): the search \
                 in the variables of the cell runs on one surface",
            ));
        }
    }
    let first = structures[0].clone();
    let biases = super::oniom_micro::build_terms(&config.biases, &first)?;
    check_chain_terms(&biases, "ieipmain", &warm_start_label)?;
    let evaluations_before = registry.counts();
    let wrapped = super::oniom_micro::wrap_constrained(
        &config.calculator_options,
        Some("a variable cell (-vc; ONIOM has no stress)"),
        &config.constraints,
        &biases,
        &[],
        super::oniom_micro::forces_of(&config.extra.criteria),
        first.clone(),
        &config.calculator,
        registry,
    )?;
    if wrapped.electrons {
        first.spin_charge.check_electrons(&first.elements)?;
    }
    let map = cell.coordinates(&first)?;
    let mut points = super::nebmain_cell::joined_points(&map, &structures)?;
    // Every point of the search holds the values of the first structure.
    let held = if others.is_empty() {
        None
    } else {
        Some(
            cell_constraints(&map, &others, &first)
                .context("the constraints at the first structure")?,
        )
    };
    let mut put_back = Vec::new();
    if let Some(set) = &held {
        for (k, p) in points.iter_mut().enumerate() {
            let on = set.restore(p)?;
            let moved = col_norm(&(&on - &*p));
            if moved > 1e-6 {
                put_back.push((k, moved));
            }
            *p = on;
        }
    }
    let mut frozen: Vec<usize> = frozen_atoms
        .iter()
        .flat_map(|a| (0..3).map(move |k| 3 * a + k))
        .collect();
    frozen.sort_unstable();
    frozen.dedup();
    let (dir, log) = open_run(config, base, console)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", wrapped.calc.as_ref());
    log.info(&format!("Calculator: {}", wrapped.calc.description()));
    for paper in wrapped.calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&config.calculator_options.software_path_report());
    log.info(&format!(
        "Variable cell (-vc): the search runs in the {} variables of the atoms and the cell made \
         from the first structure (quasi-Cartesian coordinates of the atoms and scaled lattice \
         vectors, weight {}); external pressure {} GPa: the surface is the enthalpy E + P V.",
        map.dimension(),
        crate::core_infra::log::python_float(cell.weight),
        crate::core_infra::log::python_float(cell.pressure_gpa)
    ));
    log.info("Reference: M. Gubler, M. Krummenacher, H. Huber, S. Goedecker, J. Comput. Phys. X 2023, 17, 100131 (variables of the variable-cell optimization)");
    for (k, moved) in &put_back {
        log.warn(&format!(
            "Structure {} was put onto the constraints at the values of the first structure \
             (moved {moved:.3e} in the variables): every point of the search holds the same \
             values.",
            k + 1
        ));
    }
    if let Some(set) = &held {
        log.info(&format!(
            "Constraints: {} held at every point of the search in the variables of the atoms \
             and the cell (the values of the first structure).",
            set.len()
        ));
    }
    if !frozen.is_empty() {
        log.info(&format!(
            "Fixed atoms (-fix): {} variables (their fractional coordinates).",
            frozen.len()
        ));
    }
    let surface = CellSearchSurface::new(
        &journaled,
        wrapped.terms.clone(),
        map.clone(),
        first.clone(),
        config.excited_states[0],
        held.clone(),
    );
    let restore: Option<StepMap> = held.clone().map(|set| {
        let map: StepMap = Arc::new(move |_from: &Col<f64>, to: &Col<f64>| set.restore(to));
        map
    });
    let to_cartesian = {
        let map = map.clone();
        Arc::new(move |x: &Col<f64>| map.system(x).ok().map(|s| s.coords_flat_bohr()))
    };
    let geometry = Geometry {
        bohr_lengths: true,
        frozen,
        fragments: Some(fragment_test(first.clone(), Some(to_cartesian.clone()))),
        step_map: restore.clone(),
        ..Geometry::default()
    };
    let stems = [
        stem_of(&config.inputs[0]),
        stem_of(config.inputs.get(1).unwrap_or(&config.inputs[0])),
    ];
    let run = SearchRun {
        save_steps: true,
        dir: &dir,
        stems: stems.clone(),
        log: &log,
        figures: config.extra.figures,
        commands: &super::pair_settings::COMMAND_PARSERS,
    };
    let mut outcome = run_method(config, &surface, &points, &geometry, &run)?;
    if let Some(map) = &restore {
        for (x, energy) in &mut outcome.transition_states {
            let on = map(x, x)?;
            if col_norm(&(&on - &*x)) > 1e-12 {
                *energy = None;
            }
            *x = on;
        }
    }
    fill_energies(&mut outcome, &surface, &log)?;
    check_fragments(
        &mut outcome,
        &fragment_test(first.clone(), Some(to_cartesian)),
        &run,
    )?;
    let write = |k: usize, x: &Col<f64>, energy: Option<f64>| -> MoptResult<PathBuf> {
        let suffix = if k == 0 {
            String::new()
        } else {
            format!("_{}", k + 1)
        };
        let path = dir.join(format!("{}_ts_guess{suffix}.xyz", stems[0]));
        write_xyz_file(&path, &map.system(x)?, energy.map(Quantity::new))?;
        Ok(path)
    };
    let mut result = close(&outcome, &write, None, &dir, &log)?;
    let ends = (
        &points[0],
        (points.len() > 1).then(|| &points[points.len() - 1]),
    );
    result.path = super::ieipmain_path::write_path_of(
        &config.method,
        &outcome,
        ends,
        &|x| map.system(x),
        &|s| {
            (s.n_atoms() == first.n_atoms() && s.has_pbc())
                .then(|| map.variables(s).ok())
                .flatten()
        },
        &stems[0],
        &dir,
    )?;
    if let Some(path) = &result.path {
        log.info(&format!("Path file path: {}", path.display()));
    }
    let follow = &config.extra.follow_up;
    if follow.on && !result.interrupted && !result.transition_states.is_empty() {
        // The optimizations from the estimates are those of optmain -vc (the options of the
        // cell are handed on); the constraints are resolved at every estimate, which holds
        // the values of the search.
        let s = first.spin_charge;
        result.optimized = super::ieipmain_followup::optimize_estimates(
            follow,
            config,
            &result.transition_states,
            (s.charge, s.multiplicity, config.excited_states[0]),
            None,
            &config.constraints,
            &dir,
            console,
            registry,
            &log,
        )?;
    }
    let (replayed, unused) = journal.replay_counts();
    if replayed > 0 {
        log.info(&format!(
            "Restart: {replayed} evaluations taken from {JOURNAL_FILE}{}.",
            if unused > 0 {
                format!(" ({unused} recorded ones no longer matched and were recomputed)")
            } else {
                String::new()
            }
        ));
    }
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(result)
}
