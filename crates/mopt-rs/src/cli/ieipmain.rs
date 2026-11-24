//! `ieipmain`: searches for transition states from two structures (the improved elastic image
//! pair method, by default; the two-point scaled hypersphere search, `-2pshs`; the growing
//! Newton trajectory, `-gnt`) or from one (the anharmonic downward distortion following,
//! `-addf`; the dimer method, `-use_dimer`; the spring pair method, `-use_spm`), and the
//! optimization of a model function of the states of two inputs (`-mf`, which runs `optmain`).
//!
//! The calculator options, the bias potentials, `-gfix`, `--config`, `--dump-config`, and
//! `--restart` are those of the other commands; objectives of plain variables (`-os mb2d`,
//! `-os expr:...`) take variables files as inputs. The estimate of the transition state is
//! written as `<stem>_ts_guess.xyz`; a search of order 1 from it (`optmain FILE -order 1`)
//! and a vibrational analysis confirm it.
//!
//! MultiOptPy: `ieip.py:iEIP` (`optimize`, `run`), `interface.py:call_ieipparser`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::Col;

use super::constraints::{pending_constraints, resolve_for_run};
use super::ieipmain_methods::{
    check_fragments, fragment_test, run_method, Geometry, MethodOutcome,
};
pub use super::ieipmain_settings::{config_from_cli, IeipMethod, IeipmainConfig};
use super::variables::{format_variables, read_variables_file};
use crate::coord::alignment::kabsch_align_bohr;
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{read_xyz_file, write_xyz_file};
use crate::math::linalg::col_norm;
use crate::path::manifold::PointManifold;
use crate::path::points::{PointRequest, StepMap};
use crate::pes::constraint::{breaks_rigid_body, ConstraintKind, ConstraintSet};
use crate::pes::objective::{check_dimension, ObjectiveRegistry};
use crate::pes::registry::CalculatorRegistry;
use crate::pes::term::frame_invariant;
use crate::workflow::chain::{check_chain_terms, warm_start_label};
use crate::workflow::journal::{RunJournal, JOURNAL_FILE};
use crate::workflow::pair_search::internal::is_breakdown;
use crate::workflow::pair_search::{
    ImageState, SearchRun, SearchSurface, StructureSurface, VariableSurface,
};

/// Result of a run.
#[derive(Debug, Clone)]
pub struct IeipmainResult {
    /// The run directory.
    pub directory: PathBuf,
    /// Whether the method ended on its criteria.
    pub converged: bool,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
    /// The files of the estimates of transition states (of a crossing for DS-AFIR between
    /// two structures in two states).
    pub transition_states: Vec<PathBuf>,
    /// The optimizations from the estimates (`-ieip_opt optimize=on`).
    pub optimized: Vec<super::ieipmain_followup::OptimizedEstimate>,
    /// The points of the search from the first structure to the second (`<stem>_path.xyz`),
    /// for the methods between structures.
    pub path: Option<PathBuf>,
}

/// MultiOptPy-style run directory `<date>/<stem>_iEIP_<timestamp>/` under `base`.
pub fn run_directory(base: &Path, input: &Path, now: &DateTime) -> PathBuf {
    // The name of the directory or file alone (an input such as `../pair` stays inside).
    let name = if input.is_dir() {
        input.file_name().map(|n| n.to_string_lossy().into_owned())
    } else {
        input.file_stem().map(|n| n.to_string_lossy().into_owned())
    };
    base.join(now.date_dir()).join(format!(
        "{}_iEIP_{}",
        name.unwrap_or_else(|| "input".to_string()),
        now.timestamp()
    ))
}

pub(super) fn stem_of(path: &Path) -> String {
    path.file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned())
}

/// Run `ieipmain` with the calculators of `registry`, writing the run directory under `base`.
pub fn run_with(
    config: &IeipmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<IeipmainResult> {
    if let IeipMethod::ModelFunction(name) = &config.method {
        return run_model_function(config, name, base, console, registry);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.calculator_options.pool_threads(config.threads))
        .build()
        .map_err(|e| MoptError::config(format!("thread pool: {e}")))?;
    pool.install(|| {
        let objectives = ObjectiveRegistry::builtin();
        match (
            objectives.contains(&config.calculator),
            &config.variable_cell,
        ) {
            (true, Some(_)) => Err(MoptError::unsupported(
                "-vc needs structures in a periodic cell; plain variables have no cell",
            )),
            (true, None) => run_variables(config, base, console, &objectives),
            (false, Some(cell)) => {
                super::ieipmain_cell::run_cell(config, cell, base, console, registry)
            }
            (false, None) => run_structures(config, base, console, registry),
        }
    })
    .context(config.input.display())
}

pub(super) fn open_run(
    config: &IeipmainConfig,
    base: &Path,
    console: Option<Level>,
) -> MoptResult<(PathBuf, Logger)> {
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => run_directory(base, &config.input, &DateTime::now_utc()),
    };
    std::fs::create_dir_all(&dir)?;
    let level = console.map(|l| if config.debug { Level::Debug } else { l });
    let log = level.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} ieipmain: {}",
        crate::VERSION,
        config
            .inputs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    ));
    log.info(&format!("Method: {}", config.method.title()));
    for other in &config.ignored {
        log.warn(&format!(
            "Also asked for: {other}; only one method runs (MultiOptPy's order: -mf, -gnt, \
             -addf, -2pshs, -use_dimer, -use_spm)."
        ));
    }
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document),
    )?;
    Ok((dir, log))
}

/// What is said and written at the end of a run.
pub(super) fn close(
    outcome: &MethodOutcome,
    write: &dyn Fn(usize, &Col<f64>, Option<f64>) -> MoptResult<PathBuf>,
    crossing: Option<&str>,
    dir: &Path,
    log: &Logger,
) -> MoptResult<IeipmainResult> {
    log.info(&format!("End: {}.", outcome.end));
    let mut files = Vec::new();
    for (k, (x, energy)) in outcome.transition_states.iter().enumerate() {
        let path = write(k, x, *energy)?;
        log.info(&format!(
            "{} estimate file path: {}",
            if crossing.is_some() {
                "Crossing"
            } else {
                "Transition state"
            },
            path.display()
        ));
        files.push(path);
    }
    match (files.is_empty(), crossing) {
        (true, _) => {}
        (false, Some(how)) => log.info(how),
        (false, None) => log.info(
            "A search of order 1 from an estimate (optmain FILE -order 1) and a vibrational \
             analysis (freqmain) confirm the transition state.",
        ),
    }
    if outcome.interrupted {
        log.info(&format!(
            "Continue with: ieipmain --restart {}",
            dir.display()
        ));
    }
    Ok(IeipmainResult {
        directory: dir.to_path_buf(),
        converged: outcome.converged,
        interrupted: outcome.interrupted,
        transition_states: files,
        optimized: Vec::new(),
        path: None,
    })
}

/// The structures of the inputs, each with its state, the second laid onto the first.
fn read_inputs(config: &IeipmainConfig, rigid: bool) -> MoptResult<Vec<MolecularSystem>> {
    let mut structures = Vec::with_capacity(config.inputs.len());
    for (k, path) in config.inputs.iter().enumerate() {
        let spin = config.states[k.min(1)].of_file(path)?;
        let sys = read_xyz_file(path, spin, None).context(path.display())?;
        structures.push(sys);
    }
    if let [a, b] = structures.as_slice() {
        if a.elements != b.elements {
            return Err(MoptError::config(
                "the two structures must have the same atoms in the same order",
            ));
        }
        let b = if a.has_pbc() {
            b.unwrapped_to(a)?
        } else if rigid {
            kabsch_align_bohr(b, a, false)?.0
        } else {
            b.clone()
        };
        structures[1] = b;
    }
    Ok(structures)
}

fn run_structures(
    config: &IeipmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<IeipmainResult> {
    let spin = config.states[0].of_file(&config.inputs[0])?;
    let first = read_xyz_file(&config.inputs[0], spin, None)?;
    let evaluations_before = registry.counts();
    // ONIOM microiterations: the methods move the core on the relaxed surface, every
    // structure with its own environment (each relaxation starts from that of the nearest
    // core structure evaluated; the first ones from the structures of the input).
    // Constraints and bias potentials are taken to the core.
    let biases = super::oniom_micro::build_terms(&config.biases, &first)?;
    // The search is never told which evaluation is accepted: terms of the time or of
    // accumulated state are refused (warm starts are copied for every evaluation).
    check_chain_terms(&biases, "ieipmain", &warm_start_label)?;
    let wrapped = super::oniom_micro::wrap_constrained(
        &config.calculator_options,
        None,
        &config.constraints,
        &biases,
        &[],
        super::oniom_micro::forces_of(&config.extra.criteria),
        first.clone(),
        &config.calculator,
        registry,
    )?;
    let calc = &wrapped.calc;
    let electrons = wrapped.electrons;
    let mut frozen = Vec::new();
    let mut others = Vec::new();
    for c in &wrapped.constraints {
        match &c.kind {
            ConstraintKind::Frozen(atoms) => frozen.extend(
                atoms
                    .iter()
                    .flat_map(|a| (0..3).map(move |k| 3 * a.index() + k)),
            ),
            _ => others.push(c.clone()),
        }
    }
    frozen.sort_unstable();
    frozen.dedup();
    // Rigid-body motion is left out only where the calculator, every bias potential, and
    // every constraint are blind to it.
    let invariant = frame_invariant(
        calc.is_rigid_body_invariant(),
        wrapped.terms.iter().map(|(_, t)| t.as_ref()),
    ) && !first.has_pbc()
        && !breaks_rigid_body(&others);
    let rigid = invariant && frozen.is_empty();
    let whole = read_inputs(config, rigid)?;
    // The constraints of a follow-up run, in the numbering of the whole system: the values
    // of the first structure, which the search holds.
    let follow_constraints = resolve_for_run(&config.constraints, &whole[0])
        .context("the constraints at the first structure")?;
    for (k, s) in whole.iter().enumerate() {
        if electrons {
            s.spin_charge
                .check_electrons(&s.elements)
                .context(format!("structure {}", k + 1))?;
        }
    }
    let structures = whole
        .into_iter()
        .map(|s| wrapped.structure(s))
        .collect::<MoptResult<Vec<_>>>()?;
    // Every point of the search holds the values of the first structure (explicit values as
    // given): resolved there once, and the structures of the input put onto them.
    let resolved = resolve_for_run(&wrapped.constraints, &structures[0])
        .context("the constraints at the first structure")?;
    if !pending_constraints(&resolved).is_empty() {
        return Err(MoptError::unsupported(
            "-pc eigvec: the direction of a Hessian eigenvector is resolved at the first \
             Hessian of an optimization; the searches of ieipmain have none to resolve it at",
        ));
    }
    let held = if resolved.is_empty() {
        None
    } else {
        Some(ConstraintSet::new(&resolved, &structures[0], None)?)
    };
    let mut put_back = Vec::new();
    let structures = match &held {
        Some(set) => structures
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let on = set.restore(s)?;
                let moved = col_norm(&(on.coords_flat_bohr() - s.coords_flat_bohr()));
                if moved > 1e-6 {
                    put_back.push((k, moved));
                }
                Ok(on)
            })
            .collect::<MoptResult<Vec<_>>>()?,
        None => structures,
    };
    // A combination that is refused leaves no run directory behind.
    super::ieipmain_coords::check(
        config,
        !frozen.is_empty() || !others.is_empty(),
        first.has_pbc(),
    )?;
    let (dir, log) = open_run(config, base, console)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    log.info(&format!("Calculator: {}", calc.description()));
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&config.calculator_options.software_path_report());
    if let Some(note) = &wrapped.note {
        log.info(note);
    }
    super::oniom_micro::report_relaxations(wrapped.micro.as_deref(), &log);
    for (k, moved) in &put_back {
        log.warn(&format!(
            "Structure {} was put onto the constraints at the values of the first structure \
             (moved {moved:.3e} Bohr): every point of the search holds the same values.",
            k + 1
        ));
    }
    if let Some(set) = &held {
        log.info(&format!(
            "Constraints: {} held at every point of the search (the values of the first \
             structure; the gradient and the Hessian without their directions{}).",
            set.len(),
            if others.is_empty() {
                ""
            } else {
                ", the points put back onto them after every step"
            }
        ));
    }
    let mut images = Vec::new();
    for (k, s) in structures.iter().enumerate() {
        let constraints = held.clone();
        log.info(&format!(
            "Structure {} ({}): charge {}, multiplicity {}, electronic state {}",
            k + 1,
            config.inputs[k].display(),
            s.spin_charge.charge,
            s.spin_charge.multiplicity,
            config.excited_states[k]
        ));
        images.push(ImageState {
            template: s.clone(),
            excited_state: config.excited_states[k],
            constraints,
        });
    }
    if images.len() == 1 {
        images.push(images[0].clone());
    }
    let surface = StructureSurface::new(&journaled, images)?.with_terms(wrapped.terms.clone());
    // -mf NAME with a method of mopt-rs: the method on that function of the two states.
    let surface = super::ieipmain_states::on_function(config, &structures, surface, &log)?;
    let crossing = super::ieipmain_methods::crossing_states(config, &structures, &surface, &log);
    let advice = crossing.map(|states| {
        format!(
            "The estimate is the mean of the two ends where they met; a search for the \
             crossing from it (optmain FILE {}) and its seam analysis (-freq) confirm it \
             (-ieip_opt optimize=on runs it).",
            states.options()
        )
    });
    if !frozen.is_empty() {
        log.info(&format!(
            "Fixed atoms (-fix): {} coordinates.",
            frozen.len()
        ));
    }
    let template = structures[0].clone();
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
    let points: Vec<Col<f64>> = structures
        .iter()
        .map(MolecularSystem::coords_flat_bohr)
        .collect();
    // Constraints beyond fixed atoms: every step of the method ends on them.
    let restore: Option<StepMap> = match (&held, others.is_empty()) {
        (Some(set), false) => {
            let (set, at) = (set.clone(), template.clone());
            Some(Arc::new(move |_from: &Col<f64>, to: &Col<f64>| {
                Ok(set
                    .restore(&at.with_flat_coords_bohr(to)?)?
                    .coords_flat_bohr())
            }))
        }
        _ => None,
    };
    // The same constraints as a manifold of the points, for the methods whose points are
    // not made by steps from the point before (strings, hyperspheres, the corrector of GNT).
    let manifold = match (&held, others.is_empty()) {
        (Some(set), false) => Some(Arc::new(PointManifold::new(set.clone(), template.clone())?)),
        _ => None,
    };
    // -coord: the methods move in the coordinates (or take their steps through them); where
    // the coordinates break down, they are set up anew or the next ones take over.
    let mut plan = super::ieipmain_coords::CoordinatePlan::new(config);
    let mut outcome = loop {
        let setup = plan.set_up(config, &surface, &structures, &points, &log)?;
        let (moving, geometry): (&dyn SearchSurface, Geometry) = match &setup.internal {
            Some(internal) => (
                internal,
                Geometry {
                    bohr_lengths: true,
                    fragments: Some(fragment_test(
                        template.clone(),
                        Some(internal.to_cartesian()),
                    )),
                    ..Geometry::default()
                },
            ),
            None => (
                &surface,
                Geometry {
                    template: Some(template.clone()),
                    rigid,
                    invariant,
                    frozen: frozen.clone(),
                    fragments: Some(fragment_test(template.clone(), None)),
                    step_map: setup.step_map.clone().or_else(|| restore.clone()),
                    manifold: manifold.clone(),
                    ..Geometry::default()
                },
            ),
        };
        match run_method(config, moving, &setup.points, &geometry, &run) {
            Ok(mut found) => {
                // An estimate made of points (the midpoint of the images of iEIP) is put
                // onto the constraints as well.
                if let (Some(map), None) = (&restore, &setup.internal) {
                    for (x, energy) in &mut found.transition_states {
                        let on = map(x, x)?;
                        if col_norm(&(&on - &*x)) > 1e-12 {
                            *energy = None;
                        }
                        *x = on;
                    }
                }
                fill_energies(&mut found, moving, &log)?;
                if let Some(internal) = &setup.internal {
                    for (x, _) in &mut found.transition_states {
                        *x = internal.cartesian(x)?;
                    }
                    for (x, _) in found.path.iter_mut().flatten() {
                        *x = internal.cartesian(x)?;
                    }
                }
                break found;
            }
            Err(e) if plan.active() && is_breakdown(&e) => {
                log.warn(&format!("The coordinates broke down: {e}"));
                plan.broke_down(setup.last_structure(), &log);
            }
            Err(e) => return Err(e),
        }
    };
    if !super::ieipmain_coords::is_cartesian(&config.coordinates) {
        log.info(&plan.summary());
    }
    check_fragments(&mut outcome, &fragment_test(template.clone(), None), &run)?;
    // The estimates (the midpoint of iEIP is the mean of two points on the constraints, off
    // them to second order) are put onto the constraints before they are written.
    if let Some(map) = &restore {
        for (x, _) in &mut outcome.transition_states {
            *x = map(x, x)?;
        }
    }
    let write = |k: usize, x: &Col<f64>, energy: Option<f64>| -> MoptResult<PathBuf> {
        let suffix = if k == 0 {
            String::new()
        } else {
            format!("_{}", k + 1)
        };
        let kind = if crossing.is_some() {
            "crossing_guess"
        } else {
            "ts_guess"
        };
        let path = dir.join(format!("{}_{kind}{suffix}.xyz", stems[0]));
        write_xyz_file(
            &path,
            &template.with_flat_coords_bohr(x)?,
            energy.map(Quantity::new),
        )?;
        Ok(path)
    };
    let mut result = close(&outcome, &write, advice.as_deref(), &dir, &log)?;
    // The points of the search from the first structure to the second, under one name.
    result.path =
        super::ieipmain_path::write_path(&config.method, &outcome, &structures, &stems[0], &dir)?;
    if let Some(path) = &result.path {
        log.info(&format!("Path file path: {}", path.display()));
    }
    // On a function of two states the low points of the path are candidates for crossings,
    // and the optimizations after the search start from them.
    let mut crossing = crossing;
    if config.surface_function.is_some() {
        let found = super::ieipmain_states::crossing_candidates(
            config,
            &structures,
            &surface,
            &outcome,
            &stems[0],
            &dir,
            &log,
        )?;
        if let Some((candidates, states)) = found {
            result.transition_states = candidates;
            crossing = Some(states);
        }
    }
    let follow = &config.extra.follow_up;
    if follow.on && !result.interrupted && !result.transition_states.is_empty() {
        let first = structures[0].spin_charge;
        let optimized = super::ieipmain_followup::optimize_estimates(
            follow,
            config,
            &result.transition_states,
            (first.charge, first.multiplicity, config.excited_states[0]),
            crossing.as_ref(),
            &follow_constraints,
            &dir,
            console,
            registry,
            &log,
        )?;
        result.optimized = optimized;
    }
    if let Some(relaxed) = &wrapped.micro {
        // The files of the run with the whole system.
        super::oniom_micro::finish(relaxed, None, &dir, &log)?;
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

/// The energy of an estimate that the method did not evaluate (the midpoint of iEIP).
pub(super) fn fill_energies(
    outcome: &mut MethodOutcome,
    surface: &dyn SearchSurface,
    log: &Logger,
) -> MoptResult<()> {
    if outcome.interrupted {
        return Ok(());
    }
    for (x, energy) in &mut outcome.transition_states {
        if energy.is_none() {
            let v = surface.evaluate(x, 0, &PointRequest::energy(x.clone()))?;
            log.info(&format!(
                "Energy at the estimate of the transition state: {:.10} (with bias potentials {:.10})",
                v.energy, v.biased_energy
            ));
            *energy = Some(v.biased_energy);
        }
    }
    crate::workflow::pair_search::log_notes(surface, log);
    Ok(())
}

fn run_variables(
    config: &IeipmainConfig,
    base: &Path,
    console: Option<Level>,
    objectives: &ObjectiveRegistry,
) -> MoptResult<IeipmainResult> {
    if !config.biases.is_empty()
        || !config.constraints.is_empty()
        || config.excited_states != [0, 0]
    {
        return Err(MoptError::parse(
            "bias potentials, constraints, and -es need atoms; they cannot be used with an \
             objective of plain variables",
        ));
    }
    let evaluations_before = objectives.counts();
    let objective = objectives.create(&config.calculator, &config.calculator_options)?;
    let mut points = Vec::new();
    let mut names = None;
    let mut json = false;
    for path in &config.inputs {
        let vars = read_variables_file(path)?;
        check_dimension(objective.as_ref(), vars.values.len())?;
        names = names.or(vars.names.clone());
        json = vars.json;
        points.push(Col::from_fn(vars.values.len(), |i| vars.values[i]));
    }
    if points.len() == 2 && points[0].nrows() != points[1].nrows() {
        return Err(MoptError::config(
            "the two variables files have different numbers of variables",
        ));
    }
    let n = points[0].nrows();
    let names = names.unwrap_or_else(|| objective.variable_names(n));
    let (dir, log) = open_run(config, base, console)?;
    log.info(&format!(
        "Objective: {} ({n} plain variables: {})",
        objective.description(),
        names.join(", ")
    ));
    if config.restart_dir.is_some() {
        log.info("Restart: the run is repeated from the start (objectives are cheap).");
    }
    let surface = VariableSurface(objective.as_ref());
    if !super::ieipmain_coords::is_cartesian(&config.coordinates) {
        return Err(MoptError::unsupported(format!(
            "-coord {}: plain variables are their own coordinates",
            config.coordinates
        )));
    }
    let geometry = Geometry::default();
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
    fill_energies(&mut outcome, &surface, &log)?;
    let write = |k: usize, x: &Col<f64>, energy: Option<f64>| -> MoptResult<PathBuf> {
        let suffix = if k == 0 {
            String::new()
        } else {
            format!("_{}", k + 1)
        };
        let ext = if json { "json" } else { "txt" };
        let path = dir.join(format!("{}_ts_guess{suffix}.{ext}", stems[0]));
        let values: Vec<f64> = x.iter().copied().collect();
        std::fs::write(
            &path,
            format_variables(&names, &values, json, energy.unwrap_or(f64::NAN)),
        )?;
        Ok(path)
    };
    let result = close(&outcome, &write, None, &dir, &log)?;
    log.info(
        &objectives
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(result)
}

/// `-mf NAME`: `optmain -mf` at the first structure with the states of the two inputs
/// (MultiOptPy moves both structures by the step of the first, so that only the first
/// structure matters; mopt-rs says so when the second differs).
fn run_model_function(
    config: &IeipmainConfig,
    name: &str,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<IeipmainResult> {
    let mut p = config.arguments.clone();
    p.positionals = vec![config.inputs[0].display().to_string()];
    let state = |k: usize| config.states[k].of_file(&config.inputs[k.min(config.inputs.len() - 1)]);
    let (s1, s2) = (state(0)?, state(1)?);
    p.values.remove("model_function_mode");
    p.present.remove("model_function_mode");
    p.values.insert(
        "model_function".to_string(),
        vec![
            name.to_string(),
            s2.charge.to_string(),
            s2.multiplicity.to_string(),
            config.excited_states[1].to_string(),
        ],
    );
    p.present.insert("model_function".to_string());
    for (key, value) in [
        ("electronic_charge", s1.charge.to_string()),
        ("spin_multiplicity", s1.multiplicity.to_string()),
        ("excited_state", config.excited_states[0].to_string()),
        ("NSTEP", config.max_iterations.to_string()),
    ] {
        p.values.insert(key.to_string(), vec![value]);
        p.present.insert(key.to_string());
    }
    p.values.remove("unpaired_electrons");
    p.present.remove("unpaired_electrons");
    let mut optmain = super::optmain::OptmainConfig::from_args(&p)?;
    optmain.restart_dir.clone_from(&config.restart_dir);
    if let Some(level) = console {
        let log = Logger::console(level);
        log.info(&format!(
            "ieipmain -mf {name}: optmain -mf at {} with the states (charge {}, multiplicity \
             {}, state {}) and (charge {}, multiplicity {}, state {}).",
            config.inputs[0].display(),
            s1.charge,
            s1.multiplicity,
            config.excited_states[0],
            s2.charge,
            s2.multiplicity,
            config.excited_states[1]
        ));
        if let [a, b] = config.inputs.as_slice() {
            let read = |path: &Path| std::fs::read_to_string(path).unwrap_or_default();
            if read(a) != read(b) {
                log.warn(&format!(
                    "{} differs from {}: only the first structure is optimized (both states \
                     at it), as in MultiOptPy.",
                    b.display(),
                    a.display()
                ));
            }
        }
    }
    let level = console.map(|l| if config.debug { Level::Debug } else { l });
    let runs = super::optmain::run_with(&optmain, base, level, registry)?;
    let run = runs
        .into_iter()
        .next()
        .ok_or_else(|| MoptError::calc("optmain gave no result"))?;
    Ok(IeipmainResult {
        directory: run.directory,
        converged: run.converged,
        interrupted: run.interrupted,
        transition_states: Vec::new(),
        optimized: Vec::new(),
        path: None,
    })
}
