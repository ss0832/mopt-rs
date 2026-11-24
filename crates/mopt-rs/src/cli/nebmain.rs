//! `nebmain`: a chain of images between structures (or between points of an objective of
//! plain variables) relaxed towards a reaction path.
//!
//! Without further options the chain is relaxed the way `nebmain` is used in the articles
//! of MultiOptPy's author (`-ns 20 -nd 0.5 -adpred 2`; a model Hessian, `lindh2007d3`, in
//! the place of their `-modelhess o1numhess`, which can be asked for): 20
//! iterations, model Hessians of the images at the start, half a step along the
//! forces and half a quasi-Newton step, images 0.5 Angstrom apart, and the images put
//! towards the predicted maxima of the energy profile every second iteration. MultiOptPy's
//! own defaults are `-ns 10 -modelhess none -nd 0 -adpred 0`.
//!
//! The input is a file of several structures, several files, or a directory of numbered
//! files (structures), or two or more variables files (`-os mb2d`, `-os expr:...`). The
//! calculator options, the bias potentials, the constraints (`-fix`, `-pc`, `-gfix`,
//! `-uc`), `-es`, `--config`, `--dump-config`, and `--restart` are those of `optmain`.
//! The parts are in [`crate::path::band`] and [`crate::workflow::chain`]; the files are
//! those of MultiOptPy ([`crate::workflow::chain_files`]) and the figures of
//! [`crate::workflow::chain_figures`], drawn anew after every iteration.
//!
//! MultiOptPy: `neb.py:NEB.run`, `interface.py:call_nebparser`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use faer::Col;

use super::nebmain_followup::{optimize_images, OptimizedImage};
pub use super::nebmain_settings::{
    NebmainConfig, DEFAULT_FIRST_PATH, DEFAULT_GP_LENGTH, DEFAULT_GP_LENGTH_DISTANCES,
    DEFAULT_ITERATIONS, DEFAULT_MODEL_HESSIAN, DEFAULT_PREDICTED_EVERY, DEFAULT_SPACING_ANGSTROM,
};
use super::spec::NEBMAIN;
use super::variables::read_variables_file;
use crate::coord::system::CoordinateRegistry;
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::path::band::machine::{ChainEnd, ChainRelaxation, ChainReport, ChainSettings};
use crate::path::band::redistribute::bezier_by_spacing;
use crate::path::band::surrogate::Features;
use crate::path::band::{ChainKind, Compatibility};
use crate::path::interp::{FirstPath, PathFrame, PathSettings};
use crate::pes::constraint::{Constraint, ConstraintKind};
use crate::pes::counting::EvaluationCounts;
use crate::pes::objective::{check_dimension, ObjectiveRegistry};
use crate::pes::registry::CalculatorRegistry;
use crate::pes::term::frame_invariant;
use crate::workflow::chain::{
    check_chain_terms, relax_chain, warm_start_label, CalculatorSurface, ChainConstraints,
    ChainObserver, ChainSurface, Flow, KeepConstraints, ObjectiveSurface, SharedChainConstraints,
};
use crate::workflow::chain_figures::{ChainFigures, SurfaceMap, MAP_POINTS};
use crate::workflow::chain_files::{ChainCandidates, ChainFiles, ChainPoints};
use crate::workflow::chain_input::{
    first_chain, first_chain_in, first_structure_file, join_in_the_cell_centred, read_structures,
    FirstChain,
};
use crate::workflow::chain_internal::InternalSurface;
use crate::workflow::chain_surrogate::SurrogateSurface;
use crate::workflow::journal::{RunJournal, JOURNAL_FILE};
use crate::workflow::observers::StopFile;

/// Result of a run.
#[derive(Debug, Clone)]
pub struct NebmainResult {
    /// The run directory.
    pub directory: PathBuf,
    /// Iterations done.
    pub iterations: usize,
    /// How the relaxation ended (`None`: stopped by an observer or a `STOP` file).
    pub end: Option<ChainEnd>,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
    /// Energies of the images of the last chain that was evaluated (Hartree).
    pub energies: Vec<f64>,
    /// The candidates for transition states and the file of the last chain.
    pub candidates: Option<ChainCandidates>,
    /// The images that were optimized after the relaxation (`-neb_opt optimize=`).
    pub optimized: Vec<OptimizedImage>,
}

/// MultiOptPy-style run directory `<date>/<stem>_NEB_<timestamp>/` under `base`.
pub fn run_directory(base: &Path, input: &Path, now: &DateTime) -> PathBuf {
    let stem = input.with_extension("");
    base.join(now.date_dir()).join(format!(
        "{}_NEB_{}",
        stem.to_string_lossy(),
        now.timestamp()
    ))
}

/// Run `nebmain` with the calculators of `registry`, writing the run directory under
/// `base`.
pub fn run_with(
    config: &NebmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<NebmainResult> {
    run_with_observers(config, base, console, registry, &mut [])
}

/// Notification after the band trajectory and its energies have been saved.
pub(super) type SavedBand<'a> = dyn FnMut(usize, &ChainFiles, &[f64]) + Send + 'a;

/// [`run_with`] with further observers: they are called after every iteration, after the
/// files, the figures, and the log were written, and may stop the run or replace the
/// chain (any processing between two iterations). They run on a thread of the pool of
/// the command, so they must be `Send`.
pub fn run_with_observers(
    config: &NebmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    extra: &mut [&mut (dyn ChainObserver + Send)],
) -> MoptResult<NebmainResult> {
    run_observed(config, base, console, registry, extra, &mut |_, _, _| {})
}

pub(super) fn run_observed(
    config: &NebmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    extra: &mut [&mut (dyn ChainObserver + Send)],
    saved: &mut SavedBand<'_>,
) -> MoptResult<NebmainResult> {
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
            (true, None) => run_objective(config, base, console, &objectives, extra, saved),
            (false, Some(cell)) => run_cell(config, cell, base, console, registry, extra, saved),
            (false, None) => run_structures(config, base, console, registry, extra, saved),
        }
    })
    .context(config.inputs[0].display())
}

fn stem_of(input: &Path) -> String {
    input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned())
}

fn open_run(
    config: &NebmainConfig,
    base: &Path,
    console: Option<Level>,
) -> MoptResult<(PathBuf, Logger)> {
    let input = &config.inputs[0];
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => run_directory(base, input, &DateTime::now_utc()),
    };
    std::fs::create_dir_all(&dir)?;
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} nebmain: {}",
        crate::VERSION,
        config
            .inputs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    ));
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document),
    )?;
    Ok((dir, log))
}

/// What is said about the settings at the start of a run.
fn describe(config: &NebmainConfig, chain: &ChainSettings, images: usize, log: &Logger) {
    log.info(&format!(
        "Chain: {images} images, {} iterations, ends {}",
        chain.max_iterations,
        match (chain.ends.first, chain.ends.last) {
            (false, false) => "optimized (-fe 0)",
            (true, false) => "first fixed, last optimized (-fe 1)",
            (false, true) => "first optimized, last fixed (-fe 2)",
            (true, true) => "fixed (-fe 3)",
        }
    ));
    match (&chain.rfo, config.exact_hessians, &config.model_hessian) {
        (None, _, _) => log.info("Steps: along the forces (no Hessians)."),
        (Some(rfo), exact, model) => log.info(&format!(
            "Steps: {:.2} quasi-Newton, {:.2} along the forces; Hessians of the images {} every {} iterations.",
            rfo.ratio,
            1.0 - rfo.ratio,
            if exact {
                "from the calculator (-fc)".to_string()
            } else if chain.kind == ChainKind::Variables {
                // No atoms, no model: the second derivatives of the objective.
                "of the objective".to_string()
            } else {
                format!("from -modelhess {}", model.as_deref().unwrap_or("?"))
            },
            chain.hessian_every.unwrap_or(0)
        )),
    }
    let mut placed = Vec::new();
    if let Some(every) = chain.equal_spacing_every {
        placed.push(format!("equal spacing every {every} iterations (-ad)"));
    }
    if let Some(every) = chain.predicted_maxima_every {
        placed.push(format!(
            "towards the predicted maxima every {every} iterations (-adpred){}",
            match chain.gathering {
                Some(g) => format!(", {:.1} times as dense around them", g.weight + 1.0),
                None => String::new(),
            }
        ));
    }
    log.info(&format!(
        "Redistribution: {}",
        if placed.is_empty() {
            "none".to_string()
        } else {
            placed.join("; ")
        }
    ));
    if let Some(after) = chain.climb_after {
        log.info(&format!(
            "Climbing images after iteration {after} (-cineb)."
        ));
    }
    match chain.force_tolerance {
        Some(tolerance) => log.info(&format!(
            "The run ends when the largest force of the images that move is below {tolerance:.1e} \
             Hartree/Bohr (-neb_opt tolerance=; 0: all iterations are run)."
        )),
        None => log.info("All iterations are run (-neb_opt tolerance=0)."),
    }
    if let Some(diis) = chain.diis {
        log.info(&format!(
            "Extrapolation over the last {} chains once the largest force is below {:.1e} \
             (-neb_opt diis=, diis_start=).",
            diis.points, diis.start_force
        ));
    }
    if let Some(added) = chain.added {
        log.info(&format!(
            "Adaptive chain (-aneb): {} images on either side of every maximum after every {} iterations.",
            added.images, added.every
        ));
    }
    if let Some(rfo) = &chain.rfo {
        if rfo.global.is_some() {
            log.info(
                "Quasi-Newton steps: one for the whole chain (-gqnt, -neb_opt global=on; experimental).",
            );
        } else if rfo.sweep {
            log.info(
                "Quasi-Newton steps: from the highest image down, the step of the neighbour uphill counted in (-neb_opt sweep=on).",
            );
        }
    }
    if chain.compatibility == Compatibility::MultiOptPy {
        log.info("Form of the steps: MultiOptPy's (-neb_opt form=multioptpy).");
    }
}

/// How the energies of a chain are written: `factor` times the difference in `unit`, the
/// values themselves in `absolute` (kcal/mol and Hartree for structures; the objective of
/// plain variables as it is).
#[derive(Clone, Copy)]
struct EnergyUnits {
    factor: f64,
    unit: &'static str,
    absolute: &'static str,
}

impl EnergyUnits {
    /// The enthalpy of the images of a variable cell.
    const ENTHALPY: Self = Self {
        factor: Units::HARTREE_TO_KCAL_MOL,
        unit: "kcal/mol",
        absolute: "Hartree",
    };

    fn of(points: &ChainPoints) -> Self {
        match points {
            ChainPoints::Structures(_) => Self {
                factor: Units::HARTREE_TO_KCAL_MOL,
                unit: "kcal/mol",
                absolute: "Hartree",
            },
            ChainPoints::Variables(_) => Self {
                factor: 1.0,
                unit: "(objective)",
                absolute: "(objective)",
            },
        }
    }
}

/// The line of an iteration in the log.
fn progress(report: &ChainReport, units: EnergyUnits, log: &Logger) {
    let first = report.energies[0];
    let (top, highest) =
        report
            .energies
            .iter()
            .enumerate()
            .fold(
                (0, f64::NEG_INFINITY),
                |m, (i, e)| if *e > m.1 { (i, *e) } else { m },
            );
    let (worst, force) = report
        .force_max
        .iter()
        .enumerate()
        .fold((0, 0.0_f64), |m, (i, f)| if *f > m.1 { (i, *f) } else { m });
    log.info(&format!(
        "ITR. {:4}  highest image {top:3}: {:10.4} {} above the first ({:.10} {})  largest force {force:9.3e} (image {worst})  maxima {:?}{}{}",
        report.iteration,
        units.factor * (highest - first),
        units.unit,
        highest,
        units.absolute,
        report.maxima,
        if report.climbing.is_empty() {
            String::new()
        } else {
            format!("  climbing {:?}", report.climbing)
        },
        if report.redistributed { "  redistributed" } else { "" }
    ));
    for failure in &report.failures {
        let what = match (failure.retaken_at, failure.dropped) {
            (Some(share), _) if share > 0.0 => format!(
                "its step was taken again at {share} of its length, where the calculation went through"
            ),
            (Some(_), _) => {
                "it was put back where it was calculated in the iteration before".to_string()
            }
            (None, _) => "it was taken out of the chain".to_string(),
        };
        log.warn(&format!(
            "The calculation of image {} failed ({}); {what} (-neb_opt failed=retry|drop|stop).",
            failure.image, failure.message
        ));
    }
    if report.added > 0 {
        log.info(&format!(
            "{} images were added around the maxima (-aneb): the chain has {} images now.",
            report.added,
            report.chain.len() + report.added
        ));
    }
    // What the calculators said: warnings, <S**2>, the output of a program.
    for note in &report.notes {
        // A warning has its mark already; the log writes it once.
        match note.strip_prefix("Warning: ") {
            Some(warning) => log.warn(warning),
            None => log.info(note),
        }
    }
    // The images one by one (E with the bias potentials, which the chain is relaxed on;
    // without them in the last column when there are any).
    let biased = !report.bare_energies.is_empty();
    let mut table = format!(
        "  image   E - E(first)          E             force        force     to the next{}\n\
         \x20       {:>12}   {:>14}          largest        RMS        image{}\n",
        if biased { "   E without bias" } else { "" },
        units.unit,
        units.absolute,
        if biased {
            format!("  {:>16}", units.absolute)
        } else {
            String::new()
        }
    );
    for (image, energy) in report.energies.iter().enumerate() {
        let mut marks = Vec::new();
        if report.maxima.contains(&image) {
            marks.push("maximum");
        }
        if report.minima.contains(&image) {
            marks.push("minimum");
        }
        if report.climbing.contains(&image) {
            marks.push("climbing");
        }
        if report.held.contains(&image) {
            marks.push("held");
        }
        let _ = writeln!(
            table,
            "  {image:5} {:14.4} {:18.10} {:12.3e} {:12.3e} {:>12}{}  {}",
            units.factor * (energy - first),
            energy,
            report.force_max.get(image).copied().unwrap_or(f64::NAN),
            report.force_rms.get(image).copied().unwrap_or(f64::NAN),
            report
                .to_next
                .get(image)
                .filter(|d| d.is_finite() && image + 1 < report.energies.len())
                .map_or_else(|| "-".to_string(), |d| format!("{d:.4}")),
            report
                .bare_energies
                .get(image)
                .map_or_else(String::new, |e| format!("  {e:18.10}")),
            marks.join(", ")
        );
    }
    log.info(table.trim_end_matches('\n'));
}

/// Run the relaxation with the observers of the command and report what it left.
#[allow(clippy::too_many_arguments)]
fn relax(
    config: &NebmainConfig,
    settings: ChainSettings,
    chain: Vec<Col<f64>>,
    surface: &dyn ChainSurface,
    points: ChainPoints,
    map: Option<SurfaceMap>,
    keep: Option<KeepConstraints>,
    internal: Option<&InternalSurface<'_>>,
    cell: Option<&crate::opt::cell::VariableCell>,
    name: &str,
    dir: &Path,
    log: &Logger,
    extra: &mut [&mut (dyn ChainObserver + Send)],
    saved: &mut SavedBand<'_>,
) -> MoptResult<NebmainResult> {
    describe(config, &settings, chain.len(), log);
    let tolerance = settings.force_tolerance;
    log.info(&format!("Surface: {}", surface.description()));
    // In internal coordinates the points of the chain are not structures: the steps
    // know nothing of atoms, and the files get the structures of the points.
    let template = match (&points, internal) {
        (ChainPoints::Structures(template), None) => Some((**template).clone()),
        _ => None,
    };
    let (energy_axis, units) = match (&points, cell) {
        // The points of a variable cell are structures with their cells; the profile is the
        // enthalpy.
        (_, Some(_)) => (
            (Units::HARTREE_TO_KCAL_MOL, "Enthalpy (kcal/mol)"),
            EnergyUnits::ENTHALPY,
        ),
        (ChainPoints::Structures(_), None) => (
            (Units::HARTREE_TO_KCAL_MOL, "Energy (kcal/mol)"),
            EnergyUnits::of(&points),
        ),
        (ChainPoints::Variables(_), None) => {
            ((1.0, "Value of the objective"), EnergyUnits::of(&points))
        }
    };
    let mut machine = ChainRelaxation::new(settings, chain)?;
    if let Some(template) = template {
        machine = machine.with_structure(template)?;
    }
    let mut files = match cell {
        Some(map) => ChainFiles::variable_cell(dir, name, map)?,
        None => ChainFiles::new(dir, name, points)?,
    };
    let mut figures = ChainFigures::new(dir)
        .keep_every_iteration(false)
        .energy_axis(energy_axis.0, energy_axis.1);
    if let Some(map) = map {
        figures = figures.on_surface(map);
    }
    let mut stop = StopFile::new(dir, log.clone());
    let mut interrupted = false;
    let mut energies = Vec::new();
    let mut largest_force = 0.0;
    let draw = config.figures;
    let mut keep = keep;
    let outcome = {
        let mut command = |report: &ChainReport, machine: &mut ChainRelaxation| {
            progress(report, units, log);
            let mut flow = match internal {
                Some(internal) => {
                    let written = ChainReport {
                        chain: internal.structures(&report.chain)?,
                        ..report.clone()
                    };
                    files.after_iteration(&written, machine)?
                }
                None => files.after_iteration(report, machine)?,
            };
            if draw {
                figures.after_iteration(report, machine)?;
            }
            saved(report.iteration, &files, &report.energies);
            if let Some(keep) = keep.as_mut() {
                keep.after_iteration(report, machine)?;
            }
            energies.clone_from(&report.energies);
            largest_force = report.largest_force;
            log.flush();
            for observer in extra.iter_mut() {
                if observer.after_iteration(report, machine)? == Flow::Stop {
                    flow = Flow::Stop;
                }
            }
            if stop.check() {
                interrupted = true;
                flow = Flow::Stop;
            }
            Ok(flow)
        };
        relax_chain(&mut machine, surface, &mut [&mut command])?
    };
    let candidates = files.candidates();
    match outcome.end {
        Some(ChainEnd::Converged) => log.info(&format!(
            "The chain is relaxed: the forces of the images that move are below the tolerance after {} iterations.",
            outcome.iterations
        )),
        Some(ChainEnd::IterationLimit) => {
            log.info(&format!(
                "{} iterations were run (-ns).",
                outcome.iterations
            ));
            if let Some(tolerance) = tolerance {
                log.info(&format!(
                    "The largest force of the images that move is {largest_force:.3e} Hartree/Bohr, \
                     above the tolerance of {tolerance:.1e}: the chain is not relaxed yet. \
                     `nebmain --restart {}` with a larger -ns goes on.",
                    dir.display()
                ));
            }
        }
        None => log.info(&format!(
            "The run ended after {} iterations before the chain was through.",
            outcome.iterations
        )),
    }
    if let Some(found) = &candidates {
        let first = energies.first().copied().unwrap_or(0.0);
        for (k, image) in found.images.iter().enumerate() {
            let above = energies.get(*image).map_or(f64::NAN, |e| e - first);
            log.info(&format!(
                "Candidate for a transition state: image {image}, {:.4} {} above the first image{}",
                units.factor * above,
                units.unit,
                found
                    .files
                    .get(k)
                    .map_or_else(String::new, |path| format!(" ({})", path.display()))
            ));
        }
        log.info(&format!("Last chain: {}", found.path.display()));
    }
    if interrupted {
        log.info(&format!(
            "Continue with: nebmain --restart {}",
            dir.display()
        ));
    }
    log.flush();
    Ok(NebmainResult {
        directory: dir.to_path_buf(),
        iterations: outcome.iterations,
        end: outcome.end,
        interrupted,
        energies,
        candidates,
        optimized: Vec::new(),
    })
}

/// What the model of `-neb_opt gp=on` did, for the log.
fn report_model(model: &SurrogateSurface<'_>, log: &Logger) {
    let counts = model.counts();
    log.info(&format!(
        "Gaussian process (-neb_opt gp=on): {} image values calculated, {} answered from \
         the model in {} iteration(s); its last measured error of a gradient component{}; \
         {} chain(s) that looked relaxed with its values were calculated to check.",
        counts.calculated,
        counts.predicted,
        counts.model_iterations,
        counts
            .error
            .map_or_else(|| ": none".to_string(), |e| format!(" {e:.2e}")),
        counts.checks
    ));
}

/// The internal coordinates to try for a chain, in order: those of `-coord`, then the
/// others that a chain can be relaxed in (empty for Cartesian coordinates).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn internal_coordinates(
    config: &NebmainConfig,
    settings: &ChainSettings,
    constrained: bool,
) -> MoptResult<Vec<String>> {
    let name = config.coordinates.trim().to_ascii_lowercase();
    if name.is_empty() || name == "cart" || name == "cartesian" {
        return Ok(Vec::new());
    }
    let family = name.split(':').next().unwrap_or("");
    let others: &[&str] = match family {
        "dic" | "delocalized" => &["tric-cart"],
        "tric" => &["tric-cart", "dic"],
        "tric-cart" => &["dic"],
        // Spherical coordinates: the lengths of the angles change with the partial norms,
        // so distances and tangents along a chain in them are far from those of the
        // structures (HCN about a structure off the path: forces of 1 Hartree/Bohr after
        // 150 iterations; docs/source/coordinate_systems.md).
        "spherical" | "hyperspherical" | "polar" => {
            return Err(MoptError::unsupported(format!(
                "-coord {}: spherical coordinates are for optimizations; a chain in them does \
                 not relax (the lengths of their angles change along it). Use cart, dic, \
                 tric, or tric-cart for a chain",
                config.coordinates
            )))
        }
        _ => {
            return Err(MoptError::config(format!(
                "-coord {}: a chain needs coordinates that are independent of each other \
                 and the same for all images: cart, dic, tric, or tric-cart[:WEIGHT]",
                config.coordinates
            )))
        }
    };
    if constrained || !settings.fixed_components.is_empty() || config.per_atom {
        return Err(MoptError::unsupported(
            "-coord with internal coordinates cannot be used together with -fix, -pc, \
             -gfix, -uc, or projection=atom yet (they are defined on the Cartesian \
             coordinates of the atoms)",
        ));
    }
    if settings.added.is_some() {
        return Err(MoptError::unsupported(
            "-coord with internal coordinates and -aneb cannot be used together yet",
        ));
    }
    let mut names = vec![name];
    names.extend(others.iter().map(|other| (*other).to_string()));
    Ok(names)
}

fn run_structures(
    config: &NebmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    extra: &mut [&mut (dyn ChainObserver + Send)],
    saved: &mut SavedBand<'_>,
) -> MoptResult<NebmainResult> {
    let spin = config
        .state
        .of_file(&first_structure_file(&config.inputs)?)?;
    let mut structures = read_structures(&config.inputs, spin)?;
    // The bias potentials at the first structure (which the cell does not move). A chain is
    // never told which evaluation is accepted: terms of the time or of accumulated state are
    // refused, the others evaluated by every image on its own copy.
    let biases = super::oniom_micro::build_terms(&config.biases, &structures[0])?;
    check_chain_terms(&biases, "nebmain", &warm_start_label)?;
    // A shift along the cell is free only where no term sees the position in the cell and
    // no constraint holds a position (frozen atoms, Cartesian components).
    let free_shifts = frame_invariant(true, biases.iter().map(|(_, t)| t.as_ref()))
        && !crate::pes::constraint::breaks_rigid_body(&config.constraints);
    let periodic_shifts = join_in_the_cell_centred(&mut structures, free_shifts)?;
    let whole = structures[0].clone();
    let evaluations_before = registry.counts();
    // ONIOM microiterations: the chain is one of the core on the relaxed surface, every image
    // with its own environment (each relaxation starts from that of the nearest core
    // structure evaluated, the image's own one of the iteration before; the first one from
    // the structure the image was made from). Constraints and bias potentials on the core are
    // taken to it.
    let wrapped = super::oniom_micro::wrap_constrained(
        &config.calculator_options,
        None,
        &config.constraints,
        &biases,
        &[],
        // The chain is at rest at its tolerance of the forces; the environment of every
        // image is relaxed to the same.
        match config.chain.force_tolerance {
            Some(t) => (t, f64::INFINITY),
            None => super::oniom_micro::NORMAL_FORCES,
        },
        whole.clone(),
        &config.calculator,
        registry,
    )?;
    let (template, calc, micro, micro_note, constraints, terms) = (
        wrapped.start,
        wrapped.calc,
        wrapped.micro,
        wrapped.note,
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
    let calc = crate::pes::o1numhess::wrap_if_requested(
        calc,
        config.model_hessian.as_deref(),
        config.exact_hessians,
    );
    if calc.uses_electrons() {
        template.spin_charge.check_electrons(&template.elements)?;
    }
    let (dir, log) = open_run(config, base, console)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&config.calculator_options.software_path_report());
    if let Some(note) = &micro_note {
        log.info(note);
    }
    super::oniom_micro::report_relaxations(micro.as_deref(), &log);

    // The first chain (-p, -ad, -nd, -ndb), the fixed atoms, the alignment, and the
    // constraints of the images, as the Python relax_band sets them up too.
    let rigid = frame_invariant(
        calc.is_rigid_body_invariant(),
        terms.iter().map(|(_, t)| t.as_ref()),
    );
    let StructureChain {
        mut settings,
        chain,
        constraints: shared,
        notes,
    } = structure_chain(
        config,
        &structures,
        &template,
        &constraints,
        rigid,
        periodic_shifts,
        Some(dir.clone()),
    )?;
    for note in &notes {
        log.info(note);
    }
    let others = constraints
        .iter()
        .filter(|c| !matches!(c.kind, ConstraintKind::Frozen(_)))
        .count();

    let mut surface = CalculatorSurface::new(&journaled, template.clone())
        .with_terms(terms)
        .with_excited_state(config.excited_state);
    if let Some(shared) = &shared {
        surface = surface.with_constraints(Arc::clone(shared));
    }
    if let Some(name) = config.model_function.first() {
        // Two electronic states combined as in optmain -mf, for every image. A wrong
        // name is an error here, before anything is calculated.
        let (spec, root) = (config.model_function.clone(), config.excited_state);
        super::optmain::state_surface(&spec, root)?;
        log.info(&format!(
            "Mode: Model Function ({name}); every image on the combined surface."
        ));
        if root > 0 {
            log.info(&format!("Electronic state: {root}"));
        }
        super::optmain::report_excitations(calc.as_ref(), &template, &dir, &log)?;
        surface = surface.with_states(std::sync::Arc::new(move || {
            super::optmain::state_surface(&spec, root)?
                .ok_or_else(|| MoptError::config("-mf: no surface of states"))
        }));
    }
    if let (false, Some(name)) = (config.exact_hessians, &config.model_hessian) {
        if !crate::pes::o1numhess::is_o1numhess(name) && settings.rfo.is_some() {
            surface = surface.with_model_hessian(crate::model_hessian::resolve(name)?);
        }
    }
    if let Some(dmf) = &config.dmf {
        // The direct MaxFlux method on the same surface, from the same first chain.
        super::nebmain_dmf::refuse_chain_options(config)?;
        if others > 0 {
            return Err(MoptError::unsupported(
                "-dmf: the constraints -pc, -gfix, and -uc are held image by image, and the \
                 direct MaxFlux method moves a fitted path, not images (the points of its \
                 path cannot be put back onto constraints); it keeps -fix only",
            ));
        }
        let mut dmf = dmf.clone();
        dmf.fixed.clone_from(&settings.fixed_components);
        let result = super::nebmain_dmf::relax_observed(
            config,
            dmf,
            &chain,
            &surface,
            ChainPoints::Structures(Box::new(template.clone())),
            None,
            &stem_of(&config.inputs[0]),
            &dir,
            &log,
            saved,
        )?;
        if let Some(relaxed) = &micro {
            super::oniom_micro::finish(relaxed, None, &dir, &log)?;
        }
        return close_run(
            config,
            result,
            &whole,
            console,
            registry,
            &log,
            &evaluations_before,
            &journal,
        );
    }
    // The first chain on the constraints (explicit values are held from the start).
    let mut keep = shared.map(KeepConstraints::new);
    let chain = match keep.as_mut() {
        Some(keep) => keep.restore_first(&chain)?,
        None => chain,
    };
    // -coord: the chain as vectors of internal coordinates. Coordinates that cannot be
    // set up for this chain (the orientation of a linear fragment in tric, a structure
    // that has no such coordinates) give way to the next ones, and at last to Cartesian
    // coordinates: the run goes on in what works and says so.
    let candidates = internal_coordinates(config, &settings, keep.is_some())?;
    let mut internal = None;
    let mut chain = chain;
    if !candidates.is_empty() {
        // Set up at the image in the middle of the first chain: coordinates of an end
        // structure are often poor on the way (a linear end, bonds that are not there
        // yet).
        let middle = template.with_flat_coords_bohr(&chain[chain.len() / 2])?;
        for name in &candidates {
            let coordinates = CoordinateRegistry::default().create(name)?;
            match InternalSurface::new(&surface, coordinates, middle.clone(), &chain) {
                Ok((made, points)) => {
                    log.info(&format!(
                        "Coordinates (-coord {name}): the chain is relaxed in {} internal \
                         coordinates; distances along it and forces are in these \
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
                Err(error) => log.warn(&format!(
                    "Coordinates (-coord {name}) cannot be set up for this chain: {error}"
                )),
            }
        }
        if internal.is_none() {
            log.warn(
                "None of the internal coordinates can be set up for this chain: it is \
                 relaxed in Cartesian coordinates.",
            );
        }
    }
    let behind: &dyn ChainSurface = match &internal {
        Some(internal) => internal,
        None => &surface,
    };
    // -neb_opt gp=on: a model answers for the images near calculated points.
    let plain = internal.is_none() && template.active_cell().is_none();
    let modelled =
        chain_surrogate(config, true, plain)?.map(|model| SurrogateSurface::new(behind, model));
    let result = relax(
        config,
        settings,
        chain,
        modelled.as_ref().map_or(behind, |m| m as &dyn ChainSurface),
        ChainPoints::Structures(Box::new(template.clone())),
        None,
        keep,
        internal.as_ref(),
        None,
        &stem_of(&config.inputs[0]),
        &dir,
        &log,
        extra,
        saved,
    )?;
    if let Some(model) = &modelled {
        report_model(model, &log);
    }
    if let Some(relaxed) = &micro {
        // The files of the chain with the whole system, before the optimizations after it
        // read them.
        super::oniom_micro::finish(relaxed, None, &dir, &log)?;
    }
    close_run(
        config,
        result,
        &whole,
        console,
        registry,
        &log,
        &evaluations_before,
        &journal,
    )
}

/// `nebmain -vc`: the chain of the atoms and the cell ([`super::nebmain_cell`]), with the
/// files of a chain of structures (every image with its lattice) and the cells of the images
/// after every iteration.
fn run_cell(
    config: &NebmainConfig,
    cell: &super::variable_cell::VariableCellSettings,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    extra: &mut [&mut (dyn ChainObserver + Send)],
    saved: &mut SavedBand<'_>,
) -> MoptResult<NebmainResult> {
    super::nebmain_cell::check_options(config, cell)?;
    let spin = config
        .state
        .of_file(&first_structure_file(&config.inputs)?)?;
    let structures = read_structures(&config.inputs, spin)?;
    let biases = super::oniom_micro::build_terms(&config.biases, &structures[0])?;
    check_chain_terms(&biases, "nebmain", &warm_start_label)?;
    let whole = structures[0].clone();
    let evaluations_before = registry.counts();
    let wrapped = super::oniom_micro::wrap_constrained(
        &config.calculator_options,
        Some("a variable cell (-vc; ONIOM has no stress)"),
        &config.constraints,
        &biases,
        &[],
        super::oniom_micro::NORMAL_FORCES,
        whole.clone(),
        &config.calculator,
        registry,
    )?;
    let (calc, constraints, terms) = (wrapped.calc, wrapped.constraints, wrapped.terms);
    if calc.uses_electrons() {
        whole.spin_charge.check_electrons(&whole.elements)?;
    }
    let (dir, log) = open_run(config, base, console)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&config.calculator_options.software_path_report());
    log.info("Reference: D. Sheppard, P. Xiao, W. Chemelewski, D. D. Johnson, G. Henkelman, J. Chem. Phys. 2012, 136, 074103 (the cell in the images of a chain)");
    log.info("Reference: M. Gubler, M. Krummenacher, H. Huber, S. Goedecker, J. Comput. Phys. X 2023, 17, 100131 (variables of the variable-cell optimization)");
    let super::nebmain_cell::CellChain {
        settings,
        chain,
        surface,
        keep,
        notes,
    } = super::nebmain_cell::cell_chain(
        config,
        cell,
        &structures,
        &journaled,
        terms,
        &constraints,
        Some(dir.clone()),
    )?;
    for note in &notes {
        log.info(note);
    }
    if let Some(name) = config.model_function.first() {
        log.info(&format!(
            "Mode: Model Function ({name}); every image on the two states combined in the \
             variables of the cell (their enthalpies and gradients in the variables)."
        ));
    }
    let map = surface.map().clone();
    let modelled =
        chain_surrogate(config, false, false)?.map(|model| SurrogateSurface::new(&surface, model));
    let mut tables = super::nebmain_cell::CellTables::new(&map, &dir, log.clone())?;
    let mut keep = keep;
    let mut observers: Vec<&mut (dyn ChainObserver + Send)> = vec![&mut tables];
    if let Some(keep) = keep.as_mut() {
        observers.push(keep);
    }
    for observer in extra.iter_mut() {
        observers.push(&mut **observer);
    }
    let result = relax(
        config,
        settings,
        chain,
        modelled
            .as_ref()
            .map_or(&surface as &dyn ChainSurface, |m| m as &dyn ChainSurface),
        ChainPoints::Variables(map.names()),
        None,
        None,
        None,
        Some(&map),
        &stem_of(&config.inputs[0]),
        &dir,
        &log,
        &mut observers,
        saved,
    )?;
    if let Some(model) = &modelled {
        report_model(model, &log);
    }
    close_run(
        config,
        result,
        &whole,
        console,
        registry,
        &log,
        &evaluations_before,
        &journal,
    )
}

/// The end of a run of structures: the optimizations after it, the counts of the
/// evaluations, and what the restart took from the journal.
#[allow(clippy::too_many_arguments)]
fn close_run(
    config: &NebmainConfig,
    mut result: NebmainResult,
    template: &MolecularSystem,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    log: &Logger,
    evaluations_before: &EvaluationCounts,
    journal: &RunJournal,
) -> MoptResult<NebmainResult> {
    if let (Some(follow), false) = (&config.follow_up, result.interrupted) {
        // The candidates for stationary points go on to optmain.
        result.optimized =
            optimize_images(follow, config, &result, template, console, registry, log)?;
    }
    log.info(&registry.counts().since(evaluations_before).line("this run"));
    log.flush();
    let (replayed, unused) = journal.replay_counts();
    if replayed > 0 {
        log.info(&format!(
            "Restart: {replayed} evaluations taken from {}{}.",
            JOURNAL_FILE,
            if unused > 0 {
                format!(" ({unused} recorded ones no longer matched and were recomputed)")
            } else {
                String::new()
            }
        ));
        log.flush();
    }
    Ok(result)
}

fn run_objective(
    config: &NebmainConfig,
    base: &Path,
    console: Option<Level>,
    objectives: &ObjectiveRegistry,
    extra: &mut [&mut (dyn ChainObserver + Send)],
    saved: &mut SavedBand<'_>,
) -> MoptResult<NebmainResult> {
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
    if config.follow_up.is_some() {
        return Err(MoptError::unsupported(
            "-neb_opt optimize: the optimizations after the chain are for chains of \
             structures; for plain variables run optmain on the points of the last chain",
        ));
    }
    if config.inputs.len() < 2 {
        return Err(MoptError::config(
            "a chain of plain variables needs two or more variables files (the points of the \
             first chain in order)",
        ));
    }
    let evaluations_before = objectives.counts();
    let objective = objectives.create(&config.calculator, &config.calculator_options)?;
    let mut points = Vec::with_capacity(config.inputs.len());
    let mut names: Option<Vec<String>> = None;
    for input in &config.inputs {
        let vars = read_variables_file(input)?;
        check_dimension(objective.as_ref(), vars.values.len())?;
        if points
            .first()
            .is_some_and(|first: &Col<f64>| first.nrows() != vars.values.len())
        {
            return Err(MoptError::config(format!(
                "{}: another number of variables than the first file",
                input.display()
            )));
        }
        if names.is_none() {
            names = vars.names.clone();
        }
        points.push(Col::from_fn(vars.values.len(), |k| vars.values[k]));
    }
    let n = points[0].nrows();
    let names = names.unwrap_or_else(|| objective.variable_names(n));
    let (dir, log) = open_run(config, base, console)?;
    log.info(&format!(
        "Mode: chain of points of {n} plain variables ({})",
        names.join(", ")
    ));
    if config.restart_dir.is_some() {
        log.info("Restart: the run is repeated from the first chain (objectives are cheap).");
    }
    // The first chain: -p, -ad, -nd (in the unit of the variables; none unless given).
    let (settings, chain) = variable_chain(config, &points, Some(dir.clone()))?;
    // Two variables: the chains are drawn on a map of the objective around the first one,
    // with a margin of 0.4 of its larger extent on both axes (a path can bend far from a
    // first chain on a straight line). The map is a drawing: its evaluations are not
    // counted.
    let map = if n == 2 && config.figures {
        let extent = |k: usize| {
            chain
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |m, p| {
                    (m.0.min(p[k]), m.1.max(p[k]))
                })
        };
        let (along_x, along_y) = (extent(0), extent(1));
        let margin = 0.4 * (along_x.1 - along_x.0).max(along_y.1 - along_y.0).max(1e-6);
        let bounds = |k: usize| {
            let (lo, hi) = if k == 0 { along_x } else { along_y };
            (lo - margin, hi + margin)
        };
        let drawn = objectives.create_uncounted(&config.calculator, &config.calculator_options)?;
        let values: Vec<f64> = chain
            .iter()
            .filter_map(|p| drawn.evaluate(p, false).ok().map(|found| found.value))
            .collect();
        let (lo, hi) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |m, v| {
                (m.0.min(*v), m.1.max(*v))
            });
        Some(SurfaceMap {
            x: bounds(0),
            y: bounds(1),
            ceiling: (hi > lo).then_some(hi + 0.5 * (hi - lo)),
            points: MAP_POINTS,
            value: Box::new(move |x, y| {
                drawn
                    .evaluate(&Col::from_fn(2, |c| [x, y][c]), false)
                    .map_or(f64::NAN, |found| found.value)
            }),
        })
    } else {
        None
    };
    let surface = ObjectiveSurface(objective.as_ref());
    if let Some(dmf) = &config.dmf {
        super::nebmain_dmf::refuse_chain_options(config)?;
        let result = super::nebmain_dmf::relax_observed(
            config,
            dmf.clone(),
            &chain,
            &surface,
            ChainPoints::Variables(names),
            map,
            &stem_of(&config.inputs[0]),
            &dir,
            &log,
            saved,
        )?;
        log.info(
            &objectives
                .counts()
                .since(&evaluations_before)
                .line("this run"),
        );
        log.flush();
        return Ok(result);
    }
    let modelled =
        chain_surrogate(config, false, false)?.map(|model| SurrogateSurface::new(&surface, model));
    let result = relax(
        config,
        settings,
        chain,
        modelled
            .as_ref()
            .map_or(&surface as &dyn ChainSurface, |m| m as &dyn ChainSurface),
        ChainPoints::Variables(names),
        map,
        None,
        None,
        None,
        &stem_of(&config.inputs[0]),
        &dir,
        &log,
        extra,
        saved,
    )?;
    if let Some(model) = &modelled {
        report_model(model, &log);
    }
    log.info(
        &objectives
            .counts()
            .since(&evaluations_before)
            .line("this run"),
    );
    log.flush();
    Ok(result)
}

/// The chain of structures of a `nebmain` run as it is set up before the relaxation: the
/// first chain from `structures` (`-p`, `-ad`, `-nd`, which is 0.5 Angstrom unless given,
/// `-neb_opt path=`, `-ndb`), the settings of the relaxation for structures like `template`
/// (fixed atoms of `-fix`, the alignment, which is off when `rigid_body_invariant` is not, the
/// shifts along the periodic directions `periodic_shifts`, the projection atom by atom,
/// `hessian_directory` for the Hessians of the images), and the constraints of the images
/// (each image holds its values in the first chain; frozen atoms where `template` has them),
/// with what is said about them for the log. Shared by `nebmain` and the Python `relax_band`.
pub struct StructureChain {
    /// The settings of the relaxation.
    pub settings: ChainSettings,
    /// The first chain (not yet on the constraints: [`KeepConstraints::restore_first`]).
    pub chain: Vec<Col<f64>>,
    /// The constraints of the images (`None` without constraints).
    pub constraints: Option<SharedChainConstraints>,
    /// What is said about the chain, in order, for the log.
    pub notes: Vec<String>,
}

/// See [`StructureChain`].
///
/// MultiOptPy: `neb.py:NEB.run` (the first chain and the settings).
pub fn structure_chain(
    config: &NebmainConfig,
    structures: &[MolecularSystem],
    template: &MolecularSystem,
    constraints: &[Constraint],
    rigid_body_invariant: bool,
    periodic_shifts: Option<Vec<[f64; 3]>>,
    hessian_directory: Option<PathBuf>,
) -> MoptResult<StructureChain> {
    let points: Vec<Col<f64>> = structures.iter().map(|s| s.coords_flat_bohr()).collect();
    let spacing = match config.spacing {
        // With -ndb the images are placed by it; the distance of -nd is used when given.
        None if config.bezier_spacing.is_some() => None,
        None => Some(DEFAULT_SPACING_ANGSTROM),
        Some(given) => (given > 0.0).then_some(given),
    };
    let how = FirstChain {
        // The direct MaxFlux method starts from the same first chain as a chain: between
        // two structures as many images as the spacing asks for (one with -nd 0, so that
        // there is a chain to fit the path to). One image alone between HCN and HNC gave
        // a first path 40 kcal/mol above that of the chain, and the method stayed over
        // another pass (docs/source/migration_paths.md).
        inserted: if config.dmf.is_some() && points.len() == 2 && spacing.is_none() {
            config.inserted.max(1)
        } else {
            config.inserted
        },
        equal_spacing: config.first_equal_spacing,
        spacing: spacing.map(|angstrom| angstrom * Units::ANGSTROM_TO_BOHR),
        path: config.first_path.unwrap_or(DEFAULT_FIRST_PATH),
    };
    // Atoms of -fix stay where they are in the images that are made, too.
    let mut fixed_components = Vec::new();
    for constraint in constraints {
        if let ConstraintKind::Frozen(atoms) = &constraint.kind {
            for atom in atoms
                .iter()
                .filter(|atom| atom.index() < template.n_atoms())
            {
                fixed_components.extend((0..3).map(|k| 3 * atom.index() + k));
            }
        }
    }
    let frame = PathFrame {
        elements: template.elements.clone(),
        cell: template.active_cell().cloned(),
        fixed_components,
    };
    let (mut chain, mut notes) = first_chain_in(
        &points,
        ChainKind::Atoms,
        &how,
        &frame,
        &PathSettings::default(),
    )?;
    if let Some(distance) = config.bezier_spacing {
        let before = chain.len();
        chain = bezier_by_spacing(&chain, distance * Units::ANGSTROM_TO_BOHR, ChainKind::Atoms);
        notes.push(format!(
            "The first chain is put on the Bezier curve of its {before} images (-ndb \
             {distance}): {} images. The curve passes near the images between the ends, not \
             through them.",
            chain.len()
        ));
        if chain.len() < 3 {
            return Err(MoptError::config(format!(
                "-ndb {distance}: the first chain has {} images; a relaxation needs at least \
                 three (a smaller distance)",
                chain.len()
            )));
        }
    }
    notes.push(format!(
        "{} structures read; first chain of {} images{}",
        structures.len(),
        chain.len(),
        spacing.map_or_else(String::new, |d| format!(", {d} Angstrom apart (-nd)"))
    ));

    // -fix: the atoms stay where they are in the first chain (their forces are left out of
    // the forces of the chain); all constraints take their directions out of the gradients
    // and Hessians and are restored after every iteration.
    let mut settings = config.chain.clone();
    settings.kind = ChainKind::Atoms;
    if !rigid_body_invariant {
        // A core in a held ONIOM environment, or a bias potential at a fixed place or
        // direction, has a physical position and orientation.
        settings.align = false;
        notes.push(
            "Chain alignment disabled: the calculator or a bias potential uses a fixed \
             reference frame (the images are neither centred nor turned)."
                .to_string(),
        );
        if periodic_shifts.is_some() {
            notes.push(
                "Periodic cell: the structures are not shifted to one centroid (a bias \
                 potential sees the position in the cell)."
                    .to_string(),
            );
        }
    }
    if periodic_shifts.is_some() {
        // The cell does not turn with the structures.
        notes.push(
            "Periodic cell: all images in the cell of the first structure; from structure \
             to structure every atom follows its nearest periodic image; the images are \
             shifted to one centroid along the periodic directions and not turned."
                .to_string(),
        );
    }
    settings.periodic_shifts = periodic_shifts;
    if config.per_atom {
        settings.block = Some(crate::path::band::BLOCK_ATOM);
    }
    if let Some(rfo) = settings.rfo.as_mut() {
        rfo.directory = hessian_directory;
    }
    for constraint in constraints {
        if let ConstraintKind::Frozen(atoms) = &constraint.kind {
            for atom in atoms {
                if atom.index() >= template.n_atoms() {
                    return Err(MoptError::config(format!(
                        "-fix: atom {} of {} atoms",
                        atom.one_based(),
                        template.n_atoms()
                    )));
                }
                settings
                    .fixed_components
                    .extend((0..3).map(|k| 3 * atom.index() + k));
            }
        }
    }
    settings.fixed_components.sort_unstable();
    settings.fixed_components.dedup();
    // Every image holds its own values: explicit ones, or its own in the first chain.
    let shared = if constraints.is_empty() {
        None
    } else {
        // Frozen atoms stay exactly where the first structure has them, in every image.
        let constraints = constraints
            .iter()
            .map(|c| match c.kind {
                ConstraintKind::Frozen(_) => c.resolve(template, None),
                _ => Ok(c.clone()),
            })
            .collect::<MoptResult<Vec<_>>>()?;
        let held = ChainConstraints::new(&constraints, template, &chain)?;
        if held.breaks_rigid_body() {
            // Positions and orientations are held: the structures are not turned onto
            // each other.
            settings.align = false;
        }
        notes.push(format!(
            "Constraints: {} per image, each image holding its values in the first chain \
             (explicit values in all images; an image put elsewhere along the chain holds \
             the values interpolated at its place).",
            held.per_image()
        ));
        Some(Arc::new(Mutex::new(held)))
    };
    if !settings.fixed_components.is_empty() {
        notes.push(format!(
            "Fixed atoms (-fix): {} coordinates of every image; their forces are not \
             forces of the chain.",
            settings.fixed_components.len()
        ));
    }
    Ok(StructureChain {
        settings,
        chain,
        constraints: shared,
        notes,
    })
}

/// The settings and the first chain of a `nebmain` run on plain variables from the points
/// `points` (`-p`, `-ad`, `-nd` in the unit of the variables and none unless given, `-ndb`;
/// the images on the straight line), `hessian_directory` for the Hessians of the images.
/// Shared by `nebmain` and the Python `relax_band`.
///
/// MultiOptPy: `neb.py:NEB.run` (the first chain and the settings).
pub fn variable_chain(
    config: &NebmainConfig,
    points: &[Col<f64>],
    hessian_directory: Option<PathBuf>,
) -> MoptResult<(ChainSettings, Vec<Col<f64>>)> {
    let coordinates = config.coordinates.trim().to_ascii_lowercase();
    if !["", "cart", "cartesian"].contains(&coordinates.as_str()) {
        return Err(MoptError::config(format!(
            "-coord {}: plain variables are their own coordinates",
            config.coordinates
        )));
    }
    if let Some(asked) = config.first_path.filter(|path| *path != FirstPath::Linear) {
        return Err(MoptError::config(format!(
            "-neb_opt path={}: the images between plain variables are on the straight line",
            asked.name()
        )));
    }
    let how = FirstChain {
        // The direct MaxFlux method needs the two ends only; a chain of them gets one
        // image between.
        inserted: if config.dmf.is_some() && points.len() == 2 {
            config.inserted.max(1)
        } else {
            config.inserted
        },
        equal_spacing: config.first_equal_spacing,
        spacing: config.spacing.filter(|d| *d > 0.0),
        path: FirstPath::Linear,
    };
    let mut chain = first_chain(points, ChainKind::Variables, &how)?;
    if let Some(distance) = config.bezier_spacing {
        chain = bezier_by_spacing(&chain, distance, ChainKind::Variables);
        if chain.len() < 3 {
            return Err(MoptError::config(format!(
                "-ndb {distance}: the first chain has {} points; a relaxation needs at least three",
                chain.len()
            )));
        }
    }
    let mut settings = config.chain.clone();
    settings.kind = ChainKind::Variables;
    if let Some(rfo) = settings.rfo.as_mut() {
        rfo.directory = hessian_directory;
    }
    Ok((settings, chain))
}

/// The model of `-neb_opt gp=on` for a chain (`None` without it): its length scale and what
/// it sees of a point, for structures (`structures`; `plain` when they are in Cartesian
/// coordinates without a cell, which inverse distances need) or for plain variables.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn chain_surrogate(
    config: &NebmainConfig,
    structures: bool,
    plain: bool,
) -> MoptResult<Option<crate::path::band::surrogate::SurrogateSettings>> {
    if !structures && config.surrogate_features == Some(Features::InverseDistances) {
        return Err(MoptError::config(
            "-neb_opt gp_features=distances: plain variables have no atoms",
        ));
    }
    let Some(mut model) = config.surrogate else {
        return Ok(None);
    };
    if !structures {
        model.length_scale = config.surrogate_length.unwrap_or(DEFAULT_GP_LENGTH);
        return Ok(Some(model));
    }
    // gp_features=distances: structures in Cartesian coordinates without a cell seen by
    // their inverse distances, which no shift or turn changes.
    model.features = config.surrogate_features.unwrap_or(Features::Points);
    if model.features == Features::InverseDistances && !plain {
        return Err(MoptError::unsupported(
            "-neb_opt gp_features=distances is for structures in Cartesian coordinates \
             without a periodic cell",
        ));
    }
    model.length_scale = match model.features {
        Features::InverseDistances => config
            .surrogate_length
            .unwrap_or(DEFAULT_GP_LENGTH_DISTANCES),
        Features::Points => {
            config.surrogate_length.unwrap_or(DEFAULT_GP_LENGTH) * Units::ANGSTROM_TO_BOHR
        }
    };
    Ok(Some(model))
}

/// Thin wrapper used by tests and embedding code: settings for `args`.
pub fn config_from_cli(args: &[&str]) -> MoptResult<NebmainConfig> {
    let p = super::option::parser("nebmain", NEBMAIN).parse(args)?;
    NebmainConfig::from_args(&p)
}

/// The settings of the options `args` of `nebmain` without its inputs (a caller that has the
/// structures or the points itself, the Python `relax_band`): the same parser and defaults.
///
/// MultiOptPy: `interface.py:call_nebparser`.
pub fn options_from_cli(args: &[&str]) -> MoptResult<NebmainConfig> {
    let p = super::option::parser("nebmain", NEBMAIN).parse(args)?;
    if let Some(input) = p.positionals.first() {
        return Err(MoptError::config(format!(
            "'{input}': the options of nebmain only; the images are given as arrays"
        )));
    }
    NebmainConfig::from_options(&p)
}
