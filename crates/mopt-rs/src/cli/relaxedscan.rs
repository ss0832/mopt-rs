//! `relaxedscan`: scans of chosen coordinates on a grid, each point optimized under
//! constraints (or a single point with `-rigid`); the energy curves and maps and the
//! transition-state candidates on them.
//!
//! The command is a thin driver over `optmain`: every point is a constrained optimization
//! with every optmain option ([`super::relaxedscan_points`]), the coordinates of the scan
//! being added to the constraints of the run. The points are those of
//! [`crate::workflow::scan::grid`] (MultiOptPy's joint scan, or a map with `-grid`), each
//! starting from its nearest optimized neighbour; the coordinates are geometric ones
//! (`bond`, `angle`, `dihedral`, ...), expressions, plain variables of an objective, or the
//! ReSPer coordinates of a set of reference structures ([`super::relaxedscan_resper`]). The
//! run directory `<stem>_RScan_<date>/` holds the scan's `mopt.log` (one line per point),
//! `settings.json`, `scan_state.json` (a restart continues with the points not done),
//! MultiOptPy's energy profile and structure files, `scan_points.csv`, the trajectories, the
//! figures, the stationary points ([`super::relaxedscan_outputs`]), the optimizations of the
//! transition-state candidates ([`super::relaxedscan_followup`]), and a page of the scan
//! ([`super::relaxedscan_report`]).
//!
//! MultiOptPy: `Entrypoints/relaxed_scan.py:run_relaxedscan`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::relaxedscan_points::{read_structure, Structure};
use super::relaxedscan_resper::{load_points, load_structures};
pub(crate) use super::relaxedscan_scan::{distance, structure_text, Scan, Surface};
use super::relaxedscan_scan::{run_point, surface_of, variable_indices};
use super::relaxedscan_settings::ScanTarget;
pub use super::relaxedscan_settings::{config_from_cli, RelaxedscanConfig};
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::pes::objective::ObjectiveRegistry;
use crate::pes::registry::CalculatorRegistry;
use crate::pes::term::{History, PesTerm};
use crate::workflow::observers::STOP_FILE;
use crate::workflow::scan::continuity::jumps;
use crate::workflow::scan::output::{write_points, write_profile, POINTS_FILE};
use crate::workflow::scan::{GridKind, ScanAxis, ScanGrid, ScanPoint, ScanState};

/// What a scan ended with.
#[derive(Debug, Clone)]
pub struct RelaxedscanResult {
    /// The run directory.
    pub directory: PathBuf,
    /// Whether it was interrupted (`--restart` continues it).
    pub interrupted: bool,
    /// The points, in the order they were done.
    pub points: Vec<ScanPoint>,
    /// The stationary points of the scanned surface.
    pub candidates: Vec<crate::workflow::scan::Candidate>,
    /// The optimizations of the transition-state candidates (`ts_optimize=on`).
    pub optimized: Vec<super::relaxedscan_followup::OptimizedCandidate>,
}

/// MultiOptPy's run directory `<stem>_RScan_<YYYY_MM_DD_HH_MM_SS>/` under `base`.
pub fn scan_directory(base: &Path, input: &Path, now: &DateTime) -> PathBuf {
    base.join(format!("{}_RScan_{}", stem_of(input), date_of(now)))
}

fn date_of(now: &DateTime) -> String {
    format!(
        "{}_{:02}_{:02}_{:02}",
        now.date_dir(),
        now.hour,
        now.minute,
        now.second
    )
}

fn stem_of(input: &Path) -> String {
    input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned())
}

/// Run `relaxedscan`, writing the run directory under `base`; progress goes to the console
/// up to `console` (nothing when `None`) and to `mopt.log` of the scan.
pub fn run(
    config: &RelaxedscanConfig,
    base: &Path,
    console: Option<Level>,
) -> MoptResult<RelaxedscanResult> {
    run_with(config, base, console, &CalculatorRegistry::native())
}

/// [`run`] with the calculators of `registry`.
pub fn run_with(
    config: &RelaxedscanConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<RelaxedscanResult> {
    let objectives = ObjectiveRegistry::builtin();
    let evaluations_before = (registry.counts(), objectives.counts());
    let surface = surface_of(config, &objectives)?;
    let plain = matches!(surface, Surface::Plain { .. });
    // The user's constraints resolved once, at the input: every point (whatever structure it
    // starts from: a neighbour, a ReSPer pre-image) and every follow-up holds the values of
    // the input, not those of its own start.
    let resolved;
    let config = match &surface {
        Surface::Atoms { start, .. } => {
            let mut c = config.clone();
            c.optmain.settings.constraints =
                super::constraints::resolve_for_run(&config.optmain.settings.constraints, start)
                    .context("constraints at the input")?;
            if config.optmain.variable_cell.is_some() {
                // In a variable cell a frozen atom keeps its fractional coordinates, which a
                // Cartesian anchor cannot carry from one cell to the next: each point holds
                // those of its start, and since every point starts from the input or from a
                // point that held them, they are those of the input.
                for k in &mut c.optmain.settings.constraints {
                    if matches!(k.kind, crate::pes::constraint::ConstraintKind::Frozen(_)) {
                        k.anchor = None;
                    }
                }
            }
            let terms = super::oniom_micro::build_terms(&config.optmain.biases, start)?;
            refuse_restarting_terms(&terms)?;
            resolved = c;
            &resolved
        }
        Surface::Plain { .. } => config,
    };
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => scan_directory(base, &config.input, &DateTime::now_utc()),
    };
    std::fs::create_dir_all(&dir)?;
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    let stem = stem_of(&config.input);
    log.info(&format!(
        "mopt-rs {} relaxedscan: {}",
        crate::VERSION,
        config.input.display()
    ));
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document),
    )?;
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    let variables = variable_indices(&config.scan.targets, &surface)?;
    let axes: Vec<ScanAxis> = config
        .scan
        .targets
        .iter()
        .zip(&config.scan.samples)
        .map(|(t, &n)| ScanAxis::linspace(t.label(), t.unit(plain), t.start, t.end, n))
        .collect::<MoptResult<_>>()?;
    let grid = if config.scan.grid {
        ScanGrid::product(axes, config.scan.order)?
    } else {
        ScanGrid::joint(axes)?
    };
    let resper = match (&config.scan.resper, &surface) {
        (None, _) => None,
        (Some(s), Surface::Atoms { start, .. }) => {
            // With ONIOM microiterations the map is made of the atoms of the core unless
            // atoms are given (resper_atoms): the environment is relaxed at every point and
            // does not define the coordinate.
            let core = super::oniom_micro::core_atoms(&config.optmain.calculator_options, start)?;
            match core.filter(|_| s.atoms.is_empty()) {
                Some(atoms) => {
                    log.info(&format!(
                        "ReSPer map of the {} atoms of the ONIOM core (the environment is \
                         relaxed at every point; resper_atoms= chooses other atoms).",
                        atoms.len()
                    ));
                    let of_core =
                        super::relaxedscan_settings::ResperSettings { atoms, ..s.clone() };
                    Some(load_structures(&of_core, start, &log)?)
                }
                None => Some(load_structures(s, start, &log)?),
            }
        }
        (Some(s), Surface::Plain { names, .. }) => Some(load_points(s, names, &log)?),
    };
    log.info(&format!(
        "Scan: {} ({} point(s){}), {}.",
        grid.axes()
            .iter()
            .map(|a| format!(
                "{} from {} to {} {}",
                a.label,
                a.values[0],
                a.values[a.values.len() - 1],
                a.unit
            ))
            .collect::<Vec<_>>()
            .join("; "),
        grid.len(),
        match grid.kind() {
            GridKind::Joint if grid.axes().len() > 1 => ", the coordinates together",
            GridKind::Joint => "",
            GridKind::Product => ", every combination",
        },
        if config.scan.relax {
            "every point optimized under the constraints"
        } else {
            "single points without relaxation"
        }
    ));
    let mut state = ScanState::load(&dir)?.unwrap_or_default();
    if state.profile.is_empty() {
        state.profile = format!(
            "{stem}_scan_energy_profile_{}.csv",
            date_of(&DateTime::now_utc())
        );
    }
    if !state.done.is_empty() {
        log.info(&format!(
            "Restart: {} of {} point(s) done; the scan goes on with the others.",
            state.done.len(),
            grid.len()
        ));
    }
    let scan = Scan {
        config,
        surface,
        grid,
        resper,
        variables,
        dir: dir.clone(),
        stem,
        log: log.clone(),
    };
    let mut structures: HashMap<usize, Structure> = HashMap::new();
    for p in state.done.iter().filter(|p| !p.skipped) {
        let s = read_structure(&dir.join(&p.file), &scan.surface.start())?;
        structures.insert(p.point, s);
    }
    let interrupted = run_points(&scan, registry, console, &mut state, &mut structures)?;
    let mut result = RelaxedscanResult {
        directory: dir.clone(),
        interrupted,
        points: state.done.clone(),
        candidates: Vec::new(),
        optimized: Vec::new(),
    };
    if !interrupted {
        let map = scan.grid.kind() == GridKind::Product && scan.grid.axes().len() >= 2;
        let copies = !scan.periodic_copies().is_empty();
        if config.scan.relax && config.scan.repair && (map || copies) {
            result.interrupted = repair(&scan, registry, console, &mut state, &mut structures)?;
        }
        result.points = state.done.clone();
    }
    if !result.interrupted {
        let written = super::relaxedscan_outputs::write_all(&scan, &state, &structures)?;
        result.candidates.clone_from(&written.candidates);
        if config.scan.ts_optimize {
            result.optimized = super::relaxedscan_followup::optimize_candidates(
                &scan,
                &result.candidates,
                &state,
                registry,
                &objectives,
            )?;
        }
        if config.scan.html {
            if let Err(e) = super::relaxedscan_report::write(&scan, &state, &written, &result) {
                log.warn(&format!("The page of the scan was not written: {e}"));
            }
        }
    } else {
        log.info(&format!(
            "The scan was interrupted; continue with: relaxedscan --restart {}",
            dir.display()
        ));
    }
    log.info(
        &registry
            .counts()
            .since(&evaluations_before.0)
            .line("this scan"),
    );
    if plain {
        log.info(
            &objectives
                .counts()
                .since(&evaluations_before.1)
                .line("the objective in this scan"),
        );
    }
    log.flush();
    Ok(result)
}

/// Refuses bias potentials whose value depends on more than the structure in a way a scan
/// cannot keep: every point is an optimization of its own, so a term with state
/// (metadynamics deposits, the warm start of an inner problem that can change its value) or a
/// term of the simulated time would start again at every point, and the energies of the points
/// would not lie on one surface. A ramp over the iterations restarts at every point as well:
/// it is allowed, and every point sees the ramp from its first iteration (the energy of a
/// point is that of the ramp at the iteration where its optimization ended).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
fn refuse_restarting_terms(terms: &[(String, Arc<dyn PesTerm>)]) -> MoptResult<()> {
    for (label, t) in terms {
        let why = match t.history() {
            History::Stateful => "keeps a state accumulated over its evaluations",
            History::TimeDependent => "changes with the simulated time",
            History::Static | History::IterationRamp => continue,
        };
        return Err(MoptError::unsupported(format!(
            "relaxedscan: the bias potential {label} ('{}') {why}; every point of a scan is an \
             optimization of its own, where it would start again, so the points would not lie \
             on one surface (use a bias that is a function of the structure)",
            t.name()
        )));
    }
    Ok(())
}

/// The state of the scan, `scan_points.csv`, and MultiOptPy's energy profile as they are now.
fn write_tables(scan: &Scan<'_>, state: &ScanState) -> MoptResult<()> {
    state.save(&scan.dir)?;
    write_points(&scan.dir.join(POINTS_FILE), &scan.grid, state)?;
    let kinds: Vec<String> = scan.targets().iter().map(ScanTarget::kind_name).collect();
    let atoms: Vec<String> = scan.targets().iter().map(ScanTarget::atoms_text).collect();
    write_profile(
        &scan.dir.join(&state.profile),
        &kinds,
        &atoms,
        &scan.grid,
        state,
    )
}

/// The points not done yet, in the order of the grid; true when the scan was interrupted.
fn run_points(
    scan: &Scan<'_>,
    registry: &CalculatorRegistry,
    console: Option<Level>,
    state: &mut ScanState,
    structures: &mut HashMap<usize, Structure>,
) -> MoptResult<bool> {
    let config = scan.config;
    let stop = scan.dir.join(STOP_FILE);
    let tables = |state: &ScanState| write_tables(scan, state);
    tables(state)?;
    let mut reference = state.done.iter().map(|p| p.energy).find(|e| e.is_finite());
    let total = scan.grid.len();
    for (rank, &point) in scan.grid.order().iter().enumerate() {
        if state.result(point).is_some() {
            continue;
        }
        if stop.exists() {
            let _ = std::fs::remove_file(&stop);
            scan.log.info(&format!(
                "{} found: the scan is interrupted.",
                stop.display()
            ));
            return Ok(true);
        }
        let outcome = run_point(scan, point, None, registry, console, state, structures)?;
        let Some((result, outcome)) = outcome else {
            // Skipped or failed: recorded as such.
            tables(state)?;
            continue;
        };
        if outcome.interrupted {
            // Not done: a restart continues this point from its journal.
            return Ok(true);
        }
        let e = result.energy;
        let zero = *reference.get_or_insert(e);
        scan.log.info(&format!(
            "point {:>4}/{total}  {}  E = {e:.10}  dE = {:9.4} kcal/mol  {}{}",
            rank + 1,
            scan.describe_point(point),
            (e - zero) * Units::HARTREE_TO_KCAL_MOL,
            if !config.scan.relax {
                "single point".to_string()
            } else if result.converged {
                format!("converged in {} iteration(s)", result.iterations)
            } else {
                format!("NOT converged after {} iteration(s)", result.iterations)
            },
            match (result.start_from, result.extrapolated) {
                (_, true) => "  (beyond the references)".to_string(),
                (Some(from), false) => format!("  (from point {})", from + 1),
                (None, false) => String::new(),
            }
        ));
        if let Some(err) = result.preimage_error {
            scan.log.debug(&format!(
                "point {}: the interpolated structure lands {err:.3e} from the grid point",
                point + 1
            ));
        }
        state.record(result);
        tables(state)?;
    }
    Ok(false)
}

/// Rounds of the repair of jumps.
const REPAIR_ROUNDS: usize = 2;
/// Smallest lowering (Hartree) for which a point optimized again is taken.
const REPAIR_GAIN: f64 = 1e-6;
/// Two copies of one point (-180 and 180 degrees) whose energies differ by more than this
/// (Hartree) lie on different branches; converged copies of the scans measured (running.md)
/// agree to 2e-6.
const COPY_GAP: f64 = 1e-5;

/// Optimize the higher point of every jump between branches (and of every point of a
/// periodic dimension reached twice with two energies) again from the lower one, and keep
/// the result where it is lower; true when interrupted.
fn repair(
    scan: &Scan<'_>,
    registry: &CalculatorRegistry,
    console: Option<Level>,
    state: &mut ScanState,
    structures: &mut HashMap<usize, Structure>,
) -> MoptResult<bool> {
    let map = scan.grid.kind() == GridKind::Product && scan.grid.axes().len() >= 2;
    for round in 0..REPAIR_ROUNDS {
        let energies = state.energies(&scan.grid, true);
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        if map {
            let found = jumps(
                &scan.grid,
                |a, b| distance(structures.get(&a)?, structures.get(&b)?),
                scan.config.scan.jump_factor,
            );
            if !found.is_empty() {
                scan.log.info(&format!(
                    "Map: {} pair(s) of neighbours on different branches (structures {:.3} and \
                     more apart, {:.1} times the median {:.4}); the higher points are optimized \
                     again from the lower ones.",
                    found.len(),
                    found
                        .iter()
                        .map(|j| j.distance)
                        .fold(f64::INFINITY, f64::min),
                    scan.config.scan.jump_factor,
                    found[0].median
                ));
            } else if round == 0 {
                scan.log.info("Map: no jumps between branches.");
            }
            pairs.extend(found.iter().map(|j| (j.a, j.b)));
        }
        // The first and the last value of a dihedral over a whole turn are one structure.
        let copies: Vec<(usize, usize)> = scan
            .periodic_copies()
            .into_iter()
            .filter(|&(a, b)| (energies[a] - energies[b]).abs() > COPY_GAP)
            .collect();
        if !copies.is_empty() {
            scan.log.info(&format!(
                "{} point(s) at -180 and 180 degrees (one structure) with energies that differ; \
                 the higher ones are optimized again from the lower ones.",
                copies.len()
            ));
        }
        pairs.extend(copies);
        if pairs.is_empty() {
            return Ok(false);
        }
        let mut changed = false;
        for (a, b) in pairs {
            let (high, low) = if energies[a] >= energies[b] {
                (a, b)
            } else {
                (b, a)
            };
            if !(energies[high].is_finite() && energies[low].is_finite()) {
                continue;
            }
            let old = state.result(high).cloned();
            let old_structure = structures.get(&high).cloned();
            let Some((result, outcome)) =
                run_point(scan, high, Some(low), registry, console, state, structures)?
            else {
                continue;
            };
            let better = !outcome.interrupted
                && old
                    .as_ref()
                    .is_none_or(|o| result.biased < o.biased - REPAIR_GAIN);
            if better {
                scan.log.info(&format!(
                    "point {} ({}): {:.4} kcal/mol lower from point {}.",
                    high + 1,
                    scan.describe_point(high),
                    old.as_ref().map_or(0.0, |o| (o.biased - result.biased)
                        * Units::HARTREE_TO_KCAL_MOL),
                    low + 1
                ));
                state.record(result);
                changed = true;
            } else if let (Some(o), Some(s)) = (old, old_structure) {
                // The earlier result stays: its structure back in place.
                let comment = format!("RelaxedScan_{} {}", high + 1, scan.describe_point(high));
                std::fs::write(
                    scan.dir.join(&o.file),
                    structure_text(&scan.surface, &s, &comment, o.biased),
                )?;
                structures.insert(high, s);
            }
            write_tables(scan, state)?;
            if outcome.interrupted {
                return Ok(true);
            }
        }
        if !changed {
            return Ok(false);
        }
    }
    Ok(false)
}
