//! `nebmain -dmf`: the path by the direct MaxFlux method instead of a chain of images.
//!
//! The first chain (read and made as for a chain relaxation) becomes a spline path, and
//! [`crate::path::dmf::DirectMaxFlux`] moves it on the surface of the chain (calculator,
//! bias potentials, `-mf`, `-es`; atoms of `-fix` stay). Every iteration writes the points
//! that were evaluated as a chain (`path_ITR_<k>_<name>/`, `energy_plot.csv`, and in
//! `path_length.csv` their lengths along the path), the
//! estimate of the transition state (`<name>_dmf_highest.xyz`, one frame per iteration),
//! the figures, and a table in the log; at the end the path (`<name>_dmf_path.xyz`) and
//! the estimate (`<name>_ts_guess.xyz`). `-neb_opt optimize=` goes on from the points as
//! from the images of a chain.
//!
//! MultiOptPy: `neb.py:NEB` with `-dmf` (`MEP/pathopt_dmf_force.py:CaluculationDMF`, a
//! force on the images of a chain, not ported).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;

use faer::Col;

use super::nebmain::NebmainResult;
use super::nebmain_settings::NebmainConfig;
use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::log::{python_float, Logger};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::opt::state_machine::Termination;
use crate::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};
use crate::path::band::machine::ChainEnd;
use crate::path::dmf::arc::ArcLength;
use crate::path::dmf::{DirectMaxFlux, DmfEnd, DmfReport, DmfSettings};
use crate::workflow::chain::{ChainSurface, Flow};
use crate::workflow::chain_figures::{ChainFigures, SurfaceMap};
use crate::workflow::chain_files::{ChainFiles, ChainPoints};
use crate::workflow::dmf::run_dmf;
use crate::workflow::observers::StopFile;

/// Points of the path in the figures and in `<name>_dmf_path.xyz`.
const DRAWN: usize = 41;

/// The settings of `-dmf` and of the `dmf_` keys of `-neb_opt` (`None` without `-dmf`).
pub fn settings_from(p: &ParsedArgs) -> MoptResult<Option<DmfSettings>> {
    let keys: Vec<(&str, &str)> = p
        .get_list("neb_options")
        .iter()
        .filter_map(|text| text.split_once('='))
        .map(|(key, value)| (key.trim(), value.trim()))
        .filter(|(key, _)| key.starts_with("dmf_"))
        .collect();
    if !p.get_bool("DMF", false) {
        if let Some((key, _)) = keys.first() {
            return Err(MoptError::parse(format!(
                "-neb_opt {key}: a setting of the direct MaxFlux method, which needs -dmf"
            )));
        }
        return Ok(None);
    }
    let mut settings = DmfSettings {
        max_iterations: p
            .get_usize("NSTEP", super::nebmain_settings::DEFAULT_DMF_ITERATIONS)?
            .max(1),
        ..DmfSettings::default()
    };
    let number = |key: &str, value: &str| -> MoptResult<f64> {
        value
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| MoptError::parse(format!("-neb_opt {key}: '{value}' is not a number")))
    };
    let whole = |key: &str, value: &str| -> MoptResult<usize> {
        value.parse::<usize>().map_err(|_| {
            MoptError::parse(format!("-neb_opt {key}: '{value}' is not a whole number"))
        })
    };
    for (key, value) in keys {
        match key {
            "dmf_beta" => {
                let per_ev = number(key, value)?;
                if per_ev <= 0.0 {
                    return Err(MoptError::parse(
                        "-neb_opt dmf_beta: a positive number (per eV)",
                    ));
                }
                settings.beta = per_ev * Units::HARTREE_TO_EV;
            }
            "dmf_points" => {
                settings.points = whole(key, value)?;
                if settings.points == 0 {
                    return Err(MoptError::parse("-neb_opt dmf_points: one at least"));
                }
            }
            "dmf_guards" => settings.guards = whole(key, value)?,
            "dmf_segments" => {
                settings.segments = whole(key, value)?;
                if settings.segments < 2 {
                    return Err(MoptError::parse("-neb_opt dmf_segments: two at least"));
                }
            }
            "dmf_moves" => {
                settings.moves = match value {
                    "on" => settings.moves.or_else(|| Some(Default::default())),
                    "off" => None,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-neb_opt dmf_moves: on or off, not '{other}'"
                        )))
                    }
                }
            }
            "dmf_tolerance" => {
                settings.tolerance = number(key, value)?;
                if settings.tolerance <= 0.0 {
                    return Err(MoptError::parse(
                        "-neb_opt dmf_tolerance: a positive number",
                    ));
                }
            }
            "dmf_step" => {
                settings.max_step = number(key, value)?;
                if settings.max_step <= 0.0 {
                    return Err(MoptError::parse("-neb_opt dmf_step: a positive length"));
                }
            }
            other => {
                return Err(MoptError::parse(format!(
                    "-neb_opt: no setting '{other}' of the direct MaxFlux method (dmf_beta, \
                     dmf_points, dmf_guards, dmf_segments, dmf_moves, dmf_tolerance, dmf_step)"
                )))
            }
        }
    }
    Ok(Some(settings))
}

/// What a chain relaxation has and the direct MaxFlux method does not.
pub fn refuse_chain_options(config: &NebmainConfig) -> MoptResult<()> {
    let coordinates = config.coordinates.trim().to_ascii_lowercase();
    let refused = [
        (
            !["", "cart", "cartesian"].contains(&coordinates.as_str()),
            "-coord",
        ),
        (config.surrogate.is_some(), "-neb_opt gp=on"),
        (config.chain.added.is_some(), "-aneb"),
        (config.chain.climb_after.is_some(), "-cineb"),
        (config.chain.springs.is_some(), "-ewbneb"),
    ];
    for (asked, what) in refused {
        if asked {
            return Err(MoptError::unsupported(format!(
                "-dmf: {what} is for chains of images; the direct MaxFlux method moves a \
                 smooth path"
            )));
        }
    }
    Ok(())
}

fn describe(settings: &DmfSettings, images: usize, log: &Logger) {
    log.info(&format!(
        "Direct MaxFlux method (-dmf): the path is a B-spline of degree {} in {} segments \
         through the first chain of {images} images; beta {} per eV.",
        settings.degree,
        settings.segments,
        python_float(settings.beta * Units::EV_TO_HARTREE),
    ));
    log.info(&format!(
        "Energies at {} point(s) {} and {} guard(s) between each end and them; the ends once.",
        settings.points,
        if settings.moves.is_some() {
            "that move towards the top of the energy profile"
        } else {
            "at equal shares of the length"
        },
        settings.guards
    ));
    log.info(&format!(
        "The run ends when the largest component of the gradient of the objective is below \
         {:.3e} Hartree/Bohr, or after {} iterations.",
        settings.tolerance, settings.max_iterations
    ));
    log.info(
        "Reference: S.-i. Koda and S. Saito, J. Chem. Theory Comput. 2024, 20, 2798 (direct \
         MaxFlux method)",
    );
}

/// What a point of the path is for.
fn role(settings: &DmfSettings, index: usize, count: usize) -> &'static str {
    let guards = settings.guards;
    if index == 0 || index + 1 == count {
        "end"
    } else if settings.moves.is_none() {
        "point"
    } else if index <= guards || index + 1 + guards >= count {
        "guard"
    } else {
        "near the top"
    }
}

/// How the energies are written: `factor` times the difference, in `unit` (kcal/mol for
/// structures; the objective of plain variables as it is).
#[derive(Clone, Copy)]
struct Scale {
    factor: f64,
    unit: &'static str,
    absolute: &'static str,
}

impl Scale {
    fn of(structures: bool) -> Self {
        if structures {
            Self {
                factor: Units::HARTREE_TO_KCAL_MOL,
                unit: "kcal/mol",
                absolute: "Hartree",
            }
        } else {
            Self {
                factor: 1.0,
                unit: "(objective)",
                absolute: "(objective)",
            }
        }
    }
}

fn progress(
    settings: &DmfSettings,
    report: &DmfReport,
    notes: &[String],
    scale: Scale,
    log: &Logger,
) {
    let first = report.energies[0];
    log.info(&format!(
        "DMF ITR. {:4}  top of the path {:10.4} {} above the first end at {:.4} of its \
         length ({:.10} {})  largest gradient {:9.3e}  objective {:.10}  length {:.4}{}",
        report.iteration,
        scale.factor * (report.e_max - first),
        scale.unit,
        report.t_max,
        report.e_max,
        scale.absolute,
        report.largest_gradient,
        report.objective,
        report.length,
        match (report.refitted, report.taken_back) {
            (_, true) => "  the top rose: the step is taken back and halved",
            (true, false) => "  spline fitted anew",
            (false, false) => "",
        }
    ));
    for note in notes {
        // A warning has its mark already; the log writes it once.
        match note.strip_prefix("Warning: ") {
            Some(warning) => log.warn(warning),
            None => log.info(note),
        }
    }
    let mut table = format!(
        "  point   share of     E - E(first)          E\n\
         \x20         the length   {:>12}   {:>14}\n",
        scale.unit, scale.absolute
    );
    for (k, (share, energy)) in report.t.iter().zip(&report.energies).enumerate() {
        let _ = writeln!(
            table,
            "  {k:5}   {share:9.4}   {:14.4} {:18.10}  {}",
            scale.factor * (energy - first),
            energy,
            role(settings, k, report.t.len())
        );
    }
    log.info(table.trim_end_matches('\n'));
}

/// One structure or one point of variables, as a frame of an XYZ file or a row of a table.
fn point_text(points: &ChainPoints, x: &Col<f64>, comment: &str) -> String {
    match points {
        ChainPoints::Structures(template) => {
            let mut text = format!("{}\n{comment}\n", template.n_atoms());
            for (atom, element) in template.elements.iter().enumerate() {
                let at = |k: usize| x[3 * atom + k] * Units::BOHR_TO_ANGSTROM;
                text.push_str(&crate::io::xyz::multioptpy_row(
                    element.symbol(),
                    "   ",
                    [at(0), at(1), at(2)],
                ));
            }
            text
        }
        ChainPoints::Variables(_) => {
            let values: Vec<String> = x.iter().map(|v| python_float(*v)).collect();
            format!("{comment},{}\n", values.join(","))
        }
    }
}

fn append(path: &Path, text: &str) -> MoptResult<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(text.as_bytes())?;
    Ok(())
}

/// Curvature put on the coordinates held by `-fix` in the Hessian of an end that is
/// minimized, so that the steps leave them where they are (Hartree/Bohr^2).
const HELD: f64 = 1e3;

/// Minimizes the ends of `chain` that `-fe` leaves free, on `surface` (the method needs
/// minima at its ends: its objective is measured from them). The coordinates `fixed`
/// stay where they are; `factor` turns energies into the unit named with it.
fn minimize_ends(
    config: &NebmainConfig,
    chain: &mut [Col<f64>],
    surface: &dyn ChainSurface,
    fixed: &[usize],
    factor: (f64, &str),
    log: &Logger,
) -> MoptResult<()> {
    let last = chain.len() - 1;
    for (index, held) in [(0, config.chain.ends.first), (last, config.chain.ends.last)] {
        let label = if index == 0 { "first" } else { "last" };
        if held {
            log.info(&format!("The {label} end stays as it is read (-fe)."));
            continue;
        }
        let start = surface.evaluate(&chain[index], index, true)?;
        let mut optimizer = VectorOptimizer::new(VectorSettings::default(), chain[index].clone())?;
        if let Some(mut hessian) = start.hessian {
            for &k in fixed {
                for j in 0..hessian.ncols() {
                    hessian[(k, j)] = 0.0;
                    hessian[(j, k)] = 0.0;
                }
                hessian[(k, k)] = HELD;
            }
            optimizer.set_hessian(hessian)?;
        }
        let held_gradient = |mut gradient: Col<f64>| {
            for &k in fixed {
                gradient[k] = 0.0;
            }
            gradient
        };
        let mut report = optimizer.tell(start.energy, held_gradient(start.gradient), None)?;
        let mut end = None;
        while end.is_none() {
            match optimizer.ask() {
                VectorAction::Finished(termination) => end = Some(termination),
                VectorAction::Evaluate { x, need_hessian } => {
                    let found = surface.evaluate(&x, index, need_hessian)?;
                    report = optimizer.tell(
                        found.energy,
                        held_gradient(found.gradient),
                        found.hessian,
                    )?;
                }
            }
        }
        let converged = end == Some(Termination::Converged);
        let line =
            format!(
            "The {label} end minimized in {} iterations ({}): {:+.4} {} from the structure that \
             was read.",
            report.iteration + 1,
            if converged { "converged" } else { "not converged" },
            factor.0 * (report.energy - start.energy),
            factor.1,
        );
        if converged {
            log.info(&line);
        } else {
            log.warn(&line);
        }
        chain[index] = report.x;
    }
    Ok(())
}

/// Moves the first chain `chain` on `surface` by the direct MaxFlux method and reports
/// what it left.
#[allow(clippy::too_many_arguments)]
pub fn relax_by_max_flux(
    config: &NebmainConfig,
    settings: DmfSettings,
    chain: &[Col<f64>],
    surface: &dyn ChainSurface,
    points: ChainPoints,
    map: Option<SurfaceMap>,
    name: &str,
    dir: &Path,
    log: &Logger,
) -> MoptResult<NebmainResult> {
    relax_observed(
        config,
        settings,
        chain,
        surface,
        points,
        map,
        name,
        dir,
        log,
        &mut |_, _, _| {},
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn relax_observed(
    config: &NebmainConfig,
    settings: DmfSettings,
    chain: &[Col<f64>],
    surface: &dyn ChainSurface,
    points: ChainPoints,
    map: Option<SurfaceMap>,
    name: &str,
    dir: &Path,
    log: &Logger,
    saved: &mut super::nebmain::SavedBand<'_>,
) -> MoptResult<NebmainResult> {
    describe(&settings, chain.len(), log);
    log.info(&format!("Surface: {}", surface.description()));
    let structures = matches!(points, ChainPoints::Structures(_));
    let (factor, axis) = if structures {
        (Units::HARTREE_TO_KCAL_MOL, "Energy (kcal/mol)")
    } else {
        (1.0, "Value of the objective")
    };
    let mut chain = chain.to_vec();
    let unit = if structures {
        "kcal/mol"
    } else {
        "in the objective"
    };
    minimize_ends(
        config,
        &mut chain,
        surface,
        &settings.fixed,
        (factor, unit),
        log,
    )?;
    let mut machine = DirectMaxFlux::new(settings.clone(), &chain)?;
    let mut files = ChainFiles::new(dir, name, points.clone())?;
    let mut figures = ChainFigures::new(dir)
        .energy_axis(factor, axis)
        .convergence_title(
            "Largest component of the gradient of the objective",
            "gradient",
        );
    if let Some(map) = map {
        figures = figures.on_surface(map);
    }
    let highest_file = if structures {
        dir.join(format!("{name}_dmf_highest.xyz"))
    } else {
        let file = dir.join("dmf_highest.csv");
        let names = match &points {
            ChainPoints::Variables(names) => names.join(","),
            ChainPoints::Structures(_) => String::new(),
        };
        std::fs::write(&file, format!("iteration,{names}\n"))?;
        file
    };
    if structures && highest_file.exists() {
        std::fs::remove_file(&highest_file)?;
    }
    let mut stop = StopFile::new(dir, log.clone());
    let mut interrupted = false;
    let mut written = None;
    let draw = config.figures;
    let outcome = run_dmf(&mut machine, surface, &mut |machine, notes| {
        let Some(report) = machine.report() else {
            return Ok(Flow::Continue);
        };
        progress(
            machine.settings(),
            report,
            notes,
            Scale::of(structures),
            log,
        );
        written = Some(report.iteration);
        let lengths: Vec<f64> = report.t.iter().map(|t| t * report.length).collect();
        files.write_points(report.iteration, &report.points, &report.energies, &lengths)?;
        let comment = if structures {
            format!(
                "iteration {}, {} Hartree",
                report.iteration,
                python_float(report.e_max)
            )
        } else {
            report.iteration.to_string()
        };
        append(
            &highest_file,
            &point_text(&points, &report.highest, &comment),
        )?;
        if draw {
            // The energy profile of the points, along the length of the path; the path
            // as it is after the step.
            let shares: Vec<f64> = (0..DRAWN).map(|k| k as f64 / (DRAWN - 1) as f64).collect();
            let lengths: Vec<f64> = shares.iter().map(|s| s * report.length).collect();
            let energies: Vec<f64> = shares
                .iter()
                .map(|s| report.profile.at(*s) + report.reference)
                .collect();
            let arc = ArcLength::new(machine.path())?;
            let curve: Vec<Col<f64>> = shares
                .iter()
                .map(|s| machine.path().at(arc.parameter(*s)))
                .collect();
            figures.record(
                report.iteration,
                &lengths,
                &energies,
                &curve,
                report.largest_gradient,
            )?;
        }
        saved(report.iteration, &files, &report.energies);
        log.flush();
        if stop.check() {
            interrupted = true;
            return Ok(Flow::Stop);
        }
        Ok(Flow::Continue)
    })?;

    let end = outcome.end.map(|end| match end {
        DmfEnd::Converged | DmfEnd::Settled => ChainEnd::Converged,
        DmfEnd::IterationLimit => ChainEnd::IterationLimit,
    });
    let Some(last) = outcome.last else {
        return Err(MoptError::calc(
            "the direct MaxFlux method ran no iteration",
        ));
    };
    if written != Some(last.iteration) {
        // The candidates are the points of the path that is left.
        let lengths: Vec<f64> = last.t.iter().map(|t| t * last.length).collect();
        files.write_points(last.iteration, &last.points, &last.energies, &lengths)?;
        saved(last.iteration, &files, &last.energies);
    }
    match outcome.end {
        Some(DmfEnd::Converged) => log.info(&format!(
            "The path is relaxed: the gradient of the objective is below the tolerance after \
             {} iterations.",
            outcome.iterations
        )),
        Some(DmfEnd::IterationLimit) => log.info(&format!(
            "{} iterations were run (-ns). The largest component of the gradient of the \
             objective is {:.3e}, above the tolerance of {:.3e}: the path is not relaxed yet. \
             `nebmain --restart {}` with a larger -ns goes on.",
            outcome.iterations,
            last.largest_gradient,
            settings.tolerance,
            dir.display()
        )),
        Some(DmfEnd::Settled) => log.info(&format!(
            "The path is settled: the top of the path did not come down by {} of the barrier \
             in {} iterations; what is left is the path of iteration {} with the lowest top \
             (largest component of the gradient of the objective {:.3e}, tolerance {:.3e}).",
            python_float(settings.settle),
            settings.patience,
            last.iteration,
            last.largest_gradient,
            settings.tolerance
        )),
        None => log.info(&format!(
            "The run ended after {} iterations before the path was through.",
            outcome.iterations
        )),
    }
    // The path and the estimate of the transition state.
    let arc = ArcLength::new(machine.path())?;
    let mut path_text = String::new();
    if !structures {
        let names = match &points {
            ChainPoints::Variables(names) => names.join(","),
            ChainPoints::Structures(_) => String::new(),
        };
        path_text = format!("share,{names}\n");
    }
    for k in 0..DRAWN {
        let share = k as f64 / (DRAWN - 1) as f64;
        let x = machine.path().at(arc.parameter(share));
        path_text.push_str(&point_text(
            &points,
            &x,
            &if structures {
                format!("share of the length {share:.4}")
            } else {
                python_float(share)
            },
        ));
    }
    let (path_file, guess_file) = if structures {
        (
            dir.join(format!("{name}_dmf_path.xyz")),
            dir.join(format!("{name}_ts_guess.xyz")),
        )
    } else {
        (dir.join("dmf_path.csv"), dir.join("ts_guess.csv"))
    };
    std::fs::write(&path_file, path_text)?;
    let guess = if structures {
        point_text(
            &points,
            &last.highest,
            &format!(
                "{} {}",
                match &points {
                    ChainPoints::Structures(t) => t.spin_charge.charge,
                    ChainPoints::Variables(_) => 0,
                },
                match &points {
                    ChainPoints::Structures(t) => t.spin_charge.multiplicity,
                    ChainPoints::Variables(_) => 1,
                }
            ),
        )
    } else {
        let names = match &points {
            ChainPoints::Variables(names) => names.join(","),
            ChainPoints::Structures(_) => String::new(),
        };
        format!(
            "share,{names}\n{}",
            point_text(&points, &last.highest, &python_float(last.t_max))
        )
    };
    std::fs::write(&guess_file, guess)?;
    let first = last.energies[0];
    log.info(&format!(
        "Estimate of the transition state: the top of the path, {:.4} {} above the \
         first end at {:.4} of its length ({})",
        Scale::of(structures).factor * (last.e_max - first),
        Scale::of(structures).unit,
        last.t_max,
        guess_file.display()
    ));
    log.info(&format!("The path: {}", path_file.display()));
    let candidates = files.candidates();
    if let Some(found) = &candidates {
        log.info(&format!("Last points: {}", found.path.display()));
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
        end,
        interrupted,
        energies: last.energies.clone(),
        candidates,
        optimized: Vec::new(),
    })
}
