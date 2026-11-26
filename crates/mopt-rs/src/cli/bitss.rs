//! `optmain -mf bitss REFERENCE`: binary-image transition state search between the input
//! structure and the structure of the file `REFERENCE` (`path::bitss`).
//!
//! The two structures are the two images. Each is evaluated on the surface of the
//! calculator with the bias potentials; the search itself is the state machine
//! [`Bitss`]. The screen and the files show the pair as one structure of `2N` atoms, as
//! MultiOptPy does; at the end the midpoint of the images is written as the estimate of
//! the transition state.
//!
//! MultiOptPy: the BITSS branch of `optimization.py:ModelFunctionHandler` and
//! `finalize_bitss_trajectory`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::Col;
use rayon::prelude::*;

use super::optmain::{JobSummary, OptmainConfig, OptmainResult};
use crate::coord::alignment::kabsch_align_bohr;
use crate::core_infra::log::{python_float, Logger};
use crate::domain::{EvalRequest, MolecularSystem, Quantity, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{read_xyz_file, write_xyz_file};
use crate::model_hessian::ModelHessian;
use crate::opt::state_machine::{IterationReport, Termination};
use crate::opt::vector::VectorSettings;
use crate::path::bitss::{
    Bitss, BitssAction, BitssEnd, BitssPoint, BitssReport, BitssSettings, PointRole, PointValue,
};
use crate::pes::calculator::{evaluate_checked, Calculator};
use crate::pes::constraint::ConstraintKind;
use crate::pes::term::{PesTerm, TermContext};
use crate::workflow::driver::Outcome;
use crate::workflow::observers::{labelled_geometry_blocks, ProgressLog, RunFiles, StopFile};
use crate::workflow::trajectories::PairTrajectories;
use crate::workflow::Observer;

/// The settings of `-bitss KEY=VALUE ...` on top of the defaults.
///
/// Keys: `f` (reduction of the separation per stage), `alpha`, `beta`, `interval`
/// (iterations between two settings of the coefficients), `points` (points between the
/// images for the barrier estimate), `min_separation` (Bohr), `stages`, `step_share`
/// (longest step as a share of the separation asked for; `none`: no such limit),
/// `midpoint_shift` (end when the midpoint moved by less than this share of the reduction
/// of a stage; `none`: not tested), `midpoint_gradient` (`true` / `false`: end on the forces
/// at the midpoint).
pub fn parse_settings(tokens: &[String]) -> MoptResult<BitssSettings> {
    let mut s = BitssSettings::default();
    for token in tokens {
        let (key, value) = token
            .split_once('=')
            .ok_or_else(|| MoptError::parse(format!("-bitss expects KEY=VALUE, got '{token}'")))?;
        let number = || {
            value
                .parse::<f64>()
                .map_err(|_| MoptError::parse(format!("-bitss {key}: '{value}' is not a number")))
        };
        let count = || {
            value.parse::<usize>().map_err(|_| {
                MoptError::parse(format!("-bitss {key}: '{value}' is not a whole number"))
            })
        };
        let share = || {
            if value.eq_ignore_ascii_case("none") {
                Ok(None)
            } else {
                number().map(Some)
            }
        };
        match key.to_ascii_lowercase().as_str() {
            "f" | "reduction" => s.reduction = number()?,
            "alpha" => s.alpha = number()?,
            "beta" => s.beta = number()?,
            "interval" => s.coefficient_interval = count()?,
            "points" => s.interpolation_points = count()?,
            "min_separation" => s.min_separation = number()?,
            "stages" => s.max_stages = count()?,
            "step_share" => s.step_share = share()?,
            "midpoint_shift" => s.midpoint_shift = share()?,
            "midpoint_gradient" => {
                s.stop_on_midpoint_gradient = match value.to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => true,
                    "false" | "no" | "0" => false,
                    _ => {
                        return Err(MoptError::parse(format!(
                            "-bitss midpoint_gradient: '{value}' is not true or false"
                        )))
                    }
                }
            }
            _ => {
                return Err(MoptError::parse(format!(
                    "-bitss: unknown key '{key}' (f, alpha, beta, interval, points, \
                     min_separation, stages, step_share, midpoint_shift, midpoint_gradient)"
                )))
            }
        }
    }
    Ok(s)
}

/// The BITSS parameters and the optimizer used inside every stage.
pub(super) fn effective_settings(config: &OptmainConfig) -> BitssSettings {
    let s = &config.settings;
    let mut settings = config.bitss.clone();
    settings.minimizer = VectorSettings {
        methods: s.methods.clone(),
        saddle_order: 0,
        max_iterations: s.max_iterations,
        criteria: s.criteria,
        trust: s.trust,
        exact_hessian_interval: s.exact_hessian_interval,
        ccqn: None,
        mode_kill: None,
        project_updates: s.project_updates,
    };
    settings
}

/// The surface of one image: the calculator and the bias potentials.
struct Surface<'a> {
    calc: &'a dyn Calculator,
    terms: &'a [(String, Arc<dyn PesTerm>)],
    start: &'a MolecularSystem,
}

/// A point on the surface, with the energy of the calculator alone.
struct Evaluated {
    value: PointValue,
    raw_energy: f64,
    raw_gradient: Option<Col<f64>>,
}

impl Surface<'_> {
    fn evaluate(&self, point: &BitssPoint, iteration: usize) -> MoptResult<Evaluated> {
        let sys = self.start.with_flat_coords_bohr(&point.x)?;
        let request = EvalRequest {
            need_gradient: point.gradient,
            need_hessian: point.hessian,
            excited_state: 0,
            need_stress: false,
            for_stability: false,
        };
        let out = evaluate_checked(self.calc, &sys, &request)?;
        let mut energy = out.energy.value();
        let mut gradient = point.gradient.then(|| out.gradient_bohr.clone());
        let mut hessian = if point.hessian {
            Some(out.hessian_bohr.clone().ok_or_else(|| {
                MoptError::calc(format!(
                    "calculator '{}' returned no Hessian",
                    self.calc.name()
                ))
            })?)
        } else {
            None
        };
        let ctx = TermContext::new(iteration, self.start, None);
        for (label, term) in self.terms {
            let bias = term
                .evaluate(&sys, &ctx, point.hessian)
                .context(label.clone())?;
            energy += bias.energy.value();
            if let Some(g) = gradient.as_mut() {
                *g += &bias.gradient_bohr;
            }
            if let Some(h) = hessian.as_mut() {
                let of_bias = bias.hessian_bohr.as_ref().ok_or_else(|| {
                    MoptError::calc(format!("bias potential '{label}' returned no Hessian"))
                })?;
                *h += of_bias;
            }
        }
        Ok(Evaluated {
            value: PointValue {
                energy,
                gradient,
                hessian,
            },
            raw_energy: out.energy.value(),
            raw_gradient: point.gradient.then_some(out.gradient_bohr),
        })
    }

    fn commit(&self, x: &Col<f64>, iteration: usize) -> MoptResult<()> {
        if self.terms.is_empty() {
            return Ok(());
        }
        let sys = self.start.with_flat_coords_bohr(x)?;
        let ctx = TermContext::new(iteration, self.start, None);
        for (_, term) in self.terms {
            term.commit(&sys, &ctx);
        }
        Ok(())
    }
}

/// The two images as one structure of `2N` atoms (the first image, then the second).
fn pair_system(
    start: &MolecularSystem,
    first: &Col<f64>,
    second: &Col<f64>,
) -> MoptResult<MolecularSystem> {
    let n = first.nrows();
    let elements = start
        .elements
        .iter()
        .chain(&start.elements)
        .copied()
        .collect();
    let flat = Col::from_fn(2 * n, |k| if k < n { first[k] } else { second[k - n] });
    MolecularSystem::from_flat_bohr(elements, &flat, start.spin_charge, start.cell.clone())
}

/// One frame of MultiOptPy's BITSS trajectory (Angstrom).
fn frame(
    start: &MolecularSystem,
    x: &Col<f64>,
    comment: &str,
    atoms: &[crate::domain::AtomId],
) -> String {
    let mut text = format!("{}\n{comment}\n", atoms.len());
    for &atom in atoms {
        let i = atom.index();
        let element = start.element(atom);
        let _ = writeln!(
            text,
            "{:<2} {:17.12} {:17.12} {:17.12}",
            element.symbol(),
            x[3 * i] * Units::BOHR_TO_ANGSTROM,
            x[3 * i + 1] * Units::BOHR_TO_ANGSTROM,
            x[3 * i + 2] * Units::BOHR_TO_ANGSTROM
        );
    }
    text
}

/// The lines of a report for the log, besides what every optimization shows.
fn stage_lines(r: &BitssReport) -> Vec<String> {
    let mut lines = Vec::new();
    if r.new_stage && r.end.is_none() {
        lines.push(format!(
            "BITSS stage {}: separation asked for {:.6} Bohr",
            r.stage, r.target_separation
        ));
    }
    if let Some(m) = &r.midpoint {
        let forces = match (m.max_force, m.rms_force) {
            (Some(max), Some(rms)) => {
                format!(", max force {max:.6}, RMS force {rms:.6}")
            }
            _ => String::new(),
        };
        lines.push(format!("BITSS midpoint: energy {:.10}{forces}", m.energy));
        if r.end.is_none() {
            lines.push(format!(
                "BITSS coefficients: barrier estimate {:.6e}, kappa_e {:.6e}, kappa_d {:.6e}",
                r.coefficients.barrier, r.coefficients.kappa_e, r.coefficients.kappa_d
            ));
        }
    }
    lines.push(format!(
        "BITSS separation: {:.6} Bohr (asked for {:.6}); energies of the images {:.10} {:.10}",
        r.separation, r.target_separation, r.energies[0], r.energies[1]
    ));
    lines
}

fn end_line(end: BitssEnd) -> &'static str {
    match end {
        BitssEnd::MidpointGradient => {
            "BITSS ended: the forces at the midpoint of the images are below the thresholds."
        }
        BitssEnd::Separation => "BITSS ended: the images are closer than the smallest separation.",
        BitssEnd::MidpointShift => {
            "BITSS ended: the midpoint of the images hardly moved during the last stage."
        }
        BitssEnd::NoBarrier => {
            "BITSS failed: the midpoint is not higher in energy than both images; the pair \
             has left the ridge and lies in one basin. A smaller reduction of the separation \
             (-bitss f=0.25) or shorter steps (-bitss step_share=0.25) may keep it there."
        }
        BitssEnd::StageLimit => "BITSS stopped: the number of stages is used up.",
        BitssEnd::IterationLimit => "BITSS stopped: the number of iterations is used up.",
    }
}

/// Files of the two images and of the estimate of the transition state. While the search
/// runs, every iteration appends each image to a trajectory of its own
/// (`<stem>_image_1_traj.xyz`, `<stem>_image_2_traj.xyz`); at the end the two become
/// `<stem>_traj.xyz`.
struct ImageFiles {
    dir: PathBuf,
    stem: String,
    images: PairTrajectories,
    atoms: Vec<crate::domain::AtomId>,
    path: Option<PairTrajectories>,
}

impl ImageFiles {
    fn new(
        dir: &Path,
        stem: &str,
        atoms: Vec<crate::domain::AtomId>,
        keep_path: bool,
    ) -> MoptResult<Self> {
        let names = [
            format!("{stem}_image_1_traj.xyz"),
            format!("{stem}_image_2_traj.xyz"),
        ];
        Ok(Self {
            dir: dir.to_path_buf(),
            stem: stem.to_string(),
            images: PairTrajectories::start(dir, [&names[0], &names[1]])?,
            atoms,
            path: if keep_path {
                Some(PairTrajectories::start(
                    dir,
                    ["first_path_image.xyz", "second_path_image.xyz"],
                )?)
            } else {
                None
            },
        })
    }

    fn ts_guess_path(&self) -> PathBuf {
        self.dir.join(format!("{}_ts_guess.xyz", self.stem))
    }

    /// The two images of an iteration, each to its trajectory.
    fn push(&mut self, start: &MolecularSystem, images: &[Col<f64>; 2]) -> MoptResult<()> {
        let k = self.images.frames()[0];
        for (i, x) in images.iter().enumerate() {
            let comment = format!("BITSS_Step {k} image {}", i + 1);
            self.images
                .push(i, &frame(start, x, &comment, &self.atoms))?;
            if let Some(path) = &mut self.path {
                path.push(
                    i,
                    &frame(start, x, &comment, &start.atoms().collect::<Vec<_>>()),
                )?;
            }
        }
        Ok(())
    }

    /// MultiOptPy's `<stem>_traj.xyz` of a BITSS run: the first image from its start to its
    /// end, then the second from its end back to its start, so that the frames lead from
    /// one starting structure to the other. The trajectories of the images are removed.
    fn join(self, path: &Path) -> MoptResult<PathBuf> {
        self.images
            .join(path, false, &|step, _| format!("BITSS_Step {step}"))?;
        if let Some(images) = self.path {
            let file = self.dir.join(format!("{}_path.xyz", self.stem));
            images.join(&file, false, &|step, _| format!("BITSS_Step {step}"))?;
            Ok(file)
        } else {
            Ok(path.to_owned())
        }
    }
}

/// Run the search for one input.
#[allow(clippy::too_many_arguments)]
pub fn run(
    config: &OptmainConfig,
    reference: &Path,
    terms: &[(String, Arc<dyn PesTerm>)],
    input: &Path,
    start: &MolecularSystem,
    calc: &dyn Calculator,
    dir: &Path,
    log: &Logger,
) -> MoptResult<OptmainResult> {
    let s = &config.settings;
    let mut frozen = Vec::new();
    for constraint in &s.constraints {
        match &constraint.kind {
            ConstraintKind::Frozen(atoms) => {
                frozen.extend(atoms.iter().flat_map(|a| (0..3).map(|k| 3 * a.index() + k)));
            }
            _ => {
                return Err(MoptError::unsupported(
                    "-mf bitss takes fixed atoms (-fix) only; other constraints are not \
                     available with it",
                ))
            }
        }
    }
    let other = read_xyz_file(reference, start.spin_charge, None)
        .context(format!("BITSS reference {}", reference.display()))?;
    if other.elements != start.elements {
        return Err(MoptError::config(
            "BITSS: the input and the reference must have the same atoms in the same order",
        ));
    }
    // Without fixed atoms the reference is laid onto the input (in a cell: its atoms are
    // taken to the images next to those of the input), so that the separation of the
    // images is that of their shapes.
    let other = if frozen.is_empty() && (calc.is_rigid_body_invariant() || start.has_pbc()) {
        kabsch_align_bohr(&other, start, false)?.0
    } else if start.has_pbc() {
        other.unwrapped_to(start)?
    } else {
        other
    };
    let settings = effective_settings(config);
    let first = start.coords_flat_bohr();
    let second = other.coords_flat_bohr();
    let mut bitss = Bitss::new(settings.clone(), &first, &second)?.with_frozen(&frozen)?;
    let model: Option<Box<dyn ModelHessian>> = match s.model_hessian.as_deref() {
        Some(name) if s.exact_hessian_interval.is_none() => {
            Some(crate::model_hessian::resolve(name)?)
        }
        _ => None,
    };

    log.info(&format!(
        "Mode: Model Function Optimization (bitss): binary-image transition state search \
         between {} and {}",
        input.display(),
        reference.display()
    ));
    log.info(
        "Reference: S. J. Avis, J. R. Panter, H. Kusumaatmaja, J. Chem. Phys. 2022, 157, \
         124107 (binary-image transition state search)",
    );
    log.info(&format!(
        "BITSS: reduction of the separation per stage {}, alpha {}, beta {}, coefficients \
         set every {} iterations, {} point(s) between the images for the barrier estimate, \
         first separation {:.6} Bohr",
        python_float(settings.reduction),
        python_float(settings.alpha),
        python_float(settings.beta),
        settings.coefficient_interval,
        settings.interpolation_points,
        bitss.target_separation()
    ));
    if s.saddle_order != 0 {
        log.info("BITSS minimizes its function: -order is not used.");
    }
    if s.coordinates != "cart" {
        log.info("BITSS works in the Cartesian coordinates of the two images: -coord is not used.");
    }
    let pair = pair_system(start, &first, &second)?;
    log.info("BITSS Mode detected: Expanding state to 2N atoms.");
    let labels = ["State1", "State2"];
    log.info(&format!(
        "{}\n",
        labelled_geometry_blocks(&pair, &labels.map(str::to_string))
    ));

    let stem = input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    // The two images are shown one after the other in every iteration, as MultiOptPy does.
    let mut progress = ProgressLog::new(log.clone(), s.criteria).in_parts(&labels);
    let mut files = RunFiles::new(dir, &stem)?.without_trajectory();
    let mut stop = StopFile::new(dir, log.clone());
    let selected = config
        .calculator_options
        .oniom
        .as_ref()
        .and_then(|spec| spec.trajectory_atoms());
    let keep_path = selected.is_some() && config.keep_full_path;
    let mut images = ImageFiles::new(
        dir,
        &stem,
        selected.unwrap_or_else(|| start.atoms().collect()),
        keep_path,
    )?;
    let surface = Surface { calc, terms, start };
    let mut last: Option<(MolecularSystem, f64)> = None;
    let mut last_raw: Option<[(f64, Col<f64>); 2]> = None;
    let mut shown = 0_usize;
    let end: Option<BitssEnd> = loop {
        let points = match bitss.ask() {
            BitssAction::Finished(end) => break Some(end),
            BitssAction::Evaluate(points) => points,
        };
        let iteration = bitss.iterations();
        let evaluated: Vec<Evaluated> = if points.len() > 1 {
            points
                .par_iter()
                .map(|p| surface.evaluate(p, iteration))
                .collect::<MoptResult<_>>()?
        } else {
            points
                .iter()
                .map(|p| surface.evaluate(p, iteration))
                .collect::<MoptResult<_>>()?
        };
        // The energies of the calculator alone at the two images, when they were evaluated.
        let raw: Option<[(f64, Col<f64>); 2]> = match (points.first(), evaluated.get(..2)) {
            (Some(p), Some([a, b])) if p.role == PointRole::First => a
                .raw_gradient
                .clone()
                .zip(b.raw_gradient.clone())
                .map(|(ga, gb)| [(a.raw_energy, ga), (b.raw_energy, gb)]),
            _ => None,
        };
        if raw.is_some() {
            last_raw.clone_from(&raw);
        }
        // A model of the curvature of the surface at the images, whenever the line between
        // them is surveyed (a new minimization starts there).
        let survey = points
            .iter()
            .any(|p| matches!(p.role, PointRole::Between(_)));
        if let (Some(model), Some([(_, ga), (_, gb)]), true) = (&model, &last_raw, survey) {
            let (x1, x2) = bitss.images();
            let h1 = model.hessian(&start.with_flat_coords_bohr(&x1)?, ga)?;
            let h2 = model.hessian(&start.with_flat_coords_bohr(&x2)?, gb)?;
            bitss.set_curvature_guess(h1, h2)?;
        }
        let report = bitss.tell(evaluated.into_iter().map(|e| e.value).collect())?;
        let lines = stage_lines(&report);
        let pair = pair_system(start, &report.images[0], &report.images[1])?;
        images.push(start, &report.images)?;
        let Some(step) = &report.step else {
            // The search ended at the survey of the line between the images.
            for line in &lines {
                log.info(line);
            }
            last = Some((pair, report.function));
            continue;
        };
        surface.commit(&report.images[0], report.iteration)?;
        surface.commit(&report.images[1], report.iteration)?;
        let mut components = vec![
            ("State1".to_string(), Quantity::new(report.energies[0])),
            ("State2".to_string(), Quantity::new(report.energies[1])),
        ];
        let n = report.images[0].nrows();
        let raw_gradient = match &raw {
            Some([(e1, g1), (e2, g2)]) => {
                if !terms.is_empty() {
                    components.push(("State1_pes".to_string(), Quantity::new(*e1)));
                    components.push(("State2_pes".to_string(), Quantity::new(*e2)));
                }
                Col::from_fn(2 * n, |k| if k < n { g1[k] } else { g2[k - n] })
            }
            None => Col::from_fn(2 * n, |k| {
                if k < n {
                    report.gradients[0][k]
                } else {
                    report.gradients[1][k - n]
                }
            }),
        };
        let mut notes = lines;
        notes.extend(report.notes.iter().cloned());
        let shown_report = IterationReport {
            crossing: None,
            iteration: shown,
            system: pair.clone(),
            energy: Quantity::new(report.energies[0] + report.energies[1]),
            biased_energy: Quantity::new(report.function),
            gradient: report.gradient.clone(),
            raw_gradient,
            components,
            convergence: step.convergence,
            step_norm: Quantity::new(step.step_norm),
            trust_radius: Quantity::new(step.trust_radius),
            notes,
            debug_notes: Vec::new(),
            program_outputs: Vec::new(),
            coordinates: None,
            spin_squares: Vec::new(),
            hessian_eigenvalues: None,
        };
        shown += 1;
        progress.on_iteration_with_hessian(&shown_report, bitss.updated_hessian())?;
        files.on_iteration(&shown_report)?;
        last = Some((pair.clone(), report.function));
        if stop.should_stop(&pair) {
            break None;
        }
    };
    let (system, function) =
        last.ok_or_else(|| MoptError::config("the search ended before an evaluation"))?;
    let converged = end.is_some_and(BitssEnd::converged);
    let outcome = Outcome {
        termination: end.map(|e| {
            if e.converged() {
                Termination::Converged
            } else {
                Termination::MaxIterations
            }
        }),
        system,
        energy: Quantity::new(function),
        biased_energy: Quantity::new(function),
        iterations: bitss.iterations(),
    };
    progress.on_finish(&outcome)?;
    files.on_finish(&outcome)?;
    // As MultiOptPy: the trajectory leads from one image to the other.
    let ts_guess_path = images.ts_guess_path();
    let trajectory = images.join(&files.trajectory_path())?;
    if let Some(end) = end {
        log.info(end_line(end));
    }
    let guess = start.with_flat_coords_bohr(&bitss.transition_state_estimate())?;
    let surveyed = bitss.midpoint().map(|m| m.energy);
    let guess_energy = if end.is_some() { surveyed } else { None };
    write_xyz_file(&ts_guess_path, &guess, guess_energy.map(Quantity::new))?;
    let (x1, x2) = bitss.images();
    for (k, x) in [x1, x2].iter().enumerate() {
        write_xyz_file(
            dir.join(format!("{stem}_bitss_image_{}.xyz", k + 1)),
            &start.with_flat_coords_bohr(x)?,
            None,
        )?;
    }
    if converged {
        log.info(&format!(
            "The midpoint of the two images is the estimate of the transition state{}; a \
             search of order 1 from it (optmain {} -order 1 ...) and a vibrational analysis \
             confirm it.",
            guess_energy.map_or_else(String::new, |e| format!(" (energy {e:.10})")),
            ts_guess_path.display()
        ));
    }
    log.info(&format!(
        "Trial of geometry optimization ({}) was completed.",
        input.display()
    ));
    if config.inputs.last().is_some_and(|last| last == input) {
        log.info("All calculations were completed.");
    }
    log.info(&format!(
        "Transition state estimate file path: {}",
        ts_guess_path.display()
    ));
    log.info(&format!(
        "Optimized structure file path: {}",
        files.optimized_path().display()
    ));
    log.info(&format!(
        "Trajectory file path: {}",
        files.trajectory_path().display()
    ));
    log.flush();
    Ok(OptmainResult {
        directory: dir.to_path_buf(),
        optimized: ts_guess_path,
        converged,
        interrupted: stop.stopped(),
        summary: Some(JobSummary {
            input: input.to_path_buf(),
            trajectory,
            system: guess,
            energy: Quantity::new(surveyed.unwrap_or(function)),
            biased_energy: Quantity::new(function),
            symmetry: None,
            iterations: outcome.iterations,
            irc_endpoints: Vec::new(),
            imaginary_modes: None,
        }),
    })
}
