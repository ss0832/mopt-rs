//! `run_autots` on an objective of plain variables (`"othersoft": "mb2d"` in `step1_settings`,
//! any objective of `optmain -os`, any number of variables): the four steps on the variables,
//! in this process, with files of variables in the place of structures.
//!
//! - **The input**: frames of plain variables, the reactant first and the product last (the
//!   number of variables, a comment line, and `name value` lines, as `run_mapper` writes its
//!   paths), or one variables file with `--skip_to_step4` (the transition state).
//! - **Step 1**: the two-point method `step1_method` of the registry of two-point methods
//!   between the first and the last point (`dsafir` without it: AFIR pushes need atoms, and
//!   DS-AFIR is its two-ended form); the keys of `step1_settings` that the method has are its
//!   settings. With `--skip_step1` the input is the path.
//! - **Step 2**: the chain relaxation of `nebmain` (`neb` of the registry, `bitss` with
//!   `"step2_method": "bitss"`) from the path of step 1, with `step2_settings` as the settings
//!   of `nebmain`; the maxima of the relaxed chain are the candidates, the `top_n_candidates`
//!   highest.
//! - **Step 3**: every candidate optimized at saddle order 1 (`step3_settings`, the optimizer
//!   of plain variables), and its index checked on the Hessian of the objective (one negative
//!   eigenvalue makes a transition state); the same transition state found twice goes on once.
//! - **Step 4** (`--run_step4`): the IRC with unit masses (`intrinsic_reaction_coordinates` of
//!   `step4_settings`), its two ends optimized to minima (`step4b_opt_method`).
//!
//! The files are variables files (`<stem>_ts_final_<k>.txt`, `<stem>_ts<k>_end_<j>.txt`,
//! readable by `optmain -os ...`) and frames of variables (`<stem>_step1_path.txt`,
//! `<stem>_step2_path.txt`, `<stem>_ts<k>_irc.txt`); `autots_summary.csv` gives the values of
//! the objective as they are, and `autots_report.html` draws the paths on the variables (on
//! the surface itself for two). The objective is not journaled: `--restart` runs the steps
//! again (their values are the same every time).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's AutoTS works on molecules only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::Col;

use super::autots::{
    step_args, AutotsConfig, AutotsResult, FirstPathMethod, PathMethod, PathStage, TransitionState,
};
use super::optmain::OptmainConfig;
use super::spec::OPTMAIN;
use super::variables::{format_variables, read_variables_file, vector_settings};
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::Logger;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::irc::{Irc, IrcEvent, IrcSettings, IrcSpace};
use crate::opt::state_machine::Termination;
use crate::opt::vector::VectorOptimizer;
use crate::pes::counting::EvaluationCounts;
use crate::pes::objective::{check_dimension, Objective, ObjectiveRegistry};
use crate::workflow::irc::{run_irc, IrcObserver, ObjectiveSurface};
use crate::workflow::mapper::configuration::{variable_frames, Configuration, VariablePoint};
use crate::workflow::pair_search::{
    PairInput, PairMethodRegistry, PairSettings, SearchRun, VariableSurface,
};
use crate::workflow::run_vector_optimization_until;

/// The two-point method of step 1 on plain variables when `step1_method` names none.
pub const DEFAULT_FIRST_METHOD: &str = "dsafir";

/// Two transition states closer than this (in the units of the variables) and within
/// [`SAME_VALUE`] are the same (the `variable_tolerance` of the mapper).
pub const SAME_DISTANCE: f64 = 0.01;
/// See [`SAME_DISTANCE`].
pub const SAME_VALUE: f64 = 1e-6;

/// A point of the objective with its value.
#[derive(Debug, Clone)]
pub struct Point {
    /// The variables.
    pub x: Col<f64>,
    /// The value of the objective.
    pub value: f64,
}

/// What the steps on plain variables made, for the report.
#[derive(Debug, Clone, Default)]
pub struct VariablesRun {
    /// The names of the variables.
    pub names: Vec<String>,
    /// The path of step 1 (or the input path).
    pub first_path: Vec<Point>,
    /// The relaxed chain of step 2.
    pub chain: Vec<Point>,
    /// The candidates of step 2 (their places in `chain`).
    pub candidates: Vec<Point>,
    /// Evaluated points of each Step 3 optimization when an HTML report is requested.
    pub optimizations: Vec<Vec<Point>>,
    /// The optimized candidates of step 3: the point, whether it converged, the negative
    /// curvatures of the Hessian there, and an earlier one it is the same as.
    pub saddles: Vec<(Point, bool, usize, Option<usize>)>,
    /// The IRC of every transition state (by its index in `saddles`): its points from the
    /// backward end to the forward end, and the optimized ends.
    pub ircs: Vec<(usize, Vec<Point>, Vec<Point>)>,
}

/// The objective of the settings of step 1, when it is one of plain variables.
pub fn objective_of(
    config: &AutotsConfig,
) -> Option<(String, crate::pes::registry::CalculatorOptions)> {
    let name = config.steps[0].get("othersoft")?.as_str()?.to_string();
    if !ObjectiveRegistry::builtin().contains(&name) {
        return None;
    }
    let mut options = crate::pes::registry::CalculatorOptions::default();
    if let Some(items) = config.steps[0]
        .get("calc_option")
        .and_then(JsonValue::as_string_vec)
    {
        for item in items {
            if let Some((k, v)) = item.split_once('=') {
                options
                    .extra
                    .push((k.trim().to_string(), v.trim().to_string()));
            }
        }
    }
    Some((name, options))
}

/// The points of the input: frames of variables, or one variables file.
fn read_points(path: &Path, objective: &dyn Objective) -> MoptResult<(Vec<String>, Vec<Col<f64>>)> {
    let text =
        std::fs::read_to_string(path).map_err(|e| MoptError::from(e).context(path.display()))?;
    if let Ok(frames) = variable_frames(&text) {
        if let Some((_, first)) = frames.first() {
            let names = first.names.to_vec();
            let points = frames.into_iter().map(|(_, p)| p.values).collect();
            return Ok((names, points));
        }
    }
    let vars = read_variables_file(path)?;
    let n = vars.values.len();
    check_dimension(objective, n)?;
    let names = vars.names.unwrap_or_else(|| objective.variable_names(n));
    Ok((names, vec![Col::from_fn(n, |i| vars.values[i])]))
}

/// The optimizer settings of a step (its block with `extra` over it).
fn step_config(
    settings: &super::autots_workflow::StepSettings,
    input: &Path,
    extra: &[(&str, JsonValue)],
    drop: &[&str],
    name: &str,
) -> MoptResult<OptmainConfig> {
    let args = step_args(settings, &[input], extra, drop, OPTMAIN).context(name)?;
    OptmainConfig::from_args(&args).context(name)
}

/// An optimization of `start` with `config` (`order` 1: a transition state); the point it
/// ended at and whether it converged.
fn optimize(
    objective: &dyn Objective,
    config: &OptmainConfig,
    start: &Col<f64>,
    mut history: Option<&mut Vec<Point>>,
) -> MoptResult<(Point, bool)> {
    let values: Vec<f64> = start.iter().copied().collect();
    let settings = vector_settings(config, &values)?;
    let mut opt = VectorOptimizer::new(settings, start.clone())?;
    let outcome = run_vector_optimization_until(
        &mut opt,
        objective,
        &mut |step| {
            if let Some(points) = history.as_mut() {
                points.push(Point {
                    x: step.x.clone(),
                    value: step.energy,
                });
            }
            Ok(())
        },
        &mut || false,
    )?;
    Ok((
        Point {
            x: outcome.x,
            value: outcome.value,
        },
        outcome.termination == Some(Termination::Converged),
    ))
}

/// The negative curvatures of the Hessian of the objective at `x` by the one criterion of an
/// imaginary mode (`opt::mode_kill`: an eigenvalue below minus
/// [`crate::opt::mode_kill::IMAGINARY_CURVATURE`], unit masses for plain variables), as every
/// count of the mapper takes it.
fn negative_curvatures(objective: &dyn Objective, x: &Col<f64>) -> MoptResult<usize> {
    let v = objective.evaluate(x, true)?;
    let h = v.hessian.ok_or_else(|| {
        MoptError::calc(format!("objective '{}' gave no Hessian", objective.name()))
    })?;
    crate::opt::mode_kill::imaginary_modes(&h, None, None)
}

/// The points of an IRC as they come.
#[derive(Default)]
struct IrcPoints {
    sides: Vec<(crate::irc::types::Branch, Vec<Point>)>,
    transition_state: bool,
    /// The start: its imaginary modes, whether it is stationary, its largest gradient.
    start: Option<(usize, bool, f64)>,
}

impl IrcObserver for IrcPoints {
    fn on_event(&mut self, event: &IrcEvent) -> MoptResult<()> {
        match event {
            IrcEvent::Start(r) => {
                self.transition_state = r.kind == crate::irc::StartKind::TransitionState;
                self.start = Some((r.imaginary, r.stationary, r.max_gradient));
            }
            IrcEvent::Step { branch, record } => {
                let point = Point {
                    x: record.x.clone(),
                    value: record.energy,
                };
                match self.sides.iter_mut().find(|(b, _)| b == branch) {
                    Some((_, points)) => points.push(point),
                    None => self.sides.push((*branch, vec![point])),
                }
            }
            IrcEvent::FirstStep(_) | IrcEvent::BranchFinished(_) => {}
        }
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        false
    }
}

/// The frames of `points` (the number of variables, the comment, `name value` lines).
pub(super) fn frames_text(names: &[String], points: &[Point], label: &str) -> String {
    let shared = Arc::new(names.to_vec());
    let mut out = String::new();
    for (k, p) in points.iter().enumerate() {
        if let Ok(c) = Configuration::variables(p.x.clone(), Arc::clone(&shared)) {
            out.push_str(&c.frame(&format!("{label} {} energy={:.12}", k + 1, p.value)));
        }
    }
    out
}

/// A variables file of `point`.
fn variables_text(names: &[String], point: &Point) -> String {
    let values: Vec<f64> = point.x.iter().copied().collect();
    format_variables(names, &values, false, point.value)
}

fn distance(a: &Col<f64>, b: &Col<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt()
}

/// Run the four steps of `config` on the objective `name` (with `options`) in the working
/// directory `work`, from the copy of the input `input`; `result` gets what they made, the
/// evaluations of every step among it, and the returned run is what the report draws.
pub fn run(
    config: &AutotsConfig,
    work: &Path,
    input: &Path,
    log: &Logger,
    objective_name: &str,
    options: &crate::pes::registry::CalculatorOptions,
    result: &mut AutotsResult,
) -> MoptResult<VariablesRun> {
    let objectives = ObjectiveRegistry::builtin();
    let objective = objectives.create(objective_name, options)?;
    let objective = objective.as_ref();
    let (names, points) = read_points(input, objective)?;
    let stem = input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    log.info(&format!(
        "AutoTS on plain variables: {} ({} variables: {}), {} input point(s).",
        objective.description(),
        names.len(),
        names.join(", "),
        points.len()
    ));
    if config.workflow.is_some() {
        log.warn(
            "AutoTS on plain variables: the workflow block is not used; every step runs once \
             with its settings.",
        );
    }
    let mut run = VariablesRun {
        names: names.clone(),
        ..VariablesRun::default()
    };
    let mut counted = objectives.counts();
    let mut tally = |result: &mut AutotsResult, step: usize| {
        let now = objectives.counts();
        result.evaluations[step] = result.evaluations[step].plus(&now.since(&counted));
        counted = now;
    };
    let step3 = config.steps[2].clone();
    let step4 = config.steps[3].clone();
    let saddle_config = step_config(
        &step3,
        input,
        &[("saddle_order", JsonValue::String("1".into()))],
        &[],
        "step3_settings",
    )?;
    result.effective_settings.insert(
        "Objective".into(),
        format!("{objective_name}\n{options:#?}"),
    );
    result.effective_settings.insert(
        "Step 3 · saddle optimizer".into(),
        super::autots_settings::optimization(&saddle_config),
    );
    let mut tss: Vec<Col<f64>> = Vec::new();
    if config.skip_to_step4 {
        let start = points
            .first()
            .ok_or_else(|| MoptError::parse("run_autots: the input has no point"))?;
        let value = objective.evaluate(start, false)?.value;
        // The input is judged by the start of its IRC (one decision, `irc::start`); its
        // negative curvatures by the Hessian of the objective, as in step 3.
        let negative = negative_curvatures(objective, start)?;
        run.saddles.push((
            Point {
                x: start.clone(),
                value,
            },
            true,
            negative,
            None,
        ));
        result.transition_states.push(TransitionState {
            file: input.to_path_buf(),
            energy: value,
            biased_energy: value,
            converged: true,
            imaginary_modes: Some(negative),
            same_as: None,
            meta_irc: None,
            ends: Vec::new(),
            end_checks: Vec::new(),
            directory: work.to_path_buf(),
            irc_directory: None,
        });
        tss.push(start.clone());
    } else {
        // Step 1: the first path.
        let first_path: Vec<Point> = if config.skip_step1 {
            log.info("AutoTS step 1 skipped: the input is the path.");
            result.first_path_method = "given path".to_string();
            points
                .iter()
                .map(|x| {
                    objective.evaluate(x, false).map(|v| Point {
                        x: x.clone(),
                        value: v.value,
                    })
                })
                .collect::<MoptResult<Vec<_>>>()?
        } else {
            let (Some(a), Some(b)) = (points.first(), points.last()) else {
                return Err(MoptError::parse("run_autots: the input has no point"));
            };
            if points.len() < 2 {
                return Err(MoptError::parse(
                    "run_autots on plain variables: step 1 needs the reactant and the product, \
                     the first and the last frame of the input (AFIR pushes need atoms), or \
                     --skip_to_step4 with the transition state",
                ));
            }
            let method = match config.first_method {
                FirstPathMethod::Bias => DEFAULT_FIRST_METHOD,
                FirstPathMethod::Bitss => "bitss",
                FirstPathMethod::Pair(m) => m.name,
            };
            log.info(&format!(
                "AutoTS step 1: the first path by {method} between the first and the last point."
            ));
            let registry = PairMethodRegistry::default();
            let known = registry.get(method)?.keys();
            let settings: Vec<(String, String)> = config.steps[0]
                .iter()
                .filter(|(k, _)| known.iter().any(|(key, _)| key == k))
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.as_str()
                            .map_or_else(|| v.to_string_compact(), str::to_string),
                    )
                })
                .collect();
            let path = pair_path(
                &registry,
                method,
                &[a.clone(), b.clone()],
                settings,
                objective,
                work,
                &format!("{stem}_step1"),
                log,
            )?;
            if let Some(note) = super::autots_pair::named(method).and_then(|m| m.seed_note()) {
                log.info(note);
                result.notes.push(note.to_string());
            }
            result.effective_settings.insert(
                format!("Step 1 · {method}"),
                std::fs::read_to_string(
                    work.join(format!("{stem}_step1"))
                        .join(crate::workflow::pair_search::METHOD_SETTINGS_FILE),
                )?,
            );
            result.first_path_method = method.to_string();
            path
        };
        let first_file = work.join(format!("{stem}_step1_path.txt"));
        std::fs::write(
            &first_file,
            frames_text(&names, &first_path, "step 1 point"),
        )?;
        result.afir_path = Some(first_file);
        run.first_path = first_path;
        tally(result, 0);

        // Step 2: the chain from the first path.
        let method = match config.path_method {
            PathMethod::Chain => "neb",
            PathMethod::Bitss => "bitss",
        };
        log.info(&format!(
            "AutoTS step 2: the path by {method} from the path of step 1."
        ));
        let registry = PairMethodRegistry::default();
        let mut settings = Vec::new();
        if method == "neb" {
            if let Some(block) = config.document.get("step2_settings") {
                settings.push(("settings".to_string(), block.to_string_compact()));
            }
        }
        let starts: Vec<Col<f64>> = if method == "neb" {
            run.first_path.iter().map(|p| p.x.clone()).collect()
        } else {
            let (Some(a), Some(b)) = (run.first_path.first(), run.first_path.last()) else {
                return Err(MoptError::calc("run_autots: step 1 gave no path"));
            };
            vec![a.x.clone(), b.x.clone()]
        };
        let (chain, estimates, path_method) = pair_search(
            &registry,
            method,
            &starts,
            settings,
            objective,
            work,
            &format!("{stem}_step2"),
            log,
            config.html_report.then_some(&mut result.band_profiles),
        )?;
        result.effective_settings.insert(
            format!("Step 2 · {method}"),
            std::fs::read_to_string(
                work.join(format!("{stem}_step2"))
                    .join(crate::workflow::pair_search::METHOD_SETTINGS_FILE),
            )?,
        );
        let chain_file = work.join(format!("{stem}_step2_path.txt"));
        std::fs::write(&chain_file, frames_text(&names, &chain, "step 2 image"))?;
        let mut estimates = estimates;
        estimates.sort_by(|a, b| b.value.total_cmp(&a.value));
        estimates.truncate(config.top_n.max(1));
        let mut stage = PathStage {
            method: path_method,
            path: Some(chain_file),
            energies: chain.iter().map(|p| p.value).collect(),
            ..PathStage::default()
        };
        for (k, e) in estimates.iter().enumerate() {
            let file = work.join(format!("{stem}_ts_guess_{}.txt", k + 1));
            std::fs::write(&file, variables_text(&names, e))?;
            stage.estimates.push(file.clone());
            result.candidates.push(file);
        }
        result.path_stage = Some(stage);
        run.chain = chain;
        run.candidates = estimates.clone();
        tally(result, 1);

        // Step 3: the transition states.
        for (k, guess) in estimates.iter().enumerate() {
            let mut history = Vec::new();
            let (ts, converged) = optimize(
                objective,
                &saddle_config,
                &guess.x,
                config.html_report.then_some(&mut history),
            )?;
            if config.html_report {
                run.optimizations.push(history);
            }
            let negative = negative_curvatures(objective, &ts.x)?;
            let same_as = run.saddles.iter().position(|(p, ok, n, _)| {
                *ok && *n == 1
                    && distance(&p.x, &ts.x) < SAME_DISTANCE
                    && (p.value - ts.value).abs() < SAME_VALUE
            });
            let file = work.join(format!("{stem}_ts_final_{}.txt", k + 1));
            std::fs::write(&file, variables_text(&names, &ts))?;
            log.info(&format!(
                "AutoTS step 3: candidate {} {} at value {:.10} with {negative} negative curvature(s){}.",
                k + 1,
                if converged { "converged" } else { "did not converge" },
                ts.value,
                same_as.map_or_else(String::new, |j| format!(", the same as transition state {}", j + 1))
            ));
            result.transition_states.push(TransitionState {
                file,
                energy: ts.value,
                biased_energy: ts.value,
                converged,
                imaginary_modes: Some(negative),
                same_as: same_as.map(|j| j + 1),
                meta_irc: None,
                ends: Vec::new(),
                end_checks: Vec::new(),
                directory: work.to_path_buf(),
                irc_directory: None,
            });
            if converged && negative == 1 && same_as.is_none() {
                tss.push(ts.x.clone());
            } else {
                tss.push(Col::zeros(0));
            }
            run.saddles.push((ts, converged, negative, same_as));
        }
        tally(result, 2);
    }

    // Step 4: the IRC of every transition state, its ends optimized.
    if config.run_step4 {
        let irc_config = step_config(
            &step4,
            input,
            &[("saddle_order", JsonValue::String("1".into()))],
            &["step4b_opt_method"],
            "step4_settings",
        )?;
        let irc_settings = match irc_config.irc.clone() {
            Some(s) => s,
            None => IrcSettings::from_multioptpy("0.1", "300", "lqa")?,
        };
        let end_method = step4.get("step4b_opt_method").cloned().unwrap_or_else(|| {
            JsonValue::Array(vec![JsonValue::String("rsirfo_block_fsb".into())])
        });
        // The ends are minimized with the mode kill and checked by their Hessian, as those of
        // structures are (`EndCheck`), unless check_frequencies is off; keys of step4_settings
        // win.
        let mut more = vec![
            ("saddle_order", JsonValue::String("0".into())),
            ("opt_method", end_method),
        ];
        if config.check_frequencies {
            for (key, value) in [
                ("mode_kill", JsonValue::Bool(true)),
                ("mode_kill_method", JsonValue::String("classic".into())),
            ] {
                if !step4.contains_key(key) {
                    more.push((key, value));
                }
            }
        }
        let minimum_config = step_config(
            &step4,
            input,
            &more,
            &["step4b_opt_method", "intrinsic_reaction_coordinates"],
            "step4_settings",
        )?;
        result
            .effective_settings
            .insert("Step 4 · IRC".into(), format!("{irc_settings:#?}"));
        result.effective_settings.insert(
            "Step 4 · endpoint optimizer".into(),
            super::autots_settings::optimization(&minimum_config),
        );
        for (k, x) in tss.iter().enumerate() {
            if x.nrows() == 0 {
                continue;
            }
            let c = &saddle_config.settings.criteria;
            let settings = irc_settings
                .clone()
                .accepting_converged(c.max_force.value(), c.rms_force.value());
            let mut irc = Irc::new(settings, IrcSpace::variables(x.nrows()), x.clone())?;
            let mut points = IrcPoints::default();
            run_irc(&mut irc, &ObjectiveSurface(objective), &mut [&mut points])?;
            if !points.transition_state && !config.skip_to_step4 {
                log.warn(&format!(
                    "AutoTS step 4: transition state {} is not one by its Hessian; no IRC.",
                    k + 1
                ));
                continue;
            }
            // The input of --skip_to_step4 that is no first-order saddle point: the meta-IRC
            // from it (the choice of the start of the IRC), said as such.
            if !points.transition_state {
                let (imaginary, stationary, largest) = points.start.unwrap_or((0, false, f64::NAN));
                let what = match imaginary {
                    0 => "no negative curvature".to_string(),
                    1 => "one negative curvature".to_string(),
                    n => format!("{n} negative curvatures"),
                };
                let gradient = if stationary {
                    "a stationary point".to_string()
                } else {
                    format!("not a stationary point (largest gradient component {largest:.3e})")
                };
                let notice = format!(
                    "AutoTS step 4: the input point is not a first-order saddle point: by the \
                     Hessian of the objective it has {what} (the one criterion, \
                     opt::mode_kill::IMAGINARY_CURVATURE) and it is {gradient}; a meta-IRC (the \
                     descent from the point itself) was computed, not the IRC of a transition \
                     state."
                );
                log.warn(&notice);
                result.notes.push(notice.clone());
                if let Some(t) = result.transition_states.get_mut(k) {
                    t.meta_irc = Some(notice);
                    t.imaginary_modes = Some(imaginary);
                }
            }
            let ts_value = objective.evaluate(x, false)?.value;
            let backward: Vec<Point> = points
                .sides
                .iter()
                .find(|(b, _)| *b == crate::irc::types::Branch::Backward)
                .map(|(_, p)| p.clone())
                .unwrap_or_default();
            // (The one path of a meta-IRC follows its start, as the forward side does.)
            let forward: Vec<Point> = points
                .sides
                .iter()
                .find(|(b, _)| {
                    matches!(
                        b,
                        crate::irc::types::Branch::Forward | crate::irc::types::Branch::Single
                    )
                })
                .map(|(_, p)| p.clone())
                .unwrap_or_default();
            let mut path: Vec<Point> = backward.iter().rev().cloned().collect();
            path.push(Point {
                x: x.clone(),
                value: ts_value,
            });
            path.extend(forward.iter().cloned());
            let (file_name, label) = if points.transition_state {
                (format!("{stem}_ts{}_irc.txt", k + 1), "IRC point")
            } else {
                (format!("{stem}_ts{}_meta_irc.txt", k + 1), "meta-IRC point")
            };
            std::fs::write(work.join(file_name), frames_text(&names, &path, label))?;
            let mut ends = Vec::new();
            for (j, side) in [&backward, &forward].iter().enumerate() {
                let Some(last) = side.last() else {
                    continue;
                };
                let (end, converged) = optimize(objective, &minimum_config, &last.x, None)?;
                let negative = if config.check_frequencies && converged {
                    Some(negative_curvatures(objective, &end.x)?)
                } else {
                    None
                };
                let check =
                    super::autots::EndCheck::of(config.check_frequencies, converged, negative);
                let file = work.join(format!("{stem}_ts{}_end_{}.txt", k + 1, j + 1));
                std::fs::write(&file, variables_text(&names, &end))?;
                match check {
                    super::autots::EndCheck::NotVerified { .. } => {
                        let warning = format!(
                            "AutoTS step 4: end {} of the IRC of transition state {} ({}): {}.",
                            j + 1,
                            k + 1,
                            file.display(),
                            check.describe()
                        );
                        log.warn(&warning);
                        result.notes.push(warning);
                    }
                    _ if !converged => log.warn(&format!(
                        "AutoTS step 4: end {} of the IRC of transition state {} did not converge \
                         to a minimum.",
                        j + 1,
                        k + 1
                    )),
                    _ => {}
                }
                if let Some(t) = result.transition_states.get_mut(k) {
                    t.ends.push((file, end.value));
                    t.end_checks.push(check);
                }
                ends.push(end);
            }
            log.info(&format!(
                "AutoTS step 4: the IRC of transition state {} ({} points) ends at values {}.",
                k + 1,
                path.len(),
                ends.iter()
                    .map(|e| format!("{:.10}", e.value))
                    .collect::<Vec<_>>()
                    .join(" and ")
            ));
            run.ircs.push((k, path, ends));
        }
        tally(result, 3);
    }
    Ok(run)
}

/// A search of the registry on the objective from `points`; its path (or nothing) and its
/// estimates of transition states.
#[allow(clippy::too_many_arguments)]
fn pair_search(
    registry: &PairMethodRegistry,
    method: &str,
    points: &[Col<f64>],
    settings: Vec<(String, String)>,
    objective: &dyn Objective,
    work: &Path,
    stem: &str,
    log: &Logger,
    profiles: Option<&mut Vec<super::autots::BandProfile>>,
) -> MoptResult<(Vec<Point>, Vec<Point>, String)> {
    let dir = work.join(stem);
    std::fs::create_dir_all(&dir)?;
    let search_log = Logger::silent();
    search_log.add_file(dir.join("search.log"), crate::core_infra::log::Level::Debug)?;
    let run = SearchRun {
        save_steps: true,
        dir: &dir,
        stems: [stem.to_string(), stem.to_string()],
        log: &search_log,
        figures: false,
        commands: &super::pair_settings::COMMAND_PARSERS,
    };
    let input = PairInput {
        points: points.to_vec(),
        structures: false,
        ..PairInput::default()
    };
    use crate::path::band::machine::{ChainRelaxation, ChainReport};
    use crate::workflow::chain::{ChainObserver, Flow};
    let mut observer = profiles.map(|profiles| {
        move |band: &ChainReport, _: &mut ChainRelaxation| {
            profiles.push(super::autots::BandProfile {
                run: 1,
                iteration: band.iteration,
                energies: band.energies.clone(),
            });
            Ok(Flow::Continue)
        }
    });
    let r = registry.run_observed(
        method,
        &input,
        &PairSettings::new(settings),
        &VariableSurface(objective),
        &run,
        observer.as_mut().map(|o| o as &mut dyn ChainObserver),
    )?;
    log.info(&format!(
        "AutoTS: {} {} ({} estimate(s); {} gradients, {} Hessians; its log is {})",
        r.method,
        r.end,
        r.transition_states.len(),
        r.counts.gradients,
        r.counts.hessians,
        dir.join("search.log").display()
    ));
    let path = r
        .path
        .unwrap_or_default()
        .into_iter()
        .map(|(x, value)| Point { x, value })
        .collect();
    let estimates = r
        .transition_states
        .into_iter()
        .map(|(x, value)| Point { x, value })
        .collect();
    Ok((path, estimates, r.method))
}

/// [`pair_search`] for the path of step 1 (its estimates are not used: step 2 relaxes it).
#[allow(clippy::too_many_arguments)]
fn pair_path(
    registry: &PairMethodRegistry,
    method: &str,
    points: &[Col<f64>],
    settings: Vec<(String, String)>,
    objective: &dyn Objective,
    work: &Path,
    stem: &str,
    log: &Logger,
) -> MoptResult<Vec<Point>> {
    let (path, estimates, _) = pair_search(
        registry, method, points, settings, objective, work, stem, log, None,
    )?;
    let mut out = path;
    if out.is_empty() {
        if let Some(top) = estimates
            .into_iter()
            .max_by(|a, b| a.value.total_cmp(&b.value))
        {
            out.push(top);
        } else {
            return Err(MoptError::calc(format!(
                "run_autots: {method} left no path or estimate"
            )));
        }
    }
    if let Some(a) = points.first() {
        if out.first().is_none_or(|p| &p.x != a) {
            out.insert(
                0,
                Point {
                    x: a.clone(),
                    value: objective.evaluate(a, false)?.value,
                },
            );
        }
    }
    if let Some(b) = points.last() {
        if out.last().is_none_or(|p| &p.x != b) {
            out.push(Point {
                x: b.clone(),
                value: objective.evaluate(b, false)?.value,
            });
        }
    }
    Ok(out)
}

/// The variables of a point of the input, for the summary of the page.
pub fn point_of(names: &[String], p: &Point) -> VariablePoint {
    VariablePoint {
        values: p.x.clone(),
        names: Arc::new(names.to_vec()),
    }
}

/// `autots_summary.csv` of a run on plain variables: the values of the objective as they are.
pub fn summary_csv(result: &AutotsResult) -> String {
    let mut table = String::from(
        "ts,file,value,converged,negative_curvatures,same_as,end1_value,end2_value,\
         barrier_from_end1,barrier_from_end2,end1_check,end2_check,path\n",
    );
    for (k, ts) in result.transition_states.iter().enumerate() {
        let end = |j: usize| ts.ends.get(j).map(|e| e.1);
        // The start of a meta-IRC is no transition state: no barriers from it.
        let barrier = |j: usize| {
            end(j)
                .filter(|_| ts.meta_irc.is_none())
                .map(|e| ts.energy - e)
        };
        let path = match (&ts.meta_irc, ts.ends.is_empty()) {
            (Some(_), _) => "meta_irc",
            (None, false) => "irc",
            (None, true) => "",
        };
        let opt = |v: Option<f64>| v.map_or_else(String::new, |x| format!("{x:.12}"));
        let check = |j: usize| {
            ts.end_checks
                .get(j)
                .map_or("", super::autots::EndCheck::word)
        };
        let _ = writeln!(
            table,
            "{},{},{:.12},{},{},{},{},{},{},{},{},{},{}",
            k + 1,
            ts.file
                .file_name()
                .map_or_else(String::new, |f| f.to_string_lossy().into_owned()),
            ts.energy,
            ts.converged,
            ts.imaginary_modes
                .map_or_else(String::new, |n| n.to_string()),
            ts.same_as.map_or_else(String::new, |n| n.to_string()),
            opt(end(0)),
            opt(end(1)),
            opt(barrier(0)),
            opt(barrier(1)),
            check(0),
            check(1),
            path
        );
    }
    table
}

/// The files of a run (for the settings of the page).
pub fn files_of(work: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(work)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "txt"))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// The total of the evaluations of every step.
pub fn total(result: &AutotsResult) -> EvaluationCounts {
    result
        .evaluations
        .iter()
        .fold(EvaluationCounts::default(), |a, b| a.plus(b))
}
