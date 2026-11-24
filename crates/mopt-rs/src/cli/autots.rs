//! `run_autots`: MultiOptPy's automated search for transition states (AutoTS, the four steps
//! of `run_autots`), with the commands of mopt-rs as its steps and one calculator registry for
//! all of them.
//!
//! The settings file (`-cfg config.json`, MultiOptPy's format) has `work_dir`,
//! `top_n_candidates`, and one block of settings per step whose keys are the long option names
//! of the command of the step (`step1_settings` .. `step4_settings`); the command line
//! (`-ma`, `-n`, `--skip_step1`, `--run_step4`, `--skip_to_step4`) goes over the file. The
//! steps, with MultiOptPy's file names in `work_dir`:
//!
//! 1. The first path (`step1_settings`), by `step1_method`:
//!    - `afir` (the default, MultiOptPy's step 1): `optmain` from the input with AFIR (`-ma`)
//!      or any other bias potential of `optmain` that drives the reaction; a step 1 without
//!      a bias potential is refused;
//!    - `bitss` (mopt-rs): BITSS between the reactant and the product, the first and the last
//!      structure of the input (`optmain -mf bitss`); the first path is the path of its
//!      optimization as it is, like AFIR's: the first image from the reactant to where the two
//!      met, then the second from there back to the product.
//!
//!    `<stem>_step1_traj.xyz`.
//! 2. The path (`step2_settings`): `nebmain` from the AFIR path (every method of `nebmain`,
//!    `-dmf` included), or BITSS between the two ends of the path (`"step2_method": "bitss"`,
//!    mopt-rs). The candidates for transition states, at most `top_n_candidates`, highest
//!    first: `<stem>_step3_TS_Opt_Inputs/<stem>_ts_guess_<k>.xyz`.
//! 3. Optimization of each candidate to a transition state (`optmain -order 1` with
//!    `step3_settings`): `<stem>_ts_final_<k>.xyz`. mopt-rs adds a vibrational analysis of
//!    every one (`optmain -freq`; `"check_frequencies": false` in the file turns it off), and a
//!    structure that converged with no or several imaginary frequencies is reported and not
//!    taken to step 4; the same transition state found twice goes on once.
//! 4. With `--run_step4`: the IRC of each transition state (`optmain -order 1` with
//!    `step4_settings`, which has `intrinsic_reaction_coordinates`), the two ends optimized to
//!    minima (`step4b_opt_method`, `rsirfo_block_fsb` without it):
//!    `<ts>_IRC_Endpoint_<j>.xyz`, and in `<ts>_results/` the transition state, the optimized
//!    ends (`endpoint_<j>_opt.xyz`), and `energy_profile.txt`.
//!
//! `autots_summary.csv` in `work_dir` lists every transition state with its energy, imaginary
//! frequencies, the energies of the ends of its IRC, and the barriers; `autots_report.html`
//! shows the whole run on one page ([`super::autots_report`]; on by default,
//! `"html_report": false` in the file turns it off). Both are written also when a step fails.
//!
//! MultiOptPy: `Wrapper/autots.py:AutoTSWorkflow` (v1) and `Entrypoints/autots.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::autots_workflow::{joined_trajectories, run_label, StepSettings, Workflow};
use super::nebmain::{self};
use super::nebmain_settings::NebmainConfig;
use super::optmain::{self, OptmainConfig, OptmainResult};
use super::settings::{parse_settings, settings_json, INPUT_KEY, RUN_SETTINGS_FILE};
use super::spec::{NEBMAIN, OPTMAIN, RUN_AUTOTS};
use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{AtomId, SpinCharge, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{read_xyz_file, read_xyz_trajectory_file, write_xyz_file};
use crate::pes::counting::EvaluationCounts;
use crate::pes::registry::CalculatorRegistry;

/// How the first path (step 1) is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstPathMethod {
    /// `optmain` with a bias potential (AFIR, MultiOptPy's step 1).
    Bias,
    /// BITSS between the reactant and the product of the input (mopt-rs).
    Bitss,
    /// A search of `ieipmain` between the reactant and the product of the input, whose
    /// path is the first path (mopt-rs; `cli::autots_pair`).
    Pair(&'static super::autots_pair::PairMethod),
}

/// How the path of step 2 is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathMethod {
    /// `nebmain` from the AFIR path (MultiOptPy's step 2; every method of `nebmain`).
    Chain,
    /// BITSS between the two ends of the AFIR path (mopt-rs).
    Bitss,
}

/// The settings of a run.
#[derive(Debug, Clone)]
pub struct AutotsConfig {
    /// The input structure (the AFIR start; the path with `--skip_step1`; the transition
    /// state with `--skip_to_step4`).
    pub input: PathBuf,
    /// The working directory (`work_dir`, relative to the directory of the run).
    pub work_dir: PathBuf,
    /// Candidates of step 2 that go to step 3.
    pub top_n: usize,
    /// `step1_settings` .. `step4_settings`.
    pub steps: [BTreeMap<String, JsonValue>; 4],
    /// The `workflow` block (several runs of a step, each with its settings); `None`: the
    /// four steps once each.
    pub workflow: Option<Workflow>,
    /// The first path of step 1.
    pub first_method: FirstPathMethod,
    /// The path of step 2.
    pub path_method: PathMethod,
    /// `--skip_step1`.
    pub skip_step1: bool,
    /// `--run_step4`.
    pub run_step4: bool,
    /// `--skip_to_step4`.
    pub skip_to_step4: bool,
    /// The vibrational analysis of the transition states of step 3 (mopt-rs).
    pub check_frequencies: bool,
    /// `autots_report.html`, the page of the whole run (mopt-rs; on by default).
    pub html_report: bool,
    /// The settings of the command line as a settings document (`settings.json` of the run
    /// directory, which `--restart` reads).
    pub settings: JsonValue,
    /// The settings file as it was read (kept in the run directory as `autots_config.json`).
    pub document: JsonValue,
    /// `--restart WORK_DIR`: the run made again in that directory, with the evaluations of the
    /// run before taken from its journals.
    pub restart_dir: Option<PathBuf>,
    /// What the settings say beyond the steps, for the log (the options shared by every step
    /// that a step overrides, the AFIR push of `-ma` over the runs of a workflow).
    pub notes: Vec<String>,
}

/// The settings file kept in the run directory.
const KEPT_CONFIG: &str = "autots_config.json";

fn object(doc: &JsonValue, key: &str) -> MoptResult<BTreeMap<String, JsonValue>> {
    match doc.get(key) {
        None => Ok(BTreeMap::new()),
        Some(v) => v
            .as_object()
            .cloned()
            .ok_or_else(|| MoptError::parse(format!("run_autots: '{key}' must be an object"))),
    }
}

impl AutotsConfig {
    /// The settings of the command line `p` and its settings file.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        let input = p
            .positionals
            .first()
            .map(PathBuf::from)
            .ok_or_else(|| MoptError::parse("run_autots needs an input file"))?;
        let given = p.get_opt_str("config_file");
        let path = PathBuf::from(given.unwrap_or("./config.json"));
        let doc = if path.is_file() {
            JsonValue::parse(&std::fs::read_to_string(&path)?).context(path.display())?
        } else if given.is_some() && given != Some("./config.json") {
            return Err(MoptError::parse(format!(
                "run_autots: no settings file {}",
                path.display()
            )));
        } else {
            JsonValue::Object(BTreeMap::new())
        };
        let mut steps = [
            object(&doc, "step1_settings")?,
            object(&doc, "step2_settings")?,
            object(&doc, "step3_settings")?,
            object(&doc, "step4_settings")?,
        ];
        // The state that the input file names, for the steps whose settings give none (their
        // inputs are files written by the steps before, which have no such line).
        let named = crate::io::xyz::state_of_xyz_file(&input).ok().flatten();
        let with_state = |step: &mut StepSettings| {
            if let Some((charge, multiplicity)) = named {
                step.entry("electronic_charge".to_string())
                    .or_insert(JsonValue::Number(charge as f64));
                if !step.contains_key("unpaired_electrons") {
                    step.entry("spin_multiplicity".to_string())
                        .or_insert(JsonValue::Number(multiplicity as f64));
                }
            }
        };
        steps.iter_mut().for_each(with_state);
        let mut workflow = Workflow::parse(&doc, &with_state)?;
        let mut notes = Vec::new();
        // The constraints and bias potentials shared by every step (the command line over the
        // block of the settings file), merged into every step and every run of a workflow; -ma
        // is the AFIR push of step 1, not a bias of every step.
        let shared = super::mapper_settings::shared_options(p, &doc, &["manual_AFIR"], &mut notes)?;
        if shared.contains_key("manual_AFIR") {
            return Err(MoptError::parse(format!(
                "run_autots: manual_AFIR in {}: the AFIR of run_autots is the push of step 1 \
                 (-ma, or manual_AFIR of step1_settings), not a bias of every step",
                super::mapper_settings::SHARED_BLOCK
            )));
        }
        if shared.contains_key("variable_cell_fix") {
            return Err(MoptError::unsupported(
                "run_autots: -vc_fix (variable_cell_fix) holds the cell of a variable-cell \
                 optimization, and the chain of step 2 (nebmain) has no variable cell; with \
                 constraints the IRC of a variable cell holds none either (optmain -vc -irc)",
            ));
        }
        for (k, step) in steps.iter_mut().enumerate() {
            super::mapper_settings::merge_shared(
                step,
                &shared,
                &format!("step{}_settings", k + 1),
                &mut notes,
            );
        }
        if let Some(w) = workflow.as_mut() {
            for (k, runs) in w.runs.iter_mut().enumerate() {
                let count = runs.len();
                for (r, run) in runs.iter_mut().enumerate() {
                    let name = format!(
                        "the workflow block, step {}{}",
                        k + 1,
                        super::autots_workflow::run_label(r, count)
                    );
                    super::mapper_settings::merge_shared(run, &shared, &name, &mut notes);
                }
            }
        }
        let afir = p.get_list("manual_AFIR");
        if !afir.is_empty() {
            let value =
                JsonValue::Array(afir.iter().map(|s| JsonValue::String(s.clone())).collect());
            steps[0].insert("manual_AFIR".to_string(), value.clone());
            // The command line overrides the settings file: with a workflow block, every run of
            // step 1 is pushed by it.
            if let Some(w) = workflow.as_mut() {
                for run in &mut w.runs[0] {
                    if run
                        .insert("manual_AFIR".to_string(), value.clone())
                        .is_some()
                    {
                        notes.push(
                            "-ma replaces the manual_AFIR of a run of step 1 of the workflow \
                             block."
                                .to_string(),
                        );
                    }
                }
            }
        }
        let top_n = match p.get_opt_str("top_n").filter(|s| !s.is_empty()) {
            Some(text) => text
                .parse()
                .map_err(|_| MoptError::parse(format!("-n: '{text}' is not a number")))?,
            None => doc
                .get("top_n_candidates")
                .and_then(JsonValue::as_f64)
                .map_or(3, |v| v.max(1.0) as usize),
        };
        let path_method = match doc.get("step2_method").and_then(JsonValue::as_str) {
            None | Some("chain" | "neb") => PathMethod::Chain,
            Some("bitss") => PathMethod::Bitss,
            Some(other) => {
                return Err(MoptError::parse(format!(
                    "run_autots: step2_method is chain or bitss, not '{other}'"
                )))
            }
        };
        let first_method = match doc.get("step1_method").and_then(JsonValue::as_str) {
            None | Some("afir" | "bias") => FirstPathMethod::Bias,
            Some("bitss") => FirstPathMethod::Bitss,
            Some(other) => match super::autots_pair::named(other) {
                Some(method) => FirstPathMethod::Pair(method),
                None => {
                    return Err(MoptError::parse(format!(
                        "run_autots: step1_method is afir, bitss, or a search between two \
                         structures ({}), not '{other}'",
                        super::autots_pair::names()
                    )))
                }
            },
        };
        let flag = |key: &str| {
            p.get_bool(key, false) || doc.get(key).and_then(JsonValue::as_bool).unwrap_or(false)
        };
        Ok(Self {
            input,
            work_dir: PathBuf::from(
                doc.get("work_dir")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("autots_workflow"),
            ),
            top_n,
            steps,
            workflow,
            first_method,
            path_method,
            skip_step1: flag("skip_step1"),
            run_step4: flag("run_step4"),
            skip_to_step4: flag("skip_to_step4"),
            check_frequencies: doc
                .get("check_frequencies")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true),
            html_report: doc
                .get("html_report")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true),
            settings: settings_json(p, RUN_AUTOTS),
            document: doc,
            restart_dir: None,
            notes,
        })
    }
}

impl AutotsConfig {
    /// The settings of the runs of step `step` (0: step 1), in their order: those of the
    /// `workflow` block, or the block of the step once. None: the step does not run.
    pub fn runs(&self, step: usize) -> Vec<StepSettings> {
        match &self.workflow {
            Some(workflow) => workflow.runs[step].clone(),
            None => vec![self.steps[step].clone()],
        }
    }
}

/// A transition state of step 3.
#[derive(Debug, Clone)]
pub struct TransitionState {
    /// `<stem>_ts_final_<k>.xyz`.
    pub file: PathBuf,
    /// Its energy (Hartree), without the bias potentials of the surface.
    pub energy: f64,
    /// Its energy with the bias potentials of the surface (Hartree; `energy` without any).
    pub biased_energy: f64,
    /// Whether its optimization converged.
    pub converged: bool,
    /// Its imaginary frequencies, when analysed.
    pub imaginary_modes: Option<usize>,
    /// The input of `--skip_to_step4` that is no first-order saddle point: the notice that its
    /// path is a meta-IRC (it is then never shown as a transition state).
    pub meta_irc: Option<String>,
    /// The number of an earlier transition state it is the same as.
    pub same_as: Option<usize>,
    /// The optimized ends of its IRC with their energies (step 4).
    pub ends: Vec<(PathBuf, f64)>,
    /// What the check of every end of `ends` found, in the same order (step 4).
    pub end_checks: Vec<EndCheck>,
    /// The run directory of its optimization (with the vibrational analysis).
    pub directory: PathBuf,
    /// The run directory of its IRC (step 4).
    pub irc_directory: Option<PathBuf>,
}

/// What the check of an end of an IRC (step 4) found. The ends are optimized to minima with the
/// mode kill of `optmain` (a converged point with an imaginary mode is left along it and
/// optimized on) and then judged by the one criterion of an imaginary mode
/// ([`crate::opt::mode_kill::IMAGINARY_CURVATURE`], with the zero modes projected out as the
/// vibrational analysis does; on plain variables the eigenvalues of the Hessian of the
/// objective with unit masses).
///
/// MultiOptPy: no counterpart (`Entrypoints/autots.py` optimizes the ends without a check).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndCheck {
    /// A minimum: the optimization converged and the Hessian has no imaginary mode.
    Minimum,
    /// Not verified as a minimum: the optimization did not converge, or imaginary modes are
    /// left (their number, where the Hessian was analysed).
    NotVerified {
        /// Whether the optimization converged.
        converged: bool,
        /// The imaginary modes of its last structure, where they were counted.
        imaginary_modes: Option<usize>,
    },
    /// Not checked: `check_frequencies` is off.
    NotChecked,
}

impl EndCheck {
    /// The check of an end from the end of its optimization (`check`: whether the run checks
    /// the ends).
    pub fn of(check: bool, converged: bool, imaginary_modes: Option<usize>) -> Self {
        match (check, converged, imaginary_modes) {
            (false, _, _) => Self::NotChecked,
            (true, true, Some(0)) => Self::Minimum,
            (true, converged, imaginary_modes) => Self::NotVerified {
                converged,
                imaginary_modes,
            },
        }
    }

    /// In words, for the log, the summary and the report.
    pub fn describe(&self) -> String {
        match self {
            Self::Minimum => "minimum (no imaginary mode)".to_string(),
            Self::NotChecked => "not checked (check_frequencies is off)".to_string(),
            Self::NotVerified {
                converged,
                imaginary_modes,
            } => {
                let modes = match imaginary_modes {
                    Some(n) => format!("{n} imaginary mode(s)"),
                    None => "no vibrational analysis".to_string(),
                };
                format!(
                    "end not verified: {}, {modes}",
                    if *converged {
                        "converged"
                    } else {
                        "the optimization did not converge"
                    }
                )
            }
        }
    }

    /// A word for the summary table: `minimum`, `not_verified` or `not_checked`.
    pub fn word(&self) -> &'static str {
        match self {
            Self::Minimum => "minimum",
            Self::NotVerified { .. } => "not_verified",
            Self::NotChecked => "not_checked",
        }
    }
}

/// What step 2 made, whatever its method: the report draws it from this.
#[derive(Debug, Clone, Default)]
pub struct PathStage {
    /// The method (`chain (nebmain)`, `direct MaxFlux (nebmain -dmf)`, `BITSS`, ...).
    pub method: String,
    /// The path it made, as frames (the last chain, the string, the trajectory of the two
    /// images of BITSS), when there is one.
    pub path: Option<PathBuf>,
    /// The energies of the frames of `path` (Hartree), when they are known.
    pub energies: Vec<f64>,
    /// The frames of `path` that became candidates.
    pub candidates: Vec<usize>,
    /// Its estimates of transition states (structure files), highest first.
    pub estimates: Vec<PathBuf>,
}

/// Image energies of one evaluated NEB or direct MaxFlux band, in Hartree.
#[derive(Debug, Clone)]
pub struct BandProfile {
    /// One-based Step 2 run in a workflow.
    pub run: usize,
    /// Iteration number reported by the path method.
    pub iteration: usize,
    /// Energies in image order.
    pub energies: Vec<f64>,
}

/// The result of a run.
#[derive(Debug, Clone, Default)]
pub struct AutotsResult {
    /// The working directory.
    pub directory: PathBuf,
    /// Settings resolved at dispatch, grouped by stage and run.
    pub effective_settings: BTreeMap<String, String>,
    /// The first path (step 1, or the input with `--skip_step1`).
    pub afir_path: Option<PathBuf>,
    /// The run directory of the optimization of step 1.
    pub afir_directory: Option<PathBuf>,
    /// What made the first path: the bias potentials of step 1 (`manual_AFIR` for AFIR), or
    /// `given path` with `--skip_step1`.
    pub first_path_method: String,
    /// What step 2 made.
    pub path_stage: Option<PathStage>,
    /// Evaluated bands in calculation order, including the initial path of each run.
    pub band_profiles: Vec<BandProfile>,
    /// The candidates of step 2.
    pub candidates: Vec<PathBuf>,
    /// The transition states of step 3 (or the input of `--skip_to_step4`).
    pub transition_states: Vec<TransitionState>,
    /// What the run warns about as a whole (steps 3 and 4 on different surfaces), for
    /// the report.
    pub notes: Vec<String>,
    /// The hard constraints and the bias potentials of the surface of the transition-state
    /// search (step 3) in words, for the report (empty where it has none).
    pub surface: Vec<String>,
    /// The error the run stopped with, if it did (the report shows what was done before
    /// it and says where it stopped).
    pub error: Option<String>,
    /// While the run is going on: what it is doing (the report and the summary are written
    /// again after every step, so that a run that is stopped leaves what it had done);
    /// `None` when it has ended.
    pub progress: Option<String>,
    /// The evaluations of every step (steps 1 to 4), microiterations included.
    pub evaluations: [EvaluationCounts; 4],
    /// The counts of the registry when they were last given to a step, and the step that
    /// counts now (0: step 1).
    pub(crate) counted: (EvaluationCounts, usize),
}

/// Give the evaluations since the last time to the step that counts now, and let step
/// `next` (0: step 1) count from here.
fn tally(result: &mut AutotsResult, now: EvaluationCounts, next: usize) {
    let (last, step) = result.counted;
    let k = step.min(3);
    result.evaluations[k] = result.evaluations[k].plus(&now.since(&last));
    result.counted = (now, next.min(3));
}

/// The evaluations of every step in one line of the log.
fn evaluations_line(result: &AutotsResult) -> String {
    let all = result
        .evaluations
        .iter()
        .fold(EvaluationCounts::default(), |a, b| a.plus(b));
    let mut parts: Vec<String> = result
        .evaluations
        .iter()
        .enumerate()
        .map(|(k, e)| {
            format!(
                "step {}: {} gradient(s), {} Hessian(s){}",
                k + 1,
                e.gradients,
                e.hessians,
                if e.micro_relaxations > 0 {
                    format!(", {} microiteration(s)", e.micro_iterations)
                } else {
                    String::new()
                }
            )
        })
        .collect();
    parts.push(all.line("the whole run"));
    format!("AutoTS: evaluations by step: {}", parts.join("; "))
}

/// Write `content` to `path` through a file next to it, so that a run stopped while it
/// writes leaves the last whole file.
fn write_whole(path: &Path, content: &[u8]) -> MoptResult<()> {
    let part = path.with_extension("part");
    std::fs::write(&part, content)?;
    std::fs::rename(&part, path)?;
    Ok(())
}

/// [`run_with`] on the objective of plain variables `objective` (with `options`).
fn run_variables(
    config: &AutotsConfig,
    work: &Path,
    input: &Path,
    log: &Logger,
    objective: &str,
    options: &crate::pes::registry::CalculatorOptions,
    mut result: AutotsResult,
) -> MoptResult<AutotsResult> {
    let outcome =
        super::autots_variables::run(config, work, input, log, objective, options, &mut result);
    let run = match &outcome {
        Ok(run) => run.clone(),
        Err(e) => {
            result.error = Some(e.to_string());
            super::autots_variables::VariablesRun::default()
        }
    };
    log.info(&evaluations_line(&result));
    write_whole(
        &work.join("autots_summary.csv"),
        super::autots_variables::summary_csv(&result).as_bytes(),
    )?;
    write_effective_settings(&result)?;
    if config.html_report {
        // An objective of its own for the surface of the page: its values are not counted.
        let drawn = crate::pes::objective::ObjectiveRegistry::builtin()
            .create(objective, options)
            .ok();
        let report = work.join("autots_report.html");
        match super::autots_variables_report::write(
            config,
            &result,
            &run,
            drawn.as_deref(),
            objective,
            &report,
        ) {
            Ok(()) => log.info(&format!("AutoTS: the report is {}.", report.display())),
            Err(e) => log.warn(&format!("AutoTS: the report could not be written: {e}")),
        }
    }
    outcome?;
    log.info(&format!(
        "AutoTS: {} transition state(s); the summary is {}.",
        result.transition_states.len(),
        work.join("autots_summary.csv").display()
    ));
    Ok(result)
}

fn write_effective_settings(result: &AutotsResult) -> MoptResult<()> {
    write_whole(
        &result.directory.join("effective_settings.txt"),
        super::autots_settings::text(&result.effective_settings).as_bytes(),
    )
}

/// The summary (`autots_summary.csv`) and the report (`autots_report.html`) of what the
/// run has so far; the path of the report when it was written.
fn write_outputs(config: &AutotsConfig, result: &AutotsResult) -> MoptResult<Option<PathBuf>> {
    let work = &result.directory;
    write_effective_settings(result)?;
    let mut table = String::from(
        "ts,file,energy_hartree,converged,imaginary_modes,same_as,end1_hartree,end2_hartree,\
         barrier_from_end1_kcal_mol,barrier_from_end2_kcal_mol,end1_check,end2_check,path\n",
    );
    for (k, ts) in result.transition_states.iter().enumerate() {
        let end = |j: usize| ts.ends.get(j).map(|e| e.1);
        // The start of a meta-IRC is no transition state: no barriers from it.
        let barrier = |j: usize| {
            end(j)
                .filter(|_| ts.meta_irc.is_none())
                .map(|e| Units::HARTREE_TO_KCAL_MOL * (ts.energy - e))
        };
        let path = match (&ts.meta_irc, &ts.irc_directory) {
            (Some(_), _) => "meta_irc",
            (None, Some(_)) => "irc",
            (None, None) => "",
        };
        let opt =
            |v: Option<f64>, digits: usize| v.map_or_else(String::new, |x| format!("{x:.digits$}"));
        let check = |j: usize| ts.end_checks.get(j).map_or("", EndCheck::word);
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
            opt(end(0), 12),
            opt(end(1), 12),
            opt(barrier(0), 4),
            opt(barrier(1), 4),
            check(0),
            check(1),
            path
        );
    }
    write_whole(&work.join("autots_summary.csv"), table.as_bytes())?;
    if !config.html_report {
        return Ok(None);
    }
    let report = work.join("autots_report.html");
    let part = work.join("autots_report.html.part");
    super::autots_report::write(config, result, &part)?;
    std::fs::rename(&part, &report)?;
    Ok(Some(report))
}

/// The report of the run so far, with `stage` as what the run is doing now. A report that
/// cannot be written is said and does not stop the run.
fn snapshot(config: &AutotsConfig, result: &mut AutotsResult, log: &Logger, stage: String) {
    result.progress = Some(stage);
    if let Err(e) = write_outputs(config, result) {
        log.warn(&format!(
            "AutoTS: the report of the run so far could not be written: {e}"
        ));
    }
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned())
}

/// The settings of a step for one run of a command: the step's block with `extra` over it,
/// without the keys of `drop`, with `inputs`.
pub(super) fn step_args(
    step: &BTreeMap<String, JsonValue>,
    inputs: &[&Path],
    extra: &[(&str, JsonValue)],
    drop: &[&str],
    options: &[super::option::CliOption],
) -> MoptResult<ParsedArgs> {
    let mut m = step.clone();
    for d in drop {
        m.remove(*d);
    }
    for (k, v) in extra {
        m.insert((*k).to_string(), v.clone());
    }
    m.insert(
        INPUT_KEY.to_string(),
        JsonValue::Array(
            inputs
                .iter()
                .map(|p| JsonValue::String(p.to_string_lossy().into_owned()))
                .collect(),
        ),
    );
    parse_settings(&JsonValue::Object(m).to_string_compact(), options)
}

fn text(v: &str) -> JsonValue {
    JsonValue::String(v.to_string())
}

/// The `optmain` settings of the step block `step` for the input `input`, as a step of the run
/// reads them (with the options shared by every step merged in by [`AutotsConfig::from_args`]).
pub fn step_optmain_config(step: &StepSettings, input: &Path) -> MoptResult<OptmainConfig> {
    OptmainConfig::from_args(&step_args(step, &[input], &[], &[], OPTMAIN)?)
}

fn one(runs: Vec<OptmainResult>, what: &str) -> MoptResult<OptmainResult> {
    runs.into_iter()
        .next()
        .ok_or_else(|| MoptError::calc(format!("run_autots: {what} gave no result")))
}

/// Export the display trajectory and return the full coordinates needed by later stages.
fn export_step_trajectory(summary: &optmain::JobSummary, target: PathBuf) -> MoptResult<PathBuf> {
    let stem = summary
        .input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let display = summary
        .trajectory
        .with_file_name(format!("{stem}_traj.xyz"));
    std::fs::copy(&display, &target)?;
    Ok(if display == summary.trajectory {
        target
    } else {
        summary.trajectory.clone()
    })
}

/// The largest change of an interatomic distance between two structures of the same atoms
/// (Bohr; distances do not depend on rotation), or `None` when the atoms differ.
pub(crate) fn largest_distance_change(a: &Path, b: &Path) -> MoptResult<Option<f64>> {
    let sa = read_xyz_file(a, SpinCharge::default(), None)?;
    let sb = read_xyz_file(b, SpinCharge::default(), None)?;
    if sa.elements != sb.elements {
        return Ok(None);
    }
    let n = sa.n_atoms();
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..i {
            let da = sa.distance(AtomId::new(i), AtomId::new(j)).value();
            let db = sb.distance(AtomId::new(i), AtomId::new(j)).value();
            worst = worst.max((da - db).abs());
        }
    }
    Ok(Some(worst))
}

/// Two transition states are the same when no interatomic distance differs by 0.01 Bohr.
fn same_structure(a: &Path, b: &Path) -> MoptResult<bool> {
    Ok(largest_distance_change(a, b)?.is_some_and(|d| d < 1.0e-2))
}

/// The file of a run directory with the structure the built-in GFN-FF of the run sets its
/// topology up from (`-calc_opt topology=FILE` of a command that follows the run).
pub const GFNFF_TOPOLOGY_FILE: &str = "gfnff_topology.xyz";

/// Whether the settings of a step choose the built-in GFN-FF.
fn uses_gfnff(step: &StepSettings) -> bool {
    step.get("othersoft")
        .and_then(JsonValue::as_str)
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("gfnff"))
}

/// Whether the settings of a step give the topology of GFN-FF (`-calc_opt topology=FILE`).
fn has_topology(step: &StepSettings) -> bool {
    match step.get("calc_option") {
        Some(JsonValue::Array(items)) => items.iter().any(|v| {
            v.as_str()
                .is_some_and(|s| s.trim().starts_with("topology="))
        }),
        Some(JsonValue::String(s)) => s.trim().starts_with("topology="),
        _ => false,
    }
}

/// `config` with the topology of the built-in GFN-FF of every step set up from the first frame
/// of `input`, written to [`GFNFF_TOPOLOGY_FILE`] of `work` (`None`: no step uses GFN-FF, or
/// every one that does gives its topology itself).
fn gfnff_reference(
    config: &AutotsConfig,
    input: &Path,
    work: &Path,
) -> MoptResult<Option<AutotsConfig>> {
    let mut all: Vec<&StepSettings> = config.steps.iter().collect();
    if let Some(w) = &config.workflow {
        all.extend(w.runs.iter().flatten());
    }
    if !all.iter().any(|s| uses_gfnff(s) && !has_topology(s)) {
        return Ok(None);
    }
    let frames = read_xyz_trajectory_file(input, SpinCharge::default(), None)?;
    let first = frames
        .frames
        .first()
        .ok_or_else(|| MoptError::parse("run_autots: the input has no structure"))?;
    let file = work.join(GFNFF_TOPOLOGY_FILE);
    write_xyz_file(&file, &first.system, None)?;
    let value = JsonValue::String(format!("topology={}", file.display()));
    let add = |step: &mut StepSettings| {
        if !uses_gfnff(step) || has_topology(step) {
            return;
        }
        let entry = step
            .entry("calc_option".to_string())
            .or_insert_with(|| JsonValue::Array(Vec::new()));
        match entry {
            JsonValue::Array(items) => items.push(value.clone()),
            other => *other = JsonValue::Array(vec![other.clone(), value.clone()]),
        }
    };
    let mut out = config.clone();
    out.steps.iter_mut().for_each(add);
    if let Some(w) = out.workflow.as_mut() {
        w.runs.iter_mut().flatten().for_each(add);
    }
    out.notes.push(format!(
        "GFN-FF: every step takes its topology from the input of the run ({}; -calc_opt \
         topology={} gives a command that starts from a result of this run the same force \
         field).",
        file.display(),
        file.display()
    ));
    Ok(Some(out))
}

/// The constraints of a step block with their values made explicit at `input` (the input of
/// the run): `-pc NAME ATOMS` as `-pc manual NAME ATOMS VALUE`, the rigid groups of `-gfix` as
/// the distances of their pairs, `-uc EXPR` as `-uc EXPR = VALUE`; `-fix` stays (the atoms keep
/// the places of the structures every step starts from, which the steps before held). Every
/// step, and every image of the chain of step 2 (which holds the values of its own images
/// otherwise), then holds the values of the input of the run.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
fn explicit_constraints(
    step: &mut StepSettings,
    input: &crate::domain::MolecularSystem,
) -> MoptResult<()> {
    use super::constraints::{projection_constraint, user_constraint};
    let strings = |v: &JsonValue| -> Vec<String> {
        match v {
            JsonValue::Array(items) => items
                .iter()
                .map(|x| match x {
                    JsonValue::String(s) => s.clone(),
                    other => other.to_string_compact(),
                })
                .collect(),
            JsonValue::String(s) => vec![s.clone()],
            other => vec![other.to_string_compact()],
        }
    };
    let mut manual: Vec<String> = Vec::new();
    if let Some(v) = step.get("projection_constrain") {
        let tokens = strings(v);
        if tokens
            .first()
            .is_some_and(|t| t.eq_ignore_ascii_case("manual"))
        {
            manual.extend(tokens);
        } else {
            for pair in tokens.chunks(2) {
                let [name, atoms] = pair else {
                    return Err(MoptError::parse("-pc expects pairs: name atoms"));
                };
                let c = projection_constraint(name, atoms, None)?.resolve(input, None)?;
                let Some(value) = c.target else {
                    // Kinds without a value (rot, eigvec, atoms_pair) stay as given.
                    manual.extend([String::new(), name.clone(), atoms.clone()]);
                    continue;
                };
                let shown = match name.to_ascii_lowercase().as_str() {
                    "angle" | "dihedral" => value.to_degrees(),
                    _ => value * Units::BOHR_TO_ANGSTROM,
                };
                manual.extend([
                    "manual".to_string(),
                    name.clone(),
                    atoms.clone(),
                    format!("{shown:.17e}"),
                ]);
            }
        }
    }
    let rigid_groups = step.get("gradient_fix_atoms").cloned();
    if let Some(v) = &rigid_groups {
        for group in strings(v) {
            let expanded = crate::core_infra::arg_parser::parse_atom_indices_one_based(&group)?;
            for i in 0..expanded.len() {
                for j in i + 1..expanded.len() {
                    let pair = format!("{},{}", expanded[i], expanded[j]);
                    let c = projection_constraint("bond", &pair, None)?.resolve(input, None)?;
                    let value = c.target.unwrap_or_default() * Units::BOHR_TO_ANGSTROM;
                    manual.extend([
                        "manual".to_string(),
                        "bond".to_string(),
                        pair,
                        format!("{value:.17e}"),
                    ]);
                }
            }
        }
    }
    // A kind without a value stays as given; then the manual ones cannot share the option.
    if manual.iter().any(String::is_empty) {
        return Ok(());
    }
    if rigid_groups.is_some() {
        step.remove("gradient_fix_atoms");
    }
    if !manual.is_empty() {
        step.insert(
            "projection_constrain".to_string(),
            JsonValue::Array(manual.into_iter().map(JsonValue::String).collect()),
        );
    }
    if let Some(v) = step.get("user_constraint").cloned() {
        let mut out = Vec::new();
        for text in strings(&v) {
            let (body, rest) = match text.split_once(';') {
                Some((b, r)) => (b.to_string(), format!(";{r}")),
                None => (text.clone(), String::new()),
            };
            if body.contains('=') {
                out.push(text);
                continue;
            }
            let c = user_constraint(&text)?.resolve(input, None)?;
            match c.target {
                Some(value) => out.push(format!("{body} = {value:.17e}{rest}")),
                None => out.push(text),
            }
        }
        step.insert(
            "user_constraint".to_string(),
            JsonValue::Array(out.into_iter().map(JsonValue::String).collect()),
        );
    }
    Ok(())
}

/// What the IRC of the input of `--skip_to_step4` gave.
struct GivenPointIrc {
    /// Its run directory.
    directory: PathBuf,
    /// The ends of its path (two from a transition state, one from a meta-IRC).
    endpoints: Vec<PathBuf>,
    /// The energy of the input, without and with the bias potentials (Hartree).
    energy: f64,
    biased_energy: f64,
    /// Its imaginary modes by the start of the IRC (the one criterion).
    imaginary: usize,
    /// The notice that the input is no first-order saddle point and a meta-IRC was computed.
    meta_irc: Option<String>,
}

/// The IRC of the input of `--skip_to_step4` from the input itself (`ircmain` with the settings
/// of step 4): its start computes the exact Hessian of the input (or takes the one that
/// `initial_hessian` hands in) and decides, in one place (`irc::start`), between the IRC of a
/// transition state (exactly one imaginary mode by the one criterion, and a gradient within
/// the criteria of step 4) and a meta-IRC (the descent from the point itself); the latter comes
/// with a notice of what the input is.
///
/// MultiOptPy: `Entrypoints/autots.py` (`--skip_to_step4` runs the IRC of the input; MultiOptPy's
/// IRC makes the same choice in `irc.py:IRC.saddle_check`).
fn given_point_irc(
    step4: &StepSettings,
    file: &Path,
    work: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    log: &Logger,
) -> MoptResult<GivenPointIrc> {
    let skipped = [
        "step4b_opt_method",
        "saddle_order",
        "mode_kill",
        "mode_kill_method",
        "frequency_analysis",
        "force_frequency_analysis",
    ];
    let surface: StepSettings = step4
        .iter()
        .filter(|(k, _)| {
            !skipped.contains(&k.as_str())
                && super::spec::IRCMAIN
                    .iter()
                    .any(|o| o.spec.key == k.as_str())
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    match surface.get("initial_hessian") {
        Some(v) => log.info(&format!(
            "AutoTS step 4: the start of the IRC of {} is judged by the Hessian handed in ({}).",
            file.display(),
            v.as_str()
                .map_or_else(|| v.to_string_compact(), str::to_string)
        )),
        None => log.info(&format!(
            "AutoTS step 4: the exact Hessian of the input {} is computed at the start of its \
             IRC, which judges whether it is a first-order saddle point (--skip_to_step4 gives \
             no analysis of it).",
            file.display()
        )),
    }
    // A structure converged by the criteria of step 4 is a stationary start.
    let criteria = OptmainConfig::from_args(&step_args(
        step4,
        &[file],
        &[],
        &["step4b_opt_method"],
        OPTMAIN,
    )?)?
    .settings
    .criteria;
    let mut config = super::ircmain::IrcmainConfig::from_args(&step_args(
        &surface,
        &[file],
        &[],
        &[],
        super::spec::IRCMAIN,
    )?)?;
    config.irc = config
        .irc
        .clone()
        .accepting_converged(criteria.max_force.value(), criteria.rms_force.value());
    let mut start: Option<crate::irc::StartReport> = None;
    let runs = {
        let mut observer = crate::workflow::irc::FnObserver::new(|event: &crate::irc::IrcEvent| {
            if let crate::irc::IrcEvent::Start(report) = event {
                start = Some(report.clone());
            }
            Ok(true)
        });
        super::ircmain::run_with_observers(&config, work, console, registry, &mut [&mut observer])?
    };
    let run = runs
        .into_iter()
        .next()
        .ok_or_else(|| MoptError::calc("run_autots: step 4 (IRC) gave no result"))?;
    let report = start
        .ok_or_else(|| MoptError::calc("run_autots: the IRC of the input reported no start"))?;
    let meta_irc = (report.kind != crate::irc::StartKind::TransitionState).then(|| {
        let what = match report.imaginary {
            0 => "no imaginary mode".to_string(),
            1 => "one imaginary mode".to_string(),
            n => format!("{n} imaginary modes"),
        };
        let gradient = if report.stationary {
            "a stationary point".to_string()
        } else {
            format!(
                "not a stationary point (largest gradient component {:.3e} Hartree/Bohr, RMS {:.3e})",
                report.max_gradient, report.rms_gradient
            )
        };
        format!(
            "AutoTS step 4: the input {} is not a first-order saddle point: by its exact Hessian it \
             has {what} (the one criterion, opt::mode_kill::IMAGINARY_CURVATURE) and it is \
             {gradient}; a meta-IRC (the descent from the point itself) was computed, not the IRC \
             of a transition state.",
            file.display()
        )
    });
    if let Some(notice) = &meta_irc {
        log.warn(notice);
    } else {
        log.info(&format!(
            "AutoTS step 4: the input {} is a first-order saddle point by its exact Hessian.",
            file.display()
        ));
    }
    Ok(GivenPointIrc {
        directory: run.irc.directory.clone(),
        endpoints: run.irc.endpoints.clone(),
        energy: report.energy,
        biased_energy: report.biased_energy,
        imaginary: report.imaginary,
        meta_irc,
    })
}

/// The ends `endpoints` of the IRC of `ts` (the `k`-th transition state) optimized to minima and
/// checked ([`EndCheck`]); their files in `<ts>_results` with their energies.
#[allow(clippy::too_many_arguments)]
fn optimize_ends(
    config: &AutotsConfig,
    step4: &StepSettings,
    end_method: &JsonValue,
    ts: &Path,
    endpoints: &[PathBuf],
    k: usize,
    work: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    log: &Logger,
    result: &mut AutotsResult,
    meta_irc: bool,
) -> MoptResult<(Vec<(PathBuf, f64)>, Vec<EndCheck>)> {
    let ts_base = stem(ts);
    let results_dir = work.join(format!("{ts_base}_results"));
    std::fs::create_dir_all(&results_dir)?;
    std::fs::copy(ts, results_dir.join(format!("{ts_base}_ts_final.xyz")))?;
    let mut ends = Vec::new();
    let mut checks = Vec::new();
    for (j, end) in endpoints.iter().enumerate() {
        let path = if meta_irc { "meta_IRC" } else { "IRC" };
        let local = work.join(format!("{ts_base}_{path}_Endpoint_{}.xyz", j + 1));
        std::fs::copy(end, &local)?;
        // An IRC can end on a saddle point (a symmetric transition state keeps
        // its symmetry down to a symmetric stationary point of the complex): with
        // check_frequencies on (the default) every end is optimized with the mode
        // kill of optmain, which leaves a converged point along an imaginary mode
        // and optimizes on, and its last structure is judged by the vibrational
        // analysis of optmain (the one criterion of an imaginary mode). The mode
        // kill goes on with the optimizer of the ends (`classic`): its HiSD
        // continuation, a gradient descent at order 0, did not reach the minimum of
        // the symmetric SN2 complex in 1000 iterations (a well of 0.053 kcal/mol),
        // which RS-I-RFO reached in 59 (`tests/cli_autots_irc_ends.rs`). Keys of
        // step4_settings win.
        let mut more = vec![
            ("saddle_order", text("0")),
            ("opt_method", end_method.clone()),
        ];
        if config.check_frequencies {
            for (key, value) in [
                ("mode_kill", JsonValue::Bool(true)),
                ("mode_kill_method", text("classic")),
                ("frequency_analysis", JsonValue::Bool(true)),
            ] {
                if !step4.contains_key(key) {
                    more.push((key, value));
                }
            }
        }
        let args = step_args(
            step4,
            &[&local],
            &more,
            // (A Hessian handed in is that of the transition state.)
            &[
                "step4b_opt_method",
                "intrinsic_reaction_coordinates",
                "initial_hessian",
            ],
            OPTMAIN,
        )?;
        let minimum = OptmainConfig::from_args(&args)?;
        result.effective_settings.insert(
            format!("Step 4 · IRC {} endpoint {}", k + 1, j + 1),
            super::autots_settings::optimization(&minimum),
        );
        write_effective_settings(result)?;
        let opt = one(
            optmain::run_with(&minimum, work, console, registry)?,
            "step 4 (the ends)",
        )?;
        let energy = opt.summary.as_ref().map_or(f64::NAN, |s| s.energy.value());
        let file = results_dir.join(format!("endpoint_{}_opt.xyz", j + 1));
        std::fs::copy(&opt.optimized, &file)?;
        let check = EndCheck::of(
            config.check_frequencies,
            opt.converged,
            opt.summary.as_ref().and_then(|s| s.imaginary_modes),
        );
        if let EndCheck::NotVerified { .. } = check {
            let warning = format!(
                "AutoTS step 4: end {} of the IRC of transition state {} ({}): {}.",
                j + 1,
                k + 1,
                file.display(),
                check.describe()
            );
            log.warn(&warning);
            result.notes.push(warning);
        } else {
            log.info(&format!(
                "AutoTS step 4: end {} of the IRC of transition state {}: {}.",
                j + 1,
                k + 1,
                check.describe()
            ));
        }
        ends.push((file, energy));
        checks.push(check);
    }
    Ok((ends, checks))
}

/// `energy_profile.txt` of `ts` in `<ts>_results`: its energy and those of the ends (with the
/// barriers from them where `ts` is a transition state, not the start of a meta-IRC).
fn write_profile(
    work: &Path,
    ts: &Path,
    ts_energy: f64,
    ends: &[(PathBuf, f64)],
    checks: &[EndCheck],
    meta_irc: bool,
) -> MoptResult<()> {
    let ts_base = stem(ts);
    let results_dir = work.join(format!("{ts_base}_results"));
    let mut profile = if meta_irc {
        format!(
            "# {ts_base}: a meta-IRC from a point that is not a first-order saddle point; \
             energies (Hartree)\nStart {ts_energy:.12}\n"
        )
    } else {
        format!(
            "# {ts_base}: energies (Hartree) and relative to each end (kcal/mol)\nTS    {ts_energy:.12}\n"
        )
    };
    for (j, (_, e)) in ends.iter().enumerate() {
        let check = checks.get(j).map_or_else(String::new, EndCheck::describe);
        let _ = if meta_irc {
            writeln!(profile, "End{} {e:.12}  {check}", j + 1)
        } else {
            writeln!(
                profile,
                "End{} {e:.12}  barrier from it {:.4}  {check}",
                j + 1,
                Units::HARTREE_TO_KCAL_MOL * (ts_energy - e)
            )
        };
    }
    std::fs::write(results_dir.join("energy_profile.txt"), profile)?;
    Ok(())
}

/// Run AutoTS under `base` with the calculators of `registry`.
pub fn run_with(
    config: &AutotsConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<AutotsResult> {
    let work = match &config.restart_dir {
        Some(dir) => std::path::absolute(dir)?,
        None => std::path::absolute(base.join(&config.work_dir))?,
    };
    std::fs::create_dir_all(&work)?;
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(work.join("autots.log"), Level::Info)?;
    let name = stem(&config.input);
    let input = work.join(
        config
            .input
            .file_name()
            .ok_or_else(|| MoptError::parse("run_autots: the input is not a file"))?,
    );
    if std::path::absolute(&config.input)? != input {
        std::fs::copy(&config.input, &input).map_err(|e| {
            MoptError::config(format!("run_autots: {}: {e}", config.input.display()))
        })?;
    }
    // One force field across the steps: the built-in GFN-FF of every step takes its topology
    // from the input of the run, written next to the results, instead of from the input of
    // each step (a transition state has other bonds than the reactant).
    let with_topology;
    let config = match gfnff_reference(config, &input, &work)? {
        Some(c) => {
            with_topology = c;
            &with_topology
        }
        None => config,
    };
    // The constraints of every step held at the values of the input of the run.
    let explicit;
    let config = {
        let mut c = config.clone();
        // (The points of an objective of plain variables have no constraints of structures.)
        let keys = [
            "projection_constrain",
            "gradient_fix_atoms",
            "user_constraint",
        ];
        let held = config
            .steps
            .iter()
            .chain(config.workflow.iter().flat_map(|w| w.runs.iter().flatten()))
            .any(|s| keys.iter().any(|k| s.contains_key(*k)));
        let first = if !held || super::autots_variables::objective_of(config).is_some() {
            None
        } else {
            read_xyz_trajectory_file(&input, SpinCharge::default(), None)?
                .frames
                .into_iter()
                .next()
                .map(|f| f.system)
        };
        if let Some(first) = first {
            for step in c.steps.iter_mut() {
                explicit_constraints(step, &first)?;
            }
            if let Some(w) = c.workflow.as_mut() {
                for run in w.runs.iter_mut().flatten() {
                    explicit_constraints(run, &first)?;
                }
            }
        }
        explicit = c;
        &explicit
    };
    // The settings of the run, kept so that `run_autots --restart WORK_DIR` makes it again:
    // the settings file as it was read and the command line, with the copy of the input.
    let kept = work.join(KEPT_CONFIG);
    write_whole(
        &kept,
        (config.document.to_string_pretty() + "\n").as_bytes(),
    )?;
    let mut settings = config.settings.clone();
    if let JsonValue::Object(map) = &mut settings {
        map.insert(
            INPUT_KEY.to_string(),
            JsonValue::Array(vec![JsonValue::String(
                input.to_string_lossy().into_owned(),
            )]),
        );
        map.insert(
            "config_file".to_string(),
            JsonValue::String(kept.to_string_lossy().into_owned()),
        );
    }
    write_whole(
        &work.join(RUN_SETTINGS_FILE),
        (settings.to_string_pretty() + "\n").as_bytes(),
    )?;
    // A run made again: what the run before evaluated is taken where its steps ask again.
    let pool = match &config.restart_dir {
        Some(_) => Some(Arc::new(
            crate::workflow::journal::ReplayPool::from_directory(&work)?,
        )),
        None => None,
    };
    let with_pool;
    let registry = match &pool {
        Some(p) => {
            let (offered, _) = crate::pes::replay::ReplaySource::counts(p.as_ref());
            log.info(&format!(
                "AutoTS: the run in {} made again: {offered} evaluation(s) of the run before \
                 are taken where its steps ask for them again, and the steps go on from where \
                 it stopped.",
                work.display()
            ));
            with_pool = registry
                .clone()
                .with_replay(p.clone() as Arc<dyn crate::pes::replay::ReplaySource>);
            &with_pool
        }
        None => registry,
    };
    let mut result = AutotsResult {
        directory: work.clone(),
        effective_settings: BTreeMap::from([(
            "AutoTS".into(),
            super::autots_settings::workflow(
                config,
                super::autots_variables::objective_of(config).is_some(),
            ),
        )]),
        afir_path: None,
        afir_directory: None,
        first_path_method: String::new(),
        path_stage: None,
        band_profiles: Vec::new(),
        candidates: Vec::new(),
        transition_states: Vec::new(),
        notes: Vec::new(),
        surface: Vec::new(),
        error: None,
        progress: None,
        evaluations: [EvaluationCounts::default(); 4],
        counted: (registry.counts(), 0),
    };
    // Steps 3 and 4 on different surfaces: a transition state of step 3 is not a
    // stationary point of the surface of step 4.
    if config.run_step4 && !config.skip_to_step4 {
        if let Some(step4) = config.runs(3).first() {
            let surface = |settings: &StepSettings, drop: &[&str]| {
                step_args(settings, &[&input], &[], drop, OPTMAIN)
                    .and_then(|args| OptmainConfig::from_args(&args))
            };
            let runs3 = config.runs(2);
            for (r, step3) in runs3.iter().enumerate() {
                let (Ok(third), Ok(fourth)) =
                    (surface(step3, &[]), surface(step4, &["step4b_opt_method"]))
                else {
                    // Settings that cannot be read are reported by the step that uses them.
                    continue;
                };
                let differences = super::autots_pair::surface_differences(&third, &fourth);
                if !differences.is_empty() {
                    let note = format!(
                        "Steps 3{} and 4 do not run on the same surface: they differ in {}. A \
                         transition state of step 3 is not a stationary point of the surface of \
                         step 4, so its IRC starts with a gradient and its barriers mix two \
                         levels.",
                        run_label(r, runs3.len()),
                        differences.join("; ")
                    );
                    log.warn(&format!("AutoTS: {note}"));
                    result.notes.push(note);
                }
            }
        }
    }

    // Steps 2 and 3 on different surfaces (the constraints and the static biases among them):
    // the candidates of the path are not points of the surface of the search.
    if !config.skip_to_step4 && config.path_method == PathMethod::Chain {
        let runs2 = config.runs(1);
        let runs3 = config.runs(2);
        for (r2, step2) in runs2.iter().enumerate() {
            let Ok(chain) = step_args(step2, &[&input], &[], &[], super::spec::NEBMAIN)
                .and_then(|args| NebmainConfig::from_args(&args))
            else {
                continue;
            };
            for (r3, step3) in runs3.iter().enumerate() {
                let Ok(third) = step_args(step3, &[&input], &[], &[], OPTMAIN)
                    .and_then(|args| OptmainConfig::from_args(&args))
                else {
                    continue;
                };
                let differences = super::autots_pair::chain_surface_differences(&chain, &third);
                if !differences.is_empty() {
                    let note = format!(
                        "Steps 2{} and 3{} do not run on the same surface: they differ in {}. \
                         The candidates of the path are not points of the surface of the \
                         transition-state search.",
                        run_label(r2, runs2.len()),
                        run_label(r3, runs3.len()),
                        differences.join("; ")
                    );
                    log.warn(&format!("AutoTS: {note}"));
                    result.notes.push(note);
                }
            }
        }
    }
    for note in &config.notes {
        log.info(&format!("AutoTS: {note}"));
    }
    // The constraints and bias potentials of the surface (those of step 3), for the log and
    // the report.
    if let Some(Ok(third)) = config.runs(2).first().map(|s| {
        step_args(s, &[&input], &[], &[], OPTMAIN).and_then(|a| OptmainConfig::from_args(&a))
    }) {
        if !third.settings.constraints.is_empty() {
            result.surface.push(format!(
                "Constraints of the surface (held by every step that has them): {}",
                super::mapper_report_data::describe_constraints(&third.settings.constraints)
            ));
        }
        if !third.biases.is_empty() {
            result.surface.push(format!(
                "Bias potentials of the surface: {}; the energies of the steps are those with \
                 them, and the summaries of the steps give the energy without them as well",
                third
                    .biases
                    .iter()
                    .enumerate()
                    .map(|(k, b)| format!("{} ({})", b.label(k), b.key))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        for line in &result.surface {
            log.info(&format!("AutoTS: {line}"));
        }
    }

    // An objective of plain variables: the steps on the variables, in this process.
    if let Some((objective, options)) = super::autots_variables::objective_of(config) {
        return run_variables(config, &work, &input, &log, &objective, &options, result);
    }

    // The steps; what they leave in `result` is summarized and reported even when one
    // of them fails, and its error is returned after that.
    let mut steps = || -> MoptResult<()> {
        if config.skip_to_step4 {
            // Judged by the start of its IRC (step 4).
            result.transition_states.push(TransitionState {
                file: input.clone(),
                energy: f64::NAN,
                biased_energy: f64::NAN,
                converged: true,
                imaginary_modes: None,
                meta_irc: None,
                same_as: None,
                ends: Vec::new(),
                end_checks: Vec::new(),
                directory: work.clone(),
                irc_directory: None,
            });
        } else {
            // Step 1: the AFIR path.
            let runs1 = config.runs(0);
            let repeated = config.workflow.is_some();
            let path = if config.skip_step1 || runs1.is_empty() {
                log.info("AutoTS step 1 skipped: the input is the path.");
                result.first_path_method = "given path".to_string();
                input.clone()
            } else if config.first_method == FirstPathMethod::Bitss {
                log.info("AutoTS step 1: the first path by BITSS (optmain -mf bitss).");
                let given = read_xyz_trajectory_file(&input, SpinCharge::default(), None)?;
                let (Some(first), Some(last)) = (given.frames.first(), given.frames.last()) else {
                    return Err(MoptError::calc("run_autots: the input has no structures"));
                };
                if given.frames.len() < 2 {
                    return Err(MoptError::parse(
                        "run_autots: step 1 with BITSS needs the reactant and the product, the \
                         first and the last structure of the input",
                    ));
                }
                let dir = work.join(format!("{name}_step1_bitss"));
                std::fs::create_dir_all(&dir)?;
                let a = dir.join("reactant.xyz");
                let b = dir.join("product.xyz");
                write_xyz_file(&a, &first.system, None)?;
                write_xyz_file(&b, &last.system, None)?;
                let mf = JsonValue::Array(vec![
                    text("bitss"),
                    JsonValue::String(b.to_string_lossy().into_owned()),
                ]);
                if runs1.len() > 1 {
                    return Err(MoptError::parse(
                        "run_autots: step 1 with BITSS runs once (the runs of a workflow follow \
                         each other along a path that a bias potential drives)",
                    ));
                }
                let args = step_args(&runs1[0], &[&a], &[("model_function", mf)], &[], OPTMAIN)?;
                let mut step1 = OptmainConfig::from_args(&args)?;
                step1.keep_full_path = true;
                result.effective_settings.insert(
                    "Step 1 · BITSS (optmain)".into(),
                    super::autots_settings::optimization(&step1),
                );
                write_effective_settings(&result)?;
                let run = one(
                    optmain::run_with(&step1, &dir, console, registry)?,
                    "step 1 (BITSS)",
                )?;
                // The path of the optimization as it is (as AFIR's): the first image from the
                // reactant to where it ended, then the second from there back to the product.
                let summary = run
                    .summary
                    .as_ref()
                    .ok_or_else(|| MoptError::calc("run_autots: BITSS left no trajectory"))?;
                let traj = work.join(format!("{name}_step1_traj.xyz"));
                let traj = export_step_trajectory(summary, traj)?;
                result.afir_directory = Some(run.directory.clone());
                result.first_path_method = "BITSS".to_string();
                log.info(&format!(
                    "AutoTS step 1: the first path is {} (the two images of BITSS from the \
                     reactant and from the product; {}).",
                    traj.display(),
                    if run.converged {
                        "ended on its criteria"
                    } else {
                        "did not end on its criteria"
                    }
                ));
                traj
            } else if let FirstPathMethod::Pair(method) = config.first_method {
                log.info(&format!(
                    "AutoTS step 1: the first path by the {}.",
                    method.title
                ));
                let given = read_xyz_trajectory_file(&input, SpinCharge::default(), None)?;
                let (Some(first), Some(last)) = (given.frames.first(), given.frames.last()) else {
                    return Err(MoptError::calc("run_autots: the input has no structures"));
                };
                if given.frames.len() < 2 {
                    return Err(MoptError::parse(format!(
                        "run_autots: step 1 with the {} needs the reactant and the product, \
                         the first and the last structure of the input",
                        method.title
                    )));
                }
                if runs1.len() > 1 {
                    return Err(MoptError::parse(
                        "run_autots: step 1 with a search between two structures runs once (the \
                         runs of a workflow follow each other along a path that a bias potential \
                         drives)",
                    ));
                }
                let dir = work.join(format!("{name}_step1_{}", method.name));
                std::fs::create_dir_all(&dir)?;
                let a = dir.join("reactant.xyz");
                let b = dir.join("product.xyz");
                write_xyz_file(&a, &first.system, None)?;
                write_xyz_file(&b, &last.system, None)?;
                let run = {
                    let c = super::autots_pair::configuration(method, &runs1[0], &a, &b)?;
                    result.effective_settings.insert(
                        format!("Step 1 · {} (ieipmain)", method.name),
                        super::autots_settings::pair(&c),
                    );
                    write_effective_settings(&result)?;
                    super::ieipmain::run_with(&c, &dir, console, registry)?
                };
                let method_settings = run
                    .directory
                    .join(crate::workflow::pair_search::METHOD_SETTINGS_FILE);
                if method_settings.is_file() {
                    result.effective_settings.insert(
                        format!("Step 1 · {} solver", method.name),
                        std::fs::read_to_string(method_settings)?,
                    );
                }
                let found = run.path.clone().ok_or_else(|| {
                    MoptError::calc(format!(
                        "run_autots: the {} left no path between the two structures (it ended \
                         without points; its log is in {})",
                        method.title,
                        run.directory.display()
                    ))
                })?;
                let traj = work.join(format!("{name}_step1_traj.xyz"));
                std::fs::copy(&found, &traj)?;
                result.afir_directory = Some(run.directory.clone());
                result.first_path_method = method.title.to_string();
                log.info(&format!(
                    "AutoTS step 1: the first path is {} (the points of the search from the \
                     reactant to the product; the search {}).",
                    traj.display(),
                    if run.converged {
                        "ended on its criteria"
                    } else {
                        "did not end on its criteria"
                    }
                ));
                traj
            } else {
                // Every run starts from the last structure of the run before it; the
                // trajectories of the runs together are the first path.
                let mut parts: Vec<PathBuf> = Vec::new();
                let mut displays = Vec::new();
                let mut start = input.clone();
                let mut methods: Vec<String> = Vec::new();
                for (k, settings) in runs1.iter().enumerate() {
                    let label = run_label(k, runs1.len());
                    let args = step_args(settings, &[&start], &[], &[], OPTMAIN)?;
                    let mut step1 = OptmainConfig::from_args(&args)?;
                    step1.keep_full_path = true;
                    result.effective_settings.insert(
                        format!("Step 1{label} · bias-driven optimization"),
                        super::autots_settings::optimization(&step1),
                    );
                    write_effective_settings(&result)?;
                    if step1.biases.is_empty() {
                        return Err(MoptError::parse(format!(
                            "run_autots: step 1{label} needs a bias potential that drives the \
                             reaction: -ma (AFIR, manual_AFIR in step1_settings) or another bias \
                             of optmain"
                        )));
                    }
                    let mut keys: Vec<String> =
                        step1.biases.iter().map(|b| b.key.clone()).collect();
                    keys.dedup();
                    let method = keys.join(", ");
                    log.info(&format!(
                        "AutoTS step 1{label}: the first path (optmain with {method})."
                    ));
                    if !methods.contains(&method) {
                        methods.push(method);
                    }
                    let run = match optmain::run_with(&step1, &work, console, registry) {
                        Ok(runs) => one(runs, "step 1")?,
                        Err(e) if k > 0 => {
                            log.warn(&format!(
                                "AutoTS step 1{label} failed; the path of the runs before it is \
                                 used: {e}"
                            ));
                            break;
                        }
                        Err(e) => return Err(e),
                    };
                    let summary = run
                        .summary
                        .ok_or_else(|| MoptError::calc("run_autots: step 1 has no trajectory"))?;
                    if k == 0 {
                        result.afir_directory = Some(run.directory.clone());
                    }
                    let traj = if repeated {
                        work.join(format!("{name}_step1_run{}_traj.xyz", k + 1))
                    } else {
                        work.join(format!("{name}_step1_traj.xyz"))
                    };
                    parts.push(export_step_trajectory(&summary, traj.clone())?);
                    displays.push(traj);
                    if repeated {
                        let last = work.join(format!("{name}_step1_run{}_final.xyz", k + 1));
                        std::fs::copy(&run.optimized, &last)?;
                        start = last;
                    }
                }
                result.first_path_method = methods.join("; then ");
                let traj = if repeated {
                    let joined = work.join(format!("{name}_step1_combined_traj.xyz"));
                    let structures = joined_trajectories(&displays, &joined)?;
                    let joined = if parts == displays {
                        joined
                    } else {
                        let path = work.join(format!("{name}_step1_combined_path.xyz"));
                        joined_trajectories(&parts, &path)?;
                        path
                    };
                    log.info(&format!(
                        "AutoTS step 1: the trajectories of {} run(s) are joined ({structures} \
                         structures).",
                        parts.len()
                    ));
                    joined
                } else {
                    parts.swap_remove(0)
                };
                log.info(&format!(
                    "AutoTS step 1: the first path is {}.",
                    traj.display()
                ));
                traj
            };
            if !config.skip_step1 && !runs1.is_empty() {
                if let FirstPathMethod::Pair(method) = config.first_method {
                    if let Some(note) = method.seed_note() {
                        log.info(note);
                        result.notes.push(note.to_string());
                    }
                }
            }
            result.afir_path = Some(path.clone());
            tally(&mut result, registry.counts(), 1);
            snapshot(
                config,
                &mut result,
                &log,
                "step 2 (the path) is running".to_string(),
            );

            // Step 2: the path and its candidates.
            let inputs_dir = work.join(format!("{name}_step3_TS_Opt_Inputs"));
            std::fs::create_dir_all(&inputs_dir)?;
            let mut candidates: Vec<(f64, PathBuf)> = Vec::new();
            let runs2 = config.runs(1);
            if runs2.is_empty() {
                // Only the steps of the workflow run.
                log.info(
                    "AutoTS: step 2 is not in the workflow; the run ends with the first path.",
                );
                return Ok(());
            }
            if config.path_method == PathMethod::Bitss && runs2.len() > 1 {
                return Err(MoptError::parse(
                    "run_autots: step 2 with BITSS runs once (the runs of a workflow relax a \
                     path one after the other)",
                ));
            }
            match config.path_method {
                PathMethod::Chain => {
                    // With several runs each relaxes the path of the run before it (or each
                    // the first path, `mode: independent`); the candidates are those of the
                    // last run (or of all runs).
                    let independent = config
                        .workflow
                        .as_ref()
                        .is_some_and(|w| w.independent_paths);
                    let mut source = path.clone();
                    for (k, settings) in runs2.iter().enumerate() {
                        let label = run_label(k, runs2.len());
                        log.info(&format!("AutoTS step 2{label}: the path (nebmain)."));
                        let start = if repeated {
                            let first =
                                work.join(format!("{name}_step2_run{}_init_path.xyz", k + 1));
                            std::fs::copy(if independent { &path } else { &source }, &first)?;
                            first
                        } else {
                            source.clone()
                        };
                        let args = step_args(settings, &[&start], &[], &[], NEBMAIN)?;
                        let completed_stage = result.path_stage.clone();
                        let completed_profiles = result.band_profiles.len();
                        let relaxed = NebmainConfig::from_args(&args).and_then(|c| {
                            result.effective_settings.insert(
                                format!(
                                    "Step 2{label} · {} (nebmain)",
                                    if c.dmf.is_some() { "DMF" } else { "NEB" }
                                ),
                                super::autots_settings::chain(&c),
                            );
                            write_effective_settings(&result)?;
                            if !config.html_report {
                                return nebmain::run_with(&c, &work, console, registry);
                            }
                            let method = if c.dmf.is_some() {
                                "direct MaxFlux"
                            } else {
                                "NEB"
                            };
                            nebmain::run_observed(
                                &c,
                                &work,
                                console,
                                registry,
                                &mut [],
                                &mut |iteration, files, energies| {
                                    if let Some(band) = files.candidates() {
                                        result.band_profiles.push(BandProfile {
                                            run: k + 1,
                                            iteration,
                                            energies: energies.to_vec(),
                                        });
                                        result.path_stage = Some(PathStage {
                                            method: format!("{method}{label}"),
                                            path: Some(band.path),
                                            energies: energies.to_vec(),
                                            ..PathStage::default()
                                        });
                                        tally(&mut result, registry.counts(), 1);
                                        let stage = format!(
                                            "step 2{label}: {method} iteration {iteration} saved"
                                        );
                                        snapshot(config, &mut result, &log, stage);
                                    }
                                },
                            )
                        });
                        let run = match relaxed {
                            Ok(run) => run,
                            Err(e) if k > 0 => {
                                result.path_stage = completed_stage;
                                result.band_profiles.truncate(completed_profiles);
                                log.warn(&format!(
                                    "AutoTS step 2{label} failed; the runs before it stand: {e}"
                                ));
                                break;
                            }
                            Err(e) => return Err(e),
                        };
                        let found = run.candidates.ok_or_else(|| {
                            MoptError::calc("run_autots: step 2 gave no candidates")
                        })?;
                        let mut files = found.files.clone();
                        if repeated {
                            let last =
                                work.join(format!("{name}_step2_run{}_final_path.xyz", k + 1));
                            std::fs::copy(&found.path, &last)?;
                            log.info(&format!(
                                "AutoTS step 2{label}: the relaxed path is {}.",
                                last.display()
                            ));
                            source = last;
                            let dir = work.join(format!("{name}_step2_run{}_candidates", k + 1));
                            std::fs::create_dir_all(&dir)?;
                            for (i, file) in files.iter_mut().enumerate() {
                                let local =
                                    dir.join(format!("{name}_s2_run{}_guess_{}.xyz", k + 1, i + 1));
                                std::fs::copy(&*file, &local)?;
                                *file = local;
                            }
                        }
                        if !independent {
                            candidates.clear();
                        }
                        for (image, file) in found.images.iter().zip(&files) {
                            let energy = run.energies.get(*image).copied().unwrap_or(f64::NAN);
                            candidates.push((energy, file.clone()));
                        }
                        let dmf = settings
                            .get("DMF")
                            .and_then(JsonValue::as_bool)
                            .unwrap_or(false);
                        result.path_stage = Some(PathStage {
                            method: format!(
                                "{}{label}",
                                if dmf {
                                    "direct MaxFlux (nebmain -dmf)"
                                } else {
                                    "chain (nebmain)"
                                }
                            ),
                            path: Some(found.path.clone()),
                            energies: run.energies.clone(),
                            candidates: found.images.clone(),
                            estimates: files,
                        });
                        if k + 1 < runs2.len() {
                            snapshot(
                                config,
                                &mut result,
                                &log,
                                format!("step 2: run {} of {} is running", k + 2, runs2.len()),
                            );
                        }
                    }
                }
                PathMethod::Bitss => {
                    log.info(
                        "AutoTS step 2: BITSS between the ends of the path (optmain -mf bitss).",
                    );
                    let traj = read_xyz_trajectory_file(&path, SpinCharge::default(), None)?;
                    let (Some(first), Some(last)) = (traj.frames.first(), traj.frames.last())
                    else {
                        return Err(MoptError::calc("run_autots: the path has no structures"));
                    };
                    let dir = work.join(format!("{name}_step2_bitss"));
                    std::fs::create_dir_all(&dir)?;
                    let a = dir.join("reactant.xyz");
                    let b = dir.join("product.xyz");
                    write_xyz_file(&a, &first.system, None)?;
                    write_xyz_file(&b, &last.system, None)?;
                    let mf = JsonValue::Array(vec![
                        text("bitss"),
                        JsonValue::String(b.to_string_lossy().into_owned()),
                    ]);
                    let args =
                        step_args(&runs2[0], &[&a], &[("model_function", mf)], &[], OPTMAIN)?;
                    let step2 = OptmainConfig::from_args(&args)?;
                    result.effective_settings.insert(
                        "Step 2 · BITSS (optmain)".into(),
                        super::autots_settings::optimization(&step2),
                    );
                    write_effective_settings(&result)?;
                    let run = one(
                        optmain::run_with(&step2, &dir, console, registry)?,
                        "step 2 (BITSS)",
                    )?;
                    let guess = run.directory.join("reactant_ts_guess.xyz");
                    if !guess.is_file() {
                        return Err(MoptError::calc(format!(
                            "run_autots: BITSS wrote no estimate ({})",
                            guess.display()
                        )));
                    }
                    let energy = run.summary.as_ref().map_or(f64::NAN, |s| s.energy.value());
                    result.path_stage = Some(PathStage {
                        method: "BITSS".to_string(),
                        path: run.summary.as_ref().map(|s| s.trajectory.clone()),
                        energies: Vec::new(),
                        candidates: Vec::new(),
                        estimates: vec![guess.clone()],
                    });
                    candidates.push((energy, guess));
                }
            }
            candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
            candidates.truncate(config.top_n);
            for (k, (energy, file)) in candidates.iter().enumerate() {
                let local = inputs_dir.join(format!("{name}_ts_guess_{}.xyz", k + 1));
                std::fs::copy(file, &local)?;
                log.info(&format!(
                    "AutoTS step 2: candidate {} at {energy:.10} Hartree ({}).",
                    k + 1,
                    local.display()
                ));
                result.candidates.push(local);
            }

            let stage = format!(
                "step 3: the transition state of candidate 1 of {} is being optimized",
                result.candidates.len()
            );
            tally(&mut result, registry.counts(), 2);
            snapshot(config, &mut result, &log, stage);

            // Step 3: the transition states. A later run of the step takes up the
            // candidates that have no transition state yet (converged, with one imaginary
            // frequency where that is looked at).
            let runs3 = config.runs(2);
            let guesses = result.candidates.clone();
            let mut found: Vec<Option<usize>> = vec![None; guesses.len()];
            for (r, settings) in runs3.iter().enumerate() {
                let label = run_label(r, runs3.len());
                for (k, guess) in guesses.iter().enumerate() {
                    let done = found[k].is_some_and(|at| {
                        let ts = &result.transition_states[at];
                        ts.converged && ts.imaginary_modes.is_none_or(|n| n == 1)
                    });
                    if done {
                        continue;
                    }
                    log.info(&format!(
                        "AutoTS step 3{label}: candidate {} to a transition state (optmain -order \
                         1).",
                        k + 1
                    ));
                    let mut extra = vec![("saddle_order", text("1"))];
                    if config.check_frequencies {
                        extra.push(("frequency_analysis", JsonValue::Bool(true)));
                    }
                    let args = step_args(settings, &[guess], &extra, &[], OPTMAIN)?;
                    let step3 = OptmainConfig::from_args(&args)?;
                    result.effective_settings.insert(
                        format!("Step 3{label} · candidate {}", k + 1),
                        super::autots_settings::optimization(&step3),
                    );
                    write_effective_settings(&result)?;
                    let run = match optmain::run_with(&step3, &work, console, registry) {
                        Ok(runs) => one(runs, "step 3")?,
                        Err(e) => {
                            log.warn(&format!(
                                "AutoTS step 3{label}: candidate {} failed: {e}",
                                k + 1
                            ));
                            continue;
                        }
                    };
                    let file = if repeated {
                        work.join(format!("{name}_s3_run{}_ts_final_{}.xyz", r + 1, k + 1))
                    } else {
                        work.join(format!("{name}_ts_final_{}.xyz", k + 1))
                    };
                    std::fs::copy(&run.optimized, &file)?;
                    let (energy, biased_energy, imaginary_modes) = run
                        .summary
                        .as_ref()
                        .map_or((f64::NAN, f64::NAN, None), |s| {
                            (s.energy.value(), s.biased_energy.value(), s.imaginary_modes)
                        });
                    let mut same_as = None;
                    for (j, earlier) in result.transition_states.iter().enumerate() {
                        if found[k] != Some(j)
                            && (earlier.energy - energy).abs() < 1.0e-6
                            && same_structure(&earlier.file, &file)?
                        {
                            same_as = Some(j + 1);
                            break;
                        }
                    }
                    let state = match (run.converged, imaginary_modes) {
                        (false, _) => "did not converge".to_string(),
                        (true, Some(1)) => "converged, one imaginary frequency".to_string(),
                        (true, Some(n)) => format!(
                            "converged with {n} imaginary frequencies: not a transition state"
                        ),
                        (true, None) => "converged".to_string(),
                    };
                    log.info(&format!(
                        "AutoTS step 3{label}: {} at {energy:.10} Hartree, {state}{}.",
                        file.display(),
                        same_as.map_or_else(String::new, |j| format!(
                            ", the same as transition state {j}"
                        ))
                    ));
                    let ts = TransitionState {
                        file,
                        energy,
                        biased_energy,
                        converged: run.converged,
                        imaginary_modes,
                        same_as,
                        meta_irc: None,
                        ends: Vec::new(),
                        end_checks: Vec::new(),
                        directory: run.directory.clone(),
                        irc_directory: None,
                    };
                    match found[k] {
                        Some(at) => result.transition_states[at] = ts,
                        None => {
                            found[k] = Some(result.transition_states.len());
                            result.transition_states.push(ts);
                        }
                    }
                    snapshot(
                        config,
                        &mut result,
                        &log,
                        format!(
                            "step 3{label}: {} of {} candidates done",
                            k + 1,
                            guesses.len()
                        ),
                    );
                }
            }
        }

        // Step 4: IRC and the ends.
        let runs4 = config.runs(3);
        if (config.run_step4 || config.skip_to_step4) && runs4.is_empty() {
            log.info("AutoTS step 4 is not in the workflow: no IRC.");
        } else if config.run_step4 || config.skip_to_step4 {
            if runs4.len() > 1 {
                log.info(
                    "AutoTS step 4 runs once, with its first settings (the IRC of a transition \
                     state is one calculation).",
                );
            }
            let step4 = &runs4[0];
            if !step4.contains_key("intrinsic_reaction_coordinates") {
                return Err(MoptError::parse(
                    "run_autots: step 4 needs intrinsic_reaction_coordinates in step4_settings",
                ));
            }
            let end_method = step4
                .get("step4b_opt_method")
                .cloned()
                .unwrap_or_else(|| JsonValue::Array(vec![text("rsirfo_block_fsb")]));
            for k in 0..result.transition_states.len() {
                let ts = result.transition_states[k].clone();
                // With the check on, only a stationary point with exactly one imaginary
                // frequency goes on; one whose analysis is missing is not known to be a
                // transition state and does not either.
                // The input of --skip_to_step4 is judged by the start of its IRC.
                let checked = if config.check_frequencies {
                    ts.imaginary_modes == Some(1)
                } else {
                    ts.imaginary_modes.is_none_or(|n| n == 1)
                };
                let usable =
                    config.skip_to_step4 || (ts.converged && ts.same_as.is_none() && checked);
                if !usable {
                    log.info(&format!("AutoTS step 4: {} is skipped.", ts.file.display()));
                    continue;
                }
                if config.skip_to_step4 {
                    tally(&mut result, registry.counts(), 3);
                    match given_point_irc(step4, &ts.file, &work, console, registry, &log) {
                        Ok(given) => {
                            let t = &mut result.transition_states[k];
                            t.imaginary_modes = Some(given.imaginary);
                            t.energy = given.energy;
                            t.biased_energy = given.biased_energy;
                            t.irc_directory = Some(given.directory.clone());
                            if let Some(notice) = &given.meta_irc {
                                t.meta_irc = Some(notice.clone());
                                result.notes.push(notice.clone());
                            }
                            let (ends, checks) = optimize_ends(
                                config,
                                step4,
                                &end_method,
                                &ts.file,
                                &given.endpoints,
                                k,
                                &work,
                                console,
                                registry,
                                &log,
                                &mut result,
                                given.meta_irc.is_some(),
                            )?;
                            write_profile(
                                &work,
                                &ts.file,
                                given.energy,
                                &ends,
                                &checks,
                                given.meta_irc.is_some(),
                            )?;
                            result.transition_states[k].ends = ends;
                            result.transition_states[k].end_checks = checks;
                        }
                        Err(e @ (MoptError::Convergence(_) | MoptError::Calculator(_))) => {
                            let warning = format!(
                                "AutoTS step 4: the IRC of {} failed: {e}; it has no ends.",
                                ts.file.display()
                            );
                            log.warn(&warning);
                            result.notes.push(warning);
                        }
                        Err(e) => return Err(e),
                    }
                    continue;
                }
                log.info(&format!("AutoTS step 4: the IRC of {}.", ts.file.display()));
                tally(&mut result, registry.counts(), 3);
                snapshot(
                    config,
                    &mut result,
                    &log,
                    format!("step 4: the IRC of transition state {} is running", k + 1),
                );
                let args = step_args(
                    step4,
                    &[&ts.file],
                    &[("saddle_order", text("1"))],
                    &["step4b_opt_method"],
                    OPTMAIN,
                )?;
                let irc = OptmainConfig::from_args(&args)?;
                result.effective_settings.insert(
                    format!("Step 4 · IRC {}", k + 1),
                    super::autots_settings::optimization(&irc),
                );
                write_effective_settings(&result)?;
                // An IRC that fails numerically (it cannot follow the path, or the calculator
                // fails along it) ends the IRC of this transition state, not the run: it is
                // reported for it, and the other transition states go on. Other errors (a
                // refused combination, a file) stop the run.
                let run = match optmain::run_with(&irc, &work, console, registry) {
                    Ok(runs) => one(runs, "step 4 (IRC)")?,
                    Err(e @ (MoptError::Convergence(_) | MoptError::Calculator(_))) => {
                        let warning = format!(
                            "AutoTS step 4: the IRC of transition state {} ({}) failed: {e}; it \
                             has no ends.",
                            k + 1,
                            ts.file.display()
                        );
                        log.warn(&warning);
                        result.notes.push(warning);
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                // The transition state keeps the energies of its file (step 3), with and
                // without the bias: the run of step 4 optimizes again before its IRC, and its
                // last structure (with its environment, under microiterations) is another one
                // where that run did not converge at once.
                let ts_energy = ts.energy;
                result.transition_states[k].irc_directory = Some(run.directory.clone());
                let endpoints = run.summary.map(|s| s.irc_endpoints).unwrap_or_default();
                let (ends, checks) = optimize_ends(
                    config,
                    step4,
                    &end_method,
                    &ts.file,
                    &endpoints,
                    k,
                    &work,
                    console,
                    registry,
                    &log,
                    &mut result,
                    false,
                )?;
                write_profile(&work, &ts.file, ts_energy, &ends, &checks, false)?;
                result.transition_states[k].ends = ends;
                result.transition_states[k].end_checks = checks;
            }
        }
        Ok(())
    };
    let outcome = steps();
    if let Err(e) = &outcome {
        result.error = Some(e.to_string());
    }

    // The summary and the report.
    let step_now = result.counted.1;
    tally(&mut result, registry.counts(), step_now);
    log.info(&evaluations_line(&result));
    result.progress = None;
    match write_outputs(config, &result) {
        Ok(Some(report)) => log.info(&format!("AutoTS: the report is {}.", report.display())),
        Ok(None) => {}
        Err(e) => log.warn(&format!("AutoTS: the report could not be written: {e}")),
    }
    if let Some(p) = &pool {
        let (offered, taken) = crate::pes::replay::ReplaySource::counts(p.as_ref());
        log.info(&format!(
            "AutoTS: {taken} evaluation(s) of the run before were taken instead of computed \
             ({} recorded one(s) were not asked for again).",
            offered.saturating_sub(taken)
        ));
    }
    if let Err(e) = outcome {
        log.warn(&format!(
            "AutoTS: stopped by an error; the summary and the report show what was done: {e}"
        ));
        return Err(e);
    }
    log.info(&format!(
        "AutoTS: {} transition state(s); the summary is {}.",
        result.transition_states.len(),
        work.join("autots_summary.csv").display()
    ));
    Ok(result)
}
