//! `ircmain`: the intrinsic reaction coordinate from a given structure (a transition state;
//! any other structure gives a meta-IRC down its gradient), or from a point of an objective
//! of plain variables (`-os mb2d`, `-os expr:...` with a variables file).
//!
//! The calculator options, the bias potentials, `-es`, `-mf` (functions of the states),
//! the constraints (`-fix`, `-pc`, `-gfix`, `-uc`: the path within them), `-fc`,
//! `--config`, `--dump-config`, and `--restart` are those of `optmain`; the IRC itself is
//! the one of `optmain -irc`
//! ([`super::irc`]). `-path_analysis` hands the combined path to the analysis of
//! `freqmain -path`.
//!
//! MultiOptPy: no counterpart as a command (`optmain -irc` after an optimization;
//! `irc.py:IRC`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use super::bias::{parse_biases, BiasSpec};
use super::calculator::{calculator_name, calculator_options};
use super::charge_spin::{state_spec, StateSpec};
use super::constraints::parse_constraints;
use super::freqmain::FreqmainConfig;
use super::irc::{
    irc_states, run_structure, run_variables, settings_from_args, with_common_options, IrcRun,
    IrcSurfaceParts,
};
use super::option::reject_unavailable;
use super::settings::settings_json;
use super::spec::IRCMAIN;
use super::variables::read_variables_file;
use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::read_xyz_file;
use crate::irc::IrcSettings;
use crate::opt::cell::{gigapascal_per_atomic_unit, DEFAULT_LATTICE_WEIGHT};
use crate::pes::constraint::{Constraint, ConstraintSet};
use crate::pes::objective::{check_dimension, ObjectiveRegistry};
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};
use crate::workflow::irc::IrcObserver;
use crate::workflow::journal::{RunJournal, JOURNAL_FILE};

/// Step size of `ircmain` without `-irc_step` (amu^(1/2) Bohr; measured, see
/// `docs/source/irc.md`).
pub const DEFAULT_STEP_SIZE: &str = "0.1";
/// `MAX_STEP` of `ircmain` without `-irc_max`.
pub const DEFAULT_MAX_STEP: &str = "300";

/// Typed `ircmain` settings.
#[derive(Debug, Clone)]
pub struct IrcmainConfig {
    /// Input structures or variables files.
    pub inputs: Vec<PathBuf>,
    /// The settings in effect as a settings document (`settings.json` of each run).
    pub settings_document: String,
    /// Run directory to continue (`--restart`); `None`: a new one.
    pub restart_dir: Option<PathBuf>,
    /// Calculator name (`-os`, `-xtb`, `-dxtb`; default PySCF).
    pub calculator: String,
    /// Calculator settings.
    pub calculator_options: CalculatorOptions,
    /// `-elec` / `-spin`.
    pub state: StateSpec,
    /// The IRC (`-irc_step`, `-irc_max`, `-irc_method` or `-irc`, with `-irc_opt`, `-fc`).
    pub irc: IrcSettings,
    /// `-es`: the electronic state (0: the ground state).
    pub excited_state: usize,
    /// `-mf NAME [CHARGE MULTIPLICITY]`: the states combined into the surface.
    pub model_function: Vec<String>,
    /// `-fix`, `-pc`, `-gfix`, `-uc`: constraints the path keeps.
    pub constraints: Vec<Constraint>,
    /// `-vc` with `-vc_weight` and `-vc_pressure` (Hartree/Bohr^3): the IRC of the atoms
    /// and the cell together.
    pub variable_cell: Option<(f64, f64)>,
    /// `-ih FILE`: the first Hessian that is asked for is read from this file
    /// (`io::hessian_file`).
    pub initial_hessian: Option<PathBuf>,
    /// Bias potentials (`-ma`, `-kp`, ...).
    pub biases: Vec<BiasSpec>,
    /// `-modelhess` (`o1numhess`: Hessians from gradients).
    pub model_hessian: Option<String>,
    /// `-core`.
    pub threads: usize,
    /// `-path_analysis`: the settings of the analysis of the combined path.
    pub path_analysis: Option<Box<FreqmainConfig>>,
}

impl IrcmainConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        reject_unavailable(p, IRCMAIN)?;
        let mut inputs: Vec<PathBuf> = Vec::new();
        for name in &p.positionals {
            let found = crate::core_infra::glob::expand(name)?;
            if found.is_empty() {
                return Err(MoptError::config(format!("no file matches '{name}'")));
            }
            inputs.extend(found);
        }
        if inputs.is_empty() {
            return Err(MoptError::config(
                "ircmain needs an input structure (xyz) or a variables file",
            ));
        }
        let variable_cell = if p.get_bool("variable_cell", false) {
            if !p.get_list("variable_cell_afir").is_empty() {
                return Err(MoptError::unsupported(
                    "-vc_afir pushes the cell in a search; the IRC of the atoms and the cell \
                     follows the surface itself",
                ));
            }
            Some((
                p.get_f64("variable_cell_weight", DEFAULT_LATTICE_WEIGHT)?,
                p.get_f64("variable_cell_pressure", 0.0)? / gigapascal_per_atomic_unit(),
            ))
        } else {
            None
        };
        let model_function = p.get_list("model_function").to_vec();
        // A wrong or unsuitable -mf is an error here, before anything is calculated.
        irc_states(&model_function, 0)?;
        let irc = match settings_from_args(p)? {
            Some(settings) => settings,
            None => with_common_options(
                IrcSettings::from_multioptpy(
                    p.get_str("irc_step_size", DEFAULT_STEP_SIZE),
                    p.get_str("irc_max_step", DEFAULT_MAX_STEP),
                    p.get_str("irc_method", "lqa"),
                )?,
                p,
            )?,
        };
        let path_analysis = if p.get_bool("path_analysis", false) {
            let mut analysis = FreqmainConfig::from_args(p)?;
            analysis.path = true;
            Some(Box::new(analysis))
        } else {
            None
        };
        Ok(Self {
            inputs,
            settings_document: settings_json(p, IRCMAIN).to_string_pretty(),
            restart_dir: None,
            calculator: calculator_name(p)?,
            calculator_options: calculator_options(p)?,
            state: state_spec(p)?,
            irc,
            excited_state: p.get_usize("excited_state", 0)?,
            model_function,
            constraints: parse_constraints(p)?,
            variable_cell,
            initial_hessian: p.get_opt_str("initial_hessian").map(PathBuf::from),
            biases: parse_biases(p)?,
            model_hessian: p.get_opt_str("use_model_hessian").map(str::to_string),
            threads: p.get_usize("N_THREAD", 8)?.max(1),
            path_analysis,
        })
    }
}

/// Result of one input.
#[derive(Debug, Clone)]
pub struct IrcmainResult {
    /// The IRC and its files.
    pub irc: IrcRun,
    /// The directory of the analysis of the path (`-path_analysis`).
    pub path_analysis: Option<PathBuf>,
}

/// MultiOptPy-style run directory `<date>/<stem>_IRC_<timestamp>/` under `base`.
pub fn run_directory(base: &Path, input: &Path, now: &DateTime) -> PathBuf {
    let stem = input.with_extension("");
    base.join(now.date_dir()).join(format!(
        "{}_IRC_{}",
        stem.to_string_lossy(),
        now.timestamp()
    ))
}

/// Run `ircmain` for every input with the calculators of `registry`, writing run
/// directories under `base`.
pub fn run_with(
    config: &IrcmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<Vec<IrcmainResult>> {
    run_with_observers(config, base, console, registry, &mut [])
}

/// [`run_with`] with further observers: they see every event of the IRC after the files
/// and the log were written (processing after every step) and can end the run.
pub fn run_with_observers(
    config: &IrcmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    extra: &mut [&mut dyn IrcObserver],
) -> MoptResult<Vec<IrcmainResult>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.calculator_options.pool_threads(config.threads))
        .build()
        .map_err(|e| MoptError::config(format!("thread pool: {e}")))?;
    let mut results = Vec::with_capacity(config.inputs.len());
    for input in &config.inputs {
        let result = pool
            .install(|| run_one(config, input, base, console, registry, extra))
            .context(input.display())?;
        results.push(result);
    }
    Ok(results)
}

fn stem_of(input: &Path) -> String {
    input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned())
}

fn open_run(
    config: &IrcmainConfig,
    input: &Path,
    base: &Path,
    console: Option<Level>,
) -> MoptResult<(PathBuf, Logger)> {
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => run_directory(base, input, &DateTime::now_utc()),
    };
    std::fs::create_dir_all(&dir)?;
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} ircmain: {}",
        crate::VERSION,
        input.display()
    ));
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document),
    )?;
    Ok((dir, log))
}

fn run_one(
    config: &IrcmainConfig,
    input: &Path,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    extra: &mut [&mut dyn IrcObserver],
) -> MoptResult<IrcmainResult> {
    let objectives = ObjectiveRegistry::builtin();
    if objectives.contains(&config.calculator) {
        return run_objective(config, input, base, console, &objectives, extra);
    }
    let spin = config.state.of_file(input)?;
    let start = read_xyz_file(input, spin, None)?;
    let evaluations_before = registry.counts();
    // ONIOM microiterations: the IRC of the core with the environment relaxed (the
    // microiterative IRC).
    let uncovered = config
        .variable_cell
        .is_some()
        .then_some("a variable cell (-vc; ONIOM has no stress)");
    // Constraints and bias potentials on the core are taken to it.
    let biases = super::oniom_micro::build_terms(&config.biases, &start)?;
    let wrapped = super::oniom_micro::wrap_constrained(
        &config.calculator_options,
        uncovered,
        &config.constraints,
        &biases,
        &[],
        // A side of the path ends in its minimum at the thresholds of the approach; the
        // environment is relaxed to the tighter of those and the thresholds of the path.
        (
            config.irc.max_force.min(config.irc.approach_max_force),
            config.irc.rms_force.min(config.irc.approach_rms_force),
        ),
        start,
        &config.calculator,
        registry,
    )?;
    let (start, calc, micro, micro_note, constraints, terms) = (
        wrapped.start,
        wrapped.calc,
        wrapped.micro,
        wrapped.note,
        wrapped.constraints,
        wrapped.terms,
    );
    let calc = crate::pes::o1numhess::wrap_if_requested(
        calc,
        config.model_hessian.as_deref(),
        config.irc.exact_hessian_interval.is_some(),
    );
    let calc = super::oniom_micro::first_hessian(
        calc,
        config.initial_hessian.as_deref(),
        &start,
        micro.as_deref(),
    )?;
    if calc.uses_electrons() {
        start.spin_charge.check_electrons(&start.elements)?;
    }
    let (dir, log) = open_run(config, input, base, console)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    log.info(&format!("Calculator: {}", calc.description()));
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&config.calculator_options.software_path_report());
    if let Some(note) = &micro_note {
        log.info(note);
    }
    super::oniom_micro::report_relaxations(micro.as_deref(), &log);
    if let Some(relaxed) = &micro {
        // The start is judged as the search for the transition state judged its end.
        log.info(&format!(
            "ONIOM microiterations: the Hessian at the start is that of a check of stability, \
             with the environment relaxed to {} (the tolerance of the checks, the one of the \
             verification of a converged saddle-point search).",
            super::oniom_micro::describe_check_tolerance(relaxed)
        ));
    }
    if config.excited_state > 0 {
        log.info(&format!("Electronic state: {}", config.excited_state));
    }
    if let Some((weight, pressure)) = config.variable_cell {
        if !config.model_function.is_empty() || !config.constraints.is_empty() {
            return Err(MoptError::unsupported(
                "-vc: the IRC of the atoms and the cell follows one state without constraints; \
                 -mf and the constraints are not available with it",
            ));
        }
        let irc = super::irc::run_cell(
            &config.irc,
            &journaled,
            &terms,
            &start,
            config.excited_state,
            0,
            weight,
            pressure,
            &stem_of(input),
            &dir,
            &log,
            extra,
        )?;
        log.info(
            &registry
                .counts()
                .since(&evaluations_before)
                .line("this input"),
        );
        log.flush();
        return Ok(IrcmainResult {
            irc,
            path_analysis: None,
        });
    }
    let states = irc_states(&config.model_function, config.excited_state)?;
    if let Some(name) = config.model_function.first() {
        log.info(&format!("Mode: Model Function ({name})"));
        super::optmain::report_excitations(calc.as_ref(), &start, &dir, &log)?;
    }
    let constraints = if constraints.is_empty() {
        None
    } else {
        Some(ConstraintSet::new(&constraints, &start, None)?)
    };
    let parts = IrcSurfaceParts {
        terms,
        excited_state: config.excited_state,
        states,
        constraints,
        // A given structure: the bias potentials as at the start of a ramp.
        iteration: 0,
    };
    let irc = run_structure(
        &config.irc,
        &journaled,
        parts,
        &start,
        &stem_of(input),
        &dir,
        &log,
        extra,
    )?;
    // With microiterations the files of the IRC hold the whole system.
    if let Some(relaxed) = &micro {
        super::oniom_micro::finish(relaxed, None, &dir, &log)?;
    }
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
    }
    if irc.interrupted {
        log.info(&format!(
            "Continue with: ircmain --restart {}",
            dir.display()
        ));
    }
    for path in &irc.endpoints {
        log.info(&format!("IRC end point file path: {}", path.display()));
    }
    let mut path_analysis = None;
    if let (Some(analysis), false) = (&config.path_analysis, irc.interrupted) {
        match &irc.combined_path {
            Some(path) => {
                let target = dir.join("path_analysis");
                let mut analysis = (**analysis).clone();
                analysis.inputs = vec![path.clone()];
                analysis.restart_dir = Some(target.clone());
                log.info(&format!(
                    "Analysis of the path (freqmain -path) of {} in {}",
                    path.display(),
                    target.display()
                ));
                log.flush();
                super::freqmain::run_with(&analysis, &dir, console, registry)?;
                path_analysis = Some(target);
            }
            None => log.info(
                "-path_analysis needs the two sides of a transition state; there is no \
                 combined path to analyse.",
            ),
        }
    }
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(IrcmainResult { irc, path_analysis })
}

fn run_objective(
    config: &IrcmainConfig,
    input: &Path,
    base: &Path,
    console: Option<Level>,
    objectives: &ObjectiveRegistry,
    extra: &mut [&mut dyn IrcObserver],
) -> MoptResult<IrcmainResult> {
    if !config.biases.is_empty()
        || config.excited_state > 0
        || !config.model_function.is_empty()
        || !config.constraints.is_empty()
        || config.variable_cell.is_some()
    {
        return Err(MoptError::parse(
            "bias potentials, -es, -mf, and constraints need atoms; they cannot be used with \
             an objective of plain variables",
        ));
    }
    let evaluations_before = objectives.counts();
    let objective = objectives.create(&config.calculator, &config.calculator_options)?;
    let vars = read_variables_file(input)?;
    let n = vars.values.len();
    check_dimension(objective.as_ref(), n)?;
    let names = vars
        .names
        .clone()
        .unwrap_or_else(|| objective.variable_names(n));
    let (dir, log) = open_run(config, input, base, console)?;
    log.info(&format!(
        "Mode: IRC of {n} plain variables ({}), unit masses",
        names.join(", ")
    ));
    log.info(&format!("Objective: {}", objective.description()));
    if config.restart_dir.is_some() {
        log.info("Restart: the run is repeated from the starting point (objectives are cheap).");
    }
    let irc = run_variables(
        &config.irc,
        objective.as_ref(),
        &names,
        &vars.values,
        &stem_of(input),
        &dir,
        &log,
        extra,
    )?;
    if irc.interrupted {
        log.info(&format!(
            "Continue with: ircmain --restart {}",
            dir.display()
        ));
    }
    if config.path_analysis.is_some() {
        log.info("-path_analysis needs atoms; it is not run for plain variables.");
    }
    log.info(
        &objectives
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(IrcmainResult {
        irc,
        path_analysis: None,
    })
}

/// Thin wrapper used by tests and embedding code: settings for `args`.
pub fn config_from_cli(args: &[&str]) -> MoptResult<IrcmainConfig> {
    let p = super::option::parser("ircmain", IRCMAIN).parse(args)?;
    IrcmainConfig::from_args(&p)
}
