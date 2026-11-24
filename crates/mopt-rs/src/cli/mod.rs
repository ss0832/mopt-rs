//! The ten MultiOptPy commands (`optmain`, `nebmain`, `confsearch`, ...), `freqmain`, `ircmain`,
//! `run_kinetics`, and the single entry point
//! `mopt <command> [options]` (`mopt-py <command> [options]` from Python).
//!
//! Exit codes: 0 success, 1 error, 2 usage error or a feature that is not implemented yet,
//! 3 interrupted (continue with `--restart`).
//!
//! MultiOptPy: `Entrypoints/core.py` (`run_optmain`, `run_nebmain`, ...) and
//!  `interface.py:init_parser`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod autots;
pub mod autots_pair;
pub mod autots_report;
mod autots_settings;
pub mod autots_static;
pub mod autots_variables;
pub mod autots_variables_report;
pub mod autots_workflow;
pub mod bias;
pub mod bias_constants;
pub mod bitss;
pub mod calculator;
pub mod charge_spin;
// The conformation search.
pub mod confsearch;
pub mod confsearch_files;
pub mod confsearch_jobs;
pub mod confsearch_settings;
pub mod constraints;
pub mod etmain;
pub mod follow;
pub mod freqmain;
// `ieipmain`: the searches from two structures or one.
pub mod ieipmain;
pub mod ieipmain_cell;
pub mod ieipmain_coords;
pub mod ieipmain_followup;
pub mod ieipmain_methods;
pub mod ieipmain_path;
pub mod ieipmain_settings;
pub mod ieipmain_states;
pub mod irc;
pub mod ircmain;
// `run_mapper`: the reaction-network mapper.
pub mod mapper;
mod mapper_job_report;
pub mod mapper_jobs;
pub mod mapper_kinetics;
pub mod mapper_merge;
mod mapper_minima;
pub mod mapper_oniom;
pub mod mapper_points;
mod mapper_progress;
pub mod mapper_remap;
pub mod mapper_report;
pub mod mapper_report_data;
pub mod mapper_report_figures;
mod mapper_report_kinetics;
pub mod mapper_report_pack;
pub mod mapper_report_script;
mod mapper_searches;
pub mod mapper_settings;
pub mod mapper_values;
// Molecular dynamics.
pub mod diabatic_path;
pub mod mdmain;
pub mod mdmain_analysis;
pub mod mdmain_coordinates;
pub mod mdmain_cv;
pub mod mdmain_settings;
pub mod mdmain_umbrella;
pub mod mdmain_variables;
pub mod mdmain_wigner;
pub mod nebmain;
pub mod nebmain_cell;
pub mod nebmain_dmf;
pub mod nebmain_followup;
pub mod nebmain_settings;
pub mod oniom;
pub mod oniom_micro;
pub mod option;
pub mod optmain;
pub mod orbital_path;
pub mod pair_settings;
// `orientsearch`: orientations of the parts of a complex.
pub mod orientsearch;
pub mod orientsearch_energies;
pub mod orientsearch_output;
pub mod orientsearch_settings;
// `relaxedscan`: scans of coordinates, relaxed or rigid.
pub mod relaxedscan;
pub mod relaxedscan_followup;
pub mod relaxedscan_outputs;
pub mod relaxedscan_points;
pub mod relaxedscan_report;
pub mod relaxedscan_resper;
pub mod relaxedscan_scan;
pub mod relaxedscan_settings;
pub mod relaxedscan_targets;
// `run_automecp`: crossing points of two states from one structure (a run of the mapper).
pub mod automecp;
// Its page: the steps of the search, made like that of `run_autots`.
pub mod automecp_report;
// `run_kinetics` and `run_rcmc`: the kinetics of a network file.
pub mod run_kinetics;
pub mod settings;
pub mod spec;
pub mod variable_cell;
pub mod variables;

use std::path::Path;

use crate::core_infra::arg_parser as core_infra_args;
use crate::core_infra::log::{Level, Logger};
use crate::error::MoptError;
use crate::pes::counting::EvaluationCounts;
use crate::pes::registry::CalculatorRegistry;
use option::{help, parser};
use spec::COMMANDS;

/// Exit code for success.
pub const EXIT_OK: i32 = 0;
/// Exit code for errors during a run.
pub const EXIT_ERROR: i32 = 1;
/// Exit code for usage errors and features that are not implemented yet.
pub const EXIT_USAGE: i32 = 2;
/// Exit code for runs stopped by a `STOP` file (continue with `--restart RUN_DIR`).
pub const EXIT_INTERRUPTED: i32 = 3;

/// Names of the commands.
pub fn command_names() -> Vec<&'static str> {
    COMMANDS.iter().map(|(n, _, _, _)| *n).collect()
}

fn exit_code(e: &MoptError) -> i32 {
    match e {
        MoptError::Unsupported(_) | MoptError::Parse(_) => EXIT_USAGE,
        _ => EXIT_ERROR,
    }
}

/// Run `command` with `args` (without the program name), writing run directories under the
/// current directory. Returns the process exit code.
pub fn run_command(command: &str, args: &[String]) -> i32 {
    run_command_with(command, args, &CalculatorRegistry::native())
}

/// [`run_command`] with the calculators of `registry` (the Python package adds its engines).
pub fn run_command_with(command: &str, args: &[String], registry: &CalculatorRegistry) -> i32 {
    run_command_with_parts(command, args, registry, &mapper::MapperParts::default())
}

/// [`run_command_with`] with the parts of the mapper `parts` for `run_mapper` (the Python
/// package adds the exploration methods of Python classes, `method=python:MODULE:CLASS`).
pub fn run_command_with_parts(
    command: &str,
    args: &[String],
    registry: &CalculatorRegistry,
    parts: &mapper::MapperParts,
) -> i32 {
    crate::core_infra::parallelism::deterministic_linear_algebra();
    let console = Logger::console(Level::Info);
    let evaluations_before = registry.counts();
    let Some(&(name, implemented, feature, options)) =
        COMMANDS.iter().find(|(n, _, _, _)| *n == command)
    else {
        console.error(&format!(
            "unknown command '{command}' (commands: {})",
            command_names().join(", ")
        ));
        return EXIT_USAGE;
    };
    let (parsed, restart) = match parse_with_settings(name, options, args) {
        Ok(Parsed::Run(p, r)) => (p, r),
        Ok(Parsed::Dumped(path)) => {
            console.info(&format!("{name}: settings written to {path}"));
            return EXIT_OK;
        }
        Err(e) => {
            console.error(&format!("{name}: {e}"));
            return EXIT_USAGE;
        }
    };
    if parsed.help_requested {
        console.info(&help(name, options, implemented, feature));
        return EXIT_OK;
    }
    if !implemented {
        console.error(&format!(
            "`{name}` ({feature}) is not implemented yet in mopt-rs {}; its options are parsed only.",
            crate::VERSION
        ));
        return EXIT_USAGE;
    }
    let result = match name {
        "optmain" => optmain::OptmainConfig::from_args(&parsed)
            .and_then(|mut c| {
                if restart.is_some() && c.inputs.len() != 1 {
                    return Err(MoptError::parse("--restart continues one run (one input)"));
                }
                c.restart_dir = restart.clone();
                let level = if c.debug { Level::Debug } else { Level::Info };
                optmain::run_with(&c, Path::new("."), Some(level), registry)
            })
            .map(|runs| runs.iter().any(|r| r.interrupted)),
        "freqmain" => freqmain::FreqmainConfig::from_args(&parsed)
            .and_then(|mut c| {
                if restart.is_some() && c.inputs.len() != 1 {
                    return Err(MoptError::parse("--restart continues one run (one input)"));
                }
                c.restart_dir = restart.clone();
                freqmain::run_with(&c, Path::new("."), Some(Level::Info), registry)
            })
            .map(|_| false),
        "etmain" => etmain::EtmainConfig::from_args(&parsed)
            .and_then(|mut c| {
                if restart.is_some() && c.base.inputs.len() != 1 {
                    return Err(MoptError::parse("--restart continues one run (one input)"));
                }
                c.base.restart_dir = restart.clone();
                etmain::run_with(&c, Path::new("."), Some(Level::Info), registry)
            })
            .map(|_| false),
        "ircmain" => ircmain::IrcmainConfig::from_args(&parsed)
            .and_then(|mut c| {
                if restart.is_some() && c.inputs.len() != 1 {
                    return Err(MoptError::parse("--restart continues one run (one input)"));
                }
                c.restart_dir = restart.clone();
                ircmain::run_with(&c, Path::new("."), Some(Level::Info), registry)
            })
            .map(|runs| runs.iter().any(|r| r.irc.interrupted)),
        "nebmain" => nebmain::NebmainConfig::from_args(&parsed)
            .and_then(|mut c| {
                c.restart_dir = restart.clone();
                let level = if parsed.get_bool("debug", false) {
                    Level::Debug
                } else {
                    Level::Info
                };
                nebmain::run_with(&c, Path::new("."), Some(level), registry)
            })
            .map(|run| run.interrupted),
        "confsearch" => confsearch::ConfsearchConfig::from_args(&parsed)
            .and_then(|mut c| {
                c.restart_dir = restart.clone();
                confsearch::run_with(&c, Path::new("."), Some(Level::Info), registry)
            })
            .map(|run| run.interrupted),
        "run_autots" => autots::AutotsConfig::from_args(&parsed)
            .and_then(|mut c| {
                c.restart_dir = restart.clone();
                autots::run_with(&c, Path::new("."), Some(Level::Info), registry)
            })
            .map(|_| false),
        "run_mapper" => match parsed.get_opt_str("report") {
            // The report of a run again from its files: nothing is computed.
            Some(dir) => mapper_report::rewrite(Path::new(dir), parsed.get_list("report_options"))
                .map(|(file, notes)| {
                    for note in &notes {
                        console.warn(&format!("run_mapper --report: {note}"));
                    }
                    console.info(&format!(
                        "run_mapper: the report is {}{}",
                        file.display(),
                        if notes.is_empty() {
                            String::new()
                        } else {
                            format!(
                                " ({} note(s) of missing information, listed on the page)",
                                notes.len()
                            )
                        }
                    ));
                    false
                }),
            None => mapper::MapperConfig::from_args(&parsed, restart.as_deref())
                .and_then(|c| {
                    mapper::run_with_parts(
                        &c,
                        Path::new("."),
                        Some(c.console),
                        registry,
                        parts,
                        &mut [],
                    )
                })
                .map(|run| run.interrupted),
        },
        "run_automecp" => automecp::AutomecpConfig::from_args(&parsed)
            .and_then(|mut c| {
                c.restart_dir = restart.clone();
                automecp::run_with(&c, Path::new("."), Some(Level::Info), registry)
            })
            .map(|run| {
                // Crossing points are nodes; the summary has a line per crossing edge.
                let found = run.mapper.network.crossing_points();
                console.info(&format!(
                    "run_automecp: {found} crossing point(s) with their paths; {} lists them, \
                     {} shows the search, and {} the run of the mapper behind it.",
                    run.summary.display(),
                    run.report.display(),
                    run.mapper
                        .directory
                        .join(mapper_report::REPORT_FILE)
                        .display()
                ));
                run.mapper.interrupted
            }),
        "run_kinetics" => run_kinetics::KineticsConfig::from_args(&parsed)
            .and_then(|c| run_kinetics::run_with(&c, Path::new("."), Some(c.console)))
            .map(|_| false),
        "run_rcmc" => run_kinetics::KineticsConfig::from_rcmc_args(&parsed)
            .and_then(|c| run_kinetics::run_with(&c, Path::new("."), Some(c.console)))
            .map(|_| false),
        "ieipmain" => ieipmain::IeipmainConfig::from_args(&parsed)
            .and_then(|mut c| {
                c.restart_dir = restart.clone();
                ieipmain::run_with(&c, Path::new("."), Some(Level::Info), registry)
            })
            .map(|run| run.interrupted),
        "mdmain" => mdmain::MdmainConfig::from_args(&parsed)
            .and_then(|mut c| {
                c.restart_dir = restart.clone();
                let level = if parsed.get_bool("debug", false) {
                    Level::Debug
                } else {
                    Level::Info
                };
                mdmain::run_with(&c, Path::new("."), Some(level), registry)
            })
            .map(|runs| runs.iter().any(|r| r.interrupted)),
        "orientsearch" => orientsearch::OrientsearchConfig::from_args(&parsed)
            .and_then(|mut c| {
                c.restart_dir = restart.clone();
                orientsearch::run_with(&c, Path::new("."), Some(Level::Info), registry)
            })
            .map(|run| run.interrupted),
        "relaxedscan" => relaxedscan::RelaxedscanConfig::from_args(&parsed)
            .and_then(|mut c| {
                c.restart_dir = restart.clone();
                let level = if c.optmain.debug {
                    Level::Debug
                } else {
                    Level::Info
                };
                relaxedscan::run_with(&c, Path::new("."), Some(level), registry)
            })
            .map(|run| run.interrupted),
        _ => Err(MoptError::unsupported(format!(
            "`{name}` has no driver in this build"
        ))),
    };
    let evaluations = registry.counts().since(&evaluations_before);
    if evaluations != EvaluationCounts::default() {
        console.info(&evaluations.line("in total"));
    }
    match result {
        Ok(false) => {
            console.info("All calculations were completed.");
            EXIT_OK
        }
        Ok(true) => {
            console.info("The run was interrupted; `--restart RUN_DIR` continues it.");
            EXIT_INTERRUPTED
        }
        Err(e) => {
            console.error(&format!("{name}: {e}"));
            exit_code(&e)
        }
    }
}

enum Parsed {
    Run(core_infra_args::ParsedArgs, Option<std::path::PathBuf>),
    Dumped(String),
}

/// Command line and settings file (`--config`) combined; `--dump-config` writes them instead
/// of running.
fn parse_with_settings(
    name: &str,
    options: &[option::CliOption],
    args: &[String],
) -> Result<Parsed, MoptError> {
    let flags = settings::split_settings_flags(args)?;
    let cli = parser(name, options).parse(&flags.args)?;
    let restart = flags.restart.as_ref().map(std::path::PathBuf::from);
    let config = match (&flags.config, &restart) {
        (Some(_), Some(_)) => {
            return Err(MoptError::parse(
                "--restart reads the settings of the run directory; give no --config",
            ))
        }
        (Some(f), None) => Some(std::path::PathBuf::from(f)),
        (None, Some(d)) => Some(d.join(settings::RUN_SETTINGS_FILE)),
        (None, None) => None,
    };
    let parsed = match &config {
        Some(f) => settings::merge(settings::read_settings(f, options)?, cli),
        None => cli,
    };
    if let Some(path) = flags.dump {
        let doc = settings::settings_json(&parsed, options).to_string_pretty();
        std::fs::write(&path, doc + "\n")?;
        return Ok(Parsed::Dumped(path));
    }
    Ok(Parsed::Run(parsed, restart))
}

/// Entry point for `argv` (including the program name): the command is the program name itself
/// (`optmain ...`) or the first argument (`mopt optmain ...`).
pub fn main_from_argv(argv: &[String]) -> i32 {
    main_from_argv_with(argv, &CalculatorRegistry::native())
}

/// [`main_from_argv`] with the calculators of `registry`.
pub fn main_from_argv_with(argv: &[String], registry: &CalculatorRegistry) -> i32 {
    crate::core_infra::parallelism::deterministic_linear_algebra();
    let program = argv
        .first()
        .and_then(|p| Path::new(p).file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    if command_names().contains(&program.as_str()) {
        return run_command_with(&program, &argv[1..], registry);
    }
    match argv.get(1).map(String::as_str) {
        Some("--version" | "-V") => {
            println!("mopt-rs {}", crate::VERSION);
            EXIT_OK
        }
        Some(cmd) if !cmd.starts_with('-') => run_command_with(cmd, &argv[2..], registry),
        _ => {
            println!(
                "usage: {program} <command> [options]\n\nmopt-rs {} (successor implementation of \
                 MultiOptPy)\n\ncommands:",
                crate::VERSION
            );
            for (n, done, what, _) in COMMANDS {
                let state = if *done { "" } else { "  (not implemented yet)" };
                println!("  {n}: {what}{state}");
            }
            println!("\n`{program} <command> -h` lists the options of a command.");
            if argv.len() > 1 {
                EXIT_OK
            } else {
                EXIT_USAGE
            }
        }
    }
}

/// Whether `command` is implemented in this version.
pub fn implemented(command: &str) -> bool {
    COMMANDS
        .iter()
        .any(|(n, done, _, _)| *n == command && *done)
}
