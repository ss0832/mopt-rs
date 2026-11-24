//! The child side of a job: `mopt mcp run-job COMMAND [ARGS...]` runs the command exactly as
//! `mopt COMMAND [ARGS...]` does, in the job directory the server chose as its working
//! directory, and records the exit code in `mcp_exit_code` there.
//!
//! Only the commands of the option tables run; a panic of the command is caught and recorded
//! as an error (exit code 1).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::jobs::{EXIT_FILE, JOB_FILE};
use crate::cli::{command_names, run_command, EXIT_ERROR, EXIT_USAGE};

/// The argument of `mopt mcp` that runs a job.
pub const RUN_JOB: &str = "run-job";

/// Run `args` = `COMMAND [ARGS...]` in the current directory; returns the exit code.
pub fn run_job(args: &[String]) -> i32 {
    let Some((command, rest)) = args.split_first() else {
        eprintln!("mopt mcp {RUN_JOB}: give the command and its arguments");
        return EXIT_USAGE;
    };
    if !command_names().contains(&command.as_str()) {
        eprintln!(
            "mopt mcp {RUN_JOB}: '{command}' is not a command of mopt-rs ({})",
            command_names().join(", ")
        );
        return EXIT_USAGE;
    }
    let rest = rest.to_vec();
    let name = command.clone();
    let code = std::panic::catch_unwind(move || run_command(&name, &rest)).unwrap_or_else(|_| {
        eprintln!("mopt mcp {RUN_JOB}: the command {command} stopped with an internal error");
        EXIT_ERROR
    });
    // The record is written only in the directory of a job.
    if Path::new(JOB_FILE).is_file() {
        if let Err(e) = std::fs::write(EXIT_FILE, format!("{code}\n")) {
            eprintln!("mopt mcp {RUN_JOB}: {EXIT_FILE}: {e}");
        }
    }
    code
}
