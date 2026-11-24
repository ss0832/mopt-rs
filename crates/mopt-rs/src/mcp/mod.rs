//! A Model Context Protocol (MCP) server for mopt-rs: `mopt mcp [--root DIR]`.
//!
//! The server speaks JSON-RPC 2.0 over stdio, one message per line, and offers every command
//! of the option tables ([`crate::cli::spec::COMMANDS`]) as a tool that starts a job (or a
//! task), the job tools (status, wait, stop, cancel, restart, result, list), synchronous tools
//! for components (the commands and their help and schemas, the calculators, a single point,
//! model Hessians, normal modes, coordinate systems, first images, superposition, bond graphs,
//! dispersion, Hessian updates, units, elements), the documentation, the files of run
//! directories and the status of jobs as resources (with subscriptions), and two prompts.
//!
//! - [`protocol`]: messages, error codes, protocol versions (the stateless revision
//!   `2026-07-28` with per-request `_meta`, and the revisions with the `initialize`
//!   handshake, `2025-11-25` back to `2024-11-05`).
//! - [`server`]: the loop over standard input (a reading thread, requests answered later,
//!   subscriptions, progress, cancellation) and the routing of the methods.
//! - [`schema`]: the tool definitions, generated at run time from the option tables.
//! - [`inputs`]: the positional inputs of the commands (`cli::spec::POSITIONALS`).
//! - [`tools`]: the calls of the tools.
//! - [`compute`]: the component tools that compute from structures given as text, shared with
//!   the server of the Python package.
//! - [`tasks`]: jobs as tasks (the tasks extension of `2026-07-28`, the tasks of `2025-11-25`).
//! - [`pagination`]: pages of the lists and their opaque cursors.
//! - [`jobs`]: jobs as child processes of this binary in directories under the root.
//! - [`job_settings`]: the arguments of a command tool as a settings file, and the copies of
//!   the inputs.
//! - [`run_files`]: what every command leaves in a run directory (logs, evaluation totals,
//!   files).
//! - [`runner`]: the child side of a job (`mopt mcp run-job COMMAND ...`).
//! - [`resources`]: documentation pages and files of run directories.
//! - [`components`]: the synchronous tools for components.
//! - [`prompts`]: prompt templates.
//! - [`paths`]: every path is checked by its text, resolved, checked to stay inside the root,
//!   and checked again when it is opened.
//! - [`policy`]: what is refused unless the server runs with `--allow-external` (external
//!   programs, code named by settings).
//! - [`base64`]: the encoding of binary resource contents.
//!
//! Nothing but protocol messages goes to standard output; the server logs to standard error.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod base64;
pub mod components;
pub mod compute;
pub mod inputs;
pub mod job_settings;
pub mod jobs;
pub mod pagination;
pub mod paths;
pub mod policy;
pub mod prompts;
pub mod protocol;
pub mod resources;
pub mod run_files;
pub mod runner;
pub mod schema;
pub mod server;
pub mod tasks;
pub mod tools;

use std::path::PathBuf;

/// Name of the server in `serverInfo`.
pub const SERVER_NAME: &str = "mopt-rs";

const USAGE: &str =
    "usage: mopt mcp [--root DIR] [--max-jobs N] [--allow-external] [--full-schemas]
                [--page-size N]

Model Context Protocol server of mopt-rs over stdio (JSON-RPC 2.0, one message per line).
Every implemented command is a tool that starts a job in DIR/mcp_jobs/JOB_ID/; the job
tools follow, wait for, stop, cancel, restart and read jobs; the documentation, the files of
run directories and the status of jobs are resources. Nothing outside DIR (default: the
current directory) is read or written on behalf of a request.

options:
  --root DIR        the directory the server may read and write (default: .)
  --max-jobs N      jobs of this server that may run at a time (default: 4)
  --allow-external  allow settings that start programs outside mopt-rs or load code
                    (ASE calculators, external programs, hook=, software_path.conf, ...);
                    refused by default
  --full-schemas    list every option of every command in tools/list (default: the
                    compact schemas; command_help gives the options)
  --page-size N     entries per page of the lists (default: 100)
  -h, --help        show this help message and exit";

fn usage_error(message: &str) -> i32 {
    eprintln!("mopt mcp: {message}\n\n{USAGE}");
    crate::cli::EXIT_USAGE
}

/// `mopt mcp ARGS` (without `mcp`) without documentation of its own (see [`main_with_docs`]).
pub fn main(args: &[String]) -> i32 {
    main_with_docs(args, &[])
}

/// `mopt mcp ARGS` (without `mcp`): the server with the documentation `docs` (the pages the
/// binary embeds; empty: those of the repository that holds the root), or the child side of a
/// job (`mopt mcp run-job COMMAND [ARGS...]`). Returns the exit code.
pub fn main_with_docs(args: &[String], docs: resources::EmbeddedDocs) -> i32 {
    // The jobs and the in-process tools give results that do not depend on the threads.
    crate::core_infra::parallelism::deterministic_linear_algebra();
    if args.first().map(String::as_str) == Some(runner::RUN_JOB) {
        return runner::run_job(&args[1..]);
    }
    let mut root = PathBuf::from(".");
    let mut options = server::ServerOptions::default();
    let mut k = 0;
    while k < args.len() {
        let (flag, inline) = match args[k].split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (args[k].as_str(), None),
        };
        let mut value = || -> Option<String> {
            inline.clone().or_else(|| {
                k += 1;
                args.get(k).cloned()
            })
        };
        match flag {
            "-h" | "--help" => {
                // The help goes to standard error: standard output is the protocol's.
                eprintln!("{USAGE}");
                return crate::cli::EXIT_OK;
            }
            "--root" => match value() {
                Some(dir) => root = PathBuf::from(dir),
                None => return usage_error("--root needs a directory"),
            },
            "--max-jobs" => match value().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if n >= 1 => options.max_jobs = n,
                _ => return usage_error("--max-jobs needs a number of at least 1"),
            },
            "--allow-external" => options.allow_external = true,
            "--full-schemas" => options.listing = schema::Listing::Full,
            "--page-size" => match value().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if (1..=pagination::MAX_PAGE_SIZE).contains(&n) => options.page_size = n,
                _ => {
                    return usage_error(&format!(
                        "--page-size needs a number from 1 to {}",
                        pagination::MAX_PAGE_SIZE
                    ))
                }
            },
            other => return usage_error(&format!("unknown argument '{other}'")),
        }
        k += 1;
    }
    match server::Server::new(&root, &options, docs) {
        Ok(mut server) => {
            // The input is read by a thread of the server, which needs a reader it can own.
            let stdin = std::io::BufReader::new(std::io::stdin());
            let stdout = std::io::stdout();
            match server.serve(stdin, stdout.lock()) {
                Ok(()) => crate::cli::EXIT_OK,
                Err(e) => {
                    eprintln!("mopt mcp: {e}");
                    crate::cli::EXIT_ERROR
                }
            }
        }
        Err(e) => {
            eprintln!("mopt mcp: {e}");
            crate::cli::EXIT_USAGE
        }
    }
}
