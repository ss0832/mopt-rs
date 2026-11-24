//! The positional inputs of the commands.
//!
//! `tools/cli/gen_cli.py` writes them with the option tables: to
//! [`crate::cli::spec::POSITIONALS`] for the Rust server and to
//! `python/mopt_rs/mcp/commands.json` for the Python server, from the same data
//! (`crates/mopt-rs/tests/mcp_tables.rs` fails when the two differ). A command without an entry
//! gets the generic input of [`GENERIC`], so that a command that becomes implemented in the
//! tables is a tool without a change here.
//!
//! MultiOptPy: the positional arguments of the parsers in `interface.py` and `Entrypoints/`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::cli::spec::POSITIONALS;

/// The positional input of a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Positional {
    /// Its name in the parser (`INPUT`, `input_file`, ...).
    pub name: &'static str,
    /// Whether it takes several values (`nargs="*"`).
    pub multiple: bool,
    /// Its help text.
    pub help: &'static str,
}

/// The input of a command without an entry: several files or directories.
pub const GENERIC: Positional = Positional {
    name: "INPUT",
    multiple: true,
    help: "the input files or directories of the command",
};

/// The positional input of `command`.
pub fn positional(command: &str) -> Positional {
    POSITIONALS
        .iter()
        .find(|(c, _, _, _)| *c == command)
        .map_or(GENERIC, |(_, name, multiple, help)| Positional {
            name,
            multiple: *multiple,
            help,
        })
}
