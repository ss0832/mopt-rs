//! Zero-dependency internal infrastructure modules (JSON, CLI parser, PRNG, SVG/PNG plotting).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod arg_parser;
pub mod clock;
pub mod glob;
pub mod json;
pub mod log;
// The memory a dense matrix may take.
pub mod memory;
// The process-wide parallelism of the linear algebra.
pub mod parallelism;
pub mod rng;
// SQLite database files without a library.
pub mod sqlite;
pub mod viz;
