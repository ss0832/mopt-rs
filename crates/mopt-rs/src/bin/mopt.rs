//! `mopt` — command-line entry point of mopt-rs: `mopt <command> [options]`, or one of the ten
//! command names when the binary is invoked under that name (`optmain ...`).
//!
//! MultiOptPy: `Entrypoints/core.py` (the command entry points).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![forbid(unsafe_code)]

/// The pages of the documentation (`docs/source/*.md`), which `mopt mcp` serves as resources
/// wherever the binary runs (`crates/mopt-rs/tests/mcp_server.rs` checks that every page is
/// here).
const DOCS: &[(&str, &str)] = &[
    (
        "architecture",
        include_str!("../../../../docs/source/architecture.md"),
    ),
    (
        "benchmarks",
        include_str!("../../../../docs/source/benchmarks.md"),
    ),
    (
        "bias_potentials",
        include_str!("../../../../docs/source/bias_potentials.md"),
    ),
    (
        "calculators",
        include_str!("../../../../docs/source/calculators.md"),
    ),
    (
        "cli_reference",
        include_str!("../../../../docs/source/cli_reference.md"),
    ),
    (
        "coordinate_systems",
        include_str!("../../../../docs/source/coordinate_systems.md"),
    ),
    (
        "coverage",
        include_str!("../../../../docs/source/coverage.md"),
    ),
    (
        "electron_transfer",
        include_str!("../../../../docs/source/electron_transfer.md"),
    ),
    (
        "excited_states",
        include_str!("../../../../docs/source/excited_states.md"),
    ),
    (
        "high_index_saddles",
        include_str!("../../../../docs/source/high_index_saddles.md"),
    ),
    ("irc", include_str!("../../../../docs/source/irc.md")),
    (
        "kinetics",
        include_str!("../../../../docs/source/kinetics.md"),
    ),
    (
        "kinetics_beyond_tst",
        include_str!("../../../../docs/source/kinetics_beyond_tst.md"),
    ),
    ("mapper", include_str!("../../../../docs/source/mapper.md")),
    (
        "mapper_design",
        include_str!("../../../../docs/source/mapper_design.md"),
    ),
    ("mcp", include_str!("../../../../docs/source/mcp.md")),
    (
        "migration",
        include_str!("../../../../docs/source/migration.md"),
    ),
    (
        "migration_optimizers",
        include_str!("../../../../docs/source/migration_optimizers.md"),
    ),
    (
        "migration_paths",
        include_str!("../../../../docs/source/migration_paths.md"),
    ),
    (
        "migration_safeguards",
        include_str!("../../../../docs/source/migration_safeguards.md"),
    ),
    (
        "migration_surfaces",
        include_str!("../../../../docs/source/migration_surfaces.md"),
    ),
    (
        "migration_vibrations",
        include_str!("../../../../docs/source/migration_vibrations.md"),
    ),
    (
        "migration_workflows",
        include_str!("../../../../docs/source/migration_workflows.md"),
    ),
    (
        "molecular_dynamics",
        include_str!("../../../../docs/source/molecular_dynamics.md"),
    ),
    ("oniom", include_str!("../../../../docs/source/oniom.md")),
    (
        "open_issues",
        include_str!("../../../../docs/source/open_issues.md"),
    ),
    (
        "optimization",
        include_str!("../../../../docs/source/optimization.md"),
    ),
    (
        "python_workflows",
        include_str!("../../../../docs/source/python_workflows.md"),
    ),
    (
        "quickstart",
        include_str!("../../../../docs/source/quickstart.md"),
    ),
    (
        "reaction_paths",
        include_str!("../../../../docs/source/reaction_paths.md"),
    ),
    (
        "running",
        include_str!("../../../../docs/source/running.md"),
    ),
    (
        "searches",
        include_str!("../../../../docs/source/searches.md"),
    ),
    (
        "vibrations",
        include_str!("../../../../docs/source/vibrations.md"),
    ),
];

fn main() {
    // Before anything computes: results that do not depend on the number of threads.
    mopt_rs::core_infra::parallelism::deterministic_linear_algebra();
    let argv: Vec<String> = std::env::args().collect();
    // `mopt mcp [--root DIR]`: the Model Context Protocol server over stdio.
    if argv.get(1).map(String::as_str) == Some("mcp") {
        std::process::exit(mopt_rs::mcp::main_with_docs(&argv[2..], DOCS));
    }
    std::process::exit(mopt_rs::cli::main_from_argv(&argv));
}
