//! What Rust offers, Python offers too: the Rust side of `tools/api/parity.json`.
//!
//! The table pairs modules and items of this library with the Python names that reach them,
//! lists what exists on one side by nature (with the reason), and what Python cannot reach yet.
//! This test checks that every module of the library is accounted for, that every Rust path of
//! the table names a module or a public item that exists, and that every list has reasons.
//! `tests/test_api_parity.py` checks the Python names and that the registries work from Python.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use mopt_rs::core_infra::json::JsonValue;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source() -> PathBuf {
    root().join("crates/mopt-rs/src")
}

fn table() -> JsonValue {
    let text = std::fs::read_to_string(root().join("tools/api/parity.json")).unwrap();
    JsonValue::parse(&text).unwrap()
}

/// Modules of the library down to the second level (`opt`, `opt::step`, ...).
fn modules() -> Vec<String> {
    let mut out = Vec::new();
    let mut tops: Vec<PathBuf> = std::fs::read_dir(source())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    tops.sort();
    for top in tops {
        let stem = top.file_stem().unwrap().to_string_lossy().into_owned();
        if top.is_file() {
            if top.extension().is_some_and(|e| e == "rs") && stem != "lib" && stem != "mod" {
                out.push(stem);
            }
            continue;
        }
        out.push(stem.clone());
        let mut children: Vec<PathBuf> = std::fs::read_dir(&top)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        children.sort();
        for child in children {
            let name = child.file_stem().unwrap().to_string_lossy().into_owned();
            let is_module = child.is_dir() || child.extension().is_some_and(|e| e == "rs");
            if is_module && name != "mod" {
                out.push(format!("{stem}::{name}"));
            }
        }
    }
    out
}

fn accounts_for(key: &str, module: &str) -> bool {
    key == module
        || key.starts_with(&format!("{module}::"))
        || module.starts_with(&format!("{key}::"))
}

/// Source files of the module `path` (`a::b` is `a/b.rs` or `a/b/mod.rs`; empty: `lib.rs`).
fn module_files(path: &[&str]) -> Vec<PathBuf> {
    if path.is_empty() {
        return vec![source().join("lib.rs")];
    }
    let dir = path.iter().fold(source(), |p, s| p.join(s));
    [dir.with_extension("rs"), dir.join("mod.rs")]
        .into_iter()
        .filter(|f| f.is_file())
        .collect()
}

/// Whether `path` is a module, or a public item (also a re-export) of the module before it.
fn exists(path: &str) -> bool {
    let parts: Vec<&str> = path.split("::").collect();
    // A module: a file, a directory with `mod.rs`, or a directory of binaries.
    if !module_files(&parts).is_empty() || parts.iter().fold(source(), |p, s| p.join(s)).is_dir() {
        return true;
    }
    let (item, module) = parts.split_last().unwrap();
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    module_files(module).iter().any(|file| {
        let text = std::fs::read_to_string(file).unwrap();
        let declared = text.lines().any(|line| {
            let line = line.trim_start();
            ["struct", "enum", "trait", "fn", "type", "const", "static"]
                .iter()
                .any(|kind| {
                    line.strip_prefix(&format!("pub {kind} {item}"))
                        .is_some_and(|rest| !rest.starts_with(is_word))
                })
        });
        // A re-export: `pub use a::{B, C};`, possibly over several lines.
        let reexported = text.split(';').any(|statement| {
            statement.trim_start().lines().last().is_some()
                && statement
                    .rsplit("pub use")
                    .next()
                    .filter(|_| statement.contains("pub use"))
                    .is_some_and(|names| names.split(|c: char| !is_word(c)).any(|w| w == *item))
        });
        declared || reexported
    })
}

fn keys(value: &JsonValue, name: &str) -> Vec<String> {
    value
        .get(name)
        .and_then(JsonValue::as_object)
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

fn rust_paths(table: &JsonValue) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (feature, entry) in table
        .get("components")
        .and_then(JsonValue::as_object)
        .unwrap()
    {
        for path in entry
            .get("rust")
            .and_then(JsonValue::as_string_vec)
            .unwrap()
        {
            out.push((feature.clone(), path));
        }
    }
    out
}

#[test]
fn every_module_is_accounted_for() {
    let table = table();
    let mut all: Vec<String> = rust_paths(&table).into_iter().map(|(_, p)| p).collect();
    all.extend(keys(&table, "pending"));
    all.extend(keys(&table, "rust_only"));
    let missing: Vec<String> = modules()
        .into_iter()
        .filter(|m| !all.iter().any(|k| accounts_for(k, m)))
        .collect();
    assert!(
        missing.is_empty(),
        "add these modules to tools/api/parity.json: {missing:?}"
    );
}

#[test]
fn rust_paths_of_the_table_exist() {
    let table = table();
    for (feature, path) in rust_paths(&table) {
        assert!(
            exists(&path),
            "'{feature}': no module or public item '{path}'"
        );
    }
    for name in ["pending", "rust_only"] {
        for path in keys(&table, name) {
            assert!(exists(&path), "{name}: no module '{path}'");
        }
    }
    // Every component names something on both sides, and every exception has its reason.
    for (feature, entry) in table
        .get("components")
        .and_then(JsonValue::as_object)
        .unwrap()
    {
        for side in ["rust", "python"] {
            let names = entry.get(side).and_then(JsonValue::as_string_vec).unwrap();
            assert!(!names.is_empty(), "'{feature}' names nothing in {side}");
        }
    }
    for name in ["pending", "rust_only", "python_only"] {
        for (key, reason) in table.get(name).and_then(JsonValue::as_object).unwrap() {
            assert!(
                reason.as_str().is_some_and(|r| !r.trim().is_empty()),
                "{name}: '{key}' has no reason"
            );
        }
    }
}

#[test]
fn the_check_itself_notices_missing_items() {
    assert!(exists("opt::vector::VectorOptimizer"));
    assert!(exists("pes::native"));
    assert!(exists("VERSION"));
    assert!(!exists("opt::vector::NoSuchOptimizer"));
    assert!(!exists("no_such_module"));
    assert!(accounts_for("opt::step::StepGenerator", "opt::step"));
    assert!(accounts_for("cli", "cli::spec"));
    assert!(!accounts_for("opt::step", "opt::update"));
}
