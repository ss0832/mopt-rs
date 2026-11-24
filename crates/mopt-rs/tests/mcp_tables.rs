//! The tools of the MCP server cover the option tables, and the tables of the Python server
//! (`python/mopt_rs/mcp/commands.json`, written by `tools/cli/gen_cli.py`) are those of
//! `cli/spec.rs`.
//!
//! Every implemented command must be a tool and every implemented option a property of its
//! full schema (`--full-schemas`, and `command_help` with `schema`); options that are not
//! implemented must not be offered. The compact schemas of `tools/list` take the other options
//! by name. The JSON tables must hold exactly the commands and options of `spec.rs` and the
//! positional inputs of `cli::spec::POSITIONALS`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use mopt_rs::cli::spec::COMMANDS;
use mopt_rs::core_infra::arg_parser::ArgArity;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::mcp::compute;
use mopt_rs::mcp::inputs;
use mopt_rs::mcp::schema::{
    self, Listing, COMMON_OPTIONS, COMPONENT_TOOLS, FILES_PROPERTY, INPUT_PROPERTY, JOB_TOOLS,
};

fn tools() -> Vec<JsonValue> {
    schema::all_tools(COMMANDS, Listing::Full)
}

fn properties(tool: &JsonValue) -> &std::collections::BTreeMap<String, JsonValue> {
    tool.get("inputSchema")
        .and_then(|s| s.get("properties"))
        .and_then(JsonValue::as_object)
        .unwrap()
}

#[test]
fn every_implemented_command_and_option_is_in_a_tool() {
    let tools = tools();
    let mut gaps = Vec::new();
    let mut offered_but_unavailable = Vec::new();
    for (command, implemented, _, options) in COMMANDS {
        let tool = tools
            .iter()
            .find(|t| t.get("name").and_then(JsonValue::as_str) == Some(command));
        match (implemented, tool) {
            (true, None) => gaps.push(format!("command {command}")),
            (false, Some(_)) => offered_but_unavailable.push(format!("command {command}")),
            (false, None) => {}
            (true, Some(tool)) => {
                let props = properties(tool);
                for o in *options {
                    match (o.implemented, props.contains_key(o.spec.key)) {
                        (true, false) => gaps.push(format!("{command} {}", o.spec.key)),
                        (false, true) => {
                            offered_but_unavailable.push(format!("{command} {}", o.spec.key))
                        }
                        _ => {}
                    }
                    assert!(
                        o.spec.key != INPUT_PROPERTY && o.spec.key != FILES_PROPERTY,
                        "{command}: the option {} has the name of a property of the server",
                        o.spec.key
                    );
                }
                assert!(props.contains_key(INPUT_PROPERTY), "{command}");
            }
        }
    }
    assert!(gaps.is_empty(), "not offered as tools: {gaps:?}");
    assert!(
        offered_but_unavailable.is_empty(),
        "offered although not implemented: {offered_but_unavailable:?}"
    );
}

#[test]
fn tool_names_are_unique_and_valid() {
    let names: Vec<String> = tools()
        .iter()
        .map(|t| {
            t.get("name")
                .and_then(JsonValue::as_str)
                .unwrap()
                .to_string()
        })
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len(), "{names:?}");
    for n in &names {
        assert!(
            !n.is_empty()
                && n.len() <= 128
                && n.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "_-.".contains(c)),
            "{n}"
        );
    }
    for fixed in JOB_TOOLS
        .iter()
        .chain(COMPONENT_TOOLS.iter())
        .chain(compute::TOOLS.iter())
    {
        assert!(names.iter().any(|n| n == fixed), "{fixed}");
    }
    // Every schema is an object schema that refuses unknown properties.
    for t in tools() {
        let s = t.get("inputSchema").unwrap();
        assert_eq!(s.get("type").and_then(JsonValue::as_str), Some("object"));
        assert_eq!(s.get("additionalProperties"), Some(&JsonValue::Bool(false)));
    }
}

#[test]
fn the_compact_listing_names_the_inputs_and_takes_the_other_options_by_name() {
    let compact = schema::all_tools(COMMANDS, Listing::Compact);
    let full = tools();
    let names = |v: &[JsonValue]| -> Vec<String> {
        v.iter()
            .map(|t| {
                t.get("name")
                    .and_then(JsonValue::as_str)
                    .unwrap()
                    .to_string()
            })
            .collect()
    };
    assert_eq!(
        names(&compact),
        names(&full),
        "the same tools in both listings"
    );
    let size = |v: &[JsonValue]| JsonValue::Array(v.to_vec()).to_string_compact().len();
    // The compact listing is a small fraction of the full one.
    assert!(
        size(&compact) * 5 < size(&full),
        "compact {} bytes, full {} bytes",
        size(&compact),
        size(&full)
    );
    for (command, implemented, _, options) in COMMANDS {
        if !implemented {
            continue;
        }
        let tool = compact
            .iter()
            .find(|t| t.get("name").and_then(JsonValue::as_str) == Some(command))
            .unwrap();
        let props = properties(tool);
        assert!(props.contains_key(INPUT_PROPERTY) && props.contains_key(FILES_PROPERTY));
        for key in props.keys() {
            assert!(
                key == INPUT_PROPERTY
                    || key == FILES_PROPERTY
                    || COMMON_OPTIONS.contains(&key.as_str()),
                "{command}: {key}"
            );
        }
        for o in options.iter().filter(|o| o.implemented) {
            if COMMON_OPTIONS.contains(&o.spec.key) {
                assert!(props.contains_key(o.spec.key), "{command}: {}", o.spec.key);
            }
        }
        let additional = tool
            .get("inputSchema")
            .and_then(|s| s.get("additionalProperties"))
            .unwrap();
        assert!(
            additional.as_object().is_some(),
            "{command}: other options by name"
        );
        // command_help with schema gives the full schema of the tool.
        let args =
            JsonValue::parse(&format!(r#"{{"command":"{command}","schema":true}}"#)).unwrap();
        let help = mopt_rs::mcp::components::command_help(COMMANDS, &args).unwrap();
        let full_tool = full
            .iter()
            .find(|t| t.get("name").and_then(JsonValue::as_str) == Some(command))
            .unwrap();
        assert_eq!(
            help.get("inputSchema"),
            full_tool.get("inputSchema"),
            "{command}"
        );
    }
    // Options are looked up by long name, flag, or a part of either.
    let args =
        JsonValue::parse(r#"{"command":"optmain","options":["-ns","modelhess","no_such_option"]}"#)
            .unwrap();
    let help = mopt_rs::mcp::components::command_help(COMMANDS, &args).unwrap();
    let matches = help.get("matches").and_then(JsonValue::as_array).unwrap();
    let keys = |k: usize| -> Vec<String> {
        matches[k]
            .get("options")
            .and_then(JsonValue::as_array)
            .unwrap()
            .iter()
            .map(|o| {
                o.get("key")
                    .and_then(JsonValue::as_str)
                    .unwrap()
                    .to_string()
            })
            .collect()
    };
    assert_eq!(keys(0), vec!["NSTEP".to_string()]);
    assert!(
        keys(1).contains(&"use_model_hessian".to_string()),
        "{:?}",
        keys(1)
    );
    assert!(keys(2).is_empty());
    let entry = &matches[0]
        .get("options")
        .and_then(JsonValue::as_array)
        .unwrap()[0];
    assert!(entry.get("schema").is_some() && entry.get("help").is_some());
}

fn arity_name(a: ArgArity) -> (&'static str, Option<&'static str>) {
    match a {
        ArgArity::Flag => ("flag", None),
        ArgArity::Single => ("single", None),
        ArgArity::Optional(implicit) => ("optional", Some(implicit)),
        ArgArity::ZeroOrMore => ("zero_or_more", None),
        ArgArity::OneOrMore => ("one_or_more", None),
    }
}

#[test]
fn the_tables_of_the_python_server_are_those_of_spec_rs() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/mopt_rs/mcp/commands.json");
    let doc = JsonValue::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let commands = doc.get("commands").and_then(JsonValue::as_array).unwrap();
    let names: Vec<&str> = commands
        .iter()
        .map(|c| c.get("name").and_then(JsonValue::as_str).unwrap())
        .collect();
    let expected: Vec<&str> = COMMANDS.iter().map(|(n, _, _, _)| *n).collect();
    assert_eq!(names, expected, "run tools/cli/gen_cli.py");
    let mut differences = Vec::new();
    for (c, (name, implemented, feature, options)) in commands.iter().zip(COMMANDS) {
        let s = |key: &str| {
            c.get(key)
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_string()
        };
        if c.get("implemented").and_then(JsonValue::as_bool) != Some(*implemented)
            || s("feature") != *feature
        {
            differences.push(format!("{name}: implemented or feature"));
        }
        let p = c.get("positional").unwrap();
        let input = inputs::positional(name);
        let json_input = (
            p.get("name")
                .and_then(JsonValue::as_str)
                .unwrap_or_default(),
            p.get("multiple")
                .and_then(JsonValue::as_bool)
                .unwrap_or_default(),
            p.get("help")
                .and_then(JsonValue::as_str)
                .unwrap_or_default(),
        );
        if json_input != (input.name, input.multiple, input.help) {
            differences.push(format!(
                "{name}: positional {json_input:?} != {input:?} (cli/spec.rs POSITIONALS)"
            ));
        }
        let rows = c.get("options").and_then(JsonValue::as_array).unwrap();
        if rows.len() != options.len() {
            differences.push(format!(
                "{name}: {} options != {}",
                rows.len(),
                options.len()
            ));
            continue;
        }
        for (row, o) in rows.iter().zip(options.iter()) {
            let t = |key: &str| {
                row.get(key)
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .to_string()
            };
            let flags = row
                .get("flags")
                .and_then(JsonValue::as_string_vec)
                .unwrap_or_default();
            let (arity, implicit) = arity_name(o.spec.arity);
            let same = t("key") == o.spec.key
                && flags
                    == o.spec
                        .flags
                        .iter()
                        .map(|f| f.to_string())
                        .collect::<Vec<_>>()
                && t("arity") == arity
                && row.get("implicit").and_then(JsonValue::as_str) == implicit
                && t("help") == o.spec.help
                && t("default") == o.default
                && row.get("implemented").and_then(JsonValue::as_bool) == Some(o.implemented)
                && t("feature") == o.feature;
            if !same {
                differences.push(format!("{name} {}", o.spec.key));
            }
        }
    }
    assert!(
        differences.is_empty(),
        "commands.json differs from spec.rs (run tools/cli/gen_cli.py): {differences:?}"
    );
}
