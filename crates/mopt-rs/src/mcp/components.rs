//! The synchronous tools for components: the commands, the help and the schema of a command,
//! the calculators of the registry, a single point of one structure, and the structure files
//! of the tools of [`super::compute`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::panic::AssertUnwindSafe;

use super::inputs;
use super::paths::{path_like_parts, Sandbox};
use super::policy::Policy;
use super::protocol::{count, object, text};
use super::schema::{command_tool, option_schema, CommandTable, Listing};
use crate::cli::option::help;
use crate::core_infra::json::JsonValue;
use crate::domain::{EvalRequest, SpinCharge};
use crate::io::xyz::parse_xyz_str;
use crate::pes::objective::ObjectiveRegistry;
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry, NATIVE_CALCULATORS};

/// The commands of `commands` with whether they are implemented and the tool of each.
pub fn list_commands(commands: CommandTable<'_>) -> JsonValue {
    let entries = commands
        .iter()
        .map(|(name, implemented, feature, options)| {
            let input = inputs::positional(name);
            object([
                ("command", text(name)),
                ("implemented", JsonValue::Bool(*implemented)),
                ("feature", text(feature)),
                ("options", count(options.len())),
                (
                    "implemented_options",
                    count(options.iter().filter(|o| o.implemented).count()),
                ),
                ("input", text(format!("{} ({})", input.name, input.help))),
                (
                    "tool",
                    if *implemented {
                        text(name)
                    } else {
                        JsonValue::Null
                    },
                ),
            ])
        })
        .collect();
    object([("commands", JsonValue::Array(entries))])
}

/// The most options one query of `command_help` returns.
pub const HELP_MATCHES: usize = 40;

/// The options of `options` that the query `q` names: its long name or a flag exactly, or else
/// those whose names hold it (without dashes, ignoring case).
fn matching<'a>(
    options: &'a [crate::cli::option::CliOption],
    q: &str,
) -> Vec<&'a crate::cli::option::CliOption> {
    let bare = q.trim_start_matches('-').to_ascii_lowercase();
    let names = |o: &crate::cli::option::CliOption| -> Vec<String> {
        std::iter::once(o.spec.key.to_ascii_lowercase())
            .chain(
                o.spec
                    .flags
                    .iter()
                    .map(|f| f.trim_start_matches('-').to_ascii_lowercase()),
            )
            .collect()
    };
    let exact: Vec<_> = options
        .iter()
        .filter(|o| names(o).contains(&bare))
        .collect();
    if !exact.is_empty() || bare.is_empty() {
        return exact;
    }
    options
        .iter()
        .filter(|o| names(o).iter().any(|n| n.contains(&bare)))
        .take(HELP_MATCHES)
        .collect()
}

/// The help of `command` (the arguments `args` of `command_help`): its `--help` text, or the
/// options that `options` names with their schemas, and with `schema` the full input schema of
/// its tool.
pub fn command_help(commands: CommandTable<'_>, args: &JsonValue) -> Result<JsonValue, String> {
    let command = args
        .get("command")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| "give the command".to_string())?;
    let (name, implemented, feature, options) = commands
        .iter()
        .find(|(n, _, _, _)| *n == command)
        .ok_or_else(|| format!("no command '{command}'"))?;
    let mut map = std::collections::BTreeMap::new();
    map.insert("command".to_string(), text(name));
    let queries = match args.get("options") {
        None | Some(JsonValue::Null) => None,
        Some(v) => Some(
            v.as_string_vec()
                .filter(|q| q.len() <= 100)
                .ok_or_else(|| "options is a list of at most 100 option names".to_string())?,
        ),
    };
    match queries {
        None => {
            map.insert(
                "help".to_string(),
                text(help(name, options, *implemented, feature)),
            );
        }
        Some(queries) => {
            let mut found = Vec::new();
            for q in &queries {
                let hits = matching(options, q);
                let entries: Vec<JsonValue> = hits
                    .iter()
                    .map(|o| {
                        let mut e = std::collections::BTreeMap::new();
                        e.insert("key".to_string(), text(o.spec.key));
                        e.insert(
                            "flags".to_string(),
                            JsonValue::Array(o.spec.flags.iter().map(text).collect()),
                        );
                        e.insert("help".to_string(), text(o.spec.help.trim()));
                        e.insert("default".to_string(), text(o.default));
                        e.insert("implemented".to_string(), JsonValue::Bool(o.implemented));
                        if o.implemented {
                            e.insert("schema".to_string(), option_schema(o));
                        } else {
                            e.insert("feature".to_string(), text(o.feature));
                        }
                        JsonValue::Object(e)
                    })
                    .collect();
                found.push(object([
                    ("query", text(q)),
                    ("options", JsonValue::Array(entries)),
                ]));
            }
            map.insert("matches".to_string(), JsonValue::Array(found));
        }
    }
    if args.get("schema").and_then(JsonValue::as_bool) == Some(true) {
        let full = command_tool(name, feature, options, Listing::Full);
        map.insert(
            "inputSchema".to_string(),
            full.get("inputSchema").cloned().unwrap_or(JsonValue::Null),
        );
    }
    Ok(JsonValue::Object(map))
}

/// The text of the structure file `path` under the root (at most [`STRUCTURE_LIMIT`] bytes),
/// checked and opened through the sandbox.
pub fn read_structure(sandbox: &Sandbox, path: &str) -> Result<String, String> {
    let file = sandbox.existing(path)?;
    let (opened, meta) = sandbox.open_regular(&file)?;
    if meta.len() > STRUCTURE_LIMIT {
        return Err(format!("{path} has more than {STRUCTURE_LIMIT} bytes"));
    }
    let mut text = String::new();
    std::io::Read::read_to_string(&mut std::io::Read::take(opened, STRUCTURE_LIMIT), &mut text)
        .map_err(|e| format!("{path}: {e}"))?;
    Ok(text)
}

/// The structure texts of the tool `name` of [`super::compute`] from `args`: each structure
/// argument as a file under the root or as its text, never both.
pub fn structure_texts(
    sandbox: &Sandbox,
    name: &str,
    args: &JsonValue,
) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut out = std::collections::BTreeMap::new();
    for (file, xyz) in super::compute::structure_arguments(name) {
        let given = (
            args.get(file).and_then(JsonValue::as_str),
            args.get(xyz).and_then(JsonValue::as_str),
        );
        match given {
            (Some(path), None) => {
                out.insert(file.to_string(), read_structure(sandbox, path)?);
            }
            (None, Some(content)) => {
                out.insert(file.to_string(), content.to_string());
            }
            (Some(_), Some(_)) => {
                return Err(format!("give {file} or {xyz}, not both"));
            }
            (None, None) => {}
        }
        for key in [file, xyz] {
            if args
                .get(key)
                .is_some_and(|v| !v.is_null() && v.as_str().is_none())
            {
                return Err(format!("{key} is a string"));
            }
        }
    }
    Ok(out)
}

/// The calculators of `registry`.
pub fn list_calculators(registry: &CalculatorRegistry) -> JsonValue {
    object([
        (
            "calculators",
            JsonValue::Array(registry.names().iter().map(text).collect()),
        ),
        (
            "built_in",
            JsonValue::Array(NATIVE_CALCULATORS.iter().map(text).collect()),
        ),
        (
            "objectives",
            JsonValue::Array(
                ObjectiveRegistry::builtin()
                    .names()
                    .iter()
                    .map(text)
                    .collect(),
            ),
        ),
        (
            "note",
            text(
                "these are the calculators of the Rust binary; the quantum-chemistry programs \
                 (PySCF, Psi4, tblite, xtb, ASE calculators, neural network potentials) are \
                 those of the Python package: use its server, python -m mopt_rs.mcp",
            ),
        ),
    ])
}

fn integer(args: &JsonValue, key: &str, default: i64) -> Result<i64, String> {
    match args.get(key) {
        None | Some(JsonValue::Null) => Ok(default),
        Some(JsonValue::Number(n)) if n.fract() == 0.0 => Ok(*n as i64),
        Some(_) => Err(format!("{key} is an integer")),
    }
}

/// The largest structure file a single point reads.
pub const STRUCTURE_LIMIT: u64 = 16 << 20;
/// The most atoms of a single point, which the server computes while it waits (larger
/// structures are jobs).
pub const SINGLE_POINT_ATOMS: usize = 2000;
/// The most atoms of a single point with the Hessian (its 9 N^2 numbers are the answer).
pub const SINGLE_POINT_HESSIAN_ATOMS: usize = 300;

/// Energy and gradient of the structure of `args` with the calculator of `args`.
pub fn single_point(
    sandbox: &Sandbox,
    policy: &Policy,
    registry: &CalculatorRegistry,
    args: &JsonValue,
) -> Result<JsonValue, String> {
    let calculator = args
        .get("calculator")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| "give the calculator (see list_calculators)".to_string())?;
    policy.check_calculator(calculator)?;
    let content = match (
        args.get("structure").and_then(JsonValue::as_str),
        args.get("xyz").and_then(JsonValue::as_str),
    ) {
        (Some(path), None) => read_structure(sandbox, path)?,
        (None, Some(xyz)) => xyz.to_string(),
        _ => {
            return Err(
                "give the structure as one of structure (a file) or xyz (its text)".to_string(),
            )
        }
    };
    let state = SpinCharge::checked(
        integer(args, "charge", 0)?,
        integer(args, "multiplicity", 1)?,
    )
    .map_err(|e| e.to_string())?;
    let system = parse_xyz_str(&content, state, None).map_err(|e| e.to_string())?;
    let wants_hessian = args
        .get("hessian")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let limit = if wants_hessian {
        SINGLE_POINT_HESSIAN_ATOMS
    } else {
        SINGLE_POINT_ATOMS
    };
    if system.n_atoms() > limit {
        return Err(format!(
            "a single point takes at most {limit} atoms{}; run larger structures as jobs \
             (freqmain for Hessians)",
            if wants_hessian {
                " with the Hessian"
            } else {
                ""
            }
        ));
    }
    let mut options = CalculatorOptions::default();
    if let Some(settings) = args.get("settings") {
        let map = settings
            .as_object()
            .ok_or_else(|| "settings is an object of KEY: VALUE".to_string())?;
        for (key, value) in map {
            let v = match value {
                JsonValue::String(s) => s.clone(),
                JsonValue::Number(n) => format!("{n}"),
                JsonValue::Bool(b) => b.to_string(),
                _ => return Err(format!("settings {key}: a string, a number or a boolean")),
            };
            let token = format!("{key}={v}");
            policy.check_option("calc_option", &[token.as_str()])?;
            for part in path_like_parts(&token) {
                sandbox.check_value_part(sandbox.root(), part)?;
            }
            options.extra.push((key.clone(), v));
        }
    }
    let hessian = args
        .get("hessian")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let request = if hessian {
        EvalRequest::ENERGY_GRAD_HESS
    } else {
        EvalRequest::ENERGY_GRAD
    };
    // A failure inside a calculator must not end the server.
    let evaluated = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let calc = registry.create(calculator, &options, &system)?;
        let out = calc.evaluate(&system, &request)?;
        Ok::<_, crate::error::MoptError>((out, calc.take_warnings()))
    }))
    .map_err(|_| format!("the calculator {calculator} stopped with an internal error"))?;
    let (out, warnings) = evaluated.map_err(|e| e.to_string())?;
    let g = &out.gradient_bohr;
    let n = system.n_atoms();
    let rows: Vec<JsonValue> = (0..n)
        .map(|i| JsonValue::Array((0..3).map(|k| JsonValue::Number(g[3 * i + k])).collect()))
        .collect();
    let max = (0..g.nrows()).map(|k| g[k].abs()).fold(0.0, f64::max);
    let rms = if g.nrows() > 0 {
        ((0..g.nrows()).map(|k| g[k] * g[k]).sum::<f64>() / g.nrows() as f64).sqrt()
    } else {
        0.0
    };
    let mut map = std::collections::BTreeMap::new();
    map.insert("calculator".to_string(), text(calculator));
    map.insert(
        "elements".to_string(),
        JsonValue::Array(system.elements.iter().map(|e| text(e.symbol())).collect()),
    );
    map.insert(
        "energy_hartree".to_string(),
        JsonValue::Number(out.energy.value()),
    );
    map.insert(
        "gradient_hartree_per_bohr".to_string(),
        JsonValue::Array(rows),
    );
    map.insert("max_abs_gradient".to_string(), JsonValue::Number(max));
    map.insert("rms_gradient".to_string(), JsonValue::Number(rms));
    if let Some(h) = &out.hessian_bohr {
        let flat: Vec<JsonValue> = (0..h.nrows())
            .flat_map(|i| (0..h.ncols()).map(move |j| (i, j)))
            .map(|(i, j)| JsonValue::Number(h[(i, j)]))
            .collect();
        map.insert(
            "hessian_hartree_per_bohr2".to_string(),
            JsonValue::Array(flat),
        );
    }
    map.insert(
        "warnings".to_string(),
        JsonValue::Array(warnings.iter().map(text).collect()),
    );
    Ok(JsonValue::Object(map))
}
