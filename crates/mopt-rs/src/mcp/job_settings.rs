//! The arguments of a command tool as a settings file of the command, and the copies of the
//! input files into the job directory.
//!
//! The arguments use the long option names of the tables (the keys of settings files; any
//! flag spelling such as `opt` or `-opt` is accepted as well); a flag is `true`, one value is
//! a string or a number, several values an array, and an optional value may be `true` for the
//! flag alone. The job runs the command with `--config mcp_settings.json`, the same document
//! as a settings file of the command line, so that the arguments mean what the flags mean.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::paths::{path_like_parts, Sandbox};
use super::policy::Policy;
use super::protocol::text;
use super::schema::{FILES_PROPERTY, INPUT_PROPERTY};
use crate::cli::option::CliOption;
use crate::cli::settings::{INPUT_KEY, VERSION_KEY};
use crate::core_infra::arg_parser::ArgArity;
use crate::core_infra::json::JsonValue;

/// The settings file of a job, read by the command with `--config`.
pub const SETTINGS_FILE: &str = "mcp_settings.json";
/// At most this many bytes are copied into one job directory.
pub const COPY_LIMIT_BYTES: u64 = 1 << 30;
/// At most this many files are copied into one job directory.
pub const COPY_LIMIT_FILES: usize = 20_000;

/// What a command tool was asked for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobRequest {
    /// The inputs (paths as given).
    pub inputs: Vec<String>,
    /// Further files (paths as given).
    pub files: Vec<String>,
    /// The options, by key, as values of a settings file.
    pub settings: BTreeMap<String, JsonValue>,
}

/// The option of `options` named `key` (its key or any of its flags, without dashes).
pub fn find_option<'a>(options: &'a [CliOption], key: &str) -> Option<&'a CliOption> {
    let bare = key.trim_start_matches('-');
    options.iter().find(|o| o.spec.key == bare).or_else(|| {
        options.iter().find(|o| {
            o.spec
                .flags
                .iter()
                .any(|f| f.trim_start_matches('-') == bare)
        })
    })
}

fn strings(value: &JsonValue, what: &str) -> Result<Vec<String>, String> {
    match value {
        JsonValue::String(s) => Ok(vec![s.clone()]),
        JsonValue::Array(items) => items
            .iter()
            .map(|i| {
                i.as_str()
                    .map(String::from)
                    .ok_or_else(|| format!("{what}: every entry is a path (a string)"))
            })
            .collect(),
        JsonValue::Null => Ok(Vec::new()),
        _ => Err(format!("{what}: a path or a list of paths")),
    }
}

/// A scalar as the text of the command line.
fn token(value: &JsonValue, key: &str) -> Result<String, String> {
    match value {
        JsonValue::String(s) => Ok(s.clone()),
        JsonValue::Number(n) if n.fract() == 0.0 && n.abs() < 1e15 => Ok(format!("{}", *n as i64)),
        JsonValue::Number(n) if n.is_finite() => Ok(format!("{n}")),
        JsonValue::Bool(b) => Ok(b.to_string()),
        _ => Err(format!("{key}: a string or a number")),
    }
}

/// The value of a settings file for the option `o` given `value` (`None`: not given).
fn setting(o: &CliOption, value: &JsonValue) -> Result<Option<JsonValue>, String> {
    let key = o.spec.key;
    if value.is_null() {
        return Ok(None);
    }
    match o.spec.arity {
        ArgArity::Flag => match value {
            JsonValue::Bool(true) => Ok(Some(JsonValue::Bool(true))),
            JsonValue::Bool(false) => Ok(None),
            JsonValue::String(s) if s.eq_ignore_ascii_case("true") => {
                Ok(Some(JsonValue::Bool(true)))
            }
            JsonValue::String(s) if s.eq_ignore_ascii_case("false") => Ok(None),
            _ => Err(format!("{key} is a flag: true or false")),
        },
        ArgArity::Single => match value {
            JsonValue::Array(items) if items.len() == 1 => Ok(Some(text(token(&items[0], key)?))),
            JsonValue::Array(_) => Err(format!("{key} takes one value")),
            JsonValue::Bool(_) => Err(format!("{key} takes one value (a string or a number)")),
            other => Ok(Some(text(token(other, key)?))),
        },
        ArgArity::Optional(implicit) => match value {
            JsonValue::Bool(true) => Ok(Some(text(implicit))),
            // A switch whose bare flag means "true" (`-spng`, on by default): false switches
            // it off, as `"save_pict": false` in a settings file does.
            JsonValue::Bool(false) if implicit == "true" => Ok(Some(text("false"))),
            JsonValue::Bool(false) => Ok(None),
            JsonValue::Array(items) if items.len() == 1 => Ok(Some(text(token(&items[0], key)?))),
            JsonValue::Array(_) => Err(format!("{key} takes one value or true")),
            other => Ok(Some(text(token(other, key)?))),
        },
        ArgArity::ZeroOrMore | ArgArity::OneOrMore => {
            let items: Vec<String> = match value {
                JsonValue::Array(items) => items
                    .iter()
                    .map(|i| token(i, key))
                    .collect::<Result<_, _>>()?,
                other => vec![token(other, key)?],
            };
            if o.spec.arity == ArgArity::OneOrMore && items.is_empty() {
                return Err(format!("{key} takes at least one value"));
            }
            Ok(Some(JsonValue::Array(
                items.into_iter().map(text).collect(),
            )))
        }
    }
}

/// The request in the arguments `args` of the tool of `command` with `options`.
pub fn request_of(
    command: &str,
    options: &[CliOption],
    args: &JsonValue,
) -> Result<JobRequest, String> {
    let Some(map) = args.as_object() else {
        return Err("the arguments are an object".to_string());
    };
    let mut out = JobRequest::default();
    let mut unknown = Vec::new();
    let mut unavailable = Vec::new();
    for (key, value) in map {
        if key == INPUT_PROPERTY || key == "input" {
            out.inputs = strings(value, INPUT_PROPERTY)?;
            continue;
        }
        if key == FILES_PROPERTY {
            out.files = strings(value, FILES_PROPERTY)?;
            continue;
        }
        match find_option(options, key) {
            None => unknown.push(key.clone()),
            Some(o) if !o.implemented => {
                unavailable.push(format!("{} ({})", o.spec.flags.join("/"), o.feature));
            }
            Some(o) => {
                if let Some(v) = setting(o, value)? {
                    out.settings.insert(o.spec.key.to_string(), v);
                }
            }
        }
    }
    if !unknown.is_empty() {
        return Err(format!(
            "{command} has no option {} (the properties are the long option names of the \
             command line, as in its input schema and in command_help)",
            unknown.join(", ")
        ));
    }
    if !unavailable.is_empty() {
        return Err(format!(
            "not available in this version of mopt-rs: {}",
            unavailable.join(", ")
        ));
    }
    if out.inputs.is_empty() {
        return Err(format!("{command}: give the inputs as {INPUT_PROPERTY}"));
    }
    Ok(out)
}

/// Every part of a value of `settings` that may be a path must be relative, without `..`, and
/// stay inside the root as seen from the job directory `base`.
pub fn check_paths(
    sandbox: &Sandbox,
    base: &Path,
    settings: &BTreeMap<String, JsonValue>,
) -> Result<(), String> {
    for (key, value) in settings {
        let values: Vec<&str> = match value {
            JsonValue::String(s) => vec![s.as_str()],
            JsonValue::Array(items) => items.iter().filter_map(JsonValue::as_str).collect(),
            _ => Vec::new(),
        };
        for v in values {
            for part in path_like_parts(v) {
                sandbox
                    .check_value_part(base, part)
                    .map_err(|e| format!("{key}: {e}"))?;
            }
        }
    }
    Ok(())
}

/// The same checks for every string of a JSON document copied into a job (a settings file of a
/// command, which names paths and calculators as options do).
pub fn check_json_paths(sandbox: &Sandbox, base: &Path, doc: &JsonValue) -> Result<(), String> {
    match doc {
        JsonValue::Object(map) => map
            .values()
            .try_for_each(|v| check_json_paths(sandbox, base, v)),
        JsonValue::Array(items) => items
            .iter()
            .try_for_each(|v| check_json_paths(sandbox, base, v)),
        JsonValue::String(s) => path_like_parts(s)
            .into_iter()
            .try_for_each(|part| sandbox.check_value_part(base, part)),
        _ => Ok(()),
    }
}

/// What was copied into a job directory so far.
#[derive(Debug, Clone, Default)]
pub struct CopyBudget {
    /// Bytes.
    pub bytes: u64,
    /// Files.
    pub files: usize,
    /// Entries left out (symbolic links that lead out of the root or to directories, and
    /// entries that are neither files nor directories).
    pub skipped: Vec<String>,
    /// The JSON documents among the copied files (checked as settings files).
    pub json: Vec<JsonValue>,
}

/// What a copy must respect.
#[derive(Debug, Clone, Copy)]
pub struct CopyRules<'a> {
    /// The root.
    pub sandbox: &'a Sandbox,
    /// The directory of the jobs (never copied).
    pub jobs_dir: &'a Path,
    /// What may not be copied (files that name programs).
    pub policy: &'a Policy,
}

/// Copy the existing file or directory `source` (canonical, inside the root) into `job_dir`
/// at its path relative to the root; returns that path. Every file is checked again when it
/// is opened, and written as a new regular file (never through a link).
pub fn copy_into(
    rules: CopyRules<'_>,
    source: &Path,
    job_dir: &Path,
    budget: &mut CopyBudget,
) -> Result<String, String> {
    let rel = rules
        .sandbox
        .relative(source)
        .filter(|r| !r.is_empty())
        .ok_or_else(|| "the root itself cannot be an input; name its files".to_string())?;
    if source.is_dir() && rules.jobs_dir.starts_with(source) {
        return Err(format!(
            "{rel}: a directory that holds the jobs cannot be copied into a job"
        ));
    }
    let target = job_dir.join(&rel);
    copy_entry(rules, source, &target, budget, 0).map_err(|e| format!("{rel}: {e}"))?;
    Ok(rel)
}

fn copy_entry(
    rules: CopyRules<'_>,
    source: &Path,
    target: &Path,
    budget: &mut CopyBudget,
    depth: usize,
) -> Result<(), String> {
    if depth > 32 {
        return Err("directories nested too deeply".to_string());
    }
    let meta = std::fs::symlink_metadata(source).map_err(|e| e.to_string())?;
    if meta.is_dir() {
        std::fs::create_dir_all(target).map_err(|e| e.to_string())?;
        for entry in super::run_files::sorted_entries(source) {
            let Some(name) = entry.file_name() else {
                continue;
            };
            let Ok(kind) = std::fs::symlink_metadata(&entry) else {
                continue;
            };
            let resolved = if kind.file_type().is_symlink() {
                match std::fs::canonicalize(&entry) {
                    Ok(r) if rules.sandbox.contains(&r) && r.is_file() => r,
                    _ => {
                        budget
                            .skipped
                            .push(rules.sandbox.relative(&entry).unwrap_or_default());
                        continue;
                    }
                }
            } else if kind.is_dir() || kind.is_file() {
                entry.clone()
            } else {
                budget
                    .skipped
                    .push(rules.sandbox.relative(&entry).unwrap_or_default());
                continue;
            };
            copy_entry(rules, &resolved, &target.join(name), budget, depth + 1)?;
        }
        return Ok(());
    }
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    rules.policy.check_file_name(&name)?;
    let (mut from, opened) = rules.sandbox.open_regular(source)?;
    budget.bytes += opened.len();
    budget.files += 1;
    if budget.bytes > COPY_LIMIT_BYTES || budget.files > COPY_LIMIT_FILES {
        return Err(format!(
            "more than {COPY_LIMIT_FILES} files or {COPY_LIMIT_BYTES} bytes to copy"
        ));
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut to = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)
        .map_err(|e| e.to_string())?;
    std::io::copy(&mut from, &mut to).map_err(|e| e.to_string())?;
    if name.to_ascii_lowercase().ends_with(".json") {
        let text = std::fs::read_to_string(target).map_err(|e| e.to_string())?;
        if let Ok(doc) = JsonValue::parse(&text) {
            budget.json.push(doc);
        }
    }
    Ok(())
}

/// The settings file of a job: the inputs (relative paths in the job directory) and the
/// options.
pub fn settings_document(inputs: &[String], settings: &BTreeMap<String, JsonValue>) -> JsonValue {
    let mut map = settings.clone();
    map.insert(
        INPUT_KEY.to_string(),
        JsonValue::Array(inputs.iter().map(text).collect()),
    );
    map.insert(VERSION_KEY.to_string(), text(crate::VERSION));
    JsonValue::Object(map)
}

/// The paths of `copied` (relative to the root) inside `job_dir`.
pub fn copied_paths(job_dir: &Path, copied: &[String]) -> Vec<PathBuf> {
    copied.iter().map(|c| job_dir.join(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::option::parser;
    use crate::cli::settings::parse_settings;
    use crate::cli::spec::OPTMAIN;
    use crate::core_infra::json::JsonValue;

    #[test]
    fn arguments_mean_what_the_command_line_means() {
        let args = JsonValue::parse(
            r#"{"INPUT": "a.xyz", "opt": ["rsirfo_bfgs"], "NSTEP": 50, "unrestrict": true,
                "electronic_charge": -1, "use_model_hessian": true, "othersoft": "lj"}"#,
        )
        .unwrap();
        let r = request_of("optmain", OPTMAIN, &args).unwrap();
        assert_eq!(r.inputs, vec!["a.xyz".to_string()]);
        let doc = settings_document(&r.inputs, &r.settings).to_string_pretty();
        let from_file = parse_settings(&doc, OPTMAIN).unwrap();
        let from_flags = parser("optmain", OPTMAIN)
            .parse([
                "a.xyz",
                "-opt",
                "rsirfo_bfgs",
                "-ns",
                "50",
                "-u",
                "-elec",
                "-1",
                "-modelhess",
                "-os",
                "lj",
            ])
            .unwrap();
        assert_eq!(from_file.positionals, from_flags.positionals);
        assert_eq!(from_file.values, from_flags.values);
        assert_eq!(from_file.present, from_flags.present);
    }

    #[test]
    fn save_pict_false_switches_the_figures_off_as_the_settings_file_does() {
        use crate::cli::nebmain::NebmainConfig;
        use crate::cli::spec::NEBMAIN;
        let dir = std::env::temp_dir().join(format!("mopt_mcp_spng_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a.txt"), dir.join("b.txt"));
        std::fs::write(&a, "x -0.558\ny 1.442\n").unwrap();
        std::fs::write(&b, "x 0.623\ny 0.028\n").unwrap();
        let (a, b) = (a.to_string_lossy(), b.to_string_lossy());
        let figures = |value: &str| {
            let args = JsonValue::parse(&format!(r#"{{"INPUT": ["{a}", "{b}"]{value}}}"#)).unwrap();
            let r = request_of("nebmain", NEBMAIN, &args).unwrap();
            let doc = settings_document(&r.inputs, &r.settings).to_string_pretty();
            NebmainConfig::from_args(&parse_settings(&doc, NEBMAIN).unwrap())
                .unwrap()
                .figures
        };
        let file = |text: &str| {
            let doc = format!(r#"{{"INPUT": ["{a}", "{b}"], "save_pict": {text}}}"#);
            NebmainConfig::from_args(&parse_settings(&doc, NEBMAIN).unwrap())
                .unwrap()
                .figures
        };
        assert!(figures(""));
        assert!(figures(r#", "save_pict": true"#));
        assert!(!figures(r#", "save_pict": false"#));
        assert!(!figures(r#", "save_pict": "off""#));
        // The same as the settings file.
        assert_eq!(figures(r#", "save_pict": false"#), file("false"));
        assert_eq!(figures(r#", "save_pict": true"#), file("true"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_and_unavailable_options_are_refused() {
        let bad = JsonValue::parse(r#"{"INPUT": ["a.xyz"], "no_such_option": 1}"#).unwrap();
        assert!(request_of("optmain", OPTMAIN, &bad)
            .unwrap_err()
            .contains("no_such_option"));
        let missing = JsonValue::parse(r#"{"othersoft": "lj"}"#).unwrap();
        assert!(request_of("optmain", OPTMAIN, &missing)
            .unwrap_err()
            .contains("INPUT"));
        let flag = JsonValue::parse(r#"{"INPUT": ["a.xyz"], "unrestrict": 3}"#).unwrap();
        assert!(request_of("optmain", OPTMAIN, &flag).is_err());
        if let Some(o) = OPTMAIN.iter().find(|o| !o.implemented) {
            let text = format!(r#"{{"INPUT": ["a.xyz"], "{}": "x"}}"#, o.spec.key);
            let args = JsonValue::parse(&text).unwrap();
            let e = request_of("optmain", OPTMAIN, &args).unwrap_err();
            assert!(e.contains("not available"), "{e}");
        }
    }
}
