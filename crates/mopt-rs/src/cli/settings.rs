//! Settings files: every command runs from command-line flags, from a settings file, or both.
//!
//! A settings file is a JSON object whose keys are the long option names (without dashes, as in
//! the option tables: `"opt_method"`, `"othersoft"`, ...; any flag spelling such as `"opt"` or
//! `"-opt"` is accepted) and whose values are what the flag takes:
//!
//! ```json
//! {
//!   "INPUT": ["water.xyz"],
//!   "othersoft": "pyscf",
//!   "functional": "b3lyp",
//!   "basisset": "def2-svp",
//!   "opt_method": ["rsirfo_bfgs"],
//!   "projection_constrain": ["manual", "bond", "1,2", "1.0"],
//!   "unrestrict": true
//! }
//! ```
//!
//! Strings and numbers are one value, arrays are several, `true`/`false` switch flags. Values
//! given on the command line replace those of the file (`default < file < command line`).
//! `--config FILE` reads a file; `--dump-config FILE` writes the settings in effect (also
//! usable as a template). Run directories keep a copy (`settings.json`), which a restart reads.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy reads JSON settings only in
//!  `Entrypoints` for run_autots / run_mapper.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::Path;

use super::option::CliOption;
use crate::core_infra::arg_parser::{ArgArity, ParsedArgs};
use crate::core_infra::json::JsonValue;
use crate::error::{MoptError, MoptResult};

/// The flag reading a settings file.
pub const CONFIG_FLAG: &str = "--config";
/// The flag writing the settings in effect.
pub const DUMP_FLAG: &str = "--dump-config";
/// The flag continuing an interrupted run from its directory.
pub const RESTART_FLAG: &str = "--restart";
/// File name of the settings kept in each run directory.
pub const RUN_SETTINGS_FILE: &str = "settings.json";
/// Key of the positional inputs.
pub const INPUT_KEY: &str = "INPUT";
/// Key recording the version that wrote a file (ignored when reading).
pub const VERSION_KEY: &str = "mopt_rs_version";

/// `--config FILE` / `--dump-config FILE` taken out of `args`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SettingsFlags {
    /// Settings file to read.
    pub config: Option<String>,
    /// Where to write the settings in effect.
    pub dump: Option<String>,
    /// Run directory to continue.
    pub restart: Option<String>,
    /// The remaining arguments.
    pub args: Vec<String>,
}

/// Split the settings flags from the other arguments (`--config F` or `--config=F`).
pub fn split_settings_flags(args: &[String]) -> MoptResult<SettingsFlags> {
    let mut out = SettingsFlags::default();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if [CONFIG_FLAG, DUMP_FLAG, RESTART_FLAG].contains(&f) => {
                (f, Some(v.to_string()))
            }
            _ => (a.as_str(), None),
        };
        if [CONFIG_FLAG, DUMP_FLAG, RESTART_FLAG].contains(&flag) {
            let value = match inline {
                Some(v) => v,
                None => {
                    i += 1;
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| MoptError::parse(format!("{flag} needs a file name")))?
                }
            };
            match flag {
                CONFIG_FLAG => out.config = Some(value),
                DUMP_FLAG => out.dump = Some(value),
                _ => out.restart = Some(value),
            }
        } else {
            out.args.push(a.clone());
        }
        i += 1;
    }
    Ok(out)
}

fn find<'a>(options: &'a [CliOption], key: &str) -> Option<&'a CliOption> {
    let bare = key.trim_start_matches('-');
    options.iter().find(|o| {
        o.spec.key == bare
            || o.spec
                .flags
                .iter()
                .any(|f| f.trim_start_matches('-') == bare)
    })
}

fn scalar(v: &JsonValue, key: &str) -> MoptResult<String> {
    Ok(match v {
        JsonValue::String(s) => s.clone(),
        JsonValue::Number(n) if n.fract() == 0.0 && n.abs() < 1e15 => format!("{}", *n as i64),
        JsonValue::Number(n) => format!("{n}"),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Null => "None".to_string(),
        _ => {
            return Err(MoptError::parse(format!(
                "settings '{key}': nested arrays and objects are not option values"
            )))
        }
    })
}

fn tokens(v: &JsonValue, key: &str) -> MoptResult<Vec<String>> {
    match v {
        JsonValue::Array(items) => items.iter().map(|x| scalar(x, key)).collect(),
        other => Ok(vec![scalar(other, key)?]),
    }
}

/// The options of a settings document (a JSON object) for a command with `options`.
pub fn parse_settings(text: &str, options: &[CliOption]) -> MoptResult<ParsedArgs> {
    let JsonValue::Object(map) = JsonValue::parse(text)? else {
        return Err(MoptError::parse("a settings file is a JSON object"));
    };
    let mut out = ParsedArgs::default();
    let mut unknown = Vec::new();
    for (key, value) in &map {
        if key == VERSION_KEY {
            continue;
        }
        if key == INPUT_KEY || key == "input" {
            out.positionals = tokens(value, key)?;
            continue;
        }
        let Some(o) = find(options, key) else {
            unknown.push(key.clone());
            continue;
        };
        let k = o.spec.key.to_string();
        match (o.spec.arity, value) {
            (ArgArity::Flag, JsonValue::Bool(false) | JsonValue::Null) => {}
            (ArgArity::Flag, JsonValue::Bool(true)) => {
                out.present.insert(k);
            }
            (ArgArity::Flag, other) => {
                out.present.insert(k.clone());
                out.values.insert(k, vec![scalar(other, key)?]);
            }
            (ArgArity::Single | ArgArity::Optional(_), JsonValue::Array(items))
                if items.len() != 1 =>
            {
                return Err(MoptError::parse(format!(
                    "settings '{key}' takes one value, got {}",
                    items.len()
                )))
            }
            (_, v) => {
                out.present.insert(k.clone());
                out.values.insert(k, tokens(v, key)?);
            }
        }
    }
    if !unknown.is_empty() {
        return Err(MoptError::parse(format!(
            "unknown settings: {} (keys are option names such as 'opt_method')",
            unknown.join(", ")
        )));
    }
    Ok(out)
}

/// Read a settings file.
pub fn read_settings(path: &Path, options: &[CliOption]) -> MoptResult<ParsedArgs> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| MoptError::config(format!("settings file {}: {e}", path.display())))?;
    parse_settings(&text, options)
        .map_err(|e| MoptError::parse(format!("settings file {}: {e}", path.display())))
}

/// `file` with the options given on the command line (`cli`) replacing its values.
pub fn merge(file: ParsedArgs, cli: ParsedArgs) -> ParsedArgs {
    let mut out = file;
    for key in &cli.present {
        out.present.insert(key.clone());
        match cli.values.get(key) {
            Some(v) => {
                out.values.insert(key.clone(), v.clone());
            }
            None => {
                out.values.remove(key);
            }
        }
    }
    if !cli.positionals.is_empty() {
        out.positionals = cli.positionals;
    }
    out.help_requested |= cli.help_requested;
    out
}

/// The settings in effect as a settings document (explicitly given options only).
pub fn settings_json(parsed: &ParsedArgs, options: &[CliOption]) -> JsonValue {
    let mut map = BTreeMap::new();
    map.insert(
        VERSION_KEY.to_string(),
        JsonValue::String(crate::VERSION.to_string()),
    );
    if !parsed.positionals.is_empty() {
        map.insert(
            INPUT_KEY.to_string(),
            JsonValue::Array(
                parsed
                    .positionals
                    .iter()
                    .cloned()
                    .map(JsonValue::String)
                    .collect(),
            ),
        );
    }
    for o in options {
        let key = o.spec.key;
        if !parsed.present.contains(key) {
            continue;
        }
        let values = parsed.values.get(key);
        let v = match (o.spec.arity, values) {
            (ArgArity::Flag, None) => JsonValue::Bool(true),
            (ArgArity::Single | ArgArity::Optional(_) | ArgArity::Flag, Some(v))
                if v.len() == 1 =>
            {
                JsonValue::String(v[0].clone())
            }
            (_, Some(v)) => JsonValue::Array(v.iter().cloned().map(JsonValue::String).collect()),
            (_, None) => JsonValue::Array(Vec::new()),
        };
        map.insert(key.to_string(), v);
    }
    JsonValue::Object(map)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cli::option::parser;
    use crate::cli::spec::OPTMAIN;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn files_and_command_lines_agree_and_merge() {
        let cli = parser("optmain", OPTMAIN)
            .parse(s(&[
                "a.xyz", "-opt", "fire", "-pc", "manual", "bond", "1,2", "1.0", "-u", "-elec", "-1",
            ]))
            .unwrap();
        let doc = settings_json(&cli, OPTMAIN).to_string_pretty();
        let back = parse_settings(&doc, OPTMAIN).unwrap();
        assert_eq!(back.positionals, cli.positionals);
        assert_eq!(back.values, cli.values);
        assert_eq!(back.present, cli.present);
        let file = parse_settings(
            r#"{"INPUT": "b.xyz", "opt": ["rsirfo_bfgs"], "-elec": 1, "unrestrict": false,
                "basisset": "def2-svp"}"#,
            OPTMAIN,
        )
        .unwrap();
        let over = parser("optmain", OPTMAIN)
            .parse(s(&["-elec", "2"]))
            .unwrap();
        let m = merge(file, over);
        assert_eq!(m.positionals, s(&["b.xyz"]));
        assert_eq!(
            m.get_i64("electronic_charge", 0).unwrap(),
            2,
            "command line wins"
        );
        assert_eq!(m.get_list("opt_method"), s(&["rsirfo_bfgs"]));
        assert!(!m.get_bool("unrestrict", false));
        assert!(parse_settings(r#"{"no_such_option": 1}"#, OPTMAIN).is_err());
        assert!(parse_settings(r#"{"basisset": ["a", "b"]}"#, OPTMAIN).is_err());
        let f = split_settings_flags(&s(&["x.xyz", "--config", "c.json", "--dump-config=d.json"]))
            .unwrap();
        assert_eq!(
            (f.config.as_deref(), f.dump.as_deref(), f.args),
            (Some("c.json"), Some("d.json"), s(&["x.xyz"]))
        );
    }
}
