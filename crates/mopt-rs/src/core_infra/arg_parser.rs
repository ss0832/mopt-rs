//! MultiOptPy-compatible command-line argument normalizer, flag parser, and atom-range parser.
//!
//! MultiOptPy: `argparse` as used by `interface.py:parser2args`, and
//!  `force_data_parser.num_parse`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{MoptError, MoptResult};

/// Arity specification for a CLI option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgArity {
    /// Boolean switch (`action='store_true'`), takes 0 values.
    Flag,
    /// Single required value (`nargs=1`).
    Single,
    /// Optional single value (`nargs='?'`), falls back to `implicit` when omitted.
    Optional(&'static str),
    /// Zero or more values (`nargs='*'`) consumed up to the next recognized flag.
    ZeroOrMore,
    /// One or more values (`nargs='+'`).
    OneOrMore,
}

/// Specification for a single CLI option.
#[derive(Debug, Clone)]
pub struct OptSpec {
    /// Canonical destination key (e.g. `"opt_method"`, `"usextb"`).
    pub key: &'static str,
    /// Recognized flag aliases including leading hyphens (e.g. `&["-opt", "--opt_method"]`).
    pub flags: &'static [&'static str],
    /// Value arity behavior.
    pub arity: ArgArity,
    /// Short English description for `--help`.
    pub help: &'static str,
}

/// Parsed command-line arguments mapped by canonical destination key.
#[derive(Debug, Clone, Default)]
pub struct ParsedArgs {
    /// Positional arguments in order of appearance.
    pub positionals: Vec<String>,
    /// Explicitly supplied options mapped to their collected string tokens.
    pub values: BTreeMap<String, Vec<String>>,
    /// Set of boolean switches or options explicitly passed on the command line.
    pub present: BTreeSet<String>,
    /// Whether `-h` or `--help` was requested.
    pub help_requested: bool,
}

impl ParsedArgs {
    /// Check if an option was explicitly passed on the command line.
    pub fn is_present(&self, key: &str) -> bool {
        self.present.contains(key)
    }

    /// Return boolean flag state (`true` if present and not `"false"`).
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        if !self.present.contains(key) {
            return default;
        }
        match self.values.get(key).and_then(|v| v.first()) {
            Some(s) => !matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "false" | "0" | "no" | "off"
            ),
            None => true,
        }
    }

    /// Return first string value if present, else `default`.
    pub fn get_str<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.values
            .get(key)
            .and_then(|v| v.first())
            .map(String::as_str)
            .unwrap_or(default)
    }

    /// Return optional string value (`None` if absent or `"None"`).
    pub fn get_opt_str(&self, key: &str) -> Option<&str> {
        let s = self.values.get(key)?.first()?.as_str();
        if s.eq_ignore_ascii_case("none") || s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    /// Parse integer option or return `default`.
    pub fn get_i64(&self, key: &str, default: i64) -> MoptResult<i64> {
        match self.values.get(key).and_then(|v| v.first()) {
            Some(s) => s
                .parse::<i64>()
                .map_err(|e| MoptError::parse(format!("Invalid integer for '{key}': '{s}' ({e})"))),
            None => Ok(default),
        }
    }

    /// Parse `usize` option or return `default`.
    pub fn get_usize(&self, key: &str, default: usize) -> MoptResult<usize> {
        match self.values.get(key).and_then(|v| v.first()) {
            Some(s) => s.parse::<usize>().map_err(|e| {
                MoptError::parse(format!(
                    "Invalid non-negative integer for '{key}': '{s}' ({e})"
                ))
            }),
            None => Ok(default),
        }
    }

    /// Parse `f64` option or return `default`.
    pub fn get_f64(&self, key: &str, default: f64) -> MoptResult<f64> {
        match self.values.get(key).and_then(|v| v.first()) {
            Some(s) => s
                .parse::<f64>()
                .map_err(|e| MoptError::parse(format!("Invalid float for '{key}': '{s}' ({e})"))),
            None => Ok(default),
        }
    }

    /// Return optional `f64` if explicitly provided.
    pub fn get_opt_f64(&self, key: &str) -> MoptResult<Option<f64>> {
        match self.values.get(key).and_then(|v| v.first()) {
            Some(s) => s
                .parse::<f64>()
                .map(Some)
                .map_err(|e| MoptError::parse(format!("Invalid float for '{key}': '{s}' ({e})"))),
            None => Ok(None),
        }
    }

    /// Return slice of string values for multi-token options (`ZeroOrMore`).
    pub fn get_list(&self, key: &str) -> &[String] {
        self.values.get(key).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// Configurable CLI parser supporting MultiOptPy single-hyphen multi-character flags
/// and negative numeric arguments (such as `-elec -1` or `-ma -150.0 1 5`).
pub struct CliParser {
    program: String,
    description: String,
    specs: Vec<OptSpec>,
}

impl CliParser {
    /// Create a new CLI parser with program name and summary description.
    pub fn new(program: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            description: description.into(),
            specs: Vec::new(),
        }
    }

    /// Register an option specification.
    pub fn add_opt(&mut self, spec: OptSpec) -> &mut Self {
        self.specs.push(spec);
        self
    }

    /// Register a slice of option specifications.
    pub fn add_opts(&mut self, specs: &[OptSpec]) -> &mut Self {
        self.specs.extend_from_slice(specs);
        self
    }

    /// Format a `--help` message describing all registered options.
    pub fn format_help(&self) -> String {
        let mut out = format!("{} - {}\n\nOptions:\n", self.program, self.description);
        out.push_str("  -h, --help                Show this help message and exit\n");
        for spec in &self.specs {
            let flags = spec.flags.join(", ");
            out.push_str(&format!("  {flags:<26} {}\n", spec.help));
        }
        out
    }

    fn find_spec(&self, token: &str) -> Option<&OptSpec> {
        self.specs.iter().find(|spec| spec.flags.contains(&token))
    }

    /// Determine whether `token` is a recognized option flag (not a negative number literal).
    fn is_flag_token(&self, token: &str) -> bool {
        if token == "-h" || token == "--help" {
            return true;
        }
        if self.find_spec(token).is_some() {
            return true;
        }
        if let Some((head, _)) = token.split_once('=') {
            if self.find_spec(head).is_some() {
                return true;
            }
        }
        if token.starts_with('-') && token.len() > 1 {
            // Distinguish negative numbers like "-1", "-0.5", "-1.2e-3" from unknown flags,
            // and lists of numbers that start with one, such as the range "-180,180" of a
            // scan (argparse takes those for unknown options).
            if token
                .split(',')
                .all(|part| part.trim().parse::<f64>().is_ok())
            {
                return false;
            }
            return true;
        }
        false
    }

    /// Parse an iterator of raw CLI arguments (excluding `argv[0]`).
    pub fn parse<I, S>(&self, raw_args: I) -> MoptResult<ParsedArgs>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let tokens: Vec<String> = raw_args
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        let mut out = ParsedArgs::default();
        let mut idx = 0;

        while idx < tokens.len() {
            let tok = &tokens[idx];
            if tok == "-h" || tok == "--help" {
                out.help_requested = true;
                idx += 1;
                continue;
            }
            if tok == "--" {
                for rem in &tokens[idx + 1..] {
                    out.positionals.push(rem.clone());
                }
                break;
            }

            // Support `--flag=value` or `-flag=value`
            let (flag_part, inline_val) = match tok.split_once('=') {
                Some((head, val)) if self.find_spec(head).is_some() => (head, Some(val)),
                _ => (tok.as_str(), None),
            };

            if let Some(spec) = self.find_spec(flag_part) {
                out.present.insert(spec.key.to_string());
                match spec.arity {
                    ArgArity::Flag => {
                        if let Some(v) = inline_val {
                            out.values
                                .entry(spec.key.to_string())
                                .or_default()
                                .push(v.to_string());
                        }
                        idx += 1;
                    }
                    ArgArity::Single => {
                        if let Some(v) = inline_val {
                            out.values.insert(spec.key.to_string(), vec![v.to_string()]);
                            idx += 1;
                        } else if idx + 1 < tokens.len() {
                            out.values
                                .insert(spec.key.to_string(), vec![tokens[idx + 1].clone()]);
                            idx += 2;
                        } else {
                            return Err(MoptError::parse(format!(
                                "Option '{flag_part}' requires an argument"
                            )));
                        }
                    }
                    ArgArity::Optional(implicit_default) => {
                        if let Some(v) = inline_val {
                            out.values.insert(spec.key.to_string(), vec![v.to_string()]);
                            idx += 1;
                        } else if idx + 1 < tokens.len() && !self.is_flag_token(&tokens[idx + 1]) {
                            out.values
                                .insert(spec.key.to_string(), vec![tokens[idx + 1].clone()]);
                            idx += 2;
                        } else {
                            out.values
                                .insert(spec.key.to_string(), vec![implicit_default.to_string()]);
                            idx += 1;
                        }
                    }
                    ArgArity::ZeroOrMore | ArgArity::OneOrMore => {
                        // As in argparse, a repeated option replaces the earlier values.
                        let mut entry = Vec::new();
                        if let Some(v) = inline_val {
                            entry.push(v.to_string());
                            idx += 1;
                        } else {
                            idx += 1;
                            while idx < tokens.len() && !self.is_flag_token(&tokens[idx]) {
                                entry.push(tokens[idx].clone());
                                idx += 1;
                            }
                        }
                        if spec.arity == ArgArity::OneOrMore && entry.is_empty() {
                            return Err(MoptError::parse(format!(
                                "Option '{flag_part}' requires at least one argument"
                            )));
                        }
                        out.values.insert(spec.key.to_string(), entry);
                    }
                }
            } else if self.is_flag_token(tok) {
                return Err(MoptError::parse(format!(
                    "Unrecognized CLI option: '{tok}'"
                )));
            } else {
                out.positionals.push(tok.clone());
                idx += 1;
            }
        }

        Ok(out)
    }
}

/// The value of a switch of `option`: `true`, `on`, `yes`, `1` or `false`, `off`, `no`, `0`
/// (any case, surrounding blanks ignored); anything else is refused with the option named.
/// The one parser of the words of switches in settings such as `-scan_opt ts_optimize=on`.
///
/// MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's switches are flags).
pub fn parse_switch(option: &str, value: &str) -> MoptResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        other => Err(MoptError::parse(format!(
            "{option}: '{other}' is not a switch (true/false, on/off, yes/no, 1/0)"
        ))),
    }
}

/// A finite number of `option` (`NaN` and the infinities are refused with the option named).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn parse_finite(option: &str, value: &str) -> MoptResult<f64> {
    let v = value
        .trim()
        .parse::<f64>()
        .map_err(|_| MoptError::parse(format!("{option}: '{}' is not a number", value.trim())))?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(MoptError::parse(format!(
            "{option}: '{}' is not a finite number",
            value.trim()
        )))
    }
}

/// The atoms of a list of `option` such as `1,3,5-8` as indices starting at 0.
///
/// The numbers start at 1; `a-b` is the range from `a` to `b` (`a <= b`); blanks are ignored;
/// an atom given twice counts once, in the place where it first appears; an empty list or
/// `none` is no atom. `0` is refused, and so is a number above `n_atoms` when the number of
/// atoms is known. (MultiOptPy's `num_parse` keeps repeats and turns `0` into atom -1; only
/// its `-gi` gives a lone number the meaning "off", which [`parse_atom_indices_zero_based`]
/// keeps for the commands that rely on it.)
///
/// MultiOptPy: `interface.py:num_parse`.
pub fn parse_atom_list(option: &str, expr: &str, n_atoms: Option<usize>) -> MoptResult<Vec<usize>> {
    let trimmed = expr.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        return Ok(Vec::new());
    }
    let numbers = atom_numbers(trimmed).map_err(|e| MoptError::parse(format!("{option}: {e}")))?;
    let mut out: Vec<usize> = Vec::with_capacity(numbers.len());
    for number in numbers {
        if number == 0 {
            return Err(MoptError::parse(format!(
                "{option}: atom numbers start at 1, got 0 in '{trimmed}'"
            )));
        }
        if let Some(n) = n_atoms.filter(|n| number > *n) {
            return Err(MoptError::parse(format!(
                "{option}: atom {number} in '{trimmed}', but the structure has {n} atoms"
            )));
        }
        if !out.contains(&(number - 1)) {
            out.push(number - 1);
        }
    }
    Ok(out)
}

/// The numbers of a list `1,3,5-8` as given (ranges expanded; a range must ascend).
fn atom_numbers(expr: &str) -> MoptResult<Vec<usize>> {
    let mut out = Vec::new();
    for part in expr.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let number = |s: &str| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| MoptError::parse(format!("'{part}' is not an atom number or a range")))
        };
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (number(a)?, number(b)?);
                if a > b {
                    return Err(MoptError::parse(format!("the range '{part}' must ascend")));
                }
                out.extend(a..=b);
            }
            None => out.push(number(part)?),
        }
    }
    Ok(out)
}

/// Parse MultiOptPy 1-based atom index expressions such as `"1,3,5-8"` into 0-based `usize` indices.
///
/// The older reader: a lone `0` is no atom (MultiOptPy's off value of `-gi`), and repeats are
/// kept. New code reads lists with [`parse_atom_list`], which names the option and refuses
/// `0` and atoms beyond the structure.
pub fn parse_atom_indices_zero_based(expr: &str) -> MoptResult<Vec<usize>> {
    let trimmed = expr.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") || trimmed == "0" {
        return Ok(Vec::new());
    }
    let mut indices = Vec::new();
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start_s, end_s)) = part.split_once('-') {
            let start_1 = start_s.trim().parse::<usize>().map_err(|_| {
                MoptError::parse(format!("Invalid start atom index in range '{part}'"))
            })?;
            let end_1 = end_s.trim().parse::<usize>().map_err(|_| {
                MoptError::parse(format!("Invalid end atom index in range '{part}'"))
            })?;
            if start_1 == 0 || end_1 == 0 || start_1 > end_1 {
                return Err(MoptError::parse(format!(
                    "Atom range must be 1-based with start <= end, got '{part}'"
                )));
            }
            for idx_1 in start_1..=end_1 {
                indices.push(idx_1 - 1);
            }
        } else {
            let idx_1 = part
                .parse::<usize>()
                .map_err(|_| MoptError::parse(format!("Invalid 1-based atom index '{part}'")))?;
            if idx_1 == 0 {
                return Err(MoptError::parse("Atom indices are 1-based and cannot be 0"));
            }
            indices.push(idx_1 - 1);
        }
    }
    Ok(indices)
}

/// Parse MultiOptPy 1-based atom index expressions preserving 1-based numbering where required.
pub fn parse_atom_indices_one_based(expr: &str) -> MoptResult<Vec<usize>> {
    Ok(parse_atom_indices_zero_based(expr)?
        .into_iter()
        .map(|i| i + 1)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_numbers_and_atom_lists_have_one_reading() {
        for word in ["true", "ON", " yes ", "1"] {
            assert!(parse_switch("-x", word).unwrap(), "{word}");
        }
        for word in ["False", "off", "NO", " 0"] {
            assert!(!parse_switch("-x", word).unwrap(), "{word}");
        }
        let e = parse_switch("-scan_opt ts_optimize", "maybe").unwrap_err();
        assert!(e.to_string().contains("-scan_opt ts_optimize"), "{e}");
        assert_eq!(parse_finite("-dt", " 0.5 ").unwrap(), 0.5);
        for bad in ["nan", "inf", "-inf", "x"] {
            let e = parse_finite("-dt", bad).unwrap_err().to_string();
            assert!(e.contains("-dt"), "{e}");
        }
        // Atom lists: ranges, repeats once in their first place, 0 and atoms beyond refused.
        assert_eq!(
            parse_atom_list("-a", "3,1-2, 2,5", None).unwrap(),
            vec![2, 0, 1, 4]
        );
        assert!(parse_atom_list("-a", "none", None).unwrap().is_empty());
        assert!(parse_atom_list("-a", "", Some(3)).unwrap().is_empty());
        let e = parse_atom_list("-partial_atoms", "0", None)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("-partial_atoms") && e.contains("start at 1"),
            "{e}"
        );
        let e = parse_atom_list("-a", "1,4", Some(3))
            .unwrap_err()
            .to_string();
        assert!(e.contains("3 atoms"), "{e}");
        assert!(parse_atom_list("-a", "3-1", None).is_err());
        assert!(parse_atom_list("-a", "1,x", None).is_err());
    }

    #[test]
    fn cli_parser_handles_negative_numbers_and_optional_flags() {
        let mut p = CliParser::new("optmain", "Geometry optimizer");
        p.add_opts(&[
            OptSpec {
                key: "elec",
                flags: &["-elec", "--electronic_charge"],
                arity: ArgArity::Single,
                help: "Electronic charge",
            },
            OptSpec {
                key: "manual_afir",
                flags: &["-ma", "--manual_AFIR"],
                arity: ArgArity::ZeroOrMore,
                help: "Manual AFIR parameters",
            },
            OptSpec {
                key: "model_hess",
                flags: &["-modelhess", "--use_model_hessian"],
                arity: ArgArity::Optional("fischerd3old"),
                help: "Model Hessian kind",
            },
            OptSpec {
                key: "freq",
                flags: &["-freq", "--frequency_analysis"],
                arity: ArgArity::Flag,
                help: "Frequency analysis",
            },
        ]);

        let parsed = p
            .parse([
                "SN2.xyz",
                "-elec",
                "-1",
                "-ma",
                "-150.0",
                "1-3",
                "6",
                "-modelhess",
                "-freq",
            ])
            .expect("parse args");
        assert_eq!(parsed.positionals, vec!["SN2.xyz"]);
        assert_eq!(parsed.get_i64("elec", 0).unwrap(), -1);
        assert_eq!(parsed.get_list("manual_afir"), &["-150.0", "1-3", "6"]);
        assert_eq!(parsed.get_str("model_hess", ""), "fischerd3old");
        assert!(parsed.get_bool("freq", false));
        // A list of numbers that starts with a negative one is a value (the range of a scan).
        let parsed = p
            .parse([
                "-ma",
                "dihedral",
                "1,2,3,4",
                "-180,180",
                "-0.5,-1e-3",
                "-freq",
            ])
            .expect("parse args");
        assert_eq!(
            parsed.get_list("manual_afir"),
            &["dihedral", "1,2,3,4", "-180,180", "-0.5,-1e-3"]
        );
        assert!(p.parse(["-ma", "-1,x"]).is_err());
        assert_eq!(
            parse_atom_indices_zero_based("1,3-5").unwrap(),
            vec![0, 2, 3, 4]
        );
    }
}
