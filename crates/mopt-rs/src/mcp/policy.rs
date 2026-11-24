//! What the server refuses to run unless it was started with `--allow-external`: settings that
//! start programs outside mopt-rs or load code.
//!
//! Without the flag, a job or a single point is refused when its options (or the JSON files
//! copied into the job, such as the settings file of `run_autots -cfg`) hold
//!
//! - a calculator (`-os`, the calculators of `-oniom` and `-oniom_middle`) other than the
//!   built-in potentials, the objectives of plain variables, and the libraries `pyscf`,
//!   `psi4`, `tblite`, `dxtb` that run inside the process (`ase:module:Class`, the ASE
//!   profiles, external programs, `multioptpy:NAME`, plugins, OpenMM need the flag);
//! - a setting that loads code or a system description, at any depth of `KEY=VALUE`: `hook`,
//!   `hook.*`, `excited_setup`, `state_reader`, `factory`, `system` (also as `low.system=`,
//!   `middle.system=`, `low.calc_option=hook=...`);
//! - a reference to code, `module:name`, in the settings of calculators (`-calc_opt`,
//!   `-oniom_opt`, the settings of a single point), where the engines take such references;
//!   other options take `NAME:VALUE` forms of their own (`hyperspherical:FILE`, `dvv:v0`);
//! - the option `-osp` (`software_path_file`), or a copied file named `software_path.conf`,
//!   `mopt_profiles.json` or `mopt_calculator_options.json` (paths of programs, factories and
//!   settings of calculators that the commands read from their working directory).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use crate::core_infra::json::JsonValue;
use crate::pes::objective::ObjectiveRegistry;
use crate::pes::registry::CalculatorRegistry;

/// The libraries that run inside the process of a job and load nothing named by a request.
pub const IN_PROCESS_ENGINES: [&str; 4] = ["pyscf", "psi4", "tblite", "dxtb"];
/// Heads of `head:argument` values that are not references to code.
pub const SAFE_HEADS: [&str; 3] = ["tblite", "dxtb", "expr"];
/// Settings (`KEY=VALUE`) that load code or a system description.
pub const CODE_KEYS: [&str; 5] = ["hook", "excited_setup", "state_reader", "factory", "system"];
/// Options that name the paths of external programs.
pub const PROGRAM_OPTIONS: [&str; 2] = ["software_path_file", "osp"];
/// Files that name programs, factories or settings of calculators when they are in the
/// working directory of a job.
pub const PROGRAM_FILES: [&str; 3] = [
    "software_path.conf",
    "mopt_profiles.json",
    "mopt_calculator_options.json",
];
/// Options that hold the settings of calculators (`KEY=VALUE`), whose values may name code.
pub const SETTINGS_OPTIONS: [&str; 4] = ["calc_option", "calc_opt", "oniom_options", "oniom_opt"];
/// Options whose value is a calculator name.
const CALCULATOR_OPTIONS: [&str; 2] = ["othersoft", "os"];
/// Options whose third value is a calculator name (`ATOMS LINK CALCULATOR`).
const LAYER_OPTIONS: [&str; 4] = ["oniom_flag", "oniom", "oniom_middle", "oniom_middle_flag"];

/// The policy of a server.
#[derive(Debug, Clone)]
pub struct Policy {
    /// Whether settings that start external programs or load code are allowed.
    pub allow_external: bool,
    calculators: Vec<String>,
}

fn is_identifier_path(s: &str) -> bool {
    !s.is_empty()
        && s.split('.').all(|seg| {
            let mut chars = seg.chars();
            matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}

/// Whether `value` reads as a reference to code, `module.path:name`.
pub fn is_code_reference(value: &str) -> bool {
    let v = value.trim();
    let Some((head, tail)) = v.split_once(':') else {
        return false;
    };
    if SAFE_HEADS.contains(&head.to_ascii_lowercase().as_str()) {
        return false;
    }
    // `ase:module:Class` and `module:name`.
    let tail = tail.split_once(':').map_or(tail, |(m, _)| m);
    is_identifier_path(head) && is_identifier_path(tail.trim())
}

/// Whether the setting name `name` (`hook`, `low.system`, `hook.cmd`, ...) loads code.
pub fn is_code_key(name: &str) -> bool {
    let bare = name.trim().trim_start_matches('-');
    let last = bare
        .strip_prefix("low.")
        .or_else(|| bare.strip_prefix("middle."))
        .unwrap_or(bare);
    CODE_KEYS.contains(&last) || last.split('.').next() == Some("hook")
}

impl Policy {
    /// The policy, with the calculator names of the Rust registry and of the objectives.
    pub fn new(allow_external: bool) -> Self {
        let mut calculators = CalculatorRegistry::native().names();
        calculators.extend(ObjectiveRegistry::builtin().names());
        calculators.extend(IN_PROCESS_ENGINES.iter().map(|s| s.to_string()));
        Self {
            allow_external,
            calculators,
        }
    }

    /// Whether the calculator `name` runs without the flag.
    pub fn safe_calculator(&self, name: &str) -> bool {
        let head = name
            .split(':')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        head.is_empty() || head == "none" || self.calculators.contains(&head)
    }

    fn refused(&self, what: String) -> Result<(), String> {
        if self.allow_external {
            Ok(())
        } else {
            Err(format!(
                "{what}: settings that start programs outside mopt-rs or load code are refused \
                 by this server (start it with --allow-external to allow them)"
            ))
        }
    }

    /// Check the calculator `name`.
    pub fn check_calculator(&self, name: &str) -> Result<(), String> {
        if self.safe_calculator(name) {
            Ok(())
        } else {
            self.refused(format!("the calculator '{name}'"))
        }
    }

    /// Check one token of a value of the option `key`: every setting name (`KEY=` at any
    /// depth) that loads code, and, in the settings of calculators (`code_values`), every
    /// piece between `=` and `,` that names code.
    pub fn check_token(&self, key: &str, token: &str, code_values: bool) -> Result<(), String> {
        if code_values
            && (is_code_reference(token) || token.split(['=', ',']).any(is_code_reference))
        {
            return self.refused(format!("{key}: '{token}' names code"));
        }
        let pieces: Vec<&str> = token.split('=').collect();
        for setting in &pieces[..pieces.len().saturating_sub(1)] {
            if let Some(name) = setting.split(',').find(|n| is_code_key(n)) {
                return self.refused(format!("{key}: the setting '{}'", name.trim()));
            }
        }
        Ok(())
    }

    /// Check the option `key` with the tokens `values`.
    pub fn check_option(&self, key: &str, values: &[&str]) -> Result<(), String> {
        let bare = key.trim_start_matches('-');
        if PROGRAM_OPTIONS.contains(&bare) {
            return self.refused(format!("the option {bare} (paths of external programs)"));
        }
        if is_code_key(bare) {
            return self.refused(format!("the setting '{bare}'"));
        }
        if CALCULATOR_OPTIONS.contains(&bare) {
            for v in values {
                self.check_calculator(v)?;
            }
        }
        if LAYER_OPTIONS.contains(&bare) {
            if let Some(v) = values.get(2) {
                self.check_calculator(v)?;
            }
        }
        let code_values = SETTINGS_OPTIONS.contains(&bare);
        for v in values {
            self.check_token(bare, v, code_values)?;
        }
        Ok(())
    }

    /// Check the settings of a job (key: string or array of strings).
    pub fn check_settings(&self, settings: &BTreeMap<String, JsonValue>) -> Result<(), String> {
        for (key, value) in settings {
            self.check_option(key, &strings(value))?;
        }
        Ok(())
    }

    /// Check a copied file named `name` (its last component).
    pub fn check_file_name(&self, name: &str) -> Result<(), String> {
        let lower = name.to_ascii_lowercase();
        if PROGRAM_FILES.contains(&lower.as_str()) {
            return self.refused(format!("the file {name}"));
        }
        Ok(())
    }

    /// Check a JSON document copied into a job (a settings file of a command): every key is
    /// checked as an option with its values, every string as a token.
    pub fn check_json(&self, doc: &JsonValue) -> Result<(), String> {
        match doc {
            JsonValue::Object(map) => {
                for (key, value) in map {
                    self.check_option(key, &strings(value))?;
                    self.check_json(value)?;
                }
                Ok(())
            }
            JsonValue::Array(items) => items.iter().try_for_each(|i| self.check_json(i)),
            JsonValue::String(s) => self.check_token("a settings file", s, false),
            _ => Ok(()),
        }
    }
}

fn strings(value: &JsonValue) -> Vec<&str> {
    match value {
        JsonValue::String(s) => vec![s.as_str()],
        JsonValue::Array(items) => items.iter().filter_map(JsonValue::as_str).collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_and_programs_are_refused_without_the_flag() {
        let p = Policy::new(false);
        for ok in [
            "lj",
            "gfn1xtb",
            "tblite:GFN2-xTB",
            "pyscf",
            "mb",
            "mb2d",
            "expr:(q1-1)^2 + q2^2",
            "",
        ] {
            assert!(p.check_calculator(ok).is_ok(), "{ok}");
        }
        for bad in [
            "ase:ase.calculators.emt:EMT",
            "orca",
            "multioptpy:gfn2",
            "openmm",
            "mace_mp",
            "engine:m:C",
        ] {
            assert!(p.check_calculator(bad).is_err(), "{bad}");
        }
        for bad in [
            "hook=os:system",
            "hook.label=x",
            "excited_setup=fake:choose",
            "state_reader=file",
            "low.system=model.xml",
            "factory=pkg.mod:make",
            "x=pkg.mod:func",
            "pkg.mod:func",
            "low.calc_option=hook=os:system",
            "x=a.b:c,d.e:f",
            "conv_tol=1e-9,hook=m:f",
        ] {
            assert!(p.check_option("calc_option", &[bad]).is_err(), "{bad}");
        }
        for ok in [
            "conv_tol=1e-10",
            "density_fit=True",
            "cutoff=8.0",
            "1:2",
            "tblite:GFN1-xTB",
            "scf.with_solvent.eps=78.4",
        ] {
            assert!(p.check_option("calc_option", &[ok]).is_ok(), "{ok}");
        }
        // The NAME:VALUE forms of other options are not code.
        for (key, ok) in [
            ("coordinate_system", "hyperspherical:ref.xyz"),
            ("irc_method", "dvv:v0"),
            ("md_options", "file:traj.xyz"),
            ("gnt_vec", "default:calculate"),
        ] {
            assert!(p.check_option(key, &[ok]).is_ok(), "{key} {ok}");
        }
        assert!(p.check_option("md_options", &["hook=os:system"]).is_err());
        assert!(p
            .check_option("oniom_flag", &["1-3", "auto", "ase:m:C"])
            .is_err());
        assert!(p
            .check_option("oniom_flag", &["1-3", "auto", "gfnff"])
            .is_ok());
        assert!(p.check_option("software_path_file", &["s.conf"]).is_err());
        assert!(p.check_file_name("software_path.conf").is_err());
        assert!(p.check_file_name("Mopt_Profiles.json").is_err());
        assert!(p.check_file_name("mopt_calculator_options.json").is_err());
        let doc = JsonValue::parse(r#"{"step1": {"othersoft": "ase:m:C"}}"#).unwrap();
        assert!(p.check_json(&doc).is_err());
        let doc = JsonValue::parse(r#"{"pyscf": {"hook": "os:system"}}"#).unwrap();
        assert!(p.check_json(&doc).is_err());
        let doc = JsonValue::parse(
            r#"{"step1": {"othersoft": "gfn1xtb", "opt_method": ["rsirfo_bfgs"]}}"#,
        )
        .unwrap();
        assert!(p.check_json(&doc).is_ok());
        let open = Policy::new(true);
        assert!(open.check_calculator("ase:ase.calculators.emt:EMT").is_ok());
        assert!(open
            .check_option("calc_option", &["hook=os:system"])
            .is_ok());
    }
}
