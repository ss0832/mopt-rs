//! Calculator selection and settings on the command line, the same for every command.
//!
//! - `-os NAME` names the calculator (default: PySCF); `-xtb METHOD` is `tblite:METHOD` and
//!   `-dxtb METHOD` is `dxtb:METHOD`, as in MultiOptPy.
//! - `-func`, `-bs`, `-sub_bs EL BASIS ...`, `-ecp EL ECP ...`, `-u`, `-grid`, `-core`, `-mem`,
//!   `-cpcm`/`-alpb` and `software_path.conf` (`-osp`) become [`CalculatorOptions`]; each
//!   program maps them to its own input.
//! - `-calc_opt key=value ...` (mopt-rs) passes program-specific settings, e.g.
//!   `-calc_opt density_fit=True conv_tol=1e-10`.
//!
//! MultiOptPy: `interface.py:optimizeparser` (`-os`, `-xtb`, `-dxtb`, `-func`, `-bs`, ...) and the
//!  calculator choice of `optimization.py:OptimizationConfig`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use crate::core_infra::arg_parser::ParsedArgs;
use crate::error::{MoptError, MoptResult};
use crate::pes::registry::{CalculatorOptions, DEFAULT_CALCULATOR};

fn pairs(tokens: &[String], flag: &str) -> MoptResult<Vec<(String, String)>> {
    if !tokens.len().is_multiple_of(2) {
        return Err(MoptError::parse(format!(
            "{flag} takes pairs 'element name', got {} values",
            tokens.len()
        )));
    }
    Ok(tokens
        .chunks(2)
        .map(|c| (c[0].clone(), c[1].clone()))
        .collect())
}

fn set(key: &str, p: &ParsedArgs) -> Option<String> {
    p.get_opt_str(key)
        .map(str::trim)
        .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("none"))
        .map(str::to_string)
}

/// `software_path.conf`: `name::path` per line (`#` comments); a missing file is no entries
/// (the built-in and installed engines are used as they are).
pub fn read_software_paths(path: &Path) -> MoptResult<Vec<(String, String)>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once("::"))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect())
}

/// `-calc_opt key=value` pairs.
pub fn parse_calc_options(tokens: &[String]) -> MoptResult<Vec<(String, String)>> {
    tokens
        .iter()
        .map(|t| {
            t.split_once('=')
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .filter(|(k, _)| !k.is_empty())
                .ok_or_else(|| MoptError::parse(format!("-calc_opt expects key=value, got '{t}'")))
        })
        .collect()
}

/// The calculator name of a command line.
pub fn calculator_name(p: &ParsedArgs) -> MoptResult<String> {
    let os = set("othersoft", p);
    let xtb = set("usextb", p).map(|m| format!("tblite:{m}"));
    let dxtb = set("usedxtb", p).map(|m| format!("dxtb:{m}"));
    // MultiOptPy's flag for PySCF, which is the default here: the same as `-os pyscf`.
    let pyscf = p.get_bool("pyscf", false).then(|| "pyscf".to_string());
    let mut given: Vec<String> = [os, xtb, dxtb, pyscf].into_iter().flatten().collect();
    given.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    match given.len() {
        0 => Ok(DEFAULT_CALCULATOR.to_string()),
        // Import paths (`ase:module:Class`) are case-sensitive; the registry folds the name.
        1 => Ok(given[0].clone()),
        _ => Err(MoptError::parse(format!(
            "give one of -os, -xtb, -dxtb, -pyscf (got {})",
            given.join(", ")
        ))),
    }
}

/// The calculator settings of a command line.
pub fn calculator_options(p: &ParsedArgs) -> MoptResult<CalculatorOptions> {
    let solvation = match (set("alpb_solv_model", p), set("cpcm_solv_model", p)) {
        (Some(_), Some(_)) => return Err(MoptError::parse("give either -alpb or -cpcm")),
        (Some(s), None) => Some(("alpb".to_string(), s)),
        (None, Some(s)) => Some(("cpcm".to_string(), s)),
        (None, None) => None,
    };
    let spc = p.get_str("software_path_file", "./software_path.conf");
    Ok(CalculatorOptions {
        method: set("functional", p),
        basis: set("basisset", p),
        sub_basis: pairs(p.get_list("sub_basisset"), "-sub_bs")?,
        ecp: pairs(p.get_list("effective_core_potential"), "-ecp")?,
        unrestricted: p.get_bool("unrestrict", false),
        grid_level: p
            .get_opt_str("dft_grid")
            .map(|_| p.get_i64("dft_grid", 3))
            .transpose()?,
        threads: p
            .get_opt_str("N_THREAD")
            .map(|_| p.get_usize("N_THREAD", 8))
            .transpose()?,
        memory: set("SET_MEMORY", p),
        solvation,
        software_paths: read_software_paths(Path::new(spc))?,
        software_path_file: Path::new(spc).is_file().then(|| spc.to_string()),
        extra: parse_calc_options(p.get_list("calc_option"))?,
        oniom: super::oniom::oniom_spec(p)?,
        follow_states: p.get_bool("follow_states", false),
        parallel: p.get_usize("parallel", 0)?,
        model_of: None,
        stability: p.get_usize("scf_stability", 0)?,
        stability_external: match p.get_str("scf_stability_external", "report") {
            mode @ ("off" | "report" | "switch") => mode.to_string(),
            other => {
                return Err(MoptError::parse(format!(
                    "-stability_external is off, report, or switch, not '{other}'"
                )))
            }
        },
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn calc_options_and_pairs() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(
            parse_calc_options(&s(&["density_fit=True", "conv_tol = 1e-10"])).unwrap(),
            vec![
                ("density_fit".to_string(), "True".to_string()),
                ("conv_tol".to_string(), "1e-10".to_string())
            ]
        );
        assert!(parse_calc_options(&s(&["novalue"])).is_err());
        assert!(pairs(&s(&["I"]), "-ecp").is_err());
        let dir = std::env::temp_dir().join(format!("mopt_spc_{}", std::process::id()));
        std::fs::write(&dir, "# paths\norca::/opt/orca/orca\n\nbad line\n").unwrap();
        assert_eq!(
            read_software_paths(&dir).unwrap(),
            vec![("orca".to_string(), "/opt/orca/orca".to_string())]
        );
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn one_calculator_is_chosen() {
        let name = |args: &[&str]| {
            let p = crate::cli::option::parser("optmain", crate::cli::spec::OPTMAIN)
                .parse(args)
                .unwrap();
            calculator_name(&p)
        };
        assert_eq!(name(&["a.xyz"]).unwrap(), "pyscf");
        // MultiOptPy's flag for PySCF.
        assert_eq!(name(&["a.xyz", "-pyscf"]).unwrap(), "pyscf");
        assert_eq!(name(&["a.xyz", "-pyscf", "-os", "pyscf"]).unwrap(), "pyscf");
        assert_eq!(
            name(&["a.xyz", "-xtb", "GFN2-xTB"]).unwrap(),
            "tblite:GFN2-xTB"
        );
        assert_eq!(
            name(&["a.xyz", "-os", "ase:pkg.mod:Class"]).unwrap(),
            "ase:pkg.mod:Class"
        );
        for clash in [
            &["a.xyz", "-pyscf", "-os", "lj"][..],
            &["a.xyz", "-pyscf", "-xtb", "GFN2-xTB"],
            &["a.xyz", "-os", "lj", "-dxtb", "GFN1-xTB"],
        ] {
            assert!(name(clash).is_err(), "{clash:?}");
        }
    }
}
