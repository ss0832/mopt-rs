//! Charge and spin multiplicity on the command line, the same for every command and calculator.
//!
//! - `-elec Q` is the total (formal) charge and `-spin M` the spin multiplicity `2S + 1`, for
//!   every calculator. MultiOptPy asks for `2S` instead when PySCF is used; mopt-rs converts in the
//!   PySCF adapter, so one input means the same state everywhere.
//! - `-unpaired N` (mopt-rs) gives the spin state as the number of unpaired electrons `2S`
//!   instead (as the "spin" of PySCF, xTB's `uhf`, and several Rust engines count it); with both,
//!   they must agree. It sets the state only; the unrestricted reference for singlets is `-u`.
//! - Commands with several states (`ieipmain`: reactant and product) take one value per state,
//!   `-elec Q1 Q2 -spin M1 M2`; a single value applies to every state.
//! - Other places that name a state (`-mf NAME Q M`, `-addint E Q M`) use the same parser.
//! - A value given on the command line wins over one read from the structure file (the line
//!   `charge multiplicity` of MultiOptPy's `.xyz` files), which wins over the default (neutral
//!   singlet). Charge and multiplicity are resolved separately. In MultiOptPy the line of the
//!   file wins over the command line; here the command line is the last word.
//!
//! MultiOptPy: `interface.py:optimizeparser` (`-elec`, `-spin`) and their use in
//!  `Calculator/*_calculation_tools.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::core_infra::arg_parser::ParsedArgs;
use crate::domain::SpinCharge;
use crate::error::{MoptError, MoptResult};

/// Charge and multiplicity given explicitly (each may be missing).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StateSpec {
    /// `-elec`.
    pub charge: Option<i64>,
    /// `-spin` (multiplicity `2S + 1`).
    pub multiplicity: Option<i64>,
}

impl StateSpec {
    /// The state with missing values taken from `fallback` (a structure file or the default).
    pub fn resolve(&self, fallback: SpinCharge) -> MoptResult<SpinCharge> {
        SpinCharge::checked(
            self.charge.unwrap_or(i64::from(fallback.charge)),
            self.multiplicity
                .unwrap_or(i64::from(fallback.multiplicity)),
        )
    }
}

impl StateSpec {
    /// The state of the structure file `input`: what the command line gave, then the line
    /// `charge multiplicity` of the file, then the neutral singlet.
    pub fn of_file(&self, input: &std::path::Path) -> MoptResult<SpinCharge> {
        let in_file = crate::io::xyz::state_of_xyz_file(input)?;
        let default = SpinCharge::default();
        SpinCharge::checked(
            self.charge
                .or(in_file.map(|(charge, _)| charge))
                .unwrap_or(i64::from(default.charge)),
            self.multiplicity
                .or(in_file.map(|(_, multiplicity)| multiplicity))
                .unwrap_or(i64::from(default.multiplicity)),
        )
    }
}

impl StateSpec {
    /// What [`Self::of_file`] takes from the line `charge multiplicity` of `input`, for the
    /// log; `None` when the command line gives both values or the file has no such line.
    pub fn file_note(&self, input: &std::path::Path) -> Option<String> {
        let (charge, multiplicity) = crate::io::xyz::state_of_xyz_file(input).ok()??;
        let what = match (self.charge, self.multiplicity) {
            (None, None) => format!("Charge {charge} and multiplicity {multiplicity} are"),
            (None, Some(_)) => format!("Charge {charge} is"),
            (Some(_), None) => format!("Multiplicity {multiplicity} is"),
            (Some(_), Some(_)) => return None,
        };
        Some(format!(
            "{what} read from the line `{charge} {multiplicity}` of {} (the command line \
             gives none).",
            input.display()
        ))
    }
}

fn int(token: &str, what: &str) -> MoptResult<i64> {
    token
        .trim()
        .parse::<i64>()
        .map_err(|e| MoptError::parse(format!("{what}: '{token}' is not an integer ({e})")))
}

/// A `charge multiplicity` pair from two tokens (as in `-mf NAME Q M`).
pub fn parse_pair(charge: &str, multiplicity: &str, context: &str) -> MoptResult<SpinCharge> {
    SpinCharge::checked(
        int(charge, &format!("{context} charge"))?,
        int(multiplicity, &format!("{context} multiplicity"))?,
    )
}

fn values(p: &ParsedArgs, key: &str, what: &str) -> MoptResult<Vec<i64>> {
    if !p.is_present(key) {
        return Ok(Vec::new());
    }
    p.get_list(key).iter().map(|t| int(t, what)).collect()
}

/// `-elec` / `-spin` for `n` states (one value applies to all).
pub fn state_specs(p: &ParsedArgs, n: usize) -> MoptResult<Vec<StateSpec>> {
    let q = values(p, "electronic_charge", "-elec")?;
    let m = values(p, "spin_multiplicity", "-spin")?;
    let u = values(p, "unpaired_electrons", "-unpaired")?;
    let pick = |v: &[i64], k: usize, flag: &str| -> MoptResult<Option<i64>> {
        match v.len() {
            0 => Ok(None),
            1 => Ok(Some(v[0])),
            len if len == n => Ok(Some(v[k])),
            len => Err(MoptError::parse(format!(
                "{flag} takes one value or {n} (one per state), got {len}"
            ))),
        }
    };
    (0..n)
        .map(|k| {
            let from_spin = pick(&m, k, "-spin")?;
            let from_unpaired = pick(&u, k, "-unpaired")?
                .map(|v| {
                    if v < 0 {
                        Err(MoptError::parse(format!(
                            "-unpaired: the number of unpaired electrons cannot be negative ({v})"
                        )))
                    } else {
                        Ok(v + 1)
                    }
                })
                .transpose()?;
            let multiplicity = match (from_spin, from_unpaired) {
                (Some(a), Some(b)) if a != b => {
                    return Err(MoptError::parse(format!(
                        "-spin {a} (multiplicity) and -unpaired {} (unpaired electrons) disagree",
                        b - 1
                    )))
                }
                (a, b) => a.or(b),
            };
            Ok(StateSpec {
                charge: pick(&q, k, "-elec")?,
                multiplicity,
            })
        })
        .collect()
}

/// `p` with the state that its first input names in a line `charge multiplicity` written
/// into the arguments, where the command line gives none (a directory, a file without such
/// a line, or a file that cannot be read leaves `p` as it is). For the commands that start
/// further runs on files they write themselves (the points of a scan, the samples of a
/// search, the optimizations after a path): those files have no such line, and the runs
/// get their state from the arguments that are handed on.
pub fn with_state_of_input(p: &ParsedArgs) -> ParsedArgs {
    let mut out = p.clone();
    let named = p
        .positionals
        .first()
        .and_then(|first| crate::io::xyz::state_of_xyz_file(first).ok().flatten());
    let Some((charge, multiplicity)) = named else {
        return out;
    };
    let mut give = |key: &str, value: i64| {
        out.values.insert(key.to_string(), vec![value.to_string()]);
        out.present.insert(key.to_string());
    };
    if !p.is_present("electronic_charge") {
        give("electronic_charge", charge);
    }
    if !p.is_present("spin_multiplicity") && !p.is_present("unpaired_electrons") {
        give("spin_multiplicity", multiplicity);
    }
    out
}

/// `-elec` / `-spin` of a command with one state.
pub fn state_spec(p: &ParsedArgs) -> MoptResult<StateSpec> {
    Ok(state_specs(p, 1)?.remove(0))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::Element;

    #[test]
    fn pairs_and_resolution() {
        assert_eq!(
            parse_pair("-1", "2", "-mf").unwrap(),
            SpinCharge::new(-1, 2)
        );
        assert!(
            parse_pair("0", "0", "-mf").is_err(),
            "multiplicity 0 is rejected"
        );
        assert!(parse_pair("x", "1", "-mf").is_err());
        let file = SpinCharge::new(1, 2);
        let only_charge = StateSpec {
            charge: Some(0),
            multiplicity: None,
        };
        assert_eq!(only_charge.resolve(file).unwrap(), SpinCharge::new(0, 2));
        assert_eq!(StateSpec::default().resolve(file).unwrap(), file);
    }

    #[test]
    fn the_state_of_a_file_under_the_command_line() {
        let dir = std::env::temp_dir().join(format!("mopt_charge_spin_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let anion = dir.join("anion.xyz");
        std::fs::write(&anion, "2\n-1 1\nO 0 0 0\nH 0 0 0.96\n").unwrap();
        let plain = dir.join("plain.xyz");
        std::fs::write(&plain, "2\n\nO 0 0 0\nH 0 0 0.96\n").unwrap();
        let nothing = StateSpec::default();
        assert_eq!(nothing.of_file(&anion).unwrap(), SpinCharge::new(-1, 1));
        assert_eq!(nothing.of_file(&plain).unwrap(), SpinCharge::default());
        // The command line is the last word, value by value.
        let neutral = StateSpec {
            charge: Some(0),
            multiplicity: None,
        };
        assert_eq!(neutral.of_file(&anion).unwrap(), SpinCharge::new(0, 1));
        let triplet = StateSpec {
            charge: None,
            multiplicity: Some(3),
        };
        assert_eq!(triplet.of_file(&anion).unwrap(), SpinCharge::new(-1, 3));
        // What the log says about it.
        assert!(nothing
            .file_note(&anion)
            .unwrap()
            .starts_with("Charge -1 and multiplicity 1 are read from the line `-1 1`"));
        assert!(neutral
            .file_note(&anion)
            .unwrap()
            .starts_with("Multiplicity 1 is"));
        assert!(triplet
            .file_note(&anion)
            .unwrap()
            .starts_with("Charge -1 is"));
        assert!(nothing.file_note(&plain).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_state_of_the_input_goes_into_the_arguments() {
        use crate::cli::option::parser;
        use crate::cli::spec::RELAXEDSCAN;
        let dir = std::env::temp_dir().join(format!("mopt_state_args_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let anion = dir.join("anion.xyz");
        std::fs::write(&anion, "2\n-1 1\nO 0 0 0\nH 0 0 0.96\n").unwrap();
        let file = anion.to_string_lossy().into_owned();
        let p = parser("relaxedscan", RELAXEDSCAN);
        let filled = with_state_of_input(&p.parse([file.as_str()]).unwrap());
        assert_eq!(
            state_spec(&filled).unwrap(),
            StateSpec {
                charge: Some(-1),
                multiplicity: Some(1)
            }
        );
        // What the command line gives stays, value by value.
        let given = with_state_of_input(&p.parse([file.as_str(), "-unpaired", "2"]).unwrap());
        assert_eq!(
            state_spec(&given).unwrap(),
            StateSpec {
                charge: Some(-1),
                multiplicity: Some(3)
            }
        );
        // No such file or no such line: nothing is added.
        let missing = with_state_of_input(&p.parse(["no_such_file.xyz"]).unwrap());
        assert_eq!(state_spec(&missing).unwrap(), StateSpec::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unpaired_electrons_instead_of_multiplicity() {
        use crate::cli::option::parser;
        use crate::cli::spec::OPTMAIN;
        let p = parser("optmain", OPTMAIN);
        let spec = |args: &[&str]| {
            let mut v = vec!["x.xyz"];
            v.extend_from_slice(args);
            state_spec(&p.parse(v).unwrap())
        };
        assert_eq!(spec(&["-unpaired", "2"]).unwrap().multiplicity, Some(3));
        assert_eq!(spec(&["-unpaired", "0"]).unwrap().multiplicity, Some(1));
        assert_eq!(
            spec(&["-spin", "3", "-unpaired", "2"])
                .unwrap()
                .multiplicity,
            Some(3)
        );
        assert!(
            spec(&["-spin", "2", "-unpaired", "2"]).is_err(),
            "disagreeing values"
        );
        assert!(spec(&["-unpaired", "-1"]).is_err());
        assert_eq!(spec(&[]).unwrap().multiplicity, None);
    }

    #[test]
    fn electron_parity() {
        let water = [Element::O, Element::H, Element::H];
        assert!(SpinCharge::new(0, 1).check_electrons(&water).is_ok());
        assert!(SpinCharge::new(0, 3).check_electrons(&water).is_ok());
        assert!(SpinCharge::new(0, 2).check_electrons(&water).is_err());
        assert!(SpinCharge::new(1, 2).check_electrons(&water).is_ok());
        assert!(SpinCharge::new(11, 1).check_electrons(&water).is_err());
        assert_eq!(SpinCharge::new(0, 3).unpaired_electrons(), 2);
        assert_eq!(
            SpinCharge::from_unpaired_electrons(0, 2),
            SpinCharge::new(0, 3)
        );
    }
}
