//! What `ieipmain` does with its estimates when it is asked to go on (`-ieip_opt
//! optimize=on`; nothing without the setting): every estimate is optimized with `optmain`
//! in the run directory.
//!
//! - An estimate of a transition state: `optmain -order 1` in the state of the first
//!   structure, with the Hessian of the calculator at its first point (`optimize_hessian=
//!   none`: as `optmain` starts without options) and a vibrational analysis of what it
//!   finds (`optimize_check=off`: none). A structure with no or several imaginary
//!   frequencies is reported.
//! - An estimate of a crossing (DS-AFIR between two structures in two states): `optmain -mf
//!   opt_meci` for two states of one charge and multiplicity (a conical intersection),
//!   `-mf opt_mesx` otherwise, with the states of the two structures, and the analysis of
//!   the seam at what it finds. Whether the crossing leads to the two structures is then
//!   looked at on both surfaces (`optimize_irc=off`: not): a meta-IRC (`ircmain`, the path
//!   of steepest descent from a point that is not stationary) from the crossing in the
//!   state of each structure, whose end is compared with that structure.
//!
//! The calculator and its settings, the bias potentials, and the constraints of the run
//! (resolved at the first structure: the values the search held) are handed on;
//! `optimize_settings=FILE` names a settings file of `optmain` with anything else.
//! `optimized_estimates.csv` lists the optimizations, and the log has a line for each.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; `run_autots` does the steps one after the
//! other from its settings file).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::ieipmain_settings::IeipmainConfig;
use super::ircmain::{self, IrcmainConfig};
use super::nebmain_followup::CONSTRAINT_KEYS;
use super::optmain::{self, OptmainConfig};
use super::settings::{merge, parse_settings, read_settings, INPUT_KEY};
use super::spec::{IRCMAIN, OPTMAIN};
use crate::coord::alignment::kabsch_align_bohr;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{SpinCharge, Units};
use crate::error::{MoptResult, ResultExt};
use crate::io::xyz::read_xyz_file;
use crate::pes::constraint::Constraint;
use crate::pes::registry::CalculatorRegistry;

/// RMSD (Angstrom, after superposition, atoms in their order) below which the end of a path
/// from a crossing is called the structure it is compared with. A judgment value: the ends
/// of paths that reach the same minimum differ by the convergence of the path, a few 1e-3
/// Angstrom; other minima of small molecules are further apart than this.
pub const SAME_STRUCTURE_ANGSTROM: f64 = 0.1;

/// `-fc` of an optimization that gets the Hessian of the calculator at its first point
/// only: an interval no run reaches.
const AT_THE_START_ONLY: &str = "1000000";
/// Settings of the search that are not handed on: its iterations, its method of the
/// corrector, and the states of its two structures, which the optimization gets one by one.
const NOT_HANDED_ON: &[&str] = &[
    INPUT_KEY,
    "NSTEP",
    "opt_method",
    "calc_exact_hess",
    "use_model_hessian",
    "calc_model_hess",
    "saddle_order",
    "initial_hessian",
    "electronic_charge",
    "spin_multiplicity",
    "excited_state",
    "model_function",
    "unpaired_electrons",
];

/// The optimizations after a search (`-ieip_opt optimize=`, `optimize_hessian=`,
/// `optimize_check=`, `optimize_settings=`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FollowUp {
    /// Whether the estimates are optimized.
    pub on: bool,
    /// A transition state search starts from the Hessian of the calculator.
    pub exact_hessian: bool,
    /// The structures found get a vibrational (or seam) analysis.
    pub check: bool,
    /// From a crossing found, the paths of steepest descent on the two surfaces (meta-IRC)
    /// are followed and their ends compared with the two structures.
    pub descend: bool,
    /// A settings file of `optmain` with further settings.
    pub settings: Option<PathBuf>,
}

impl Default for FollowUp {
    fn default() -> Self {
        Self {
            on: false,
            exact_hessian: true,
            check: true,
            descend: true,
            settings: None,
        }
    }
}

/// The two states of a crossing: charge, multiplicity, and electronic state of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossingStates {
    /// The state of the first structure.
    pub first: (i32, u32, usize),
    /// The state of the second structure.
    pub second: (i32, u32, usize),
}

impl CrossingStates {
    /// The model function of `optmain` that finds the crossing: the projection for a
    /// conical intersection between two states of one charge and multiplicity, the one for
    /// a seam of crossing otherwise.
    pub fn model_function(&self) -> &'static str {
        if (self.first.0, self.first.1) == (self.second.0, self.second.1) {
            "opt_meci"
        } else {
            "opt_mesx"
        }
    }

    /// The options of `optmain` for the search of the crossing.
    pub fn options(&self) -> String {
        format!(
            "-elec {} -spin {} -es {} -mf {} {} {} {}",
            self.first.0,
            self.first.1,
            self.first.2,
            self.model_function(),
            self.second.0,
            self.second.1,
            self.second.2
        )
    }
}

/// One optimization after the search.
#[derive(Debug, Clone)]
pub struct OptimizedEstimate {
    /// The estimate it started from.
    pub estimate: PathBuf,
    /// Whether it looked for a crossing (a transition state otherwise).
    pub crossing: bool,
    /// Whether it converged.
    pub converged: bool,
    /// Its iterations.
    pub iterations: usize,
    /// The energy of what it found (with bias potentials; of a crossing: the mean of the
    /// two states as `optmain` reports it).
    pub energy: f64,
    /// Imaginary frequencies of what it found (transition states with a check).
    pub imaginary_modes: Option<usize>,
    /// The optimized structure.
    pub optimized: PathBuf,
    /// The directory of the optimization.
    pub directory: PathBuf,
    /// From a crossing: the end of the meta-IRC in the state of each of the two structures
    /// and its RMSD (Angstrom) from that structure.
    pub descents: Vec<(PathBuf, f64)>,
}

/// The settings of the `ircmain` run from the crossing `input` in the state `state`.
fn ircmain_config(
    config: &IeipmainConfig,
    input: &Path,
    state: (i32, u32, usize),
    constraints: &[Constraint],
) -> MoptResult<IrcmainConfig> {
    let text = |value: &str| JsonValue::String(value.to_string());
    let mut settings = BTreeMap::new();
    let document = JsonValue::parse(&config.settings_document)?;
    if let Some(given) = document.as_object() {
        for (key, value) in given {
            let shared = IRCMAIN.iter().any(|option| option.spec.key == key);
            // The constraints go on resolved, not as options to resolve anew.
            let constraint = CONSTRAINT_KEYS.contains(&key.as_str());
            if shared && !NOT_HANDED_ON.contains(&key.as_str()) && !constraint {
                settings.insert(key.clone(), value.clone());
            }
        }
    }
    settings.insert(
        INPUT_KEY.to_string(),
        JsonValue::Array(vec![text(&input.to_string_lossy())]),
    );
    settings.insert("electronic_charge".to_string(), text(&state.0.to_string()));
    settings.insert("spin_multiplicity".to_string(), text(&state.1.to_string()));
    settings.insert("excited_state".to_string(), text(&state.2.to_string()));
    let parsed = parse_settings(&JsonValue::Object(settings).to_string_compact(), IRCMAIN)?;
    let mut out = IrcmainConfig::from_args(&parsed)?;
    out.constraints = constraints.to_vec();
    Ok(out)
}

/// From the crossing `crossing`: the path of steepest descent in the state of each of the
/// two structures of the search, and how far its end is from that structure.
#[allow(clippy::too_many_arguments)]
fn descend_on_both_surfaces(
    config: &IeipmainConfig,
    crossing: &Path,
    states: &CrossingStates,
    constraints: &[Constraint],
    dir: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    log: &Logger,
) -> MoptResult<Vec<(PathBuf, f64)>> {
    let mut found = Vec::new();
    for (k, state) in [states.first, states.second].into_iter().enumerate() {
        let settings = ircmain_config(config, crossing, state, constraints)?;
        log.info(&format!(
            "From the crossing down the surface of structure {} (charge {}, multiplicity {}, \
             state {}): ircmain {}.",
            k + 1,
            state.0,
            state.1,
            state.2,
            crossing.display()
        ));
        log.flush();
        let runs = ircmain::run_with(&settings, dir, console, registry)
            .context(format!("the path from the crossing in state {}", k + 1))?;
        let Some(end) = runs
            .into_iter()
            .next()
            .and_then(|run| run.irc.endpoints.into_iter().next())
        else {
            log.warn(&format!(
                "The path from the crossing in the state of structure {} gave no end point.",
                k + 1
            ));
            continue;
        };
        let spin = SpinCharge {
            charge: state.0,
            multiplicity: state.1,
        };
        let reached = read_xyz_file(&end, spin, None).context(end.display())?;
        let wanted = config.inputs.get(k).unwrap_or(&config.inputs[0]);
        let reference = read_xyz_file(wanted, spin, None).context(wanted.display())?;
        let rmsd = kabsch_align_bohr(&reached, &reference, false)?.1 * Units::BOHR_TO_ANGSTROM;
        log.info(&format!(
            "The path in the state of structure {} ends {rmsd:.4} Angstrom (RMSD) from that \
             structure: {} ({}).",
            k + 1,
            if rmsd < SAME_STRUCTURE_ANGSTROM {
                "the crossing leads to it"
            } else {
                "another structure"
            },
            end.display()
        ));
        found.push((end, rmsd));
    }
    Ok(found)
}

/// The settings of the `optmain` run from `input`.
fn optmain_config(
    config: &IeipmainConfig,
    follow: &FollowUp,
    input: &Path,
    first_state: (i32, u32, usize),
    crossing: Option<&CrossingStates>,
    constraints: &[Constraint],
) -> MoptResult<OptmainConfig> {
    let text = |value: &str| JsonValue::String(value.to_string());
    let mut settings = BTreeMap::new();
    let document = JsonValue::parse(&config.settings_document)?;
    if let Some(given) = document.as_object() {
        for (key, value) in given {
            let shared = OPTMAIN.iter().any(|option| option.spec.key == key);
            let constraint = CONSTRAINT_KEYS.contains(&key.as_str());
            if shared && !NOT_HANDED_ON.contains(&key.as_str()) && !constraint {
                settings.insert(key.clone(), value.clone());
            }
        }
    }
    settings.insert(
        INPUT_KEY.to_string(),
        JsonValue::Array(vec![text(&input.to_string_lossy())]),
    );
    settings.insert(
        "electronic_charge".to_string(),
        text(&first_state.0.to_string()),
    );
    settings.insert(
        "spin_multiplicity".to_string(),
        text(&first_state.1.to_string()),
    );
    settings.insert(
        "excited_state".to_string(),
        text(&first_state.2.to_string()),
    );
    match crossing {
        Some(states) => {
            settings.insert(
                "model_function".to_string(),
                JsonValue::Array(vec![
                    text(states.model_function()),
                    text(&states.second.0.to_string()),
                    text(&states.second.1.to_string()),
                    text(&states.second.2.to_string()),
                ]),
            );
        }
        None => {
            settings.insert("saddle_order".to_string(), text("1"));
            if follow.exact_hessian {
                settings.insert("calc_exact_hess".to_string(), text(AT_THE_START_ONLY));
            }
        }
    }
    if follow.check {
        settings.insert("frequency_analysis".to_string(), JsonValue::Bool(true));
    }
    let mut parsed = parse_settings(&JsonValue::Object(settings).to_string_compact(), OPTMAIN)?;
    if let Some(file) = &follow.settings {
        // What the file says goes before what the search hands on, but for the input.
        let mut further = read_settings(file, OPTMAIN).context(file.display())?;
        further.positionals.clear();
        parsed = merge(parsed, further);
    }
    let mut out = OptmainConfig::from_args(&parsed)?;
    // Those of the search first; a settings file may add more.
    let added = std::mem::take(&mut out.settings.constraints);
    out.settings.constraints = constraints.to_vec();
    out.settings.constraints.extend(added);
    Ok(out)
}

/// Optimize the estimates `files` of the run in `dir`: to crossings of `crossing`, or to
/// transition states in the state `first_state` (charge, multiplicity, electronic state).
#[allow(clippy::too_many_arguments)]
pub fn optimize_estimates(
    follow: &FollowUp,
    config: &IeipmainConfig,
    files: &[PathBuf],
    first_state: (i32, u32, usize),
    crossing: Option<&CrossingStates>,
    constraints: &[Constraint],
    dir: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    log: &Logger,
) -> MoptResult<Vec<OptimizedEstimate>> {
    let mut done = Vec::new();
    let kind = if crossing.is_some() {
        "a crossing"
    } else {
        "a transition state"
    };
    for (k, file) in files.iter().enumerate() {
        // With the full path of the estimate the optimization gets its directory next to
        // it, in the directory of the search.
        let file = &std::path::absolute(file)?;
        let settings = optmain_config(config, follow, file, first_state, crossing, constraints)?;
        log.info(&format!(
            "Estimate {} to {kind}: optmain {}{} ({}).",
            k + 1,
            file.display(),
            crossing.map_or_else(|| " -order 1".to_string(), |c| format!(" {}", c.options())),
            settings.settings.methods.join(" ")
        ));
        log.flush();
        let runs = optmain::run_with(&settings, dir, console, registry)
            .context(format!("the optimization of estimate {}", k + 1))?;
        let Some(run) = runs.into_iter().next() else {
            continue;
        };
        let (energy, iterations) = run
            .summary
            .as_ref()
            .map_or((f64::NAN, 0), |s| (s.biased_energy.value(), s.iterations));
        let imaginary_modes = run.summary.as_ref().and_then(|s| s.imaginary_modes);
        log.info(&format!(
            "Estimate {} to {kind}: {} after {iterations} iteration(s), energy {energy:.10} \
             ({}).",
            k + 1,
            if run.converged {
                "converged"
            } else {
                "not converged"
            },
            run.optimized.display()
        ));
        if crossing.is_none() {
            if let Some(found) = imaginary_modes.filter(|n| run.converged && *n != 1) {
                log.warn(&format!(
                    "Estimate {}: the optimization to a transition state ended at a \
                     structure with {found} imaginary frequenc{} (1 expected): {}.",
                    k + 1,
                    if found == 1 { "y" } else { "ies" },
                    if found == 0 {
                        "a minimum, not a transition state"
                    } else {
                        "a saddle point of higher order"
                    }
                ));
            }
        } else if follow.check && run.converged {
            log.info(&format!(
                "The analysis of the seam at the crossing: {}",
                run.directory.join("seam.txt").display()
            ));
        }
        let mut descents = Vec::new();
        if let (Some(states), true, true) = (crossing, follow.descend, run.converged) {
            descents = descend_on_both_surfaces(
                config,
                &run.optimized,
                states,
                constraints,
                dir,
                console,
                registry,
                log,
            )?;
        }
        done.push(OptimizedEstimate {
            estimate: file.clone(),
            crossing: crossing.is_some(),
            converged: run.converged,
            iterations,
            energy,
            imaginary_modes,
            optimized: run.optimized,
            directory: run.directory,
            descents,
        });
    }
    let mut table = String::from(
        "estimate,kind,converged,iterations,energy_hartree,imaginary_modes,file,\
         rmsd_to_structure_1_angstrom,rmsd_to_structure_2_angstrom\n",
    );
    for (k, e) in done.iter().enumerate() {
        let _ = writeln!(
            table,
            "{},{},{},{},{:.12},{},{},{},{}",
            k + 1,
            if e.crossing {
                "crossing"
            } else {
                "transition_state"
            },
            e.converged,
            e.iterations,
            e.energy,
            e.imaginary_modes
                .map_or_else(String::new, |n| n.to_string()),
            e.optimized.display(),
            e.descents
                .first()
                .map_or_else(String::new, |d| format!("{:.6}", d.1)),
            e.descents
                .get(1)
                .map_or_else(String::new, |d| format!("{:.6}", d.1))
        );
    }
    std::fs::write(dir.join("optimized_estimates.csv"), table)?;
    Ok(done)
}
