//! Calculator names to calculators, for every command.
//!
//! The registry maps a name (`-os`, `-xtb`, `-dxtb`) and the program-independent
//! [`CalculatorOptions`] to a [`Calculator`]. The Rust potentials are registered by
//! [`CalculatorRegistry::native`]; other programs are added by whoever embeds mopt-rs, most
//! importantly the Python package, which installs a fallback that builds any engine of
//! `mopt_rs.calculators` (PySCF, Psi4, tblite, dxtb, ASE programs). A name nobody provides is
//! an error that says where it is available.
//!
//! MultiOptPy: `optimization.py:Optimize._init_calculation_module` and `_create_calculation` (the
//!  choice by `-os`, `-xtb`, `-dxtb`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use super::calculator::Calculator;
use super::counting::{CountingCalculator, EvaluationCounters, EvaluationCounts};
use super::native::meam::{Lattice, MeamPair, DEFAULT_CMAX, DEFAULT_CMIN};
use super::native::{
    Eam, Emt, Gfn0Xtb, Gfn1Xtb, Gfnff, LennardJones, Meam, MullerBrown, Tersoff,
    TruncatedLennardJones,
};
use crate::domain::{Angstrom, Element, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};

/// The potentials implemented in Rust.
pub const NATIVE_CALCULATORS: [&str; 9] = [
    "lj",
    "emt",
    "eam",
    "meam",
    "tersoff",
    "muller_brown",
    "gfn0xtb",
    "gfn1xtb",
    "gfnff",
];

/// The quantum-chemistry engine used when no calculator is named.
pub const DEFAULT_CALCULATOR: &str = "pyscf";

/// Program-independent calculator settings; each program maps them to its own input.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CalculatorOptions {
    /// Functional or method (`-func`); `None`: the program's default.
    pub method: Option<String>,
    /// Basis set (`-bs`).
    pub basis: Option<String>,
    /// Per-element basis sets (`-sub_bs I LanL2DZ`).
    pub sub_basis: Vec<(String, String)>,
    /// Per-element effective core potentials (`-ecp`).
    pub ecp: Vec<(String, String)>,
    /// Unrestricted reference also for singlets (`-u`).
    pub unrestricted: bool,
    /// DFT integration grid level (`-grid`).
    pub grid_level: Option<i64>,
    /// Threads (`-core`).
    pub threads: Option<usize>,
    /// Memory (`-mem`, e.g. `2GB`).
    pub memory: Option<String>,
    /// Implicit solvation for tight binding: (`alpb` or `cpcm`, solvent).
    pub solvation: Option<(String, String)>,
    /// `software_path.conf` entries (`name::path`); they override the paths of the engines
    /// built into mopt-rs and of the installed packages, which are used by default.
    pub software_paths: Vec<(String, String)>,
    /// The `software_path.conf` that was read (`None`: not found, nothing overridden).
    pub software_path_file: Option<String>,
    /// Program-specific settings (`-calc_opt key=value`), values as written.
    pub extra: Vec<(String, String)>,
    /// `-stability N`: the SCF solution is tested for instabilities every `N` evaluations
    /// and moved to a stabler one where it is unstable (0: never; calculators with an SCF
    /// only, `mopt_rs.calculators.stability`).
    pub stability: usize,
    /// `-stability_external`: what is done about an instability towards another kind of
    /// wavefunction (`off`, `report`, `switch`; empty: `report`).
    pub stability_external: String,
    /// `-es_follow`: an excited state (`-es N`, the states of `-mf`) is the state that was
    /// number `N` at the first structure and is followed from structure to structure by
    /// its overlaps ([`crate::pes::followed`]), not the state number `N` in the order of
    /// the energies at every structure.
    pub follow_states: bool,
    /// `-parallel N`: how many structures are calculated at the same time (0: not set).
    /// The pool of the command has that many threads ([`Self::pool_threads`]), and a
    /// calculator of the Python side that many copies in worker processes.
    pub parallel: usize,
    /// `-oniom` (with `-oniom_middle`, `-oniom_opt`): the calculator is the highest level
    /// of an ONIOM calculation on these regions ([`crate::pes::oniom`]).
    pub oniom: Option<crate::pes::oniom::OniomSpec>,
    /// For the calculator of a model system of ONIOM: what the model is made of. A
    /// calculator that needs a description of its atoms (a force field) makes that of the
    /// model from that of the whole structure with it; the others do not look at it.
    pub model_of: Option<ModelOrigin>,
}

/// What a model system of ONIOM is made of, in the atoms of the whole structure.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelOrigin {
    /// The atoms of the whole structure (from 0) in the order of the model system; the
    /// link atoms follow them.
    pub atoms: Vec<usize>,
    /// The cut bonds, one per link atom: the inner atom, the outer atom, and the factor
    /// of the placement (`g`, or the distance in Bohr).
    pub links: Vec<(usize, usize, f64)>,
    /// Whether the link atoms are at a fixed distance from the inner atom (else at the
    /// share `g` of the cut bond).
    pub at_distance: bool,
}

impl ModelOrigin {
    /// The origin of the model system of `region`.
    pub fn of(region: &crate::pes::oniom::Region) -> Self {
        Self {
            atoms: region.atoms.iter().map(|a| a.index()).collect(),
            links: region
                .links
                .iter()
                .map(|l| (l.inner.index(), l.outer.index(), l.factor))
                .collect(),
            at_distance: region.placement == crate::pes::oniom::LinkPlacement::Distance,
        }
    }
}

impl CalculatorOptions {
    /// How many structures are calculated at the same time: `-parallel N`, or the copies
    /// in worker processes asked for as `-calc_opt workers=N` (the larger one; 1: neither).
    #[must_use]
    pub fn structures_at_once(&self) -> usize {
        let workers = self
            .extra
            .iter()
            .rev()
            .find(|(key, _)| key == "workers")
            .and_then(|(_, value)| value.trim().parse::<usize>().ok())
            .unwrap_or(1);
        self.parallel.max(workers).max(1)
    }

    /// The threads of the pool of a command that was asked for `threads` (`-core`, the
    /// threads of one calculation): at least as many as there are structures to calculate
    /// at the same time, so that each of them has a thread that hands it out.
    #[must_use]
    pub fn pool_threads(&self, threads: usize) -> usize {
        threads.max(self.structures_at_once()).max(1)
    }

    /// The options of a lower level of an ONIOM calculation: its own settings `own`
    /// (`-oniom_opt low.KEY=VALUE`: `method` and `basis` are the method and the basis set,
    /// everything else a setting of that calculator), with the threads, the memory, and the
    /// paths of the programs of the run. What is given for the highest level (`-func`, `-bs`,
    /// `-calc_opt`, ...) belongs to its calculator and is not handed down.
    #[must_use]
    pub fn for_lower_level(&self, own: &[(String, String)]) -> Self {
        let named = |name: &str| {
            own.iter()
                .rev()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        };
        Self {
            method: named("method"),
            basis: named("basis"),
            threads: self.threads,
            memory: self.memory.clone(),
            software_paths: self.software_paths.clone(),
            software_path_file: self.software_path_file.clone(),
            extra: own
                .iter()
                .filter(|(k, _)| k != "method" && k != "basis")
                .cloned()
                .collect(),
            ..Self::default()
        }
    }

    /// What was read from `software_path.conf`, for the run log.
    pub fn software_path_report(&self) -> String {
        match &self.software_path_file {
            None => "software_path.conf: not found; using the engines built into mopt-rs and the installed Python packages"
                .to_string(),
            Some(file) if self.software_paths.is_empty() => {
                format!("software_path.conf: read {file} (no entries); using the built-in engines")
            }
            Some(file) => {
                let names: Vec<&str> = self.software_paths.iter().map(|(k, _)| k.as_str()).collect();
                format!(
                    "software_path.conf: read {file}; entries {} override the built-in engines' paths",
                    names.join(", ")
                )
            }
        }
    }
}

/// `calc` with its excited states followed between structures, where `options` ask for it
/// and the calculator has excited states.
fn followed(calc: Box<dyn Calculator>, options: &CalculatorOptions) -> Box<dyn Calculator> {
    if options.follow_states && calc.supports_excited_states() {
        Box::new(super::followed::FollowedStates::new(calc))
    } else {
        calc
    }
}

/// What a factory receives.
#[derive(Debug, Clone, Copy)]
pub struct CalculatorRequest<'a> {
    /// The full name as given (`pyscf`, `tblite:GFN1-xTB`, `ase:pm6_rs.ase:PM6`).
    pub name: &'a str,
    /// The part before `:`, lowercase.
    pub head: &'a str,
    /// The part after `:` (empty when absent).
    pub argument: &'a str,
    /// Settings.
    pub options: &'a CalculatorOptions,
    /// The starting structure (some potentials take parameters from its elements).
    pub system: &'a MolecularSystem,
}

/// Builds a calculator.
pub type CalculatorFactory =
    Arc<dyn Fn(&CalculatorRequest<'_>) -> MoptResult<Box<dyn Calculator>> + Send + Sync>;

/// Names to calculator factories.
#[derive(Clone, Default)]
pub struct CalculatorRegistry {
    factories: BTreeMap<String, CalculatorFactory>,
    fallback: Option<CalculatorFactory>,
    /// The evaluations of the calculators created here (shared by clones).
    counters: Arc<EvaluationCounters>,
    /// The evaluations of an earlier run, offered to the journals of the runs of this
    /// registry (a workflow restarted from its directory).
    replay: Option<Arc<dyn super::replay::ReplaySource>>,
}

impl std::fmt::Debug for CalculatorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculatorRegistry")
            .field("names", &self.names())
            .field("fallback", &self.fallback.is_some())
            .finish()
    }
}

/// `true` or `false` of a setting.
fn switch(what: &str, value: &str) -> MoptResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(MoptError::config(format!(
            "{what} must be true or false, got '{value}'"
        ))),
    }
}

/// A length in Angstrom of a setting.
fn length(what: &str, value: &str) -> MoptResult<f64> {
    value.trim().parse().map_err(|_| {
        MoptError::config(format!(
            "{what} must be a length in Angstrom, got '{value}'"
        ))
    })
}

/// Lennard-Jones for a run: all pairs as in MultiOptPy, or with `-calc_opt cutoff=ANGSTROM`
/// the pairs within the cutoff, switched off over `switch=ANGSTROM` below it (a tenth of
/// the cutoff by default, 0 for none).
fn lj_from(q: &CalculatorRequest<'_>) -> MoptResult<Box<dyn Calculator>> {
    let (mut cutoff, mut width) = (None, None);
    for (key, value) in &q.options.extra {
        match key.as_str() {
            "cutoff" => cutoff = Some(length("lj: cutoff", value)?),
            "switch" => width = Some(length("lj: switch", value)?),
            other => {
                return Err(MoptError::config(format!(
                    "lj: unknown setting '{other}' (known: cutoff, switch)"
                )))
            }
        }
    }
    match (cutoff, width) {
        (None, None) => Ok(Box::new(LennardJones)),
        (None, Some(_)) => Err(MoptError::config(
            "lj: switch needs a cutoff (-calc_opt cutoff=ANGSTROM)",
        )),
        (Some(cutoff), width) => Ok(Box::new(TruncatedLennardJones::new(
            Quantity::<Angstrom>::new(cutoff).to(),
            width.map(|w| Quantity::<Angstrom>::new(w).to()),
        )?)),
    }
}

/// The embedded-atom method for a run; `-calc_opt switch=ANGSTROM` sets the width of the
/// switching range below the cutoff (0 for none).
fn eam_from(q: &CalculatorRequest<'_>) -> MoptResult<Eam> {
    let mut calc = Eam::zhou2004(&q.system.elements)?;
    for (key, value) in &q.options.extra {
        match key.as_str() {
            "switch" => calc = calc.with_switch_width(length("eam: switch", value)?)?,
            other => {
                return Err(MoptError::config(format!(
                    "eam: unknown setting '{other}' (known: switch)"
                )))
            }
        }
    }
    Ok(calc)
}

/// Effective medium theory for a run; `-calc_opt switch=ANGSTROM` switches the interactions
/// off below the cutoff of the neighbor list (not by default: as in ASE).
fn emt_from(q: &CalculatorRequest<'_>) -> MoptResult<Emt> {
    let mut calc = Emt::new();
    for (key, value) in &q.options.extra {
        match key.as_str() {
            "switch" => calc = calc.with_switch_width(length("emt: switch", value)?)?,
            other => {
                return Err(MoptError::config(format!(
                    "emt: unknown setting '{other}' (known: switch)"
                )))
            }
        }
    }
    Ok(calc)
}

/// The modified embedded-atom method for a run, with the library parameters of the elements
/// of the starting structure. `-calc_opt cutoff=ANGSTROM`, `cmin=C` and `cmax=C` (screening),
/// `zbl=false` (no nuclear repulsion at short distances), `smooth=true` (an infinitely
/// smooth step in place of the cutoff function of the model), and
/// `pair=A-B,LATTICE,EC,RE,ALPHA` (reference of a pair of different elements: `fcc`, `bcc`,
/// `hcp`, `dia`, `b1`, or `b2`, cohesive energy in eV, distance in Angstrom).
fn meam_from(q: &CalculatorRequest<'_>) -> MoptResult<Meam> {
    let mut calc = Meam::library(&q.system.elements)?;
    let number = |key: &str, value: &str| -> MoptResult<f64> {
        value
            .trim()
            .parse()
            .map_err(|_| MoptError::config(format!("meam: {key} must be a number, got '{value}'")))
    };
    let (mut cmin, mut cmax) = (DEFAULT_CMIN, DEFAULT_CMAX);
    for (key, value) in &q.options.extra {
        match key.as_str() {
            "cutoff" => calc = calc.with_cutoff(number(key, value)?)?,
            "cmin" => cmin = number(key, value)?,
            "cmax" => cmax = number(key, value)?,
            "zbl" => calc = calc.with_nuclear_repulsion(switch("meam: zbl", value)?),
            "smooth" => calc = calc.with_smooth_cutoff(switch("meam: smooth", value)?),
            "pair" => {
                let fields: Vec<&str> = value.split(',').map(str::trim).collect();
                let (names, lattice, numbers) = match fields.as_slice() {
                    [names, lattice, ec, re, alpha] => (names, lattice, [ec, re, alpha]),
                    _ => {
                        return Err(MoptError::config(format!(
                            "meam: pair expects A-B,LATTICE,EC,RE,ALPHA, got '{value}'"
                        )))
                    }
                };
                let (a, b) = names.split_once('-').ok_or_else(|| {
                    MoptError::config(format!(
                        "meam: pair expects two elements as A-B, got '{names}'"
                    ))
                })?;
                let [ec, re, alpha] = numbers;
                calc = calc.with_pair(
                    Element::parse_token(a)?,
                    Element::parse_token(b)?,
                    MeamPair {
                        lattice: Lattice::parse(lattice)?,
                        ec: number("the cohesive energy of pair", ec)?,
                        re: number("the distance of pair", re)?,
                        alpha: number("alpha of pair", alpha)?,
                    },
                )?;
            }
            other => {
                return Err(MoptError::config(format!(
                    "meam: unknown setting '{other}' (known: cutoff, cmin, cmax, zbl, smooth, \
                     pair)"
                )))
            }
        }
    }
    calc.with_screening(cmin, cmax)
}

/// GFN1-xTB for a run: `-calc_opt scc_guess=sad|previous` (where the SCC starts),
/// `diis_patience=N` (iterations without a new lowest commutator before CDIIS takes over from
/// ADIIS), `diis_depth=N`, `diis_damping=B` (the share of the last input density kept while
/// ADIIS runs), `charge_limit=E` (the atomic charge beyond which a converged SCC counts as a
/// runaway and is repeated with stronger damping), `hessian_step=BOHR`.
/// GFN0-xTB for a run: `-calc_opt hessian=analytic|numerical` (analytic by default) and
/// `-calc_opt hessian_step=BOHR` for the differences.
/// `analytic` (true) or `numerical` (central differences that follow the state) for the
/// gradients of the excited states of `method`.
fn excited_gradient(method: &str, value: &str) -> MoptResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "analytic" => Ok(true),
        "numerical" | "numeric" => Ok(false),
        other => Err(MoptError::config(format!(
            "{method}: excited_gradient must be analytic or numerical, got '{other}'"
        ))),
    }
}

fn gfn0_from(q: &CalculatorRequest<'_>) -> MoptResult<Gfn0Xtb> {
    let mut calc = Gfn0Xtb::default();
    for (key, value) in &q.options.extra {
        match key.as_str() {
            "hessian_step" => {
                calc.hessian_step_bohr = value.trim().parse::<f64>().map_err(|_| {
                    MoptError::config(format!(
                        "gfn0xtb: hessian_step must be a number, got '{value}'"
                    ))
                })?;
            }
            "hessian" => {
                calc.analytic_hessian = match value.trim().to_ascii_lowercase().as_str() {
                    "analytic" => true,
                    "numerical" | "numeric" => false,
                    other => {
                        return Err(MoptError::config(format!(
                            "gfn0xtb: hessian must be analytic or numerical, got '{other}'"
                        )))
                    }
                };
            }
            "excited_gradient" => calc.analytic_excited = excited_gradient("gfn0xtb", value)?,
            other => {
                return Err(MoptError::config(format!(
                    "gfn0xtb: unknown setting '{other}' (known: hessian, hessian_step, \
                     excited_gradient)"
                )))
            }
        }
    }
    Ok(calc)
}

fn gfn1_from(q: &CalculatorRequest<'_>) -> MoptResult<Gfn1Xtb> {
    use crate::pes::native::gfn1::{SccBroyden, SccGuess};
    let mut calc = Gfn1Xtb::default();
    let number = |key: &str, value: &str| -> MoptResult<f64> {
        value.trim().parse::<f64>().map_err(|_| {
            MoptError::config(format!("gfn1xtb: {key} must be a number, got '{value}'"))
        })
    };
    for (key, value) in &q.options.extra {
        match key.as_str() {
            "scc_guess" => {
                calc.guess = match value.trim().to_ascii_lowercase().as_str() {
                    "sad" => SccGuess::Sad,
                    "previous" | "last" => SccGuess::Previous,
                    other => {
                        return Err(MoptError::config(format!(
                            "gfn1xtb: scc_guess must be sad or previous, got '{other}'"
                        )))
                    }
                };
            }
            "diis_patience" | "diis_depth" | "diis_window" => {
                let v = number(key, value)?;
                if v.is_nan() || v < 1.0 || v.fract() != 0.0 {
                    return Err(MoptError::config(format!(
                        "gfn1xtb: {key} must be a whole number of at least 1, got '{value}'"
                    )));
                }
                if key == "diis_depth" {
                    calc.diis_depth = v as usize;
                } else if key == "diis_window" {
                    calc.diis_window = v as usize;
                } else {
                    calc.diis_patience = v as usize;
                }
            }
            "diis_rate" => {
                calc.diis_rate = if value.trim().eq_ignore_ascii_case("off") {
                    None
                } else {
                    let v = number(key, value)?;
                    if !(v > 0.0 && v <= 1.0) {
                        return Err(MoptError::config(format!(
                            "gfn1xtb: diis_rate must be above 0 and at most 1, or off, got \
                             '{value}'"
                        )));
                    }
                    Some(v)
                };
            }
            "diis_damping" => {
                let v = number(key, value)?;
                if !(0.0..1.0).contains(&v) {
                    return Err(MoptError::config(format!(
                        "gfn1xtb: diis_damping must be at least 0 and below 1, got '{value}'"
                    )));
                }
                calc.diis_damping = v;
            }
            "hessian_step" => calc.hessian_step_bohr = number(key, value)?,
            "hessian" => {
                calc.analytic_hessian = match value.trim().to_ascii_lowercase().as_str() {
                    "analytic" => true,
                    "numerical" | "numeric" => false,
                    other => {
                        return Err(MoptError::config(format!(
                            "gfn1xtb: hessian must be analytic or numerical, got '{other}'"
                        )))
                    }
                };
            }
            "excited" => {
                use crate::pes::native::gfn1::ExcitedStates;
                calc.excited = match value.trim().to_ascii_lowercase().as_str() {
                    "tda" => ExcitedStates::Tda,
                    "stda" => ExcitedStates::Stda,
                    other => {
                        return Err(MoptError::config(format!(
                            "gfn1xtb: excited must be tda or stda, got '{other}'"
                        )))
                    }
                };
            }
            "excitation_step" => calc.excitation_step_bohr = number(key, value)?,
            "excited_gradient" => calc.analytic_excited = excited_gradient("gfn1xtb", value)?,
            "charge_limit" => {
                let v = number(key, value)?;
                if v.is_nan() || v <= 0.0 {
                    return Err(MoptError::config(format!(
                        "gfn1xtb: charge_limit must be positive, got '{value}'"
                    )));
                }
                calc.charge_limit = v;
            }
            "scc_broyden" => calc.broyden = SccBroyden::parse(value)?,
            other => {
                return Err(MoptError::config(format!(
                    "gfn1xtb: unknown setting '{other}' (known: scc_guess, diis_patience, \
                     diis_depth, diis_damping, diis_rate, diis_window, charge_limit, scc_broyden, \
                     hessian, hessian_step, excited, excitation_step, excited_gradient)"
                )))
            }
        }
    }
    Ok(calc)
}

/// GFN-FF for a run: the topology from the starting structure, `-calc_opt refresh_topology=true`
/// to set it up at every structure, `-calc_opt hessian=analytic|numerical` (analytic by
/// default), `-calc_opt hessian_step=BOHR` for the Hessian differences,
/// `-calc_opt hb_switch=on|off|SHARE` for the end of the hydrogen and halogen bond terms at
/// the thresholds of their lists (`on`: switched off smoothly over the last 10 % of the
/// threshold, the default; a number: over that share; `off`: dropped at the threshold, as a
/// single point of xtb).
fn gfnff_from(q: &CalculatorRequest<'_>) -> MoptResult<Gfnff> {
    // `-calc_opt topology=FILE`: the topology of a reference structure (that of the run whose
    // result is the input here), so that the steps of a workflow see one force field.
    let reference = q
        .options
        .extra
        .iter()
        .rev()
        .find(|(k, _)| k == "topology")
        .map(|(_, v)| Gfnff::from_reference_file(std::path::Path::new(v.trim()), q.system))
        .transpose()?;
    let has_reference = reference.is_some();
    let mut calc = reference.unwrap_or_else(|| Gfnff::starting_from(q.system));
    for (key, value) in &q.options.extra {
        match key.as_str() {
            "topology" => {}
            "refresh_topology" => {
                calc.refresh_topology = switch("gfnff: refresh_topology", value)?;
            }
            "hessian_step" => {
                calc.hessian_step_bohr = value.trim().parse().map_err(|_| {
                    MoptError::config(format!(
                        "gfnff: hessian_step must be a number, got '{value}'"
                    ))
                })?;
            }
            "hessian" => {
                calc.analytic_hessian = match value.trim().to_ascii_lowercase().as_str() {
                    "analytic" => true,
                    "numerical" | "numeric" => false,
                    other => {
                        return Err(MoptError::config(format!(
                            "gfnff: hessian must be analytic or numerical, got '{other}'"
                        )))
                    }
                };
            }
            "hb_switch" => {
                use crate::pes::native::gfnff::{HbCutoff, HB_SWITCH_SHARE};
                calc.hb_cutoff = match value.trim() {
                    "on" | "true" | "yes" => HbCutoff::Switched(HB_SWITCH_SHARE),
                    "off" | "false" | "no" => HbCutoff::Sharp,
                    share => match share.parse::<f64>() {
                        Ok(v) if v > 0.0 && v <= 1.0 => HbCutoff::Switched(v),
                        _ => {
                            return Err(MoptError::config(format!(
                                "gfnff: hb_switch is on, off, or a share of the threshold \
                                 between 0 and 1, got '{value}'"
                            )))
                        }
                    },
                };
            }
            other => {
                return Err(MoptError::config(format!(
                    "gfnff: unknown setting '{other}' (known: topology, refresh_topology, \
                     hessian, hessian_step, hb_switch)"
                )))
            }
        }
    }
    if calc.refresh_topology && has_reference {
        return Err(MoptError::config(
            "gfnff: topology=FILE keeps the topology of that structure, refresh_topology=true \
             sets it up anew at every structure: one of the two",
        ));
    }
    Ok(calc)
}

impl CalculatorRegistry {
    /// The Rust potentials only.
    pub fn native() -> Self {
        // Every use of the library that makes its calculators here gets results that do not
        // depend on the number of threads.
        crate::core_infra::parallelism::deterministic_linear_algebra();
        let mut r = Self::default();
        r.register("lj", lj_from);
        r.register("emt", |q| Ok(Box::new(emt_from(q)?)));
        r.register("eam", |q| Ok(Box::new(eam_from(q)?)));
        r.register("meam", |q| Ok(Box::new(meam_from(q)?)));
        r.register("tersoff", |q| {
            Ok(Box::new(Tersoff::tersoff_1989(&q.system.elements)?))
        });
        r.register("muller_brown", |_| Ok(Box::new(MullerBrown::default())));
        r.alias("mb", "muller_brown");
        r.register("gfn0xtb", |q| Ok(Box::new(gfn0_from(q)?)));
        r.register("gfn1xtb", |q| Ok(Box::new(gfn1_from(q)?)));
        r.register("gfnff", |q| Ok(Box::new(gfnff_from(q)?)));
        r
    }

    /// Register `factory` under `name` (case-insensitive; replaces an earlier one).
    pub fn register(
        &mut self,
        name: &str,
        factory: impl Fn(&CalculatorRequest<'_>) -> MoptResult<Box<dyn Calculator>>
            + Send
            + Sync
            + 'static,
    ) {
        self.factories
            .insert(name.to_ascii_lowercase(), Arc::new(factory));
    }

    /// `alias` builds what `name` builds.
    pub fn alias(&mut self, alias: &str, name: &str) {
        if let Some(f) = self.factories.get(&name.to_ascii_lowercase()).cloned() {
            self.factories.insert(alias.to_ascii_lowercase(), f);
        }
    }

    /// The factory for names without their own entry (e.g. the Python engines).
    pub fn set_fallback(
        &mut self,
        factory: impl Fn(&CalculatorRequest<'_>) -> MoptResult<Box<dyn Calculator>>
            + Send
            + Sync
            + 'static,
    ) {
        self.fallback = Some(Arc::new(factory));
    }

    /// Registered names (without the fallback's).
    pub fn names(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// The registry whose runs take the evaluations of `source` where they ask for them
    /// again, before they compute (a workflow restarted from its directory).
    pub fn with_replay(mut self, source: Arc<dyn super::replay::ReplaySource>) -> Self {
        self.replay = Some(source);
        self
    }

    /// The source of earlier evaluations of the runs of this registry, if any.
    pub fn replay(&self) -> Option<Arc<dyn super::replay::ReplaySource>> {
        self.replay.clone()
    }

    /// The evaluations the calculators created by this registry (and its clones) made so
    /// far.
    pub fn counts(&self) -> EvaluationCounts {
        self.counters.counts()
    }

    /// The counters of this registry (shared with its clones), for parts that count into them
    /// themselves (the relaxations of an ONIOM environment).
    pub fn counters(&self) -> Arc<EvaluationCounters> {
        self.counters.clone()
    }

    /// The calculator `name` (empty or `none`: [`DEFAULT_CALCULATOR`]) for `system`. With an
    /// ONIOM spec in `options` this is refused: the ONIOM surface is made by
    /// [`Self::create_oniom`], which hands out a type that is not a calculator until the run
    /// says what it is for (so that nothing moves structures on it without microiterations
    /// by oversight).
    pub fn create(
        &self,
        name: &str,
        options: &CalculatorOptions,
        system: &MolecularSystem,
    ) -> MoptResult<Box<dyn Calculator>> {
        if options.oniom.is_some() {
            return Err(MoptError::unsupported(
                "ONIOM: this part of mopt-rs makes its calculator without saying what the \
                 ONIOM surface is for (CalculatorRegistry::create_oniom); every ONIOM run \
                 but the molecular dynamics has microiterations, and none is started \
                 without them by oversight",
            ));
        }
        self.create_one(name, options, system)
    }

    /// The ONIOM surface of whole structures like `system`, with `name` as its highest level
    /// and the spec of `options`.
    pub fn create_oniom(
        &self,
        name: &str,
        options: &CalculatorOptions,
        system: &MolecularSystem,
    ) -> MoptResult<crate::pes::oniom::WholeOniom> {
        let Some(spec) = &options.oniom else {
            return Err(MoptError::config(
                "create_oniom: the calculator options have no ONIOM spec",
            ));
        };
        let plain = CalculatorOptions {
            oniom: None,
            ..options.clone()
        };
        let regions = spec.resolve(system)?;
        let mut names = vec![name.to_string()];
        names.extend(spec.lower_levels.iter().cloned());
        let mut oniom = crate::pes::oniom::OniomCalculator::build(
            names.len(),
            regions,
            system,
            |level, part, region| {
                // The highest level with the options of the run, each lower one with its
                // own; the calculator of a model system is told what the model is made of.
                let mut own = if level == 0 {
                    plain.clone()
                } else {
                    plain.for_lower_level(spec.options_of_lower_level(level - 1))
                };
                own.model_of = region.map(ModelOrigin::of);
                self.create_uncounted(&names[level], &own, part)
            },
        )?;
        if let Some(embedding) = &spec.embedding {
            oniom = oniom.with_electronic_embedding(embedding, system)?;
        }
        let model = oniom.model_part();
        Ok(
            crate::pes::oniom::WholeOniom::new(Box::new(CountingCalculator::new(
                Box::new(oniom),
                self.counters.clone(),
            )))
            .with_model_part(model),
        )
    }

    /// [`Self::create`] without ONIOM and without counting its evaluations (a part of a
    /// calculator that is counted as a whole).
    pub fn create_uncounted(
        &self,
        name: &str,
        options: &CalculatorOptions,
        system: &MolecularSystem,
    ) -> MoptResult<Box<dyn Calculator>> {
        crate::core_infra::parallelism::deterministic_linear_algebra();
        let (factory, req_name, head, argument) = self.factory_for(name)?;
        let req = CalculatorRequest {
            name: &req_name,
            head: &head,
            argument: &argument,
            options,
            system,
        };
        factory(&req).map(|calc| followed(calc, options))
    }

    /// The factory of `name` with the parts of the name.
    #[allow(clippy::type_complexity)]
    fn factory_for(&self, name: &str) -> MoptResult<(CalculatorFactory, String, String, String)> {
        let name = name.trim();
        let name = if name.is_empty() || name.eq_ignore_ascii_case("none") {
            DEFAULT_CALCULATOR
        } else {
            name
        };
        let (head, argument) = name.split_once(':').unwrap_or((name, ""));
        let head = head.to_ascii_lowercase();
        match self.factories.get(&head).or(self.fallback.as_ref()) {
            Some(f) => Ok((f.clone(), name.to_string(), head, argument.to_string())),
            None => Err(MoptError::unsupported(format!(
                "calculator '{name}' is provided by the Python package mopt-rs: run the \
                 pip-installed command (e.g. `optmain`) or `mopt-py <command>`; this program \
                 has only {}",
                self.names().join(", ")
            ))),
        }
    }

    /// One calculator `name`, counted.
    fn create_one(
        &self,
        name: &str,
        options: &CalculatorOptions,
        system: &MolecularSystem,
    ) -> MoptResult<Box<dyn Calculator>> {
        let name = name.trim();
        let name = if name.is_empty() || name.eq_ignore_ascii_case("none") {
            DEFAULT_CALCULATOR
        } else {
            name
        };
        let (head, argument) = name.split_once(':').unwrap_or((name, ""));
        let head = head.to_ascii_lowercase();
        let req = CalculatorRequest {
            name,
            head: &head,
            argument,
            options,
            system,
        };
        let counted = |calc: Box<dyn Calculator>| -> Box<dyn Calculator> {
            let mut notes = Vec::new();
            if options.follow_states && !calc.supports_excited_states() {
                notes.push(format!(
                    "-es_follow: {} has no excited states; nothing is followed",
                    calc.name()
                ));
            }
            let mut counting =
                CountingCalculator::new(followed(calc, options), self.counters.clone());
            if options.stability > 0 && !counting.checks_stability() {
                notes.push(format!(
                    "-stability {}: {} has no SCF solution to test; nothing is tested",
                    options.stability,
                    counting.name()
                ));
            }
            if !notes.is_empty() {
                counting = counting.with_note(notes.join("; "));
            }
            Box::new(counting)
        };
        if let Some(f) = self.factories.get(&head) {
            return f(&req).map(counted);
        }
        match &self.fallback {
            Some(f) => f(&req).map(counted),
            None => Err(MoptError::unsupported(format!(
                "calculator '{name}' is provided by the Python package mopt-rs: run the \
                 pip-installed command (e.g. `optmain`) or `mopt-py <command>`; this program \
                 has only {}",
                self.names().join(", ")
            ))),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};

    #[test]
    fn the_pool_has_a_thread_for_every_worker_copy() {
        let with = |extra: &[(&str, &str)]| CalculatorOptions {
            extra: extra
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
            ..CalculatorOptions::default()
        };
        assert_eq!(with(&[]).pool_threads(8), 8);
        assert_eq!(with(&[]).pool_threads(0), 1);
        // -core 1 with four copies: four structures at a time, one thread each.
        assert_eq!(with(&[("workers", "4")]).pool_threads(1), 4);
        assert_eq!(with(&[("workers", "2")]).pool_threads(8), 8);
        // The last setting counts; what is no number is left to the calculator to refuse.
        assert_eq!(
            with(&[("workers", "2"), ("workers", "6")]).pool_threads(1),
            6
        );
        assert_eq!(with(&[("workers", "many")]).pool_threads(3), 3);
        // -parallel says the same for every calculator.
        let parallel = CalculatorOptions {
            parallel: 5,
            ..with(&[("workers", "2")])
        };
        assert_eq!(parallel.structures_at_once(), 5);
        assert_eq!(parallel.pool_threads(1), 5);
        assert_eq!(with(&[]).structures_at_once(), 1);
    }

    fn argon() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::from_atomic_number(18).unwrap(); 2],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn names_aliases_fallback_and_default() {
        let mut r = CalculatorRegistry::native();
        let sys = argon();
        let o = CalculatorOptions::default();
        assert_eq!(r.create("LJ", &o, &sys).unwrap().name(), "lj");
        assert!(r.create("mb", &o, &sys).is_ok());
        let e = r.create("none", &o, &sys).err().unwrap().to_string();
        assert!(e.contains("pyscf") && e.contains("mopt-py"), "{e}");
        r.set_fallback(|q| {
            assert_eq!((q.head, q.argument), ("tblite", "GFN1-xTB"));
            Ok(Box::new(LennardJones))
        });
        assert!(r.create("tblite:GFN1-xTB", &o, &sys).is_ok());
    }
}
