//! The `mapper_settings` block of the settings file of `run_mapper`, with the `-map_opt`
//! settings and the options of the command line over it, as [`MapperSettings`]: MultiOptPy's
//! keys with their types checked as MultiOptPy checks them, the keys of mopt-rs, and the parts
//! that are not implemented yet (or only fit MultiOptPy's design) refused by name.
//!
//! MultiOptPy: `Entrypoints/mapper.py:run_mapper` (`merge_config`) and
//!  `Wrapper/mapper.py:ReactionNetworkMapper._validate_mapper_settings`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::json::JsonValue;
use crate::error::{MoptError, MoptResult};
use crate::workflow::mapper::{AfirSettings, Schedule, ScheduleMode};

/// Two points of plain variables closer than this (in the units of the variables) are the
/// same, as structures within `rmsd_threshold` are: a hundredth of the smallest distance
/// between two stationary points of the same kind of the built-in objectives (0.80, two minima
/// of `mb2d`; the others are 1.2 and more apart), and far above where optimizations of the same
/// point end (`docs/source/mapper.md`).
pub const DEFAULT_VARIABLE_TOLERANCE: f64 = 0.01;

/// Keys of `mapper_settings` that MultiOptPy knows.
const MULTIOPTPY_KEYS: &[&str] = &[
    "temperature_K",
    "rmsd_threshold",
    "max_iterations",
    "afir_gamma_kJmol",
    "max_pairs",
    "dist_lower_ang",
    "dist_upper_ang",
    "output_dir",
    "rng_seed",
    "active_atoms",
    "include_negative_gamma",
    "excluded_node_ids",
    "exclude_bond_rearrangement",
    "use_rcmc",
    "rcmc_temperature_K",
    "rcmc_reaction_time_s",
    "rcmc_start_node_id",
    "n_parallel",
    "worker_timeout_s",
    "energy_tolerance_kcalmol",
];

/// Keys of `mapper_settings` and `-map_opt` that mopt-rs adds (`method_KEY` as well).
const MOPT_KEYS: &[&str] = &[
    "method",
    "priority",
    "stop",
    "store",
    "html_report",
    "save_job_steps",
    "mirror",
    "refine",
    "crossings",
    "states",
    "crossing_pairs",
    "crossing_afir",
    "crossing_smf_alpha",
    "crossing_amf_alpha",
    "crossing_amf_branch",
    "crossing_step",
    "crossing_gap",
    "crossing_candidate_prominence",
    "crossing_candidate_distance",
    "crossing_parallel_stop",
    "crossing_parallel_gradient",
    "crossing_parallel_steps",
    "crossing_seam_check",
    "crossing_direct",
    "crossing_pushes",
    "crossing_from",
    "ts_states",
    "kinetics",
    "workers",
    "batch_size",
    "schedule",
    "verify_minima",
    "irc_hessians",
    "irc_end_threshold_cm1",
    "bifurcation_threshold_cm1",
    "variable_tolerance",
    "channels",
    "channel_margin",
    // Which atoms of an ONIOM network decide the identity of its nodes.
    "identity_scope",
    // The pushes of the cell itself (a variable cell), and the margin of the bonded pairs.
    "cell_pushes",
    "pair_bond_scale",
    // Pushes of pairs of molecules rather than of atoms.
    "afir_fragments",
    // The goal of a goal-directed search and its stop, the acceptance filter of new minima,
    // and the settings of priority=rrt.
    "higher_saddles",
    "max_saddle_order",
    "upward",
    "upward_modes",
    "confirm_bifurcations",
    "ts_conformers",
    "path_peaks",
    "entrance_structures",
    "entrance_contact",
    "max_bond_changes",
    "max_species_changes",
    "irc_hessian_interval",
    "confirm_dissociations",
    "dissociation_scan_step",
    "dissociation_scan_tolerance",
    "goal",
    "goal_stop",
    "goal_match",
    "accept_temperature",
    "rrt_goal_weight",
    "rrt_locality",
    // The settings of priority=kinetics (workflow::mapper::priority::KINETIC_KEYS).
    "kinetics_score",
    "kinetics_solver",
    "kinetics_time",
    "kinetics_temperature",
    "kinetics_start",
    "kinetics_energy",
    "kinetics_clamp",
    "kinetics_time_rule",
    "kinetics_tolerance",
    "kinetics_rate_theory",
    "kinetics_corrections",
    "kinetics_kappa",
];

/// The largest RMSD of two structures that are the same node (Angstrom), and the largest
/// difference of their energies ([`DEFAULT_ENERGY_TOLERANCE`]): the thresholds of the conformer
/// ensembles of CREST (P. Pracht, F. Bohle, S. Grimme, Phys. Chem. Chem. Phys. 2020, 22, 7169).
/// MultiOptPy's 0.30 Angstrom and 1 kcal/mol take different minima for one: in two runs of
/// 1000 jobs on the urea synthesis with GFN1-xTB (`docs/source/mapper_design.md`), 97 % of
/// the minima that the jobs reached agreed with their node to 0.005 Angstrom and 1e-4
/// kcal/mol, and of the others 10 and 4 were within 0.30 Angstrom and 1 kcal/mol of a node:
/// one of each is that node (0.003 and 0.04 Angstrom away), the others lie 0.09 to 0.30
/// Angstrom from it and are other minima (one pair in both runs). With these thresholds only
/// the first two are taken for their node.
pub const DEFAULT_RMSD_THRESHOLD: f64 = 0.125;

/// The largest difference of the energies of two structures that are the same node
/// (kcal/mol); see [`DEFAULT_RMSD_THRESHOLD`].
pub const DEFAULT_ENERGY_TOLERANCE: f64 = 0.05;

/// The typed settings of the mapper.
#[derive(Debug, Clone, PartialEq)]
pub struct MapperSettings {
    /// Temperature of the Boltzmann priority (K).
    pub temperature: f64,
    /// Largest RMSD of two structures that are the same (Angstrom).
    pub rmsd_threshold: f64,
    /// Largest distance of two points of plain variables that are the same (in the units of
    /// the variables).
    pub variable_tolerance: f64,
    /// Largest energy difference of two structures that are the same (kcal/mol).
    pub energy_tolerance: f64,
    /// Jobs of the run (0: no limit).
    pub max_jobs: usize,
    /// Whether the limit of jobs was given (`--max_iter`, `max_iterations`), not the default.
    pub max_jobs_given: bool,
    /// The AFIR pushes.
    pub afir: AfirSettings,
    /// Transition-state guesses per job.
    pub top_n: usize,
    /// Seed of the random choices.
    pub seed: u64,
    /// Nodes that are never explored.
    pub excluded: Vec<usize>,
    /// New minima with other bonds than their input are not explored.
    pub exclude_bond_rearrangement: bool,
    /// The way a minimum is explored (empty: the default of the surface, `afir_ts` for
    /// structures and `addf` for plain variables, chosen by [`super::mapper::MapperConfig`]).
    pub method: String,
    /// The choice of the next minimum.
    pub priority: String,
    /// Further stop rules.
    pub stop: Vec<String>,
    /// Further stores besides the files of the run directory.
    pub stores: Vec<String>,
    /// The HTML report.
    pub html_report: bool,
    /// Save live per-job progress and completed NEB iterations.
    pub save_job_steps: bool,
    /// Whether a mirror image is the same minimum.
    pub mirror: bool,
    /// Settings of a registered method (`method_KEY=VALUE`, without the prefix).
    pub method_extra: BTreeMap<String, String>,
    /// How the jobs run: how many at the same time, chosen together, and their time limit.
    pub schedule: Schedule,
    /// Every minimum is verified by its Hessian (`verify_minima`, on by default).
    pub verify_minima: bool,
    /// The IRC asks for an exact Hessian at every tenth point (`irc_hessians`, on by default; `calc_exact_hess` of
    /// `step4_settings` goes first).
    pub irc_hessians: bool,
    /// Magnitude of an imaginary mode that triggers a TS–TS link at an optimized IRC endpoint (cm^-1).
    pub irc_end_threshold_cm1: f64,
    /// How negative the curvature across the path at an exact Hessian of the IRC must be to
    /// record a possible bifurcation (cm^-1, `bifurcation_threshold_cm1`).
    pub bifurcation_threshold_cm1: f64,
    /// Dissociation channels are recognized (`channels`, on by default).
    pub channels: bool,
    /// Whether `channels` was given (not the default).
    pub channels_given: bool,
    /// The margin of their separation test over the sums of the van der Waals radii
    /// (`channel_margin`, Angstrom).
    pub channel_margin: f64,
    /// Which atoms of an ONIOM network decide whether two structures are one node
    /// (`identity_scope`; `None`: not given, the core).
    pub identity_scope: Option<super::mapper_oniom::IdentityScope>,
    /// Settings of the priority (`kinetics_KEY=VALUE` and the `rcmc_*` settings of MultiOptPy,
    /// without the prefix); with `kinetics`, also those of the kinetics at the end.
    pub priority_options: BTreeMap<String, String>,
    /// The kinetics of the network runs when the exploration ends (`kinetics`, off by
    /// default), with the settings `kinetics_*`.
    pub kinetics: bool,
    /// The goal structure of a goal-directed search (`goal=FILE`).
    pub goal: Option<PathBuf>,
    /// The run ends when the goal is joined to a start (`goal_stop`, on by default).
    pub goal_stop: bool,
    /// A minimum with the bond graph of the goal joined to a start counts as the goal
    /// (`goal_match=bonds`; `structure`, the default: the goal structure itself).
    pub goal_bonds: bool,
    /// The temperature (K) of the acceptance filter of new minima (`accept_temperature`;
    /// with `priority=rrt` [`DEFAULT_ACCEPT_TEMPERATURE`] when it is not given).
    pub accept_temperature: Option<f64>,
    /// Settings of `priority=rrt` (`rrt_KEY=VALUE`, without the prefix).
    pub rrt_options: BTreeMap<String, String>,
    /// Saddle points of order 2 or more are optimized and descended from (`higher_saddles`,
    /// off by default; experimental), up to this order (`max_saddle_order`, 2).
    pub higher_saddles: Option<usize>,
    /// Possible bifurcations of the IRCs are confirmed by minimizations across the path
    /// (`confirm_bifurcations`, off by default).
    pub confirm_bifurcations: bool,
    /// The conformers of every transition state are searched with its saddle direction held
    /// (`ts_conformers`, off by default; the settings of the search are the block
    /// `ts_conformer_settings` of the settings file).
    pub ts_conformers: bool,
    /// Which path peaks are saved (`path_peaks`: `failed`, the default; `all`; `off`).
    pub path_peaks: crate::workflow::mapper::PeakMode,
    /// The structures of the entrance channel of the first input (`entrance_structures`; 0:
    /// none) and how close its molecules are placed (`entrance_contact`, times the sum of the
    /// van der Waals radii of their nearest atoms).
    pub entrance_structures: usize,
    /// See [`Self::entrance_structures`].
    pub entrance_contact: f64,
    /// New minima are not explored when the bonds (the chemical species) changed more than
    /// this many times on the way from their input (`max_bond_changes`,
    /// `max_species_changes`; `None`: no limit).
    pub max_bond_changes: Option<usize>,
    /// See [`Self::max_bond_changes`].
    pub max_species_changes: Option<usize>,
    /// The points between the exact Hessians of the IRC where it computes them on its own
    /// (`irc_hessians`): `irc_hessian_interval`, 10 by default.
    pub irc_hessian_interval: usize,
    /// The relaxed scan that confirms a barrierless dissociation (`confirm_dissociations`, off
    /// by default; `dissociation_scan_step` in Angstrom, `dissociation_scan_tolerance` in
    /// kcal/mol).
    pub dissociation_scan: Option<crate::workflow::mapper::ScanSettings>,
    /// The upward searches from the transition states of every minimum are added to the tasks
    /// of the method (`upward`, off by default; experimental): `upward_modes` modes, both ways.
    pub upward: Option<crate::workflow::mapper::UpwardSettings>,
    /// The electronic states of the run, PES1, PES2, ... (`states=C,M[,ROOT];...`; empty: one
    /// state, that of the inputs).
    pub states: Vec<crate::workflow::mapper::ElectronicState>,
    /// The crossing points between the states are searched from every minimum (`crossings`,
    /// on by default with two or more states).
    pub crossings: bool,
    /// Which pairs of states (`crossing_pairs`: `all`, the default, or `adjacent`).
    pub crossing_pairs: crate::workflow::mapper::CrossingPairs,
    /// The AFIR paths towards the crossing points of every pair of states (`crossing_afir`:
    /// `gp`, the default with states, `model`, `both`, `off`).
    pub crossing_afir: crate::workflow::mapper::CrossingAfir,
    /// The crossing point is also searched from every minimum itself (`crossing_direct`, off
    /// by default).
    pub crossing_direct: bool,
    /// The states (their indices) on which the method searches transition states
    /// (`ts_states`, PES numbers from 1, or `none`; `None`: all).
    pub ts_states: Option<Vec<usize>>,
    /// The AFIR pushes of the paths towards crossing points, given by their fragments
    /// (`crossing_pushes=GAMMA:ATOMS:ATOMS;...`; `None`: the pairs of atoms of `afir_ts`).
    pub crossing_pushes: Option<Vec<crate::workflow::mapper::FragmentPush>>,
    /// Only the minima of the inputs have crossing tasks (`crossing_from=inputs`; `all` by
    /// default).
    pub crossing_inputs_only: bool,
    /// The model functions of those paths.
    pub crossing_models: super::mapper_settings::CrossingModels,
}

/// The AFIR pushes of `crossing_pushes=GAMMA:ATOMS:ATOMS;...`: the force parameter (kJ/mol,
/// negative for a pull) and the atoms of the two fragments (from 1, `,` between them and `-`
/// for ranges, as `-ma`).
pub fn parse_pushes(text: &str) -> MoptResult<Vec<crate::workflow::mapper::FragmentPush>> {
    let mut out = Vec::new();
    for item in text.split(';').map(str::trim).filter(|t| !t.is_empty()) {
        let bad = || {
            MoptError::parse(format!(
                "crossing_pushes: '{item}' is not GAMMA:ATOMS:ATOMS (e.g. 150:1,4:7-8)"
            ))
        };
        let parts: Vec<&str> = item.split(':').map(str::trim).collect();
        let [gamma, a, b] = parts[..] else {
            return Err(bad());
        };
        let gamma: f64 = gamma
            .parse()
            .ok()
            .filter(|g: &f64| g.is_finite() && *g != 0.0)
            .ok_or_else(bad)?;
        let fragment = |t: &str| -> MoptResult<Vec<usize>> {
            let atoms = crate::core_infra::arg_parser::parse_atom_indices_zero_based(t)?;
            if atoms.is_empty() {
                return Err(bad());
            }
            Ok(atoms)
        };
        out.push(crate::workflow::mapper::FragmentPush {
            gamma,
            a: fragment(a)?,
            b: fragment(b)?,
        });
    }
    if out.is_empty() {
        return Err(MoptError::parse(
            "crossing_pushes lists AFIR pushes as GAMMA:ATOMS:ATOMS, ';' between them",
        ));
    }
    Ok(out)
}

/// The states of `states=C,M[,ROOT];C,M[,ROOT];...`: the charge, the multiplicity, and the root
/// (0, the ground state of that multiplicity, when it is not given) of each, two or more and
/// all different.
pub fn parse_states(text: &str) -> MoptResult<Vec<crate::workflow::mapper::ElectronicState>> {
    let mut out: Vec<crate::workflow::mapper::ElectronicState> = Vec::new();
    for item in text.split(';').map(str::trim).filter(|t| !t.is_empty()) {
        let parts: Vec<&str> = item.split(',').map(str::trim).collect();
        let bad = || {
            MoptError::parse(format!(
                "states: '{item}' is not CHARGE,MULTIPLICITY or CHARGE,MULTIPLICITY,ROOT"
            ))
        };
        if !(2..=3).contains(&parts.len()) {
            return Err(bad());
        }
        let charge: i32 = parts[0].parse().map_err(|_| bad())?;
        let multiplicity: u32 = parts[1]
            .parse()
            .ok()
            .filter(|&m: &u32| m >= 1)
            .ok_or_else(bad)?;
        let excited_state: usize = match parts.get(2) {
            Some(r) => r.parse().map_err(|_| bad())?,
            None => 0,
        };
        let state = crate::workflow::mapper::ElectronicState {
            charge,
            multiplicity,
            excited_state,
        };
        if out.contains(&state) {
            return Err(MoptError::parse(format!("states: '{item}' is given twice")));
        }
        out.push(state);
    }
    if out.len() < 2 {
        return Err(MoptError::parse(
            "states lists two or more states, ';' between them (e.g. states=0,1;0,3 for a \
             singlet and a triplet, states=0,1;0,1,1 for the ground state and the first excited \
             state of the singlet)",
        ));
    }
    Ok(out)
}

/// The temperature of the acceptance filter of the goal-directed search when it is not given:
/// the middle of the 5000 to 10000 K of Nakao, Harabuchi, Maeda, Tsuda, Phys. Chem. Chem. Phys.
/// 2022 (`docs/source/mapper_design.md`).
pub const DEFAULT_ACCEPT_TEMPERATURE: f64 = 7500.0;

/// The priority and its settings: `--use_rcmc` (`use_rcmc`) selects `kinetics` with the score
/// `population` and the solver `rcmc` (unless `kinetics_solver` names another) as MultiOptPy's
/// RCMC queue, and `rcmc_temperature_K`, `rcmc_reaction_time_s`,
/// `rcmc_start_node_id` (`--rcmc_temperature`, `--rcmc_time`, `--rcmc_start_node`) are its
/// `temperature`, `time`, `start`; as in MultiOptPy they are not used by other priorities. The
/// command line goes over `kinetics_*`, which goes over the `rcmc_*` keys of the file.
fn kinetic_priority(
    p: &ParsedArgs,
    v: &Values,
    default: &str,
) -> MoptResult<(String, BTreeMap<String, String>)> {
    let given = v.text("priority")?;
    let use_rcmc = p.get_bool("use_rcmc", false) || v.flag("use_rcmc")?.unwrap_or(false);
    let priority = match (given, use_rcmc) {
        (Some(g), true) if !g.eq_ignore_ascii_case("kinetics") => {
            return Err(MoptError::parse(format!(
                "use_rcmc selects priority=kinetics; it cannot be combined with priority={g}"
            )))
        }
        (Some(g), _) => g,
        (None, true) => "kinetics".to_string(),
        (None, false) => default.to_string(),
    };
    let as_text = |x: &JsonValue| match x {
        JsonValue::String(s) => s.clone(),
        JsonValue::Number(n) if n.fract() == 0.0 && n.abs() < 1e15 => format!("{}", *n as i64),
        JsonValue::Number(n) => format!("{n:e}"),
        other => other.to_string_compact(),
    };
    let mut options: BTreeMap<String, String> = v
        .map
        .iter()
        .filter(|(_, x)| !x.is_null())
        .filter_map(|(k, x)| {
            k.strip_prefix("kinetics_")
                .map(|key| (key.to_string(), as_text(x)))
        })
        .collect();
    // MultiOptPy's RCMC queue: the populations by the published RCMC unless a solver is
    // named.
    if use_rcmc {
        options
            .entry("solver".to_string())
            .or_insert_with(|| "rcmc".to_string());
    }
    if priority.eq_ignore_ascii_case("kinetics") {
        for (cli, key, target) in [
            ("rcmc_temperature", "rcmc_temperature_K", "temperature"),
            ("rcmc_time", "rcmc_reaction_time_s", "time"),
            ("rcmc_start_node", "rcmc_start_node_id", "start"),
        ] {
            if let Some(value) = p.get_opt_str(cli) {
                options.insert(target.to_string(), value.to_string());
            } else if let Some(x) = v.map.get(key).filter(|x| !x.is_null()) {
                options
                    .entry(target.to_string())
                    .or_insert_with(|| as_text(x));
            }
        }
    }
    Ok((priority, options))
}

pub(super) fn object(doc: &JsonValue, key: &str) -> MoptResult<BTreeMap<String, JsonValue>> {
    match doc.get(key) {
        None | Some(JsonValue::Null) => Ok(BTreeMap::new()),
        Some(v) => v
            .as_object()
            .cloned()
            .ok_or_else(|| MoptError::parse(format!("run_mapper: '{key}' must be an object"))),
    }
}

/// `mapper_settings` with the `-map_opt` settings over it.
struct Values {
    map: BTreeMap<String, JsonValue>,
}

impl Values {
    fn number(&self, key: &str) -> MoptResult<Option<f64>> {
        match self.map.get(key) {
            None | Some(JsonValue::Null) => Ok(None),
            Some(JsonValue::Number(x)) => Ok(Some(*x)),
            Some(JsonValue::String(s)) => s.trim().parse().map(Some).map_err(|_| {
                MoptError::parse(format!("mapper_settings.{key} must be a number, not '{s}'"))
            }),
            Some(other) => Err(MoptError::parse(format!(
                "mapper_settings.{key} must be a number, not {}",
                other.to_string_compact()
            ))),
        }
    }

    fn whole(&self, key: &str) -> MoptResult<Option<usize>> {
        match self.number(key)? {
            None => Ok(None),
            Some(x) if x >= 0.0 && x.fract() == 0.0 => Ok(Some(x as usize)),
            Some(x) => Err(MoptError::parse(format!(
                "mapper_settings.{key} must be a whole number of at least 0, not {x}"
            ))),
        }
    }

    /// A limit that may be none: a whole number, or `inf`, `infinity`, `none` (no limit, the
    /// same as leaving the key out).
    fn limit(&self, key: &str) -> MoptResult<Option<usize>> {
        if let Some(JsonValue::String(s)) = self.map.get(key) {
            if ["inf", "infinity", "none", "off"].contains(&s.trim().to_ascii_lowercase().as_str())
            {
                return Ok(None);
            }
        }
        self.whole(key)
    }

    fn flag(&self, key: &str) -> MoptResult<Option<bool>> {
        match self.map.get(key) {
            None | Some(JsonValue::Null) => Ok(None),
            Some(JsonValue::Bool(b)) => Ok(Some(*b)),
            Some(JsonValue::String(s)) => match s.to_ascii_lowercase().as_str() {
                "true" | "on" | "yes" | "1" => Ok(Some(true)),
                "false" | "off" | "no" | "0" | "none" => Ok(Some(false)),
                _ => Err(MoptError::parse(format!(
                    "mapper_settings.{key} is true or false, not '{s}'"
                ))),
            },
            Some(other) => Err(MoptError::parse(format!(
                "mapper_settings.{key} is true or false, not {}",
                other.to_string_compact()
            ))),
        }
    }

    fn text(&self, key: &str) -> MoptResult<Option<String>> {
        match self.map.get(key) {
            None | Some(JsonValue::Null) => Ok(None),
            Some(JsonValue::String(s)) => Ok(Some(s.clone())),
            Some(other) => Err(MoptError::parse(format!(
                "mapper_settings.{key} must be a text, not {}",
                other.to_string_compact()
            ))),
        }
    }

    fn numbers(&self, key: &str) -> MoptResult<Option<Vec<usize>>> {
        match self.map.get(key) {
            None | Some(JsonValue::Null) => Ok(None),
            Some(JsonValue::Array(items)) => items
                .iter()
                .map(|x| {
                    x.as_f64()
                        .filter(|v| *v >= 0.0 && v.fract() == 0.0)
                        .map(|v| v as usize)
                        .ok_or_else(|| {
                            MoptError::parse(format!(
                                "mapper_settings.{key} is a list of whole numbers"
                            ))
                        })
                })
                .collect::<MoptResult<Vec<_>>>()
                .map(Some),
            Some(other) => Err(MoptError::parse(format!(
                "mapper_settings.{key} is a list of whole numbers, not {}",
                other.to_string_compact()
            ))),
        }
    }
}

fn parse_list<T: std::str::FromStr>(tokens: &[String], what: &str) -> MoptResult<Vec<T>> {
    tokens
        .iter()
        .flat_map(|t| t.split(','))
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            t.trim()
                .parse()
                .map_err(|_| MoptError::parse(format!("{what}: '{t}' is not a whole number")))
        })
        .collect()
}

/// Whether a refused part is asked for by `value`.
fn asked(value: Option<String>) -> bool {
    value.is_some_and(|v| {
        !matches!(
            v.to_ascii_lowercase().as_str(),
            "" | "off" | "false" | "no" | "none"
        )
    })
}

/// How the jobs run: `--max_workers` over `workers` (`-map_opt` or `mapper_settings`) over
/// MultiOptPy's `n_parallel` (1: one job at a time, as MultiOptPy); `batch_size` (the number
/// of workers when it is not given); `schedule` (`batch`, the default, or `rolling`); and
/// MultiOptPy's `worker_timeout_s` (seconds, no limit when it is not given).
fn schedule(p: &ParsedArgs, v: &Values) -> MoptResult<Schedule> {
    let at_least_one = |key: &str, n: Option<usize>| -> MoptResult<Option<usize>> {
        match n {
            Some(0) => Err(MoptError::parse(format!(
                "{key} must be a whole number of at least 1 (got 0)"
            ))),
            other => Ok(other),
        }
    };
    let cli = match p.get_opt_str("max_workers") {
        Some(_) => Some(p.get_usize("max_workers", 1)?),
        None => None,
    };
    let workers = at_least_one("--max_workers", cli)?
        .or(at_least_one("workers", v.whole("workers")?)?)
        .or(at_least_one(
            "mapper_settings.n_parallel",
            v.whole("n_parallel")?,
        )?)
        .unwrap_or(1);
    let batch = at_least_one("batch_size", v.whole("batch_size")?)?.unwrap_or(workers);
    let mode = match v.text("schedule")? {
        Some(name) => ScheduleMode::parse(&name)?,
        None => ScheduleMode::Batch,
    };
    let time_limit = match v.number("worker_timeout_s")? {
        None => None,
        Some(s) if s.is_finite() && s > 0.0 => Some(std::time::Duration::from_secs_f64(s)),
        Some(s) => {
            return Err(MoptError::parse(format!(
                "mapper_settings.worker_timeout_s is a time in seconds above 0 (got {s}); leave \
                 it out for no limit"
            )))
        }
    };
    let schedule = Schedule {
        workers,
        batch,
        mode,
        time_limit,
    };
    schedule.check()?;
    Ok(schedule)
}

/// The settings of `mapper_settings` with `-map_opt` and the command line `p` over them.
pub(super) fn mapper_settings(
    p: &ParsedArgs,
    doc: &JsonValue,
) -> MoptResult<(MapperSettings, PathBuf)> {
    let mut map = object(doc, "mapper_settings")?;
    for item in p.get_list("mapper_options") {
        let (k, v) = item
            .split_once('=')
            .ok_or_else(|| MoptError::parse(format!("-map_opt takes KEY=VALUE, not '{item}'")))?;
        let k = k.trim();
        if !MOPT_KEYS.contains(&k) && !k.starts_with("method_") {
            return Err(MoptError::parse(format!(
                "-map_opt {k}: unknown key (keys: {}, method_KEY; MultiOptPy's keys stand in \
                 mapper_settings of the settings file or have their own options)",
                MOPT_KEYS.join(", ")
            )));
        }
        map.insert(k.to_string(), JsonValue::String(v.trim().to_string()));
    }
    let unknown: Vec<&String> = map
        .keys()
        .filter(|k| {
            !MULTIOPTPY_KEYS.contains(&k.as_str())
                && !MOPT_KEYS.contains(&k.as_str())
                && !k.starts_with("method_")
        })
        .collect();
    if !unknown.is_empty() {
        return Err(MoptError::parse(format!(
            "unknown key(s) in mapper_settings: {} (known: {}, {}, method_KEY)",
            unknown
                .iter()
                .map(|k| k.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            MULTIOPTPY_KEYS.join(", "),
            MOPT_KEYS.join(", ")
        )));
    }
    let v = Values { map };
    // The goal of a goal-directed search: the priority rrt unless another is chosen (the
    // method connect chooses its jobs by itself, and any priority does for it).
    let goal = v.text("goal")?.map(PathBuf::from);
    let connect = v
        .text("method")?
        .is_some_and(|m| m.eq_ignore_ascii_case(crate::workflow::mapper::Connect::NAME));
    let default_priority = if goal.is_some() && !connect {
        "rrt"
    } else {
        "boltzmann"
    };
    // MultiOptPy's RCMC priority is the kinetic priority.
    let (priority, priority_options) = kinetic_priority(p, &v, default_priority)?;
    let rrt = priority.eq_ignore_ascii_case("rrt");
    if rrt && goal.is_none() {
        return Err(MoptError::parse(
            "priority=rrt grows the network towards a goal: give the goal structure with \
             goal=FILE",
        ));
    }
    if connect && goal.is_none() {
        return Err(MoptError::parse(
            "method=connect joins a start to a goal: give the goal structure with goal=FILE",
        ));
    }
    let rrt_options: BTreeMap<String, String> = v
        .map
        .iter()
        .filter(|(_, x)| !x.is_null())
        .filter_map(|(k, x)| {
            k.strip_prefix("rrt_").map(|key| {
                (
                    key.to_string(),
                    x.as_str()
                        .map_or_else(|| x.to_string_compact(), str::to_string),
                )
            })
        })
        .collect();
    if !rrt && !rrt_options.is_empty() {
        return Err(MoptError::parse(format!(
            "rrt_{} is a setting of priority=rrt, which is not chosen",
            rrt_options.keys().next().map_or("", String::as_str)
        )));
    }
    let accept_temperature = match v.number("accept_temperature")? {
        Some(t) if !(t.is_finite() && t > 0.0) => {
            return Err(MoptError::parse(format!(
                "accept_temperature is a temperature above 0 K (got {t})"
            )))
        }
        Some(t) => Some(t),
        None if rrt => Some(DEFAULT_ACCEPT_TEMPERATURE),
        None => None,
    };
    // What is not implemented yet, or only fits MultiOptPy's design, is refused by name.
    let schedule = schedule(p, &v)?;
    if asked(v.text("refine")?) {
        // MultiOptPy has no such setting either; the name was kept for the refinement of a
        // network on another surface, which is a run of its own.
        return Err(MoptError::unsupported(
            "run_mapper: refine is no setting of a run: the network of a run is drawn again on \
             another surface by a run of its own, run_mapper -remap RUN_DIR",
        ));
    }
    let states = match v.text("states")? {
        Some(t) if asked(Some(t.clone())) => parse_states(&t)?,
        _ => Vec::new(),
    };
    let crossings = match v.flag("crossings")? {
        Some(true) if states.is_empty() => {
            return Err(MoptError::parse(
                "crossings needs the states of the run (states=C,M;C,M)",
            ))
        }
        Some(on) => on,
        None => !states.is_empty(),
    };
    let crossing_pairs = match v.text("crossing_pairs")? {
        Some(_) if states.is_empty() => {
            return Err(MoptError::parse(
                "crossing_pairs needs the states of the run (states=C,M;C,M;C,M)",
            ))
        }
        Some(t) => crate::workflow::mapper::CrossingPairs::parse(&t)?,
        None => crate::workflow::mapper::CrossingPairs::All,
    };
    let crossing_afir = match v.text("crossing_afir")? {
        Some(t) => crate::workflow::mapper::CrossingAfir::parse(&t)?,
        None => crate::workflow::mapper::CrossingAfir::Gp,
    };
    let crossing_direct = v.flag("crossing_direct")?.unwrap_or(false);
    let crossing_pushes = match v.text("crossing_pushes")? {
        Some(t) => Some(parse_pushes(&t)?),
        None => None,
    };
    let crossing_inputs_only = match v
        .text("crossing_from")?
        .as_deref()
        .map(str::to_ascii_lowercase)
    {
        None => false,
        Some(f) if f == "all" => false,
        Some(f) if f == "inputs" => true,
        Some(f) => {
            return Err(MoptError::parse(format!(
                "crossing_from is all (every minimum, the default) or inputs, not '{f}'"
            )))
        }
    };
    let ts_states = match v.text("ts_states")? {
        Some(t) if t.trim().eq_ignore_ascii_case("none") => Some(Vec::new()),
        Some(t) => {
            let mut list = Vec::new();
            for item in t.split(',').map(str::trim).filter(|x| !x.is_empty()) {
                let k: usize = item
                    .parse()
                    .ok()
                    .filter(|&k: &usize| k >= 1 && k <= states.len())
                    .ok_or_else(|| {
                        MoptError::parse(format!(
                            "ts_states lists numbers of the states of the run (1 to {}), ',' \
                             between them, not '{item}'",
                            states.len()
                        ))
                    })?;
                if !list.contains(&(k - 1)) {
                    list.push(k - 1);
                }
            }
            Some(list)
        }
        None => None,
    };
    if (crossing_pushes.is_some() || crossing_inputs_only) && (states.is_empty() || !crossings) {
        return Err(MoptError::parse(
            "crossing_pushes and crossing_from belong to the search for crossing points: they \
             need states=... (and not crossings=off)",
        ));
    }
    if crossings
        && !states.is_empty()
        && crossing_afir == crate::workflow::mapper::CrossingAfir::Off
        && !crossing_direct
    {
        return Err(MoptError::parse(
            "crossings without a search: crossing_afir=off needs crossing_direct=on (or \
             crossings=off)",
        ));
    }
    let mut crossing_models = super::mapper_settings::CrossingModels::default();
    for (key, slot) in [
        ("crossing_smf_alpha", &mut crossing_models.smf_alpha),
        ("crossing_amf_alpha", &mut crossing_models.amf_alpha),
    ] {
        if let Some(x) = v.number(key)? {
            if !(x.is_finite() && x > 0.0) {
                return Err(MoptError::parse(format!(
                    "{key} is an energy above 0 (Hartree), not {x}"
                )));
            }
            *slot = x;
        }
    }
    if let Some(x) = v.number("crossing_step")? {
        if !(x.is_finite() && x > 0.0) {
            return Err(MoptError::parse(format!(
                "crossing_step is a length above 0 (Angstrom), not {x}"
            )));
        }
        crossing_models.max_step = x * crate::domain::Units::ANGSTROM_TO_BOHR;
    }
    if let Some(x) = v.number("crossing_gap")? {
        if !(x.is_finite() && x > 0.0) {
            return Err(MoptError::parse(format!(
                "crossing_gap is an energy above 0 (Hartree), not {x}"
            )));
        }
        crossing_models.gap_limit = x;
    }
    if let Some(x) = v.number("crossing_candidate_prominence")? {
        if !(x.is_finite() && x >= 0.0) {
            return Err(MoptError::parse(format!(
                "crossing_candidate_prominence is an energy of 0 or more (Hartree), not {x}"
            )));
        }
        crossing_models.candidates.prominence = x;
    }
    if let Some(x) = v.number("crossing_candidate_distance")? {
        if !(x.is_finite() && x >= 0.0) {
            return Err(MoptError::parse(format!(
                "crossing_candidate_distance is a length of 0 or more (Angstrom), not {x}"
            )));
        }
        crossing_models.candidates.distance = x * crate::domain::Units::ANGSTROM_TO_BOHR;
    }
    // The end of a search where the two states run parallel: off unless asked for.
    let mut parallel = super::mapper_settings::ParallelStop {
        gradient: super::mapper_settings::DEFAULT_PARALLEL_GRADIENT,
        steps: super::mapper_settings::DEFAULT_PARALLEL_STEPS,
    };
    if let Some(x) = v.number("crossing_parallel_gradient")? {
        if !(x.is_finite() && x > 0.0) {
            return Err(MoptError::parse(format!(
                "crossing_parallel_gradient is above 0 (Hartree/Bohr), not {x}"
            )));
        }
        parallel.gradient = x;
    }
    if let Some(n) = v.whole("crossing_parallel_steps")? {
        if n == 0 {
            return Err(MoptError::parse(
                "crossing_parallel_steps is a number of iterations of 1 or more",
            ));
        }
        parallel.steps = n;
    }
    if v.flag("crossing_parallel_stop")?.unwrap_or(false) {
        crossing_models.parallel = Some(parallel);
    }
    // The check within the seam of every accepted crossing point: on unless switched off.
    if let Some(on) = v.flag("crossing_seam_check")? {
        crossing_models.seam_check = on;
    }
    match v
        .text("crossing_amf_branch")?
        .as_deref()
        .map(str::to_ascii_lowercase)
    {
        None => {}
        Some(b) if b == "upper" => crossing_models.amf_upper = true,
        Some(b) if b == "lower" => crossing_models.amf_upper = false,
        Some(b) => {
            return Err(MoptError::parse(format!(
                "crossing_amf_branch is upper (the default) or lower, not '{b}'"
            )))
        }
    }
    let model_keys = [
        "crossing_afir",
        "crossing_smf_alpha",
        "crossing_amf_alpha",
        "crossing_amf_branch",
        "crossing_step",
        "crossing_gap",
        "crossing_candidate_prominence",
        "crossing_candidate_distance",
        "crossing_parallel_stop",
        "crossing_parallel_gradient",
        "crossing_parallel_steps",
        "crossing_seam_check",
        "crossing_direct",
        "crossing_pushes",
        "crossing_from",
        "ts_states",
    ];
    if states.is_empty() {
        for key in model_keys {
            if v.text(key)?.is_some() {
                return Err(MoptError::parse(format!(
                    "{key} needs the states of the run (states=C,M;C,M)"
                )));
            }
        }
    }
    let cli_f64 = |key: &str| p.get_opt_f64(key);
    let cli_usize = |key: &str| -> MoptResult<Option<usize>> {
        match p.get_opt_str(key) {
            Some(_) => p.get_usize(key, 0).map(Some),
            None => Ok(None),
        }
    };
    let defaults = AfirSettings::default();
    let seed = cli_usize("rng_seed")?
        .or(v.whole("rng_seed")?)
        .unwrap_or(42) as u64;
    let active = match p.get_list("active_atoms") {
        [] => v.numbers("active_atoms")?,
        given => Some(parse_list::<usize>(given, "--active_atoms")?),
    };
    let active = match active {
        Some(list) => Some(
            list.iter()
                .map(|&a| {
                    a.checked_sub(1)
                        .ok_or_else(|| MoptError::parse("active atoms are numbered from 1"))
                })
                .collect::<MoptResult<Vec<_>>>()?,
        ),
        None => None,
    };
    let afir = AfirSettings {
        gamma_kj_mol: cli_f64("afir_gamma")?
            .or(v.number("afir_gamma_kJmol")?)
            .unwrap_or(defaults.gamma_kj_mol),
        max_pairs: cli_usize("max_pairs")?
            .or(v.whole("max_pairs")?)
            .unwrap_or(defaults.max_pairs),
        dist_lower: cli_f64("dist_lower")?
            .or(v.number("dist_lower_ang")?)
            .unwrap_or(defaults.dist_lower),
        dist_upper: cli_f64("dist_upper")?
            .or(v.number("dist_upper_ang")?)
            .unwrap_or(defaults.dist_upper),
        active_atoms: active,
        negative_gamma: p.get_bool("negative_gamma", false)
            || v.flag("include_negative_gamma")?.unwrap_or(false),
        cell_pushes: v.flag("cell_pushes")?.unwrap_or(false),
        covalent_margin: v
            .number("pair_bond_scale")?
            .unwrap_or(defaults.covalent_margin),
        fragments: match v.text("afir_fragments")? {
            Some(t) => crate::workflow::mapper::afir::AfirFragments::parse(&t)?,
            None => defaults.fragments,
        },
        seed,
    };
    afir.check()?;
    let excluded = match p.get_list("exclude_nodes") {
        [] => v.numbers("excluded_node_ids")?.unwrap_or_default(),
        given => parse_list::<usize>(given, "--exclude_nodes")?,
    };
    let top_n = match cli_usize("top_n")? {
        Some(n) => n,
        None => doc
            .get("top_n_candidates")
            .and_then(JsonValue::as_f64)
            .map_or(3, |x| x.max(1.0) as usize),
    };
    let list = |key: &str, sep: char| -> MoptResult<Vec<String>> {
        Ok(v.text(key)?
            .map(|t| {
                t.split(sep)
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default())
    };
    let settings = MapperSettings {
        temperature: cli_f64("temperature")?
            .or(v.number("temperature_K")?)
            .unwrap_or(300.0),
        rmsd_threshold: cli_f64("rmsd_threshold")?
            .or(v.number("rmsd_threshold")?)
            .unwrap_or(DEFAULT_RMSD_THRESHOLD),
        energy_tolerance: v
            .number("energy_tolerance_kcalmol")?
            .unwrap_or(DEFAULT_ENERGY_TOLERANCE),
        variable_tolerance: match v.number("variable_tolerance")? {
            Some(t) if !(t.is_finite() && t > 0.0) => {
                return Err(MoptError::parse(format!(
                    "variable_tolerance is a distance above 0 (got {t})"
                )))
            }
            Some(t) => t,
            None => DEFAULT_VARIABLE_TOLERANCE,
        },
        max_jobs: cli_usize("max_iter")?
            .or(v.whole("max_iterations")?)
            .unwrap_or(50),
        max_jobs_given: cli_usize("max_iter")?
            .or(v.whole("max_iterations")?)
            .is_some(),
        afir,
        top_n,
        seed,
        excluded,
        exclude_bond_rearrangement: p.get_bool("exclude_bond_rearrangement", false)
            || v.flag("exclude_bond_rearrangement")?.unwrap_or(false),
        method: v.text("method")?.unwrap_or_default(),
        priority,
        priority_options,
        stop: list("stop", ';')?,
        stores: list("store", ',')?,
        html_report: v.flag("html_report")?.unwrap_or(true),
        save_job_steps: v.flag("save_job_steps")?.unwrap_or(true),
        mirror: v.flag("mirror")?.unwrap_or(false),
        method_extra: v
            .map
            .iter()
            .filter_map(|(k, x)| {
                k.strip_prefix("method_").map(|key| {
                    (
                        key.to_string(),
                        x.as_str()
                            .map_or_else(|| x.to_string_compact(), str::to_string),
                    )
                })
            })
            .collect(),
        schedule,
        kinetics: v.flag("kinetics")?.unwrap_or(false),
        // The upward search finds saddle points of order 2 and descends from them: it needs
        // them followed.
        higher_saddles: match (
            v.flag("higher_saddles")?.or_else(|| {
                let upward_method =
                    v.text("method").ok().flatten().is_some_and(|m| {
                        m.eq_ignore_ascii_case(crate::workflow::mapper::Upward::NAME)
                    });
                let upward_on = v.flag("upward").ok().flatten() == Some(true);
                (upward_method || upward_on).then_some(true)
            }),
            v.whole("max_saddle_order")?,
        ) {
            (Some(true), order) => {
                let order =
                    order.unwrap_or(crate::workflow::mapper::method::DEFAULT_MAX_SADDLE_ORDER);
                if !(2..=usize::from(crate::workflow::mapper::network::MAX_TAGGED_ORDER))
                    .contains(&order)
                {
                    return Err(MoptError::parse(format!(
                        "max_saddle_order is from 2 to {} (got {order})",
                        crate::workflow::mapper::network::MAX_TAGGED_ORDER
                    )));
                }
                Some(order)
            }
            (_, Some(_)) => {
                return Err(MoptError::parse(
                    "max_saddle_order is a setting of higher_saddles=on",
                ))
            }
            _ => None,
        },
        confirm_bifurcations: v.flag("confirm_bifurcations")?.unwrap_or(false),
        ts_conformers: v.flag("ts_conformers")?.unwrap_or(false),
        path_peaks: match v.map.get("path_peaks") {
            None | Some(JsonValue::Null) => crate::workflow::mapper::PeakMode::default(),
            Some(JsonValue::Bool(true)) => crate::workflow::mapper::PeakMode::Failed,
            Some(JsonValue::Bool(false)) => crate::workflow::mapper::PeakMode::Off,
            Some(_) => crate::workflow::mapper::PeakMode::parse(
                &v.text("path_peaks")?.unwrap_or_default(),
            )?,
        },
        entrance_structures: v.whole("entrance_structures")?.unwrap_or(0),
        entrance_contact: v
            .number("entrance_contact")?
            .unwrap_or(crate::workflow::mapper::entrance::DEFAULT_ENTRANCE_CONTACT),
        max_bond_changes: v.limit("max_bond_changes")?,
        max_species_changes: v.limit("max_species_changes")?,
        irc_hessian_interval: match v.whole("irc_hessian_interval")? {
            None => super::mapper_settings::IRC_HESSIAN_INTERVAL,
            Some(0) => {
                return Err(MoptError::parse(
                    "irc_hessian_interval is at least 1 (irc_hessians=off turns the exact \
                     Hessians of the IRC off)",
                ))
            }
            Some(n) => n,
        },
        dissociation_scan: {
            use crate::workflow::mapper::scan::{DEFAULT_SCAN_STEP, DEFAULT_SCAN_TOLERANCE};
            let step = v.number("dissociation_scan_step")?;
            let tolerance = v.number("dissociation_scan_tolerance")?;
            let on = v.flag("confirm_dissociations")?.unwrap_or(false);
            if !on && (step.is_some() || tolerance.is_some()) {
                return Err(MoptError::parse(
                    "dissociation_scan_step and dissociation_scan_tolerance are settings of \
                     confirm_dissociations=on",
                ));
            }
            let (step, tolerance) = (
                step.unwrap_or(DEFAULT_SCAN_STEP),
                tolerance.unwrap_or(DEFAULT_SCAN_TOLERANCE),
            );
            if !(step.is_finite() && step > 0.0 && tolerance.is_finite() && tolerance >= 0.0) {
                return Err(MoptError::parse(format!(
                    "dissociation_scan_step must be positive and dissociation_scan_tolerance \
                     at least 0 (got {step} and {tolerance})"
                )));
            }
            on.then(|| crate::workflow::mapper::ScanSettings {
                step: step * crate::domain::Units::ANGSTROM_TO_BOHR,
                tolerance: tolerance * crate::domain::Units::KCAL_MOL_TO_HARTREE,
                ..crate::workflow::mapper::ScanSettings::default()
            })
        },
        states,
        crossings,
        crossing_pairs,
        crossing_afir,
        crossing_direct,
        crossing_pushes,
        crossing_inputs_only,
        ts_states,
        crossing_models,
        upward: if v.flag("upward")?.unwrap_or(false) {
            Some(crate::workflow::mapper::UpwardSettings {
                modes: v
                    .whole("upward_modes")?
                    .filter(|&n| n >= 1)
                    .unwrap_or(crate::workflow::mapper::upward::DEFAULT_MODES),
                displacement: crate::workflow::mapper::upward::DEFAULT_DISPLACEMENT,
            })
        } else {
            None
        },
        goal_stop: v.flag("goal_stop")?.unwrap_or(true),
        goal_bonds: match v
            .text("goal_match")?
            .as_deref()
            .map(str::to_ascii_lowercase)
        {
            None => false,
            Some(m) if m == "structure" => false,
            Some(m) if m == "bonds" => true,
            Some(m) => {
                return Err(MoptError::parse(format!(
                    "goal_match is structure (the goal structure itself, the default) or bonds \
                     (any minimum with its bond graph), not '{m}'"
                )))
            }
        },
        goal,
        accept_temperature,
        rrt_options,
        verify_minima: v.flag("verify_minima")?.unwrap_or(true),
        irc_hessians: v.flag("irc_hessians")?.unwrap_or(true),
        irc_end_threshold_cm1: v
            .number("irc_end_threshold_cm1")?
            .unwrap_or(crate::workflow::mapper::method::DEFAULT_IRC_END_THRESHOLD_CM1),
        bifurcation_threshold_cm1: v
            .number("bifurcation_threshold_cm1")?
            .unwrap_or(crate::workflow::mapper::DEFAULT_BIFURCATION_THRESHOLD_CM1),
        channels: v.flag("channels")?.unwrap_or(true),
        channels_given: v.flag("channels")?.is_some(),
        channel_margin: v
            .number("channel_margin")?
            .unwrap_or(crate::workflow::mapper::channels::DEFAULT_CHANNEL_MARGIN_ANGSTROM),
        identity_scope: v
            .text("identity_scope")?
            .map(|t| super::mapper_oniom::IdentityScope::parse(&t))
            .transpose()?,
    };
    if !settings.irc_end_threshold_cm1.is_finite() || settings.irc_end_threshold_cm1 < 0.0 {
        return Err(MoptError::parse(
            "run_mapper: irc_end_threshold_cm1 must be finite and nonnegative (cm^-1)",
        ));
    }
    if !(settings.bifurcation_threshold_cm1.is_finite() && settings.bifurcation_threshold_cm1 > 0.0)
    {
        return Err(MoptError::parse(format!(
            "run_mapper: bifurcation_threshold_cm1 must be above zero (cm^-1; got {})",
            settings.bifurcation_threshold_cm1
        )));
    }
    if !(settings.channel_margin.is_finite() && settings.channel_margin >= 0.0) {
        return Err(MoptError::parse(format!(
            "run_mapper: channel_margin must be zero or more (Angstrom; got {})",
            settings.channel_margin
        )));
    }
    for (what, x) in [
        ("the temperature (temperature_K)", settings.temperature),
        ("rmsd_threshold", settings.rmsd_threshold),
        ("energy_tolerance_kcalmol", settings.energy_tolerance),
    ] {
        if !(x.is_finite() && x > 0.0) {
            return Err(MoptError::parse(format!(
                "run_mapper: {what} must be positive (got {x})"
            )));
        }
    }
    let output = p
        .get_opt_str("output_dir")
        .map(str::to_string)
        .or(v.text("output_dir")?)
        .unwrap_or_else(|| "mapper_output".to_string());
    Ok((settings, PathBuf::from(output)))
}
