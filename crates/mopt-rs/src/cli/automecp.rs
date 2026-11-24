//! `run_automecp`: the automated search for minimum-energy crossing points between two
//! electronic states from one structure, the counterpart of `run_autots` for crossing points
//! (MECP between states of different spin or charge, MECI between two roots of one spin).
//!
//! The settings file (`-cfg config.json`) has one block of `optmain` settings per step, as
//! that of `run_autots`, and the command line (`-ma`, `-state2`, `-afir`) goes over it:
//!
//! 1. The input is minimized on its own state (`step3_settings`), and from that minimum every
//!    AFIR push of `-ma GAMMA FRAGMENT1 FRAGMENT2 ...` (`manual_AFIR` in the file) gives a path
//!    towards the crossing points of the two states (`step1_settings`): with the force on the
//!    gradient projection of the two states (`-afir gp`, the default) or on a model function of
//!    them (`-afir model`). The candidates are the end of the path and the local minima of the
//!    model function along it.
//! 2. Every candidate is optimized to the crossing point by the gradient projection without
//!    the force (`step2_settings`).
//! 3. From every crossing point a meta-IRC runs down on each of the two states, and its end is
//!    minimized (`step3_settings`, with `intrinsic_reaction_coordinates` for the IRC).
//!
//! The second state is `-state2 CHARGE MULTIPLICITY [ROOT]` (`state2` in the file); the first
//! is that of the input (`electronic_charge`, `spin_multiplicity` or `unpaired_electrons` of
//! `step1_settings`) with the root `-root1` (or `excited_state` of `step1_settings`; 0 by
//! default). The settings of the mapper run that the command makes (`states`,
//! `crossing_pushes`, `crossing_from`, `ts_states`, `crossing_afir`) are refused in `-map_opt`
//! and, unless they say the same, in `mapper_settings`. Without `opt_method` and
//! `calc_exact_hess` in `step2_settings` the crossing points are optimized with
//! [`super::mapper_settings::CROSSING_METHOD`] and no exact Hessians, the defaults of every
//! crossing search of the mapper; every crossing point that is found is checked within the seam
//! (`crossing_seam_check`, on by default). A restart goes on with the recorded settings and
//! warns of every option given that differs from them ([`restart_notes`]); a run that ends
//! without a crossing point is an error after its files are written.
//!
//! Nothing is computed here: the run is a run of the reaction-network mapper
//! ([`super::mapper`]) on the two states, with the crossing tasks of the given pushes from the
//! input alone and no search for transition states (`states`, `crossing_pushes`,
//! `crossing_from=inputs`, `ts_states=none`). It has the mapper's files, report, interruption
//! and restart in `RUN_DIR/mapper`, and beside them [`SUMMARY_FILE`] with one line per crossing
//! point and the page of the search ([`super::automecp_report`], made like that of
//! `run_autots`: the paths of the pushes, the candidates, the crossing points, and the way
//! between the two minima of each).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::mapper::{self, MapperConfig, MapperRun};
use super::option::parser;
use super::settings::{read_settings, settings_json, RUN_SETTINGS_FILE};
use super::spec::{RUN_AUTOMECP, RUN_MAPPER};
use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::pes::registry::CalculatorRegistry;
use crate::workflow::mapper::{surface_name, Network, NodeKind, SeamCheck};

/// The run directory when `--output_dir` is not given.
pub const DEFAULT_OUTPUT_DIR: &str = "automecp_output";
/// The directory of the mapper run inside the run directory.
pub const MAPPER_DIR: &str = "mapper";
/// The settings of the mapper run, written to the run directory.
pub const MAPPER_CONFIG_FILE: &str = "mapper_config.json";
/// The crossing points found, one line each.
pub const SUMMARY_FILE: &str = "automecp_summary.csv";

/// The settings of a run.
#[derive(Debug, Clone)]
pub struct AutomecpConfig {
    /// The input structure file.
    pub input: PathBuf,
    /// The settings file (`-cfg`).
    pub config_file: PathBuf,
    /// The AFIR pushes of the command line (`-ma`: triples of the force parameter and two
    /// fragments); empty: those of the file.
    pub pushes: Vec<String>,
    /// The second state of the command line (`-state2`: charge, multiplicity, and root);
    /// empty: that of the file.
    pub state2: Vec<String>,
    /// The root of the first state (`-root1`); `None`: `excited_state` of `step1_settings`,
    /// else 0.
    pub first_root: Option<usize>,
    /// The kind of AFIR path (`-afir`); `None`: that of the file, or the mapper's default.
    pub afir: Option<String>,
    /// Further settings of the mapper (`-map_opt KEY=VALUE`).
    pub mapper_options: Vec<String>,
    /// The run directory (`--output_dir`, relative to the base directory).
    pub output_dir: PathBuf,
    /// The run directory to continue (`--restart`).
    pub restart_dir: Option<PathBuf>,
    /// The settings of the command as they are used (written to the run directory).
    pub settings_document: String,
}

impl AutomecpConfig {
    /// From the command line.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        let input = p
            .positionals
            .first()
            .map(PathBuf::from)
            .ok_or_else(|| MoptError::parse("run_automecp needs an input structure (xyz)"))?;
        Ok(Self {
            input,
            config_file: PathBuf::from(p.get_opt_str("config_file").unwrap_or("./config.json")),
            pushes: p.get_list("manual_AFIR").to_vec(),
            state2: p.get_list("state2").to_vec(),
            first_root: match p.get_opt_str("state1_root") {
                None => None,
                Some(t) => Some(t.trim().parse().map_err(|_| {
                    MoptError::parse(format!(
                        "run_automecp: -root1 is the number of a root (0: the lowest state of \
                         the charge and multiplicity), not '{t}'"
                    ))
                })?),
            },
            afir: p.get_opt_str("crossing_afir").map(str::to_string),
            mapper_options: p.get_list("mapper_options").to_vec(),
            output_dir: PathBuf::from(p.get_opt_str("output_dir").unwrap_or(DEFAULT_OUTPUT_DIR)),
            restart_dir: None,
            settings_document: settings_json(p, RUN_AUTOMECP).to_string_pretty(),
        })
    }
}

/// What a run gave.
#[derive(Debug)]
pub struct AutomecpRun {
    /// The run directory.
    pub directory: PathBuf,
    /// The run of the mapper (its directory, the network).
    pub mapper: MapperRun,
    /// The file of the crossing points.
    pub summary: PathBuf,
    /// The page of the run ([`super::automecp_report`]).
    pub report: PathBuf,
    /// On a restart: the options given that the run does not take ([`restart_notes`]).
    pub restart_notes: Vec<String>,
}

/// The settings of the mapper run that the command makes from its own options, with the
/// options they come from.
const OWN_SETTINGS: [(&str, &str); 5] = [
    ("states", "the state of step1_settings (-root1) and -state2"),
    ("crossing_pushes", "-ma (manual_AFIR)"),
    ("crossing_from", "the command itself (inputs)"),
    ("ts_states", "the command itself (none)"),
    ("crossing_afir", "-afir (crossing_afir)"),
];

/// A value of the settings file as text (a text as it is).
fn setting_text(v: &JsonValue) -> String {
    match v {
        JsonValue::String(s) => s.clone(),
        JsonValue::Number(n) if n.fract() == 0.0 => format!("{n:.0}"),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::Bool(b) => b.to_string(),
        other => other.to_string_pretty(),
    }
}

/// Whether the setting `given` says `value` (without spaces, in any case).
fn same_setting(given: &JsonValue, value: &str) -> bool {
    let plain = |s: &str| -> String {
        s.chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .to_ascii_lowercase()
    };
    plain(&setting_text(given)) == plain(value)
}

/// A list of numbers or texts of the settings file as texts.
fn texts(v: &JsonValue, what: &str) -> MoptResult<Vec<String>> {
    let bad = || {
        MoptError::parse(format!(
            "run_automecp: '{what}' is a list of numbers or texts"
        ))
    };
    v.as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|x| match x {
            JsonValue::String(s) => Ok(s.clone()),
            JsonValue::Number(n) if n.fract() == 0.0 => Ok(format!("{n:.0}")),
            JsonValue::Number(n) => Ok(n.to_string()),
            _ => Err(bad()),
        })
        .collect()
}

/// The settings file of the mapper run that `config` asks for, from its settings file
/// `document`: the two states, the pushes, and the steps (`step1_settings` as it is,
/// `step2_settings` as the mapper's `step3_settings`, `step3_settings` as its
/// `step4_settings`).
pub fn mapper_document(config: &AutomecpConfig, document: &JsonValue) -> MoptResult<JsonValue> {
    let top = document.as_object().ok_or_else(|| {
        MoptError::parse("run_automecp: the settings file is an object of settings")
    })?;
    for key in top.keys() {
        if ![
            "step1_settings",
            "step2_settings",
            "step3_settings",
            "manual_AFIR",
            "state2",
            "crossing_afir",
            "mapper_settings",
        ]
        .contains(&key.as_str())
        {
            return Err(MoptError::parse(format!(
                "run_automecp: unknown key '{key}' of the settings file (keys: step1_settings, \
                 step2_settings, step3_settings, manual_AFIR, state2, crossing_afir, \
                 mapper_settings)"
            )));
        }
    }
    let step = |key: &str| -> MoptResult<BTreeMap<String, JsonValue>> {
        match top.get(key) {
            None => Ok(BTreeMap::new()),
            Some(JsonValue::Object(m)) => Ok(m.clone()),
            Some(_) => Err(MoptError::parse(format!(
                "run_automecp: '{key}' is an object of optmain settings"
            ))),
        }
    };
    let mut step1 = step("step1_settings")?;
    // The first state: that of the input.
    let whole = |key: &str| -> MoptResult<Option<i64>> {
        match step1.get(key) {
            None => Ok(None),
            Some(v) => match v {
                JsonValue::Number(n) if n.fract() == 0.0 => Ok(Some(*n as i64)),
                JsonValue::String(s) => s.trim().parse().map(Some).map_err(|_| {
                    MoptError::parse(format!(
                        "run_automecp: step1_settings.{key} is a whole number"
                    ))
                }),
                _ => Err(MoptError::parse(format!(
                    "run_automecp: step1_settings.{key} is a whole number"
                ))),
            },
        }
    };
    let charge = whole("electronic_charge")?.unwrap_or(0);
    // The multiplicity, or the unpaired electrons that give it (as optmain takes them).
    let multiplicity = match (whole("spin_multiplicity")?, whole("unpaired_electrons")?) {
        (Some(m), Some(u)) if m != u + 1 => {
            return Err(MoptError::parse(format!(
                "run_automecp: step1_settings gives the multiplicity {m} and {u} unpaired \
                 electrons, which contradict each other"
            )))
        }
        (Some(m), _) => m,
        (None, Some(u)) => u + 1,
        (None, None) => 1,
    };
    // The root of the first state: -root1, or excited_state of step1_settings, which leaves
    // the block (the states of the mapper run say which surface every step is on).
    let root = match (config.first_root, whole("excited_state")?) {
        (Some(a), Some(b)) if i64::try_from(a).ok() != Some(b) => {
            return Err(MoptError::parse(format!(
                "run_automecp: -root1 {a} and excited_state {b} of step1_settings name two \
                 roots for the first state; give one of them"
            )))
        }
        (Some(a), _) => i64::try_from(a).map_err(|_| MoptError::parse("-root1 is too large"))?,
        (None, Some(b)) => b,
        (None, None) => 0,
    };
    if root < 0 {
        return Err(MoptError::parse(format!(
            "run_automecp: the root of the first state is 0 or more, not {root}"
        )));
    }
    step1.remove("excited_state");
    let first = if root > 0 {
        format!("{charge},{multiplicity},{root}")
    } else {
        format!("{charge},{multiplicity}")
    };
    // The second state.
    let state2 = if config.state2.is_empty() {
        match top.get("state2") {
            Some(v) => texts(v, "state2")?,
            None => Vec::new(),
        }
    } else {
        config.state2.clone()
    };
    if !(2..=3).contains(&state2.len()) {
        return Err(MoptError::parse(
            "run_automecp needs the second state: -state2 CHARGE MULTIPLICITY [ROOT] (or state2 \
             in the settings file), e.g. -state2 0 3 for a triplet",
        ));
    }
    // The pushes: triples of the force parameter and the two fragments.
    let pushes = if config.pushes.is_empty() {
        match top.get("manual_AFIR") {
            Some(v) => texts(v, "manual_AFIR")?,
            None => Vec::new(),
        }
    } else {
        config.pushes.clone()
    };
    if pushes.is_empty() || pushes.len() % 3 != 0 {
        return Err(MoptError::parse(
            "run_automecp needs the AFIR pushes: -ma GAMMA FRAGMENT1 FRAGMENT2 (triples; or \
             manual_AFIR in the settings file), e.g. -ma 150 1,4 7,8",
        ));
    }
    let pushes: Vec<String> = pushes
        .chunks(3)
        .map(|c| format!("{}:{}:{}", c[0], c[1], c[2]))
        .collect();
    let mut settings = match top.get("mapper_settings") {
        None => BTreeMap::new(),
        Some(JsonValue::Object(m)) => m.clone(),
        Some(_) => {
            return Err(MoptError::parse(
                "run_automecp: 'mapper_settings' is an object of mapper settings",
            ))
        }
    };
    // What the command sets itself is not given in -map_opt.
    for item in &config.mapper_options {
        for pair in item.split_whitespace() {
            let key = pair.split_once('=').map_or(pair, |(k, _)| k).trim();
            if let Some((_, by)) = OWN_SETTINGS.iter().find(|(own, _)| *own == key) {
                return Err(MoptError::parse(format!(
                    "run_automecp: -map_opt {key}: run_automecp makes this setting of the \
                     mapper run from {by}; give it there"
                )));
            }
        }
    }
    let afir = match (&config.afir, top.get("crossing_afir")) {
        (Some(a), _) => Some(a.clone()),
        (None, None) => None,
        (None, Some(JsonValue::String(a))) => Some(a.clone()),
        (None, Some(_)) => {
            return Err(MoptError::parse(
                "run_automecp: crossing_afir of the settings file is a text (gp, model, both)",
            ))
        }
    };
    let mut values = vec![
        ("states", format!("{first};{}", state2.join(","))),
        ("crossing_pushes", pushes.join(";")),
        ("crossing_from", "inputs".to_string()),
        ("ts_states", "none".to_string()),
    ];
    values.extend(afir.map(|a| ("crossing_afir", a)));
    for (key, value) in values {
        // A setting of mapper_settings that the command makes as well: the same value, or a
        // contradiction that is refused (not written over).
        if let Some(given) = settings.get(key) {
            if !same_setting(given, &value) {
                let by = OWN_SETTINGS
                    .iter()
                    .find(|(own, _)| *own == key)
                    .map_or("its options", |(_, by)| by);
                return Err(MoptError::parse(format!(
                    "run_automecp: mapper_settings.{key} is '{}', but run_automecp makes it \
                     '{value}' from {by}; leave it out of mapper_settings",
                    setting_text(given)
                )));
            }
        }
        settings.insert(key.to_string(), JsonValue::String(value));
    }
    // Separated molecules are a start like any, not a dissociation channel.
    settings
        .entry("channels".to_string())
        .or_insert(JsonValue::Bool(false));
    let mut out = BTreeMap::new();
    out.insert("mapper_settings".to_string(), JsonValue::Object(settings));
    out.insert("step1_settings".to_string(), JsonValue::Object(step1));
    for (theirs, ours) in [
        ("step3_settings", "step2_settings"),
        ("step4_settings", "step3_settings"),
    ] {
        // Without opt_method and calc_exact_hess in step2_settings the mapper optimizes the
        // crossing points with its defaults for them (`mapper_settings::CROSSING_METHOD`).
        let block = step(ours)?;
        if !block.is_empty() {
            out.insert(theirs.to_string(), JsonValue::Object(block));
        }
    }
    Ok(JsonValue::Object(out))
}

/// The crossing edges of `network` as CSV, one line each (a crossing point with a pair of
/// minima; a point reached with other minima has a line per pair): the crossing point, its
/// kind, its two states, its energy (the mean of the two states), the minima of its two sides
/// with their energies, how far it lies above each, the norm of the gradient difference, the
/// file of its meta-IRC paths, the gap (the first state minus the second), and the check within
/// the seam (the imaginary modes, and what that makes of the point).
pub fn summary_csv(network: &Network) -> String {
    let mut out = String::from(
        "crossing,kind,state_1,state_2,energy_hartree,end_1,end_1_energy_hartree,\
         above_end_1_kcal_mol,end_2,end_2_energy_hartree,above_end_2_kcal_mol,\
         gradient_difference_hartree_bohr,paths,gap_hartree,seam_imaginary_modes,seam\n",
    );
    for e in &network.crossing_edges {
        let Some(node) = network.nodes.get(e.crossing) else {
            continue;
        };
        let NodeKind::Crossing { kind, states } = &node.kind else {
            continue;
        };
        let name = |s| {
            network
                .states
                .iter()
                .position(|x| x == s)
                .map_or_else(|| "?".to_string(), surface_name)
        };
        let _ = write!(
            out,
            "{},{},{},{},{:.10}",
            node.label(),
            kind.tag(),
            name(&states[0]),
            name(&states[1]),
            node.energy
        );
        for side in 0..2 {
            match e.ends[side].and_then(|id| network.nodes.get(id)) {
                Some(end) => {
                    let _ = write!(
                        out,
                        ",{},{:.10},{:.3}",
                        end.label(),
                        end.energy,
                        (node.energy - end.energy) * Units::HARTREE_TO_KCAL_MOL
                    );
                }
                None => out.push_str(",,,"),
            }
        }
        let (modes, seam) = match &e.seam {
            Some(s @ SeamCheck::Checked(n)) => (n.to_string(), s.describe()),
            Some(s) => (String::new(), s.describe()),
            None => (String::new(), String::new()),
        };
        let _ = writeln!(
            out,
            ",{},{},{},{},\"{}\"",
            e.gradient_difference
                .map_or_else(String::new, |g| format!("{g:.6}")),
            e.irc.file.as_deref().unwrap_or(""),
            e.gap.map_or_else(String::new, |g| format!("{g:.6e}")),
            modes,
            seam.replace('"', "'")
        );
    }
    out
}

/// What a restart of the run in `dir` with `config` does not take: a run goes on with the
/// settings it recorded (`mapper_config.json` and the settings of its mapper run), so every
/// option of `config` that differs from the settings the run started with (its
/// `settings.json`) is not used. One note per such option; none when the run has no
/// recorded settings.
pub fn restart_notes(config: &AutomecpConfig, dir: &Path) -> MoptResult<Vec<String>> {
    let path = dir.join(RUN_SETTINGS_FILE);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let recorded = AutomecpConfig::from_args(&read_settings(&path, RUN_AUTOMECP)?)?;
    let mut notes = Vec::new();
    let mut compare = |option: &str, given: String, kept: String| {
        if given != kept {
            notes.push(format!(
                "--restart: {option} is taken from the run being continued ({}); the {} given \
                 now is not used",
                if kept.is_empty() { "none" } else { &kept },
                if given.is_empty() { "none" } else { &given }
            ));
        }
    };
    compare(
        "the input",
        config.input.display().to_string(),
        recorded.input.display().to_string(),
    );
    compare(
        "-cfg",
        config.config_file.display().to_string(),
        recorded.config_file.display().to_string(),
    );
    compare("-ma", config.pushes.join(" "), recorded.pushes.join(" "));
    compare(
        "-state2",
        config.state2.join(" "),
        recorded.state2.join(" "),
    );
    let root = |r: Option<usize>| r.map_or_else(String::new, |r| r.to_string());
    compare("-root1", root(config.first_root), root(recorded.first_root));
    compare(
        "-afir",
        config.afir.clone().unwrap_or_default(),
        recorded.afir.clone().unwrap_or_default(),
    );
    compare(
        "-map_opt",
        config.mapper_options.join(" "),
        recorded.mapper_options.join(" "),
    );
    Ok(notes)
}

/// Run `config` under `base` with the calculators of `registry`; `console` is the level of
/// the screen (`None`: nothing).
pub fn run_with(
    config: &AutomecpConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<AutomecpRun> {
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => base.join(&config.output_dir),
    };
    let mapper_dir = dir.join(MAPPER_DIR);
    let mut restart = Vec::new();
    let mapper_config = if config.restart_dir.is_some() && mapper_dir.is_dir() {
        // The mapper run goes on with the settings it recorded; what is given now and differs
        // from them is said.
        restart = restart_notes(config, &dir)?;
        if let Some(level) = console {
            let log = Logger::console(level);
            for note in &restart {
                log.warn(&format!("run_automecp {note}"));
            }
        }
        let settings = read_settings(&mapper_dir.join(RUN_SETTINGS_FILE), RUN_MAPPER)?;
        MapperConfig::from_args(&settings, Some(&mapper_dir))?
    } else {
        if config.restart_dir.is_none() && dir.exists() {
            return Err(MoptError::config(format!(
                "run_automecp: the run directory {} exists; continue its run with --restart {} \
                 or give another --output_dir",
                dir.display(),
                dir.display()
            )));
        }
        let path = if config.config_file.is_absolute() {
            config.config_file.clone()
        } else {
            base.join(&config.config_file)
        };
        let content = std::fs::read_to_string(&path)
            .map_err(|e| MoptError::from(e).context(path.display()))?;
        let document = JsonValue::parse(&content).context(path.display())?;
        let translated = mapper_document(config, &document)?;
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            dir.join(RUN_SETTINGS_FILE),
            format!("{}\n", config.settings_document),
        )?;
        let file = dir.join(MAPPER_CONFIG_FILE);
        std::fs::write(&file, format!("{}\n", translated.to_string_pretty()))?;
        let input = if config.input.is_absolute() {
            config.input.clone()
        } else {
            base.join(&config.input)
        };
        let mut args = vec![
            input.display().to_string(),
            "-cfg".to_string(),
            file.display().to_string(),
            "--output_dir".to_string(),
            MAPPER_DIR.to_string(),
        ];
        if !config.mapper_options.is_empty() {
            args.push("-map_opt".to_string());
            args.extend(config.mapper_options.iter().cloned());
        }
        MapperConfig::from_args(&parser("run_mapper", RUN_MAPPER).parse(args)?, None)?
    };
    let run = mapper::run_with(&mapper_config, &dir, console, registry)?;
    let summary = dir.join(SUMMARY_FILE);
    std::fs::write(&summary, summary_csv(&run.network))?;
    // The page of the run, written again by every session of it (a stopped run has what it
    // had done).
    let name = mapper_config
        .inputs
        .first()
        .and_then(|p| p.file_stem())
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    let report = super::automecp_report::write(
        &dir,
        &run.directory,
        &name,
        &run.network,
        run.reason.as_deref().filter(|_| !run.interrupted),
    )?;
    // A search that ended without any crossing point has failed (its files say what it did).
    if !run.interrupted && run.network.crossing_points() == 0 {
        return Err(MoptError::conv(format!(
            "run_automecp: no crossing point was found{}; {} and {} show what was done",
            run.reason
                .as_deref()
                .map_or_else(String::new, |r| format!(" (the run ended: {r})")),
            summary.display(),
            report.display()
        )));
    }
    Ok(AutomecpRun {
        directory: dir,
        mapper: run,
        summary,
        report,
        restart_notes: restart,
    })
}
