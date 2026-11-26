//! The settings of `run_mapper`, converted once from the command line and MultiOptPy's settings
//! file (`-cfg config.json`) into typed values: command line over the file over the defaults.
//!
//! The settings file has `mapper_settings` (MultiOptPy's keys, and the keys of `-map_opt`) and
//! `step1_settings` to `step4_settings` (the long option names of `optmain`, as for
//! `run_autots`): step 1 is the AFIR optimization, step 3 the transition-state search, step 4
//! the IRC (`intrinsic_reaction_coordinates`) and the minimizations of its ends and of the
//! inputs (`step4b_opt_method`). What sets the surface (the calculator and its settings, the
//! charge and the multiplicity, the excited state) is given once in step 1 and taken by steps 3
//! and 4 where they do not set it; the surfaces of the steps must be the same. Keys that only
//! `run_autots` uses are reported and not used; unknown keys and the parts that are not
//! implemented yet are refused.
//!
//! The defaults are MultiOptPy's (`Entrypoints/mapper.py:merge_config`): 300 K, 50 jobs, AFIR
//! with 100 kJ/mol between atoms 1.5 to 5.0 Angstrom apart, all pairs, seed 42, three guesses
//! per job. The thresholds of two structures that are the same are tighter than MultiOptPy's
//! 0.30 Angstrom and 1 kcal/mol, which took different minima for one
//! ([`super::mapper_values::DEFAULT_RMSD_THRESHOLD`]).
//!
//! MultiOptPy: `Entrypoints/mapper.py:run_mapper` (`build_parser`, `merge_config`) and
//!  `Wrapper/mapper.py:ReactionNetworkMapper._validate_mapper_settings`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::autots::step_args;
use super::confsearch_settings::ConfsearchConfig;
use super::mapper_report_data::ReportOptions;
use super::option::reject_unavailable;
use super::optmain::OptmainConfig;
use super::settings::settings_json;
use super::spec::{CONFSEARCH, OPTMAIN, RUN_MAPPER};
use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::Level;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::irc::IrcSettings;

pub use super::mapper_values::MapperSettings;
use super::mapper_values::{mapper_settings, object};

/// The points between the exact Hessians of the IRC by default
/// (`irc_hessians`, `irc_hessian_interval`; user decisions, 2026-11-18 and 2026-11-19): they
/// show where a valley turns into a ridge (a possible bifurcation) for about a tenth more
/// cost with GFN1-xTB. An exact Hessian at every point costs too much to be the default.
pub const IRC_HESSIAN_INTERVAL: usize = 10;

/// The copy of the settings file in the run directory, which a restart reads.
pub const CONFIG_SNAPSHOT: &str = "config_snapshot.json";

/// Top-level keys of a settings file of `run_autots` that the mapper does not use.
const AUTOTS_ONLY: &[&str] = &[
    "initial_mol_file",
    "work_dir",
    "software_path_file_source",
    "run_step4",
    "skip_step1",
    "skip_to_step4",
    "check_frequencies",
    "html_report",
    "step1_method",
    "step2_method",
    "workflow",
];

/// Options of `optmain` that set the surface: step 1 gives them to steps 3 and 4.
const SURFACE_KEYS: &[&str] = &[
    "othersoft",
    "usextb",
    "usedxtb",
    "pyscf",
    "electronic_charge",
    "spin_multiplicity",
    "unpaired_electrons",
    "excited_state",
    "functional",
    "basisset",
    "sub_basisset",
    "effective_core_potential",
    "dft_grid",
    "unrestrict",
    "calc_option",
    "software_path_file",
    "cpcm_solv_model",
    "alpb_solv_model",
    "SET_MEMORY",
    "N_THREAD",
    "oniom_flag",
    "oniom_middle",
    "oniom_options",
    "model_function",
    "variable_cell",
    "variable_cell_pressure",
    "variable_cell_weight",
];

/// The block of the settings file of `run_mapper` and `run_autots` with the options shared by
/// every step ([`SHARED_KEYS`]).
pub const SHARED_BLOCK: &str = "shared_settings";

/// The options of `optmain` (by destination) that `run_mapper` and `run_autots` take as shared
/// by every step: the hard constraints, the bias potentials, and `-update_proj`
/// (`tools/cli/gen_cli.py`, `WORKFLOW_SHARED`).
pub const SHARED_KEYS: &[&str] = &[
    "fix_atoms",
    "projection_constrain",
    "gradient_fix_atoms",
    "user_constraint",
    "variable_cell_fix",
    "user_bias_expression",
    "manual_AFIR",
    "repulsive_potential",
    "repulsive_potential_v2",
    "repulsive_potential_gaussian",
    "cone_potential",
    "flux_potential",
    "keep_pot",
    "keep_pot_v2",
    "anharmonic_keep_pot",
    "keep_angle",
    "keep_angle_v2",
    "universal_potential",
    "keep_dihedral_angle",
    "keep_out_of_plain_angle",
    "keep_dihedral_angle_v2",
    "keep_dihedral_angle_cos",
    "keep_out_of_plain_angle_v2",
    "void_point_pot",
    "bond_range_potential",
    "well_pot",
    "wall_well_pot",
    "void_point_well_pot",
    "around_well_pot",
    "spacer_model_potential",
    "metadynamics",
    "linear_mechano_force_pot",
    "linear_mechano_force_pot_v2",
    "asymmetric_ellipsoidal_repulsive_potential",
    "asymmetric_ellipsoidal_repulsive_potential_v2",
    "nano_reactor_potential",
    "update_projection",
];

/// The options shared by every step: the block [`SHARED_BLOCK`] of `doc` with those of the
/// command line `p` over it (`skip`: keys of `p` that are not shared there, such as the AFIR
/// push of step 1 of `run_autots`). What the command line replaces is said in `notes`.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn shared_options(
    p: &ParsedArgs,
    doc: &JsonValue,
    skip: &[&str],
    notes: &mut Vec<String>,
) -> MoptResult<BTreeMap<String, JsonValue>> {
    let mut shared = object(doc, SHARED_BLOCK)?;
    for key in shared.keys() {
        if !SHARED_KEYS.contains(&key.as_str()) {
            return Err(MoptError::parse(format!(
                "{SHARED_BLOCK}: '{key}' is not an option shared by the steps (the constraints, \
                 the bias potentials, and -update_proj of optmain: {})",
                SHARED_KEYS.join(", ")
            )));
        }
    }
    for key in SHARED_KEYS {
        if skip.contains(key) || !p.is_present(key) {
            continue;
        }
        let value = JsonValue::Array(
            p.get_list(key)
                .iter()
                .map(|s| JsonValue::String(s.clone()))
                .collect(),
        );
        if shared.insert((*key).to_string(), value).is_some() {
            notes.push(format!(
                "The command line gives '{key}', which replaces that of {SHARED_BLOCK} of the \
                 settings file."
            ));
        }
    }
    Ok(shared)
}

/// `block` (the settings of the step `name`) with the shared options `shared` merged in: a key
/// that the block sets itself wins, and `notes` says so.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn merge_shared(
    block: &mut BTreeMap<String, JsonValue>,
    shared: &BTreeMap<String, JsonValue>,
    name: &str,
    notes: &mut Vec<String>,
) {
    for (key, value) in shared {
        match block.get(key) {
            Some(own) if own != value => notes.push(format!(
                "{name} sets '{key}' itself; it overrides the value shared by every step."
            )),
            Some(_) => {}
            None => {
                block.insert(key.clone(), value.clone());
            }
        }
    }
}

/// The settings of the model functions of the AFIR paths towards crossing points
/// (`crossing_smf_alpha`, `crossing_amf_alpha`, `crossing_amf_branch`) and the largest step of
/// the crossing searches (`crossing_step`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingModels {
    /// `alpha` of the seam model function (Hartree).
    pub smf_alpha: f64,
    /// `alpha` of the Avoiding model function (Hartree).
    pub amf_alpha: f64,
    /// The upper adiabatic state of the Avoiding model function (`false`: the lower).
    pub amf_upper: bool,
    /// The largest step of the optimizations on them (Bohr).
    pub max_step: f64,
    /// The largest gap of the two states at a crossing point (Hartree; `crossing_gap`): a
    /// search that converges with a larger one has found no crossing point.
    pub gap_limit: f64,
    /// Which extrema along an AFIR path are candidates of crossing points
    /// (`crossing_candidate_prominence`, `crossing_candidate_distance`).
    pub candidates: crate::workflow::mapper::CandidateFilter,
    /// End a crossing search where the two states run parallel (`crossing_parallel_stop`;
    /// `None`: never).
    pub parallel: Option<ParallelStop>,
    /// Whether `step3_settings` name the method of the optimizations (`opt_method`). Where
    /// they name it or [`Self::own_exact_hessians`], the crossing searches take the settings
    /// of the step as they are; otherwise [`CROSSING_METHOD`] and
    /// [`CROSSING_EXACT_HESSIANS`].
    pub own_method: bool,
    /// Whether `step3_settings` say when exact Hessians are computed (`calc_exact_hess`).
    pub own_exact_hessians: bool,
    /// The check to second order within the seam of every accepted crossing point
    /// (`crossing_seam_check`, on by default).
    pub seam_check: bool,
}

/// The method of the optimization of a crossing point when the settings of the step name none
/// (`run_mapper` with states, `run_automecp`). A crossing point is a minimum of the projected
/// surface; measured on the two-state model surfaces of `tests/crossing_searches.rs`
/// (`docs/source/mapper.md`, "The search for a crossing point"): the minimizer without exact
/// Hessians converged everywhere with fewer gradients than the settings of the
/// transition-state step the searches took before (`rsirfo_block_bofill` with two exact
/// Hessians every five iterations).
pub const CROSSING_METHOD: &str = "rsirfo_block_fsb";
/// The interval of the exact Hessians of the optimization of a crossing point when the settings
/// of the step give none: none ([`CROSSING_METHOD`]).
pub const CROSSING_EXACT_HESSIANS: Option<usize> = None;

/// When a crossing search is ended because its two states run parallel
/// (`crossing_parallel_stop=on`): where the gap is closed and the difference of the two
/// gradients is small, the states are degenerate over a region (fragments apart, for
/// example), the direction of the difference gradient is noise, and the search cannot
/// converge to a point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParallelStop {
    /// The difference of the two gradients below which the states count as parallel
    /// (Hartree/Bohr; `crossing_parallel_gradient`).
    pub gradient: f64,
    /// The iterations in a row with a closed gap and such a difference that end the search
    /// (`crossing_parallel_steps`).
    pub steps: usize,
}

/// The difference of the two gradients below which two states with a closed gap count as
/// parallel when `crossing_parallel_gradient` is not given (Hartree/Bohr). With the largest
/// gap of a crossing point (1e-3 Hartree) the gap then stays closed over more than 0.1 Bohr,
/// half a largest step of the search; see `docs/source/mapper_design.md` for the measurement.
pub const DEFAULT_PARALLEL_GRADIENT: f64 = 0.01;
/// The iterations in a row that end the search when `crossing_parallel_steps` is not given.
pub const DEFAULT_PARALLEL_STEPS: usize = 10;

/// `alpha` of the seam model function when it is not given (Hartree): 30 kJ/mol, the standard
/// value of the searches on it (Harabuchi, Taketsugu, Maeda 2017; `-mf seam` keeps MultiOptPy's
/// 0.05 Hartree).
pub const DEFAULT_SMF_ALPHA: f64 = 30.0 * crate::domain::Units::KJ_MOL_TO_HARTREE;
/// `alpha` of the Avoiding model function when it is not given (Hartree): that of `-mf
/// avoiding`, 30 kJ/mol.
pub const DEFAULT_AMF_ALPHA: f64 = crate::pes::combine::AVOIDING_ALPHA;
/// The largest step of the optimizations on the model functions when it is not given (Bohr;
/// 0.1 Angstrom): their penalty and coupling terms change fast across the seam.
pub const DEFAULT_MODEL_STEP: f64 = 0.188_972_612_462_577_5;

/// The largest gap (Hartree) between the two states at the end of a converged crossing search
/// for which the end is a crossing point, when `crossing_gap` does not give one; for an MECP
/// and an MECI alike. It is a condition of its own: at convergence of the projected gradient
/// `2 (E_1 - E_2) x + P (g_1 + g_2) / 2` the gap is bounded only by `sqrt(3N)` times the force
/// threshold over two (a vector of `3N` coordinates below the threshold in RMS has a norm below
/// `sqrt(3N)` times it), which is above this limit from about 34 atoms with the default RMS
/// threshold of 2e-4 Hartree/Bohr.
pub const DEFAULT_CROSSING_GAP: f64 = 1e-3;

impl Default for CrossingModels {
    fn default() -> Self {
        Self {
            smf_alpha: DEFAULT_SMF_ALPHA,
            amf_alpha: DEFAULT_AMF_ALPHA,
            amf_upper: true,
            max_step: DEFAULT_MODEL_STEP,
            gap_limit: DEFAULT_CROSSING_GAP,
            candidates: crate::workflow::mapper::CandidateFilter::default(),
            parallel: None,
            own_method: false,
            own_exact_hessians: false,
            seam_check: true,
        }
    }
}

/// The `optmain` settings of the steps, on one surface.
#[derive(Debug, Clone)]
pub struct StepConfigs {
    /// Step 1: the AFIR optimization (also the surface of the run).
    pub biased: OptmainConfig,
    /// Step 3: the transition-state search.
    pub saddle: OptmainConfig,
    /// Step 4: the minimizations (the ends of an IRC, the inputs).
    pub minimum: OptmainConfig,
    /// Step 4: the IRC.
    pub irc: IrcSettings,
    /// Compute the Hessian at convergence (`verify_minima`). Mode kill is opt-in.
    pub verify_minima: bool,
    /// Save completed NEB iterations in the job directory.
    pub save_job_steps: bool,
    /// The IRC asks for an exact Hessian at every [`IRC_HESSIAN_INTERVAL`]-th point unless
    /// `calc_exact_hess` says otherwise
    /// (`irc_hessians`).
    pub irc_hessians: bool,
    /// Magnitude of an imaginary mode that triggers a TS窶典S link at an optimized IRC endpoint (cm^-1).
    pub irc_end_threshold_cm1: f64,
    /// The threshold of a possible bifurcation (cm^-1, `bifurcation_threshold_cm1`).
    pub bifurcation_threshold_cm1: f64,
    /// When the molecules of a structure are a dissociation channel (`channels`,
    /// `channel_margin`).
    pub channels: crate::workflow::mapper::ChannelSettings,
    /// The saddle points of order 2 or more (`higher_saddles`; `None`: off).
    pub higher_saddles: Option<crate::workflow::mapper::HigherSaddleSettings>,
    /// The possible bifurcations of the IRCs are confirmed (`confirm_bifurcations`).
    pub confirm_bifurcations: bool,
    /// The electronic states of the run (`states`; empty for one state).
    pub states: Vec<crate::workflow::mapper::ElectronicState>,
    /// The steps on each state, in the order of `states` (the root of the state as the excited
    /// state of every step; the charge and multiplicity come with the structures).
    pub on_states: Vec<StepConfigs>,
    /// The model functions of the AFIR paths towards crossing points (`crossing_afir`).
    pub crossing_models: super::mapper_settings::CrossingModels,
    /// The settings of the conformer searches of the transition states (`ts_conformers=on`):
    /// those of `confsearch -tsconf`, on the surface of the run.
    pub ts_conformers: Option<Box<ConfsearchConfig>>,
    /// Which path peaks the run saves (`path_peaks`).
    pub path_peaks: crate::workflow::mapper::PeakMode,
    /// The points between those exact Hessians (`irc_hessian_interval`).
    pub irc_hessian_interval: usize,
    /// The scan that confirms a barrierless dissociation (`confirm_dissociations`).
    pub dissociation_scan: Option<crate::workflow::mapper::ScanSettings>,
    /// Atoms of the environment of an ONIOM surface held during its relaxations (`-fix` of
    /// the steps), apart from the settings of the steps, whose optimizations move the core.
    pub held: Vec<crate::pes::constraint::Constraint>,
}

impl StepConfigs {
    /// The settings of `stage`.
    pub fn config(&self, stage: crate::workflow::mapper::Stage) -> &OptmainConfig {
        match stage {
            crate::workflow::mapper::Stage::Biased => &self.biased,
            crate::workflow::mapper::Stage::Saddle => &self.saddle,
            crate::workflow::mapper::Stage::Minimum | crate::workflow::mapper::Stage::IrcEnd => {
                &self.minimum
            }
        }
    }

    /// The constraints of the run (those of step 1).
    pub fn constraints(&self) -> &[crate::pes::constraint::Constraint] {
        &self.biased.settings.constraints
    }

    /// The atoms that the constraints of the run name, for the identity of the nodes of a
    /// structure of `atoms` atoms: a function of the structure (`-uc`, a cell constraint) names
    /// no atoms, so every atom is held then.
    pub fn held_atoms(&self, atoms: usize) -> crate::workflow::mapper::HeldAtoms {
        use crate::pes::constraint::ConstraintKind;
        let constraints = self.constraints();
        let mut held: Vec<usize> = Vec::new();
        for c in constraints {
            let named: Vec<usize> = match &c.kind {
                ConstraintKind::Distance(a, b) => a.iter().chain(b).map(|x| x.index()).collect(),
                ConstraintKind::Angle(a) => a.iter().map(|x| x.index()).collect(),
                ConstraintKind::Dihedral(a) => a.iter().map(|x| x.index()).collect(),
                ConstraintKind::Coordinate(a, _) => vec![a.index()],
                ConstraintKind::Frozen(a) | ConstraintKind::Orientation(a) => {
                    a.iter().map(|x| x.index()).collect()
                }
                ConstraintKind::AtomPair(a) => a.iter().map(|x| x.index()).collect(),
                ConstraintKind::EigenVector(_) | ConstraintKind::Function(_) => {
                    (0..atoms).collect()
                }
            };
            held.extend(named);
        }
        held.sort_unstable();
        held.dedup();
        crate::workflow::mapper::HeldAtoms {
            atoms: held,
            fixed_frame: crate::pes::constraint::breaks_rigid_body(constraints),
        }
    }

    /// These steps with the constraints of every step resolved once at `first`, the first
    /// input of the run ([`super::constraints::resolve_for_run`]): every stage of every job
    /// holds the values of that structure.
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs).
    pub fn resolved_at(mut self, first: &crate::domain::MolecularSystem) -> MoptResult<Self> {
        let resolve = |c: &mut OptmainConfig| -> MoptResult<()> {
            c.settings.constraints =
                super::constraints::resolve_for_run(&c.settings.constraints, first)
                    .context("run_mapper: the constraints at the first input")?;
            Ok(())
        };
        for c in [&mut self.biased, &mut self.saddle, &mut self.minimum] {
            resolve(c)?;
        }
        if let Some(ts) = self.ts_conformers.as_mut() {
            resolve(&mut ts.optmain)?;
        }
        self.on_states = self
            .on_states
            .into_iter()
            .map(|s| s.resolved_at(first))
            .collect::<MoptResult<_>>()?;
        Ok(self)
    }

    /// These steps with the states `states`: the steps on each of them, with its root as the
    /// excited state (a run whose steps name an excited state of their own or a model
    /// function is refused: the states say which surfaces there are).
    pub fn with_states(
        mut self,
        states: Vec<crate::workflow::mapper::ElectronicState>,
    ) -> MoptResult<Self> {
        if states.is_empty() {
            return Ok(self);
        }
        for (name, c) in [
            ("step1_settings", &self.biased),
            ("step3_settings", &self.saddle),
            ("step4_settings", &self.minimum),
        ] {
            if c.excited_state != 0 || !c.model_function.is_empty() {
                return Err(MoptError::config(format!(
                    "run_mapper: {name} names an excited state or a model function, and the \
                     run has its states (states=...): the states say on which surfaces the \
                     network lies"
                )));
            }
        }
        let mut on = Vec::with_capacity(states.len());
        for s in &states {
            let mut one = self.clone();
            for c in [&mut one.biased, &mut one.saddle, &mut one.minimum] {
                c.excited_state = s.excited_state;
            }
            on.push(one);
        }
        self.states = states;
        self.on_states = on;
        Ok(self)
    }
}

/// The settings of a run.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// The input structure files (every frame of each is an input structure).
    pub inputs: Vec<PathBuf>,
    /// The settings file as it is used (written to the run directory).
    pub document: JsonValue,
    /// The steps.
    pub steps: StepConfigs,
    /// The mapper.
    pub settings: MapperSettings,
    /// The run directory (`--output_dir`, relative to the base directory).
    pub output_dir: PathBuf,
    /// The run directory to continue (`--restart`).
    pub restart_dir: Option<PathBuf>,
    /// The network that is drawn again on the surface of this run (`-remap`): its minima are
    /// the inputs.
    pub remap: Option<super::mapper_remap::FirstNetwork>,
    /// The network whose nodes get the energies of the surface of this run at their
    /// structures (`-single_points`).
    pub single_points: Option<super::mapper_remap::FirstNetwork>,
    /// The networks that are written as one (`-merge`), the first first.
    pub merge: Vec<super::mapper_remap::FirstNetwork>,
    /// What the screen shows (`--log_level`).
    pub console: Level,
    /// The options in effect (`settings.json` of the run).
    pub settings_document: String,
    /// What of the settings file is not used, for the log.
    pub notes: Vec<String>,
    /// The settings of the HTML report (`-report_opt`).
    pub report: ReportOptions,
}

/// The block of step `k` (1 to 4) with the surface of step 1 where it gives none.
fn step_block(
    doc: &JsonValue,
    k: usize,
    osp: Option<&str>,
) -> MoptResult<BTreeMap<String, JsonValue>> {
    let first = object(doc, "step1_settings")?;
    let mut block = object(doc, &format!("step{k}_settings"))?;
    if k != 1 {
        for key in SURFACE_KEYS {
            if let Some(v) = first.get(*key) {
                block.entry((*key).to_string()).or_insert_with(|| v.clone());
            }
        }
    }
    if let Some(path) = osp {
        block.insert(
            "software_path_file".to_string(),
            JsonValue::String(path.to_string()),
        );
    }
    Ok(block)
}

fn text(v: &str) -> JsonValue {
    JsonValue::String(v.to_string())
}

/// The block of the settings file with the settings of the conformer searches of the
/// transition states (`ts_conformers=on`): the options of `confsearch`.
pub const TS_CONFORMER_BLOCK: &str = "ts_conformer_settings";

/// The settings of the conformer searches of the transition states: the block
/// [`TS_CONFORMER_BLOCK`] as the options of `confsearch -tsconf` for `input`, on the surface of
/// `step1_settings` (`biased`), with the seed of the run unless the block has one (a restart
/// repeats the searches). The check of the reaction by an IRC inside the search is off: the
/// mapper runs the IRC of every conformer itself.
fn ts_conformer_config(
    doc: &JsonValue,
    osp: Option<&str>,
    input: &Path,
    seed: u64,
    biased: &OptmainConfig,
    shared: &BTreeMap<String, JsonValue>,
    notes: &mut Vec<String>,
) -> MoptResult<ConfsearchConfig> {
    let name = TS_CONFORMER_BLOCK;
    let first = object(doc, "step1_settings")?;
    let mut block = object(doc, name)?;
    for key in SURFACE_KEYS {
        if let Some(v) = first.get(*key) {
            block.entry((*key).to_string()).or_insert_with(|| v.clone());
        }
    }
    merge_shared(&mut block, shared, name, notes);
    if let Some(path) = osp {
        block.insert("software_path_file".to_string(), text(path));
    }
    let mut options = match block.remove("confsearch_options") {
        None | Some(JsonValue::Null) => Vec::new(),
        Some(JsonValue::Array(items)) => items,
        Some(one) => vec![one],
    };
    let has_seed = options.iter().any(|o| {
        o.as_str()
            .is_some_and(|t| t.trim().to_ascii_lowercase().starts_with("seed="))
    });
    if !has_seed {
        options.push(text(&format!("seed={seed}")));
    }
    block.insert("confsearch_options".to_string(), JsonValue::Array(options));
    let args = step_args(
        &block,
        &[input],
        &[("ts_conformers", JsonValue::Bool(true))],
        &[],
        CONFSEARCH,
    )
    .context(name)?;
    let mut config = ConfsearchConfig::from_args(&args).context(name)?;
    if let Some(t) = config.ts.as_mut() {
        t.check_reaction = false;
    }
    check_step(name, &config.optmain)?;
    // The state as the input has it: the search writes the state of the input file into its
    // settings, where step 1 leaves it to the file.
    let mut differences = super::autots_pair::surface_differences(biased, &config.optmain);
    differences.retain(|d| !d.starts_with("charge and multiplicity"));
    // The enthalpy of a variable cell is the surface of every step.
    let cell = |c: &OptmainConfig| c.variable_cell.as_ref().map(|v| (v.pressure_gpa, v.weight));
    if cell(biased) != cell(&config.optmain) {
        let shown = |v: Option<(f64, f64)>| match v {
            Some((p, w)) => format!("a variable cell at {p} GPa with the weight {w}"),
            None => "a fixed cell".to_string(),
        };
        differences.push(format!(
            "the cell (-vc: {} against {})",
            shown(cell(biased)),
            shown(cell(&config.optmain))
        ));
    }
    let (ours, theirs) = (
        biased.state.of_file(input)?,
        config.optmain.state.of_file(input)?,
    );
    if ours != theirs {
        differences.push(format!(
            "charge and multiplicity ({} {} against {} {})",
            ours.charge, ours.multiplicity, theirs.charge, theirs.multiplicity
        ));
    }
    if !differences.is_empty() {
        return Err(MoptError::unsupported(format!(
            "run_mapper: step1_settings and {name} do not run on the same surface (they \
             differ in {})",
            differences.join("; ")
        )));
    }
    Ok(config)
}

/// Refuse what the steps of the mapper do not support yet.
fn check_step(name: &str, c: &OptmainConfig) -> MoptResult<()> {
    let refuse = |what: &str| {
        Err(MoptError::unsupported(format!(
            "run_mapper: {name} has {what}, which the mapper does not support yet"
        )))
    };
    if !c.model_function.is_empty() {
        return refuse("-mf (a function of several states; the search for crossing points is not implemented yet)");
    }
    if c.variable_cell
        .as_ref()
        .is_some_and(|v| !v.forces.is_empty())
    {
        return refuse(
            "-vc_afir (forces on the cell itself; the mapper pushes pairs of atoms, and the \
             stress of the push moves the cell)",
        );
    }
    let oniom = c.calculator_options.oniom.is_some();
    if oniom && !c.biases.is_empty() {
        return Err(MoptError::unsupported(format!(
            "run_mapper: {name} has bias potentials on an ONIOM surface: the calculator with \
             the biases added does not hand on the environment that the microiterations \
             relaxed, which every calculation of the mapper on an ONIOM surface starts from"
        )));
    }
    // On an ONIOM surface -fix holds atoms of the environment during the relaxations (the
    // distant part of a protein); the mapper checks that none of them is in the core.
    let held_environment = c
        .settings
        .constraints
        .iter()
        .all(|k| matches!(k.kind, crate::pes::constraint::ConstraintKind::Frozen(_)));
    if oniom && !c.settings.constraints.is_empty() && !held_environment {
        return refuse(
            "constraints other than -fix on an ONIOM surface (-fix holds atoms of the \
             environment; the core is explored freely)",
        );
    }
    if let Some(k) = c.settings.constraints.iter().find(|k| {
        matches!(
            k.kind,
            crate::pes::constraint::ConstraintKind::EigenVector(_)
        )
    }) {
        return Err(MoptError::unsupported(format!(
            "run_mapper: {name} has the constraint {:?}: an eigenvector constraint is resolved \
             with the Hessian at the start, and the constraints of a network are resolved once \
             at its first input, which has none",
            k.kind
        )));
    }
    Ok(())
}

impl MapperConfig {
    /// The settings of the command line `p`; with `restart`, the settings file is the copy in
    /// that run directory.
    pub fn from_args(p: &ParsedArgs, restart: Option<&Path>) -> MoptResult<Self> {
        reject_unavailable(p, RUN_MAPPER)?;
        let mut inputs = Vec::new();
        for name in &p.positionals {
            let found = crate::core_infra::glob::expand(name)?;
            if found.is_empty() {
                return Err(MoptError::config(format!("no file matches '{name}'")));
            }
            inputs.extend(found);
        }
        // The network of another run drawn again (-remap): its minima are the inputs, and the
        // structure file of its first minimum is what the settings of the steps are made with.
        let remap = match p.get_opt_str("remap") {
            Some(path) => Some(super::mapper_remap::FirstNetwork::read(Path::new(path))?),
            None => None,
        };
        let single_points = match p.get_opt_str("single_points") {
            Some(path) => Some(super::mapper_remap::FirstNetwork::read(Path::new(path))?),
            None => None,
        };
        // The networks of several runs written as one (-merge).
        let merge = p
            .get_list("merge")
            .iter()
            .map(|path| super::mapper_remap::FirstNetwork::read(Path::new(path)))
            .collect::<MoptResult<Vec<_>>>()?;
        if [remap.is_some(), single_points.is_some(), !merge.is_empty()]
            .iter()
            .filter(|given| **given)
            .count()
            > 1
        {
            return Err(MoptError::parse(
                "run_mapper: -remap draws a network again, -single_points keeps its \
                 structures, and -merge writes several networks as one: one of the three",
            ));
        }
        if !merge.is_empty() && restart.is_some() {
            return Err(MoptError::parse(
                "run_mapper: -merge writes a directory of its own and runs nothing that \
                 --restart could continue",
            ));
        }
        let first = match (
            remap.as_ref().or(single_points.as_ref()).or(merge.first()),
            inputs.first(),
        ) {
            (Some(_), Some(_)) => {
                return Err(MoptError::parse(
                    "run_mapper: -remap, -single_points and -merge take the structures of \
                     their network; no input file is given with it",
                ))
            }
            (Some(network), None) => network.first_file()?,
            (None, Some(file)) => file.clone(),
            (None, None) => {
                return Err(MoptError::parse(
                    "run_mapper needs an input structure (xyz)",
                ))
            }
        };
        let given = p.get_opt_str("config_file");
        let path = match restart {
            Some(dir) => dir.join(CONFIG_SNAPSHOT),
            None => PathBuf::from(given.unwrap_or("./config.json")),
        };
        let document = if path.is_file() {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| MoptError::from(e).context(path.display()))?;
            JsonValue::parse(&content).context(path.display())?
        } else if restart.is_some() || given.is_some_and(|g| g != "./config.json") {
            return Err(MoptError::config(format!(
                "run_mapper: no settings file {}",
                path.display()
            )));
        } else {
            JsonValue::Object(BTreeMap::new())
        };
        let Some(top) = document.as_object() else {
            return Err(MoptError::parse(
                "run_mapper: the settings file is a JSON object",
            ));
        };
        let mut notes = Vec::new();
        for key in top.keys() {
            let known = [
                "mapper_settings",
                "top_n_candidates",
                TS_CONFORMER_BLOCK,
                SHARED_BLOCK,
            ]
            .contains(&key.as_str())
                || (1..=4).any(|k| *key == format!("step{k}_settings"));
            if AUTOTS_ONLY.contains(&key.as_str()) {
                notes.push(format!(
                    "The key '{key}' of the settings file belongs to run_autots and is not used."
                ));
            } else if !known {
                return Err(MoptError::parse(format!(
                    "run_mapper: unknown key '{key}' in the settings file (mapper_settings, \
                     step1_settings to step4_settings, {TS_CONFORMER_BLOCK}, {SHARED_BLOCK}, \
                     top_n_candidates)"
                )));
            }
        }
        let (mut settings, output_dir) = mapper_settings(p, &document)?;
        let has_step2 = top.contains_key("step2_settings");
        let osp = p
            .is_present("software_path_file")
            .then(|| p.get_str("software_path_file", "").to_string());
        let mut step1 = step_block(&document, 1, osp.as_deref())?;
        if step1.remove("manual_AFIR").is_some() {
            notes.push(
                "manual_AFIR of step1_settings is replaced by the AFIR pushes of every job (as \
                 in MultiOptPy); a static AFIR bias of the surface of every step is -ma or \
                 manual_AFIR of shared_settings."
                    .to_string(),
            );
        }
        // The constraints and bias potentials shared by every step (the command line over
        // the block of the settings file), merged into the blocks of the steps.
        let shared = shared_options(p, &document, &[], &mut notes)?;
        merge_shared(&mut step1, &shared, "step1_settings", &mut notes);
        // The relaxation of the paths (step 2) holds the constraints and biases of the run:
        // its block may repeat them, not change them.
        let step2 = object(&document, "step2_settings")?;
        for key in SHARED_KEYS {
            if let Some(own) = step2.get(*key) {
                if step1.get(*key) != Some(own) {
                    return Err(MoptError::unsupported(format!(
                        "run_mapper: step2_settings sets '{key}' to another value than the other \
                         steps; the relaxation of a path holds the constraints and bias \
                         potentials of the run (give them once, in {SHARED_BLOCK} or on the \
                         command line)"
                    )));
                }
            }
        }
        let input: &Path = &first;
        let build = |block: &BTreeMap<String, JsonValue>,
                     extra: &[(&str, JsonValue)],
                     drop: &[&str],
                     name: &str|
         -> MoptResult<OptmainConfig> {
            let args = step_args(block, &[input], extra, drop, OPTMAIN).context(name)?;
            OptmainConfig::from_args(&args).context(name)
        };
        let mut biased = build(
            &step1,
            &[("saddle_order", text("0"))],
            &[],
            "step1_settings",
        )?;
        // The default way of exploring: AFIR pushes between atoms for structures, ADDF for
        // the points of an objective of plain variables (which have no atoms to push).
        let variables =
            crate::pes::objective::ObjectiveRegistry::builtin().contains(&biased.calculator);
        if single_points.is_some() {
            // A run of single points explores nothing.
            let name = super::mapper_points::POINTS_METHOD;
            if !settings.method.is_empty() && !settings.method.eq_ignore_ascii_case(name) {
                return Err(MoptError::parse(format!(
                    "run_mapper: -single_points explores nothing, method={} has no place",
                    settings.method
                )));
            }
            settings.method = name.to_string();
        } else if remap.is_some() {
            // The method of a remap is `remap`, and without a limit of jobs given it runs
            // every item of the first network.
            let name = crate::workflow::mapper::Remap::NAME;
            if !settings.method.is_empty() && !settings.method.eq_ignore_ascii_case(name) {
                return Err(MoptError::parse(format!(
                    "run_mapper: -remap runs the method {name}, not method={}",
                    settings.method
                )));
            }
            settings.method = name.to_string();
            if !settings.max_jobs_given {
                settings.max_jobs = 0;
            }
        } else if settings
            .method
            .eq_ignore_ascii_case(crate::workflow::mapper::Remap::NAME)
        {
            return Err(MoptError::parse(
                "run_mapper: method=remap draws the network of another run again; give it with \
                 -remap RUN_DIR",
            ));
        }
        if settings.method.is_empty() {
            settings.method = if variables {
                crate::workflow::mapper::AddfMap::NAME
            } else {
                crate::workflow::mapper::AfirTs::NAME
            }
            .to_string();
        }
        if has_step2 {
            // The settings of nebmain for the relaxation of the AFIR paths of afir_ts.
            let off = settings.method_extra.get("relax").is_some_and(|v| {
                ["off", "none", "false", "no", "0"].contains(&v.to_ascii_lowercase().as_str())
            });
            if !settings
                .method
                .eq_ignore_ascii_case(crate::workflow::mapper::afir::AfirTs::NAME)
                || off
            {
                notes.push(
                    "step2_settings (the settings of nebmain that relaxes the AFIR paths of \
                     afir_ts) are not used: another method, or method_relax=off."
                        .to_string(),
                );
            }
        }
        let mut step3 = step_block(&document, 3, osp.as_deref())?;
        merge_shared(&mut step3, &shared, "step3_settings", &mut notes);
        // The crossing searches take the method and the exact Hessians of the step only where
        // it names them (`CrossingModels::own_method`).
        settings.crossing_models.own_method = step3.contains_key("opt_method");
        settings.crossing_models.own_exact_hessians = step3.contains_key("calc_exact_hess");
        step3
            .entry("calc_exact_hess".to_string())
            .or_insert_with(|| text("5"));
        let mut saddle = build(
            &step3,
            &[("saddle_order", text("1"))],
            &[],
            "step3_settings",
        )?;
        let mut step4 = step_block(&document, 4, osp.as_deref())?;
        merge_shared(&mut step4, &shared, "step4_settings", &mut notes);
        let end_method = step4
            .get("step4b_opt_method")
            .cloned()
            .unwrap_or_else(|| JsonValue::Array(vec![text("rsirfo_block_fsb")]));
        let mut minimum = build(
            &step4,
            &[("saddle_order", text("0")), ("opt_method", end_method)],
            &["step4b_opt_method", "intrinsic_reaction_coordinates"],
            "step4_settings",
        )?;
        let irc = build(
            &step4,
            &[("saddle_order", text("1"))],
            &["step4b_opt_method"],
            "step4_settings",
        )?
        .irc;
        let mut irc = match irc {
            Some(s) => s,
            None => IrcSettings::from_multioptpy("0.1", "300", "lqa")?,
        };
        irc.exact_hessian_interval = minimum.settings.exact_hessian_interval;
        irc.project_updates = minimum.settings.project_updates;
        for (name, c) in [
            ("step1_settings", &biased),
            ("step3_settings", &saddle),
            ("step4_settings", &minimum),
        ] {
            check_step(name, c)?;
        }
        for (name, c) in [("step3_settings", &saddle), ("step4_settings", &minimum)] {
            let differences = super::autots_pair::surface_differences(&biased, c);
            if !differences.is_empty() {
                return Err(MoptError::unsupported(format!(
                    "run_mapper: step1_settings and {name} do not run on the same surface (they \
                     differ in {}); a network is explored on one surface, and it is drawn \
                     again on another by a run of its own (-remap RUN_DIR)",
                    differences.join("; ")
                )));
            }
        }
        // -fix of an ONIOM run holds atoms of the environment: the surface holds them, and they
        // leave the settings of the optimizations, which move the core (numbered as the core).
        // Elsewhere the constraints stay in the settings of every step: every optimization
        // holds them (resolved once at the first input, `StepConfigs::resolved_at`).
        let held = if biased.calculator_options.oniom.is_some() {
            saddle.settings.constraints.clear();
            minimum.settings.constraints.clear();
            std::mem::take(&mut biased.settings.constraints)
        } else {
            Vec::new()
        };
        // What needs molecules is refused for the points of an objective of plain variables.
        if variables
            && (!biased.settings.constraints.is_empty()
                || !biased.biases.is_empty()
                || !saddle.biases.is_empty()
                || !minimum.biases.is_empty())
        {
            return Err(MoptError::unsupported(
                "run_mapper: constraints and bias potentials on an objective of plain variables: \
                 the optimizations of plain variables of the mapper hold none (the constraints \
                 and biases are those of structures)",
            ));
        }
        if variables {
            for (given, name) in [
                (settings.entrance_structures > 0, "entrance_structures"),
                (settings.max_bond_changes.is_some(), "max_bond_changes"),
                (
                    settings.max_species_changes.is_some(),
                    "max_species_changes",
                ),
                (
                    settings.dissociation_scan.is_some(),
                    "confirm_dissociations",
                ),
            ] {
                if given {
                    return Err(MoptError::config(format!(
                        "run_mapper: {name} needs the molecules of a structure; the points of \
                         an objective of plain variables have none"
                    )));
                }
            }
        }
        // Hard constraints and static biases are a part of the surface of every stage
        // (`docs/source/mapper_design.md`, "Constraints and bias potentials of the user"); what
        // cannot hold them is refused by name.
        let constrained = !biased.settings.constraints.is_empty();
        let biased_surface = !biased.biases.is_empty();
        if constrained || biased_surface {
            let what = match (constrained, biased_surface) {
                (true, true) => "constraints and bias potentials",
                (true, false) => "constraints",
                _ => "bias potentials",
            };
            if settings.channels && settings.channels_given {
                return Err(MoptError::unsupported(format!(
                    "run_mapper: channels=on with {what}: the fragments of a dissociation \
                     channel are optimized alone, where {what} over the whole structure have no \
                     meaning"
                )));
            }
            if settings.channels {
                settings.channels = false;
                notes.push(format!(
                    "Dissociation channels are off: the run has {what}, which the fragments of a \
                     channel, optimized alone, cannot hold."
                ));
            }
            let refused = [
                (
                    settings.entrance_structures > 0,
                    "entrance_structures (molecules turned and placed at random break the held \
                     values)",
                ),
                (
                    settings
                        .method
                        .eq_ignore_ascii_case(crate::workflow::mapper::Nanoreactor::NAME),
                    "method=nanoreactor (its walls and dynamics are a surface of their own)",
                ),
                (
                    !settings.states.is_empty(),
                    "several electronic states (states: the crossing searches do not hold them \
                     yet)",
                ),
                (
                    constrained && (settings.higher_saddles.is_some() || settings.upward.is_some()),
                    "higher_saddles or upward (the high-index saddle dynamics does not hold \
                     constraints)",
                ),
                (
                    constrained && biased.variable_cell.is_some(),
                    "a variable cell (variable_cell: the IRC of the atoms and the cell holds no \
                     constraints, as optmain -vc -irc refuses them)",
                ),
                (
                    constrained
                        && settings
                            .method_extra
                            .get("relax")
                            .is_some_and(|v| v.eq_ignore_ascii_case("dmf")),
                    "method_relax=dmf (the direct MaxFlux method does not hold constraints)",
                ),
            ];
            if let Some((_, name)) = refused.iter().find(|(given, _)| *given) {
                return Err(MoptError::unsupported(format!(
                    "run_mapper: {what} with {name}"
                )));
            }
        }
        if settings.entrance_structures > 0 {
            if remap.is_some() || single_points.is_some() || !merge.is_empty() {
                return Err(MoptError::config(
                    "run_mapper: entrance_structures belongs to an exploration; -remap, \
                     -single_points and -merge take the inputs of their first network",
                ));
            }
            if !settings.states.is_empty() {
                return Err(MoptError::unsupported(
                    "run_mapper: entrance_structures with several states (states=...) is not \
                     available yet",
                ));
            }
        }
        // The conformer searches of the transition states (ts_conformers=on).
        let ts_conformers = if settings.ts_conformers {
            if variables {
                return Err(MoptError::config(
                    "run_mapper: ts_conformers=on searches the conformers of molecules; the \
                     points of an objective of plain variables have none",
                ));
            }
            if remap.is_some() || single_points.is_some() {
                return Err(MoptError::config(
                    "run_mapper: ts_conformers=on belongs to an exploration; -remap and \
                     -single_points take the transition states of their first network",
                ));
            }
            if !settings.states.is_empty() {
                return Err(MoptError::unsupported(
                    "run_mapper: ts_conformers=on with several states (states=...) is not \
                     available yet",
                ));
            }
            Some(Box::new(ts_conformer_config(
                &document,
                osp.as_deref(),
                input,
                settings.seed,
                &biased,
                &shared,
                &mut notes,
            )?))
        } else {
            if top.contains_key(TS_CONFORMER_BLOCK) {
                notes.push(format!(
                    "{TS_CONFORMER_BLOCK} (the settings of the conformer searches of the \
                     transition states) are not used: ts_conformers is off."
                ));
            }
            None
        };
        let console = match p.get_opt_str("log_level") {
            Some(level) => Level::parse(level)?,
            None => Level::Info,
        };
        Ok(Self {
            inputs,
            document,
            steps: StepConfigs {
                biased,
                saddle,
                minimum,
                irc,
                verify_minima: settings.verify_minima,
                save_job_steps: settings.save_job_steps,
                irc_hessians: settings.irc_hessians && !step4.contains_key("calc_exact_hess"),
                irc_end_threshold_cm1: settings.irc_end_threshold_cm1,
                bifurcation_threshold_cm1: settings.bifurcation_threshold_cm1,
                channels: crate::workflow::mapper::ChannelSettings {
                    enabled: settings.channels,
                    margin: settings.channel_margin * crate::domain::Units::ANGSTROM_TO_BOHR,
                    mirror_images: settings.mirror,
                    ..crate::workflow::mapper::ChannelSettings::default()
                },
                higher_saddles: settings.higher_saddles.map(|max_order| {
                    crate::workflow::mapper::HigherSaddleSettings {
                        max_order,
                        step: crate::workflow::mapper::method::DEFAULT_DESCENT_STEP,
                    }
                }),
                confirm_bifurcations: settings.confirm_bifurcations,
                states: Vec::new(),
                on_states: Vec::new(),
                crossing_models: settings.crossing_models,
                ts_conformers,
                path_peaks: settings.path_peaks,
                irc_hessian_interval: settings.irc_hessian_interval,
                dissociation_scan: settings.dissociation_scan,
                held,
            }
            .with_states(settings.states.clone())?,
            settings,
            output_dir,
            restart_dir: restart.map(Path::to_path_buf),
            remap,
            single_points,
            merge,
            console,
            settings_document: settings_json(p, RUN_MAPPER).to_string_pretty(),
            notes,
            report: ReportOptions::parse(p.get_list("report_options"))?,
        })
    }
}

#[cfg(test)]
#[path = "mapper_settings_tests.rs"]
mod tests;
