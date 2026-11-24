//! `mdmain`: molecular dynamics of a structure on the surface of any calculator (with bias
//! potentials, the nanoreactor among them, and two electronic states combined by `-mf`), in
//! a fixed or a periodic cell (constant pressure with a barostat), with constraints and
//! frozen modes; or of an objective of plain variables (`-os mb2d`, `-os expr:...` with a
//! variables file; unit masses).
//!
//! The calculator options, the bias potentials, `-fix`, `-pc`, `-gfix`, `-uc`, `-es`, `-mf`,
//! `--config`, `--dump-config`, and `--restart` are those of the other commands. A restart
//! replays the evaluations of the journal with the same seed and continues from where the
//! run stopped. `-traj N` runs N trajectories, each in its own run directory with the seed
//! one higher than the one before.
//!
//! MultiOptPy: `moleculardynamics.py:MD` (`md`, `run`) with the options of
//! `interface.py:mdparser`, and `Wrapper/md_wrapper.py:MDJob` (from Python).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::bias::{parse_biases, BiasSpec};
use super::calculator::{calculator_name, calculator_options};
use super::charge_spin::{state_spec, StateSpec};
use super::mdmain_settings::{md_settings, RunOptions};
use super::option::reject_unavailable;
use super::settings::settings_json;
use super::spec::MDMAIN;
use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{AtomId, MolecularSystem, PeriodicCell};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::read_xyz_file;
use crate::md::{MdMachine, MdSettings, MdSpace, MolecularDynamics, RigidMotion};
use crate::pes::combine::PesExpr;
use crate::pes::objective::ObjectiveRegistry;
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};
use crate::pes::term::{frame_invariant, History, PesTerm};
use crate::workflow::driver::CalculatorSet;
use crate::workflow::journal::{RunJournal, JOURNAL_FILE};
use crate::workflow::md::{run_md, MdObserver, MoleculeMdSurface};
use crate::workflow::md_files::{MdFiles, MdLayout};
use crate::workflow::md_log::{MdGeometryInfo, MdLog, MdMaps};
use crate::workflow::observers::StopFile;

/// Typed `mdmain` settings.
#[derive(Debug, Clone)]
pub struct MdmainConfig {
    /// Input structures or variables files.
    pub inputs: Vec<PathBuf>,
    /// The options as parsed (for the settings document of each run).
    pub parsed: ParsedArgs,
    /// Run directory to continue (`--restart`); `None`: a new one.
    pub restart_dir: Option<PathBuf>,
    /// Calculator name.
    pub calculator: String,
    /// Calculator settings.
    pub calculator_options: CalculatorOptions,
    /// `-elec` / `-spin`.
    pub state: StateSpec,
    /// The settings of the dynamics (the seed of the first trajectory).
    pub md: MdSettings,
    /// Files, log, conserved motions, wrapping.
    pub run: RunOptions,
    /// `-traj`.
    pub trajectories: usize,
    /// `-mf NAME [charge multiplicity]`.
    pub model_function: Vec<String>,
    /// `-es`.
    pub excited_state: usize,
    /// Bias potentials.
    pub biases: Vec<BiasSpec>,
    /// `-pbc`: the cell of the structure.
    pub cell: Option<PeriodicCell>,
    /// `-gi`.
    pub geometry_info: Vec<AtomId>,
    /// `-cmds`, `-pca`.
    pub maps: (bool, bool),
    /// `-core`.
    pub threads: usize,
    /// Options read without effect (`-order`, `-addint`), for the log.
    pub without_effect: Vec<&'static str>,
    /// `-umbrella`: umbrella sampling.
    pub umbrella: Option<super::mdmain_umbrella::UmbrellaOption>,
    /// `-analysis`: the trajectory analysis at the end of a run (or of `-analyze_only`).
    pub analysis: Option<crate::workflow::md_analysis::AnalysisSettings>,
    /// `-analyze_only`: the inputs are earlier runs or trajectories, analyzed without a run.
    pub analyze_only: bool,
    /// `-coord`: the coordinate system of the dynamics (`cart` by default).
    pub coordinates: String,
    /// `-cv`: collective variables written at every step.
    pub cvs: Vec<String>,
    /// `-cv_bias`: harmonic restraints of collective variables.
    pub cv_biases: Vec<super::mdmain_cv::CvBias>,
}

fn clock_seed() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // Low bits of the clock, folded; 0 is avoided only for readability.
    ((now as u64) ^ ((now >> 64) as u64)) % 1_000_000_007 + 1
}

/// The barostat that `-vc` selects when `-mt` and `-md_opt barostat=` name none: the
/// Martyna-Tobias-Klein barostat with a fully flexible cell (`-mt npt_aniso`), with the
/// Nose-Hoover chain of the default ensemble; the cell changes as a whole, as under `-vc` of
/// the other commands.
pub const VARIABLE_CELL_BAROSTAT: &str = "mtk_aniso";

/// `-vc` (the common spelling of constant pressure, `-vc_pressure` in GPa as in the other
/// commands): the barostat of `-mt npt*` or `-md_opt barostat=`, or
/// [`VARIABLE_CELL_BAROSTAT`] with the default ensemble; the pressure of `-vc_pressure`
/// (0 GPa without it, as `optmain -vc`) unless `-press` gives it. What the dynamics of the
/// cell cannot take is refused with the reason. Without `-vc`, `-mt npt*` and `-press` work as
/// before.
///
/// MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy reads `-press` without using it).
fn variable_cell(p: &ParsedArgs, variables: bool, md: &mut MdSettings) -> MoptResult<()> {
    let Some(cell) = super::variable_cell::VariableCellSettings::from_args(p)? else {
        for key in [
            "variable_cell_pressure",
            "variable_cell_weight",
            "variable_cell_afir",
        ] {
            if p.is_present(key) {
                return Err(MoptError::config(format!(
                    "--{key} is a setting of -vc; it needs -vc"
                )));
            }
        }
        return Ok(());
    };
    if variables {
        return Err(MoptError::unsupported(
            "-vc needs structures in a periodic cell; plain variables have no cell",
        ));
    }
    if !cell.forces.is_empty() {
        return Err(MoptError::unsupported(
            "-vc_afir with mdmain: the artificial forces on the cell push it in an optimization; \
             in the dynamics the cell follows the pressure through the barostat",
        ));
    }
    if p.is_present("variable_cell_weight") {
        return Err(MoptError::unsupported(
            "-vc_weight with mdmain: the weight of the lattice variables of an optimization; the \
             inertia of the cell in the dynamics is the time constant of the barostat (-md_opt \
             tau_p=)",
        ));
    }
    if !p.get_list("variable_cell_fix").is_empty()
        || p.get_list("user_constraint")
            .iter()
            .any(|t| super::constraints::uses_cell_anywhere(t))
    {
        return Err(MoptError::unsupported(
            "-vc_fix (or -uc of a function of the cell) with mdmain: the barostat moves the cell \
             by its equations of motion and holds no constraint of the cell; the constraints of \
             the atoms (-fix, -pc, -gfix, -uc) are held",
        ));
    }
    if md.barostat.is_none() {
        if p.is_present("mdtype") {
            return Err(MoptError::unsupported(format!(
                "-vc with -mt {}: an ensemble without a barostat; give a constant-pressure \
                 ensemble (-mt npt, npt_aniso, npt_berendsen, npt_berendsen_aniso, npt_crescale, \
                 npt_crescale_aniso) or leave -mt out (-vc then runs -mt npt_aniso)",
                p.get_str("mdtype", "")
            )));
        }
        md.barostat = Some(VARIABLE_CELL_BAROSTAT.to_string());
    }
    if p.is_present("variable_cell_pressure") && p.is_present("pressure") {
        return Err(MoptError::config(
            "the pressure is given twice: -vc_pressure (GPa) or -press (kPa), one of them",
        ));
    }
    if !p.is_present("pressure") {
        md.pressure = cell.pressure_gpa / crate::md::barostat::gigapascal_per_atomic_unit();
    }
    md.validate()
}

impl MdmainConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        reject_unavailable(p, MDMAIN)?;
        let mut inputs: Vec<PathBuf> = Vec::new();
        for name in &p.positionals {
            let found = crate::core_infra::glob::expand(name)?;
            if found.is_empty() {
                return Err(MoptError::config(format!("no file matches '{name}'")));
            }
            inputs.extend(found);
        }
        if inputs.is_empty() {
            return Err(MoptError::config(
                "mdmain needs an input structure (xyz) or a variables file",
            ));
        }
        let calculator = calculator_name(p)?;
        let variables = ObjectiveRegistry::builtin().contains(&calculator);
        let (mut md, run) = md_settings(p, variables)?;
        if !run.seed_given {
            md.seed = clock_seed();
        }
        variable_cell(p, variables, &mut md)?;
        let model_function = p.get_list("model_function").to_vec();
        if md.barostat.is_some() && !model_function.is_empty() {
            return Err(MoptError::unsupported(
                "constant pressure needs the stress of one surface; -mf combines two states",
            ));
        }
        let cell = match p.get_opt_str("periodic_boundary_condition").map(str::trim) {
            Some(spec) if !spec.is_empty() && spec != "[]" => {
                Some(PeriodicCell::parse_cli_spec(spec).context("-pbc")?)
            }
            _ => None,
        };
        let geometry_info = match p.get_list("geom_info").first() {
            Some(list) => parse_atom_indices_one_based(list)?
                .into_iter()
                .map(AtomId::from_one_based)
                .collect::<MoptResult<Vec<_>>>()?,
            None => Vec::new(),
        };
        let mut without_effect = Vec::new();
        if p.get_usize("saddle_order", 0)? != 0 {
            without_effect.push("-order");
        }
        if !p.get_list("additional_inputs").is_empty() {
            without_effect.push("-addint");
        }
        let analyze_only = p.get_bool("analyze_only", false);
        let analysis = super::mdmain_analysis::parse_analysis(
            analyze_only || p.present.contains("trajectory_analysis"),
            p.get_list("trajectory_analysis"),
        )?;
        let coordinates = p.get_str("coordinate_system", "cart").trim().to_string();
        if !super::mdmain_coordinates::is_cartesian(&coordinates) {
            // An unknown name is reported before anything runs.
            crate::coord::system::CoordinateRegistry::default()
                .create(&coordinates)
                .context("-coord")?;
            if variables {
                return Err(MoptError::unsupported(
                    "-coord is a coordinate system of atoms; plain variables are the \
                     coordinates of their own dynamics",
                ));
            }
        }
        let excited_state = p.get_usize("excited_state", 0)?;
        if let Some(h) = md.hopping.as_mut() {
            h.start = excited_state;
            h.validate()
                .context("surface hopping (-md_opt hopping_states, -es)")?;
            if variables {
                return Err(MoptError::unsupported(
                    "surface hopping needs a calculator of electronic states; plain variables \
                     have one surface",
                ));
            }
            if !model_function.is_empty() {
                return Err(MoptError::unsupported(
                    "surface hopping runs on one electronic state at a time; -mf combines two \
                     states into one surface",
                ));
            }
            if p.present.contains("umbrella_sampling") {
                return Err(MoptError::unsupported(
                    "-umbrella runs on the surface of one electronic state; surface hopping is \
                     not available with it",
                ));
            }
        }
        let cvs = p.get_list("collective_variables").to_vec();
        super::mdmain_cv::check_cvs(&cvs)?;
        Ok(Self {
            cvs,
            cv_biases: super::mdmain_cv::parse_cv_biases(p.get_list("cv_bias"))?,
            coordinates,
            analysis,
            analyze_only,
            inputs,
            parsed: p.clone(),
            restart_dir: None,
            calculator,
            calculator_options: calculator_options(p)?,
            state: state_spec(p)?,
            md,
            run,
            trajectories: p.get_usize("TRAJECTORY", 1)?.max(1),
            model_function,
            excited_state,
            biases: parse_biases(p)?,
            cell,
            geometry_info,
            maps: (p.get_bool("cmds", false), p.get_bool("pca", false)),
            threads: p.get_usize("N_THREAD", 8)?.max(1),
            without_effect,
            umbrella: super::mdmain_umbrella::parse_umbrella(p.get_list("umbrella_sampling"))?,
        })
    }

    /// The settings document of the run of trajectory `k` (one trajectory, its seed).
    pub fn settings_document(&self, k: usize) -> String {
        let mut p = self.parsed.clone();
        p.values
            .insert("TRAJECTORY".to_string(), vec!["1".to_string()]);
        p.present.insert("TRAJECTORY".to_string());
        let mut options: Vec<String> = p
            .get_list("md_options")
            .iter()
            .filter(|t| !t.trim().to_ascii_lowercase().starts_with("seed="))
            .cloned()
            .collect();
        options.push(format!("seed={}", self.seed_of(k)));
        // The interval of the states in the log is written as well, given or not.
        if !options.iter().any(|t| {
            t.trim()
                .to_ascii_lowercase()
                .starts_with("structure_every=")
        }) {
            options.push(format!("structure_every={}", self.run.structure_every));
        }
        p.values.insert("md_options".to_string(), options);
        p.present.insert("md_options".to_string());
        settings_json(&p, MDMAIN).to_string_pretty()
    }

    /// The seed of trajectory `k`.
    pub fn seed_of(&self, k: usize) -> u64 {
        self.md.seed.wrapping_add(k as u64)
    }
}

/// Result of one trajectory.
#[derive(Debug, Clone)]
pub struct MdmainResult {
    /// Run directory.
    pub directory: PathBuf,
    /// The trajectory file.
    pub trajectory: PathBuf,
    /// The last structure (or values).
    pub last: PathBuf,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
    /// Steps made.
    pub steps: usize,
}

/// MultiOptPy's run directory `<date>/<stem>_MD_ASE_<timestamp>/` under `base` (with
/// `_traj<k>` for the trajectories after the first of `-traj`).
pub fn run_directory(base: &Path, input: &Path, now: &DateTime, k: usize) -> PathBuf {
    let stem = input.with_extension("");
    let suffix = if k > 0 {
        format!("_traj{k}")
    } else {
        String::new()
    };
    base.join(now.date_dir()).join(format!(
        "{}_MD_ASE_{}{suffix}",
        stem.to_string_lossy(),
        now.timestamp()
    ))
}

/// Run `mdmain` for every input and trajectory with the calculators of `registry`.
pub fn run_with(
    config: &MdmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<Vec<MdmainResult>> {
    run_with_observers(config, base, console, registry, &mut [])
}

/// [`run_with`] with further observers of every step (hooks; after the files and the log).
pub fn run_with_observers(
    config: &MdmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    extra: &mut [&mut dyn MdObserver],
) -> MoptResult<Vec<MdmainResult>> {
    if config.analyze_only {
        return super::mdmain_analysis::analyze_inputs(config, console);
    }
    if config.restart_dir.is_some() && (config.inputs.len() != 1 || config.trajectories != 1) {
        return Err(MoptError::parse(
            "--restart continues one run (one input, one trajectory)",
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.calculator_options.pool_threads(config.threads))
        .build()
        .map_err(|e| MoptError::config(format!("thread pool: {e}")))?;
    let mut results = Vec::new();
    for input in &config.inputs {
        let first = results.len();
        for k in 0..config.trajectories {
            let result = pool
                .install(|| run_one(config, input, k, base, console, registry, extra))
                .context(input.display())?;
            let stopped = result.interrupted;
            results.push(result);
            if stopped {
                return Ok(results);
            }
        }
        if config.md.hopping.is_some() && config.trajectories > 1 {
            hopping_populations(&results[first..], &stem_of(input), console)?;
        }
    }
    Ok(results)
}

/// The populations of the electronic states over the trajectories `runs` of one input,
/// written to the directory of the first of them (and its log).
fn hopping_populations(
    runs: &[MdmainResult],
    stem: &str,
    console: Option<Level>,
) -> MoptResult<()> {
    use crate::workflow::md_hopping_output::{populations, read_states};
    let Some(first) = runs.first() else {
        return Ok(());
    };
    let tables = runs
        .iter()
        .map(|r| read_states(&r.directory.join(format!("{stem}_states.csv"))))
        .collect::<MoptResult<Vec<_>>>()?;
    let p = populations(&tables)?;
    let (csv, png) = p.write(&first.directory)?;
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(first.directory.join("mopt.log"), Level::Debug)?;
    let last = p
        .shares
        .last()
        .map(|row| {
            p.states
                .iter()
                .zip(row)
                .map(|(s, x)| format!("state {s}: {x:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    log.info(&format!(
        "Surface hopping over {} trajectories: populations at the end {last}; {} and {}",
        p.trajectories,
        csv.display(),
        png.display()
    ));
    log.flush();
    Ok(())
}

pub(super) fn stem_of(input: &Path) -> String {
    input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned())
}

pub(super) fn open_run(
    config: &MdmainConfig,
    input: &Path,
    k: usize,
    base: &Path,
    console: Option<Level>,
) -> MoptResult<(PathBuf, Logger)> {
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => run_directory(base, input, &DateTime::now_utc(), k),
    };
    std::fs::create_dir_all(&dir)?;
    // A restart writes every file anew from the start.
    let _ = std::fs::remove_file(dir.join("geometry_info.csv"));
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} mdmain: {} (trajectory {} of {})",
        crate::VERSION,
        input.display(),
        k + 1,
        config.trajectories
    ));
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document(k)),
    )?;
    for option in &config.without_effect {
        log.info(&format!(
            "{option} has no effect in mdmain (it has none in MultiOptPy either)."
        ));
    }
    Ok((dir, log))
}

/// The settings of trajectory `k`.
pub(super) fn settings_of(config: &MdmainConfig, k: usize) -> MdSettings {
    MdSettings {
        seed: config.seed_of(k),
        ..config.md.clone()
    }
}

fn run_one(
    config: &MdmainConfig,
    input: &Path,
    k: usize,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    extra: &mut [&mut dyn MdObserver],
) -> MoptResult<MdmainResult> {
    let objectives = ObjectiveRegistry::builtin();
    if objectives.contains(&config.calculator) {
        return super::mdmain_variables::run_objective(
            config,
            input,
            k,
            base,
            console,
            &objectives,
            extra,
        );
    }
    let spin = config.state.of_file(input)?;
    let mut start = read_xyz_file(input, spin, config.cell.as_ref())?;
    let evaluations_before = registry.counts();
    let calc = super::oniom_micro::for_dynamics(
        &config.calculator,
        &config.calculator_options,
        &start,
        registry,
    )?;
    if calc.uses_electrons() {
        start.spin_charge.check_electrons(&start.elements)?;
    }
    let (dir, log) = open_run(config, input, k, base, console)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    log.info(&format!("Calculator: {}", calc.description()));
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&config.calculator_options.software_path_report());
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    let mut terms: Vec<(String, Arc<dyn PesTerm>)> = Vec::new();
    for spec in &config.biases {
        let n = counts.entry(spec.key.as_str()).or_insert(0);
        terms.push((spec.label(*n), spec.build(&start)?));
        *n += 1;
    }
    terms.extend(super::mdmain_cv::cv_terms(config)?);
    let mut expr =
        match super::optmain::state_surface(&config.model_function, config.excited_state)? {
            Some(states) => {
                if let Some(name) = config.model_function.first() {
                    log.info(&format!(
                        "Mode: Model Function ({name}); the dynamics runs on the combined surface."
                    ));
                }
                if config.excited_state > 0 {
                    log.info(&format!("Electronic state: {}", config.excited_state));
                }
                states
            }
            None => PesExpr::single(),
        };
    for (label, term) in &terms {
        expr = expr.plus_term(label, Arc::clone(term));
    }
    let mut settings = settings_of(config, k);
    // The constraints resolved at the input structure: frozen atoms and the values held
    // "at the start" are those of the input, whatever start the dynamics is given (a
    // Wigner sample, a restart).
    settings.constraints = super::constraints::resolve_for_run(&settings.constraints, &start)
        .context("the constraints at the input structure")?;
    // Rigid motions are conserved only where the calculator and every term are blind to
    // them (the constraints that pin the structure are seen by the dynamics itself).
    let invariant = frame_invariant(
        calc.is_rigid_body_invariant(),
        terms.iter().map(|(_, t)| t.as_ref()),
    );
    report_drift(&terms, &log);
    if config.run.rigid_auto {
        settings.rigid = match (invariant, start.has_pbc()) {
            (false, _) => RigidMotion::None,
            (true, true) => RigidMotion::Translation,
            (true, false) => RigidMotion::TranslationRotation,
        };
    }
    if let Some(umbrella) = &config.umbrella {
        if !config.model_function.is_empty() || config.excited_state > 0 {
            return Err(MoptError::unsupported(
                "-umbrella runs on the surface of one electronic state; -mf and -es are not \
                 available with it",
            ));
        }
        if crate::pes::bias::ExpressionBias::parse(&umbrella.cv).is_ok_and(|e| e.uses_cartesian()) {
            settings.rigid = RigidMotion::None;
        }
        if !super::mdmain_coordinates::is_cartesian(&config.coordinates) {
            log.info("Umbrella sampling: the windows run in Cartesian coordinates.");
        }
        let stem = stem_of(input);
        let run = super::mdmain_umbrella::run(
            umbrella,
            &settings,
            config.run.write_every,
            &journaled,
            &terms,
            &start,
            &stem,
            &dir,
            &log,
        )?;
        if run.interrupted {
            log.info(&format!(
                "Continue with: mdmain --restart {}",
                dir.display()
            ));
        }
        log.info(
            &registry
                .counts()
                .since(&evaluations_before)
                .line("this input"),
        );
        log.flush();
        let last = dir.join(format!(
            "umbrella_window_{}",
            run.windows.len().saturating_sub(1)
        ));
        return Ok(MdmainResult {
            trajectory: last.join(format!("{stem}_traj.xyz")),
            last: last.join(format!("{stem}_final.xyz")),
            directory: dir,
            interrupted: run.interrupted,
            steps: run.windows.len() * settings.steps,
        });
    }
    if let Some(w) = &config.run.wigner {
        let surface = super::mdmain_wigner::WignerSurface {
            terms: &terms,
            constraints: &settings.constraints,
            frame_invariant: invariant,
        };
        let (sampled, v) = super::mdmain_wigner::wigner_start(
            w,
            settings.temperature,
            settings.seed,
            &start,
            &journaled,
            &surface,
            &log,
        )?;
        start = sampled;
        settings.velocities = crate::md::InitialVelocities::Given(v);
    }
    // Dynamics in the coordinates of -coord, or in Cartesian coordinates.
    let mut generalized =
        super::mdmain_coordinates::machine(config, &settings, &start, invariant, &log)?;
    let mut cartesian = None;
    match &generalized {
        Some(g) => super::mdmain_coordinates::announce(g, &log),
        None => {
            let space = MdSpace::atoms(&start.masses_amu())?;
            let given = matches!(settings.velocities, crate::md::InitialVelocities::Given(_));
            let md = MolecularDynamics::new(
                settings,
                space,
                start.coords_flat_bohr(),
                Some(start.clone()),
            )?;
            super::mdmain_variables::announce(&md, &log);
            if given && md.rigid() != RigidMotion::None {
                log.info(&format!(
                    "Starting velocities (Wigner sample): their part along the conserved \
                     motions ({}) is removed, since those motions are not counted in the \
                     degrees of freedom of the temperature.",
                    md.rigid().name()
                ));
            }
            cartesian = Some(md);
        }
    }
    let md: &mut dyn MdMachine = match (generalized.as_mut(), cartesian.as_mut()) {
        (Some(g), _) => g,
        (None, Some(c)) => c,
        (None, None) => return Err(MoptError::config("no dynamics was set up")),
    };
    let steps = config.md.steps;
    let mut surface = MoleculeMdSurface::new(
        CalculatorSet::single(&journaled),
        expr,
        start.clone(),
        terms.clone(),
    )?
    .telling(log.clone());
    if let Some(h) = &config.md.hopping {
        if h.states.iter().any(|&s| s > 0) && !calc.supports_excited_states() {
            return Err(MoptError::unsupported(format!(
                "surface hopping among the states {:?}: the calculator {} has no excited states",
                h.states,
                calc.name()
            )));
        }
        announce_hopping(h, &log);
        let mut surfaces = Vec::new();
        for &state in &h.states {
            let mut e = super::optmain::state_surface(&[], state)?.unwrap_or_else(PesExpr::single);
            for (label, term) in &terms {
                e = e.plus_term(label, Arc::clone(term));
            }
            surfaces.push((state, e));
        }
        surface = surface.with_states(surfaces)?;
    }
    let stem = stem_of(input);
    let mut files = MdFiles::new(
        &dir,
        &stem,
        MdLayout::Atoms(Box::new(start.clone())),
        config.run.write_every,
    )?;
    files.select_trajectory_atoms(
        config
            .calculator_options
            .oniom
            .as_ref()
            .and_then(|spec| spec.trajectory_atoms()),
    );
    let mut printed = MdLog::new(log.clone(), config.run.print_every, steps).with_state(
        config.run.structure_every,
        MdLayout::Atoms(Box::new(start.clone())),
    );
    let mut stop = StopFile::new(&dir, log.clone());
    let mut info = MdGeometryInfo::new(config.geometry_info.clone(), start.clone(), &dir);
    let mut maps = MdMaps::new(
        config.maps.0,
        config.maps.1,
        start.clone(),
        config.run.write_every,
        steps,
        &dir,
        log.clone(),
    );
    let mut coordinates = super::mdmain_coordinates::output(config, &start, &dir, &stem)?;
    let mut cvs = super::mdmain_cv::cv_observer(config, &start, &dir, &stem)?;
    let mut flux = match config.run.heat_flux {
        true => Some(crate::workflow::md_flux::HeatFlux::new(
            calc.as_ref(),
            &start,
            &dir,
            &stem,
        )?),
        false => None,
    };
    let end = {
        let mut observers: Vec<&mut dyn MdObserver> =
            vec![&mut printed, &mut files, &mut info, &mut stop];
        if config.maps.0 || config.maps.1 {
            observers.push(&mut maps);
        }
        if let Some(c) = coordinates.as_mut() {
            observers.push(c);
        }
        if let Some(f) = flux.as_mut() {
            observers.push(f);
        }
        if let Some(c) = cvs.as_mut() {
            observers.push(c);
        }
        for o in extra.iter_mut() {
            observers.push(&mut **o);
        }
        run_md(md, &surface, &mut observers)?
    };
    let (replayed, unused) = journal.replay_counts();
    if replayed > 0 {
        log.info(&format!(
            "Restart: {replayed} evaluations taken from {JOURNAL_FILE}{}.",
            if unused > 0 {
                format!(" ({unused} recorded ones no longer matched and were recomputed)")
            } else {
                String::new()
            }
        ));
    }
    super::mdmain_variables::finish(config, md, end.is_none(), &dir, &files, &log)?;
    super::mdmain_analysis::analyze_run(
        config,
        &files.trajectory_path(),
        files.velocities_path(),
        &dir,
        &log,
    )?;
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(MdmainResult {
        directory: dir,
        trajectory: files.trajectory_path(),
        last: files.final_path(),
        interrupted: end.is_none(),
        steps: md.evaluations().saturating_sub(1),
    })
}

/// One line naming the bias potentials that make the conserved quantity of the dynamics
/// drift: those that change with the simulated time or the step (walls that move, ramps,
/// `t` and `it` in `-ubias`), and those that change with what they have seen (metadynamics
/// deposits). Without them the conserved quantity of an NVE run is conserved up to the error
/// of the integrator.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub(super) fn report_drift(terms: &[(String, Arc<dyn PesTerm>)], log: &Logger) {
    let drifting: Vec<String> = terms
        .iter()
        .filter(|(_, t)| t.history() != History::Static)
        .map(|(label, t)| format!("{label} ({:?})", t.history()))
        .collect();
    if !drifting.is_empty() {
        log.info(&format!(
            "The conserved quantity drifts with the bias potentials that change with the time \
             or with what they have seen: {} (their work on the system is not part of it).",
            drifting.join(", ")
        ));
    }
}

/// The settings and the papers of surface hopping, for the log.
fn announce_hopping(h: &crate::md::HoppingSettings, log: &Logger) {
    log.info(&format!(
        "Surface hopping (Zhu-Nakamura, without nonadiabatic couplings) among the electronic \
         states {:?} from state {}: hops considered at minima of the gap below {:.4} eV, \
         diabatic forces {}, frustrated hops: {}.",
        h.states,
        h.start,
        h.gap * crate::domain::Units::HARTREE_TO_EV,
        h.forces.name(),
        h.frustrated.name()
    ));
    for paper in [
        "T. Ishida, S. Nanbu, H. Nakamura, Int. Rev. Phys. Chem. 2017, 36, 229 (Zhu-Nakamura \
         formulas and ZN-TSH)",
        "L. Yu, C. Xu, Y. Lei, C. Zhu, Z. Wen, Phys. Chem. Chem. Phys. 2014, 16, 25883 \
         (hopping without nonadiabatic couplings)",
        "C. Zhu, H. Nakamura, J. Chem. Phys. 1994, 101, 10630 (Zhu-Nakamura theory)",
    ] {
        log.info(&format!("Reference: {paper}"));
    }
}

/// Thin wrapper used by tests and embedding code: settings for `args`.
pub fn config_from_cli(args: &[&str]) -> MoptResult<MdmainConfig> {
    let p = super::option::parser("mdmain", MDMAIN).parse(args)?;
    MdmainConfig::from_args(&p)
}

/// The structure of `input` with the cell of `-pbc` (for embedding code).
pub fn read_structure(config: &MdmainConfig, input: &Path) -> MoptResult<MolecularSystem> {
    let spin = config.state.of_file(input)?;
    read_xyz_file(input, spin, config.cell.as_ref())
}
