//! `optmain`: geometry optimization of minima and saddle points.
//!
//! With an objective of plain variables as the calculator (`-os mb2d`, `-os expr:...`) the
//! input is a variables file and the run is handled by [`super::variables`].
//!
//! MultiOptPy: `optimization.py:Optimize` (`run`, `_run_single_job`, `_setup_directory`,
//!  `_print_info`) with the options of `interface.py:optimizeparser`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::bias::{parse_biases, BiasSpec};
use super::calculator::{calculator_name, calculator_options};
use super::charge_spin::{parse_pair, state_spec, StateSpec};
use super::constraints::parse_constraints;
use super::option::reject_unavailable;
use super::settings::settings_json;
use super::spec::OPTMAIN;
use super::variable_cell::{parse_cell_forces, VariableCellSettings};
use crate::coord::connectivity::ShapeCondition;
use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{AtomId, Bohr, MolecularSystem, Quantity, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::read_xyz_file;
use crate::opt::cell::DEFAULT_LATTICE_WEIGHT;
use crate::opt::convergence::ConvergenceCriteria;
use crate::opt::crossing_check::CrossingCheckSettings;
use crate::opt::mode_kill::ModeKillSettings;
use crate::opt::state_machine::{GeometryOptimizer, OptimizerSettings};
use crate::opt::step::ccqn::{CcqnAxis, CcqnSettings, ConeSolver};
use crate::opt::trust::TrustSettings;
use crate::pes::calculator::Calculator;
use crate::pes::combine::{model_function, two_states_of, PesExpr, TaskSpec};
use crate::pes::objective::ObjectiveRegistry;
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};
use crate::vib::{
    self, Direction, EnthalpyTreatment, FrequencySettings, QuasiHarmonic, RotorMoment,
    ThermoSettings,
};
use crate::workflow::driver::CalculatorSet;
use crate::workflow::journal::{RunJournal, JOURNAL_FILE};
use crate::workflow::observers::{
    GeometryInfo, NegativeEigenvalueGuard, ProgressLog, RunFiles, ShapeGuard, StopFile,
};
use crate::workflow::path_analysis::{KoopmanLog, PathMaps};
use crate::workflow::{run_optimization, Observer};

/// Calculators available without external programs (`-os`).
pub use crate::pes::registry::NATIVE_CALCULATORS;

/// Typed `optmain` settings.
#[derive(Debug, Clone)]
pub struct OptmainConfig {
    /// Input structures (`INPUT`).
    pub inputs: Vec<PathBuf>,
    /// The settings in effect as a settings document (`settings.json` of each run).
    pub settings_document: String,
    /// Run directory to continue (`--restart`); `None`: a new one.
    pub restart_dir: Option<PathBuf>,
    /// Calculator name (`-os`, `-xtb`, `-dxtb`; default PySCF).
    pub calculator: String,
    /// Calculator settings (`-func`, `-bs`, `-calc_opt`, ...).
    pub calculator_options: CalculatorOptions,
    /// Keep a full-coordinate path when another workflow needs optimization frames as input.
    pub(crate) keep_full_path: bool,
    /// `-elec` / `-spin`.
    pub state: StateSpec,
    /// Optimizer settings.
    pub settings: OptimizerSettings,
    /// Use HiSD at the requested order after a standalone mode kill.
    pub(crate) hisd_mode_kill: bool,
    /// `-mf NAME [charge multiplicity]`.
    pub model_function: Vec<String>,
    /// `-es`: the electronic state (0: the ground state).
    pub excited_state: usize,
    /// `-ih FILE`: the first Hessian that is asked for is read from this file
    /// (`io::hessian_file`).
    pub initial_hessian: Option<PathBuf>,
    /// `-follow`: a direction (xyz file of displacements) whose closest Hessian mode a
    /// transition-state search takes uphill and then tracks.
    pub follow: Option<PathBuf>,
    /// `-sc`.
    pub shape_conditions: Vec<ShapeCondition>,
    /// `-gi`: atoms whose distances, angles, and dihedral angles are written at every
    /// iteration (fewer than two: nothing).
    pub geometry_info: Vec<AtomId>,
    /// `-negeigval`: stop a saddle-point search without a negative eigenvalue at the start.
    pub detect_negative_eigenvalues: bool,
    /// `-debug`: show the messages of the inner solvers on the screen as well (they are
    /// always in `mopt.log`).
    pub debug: bool,
    /// `-vc`: optimize the cell together with the atoms.
    pub variable_cell: Option<VariableCellSettings>,
    /// `-cmds`: map of the structures of the run by multidimensional scaling.
    pub cmds: bool,
    /// `-pca`: principal components of the structures of the run.
    pub pca: bool,
    /// `-km`: dynamic modes of the last iterations at every iteration.
    pub koopman: bool,
    /// `-bitss`: settings of the binary-image transition state search (`-mf bitss`).
    pub bitss: crate::path::bitss::BitssSettings,
    /// `-core`.
    pub threads: usize,
    /// `-nosymm`.
    pub no_symmetry: bool,
    /// `-freq`: vibrational analysis and thermochemistry of the converged structure.
    pub frequency_analysis: bool,
    /// `-force_freq`: the same also when the optimization did not converge.
    pub force_frequency_analysis: bool,
    /// `-temp`, `-press`, `-freq_cutoff`, `-freq_corr`, `-freq_lowest`, `-symm_tol`.
    pub frequency: FrequencySettings,
    /// `-project_gradient`.
    pub project_gradient: bool,
    /// CCQN reactive bonds (`-ccqn_bonds`).
    pub ccqn_bonds: Vec<(AtomId, AtomId)>,
    /// CCQN product structure (`-ccqn_product`).
    pub ccqn_product: Option<PathBuf>,
    /// `-ccqn_step` as given (angstrom for molecules, the unit of the variables otherwise).
    pub ccqn_step: Option<f64>,
    /// Bias potentials (`-ma`, `-kp`, ..., `-nrp`).
    pub biases: Vec<BiasSpec>,
    /// The bias options as written (flag and arguments), for `-bias_derivatives`.
    pub bias_options: Vec<(String, Vec<String>)>,
    /// `-bias_derivatives`: the derivatives in the constants of the bias options after a
    /// converged optimization.
    pub bias_derivatives: bool,
    /// `-irc STEP_SIZE MAX_STEP METHOD` (with `-irc_opt`, `-fc`): the IRC from the
    /// optimized structure.
    pub irc: Option<crate::irc::IrcSettings>,
    /// A further file whose appearance stops the run as the `STOP` file of its directory
    /// does (a command that runs optmain for each of its jobs, such as `relaxedscan`, watches
    /// its own); none from the command line.
    pub stop_file: Option<PathBuf>,
    /// Write the Hessian the optimizer has at the end (Cartesian, Hartree/Bohr^2, NumPy) to
    /// this file, where its coordinates are Cartesian: the start of the next point of a scan
    /// (`-ih`); none from the command line.
    pub final_hessian: Option<PathBuf>,
}

fn first_f64(p: &ParsedArgs, key: &str, default: f64) -> MoptResult<f64> {
    match p.get_list(key).first() {
        Some(s) => s
            .parse()
            .map_err(|e| MoptError::parse(format!("invalid number for {key}: '{s}' ({e})"))),
        None => Ok(default),
    }
}

fn atom_list(expr: &str) -> MoptResult<Vec<AtomId>> {
    parse_atom_indices_one_based(expr)?
        .into_iter()
        .map(AtomId::from_one_based)
        .collect()
}

fn parse_bonds(tokens: &[String]) -> MoptResult<Vec<(AtomId, AtomId)>> {
    tokens
        .iter()
        .map(|t| {
            let idx = parse_atom_indices_one_based(t)?;
            match idx.as_slice() {
                [a, b] => Ok((AtomId::from_one_based(*a)?, AtomId::from_one_based(*b)?)),
                _ => Err(MoptError::parse(format!(
                    "-ccqn_bonds expects atom pairs such as 1,2; got '{t}'"
                ))),
            }
        })
        .collect()
}

impl OptmainConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        reject_unavailable(p, OPTMAIN)?;
        // Names with wildcards stand for the files that match (MultiOptPy's glob).
        let mut inputs: Vec<PathBuf> = Vec::new();
        for name in &p.positionals {
            let found = crate::core_infra::glob::expand(name)?;
            if found.is_empty() {
                return Err(MoptError::config(format!("no file matches '{name}'")));
            }
            inputs.extend(found);
        }
        if inputs.is_empty() {
            return Err(MoptError::config("optmain needs an input structure (xyz)"));
        }
        // Without -opt: by the order of the stationary point (MultiOptPy: FIRELARS).
        let mut methods = match p.get_list("opt_method") {
            [] => crate::opt::registry::default_methods(p.get_usize("saddle_order", 0)?),
            m => m.to_vec(),
        };
        if p.get_opt_str("follow_direction").is_some() {
            // A direction to follow takes the mode-following variant of RS-I-RFO.
            for m in &mut methods {
                if m.to_ascii_lowercase().starts_with("rsirfo_") {
                    *m = format!("mf_{m}");
                }
            }
        }
        let fc = p.get_i64("calc_exact_hess", -1)?;
        // A search on the gradient projection of two states with RS-I-RFO: the largest trust
        // radius is [`PROJECTION_MAX_TRUST_ANGSTROM`] unless `-tr` gives one.
        let on_projection = p
            .get_list("model_function")
            .first()
            .is_some_and(|name| is_gradient_projection(name))
            && methods
                .iter()
                .any(|m| m.to_ascii_lowercase().contains("rsirfo"));
        let max_tr = p.get_opt_f64("max_trust_radius")?.or_else(|| {
            on_projection.then_some(PROJECTION_MAX_TRUST_ANGSTROM * Units::ANGSTROM_TO_BOHR)
        });
        if p.get_bool("tight_convergence_criteria", false)
            && p.get_bool("loose_convergence_criteria", false)
        {
            return Err(MoptError::config(
                "-tcc (tight criteria) and -lcc (loose criteria) contradict each other; give one \
                 of them",
            ));
        }
        let settings = OptimizerSettings {
            methods,
            saddle_order: p.get_usize("saddle_order", 0)?,
            max_iterations: p.get_usize("NSTEP", 1000)?,
            criteria: ConvergenceCriteria::from_flags(
                p.get_bool("tight_convergence_criteria", false),
                p.get_bool("loose_convergence_criteria", false),
                !p.get_list("projection_constrain").is_empty(),
            ),
            trust: TrustSettings {
                max: max_tr.map(Quantity::<Bohr>::new),
                min: Quantity::new(p.get_f64("min_trust_radius", 0.01)?),
            },
            exact_hessian_interval: usize::try_from(fc).ok().filter(|&n| n > 0),
            coordinates: p.get_str("coordinate_system", "cart").to_string(),
            dissociation_distance: Some(Quantity::<Bohr>::new(
                first_f64(p, "dissociate_check", 10.0)? * Units::ANGSTROM_TO_BOHR,
            )),
            constraints: parse_constraints(p)?,
            model_hessian: p.get_opt_str("use_model_hessian").map(str::to_string),
            model_hessian_interval: p.get_usize("calc_model_hess", 50)?,
            mode_kill: p
                .get_bool("mode_kill", false)
                .then(ModeKillSettings::default),
            crossing_check: (!p.get_bool("no_crossing_check", false))
                .then(CrossingCheckSettings::default),
            projection_trust: match p
                .get_str("projection_trust", "energy")
                .to_ascii_lowercase()
                .as_str()
            {
                "energy" => false,
                "merit" => true,
                other => {
                    return Err(MoptError::config(format!(
                        "-projection_trust is energy or merit, not '{other}'"
                    )))
                }
            },
            step_factor: match p.get_str("DELTA", "x") {
                "x" => None,
                text => Some(text.trim().parse::<f64>().map_err(|_| {
                    MoptError::config(format!("-d expects a number, got '{text}'"))
                })?),
            },
            fragments: p
                .get_list("opt_fragment")
                .iter()
                .map(|group| atoms_of(group))
                .collect::<MoptResult<_>>()?,
            ..OptimizerSettings::default()
        };
        let hisd_mode_kill = match p.get_opt_str("mode_kill_method") {
            None => settings.saddle_order == 0,
            Some("classic") => false,
            Some("hisd") => true,
            Some(other) => {
                return Err(MoptError::config(format!(
                    "unknown -modekill_method '{other}'; choose classic or hisd"
                )))
            }
        };
        let threads = p.get_usize("N_THREAD", 8)?;
        let ccqn_bonds = parse_bonds(p.get_list("ccqn_reactive_bonds"))?;
        Ok(Self {
            inputs,
            settings_document: settings_json(p, OPTMAIN).to_string_pretty(),
            restart_dir: None,
            calculator: calculator_name(p)?,
            calculator_options: calculator_options(p)?,
            keep_full_path: false,
            hisd_mode_kill,
            state: state_spec(p)?,
            settings,
            model_function: p.get_list("model_function").to_vec(),
            excited_state: p.get_usize("excited_state", 0)?,
            initial_hessian: p.get_opt_str("initial_hessian").map(PathBuf::from),
            follow: p.get_opt_str("follow_direction").map(PathBuf::from),
            shape_conditions: ShapeCondition::parse_cli(p.get_list("shape_conditions"))?,
            geometry_info: match p.get_list("geom_info").first() {
                Some(list) => atoms_of(list)?,
                None => Vec::new(),
            },
            detect_negative_eigenvalues: p.get_bool("detect_negative_eigenvalues", false),
            debug: p.get_bool("debug", false),
            cmds: p.get_bool("cmds", false),
            pca: p.get_bool("pca", false),
            koopman: p.get_bool("koopman", false),
            bitss: super::bitss::parse_settings(p.get_list("bitss_options"))?,
            variable_cell: if p.get_bool("variable_cell", false) {
                Some(VariableCellSettings {
                    pressure_gpa: p.get_f64("variable_cell_pressure", 0.0)?,
                    weight: p.get_f64("variable_cell_weight", DEFAULT_LATTICE_WEIGHT)?,
                    forces: parse_cell_forces(p.get_list("variable_cell_afir"))?,
                })
            } else {
                None
            },
            threads: threads.max(1),
            no_symmetry: p.get_bool("no_symmetry_analyzer", false),
            frequency_analysis: p.get_bool("frequency_analysis", false),
            force_frequency_analysis: p.get_bool("force_frequency_analysis", false),
            frequency: frequency_settings(p)?,
            project_gradient: p.get_bool("project_gradient", false),
            ccqn_bonds,
            ccqn_product: p.get_opt_str("ccqn_product").map(PathBuf::from),
            ccqn_step: p.get_opt_f64("ccqn_step")?,
            biases: parse_biases(p)?,
            bias_options: super::bias::present_bias_options(p),
            bias_derivatives: p.get_bool("bias_derivatives", false),
            irc: super::irc::settings_from_args(p)?,
            stop_file: None,
            final_hessian: None,
        })
        .and_then(|c| c.with_ccqn(p))
    }

    fn with_ccqn(mut self, p: &ParsedArgs) -> MoptResult<Self> {
        let uses_ccqn = self
            .settings
            .methods
            .iter()
            .any(|m| m.to_ascii_lowercase().contains("ccqn"));
        if !uses_ccqn {
            return Ok(self);
        }
        let axis_count =
            usize::from(!self.ccqn_bonds.is_empty()) + usize::from(self.ccqn_product.is_some());
        if axis_count != 1 {
            return Err(MoptError::config(
                "CCQN needs exactly one axis: -ccqn_bonds or -ccqn_product",
            ));
        }
        // The axis is completed per input (the product needs the start structure).
        let mut c = CcqnSettings::new(CcqnAxis::ReactiveBonds(self.ccqn_bonds.clone()));
        if let Some(cphi) = p.get_opt_f64("ccqn_cos_phi")? {
            c.cos_phi = cphi;
        }
        if let Some(step) = self.ccqn_step {
            let step = step * Units::ANGSTROM_TO_BOHR;
            c.uphill_radius = step;
            c.trust.initial = step;
            c.trust.min = c.trust.min.min(step);
            c.trust.max = c.trust.max.max(step);
        }
        c.cone_solver = match p
            .get_str("ccqn_cone_solver", "exact")
            .to_ascii_lowercase()
            .as_str()
        {
            "exact" => ConeSolver::Exact,
            "alm" => ConeSolver::Alm,
            other => {
                return Err(MoptError::parse(format!(
                    "-ccqn_solver must be exact or alm, got '{other}'"
                )))
            }
        };
        if self.ccqn_product.is_some() && p.get_opt_f64("ccqn_cos_phi")?.is_none() {
            c.cos_phi = 0.5;
        }
        self.settings.ccqn = Some(c);
        Ok(self)
    }
}

/// `-temp`, `-press`, `-freq_cutoff`, `-freq_corr`, and `-nosymm` as settings of a vibrational
/// analysis (the multiplicity is filled in from the structure).
pub fn frequency_settings(p: &ParsedArgs) -> MoptResult<FrequencySettings> {
    let defaults = FrequencySettings::default();
    Ok(FrequencySettings {
        thermo: ThermoSettings {
            temperature: p.get_f64("temperature", 298.15)?,
            pressure: p.get_f64("pressure", 101_325.0)?,
            cutoff: p.get_f64("frequency_analysis_cutoff", 100.0)?,
            quasi_harmonic: QuasiHarmonic::parse(
                p.get_str("frequency_correction_method", "grimme"),
            )?,
            lowest: p.get_f64("frequency_lowest", defaults.thermo.lowest)?,
            enthalpy: match p.get_opt_str("frequency_correction_enthalpy") {
                Some(name) => Some(EnthalpyTreatment::parse(name)?),
                None => None,
            },
            enthalpy_cutoff: p.get_opt_f64("frequency_cutoff_enthalpy")?,
            concentration: p.get_opt_f64("concentration")?,
            frequency_scale: p.get_f64("frequency_scale", 1.0)?,
            zero_point_scale: p.get_opt_f64("zero_point_scale")?,
            invert_below: p.get_opt_f64("frequency_invert")?,
            rotor_moment: RotorMoment::parse(p.get_str("frequency_rotor_moment", "global"))?,
            ..defaults.thermo
        },
        symmetry_tolerance: if p.get_bool("no_symmetry_analyzer", false) {
            None
        } else {
            Some(p.get_f64("symmetry_tolerance", 0.01)?)
        },
        ..defaults
    })
}

/// The surface of `-es ROOT` and `-mf NAME [charge multiplicity]`; `None` for the ground
/// state of the calculator alone.
///
/// Without `-mf`: the electronic state `ROOT` (0: the ground state). With `-mf`: two states
/// of the whole structure combined by the model function `NAME`. The first is the state
/// `ROOT` of the structure as given. The second has the given charge and multiplicity in
/// its lowest state (a crossing of states of different spin), or, without them, is the next
/// state above the first with the same spin (a conical intersection of the states `ROOT`
/// and `ROOT + 1`).
pub fn state_surface(spec: &[String], root: usize) -> MoptResult<Option<PesExpr>> {
    let Some(name) = spec.first() else {
        return Ok((root > 0).then(|| PesExpr::Task(TaskSpec::whole("pes").excited_state(root))));
    };
    let state1 = TaskSpec::whole("State1").excited_state(root);
    let state2 = match &spec[1..] {
        [] => TaskSpec::whole("State2").excited_state(root + 1),
        [c, m] => TaskSpec::whole("State2").spin_charge(parse_pair(c, m, "-mf")?),
        // A third number: the electronic state of the second state (`ieipmain -mf` passes
        // the charge, multiplicity, and `-es` of its second structure).
        [c, m, state, ..] => TaskSpec::whole("State2")
            .spin_charge(parse_pair(c, m, "-mf")?)
            .excited_state(state.trim().parse::<usize>().map_err(|_| {
                MoptError::parse(format!("-mf: the state '{state}' is not a whole number"))
            })?),
        [_] => {
            return Err(MoptError::parse(
                "-mf takes a name and optionally a charge and multiplicity",
            ))
        }
    };
    if name.eq_ignore_ascii_case("bitss") {
        return Err(MoptError::unsupported(
            "-mf bitss is a search with two structures (optmain INPUT -mf bitss REFERENCE), not a surface of one structure",
        ));
    }
    Ok(Some(two_states_of(
        model_function(name, None, None)?,
        state1,
        state2,
    )?))
}

/// The largest trust radius (Angstrom) of an RS-I-RFO search on the gradient projection of two
/// states (`-mf opt_mesx`, `-mf opt_meci`) when `-tr` does not give one: the difference
/// gradient turns within a short distance next to a seam, and with it the projection, so that
/// the quadratic model of a step holds over a shorter distance than on one surface.
pub const PROJECTION_MAX_TRUST_ANGSTROM: f64 = 0.1;

/// Whether the model function `name` of `-mf` is a gradient projection of two states.
pub fn is_gradient_projection(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "opt_mesx" | "opt_mesx_2" | "mesx" | "opt_meci" | "meci"
    )
}

/// The lowest excitation energies of `sys` as the calculator tells them, written to
/// `excitations.txt` in `dir` and to the log; nothing if it does not tell them.
pub fn report_excitations(
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    dir: &Path,
    log: &Logger,
) -> MoptResult<()> {
    let Some(energies) = calc.excitation_energies(sys)? else {
        return Ok(());
    };
    let mut lines = vec![
        "Excited states at the starting structure".to_string(),
        format!(
            "{:>6} {:>14} {:>12} {:>14}",
            "state", "Hartree", "eV", "cm^-1"
        ),
    ];
    for (k, energy) in energies.iter().take(10).enumerate() {
        lines.push(format!(
            "{:>6} {:>14.8} {:>12.4} {:>14.1}",
            k + 1,
            energy,
            energy * Units::HARTREE_TO_EV,
            energy * Units::HARTREE_TO_WAVENUMBER
        ));
    }
    for line in &lines {
        log.info(line);
    }
    std::fs::write(dir.join("excitations.txt"), lines.join("\n") + "\n")?;
    Ok(())
}

/// The vibrational problem of `optmain -freq` at `sys` on the optimizer's surface (away from a
/// crossing of states): the Hessian whose modes are analysed and the directions left out with
/// the rigid-body motions, so that zero modes are never counted as imaginary.
pub(super) struct FrequencyProblem {
    /// Energy of the surface (Hartree).
    pub energy: Quantity<crate::domain::Hartree>,
    /// Cartesian gradient of the surface (Hartree / Bohr).
    pub gradient: faer::Col<f64>,
    /// The Hessian of the surface, or at a constrained point that of the Lagrangian.
    pub hessian: faer::Mat<f64>,
    /// The constraint normals (and the rigid motions that keep the constraints) to project.
    pub normals: Vec<Direction>,
    /// Whether the optimizer holds constraints (the modes are those on their surface).
    pub constrained: bool,
    /// The number of constraints the optimizer holds (`note_constraint_surface`).
    pub held: usize,
}

/// [`FrequencyProblem`] of `sys` from the evaluations `results` of the optimizer's Hessian
/// tasks: the one place where `optmain -freq` and the imaginary-mode checks of `confsearch`
/// take the Hessian and the zero modes to project.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub(super) fn frequency_problem(
    opt: &mut GeometryOptimizer,
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    results: Vec<crate::domain::EvalOutput>,
) -> MoptResult<FrequencyProblem> {
    let (energy, gradient, hessian) = opt.surface_hessian(sys, results)?;
    let held = opt
        .constraint_set()
        .map_or(0, crate::pes::constraint::ConstraintSet::len);
    // At a constrained stationary point: the vibrations on the constraint surface.
    let problem = match opt.constraint_set().filter(|c| !c.is_empty()) {
        Some(set) => {
            let invariant = super::freqmain::surface_invariant(opt, calc)?;
            let (lagrangian, normals) =
                super::freqmain::constraint_surface(set, sys, &gradient, &hessian, invariant)?;
            FrequencyProblem {
                energy,
                gradient,
                hessian: lagrangian,
                normals,
                constrained: true,
                held,
            }
        }
        None => FrequencyProblem {
            energy,
            gradient,
            hessian,
            normals: Vec::new(),
            constrained: false,
            held,
        },
    };
    Ok(problem)
}

/// Vibrational analysis of `sys` with the Hessian of the optimizer's surface: writes
/// `thermochemistry.txt`, `normal_modes.txt`, and `vibration_animation/` into `dir` and the
/// reports into the log.
pub fn write_frequency_analysis(
    opt: &mut GeometryOptimizer,
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    settings: &FrequencySettings,
    project_gradient: bool,
    dir: &Path,
    log: &Logger,
) -> MoptResult<Option<Vec<f64>>> {
    log.info("Performing vibrational analysis...");
    let tasks = opt.hessian_tasks(sys)?;
    let results = CalculatorSet::single(calc).evaluate(&tasks)?;
    let mut settings = *settings;
    settings.project_rigid = opt.removes_rigid_body_motion();
    // At a crossing of two states the vibrations are those within the seam.
    if let Some(crossing) = opt.crossing(sys, results.clone())? {
        let probed =
            super::freqmain::probed_plane(opt, calc, sys, &crossing, settings.project_rigid)?;
        super::freqmain::write_seam_reports(
            sys,
            &crossing,
            probed.as_deref(),
            &settings,
            dir,
            log,
        )?;
        return Ok(None);
    }
    let problem = frequency_problem(opt, calc, sys, results)?;
    let mut directions = if project_gradient {
        vec![Direction::Gradient(problem.gradient)]
    } else {
        Vec::new()
    };
    if problem.constrained {
        super::freqmain::announce_constraint_surface(&problem.normals, settings.project_rigid, log);
    }
    directions.extend(problem.normals);
    let (energy, hessian, held) = (problem.energy, problem.hessian, problem.held);
    let (_, wavenumbers) = super::freqmain::write_reports(
        sys,
        energy.value(),
        &hessian,
        &settings,
        &directions,
        dir,
        log,
    )?;
    if held > 0 {
        super::freqmain::note_constraint_surface(dir, held)?;
    }
    Ok(Some(wavenumbers))
}

/// Atoms of a list such as `1,2,5-7` (numbers starting at 1).
fn atoms_of(list: &str) -> MoptResult<Vec<AtomId>> {
    parse_atom_indices_one_based(list)?
        .into_iter()
        .map(AtomId::from_one_based)
        .collect()
}

/// Built-in calculator `name` (`lj`, `emt`, `tersoff`, `muller_brown`, ...) for `sys`.
pub fn native_calculator(name: &str, sys: &MolecularSystem) -> MoptResult<Box<dyn Calculator>> {
    native_calculator_with(name, sys, &[])
}

/// Built-in calculator `name` for `sys` with its settings (the `key=value` pairs of
/// `-calc_opt`, such as the cutoff of `lj` or the reference of a pair of `meam`).
pub fn native_calculator_with(
    name: &str,
    sys: &MolecularSystem,
    settings: &[(String, String)],
) -> MoptResult<Box<dyn Calculator>> {
    let options = CalculatorOptions {
        extra: settings.to_vec(),
        ..CalculatorOptions::default()
    };
    CalculatorRegistry::native().create(name, &options, sys)
}

/// MultiOptPy's run directory `<date>/<stem>_OPT_<suffix>_<timestamp>/` under `base`
/// (`suffix` is `ASE` whenever `-os` is given, as in MultiOptPy).
pub fn run_directory(base: &Path, input: &Path, now: &DateTime) -> PathBuf {
    let stem = input.with_extension("");
    let stem = stem.to_string_lossy();
    base.join(now.date_dir())
        .join(format!("{stem}_OPT_ASE_{}", now.timestamp()))
}

/// Result of one input.
#[derive(Debug, Clone)]
pub struct OptmainResult {
    /// Run directory.
    pub directory: PathBuf,
    /// `<stem>_optimized.xyz` (the optimized variables file for plain variables).
    pub optimized: PathBuf,
    /// Whether the optimization converged.
    pub converged: bool,
    /// Whether the run was interrupted (a `STOP` file); `--restart` continues it.
    pub interrupted: bool,
    /// What the job ended with (`None` for plain variables).
    pub summary: Option<JobSummary>,
}

/// The end of one job of a run over several inputs (MultiOptPy's snapshot in
/// `Optimize.all_results`).
#[derive(Debug, Clone)]
pub struct JobSummary {
    /// The input file.
    pub input: PathBuf,
    /// `<stem>_traj.xyz`.
    pub trajectory: PathBuf,
    /// The final structure.
    pub system: MolecularSystem,
    /// Its energy without bias potentials.
    pub energy: Quantity<crate::domain::Hartree>,
    /// Its energy with them.
    pub biased_energy: Quantity<crate::domain::Hartree>,
    /// Point group of the final structure (converged runs).
    pub symmetry: Option<String>,
    /// Iterations taken.
    pub iterations: usize,
    /// The files of the end points of the IRC (`-irc`; MultiOptPy's
    /// `irc_terminal_struct_paths`).
    pub irc_endpoints: Vec<PathBuf>,
    /// The number of imaginary frequencies of the final structure, when `-freq` analysed it
    /// (by the one criterion of the crate, `vib::imaginary_wavenumbers`).
    pub imaginary_modes: Option<usize>,
}

/// Run `optmain` for every input, writing run directories under `base`. Progress goes to the
/// console up to `console` (nothing when `None`) and to `mopt.log` in each run directory.
pub fn run(
    config: &OptmainConfig,
    base: &Path,
    console: Option<Level>,
) -> MoptResult<Vec<OptmainResult>> {
    run_with(config, base, console, &CalculatorRegistry::native())
}

/// [`run`] with the calculators of `registry`.
pub fn run_with(
    config: &OptmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<Vec<OptmainResult>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.calculator_options.pool_threads(config.threads))
        .build()
        .map_err(|e| MoptError::config(format!("thread pool: {e}")))?;
    pool.install(|| {
        config
            .inputs
            .iter()
            .map(|input| run_one(config, input, base, console, registry).context(input.display()))
            .collect()
    })
}

fn run_one(
    config: &OptmainConfig,
    input: &Path,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<OptmainResult> {
    let objectives = ObjectiveRegistry::builtin();
    if objectives.contains(&config.calculator) {
        if config.initial_hessian.is_some() {
            return Err(MoptError::unsupported(
                "-ih is for structures; an objective of plain variables has its own second \
                 derivatives (-fc)",
            ));
        }
        return super::variables::run_one(config, input, base, console, &objectives);
    }
    if config.initial_hessian.is_some()
        && (!config.model_function.is_empty() || config.variable_cell.is_some())
    {
        return Err(MoptError::unsupported(
            "-ih gives the Hessian of one state in a fixed cell; it is not available with -mf \
             or -vc",
        ));
    }
    if config.irc.is_some() {
        // A projection for crossings has no path of steepest descent: refused before the
        // optimization.
        super::irc::irc_states(&config.model_function, config.excited_state)?;
    }
    if config.bias_derivatives && config.variable_cell.is_some() {
        return Err(MoptError::unsupported(
            "-bias_derivatives works in a fixed cell; it is not available with -vc",
        ));
    }
    let spin = config.state.of_file(input)?;
    let start = read_xyz_file(input, spin, None)?;
    // The constraints resolved once, at the input: every stage of the run (the optimization
    // and a restart of it, the IRC, the vibrational analysis, the derivatives in the bias
    // constants) holds these values, not those of its own starting structure. Eigenvector
    // constraints resolve at the first Hessian of the optimization, which hands them on
    // (`GeometryOptimizer::resolved_constraints`).
    let input_constraints =
        super::constraints::resolve_for_run(&config.settings.constraints, &start)
            .context("constraints at the input")?;
    let evaluations_before = registry.counts();
    // The bias potentials of the whole system (atom numbers of the input).
    let biases = super::oniom_micro::build_terms(&config.biases, &start)?;
    // ONIOM microiterations: the optimizer moves the core, the environment is relaxed; the
    // constraints and the bias potentials on the core are taken to it.
    let wrapped = super::oniom_micro::wrap_constrained(
        &config.calculator_options,
        super::oniom_micro::uncovered_by_optmain(config),
        &input_constraints,
        &biases,
        &super::oniom_micro::other_atoms_of_optmain(config),
        super::oniom_micro::forces_of(&config.settings.criteria),
        start,
        &config.calculator,
        registry,
    )?;

    // The other settings with atom numbers, in the numbering of the structures the run
    // moves: the fragments of -gfix, the reactive bonds and the product of CCQN.
    let fragments = config
        .settings
        .fragments
        .iter()
        .map(|group| group.iter().map(|a| wrapped.on_core(*a)).collect())
        .collect::<MoptResult<Vec<Vec<AtomId>>>>()?;
    let ccqn_bonds = config
        .ccqn_bonds
        .iter()
        .map(|(a, b)| Ok((wrapped.on_core(*a)?, wrapped.on_core(*b)?)))
        .collect::<MoptResult<Vec<(AtomId, AtomId)>>>()?;
    let ccqn_product = match &config.ccqn_product {
        Some(product) => Some(wrapped.structure(read_xyz_file(product, spin, None)?)?),
        None => None,
    };
    let log_atoms = config
        .calculator_options
        .oniom
        .as_ref()
        .map(|spec| {
            spec.high_atoms()
                .into_iter()
                .map(|atom| wrapped.on_core(atom))
                .collect::<MoptResult<Vec<_>>>()
        })
        .transpose()?;
    let trajectory_atoms = wrapped.trajectory_atoms(&config.calculator_options)?;
    let (start, calc, micro, micro_note, constraints, biases) = (
        wrapped.start,
        wrapped.calc,
        wrapped.micro,
        wrapped.note,
        wrapped.constraints,
        wrapped.terms,
    );
    if config.bias_derivatives && micro.is_some() {
        return Err(MoptError::unsupported(
            "-bias_derivatives needs the Hessian of the whole surface; with ONIOM \
             microiterations the optimizer moves the core alone (not available yet)",
        ));
    }
    let calc = crate::pes::o1numhess::wrap_if_requested(
        calc,
        config.settings.model_hessian.as_deref(),
        config.settings.exact_hessian_interval.is_some(),
    );
    let calc = super::oniom_micro::first_hessian(
        calc,
        config.initial_hessian.as_deref(),
        &start,
        micro.as_deref(),
    )?;
    if calc.uses_electrons() {
        start.spin_charge.check_electrons(&start.elements)?;
    }
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => run_directory(base, input, &DateTime::now_utc()),
    };
    std::fs::create_dir_all(&dir)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let calc_journaled = journal.wrap("main", calc.as_ref());
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "********************************\n{}\n********************************",
        input.display()
    ));
    log.info(&format!(
        "mopt-rs {} optmain: {}",
        crate::VERSION,
        input.display()
    ));
    // MultiOptPy's lines for its own potentials; the name for every other calculator.
    log.info(&match calc.name() {
        "lj" => "Use Lennard-Jones cluster potential.".to_string(),
        "emt" => "Use EMT potential.".to_string(),
        "tersoff" => "Use Tersoff potential.".to_string(),
        other => format!("Use {other}"),
    });
    log.info(&format!("Calculator: {}", calc.description()));
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&config.calculator_options.software_path_report());
    if let Some(note) = config.state.file_note(input) {
        log.info(&note);
    }
    if let Some(note) = &micro_note {
        log.info(note);
    }
    super::oniom_micro::report_relaxations(micro.as_deref(), &log);
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document),
    )?;

    if config
        .model_function
        .first()
        .is_some_and(|name| name.eq_ignore_ascii_case("bitss"))
    {
        let reference = config.model_function.get(1).ok_or_else(|| {
            MoptError::config("BITSS requires a reference geometry file path: -mf bitss FILE")
        })?;
        if config.variable_cell.is_some() {
            return Err(MoptError::unsupported(
                "-mf bitss with a variable cell (-vc) is not available",
            ));
        }
        let terms = biases.clone();
        return super::bitss::run(
            config,
            Path::new(reference),
            &terms,
            input,
            &start,
            &calc_journaled,
            &dir,
            &log,
        );
    }
    if let Some(cell) = &config.variable_cell {
        let terms = biases.clone();
        return super::variable_cell::run(
            config,
            cell,
            &terms,
            &constraints,
            input,
            &start,
            &calc_journaled,
            &dir,
            &log,
        );
    }
    let mut settings = config.settings.clone();
    settings.constraints = constraints;
    settings.fragments = fragments;
    settings.rigid_body_invariant = crate::pes::term::frame_invariant(
        calc.is_rigid_body_invariant(),
        biases.iter().map(|(_, t)| t.as_ref()),
    );
    if config.initial_hessian.is_some() && settings.exact_hessian_interval.is_none() {
        // The optimizer asks the calculator for a Hessian at its first point only, and
        // the calculator answers with the one of the file.
        settings.exact_hessian_interval = Some(HESSIAN_AT_THE_START_ONLY);
    }
    if let Some(c) = settings.ccqn.as_mut() {
        match &ccqn_product {
            Some(product) => c.axis = CcqnAxis::linear_midpoint(&start, product)?,
            None => c.axis = CcqnAxis::ReactiveBonds(ccqn_bonds),
        }
    }
    // Saddle searches on the relaxed surface of microiterations end at a verified point.
    if let Some(note) = super::oniom_micro::verified_saddle_points(&mut settings, micro.as_deref())
    {
        log.info(&note);
    }
    let mut opt = GeometryOptimizer::new(settings.clone(), start.clone())?;
    if config.hisd_mode_kill {
        opt.enable_hisd_mode_kill();
    }
    if let Some(file) = &config.follow {
        let direction = super::follow::read_direction(file, &start, micro.as_deref())?;
        opt.follow_directions(&[direction])?;
        log.info(&format!(
            "Mode following from the direction of {}.",
            file.display()
        ));
    }
    // The states first, then the bias potentials on top of them.
    if let Some(surface) = state_surface(&config.model_function, config.excited_state)? {
        if let Some(name) = config.model_function.first() {
            log.info(&format!("Mode: Model Function Optimization ({name})"));
        }
        if config.excited_state > 0 {
            log.info(&format!("Electronic state: {}", config.excited_state));
        }
        opt.set_surface(surface)?;
        report_excitations(calc.as_ref(), &start, &dir, &log)?;
    }
    for (label, term) in &biases {
        opt.add_term(label, Arc::clone(term))?;
    }

    let stem = input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    let mut progress = ProgressLog::new(log.clone(), settings.criteria);
    progress.select_geometry_atoms(log_atoms);
    log.info(&format!("{}\n", progress.geometry(&start)?));
    let mut files = RunFiles::new(&dir, &stem)?;
    if let Some(atoms) = trajectory_atoms {
        files.select_trajectory_atoms(atoms, config.keep_full_path)?;
    }
    let mut shape = ShapeGuard::new(config.shape_conditions.clone(), log.clone());
    let mut stop = StopFile::new(&dir, log.clone()).also_watching(config.stop_file.clone());
    let mut info = GeometryInfo::new(config.geometry_info.clone(), &dir);
    // Without -negeigval the guard watches a search of order 0, which it never stops.
    let watched = if config.detect_negative_eigenvalues {
        settings.saddle_order
    } else {
        0
    };
    let mut curvature = NegativeEigenvalueGuard::new(watched, &dir, log.clone());
    let mut maps = PathMaps::new(config.cmds, config.pca, &dir, log.clone());
    let mut bias_iterations = if config.bias_derivatives && !config.bias_options.is_empty() {
        Some(super::bias_constants::IterationDerivatives::new(
            &config.bias_options,
            &start,
            &dir,
        )?)
    } else {
        None
    };
    let mut modes = if config.koopman {
        Some(KoopmanLog::new(&dir, log.clone())?)
    } else {
        None
    };
    let outcome = {
        let mut observers: Vec<&mut dyn Observer> = vec![
            &mut progress,
            &mut files,
            &mut shape,
            &mut stop,
            &mut info,
            &mut curvature,
        ];
        if config.cmds || config.pca {
            observers.push(&mut maps);
        }
        if let Some(modes) = modes.as_mut() {
            observers.push(modes);
        }
        if let Some(d) = bias_iterations.as_mut() {
            observers.push(d);
        }
        run_optimization(&mut opt, &calc_journaled, &mut observers)?
    };
    let (replayed, unused) = journal.replay_counts();
    if replayed > 0 {
        log.info(&format!(
            "Restart: {replayed} evaluations taken from {}{}.",
            JOURNAL_FILE,
            if unused > 0 {
                format!(" ({unused} recorded ones no longer matched and were recomputed)")
            } else {
                String::new()
            }
        ));
    }
    if stop.stopped() {
        log.info(&format!(
            "Continue with: optmain --restart {}",
            dir.display()
        ));
    }
    let converged = outcome.termination == Some(crate::opt::state_machine::Termination::Converged);
    if let (Some(path), true) = (&config.final_hessian, opt.is_cartesian()) {
        crate::io::npy::write_npy_matrix(path, opt.hessian())?;
    }
    // The last evaluated structure, whose energies the outcome has (not a step the optimizer
    // proposed after it).
    let last = outcome.system.clone();
    let mut imaginary_modes = None;
    if (config.frequency_analysis && converged) || config.force_frequency_analysis {
        let analysis = write_frequency_analysis(
            &mut opt,
            &calc_journaled,
            &last,
            &config.frequency,
            config.project_gradient,
            &dir,
            &log,
        )?;
        // Molecules and periodic cells (the modes at the zone centre) alike.
        imaginary_modes = analysis.map(|w| crate::vib::imaginary_wavenumbers(&w));
        if !converged {
            log.warn(
                "Vibrational analysis was performed, but the optimization did not converge. \
                 The result of thermochemistry is unreliable.",
            );
        }
    }
    if config.bias_derivatives {
        if converged {
            let resolved = opt.resolved_constraints().to_vec();
            super::bias_constants::write_bias_derivatives(
                &mut opt,
                &calc_journaled,
                &last,
                &start,
                outcome.iterations,
                &config.bias_options,
                &resolved,
                &dir,
                &log,
            )?;
        } else {
            log.warn(
                "-bias_derivatives: the optimization did not converge; the derivatives are \
                 those of a stationary point and were not computed.",
            );
        }
    }
    // MultiOptPy: the IRC from the last structure of the optimization.
    let mut irc_run = None;
    if let (Some(irc), false) = (&config.irc, stop.stopped()) {
        // The constraints of the optimization, resolved at the input (eigenvector
        // constraints at its first Hessian), are kept along the path.
        let resolved = opt.resolved_constraints();
        let constraints = if resolved.is_empty() {
            None
        } else {
            Some(crate::pes::constraint::ConstraintSet::new(
                resolved, &last, None,
            )?)
        };
        let terms = biases.clone();
        // A structure the optimization converged to is a stationary point for the IRC.
        let irc = if converged {
            irc.clone().accepting_converged(
                settings.criteria.max_force.value(),
                settings.criteria.rms_force.value(),
            )
        } else {
            irc.clone()
        };
        let parts = super::irc::IrcSurfaceParts {
            terms,
            excited_state: config.excited_state,
            states: super::irc::irc_states(&config.model_function, config.excited_state)?,
            constraints,
            // The surface on which the structure was found: a ramp is not started over.
            iteration: opt.last_surface_iteration().unwrap_or(0),
        };
        let run = super::irc::run_structure(
            &irc,
            &calc_journaled,
            parts,
            &last,
            &stem,
            &dir,
            &log,
            &mut [],
        )?;
        if run.interrupted {
            log.info(&format!(
                "Continue with: optmain --restart {}",
                dir.display()
            ));
        }
        irc_run = Some(run);
    }
    let mut point_group = None;
    if converged {
        // MultiOptPy: `symmetry.txt` with the point group of the final structure.
        // Of the whole system (with microiterations `last` is the core).
        let symmetry = match config.frequency.symmetry_tolerance {
            Some(tolerance) => {
                let whole = match &micro {
                    Some(relaxed) => relaxed.full_system(&last)?,
                    None => last.clone(),
                };
                vib::point_group(&whole, tolerance).label
            }
            None => "C1".to_string(),
        };
        point_group = Some(symmetry.clone());
        std::fs::write(
            dir.join("symmetry.txt"),
            format!("Symmetry of final structure: {symmetry}"),
        )?;
        log.info(&format!("Symmetry: {symmetry}"));
    }
    // With microiterations the files and the result hold the whole system.
    let last = match &micro {
        Some(relaxed) => {
            super::oniom_micro::finish(relaxed, Some(&last), &dir, &log)?;
            relaxed.full_system(&last)?
        }
        None => last,
    };
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.info(&format!(
        "Trial of geometry optimization ({}) was completed.",
        input.display()
    ));
    if config.inputs.last().is_some_and(|last| last == input) {
        log.info("All calculations were completed.");
    }
    log.info(&format!(
        "Optimized structure file path: {}",
        files.optimized_path().display()
    ));
    log.info(&format!(
        "Trajectory file path: {}",
        files.trajectory_path().display()
    ));
    log.flush();
    Ok(OptmainResult {
        directory: dir,
        optimized: files.optimized_path(),
        converged: outcome.termination == Some(crate::opt::state_machine::Termination::Converged),
        interrupted: stop.stopped() || irc_run.as_ref().is_some_and(|r| r.interrupted),
        summary: Some(JobSummary {
            input: input.to_path_buf(),
            trajectory: files.result_path(),
            system: last,
            energy: outcome.energy,
            biased_energy: outcome.biased_energy,
            symmetry: point_group,
            iterations: outcome.iterations,
            irc_endpoints: irc_run.map(|r| r.endpoints).unwrap_or_default(),
            imaginary_modes,
        }),
    })
}

/// Interval of the Hessians of the calculator with which only the first point gets one
/// (`-ih` without `-fc`): more iterations than any run has.
pub const HESSIAN_AT_THE_START_ONLY: usize = 1_000_000_000;

/// Thin wrapper used by tests and embedding code: settings for `args`.
pub fn config_from_cli(args: &[&str]) -> MoptResult<OptmainConfig> {
    let p = super::option::parser("optmain", OPTMAIN).parse(args)?;
    OptmainConfig::from_args(&p)
}
