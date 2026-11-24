//! `freqmain`: vibrational analysis and thermochemistry of given structures, without an
//! optimization.
//!
//! For every input structure the Hessian of the surface (calculator and bias potentials) is
//! evaluated there, and the normal modes, the point group, and the thermodynamic functions
//! are written as by `optmain -freq`: `normal_modes.txt`, `thermochemistry.txt`,
//! `vibration_animation/`, `symmetry.txt`, in a run directory
//! `<date>/<stem>_FREQ_<timestamp>/`. The command takes the options of `optmain` that
//! describe the surface (`-os`, `-elec`, `-spin`, bias potentials, `-modelhess o1numhess`
//! for the Hessian, ...) and those of the analysis (`-temp`, `-press`, `-freq_cutoff`,
//! `-freq_corr`, `-freq_lowest`, `-symm_tol`, `-nosymm`, `-project_gradient`); options of the
//! optimization itself are ignored.
//!
//! `-temp_scan START,END,STEP` adds the thermodynamic functions over a range of
//! temperatures (`thermochemistry_scan.csv`), `-spc_energy` replaces the electronic energies
//! by those of another level of theory, and `-boltz` writes the relative Gibbs free energies
//! and Boltzmann populations of several inputs.
//!
//! With `-anharmonic` the cubic and quartic force constants of a minimum are computed from
//! the Hessians at structures displaced along every normal mode, and the fundamentals and
//! the zero-point energy follow from second-order perturbation theory and from the
//! vibrational self-consistent field: `anharmonic.txt` and `force_field.txt`.
//!
//! With `-mf opt_mesx` or `-mf opt_meci` (as in `optmain`) the structure is a crossing point
//! of two states, and the vibrations within the crossing seam are analysed instead:
//! `seam.txt`, `normal_modes.txt`, `vibration_animation/`.
//!
//! With `-phonon_supercell N1 N2 N3` the Hessian of that supercell of a periodic structure
//! gives the force constants and from them the phonons at any wave vector: the dispersion
//! along `-phonon_path` (`phonon_dispersion.csv`), and on `-phonon_mesh` the density of
//! states (`phonon_dos.csv`) and the thermodynamic functions of the harmonic crystal
//! (`phonon_thermochemistry.txt`).
//!
//! With `-local_modes` the local vibrational modes of internal coordinates are written to
//! `local_modes.txt` (relaxed force constant, local mode mass and wavenumber of every
//! coordinate) and `local_mode_composition.csv` (the share of every normal mode in each).
//!
//! With `-partial_atoms` the vibrations of the chosen atoms alone are analysed as well,
//! from the block of their Hessian or from the effective Hessian with the other atoms
//! relaxed (`-partial_hessian block|effective|frozen`); the results go to
//! `partial_hessian/` in the run directory: `hessian.txt` (the Hessian of the chosen atoms,
//! Hartree/Bohr^2), `normal_modes.txt`, `vibration_animation/`, and `summary.txt`.
//!
//! With `-path` the frames of the input are the points of a reaction path, given in a
//! common orientation as an intrinsic reaction coordinate gives them. At every point the
//! tangent of the path is projected out: the gradient (the line through the neighbouring
//! points where it vanishes), or with `-path_tangent points` that line everywhere, for a
//! path that is not a steepest-descent path. These files are written:
//!
//! - `path_profile.csv`: arc length, energy, RMS gradient and overlap of the gradient with
//!   the line through the neighbouring points, zero-point energy of the transverse modes,
//!   vibrationally adiabatic ground-state potential, curvature of the path, effective mass
//!   of small-curvature tunneling, the lowest transverse wavenumber with the overlap of its
//!   mode with that of the point before, and all transverse wavenumbers;
//! - `bifurcations.txt`: points where the lowest transverse curvature turns negative
//!   (valley-ridge transitions, at which the path bifurcates);
//! - `curvature_couplings.csv` and `mode_couplings.csv`: the couplings `B_kF` of the path
//!   curvature to the transverse modes and `B_kl` between the modes followed along the path
//!   (reaction-path Hamiltonian);
//! - `urva_direction.csv`, `urva_curvature.csv`, `reaction_phases.txt`: components of the
//!   path direction and curvature along internal coordinates, and the reaction phases
//!   between the minima of the curvature (unified reaction valley approach);
//! - `free_energy_profile.csv` and `variational_tst.txt`: free energies of the generalized
//!   transition states at `-temp`, the variational transition state, and the transmission
//!   coefficients of zero- and small-curvature tunneling.
//!
//! In a periodic cell the modes are those of the cell at the zone centre (translations are
//! projected out, rotations are not); the thermodynamic functions of a molecule do not apply
//! and are left out.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy analyses vibrations only at the
//!  end of an optimization (`optimization.py:Optimize._perform_vibrational_analysis`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use faer::{Col, Mat};

use super::bias::{parse_biases, BiasSpec};
use super::calculator::{calculator_name, calculator_options};
use super::charge_spin::{state_spec, StateSpec};
use super::optmain::frequency_settings;
use super::orbital_path::{OrbitalPath, OrbitalPathSettings};
use super::settings::settings_json;
use super::spec::FREQMAIN;
use crate::coord::internal::{Primitive, PrimitiveOptions, PrimitiveSet};
use crate::core_infra::arg_parser::{parse_atom_list, ParsedArgs};
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::domain::EvalRequest;
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use crate::math::linalg::{col_dot, col_norm, col_rms};
use crate::math::projection::{build_tr_rot_basis, project_out_basis_vec};
use crate::math::stable::normalized_exp;
use crate::opt::crossing_check::{self, CrossingCheckSettings};
use crate::opt::state_machine::{GeometryOptimizer, OptimizerSettings};
use crate::pes::calculator::Calculator;
use crate::pes::combine::{CrossingStates, EvalTask, DEFAULT_CALCULATOR};
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};
use crate::vib::phonon::{self, ForceConstants, PhononThermo};
use crate::vib::{
    self, Direction, ForceField, FreeEnergyProfile, FrequencySettings, InternalComponents,
    LocalModes, NormalModes, PartialHessian, PartialKind, PathPoint, PathProfile, QffSettings,
    ReactionPhase, SeamAnalysis, SecondDirection, Tunneling, VibrationalAnalysis, Vpt2,
    Vpt2Settings, Vscf, VscfSettings,
};
use crate::workflow::driver::CalculatorSet;
use crate::workflow::journal::RunJournal;
use crate::workflow::narrated::Narrated;
use crate::workflow::orbital_analysis::DEFAULT_ORBITAL_STEP;

/// An RMS gradient above this (Hartree/Bohr) is reported: the structure is not a stationary
/// point (MultiOptPy's normal criterion on the RMS force).
const STATIONARY_RMS_GRADIENT: f64 = 3e-4;

/// Where the direction of a path at its points comes from (`-path_tangent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathTangent {
    /// The gradient: a steepest-descent path. The points are used where it vanishes.
    Gradient,
    /// The line through the neighbouring points: any path.
    Points,
}

/// Settings of `freqmain`.
#[derive(Debug, Clone)]
pub struct FreqmainConfig {
    /// Input structures (xyz).
    pub inputs: Vec<PathBuf>,
    /// The settings as given, for `settings.json`.
    pub settings_document: String,
    /// `--restart`: the run directory to continue.
    pub restart_dir: Option<PathBuf>,
    /// `-os`.
    pub calculator: String,
    /// Calculator settings.
    pub calculator_options: CalculatorOptions,
    /// `-elec` / `-spin`.
    pub state: StateSpec,
    /// `-temp`, `-press`, `-freq_cutoff`, `-freq_corr`, `-freq_lowest`, `-symm_tol`, `-nosymm`.
    pub frequency: FrequencySettings,
    /// Bias potentials.
    pub biases: Vec<BiasSpec>,
    /// `-mf NAME [charge multiplicity]`: two states combined by a model function.
    pub model_function: Vec<String>,
    /// `-es`: the electronic state (0: the ground state).
    pub excited_state: usize,
    /// `-ih FILE`: the first Hessian that is asked for is read from this file
    /// (`io::hessian_file`).
    pub initial_hessian: Option<PathBuf>,
    /// `-modelhess o1numhess`: the Hessian from O1NumHess.
    pub model_hessian: Option<String>,
    /// `-project_gradient`.
    pub project_gradient: bool,
    /// `-temp_scan`: first and last temperature and the step (K).
    pub temperature_scan: Option<(f64, f64, f64)>,
    /// `-spc_energy`: electronic energies (Hartree) for the thermochemistry, one per input.
    pub single_point_energies: Vec<f64>,
    /// `-boltz`: Boltzmann populations of the inputs.
    pub boltzmann: bool,
    /// `-anharmonic`: quartic force field, VPT2, and VSCF.
    pub anharmonic: bool,
    /// `-qff_step`: displacement in the dimensionless normal coordinates.
    pub qff_step: f64,
    /// `-fermi_gap`, `-fermi_threshold`.
    pub vpt2: Vpt2Settings,
    /// `-vscf_basis`.
    pub vscf: VscfSettings,
    /// `-phonon_supercell`: cells of the supercell along the three lattice vectors.
    pub phonon_supercell: Option<[usize; 3]>,
    /// `-phonon_path`: corners of the path of wave vectors (reciprocal lattice units).
    pub phonon_path: Vec<[f64; 3]>,
    /// `-phonon_points`: steps along every line of the path.
    pub phonon_points: usize,
    /// `-phonon_mesh`: mesh of wave vectors.
    pub phonon_mesh: Option<[usize; 3]>,
    /// `-phonon_dos_width`: width of the Gaussians of the density of states (cm^-1).
    pub phonon_dos_width: f64,
    /// `-local_modes`: the internal coordinates whose local modes are wanted, as given.
    pub local_modes: Vec<String>,
    /// `-partial_atoms`: atoms (starting at 0) whose vibrations are analysed alone.
    pub partial_atoms: Vec<usize>,
    /// `-partial_hessian`: how their Hessian is taken.
    pub partial_kind: PartialKind,
    /// `-partial_free`: relative curvature below which a direction of the other atoms has
    /// no restoring force.
    pub partial_free_direction: f64,
    /// `-path`: the frames of the input are the points of a reaction path.
    pub path: bool,
    /// `-path_tangent`: where the direction of the path comes from.
    pub path_tangent: PathTangent,
    /// `-phase_prominence`: how deep a minimum of the path curvature must be to divide two
    /// reaction phases, as a fraction of the largest curvature.
    pub phase_prominence: f64,
    /// `-nro`, `-roet`, `-orbital_step`: analyses of the orbitals along the frames.
    pub orbitals: OrbitalPathSettings,
    /// `-fix`, `-pc`, `-gfix`, `-uc`: the structure is a stationary point within these
    /// constraints, and the vibrations are those on their surface (resolved at the input).
    pub constraints: Vec<crate::pes::constraint::Constraint>,
}

impl FreqmainConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        let inputs: Vec<PathBuf> = p.positionals.iter().map(PathBuf::from).collect();
        if inputs.is_empty() {
            return Err(MoptError::config("freqmain needs an input structure (xyz)"));
        }
        let mut frequency = frequency_settings(p)?;
        frequency.thermo.masses = vib::MassChoice::parse(p.get_str("freq_masses", "standard"))?;
        let config = Self {
            inputs,
            settings_document: settings_json(p, FREQMAIN).to_string_pretty(),
            restart_dir: None,
            calculator: calculator_name(p)?,
            calculator_options: calculator_options(p)?,
            state: state_spec(p)?,
            frequency,
            biases: parse_biases(p)?,
            model_function: p.get_list("model_function").to_vec(),
            excited_state: p.get_usize("excited_state", 0)?,
            initial_hessian: p.get_opt_str("initial_hessian").map(PathBuf::from),
            model_hessian: p.get_opt_str("use_model_hessian").map(str::to_string),
            project_gradient: p.get_bool("project_gradient", false),
            temperature_scan: match p.get_opt_str("temperature_scan") {
                Some(text) => Some(temperature_scan(text)?),
                None => None,
            },
            single_point_energies: p
                .get_list("single_point_energy")
                .iter()
                .map(|v| {
                    v.parse::<f64>().map_err(|_| {
                        MoptError::config(format!("-spc_energy: '{v}' is not a number"))
                    })
                })
                .collect::<MoptResult<Vec<f64>>>()?,
            boltzmann: p.get_bool("boltzmann", false),
            anharmonic: p.get_bool("anharmonic", false),
            qff_step: p.get_f64("qff_step", vib::DEFAULT_STEP)?,
            vpt2: Vpt2Settings {
                fermi_gap: p.get_f64("fermi_gap", 200.0)?,
                fermi_threshold: p.get_f64("fermi_threshold", 1.0)?,
                ..Vpt2Settings::default()
            },
            vscf: VscfSettings {
                basis: p.get_usize("vscf_basis", vib::DEFAULT_BASIS)?,
                ..VscfSettings::default()
            },
            phonon_supercell: three_counts(p.get_list("phonon_supercell"), "-phonon_supercell")?,
            phonon_path: p
                .get_list("phonon_path")
                .iter()
                .map(|text| wave_vector(text))
                .collect::<MoptResult<_>>()?,
            phonon_points: p.get_usize("phonon_points", 40)?,
            phonon_mesh: three_counts(p.get_list("phonon_mesh"), "-phonon_mesh")?,
            phonon_dos_width: p.get_f64("phonon_dos_width", 5.0)?,
            local_modes: p.get_list("local_modes").to_vec(),
            partial_atoms: match p.get_opt_str("partial_atoms") {
                Some(list) => parse_atom_list("-partial_atoms", list, None)?,
                None => Vec::new(),
            },
            partial_kind: PartialKind::parse(p.get_str("partial_hessian", "block"))?,
            partial_free_direction: p
                .get_f64("partial_free_direction", vib::DEFAULT_FREE_DIRECTION)?,
            path: p.get_bool("path", false),
            path_tangent: match p.get_opt_str("path_tangent").unwrap_or("gradient") {
                "gradient" => PathTangent::Gradient,
                "points" => PathTangent::Points,
                other => {
                    return Err(MoptError::config(format!(
                        "-path_tangent is gradient or points, got '{other}'"
                    )))
                }
            },
            phase_prominence: p.get_f64("phase_prominence", 0.0)?,
            orbitals: OrbitalPathSettings {
                nro: p.get_bool("natural_reaction_orbitals", false),
                roet: p.get_bool("reactive_orbital_energies", false),
                step: p.get_f64("orbital_step", DEFAULT_ORBITAL_STEP)?,
                diabatic: states_of(p.get_list("diabatize"))?,
            },
            constraints: super::constraints::parse_constraints(p)?,
        };
        config.check_masses()?;
        Ok(config)
    }

    /// Refuse the analyses that weight with the standard atomic weights only when other masses
    /// are chosen (`-freq_masses`): they would mix two sets of masses.
    fn check_masses(&self) -> MoptResult<()> {
        if self.frequency.thermo.masses == vib::MassChoice::Standard {
            return Ok(());
        }
        let other = [
            (self.anharmonic, "-anharmonic"),
            (!self.local_modes.is_empty(), "-local_modes"),
            (!self.partial_atoms.is_empty(), "-partial_atoms"),
            (self.phonon_supercell.is_some(), "-phonon_supercell"),
            (!self.model_function.is_empty(), "-mf"),
        ];
        match other.iter().find(|(given, _)| *given) {
            Some((_, name)) => Err(MoptError::unsupported(format!(
                "freqmain: -freq_masses {} is available for the normal modes, the \
                 thermochemistry and -path; {name} weights with the standard atomic weights only",
                self.frequency.thermo.masses.name()
            ))),
            None => Ok(()),
        }
    }
}

/// The excited states of `-diabatize N M ...` (none when it is not given): two or more
/// different numbers, 1 the lowest.
fn states_of(values: &[String]) -> MoptResult<Vec<usize>> {
    let numbers: Vec<usize> = values.iter().filter_map(|v| v.parse().ok()).collect();
    let mut different = numbers.clone();
    different.sort_unstable();
    different.dedup();
    let wrong = numbers.len() != values.len()
        || numbers.len() == 1
        || different.len() != numbers.len()
        || numbers.contains(&0);
    if wrong {
        return Err(MoptError::config(format!(
            "-diabatize takes two or more different excited states (1: the lowest), got '{}'",
            values.join(" ")
        )));
    }
    Ok(numbers)
}

/// Three positive counts of an option, or `None` when it is not given.
fn three_counts(values: &[String], option: &str) -> MoptResult<Option<[usize; 3]>> {
    if values.is_empty() {
        return Ok(None);
    }
    let counts: Vec<usize> = values
        .iter()
        .map(|v| v.parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| MoptError::config(format!("{option} takes three whole numbers")))?;
    match counts[..] {
        [a, b, c] if a > 0 && b > 0 && c > 0 => Ok(Some([a, b, c])),
        _ => Err(MoptError::config(format!(
            "{option} takes three positive whole numbers"
        ))),
    }
}

/// A wave vector `Q1,Q2,Q3` in units of the reciprocal lattice vectors.
fn wave_vector(text: &str) -> MoptResult<[f64; 3]> {
    let values: Vec<f64> = text
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| MoptError::config(format!("-phonon_path: '{text}' is not Q1,Q2,Q3")))?;
    match values[..] {
        [a, b, c] => Ok([a, b, c]),
        _ => Err(MoptError::config(format!(
            "-phonon_path: '{text}' is not Q1,Q2,Q3"
        ))),
    }
}

/// `START,END,STEP` of `-temp_scan`.
fn temperature_scan(text: &str) -> MoptResult<(f64, f64, f64)> {
    let values: Vec<f64> = text
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| MoptError::config(format!("-temp_scan START,END,STEP, got '{text}'")))?;
    match values[..] {
        [start, end, step] if start > 0.0 && end >= start && step > 0.0 => {
            if (end - start) / step > 1e5 {
                return Err(MoptError::config(
                    "-temp_scan: more than 100000 temperatures",
                ));
            }
            Ok((start, end, step))
        }
        _ => Err(MoptError::config(format!(
            "-temp_scan START,END,STEP with 0 < START <= END and STEP > 0, got '{text}'"
        ))),
    }
}

/// Result of one input.
#[derive(Debug, Clone)]
pub struct FreqmainResult {
    /// Run directory.
    pub directory: PathBuf,
    /// The analysis (`None` for a periodic cell, whose modes are in `normal_modes.txt`, and
    /// with `-path`).
    pub analysis: Option<VibrationalAnalysis>,
    /// At a crossing of two states (`-mf opt_mesx`, `-mf opt_meci`): the vibrations within
    /// the seam.
    pub seam: Option<SeamAnalysis>,
    /// `-boltz`: Boltzmann population among the inputs.
    pub population: Option<f64>,
    /// `-anharmonic`: the force field and what follows from it.
    pub anharmonic: Option<Anharmonic>,
    /// `-phonon_supercell`: the phonons of the crystal.
    pub phonons: Option<Phonons>,
    /// `-local_modes`: the local modes of the chosen internal coordinates.
    pub local_modes: Option<LocalModes>,
    /// `-partial_atoms`: the Hessian of the chosen atoms and their normal modes.
    pub partial: Option<(PartialHessian, NormalModes)>,
    /// `-path`: the path and the transverse vibrations along it.
    pub profile: Option<PathProfile>,
    /// `-path`: the remaining analyses of the path.
    pub path: Option<PathAnalysis>,
    /// `-nro`, `-roet`: the analyses of the orbitals along the frames.
    pub orbitals: Option<OrbitalPath>,
}

/// What `-phonon_supercell` computes.
#[derive(Debug, Clone)]
pub struct Phonons {
    /// The force constants.
    pub constants: ForceConstants,
    /// `-phonon_path`: distance along the path, wave vector, and wavenumbers (cm^-1).
    pub dispersion: Vec<(f64, [f64; 3], Vec<f64>)>,
    /// `-phonon_mesh`: the density of states `(wavenumber, states per cm^-1 and cell)`.
    pub density_of_states: Vec<(f64, f64)>,
    /// `-phonon_mesh`: thermodynamic functions per cell at `-temp`.
    pub thermo: Option<PhononThermo>,
}

/// What `-anharmonic` computes.
#[derive(Debug, Clone)]
pub struct Anharmonic {
    /// Quartic force field in dimensionless normal coordinates.
    pub field: ForceField,
    /// Second-order perturbation theory.
    pub vpt2: Vpt2,
    /// Vibrational self-consistent field.
    pub vscf: Vscf,
}

/// What `-path` derives from the profile of a path.
#[derive(Debug, Clone)]
pub struct PathAnalysis {
    /// Energy of the surface at every point (Hartree).
    pub energies: Vec<f64>,
    /// Overlap of the gradient with the line through the neighbouring points (1 on a
    /// steepest-descent path; NaN where the gradient vanishes).
    pub gradient_alignment: Vec<f64>,
    /// Components of the path direction and curvature along internal coordinates.
    pub components: InternalComponents,
    /// Reaction phases between the minima of the curvature.
    pub phases: Vec<ReactionPhase>,
    /// Free energies of the generalized transition states.
    pub free_energy: FreeEnergyProfile,
    /// Tunneling through the vibrationally adiabatic barrier.
    pub tunneling: Tunneling,
}

/// Write the reports of a vibrational analysis of `sys` into `dir` and the log:
/// `normal_modes.txt`, `vibration_animation/`, and, for a molecule, `thermochemistry.txt`.
/// `directions` are projected out with the rigid-body motions.
#[allow(clippy::too_many_arguments)]
pub fn write_reports(
    sys: &MolecularSystem,
    energy: f64,
    hessian: &Mat<f64>,
    settings: &FrequencySettings,
    directions: &[Direction],
    dir: &Path,
    log: &Logger,
) -> MoptResult<(Option<VibrationalAnalysis>, Vec<f64>)> {
    let mut settings = *settings;
    settings.thermo.multiplicity = sys.spin_charge.multiplicity.max(1);
    log.info(settings.thermo.masses.describe());
    let (modes, analysis, thermo) = if sys.has_pbc() {
        let modes = vib::normal_modes_with_masses(
            sys,
            hessian,
            settings.project_rigid,
            directions,
            &settings.thermo.masses.masses(sys)?,
        )?;
        log.info(
            "Periodic cell: the modes at the zone centre. Thermodynamic functions of a \
             molecule do not apply and are not written.",
        );
        (modes, None, String::new())
    } else {
        let analysis = vib::analyze_without(sys, hessian, energy, &settings, directions)?;
        let text = vib::thermochemistry_text(&analysis);
        std::fs::write(dir.join("thermochemistry.txt"), &text)?;
        (analysis.modes.clone(), Some(analysis), text)
    };
    let table = vib::normal_modes_text(sys, &modes, settings.table_cutoff);
    std::fs::write(dir.join("normal_modes.txt"), &table)?;
    let movies = dir.join("vibration_animation");
    std::fs::create_dir_all(&movies)?;
    for (name, frames) in vib::animations(sys, &modes, &settings) {
        std::fs::write(movies.join(name), frames)?;
    }
    for line in thermo.lines().chain(table.lines()) {
        log.info(line);
    }
    Ok((analysis, modes.wavenumbers))
}

/// The vibrational problem on the surface of the constraints `set` at the constrained
/// stationary point `sys` (`gradient` and `hessian` Cartesian, of the whole surface): the
/// Hessian of the Lagrangian `H - sum_k lambda_k d2c_k` with the multipliers
/// `lambda = (J J^T)^+ J g` of the gradient, and the gradients of the constraints as
/// directions to remove (they are mass-weighted as gradients, `M^-1/2 grad c_k`, so that the
/// modes are those of the motion within the constraints). At such a point the gradient is
/// not zero (it is the constraint force), and the bare Hessian is not the curvature of the
/// constrained surface.
///
/// On a surface that is `invariant` under rigid-body motions, constraints that pin the frame
/// (frozen atoms, Cartesian components, oriented groups) still leave the rigid motions that
/// keep them (`math::projection::rigid_motions_keeping`: the rotations about one frozen atom,
/// the rotation about the line through two): those are zero modes of the constrained surface,
/// not vibrations, and are among the directions returned (as displacements). Constraints that
/// do not pin the frame leave all the rigid motions, which the caller projects as usual.
///
/// MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy analyses the bare Hessian).
pub fn constraint_surface(
    set: &crate::pes::constraint::ConstraintSet,
    sys: &MolecularSystem,
    gradient: &Col<f64>,
    hessian: &Mat<f64>,
    invariant: bool,
) -> MoptResult<(Mat<f64>, Vec<Direction>)> {
    let frame = set.frame(sys, gradient)?;
    let lagrangian = hessian - &frame.curvature;
    let n = sys.dof();
    let rows: Vec<Col<f64>> = (0..frame.jacobian.nrows())
        .map(|k| Col::from_fn(n, |i| frame.jacobian[(k, i)]))
        .collect();
    let kept = if invariant && set.breaks_rigid_body() {
        crate::math::projection::rigid_motions_keeping(sys, &rows)
    } else {
        Vec::new()
    };
    let mut directions: Vec<Direction> = rows.into_iter().map(Direction::Gradient).collect();
    directions.extend(kept.into_iter().map(Direction::Displacement));
    Ok((lagrangian, directions))
}

/// Whether the surface of `opt` with the calculator `calc` is invariant under rigid-body
/// motions (`pes::term::frame_invariant` over the calculator and every term of the surface),
/// whatever its constraints pin.
pub fn surface_invariant(opt: &GeometryOptimizer, calc: &dyn Calculator) -> MoptResult<bool> {
    let terms = opt.surface().terms()?;
    Ok(crate::pes::term::frame_invariant(
        calc.is_rigid_body_invariant(),
        terms.iter().map(|t| t.as_ref()),
    ))
}

/// Refuse the analyses of `freqmain` that do not have a meaning on the surface of
/// constraints (they take every Cartesian direction, or another surface).
fn check_constrained_analysis(config: &FreqmainConfig, oniom: bool) -> MoptResult<()> {
    let other = [
        (config.path, "-path"),
        (config.anharmonic, "-anharmonic"),
        (!config.local_modes.is_empty(), "-local_modes"),
        (!config.partial_atoms.is_empty(), "-partial_atoms"),
        (config.phonon_supercell.is_some(), "-phonon_supercell"),
        (config.orbitals.any(), "-nro, -roet, -diabatize"),
        (!config.model_function.is_empty(), "-mf"),
        (
            oniom,
            "-oniom (the Hessian of the core with its environment relaxed)",
        ),
    ];
    match other.iter().find(|(given, _)| *given) {
        Some((_, name)) => Err(MoptError::unsupported(format!(
            "freqmain: constraints (-fix, -pc, -gfix, -uc) give the vibrations on the \
             constraint surface of one structure; {name} is not available with them"
        ))),
        None => Ok(()),
    }
}

/// The number of scalar constraints among the `directions` of [`constraint_surface`].
pub fn held_constraints(directions: &[Direction]) -> usize {
    directions
        .iter()
        .filter(|d| matches!(d, Direction::Gradient(_)))
        .count()
}

/// The log line of an analysis on the surface of the constraints with the `directions` of
/// [`constraint_surface`]; `rigid`: all the rigid-body motions are projected as well.
pub fn announce_constraint_surface(directions: &[Direction], rigid: bool, log: &Logger) {
    let held = held_constraints(directions);
    let kept = directions.len() - held;
    log.info(&format!(
        "Vibrational analysis on the constraint surface: the Hessian of the Lagrangian with \
         the {held} constrained direction(s) removed (mass-weighted), {}",
        if rigid {
            "and the rigid-body motions; 3N - 6 - m (3N - 5 - m linear) vibrations are left"
                .to_string()
        } else if kept > 0 {
            format!(
                "and the {kept} rigid-body motion(s) that keep the constraints (zero modes of \
                 the surface, such as rotations about frozen atoms); 3N - m - {kept} vibrations \
                 are left"
            )
        } else {
            "the constraints pin the frame; 3N - m vibrations are left".to_string()
        }
    ));
}

/// Say in `thermochemistry.txt` (when there is one in `dir`) that the analysis lacks the
/// `held` constrained modes.
pub fn note_constraint_surface(dir: &Path, held: usize) -> MoptResult<()> {
    let path = dir.join("thermochemistry.txt");
    if !path.exists() {
        return Ok(());
    }
    let mut text = std::fs::read_to_string(&path)?;
    text.push_str(&format!(
        "Note: the analysis is on the constraint surface; the {held} constrained direction(s) \
         are not vibrations here, and the thermal functions lack their modes.\n"
    ));
    std::fs::write(&path, text)?;
    Ok(())
}

/// The branching plane of the conical intersection `crossing` at `sys` with its second
/// direction from the gradients of the two states at displaced structures (see
/// [`crate::opt::crossing_check`]); `None` for states that do not interact, and when the
/// difference of the gradients does not change with the structure.
pub fn probed_plane(
    opt: &GeometryOptimizer,
    calc: &dyn Calculator,
    sys: &MolecularSystem,
    crossing: &CrossingStates,
    rigid: bool,
) -> MoptResult<Option<Vec<Col<f64>>>> {
    if crossing.plane.len() != 2 {
        return Ok(None);
    }
    let check = CrossingCheckSettings::default();
    let states = &crossing.states;
    let n = sys.dof();
    let difference = &states.gradients[0] - &states.gradients[1];
    let length = col_norm(&difference);
    let x = Col::from_fn(n, |i| difference[i] / length);
    let mut seed = crossing_check::seed(n);
    if rigid && !sys.has_pbc() {
        seed = project_out_basis_vec(&build_tr_rot_basis(sys, false), &seed);
    }
    let Some(mut direction) = crossing_check::orthogonal_unit(&seed, &x) else {
        return Ok(None);
    };
    let x0 = sys.coords_flat_bohr();
    let request = EvalRequest::ENERGY_GRAD;
    for _ in 0..check.products {
        let mut differences = Vec::with_capacity(2);
        for sign in [1.0, -1.0] {
            let moved = sys.with_flat_coords_bohr(&Col::from_fn(n, |i| {
                x0[i] + sign * check.displacement * direction[i]
            }))?;
            let tasks = opt.surface().tasks(&moved, &request)?;
            let results = CalculatorSet::single(calc).evaluate(&tasks)?;
            let found = opt
                .crossing_states(&moved, &request, results)?
                .ok_or_else(|| MoptError::calc("the surface is not a crossing of two states"))?;
            differences.push(&found.states.gradients[0] - &found.states.gradients[1]);
        }
        let w = Col::from_fn(n, |i| {
            (differences[0][i] - differences[1][i]) / (2.0 * check.displacement)
        });
        match crossing_check::orthogonal_unit(&w, &x) {
            Some(y) => direction = y,
            None => return Ok(None),
        }
    }
    Ok(Some(vec![x, direction]))
}

/// The vibrations within the seam of the two crossing states `crossing` at `sys`, written to
/// `seam.txt`, `normal_modes.txt`, and `vibration_animation/` in `dir` and to the log.
/// `probed` is the branching plane from [`probed_plane`]; without it the second direction
/// of a conical intersection is taken from the Hessians.
pub fn write_seam_reports(
    sys: &MolecularSystem,
    crossing: &CrossingStates,
    probed: Option<&[Col<f64>]>,
    settings: &FrequencySettings,
    dir: &Path,
    log: &Logger,
) -> MoptResult<SeamAnalysis> {
    let states = &crossing.states;
    let hessians = states
        .hessians
        .as_ref()
        .ok_or_else(|| MoptError::calc("the states of the crossing came without Hessians"))?;
    let (plane, second) = match probed {
        Some(plane) => (plane, SecondDirection::Given),
        None => (&crossing.plane[..], SecondDirection::FromHessians),
    };
    let seam = vib::seam_analysis(
        sys,
        [states.energies[0], states.energies[1]],
        [&states.gradients[0], &states.gradients[1]],
        [&hessians[0], &hessians[1]],
        plane,
        second,
        settings.project_rigid,
    )?;
    let kind = if seam.plane.len() == 2 {
        "conical intersection (two directions in the branching plane)"
    } else {
        "crossing of states that do not interact (one direction in the branching plane)"
    };
    let unstable = seam.modes.imaginary_modes();
    let mut lines = vec![
        format!(
            "Vibrations within the seam of {} and {}: {kind}",
            crossing.labels[0], crossing.labels[1]
        ),
        format!(
            "Energies [Eh]: {:.10} and {:.10}; gap {:.3e}; mean {:.10}",
            states.energies[0], states.energies[1], seam.gap, seam.mean_energy
        ),
        format!(
            "Difference gradient: {:.6e} Eh/Bohr; multiplier of the seam condition: {:.6}",
            seam.difference_gradient, seam.multiplier
        ),
        format!(
            "RMS of the mean gradient within the seam: {:.3e} Eh/Bohr",
            seam.seam_gradient
        ),
    ];
    if let (Some(component), Some((largest, next, overlap))) =
        (seam.coupling_component, seam.coupling_curvature)
    {
        let of_search = crossing
            .plane
            .get(1)
            .map_or(f64::NAN, |y| col_dot(y, &seam.plane[1]).abs());
        lines.push(match probed {
            Some(_) => format!(
                "Second direction of the branching plane: from the gradients at displaced \
                 structures; overlap {of_search:.4} with the direction of the search, \
                 {overlap:.4} with the eigenvector of the difference of the Hessians of the \
                 largest eigenvalue ({largest:.4e}; next: {next:.4e} Eh/Bohr^2); mean gradient \
                 along it {component:.6e} Eh/Bohr"
            ),
            None => format!(
                "Second direction of the branching plane: the eigenvector of the difference \
                 of the Hessians of the largest eigenvalue ({largest:.4e}; next: {next:.4e} \
                 Eh/Bohr^2); overlap {overlap:.4} with the direction of the search; mean \
                 gradient along it {component:.6e} Eh/Bohr"
            ),
        });
    }
    lines.push(if unstable == 0 {
        "All curvatures within the seam are positive: a minimum of the seam.".to_string()
    } else {
        format!(
            "{unstable} negative curvatures within the seam: a saddle point of the seam, not \
             its minimum."
        )
    });
    if seam.seam_gradient > STATIONARY_RMS_GRADIENT {
        lines.push(
            "The mean gradient within the seam is not small: the structure is not a \
             stationary point of the seam."
                .to_string(),
        );
        if seam.coupling_curvature.is_some() {
            lines.push(
                "A search by gradient projection estimates the second direction of the \
                 branching plane from its own steps; when that estimate differs from the \
                 direction found here (see the overlap above; a structure that kept its \
                 symmetry is the usual case), the search removes a direction in which the \
                 energy still falls and stops early. The check of the branching plane at \
                 convergence (on unless -no_crossing_check is given) corrects this; search \
                 again from this structure."
                    .to_string(),
            );
        }
    }
    if seam.gap.abs() > 1e-3 {
        lines.push(format!(
            "The two states are {:.3e} Eh apart: the structure is not on the seam.",
            seam.gap.abs()
        ));
    }
    lines.extend(seam.notes.iter().cloned());
    std::fs::write(dir.join("seam.txt"), lines.join("\n") + "\n")?;
    let table = vib::normal_modes_text(sys, &seam.modes, settings.table_cutoff);
    std::fs::write(dir.join("normal_modes.txt"), &table)?;
    let movies = dir.join("vibration_animation");
    std::fs::create_dir_all(&movies)?;
    for (name, frames) in vib::animations(sys, &seam.modes, settings) {
        std::fs::write(movies.join(name), frames)?;
    }
    for line in lines.iter().map(String::as_str).chain(table.lines()) {
        log.info(line);
    }
    Ok(seam)
}

/// Run `freqmain` for every input with the calculators of `registry`, writing run
/// directories under `base`.
pub fn run_with(
    config: &FreqmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<Vec<FreqmainResult>> {
    let energies = &config.single_point_energies;
    if !energies.is_empty() && energies.len() != config.inputs.len() {
        return Err(MoptError::config(format!(
            "-spc_energy: {} energies for {} inputs",
            energies.len(),
            config.inputs.len()
        )));
    }
    let mut results = Vec::with_capacity(config.inputs.len());
    for (index, input) in config.inputs.iter().enumerate() {
        let single_point = energies.get(index).copied();
        results.push(
            run_one(config, input, single_point, base, console, registry)
                .context(input.display())?,
        );
    }
    if config.boltzmann && results.len() > 1 {
        boltzmann_populations(config, &mut results, base)?;
    }
    Ok(results)
}

fn run_one(
    config: &FreqmainConfig,
    input: &Path,
    single_point: Option<f64>,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<FreqmainResult> {
    let spin = config.state.of_file(input)?;
    let start = read_xyz_file(input, spin, None)?;
    let evaluations_before = registry.counts();
    // ONIOM: the analysis is that of a structure whose environment is at a minimum, as in
    // every other ONIOM run; the environment is relaxed at the core of the input first (a
    // structure from an optimization with microiterations does not move).
    let relaxer = super::oniom_micro::EnvironmentRelaxer::new(
        &config.calculator,
        &config.calculator_options,
        &start,
        &[],
        config.excited_state,
        registry,
    )?;
    let (start, relaxed_note) = match &relaxer {
        Some(r) => {
            let (relaxed, note) = r.relax(&start)?;
            (relaxed, Some(note))
        }
        None => (start, None),
    };
    if !config.constraints.is_empty() {
        check_constrained_analysis(config, relaxer.is_some())?;
    }
    let frames_of = |path: &Path| -> MoptResult<Vec<MolecularSystem>> {
        read_xyz_trajectory_file(path, spin, None)?
            .frames
            .into_iter()
            .map(|f| match &relaxer {
                Some(r) => Ok(r.relax(&f.system)?.0),
                None => Ok(f.system),
            })
            .collect()
    };
    // With ONIOM: the surface of whole structures, at the structure relaxed above.
    let calc = super::oniom_micro::at_given_structures(
        &config.calculator,
        &config.calculator_options,
        &start,
        registry,
    )?;
    let calc =
        crate::pes::o1numhess::wrap_if_requested(calc, config.model_hessian.as_deref(), false);
    let calc =
        crate::pes::given_hessian::wrap_if_given(calc, config.initial_hessian.as_deref(), &start)?;
    if calc.uses_electrons() {
        start.spin_charge.check_electrons(&start.elements)?;
    }
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => {
            let now = DateTime::now_utc();
            let stem = input.with_extension("");
            base.join(now.date_dir()).join(format!(
                "{}_FREQ_{}",
                stem.to_string_lossy(),
                now.timestamp()
            ))
        }
    };
    std::fs::create_dir_all(&dir)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} freqmain: {}",
        crate::VERSION,
        input.display()
    ));
    log.info(&format!("Calculator: {}", calc.description()));
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&config.calculator_options.software_path_report());
    if let Some(note) = &relaxed_note {
        log.info(note);
    }
    // What the evaluations say (warnings, <S**2>, the output of the program) goes to the
    // log as they happen.
    let journaled = Narrated::new(&journaled, log.clone(), "evaluation");
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document),
    )?;

    // The surface is the optimizer's: the calculator and the bias potentials.
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    let mut terms = Vec::with_capacity(config.biases.len());
    for spec in &config.biases {
        let k = counts.entry(spec.key.as_str()).or_insert(0);
        terms.push((spec.label(*k), spec.build(&start)?));
        *k += 1;
    }
    let invariant = crate::pes::term::frame_invariant(
        calc.is_rigid_body_invariant(),
        terms.iter().map(|(_, t)| t.as_ref()),
    );
    let settings = OptimizerSettings {
        rigid_body_invariant: invariant,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, start)?;
    // The states first, then the bias potentials on top of them.
    if let Some(surface) =
        super::optmain::state_surface(&config.model_function, config.excited_state)?
    {
        if let Some(name) = config.model_function.first() {
            log.info(&format!("Two states combined by the model function {name}"));
        }
        if config.excited_state > 0 {
            log.info(&format!("Electronic state: {}", config.excited_state));
        }
        opt.set_surface(surface)?;
        super::optmain::report_excitations(calc.as_ref(), opt.initial(), &dir, &log)?;
    }
    for (label, term) in terms {
        opt.add_term(&label, term)?;
    }
    // The orbitals along the frames of the input; without -path this is the whole job.
    let orbitals = if config.orbitals.any() {
        let frames: Vec<MolecularSystem> = frames_of(input)?;
        let found = super::orbital_path::run(
            &config.orbitals,
            &journaled,
            &frames,
            config.frequency.thermo.temperature,
            &dir,
            &log,
        )?;
        if !config.path {
            log.flush();
            return Ok(FreqmainResult {
                directory: dir,
                analysis: None,
                seam: None,
                population: None,
                anharmonic: None,
                phonons: None,
                local_modes: None,
                partial: None,
                profile: None,
                path: None,
                orbitals: Some(found),
            });
        }
        Some(found)
    } else {
        None
    };
    if config.path {
        let frames: Vec<MolecularSystem> = frames_of(input)?;
        let (profile, analysis) =
            path_profile(config, &mut opt, &journaled, invariant, &frames, &dir, &log)?;
        log.flush();
        return Ok(FreqmainResult {
            directory: dir,
            analysis: None,
            seam: None,
            population: None,
            anharmonic: None,
            phonons: None,
            local_modes: None,
            partial: None,
            profile: Some(profile),
            path: Some(analysis),
            orbitals,
        });
    }
    let structure = opt.initial().clone();
    let tasks = opt.hessian_tasks(&structure)?;
    log.info(&format!(
        "Hessian of the structure ({} atoms): calculating",
        structure.n_atoms()
    ));
    let results = CalculatorSet::single(&journaled).evaluate(&tasks)?;
    if let Some(crossing) = opt.crossing(&structure, results.clone())? {
        if config.anharmonic || !config.local_modes.is_empty() || !config.partial_atoms.is_empty() {
            return Err(MoptError::config(
                "-anharmonic, -local_modes, and -partial_atoms are not available at a crossing \
                 of two states",
            ));
        }
        let mut frequency = config.frequency;
        frequency.project_rigid = invariant;
        let probed = probed_plane(
            &opt,
            &journaled,
            &structure,
            &crossing,
            frequency.project_rigid,
        )?;
        let seam = write_seam_reports(
            &structure,
            &crossing,
            probed.as_deref(),
            &frequency,
            &dir,
            &log,
        )?;
        log.flush();
        return Ok(FreqmainResult {
            directory: dir,
            analysis: None,
            seam: Some(seam),
            population: None,
            anharmonic: None,
            phonons: None,
            local_modes: None,
            partial: None,
            profile: None,
            path: None,
            orbitals: None,
        });
    }
    let (energy, gradient, hessian) = opt.surface_hessian(&structure, results)?;
    // With constraints: resolved at the input (eigenvector constraints with its Hessian), and
    // the vibrations on their surface.
    let constrained = if config.constraints.is_empty() {
        None
    } else {
        let resolved = config
            .constraints
            .iter()
            .map(|c| c.resolve(&structure, Some(&hessian)))
            .collect::<MoptResult<Vec<_>>>()?;
        let set = crate::pes::constraint::ConstraintSet::new(&resolved, &structure, None)?;
        let free = set
            .frame(&structure, &gradient)?
            .project_gradient(&gradient);
        let (lagrangian, normals) =
            constraint_surface(&set, &structure, &gradient, &hessian, invariant)?;
        Some((set.breaks_rigid_body(), free, lagrangian, normals))
    };
    let rms = match &constrained {
        Some((_, free, _, _)) => {
            let rms = col_rms(free);
            log.info(&format!(
                "RMS gradient within the constraints: {rms:.3e} Hartree/Bohr"
            ));
            rms
        }
        None => {
            let rms = col_rms(&gradient);
            log.info(&format!("RMS gradient: {rms:.3e} Hartree/Bohr"));
            rms
        }
    };
    let mut directions: Vec<Direction> = if config.project_gradient {
        log.info("The direction of the gradient is projected out of the modes.");
        vec![Direction::Gradient(gradient)]
    } else {
        if rms > STATIONARY_RMS_GRADIENT {
            log.warn(
                "The structure is not a stationary point: the frequencies are not those of a \
                 minimum or a transition state (-project_gradient gives the modes orthogonal \
                 to the gradient).",
            );
        }
        Vec::<Direction>::new()
    };
    let mut frequency = config.frequency;
    frequency.project_rigid = invariant;
    let held = constrained
        .as_ref()
        .map_or(0, |(_, _, _, normals)| held_constraints(normals));
    let hessian = match constrained {
        Some((pins, _, lagrangian, normals)) => {
            frequency.project_rigid = invariant && !pins;
            announce_constraint_surface(&normals, frequency.project_rigid, &log);
            directions.extend(normals);
            lagrangian
        }
        None => hessian,
    };
    let electronic = match single_point {
        Some(value) => {
            log.info(&format!(
                "Electronic energy for the thermochemistry: {value:.10} Hartree (given) instead \
                 of {:.10} Hartree",
                energy.value()
            ));
            value
        }
        None => energy.value(),
    };
    let (analysis, _) = write_reports(
        &structure,
        electronic,
        &hessian,
        &frequency,
        &directions,
        &dir,
        &log,
    )?;
    if held > 0 {
        note_constraint_surface(&dir, held)?;
    }
    let symmetry = match (frequency.symmetry_tolerance, structure.has_pbc()) {
        (Some(tolerance), false) => vib::point_group(&structure, tolerance).label,
        _ => "C1".to_string(),
    };
    std::fs::write(
        dir.join("symmetry.txt"),
        format!("Symmetry of final structure: {symmetry}"),
    )?;
    log.info(&format!("Symmetry: {symmetry}"));
    if let (Some(found), Some((first, last, step))) = (&analysis, config.temperature_scan) {
        let mut settings = frequency.thermo;
        settings.symmetry_number = found.thermo.symmetry_number;
        settings.multiplicity = structure.spin_charge.multiplicity.max(1);
        let mut csv = String::from(
            "temperature_K,zero_point_energy_hartree,enthalpy_hartree,entropy_hartree_per_K,\
             gibbs_hartree\n",
        );
        let count = ((last - first) / step + 1e-9).floor() as usize;
        for index in 0..=count {
            settings.temperature = first + step * index as f64;
            let at = vib::thermochemistry(&structure, &found.modes, electronic, &settings)?;
            csv.push_str(&format!(
                "{:.4},{:.10},{:.10},{:.12e},{:.10}\n",
                settings.temperature,
                at.zero_point_energy,
                at.enthalpy.total,
                at.entropy.total,
                at.gibbs.total
            ));
        }
        std::fs::write(dir.join("thermochemistry_scan.csv"), csv)?;
    }
    let phonons = match config.phonon_supercell {
        Some(repeats) => Some(phonon_analysis(
            config, registry, &journal, &structure, repeats, &frequency, &dir, &log,
        )?),
        None => None,
    };
    let local_modes = if config.local_modes.is_empty() {
        None
    } else {
        Some(local_mode_analysis(
            config, &structure, &hessian, &frequency, &dir, &log,
        )?)
    };
    let partial = if config.partial_atoms.is_empty() {
        None
    } else {
        Some(partial_analysis(
            config, &structure, &hessian, &frequency, &dir, &log,
        )?)
    };
    let anharmonic = if config.anharmonic {
        Some(anharmonic_analysis(
            config, &mut opt, &journaled, &structure, &hessian, &frequency, &dir, &log,
        )?)
    } else {
        None
    };
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(FreqmainResult {
        directory: dir,
        analysis,
        seam: None,
        population: None,
        anharmonic,
        phonons,
        local_modes,
        partial,
        profile: None,
        path: None,
        orbitals: None,
    })
}

/// `-phonon_supercell`: the force constants from the Hessian of the supercell, and the
/// dispersion, the density of states, and the thermodynamic functions asked for, written
/// into `dir`.
#[allow(clippy::too_many_arguments)]
fn phonon_analysis(
    config: &FreqmainConfig,
    registry: &CalculatorRegistry,
    journal: &RunJournal,
    structure: &MolecularSystem,
    repeats: [usize; 3],
    frequency: &FrequencySettings,
    dir: &Path,
    log: &Logger,
) -> MoptResult<Phonons> {
    if !config.biases.is_empty() {
        return Err(MoptError::config(
            "-phonon_supercell: bias potentials are defined for the cell given, not for its \
             supercell",
        ));
    }
    let large = phonon::supercell(structure, repeats)?;
    log.info(&format!(
        "Phonons: Hessian of the supercell of {} x {} x {} cells ({} atoms)",
        repeats[0],
        repeats[1],
        repeats[2],
        large.n_atoms()
    ));
    let calc = super::oniom_micro::at_given_structures(
        &config.calculator,
        &config.calculator_options,
        &large,
        registry,
    )?;
    let calc =
        crate::pes::o1numhess::wrap_if_requested(calc, config.model_hessian.as_deref(), false);
    let journaled = journal.wrap("phonon", calc.as_ref());
    let journaled = Narrated::new(&journaled, log.clone(), "evaluation of the supercell");
    let task = EvalTask {
        label: "phonon supercell".to_string(),
        calculator: DEFAULT_CALCULATOR.to_string(),
        system: large,
        request: EvalRequest::ENERGY_GRAD_HESS,
    };
    let hessian = CalculatorSet::single(&journaled)
        .evaluate(&[task])?
        .pop()
        .and_then(|out| out.hessian_bohr)
        .ok_or_else(|| MoptError::calc("the supercell came without a Hessian"))?;
    let constants = ForceConstants::from_supercell(structure, repeats, &hessian)?;
    log.info(&format!(
        "Acoustic sum rule: largest correction {:.3e} Eh/Bohr^2",
        constants.sum_rule_residual
    ));

    let mut lowest = f64::INFINITY;
    let mut dispersion = Vec::new();
    if config.phonon_path.len() == 1 {
        return Err(MoptError::config(
            "-phonon_path needs two wave vectors at least",
        ));
    }
    if !config.phonon_path.is_empty() {
        let mut csv = String::from("distance,q1,q2,q3");
        for branch in 1..=constants.branches() {
            csv.push_str(&format!(",branch_{branch}_cm-1"));
        }
        csv.push('\n');
        for (distance, q) in phonon::path(&config.phonon_path, config.phonon_points)? {
            let wavenumbers = constants.wavenumbers(q)?;
            csv.push_str(&format!(
                "{distance:.6},{:.6},{:.6},{:.6}",
                q[0], q[1], q[2]
            ));
            for w in &wavenumbers {
                csv.push_str(&format!(",{w:.4}"));
                lowest = lowest.min(*w);
            }
            csv.push('\n');
            dispersion.push((distance, q, wavenumbers));
        }
        std::fs::write(dir.join("phonon_dispersion.csv"), csv)?;
    }

    let mut density_of_states = Vec::new();
    let mut thermo = None;
    if let Some(mesh) = config.phonon_mesh {
        let on_mesh = phonon::mesh(mesh)?
            .into_iter()
            .map(|q| constants.wavenumbers(q))
            .collect::<MoptResult<Vec<Vec<f64>>>>()?;
        for w in on_mesh.iter().flatten() {
            lowest = lowest.min(*w);
        }
        let width = config.phonon_dos_width;
        density_of_states = phonon::density_of_states(&on_mesh, width, width / 5.0)?;
        let mut csv = String::from("wavenumber_cm-1,states_per_cm-1_and_cell\n");
        for (at, density) in &density_of_states {
            csv.push_str(&format!("{at:.4},{density:.8e}\n"));
        }
        std::fs::write(dir.join("phonon_dos.csv"), csv)?;

        let settings = &frequency.thermo;
        let found = phonon::thermodynamics(&on_mesh, settings.temperature, settings.lowest)?;
        let mut lines = vec![
            format!(
                "Harmonic crystal per cell, {} x {} x {} wave vectors, {} branches",
                mesh[0],
                mesh[1],
                mesh[2],
                constants.branches()
            ),
            format!("Temperature {:.4} [K]", found.temperature),
            format!("Zero-point energy {:.10} [Eh]", found.zero_point_energy),
            format!(
                "Vibrational internal energy {:.10} [Eh]",
                found.internal_energy
            ),
            format!("Entropy {:.10e} [Eh/K]", found.entropy),
            format!("Helmholtz free energy {:.10} [Eh]", found.free_energy),
            format!("Heat capacity Cv {:.10e} [Eh/K]", found.heat_capacity),
            format!(
                "Modes included: {}; left out (at or below {} cm^-1, or imaginary): {}",
                found.modes, settings.lowest, found.left_out
            ),
        ];
        if found.imaginary > 0 {
            lines.push(format!(
                "{} imaginary modes on the mesh: the structure is not stable against these \
                 waves, and the thermodynamic functions leave them out.",
                found.imaginary
            ));
        }
        for line in &lines {
            log.info(line);
        }
        std::fs::write(
            dir.join("phonon_thermochemistry.txt"),
            lines.join("\n") + "\n",
        )?;
        if let Some((first, last, step)) = config.temperature_scan {
            let mut csv = String::from(
                "temperature_K,zero_point_energy_hartree,internal_energy_hartree,\
                 entropy_hartree_per_K,free_energy_hartree,heat_capacity_hartree_per_K\n",
            );
            let count = ((last - first) / step + 1e-9).floor() as usize;
            for index in 0..=count {
                let at =
                    phonon::thermodynamics(&on_mesh, first + step * index as f64, settings.lowest)?;
                csv.push_str(&format!(
                    "{:.4},{:.10},{:.10},{:.12e},{:.10},{:.12e}\n",
                    at.temperature,
                    at.zero_point_energy,
                    at.internal_energy,
                    at.entropy,
                    at.free_energy,
                    at.heat_capacity
                ));
            }
            std::fs::write(dir.join("phonon_thermochemistry_scan.csv"), csv)?;
        }
        thermo = Some(found);
    }
    if lowest.is_finite() {
        log.info(&format!(
            "Lowest phonon wavenumber on the path and the mesh: {lowest:.2} cm^-1"
        ));
    }
    Ok(Phonons {
        constants,
        dispersion,
        density_of_states,
        thermo,
    })
}

/// The internal coordinates of `-local_modes` for `sys`: `1-2`, `1-2-3`, `1-2-3-4` by atom
/// numbers starting at 1, `bonds`, or `auto`.
fn local_mode_coordinates(specs: &[String], sys: &MolecularSystem) -> MoptResult<Vec<Primitive>> {
    let mut out: Vec<Primitive> = Vec::new();
    for spec in specs {
        let name = spec.to_ascii_lowercase();
        if name == "auto" || name == "bonds" {
            let generated = PrimitiveSet::generate(sys, &PrimitiveOptions::default())?;
            out.extend(
                generated
                    .primitives
                    .into_iter()
                    .filter(|p| name == "auto" || matches!(p, Primitive::Bond(_))),
            );
            continue;
        }
        let atoms: Vec<AtomId> = spec
            .split('-')
            .map(|token| AtomId::parse_one_based(token.trim())?.check(sys.n_atoms()))
            .collect::<MoptResult<_>>()
            .context(format!("-local_modes {spec}"))?;
        let distinct = (0..atoms.len()).all(|i| (0..i).all(|j| atoms[i] != atoms[j]));
        let primitive = match atoms[..] {
            [a, b] if distinct => Primitive::Bond([a, b]),
            [a, b, c] if distinct => Primitive::Angle([a, b, c]),
            [a, b, c, d] if distinct => Primitive::Dihedral([a, b, c, d]),
            _ => {
                return Err(MoptError::config(format!(
                    "-local_modes: '{spec}' is not two, three, or four different atoms joined \
                     by '-', nor 'bonds' or 'auto'"
                )))
            }
        };
        out.push(primitive);
    }
    Ok(out)
}

/// `-local_modes`: the local modes of the chosen coordinates, written to `local_modes.txt`
/// and `local_mode_composition.csv` in `dir`.
fn local_mode_analysis(
    config: &FreqmainConfig,
    structure: &MolecularSystem,
    hessian: &Mat<f64>,
    frequency: &FrequencySettings,
    dir: &Path,
    log: &Logger,
) -> MoptResult<LocalModes> {
    let coordinates = local_mode_coordinates(&config.local_modes, structure)?;
    let found = vib::local_modes(structure, hessian, frequency.project_rigid, &coordinates)?;
    // Hartree/Bohr^2 in mdyne/Angstrom, Hartree in mdyne Angstrom, Bohr in Angstrom.
    let (stretch, bend, length) = (
        Units::HARTREE_BOHR2_TO_MDYNE_ANGSTROM,
        Units::HARTREE_TO_JOULE * 1.0e18,
        Units::BOHR_TO_ANGSTROM,
    );
    let mut lines = vec![
        "Local vibrational modes (Konkoli-Cremer): the force constant of each coordinate with \
         the rest of the molecule relaxed, and the wavenumber of that motion"
            .to_string(),
        "Distances: value in Angstrom, force constant in mdyne/Angstrom, mass in amu. Angles: \
         value in degrees, force constant in mdyne Angstrom/rad^2, mass in amu Angstrom^2."
            .to_string(),
        format!(
            "{:<18} {:>12} {:>16} {:>14} {:>14}",
            "coordinate", "value", "force constant", "mass", "wavenumber"
        ),
    ];
    for mode in &found.modes {
        let distance = matches!(mode.primitive, Primitive::Bond(_));
        let (value, constant, mass) = if distance {
            (
                mode.value * length,
                mode.force_constant * stretch,
                1.0 / mode.inverse_mass,
            )
        } else if matches!(mode.primitive, Primitive::LinearBend { .. }) {
            // The component of the bend is a pure number.
            (
                mode.value,
                mode.force_constant * bend,
                length * length / mode.inverse_mass,
            )
        } else {
            (
                mode.value.to_degrees(),
                mode.force_constant * bend,
                length * length / mode.inverse_mass,
            )
        };
        lines.push(format!(
            "{:<18} {:>12.5} {:>16.6} {:>14.6} {:>14.2}",
            mode.label, value, constant, mass, mode.wavenumber
        ));
    }
    lines.extend(found.notes.iter().cloned());
    for line in &lines {
        log.info(line);
    }
    std::fs::write(dir.join("local_modes.txt"), lines.join("\n") + "\n")?;

    let mut csv = String::from("coordinate");
    for wavenumber in &found.normal_wavenumbers {
        csv.push_str(&format!(",{wavenumber:.2}"));
    }
    csv.push('\n');
    for mode in &found.modes {
        csv.push_str(&mode.label);
        for share in &mode.composition {
            if share.is_finite() {
                csv.push_str(&format!(",{share:.6}"));
            } else {
                csv.push(',');
            }
        }
        csv.push('\n');
    }
    std::fs::write(dir.join("local_mode_composition.csv"), csv)?;
    Ok(found)
}

/// `-boltz`: relative Gibbs free energies and populations of the inputs, written to
/// `boltzmann_<time>.csv` under `base`.
fn boltzmann_populations(
    config: &FreqmainConfig,
    results: &mut [FreqmainResult],
    base: &Path,
) -> MoptResult<()> {
    let gibbs: Vec<f64> = results
        .iter()
        .map(|r| {
            r.analysis
                .as_ref()
                .map(|a| a.thermo.gibbs.total)
                .ok_or_else(|| {
                    MoptError::config(
                        "-boltz needs the thermochemistry of every input (molecules, not -path)",
                    )
                })
        })
        .collect::<MoptResult<_>>()?;
    let lowest = gibbs.iter().fold(f64::INFINITY, |m, g| m.min(*g));
    let kt = vib::boltzmann_hartree() * config.frequency.thermo.temperature;
    let exponents: Vec<f64> = gibbs.iter().map(|g| -(g - lowest) / kt).collect();
    let populations = normalized_exp(&exponents);
    let mut csv = String::from("input,gibbs_hartree,relative_kcal_mol,population\n");
    for (i, result) in results.iter_mut().enumerate() {
        result.population = Some(populations[i]);
        csv.push_str(&format!(
            "{},{:.10},{:.4},{:.6}\n",
            config.inputs[i].display(),
            gibbs[i],
            (gibbs[i] - lowest) * Units::HARTREE_TO_KCAL_MOL,
            populations[i]
        ));
    }
    let now = DateTime::now_utc();
    let folder = base.join(now.date_dir());
    std::fs::create_dir_all(&folder)?;
    std::fs::write(
        folder.join(format!("boltzmann_{}.csv", now.timestamp())),
        csv,
    )?;
    Ok(())
}

/// `-anharmonic`: the quartic force field of `structure` and the anharmonic wavenumbers,
/// written to `force_field.txt` and `anharmonic.txt` in `dir`.
#[allow(clippy::too_many_arguments)]
fn anharmonic_analysis(
    config: &FreqmainConfig,
    opt: &mut GeometryOptimizer,
    calc: &dyn Calculator,
    structure: &MolecularSystem,
    hessian: &Mat<f64>,
    frequency: &FrequencySettings,
    dir: &Path,
    log: &Logger,
) -> MoptResult<Anharmonic> {
    let settings = QffSettings {
        step: config.qff_step,
        lowest: frequency.thermo.lowest,
    };
    let mut qff = vib::Qff::new(structure, hessian, frequency.project_rigid, settings)?;
    log.info(&format!(
        "Anharmonic analysis: {} vibrations, {} Hessians at structures displaced by {} in the \
         dimensionless normal coordinates.",
        qff.modes(),
        qff.total(),
        config.qff_step
    ));
    while let Some(displaced) = qff.ask()? {
        let tasks = opt.hessian_tasks(&displaced)?;
        let results = CalculatorSet::single(calc).evaluate(&tasks)?;
        let (_, _, at_displaced) = opt
            .surface_hessian(&displaced, results)
            .context(format!("displaced structure {}", qff.done() + 1))?;
        qff.tell(&at_displaced)?;
        log.info(&format!(
            "Displaced Hessian {} of {}",
            qff.done(),
            qff.total()
        ));
    }
    let field = qff.force_field()?;
    let vpt2 = vib::vpt2(&field, &config.vpt2);
    let vscf = vib::vscf(&field, &config.vscf)?;
    let m = field.modes();

    // Constants above a hundredth of a wavenumber, each once.
    let mut constants = String::from(
        "Force constants in dimensionless normal coordinates (cm^-1); modes numbered as in \
         anharmonic.txt\ncubic: i j k phi_ijk\n",
    );
    for i in 0..m {
        for j in i..m {
            for k in j..m {
                let value = field.cubic(i, j, k);
                if value.abs() >= 0.01 {
                    constants.push_str(&format!("{} {} {} {value:.4}\n", i + 1, j + 1, k + 1));
                }
            }
        }
    }
    constants.push_str("quartic: i i j k phi_iijk\n");
    for i in 0..m {
        for j in 0..m {
            for k in j..m {
                // phi_iijj once, for i <= j.
                if j == k && j < i {
                    continue;
                }
                let value = field.quartic(i, j, k);
                if value.abs() >= 0.01 {
                    constants.push_str(&format!(
                        "{} {} {} {} {value:.4}\n",
                        i + 1,
                        i + 1,
                        j + 1,
                        k + 1
                    ));
                }
            }
        }
    }
    std::fs::write(dir.join("force_field.txt"), constants)?;

    let mut lines = vec![
        format!(
            "Anharmonic vibrational analysis: quartic force field from {} displaced Hessians \
             (step {} in the dimensionless normal coordinates)",
            qff.total(),
            config.qff_step
        ),
        String::new(),
        "Fundamentals (cm^-1)".to_string(),
        format!(
            "{:>5} {:>12} {:>12} {:>12}",
            "mode", "harmonic", "VPT2", "VSCF"
        ),
    ];
    for i in 0..m {
        lines.push(format!(
            "{:>5} {:>12.2} {:>12.2} {:>12.2}",
            i + 1,
            field.harmonic[i],
            vpt2.fundamentals[i],
            vscf.fundamentals[i]
        ));
    }
    lines.push(String::new());
    lines.push(format!(
        "Zero-point energy (cm^-1): harmonic {:.2}, VPT2 {:.2}, VSCF {:.2}",
        vpt2.zero_point_harmonic, vpt2.zero_point, vscf.zero_point
    ));
    lines.push(String::new());
    lines.push("First overtones, VPT2 (cm^-1)".to_string());
    for i in 0..m {
        lines.push(format!("{:>5} {:>12.2}", i + 1, vpt2.overtones[i]));
    }
    lines.push(String::new());
    lines.push("Anharmonic constants x_ij, VPT2 (cm^-1)".to_string());
    for i in 0..m {
        let row: Vec<String> = (0..=i)
            .map(|j| format!("{:>10.3}", vpt2.constants[(i, j)]))
            .collect();
        lines.push(format!("{:>5} {}", i + 1, row.join(" ")));
    }
    for resonance in &vpt2.resonances {
        let (i, j) = resonance.pair;
        lines.push(format!(
            "Fermi resonance: modes {} + {} with mode {} (gap {:.2} cm^-1, phi {:.2} cm^-1); \
             coupled levels at {:.2} and {:.2} cm^-1",
            i + 1,
            j + 1,
            resonance.single + 1,
            resonance.gap,
            resonance.constant,
            resonance.levels.0,
            resonance.levels.1
        ));
    }
    lines.extend(vpt2.notes.iter().cloned());
    if !vscf.converged {
        lines.push("VSCF did not converge for every state.".to_string());
    }
    if vscf.unbounded {
        lines.push(
            "VSCF: the quartic force field is not bounded from below within the basis (it has \
             states below the vibrations); the modals were followed by their overlap."
                .to_string(),
        );
    }
    for line in &lines {
        log.info(line);
    }
    std::fs::write(dir.join("anharmonic.txt"), lines.join("\n") + "\n")?;
    Ok(Anharmonic { field, vpt2, vscf })
}

/// `-partial_atoms`: the vibrations of the chosen atoms alone, written to `partial_hessian/`
/// in `dir`.
fn partial_analysis(
    config: &FreqmainConfig,
    structure: &MolecularSystem,
    hessian: &Mat<f64>,
    frequency: &FrequencySettings,
    dir: &Path,
    log: &Logger,
) -> MoptResult<(PartialHessian, NormalModes)> {
    let partial = vib::partial_hessian(
        structure,
        hessian,
        &config.partial_atoms,
        config.partial_kind,
        config.partial_free_direction,
    )?;
    let project = frequency.project_rigid && partial.kind.projects_rigid();
    let modes = vib::normal_modes(&partial.system, &partial.hessian, project)?;
    let out = dir.join("partial_hessian");
    std::fs::create_dir_all(&out)?;
    let numbers: Vec<String> = partial.atoms.iter().map(|a| (a + 1).to_string()).collect();
    let mut lines = vec![
        format!(
            "Partial Hessian ({}) of the atoms {}",
            partial.kind.name(),
            numbers.join(", ")
        ),
        match partial.kind {
            PartialKind::Block => "The block of these atoms; the others stay in place. The \
                                   translations and rotations of these atoms are projected out."
                .to_string(),
            PartialKind::Effective => format!(
                "The other atoms follow without a force on them (effective Hessian). {} \
                 directions of the other atoms have no restoring force and are left out. The \
                 translations and rotations of these atoms are projected out.",
                partial.free_directions
            ),
            PartialKind::Frozen => {
                "The block of these atoms in the fixed frame of the others; nothing is \
                 projected out."
                    .to_string()
            }
        },
    ];
    if let Some((atomic, mdyne)) = vib::relaxed_stretch(&partial) {
        lines.push(format!(
            "Relaxed force constant of the distance {}-{} (local stretching mode): {atomic:.8} \
             Hartree/Bohr^2 = {mdyne:.6} mdyne/Angstrom",
            numbers[0], numbers[1]
        ));
    }
    let shown = modes.vibrations(frequency.table_cutoff, true);
    let listed: Vec<String> = shown
        .iter()
        .map(|k| format!("{:.2}", modes.wavenumbers[*k]))
        .collect();
    lines.push(format!("Wavenumbers (cm^-1): {}", listed.join(" ")));
    for line in &lines {
        log.info(line);
    }
    std::fs::write(out.join("summary.txt"), lines.join("\n") + "\n")?;
    let n = partial.hessian.nrows();
    let mut matrix = String::new();
    for i in 0..n {
        let row: Vec<String> = (0..n)
            .map(|j| format!("{:.12e}", partial.hessian[(i, j)]))
            .collect();
        matrix.push_str(&row.join(" "));
        matrix.push('\n');
    }
    std::fs::write(out.join("hessian.txt"), matrix)?;
    let table = vib::normal_modes_text(&partial.system, &modes, frequency.table_cutoff);
    std::fs::write(out.join("normal_modes.txt"), &table)?;
    let movies = out.join("vibration_animation");
    std::fs::create_dir_all(&movies)?;
    for (name, frames) in vib::animations(&partial.system, &modes, frequency) {
        std::fs::write(movies.join(name), frames)?;
    }
    Ok((partial, modes))
}

/// `-path`: the analyses of the path through `frames`, written into `dir`.
fn path_profile(
    config: &FreqmainConfig,
    opt: &mut GeometryOptimizer,
    calc: &dyn Calculator,
    rigid: bool,
    frames: &[MolecularSystem],
    dir: &Path,
    log: &Logger,
) -> MoptResult<(PathProfile, PathAnalysis)> {
    if frames.len() < 2 {
        return Err(MoptError::config(
            "-path needs a file with the points of a path (two frames at least)",
        ));
    }
    let mut energies = Vec::with_capacity(frames.len());
    let mut gradients = Vec::with_capacity(frames.len());
    let mut hessians = Vec::with_capacity(frames.len());
    for (k, frame) in frames.iter().enumerate() {
        log.info(&format!(
            "Path point {} of {}: Hessian",
            k + 1,
            frames.len()
        ));
        let tasks = opt.hessian_tasks(frame)?;
        let results = CalculatorSet::single(calc).evaluate(&tasks)?;
        let (energy, gradient, hessian) = opt
            .surface_hessian(frame, results)
            .context(format!("path point {k}"))?;
        energies.push(energy.value());
        gradients.push(gradient);
        hessians.push(hessian);
    }
    let masses = config.frequency.thermo.masses.masses(&frames[0])?;
    log.info(config.frequency.thermo.masses.describe());
    let from_points = vib::path_tangents_with_masses(frames, Some(&masses))?;
    let gradient_alignment =
        vib::gradient_alignment_with_masses(frames, &gradients, Some(&masses))?;
    let points: Vec<PathPoint> = frames
        .iter()
        .enumerate()
        .map(|(k, frame)| {
            // Where the gradient vanishes the path runs along the line through the neighbours.
            let tangent = if config.path_tangent == PathTangent::Gradient
                && col_rms(&gradients[k]) > STATIONARY_RMS_GRADIENT
            {
                Direction::Gradient(gradients[k].clone())
            } else {
                from_points[k].clone()
            };
            PathPoint {
                system: frame.clone(),
                tangent,
                hessian: hessians[k].clone(),
            }
        })
        .collect();
    let least = (0..frames.len())
        .filter(|k| gradient_alignment[*k].is_finite())
        .min_by(|a, b| gradient_alignment[*a].total_cmp(&gradient_alignment[*b]));
    if let Some(k) = least {
        log.info(&format!(
            "Overlap of the gradient with the line through the neighbouring points: {:.4} at \
             least (point {k}). It is 1 on a steepest-descent path; for another path (a scan, \
             a band of images) use -path_tangent points.",
            gradient_alignment[k]
        ));
    }
    // The rigid-body motions are projected when the whole surface (calculator and bias
    // potentials) is invariant under them.
    let profile = vib::transverse_profile_with_masses(&points, rigid, Some(&masses))?;

    // Tunneling and the free energies of the generalized transition states.
    let mut thermo = config.frequency.thermo;
    thermo.multiplicity = frames[0].spin_charge.multiplicity.max(1);
    let molecular = rigid && !frames[0].has_pbc();
    let free_energy = vib::free_energy_profile(&points, &energies, &profile, &thermo, molecular)?;
    let tunneling = vib::tunneling(&energies, &profile, thermo.temperature)?;

    let mut csv = String::from(
        "point,arc_length_sqrt_amu_bohr,energy_hartree,rms_gradient,gradient_alignment,\
         zero_point_energy_hartree,adiabatic_energy_hartree,path_curvature,sct_mass_ratio,\
         lowest_transverse_cm-1,overlap_with_previous,transverse_cm-1\n",
    );
    for k in 0..frames.len() {
        let all: Vec<String> = profile.wavenumbers[k]
            .iter()
            .map(|w| format!("{w:.4}"))
            .collect();
        csv.push_str(&format!(
            "{k},{:.8},{:.12},{:.6e},{},{:.12},{:.12},{:.8e},{:.6},{:.4},{:.6},{}\n",
            profile.arc_length[k],
            energies[k],
            col_rms(&gradients[k]),
            if gradient_alignment[k].is_finite() {
                format!("{:.6}", gradient_alignment[k])
            } else {
                String::new()
            },
            profile.zero_point_energy[k],
            tunneling.adiabatic[k],
            profile.curvature[k],
            tunneling.mass_ratio[k],
            profile.lowest_wavenumber[k],
            profile.overlap[k],
            all.join(" ")
        ));
    }
    std::fs::write(dir.join("path_profile.csv"), csv)?;

    let mut report = String::new();
    for found in &profile.bifurcations {
        let line = format!(
            "Valley-ridge transition between points {} and {} at arc length {:.6} sqrt(amu) Bohr: \
             the lowest transverse wavenumber goes from {:.2} to {:.2} cm^-1.",
            found.after,
            found.after + 1,
            found.arc_length,
            profile.lowest_wavenumber[found.after],
            profile.lowest_wavenumber[found.after + 1],
        );
        log.info(&line);
        report.push_str(&line);
        report.push('\n');
    }
    if profile.bifurcations.is_empty() {
        let line = "No valley-ridge transition: the lowest transverse curvature does not turn \
                    negative along the path.";
        log.info(line);
        report.push_str(line);
        report.push('\n');
    }
    std::fs::write(dir.join("bifurcations.txt"), report)?;

    write_couplings(&profile, dir)?;
    let primitives = vib::path_primitives(&points)?;
    let components = vib::internal_components(&points, &profile, &primitives)?;
    let phases = vib::reaction_phases(&profile.curvature, config.phase_prominence);
    write_urva(&profile, &components, &phases, dir, log)?;
    write_vtst(&profile, &energies, &free_energy, &tunneling, dir, log)?;
    Ok((
        profile,
        PathAnalysis {
            energies,
            gradient_alignment,
            components,
            phases,
            free_energy,
            tunneling,
        },
    ))
}

/// `curvature_couplings.csv` (`B_kF` of the modes in ascending order of their wavenumbers)
/// and `mode_couplings.csv` (`B_kl` of the modes followed from the first point).
fn write_couplings(profile: &PathProfile, dir: &Path) -> MoptResult<()> {
    let most = profile
        .curvature_couplings
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    let mut csv = String::from("point,arc_length_sqrt_amu_bohr,path_curvature");
    for k in 1..=most {
        csv.push_str(&format!(",B_{k}F"));
    }
    csv.push('\n');
    for (i, couplings) in profile.curvature_couplings.iter().enumerate() {
        csv.push_str(&format!(
            "{i},{:.8},{:.8e}",
            profile.arc_length[i], profile.curvature[i]
        ));
        for b in couplings {
            csv.push_str(&format!(",{b:.8e}"));
        }
        csv.push('\n');
    }
    std::fs::write(dir.join("curvature_couplings.csv"), csv)?;

    let mut csv = String::from(
        "point,arc_length_sqrt_amu_bohr,mode_k,mode_l,wavenumber_k_cm-1,wavenumber_l_cm-1,B_kl\n",
    );
    for (i, couplings) in profile.mode_couplings.iter().enumerate() {
        let track = &profile.tracks[i];
        for k in 0..couplings.nrows() {
            for l in k + 1..couplings.ncols() {
                csv.push_str(&format!(
                    "{i},{:.8},{},{},{:.4},{:.4},{:.8e}\n",
                    profile.arc_length[i],
                    k + 1,
                    l + 1,
                    profile.wavenumbers[i][track[k]],
                    profile.wavenumbers[i][track[l]],
                    couplings[(k, l)]
                ));
            }
        }
    }
    std::fs::write(dir.join("mode_couplings.csv"), csv)?;
    Ok(())
}

/// `urva_direction.csv`, `urva_curvature.csv`, and `reaction_phases.txt`.
fn write_urva(
    profile: &PathProfile,
    components: &InternalComponents,
    phases: &[ReactionPhase],
    dir: &Path,
    log: &Logger,
) -> MoptResult<()> {
    let table = |rows: &[Vec<f64>]| -> String {
        let mut csv = String::from("point,arc_length_sqrt_amu_bohr");
        for label in &components.labels {
            csv.push(',');
            csv.push_str(label);
        }
        csv.push('\n');
        for (i, row) in rows.iter().enumerate() {
            csv.push_str(&format!("{i},{:.8}", profile.arc_length[i]));
            for value in row {
                // A coordinate that is not defined at this point leaves its field empty.
                if value.is_finite() {
                    csv.push_str(&format!(",{value:.8e}"));
                } else {
                    csv.push(',');
                }
            }
            csv.push('\n');
        }
        csv
    };
    std::fs::write(dir.join("urva_direction.csv"), table(&components.direction))?;
    std::fs::write(dir.join("urva_curvature.csv"), table(&components.curvature))?;

    let mut report = String::from(
        "Reaction phases: the path between two minima of its curvature, with the point of the\n\
         largest curvature and the internal coordinates with the largest curvature components\n\
         there (direction component in parentheses).\n",
    );
    for (number, phase) in phases.iter().enumerate() {
        let peak = phase.peak;
        let mut line = format!(
            "Phase {}: points {}-{}, arc length {:.4} to {:.4} sqrt(amu) Bohr; largest curvature \
             {:.4e} at point {peak}.",
            number + 1,
            phase.start,
            phase.end,
            profile.arc_length[phase.start],
            profile.arc_length[phase.end],
            profile.curvature[peak],
        );
        let mut order: Vec<usize> = (0..components.labels.len())
            .filter(|n| components.curvature[peak][*n].is_finite())
            .collect();
        order.sort_by(|a, b| {
            components.curvature[peak][*b]
                .abs()
                .total_cmp(&components.curvature[peak][*a].abs())
        });
        for n in order.into_iter().take(5) {
            line.push_str(&format!(
                " {} {:+.3e} ({:+.3})",
                components.labels[n], components.curvature[peak][n], components.direction[peak][n]
            ));
        }
        log.info(&line);
        report.push_str(&line);
        report.push('\n');
    }
    std::fs::write(dir.join("reaction_phases.txt"), report)?;
    Ok(())
}

/// `free_energy_profile.csv` and `variational_tst.txt`.
fn write_vtst(
    profile: &PathProfile,
    energies: &[f64],
    free: &FreeEnergyProfile,
    tunneling: &Tunneling,
    dir: &Path,
    log: &Logger,
) -> MoptResult<()> {
    let mut csv = String::from(
        "point,arc_length_sqrt_amu_bohr,energy_hartree,adiabatic_energy_hartree,\
         free_energy_hartree\n",
    );
    for (k, energy) in energies.iter().enumerate() {
        csv.push_str(&format!(
            "{k},{:.8},{energy:.12},{:.12},{:.12}\n",
            profile.arc_length[k], tunneling.adiabatic[k], free.free_energy[k]
        ));
    }
    std::fs::write(dir.join("free_energy_profile.csv"), csv)?;

    let last = energies.len() - 1;
    let mut lines = vec![
        format!("Temperature: {:.2} K", free.temperature),
        format!(
            "Highest energy (dividing surface of conventional transition state theory): point \
             {} at arc length {:.6} sqrt(amu) Bohr",
            free.saddle, profile.arc_length[free.saddle]
        ),
        format!(
            "Highest free energy (variational transition state): near point {} at arc length \
             {:.6} sqrt(amu) Bohr, {:.6e} Hartree above the point of the highest energy",
            free.variational,
            free.variational_arc_length,
            free.variational_free_energy - free.free_energy[free.saddle]
        ),
        format!(
            "Rate constant of canonical variational theory over conventional theory: {:.6}",
            free.recrossing
        ),
        format!(
            "Vibrationally adiabatic barrier: top at point {}, {:.6e} Hartree above the higher \
             end of the path",
            tunneling.top,
            tunneling.adiabatic[tunneling.top] - tunneling.threshold
        ),
        format!(
            "Transmission coefficient, zero-curvature tunneling (ZCT): {:.6}",
            tunneling.zct
        ),
        format!(
            "Transmission coefficient, small-curvature tunneling (SCT): {:.6}",
            tunneling.sct
        ),
    ];
    if free.saddle == 0 || free.saddle == last {
        lines.push(
            "The highest energy is at an end of the path: the path does not contain a barrier, \
             and the comparison with conventional theory does not apply."
                .to_string(),
        );
    }
    if free.variational == 0 || free.variational == last {
        lines.push(
            "The highest free energy is at an end of the path: the variational transition \
             state may lie beyond it."
                .to_string(),
        );
    }
    lines.push(
        "Tunneling counts the energies between the top of the adiabatic barrier and the higher \
         end of the path; a path that ends high on the barrier underestimates it."
            .to_string(),
    );
    for line in &lines {
        log.info(line);
    }
    std::fs::write(dir.join("variational_tst.txt"), lines.join("\n") + "\n")?;
    Ok(())
}

/// Thin wrapper used by tests and embedding code: settings for `args`.
pub fn config_from_cli(args: &[&str]) -> MoptResult<FreqmainConfig> {
    let p = super::option::parser("freqmain", FREQMAIN).parse(args)?;
    FreqmainConfig::from_args(&p)
}
