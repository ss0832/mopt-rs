//! The settings of `ieipmain`: which method, its options, and the inputs.
//!
//! The method is chosen as MultiOptPy chooses it: `-mf` before `-gnt` before `-addf` before
//! `-2pshs` before `-use_dimer` before `-use_spm`, and iEIP without any of them. The input is
//! a directory with the structures `*_[A-Z].xyz` (MultiOptPy's form; in the order of their
//! letters, where MultiOptPy takes them in the order of the directory), or one or two structure
//! files,
//! or one or two variables files for an objective of plain variables. `-ieip_opt KEY=VALUE`
//! (mopt-rs) holds the further settings of the methods.
//!
//! MultiOptPy: `interface.py:call_ieipparser`, `ieip.py:IEIPConfig`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use super::bias::{parse_biases, BiasSpec};
use super::calculator::{calculator_name, calculator_options};
use super::charge_spin::{state_specs, StateSpec};
use super::constraints::parse_constraints;
use super::option::reject_unavailable;
use super::settings::settings_json;
use super::spec::IEIPMAIN;
use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::opt::convergence::ConvergenceCriteria;
use crate::pes::constraint::Constraint;
use crate::pes::registry::CalculatorOptions;

/// The method of a run.
#[derive(Debug, Clone, PartialEq)]
pub enum IeipMethod {
    /// The improved elastic image pair method (the default).
    Ieip,
    /// `-mf NAME`: a model function of the states of the two inputs at the first structure.
    ModelFunction(String),
    /// `-gnt`: the growing Newton trajectory.
    NewtonTrajectory,
    /// `-addf`: the anharmonic downward distortion following.
    Addf,
    /// `-2pshs`: the two-point scaled hypersphere search.
    TwoPshs,
    /// `-use_dimer`: the dimer method.
    Dimer,
    /// `-use_spm`: the spring pair method.
    SpringPair,
    /// `-gsm`: the double-ended growing string (mopt-rs).
    Gsm,
    /// `-se_gsm DRIVING...`: the single-ended growing string (mopt-rs).
    SeGsm(Vec<String>),
    /// `-fsm`: the freezing string (mopt-rs).
    Fsm,
    /// `-dsafir`: the double-sphere AFIR (mopt-rs).
    DsAfir,
    /// `-lmm`: the local minimax method (mopt-rs, experimental).
    Lmm,
}

impl IeipMethod {
    /// Its name in the log.
    pub fn title(&self) -> &str {
        match self {
            Self::Ieip => "improved elastic image pair method (iEIP)",
            Self::ModelFunction(_) => "model function of two states (-mf)",
            Self::NewtonTrajectory => "growing Newton trajectory (-gnt)",
            Self::Addf => "anharmonic downward distortion following (-addf)",
            Self::TwoPshs => "two-point scaled hypersphere search (-2pshs)",
            Self::Dimer => "dimer method (-use_dimer)",
            Self::SpringPair => "spring pair method (-use_spm)",
            Self::Gsm => "double-ended growing string method (-gsm)",
            Self::SeGsm(_) => "single-ended growing string method (-se_gsm)",
            Self::Fsm => "freezing string method (-fsm)",
            Self::DsAfir => "double-sphere AFIR (-dsafir)",
            Self::Lmm => "local minimax method (-lmm, experimental)",
        }
    }

    /// Its name in the registry of two-point methods, for those that run through it.
    pub fn registry_name(&self) -> Option<&'static str> {
        match self {
            Self::Gsm => Some("gsm"),
            Self::SeGsm(_) => Some("se_gsm"),
            Self::Fsm => Some("fsm"),
            Self::DsAfir => Some("dsafir"),
            Self::Lmm => Some("lmm"),
            _ => None,
        }
    }

    /// How many structures it takes: at least, at most.
    pub fn inputs(&self, bonds: bool) -> (usize, usize) {
        match self {
            Self::Ieip | Self::TwoPshs | Self::ModelFunction(_) => (2, 2),
            Self::Gsm | Self::Fsm | Self::DsAfir => (2, 2),
            Self::NewtonTrajectory if bonds => (1, 2),
            Self::NewtonTrajectory => (2, 2),
            Self::Addf | Self::SeGsm(_) | Self::Lmm => (1, 2),
            Self::Dimer | Self::SpringPair => (1, 2),
        }
    }
}

/// The further settings of `-ieip_opt KEY=VALUE` (defaults: see the docs of the methods).
#[derive(Debug, Clone, PartialEq)]
pub struct IeipExtra {
    /// Draw the figures after every iteration (`figures=on|off`).
    pub figures: bool,
    /// Seed of the random direction of the dimer and of the spring pair without a second
    /// structure (`seed=`).
    pub seed: u64,
    /// Force thresholds of the dimer and the spring pair (`criteria=normal|tight|loose`).
    pub criteria: ConvergenceCriteria,
    /// iEIP: tolerance of the relaxation on hyperspheres (`micro_tolerance=`, 0.01).
    pub micro_tolerance: f64,
    /// Dimer: rotations per translation (`rotations=`, 8), the angle below which they stop
    /// (`rotation_angle=`, degrees, 1), the largest translation (`step=`, Bohr, 0.1).
    pub rotations: usize,
    /// Rotation angle tolerance (degrees).
    pub rotation_angle_deg: f64,
    /// Largest translation of the dimer and the largest step of the spring pair (Bohr).
    pub max_step: f64,
    /// Spring pair: natural length (`spm_length=`, Bohr, 0.1), offset of the second particle
    /// (`spm_offset=`, 0.1), rates of the drift (`spm_drift=`, 0.5), of the spring
    /// (`spm_spring=`, 1), and of the climb (`spm_climb=`, 5), tolerance of the drift
    /// (`spm_tolerance=`, 2e-3), drift steps per cycle (`spm_drift_steps=`, 200).
    pub spm: [f64; 6],
    /// Drift steps of the spring pair per cycle.
    pub spm_drift_steps: usize,
    /// ADDF: length along the stiffest mode on the first sphere (`addf_initial=`, Angstrom,
    /// 0.03), the angle that makes two ADDs one (`addf_angle=`, rad, 0.01), the largest
    /// gradient component at the start (`addf_gradient=`, 1e-3).
    pub addf: [f64; 3],
    /// Minimization on a sphere (ADDF, 2PSHS): evaluations at most (`sphere_iterations=`,
    /// 30) and the RMS of the gradient within the sphere at which it ends
    /// (`sphere_gradient=`, 1e-4).
    pub sphere_iterations: usize,
    /// RMS tolerance of the gradient within a sphere.
    pub sphere_gradient: f64,
    /// Growing Newton trajectory: RMS of the projected gradient at which the corrector ends
    /// (`gnt_tolerance=`, 1.7e-3).
    pub gnt_tolerance: f64,
    /// Length (Bohr; the unit of the variables otherwise) of a random displacement (seeded)
    /// of the second structure where it only gives a direction or a start (`-use_dimer`,
    /// `-use_spm`, `-gnt`, `-2pshs`), so that symmetric inputs (linear molecules) can leave
    /// their symmetry (`symmetry_noise=`, 0.05; 0: none).
    pub symmetry_noise: f64,
    /// The keys of the methods of the registry with their prefixes (`gsm_nodes=11`, ...).
    pub method_keys: Vec<(String, String)>,
    /// Dimer: the translation (`dimer_translation=lbfgs|plain|optimistic|extragradient`;
    /// the last three experimental).
    pub dimer_translation: crate::path::dimer::DimerTranslation,
    /// Dimer: the rates across and along the axis of the experimental translations
    /// (`dimer_rate_across=`, `dimer_rate_along=`, 0.5 each).
    pub dimer_rates: [f64; 2],
    /// The optimizations of the estimates after the search (`optimize=on`,
    /// `optimize_hessian=exact|none`, `optimize_check=on|off`, `optimize_irc=on|off`,
    /// `optimize_settings=FILE`).
    pub follow_up: super::ieipmain_followup::FollowUp,
}

impl IeipExtra {
    /// The keys for the registry method `name` (`gsm` and `se_gsm` take `gsm_`), without
    /// their prefix, and the keys of the relaxation of the path after any method
    /// (`method_relax*`, as they are).
    pub fn keys_of(&self, name: &str) -> Vec<(String, String)> {
        let prefix = match name {
            "se_gsm" => "gsm_".to_string(),
            other => format!("{other}_"),
        };
        let mut keys: Vec<(String, String)> = self
            .method_keys
            .iter()
            .filter_map(|(k, v)| k.strip_prefix(&prefix).map(|k| (k.to_string(), v.clone())))
            .collect();
        keys.extend(self.relax_keys());
        keys
    }

    /// The keys of the relaxation of the path after the method (`method_relax*`).
    pub fn relax_keys(&self) -> Vec<(String, String)> {
        self.method_keys
            .iter()
            .filter(|(k, _)| k.starts_with("method_relax"))
            .cloned()
            .collect()
    }
}

impl Default for IeipExtra {
    fn default() -> Self {
        Self {
            figures: true,
            seed: 1,
            criteria: ConvergenceCriteria::NORMAL,
            micro_tolerance: 0.01,
            rotations: 8,
            rotation_angle_deg: 1.0,
            max_step: 0.1,
            spm: [0.1, 0.1, 0.5, 1.0, 5.0, 2e-3],
            spm_drift_steps: 200,
            addf: [0.03, 0.01, 1e-3],
            sphere_iterations: 30,
            sphere_gradient: 1e-4,
            gnt_tolerance: 1.7e-3,
            symmetry_noise: 0.05,
            method_keys: Vec::new(),
            dimer_translation: crate::path::dimer::DimerTranslation::Lbfgs,
            dimer_rates: [0.5, 0.5],
            follow_up: super::ieipmain_followup::FollowUp::default(),
        }
    }
}

/// `-ieip_opt KEY=VALUE ...` on top of the defaults.
pub fn parse_extra(tokens: &[String]) -> MoptResult<IeipExtra> {
    let mut s = IeipExtra::default();
    for token in tokens {
        let (key, value) = token.split_once('=').ok_or_else(|| {
            MoptError::parse(format!("-ieip_opt expects KEY=VALUE, got '{token}'"))
        })?;
        let number = || -> MoptResult<f64> {
            value
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    MoptError::parse(format!("-ieip_opt {key}: '{value}' is not a number"))
                })
        };
        let count = || -> MoptResult<usize> {
            value.trim().parse::<usize>().map_err(|_| {
                MoptError::parse(format!("-ieip_opt {key}: '{value}' is not a whole number"))
            })
        };
        match key.trim().to_ascii_lowercase().as_str() {
            "figures" => {
                s.figures =
                    crate::core_infra::arg_parser::parse_switch("-ieip_opt figures", value)?;
            }
            "seed" => s.seed = count()? as u64,
            "criteria" => {
                s.criteria = match value.trim().to_ascii_lowercase().as_str() {
                    "normal" => ConvergenceCriteria::NORMAL,
                    "tight" => ConvergenceCriteria::TIGHT,
                    "loose" => ConvergenceCriteria::LOOSE,
                    _ => {
                        return Err(MoptError::parse(
                            "-ieip_opt criteria: normal, tight, or loose",
                        ))
                    }
                }
            }
            "micro_tolerance" => s.micro_tolerance = number()?,
            "rotations" => s.rotations = count()?,
            "rotation_angle" => s.rotation_angle_deg = number()?,
            "step" => s.max_step = number()?,
            "spm_length" => s.spm[0] = number()?,
            "spm_offset" => s.spm[1] = number()?,
            "spm_drift" => s.spm[2] = number()?,
            "spm_spring" => s.spm[3] = number()?,
            "spm_climb" => s.spm[4] = number()?,
            "spm_tolerance" => s.spm[5] = number()?,
            "spm_drift_steps" => s.spm_drift_steps = count()?,
            "addf_initial" => s.addf[0] = number()?,
            "addf_angle" => s.addf[1] = number()?,
            "addf_gradient" => s.addf[2] = number()?,
            "sphere_iterations" => s.sphere_iterations = count()?,
            "sphere_gradient" => s.sphere_gradient = number()?,
            "gnt_tolerance" => s.gnt_tolerance = number()?,
            "symmetry_noise" => s.symmetry_noise = number()?.max(0.0),
            "dimer_translation" => {
                s.dimer_translation = crate::path::dimer::DimerTranslation::from_name(value)?
            }
            "optimize" | "optimize_check" | "optimize_irc" => {
                let on = crate::core_infra::arg_parser::parse_switch(
                    &format!("-ieip_opt {key}"),
                    value,
                )?;
                match key.trim().to_ascii_lowercase().as_str() {
                    "optimize" => s.follow_up.on = on,
                    "optimize_check" => s.follow_up.check = on,
                    _ => s.follow_up.descend = on,
                }
            }
            "optimize_hessian" => {
                s.follow_up.exact_hessian = match value.trim().to_ascii_lowercase().as_str() {
                    "exact" => true,
                    "none" => false,
                    _ => {
                        return Err(MoptError::parse(
                            "-ieip_opt optimize_hessian: exact or none",
                        ))
                    }
                }
            }
            "optimize_settings" => {
                s.follow_up.settings = Some(std::path::PathBuf::from(value.trim()));
            }
            "dimer_rate_across" => s.dimer_rates[0] = number()?,
            "dimer_rate_along" => s.dimer_rates[1] = number()?,
            // The keys of the methods of the registry, checked there.
            other
                if ["gsm_", "fsm_", "dsafir_", "lmm_", "method_relax"]
                    .iter()
                    .any(|prefix| other.starts_with(prefix)) =>
            {
                s.method_keys
                    .push((other.to_string(), value.trim().to_string()));
            }
            other => {
                return Err(MoptError::parse(format!(
                    "-ieip_opt: unknown key '{other}' (figures, seed, criteria, micro_tolerance, \
                     rotations, rotation_angle, step, spm_length, spm_offset, spm_drift, \
                     spm_spring, spm_climb, spm_tolerance, spm_drift_steps, addf_initial, \
                     addf_angle, addf_gradient, sphere_iterations, sphere_gradient, \
                     gnt_tolerance, symmetry_noise, optimize, optimize_hessian, \
                     optimize_check, optimize_irc, optimize_settings, and gsm_*, fsm_*, dsafir_*)"
                )))
            }
        }
    }
    Ok(s)
}

/// Typed `ieipmain` settings.
#[derive(Debug, Clone)]
pub struct IeipmainConfig {
    /// The input as given (a directory or files).
    pub input: PathBuf,
    /// The structure (or variables) files, in order.
    pub inputs: Vec<PathBuf>,
    /// The settings in effect as a settings document (`settings.json` of the run).
    pub settings_document: String,
    /// Run directory to continue (`--restart`).
    pub restart_dir: Option<PathBuf>,
    /// The parsed arguments (for `-mf`, which runs `optmain`).
    pub arguments: ParsedArgs,
    /// The method.
    pub method: IeipMethod,
    /// `-mf NAME` with a method of mopt-rs: the method runs on this function of the states
    /// of the two structures.
    pub surface_function: Option<String>,
    /// Further methods asked for, which MultiOptPy's order passes over.
    pub ignored: Vec<String>,
    /// Calculator name and settings.
    pub calculator: String,
    /// Calculator settings.
    pub calculator_options: CalculatorOptions,
    /// `-elec`, `-spin` of the two states.
    pub states: Vec<StateSpec>,
    /// `-es` of the two states.
    pub excited_states: [usize; 2],
    /// `-ns`.
    pub max_iterations: usize,
    /// `-beta`.
    pub beta: f64,
    /// `-mi`.
    pub microiterations: usize,
    /// `-gnt_vec` as pairs of atoms (indices from 0).
    pub gnt_bonds: Vec<(usize, usize)>,
    /// `-gnt_step` (Bohr).
    pub gnt_step: f64,
    /// `-gnt_mi`.
    pub gnt_corrector: usize,
    /// `-addf_step` (Angstrom along the stiffest mode).
    pub addf_step: f64,
    /// `-addf_num`.
    pub addf_spheres: usize,
    /// `-addf_nadd`.
    pub addf_paths: usize,
    /// `-2pshs_step` (Angstrom along the stiffest mode).
    pub twopshs_step: f64,
    /// `-2pshs_num`.
    pub twopshs_spheres: usize,
    /// `-dimer_sep` (Bohr).
    pub dimer_separation: f64,
    /// `-dimer_trial_angle` (rad).
    pub dimer_trial_angle: f64,
    /// `-dimer_maxiter`.
    pub dimer_iterations: usize,
    /// `-ieip_opt`.
    pub extra: IeipExtra,
    /// Bias potentials.
    pub biases: Vec<BiasSpec>,
    /// `-fix` and `-gfix`.
    pub constraints: Vec<Constraint>,
    /// `-core`.
    pub threads: usize,
    /// `-debug`.
    pub debug: bool,
    /// `-coord`: the coordinates the search moves in (`cart`: Cartesian).
    pub coordinates: String,
    /// `-vc` (with `-vc_pressure`, `-vc_weight`): the search in the variables of the atoms and
    /// the cell (`cli::ieipmain_cell`); `None`: the cell of the structures is held.
    pub variable_cell: Option<super::variable_cell::VariableCellSettings>,
}

/// The method, what else was asked for and does not run, and the model function that is
/// the surface of the method: `-mf NAME` with one of the methods of mopt-rs (`-dsafir`,
/// `-gsm`, `-se_gsm`, `-fsm`, `-lmm`) runs that method on the function of the states of
/// the two structures. Among MultiOptPy's methods its order holds (`-mf` first, which then
/// runs `optmain`).
fn chosen_method(p: &ParsedArgs) -> (IeipMethod, Vec<String>, Option<String>) {
    let function = p.get_opt_str("model_function_mode").map(str::to_string);
    let own = [
        p.get_bool("use_gsm", false),
        p.is_present("use_se_gsm"),
        p.get_bool("use_fsm", false),
        p.get_bool("use_dsafir", false),
        p.get_bool("use_lmm", false),
    ]
    .contains(&true);
    let on_function = function.clone().filter(|_| own);
    let mut asked = Vec::new();
    if let (Some(name), None) = (&function, &on_function) {
        asked.push(IeipMethod::ModelFunction(name.clone()));
    }
    for (key, m) in [
        ("use_gnt", IeipMethod::NewtonTrajectory),
        ("use_addf", IeipMethod::Addf),
        ("use_2pshs", IeipMethod::TwoPshs),
        ("use_dimer", IeipMethod::Dimer),
        ("use_spm", IeipMethod::SpringPair),
    ] {
        if p.get_bool(key, false) {
            asked.push(m);
        }
    }
    // The methods of mopt-rs come after MultiOptPy's.
    if p.get_bool("use_gsm", false) {
        asked.push(IeipMethod::Gsm);
    }
    if p.is_present("use_se_gsm") {
        asked.push(IeipMethod::SeGsm(p.get_list("use_se_gsm").to_vec()));
    }
    if p.get_bool("use_fsm", false) {
        asked.push(IeipMethod::Fsm);
    }
    if p.get_bool("use_dsafir", false) {
        asked.push(IeipMethod::DsAfir);
    }
    if p.get_bool("use_lmm", false) {
        asked.push(IeipMethod::Lmm);
    }
    if on_function.is_some() {
        // The method of mopt-rs runs; MultiOptPy's are named as not run.
        asked.sort_by_key(|m| m.registry_name().is_none());
    }
    let mut asked = asked.into_iter();
    let method = asked.next().unwrap_or(IeipMethod::Ieip);
    (
        method,
        asked.map(|m| m.title().to_string()).collect(),
        on_function,
    )
}

/// The structures of a directory: `*_[A-Z].xyz`, in the order of their letters (`_A` first),
/// then by name.
pub fn structures_of_directory(dir: &Path) -> MoptResult<Vec<PathBuf>> {
    let letter = |path: &PathBuf| -> Option<char> {
        let name = path.file_name().and_then(|n| n.to_str())?;
        let mut tail = name.strip_suffix(".xyz")?.chars().rev();
        match (tail.next(), tail.next()) {
            (Some(c), Some('_')) if c.is_ascii_uppercase() => Some(c),
            _ => None,
        }
    };
    let mut found: Vec<(char, PathBuf)> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter_map(|path| letter(&path).map(|c| (c, path)))
        .collect();
    found.sort();
    Ok(found.into_iter().map(|(_, path)| path).collect())
}

fn parse_bonds(text: Option<&str>) -> MoptResult<Vec<(usize, usize)>> {
    let Some(text) = text else {
        return Ok(Vec::new());
    };
    let atoms = parse_atom_indices_one_based(text)?;
    if atoms.is_empty() || atoms.len() % 2 != 0 {
        return Err(MoptError::parse(
            "-gnt_vec takes pairs of atoms (an even number), e.g. 1,2 or 1,2,3,4",
        ));
    }
    Ok(atoms.chunks(2).map(|c| (c[0] - 1, c[1] - 1)).collect())
}

/// `-dimer_trial_angle`: a number, or MultiOptPy's default text `np.pi / 32.0`.
fn trial_angle(p: &ParsedArgs) -> MoptResult<f64> {
    let text = p.get_str("dimer_trial_angle", "np.pi / 32.0").trim();
    if text.replace(' ', "") == "np.pi/32.0" {
        return Ok(std::f64::consts::PI / 32.0);
    }
    text.parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v > 0.0 && *v < std::f64::consts::FRAC_PI_2)
        .ok_or_else(|| {
            MoptError::parse(format!(
                "-dimer_trial_angle: '{text}' is not an angle in radians between 0 and pi/2"
            ))
        })
}

impl IeipmainConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        reject_unavailable(p, IEIPMAIN)?;
        let (method, ignored, surface_function) = chosen_method(p);
        let input = match p.positionals.as_slice() {
            [] => {
                return Err(MoptError::config(
                    "ieipmain needs an input: a directory with *_A.xyz and *_B.xyz, or one or \
                     two structure or variables files",
                ))
            }
            [one, ..] => PathBuf::from(one),
        };
        let mut inputs: Vec<PathBuf> = Vec::new();
        for name in &p.positionals {
            let path = PathBuf::from(name);
            if path.is_dir() {
                inputs.extend(structures_of_directory(&path)?);
            } else {
                let found = crate::core_infra::glob::expand(name)?;
                if found.is_empty() {
                    return Err(MoptError::config(format!("no file matches '{name}'")));
                }
                inputs.extend(found);
            }
        }
        let gnt_bonds = parse_bonds(p.get_opt_str("gnt_vec"))?;
        let (least, most) = method.inputs(!gnt_bonds.is_empty());
        if inputs.len() < least || inputs.len() > most {
            return Err(MoptError::config(format!(
                "the {} takes {least}{} structure(s); {} were found in {}",
                method.title(),
                if most > least {
                    format!(" or {most}")
                } else {
                    String::new()
                },
                inputs.len(),
                input.display()
            )));
        }
        let excited = p.get_list("excited_state");
        let es = |k: usize| -> MoptResult<usize> {
            match excited.get(k).or_else(|| excited.first()) {
                Some(t) => t
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| MoptError::parse(format!("-es: '{t}' is not a whole number"))),
                None => Ok(0),
            }
        };
        Ok(Self {
            input,
            inputs,
            settings_document: settings_json(p, IEIPMAIN).to_string_pretty(),
            restart_dir: None,
            arguments: p.clone(),
            method,
            ignored,
            surface_function,
            calculator: calculator_name(p)?,
            calculator_options: calculator_options(p)?,
            states: state_specs(p, 2)?,
            excited_states: [es(0)?, es(1)?],
            max_iterations: p.get_usize("NSTEP", 999)?.max(1),
            beta: p.get_f64("BETA", 1.0)?,
            microiterations: p.get_usize("microiter", 0)?,
            gnt_bonds,
            gnt_step: p.get_f64("gnt_step_len", 0.5)?,
            gnt_corrector: p.get_usize("gnt_microiter", 25)?.max(1),
            addf_step: p.get_f64("addf_step_size", 0.1)?,
            addf_spheres: p.get_usize("addf_step_num", 300)?,
            addf_paths: p.get_usize("number_of_add", 5)?,
            twopshs_step: p.get_f64("twoPshs_step_size", 0.05)?,
            twopshs_spheres: p.get_usize("twoPshs_step_num", 300)?,
            dimer_separation: p.get_f64("dimer_separation", 1e-4)?,
            dimer_trial_angle: trial_angle(p)?,
            dimer_iterations: p.get_usize("dimer_max_iterations", 1000)?,
            extra: parse_extra(p.get_list("ieip_options"))?,
            biases: parse_biases(p)?,
            constraints: parse_constraints(p)?,
            threads: p.get_usize("N_THREAD", 8)?.max(1),
            debug: p.get_bool("debug", false),
            coordinates: p.get_str("coordinate_system", "cart").trim().to_string(),
            variable_cell: super::variable_cell::VariableCellSettings::from_args(p)?,
        })
    }

    /// The length along the stiffest mode per sphere of ADDF in Bohr.
    pub fn addf_step_bohr(&self) -> f64 {
        self.addf_step * Units::ANGSTROM_TO_BOHR
    }
}

/// Settings for `args` (tests and embedding code).
pub fn config_from_cli(args: &[&str]) -> MoptResult<IeipmainConfig> {
    let p = super::option::parser("ieipmain", IEIPMAIN).parse(args)?;
    IeipmainConfig::from_args(&p)
}
