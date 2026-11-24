//! The settings of `nebmain` from its options.
//!
//! The default iteration limits are 30 for NEB and 60 for direct MaxFlux. Image spacing
//! and redistribution follow the articles of MultiOptPy's author (`-nd 0.5 -adpred 2`),
//! with the model Hessian `lindh2007d3` in place of `o1numhess`. MultiOptPy's defaults are
//! `-ns 10 -modelhess none -nd 0 -adpred 0`. `-neb_opt KEY=VALUE` has what MultiOptPy has
//! no option for.
//!
//! MultiOptPy: `interface.py:call_nebparser`, `neb.py:NEBConfig`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use super::bias::{parse_biases, BiasSpec};
use super::calculator::{calculator_name, calculator_options};
use super::charge_spin::{state_spec, StateSpec};
use super::constraints::parse_constraints;
use super::nebmain_followup::{CheckedImages, FirstHessian, FollowUp, OptimizedImages};
use super::option::reject_unavailable;
use super::settings::settings_json;
use super::spec::NEBMAIN;
use crate::core_infra::arg_parser::ParsedArgs;
use crate::error::{MoptError, MoptResult};
use crate::path::band::diis::ChainDiisSettings;
use crate::path::band::global::GlobalRfoSettings;
use crate::path::band::images::{AddedImages, FailedImage};
use crate::path::band::lup::{check_locally_updated_planes, locally_updated_planes};
use crate::path::band::machine::{ChainSettings, ChainTangent};
use crate::path::band::redistribute::Gathering;
use crate::path::band::rfo::ChainRfoSettings;
use crate::path::band::springs::EnergyWeightedSprings;
use crate::path::band::surrogate::{Features, SurrogateSettings};
use crate::path::band::trust::{FixedEnds, ImageCriteria};
use crate::path::band::Compatibility;
use crate::path::interp::FirstPath;
use crate::pes::constraint::Constraint;
use crate::pes::registry::CalculatorOptions;

/// Maximum NEB iterations without `-ns`.
pub const DEFAULT_ITERATIONS: usize = 30;
/// Maximum direct MaxFlux iterations without `-ns`.
pub const DEFAULT_DMF_ITERATIONS: usize = 60;
/// Distance of the images (Angstrom) without `-nd`, for structures (the articles).
pub const DEFAULT_SPACING_ANGSTROM: f64 = 0.5;
/// Redistribution towards the predicted maxima every this many iterations without
/// `-adpred` (the articles).
pub const DEFAULT_PREDICTED_EVERY: usize = 2;
/// `-modelhess` without the option: a model Hessian, which costs no calculation. The
/// articles use `o1numhess` (a Hessian from further gradients of every image), which can
/// be asked for. `docs/source/migration_paths.md` ("Chain relaxation") has the iterations
/// measured with the model Hessians and with `o1numhess`.
pub const DEFAULT_MODEL_HESSIAN: &str = "lindh2007d3";
/// Length scale of the Gaussian process of `-neb_opt gp=on` without `gp_length=` when
/// it sees points (Angstrom for structures; 0.4 as in the Gaussian-process optimizer).
pub const DEFAULT_GP_LENGTH: f64 = 0.4;
/// The same when it sees inverse distances (1/Bohr): the best of 0.02 to 0.5 for HCN
/// (`path::band::surrogate::tests`).
pub const DEFAULT_GP_LENGTH_DISTANCES: f64 = 0.1;
/// How the images between the structures that are read are made without `-neb_opt
/// path=`: on the geodesic in scaled interatomic distances. Measured against the straight
/// line (MultiOptPy's way), the linear synchronous transit, and the two forms of the
/// image dependent pair potential in `docs/source/migration_paths.md` ("The first chain").
pub const DEFAULT_FIRST_PATH: FirstPath = FirstPath::Geodesic;
/// Largest force of the images that move at which the run ends without `-neb_opt
/// tolerance=` (Hartree/Bohr): the largest force at which an optimization of `optmain`
/// is converged. `docs/source/migration_paths.md` ("Chain relaxation") has the iterations and
/// barriers measured with it and with other values.
pub const DEFAULT_FORCE_TOLERANCE: f64 = 3e-4;
/// MultiOptPy's value of `-cineb` and `-sd` that means "never".
const NEVER: usize = 99_999;
/// `-lup` without `-adpred` or `-ad`: the images are put back along the path towards the
/// predicted maxima every this many iterations (`None`: never, the original method). Chosen
/// by measurement on the test paths of `nebmain` (`docs/source/reaction_paths.md`, "The
/// locally updated planes").
pub const LUP_PREDICTED_EVERY: Option<usize> = Some(DEFAULT_PREDICTED_EVERY);

/// Typed `nebmain` settings.
#[derive(Debug, Clone)]
pub struct NebmainConfig {
    /// Inputs: structures (a file of several, several files, a directory) or variables
    /// files.
    pub inputs: Vec<PathBuf>,
    /// The settings in effect as a settings document (`settings.json` of the run).
    pub settings_document: String,
    /// Run directory to continue (`--restart`); `None`: a new one.
    pub restart_dir: Option<PathBuf>,
    /// Calculator name (`-os`, `-xtb`, `-dxtb`; default PySCF).
    pub calculator: String,
    /// Calculator settings.
    pub calculator_options: CalculatorOptions,
    /// `-elec` / `-spin`.
    pub state: StateSpec,
    /// `-es`: the electronic state (0: the ground state).
    pub excited_state: usize,
    /// Bias potentials (`-ma`, `-kp`, ...).
    pub biases: Vec<BiasSpec>,
    /// Constraints (`-fix`, `-pc`, `-gfix`, `-uc`).
    pub constraints: Vec<Constraint>,
    /// `-mf NAME [CHARGE MULTIPLICITY]`: the chain is relaxed on a surface that combines
    /// two electronic states, as in `optmain` (empty: on the surface of one state).
    pub model_function: Vec<String>,
    /// The relaxation (`-ns`, `-fe`, `-cineb`, `-sd`, `-aconv`, `-ad`, `-adpred`, `-rrs`,
    /// `-mfc` / `-fc`, `-neb_opt`). Kind, fixed components, and the directory of the
    /// Hessians are set when the input is read.
    pub chain: ChainSettings,
    /// `-p`: images put between every two points that follow each other.
    pub inserted: usize,
    /// `-ad` given: the first chain is spread evenly.
    pub first_equal_spacing: bool,
    /// `-nd` (in the unit of the input: Angstrom for structures): `None` when it was not
    /// given, `Some(0)` to keep the images as they are.
    pub spacing: Option<f64>,
    /// `-ndb` (Angstrom for structures): the first chain is put on the Bezier curve of its
    /// images, with as many images as this distance holds in its length.
    pub bezier_spacing: Option<f64>,
    /// `-modelhess`: where the Hessians of the images come from (`None`: no Hessians, the
    /// steps along the forces alone; `o1numhess`: from gradients). Ignored with `-fc`.
    pub model_hessian: Option<String>,
    /// `-fc N`: Hessians of the calculator every `N` iterations.
    pub exact_hessians: bool,
    /// The component of the gradient along the direction of every atom is removed, as
    /// MultiOptPy does, not along the one direction of the whole structure
    /// (`-neb_opt projection=atom`).
    pub per_atom: bool,
    /// Figures after every iteration: on by default; `-spng off` (`"save_pict": false`) or
    /// `-neb_opt figures=off` turns them off.
    pub figures: bool,
    /// `-coord`: the coordinates the chain of structures is relaxed in (`cart`, or the
    /// independent internal coordinates `dic`, `tric`, `tric-cart[:WEIGHT]`).
    pub coordinates: String,
    /// A Gaussian-process model of the surface that answers for images near calculated
    /// points (`-neb_opt gp=on`, `gp_uncertainty=`, `gp_points=`, `gp_memory=`, ...). Its
    /// length scale and what it sees of a point are in the next two fields.
    pub surrogate: Option<SurrogateSettings>,
    /// `gp_length=`: the length scale of the model, in the unit of what it sees (1/Bohr
    /// for inverse distances, Angstrom for structures as points, the unit of the
    /// variables otherwise); `None`: [`DEFAULT_GP_LENGTH_DISTANCES`] or
    /// [`DEFAULT_GP_LENGTH`].
    pub surrogate_length: Option<f64>,
    /// `gp_features=`: what the model sees of a structure (`points`, the default;
    /// `distances`: the inverse interatomic distances, for structures in Cartesian
    /// coordinates without a periodic cell).
    pub surrogate_features: Option<Features>,
    /// How the images between the structures that are read are made (`-neb_opt path=`;
    /// `-idpp` is `path=idpp`, `-cfbenm` is `path=cfbenm`). `None`: nothing was asked for, which is
    /// [`DEFAULT_FIRST_PATH`] for structures and the straight line for plain variables
    /// and for `form=multioptpy`.
    pub first_path: Option<FirstPath>,
    /// Optimizations of the candidates for stationary points with `optmain` after the
    /// relaxation (`-neb_opt optimize=`, `optimize_hessian=`, `optimize_settings=`; `None`
    /// without them).
    pub follow_up: Option<FollowUp>,
    /// `-dmf` and the `dmf_` keys of `-neb_opt`: the path by the direct MaxFlux method
    /// instead of a chain relaxation (`None` without `-dmf`).
    pub dmf: Option<crate::path::dmf::DmfSettings>,
    /// `-core`.
    pub threads: usize,
    /// `-vc` (with `-vc_pressure`, `-vc_weight`): the chain of the atoms and the cell
    /// (`cli::nebmain_cell`); `None`: every image in the cell of the first structure.
    pub variable_cell: Option<super::variable_cell::VariableCellSettings>,
}

/// One `KEY=VALUE` of `-neb_opt`.
fn neb_option(text: &str) -> MoptResult<(&str, &str)> {
    text.split_once('=')
        .map(|(key, value)| (key.trim(), value.trim()))
        .filter(|(key, value)| !key.is_empty() && !value.is_empty())
        .ok_or_else(|| MoptError::parse(format!("-neb_opt: '{text}' is not KEY=VALUE")))
}

fn number(key: &str, value: &str) -> MoptResult<f64> {
    value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| MoptError::parse(format!("-neb_opt {key}: '{value}' is not a number")))
}

/// A count given as text; `None` when the option is absent.
fn count(p: &ParsedArgs, key: &str, flag: &str) -> MoptResult<Option<usize>> {
    match p.get_opt_str(key) {
        None => Ok(None),
        Some(text) => text
            .parse::<usize>()
            .map(Some)
            .map_err(|_| MoptError::parse(format!("{flag}: '{text}' is not a whole number"))),
    }
}

impl NebmainConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        // The state that the input file names, for the runs started on files written here.
        let with_state = super::charge_spin::with_state_of_input(p);
        let p = &with_state;
        reject_unavailable(p, NEBMAIN)?;
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
                "nebmain needs structures (a file of several, several files, or a directory of \
                 numbered files) or two or more variables files",
            ));
        }
        Self::with_inputs(p, inputs)
    }

    /// The settings of the options of `p` alone, without inputs (`inputs` is empty): for a
    /// caller that has the structures or points itself (the Python `relax_band`), with the
    /// same defaults as the command.
    pub fn from_options(p: &ParsedArgs) -> MoptResult<Self> {
        reject_unavailable(p, NEBMAIN)?;
        Self::with_inputs(p, Vec::new())
    }

    /// The settings of `p` for the inputs `inputs`.
    fn with_inputs(p: &ParsedArgs, inputs: Vec<PathBuf>) -> MoptResult<Self> {
        // Where the Hessians come from: -fc before -modelhess, as in MultiOptPy.
        let exact_every = p
            .get_opt_str("calc_exact_hess")
            .map(|text| {
                text.parse::<i64>()
                    .map_err(|_| MoptError::parse(format!("-fc: '{text}' is not a whole number")))
            })
            .transpose()?
            .filter(|every| *every > 0)
            .map(|every| every as usize);
        // `-modelhess none` (MultiOptPy's default) is not the same as no `-modelhess`.
        let model_hessian = if p.is_present("use_model_hessian") {
            let name = p.get_str("use_model_hessian", "");
            let none = ["", "none", "off", "no", "0"];
            (!none.contains(&name.to_ascii_lowercase().as_str())).then(|| name.to_string())
        } else {
            Some(DEFAULT_MODEL_HESSIAN.to_string())
        };
        let with_hessians = exact_every.is_some() || model_hessian.is_some();
        let model_every = p.get_usize("calc_model_hess", 50)?;

        let dmf = super::nebmain_dmf::settings_from(p)?;
        let max_iterations = match &dmf {
            Some(settings) => settings.max_iterations,
            None => count(p, "NSTEP", "-ns")?.unwrap_or(DEFAULT_ITERATIONS),
        };
        let mut chain = ChainSettings {
            max_iterations,
            ends: FixedEnds::from_code(p.get_usize("fixedges", 0)? as i64),
            climb_after: count(p, "apply_CI_NEB", "-cineb")?.filter(|n| *n < NEVER),
            steepest_from: count(p, "steepest_descent", "-sd")?.filter(|n| *n < NEVER),
            hold_converged: p
                .get_bool("apply_convergence_criteria", false)
                .then(ImageCriteria::of_multioptpy),
            equal_spacing_every: count(p, "align_distances", "-ad")?.filter(|n| *n > 0),
            predicted_maxima_every: match count(p, "align_distances_energy_predicted", "-adpred")? {
                // Springs keep the images apart: no redistribution unless asked for.
                None if p.get_bool("EWBNEB", false) => None,
                None => Some(DEFAULT_PREDICTED_EVERY),
                Some(0) => None,
                Some(every) => Some(every),
            },
            springs: p
                .get_bool("EWBNEB", false)
                .then(EnergyWeightedSprings::default),
            gathering: Some(Gathering {
                weight: 1.0,
                width: 1.0,
            }),
            rfo: with_hessians.then(|| ChainRfoSettings {
                ratio: 0.5,
                ..ChainRfoSettings::default()
            }),
            hessian_every: with_hessians.then_some(exact_every.unwrap_or(model_every).max(1)),
            force_tolerance: Some(DEFAULT_FORCE_TOLERANCE),
            ..ChainSettings::default()
        };
        if let Some(rfo) = chain.rfo.as_mut() {
            rfo.ratio = p.get_f64("ratio_of_rfo_step", 0.5)?;
            if !(0.0..=1.0).contains(&rfo.ratio) {
                return Err(MoptError::config(format!(
                    "-rrs: the share of the quasi-Newton step is between 0 and 1, not {}",
                    rfo.ratio
                )));
            }
        }
        // -aneb N FREQ: N images on either side of the maxima after every FREQ iterations.
        let adaptive = p.get_list("ANEB");
        if !adaptive.is_empty() {
            let whole = |text: &String| {
                text.parse::<usize>()
                    .ok()
                    .filter(|v| *v >= 1)
                    .ok_or_else(|| {
                        MoptError::parse(format!(
                            "-aneb: '{text}' is not a whole number of 1 or more"
                        ))
                    })
            };
            let [images, every] = adaptive else {
                return Err(MoptError::parse(
                    "-aneb takes two numbers: the images added on either side of a maximum, and the iterations between two additions",
                ));
            };
            chain.added = Some(AddedImages {
                images: whole(images)?,
                every: whole(every)?,
            });
        }
        if p.get_bool("global_quasi_newton", false) {
            // MultiOptPy reads -gqnt and does nothing with it; here it is the step for
            // the whole chain.
            if let Some(rfo) = chain.rfo.as_mut() {
                rfo.global = Some(GlobalRfoSettings::default());
            }
        }
        // The figures of the chain: on by default, unlike MultiOptPy, which draws them only
        // with -spng (the owner's choice for mopt-rs); `-spng off` (or `"save_pict": false` in
        // a settings file) switches them off, `-spng` alone keeps MultiOptPy's meaning "on".
        let figures = match p.get_list("save_pict").first() {
            Some(value) => {
                crate::core_infra::arg_parser::parse_switch("-spng", value).map_err(|e| {
                    MoptError::parse(format!(
                        "{e}; -spng takes an optional on/off (put the input files before \
                         -spng, or write -spng=on)"
                    ))
                })?
            }
            None => true,
        };
        let (mut per_atom, mut figures) = (false, figures);
        // MultiOptPy's quadratic string forms, on the chain of mopt-rs: -qsm is its
        // projection atom by atom, -qsmv2 its tangents carried from the highest image.
        let (qsm, qsmv2) = (p.get_bool("QSM", false), p.get_bool("QSMv2", false));
        if qsm && qsmv2 {
            return Err(MoptError::config(
                "-qsm and -qsmv2 are two forms of one method; give one of them",
            ));
        }
        let mut projection_given = false;
        if qsmv2 {
            chain.tangent = ChainTangent::CarriedFromTop;
        }
        let mut tolerance_given = false;
        let springs_given = p
            .get_list("neb_options")
            .iter()
            .any(|text| text.starts_with("spring_"));
        let mut optimized = None;
        let mut with_model = false;
        let mut model = SurrogateSettings::default();
        let (mut model_length, mut model_features) = (None, None);
        let mut first_path = p
            .get_bool("use_image_dependent_pair_potential", false)
            .then_some(FirstPath::Idpp);
        if p.get_bool("use_correlated_flat_bottom_elastic_network_model", false) {
            // MultiOptPy applies it after -idpp; here one way makes the images.
            first_path = Some(FirstPath::Cfbenm);
        }
        let mut first_hessian = FirstHessian::Chain;
        let mut optimize_settings = None;
        let mut optimize_check = CheckedImages::Saddles;
        for text in p.get_list("neb_options") {
            let (key, value) = neb_option(text)?;
            match key {
                "tolerance" => {
                    let tolerance = number(key, value)?;
                    chain.force_tolerance = (tolerance > 0.0).then_some(tolerance);
                    tolerance_given = true;
                }
                "gather" => {
                    let weight = number(key, value)?;
                    let width = chain.gathering.map_or(1.0, |g| g.width);
                    chain.gathering = (weight > 0.0).then_some(Gathering { weight, width });
                }
                "gather_width" => {
                    let width = number(key, value)?;
                    if let Some(gathering) = chain.gathering.as_mut() {
                        gathering.width = width;
                    }
                }
                "gp" => {
                    with_model = match value {
                        "on" => true,
                        "off" => false,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt gp: on or off, not '{other}'"
                            )))
                        }
                    }
                }
                "gp_uncertainty" => {
                    model.uncertainty = number(key, value)?;
                    if !(0.0..1.0).contains(&model.uncertainty) {
                        return Err(MoptError::parse(format!(
                            "-neb_opt gp_uncertainty: between 0 and 1, not {value}"
                        )));
                    }
                }
                "gp_length" => {
                    let length = number(key, value)?;
                    if length <= 0.0 {
                        return Err(MoptError::parse(format!(
                            "-neb_opt gp_length: a positive length, not {value}"
                        )));
                    }
                    model_length = Some(length);
                }
                "gp_features" => {
                    model_features = Some(match value {
                        "distances" => Features::InverseDistances,
                        "points" => Features::Points,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt gp_features: distances or points, not '{other}'"
                            )))
                        }
                    });
                }
                "gp_points" => model.neighbours = (number(key, value)? as usize).max(2),
                "gp_between" => model.between = number(key, value)? as usize,
                "gp_error" => model.error_share = number(key, value)?,
                "gp_reach" => model.reach = number(key, value)?,
                "gp_memory" => model.memory = (number(key, value)? as usize).max(1),
                "spring_upper" | "spring_lower" => {
                    let constant = number(key, value)?;
                    let Some(springs) = chain.springs.as_mut() else {
                        return Err(MoptError::parse(format!(
                            "-neb_opt {key}: the chain has no springs (-ewbneb)"
                        )));
                    };
                    if constant < 0.0 {
                        return Err(MoptError::parse(format!(
                            "-neb_opt {key}: a spring constant is not negative"
                        )));
                    }
                    if key == "spring_upper" {
                        springs.upper = constant;
                    } else {
                        springs.lower = constant;
                    }
                }
                "pairs" => {
                    let pairs = number(key, value)?;
                    if let Some(rfo) = chain.rfo.as_mut() {
                        rfo.pairs = (pairs > 0.0).then_some(pairs as usize);
                    }
                }
                "failed" => {
                    chain.on_failure = match value {
                        "retry" => FailedImage::Retry,
                        "drop" => FailedImage::Drop,
                        "stop" => FailedImage::Stop,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt failed: retry, drop, or stop, not '{other}'"
                            )))
                        }
                    }
                }
                "global" => {
                    let on = match value {
                        "on" => true,
                        "off" => false,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt global: on or off, not '{other}'"
                            )))
                        }
                    };
                    if let Some(rfo) = chain.rfo.as_mut() {
                        rfo.global = on.then(|| rfo.global.unwrap_or_default());
                    }
                }
                "sweep" => {
                    let on = match value {
                        "on" => true,
                        "off" => false,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt sweep: on or off, not '{other}'"
                            )))
                        }
                    };
                    if let Some(rfo) = chain.rfo.as_mut() {
                        rfo.sweep = on;
                    }
                }
                "global_memory" => {
                    let memory_mb = number(key, value)?;
                    if let Some(global) = chain.rfo.as_mut().and_then(|rfo| rfo.global.as_mut()) {
                        global.memory_mb = memory_mb;
                    }
                }
                "diis" => {
                    let points = number(key, value)?;
                    let kept = chain.diis.unwrap_or_default();
                    chain.diis = (points >= 2.0).then_some(ChainDiisSettings {
                        points: points as usize,
                        ..kept
                    });
                }
                "diis_start" => {
                    let start_force = number(key, value)?;
                    if let Some(diis) = chain.diis.as_mut() {
                        diis.start_force = start_force;
                    }
                }
                "tangent" => {
                    chain.tangent = match value {
                        "higher" => ChainTangent::HigherNeighbour,
                        "top" => ChainTangent::CarriedFromTop,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt tangent: higher or top, not '{other}'"
                            )))
                        }
                    }
                }
                "projection" => {
                    projection_given = true;
                    per_atom = match value {
                        "whole" => false,
                        "atom" => true,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt projection: atom or whole, not '{other}'"
                            )))
                        }
                    }
                }
                "path" => first_path = Some(FirstPath::from_name(value)?),
                "optimize" => {
                    optimized = match value {
                        "on" | "all" => Some(OptimizedImages::All),
                        "saddles" => Some(OptimizedImages::Saddles),
                        "minima" => Some(OptimizedImages::Minima),
                        "off" => None,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt optimize: on, saddles, minima, or off, not '{other}'"
                            )))
                        }
                    }
                }
                "optimize_hessian" => {
                    first_hessian = match value {
                        "chain" => FirstHessian::Chain,
                        "exact" => FirstHessian::Exact,
                        "none" => FirstHessian::None,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt optimize_hessian: chain, exact, or none, not '{other}'"
                            )))
                        }
                    }
                }
                "optimize_settings" => optimize_settings = Some(PathBuf::from(value)),
                "optimize_check" => {
                    optimize_check = match value {
                        "saddles" => CheckedImages::Saddles,
                        "all" | "on" => CheckedImages::All,
                        "off" => CheckedImages::Off,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt optimize_check: saddles, all, or off, not '{other}'"
                            )))
                        }
                    }
                }
                "form" => {
                    chain.compatibility = match value {
                        "multioptpy" => Compatibility::MultiOptPy,
                        "consistent" => Compatibility::Consistent,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt form: consistent or multioptpy, not '{other}'"
                            )))
                        }
                    }
                }
                "figures" => {
                    figures = match value {
                        "on" => true,
                        "off" => false,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-neb_opt figures: on or off, not '{other}'"
                            )))
                        }
                    }
                }
                // Read with -dmf.
                dmf if dmf.starts_with("dmf_") => {}
                other => {
                    return Err(MoptError::parse(format!(
                        "-neb_opt: no setting '{other}' (tolerance, gather, gather_width, pairs, failed, sweep, global, global_memory, diis, diis_start, \
                         projection, tangent, form, figures, path, optimize, optimize_hessian, \
                         optimize_settings, optimize_check, gp, gp_uncertainty, gp_length, gp_points, \
                         gp_memory, gp_between, gp_error, gp_reach, gp_features, \
                         spring_upper, spring_lower, and with -dmf dmf_beta, dmf_points, \
                         dmf_guards, dmf_segments, dmf_moves, dmf_tolerance, dmf_step)"
                    )))
                }
            }
        }
        if chain.compatibility == Compatibility::MultiOptPy && !tolerance_given {
            // MultiOptPy runs all its iterations.
            chain.force_tolerance = None;
        }
        if let (Compatibility::MultiOptPy, Some(springs), false) =
            (chain.compatibility, chain.springs.as_mut(), springs_given)
        {
            // MultiOptPy's constants with MultiOptPy's form of the springs.
            *springs = EnergyWeightedSprings::of_multioptpy();
        }
        if chain.compatibility == Compatibility::MultiOptPy && first_path.is_none() {
            // MultiOptPy puts the images on the straight line.
            first_path = Some(FirstPath::Linear);
        }
        if qsm {
            if projection_given && !per_atom {
                return Err(MoptError::config(
                    "-qsm is the projection atom by atom; with projection=whole leave it out",
                ));
            }
            per_atom = true;
        }
        if chain.tangent == ChainTangent::CarriedFromTop && per_atom {
            return Err(MoptError::config(
                "the tangents carried from the highest image (-qsmv2, tangent=top) are those of \
                 whole images; not with the projection atom by atom (-qsm, projection=atom)",
            ));
        }
        if (qsm || qsmv2) && chain.compatibility == Compatibility::MultiOptPy {
            return Err(MoptError::unsupported(
                "-qsm and -qsmv2 run on the chain of mopt-rs; MultiOptPy's own step rule for \
                 them is not reproduced (form=multioptpy)",
            ));
        }
        if p.get_bool("LUP", false) {
            // -lup: the locally updated planes, the chain relaxed by the true force across
            // the path alone (path::band::lup). What they do not have is refused by name.
            if qsm || qsmv2 {
                return Err(MoptError::config(
                    "-lup and -qsm/-qsmv2 are different chain methods; give one of them",
                ));
            }
            if per_atom {
                return Err(MoptError::config(
                    "-lup: in the locally updated planes the planes are those of whole \
                     images (not projection=atom)",
                ));
            }
            check_locally_updated_planes(&chain)?;
            // The images are put back along the path as measured
            // (docs/source/reaction_paths.md, "The locally updated planes"), unless the
            // redistributions are given.
            let given =
                p.is_present("align_distances_energy_predicted") || p.is_present("align_distances");
            if !given {
                chain.predicted_maxima_every = LUP_PREDICTED_EVERY;
            }
            chain = locally_updated_planes(chain);
        }
        let spacing = p
            .get_opt_str("node_distance")
            .map(|text| {
                text.parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or_else(|| {
                        MoptError::parse(format!("-nd: '{text}' is not a distance (0 or more)"))
                    })
            })
            .transpose()?;
        Ok(Self {
            inputs,
            settings_document: settings_json(p, NEBMAIN).to_string_pretty(),
            restart_dir: None,
            calculator: calculator_name(p)?,
            calculator_options: calculator_options(p)?,
            state: state_spec(p)?,
            excited_state: p.get_usize("excited_state", 0)?,
            model_function: p.get_list("model_function").to_vec(),
            biases: parse_biases(p)?,
            constraints: parse_constraints(p)?,
            chain,
            inserted: p.get_usize("partition", 0)?,
            first_equal_spacing: p.get_usize("align_distances", 0)? > 0,
            spacing,
            bezier_spacing: p
                .get_opt_str("node_distance_bernstein")
                .map(|text| {
                    text.parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .ok_or_else(|| {
                            MoptError::parse(format!("-ndb: '{text}' is not a distance above 0"))
                        })
                })
                .transpose()?,
            model_hessian,
            exact_hessians: exact_every.is_some(),
            per_atom,
            figures,
            coordinates: p.get_str("coordinate_system", "cart").to_string(),
            surrogate: with_model.then_some(model),
            surrogate_length: model_length,
            surrogate_features: model_features,
            first_path,
            follow_up: optimized.map(|images| FollowUp {
                images,
                hessian: first_hessian,
                settings: optimize_settings,
                check: optimize_check,
            }),
            dmf,
            threads: p.get_usize("N_THREAD", 8)?.max(1),
            variable_cell: super::variable_cell::VariableCellSettings::from_args(p)?,
        })
    }
}
