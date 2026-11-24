//! What `nebmain` does with a relaxed chain when it is asked to go on: the images that
//! are candidates for stationary points are optimized with `optmain`.
//!
//! `-neb_opt optimize=on` (`saddles`, `minima`; nothing without the setting): the images at
//! the maxima of the energy profile are optimized to transition states (`optmain -order
//! 1`), the end images and the images at minima of the profile to minima. Each is a run
//! of `optmain` with the settings it has without options (its methods for a minimum and
//! for a saddle point, its criteria) on the surface of the chain: the calculator and its
//! settings, charge and multiplicity, the electronic state or model function, the bias
//! potentials, and the constraints of the run are handed on. `optimize_settings=FILE`
//! names a settings file of `optmain` with anything else.
//!
//! The Hessian an optimization starts from (`optimize_hessian=`):
//!
//! - `chain` (the default): the Hessian that the relaxation kept for the image (a model
//!   Hessian or one from the calculator, updated over the iterations). For a saddle
//!   point its curvature along the chain is replaced by the curvature of the energy
//!   profile there, which is negative at a maximum: the steps of the chain go across it,
//!   so the kept Hessian knows little about that direction, and the profile knows it.
//!   Where there is no such Hessian (no Hessians in the relaxation, a chain that did not
//!   come to rest, bias potentials, whose curvature the kept Hessian contains and
//!   `optmain` would add again) the optimization starts as with `exact`;
//! - `exact`: the Hessian of the calculator at the first point;
//! - `none`: as `optmain` starts without options.
//!
//! Which optimized structures get a vibrational analysis (`optimize_check=`; `optmain
//! -freq`, an exact Hessian each): `saddles` (the default), `all`, or `off`. An optimization
//! to a transition state that converged to a structure with no or several imaginary
//! frequencies (or one to a minimum with some) is reported, and `optimized_images.csv`
//! gives the number.
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

use faer::{Col, Mat};

use super::nebmain::NebmainResult;
use super::nebmain_settings::NebmainConfig;
use super::optmain::{self, OptmainConfig};
use super::settings::{merge, parse_settings, read_settings, INPUT_KEY};
use super::spec::OPTMAIN;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::npy::{read_npy_matrix, write_npy_matrix};
use crate::io::xyz::read_xyz_trajectory_file;
use crate::math::linalg::col_norm;
use crate::math::projection::{build_tr_rot_basis, project_out_basis_vec};
use crate::path::band::machine::ChainEnd;
use crate::path::band::{path_lengths, strict_minima, ChainKind};
use crate::pes::constraint::Constraint;
use crate::pes::registry::CalculatorRegistry;

/// `-fc` of an optimization that gets the Hessian of the calculator at its first point
/// only: an interval no run reaches.
const AT_THE_START_ONLY: &str = "1000000";
/// Settings of the run that are not handed on: the number of iterations and the Hessians
/// of the chain are those of the relaxation, not of the optimizations after it.
const NOT_HANDED_ON: &[&str] = &[
    INPUT_KEY,
    "NSTEP",
    "use_model_hessian",
    "calc_model_hess",
    "saddle_order",
    "initial_hessian",
];

/// Which images are optimized after the relaxation (`-neb_opt optimize=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizedImages {
    /// The images at the maxima of the profile, to transition states.
    Saddles,
    /// The end images and the images at the minima of the profile, to minima.
    Minima,
    /// Both.
    All,
}

/// The Hessian an optimization starts from (`-neb_opt optimize_hessian=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstHessian {
    /// The Hessian the relaxation kept for the image, with the curvature of the profile
    /// along the chain for a saddle point.
    Chain,
    /// The Hessian of the calculator at the first point.
    Exact,
    /// None: as `optmain` starts without options.
    None,
}

/// Which optimized structures get a vibrational analysis (`-neb_opt optimize_check=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedImages {
    /// The transition states.
    Saddles,
    /// All of them.
    All,
    /// None.
    Off,
}

impl CheckedImages {
    /// Whether an optimization to a transition state (`saddle`) or a minimum is analysed.
    pub fn includes(self, saddle: bool) -> bool {
        match self {
            Self::Saddles => saddle,
            Self::All => true,
            Self::Off => false,
        }
    }
}

/// The optimizations after the relaxation of a chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FollowUp {
    /// Which images.
    pub images: OptimizedImages,
    /// What they start from.
    pub hessian: FirstHessian,
    /// A settings file of `optmain` with further settings (`optimize_settings=`).
    pub settings: Option<PathBuf>,
    /// Which optimized structures get a vibrational analysis.
    pub check: CheckedImages,
}

impl FollowUp {
    /// The optimizations of `images` with the default first Hessian and checks.
    pub fn of(images: OptimizedImages) -> Self {
        Self {
            images,
            hessian: FirstHessian::Chain,
            settings: None,
            check: CheckedImages::Saddles,
        }
    }
}

/// An image after its optimization.
#[derive(Debug, Clone)]
pub struct OptimizedImage {
    /// Its number in the chain.
    pub image: usize,
    /// Whether it was optimized to a transition state (otherwise to a minimum).
    pub saddle: bool,
    /// Whether the optimization converged.
    pub converged: bool,
    /// Iterations taken.
    pub iterations: usize,
    /// Energy of the optimized structure with the bias potentials (Hartree).
    pub energy: f64,
    /// Distance of the optimized structure from the image (Bohr, each at its centroid).
    pub moved: f64,
    /// Whether that is further than the nearest neighbour of the image in the chain: the
    /// optimization then left the place of the image, and what it found need not be the
    /// stationary point the chain passes.
    pub left_its_place: bool,
    /// The optimized structure.
    pub optimized: PathBuf,
    /// The run directory of the optimization.
    pub directory: PathBuf,
    /// Its number of imaginary frequencies, when it was analysed.
    pub imaginary_modes: Option<usize>,
}

/// `hessian` of image `image` of `chain` with its curvature along the chain replaced by
/// the curvature of the energy profile there (second difference of `energies` over the
/// lengths along the chain). Unchanged where the profile curves upwards or the chain has
/// no direction.
pub fn with_curvature_of_the_profile(
    hessian: &Mat<f64>,
    chain: &[Col<f64>],
    energies: &[f64],
    image: usize,
    template: &MolecularSystem,
) -> MoptResult<Mat<f64>> {
    if image == 0 || image + 1 >= chain.len() || energies.len() != chain.len() {
        return Ok(hessian.clone());
    }
    let lengths = path_lengths(chain, ChainKind::Atoms);
    let (before, after) = (
        lengths[image] - lengths[image - 1],
        lengths[image + 1] - lengths[image],
    );
    if before <= 0.0 || after <= 0.0 {
        return Ok(hessian.clone());
    }
    let curvature = 2.0
        * ((energies[image + 1] - energies[image]) / after
            - (energies[image] - energies[image - 1]) / before)
        / (before + after);
    let here = template.with_flat_coords_bohr(&chain[image])?;
    let rigid = build_tr_rot_basis(&here, false);
    let along = project_out_basis_vec(&rigid, &(&chain[image + 1] - &chain[image - 1]));
    let norm = col_norm(&along);
    if curvature >= 0.0 || norm < 1e-10 || hessian.nrows() != along.nrows() {
        return Ok(hessian.clone());
    }
    let t = &along * (1.0 / norm);
    let kept = (hessian * &t)
        .iter()
        .zip(t.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    Ok(Mat::from_fn(hessian.nrows(), hessian.ncols(), |r, c| {
        hessian[(r, c)] + (curvature - kept) * t[r] * t[c]
    }))
}

/// The settings of the `optmain` run for one image, which holds `constraints` (the
/// constraints of the run resolved at the image: what the image held in the chain).
fn optmain_config(
    config: &NebmainConfig,
    follow: &FollowUp,
    input: &Path,
    saddle: bool,
    first_hessian: Option<&Path>,
    exact_at_the_start: bool,
    constraints: &[Constraint],
) -> MoptResult<OptmainConfig> {
    let text = |value: &str| JsonValue::String(value.to_string());
    let mut settings = BTreeMap::new();
    let document = JsonValue::parse(&config.settings_document)?;
    if let Some(given) = document.as_object() {
        for (key, value) in given {
            let shared = OPTMAIN.iter().any(|option| option.spec.key == key);
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
    if saddle {
        settings.insert("saddle_order".to_string(), text("1"));
    }
    if follow.check.includes(saddle) {
        settings.insert("frequency_analysis".to_string(), JsonValue::Bool(true));
    }
    if let Some(file) = first_hessian {
        settings.insert("initial_hessian".to_string(), text(&file.to_string_lossy()));
    } else if exact_at_the_start && !settings.contains_key("calc_exact_hess") {
        settings.insert("calc_exact_hess".to_string(), text(AT_THE_START_ONLY));
    }
    let mut parsed = parse_settings(&JsonValue::Object(settings).to_string_compact(), OPTMAIN)?;
    if let Some(file) = &follow.settings {
        // What the file says goes before what the relaxation hands on, but for the input.
        let mut further = read_settings(file, OPTMAIN).context(file.display())?;
        further.positionals.clear();
        parsed = merge(parsed, further);
    }
    let mut out = OptmainConfig::from_args(&parsed)?;
    // Those of the chain first; a settings file may add more.
    let added = std::mem::take(&mut out.settings.constraints);
    out.settings.constraints = constraints.to_vec();
    out.settings.constraints.extend(added);
    Ok(out)
}

/// The keys of the constraint options, which a follow-up run gets resolved.
pub(super) const CONSTRAINT_KEYS: &[&str] = &[
    "fix_atoms",
    "projection_constrain",
    "gradient_fix_atoms",
    "user_constraint",
    "variable_cell_fix",
];

/// The optimizations `follow` of the images of the chain of `result` (a chain of
/// structures with the atoms of `template`), run in the directory of the chain. A line
/// per optimization goes to `log`, and all are in `optimized_images.csv` there.
pub fn optimize_images(
    follow: &FollowUp,
    config: &NebmainConfig,
    result: &NebmainResult,
    template: &MolecularSystem,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    log: &Logger,
) -> MoptResult<Vec<OptimizedImage>> {
    let Some(candidates) = &result.candidates else {
        return Ok(Vec::new());
    };
    let frames = read_xyz_trajectory_file(&candidates.path, template.spin_charge, None)
        .context(candidates.path.display())?;
    let chain: Vec<Col<f64>> = frames
        .frames
        .iter()
        .map(|frame| frame.system.coords_flat_bohr())
        .collect();
    let energies = &result.energies;
    if chain.len() != energies.len() || chain.len() < 2 {
        return Err(MoptError::config(format!(
            "{}: {} structures for {} energies of the last chain",
            candidates.path.display(),
            chain.len(),
            energies.len()
        )));
    }
    let mut images: Vec<(usize, bool)> = Vec::new();
    if follow.images != OptimizedImages::Saddles {
        images.push((0, false));
        images.extend(strict_minima(energies).into_iter().map(|k| (k, false)));
        images.push((chain.len() - 1, false));
    }
    if follow.images != OptimizedImages::Minima {
        images.extend(candidates.images.iter().map(|k| (*k, true)));
    }
    images.sort_unstable();
    images.dedup_by_key(|entry| entry.0);

    // Absolute, so that each optimization's run directory is next to its image (optmain
    // names it after the whole input path, as MultiOptPy does).
    let directory = std::path::absolute(candidates.path.parent().unwrap_or(Path::new(".")))?;
    let base = std::path::absolute(&result.directory)?;
    let name = candidates
        .path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .and_then(|stem| stem.strip_suffix("_path").map(str::to_string))
        .ok_or_else(|| {
            MoptError::config(format!(
                "{}: not the file of a chain",
                candidates.path.display()
            ))
        })?;
    let at_rest = result.end == Some(ChainEnd::Converged);
    let lengths = path_lengths(&chain, ChainKind::Atoms);
    let evaluations_before = registry.counts();
    log.info(&format!(
        "Optimizations after the chain (-neb_opt optimize): {} image(s) with optmain.",
        images.len()
    ));
    let mut done = Vec::with_capacity(images.len());
    for (image, saddle) in images {
        let input = directory.join(format!("{name}_{image}.xyz"));
        let kept = result.directory.join(format!("tmp_hessian_{image}.npy"));
        let mut first_hessian = None;
        if follow.hessian == FirstHessian::Chain {
            let cartesian = ["", "cart", "cartesian"]
                .contains(&config.coordinates.trim().to_ascii_lowercase().as_str());
            let usable = at_rest && cartesian && config.biases.is_empty() && kept.is_file();
            let hessian = if usable {
                Some(read_npy_matrix(&kept)?)
            } else {
                None
            };
            // With ONIOM microiterations the chain kept Hessians of the core, and the
            // optimization starts from the whole structure.
            let of_the_whole = hessian
                .as_ref()
                .is_some_and(|h| h.nrows() == template.dof());
            match hessian {
                Some(mut hessian) if of_the_whole => {
                    if saddle {
                        hessian = with_curvature_of_the_profile(
                            &hessian, &chain, energies, image, template,
                        )?;
                    }
                    let file = result.directory.join(format!("first_hessian_{image}.npy"));
                    write_npy_matrix(&file, &hessian)?;
                    first_hessian = Some(file);
                }
                _ => log.info(&format!(
                    "Image {image}: no Hessian of the chain to start from ({}); the \
                     calculator's at the first point instead.",
                    if config.variable_cell.is_some() {
                        "the chain kept its Hessians in the variables of the atoms and the cell"
                    } else if !cartesian {
                        "the chain kept its Hessians in internal coordinates"
                    } else if !kept.is_file() {
                        "the relaxation kept none"
                    } else if !at_rest {
                        "the chain did not come to rest"
                    } else if !config.biases.is_empty() {
                        "it contains the bias potentials"
                    } else {
                        "the chain kept Hessians of the core of the ONIOM microiterations"
                    }
                )),
            }
        }
        let exact = follow.hessian != FirstHessian::None && first_hessian.is_none();
        // The constraints the image held: resolved at it (the image is on its values in
        // the chain), in the numbering of the whole system; on a variable cell in the cell of
        // the image (the frame of the file has it).
        let at = if config.variable_cell.is_some() {
            frames.frames[image].system.clone()
        } else {
            template.with_flat_coords_bohr(&chain[image])?
        };
        let held = super::constraints::resolve_for_run(&config.constraints, &at)
            .context(format!("the constraints of image {image}"))?;
        let settings = optmain_config(
            config,
            follow,
            &input,
            saddle,
            first_hessian.as_deref(),
            exact,
            &held,
        )?;
        let kind = if saddle {
            "a transition state"
        } else {
            "a minimum"
        };
        log.info(&format!(
            "Image {image} to {kind}: optmain {} ({}).",
            input.display(),
            settings.settings.methods.join(" ")
        ));
        log.flush();
        let runs = optmain::run_with(&settings, &result.directory, console, registry)
            .context(format!("the optimization of image {image}"))?;
        let Some(run) = runs.into_iter().next() else {
            continue;
        };
        let (energy, iterations) = run
            .summary
            .as_ref()
            .map_or((f64::NAN, 0), |s| (s.biased_energy.value(), s.iterations));
        let imaginary_modes = run.summary.as_ref().and_then(|s| s.imaginary_modes);
        let moved = run.summary.as_ref().map_or(f64::NAN, |s| {
            let pair = [chain[image].clone(), s.system.coords_flat_bohr()];
            path_lengths(&pair, ChainKind::Atoms)[1]
        });
        let nearest = [image.checked_sub(1), Some(image + 1)]
            .into_iter()
            .flatten()
            .filter(|k| *k < chain.len())
            .map(|k| (lengths[k] - lengths[image]).abs())
            .fold(f64::INFINITY, f64::min);
        let left_its_place = moved > nearest;
        log.info(&format!(
            "Image {image} to {kind}: {} after {iterations} iteration(s), {:.4} kcal/mol from \
             the image, {moved:.4} Bohr from it ({}).",
            if run.converged {
                "converged"
            } else {
                "not converged"
            },
            Units::HARTREE_TO_KCAL_MOL * (energy - energies[image]),
            run.optimized.display()
        ));
        let expected = usize::from(saddle);
        if let Some(found) = imaginary_modes.filter(|n| run.converged && *n != expected) {
            log.warn(&format!(
                "Image {image}: the optimization to {kind} ended at a structure with {found} \
                 imaginary frequenc{} ({expected} expected): {}.",
                if found == 1 { "y" } else { "ies" },
                if saddle && found == 0 {
                    "a minimum, not a transition state"
                } else if saddle {
                    "a saddle point of higher order"
                } else {
                    "not a minimum"
                }
            ));
        }
        if left_its_place {
            log.warn(&format!(
                "Image {image}: the optimization went further ({moved:.4} Bohr) than the \
                 nearest image is away ({nearest:.4} Bohr); what it found need not be the \
                 stationary point of this chain. A chain relaxed further (a smaller \
                 -neb_opt tolerance) gives a better start."
            ));
        }
        done.push(OptimizedImage {
            image,
            saddle,
            converged: run.converged,
            iterations,
            energy,
            moved,
            left_its_place,
            optimized: run.optimized,
            directory: run.directory,
            imaginary_modes,
        });
    }
    let lowest = done
        .iter()
        .filter(|point| !point.saddle && point.energy.is_finite())
        .map(|point| point.energy)
        .fold(f64::INFINITY, f64::min);
    let reference = if lowest.is_finite() {
        lowest
    } else {
        energies[0]
    };
    let mut table = String::from(
        "image,kind,converged,iterations,energy_hartree,relative_kcal_mol,moved_bohr,\
         left_its_place,imaginary_modes,file\n",
    );
    for point in &done {
        let _ = writeln!(
            table,
            "{},{},{},{},{:.12},{:.4},{:.4},{},{},{}",
            point.image,
            if point.saddle { "saddle" } else { "minimum" },
            point.converged,
            point.iterations,
            point.energy,
            Units::HARTREE_TO_KCAL_MOL * (point.energy - reference),
            point.moved,
            point.left_its_place,
            point
                .imaginary_modes
                .map_or_else(String::new, |n| n.to_string()),
            // Relative to the run directory of the chain (where this table is).
            point
                .optimized
                .strip_prefix(&base)
                .unwrap_or(&point.optimized)
                .display()
        );
        log.info(&format!(
            "  image {:>3}  {:<8} {:>12.4} kcal/mol{}",
            point.image,
            if point.saddle { "saddle" } else { "minimum" },
            Units::HARTREE_TO_KCAL_MOL * (point.energy - reference),
            match (
                point.converged,
                point
                    .imaginary_modes
                    .filter(|n| *n != usize::from(point.saddle)),
                point.left_its_place
            ) {
                (false, _, _) => "  (not converged)".to_string(),
                (true, Some(n), _) => format!("  ({n} imaginary frequencies)"),
                (true, None, true) => "  (left the place of its image)".to_string(),
                (true, None, false) => String::new(),
            }
        ));
    }
    std::fs::write(result.directory.join("optimized_images.csv"), table)?;
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("the optimizations after the chain"),
    );
    log.flush();
    Ok(done)
}
