//! The settings of `relaxedscan`: the scanned coordinates (`-scan`, read by
//! [`super::relaxedscan_targets`]), their numbers of points (`-nsample`), the kind of scan
//! (`-grid`, `-rigid`, `-fo`), the ReSPer references (`-resper`), the further settings
//! (`-scan_opt`), and the optmain settings of every point (every optmain option).
//!
//! MultiOptPy: `Entrypoints/relaxed_scan.py:run_relaxedscan` (`relaxed_scan_perser`,
//!  `make_scan_tgt`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use super::option::reject_unavailable;
use super::optmain::OptmainConfig;
use super::settings::settings_json;
use super::spec::RELAXEDSCAN;
use crate::analysis::resper::preimage::PreimageSettings;
use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::domain::AtomId;
use crate::error::{MoptError, MoptResult};
use crate::workflow::scan::GridOrder;

pub use super::relaxedscan_targets::{parse_targets, ScanTarget, TargetKind};

/// Points of a coordinate without `-nsample` (MultiOptPy's default).
pub const DEFAULT_SAMPLES: usize = 10;
/// Neighbours whose structures differ by more than this many times the median are on
/// different branches (`jump_factor`).
pub const DEFAULT_JUMP_FACTOR: f64 = 3.0;
/// Most reference structures of a ReSPer map (`resper_max`); more are thinned out evenly.
pub const DEFAULT_RESPER_MAX: usize = 200;

/// Where grid points of ReSPer coordinates beyond the references go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outside {
    /// Calculated and marked.
    Mark,
    /// Not calculated.
    Skip,
}

/// The settings of ReSPer coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct ResperSettings {
    /// The file of the reference structures.
    pub file: PathBuf,
    /// The atoms of the distances (all when empty).
    pub atoms: Vec<AtomId>,
    /// Masses in the superposition and the distances.
    pub mass_weighted: bool,
    /// Every this many frames of the file are taken.
    pub stride: usize,
    /// At most this many references.
    pub max: usize,
    /// Second derivatives of the coordinates for the constrained optimizer.
    pub curvature: bool,
    /// The pre-images of rigid scans.
    pub preimage: PreimageSettings,
    /// What happens to points beyond the references.
    pub outside: Outside,
    /// Relaxed points start from their pre-image instead of their optimized neighbour.
    pub start_from_preimage: bool,
}

/// The scan.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanSettings {
    /// The coordinates.
    pub targets: Vec<ScanTarget>,
    /// Points of each coordinate (one per target).
    pub samples: Vec<usize>,
    /// Every combination of the values (`-grid`) instead of all moving together.
    pub grid: bool,
    /// The order of the points of a grid.
    pub order: GridOrder,
    /// Optimize every point (false: single points, `-rigid`).
    pub relax: bool,
    /// Every point starts from the input (`-fo`).
    pub first_only: bool,
    /// Optimize the higher point of a jump between branches again from the lower one.
    pub repair: bool,
    /// Jumps are steps longer than this many times the median.
    pub jump_factor: f64,
    /// List the stationary points of the scanned surface.
    pub candidates: bool,
    /// Optimize the transition-state candidates (`optmain -order 1`).
    pub ts_optimize: bool,
    /// Start every relaxed point from the last Hessian of the point it starts from
    /// (`hessian=continue`).
    pub continue_hessians: bool,
    /// Figures.
    pub figures: bool,
    /// The page of the scan.
    pub html: bool,
    /// ReSPer coordinates.
    pub resper: Option<ResperSettings>,
}

/// Typed `relaxedscan` settings.
#[derive(Debug, Clone)]
pub struct RelaxedscanConfig {
    /// The input structure (or variables file).
    pub input: PathBuf,
    /// The settings in effect (`settings.json` of the scan).
    pub settings_document: String,
    /// Run directory to continue (`--restart`).
    pub restart_dir: Option<PathBuf>,
    /// The settings of the optimization of every point.
    pub optmain: OptmainConfig,
    /// The scan.
    pub scan: ScanSettings,
    /// `-tcc` and `-lcc` (the criteria of the constrained optimizations are set from them).
    pub criteria_flags: (bool, bool),
}

/// A switch of `-scan_opt` (`core_infra::arg_parser::parse_switch`).
fn on_off(key: &str, value: &str) -> MoptResult<bool> {
    crate::core_infra::arg_parser::parse_switch(&format!("-scan_opt {key}"), value)
}

fn samples(p: &ParsedArgs, targets: usize, grid: bool) -> MoptResult<Vec<usize>> {
    let text = p.get_list("number_of_samples").first().cloned();
    let Some(text) = text else {
        return Ok(vec![DEFAULT_SAMPLES; targets]);
    };
    let numbers: Vec<usize> = text
        .split(',')
        .map(|t| t.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| MoptError::parse(format!("-nsample: '{text}' is not N or N1,N2,...")))?;
    if numbers.contains(&0) {
        return Err(MoptError::parse("-nsample: every coordinate needs a point"));
    }
    match numbers.len() {
        1 => Ok(vec![numbers[0]; targets]),
        n if n == targets && grid => Ok(numbers),
        n if n == targets => Err(MoptError::parse(
            "-nsample: one count per coordinate needs -grid (without it the coordinates move \
             together)",
        )),
        n => Err(MoptError::parse(format!(
            "-nsample: {n} counts for {targets} coordinates"
        ))),
    }
}

fn resper_settings(
    p: &ParsedArgs,
    extra: &[(String, String)],
) -> MoptResult<Option<ResperSettings>> {
    let Some(file) = p.get_opt_str("resper_references") else {
        return Ok(None);
    };
    let mut s = ResperSettings {
        file: PathBuf::from(file),
        atoms: Vec::new(),
        mass_weighted: true,
        stride: 1,
        max: DEFAULT_RESPER_MAX,
        curvature: false,
        preimage: PreimageSettings::default(),
        outside: Outside::Mark,
        // Measured (running.md): from the neighbour a relaxed point can stay on another
        // branch of the level set of the coordinate, far from the references.
        start_from_preimage: true,
    };
    for (key, value) in extra {
        let number = || -> MoptResult<f64> {
            value
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| {
                    MoptError::parse(format!(
                        "-scan_opt {key}: '{value}' is not a positive number"
                    ))
                })
        };
        let count = || -> MoptResult<usize> {
            value
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|v| *v > 0)
                .ok_or_else(|| {
                    MoptError::parse(format!(
                        "-scan_opt {key}: '{value}' is not a positive count"
                    ))
                })
        };
        match key.as_str() {
            "resper_atoms" => {
                s.atoms = parse_atom_indices_one_based(value)?
                    .into_iter()
                    .map(AtomId::from_one_based)
                    .collect::<MoptResult<_>>()?;
            }
            "resper_mass" => s.mass_weighted = on_off(key, value)?,
            "resper_stride" => s.stride = count()?,
            "resper_max" => s.max = count()?.max(3),
            "resper_curvature" => s.curvature = on_off(key, value)?,
            "resper_correct" => s.preimage.correct = on_off(key, value)?,
            "resper_outside" => {
                s.outside = match value.trim().to_ascii_lowercase().as_str() {
                    "mark" => Outside::Mark,
                    "skip" => Outside::Skip,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-scan_opt resper_outside: mark or skip, got '{other}'"
                        )))
                    }
                }
            }
            "resper_outside_factor" => s.preimage.outside_factor = number()?,
            "resper_start" => {
                s.start_from_preimage = match value.trim().to_ascii_lowercase().as_str() {
                    "neighbour" | "neighbor" => false,
                    "preimage" => true,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-scan_opt resper_start: neighbour or preimage, got '{other}'"
                        )))
                    }
                }
            }
            _ => {}
        }
    }
    Ok(Some(s))
}

/// The keys of `-scan_opt`.
const KEYS: &[&str] = &[
    "order",
    "repair",
    "jump_factor",
    "candidates",
    "ts_optimize",
    "hessian",
    "figures",
    "html",
    "resper_atoms",
    "resper_mass",
    "resper_stride",
    "resper_max",
    "resper_curvature",
    "resper_correct",
    "resper_outside",
    "resper_outside_factor",
    "resper_start",
];

impl RelaxedscanConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        // The state that the input file names, for the runs started on files written here.
        let with_state = super::charge_spin::with_state_of_input(p);
        let p = &with_state;
        reject_unavailable(p, RELAXEDSCAN)?;
        let input = match p.positionals.as_slice() {
            [one] => PathBuf::from(one),
            [] => return Err(MoptError::config("relaxedscan needs an input structure")),
            _ => return Err(MoptError::config("relaxedscan scans one input structure")),
        };
        let mut extra = Vec::new();
        for token in p.get_list("scan_options") {
            let (key, value) = token.split_once('=').ok_or_else(|| {
                MoptError::parse(format!("-scan_opt expects KEY=VALUE, got '{token}'"))
            })?;
            let key = key.trim().to_ascii_lowercase();
            if !KEYS.contains(&key.as_str()) {
                return Err(MoptError::parse(format!(
                    "-scan_opt: unknown key '{key}' (known: {})",
                    KEYS.join(", ")
                )));
            }
            extra.push((key, value.trim().to_string()));
        }
        let targets = parse_targets(p.get_list("scan_tgt"))?;
        let grid = p.get_bool("scan_grid", false);
        let mut scan = ScanSettings {
            samples: samples(p, targets.len(), grid)?,
            targets,
            grid,
            order: GridOrder::Snake,
            relax: !p.get_bool("rigid_scan", false),
            first_only: p.get_bool("first_only", false),
            repair: true,
            jump_factor: DEFAULT_JUMP_FACTOR,
            candidates: true,
            ts_optimize: false,
            continue_hessians: true,
            figures: true,
            html: true,
            resper: resper_settings(p, &extra)?,
        };
        for (key, value) in &extra {
            match key.as_str() {
                "order" => scan.order = GridOrder::parse(value)?,
                "repair" => scan.repair = on_off(key, value)?,
                "jump_factor" => {
                    scan.jump_factor = value
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite() && *v > 1.0)
                        .ok_or_else(|| {
                            MoptError::parse(format!(
                                "-scan_opt jump_factor: a number above 1, got '{value}'"
                            ))
                        })?
                }
                "candidates" => scan.candidates = on_off(key, value)?,
                "ts_optimize" => scan.ts_optimize = on_off(key, value)?,
                "hessian" => {
                    scan.continue_hessians = match value.trim().to_ascii_lowercase().as_str() {
                        "continue" | "on" => true,
                        "off" | "none" => false,
                        other => {
                            return Err(MoptError::parse(format!(
                                "-scan_opt hessian: continue or off, got '{other}'"
                            )))
                        }
                    }
                }
                "figures" => scan.figures = on_off(key, value)?,
                "html" => scan.html = on_off(key, value)?,
                _ => {}
            }
        }
        let uses_resper = scan
            .targets
            .iter()
            .any(|t| matches!(t.kind, TargetKind::Resper(_)));
        if uses_resper && scan.resper.is_none() {
            return Err(MoptError::config(
                "-scan resper needs the reference structures: -resper FILE",
            ));
        }
        if scan.ts_optimize && !scan.candidates {
            return Err(MoptError::config(
                "-scan_opt ts_optimize=on needs candidates=on",
            ));
        }
        let mut optmain = OptmainConfig::from_args(p)?;
        if optmain.irc.is_some() {
            return Err(MoptError::unsupported(
                "-irc: the points of a scan are constrained minima, not transition states; \
                 -scan_opt ts_optimize=on optimizes the candidates of the scan, and ircmain \
                 follows the IRC of one",
            ));
        }
        if optmain
            .model_function
            .first()
            .is_some_and(|m| m.eq_ignore_ascii_case("bitss"))
        {
            return Err(MoptError::unsupported(
                "-mf bitss is a search between two structures, not a surface to scan",
            ));
        }
        if let Some(t) = scan
            .targets
            .iter()
            .find(|t| matches!(t.kind, TargetKind::Cell(_)))
        {
            if optmain.variable_cell.is_none() {
                return Err(MoptError::config(format!(
                    "-scan cell {}: a parameter of the cell changes only in a variable-cell \
                     optimization; add -vc",
                    t.atoms_text()
                )));
            }
            if !scan.relax {
                return Err(MoptError::unsupported(format!(
                    "-scan cell {} with relax=off: the cell of a point is brought onto its \
                     value by the variable-cell optimization of the point",
                    t.atoms_text()
                )));
            }
        }
        if optmain.variable_cell.is_some() {
            // The variable-cell optimizer holds distances, angles, dihedrals, expressions of
            // them, and frozen atoms (their fractional coordinates); what it cannot hold is
            // refused with the reason.
            for t in &scan.targets {
                let why = match &t.kind {
                    TargetKind::Resper(_) => Some(
                        "the ReSPer coordinates are those of reference structures in one cell; \
                         the cell of a variable-cell optimization changes",
                    ),
                    TargetKind::Variable(_) => None,
                    _ => match t.constraint(t.start)? {
                        Some(c) => crate::opt::cell_constraints::refused(&c.kind),
                        None => None,
                    },
                };
                if let Some(why) = why {
                    return Err(MoptError::unsupported(format!(
                        "-vc with -scan {} {}: {why}",
                        t.kind_name(),
                        t.atoms_text()
                    )));
                }
            }
            for c in &optmain.settings.constraints {
                if let Some(why) = crate::opt::cell_constraints::refused(&c.kind) {
                    return Err(MoptError::unsupported(format!(
                        "-vc with this constraint: {why}"
                    )));
                }
            }
        }
        optmain.inputs = vec![input.clone()];
        let criteria_flags = (
            p.get_bool("tight_convergence_criteria", false),
            p.get_bool("loose_convergence_criteria", false),
        );
        Ok(Self {
            input,
            settings_document: settings_json(p, RELAXEDSCAN).to_string_pretty(),
            restart_dir: None,
            optmain,
            scan,
            criteria_flags,
        })
    }

    /// The value of every target in the units of the constraints (Bohr, radian) for display:
    /// the factor from the units of the command line.
    pub fn to_atomic(target: &ScanTarget) -> f64 {
        target.to_atomic()
    }
}

/// Thin wrapper used by tests and embedding code: settings for `args`.
pub fn config_from_cli(args: &[&str]) -> MoptResult<RelaxedscanConfig> {
    let p = super::option::parser("relaxedscan", RELAXEDSCAN).parse(args)?;
    RelaxedscanConfig::from_args(&p)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn targets_counts_and_options() {
        let c = config_from_cli(&[
            "x.xyz",
            "-scan",
            "dihedral",
            "1,2,3,4",
            "-180,180",
            "bond",
            "1,2",
            "1.0,2.0",
            "-nsample",
            "13,5",
            "-grid",
            "-scan_opt",
            "order=raster",
            "repair=off",
        ])
        .unwrap();
        assert_eq!(c.scan.targets.len(), 2);
        assert_eq!(c.scan.targets[0].start, -180.0);
        assert_eq!(c.scan.targets[0].label(), "dihedral(1,2,3,4)");
        assert_eq!(c.scan.targets[0].unit(false), "degree");
        assert_eq!(c.scan.samples, vec![13, 5]);
        assert_eq!(c.scan.order, GridOrder::Raster);
        assert!(!c.scan.repair && c.scan.relax && c.scan.grid);
        // MultiOptPy's form: one count, the coordinates together.
        let j = config_from_cli(&["x.xyz", "-scan", "bond", "1,2", "1.3,2.6", "-nsample", "8"])
            .unwrap();
        assert_eq!(j.scan.samples, vec![8]);
        assert!(!j.scan.grid);
        for bad in [
            "x.xyz",
            "x.xyz -scan bond 1,2",
            "x.xyz -scan bond 1,2,3 1,2",
            "x.xyz -scan torsion 1,2,3,4 0,90",
            "x.xyz -scan bond 1,2 1",
            "x.xyz -scan bond 1,2 1,2 -nsample 3,4",
            "x.xyz -scan resper 1 0,1",
            "x.xyz -scan resper 3 0,1 -resper p.xyz",
            "x.xyz -scan bond 1,2 1,2 -scan_opt speed=fast",
            "x.xyz -scan bond 1,2 1,2 -scan_opt hessian=sometimes",
            "x.xyz -scan bond 1,2 1,2 -irc 0.1 50 lqa",
            // A variable cell holds no Cartesian component and no fixed direction.
            "x.xyz -scan x 1 0,1 -vc",
            "x.xyz -scan bond 1,2 1,2 -vc -pc rot 1-3",
        ] {
            let args: Vec<&str> = bad.split_whitespace().collect();
            assert!(config_from_cli(&args).is_err(), "{bad}");
        }
        // It holds distances, angles, dihedrals, expressions, and frozen atoms.
        for good in [
            "x.xyz -scan bond 1,2 1,2 -vc",
            "x.xyz -scan angle 1,2,3 90,100 -vc -fix 4",
        ] {
            let args: Vec<&str> = good.split_whitespace().collect();
            assert!(config_from_cli(&args).is_ok(), "{good}");
        }
        let e = config_from_cli(&[
            "x.xyz",
            "-scan",
            "expr",
            "dist(1,2)-dist(2,3)",
            "-1.0,1.0",
            "-rigid",
        ])
        .unwrap();
        assert!(!e.scan.relax);
        assert!(e.scan.targets[0].constraint(0.5).unwrap().is_some());
        assert_eq!(
            e.scan.targets[0].option_tokens(0.5).unwrap().0,
            "user_constraint"
        );
    }
}
