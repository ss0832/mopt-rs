//! `run_kinetics`: the kinetics of a reaction network file (the `reaction_network.json` of
//! `run_mapper`, a run directory of it, or MultiOptPy's network file): rate constants by a
//! chosen rate theory with corrections, a kinetic model (`canonical`, `mass_action`, `egme`,
//! `two_grain`, `semi_markov`), the populations at the output times by a chosen solver
//! (`propagate` by default; `rcmc`, `legacy_rcmc`, `expm`, `spectral`, `steady`, `rosenbrock`,
//! `kmc`), the superstates, diagnostics, and on request the comparison with the matrix
//! exponential, the degrees of rate control, the refinement tasks, the propagation of energy
//! errors, and the ranking of the states for an exploration. Files go to the run directory
//! (`--output-dir`, `kinetics_output`).
//!
//! `run_rcmc` is MultiOptPy's command with its options: the same code with the settings that
//! reproduce MultiOptPy (the Eyring rates of `rcmc.py` with its clamp, the solver
//! `legacy_rcmc`), its table on the screen, and `rcmc_K_contracted.csv` when `--output-dir` is
//! given (with the other files of `run_kinetics`).
//!
//! MultiOptPy: `Utils/rcmc.py:main` (`run_rcmc`); `run_kinetics` is new in mopt-rs.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::option::reject_unavailable;
use super::settings::settings_json;
use super::spec::{RUN_KINETICS, RUN_RCMC};
use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use crate::error::{MoptError, MoptResult};
use crate::kinetics::conditions::{Clamp, Conditions, EnergyKind};
use crate::kinetics::contract::TimeRule;
use crate::kinetics::egme::{Bath, CollisionModel, EgmeSettings};
use crate::kinetics::hot::HotSettings;
use crate::kinetics::mass_action::DEFAULT_ODE_TOLERANCE;
use crate::kinetics::observe::Score;
use crate::kinetics::rate::tst::{QrrhoSettings, SymmetryChoice};
use crate::kinetics::rate::{BellBarrier, BellSettings, BellTruncation, RateSettings};
use crate::kinetics::rrkm::RrkmSettings;
use crate::kinetics::solve::{Closure, SolverSettings, DEFAULT_TOLERANCE};
use crate::kinetics::{
    Association, KineticsParts, KineticsSettings, ModelSettings, SpeciesSettings,
};
use crate::vib::QuasiHarmonic;
use crate::workflow::kinetics::{run_job, KineticsJob, KineticsRun, Report};
use crate::workflow::kinetics_beyond::{KineticsExtras, MdTrajectory};
use crate::workflow::mapper::network::Level as NetworkLevel;

/// The settings of a run of `run_kinetics` or `run_rcmc`.
#[derive(Debug, Clone)]
pub struct KineticsConfig {
    /// The job.
    pub job: KineticsJob,
    /// What the screen shows.
    pub console: Level,
}

fn parse_initial(items: &[String]) -> MoptResult<Vec<(usize, f64)>> {
    items
        .iter()
        .flat_map(|t| t.split(','))
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            let (id, share) = t.split_once(':').unwrap_or((t, "1"));
            let id = id
                .trim()
                .trim_start_matches("EQ")
                .parse::<usize>()
                .map_err(|_| MoptError::parse(format!("--initial: '{t}' is not ID:SHARE")))?;
            let share = share
                .trim()
                .parse::<f64>()
                .map_err(|_| MoptError::parse(format!("--initial: '{t}' is not ID:SHARE")))?;
            Ok((id, share))
        })
        .collect()
}

/// `NAME:VALUE[UNIT]` of `--concentration`: mol/L by default; `M`, `mM`, `uM`, `nM`, `pM`,
/// `mol/m3`, or a partial pressure of an ideal gas (`bar`, `atm`, `Pa`), converted with
/// `c = p / RT` at the temperature of `c`.
pub fn parse_concentration(text: &str, c: &Conditions) -> MoptResult<(String, f64)> {
    let bad = || {
        MoptError::parse(format!(
            "--concentration: '{text}' is not NAME:VALUE[UNIT] (units M, mM, uM, nM, pM, \
             mol/m3, bar, atm, Pa; mol/L by default)"
        ))
    };
    let (name, value) = text.trim().rsplit_once(':').ok_or_else(bad)?;
    let value = value.trim();
    let split = (1..=value.len())
        .rev()
        .find(|&k| value.is_char_boundary(k) && value[..k].parse::<f64>().is_ok())
        .ok_or_else(bad)?;
    let x: f64 = value[..split].parse().map_err(|_| bad())?;
    let unit = value[split..].trim();
    let per_m3_to_mol_l = 1e-3;
    let gas = |pa: f64| pa / (crate::domain::Units::GAS_CONSTANT_J_MOL_K * c.temperature) * 1e-3;
    let mol_l = match unit {
        "" | "M" | "mol/L" | "mol/l" => x,
        "mM" => x * 1e-3,
        "uM" | "\u{b5}M" | "\u{3bc}M" => x * 1e-6,
        "nM" => x * 1e-9,
        "pM" => x * 1e-12,
        "mol/m3" => x * per_m3_to_mol_l,
        "bar" => gas(x * 1e5),
        "atm" => gas(x * 101_325.0),
        "Pa" => gas(x),
        _ => return Err(bad()),
    };
    if name.trim().is_empty() || !(mol_l.is_finite() && mol_l > 0.0) {
        return Err(bad());
    }
    Ok((name.trim().to_string(), mol_l))
}

/// The annotations of `--species NAME:E_HARTREE[:G_HARTREE]` and
/// `--partner EDGE:SPECIES:STATE`.
pub fn species_annotations(
    species: &[String],
    partners: &[String],
) -> MoptResult<Option<JsonValue>> {
    let items = |list: &[String]| -> Vec<String> {
        list.iter()
            .flat_map(|c| c.split(','))
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect()
    };
    let (species, partners) = (items(species), items(partners));
    if species.is_empty() && partners.is_empty() {
        return Ok(None);
    }
    let mut sp = BTreeMap::new();
    for s in &species {
        let parts: Vec<&str> = s.split(':').collect();
        let number = |x: &str| -> MoptResult<Option<JsonValue>> {
            if x.is_empty() || x == "-" {
                return Ok(None);
            }
            x.parse::<f64>()
                .map(|v| Some(JsonValue::Number(v)))
                .map_err(|_| {
                    MoptError::parse(format!(
                        "--species: '{s}' is not NAME:E_HARTREE[:G_HARTREE] ('-' for none)"
                    ))
                })
        };
        let (name, e, g) = match parts.as_slice() {
            [n, e] => (*n, number(e)?, None),
            [n, e, g] => (*n, number(e)?, number(g)?),
            _ => {
                return Err(MoptError::parse(format!(
                    "--species: '{s}' is not NAME:E_HARTREE[:G_HARTREE]"
                )))
            }
        };
        let mut m = BTreeMap::new();
        if let Some(e) = e {
            m.insert("energy_hartree".to_string(), e);
        }
        if let Some(g) = g {
            m.insert("free_energy_hartree".to_string(), g);
        }
        sp.insert(name.to_string(), JsonValue::Object(m));
    }
    let mut edges = BTreeMap::new();
    for p in &partners {
        let [edge, s, state] = p.split(':').collect::<Vec<_>>()[..] else {
            return Err(MoptError::parse(format!(
                "--partner: '{p}' is not EDGE:SPECIES:STATE (the state of the edge where the \
                 species is free)"
            )));
        };
        let mut m = BTreeMap::new();
        m.insert("partner".to_string(), JsonValue::String(s.to_string()));
        m.insert(
            "partner_free_state".to_string(),
            JsonValue::String(state.to_string()),
        );
        edges.insert(edge.to_string(), JsonValue::Object(m));
    }
    let mut doc = BTreeMap::new();
    doc.insert("species".to_string(), JsonValue::Object(sp));
    doc.insert("edges".to_string(), JsonValue::Object(edges));
    Ok(Some(JsonValue::Object(doc)))
}

fn parse_times(p: &ParsedArgs) -> MoptResult<Vec<f64>> {
    let mut times: Vec<f64> = Vec::new();
    for t in p.get_list("times").iter().flat_map(|t| t.split(',')) {
        if t.trim().is_empty() {
            continue;
        }
        times.push(
            t.trim()
                .parse()
                .map_err(|_| MoptError::parse(format!("--times: '{t}' is not a number")))?,
        );
    }
    let grid = p.get_list("time_grid");
    if !grid.is_empty() {
        let [lo, hi, n] = grid else {
            return Err(MoptError::parse("--time-grid takes TMIN TMAX N"));
        };
        let lo: f64 = lo
            .parse()
            .map_err(|_| MoptError::parse("--time-grid: TMIN is not a number"))?;
        let hi: f64 = hi
            .parse()
            .map_err(|_| MoptError::parse("--time-grid: TMAX is not a number"))?;
        let n: usize = n
            .parse()
            .map_err(|_| MoptError::parse("--time-grid: N is not a whole number"))?;
        if !(lo > 0.0 && hi > lo && n >= 2) {
            return Err(MoptError::parse(
                "--time-grid needs 0 < TMIN < TMAX and at least 2 points",
            ));
        }
        let ratio = (hi / lo).ln();
        times.extend((0..n).map(|k| lo * (ratio * k as f64 / (n - 1) as f64).exp()));
    }
    if times.is_empty() {
        times.push(p.get_f64("time", 1.0)?);
    }
    if times.iter().any(|t| !(t.is_finite() && *t > 0.0)) {
        return Err(MoptError::parse("the output times must be positive"));
    }
    times.sort_by(f64::total_cmp);
    times.dedup();
    Ok(times)
}

/// The grains and collisions of `--model egme` (the collision model is read only for it).
fn egme_settings(p: &ParsedArgs) -> MoptResult<EgmeSettings> {
    let egme = p.get_str("model", "canonical").eq_ignore_ascii_case("egme");
    let lennard_jones = match p.get_opt_str("lennard_jones") {
        None => None,
        Some(text) => {
            let v: Vec<f64> = text
                .split(':')
                .map(|x| x.trim().parse::<f64>())
                .collect::<Result<_, _>>()
                .map_err(|_| {
                    MoptError::parse(format!("--lennard-jones: '{text}' is not SIGMA:EPS"))
                })?;
            match v.as_slice() {
                [s, e] if *s > 0.0 && *e > 0.0 => Some((*s, *e)),
                _ => {
                    return Err(MoptError::parse(format!(
                        "--lennard-jones: '{text}' is not SIGMA:EPS (Angstrom, K; positive)"
                    )))
                }
            }
        }
    };
    Ok(EgmeSettings {
        grain: p.get_f64("egme_grain", 50.0)?,
        above_kt: p.get_f64("egme_above", 40.0)?,
        collision: if egme {
            Some(CollisionModel::parse(
                p.get_str("collision_model", "exponential_down"),
                p.get_opt_f64("energy_transfer")?,
            )?)
        } else {
            None
        },
        frequency: p.get_opt_f64("collision_frequency")?,
        bath: p.get_opt_str("bath").map(Bath::parse).transpose()?,
        lennard_jones,
    })
}

/// `--refine`, `--md-trajectory`, and the seed of `kmc`.
fn extras(p: &ParsedArgs) -> MoptResult<KineticsExtras> {
    let refine = if p.is_present("refine") {
        Some(
            parse_initial(p.get_list("refine"))?
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
        )
    } else {
        None
    };
    let md_trajectory = match p.get_opt_str("md_trajectory") {
        None => None,
        Some(path) => {
            let Some(fs) = p.get_opt_f64("md_time_step")? else {
                return Err(MoptError::parse(
                    "--md-trajectory needs the time between its frames (--md-time-step FS)",
                ));
            };
            if !(fs > 0.0 && fs.is_finite()) {
                return Err(MoptError::parse("--md-time-step must be positive"));
            }
            Some(MdTrajectory {
                path: PathBuf::from(path),
                time_step: fs * 1e-15,
                lag: p.get_usize("md_lag", 1)?.max(1),
            })
        }
    };
    Ok(KineticsExtras {
        seed: Some(p.get_usize("seed", 42)? as u64),
        refine,
        refine_sigma: None,
        md_trajectory,
    })
}

fn console_level(p: &ParsedArgs, default: Level) -> MoptResult<Level> {
    match p.get_opt_str("log_level") {
        Some(l) => Level::parse(l),
        None => Ok(default),
    }
}

fn network_path(p: &ParsedArgs, command: &str) -> MoptResult<PathBuf> {
    match p.positionals.as_slice() {
        [one] => Ok(PathBuf::from(one)),
        [] => Err(MoptError::parse(format!(
            "{command} needs a network file (reaction_network.json or MultiOptPy's network)"
        ))),
        _ => Err(MoptError::parse(format!(
            "{command} takes one network file"
        ))),
    }
}

impl KineticsConfig {
    /// The settings of `run_kinetics`.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        reject_unavailable(p, RUN_KINETICS)?;
        let network = network_path(p, "run_kinetics")?;
        let conditions = Conditions {
            temperature: p.get_f64("temperature", 300.0)?,
            pressure: p.get_f64("pressure", 101_325.0)?,
            concentration: p.get_opt_f64("standard_concentration")?,
            energy: EnergyKind::parse(p.get_str("energy_type", "electronic"))?,
            clamp: Clamp::parse(p.get_str("clamp", "consistent"))?,
            viscosity: p.get_opt_f64("viscosity")?,
        };
        conditions.check()?;
        let corrections: Vec<String> = p
            .get_list("corrections")
            .iter()
            .flat_map(|c| c.split(','))
            .map(|c| c.trim().to_ascii_lowercase())
            .filter(|c| !c.is_empty() && c != "none")
            .collect();
        let kappa = p.get_opt_f64("kappa")?;
        let mut corrections = corrections;
        if kappa.is_some() && !corrections.iter().any(|c| c == "kappa") {
            corrections.push("kappa".to_string());
        }
        let settings = KineticsSettings {
            conditions,
            rate_theory: p.get_str("rate_theory", "eyring").to_ascii_lowercase(),
            corrections,
            rate: RateSettings {
                qrrho: QrrhoSettings {
                    method: QuasiHarmonic::parse(p.get_str("qrrho", "grimme"))?,
                    cutoff: p.get_f64("qrrho_cutoff", 100.0)?,
                    symmetry: SymmetryChoice::parse(p.get_str("symmetry", "auto"))?,
                },
                kappa,
                bell: BellSettings {
                    truncation: BellTruncation::parse(p.get_str("bell_truncation", "auto"))?,
                    barrier: BellBarrier::parse(p.get_str("bell_barrier", "higher"))?,
                },
                friction: p.get_opt_f64("friction")?,
                rrkm: RrkmSettings {
                    cell: p.get_f64("rrkm_cell", 1.0)?,
                    ..RrkmSettings::default()
                },
            },
            model: p.get_str("model", "canonical").to_ascii_lowercase(),
            solver: p.get_str("solver", "propagate").to_ascii_lowercase(),
            solver_settings: SolverSettings {
                time_rule: TimeRule::parse(p.get_str("time_rule", "gershgorin"))?,
                tolerance: p.get_f64("tolerance", DEFAULT_TOLERANCE)?,
                closure: Closure::parse(p.get_str("closure", "a"))?,
                legacy_start_known: true,
                ode_tolerance: p.get_f64("ode_tolerance", DEFAULT_ODE_TOLERANCE)?,
                kmc_trajectories: p.get_usize("kmc_trajectories", 10_000)?,
                kmc_max_events: p.get_f64("kmc_max_events", 1e8)?.max(1.0) as u64,
                contract_theta: p.get_f64("contract_theta", 100.0)?,
                window_tolerance: p.get_f64("window_tolerance", 1e-3)?,
            },
            species: SpeciesSettings {
                concentrations: p
                    .get_list("concentration")
                    .iter()
                    .flat_map(|c| c.split(','))
                    .filter(|c| !c.trim().is_empty())
                    .map(|c| parse_concentration(c, &conditions))
                    .collect::<MoptResult<_>>()?,
                association: Association::parse(p.get_str("association", "none"))?,
                ..SpeciesSettings::default()
            },
            model_settings: ModelSettings {
                total_concentration: p.get_opt_f64("total_concentration")?,
                egme: egme_settings(p)?,
                hot: HotSettings {
                    cooling: p.get_opt_f64("cooling_rate")?,
                    threshold: p.get_f64("hot_threshold", 0.1)?,
                },
                order_tolerance: p.get_opt_f64("order_tolerance")?,
                ..ModelSettings::default()
            },
        };
        let cli_annotations = species_annotations(p.get_list("species"), p.get_list("partner"))?;
        let mut initial = parse_initial(p.get_list("initial"))?;
        if let Some(start) = p.get_opt_str("start_node") {
            if !initial.is_empty() {
                return Err(MoptError::parse("give --start-node or --initial, not both"));
            }
            initial = parse_initial(&[start.to_string()])?;
        }
        let validate = if p.is_present("validate") {
            Some(match p.get_opt_str("validate") {
                Some(t) => Some(t.parse::<f64>().map_err(|_| {
                    MoptError::parse(format!("--validate: '{t}' is not a tolerance"))
                })?),
                None => None,
            })
        } else {
            None
        };
        let sensitivity = if p.is_present("sensitivity") {
            Some(
                parse_initial(p.get_list("sensitivity"))?
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect(),
            )
        } else {
            None
        };
        let uncertainty = match p.get_opt_f64("uncertainty")? {
            Some(sigma) => Some((
                sigma,
                p.get_usize("samples", 100)?,
                p.get_usize("seed", 42)? as u64,
            )),
            None => None,
        };
        let level = NetworkLevel::from_number(p.get_usize("level", 1)? as u64)?;
        let extras = extras(p)?;
        let job = KineticsJob {
            network,
            level,
            annotations: p.get_opt_str("annotations").map(PathBuf::from),
            settings,
            times: parse_times(p)?,
            initial,
            validate,
            sensitivity,
            uncertainty,
            priority: p.get_opt_str("priority").map(Score::parse).transpose()?,
            lump_conformers: p.get_bool("lump_conformers", false),
            spectrum: p.get_bool("spectrum", false),
            output: Some(PathBuf::from(
                p.get_opt_str("output_dir").unwrap_or("kinetics_output"),
            )),
            report: Report::Kinetics,
            settings_document: Some(settings_json(p, RUN_KINETICS).to_string_pretty()),
            cli_annotations,
            extras,
        };
        Ok(Self {
            job,
            console: console_level(p, Level::Info)?,
        })
    }

    /// The settings of `run_rcmc` (MultiOptPy's options; its numbers).
    pub fn from_rcmc_args(p: &ParsedArgs) -> MoptResult<Self> {
        reject_unavailable(p, RUN_RCMC)?;
        let network = network_path(p, "run_rcmc")?;
        let conditions = Conditions {
            temperature: p.get_f64("temperature", 300.0)?,
            energy: EnergyKind::parse(p.get_str("energy_type", "electronic"))?,
            clamp: Clamp::Legacy,
            ..Conditions::default()
        };
        conditions.check()?;
        let time = p.get_f64("time", 1.0)?;
        if !(time.is_finite() && time >= 0.0) {
            return Err(MoptError::parse("--time must not be negative"));
        }
        let start = p.get_usize("start_node", 0)?;
        let settings = KineticsSettings {
            conditions,
            solver: "legacy_rcmc".to_string(),
            ..KineticsSettings::default()
        };
        // MultiOptPy's levels decide what its logging shows; the results are printed at any.
        let console = match p.get_opt_str("log_level") {
            Some(l) if l.eq_ignore_ascii_case("debug") => Level::Debug,
            _ => Level::Info,
        };
        let output = p.get_opt_str("output_dir").map(PathBuf::from);
        let job = KineticsJob {
            network,
            level: NetworkLevel::Exploration,
            annotations: None,
            settings,
            times: vec![time],
            initial: vec![(start, 1.0)],
            validate: None,
            sensitivity: None,
            uncertainty: None,
            priority: None,
            lump_conformers: false,
            spectrum: false,
            settings_document: output
                .as_ref()
                .map(|_| settings_json(p, RUN_RCMC).to_string_pretty()),
            output,
            report: Report::Rcmc,
            cli_annotations: None,
            extras: KineticsExtras::default(),
        };
        Ok(Self { job, console })
    }
}

/// Run under `base` (relative paths of the job are taken from there).
pub fn run_with(
    c: &KineticsConfig,
    base: &Path,
    console: Option<Level>,
) -> MoptResult<KineticsRun> {
    run_with_parts(c, base, console, &KineticsParts::default())
}

/// [`run_with`] with further rate theories, corrections, models, and solvers.
pub fn run_with_parts(
    c: &KineticsConfig,
    base: &Path,
    console: Option<Level>,
    parts: &KineticsParts,
) -> MoptResult<KineticsRun> {
    let mut job = c.job.clone();
    job.network = base.join(&job.network);
    job.annotations = job.annotations.map(|a| base.join(a));
    if let Some(md) = &mut job.extras.md_trajectory {
        md.path = base.join(&md.path);
    }
    job.output = job.output.map(|o| base.join(o));
    let log = console.map_or_else(Logger::silent, Logger::console);
    if let Some(dir) = &job.output {
        std::fs::create_dir_all(dir).map_err(|e| MoptError::from(e).context(dir.display()))?;
        let path = dir.join("mopt.log");
        let _ = std::fs::remove_file(&path);
        log.add_file(path, Level::Debug)?;
        log.info(&format!(
            "mopt-rs {} {}: {}",
            crate::VERSION,
            match job.report {
                Report::Rcmc => "run_rcmc",
                Report::Kinetics => "run_kinetics",
            },
            job.network.display()
        ));
    }
    let run = run_job(&job, parts, &log);
    log.flush();
    run
}
