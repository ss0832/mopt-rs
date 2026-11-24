//! Settings of `confsearch`: MultiOptPy's options (`-bf`, `-ms`, `-nl`, `-nr`, `-tgta`, `-st`,
//! `-nost`, `-pbc`, `-tabu`, `-alpha`), every option of `optmain` for the optimizations, the
//! transition-state mode (`-tsconf`), and the settings of mopt-rs as `-cs_opt KEY=VALUE`.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py:conformation_search` (the parser) and
//!  `interface.py:optimizeparser`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use super::option::reject_unavailable;
use super::optmain::OptmainConfig;
use super::settings::settings_json;
use super::spec::CONFSEARCH;
use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::workflow::confsearch::afir::{AfirSettings, PairRule};
use crate::workflow::confsearch::imtd::{crest_runs, ImtdSettings, MtdRun, Shake};
use crate::workflow::confsearch::moves::MoveOptions;
use crate::workflow::confsearch::ts::{Hold, TsSettings};
use crate::workflow::confsearch::{CommonSettings, GraphCheck, Mode};
use crate::workflow::explore::CrossoverSettings;

/// The strategy of the search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Method {
    /// MultiOptPy's sampling with AFIR forces (`method=afir`).
    Afir,
    /// iMTD-GC (`method=imtd`).
    Imtd,
    /// Trials by a move of the structure (`method=torsion`, `kick`, `lowmode`, or the name
    /// of a move a user registered; [`crate::workflow::confsearch::moves`]). The name is
    /// looked up in the registry of the run.
    Move(String),
}

/// Typed `confsearch` settings.
#[derive(Debug, Clone)]
pub struct ConfsearchConfig {
    /// The input structure.
    pub input: PathBuf,
    /// The settings of the optimizations (every option of `optmain`).
    pub optmain: OptmainConfig,
    /// Whether `-opt` was given (otherwise saddle-point searches take the methods of `-order 1`).
    pub methods_given: bool,
    /// The settings in effect as a settings document (`settings.json`, with the seed).
    pub settings_document: String,
    /// Run directory to continue (`--restart`).
    pub restart_dir: Option<PathBuf>,
    /// The strategy.
    pub method: Method,
    /// MultiOptPy's sampling.
    pub afir: AfirSettings,
    /// iMTD-GC.
    pub imtd: ImtdSettings,
    /// The settings of a move (`-cs_opt move_KEY=VALUE`).
    pub moves: MoveOptions,
    /// Settings of every strategy.
    pub common: CommonSettings,
    /// Name of the sampling priority (`-cs_opt priority=`; MultiOptPy's selection without it).
    pub priority: String,
    /// `-st`: temperature of the selection (K).
    pub sampling_temperature: f64,
    /// `-tabu`: the visit penalty `-alpha`.
    pub tabu: Option<f64>,
    /// Temperature of the populations (`-temp`, K).
    pub temperature: f64,
    /// `-tsconf`: conformers of a transition state.
    pub ts: Option<TsSettings>,
    /// `-freq`: a vibrational analysis of every conformer at the end.
    pub frequencies: bool,
    /// Every converged conformer is displaced a little and optimized once more, so that
    /// saddle points of symmetry (eclipsed methyl groups) are left (`-cs_opt nudge=`).
    pub nudge: bool,
}

/// MultiOptPy's run directory `<stem>_<bf>KJ_CS_<date>` is placed as the other commands
/// place theirs (`<date>/<stem>_<bf>KJ_CS_<timestamp>/`).
pub const RUN_TAG: &str = "KJ_CS";

fn number(key: &str, value: &str) -> MoptResult<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| MoptError::parse(format!("-cs_opt {key}={value}: not a number")))
}

fn whole(key: &str, value: &str) -> MoptResult<usize> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| MoptError::parse(format!("-cs_opt {key}={value}: not a whole number")))
}

/// A switch of `-cs_opt` (`core_infra::arg_parser::parse_switch`; `none` is off as well).
fn switch(key: &str, value: &str) -> MoptResult<bool> {
    if value.trim().eq_ignore_ascii_case("none") {
        return Ok(false);
    }
    crate::core_infra::arg_parser::parse_switch(&format!("-cs_opt {key}"), value)
}

fn atoms(key: &str, value: &str) -> MoptResult<Vec<usize>> {
    let list = parse_atom_indices_one_based(value.trim())
        .map_err(|e| MoptError::parse(format!("-cs_opt {key}={value}: {e}")))?;
    list.into_iter()
        .map(|a| {
            a.checked_sub(1)
                .ok_or_else(|| MoptError::parse(format!("-cs_opt {key}: atoms count from 1")))
        })
        .collect()
}

/// `k/N:alpha,k/N:alpha,...` with `k/N` in mEh (`;` separates as well).
fn runs(value: &str) -> MoptResult<Vec<MtdRun>> {
    if value.trim().eq_ignore_ascii_case("crest") {
        return Ok(crest_runs());
    }
    value
        .split([',', ';'])
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            let (k, a) = t.split_once(':').ok_or_else(|| {
                MoptError::parse(format!(
                    "-cs_opt mtd_runs: '{t}' is not K_PER_ATOM_MEH:ALPHA"
                ))
            })?;
            Ok(MtdRun {
                k_per_atom: number("mtd_runs", k)? * 1e-3,
                alpha: number("mtd_runs", a)?,
            })
        })
        .collect()
}

fn list(key: &str, value: &str) -> MoptResult<Vec<f64>> {
    value
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| number(key, t))
        .collect()
}

/// Apply the `-cs_opt` settings; returns whether a seed was given.
fn apply(config: &mut ConfsearchConfig, tokens: &[String]) -> MoptResult<bool> {
    let kcal = Units::KCAL_MOL_TO_HARTREE;
    let fs = Units::FS_TO_AU_TIME;
    let mut seed_given = false;
    for token in tokens {
        let (key, value) = token
            .split_once('=')
            .ok_or_else(|| MoptError::parse(format!("-cs_opt expects KEY=VALUE, got '{token}'")))?;
        let key = key.trim().to_ascii_lowercase();
        let c = &mut config.common;
        let m = &mut config.imtd;
        match key.as_str() {
            "method" => {
                config.method = match value.trim().to_ascii_lowercase().as_str() {
                    "afir" | "multioptpy" => Method::Afir,
                    "imtd" | "imtd-gc" | "imtd_gc" | "crest" => Method::Imtd,
                    "" => {
                        return Err(MoptError::parse(
                            "-cs_opt method=: afir, imtd, or a move (torsion, kick, lowmode, \
                             or one that was registered)",
                        ))
                    }
                    // A move by name: the registry of the run knows it, or refuses it.
                    other => Method::Move(other.to_string()),
                }
            }
            key if key.starts_with("move_") => {
                config
                    .moves
                    .0
                    .insert(key["move_".len()..].to_string(), value.trim().to_string());
            }
            "mode" => {
                c.mode = match value.trim().to_ascii_lowercase().as_str() {
                    "conformers" => Mode::Conformers,
                    "minima" | "eq" => Mode::Minima,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-cs_opt mode={other}: conformers or minima"
                        )))
                    }
                }
            }
            "graph" => {
                c.graph_check = match value.trim().to_ascii_lowercase().as_str() {
                    "isomorphic" => GraphCheck::Isomorphic,
                    "numbering" | "same" => GraphCheck::SameNumbering,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-cs_opt graph={other}: isomorphic or numbering"
                        )))
                    }
                }
            }
            "bond_scale" => c.bond_scale = number(&key, value)?,
            "seed" => {
                c.seed = value
                    .trim()
                    .parse::<u64>()
                    .map_err(|e| MoptError::parse(format!("-cs_opt seed={value}: {e}")))?;
                seed_given = true;
            }
            "priority" => config.priority = value.trim().to_ascii_lowercase(),
            "ethr" => c.compare.energy = number(&key, value)? * kcal,
            "rthr" => c.compare.rmsd = number(&key, value)? * Units::ANGSTROM_TO_BOHR,
            "bthr" => c.compare.rotational = number(&key, value)?,
            "mirror" => c.compare.mirror = switch(&key, value)?,
            "permutations" => c.compare.permutations = switch(&key, value)?,
            "compare_atoms" => c.compare_atoms = atoms(&key, value)?,
            "loose_window" => c.loose_window = number(&key, value)? * kcal,
            "skip_known" => c.skip_known = switch(&key, value)?,
            "nudge" => config.nudge = switch(&key, value)?,
            "verify" => c.verify = switch(&key, value)?,
            "pairs" => {
                let v = value.trim().to_ascii_lowercase();
                config.afir.pairs = match v.split_once(':') {
                    None if v == "multioptpy" => PairRule::MultiOptPy { min: 1.0, max: 8.0 },
                    None if v == "ratio" => PairRule::Ratio { min: 1.0, max: 8.0 },
                    None if v == "bonds" => PairRule::Bonds {
                        min_bonds: 3,
                        max_ratio: 8.0,
                    },
                    Some(("bonds", n)) => PairRule::Bonds {
                        min_bonds: whole(&key, n)?,
                        max_ratio: 8.0,
                    },
                    _ => {
                        return Err(MoptError::parse(format!(
                            "-cs_opt pairs={v}: multioptpy, ratio, or bonds[:N]"
                        )))
                    }
                }
            }
            "mtd_runs" => m.runs = runs(value)?,
            "mtd_time" => m.mtd_time = number(&key, value)? * 1000.0 * fs,
            "time_step" => m.time_step = number(&key, value)? * fs,
            "mtd_temperature" => m.temperature = number(&key, value)?,
            "dump" => m.dump_every = number(&key, value)? * fs,
            "vbias" => m.deposit_every = number(&key, value)? * fs,
            "known_references" => m.known_references = whole(&key, value)?,
            "shake" => {
                m.shake = match value.trim().to_ascii_lowercase().as_str() {
                    "off" | "0" | "none" => Shake::Off,
                    "h" | "hydrogens" | "1" => Shake::Hydrogens,
                    "all" | "2" => Shake::All,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-cs_opt shake={other}: off, hydrogens, or all"
                        )))
                    }
                }
            }
            "hmass" => {
                let v = number(&key, value)?;
                m.hydrogen_mass = (v > 0.0).then_some(v);
            }
            "rounds" => m.max_rounds = whole(&key, value)?.max(1),
            "bias_atoms" => m.bias_atoms = atoms(&key, value)?,
            "crossover" => {
                m.crossover = if switch(&key, value)? {
                    Some(
                        m.crossover
                            .unwrap_or((CrossoverSettings::default(), 6.0 * kcal)),
                    )
                } else {
                    None
                }
            }
            "crossover_window" => {
                let (s, _) = m.crossover.unwrap_or((CrossoverSettings::default(), 0.0));
                m.crossover = Some((s, number(&key, value)? * kcal));
            }
            "crossover_max" => {
                let (mut s, w) = m
                    .crossover
                    .unwrap_or((CrossoverSettings::default(), 6.0 * kcal));
                s.max = whole(&key, value)?;
                m.crossover = Some((s, w));
            }
            "finish" => {
                if !switch(&key, value)? {
                    m.finishing = None;
                } else if m.finishing.is_none() {
                    m.finishing = ImtdSettings::default().finishing;
                }
            }
            "finish_conformers" | "finish_temperatures" | "finish_time" => {
                let mut f = m
                    .finishing
                    .clone()
                    .or(ImtdSettings::default().finishing)
                    .ok_or_else(|| MoptError::config("no finishing dynamics"))?;
                match key.as_str() {
                    "finish_conformers" => f.conformers = whole(&key, value)?,
                    "finish_temperatures" => f.temperatures = list(&key, value)?,
                    _ => f.time = number(&key, value)? * 1000.0 * fs,
                }
                m.finishing = Some(f);
            }
            "ts_hold" | "ts_update" | "ts_window" | "ts_max" | "ts_irc" | "ts_share"
            | "ts_bond_scale" | "ts_bonds" => {
                let t = config.ts.get_or_insert_with(TsSettings::default);
                match key.as_str() {
                    "ts_hold" => {
                        t.hold = match value.trim().to_ascii_lowercase().as_str() {
                            "coordinate" => Hold::Coordinate,
                            "modes" | "freeze" => Hold::FrozenModes { update: 10 },
                            other => {
                                return Err(MoptError::parse(format!(
                                    "-cs_opt ts_hold={other}: coordinate or modes"
                                )))
                            }
                        }
                    }
                    "ts_update" => {
                        t.hold = Hold::FrozenModes {
                            update: whole(&key, value)?.max(1),
                        }
                    }
                    "ts_window" => t.window = number(&key, value)? * kcal,
                    "ts_max" => t.max_candidates = whole(&key, value)?,
                    "ts_irc" => t.check_reaction = switch(&key, value)?,
                    "ts_share" => t.reactive_share = number(&key, value)?,
                    "ts_bonds" => t.hold_bonds = switch(&key, value)?,
                    _ => t.bond_scale = number(&key, value)?,
                }
            }
            other => {
                return Err(MoptError::parse(format!(
                    "-cs_opt: unknown setting '{other}' (method, mode, graph, bond_scale, seed, \
                     priority, ethr, rthr, bthr, mirror, permutations, compare_atoms, \
                     loose_window, skip_known, nudge, verify, pairs, mtd_runs, mtd_time, time_step, \
                     mtd_temperature, dump, vbias, known_references, shake, hmass, rounds, \
                     bias_atoms, crossover, crossover_window, crossover_max, finish, \
                     finish_conformers, finish_temperatures, finish_time, ts_hold, ts_update, \
                     ts_window, ts_max, ts_irc, ts_share, ts_bonds, ts_bond_scale)"
                )))
            }
        }
    }
    Ok(seed_given)
}

fn clock_seed() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    ((now as u64) ^ ((now >> 64) as u64)) % 1_000_000_007 + 1
}

impl ConfsearchConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        // The state that the input file names, for the runs started on files written here.
        let with_state = super::charge_spin::with_state_of_input(p);
        let p = &with_state;
        reject_unavailable(p, CONFSEARCH)?;
        let mut optmain = OptmainConfig::from_args(p)?;
        // Without -tcc or -lcc the conformers are optimized with the tight criteria: with
        // the normal ones the flat torsions of butane already end 2 to 5 degrees and 0.02
        // to 0.06 kcal/mol apart, so that one conformer appears several times (running.md).
        let tight_default = !p.get_bool("tight_convergence_criteria", false)
            && !p.get_bool("loose_convergence_criteria", false);
        if tight_default {
            optmain.settings.criteria = crate::opt::convergence::ConvergenceCriteria::from_flags(
                true,
                false,
                !p.get_list("projection_constrain").is_empty(),
            );
        }
        let input = match optmain.inputs.as_slice() {
            [one] => one.clone(),
            [] => {
                return Err(MoptError::config(
                    "confsearch needs an input structure (xyz)",
                ))
            }
            _ => {
                return Err(MoptError::config(
                    "confsearch searches the conformers of one structure; give one input",
                ))
            }
        };
        let target_atoms = match p.get_list("target_atoms").first() {
            Some(list) => parse_atom_indices_one_based(list)?
                .into_iter()
                .map(|a| {
                    a.checked_sub(1)
                        .ok_or_else(|| MoptError::parse("-tgta: atoms count from 1"))
                })
                .collect::<MoptResult<_>>()?,
            None => Vec::new(),
        };
        let afir = AfirSettings {
            base_force: p.get_f64("base_force", 100.0)?,
            max_samples: p.get_usize("max_samples", 50)?,
            rank: p.get_usize("number_of_rank", 10)?,
            patience: p.get_usize("number_of_lowest", 5)?,
            target_atoms,
            no_stochastic: p.get_bool("no_stochastic", false),
            ..AfirSettings::default()
        };
        let mut common = CommonSettings::default();
        if p.get_bool("preserve_bond_connectivity", false) {
            // MultiOptPy's -pbc compares the bond tables in the numbering of the input.
            common.graph_check = GraphCheck::SameNumbering;
        }
        let tabu = p
            .get_bool("tabu_search", false)
            .then(|| p.get_f64("tabu_alpha", 0.5))
            .transpose()?;
        let mut config = Self {
            input,
            methods_given: !p.get_list("opt_method").is_empty(),
            settings_document: String::new(),
            restart_dir: None,
            method: Method::Afir,
            afir,
            imtd: ImtdSettings::default(),
            moves: MoveOptions::default(),
            common,
            priority: String::new(),
            sampling_temperature: p.get_f64("sampling_temperature", 298.15)?,
            tabu,
            temperature: p.get_f64("temperature", 298.15)?,
            ts: p.get_bool("ts_conformers", false).then(TsSettings::default),
            frequencies: optmain.frequency_analysis,
            nudge: true,
            optmain,
        };
        let seed_given = apply(&mut config, p.get_list("confsearch_options"))?;
        if config.ts.is_some() && !p.get_bool("ts_conformers", false) {
            return Err(MoptError::parse(
                "-cs_opt ts_...: the settings of transition-state conformers need -tsconf",
            ));
        }
        if !seed_given {
            config.common.seed = clock_seed();
        }
        // The settings document keeps the seed, so that a restart repeats the run.
        let mut q = p.clone();
        let mut tokens: Vec<String> = q
            .get_list("confsearch_options")
            .iter()
            .filter(|t| !t.trim().to_ascii_lowercase().starts_with("seed="))
            .cloned()
            .collect();
        tokens.push(format!("seed={}", config.common.seed));
        q.values.insert("confsearch_options".to_string(), tokens);
        q.present.insert("confsearch_options".to_string());
        config.settings_document = settings_json(&q, CONFSEARCH).to_string_pretty();
        config.check()?;
        Ok(config)
    }

    /// Combinations that cannot make sense.
    fn check(&self) -> MoptResult<()> {
        let o = &self.optmain;
        if o.variable_cell.is_some() {
            return Err(MoptError::unsupported(
                "-vc: a conformation search keeps the cell; it compares molecules, and a cell \
                 that changes with every optimization has no ensemble",
            ));
        }
        if o.initial_hessian.is_some() {
            return Err(MoptError::unsupported(
                "-ih gives the Hessian of one structure; confsearch optimizes many",
            ));
        }
        if o.model_function
            .first()
            .is_some_and(|m| m.eq_ignore_ascii_case("bitss"))
        {
            return Err(MoptError::unsupported(
                "-mf bitss is a search between two structures, not a surface",
            ));
        }
        let per_run = [
            (o.cmds, "-cmds"),
            (o.pca, "-pca"),
            (o.koopman, "-km"),
            (!o.geometry_info.is_empty(), "-gi"),
            (o.detect_negative_eigenvalues, "-negeigval"),
        ];
        if let Some((_, flag)) = per_run.iter().find(|(on, _)| *on) {
            return Err(MoptError::unsupported(format!(
                "{flag} describes the iterations of one optimization; confsearch runs many \
                 (run optmain on a conformer for it)"
            )));
        }
        if self.ts.is_none() && o.settings.saddle_order > 0 {
            return Err(MoptError::unsupported(
                "-order: conformers are minima; for the conformers of a transition state give \
                 -tsconf",
            ));
        }
        let c = &self.common;
        if !(c.compare.energy > 0.0 && c.compare.rmsd > 0.0 && c.compare.rotational > 0.0) {
            return Err(MoptError::config("ethr, rthr, and bthr must be positive"));
        }
        if self.afir.base_force <= 0.0 && self.method == Method::Afir {
            return Err(MoptError::config("-bf must be positive"));
        }
        Ok(())
    }

    /// The window of the first, loose optimizations in kcal/mol (for the log).
    pub fn loose_window_kcal(&self) -> f64 {
        self.common.loose_window * Units::HARTREE_TO_KCAL_MOL
    }
}

/// Thin wrapper used by tests and embedding code: settings for `args`.
pub fn config_from_cli(args: &[&str]) -> MoptResult<ConfsearchConfig> {
    let p = super::option::parser("confsearch", CONFSEARCH).parse(args)?;
    ConfsearchConfig::from_args(&p)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn multioptpy_options_and_the_settings_of_mopt_rs_are_read() {
        let c = config_from_cli(&[
            "x.xyz",
            "-os",
            "gfnff",
            "-ms",
            "7",
            "-nl",
            "2",
            "-nr",
            "3",
            "-bf",
            "150",
            "-tabu",
            "-alpha",
            "0.3",
            "-pbc",
            "-tgta",
            "1-4",
            "-cs_opt",
            "seed=5",
            "method=imtd",
            "mtd_runs=3:1.3,1.5:0.78",
            "ethr=0.1",
            "shake=all",
        ])
        .unwrap();
        assert_eq!(c.afir.max_samples, 7);
        assert_eq!((c.afir.rank, c.afir.patience), (3, 2));
        assert_eq!(c.afir.target_atoms, vec![0, 1, 2, 3]);
        assert_eq!(c.tabu, Some(0.3));
        assert_eq!(c.common.graph_check, GraphCheck::SameNumbering);
        assert_eq!(c.method, Method::Imtd);
        assert_eq!(c.imtd.runs.len(), 2);
        assert!((c.imtd.runs[0].k_per_atom - 3e-3).abs() < 1e-15);
        assert_eq!(c.imtd.shake, Shake::All);
        assert_eq!(c.common.seed, 5);
        assert!(c.settings_document.contains("seed=5"));
        // A seed is drawn and kept when none is given.
        let d = config_from_cli(&["x.xyz", "-os", "gfnff"]).unwrap();
        // The pairs at least 3 bonds apart by default (measured); MultiOptPy's window on request.
        assert_eq!(
            d.afir.pairs,
            PairRule::Bonds {
                min_bonds: 3,
                max_ratio: 8.0
            }
        );
        let m = config_from_cli(&["x.xyz", "-cs_opt", "pairs=multioptpy"]).unwrap();
        assert_eq!(m.afir.pairs, PairRule::MultiOptPy { min: 1.0, max: 8.0 });
        assert!(d
            .settings_document
            .contains(&format!("seed={}", d.common.seed)));
        // Refused combinations.
        for bad in [
            vec!["x.xyz", "-vc"],
            vec!["x.xyz", "-order", "1"],
            vec!["x.xyz", "-cmds"],
            vec!["x.xyz", "-cs_opt", "nonsense=1"],
            vec!["x.xyz", "-cs_opt", "ts_window=3"],
        ] {
            assert!(config_from_cli(&bad).is_err(), "{bad:?}");
        }
        assert!(config_from_cli(&["x.xyz", "-tsconf", "-cs_opt", "ts_window=3"]).is_ok());
    }
}
