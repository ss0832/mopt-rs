//! `etmain`: the rate constant of an electron transfer from one structure
//! ([`crate::workflow::electron_transfer`], `docs/source/electron_transfer.md`).
//!
//! The structure is the system with the moving electron on the donor. The fragments are
//! built from `-et_donor` and `-et_acceptor`: separate molecules when no bond joins them,
//! otherwise a molecule of donor, bridge, and acceptor (the other atoms, or `-et_bridge`),
//! whose donor-bridge and bridge-acceptor fragments have the bonds they cut capped by
//! hydrogens. `-et_charges` are the charges of the two fragments without the moving electron;
//! the multiplicities are the lowest ones unless `-et_mult` gives them. The optimizer settings
//! and the calculator are those of `optmain`.
//!
//! The barriers: `dE` of the electronic energies with the solvent shift (the thesis) and, by
//! default (`-et_free_energy off` leaves it out), the free energy of activation `dG` from the
//! harmonic thermal corrections at the initial equilibrium and within the seam at each
//! crossing (`-press`, `-freq_corr`, and the other settings of the thermochemistry of
//! `optmain`; the temperature is `-et_temperature`). The rate constants are given with both,
//! and with each the apparent free energy of activation that the Eyring equation gives it.
//!
//! The run writes `mopt.log`, `et_scan.csv` (one row per crossing), `et_summary.json`, the
//! equilibria `eq_initial.xyz` and `eq_final.xyz`, and the crossings `crossings.xyz`.
//!
//! `-et_method hopping` is the hop of a charge (`-et_carrier hole` or `electron`) between two
//! molecules instead ([`crate::workflow::hopping`]): `-et_donor` is the molecule it starts on,
//! `-et_acceptor` the other, `-et_charges` their charges without the carrier, and the
//! structure is the pair in those charges. The transfer integral of the projection method at
//! that structure, the four-point reorganization energy from the equilibria of both molecules
//! in both charges, and the Marcus rates both ways; the run writes `mopt.log`,
//! `hop_summary.json`, and the equilibria `molecule_{a,b}_{without,with}.xyz`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use super::optmain::OptmainConfig;
use crate::core_infra::arg_parser::{parse_atom_indices_one_based, ParsedArgs};
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{AtomId, MolecularSystem, Quantity, SpinCharge, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{format_xyz_system, read_xyz_file};
use crate::kinetics::rate::transfer_integral::Carrier;
use crate::pes::combine::{EdeelLayout, EdeelStates};
use crate::pes::oniom::layout::LinkPlacement;
use crate::pes::registry::CalculatorRegistry;
use crate::vib::ThermoSettings;
use crate::workflow::driver::CalculatorSet;
use crate::workflow::electron_transfer::{
    apparent_barrier, run_transfer, AutoCoupling, CouplingSource, CrossingResult, CrossingSearch,
    EdeelCoupling, ExcitedStateCoupling, ExcitedStateFormula, GivenCoupling, MullikenHushCoupling,
    TransferResult, TransferSettings,
};
use crate::workflow::hopping::{run_hopping, HoppingResult, HoppingSettings};
use crate::workflow::journal::RunJournal;
use crate::workflow::narrated::Narrated;
use crate::workflow::observers::{ProgressLog, RunFiles, StopFile};
use crate::workflow::Observer;

/// What `etmain` computes (`-et_method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtMethod {
    /// The transfer within a molecule of donor, bridge, and acceptor by EDEEL.
    Edeel,
    /// The hop of a charge between two molecules.
    Hopping,
}

impl EtMethod {
    /// `edeel` or `hopping`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "edeel" => Ok(Self::Edeel),
            "hopping" | "hop" => Ok(Self::Hopping),
            other => Err(MoptError::parse(format!(
                "-et_method is edeel or hopping, got '{other}'"
            ))),
        }
    }
}

/// Where the coupling comes from (`-et_coupling`).
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingChoice {
    /// Mulliken-Hush when the calculator gives transitions, else EDEEL's.
    Auto,
    /// EDEEL's adiabatic coupling.
    Edeel,
    /// Mulliken-Hush.
    MullikenHush,
    /// Generalized Mulliken-Hush.
    GeneralizedMullikenHush,
    /// Fragment charge difference.
    FragmentChargeDifference,
    /// A given value (Hartree).
    Given(f64),
}

/// The settings of `etmain`.
#[derive(Debug, Clone)]
pub struct EtmainConfig {
    /// The optmain settings (calculator, state, optimizer).
    pub base: OptmainConfig,
    /// `-et_method`.
    pub method: EtMethod,
    /// `-et_carrier` (of `-et_method hopping`).
    pub carrier: Carrier,
    /// The donor atoms.
    pub donor: Vec<AtomId>,
    /// The acceptor atoms.
    pub acceptor: Vec<AtomId>,
    /// `None`: found (`auto`); `Some(empty)`: separate molecules; else the bridge atoms.
    pub bridge: Option<Vec<AtomId>>,
    /// The charges of the donor and acceptor fragments without the moving electron.
    pub charges: [i32; 2],
    /// The five multiplicities, when given (`-et_method edeel`).
    pub multiplicities: Option<[u32; 5]>,
    /// The four multiplicities of the two molecules without and with the carrier, when given
    /// (`-et_method hopping`).
    pub hopping_multiplicities: Option<[u32; 4]>,
    /// The solvent reorganization energies (Hartree).
    pub solvent: Option<[f64; 2]>,
    /// `alpha_SA` of the scan.
    pub alphas: Vec<f64>,
    /// The coupling.
    pub coupling: CouplingChoice,
    /// Temperature (K).
    pub temperature: f64,
    /// The start of the final state.
    pub final_start: Option<PathBuf>,
    /// Excited states searched for the charge-transfer state.
    pub states: usize,
    /// `-et_free_energy`: the free energies of activation (Hessians at the initial
    /// equilibrium and at the crossings).
    pub free_energy: bool,
    /// The settings of their thermochemistry.
    pub thermo: ThermoSettings,
    /// `-et_crossing`: how the crossings are searched.
    pub crossing_search: CrossingSearch,
}

fn atoms(text: &str, what: &str) -> MoptResult<Vec<AtomId>> {
    parse_atom_indices_one_based(text)
        .map_err(|e| MoptError::parse(format!("{what}: {e}")))?
        .into_iter()
        .map(AtomId::from_one_based)
        .collect()
}

fn alphas(text: &str) -> MoptResult<Vec<f64>> {
    let parts: Vec<f64> = text
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f64>()
                .map_err(|_| MoptError::parse(format!("-et_alpha: '{t}' is not a number")))
        })
        .collect::<MoptResult<_>>()?;
    let [start, end, step] = parts[..] else {
        return Err(MoptError::parse("-et_alpha takes START,END,STEP"));
    };
    if !(step.is_finite() && step > 0.0) {
        return Err(MoptError::parse("-et_alpha: the step must be positive"));
    }
    let n = ((start - end).abs() / step + 1e-9).floor() as usize;
    let sign = if end < start { -1.0 } else { 1.0 };
    Ok((0..=n).map(|k| start + sign * step * k as f64).collect())
}

impl EtmainConfig {
    /// The settings of the command line `p`.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        let base = OptmainConfig::from_args(p)?;
        let donor = atoms(
            p.get_opt_str("et_donor")
                .ok_or_else(|| MoptError::config("etmain needs -et_donor"))?,
            "-et_donor",
        )?;
        let acceptor = atoms(
            p.get_opt_str("et_acceptor")
                .ok_or_else(|| MoptError::config("etmain needs -et_acceptor"))?,
            "-et_acceptor",
        )?;
        let bridge = match p.get_str("et_bridge", "auto").trim() {
            "auto" => None,
            "none" => Some(Vec::new()),
            list => Some(atoms(list, "-et_bridge")?),
        };
        let charges = p.get_list("et_charges");
        let charges = match charges {
            [] => [0, 0],
            [a, b] => [
                a.parse()
                    .map_err(|_| MoptError::parse("-et_charges: integers"))?,
                b.parse()
                    .map_err(|_| MoptError::parse("-et_charges: integers"))?,
            ],
            _ => return Err(MoptError::parse("-et_charges takes two charges")),
        };
        let method = EtMethod::parse(p.get_str("et_method", "edeel"))?;
        let given: Vec<u32> = p
            .get_list("et_mult")
            .iter()
            .map(|t| {
                t.parse::<u32>()
                    .ok()
                    .filter(|&m| m > 0)
                    .ok_or_else(|| MoptError::parse("-et_mult: positive integers"))
            })
            .collect::<MoptResult<_>>()?;
        let (multiplicities, hopping_multiplicities) = match (method, &given[..]) {
            (_, []) => (None, None),
            (EtMethod::Edeel, &[a, b, c, d, e]) => (Some([a, b, c, d, e]), None),
            (EtMethod::Hopping, &[a, b, c, d]) => (None, Some([a, b, c, d])),
            (EtMethod::Edeel, _) => {
                return Err(MoptError::parse("-et_mult takes five multiplicities"))
            }
            (EtMethod::Hopping, _) => {
                return Err(MoptError::parse(
                    "-et_mult takes four multiplicities with -et_method hopping",
                ))
            }
        };
        let solvent = match p.get_list("et_lambda_s") {
            [] => None,
            [a, b] => {
                let ev = |t: &String| -> MoptResult<f64> {
                    t.parse::<f64>()
                        .map(|v| v * Units::EV_TO_HARTREE)
                        .map_err(|_| MoptError::parse("-et_lambda_s: numbers (eV)"))
                };
                Some([ev(a)?, ev(b)?])
            }
            _ => return Err(MoptError::parse("-et_lambda_s takes two energies (eV)")),
        };
        let coupling = match p
            .get_str("et_coupling", "auto")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "auto" => CouplingChoice::Auto,
            "edeel" => CouplingChoice::Edeel,
            "mulliken_hush" | "mh" => CouplingChoice::MullikenHush,
            "gmh" | "generalized_mulliken_hush" => CouplingChoice::GeneralizedMullikenHush,
            "fcd" | "fragment_charge_difference" => CouplingChoice::FragmentChargeDifference,
            other => CouplingChoice::Given(
                other.parse::<f64>().map_err(|_| {
                    MoptError::parse(format!(
                        "-et_coupling: auto, edeel, mulliken_hush, gmh, fcd, or a value in eV, \
                         got '{other}'"
                    ))
                })? * Units::EV_TO_HARTREE,
            ),
        };
        Ok(Self {
            base,
            method,
            carrier: Carrier::parse(p.get_str("et_carrier", "hole"))?,
            donor,
            acceptor,
            bridge,
            charges,
            multiplicities,
            hopping_multiplicities,
            solvent,
            alphas: alphas(p.get_str("et_alpha", "1,0,0.1"))?,
            coupling,
            temperature: p.get_f64("et_temperature", 298.15)?,
            final_start: p.get_opt_str("et_final").map(PathBuf::from),
            states: p.get_usize("et_states", 10)?,
            free_energy: match p.get_str("et_free_energy", "on").trim() {
                "on" => true,
                "off" => false,
                other => {
                    return Err(MoptError::parse(format!(
                        "-et_free_energy is on or off, got '{other}'"
                    )))
                }
            },
            thermo: super::optmain::frequency_settings(p)?.thermo,
            crossing_search: CrossingSearch::parse(p.get_str("et_crossing", "projection"))?,
        })
    }
}

/// The settings of the command line `args` (the options of `etmain`).
pub fn config_from_cli(args: &[&str]) -> MoptResult<EtmainConfig> {
    let p = super::option::parser("etmain", super::spec::ETMAIN).parse(args)?;
    EtmainConfig::from_args(&p)
}

/// The settings of the crossing searches: those of the equilibria with the largest trust radius
/// of a gradient projection with RS-I-RFO ([`super::optmain::PROJECTION_MAX_TRUST_ANGSTROM`],
/// as `optmain -mf opt_mesx` has it) unless `-tr` gives one or the methods are others.
fn crossing_settings(
    settings: &crate::opt::state_machine::OptimizerSettings,
) -> crate::opt::state_machine::OptimizerSettings {
    let methods = if settings.methods.is_empty() {
        crate::opt::registry::default_methods(settings.saddle_order)
    } else {
        settings.methods.clone()
    };
    let rsirfo = methods
        .iter()
        .any(|m| m.to_ascii_lowercase().contains("rsirfo"));
    let mut out = settings.clone();
    if rsirfo && out.trust.max.is_none() {
        out.trust.max = Some(Quantity::new(
            super::optmain::PROJECTION_MAX_TRUST_ANGSTROM * Units::ANGSTROM_TO_BOHR,
        ));
    }
    out
}

/// The model Hessian the optimizations start from: Lindh's unless `-modelhess` names another or
/// `-fc N` asks for exact Hessians (every `N` iterations, the first included; the exact
/// Hessians of the diabatic states, caps included, and of the crossing), which then stay as
/// they are. Measured on the radical anion of 1,4-pentadiene with GFN1-xTB (normal criteria;
/// the equilibria of both states and the crossing; `docs/source/electron_transfer.md`): from
/// Lindh's 57, 58, and 46 iterations (98 s), from the exact Hessian at the start 68, 63, and
/// 44 (100 s), the barriers the same to 0.1 meV; from the unit matrix, on an earlier start,
/// 124, 117, and 114 (226 s) against 50, 48, and 42 from Lindh's.
fn start_hessian(settings: &crate::opt::state_machine::OptimizerSettings) -> Option<String> {
    if settings.model_hessian.is_some() || settings.exact_hessian_interval.is_some() {
        settings.model_hessian.clone()
    } else {
        Some("lindh".to_string())
    }
}

/// The lowest multiplicity of `electrons` electrons.
fn lowest(electrons: i64) -> u32 {
    if electrons.rem_euclid(2) == 0 {
        1
    } else {
        2
    }
}

/// The fragments, the states, and whether the molecule is bridged.
fn layout_of(
    config: &EtmainConfig,
    sys: &MolecularSystem,
) -> MoptResult<(EdeelLayout, EdeelStates)> {
    let bonded = {
        let caps = crate::pes::combine::Caps::of_bonds(sys, &config.donor, LinkPlacement::Scaled);
        !caps.links.is_empty()
    };
    let layout = match &config.bridge {
        Some(b) if b.is_empty() => EdeelLayout::separate(&config.donor, &config.acceptor)?,
        Some(b) => EdeelLayout::bridged(
            sys,
            &config.donor,
            b,
            &config.acceptor,
            LinkPlacement::Scaled,
        )?,
        None if !bonded => EdeelLayout::separate(&config.donor, &config.acceptor)?,
        None => {
            let bridge: Vec<AtomId> = sys
                .atoms()
                .filter(|a| !config.donor.contains(a) && !config.acceptor.contains(a))
                .collect();
            EdeelLayout::bridged(
                sys,
                &config.donor,
                &bridge,
                &config.acceptor,
                LinkPlacement::Scaled,
            )?
        }
    };
    let electrons = |atoms: &[AtomId], caps: usize, charge: i32| -> i64 {
        atoms
            .iter()
            .map(|a| i64::from(sys.element(*a).atomic_number()))
            .sum::<i64>()
            + caps as i64
            - i64::from(charge)
    };
    let [qd, qa] = config.charges;
    let caps =
        |f: &crate::pes::combine::EdeelFragment| f.caps.as_ref().map_or(0, |c| c.links.len());
    let whole: Vec<AtomId> = sys.atoms().collect();
    let qc = qd + qa - layout.bridge_charge;
    let m = config.multiplicities.unwrap_or([
        lowest(electrons(&whole, 0, qc)),
        lowest(electrons(&layout.donor.atoms, caps(&layout.donor), qd)),
        lowest(electrons(&layout.donor.atoms, caps(&layout.donor), qd - 1)),
        lowest(electrons(
            &layout.acceptor.atoms,
            caps(&layout.acceptor),
            qa,
        )),
        lowest(electrons(
            &layout.acceptor.atoms,
            caps(&layout.acceptor),
            qa - 1,
        )),
    ]);
    let states = EdeelStates {
        complex_without: SpinCharge::new(qc, m[0]),
        donor_without: SpinCharge::new(qd, m[1]),
        donor_with: SpinCharge::new(qd - 1, m[2]),
        acceptor_without: SpinCharge::new(qa, m[3]),
        acceptor_with: SpinCharge::new(qa - 1, m[4]),
    };
    Ok((layout, states))
}

/// What a run gives back.
#[derive(Debug, Clone)]
pub struct EtmainResult {
    /// The run directory.
    pub directory: PathBuf,
    /// The result of `-et_method edeel`.
    pub transfer: Option<TransferResult>,
    /// The result of `-et_method hopping`.
    pub hopping: Option<HoppingResult>,
}

/// The directory of a run of `input`: the one to restart in, or a new one below `base` named
/// after the input, `tag`, and the time.
fn run_directory(config: &EtmainConfig, input: &Path, base: &Path, tag: &str) -> PathBuf {
    match &config.base.restart_dir {
        Some(d) => d.clone(),
        None => {
            let now = DateTime::now_utc();
            let stem = input.with_extension("");
            base.join(now.date_dir()).join(format!(
                "{}_{tag}_{}",
                stem.to_string_lossy(),
                now.timestamp()
            ))
        }
    }
}

fn ev(h: f64) -> f64 {
    h * Units::HARTREE_TO_EV
}

fn in_ev(v: Option<f64>) -> String {
    v.map_or_else(String::new, |x| format!("{:.6e}", ev(x)))
}

fn log10_of(k: Option<f64>) -> String {
    k.map_or_else(String::new, |v| format!("{:.4}", v.log10()))
}

/// Run `etmain` on its inputs.
pub fn run_with(
    config: &EtmainConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<Vec<EtmainResult>> {
    let mut out = Vec::new();
    for input in &config.base.inputs {
        out.push(run_one(config, input, base, console, registry).context(input.display())?);
    }
    Ok(out)
}

fn run_one(
    config: &EtmainConfig,
    input: &Path,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<EtmainResult> {
    if config.base.initial_hessian.is_some() {
        return Err(MoptError::config(
            "etmain: -ih reads one Hessian for one optimization, and etmain runs several (the \
             equilibria of both states and the crossings); -fc N computes exact Hessians in \
             each of them",
        ));
    }
    if config.method == EtMethod::Hopping {
        return run_hopping_one(config, input, base, console, registry);
    }
    let spin = config.base.state.of_file(input)?;
    let start = read_xyz_file(input, spin, None)?;
    let evaluations_before = registry.counts();
    let (layout, states) = layout_of(config, &start)?;
    if start.spin_charge.charge != states.complex_without.charge - 1 {
        return Err(MoptError::config(format!(
            "etmain: the structure (charge {}) is the system with the moving electron, so its \
             charge is that of the two fragments without it ({}) less one",
            start.spin_charge.charge, states.complex_without.charge
        )));
    }
    let calc = registry.create(
        &config.base.calculator,
        &config.base.calculator_options,
        &start,
    )?;
    let dir = run_directory(config, input, base, "ET");
    std::fs::create_dir_all(&dir)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} etmain: {}",
        crate::VERSION,
        input.display()
    ));
    log.info(&format!("Calculator: {}", calc.description()));
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    let caps =
        |f: &crate::pes::combine::EdeelFragment| f.caps.as_ref().map_or(0, |c| c.links.len());
    log.info(&if layout.is_bridged() {
        format!(
            "Fragments: a molecule of donor ({} atoms), bridge ({} atoms), and acceptor ({} \
             atoms); the donor-bridge fragment of {} atoms with {} hydrogen caps, the \
             bridge-acceptor fragment of {} atoms with {}",
            config.donor.len(),
            layout.bridge.len(),
            config.acceptor.len(),
            layout.donor.atoms.len(),
            caps(&layout.donor),
            layout.acceptor.atoms.len(),
            caps(&layout.acceptor)
        )
    } else {
        format!(
            "Fragments: separate molecules (donor {} atoms, acceptor {} atoms)",
            layout.donor.atoms.len(),
            layout.acceptor.atoms.len()
        )
    });
    log.info(&format!("States (charge, multiplicity): {states:?}"));
    let journaled = Narrated::new(&journaled, log.clone(), "evaluation");
    let calcs = CalculatorSet::single(&journaled);
    let final_start = match &config.final_start {
        Some(path) => Some(read_xyz_file(path, spin, None)?),
        None => None,
    };
    let multiplicity = start.spin_charge.multiplicity;
    let coupling: Box<dyn CouplingSource> = match &config.coupling {
        CouplingChoice::Given(v) => Box::new(GivenCoupling(*v)),
        CouplingChoice::Edeel => Box::new(EdeelCoupling {
            states,
            multiplicity,
        }),
        CouplingChoice::MullikenHush => Box::new(MullikenHushCoupling {
            states,
            multiplicity,
            count: config.states,
        }),
        CouplingChoice::GeneralizedMullikenHush | CouplingChoice::FragmentChargeDifference => {
            let formula = if config.coupling == CouplingChoice::GeneralizedMullikenHush {
                ExcitedStateFormula::GeneralizedMullikenHush
            } else {
                ExcitedStateFormula::FragmentChargeDifference
            };
            let index = |atoms: &[AtomId]| atoms.iter().map(|a| a.index()).collect::<Vec<_>>();
            Box::new(ExcitedStateCoupling {
                states,
                multiplicity,
                count: config.states,
                formula,
                fragments: [index(&config.donor), index(&config.acceptor)],
            })
        }
        CouplingChoice::Auto => Box::new(AutoCoupling::new(
            MullikenHushCoupling {
                states,
                multiplicity,
                count: config.states,
            },
            EdeelCoupling {
                states,
                multiplicity,
            },
        )),
    };
    log.info(&format!("Coupling: {}", coupling.name()));
    let optimizer = crate::opt::state_machine::OptimizerSettings {
        rigid_body_invariant: calc.is_rigid_body_invariant(),
        model_hessian: start_hessian(&config.base.settings),
        ..config.base.settings.clone()
    };
    let settings = TransferSettings {
        temperature: config.temperature,
        solvent: config.solvent,
        alphas: config.alphas.clone(),
        crossing_optimizer: crossing_settings(&optimizer),
        optimizer,
        thermo: config.free_energy.then_some(config.thermo),
        crossing_search: config.crossing_search,
    };
    log.info(if config.free_energy {
        "Barriers: the electronic energies with the solvent shift, and the free energies from \
         the harmonic thermal corrections at the initial equilibrium and within the seam at \
         each crossing"
    } else {
        "Barriers: the electronic energies with the solvent shift"
    });
    // Every iteration of every optimization (the equilibria, then the crossings in the order
    // of the scan) to the log and to one trajectory.
    let mut progress = ProgressLog::new(log.clone(), settings.optimizer.criteria);
    let mut files = RunFiles::new(&dir, "et")?;
    let mut stop = StopFile::new(&dir, log.clone());
    let mut observers: [&mut dyn Observer; 3] = [&mut progress, &mut files, &mut stop];
    let transfer = run_transfer(
        &layout,
        &states,
        coupling.as_ref(),
        &start,
        final_start.as_ref(),
        &settings,
        &calcs,
        &mut observers,
    )?;
    log.info(&format!("Coupling used: {}", coupling.name()));
    write_outputs(&dir, &transfer, config.temperature, &log)?;
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(EtmainResult {
        directory: dir,
        transfer: Some(transfer),
        hopping: None,
    })
}

/// `etmain -et_method hopping` on one input: the pair in the charges of its molecules without
/// the carrier.
fn run_hopping_one(
    config: &EtmainConfig,
    input: &Path,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<EtmainResult> {
    let spin = config.base.state.of_file(input)?;
    let pair = read_xyz_file(input, spin, None)?;
    let evaluations_before = registry.counts();
    let calc = registry.create(
        &config.base.calculator,
        &config.base.calculator_options,
        &pair,
    )?;
    let dir = run_directory(config, input, base, "HOP");
    std::fs::create_dir_all(&dir)?;
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} etmain -et_method hopping: {}",
        crate::VERSION,
        input.display()
    ));
    log.info(&format!("Calculator: {}", calc.description()));
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&format!(
        "Molecules: the first of {} atoms (the {} starts on it), the second of {} atoms; \
         their charges without the carrier {} and {}",
        config.donor.len(),
        config.carrier.name(),
        config.acceptor.len(),
        config.charges[0],
        config.charges[1]
    ));
    let journaled = Narrated::new(&journaled, log.clone(), "evaluation");
    let calcs = CalculatorSet::single(&journaled);
    let settings = HoppingSettings {
        temperature: config.temperature,
        carrier: config.carrier,
        charges: config.charges,
        multiplicities: config.hopping_multiplicities,
        optimizer: crate::opt::state_machine::OptimizerSettings {
            rigid_body_invariant: calc.is_rigid_body_invariant(),
            model_hessian: start_hessian(&config.base.settings),
            ..config.base.settings.clone()
        },
    };
    // Every iteration of the four optimizations to the log and to one trajectory.
    let mut progress = ProgressLog::new(log.clone(), settings.optimizer.criteria);
    let mut files = RunFiles::new(&dir, "hop")?;
    let mut stop = StopFile::new(&dir, log.clone());
    let mut observers: [&mut dyn Observer; 3] = [&mut progress, &mut files, &mut stop];
    let hop = run_hopping(
        &pair,
        &config.donor,
        &config.acceptor,
        &settings,
        &calcs,
        &mut observers,
    )?;
    write_hopping(&dir, &hop, config, &log)?;
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(EtmainResult {
        directory: dir,
        transfer: None,
        hopping: Some(hop),
    })
}

/// The log lines and files of a hop.
fn write_hopping(
    dir: &Path,
    h: &HoppingResult,
    config: &EtmainConfig,
    log: &Logger,
) -> MoptResult<()> {
    let mev = |x: f64| x * Units::HARTREE_TO_EV * 1000.0;
    for (m, name) in h.molecules.iter().zip(["a", "b"]) {
        for (k, which) in ["without", "with"].iter().enumerate() {
            std::fs::write(
                dir.join(format!("molecule_{name}_{which}.xyz")),
                format_xyz_system(&m.equilibria[k], None),
            )?;
            if !m.converged[k] {
                log.warn(&format!(
                    "The equilibrium of molecule {name} {which} the carrier did not converge; \
                     its energies enter the reorganization energy as they are"
                ));
            }
        }
    }
    let t = &h.coupling;
    let degenerate = if t.degeneracy == [1, 1] {
        String::new()
    } else {
        format!(
            "; frontier orbitals degenerate {}-fold and {}-fold, their best-matched \
             combinations",
            t.degeneracy[0], t.degeneracy[1]
        )
    };
    log.info(&format!(
        "Transfer integral (projection method): J_eff = {:.3} meV (J = {:.3} meV, S_ab = \
         {:.5}, site energies {:.4} and {:.4} eV{degenerate})",
        mev(t.effective),
        mev(t.coupling),
        t.overlap,
        ev(t.site_energies[0]),
        ev(t.site_energies[1])
    ));
    log.info(&format!(
        "Reorganization energy (four-point): {:.4} eV from the first molecule to the second, \
         {:.4} eV back; the rates take the mean, {:.4} eV",
        ev(h.reorganization[0]),
        ev(h.reorganization[1]),
        ev(h.mean_reorganization())
    ));
    log.info(&format!(
        "Reaction energy of the hop from the first molecule to the second: {:.4} eV \
         (electronic, each molecule alone)",
        ev(h.reaction_energy)
    ));
    log.info(&format!(
        "Marcus rates at {} K: {:.4e} 1/s to the second molecule (log10 {:.3}), {:.4e} 1/s \
         back (log10 {:.3}); apparent barriers {:.4} and {:.4} eV",
        config.temperature,
        h.rates[0],
        h.rates[0].log10(),
        h.rates[1],
        h.rates[1].log10(),
        ev(h.apparent_barriers[0]),
        ev(h.apparent_barriers[1])
    ));
    let pair = |x: [f64; 2], f: &dyn Fn(f64) -> f64| format!("[{}, {}]", f(x[0]), f(x[1]));
    let converged = |m: &crate::workflow::hopping::HoppingMolecule| {
        format!("[{}, {}]", m.converged[0], m.converged[1])
    };
    let json = format!(
        "{{\n  \"carrier\": \"{}\",\n  \"temperature_K\": {},\n  \"J_eff_meV\": {},\n  \
         \"J_meV\": {},\n  \"overlap\": {},\n  \"site_energies_eV\": {},\n  \
         \"effective_site_energies_eV\": {},\n  \"degeneracy\": [{}, {}],\n  \
         \"reorganization_eV\": {},\n  \"mean_reorganization_eV\": {},\n  \
         \"reaction_energy_eV\": {},\n  \"rates_per_s\": [{:e}, {:e}],\n  \
         \"log10_rates\": [{}, {}],\n  \"apparent_barriers_eV\": {},\n  \
         \"apparent_barriers_kcal_mol\": {},\n  \"converged\": [{}, {}]\n}}\n",
        config.carrier.name(),
        config.temperature,
        mev(t.effective),
        mev(t.coupling),
        t.overlap,
        pair(t.site_energies, &ev),
        pair(t.effective_site_energies, &ev),
        t.degeneracy[0],
        t.degeneracy[1],
        pair(h.reorganization, &ev),
        ev(h.mean_reorganization()),
        ev(h.reaction_energy),
        h.rates[0],
        h.rates[1],
        h.rates[0].log10(),
        h.rates[1].log10(),
        pair(h.apparent_barriers, &ev),
        pair(h.apparent_barriers, &kcal),
        converged(&h.molecules[0]),
        converged(&h.molecules[1])
    );
    std::fs::write(dir.join("hop_summary.json"), json)?;
    Ok(())
}

fn kcal(h: f64) -> f64 {
    h / Units::KCAL_MOL_TO_HARTREE
}

/// The largest rate along the scan as JSON, with the free energy of activation it implies.
fn best_json(best: Option<(f64, f64)>, temperature: f64) -> String {
    best.map_or_else(
        || "null".to_string(),
        |(a, k)| {
            let g = apparent_barrier(10f64.powf(k), temperature);
            format!(
                "{{\"alpha_SA\": {a}, \"log10_k\": {k}, \"apparent_barrier_eV\": {}, \
                 \"apparent_barrier_kcal_mol\": {}}}",
                ev(g),
                kcal(g)
            )
        },
    )
}

fn write_outputs(dir: &Path, r: &TransferResult, temperature: f64, log: &Logger) -> MoptResult<()> {
    std::fs::write(
        dir.join("eq_initial.xyz"),
        format_xyz_system(&r.initial.0, None),
    )?;
    std::fs::write(
        dir.join("eq_final.xyz"),
        format_xyz_system(&r.final_state.0, None),
    )?;
    let mut traj = String::new();
    let mut csv = String::from(
        "alpha_SA,converged,V11_eV,V22_eV,shift_initial_eV,shift_final_eV,barrier_eV,\
         free_barrier_eV,seam_imaginary,inner_barrier_eV,coupling_eV,coupling_from_others,\
         reorganization_eV,nu_in_cm1,dE_in0_eV,lambda_in_eV,log10_Gamma,log10_k_marcus,\
         log10_k_marcus_G,apparent_barrier_marcus_eV,zn_type,log10_k_zn,log10_k_zn_G,\
         apparent_barrier_zn_eV,log10_k_mlj,solvent_eV,donor_eV,acceptor_eV\n",
    );
    let apparent = |k: Option<f64>| {
        k.map_or_else(String::new, |v| {
            format!("{:.6}", ev(apparent_barrier(v, temperature)))
        })
    };
    let at = |c: &CrossingResult| {
        c.alpha.map_or_else(
            || "without solvent".to_string(),
            |a| format!("alpha_SA = {a:.2}"),
        )
    };
    for c in &r.crossings {
        traj.push_str(&format_xyz_system(&c.system, None));
        let zn = c.zhu_nakamura.map(|z| z.rate);
        let zn_free = c.zhu_nakamura_free.map(|z| z.rate);
        let fields = [
            c.alpha.map_or_else(String::new, |a| format!("{a:.3}")),
            c.converged.to_string(),
            format!("{:.6}", ev(c.v11 - r.initial.1)),
            format!("{:.6}", ev(c.v22 - r.initial.1)),
            format!("{:.6}", ev(c.shifts[0])),
            format!("{:.6}", ev(c.shifts[1])),
            format!("{:.6}", ev(c.barrier)),
            c.free_energy_barrier
                .map_or_else(String::new, |g| format!("{:.6}", ev(g))),
            c.seam_imaginary.to_string(),
            format!("{:.6}", ev(c.inner_barrier)),
            in_ev(c.coupling),
            c.coupling_from_others.to_string(),
            in_ev(c.reorganization),
            c.inner
                .map_or_else(String::new, |p| format!("{:.1}", p.wavenumber())),
            c.inner
                .map_or_else(String::new, |p| format!("{:.6}", ev(p.reaction_energy))),
            c.inner
                .map_or_else(String::new, |p| format!("{:.6}", ev(p.reorganization))),
            c.tunnelling
                .map_or_else(String::new, |t| format!("{:.4}", t.log10())),
            log10_of(c.marcus),
            log10_of(c.marcus_free),
            apparent(c.marcus_free.or(c.marcus)),
            c.zhu_nakamura
                .map_or_else(String::new, |z| format!("{:?}", z.kind)),
            log10_of(zn),
            log10_of(zn_free),
            apparent(zn_free.or(zn)),
            log10_of(c.marcus_levich_jortner),
            format!("{:.6}", ev(c.decomposition.solvent)),
            format!("{:.6}", ev(c.decomposition.donor)),
            format!("{:.6}", ev(c.decomposition.acceptor)),
        ];
        csv.push_str(&fields.join(","));
        csv.push('\n');
        if let Some(note) = &c.coupling_note {
            log.info(&format!(
                "Crossing at {}: no coupling of its own ({note})",
                at(c)
            ));
        }
        if c.seam_imaginary > 0 {
            log.warn(&format!(
                "Crossing at {}: {} imaginary frequencies within the seam (not a minimum of \
                 it; left out of its free energy)",
                at(c),
                c.seam_imaginary
            ));
        }
        let rate =
            |k: Option<f64>| k.map_or_else(|| "-".to_string(), |v| format!("{:.2}", v.log10()));
        log.info(&format!(
            "Crossing at {}: dE = {:.4} eV ({:.2} kcal/mol){}; log10 k: Marcus {} (dG: {}), \
             Zhu-Nakamura {} (dG: {})",
            at(c),
            ev(c.barrier),
            kcal(c.barrier),
            c.free_energy_barrier.map_or_else(String::new, |g| format!(
                ", dG = {:.4} eV ({:.2} kcal/mol)",
                ev(g),
                kcal(g)
            )),
            rate(c.marcus),
            rate(c.marcus_free),
            rate(zn),
            rate(zn_free),
        ));
    }
    std::fs::write(dir.join("crossings.xyz"), traj)?;
    std::fs::write(dir.join("et_scan.csv"), csv)?;
    let number = |v: Option<f64>| v.map_or_else(|| "null".to_string(), |x| format!("{x}"));
    std::fs::write(
        dir.join("et_summary.json"),
        format!(
            "{{\n  \"crossings\": {},\n  \"temperature_K\": {temperature},\n  \
             \"initial_energy_hartree\": {},\n  \"initial_free_energy_hartree\": {},\n  \
             \"marcus_with_tunnelling\": {},\n  \"zhu_nakamura\": {},\n  \
             \"marcus_with_tunnelling_free_energy\": {},\n  \"zhu_nakamura_free_energy\": \
             {},\n  \"marcus_levich_jortner\": {}\n}}\n",
            r.crossings.len(),
            r.initial.1,
            number(r.initial_free_energy),
            best_json(r.best_marcus, temperature),
            best_json(r.best_zhu_nakamura, temperature),
            best_json(r.best_marcus_free, temperature),
            best_json(r.best_zhu_nakamura_free, temperature),
            best_json(r.best_marcus_levich_jortner, temperature),
        ),
    )?;
    let report = |name: &str, best: Option<(f64, f64)>| {
        if let Some((a, k)) = best {
            let g = apparent_barrier(10f64.powf(k), temperature);
            log.info(&format!(
                "Largest {name}: log10 k = {k:.2} at alpha_SA = {a:.2} (apparent free energy of \
                 activation {:.4} eV, {:.2} kcal/mol)",
                ev(g),
                kcal(g)
            ));
        }
    };
    report("Marcus rate with Scher-Holstein tunnelling", r.best_marcus);
    report("Zhu-Nakamura rate", r.best_zhu_nakamura);
    report(
        "Marcus rate with the free energies of activation",
        r.best_marcus_free,
    );
    report(
        "Zhu-Nakamura rate with the free energies of activation",
        r.best_zhu_nakamura_free,
    );
    report("Marcus-Levich-Jortner rate", r.best_marcus_levich_jortner);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn config(extra: &[&str]) -> EtmainConfig {
        let mut args = vec!["a.xyz", "-et_donor", "1-2", "-et_acceptor", "4-5"];
        args.extend_from_slice(extra);
        config_from_cli(&args).unwrap()
    }

    #[test]
    fn the_optimizations_start_from_lindh_unless_told_otherwise() {
        // The default: Lindh's model Hessian.
        let plain = config(&[]);
        assert_eq!(
            start_hessian(&plain.base.settings).as_deref(),
            Some("lindh")
        );
        // Exact Hessians stay exact: no model Hessian is put in their place.
        let exact = config(&["-fc", "5"]);
        assert_eq!(exact.base.settings.exact_hessian_interval, Some(5));
        assert_eq!(start_hessian(&exact.base.settings), None);
        // Another model Hessian by name.
        let swart = config(&["-modelhess", "swart"]);
        assert_eq!(
            start_hessian(&swart.base.settings).as_deref(),
            Some("swart")
        );
    }

    #[test]
    fn the_crossing_searches_have_the_largest_trust_radius_of_a_gradient_projection() {
        let cap = super::super::optmain::PROJECTION_MAX_TRUST_ANGSTROM * Units::ANGSTROM_TO_BOHR;
        let largest = |args: &[&str]| {
            let c = config(args);
            (
                c.base.settings.trust.max.map(|q| q.value()),
                crossing_settings(&c.base.settings)
                    .trust
                    .max
                    .map(|q| q.value()),
            )
        };
        // The default method (RS-I-RFO): the equilibria keep the default, the crossings get
        // the cap.
        assert_eq!(largest(&[]), (None, Some(cap)));
        // A largest trust radius given by -tr holds for both.
        assert_eq!(largest(&["-tr", "0.3"]), (Some(0.3), Some(0.3)));
        // Methods other than RS-I-RFO keep their defaults.
        assert_eq!(largest(&["-opt", "fire"]), (None, None));
    }
}
