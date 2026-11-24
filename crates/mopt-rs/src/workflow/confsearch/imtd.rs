//! The iMTD-GC conformer search, implemented from its papers (Grimme, J. Chem. Theory
//! Comput. 2019, 15, 2847; Pracht, Bohle, Grimme, Phys. Chem. Chem. Phys. 2020, 22, 7169;
//! Pracht et al., J. Chem. Phys. 2024, 160, 114110) and the documentation of CREST; no code
//! of CREST is used.
//!
//! 1. **RMSD metadynamics.** Several dynamics from the lowest conformer, each with the bias
//!    `sum_i k exp(-alpha Delta_i^2)` of the RMSD to reference structures
//!    ([`crate::pes::bias::RmsdMetadynamics`]) with its own `k` (per atom of the bias) and
//!    `alpha`: the known conformers as fixed references, and a structure of the run itself
//!    deposited at every interval. Frames are taken at a fixed interval.
//! 2. **Multi-level optimization.** Every frame is optimized with loose criteria; those within
//!    the loose energy window of the lowest are freed of duplicates and optimized with the
//!    criteria of the command, then judged by the ensemble.
//! 3. **Iteration.** A round that finds a conformer below the lowest known before starts the
//!    next round from it; a round without one ends the iterations
//!    ([`crate::workflow::explore::NoLowerMinimum`]), as do `max_rounds` rounds.
//! 4. **Finishing.** Short dynamics without a bias from the lowest conformers at raised
//!    temperatures pick up conformers near the low ones, and the genetic crossing
//!    ([`crate::workflow::explore::crossover`]) combines the conformers within a window. When
//!    either finds a lower conformer, the rounds start again from it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use rayon::prelude::*;

use super::search::Search;
use super::{
    CommonSettings, DynamicsJob, Level, RelaxJob, SearchJobs, SearchObserver, SearchResult,
};
use crate::core_infra::rng::Rng;
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::pes::bias::RmsdMetadynamics;
use crate::pes::constraint::{Constraint, ConstraintKind};
use crate::pes::term::PesTerm;
use crate::workflow::explore::{
    crossover_candidates, CrossoverSettings, MaxTrials, NoLowerMinimum, StopRules, TrialSummary,
};

/// One metadynamics of a round.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtdRun {
    /// Height `k` per atom of the bias (Hartree).
    pub k_per_atom: f64,
    /// Exponent `alpha` (Bohr^-2).
    pub alpha: f64,
}

/// The 14 runs of the example output of CREST's documentation (`k / N` = 3.0, 1.5, 0.75 mEh
/// with `alpha` = 1.300, 0.780, 0.468, 0.281 Bohr^-2, and two more).
pub fn crest_runs() -> Vec<MtdRun> {
    let mut out = Vec::new();
    for alpha in [1.300, 0.780, 0.468, 0.281] {
        for k in [3.0e-3, 1.5e-3, 0.75e-3] {
            out.push(MtdRun {
                k_per_atom: k,
                alpha,
            });
        }
    }
    out.push(MtdRun {
        k_per_atom: 1.0e-3,
        alpha: 0.100,
    });
    out.push(MtdRun {
        k_per_atom: 5.0e-3,
        alpha: 0.800,
    });
    out
}

/// Which bonds SHAKE holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shake {
    /// None.
    Off,
    /// Bonds to hydrogen atoms.
    Hydrogens,
    /// Every bond.
    All,
}

/// The short dynamics from the lowest conformers.
#[derive(Debug, Clone, PartialEq)]
pub struct FinishingDynamics {
    /// From this many of the lowest conformers.
    pub conformers: usize,
    /// At these temperatures (K), one run each.
    pub temperatures: Vec<f64>,
    /// Length of each run (atomic units).
    pub time: f64,
}

/// Settings of the search.
#[derive(Debug, Clone, PartialEq)]
pub struct ImtdSettings {
    /// The metadynamics of a round.
    pub runs: Vec<MtdRun>,
    /// Length of each (atomic units).
    pub mtd_time: f64,
    /// Time step (atomic units).
    pub time_step: f64,
    /// Temperature (K).
    pub temperature: f64,
    /// A frame every this much time (atomic units).
    pub dump_every: f64,
    /// A structure of the run is deposited as a reference every this much time (atomic units).
    pub deposit_every: f64,
    /// Largest number of deposited references.
    pub max_deposits: usize,
    /// At most this many known conformers (the lowest) are fixed references of the bias.
    pub known_references: usize,
    /// Bonds held by SHAKE.
    pub shake: Shake,
    /// Mass of hydrogen atoms in the dynamics (amu; `None`: their mass).
    pub hydrogen_mass: Option<f64>,
    /// Largest number of rounds.
    pub max_rounds: usize,
    /// The genetic crossing, and the window of its parents (Hartree above the lowest).
    pub crossover: Option<(CrossoverSettings, f64)>,
    /// The short dynamics at the end.
    pub finishing: Option<FinishingDynamics>,
    /// Atoms of the RMSD of the bias (from 0; empty: all).
    pub bias_atoms: Vec<usize>,
}

impl Default for ImtdSettings {
    fn default() -> Self {
        let fs = Units::FS_TO_AU_TIME;
        Self {
            runs: crest_runs(),
            mtd_time: 5000.0 * fs,
            time_step: 2.0 * fs,
            temperature: 300.0,
            dump_every: 100.0 * fs,
            deposit_every: 1000.0 * fs,
            max_deposits: 1000,
            known_references: 20,
            shake: Shake::Hydrogens,
            hydrogen_mass: Some(4.0),
            max_rounds: 5,
            crossover: Some((
                CrossoverSettings::default(),
                6.0 * Units::KCAL_MOL_TO_HARTREE,
            )),
            finishing: Some(FinishingDynamics {
                conformers: 4,
                temperatures: vec![400.0, 500.0],
                time: 3000.0 * fs,
            }),
            bias_atoms: Vec::new(),
        }
    }
}

/// The bonds of `sys` that `shake` holds, as constraints at their starting lengths.
pub fn shake_constraints(
    sys: &MolecularSystem,
    graph: &crate::coord::graph::MolecularGraph,
    shake: Shake,
) -> Vec<Constraint> {
    if shake == Shake::Off {
        return Vec::new();
    }
    graph
        .bonds()
        .into_iter()
        .filter(|[a, b]| {
            shake == Shake::All
                || sys.elements[*a].atomic_number() == 1
                || sys.elements[*b].atomic_number() == 1
        })
        .map(|[a, b]| {
            Constraint::at_start(ConstraintKind::Distance(
                vec![AtomId::new(a)],
                vec![AtomId::new(b)],
            ))
        })
        .collect()
}

fn steps(time: f64, dt: f64) -> MoptResult<usize> {
    let n = (time / dt).round();
    if !(n.is_finite() && n >= 1.0) {
        return Err(MoptError::config(format!(
            "a dynamics of {:.1} fs with steps of {:.2} fs has no steps",
            time * Units::AU_TIME_TO_FS,
            dt * Units::AU_TIME_TO_FS
        )));
    }
    Ok(n as usize)
}

/// The frames of `jobs` (in parallel), after their first, with their origins.
fn sample(
    search: &mut Search<'_>,
    starts: Vec<(MolecularSystem, DynamicsJob)>,
) -> MoptResult<Vec<(MolecularSystem, String)>> {
    let jobs = search.jobs;
    let runs: Vec<_> = starts
        .par_iter()
        .map(|(s, job)| (job.label.clone(), jobs.dynamics(s, job)))
        .collect();
    let mut frames = Vec::new();
    for (label, run) in runs {
        match run {
            Ok(r) => {
                if r.interrupted {
                    search.interrupted = true;
                }
                let n = r.frames.len().saturating_sub(1);
                search.note(format!("{label}: {n} snapshots"))?;
                frames.extend(
                    r.frames
                        .into_iter()
                        .enumerate()
                        .skip(1)
                        .map(|(k, f)| (f, format!("{label}, snapshot {k}"))),
                );
            }
            Err(e) => search.note(format!("{label}: the dynamics failed: {e}"))?,
        }
    }
    Ok(frames)
}

/// Run the search from `input`.
pub fn run(
    input: &MolecularSystem,
    settings: &ImtdSettings,
    common: &CommonSettings,
    jobs: &dyn SearchJobs,
    observer: &mut dyn SearchObserver,
) -> MoptResult<Option<SearchResult>> {
    if settings.runs.is_empty() {
        return Err(MoptError::config("the metadynamics need at least one run"));
    }
    let dt = settings.time_step;
    let mtd_steps = steps(settings.mtd_time, dt)?;
    let frame_every = steps(settings.dump_every, dt)?;
    let first = RelaxJob::minimize("input", Level::Final);
    let Some(mut search) = Search::start(jobs, observer, common, input, &first)? else {
        return Ok(None);
    };
    let mut rng = Rng::from_seed(common.seed);
    let n_bias = if settings.bias_atoms.is_empty() {
        input.n_atoms()
    } else {
        settings.bias_atoms.len()
    };
    let bias_atoms: Vec<AtomId> = settings
        .bias_atoms
        .iter()
        .map(|&a| AtomId::new(a))
        .collect();
    let plain = RelaxJob::minimize("candidate", Level::Final);
    let mut rules = StopRules::new()
        .with(Box::new(NoLowerMinimum))
        .with(Box::new(MaxTrials {
            max: settings.max_rounds,
        }));
    let mut round = 0usize;
    let reason;
    'search: loop {
        // Rounds of metadynamics until one finds no lower conformer.
        loop {
            let Some(lowest) = search.ensemble.lowest() else {
                return Err(MoptError::config("the ensemble lost its conformers"));
            };
            let start = search.ensemble.members()[lowest].system.clone();
            let known: Vec<MolecularSystem> = search
                .ensemble
                .by_energy()
                .into_iter()
                .take(settings.known_references)
                .map(|i| search.ensemble.members()[i].system.clone())
                .collect();
            search.note(format!(
                "Metadynamics round {}: {} runs of {:.1} ps from EQ{lowest} ({} known conformers \
                 as references)",
                round + 1,
                settings.runs.len(),
                settings.mtd_time * Units::AU_TIME_TO_FS / 1000.0,
                known.len()
            ))?;
            let constraints = shake_constraints(&start, &search.reference, settings.shake);
            let mut dynamics = Vec::new();
            for (j, run) in settings.runs.iter().enumerate() {
                let bias = RmsdMetadynamics::new(run.k_per_atom * n_bias as f64, run.alpha)?
                    .on_atoms(&bias_atoms)
                    .depositing(settings.deposit_every, settings.max_deposits)?;
                for k in &known {
                    bias.add_reference(k)?;
                }
                let bias: Arc<dyn PesTerm> = Arc::new(bias);
                dynamics.push((
                    start.clone(),
                    DynamicsJob {
                        label: format!(
                            "MTD {} of round {} (k = {:.4} Eh, alpha = {:.3} bohr^-2)",
                            j + 1,
                            round + 1,
                            run.k_per_atom * n_bias as f64,
                            run.alpha
                        ),
                        temperature: settings.temperature,
                        time_step: dt,
                        steps: mtd_steps,
                        frame_every,
                        seed: common
                            .seed
                            .wrapping_add(1000 * round as u64)
                            .wrapping_add(j as u64 + 1),
                        terms: vec![("rmsd_metadynamics".to_string(), bias)],
                        constraints: constraints.clone(),
                        hydrogen_mass: settings.hydrogen_mass,
                        freeze: None,
                    },
                ));
            }
            let frames = sample(&mut search, dynamics)?;
            let low_before = search.lowest_energy();
            let refined = search.refine(&frames, &plain)?;
            search.note(format!(
                "Round {}: {} new conformers ({} known); lowest {:.10}",
                round + 1,
                refined.new,
                search.ensemble.len(),
                search.lowest_energy()
            ))?;
            if search.interrupted {
                reason = "The stop signal is detected.".to_string();
                break 'search;
            }
            let energies: Vec<f64> = search.ensemble.members().iter().map(|m| m.energy).collect();
            let summary = TrialSummary {
                trial: round,
                energies: &energies,
                found_new: refined.new > 0,
                found_lower: search.lowest_energy() < low_before - common.compare.energy,
            };
            round += 1;
            // No lower conformer, or the last round: on to the finishing steps.
            if let Some(why) = rules.after_trial(&summary) {
                search.note(why)?;
                break;
            }
        }
        // Finishing: dynamics from the lowest conformers and the genetic crossing.
        let low_before = search.lowest_energy();
        if let Some(f) = &settings.finishing {
            let mut dynamics = Vec::new();
            let lows = search.ensemble.by_energy();
            for (c, &i) in lows.iter().take(f.conformers).enumerate() {
                let s = search.ensemble.members()[i].system.clone();
                let constraints = shake_constraints(&s, &search.reference, settings.shake);
                for (t, &temperature) in f.temperatures.iter().enumerate() {
                    dynamics.push((
                        s.clone(),
                        DynamicsJob {
                            label: format!("MD from EQ{i} at {temperature:.0} K"),
                            temperature,
                            time_step: dt,
                            steps: steps(f.time, dt)?,
                            frame_every,
                            seed: common
                                .seed
                                .wrapping_add(500_000 + 1000 * round as u64)
                                .wrapping_add((10 * c + t) as u64),
                            terms: Vec::new(),
                            constraints: constraints.clone(),
                            hydrogen_mass: settings.hydrogen_mass,
                            freeze: None,
                        },
                    ));
                }
            }
            search.note(format!(
                "Dynamics from the {} lowest conformers at {:?} K",
                f.conformers.min(lows.len()),
                f.temperatures
            ))?;
            let frames = sample(&mut search, dynamics)?;
            let refined = search.refine(&frames, &plain)?;
            search.note(format!("Dynamics: {} new conformers", refined.new))?;
        }
        if let Some((cross, window)) = &settings.crossover {
            let parents: Vec<MolecularSystem> = search
                .ensemble
                .within(*window)
                .into_iter()
                .map(|i| search.ensemble.members()[i].system.clone())
                .collect();
            if parents.len() >= 2 {
                let Some(lowest) = search.ensemble.lowest() else {
                    return Err(MoptError::config("the ensemble lost its conformers"));
                };
                let reference = search.ensemble.members()[lowest].system.clone();
                let others: Vec<MolecularSystem> = parents.into_iter().skip(1).collect();
                let crossed = crossover_candidates(&reference, &others, cross, &mut rng)?;
                search.note(format!(
                    "Genetic crossing of {} conformers: {} new structures",
                    others.len() + 1,
                    crossed.len()
                ))?;
                let starts: Vec<(MolecularSystem, String)> = crossed
                    .into_iter()
                    .enumerate()
                    .map(|(k, s)| (s, format!("crossing {k}")))
                    .collect();
                let refined = search.refine(&starts, &plain)?;
                search.note(format!("Crossing: {} new conformers", refined.new))?;
            }
        }
        if search.interrupted {
            reason = "The stop signal is detected.".to_string();
            break;
        }
        let lower = search.lowest_energy() < low_before - common.compare.energy;
        if !lower {
            reason = "No conformer below the lowest one was found.".to_string();
            break;
        }
        if round >= settings.max_rounds {
            reason = format!("{} rounds of metadynamics were run.", settings.max_rounds);
            break;
        }
        search.note("A lower conformer was found: the metadynamics start again from it.")?;
    }
    Ok(Some(search.finish(reason)))
}
