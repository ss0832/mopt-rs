//! The state every strategy of the conformation search shares: the optimized input (EQ0) and
//! its bond graph, the ensemble, the minima rejected for their bonds, and the pipeline that
//! takes candidate structures through loose optimizations, the energy window, the final
//! optimizations, and the judgement of the ensemble.
//!
//! MultiOptPy: the first optimization and the checks after each trial in
//!  `Entrypoints/conformation_search.py:run_confsearch` (`check_bond_connectivity_preserved`,
//!  `is_identical`); the loose stage and the window follow the multi-level optimization of
//!  iMTD-GC (Pracht, Bohle, Grimme, Phys. Chem. Chem. Phys. 2020, 22, 7169).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use rayon::prelude::*;

use super::{
    CandidateFate, CommonSettings, GraphCheck, Level, Mode, Rejected, RelaxJob, Relaxed,
    SearchEvent, SearchJobs, SearchObserver, SearchResult, Tally,
};
use crate::coord::graph::{whole_molecules, MolecularGraph};
use crate::coord::invariant_rmsd::direct_rmsd;
use crate::core_infra::rng::Rng;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::{
    ConformerEnsemble, EqRecord, MaxTrials, RankUnchanged, SamplingPriority, StopRules,
    TrialSummary, Verdict,
};

/// Largest displacement of an atom (Bohr) along the imaginary mode of a saddle point before
/// it is optimized again.
pub const SADDLE_STEP: f64 = 0.2;

/// When the Hessian of `sys` has an imaginary frequency, the minimum reached by optimizing
/// from `sys` displaced along that mode (either way), if it is lower; `None` for a minimum.
fn leave_saddle(
    jobs: &dyn SearchJobs,
    sys: &MolecularSystem,
    energy: f64,
) -> MoptResult<Option<Relaxed>> {
    let modes = jobs.modes(sys)?;
    // The one criterion of an imaginary mode (`opt::mode_kill::IMAGINARY_CURVATURE`).
    if modes.imaginary_modes() == 0 {
        return Ok(None);
    }
    let Some(d) = modes.displacements.first() else {
        return Ok(None);
    };
    let largest = (0..d.nrows() / 3)
        .map(|a| (0..3).map(|k| d[3 * a + k].powi(2)).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    if largest <= 0.0 {
        return Ok(None);
    }
    let x = sys.coords_flat_bohr();
    for sign in [1.0, -1.0] {
        let moved = faer::Col::from_fn(x.nrows(), |i| x[i] + sign * SADDLE_STEP / largest * d[i]);
        let start = sys.with_flat_coords_bohr(&moved)?;
        let r = jobs.relax(
            &start,
            &RelaxJob::minimize("from a saddle point", Level::Final),
        )?;
        if r.interrupted {
            return Ok(None);
        }
        if r.converged && r.energy < energy - 1e-6 {
            return Ok(Some(r));
        }
    }
    Ok(None)
}

/// What a batch of candidates gave.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Refined {
    /// New conformers.
    pub new: usize,
    /// Whether one is lower than the lowest known before the batch.
    pub lower: bool,
}

/// The bookkeeping of a search by trials (MultiOptPy's sampling and the searches by moves):
/// the termination by the ranking of the lowest conformers and by the number of trials, and
/// the choice of the conformer the next trial starts from.
pub(crate) struct Trials<'p> {
    rules: StopRules,
    ends: StopRules,
    rank: usize,
    no_stochastic: bool,
    priority: &'p mut dyn SamplingPriority,
    rng: Rng,
    /// The conformer the next trial starts from.
    pub current: usize,
}

impl<'p> Trials<'p> {
    /// Trials that end when the `rank` lowest conformers did not change for `patience`
    /// trials or after `max_samples`; the next start is chosen by `priority` with `rng`
    /// (always the first conformer with `no_stochastic`).
    pub fn new(
        rank: usize,
        patience: usize,
        max_samples: usize,
        no_stochastic: bool,
        priority: &'p mut dyn SamplingPriority,
        rng: Rng,
    ) -> Self {
        Self {
            // As in MultiOptPy, the termination by the ranking is tested before the switch
            // to the start of the next sample, the number of samples after it
            // (`switch_conformer.log` has a line for the last sample too).
            rules: StopRules::new().with(Box::new(RankUnchanged::new(rank, patience))),
            ends: StopRules::new().with(Box::new(MaxTrials { max: max_samples })),
            rank,
            no_stochastic,
            priority,
            rng,
            current: 0,
        }
    }

    /// The random numbers of the trials.
    pub fn rng(&mut self) -> &mut Rng {
        &mut self.rng
    }

    /// After trial `trial`, whose candidate became `fate` (`low_before`: the lowest energy
    /// before it): the reason to end the search, or the choice of the next start.
    pub fn after(
        &mut self,
        search: &mut Search<'_>,
        trial: usize,
        fate: &CandidateFate,
        low_before: f64,
    ) -> MoptResult<Option<String>> {
        let energies: Vec<f64> = search.ensemble.members().iter().map(|m| m.energy).collect();
        let summary = TrialSummary {
            trial,
            energies: &energies,
            found_new: matches!(fate, CandidateFate::New { .. }),
            found_lower: search.lowest_energy() < low_before,
        };
        let stop = self.rules.after_trial(&summary);
        if let Some(count) = self
            .rules
            .counter("rank_unchanged")
            .filter(|_| energies.len() > self.rank)
        {
            search
                .observer
                .on_event(&SearchEvent::Counter { trial, count })?;
        }
        if stop.is_some() {
            return Ok(stop);
        }
        // The start of the next trial.
        if search.ensemble.len() > 1 {
            let next = if self.no_stochastic {
                Some(0)
            } else {
                let pool: Vec<EqRecord> = search
                    .ensemble
                    .members()
                    .iter()
                    .map(|m| EqRecord {
                        energy: crate::domain::Quantity::new(m.energy),
                        visits: m.visits,
                    })
                    .collect();
                self.priority.on_trial(trial);
                self.priority.select(&pool, &mut self.rng)
            };
            let Some(next) = next else {
                return Ok(Some(format!(
                    "The priority {} has nothing more to sample.",
                    self.priority.name()
                )));
            };
            self.current = next;
            if let Some(m) = search.ensemble.member_mut(next) {
                m.visits += 1;
            }
            let visits = search.ensemble.members()[next].visits;
            search.observer.on_event(&SearchEvent::Switch {
                trial,
                to: next,
                visits,
                probabilities: None,
            })?;
        } else {
            self.current = 0;
        }
        Ok(self.ends.after_trial(&summary))
    }
}

pub(crate) struct Search<'a> {
    pub jobs: &'a dyn SearchJobs,
    pub observer: &'a mut dyn SearchObserver,
    pub common: &'a CommonSettings,
    pub ensemble: ConformerEnsemble,
    pub reference: MolecularGraph,
    pub rejected: Vec<Rejected>,
    pub tally: Tally,
    pub interrupted: bool,
    /// The energy of every member without the bias potentials, in the order of the members.
    pub bare: Vec<f64>,
}

impl<'a> Search<'a> {
    /// Optimize `input` to the first conformer (EQ0); `None` when a `STOP` file ended it.
    pub fn start(
        jobs: &'a dyn SearchJobs,
        observer: &'a mut dyn SearchObserver,
        common: &'a CommonSettings,
        input: &MolecularSystem,
        job: &RelaxJob,
    ) -> MoptResult<Option<Self>> {
        let mut relaxed = jobs.relax(input, job)?;
        if relaxed.interrupted {
            return Ok(None);
        }
        if !relaxed.converged || relaxed.dissociated {
            return Err(MoptError::conv(
                "the optimization of the input structure failed (MultiOptPy stops here as \
                 well); start from a structure closer to a minimum or allow more iterations (-ns)",
            ));
        }
        // A symmetric input can end at a saddle point of its symmetry: go on to the minimum.
        if common.verify && job.saddle_order == 0 {
            while let Some(lower) = leave_saddle(jobs, &relaxed.system, relaxed.energy)? {
                observer.on_event(&SearchEvent::Note(format!(
                    "The optimized input is a saddle point ({:.3} kcal/mol above the minimum it \
                     leads to); the minimum is the first conformer.",
                    (relaxed.energy - lower.energy) * crate::domain::Units::HARTREE_TO_KCAL_MOL
                )))?;
                relaxed.system = lower.system;
                relaxed.energy = lower.energy;
                relaxed.bare_energy = lower.bare_energy;
            }
        }
        let reference = MolecularGraph::from_system(&relaxed.system, common.bond_scale);
        // Refuses networks through the boundaries of a cell, which have no conformers.
        whole_molecules(&relaxed.system, &reference)?;
        let ensemble =
            ConformerEnsemble::new(common.compare).comparing(common.compare_atoms.clone());
        let mut search = Self {
            jobs,
            observer,
            common,
            ensemble,
            reference,
            rejected: Vec::new(),
            tally: Tally::default(),
            interrupted: false,
            bare: Vec::new(),
        };
        search
            .ensemble
            .add(&relaxed.system, relaxed.energy, "input")?;
        search.bare.push(relaxed.bare_energy);
        search.tally.candidates += 1;
        search.tally.new += 1;
        let fate = CandidateFate::New {
            index: 0,
            energy: relaxed.energy,
        };
        search.observer.on_event(&SearchEvent::Candidate {
            origin: "input",
            fate: &fate,
            system: Some(&relaxed.system),
        })?;
        Ok(Some(search))
    }

    /// The lowest energy of the ensemble.
    pub fn lowest_energy(&self) -> f64 {
        self.ensemble
            .lowest()
            .map_or(f64::INFINITY, |i| self.ensemble.members()[i].energy)
    }

    /// Write `text` to the log.
    pub fn note(&mut self, text: impl Into<String>) -> MoptResult<()> {
        self.observer.on_event(&SearchEvent::Note(text.into()))
    }

    /// Judge the end of an optimization; reports it and returns what became of it.
    pub fn consider(
        &mut self,
        relaxed: MoptResult<Relaxed>,
        origin: &str,
    ) -> MoptResult<CandidateFate> {
        self.tally.candidates += 1;
        let system = relaxed.as_ref().ok().map(|r| r.system.clone());
        let fate = match relaxed {
            Err(e) if e.to_string().contains(super::OFF_CONSTRAINTS) => {
                CandidateFate::OffConstraints(e.to_string())
            }
            Err(e) => CandidateFate::Failed(e.to_string()),
            Ok(r) if r.interrupted => {
                self.interrupted = true;
                CandidateFate::Failed("interrupted by the STOP file".to_string())
            }
            Ok(r) if r.dissociated => CandidateFate::Dissociated,
            Ok(r) if !r.converged => CandidateFate::NotConverged,
            Ok(r) => self.judge(&r, origin)?,
        };
        match &fate {
            CandidateFate::New { .. } => self.tally.new += 1,
            CandidateFate::Duplicate { .. } => self.tally.duplicates += 1,
            CandidateFate::Rotamer { .. } => self.tally.rotamers += 1,
            CandidateFate::Rejected { .. } => self.tally.rejected += 1,
            CandidateFate::OutsideWindow(_) => self.tally.outside += 1,
            CandidateFate::OffConstraints(_) => self.tally.off_constraints += 1,
            _ => self.tally.failed += 1,
        }
        self.observer.on_event(&SearchEvent::Candidate {
            origin,
            fate: &fate,
            system: system.as_ref(),
        })?;
        Ok(fate)
    }

    fn judge(&mut self, relaxed: &Relaxed, origin: &str) -> MoptResult<CandidateFate> {
        let (sys, energy) = (&relaxed.system, relaxed.energy);
        if self.common.mode == Mode::Conformers {
            let graph = MolecularGraph::from_system(sys, self.common.bond_scale);
            let keeps = match self.common.graph_check {
                GraphCheck::Isomorphic => graph.is_isomorphic(&self.reference),
                GraphCheck::SameNumbering => graph.same_bonds(&self.reference),
            };
            if !keeps {
                let changes = self.reference.changes_to(&graph);
                self.rejected.push(Rejected {
                    system: sys.clone(),
                    energy,
                    origin: origin.to_string(),
                    changes: changes.clone(),
                });
                return Ok(CandidateFate::Rejected {
                    system: Box::new(sys.clone()),
                    energy,
                    changes,
                });
            }
        }
        if self.common.verify && matches!(self.ensemble.classify(sys, energy)?, Verdict::New(_)) {
            if let Some(lower) = leave_saddle(self.jobs, sys, energy)? {
                self.observer.on_event(&SearchEvent::Note(format!(
                    "{origin}: a saddle point ({:.3} kcal/mol above the minimum it leads to); \
                     the minimum is judged instead",
                    (energy - lower.energy) * crate::domain::Units::HARTREE_TO_KCAL_MOL
                )))?;
                return self.judge(&lower, origin);
            }
        }
        Ok(match self.ensemble.add(sys, energy, origin)? {
            Verdict::New(index) => {
                if self.bare.len() <= index {
                    self.bare.resize(index + 1, f64::NAN);
                }
                self.bare[index] = relaxed.bare_energy;
                CandidateFate::New { index, energy }
            }
            Verdict::Duplicate { of, .. } => CandidateFate::Duplicate { of },
            Verdict::Rotamer { of, .. } => CandidateFate::Rotamer { of },
        })
    }

    /// Whether `sys` lies within the RMSD threshold of a known conformer (any energy).
    fn known(&self, sys: &MolecularSystem) -> bool {
        let Ok(whole) = whole_molecules(sys, &self.reference) else {
            return false;
        };
        let threshold = self.common.compare.rmsd;
        self.ensemble.members().iter().any(|m| {
            whole_molecules(&m.system, &self.reference)
                .and_then(|w| direct_rmsd(&whole, &w))
                .is_ok_and(|d| d < threshold)
        })
    }

    /// Take `starts` through loose optimizations, the energy window, the final optimizations
    /// (with the terms, constraints, and order of `template`), and the judgement, in the
    /// order of `starts`.
    pub fn refine(
        &mut self,
        starts: &[(MolecularSystem, String)],
        template: &RelaxJob,
    ) -> MoptResult<Refined> {
        if starts.is_empty() || self.interrupted {
            return Ok(Refined::default());
        }
        if self.jobs.stop_requested() {
            self.interrupted = true;
            return Ok(Refined::default());
        }
        let low_before = self.lowest_energy();
        let jobs = self.jobs;
        let loose_job = |label: &str| RelaxJob {
            label: label.to_string(),
            level: Level::Loose,
            ..template.clone()
        };
        let loose: Vec<MoptResult<Relaxed>> = starts
            .par_iter()
            .map(|(s, origin)| jobs.relax(s, &loose_job(origin)))
            .collect();
        if loose
            .iter()
            .any(|r| r.as_ref().is_ok_and(|r| r.interrupted))
        {
            self.interrupted = true;
            return Ok(Refined::default());
        }
        // The energy window over the loose results and the known conformers.
        let low = loose
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .filter(|r| !r.dissociated)
            .map(|r| r.energy)
            .fold(low_before, f64::min);
        let mut batch = ConformerEnsemble::new(self.common.compare)
            .comparing(self.common.compare_atoms.clone());
        let mut survivors: Vec<(MolecularSystem, String)> = Vec::new();
        for (result, (_, origin)) in loose.into_iter().zip(starts) {
            let r = match result {
                Ok(r) if !r.dissociated => r,
                failed => {
                    let _ = self.consider(failed, origin)?;
                    continue;
                }
            };
            if r.energy - low > self.common.loose_window {
                self.tally.candidates += 1;
                self.tally.outside += 1;
                let fate = CandidateFate::OutsideWindow(r.energy - low);
                self.observer.on_event(&SearchEvent::Candidate {
                    origin,
                    fate: &fate,
                    system: Some(&r.system),
                })?;
                continue;
            }
            // Duplicates among the loose structures, and those of known conformers.
            if !matches!(batch.add(&r.system, r.energy, origin)?, Verdict::New(_)) {
                continue;
            }
            if self.common.skip_known && self.known(&r.system) {
                self.tally.candidates += 1;
                self.tally.duplicates += 1;
                continue;
            }
            survivors.push((r.system, origin.clone()));
        }
        let final_job = |label: &str| RelaxJob {
            label: label.to_string(),
            level: Level::Final,
            ..template.clone()
        };
        let finals: Vec<MoptResult<Relaxed>> = survivors
            .par_iter()
            .map(|(s, origin)| jobs.relax(s, &final_job(origin)))
            .collect();
        let mut refined = Refined::default();
        for (result, (_, origin)) in finals.into_iter().zip(&survivors) {
            if let CandidateFate::New { .. } = self.consider(result, origin)? {
                refined.new += 1;
            }
            if self.interrupted {
                return Ok(refined);
            }
        }
        refined.lower = self.lowest_energy() < low_before - self.common.compare.energy;
        Ok(refined)
    }

    /// The end of the search.
    pub fn finish(self, reason: String) -> SearchResult {
        SearchResult {
            ensemble: self.ensemble,
            reference: self.reference,
            rejected: self.rejected,
            reason,
            interrupted: self.interrupted,
            tally: self.tally,
            bare_energies: self.bare,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};
    use crate::workflow::confsearch::{DynamicsJob, Sampled};
    use faer::Mat;

    /// H2 with the force constant `k` (Hartree / Bohr^2) along its bond; every relaxation
    /// ends 1e-3 Hartree lower.
    struct Bond {
        k: f64,
        relaxations: std::sync::atomic::AtomicUsize,
    }

    impl SearchJobs for Bond {
        fn relax(&self, start: &MolecularSystem, _job: &RelaxJob) -> MoptResult<Relaxed> {
            self.relaxations
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Relaxed {
                system: start.clone(),
                energy: -1e-3,
                bare_energy: -1e-3,
                converged: true,
                dissociated: false,
                iterations: 1,
                interrupted: false,
            })
        }

        fn dynamics(&self, _start: &MolecularSystem, _job: &DynamicsJob) -> MoptResult<Sampled> {
            Ok(Sampled::default())
        }

        fn hessian(&self, _sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
            let k = self.k;
            Ok(Mat::from_fn(6, 6, |i, j| match (i, j) {
                (0, 0) | (3, 3) => k,
                (0, 3) | (3, 0) => -k,
                _ => 0.0,
            }))
        }

        fn reaction_ends(
            &self,
            _ts: &MolecularSystem,
            _label: &str,
        ) -> MoptResult<Option<[(MolecularSystem, f64); 2]>> {
            Ok(None)
        }

        fn stop_requested(&self) -> bool {
            false
        }
    }

    #[test]
    fn a_soft_imaginary_mode_above_the_old_cutoff_is_a_saddle() {
        let sys = MolecularSystem::new(
            vec![Element::H, Element::H],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let mu = 0.5 * sys.masses_amu()[0];
        // A curvature of -5 cm^-1 (above the former cutoff of -20 cm^-1, below the one
        // criterion of about -0.5 cm^-1), and one of -0.1 cm^-1.
        let curvature = |nu: f64| -(nu / crate::domain::Units::EIGVAL_MW_TO_CM1).powi(2) * mu;
        let soft = Bond {
            k: curvature(5.0),
            relaxations: Default::default(),
        };
        let modes = soft.modes(&sys).unwrap();
        assert!(
            (modes.wavenumbers[0] + 5.0).abs() < 1e-6,
            "{:?}",
            modes.wavenumbers
        );
        assert_eq!(modes.imaginary_modes(), 1);
        assert!(leave_saddle(&soft, &sys, 0.0).unwrap().is_some());
        // A flat mode within the criterion is a minimum, and nothing is optimized again.
        let flat = Bond {
            k: curvature(0.1),
            relaxations: Default::default(),
        };
        assert!(leave_saddle(&flat, &sys, 0.0).unwrap().is_none());
        assert_eq!(
            flat.relaxations.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }
}
