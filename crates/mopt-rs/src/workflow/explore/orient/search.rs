//! The orientation search as an ask/tell state machine without I/O.
//!
//! `ask` gives the starting structure of the next sample (a random placement of the parts, or,
//! with `refine`, a known minimum with one part turned and shifted, the minimum chosen by a
//! [`crate::workflow::explore::SamplingPriority`]); the caller optimizes it with any optimizer
//! and calculator and `tell`s the result; converged results go into the pool of distinct minima
//! ([`crate::workflow::explore::MinimaPool`]). All random numbers come from one generator
//! seeded by `seed`, so the same results give the same samples.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py:run_orientsearch` (the loop over `-nsample`
//!  samples, each optimized by `Optimize`); the pool, the priorities, and the refinement are
//!  new.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::parts::ComplexParts;
use super::placement::{Perturbation, PlacementSettings, Sampler};
use crate::core_infra::rng::Rng;
use crate::domain::{Hartree, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::identity::{StructureMatcher, DEFAULT_BOND_SCALE};
use crate::workflow::explore::pool::{Insertion, MinimaPool};
use crate::workflow::explore::priority::{PriorityRegistry, SamplingPriority};

/// Settings of an orientation search.
#[derive(Debug, Clone, PartialEq)]
pub struct OrientSettings {
    /// `-nsample`: samples to optimize.
    pub samples: usize,
    /// How random samples are placed.
    pub placement: PlacementSettings,
    /// Share of the samples (after the first minimum) started from a known minimum.
    pub refine: f64,
    /// The rule that chooses that minimum (a name of the [`PriorityRegistry`]).
    pub priority: String,
    /// How such a start differs from its minimum.
    pub perturbation: Perturbation,
    /// When two minima are the same.
    pub matcher: StructureMatcher,
    /// Seed of the random numbers.
    pub seed: u64,
}

impl Default for OrientSettings {
    /// MultiOptPy's five samples with the defaults measured in running.md (`orientsearch`).
    fn default() -> Self {
        Self {
            samples: 5,
            placement: PlacementSettings::default(),
            refine: 0.0,
            priority: "tabu".to_string(),
            perturbation: Perturbation::default(),
            matcher: StructureMatcher {
                bond_scale: DEFAULT_BOND_SCALE,
                ..StructureMatcher::default()
            },
            seed: 1,
        }
    }
}

/// Where the start of a sample comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleOrigin {
    /// A random placement, after `redraws` draws rejected for clashes.
    Random {
        /// Draws rejected before it.
        redraws: usize,
    },
    /// Minimum `minimum` (index into the pool) with one part turned and shifted.
    Perturbed {
        /// The minimum.
        minimum: usize,
    },
}

/// A start to optimize.
#[derive(Debug, Clone)]
pub struct SampleRequest {
    /// Number of the sample (from 0).
    pub index: usize,
    /// The starting structure.
    pub start: MolecularSystem,
    /// Where it comes from.
    pub origin: SampleOrigin,
}

/// What the search asks for.
#[derive(Debug, Clone)]
pub enum OrientAction {
    /// Optimize this start and tell the result.
    Optimize(Box<SampleRequest>),
    /// All samples are done.
    Finished,
}

/// The optimized structure of a sample.
#[derive(Debug, Clone)]
pub struct SampleOutcome {
    /// The structure where the optimization ended.
    pub system: MolecularSystem,
    /// Its energy on the surface the optimization minimized, with the bias potentials (the
    /// one minima are ranked by and weighted with; [`OrientationSearch::tell_with_bare`] gives
    /// the energy without them as well).
    pub energy: Quantity<Hartree>,
    /// Whether the optimization converged.
    pub converged: bool,
}

/// What became of a sample.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleRecord {
    /// Number of the sample.
    pub index: usize,
    /// Where its start came from.
    pub origin: SampleOrigin,
    /// Whether its optimization converged (`false` also when it failed).
    pub converged: bool,
    /// Its energy, when it ended.
    pub energy: Option<Quantity<Hartree>>,
    /// The minimum it gave (converged samples only).
    pub insertion: Option<Insertion>,
}

/// The state machine.
pub struct OrientationSearch {
    settings: OrientSettings,
    sampler: Sampler,
    pool: MinimaPool,
    rng: Rng,
    priority: Box<dyn SamplingPriority>,
    pending: Option<(usize, SampleOrigin)>,
    records: Vec<SampleRecord>,
    /// The bare energy of every minimum of the pool (that of the structure the pool keeps).
    bare: Vec<Quantity<Hartree>>,
}

impl std::fmt::Debug for OrientationSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrientationSearch")
            .field("samples", &self.settings.samples)
            .field("minima", &self.pool.len())
            .field("told", &self.records.len())
            .finish()
    }
}

impl OrientationSearch {
    /// A search over the placements of `parts` of `template`.
    pub fn new(
        template: &MolecularSystem,
        parts: &ComplexParts,
        settings: OrientSettings,
        priorities: &PriorityRegistry,
    ) -> MoptResult<Self> {
        if !(0.0..=1.0).contains(&settings.refine) {
            return Err(MoptError::config(format!(
                "refine is a share of the samples from 0 to 1, got {}",
                settings.refine
            )));
        }
        let sampler = Sampler::new(
            template,
            parts,
            settings.placement,
            settings.matcher.bond_scale,
        )?;
        Ok(Self {
            priority: priorities.create(&settings.priority)?,
            rng: Rng::from_seed(settings.seed),
            pool: MinimaPool::new(settings.matcher),
            sampler,
            settings,
            pending: None,
            records: Vec::new(),
            bare: Vec::new(),
        })
    }

    /// The start of the next sample, or the end.
    pub fn ask(&mut self) -> MoptResult<OrientAction> {
        if let Some((index, _)) = self.pending {
            return Err(MoptError::config(format!(
                "tell the result of sample {index} before asking for the next one"
            )));
        }
        let index = self.records.len();
        if index >= self.settings.samples {
            return Ok(OrientAction::Finished);
        }
        let from_minimum = if self.settings.refine > 0.0 && !self.pool.is_empty() {
            let draw = self.rng.next_f64();
            if draw < self.settings.refine {
                self.priority.select(&self.pool.records(), &mut self.rng)
            } else {
                None
            }
        } else {
            None
        };
        let (start, origin) = match from_minimum {
            Some(minimum) => {
                self.pool.visit(minimum)?;
                let base = &self.pool.minima()[minimum].system;
                let start =
                    self.sampler
                        .perturb(base, &self.settings.perturbation, &mut self.rng)?;
                (start, SampleOrigin::Perturbed { minimum })
            }
            None => {
                let (start, redraws) = self.sampler.sample(&mut self.rng)?;
                (start, SampleOrigin::Random { redraws })
            }
        };
        self.pending = Some((index, origin));
        Ok(OrientAction::Optimize(Box::new(SampleRequest {
            index,
            start,
            origin,
        })))
    }

    /// The result of the sample asked for last (`None` when its optimization failed), on a
    /// surface without bias potentials (the bare energy is the energy).
    pub fn tell(&mut self, outcome: Option<SampleOutcome>) -> MoptResult<SampleRecord> {
        let bare = outcome.as_ref().map(|o| o.energy);
        self.tell_with_bare(outcome, bare)
    }

    /// [`Self::tell`] with the energy of the result without the bias potentials, `bare`
    /// (reported with the minimum; the ranking is by [`SampleOutcome::energy`]).
    pub fn tell_with_bare(
        &mut self,
        outcome: Option<SampleOutcome>,
        bare: Option<Quantity<Hartree>>,
    ) -> MoptResult<SampleRecord> {
        let (index, origin) = self
            .pending
            .take()
            .ok_or_else(|| MoptError::config("no sample was asked for"))?;
        let record = match outcome {
            None => SampleRecord {
                index,
                origin,
                converged: false,
                energy: None,
                insertion: None,
            },
            Some(o) => {
                let insertion = if o.converged {
                    let ins = self.pool.insert(o.system, o.energy, index)?;
                    if ins.new || ins.replaced {
                        if self.bare.len() <= ins.index {
                            self.bare.resize(ins.index + 1, Quantity::new(f64::NAN));
                        }
                        self.bare[ins.index] = bare.unwrap_or(o.energy);
                    }
                    Some(ins)
                } else {
                    None
                };
                SampleRecord {
                    index,
                    origin,
                    converged: o.converged,
                    energy: Some(o.energy),
                    insertion,
                }
            }
        };
        self.records.push(record.clone());
        Ok(record)
    }

    /// The distinct minima so far.
    pub fn pool(&self) -> &MinimaPool {
        &self.pool
    }

    /// The energy without the bias potentials of minimum `k` of the pool (its ranking energy
    /// when none was told).
    pub fn bare_energy(&self, k: usize) -> Quantity<Hartree> {
        self.bare
            .get(k)
            .copied()
            .filter(|e| e.value().is_finite())
            .unwrap_or_else(|| self.pool.minima()[k].energy)
    }

    /// What became of every sample told so far.
    pub fn records(&self) -> &[SampleRecord] {
        &self.records
    }

    /// The sampler of the placements.
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    /// The settings.
    pub fn settings(&self) -> &OrientSettings {
        &self.settings
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};

    fn dimer() -> MolecularSystem {
        let w = [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ];
        let mut pos: Vec<[f64; 3]> = w.to_vec();
        pos.extend(w.iter().map(|p| [p[0] + 3.0, p[1], p[2]]));
        MolecularSystem::new(
            [Element::O, Element::H, Element::H].repeat(2),
            &positions::<Angstrom>(&pos),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn asks_and_tells_in_turn_and_merges_equal_results() {
        let sys = dimer();
        let parts = ComplexParts::molecules(&sys, &[], DEFAULT_BOND_SCALE).unwrap();
        let settings = OrientSettings {
            samples: 3,
            refine: 1.0,
            ..OrientSettings::default()
        };
        let mut search =
            OrientationSearch::new(&sys, &parts, settings, &PriorityRegistry::default()).unwrap();
        let OrientAction::Optimize(first) = search.ask().unwrap() else {
            panic!("a sample first");
        };
        assert_eq!(first.index, 0);
        assert!(matches!(first.origin, SampleOrigin::Random { .. }));
        assert!(search.ask().is_err(), "tell before the next ask");
        // Every optimization "ends" in the input structure.
        let told = search
            .tell(Some(SampleOutcome {
                system: sys.clone(),
                energy: Quantity::new(-10.0),
                converged: true,
            }))
            .unwrap();
        assert!(told.insertion.unwrap().new);
        // With refine = 1 the next start is the known minimum perturbed.
        let OrientAction::Optimize(second) = search.ask().unwrap() else {
            panic!("a second sample");
        };
        assert_eq!(second.origin, SampleOrigin::Perturbed { minimum: 0 });
        let again = search
            .tell(Some(SampleOutcome {
                system: sys.clone(),
                energy: Quantity::new(-10.0),
                converged: true,
            }))
            .unwrap();
        assert!(!again.insertion.unwrap().new);
        let OrientAction::Optimize(_) = search.ask().unwrap() else {
            panic!("a third sample");
        };
        let failed = search.tell(None).unwrap();
        assert!(!failed.converged && failed.insertion.is_none());
        assert!(matches!(search.ask().unwrap(), OrientAction::Finished));
        assert_eq!(search.pool().len(), 1);
        assert_eq!(search.pool().minima()[0].hits, 2);
        // The second and the third sample started from it.
        assert_eq!(search.pool().minima()[0].visits, 2);
        assert_eq!(search.records().len(), 3);
        assert!(search.tell(None).is_err());
    }

    #[test]
    fn the_same_seed_gives_the_same_starts() {
        let sys = dimer();
        let parts = ComplexParts::molecules(&sys, &[], DEFAULT_BOND_SCALE).unwrap();
        let start = |seed: u64| {
            let settings = OrientSettings {
                seed,
                ..OrientSettings::default()
            };
            let mut s =
                OrientationSearch::new(&sys, &parts, settings, &PriorityRegistry::default())
                    .unwrap();
            match s.ask().unwrap() {
                OrientAction::Optimize(r) => r.start.coords_flat_bohr(),
                OrientAction::Finished => panic!("a sample"),
            }
        };
        assert_eq!(start(4), start(4));
        assert_ne!(start(4), start(5));
        let bad = OrientSettings {
            refine: 1.5,
            ..OrientSettings::default()
        };
        assert!(OrientationSearch::new(&sys, &parts, bad, &PriorityRegistry::default()).is_err());
    }
}
