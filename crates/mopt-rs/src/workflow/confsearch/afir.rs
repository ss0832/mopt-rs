//! MultiOptPy's sampling of conformers: in trial `i` an AFIR force of `-bf` pushes (or, with
//! the negative sign, pulls) the `i`-th pair of a shuffled list of atom pairs during an
//! optimization from the current minimum; the result is optimized without the force and
//! judged; then the priority chooses the minimum of the next trial.
//!
//! Each pair of the target atoms (`-tgta`) enters the list twice, once with each sign, when it
//! passes the pair rule:
//!
//! - [`PairRule::MultiOptPy`]: MultiOptPy's window `1 < d / (R_i + R_j) < 8` as
//!   `make_tgt_atom_pair` computes it, the distance in Angstrom over covalent radii in Bohr
//!   (a ratio 1.89 times too small, so that bonded and geminal pairs fall out);
//! - [`PairRule::Ratio`]: the same window with both lengths in the same unit;
//! - [`PairRule::Bonds`] (the default, with 3 bonds; measured): pairs at least `min_bonds`
//!   bonds apart along the bond graph (the pairs whose distance a torsion changes), within
//!   `max_ratio` times the sum of the radii; in a structure without pairs that far apart (a
//!   compact cluster) the farthest pairs along the graph.
//!
//! The differences from MultiOptPy (the last pair is used, the energy check only compares
//! conformers of similar energy, seeded random numbers, ...) are listed in
//! `docs/source/migration_workflows.md`.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py:run_confsearch` (`make_tgt_atom_pair`, the
//!  loop over the samples, `switch_conformer`, the termination).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use super::search::{Search, Trials};
use super::{
    CommonSettings, Level, RelaxJob, SearchEvent, SearchJobs, SearchObserver, SearchResult,
};
use crate::coord::graph::MolecularGraph;
use crate::core_infra::rng::Rng;
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::parameters::sum_covalent_radii_bohr;
use crate::pes::bias::Afir;
use crate::pes::term::PesTerm;
use crate::workflow::explore::SamplingPriority;

/// Which atom pairs are pushed and pulled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PairRule {
    /// MultiOptPy's window of normalized distances, computed as MultiOptPy does.
    MultiOptPy {
        /// Lower end (1.0).
        min: f64,
        /// Upper end (8.0).
        max: f64,
    },
    /// The window of distances over the sums of the covalent radii.
    Ratio {
        /// Lower end.
        min: f64,
        /// Upper end.
        max: f64,
    },
    /// Pairs at least `min_bonds` bonds apart, within `max_ratio` times the sum of the radii.
    Bonds {
        /// Fewest bonds between the atoms.
        min_bonds: usize,
        /// Largest distance over the sum of the covalent radii.
        max_ratio: f64,
    },
}

/// Settings of the sampling.
#[derive(Debug, Clone, PartialEq)]
pub struct AfirSettings {
    /// `-bf`: the collision energy of the force (kJ/mol).
    pub base_force: f64,
    /// `-ms`: trials.
    pub max_samples: usize,
    /// `-nr`: the lowest conformers watched by the termination.
    pub rank: usize,
    /// `-nl`: trials without a change of them that end the search.
    pub patience: usize,
    /// `-tgta`: atoms of the pairs (from 0; empty: all).
    pub target_atoms: Vec<usize>,
    /// Which pairs.
    pub pairs: PairRule,
    /// `-nost`: every trial starts from the first conformer.
    pub no_stochastic: bool,
}

impl Default for AfirSettings {
    /// MultiOptPy's defaults, except the pairs: those at least 3 bonds apart found as many
    /// conformers as MultiOptPy's window or more in every one of nine runs of n-pentane,
    /// n-hexane, and Ace-Ala-NMe (docs/source/searches.md, "Conformation search").
    fn default() -> Self {
        Self {
            base_force: 100.0,
            max_samples: 50,
            rank: 10,
            patience: 5,
            target_atoms: Vec::new(),
            pairs: PairRule::Bonds {
                min_bonds: 3,
                max_ratio: 8.0,
            },
            no_stochastic: false,
        }
    }
}

/// The atom pairs with their signs (`+1` pushes, `-1` pulls), shuffled by `rng`.
pub fn atom_pairs(
    sys: &MolecularSystem,
    settings: &AfirSettings,
    bond_scale: f64,
    rng: &mut Rng,
) -> MoptResult<Vec<([usize; 2], f64)>> {
    let n = sys.n_atoms();
    let targets: Vec<usize> = if settings.target_atoms.is_empty() {
        (0..n).collect()
    } else {
        settings.target_atoms.clone()
    };
    if let Some(a) = targets.iter().find(|&&a| a >= n) {
        return Err(MoptError::config(format!(
            "-tgta: atom {} is not in the structure of {n} atoms",
            a + 1
        )));
    }
    let distances = MolecularGraph::from_system(sys, bond_scale).graph_distances();
    // In a compact cluster (every pair of atoms within two bonds, as in a metal cluster)
    // no pair is `min_bonds` apart: the farthest pairs are taken then.
    let farthest = targets
        .iter()
        .enumerate()
        .flat_map(|(k, &i)| targets[k + 1..].iter().map(move |&j| (i, j)))
        .map(|(i, j)| distances[i][j])
        .max()
        .unwrap_or(0);
    let mut out = Vec::new();
    for (k, &i) in targets.iter().enumerate() {
        for &j in &targets[k + 1..] {
            let (ai, aj) = (AtomId::new(i), AtomId::new(j));
            let r = sys.distance(ai, aj).value();
            let radii = sum_covalent_radii_bohr(sys.element(ai), sys.element(aj));
            let keep = match settings.pairs {
                // Angstrom over Bohr, as MultiOptPy's calc_normalized_distance_list.
                PairRule::MultiOptPy { min, max } => {
                    let ratio = r * Units::BOHR_TO_ANGSTROM / radii;
                    min < ratio && ratio < max
                }
                PairRule::Ratio { min, max } => {
                    let ratio = r / radii;
                    min < ratio && ratio < max
                }
                PairRule::Bonds {
                    min_bonds,
                    max_ratio,
                } => distances[i][j] >= min_bonds.min(farthest) && r / radii < max_ratio,
            };
            if keep {
                out.push(([i.min(j), i.max(j)], 1.0));
                out.push(([i.min(j), i.max(j)], -1.0));
            }
        }
    }
    rng.shuffle(&mut out);
    Ok(out)
}

/// Run MultiOptPy's sampling from `input` with `priority` choosing the starts.
#[allow(clippy::too_many_arguments)]
pub fn run(
    input: &MolecularSystem,
    settings: &AfirSettings,
    common: &CommonSettings,
    priority: &mut dyn SamplingPriority,
    jobs: &dyn SearchJobs,
    observer: &mut dyn SearchObserver,
) -> MoptResult<Option<SearchResult>> {
    let mut rng = Rng::from_seed(common.seed);
    let pairs = atom_pairs(input, settings, common.bond_scale, &mut rng)?;
    let first = RelaxJob::minimize("input", Level::Final);
    let Some(mut search) = Search::start(jobs, observer, common, input, &first)? else {
        return Ok(None);
    };
    if pairs.is_empty() {
        return Err(MoptError::config(
            "Cannot make atom_pair list: no pair of the target atoms passes the pair rule",
        ));
    }
    search.observer.on_event(&SearchEvent::Pairs(&pairs))?;
    let mut trials = Trials::new(
        settings.rank,
        settings.patience,
        settings.max_samples,
        settings.no_stochastic,
        priority,
        rng,
    );
    let mut reason = "Max samples are reached.".to_string();
    for trial in 0..settings.max_samples {
        if jobs.stop_requested() {
            search.interrupted = true;
            reason = "The stop signal is detected.".to_string();
            break;
        }
        let Some(&([a, b], sign)) = pairs.get(trial) else {
            reason = "All possible atom pairs are searched.".to_string();
            break;
        };
        let gamma = sign * settings.base_force;
        let current = trials.current;
        let start = search.ensemble.members()[current].system.clone();
        search.observer.on_event(&SearchEvent::Sample {
            trial,
            from: current,
            what: format!(
                "AFIR {gamma:+.1} kJ/mol between atoms {} and {}",
                a + 1,
                b + 1
            ),
        })?;
        let force: Arc<dyn PesTerm> = Arc::new(Afir::new(
            gamma,
            vec![AtomId::new(a)],
            vec![AtomId::new(b)],
        )?);
        let origin = format!(
            "sample {trial}: AFIR {gamma:+.0} kJ/mol {}-{}",
            a + 1,
            b + 1
        );
        let biased = jobs.relax(
            &start,
            &RelaxJob {
                terms: vec![("AFIR_conformation_search".to_string(), force)],
                ..RelaxJob::minimize(format!("{origin} (biased)"), Level::Final)
            },
        );
        let low_before = search.lowest_energy();
        let fate = match biased {
            Ok(r) if r.interrupted => {
                search.interrupted = true;
                reason = "The stop signal is detected.".to_string();
                break;
            }
            Ok(r) if !r.dissociated => {
                let relaxed =
                    jobs.relax(&r.system, &RelaxJob::minimize(origin.clone(), Level::Final));
                search.consider(relaxed, &origin)?
            }
            other => search.consider(other, &origin)?,
        };
        if search.interrupted {
            reason = "The stop signal is detected.".to_string();
            break;
        }
        if let Some(r) = trials.after(&mut search, trial, &fate, low_before)? {
            reason = r;
            break;
        }
    }
    Ok(Some(search.finish(reason)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::SpinCharge;
    use crate::io::xyz::parse_xyz_str;

    #[test]
    fn the_pair_rules_choose_the_pairs_a_torsion_moves() {
        let sys = parse_xyz_str(
            "14\nbutane\nC -1.9 0.4 0.0\nC -0.6 -0.4 0.0\nC 0.6 0.4 0.0\nC 1.9 -0.4 0.0\n\
H -2.0 1.0 0.9\nH -2.0 1.0 -0.9\nH -2.8 -0.3 0.0\nH -0.6 -1.0 0.9\nH -0.6 -1.0 -0.9\n\
H 0.6 1.0 0.9\nH 0.6 1.0 -0.9\nH 2.0 -1.0 0.9\nH 2.0 -1.0 -0.9\nH 2.8 0.3 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let carbons = AfirSettings {
            target_atoms: vec![0, 1, 2, 3],
            ..AfirSettings::default()
        };
        let pairs = |rule: PairRule| {
            let s = AfirSettings {
                pairs: rule,
                ..carbons.clone()
            };
            let mut p: Vec<[usize; 2]> = atom_pairs(&sys, &s, 1.1, &mut Rng::from_seed(0))
                .unwrap()
                .into_iter()
                .map(|(p, _)| p)
                .collect();
            p.sort_unstable();
            p.dedup();
            p
        };
        // MultiOptPy's units leave out the bonded and the 1,3 pairs; consistent units keep
        // the bonded ones (ratio 1.02).
        assert_eq!(
            pairs(PairRule::MultiOptPy { min: 1.0, max: 8.0 }),
            vec![[0, 3]]
        );
        let consistent = pairs(PairRule::Ratio { min: 1.0, max: 8.0 });
        assert!(consistent.contains(&[0, 1]) && consistent.contains(&[0, 2]));
        assert_eq!(
            pairs(PairRule::Bonds {
                min_bonds: 3,
                max_ratio: 8.0
            }),
            vec![[0, 3]]
        );
        // No pair of the carbons is 5 bonds apart: the farthest ones are taken.
        assert_eq!(
            pairs(PairRule::Bonds {
                min_bonds: 5,
                max_ratio: 8.0
            }),
            vec![[0, 3]]
        );
        // Both signs of each pair.
        let all = atom_pairs(&sys, &carbons, 1.1, &mut Rng::from_seed(0)).unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|p| p.1 > 0.0) && all.iter().any(|p| p.1 < 0.0));
    }
}
