//! The distinct local minima an exploration has found, with how often each was reached and
//! sampled from, and their Boltzmann weights.
//!
//! A new structure is compared ([`super::identity::StructureMatcher`]) with the minima whose
//! energies lie within the energy tolerance, nearest energy first; the first one within the
//! RMSD tolerance takes it (and its structure when it is lower in energy), otherwise it becomes
//! a new minimum.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py` (the list of conformers `EQ<k>` with their
//!  energies and visit counts, `is_identical`, `calc_boltzmann_distribution`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::identity::{BondGraph, StructureMatcher};
use super::priority::EqRecord;
use crate::domain::{Bohr, Hartree, MolecularSystem, Quantity, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::stable::normalized_exp;

/// One distinct minimum.
#[derive(Debug, Clone)]
pub struct Minimum {
    /// The structure (the lowest of those that matched it).
    pub system: MolecularSystem,
    /// Its energy.
    pub energy: Quantity<Hartree>,
    /// The trial that found it first.
    pub first_trial: usize,
    /// The trial whose structure it holds.
    pub trial: usize,
    /// How many trials ended in it.
    pub hits: usize,
    /// How many trials were started from it.
    pub visits: usize,
    /// Its bond graph.
    pub graph: BondGraph,
}

/// What happened to a structure given to the pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Insertion {
    /// The minimum it is (index into [`MinimaPool::minima`]).
    pub index: usize,
    /// Whether it is a new minimum.
    pub new: bool,
    /// Whether it replaced the structure of a known minimum (it is lower in energy).
    pub replaced: bool,
    /// RMSD to the known minimum it matched.
    pub rmsd: Option<Quantity<Bohr>>,
}

/// The distinct minima of a search.
#[derive(Debug, Clone)]
pub struct MinimaPool {
    /// When two structures are the same.
    pub matcher: StructureMatcher,
    minima: Vec<Minimum>,
}

impl MinimaPool {
    /// An empty pool.
    pub fn new(matcher: StructureMatcher) -> Self {
        Self {
            matcher,
            minima: Vec::new(),
        }
    }

    /// The minima, in the order they were found.
    pub fn minima(&self) -> &[Minimum] {
        &self.minima
    }

    /// Number of distinct minima.
    pub fn len(&self) -> usize {
        self.minima.len()
    }

    /// Whether nothing was found yet.
    pub fn is_empty(&self) -> bool {
        self.minima.is_empty()
    }

    /// Add the result of trial `trial`.
    pub fn insert(
        &mut self,
        system: MolecularSystem,
        energy: Quantity<Hartree>,
        trial: usize,
    ) -> MoptResult<Insertion> {
        if !energy.value().is_finite() {
            return Err(MoptError::calc(format!(
                "trial {trial}: the energy is not finite"
            )));
        }
        let graph = BondGraph::of(&system, self.matcher.bond_scale);
        let mut near: Vec<usize> = (0..self.minima.len())
            .filter(|&k| {
                (self.minima[k].energy.value() - energy.value()).abs()
                    <= self.matcher.energy_tolerance.value()
            })
            .collect();
        near.sort_by(|&x, &y| {
            let dx = (self.minima[x].energy.value() - energy.value()).abs();
            let dy = (self.minima[y].energy.value() - energy.value()).abs();
            dx.total_cmp(&dy)
        });
        for k in near {
            let known = &self.minima[k];
            let Some(found) =
                self.matcher
                    .best_match_with(&known.system, &known.graph, &system, &graph)?
            else {
                continue;
            };
            if found.rmsd.value() > self.matcher.rmsd_tolerance.value() {
                continue;
            }
            let m = &mut self.minima[k];
            m.hits += 1;
            let replaced = energy.value() < m.energy.value();
            if replaced {
                m.system = system;
                m.energy = energy;
                m.trial = trial;
                m.graph = graph;
            }
            return Ok(Insertion {
                index: k,
                new: false,
                replaced,
                rmsd: Some(found.rmsd),
            });
        }
        self.minima.push(Minimum {
            system,
            energy,
            first_trial: trial,
            trial,
            hits: 1,
            visits: 0,
            graph,
        });
        Ok(Insertion {
            index: self.minima.len() - 1,
            new: true,
            replaced: false,
            rmsd: None,
        })
    }

    /// One more trial was started from minimum `index`.
    pub fn visit(&mut self, index: usize) -> MoptResult<()> {
        let m = self
            .minima
            .get_mut(index)
            .ok_or_else(|| MoptError::config(format!("no minimum {index}")))?;
        m.visits += 1;
        Ok(())
    }

    /// Indices of the minima from the lowest energy up.
    pub fn ranking(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.minima.len()).collect();
        order.sort_by(|&a, &b| {
            self.minima[a]
                .energy
                .value()
                .total_cmp(&self.minima[b].energy.value())
                .then(a.cmp(&b))
        });
        order
    }

    /// The energies and visit counts, for a [`super::SamplingPriority`].
    pub fn records(&self) -> Vec<EqRecord> {
        self.minima
            .iter()
            .map(|m| EqRecord {
                energy: m.energy,
                visits: m.visits,
            })
            .collect()
    }

    /// Boltzmann weights `exp(-E_k / k_B T) / sum` of the minima at `temperature` (K), in the
    /// order of [`Self::minima`].
    pub fn boltzmann_weights(&self, temperature: f64) -> MoptResult<Vec<f64>> {
        boltzmann_weights(
            &self
                .minima
                .iter()
                .map(|m| m.energy.value())
                .collect::<Vec<_>>(),
            temperature,
        )
    }
}

/// Boltzmann weights of `energies` (Hartree) at `temperature` (K), normalized without
/// overflow ([`normalized_exp`]).
pub fn boltzmann_weights(energies: &[f64], temperature: f64) -> MoptResult<Vec<f64>> {
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(MoptError::config(format!(
            "the temperature of Boltzmann weights must be positive, got {temperature}"
        )));
    }
    let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature;
    let exponents: Vec<f64> = energies.iter().map(|e| -e / kt).collect();
    Ok(normalized_exp(&exponents))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};

    fn water(shift: [f64; 3], swap: bool) -> MolecularSystem {
        let (h1, h2) = if swap { (2, 1) } else { (1, 2) };
        let base = [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ];
        let pos: Vec<[f64; 3]> = [0, h1, h2]
            .iter()
            .map(|&k| {
                let p = base[k];
                [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]
            })
            .collect();
        MolecularSystem::new(
            vec![Element::O, Element::H, Element::H],
            &positions::<Angstrom>(&pos),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn duplicates_are_merged_and_the_lowest_structure_is_kept() {
        let mut pool = MinimaPool::new(StructureMatcher::default());
        let a = pool
            .insert(water([0.0; 3], false), Quantity::new(-1.0), 0)
            .unwrap();
        assert!(a.new && a.index == 0);
        // The same molecule with its hydrogen atoms in the other order, moved: the same.
        let b = pool
            .insert(water([1.0, 2.0, 3.0], true), Quantity::new(-1.000_01), 1)
            .unwrap();
        assert!(!b.new && b.replaced && b.index == 0);
        assert!(b.rmsd.unwrap().value() < 1e-8);
        assert_eq!(pool.minima()[0].trial, 1);
        assert_eq!(pool.minima()[0].first_trial, 0);
        // Outside the energy window: another minimum, whatever the structure.
        let c = pool
            .insert(water([0.0; 3], false), Quantity::new(-0.99), 2)
            .unwrap();
        assert!(c.new && pool.len() == 2);
        assert_eq!(pool.ranking(), vec![0, 1]);
        assert_eq!(pool.minima()[0].hits, 2);
        pool.visit(1).unwrap();
        assert_eq!(pool.records()[1].visits, 1);
        assert!(pool.visit(5).is_err());
        assert!(pool
            .insert(water([0.0; 3], false), Quantity::new(f64::NAN), 3)
            .is_err());
    }

    #[test]
    fn boltzmann_weights_follow_the_energies() {
        let kt = Units::BOLTZMANN_HARTREE_PER_K * 298.15;
        let w = boltzmann_weights(&[-100.0, -100.0 + kt], 298.15).unwrap();
        let e = (-1.0_f64).exp();
        assert!((w[0] - 1.0 / (1.0 + e)).abs() < 1e-12);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-14);
        // Energies far apart: no overflow, the higher weight is zero.
        let w = boltzmann_weights(&[-1e4, 0.0], 298.15).unwrap();
        assert_eq!(w, vec![1.0, 0.0]);
        assert!(boltzmann_weights(&[0.0], 0.0).is_err());
    }
}
