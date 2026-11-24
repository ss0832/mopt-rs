//! The goal-directed search of the mapper (`priority=rrt`, `goal=FILE`): the network grows
//! from the inputs towards a given goal structure by alternating a fair random expansion (a
//! group of minima with the same connection pattern chosen with equal probability, then a
//! member) and an expansion towards the goal (a group, then a member, with probabilities
//! proportional to `exp(c z)`, `z` the similarity of the local atomic environments to the
//! goal). The acceptance filter of new minima and the stop at the goal are in
//! [`super::driver`]; the rules are in `docs/source/mapper_design.md`.
//!
//! After A. Nakao, Y. Harabuchi, S. Maeda, K. Tsuda, Phys. Chem. Chem. Phys. 2022, 24,
//! 10305 (doi:10.1039/d2cp01079h), the rapidly exploring random tree (RRT) combined with
//! AFIR. The paper leaves the atoms over which the local environments run to its supporting
//! information; here an environment holds the distances to every other atom, matched within
//! their elements ([`similarity`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, HashMap};

use super::network::Network;
use super::priority::{Candidate, ExplorationPriority};
use crate::core_infra::rng::Rng;
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::stable::normalized_exp;
use crate::workflow::explore::identity::BondGraph;

/// Bonds of the connection pattern within this times the sums of the covalent radii (the
/// `lambda` of the paper).
pub const CONNECTION_SCALE: f64 = 1.1;

/// The weight `c` of the similarity in step B (the paper).
pub const DEFAULT_GOAL_WEIGHT: f64 = 1.0;

/// The locality `theta` of the similarity, per Angstrom (the paper).
pub const DEFAULT_LOCALITY: f64 = 20.0;

/// The settings of `priority=rrt` (`rrt_KEY=VALUE`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrtSettings {
    /// `c`: the weight of the similarity in the choices of step B.
    pub goal_weight: f64,
    /// `theta` (per Angstrom): how fast a difference of distances lowers the similarity.
    pub locality: f64,
}

impl Default for RrtSettings {
    fn default() -> Self {
        Self {
            goal_weight: DEFAULT_GOAL_WEIGHT,
            locality: DEFAULT_LOCALITY,
        }
    }
}

impl RrtSettings {
    /// The keys (without `rrt_`).
    pub const KEYS: &'static [&'static str] = &["goal_weight", "locality"];

    /// From the `rrt_*` settings (without the prefix).
    pub fn parse(options: &BTreeMap<String, String>) -> MoptResult<Self> {
        let mut s = Self::default();
        for (key, value) in options {
            let x: f64 = value.trim().parse().map_err(|_| {
                MoptError::parse(format!("rrt_{key} must be a number, not '{value}'"))
            })?;
            match key.as_str() {
                "goal_weight" if x.is_finite() && x >= 0.0 => s.goal_weight = x,
                "locality" if x.is_finite() && x > 0.0 => s.locality = x,
                "goal_weight" | "locality" => {
                    return Err(MoptError::parse(format!(
                        "rrt_{key} must be a number above 0 (rrt_goal_weight may be 0), not {x}"
                    )))
                }
                other => {
                    return Err(MoptError::parse(format!(
                        "unknown setting rrt_{other} of priority=rrt (rrt_{})",
                        Self::KEYS.join(", rrt_")
                    )))
                }
            }
        }
        Ok(s)
    }
}

/// The values taken by the greedy matching of the entries `(value, row, column)`: the
/// smallest (or, with `largest`, the largest) entry, its row and column removed, repeated
/// (the minimum and maximum value profiles of the paper).
fn profile(
    mut entries: Vec<(f64, usize, usize)>,
    rows: usize,
    cols: usize,
    largest: bool,
) -> Vec<f64> {
    if largest {
        entries.sort_by(|a, b| b.0.total_cmp(&a.0));
    } else {
        entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let (mut row_used, mut col_used) = (vec![false; rows], vec![false; cols]);
    let mut out = Vec::with_capacity(rows.min(cols));
    for (v, r, c) in entries {
        if !row_used[r] && !col_used[c] {
            row_used[r] = true;
            col_used[c] = true;
            out.push(v);
            if out.len() == rows.min(cols) {
                break;
            }
        }
    }
    out
}

/// The distances (Angstrom, minimum image) between every two atoms of `s`.
fn distances(s: &MolecularSystem) -> Vec<Vec<f64>> {
    let n = s.n_atoms();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    s.distance(AtomId::new(i), AtomId::new(j)).value() * Units::BOHR_TO_ANGSTROM
                })
                .collect()
        })
        .collect()
}

/// The atoms of each element (by atomic number) of `s`.
fn by_element(s: &MolecularSystem) -> BTreeMap<u8, Vec<usize>> {
    let mut out: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
    for (i, e) in s.elements.iter().enumerate() {
        out.entry(e.atomic_number()).or_default().push(i);
    }
    out
}

/// The similarity of the local atomic environments of `a` and `b` (between 0 and the number
/// of atoms; `locality` per Angstrom). For every element, atom `i` of `a` and `j` of `b` are
/// compared by their distances to the other atoms: within each element the differences
/// `|d_ik - d_jl|` are matched greedily from the smallest, and `S_ij` is the mean of
/// `exp(-locality v)` over the matched values; the atoms `i` and `j` are then matched greedily
/// from the largest `S_ij`, and the matched values are summed over all elements.
pub fn similarity(a: &MolecularSystem, b: &MolecularSystem, locality: f64) -> f64 {
    let (da, db) = (distances(a), distances(b));
    let (ea, eb) = (by_element(a), by_element(b));
    let mut total = 0.0;
    for (element, atoms_a) in &ea {
        let Some(atoms_b) = eb.get(element) else {
            continue;
        };
        let mut s_entries = Vec::with_capacity(atoms_a.len() * atoms_b.len());
        for (r, &i) in atoms_a.iter().enumerate() {
            for (c, &j) in atoms_b.iter().enumerate() {
                let mut sum = 0.0;
                let mut count = 0usize;
                for (other, ka) in &ea {
                    let Some(kb) = eb.get(other) else {
                        continue;
                    };
                    let ks: Vec<usize> = ka.iter().copied().filter(|&k| k != i).collect();
                    let ls: Vec<usize> = kb.iter().copied().filter(|&l| l != j).collect();
                    if ks.is_empty() || ls.is_empty() {
                        continue;
                    }
                    let mut entries = Vec::with_capacity(ks.len() * ls.len());
                    for (x, &k) in ks.iter().enumerate() {
                        for (y, &l) in ls.iter().enumerate() {
                            entries.push(((da[i][k] - db[j][l]).abs(), x, y));
                        }
                    }
                    for v in profile(entries, ks.len(), ls.len(), false) {
                        sum += (-locality * v).exp();
                        count += 1;
                    }
                }
                let s_ij = if count == 0 { 1.0 } else { sum / count as f64 };
                s_entries.push((s_ij, r, c));
            }
        }
        total += profile(s_entries, atoms_a.len(), atoms_b.len(), true)
            .iter()
            .sum::<f64>();
    }
    total
}

/// The goal-directed priority (`priority=rrt`).
pub struct RrtPriority {
    settings: RrtSettings,
    probabilities: Vec<f64>,
    /// The similarity of each node to the goal (structures never change).
    similarities: HashMap<usize, f64>,
    /// The connection pattern of each node.
    graphs: HashMap<usize, Option<BondGraph>>,
}

impl RrtPriority {
    /// The name in the registry.
    pub const NAME: &'static str = "rrt";

    /// The priority with `settings`.
    pub fn new(settings: RrtSettings) -> Self {
        Self {
            settings,
            probabilities: Vec::new(),
            similarities: HashMap::new(),
            graphs: HashMap::new(),
        }
    }

    /// The similarity `z` of node `id` to the goal `goal`: of the local environments for
    /// structures, minus the distance (in the units of the variables) for points of plain
    /// variables.
    pub fn similarity_to_goal(
        &mut self,
        network: &Network,
        id: usize,
        goal: usize,
    ) -> MoptResult<f64> {
        if let Some(&z) = self.similarities.get(&id) {
            return Ok(z);
        }
        let (a, b) = (&network.node(id)?.system, &network.node(goal)?.system);
        let z = match (a.structure(), b.structure()) {
            (Some(sa), Some(sb)) => similarity(sa, sb, self.settings.locality),
            _ => {
                let (x, y) = (a.flat(), b.flat());
                -(0..x.nrows().min(y.nrows()))
                    .map(|k| (x[k] - y[k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
        };
        self.similarities.insert(id, z);
        Ok(z)
    }

    /// The candidates (indices) grouped by their connection pattern, in the order of their
    /// first members (each point of plain variables a group of its own).
    fn clusters(
        &mut self,
        network: &Network,
        candidates: &[Candidate],
    ) -> MoptResult<Vec<Vec<usize>>> {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (k, c) in candidates.iter().enumerate() {
            if let std::collections::hash_map::Entry::Vacant(slot) = self.graphs.entry(c.node) {
                slot.insert(
                    network
                        .node(c.node)?
                        .system
                        .structure()
                        .map(|s| BondGraph::of(s, CONNECTION_SCALE)),
                );
            }
            let mine = self.graphs.get(&c.node).and_then(Option::as_ref);
            let found = mine.and_then(|g| {
                groups.iter().position(|members| {
                    let first = candidates[members[0]].node;
                    self.graphs
                        .get(&first)
                        .and_then(Option::as_ref)
                        .is_some_and(|h| h.same_as(g))
                })
            });
            match found {
                Some(i) => groups[i].push(k),
                None => groups.push(vec![k]),
            }
        }
        Ok(groups)
    }
}

/// An index drawn from normalized probabilities.
fn draw_probability(p: &[f64], rng: &mut Rng) -> usize {
    let u = rng.next_f64();
    let mut acc = 0.0;
    for (k, w) in p.iter().enumerate() {
        acc += w;
        if u < acc {
            return k;
        }
    }
    p.len().saturating_sub(1)
}

impl ExplorationPriority for RrtPriority {
    fn last_scores(&self) -> &[f64] {
        &self.probabilities
    }

    fn name(&self) -> &str {
        Self::NAME
    }

    fn select(
        &mut self,
        network: &Network,
        candidates: &[Candidate],
        job: usize,
        rng: &mut Rng,
    ) -> MoptResult<Option<usize>> {
        if candidates.is_empty() {
            return Ok(None);
        }
        let goal = network.goal_node().ok_or_else(|| {
            MoptError::config("priority=rrt grows the network towards a goal: give goal=FILE")
        })?;
        let groups = self.clusters(network, candidates)?;
        self.probabilities.resize(candidates.len(), 0.0);
        if job.is_multiple_of(2) {
            // Step A: a group, then a member, each with equal probability.
            for g in &groups {
                for &i in g {
                    self.probabilities[i] = 1.0 / (groups.len() * g.len()) as f64;
                }
            }
            let g = &groups[rng.gen_range_usize(groups.len())];
            return Ok(Some(g[rng.gen_range_usize(g.len())]));
        }
        // Step B: a group by its best similarity, then a member by its own.
        let c = self.settings.goal_weight;
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(groups.len());
        for g in &groups {
            let mut zs = Vec::with_capacity(g.len());
            for &k in g {
                zs.push(self.similarity_to_goal(network, candidates[k].node, goal)?);
            }
            z.push(zs);
        }
        let best: Vec<f64> = z
            .iter()
            .map(|zs| c * zs.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect();
        let group_probabilities = normalized_exp(&best);
        let gi = draw_probability(&group_probabilities, rng);
        let mut pick = None;
        for (i, g) in groups.iter().enumerate() {
            let logs: Vec<f64> = z[i].iter().map(|v| c * v).collect();
            let within = normalized_exp(&logs);
            for (&index, &probability) in g.iter().zip(&within) {
                self.probabilities[index] = group_probabilities[i] * probability;
            }
            if i == gi {
                pick = Some(g[draw_probability(&within, rng)]);
            }
        }
        Ok(pick)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};

    fn system(symbols: &[&str], angstrom: &[[f64; 3]]) -> MolecularSystem {
        let elements: Vec<Element> = symbols
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect();
        MolecularSystem::new(
            elements,
            &positions::<Angstrom>(angstrom),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn the_profiles_match_greedily() {
        let entries = vec![(3.0, 0, 0), (1.0, 0, 1), (2.0, 1, 0), (5.0, 1, 1)];
        assert_eq!(profile(entries.clone(), 2, 2, false), vec![1.0, 2.0]);
        assert_eq!(profile(entries, 2, 2, true), vec![5.0, 3.0]);
    }

    #[test]
    fn a_structure_is_most_similar_to_itself_and_to_its_renumbering() {
        let water = system(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        );
        // The same molecule turned, moved, and with its hydrogens renumbered.
        let turned = system(
            &["O", "H", "H"],
            &[
                [5.0, 1.0, 0.0],
                [5.0 - 0.24, 1.0, 0.93],
                [5.0 + 0.96, 1.0, 0.0],
            ],
        );
        let bent = system(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.70, 0.66, 0.0]],
        );
        let same = similarity(&water, &water, DEFAULT_LOCALITY);
        assert!((same - 3.0).abs() < 1e-12, "{same}");
        let moved = similarity(&water, &turned, DEFAULT_LOCALITY);
        assert!((moved - 3.0).abs() < 1e-9, "{moved}");
        let other = similarity(&water, &bent, DEFAULT_LOCALITY);
        assert!(other < 3.0 && other > 0.0, "{other}");
        // Symmetric for structures of the same composition.
        assert!((other - similarity(&bent, &water, DEFAULT_LOCALITY)).abs() < 1e-12);
    }

    #[test]
    fn the_settings_are_checked() {
        let mut o = BTreeMap::new();
        o.insert("goal_weight".to_string(), "2".to_string());
        assert_eq!(RrtSettings::parse(&o).unwrap().goal_weight, 2.0);
        o.insert("locality".to_string(), "-1".to_string());
        assert!(RrtSettings::parse(&o).is_err());
        o.clear();
        o.insert("nope".to_string(), "1".to_string());
        assert!(RrtSettings::parse(&o).is_err());
    }
}
