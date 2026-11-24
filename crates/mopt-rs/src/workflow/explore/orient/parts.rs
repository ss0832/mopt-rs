//! The parts of a complex for an orientation search: which atoms move together, which keep
//! their place, and their positions made whole across the boundaries of a cell.
//!
//! The parts are MultiOptPy's `-part` lists (every list one part, the atoms that no list names
//! one more part), or, without them, the molecules of the bond graph. A part stays where it is
//! when it is named fixed or when its bonds reach its own periodic images (a slab, a crystal,
//! a polymer): such a part cannot be rotated without breaking the cell.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py:run_orientsearch` (`num_parse`, the parts of
//!  `-part` with the rest as one more part).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use crate::core_infra::arg_parser::parse_atom_indices_zero_based;
use crate::domain::{AtomId, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::identity::BondGraph;

/// How the parts were chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartSource {
    /// From lists of atoms (`-part`).
    Lists,
    /// The molecules of the bond graph.
    Molecules,
}

/// The parts of a complex.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexParts {
    /// Atoms of each part (from 0, ascending).
    pub atoms: Vec<Vec<usize>>,
    /// Whether each part keeps its place and orientation.
    pub fixed: Vec<bool>,
    /// Whether each part is bonded to its own periodic images.
    pub periodic: Vec<bool>,
    /// How the parts were chosen.
    pub source: PartSource,
}

/// A displacement larger than this (Bohr) between two routes to the same atom through the
/// bonds of a part means that the bonds reach a periodic image.
const IMAGE_MISMATCH_BOHR: f64 = 1e-3;

impl ComplexParts {
    /// The parts of MultiOptPy's `-part`: every list (`1,2,3`, `1-3`, `1-4,7`) one part, the
    /// atoms that no list names one more part; `fixed` are part numbers from 1.
    pub fn from_lists(
        sys: &MolecularSystem,
        lists: &[String],
        fixed: &[usize],
        bond_scale: f64,
    ) -> MoptResult<Self> {
        let n = sys.n_atoms();
        let mut owner = vec![usize::MAX; n];
        let mut atoms: Vec<Vec<usize>> = Vec::new();
        for list in lists {
            let mut part = parse_atom_indices_zero_based(list)?;
            if part.is_empty() {
                return Err(MoptError::parse(format!("-part: '{list}' names no atoms")));
            }
            part.sort_unstable();
            part.dedup();
            for &i in &part {
                if i >= n {
                    return Err(MoptError::parse(format!(
                        "-part: atom {} of '{list}' is beyond the {n} atoms of the structure",
                        i + 1
                    )));
                }
                if owner[i] != usize::MAX {
                    return Err(MoptError::parse(format!(
                        "-part: atom {} is in part {} and in '{list}'",
                        i + 1,
                        owner[i] + 1
                    )));
                }
                owner[i] = atoms.len();
            }
            atoms.push(part);
        }
        let rest: Vec<usize> = (0..n).filter(|&i| owner[i] == usize::MAX).collect();
        if !rest.is_empty() {
            atoms.push(rest);
        }
        Self::with_atoms(sys, atoms, fixed, PartSource::Lists, bond_scale)
    }

    /// The molecules of the bond graph of `sys` as the parts.
    pub fn molecules(sys: &MolecularSystem, fixed: &[usize], bond_scale: f64) -> MoptResult<Self> {
        let graph = BondGraph::of(sys, bond_scale);
        Self::with_atoms(
            sys,
            graph.molecules.clone(),
            fixed,
            PartSource::Molecules,
            bond_scale,
        )
    }

    fn with_atoms(
        sys: &MolecularSystem,
        atoms: Vec<Vec<usize>>,
        fixed: &[usize],
        source: PartSource,
        bond_scale: f64,
    ) -> MoptResult<Self> {
        if atoms.len() < 2 {
            return Err(MoptError::config(match source {
                PartSource::Lists => {
                    "orientsearch needs at least two parts: the lists of -part name every atom"
                }
                PartSource::Molecules => {
                    "orientsearch needs at least two parts: the structure is one molecule (give \
                     -part)"
                }
            }));
        }
        let mut is_fixed = vec![false; atoms.len()];
        for &k in fixed {
            if k == 0 || k > atoms.len() {
                return Err(MoptError::parse(format!(
                    "fixed part {k} does not exist (parts 1 to {})",
                    atoms.len()
                )));
            }
            is_fixed[k - 1] = true;
        }
        let graph = BondGraph::of(sys, bond_scale);
        let periodic: Vec<bool> = atoms
            .iter()
            .map(|part| sys.has_pbc() && reaches_images(sys, &graph, part))
            .collect();
        for (f, p) in is_fixed.iter_mut().zip(&periodic) {
            *f |= *p;
        }
        if is_fixed.iter().all(|f| *f) {
            return Err(MoptError::config(
                "orientsearch has nothing to place: every part is fixed or bonded to its own \
                 periodic images",
            ));
        }
        Ok(Self {
            atoms,
            fixed: is_fixed,
            periodic,
            source,
        })
    }

    /// Keep in place every part with one of `atoms` (from 0; the atoms that constraints pin in
    /// space: frozen atoms, Cartesian components, oriented groups), so that no placement moves
    /// them. Returns the parts (from 1) that are kept for that reason and were not before; an
    /// error naming the constraints when no part is left to place.
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy places every part and leaves the
    /// constraints to its optimization).
    pub fn hold_parts_with(&mut self, atoms: &[usize]) -> MoptResult<Vec<usize>> {
        let mut held = Vec::new();
        for (k, part) in self.atoms.iter().enumerate() {
            if !self.fixed[k] && part.iter().any(|a| atoms.contains(a)) {
                self.fixed[k] = true;
                held.push(k + 1);
            }
        }
        if self.fixed.iter().all(|f| *f) {
            return Err(MoptError::config(
                "orientsearch has nothing to place: every part is fixed, bonded to its own \
                 periodic images, or holds atoms that constraints pin in space (-fix, -pc \
                 x|y|z|rot); such a part stays where it is in the input",
            ));
        }
        Ok(held)
    }

    /// Number of parts.
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Whether there are no parts (never for a constructed value).
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// The parts that move.
    pub fn movable(&self) -> Vec<usize> {
        (0..self.len()).filter(|&k| !self.fixed[k]).collect()
    }

    /// The parts that stay.
    pub fn staying(&self) -> Vec<usize> {
        (0..self.len()).filter(|&k| self.fixed[k]).collect()
    }

    /// One line per part for the log.
    pub fn describe(&self, sys: &MolecularSystem) -> Vec<String> {
        self.atoms
            .iter()
            .enumerate()
            .map(|(k, atoms)| {
                let mut formula: Vec<(String, usize)> = Vec::new();
                for &i in atoms {
                    let symbol = sys.elements[i].symbol().to_string();
                    match formula.iter_mut().find(|(s, _)| *s == symbol) {
                        Some((_, c)) => *c += 1,
                        None => formula.push((symbol, 1)),
                    }
                }
                let formula: String = formula
                    .iter()
                    .map(|(s, c)| {
                        if *c == 1 {
                            s.clone()
                        } else {
                            format!("{s}{c}")
                        }
                    })
                    .collect();
                let state = if self.periodic[k] {
                    " (bonded to its periodic images: stays)"
                } else if self.fixed[k] {
                    " (fixed)"
                } else {
                    ""
                };
                format!("Part {}: {formula}, atoms {}{state}", k + 1, ranges(atoms))
            })
            .collect()
    }
}

/// `1-3,7` for the atoms (from 0) `[0, 1, 2, 6]`.
pub fn ranges(atoms: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut k = 0;
    while k < atoms.len() {
        let start = atoms[k];
        let mut end = start;
        while k + 1 < atoms.len() && atoms[k + 1] == end + 1 {
            k += 1;
            end = atoms[k];
        }
        out.push(if end > start {
            format!("{}-{}", start + 1, end + 1)
        } else {
            format!("{}", start + 1)
        });
        k += 1;
    }
    out.join(",")
}

/// Positions (Bohr) of `part` made whole: each bonded group of atoms followed through its
/// bonds by minimum-image steps from its first atom, and each group put at the image nearest
/// to the first atom of the part. Without a cell the positions as they are.
pub fn whole_positions(sys: &MolecularSystem, graph: &BondGraph, part: &[usize]) -> Vec<[f64; 3]> {
    let raw = |i: usize| sys.position(AtomId::new(i)).value();
    let Some(cell) = sys.active_cell() else {
        return part.iter().map(|&i| raw(i)).collect();
    };
    let inside: std::collections::HashMap<usize, usize> =
        part.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    let mut out: Vec<Option<[f64; 3]>> = vec![None; part.len()];
    let anchor = raw(part[0]);
    for start in 0..part.len() {
        if out[start].is_some() {
            continue;
        }
        let d = cell.mic_delta(raw(part[start]), anchor);
        out[start] = Some([anchor[0] + d[0], anchor[1] + d[1], anchor[2] + d[2]]);
        let mut queue = VecDeque::from([start]);
        while let Some(k) = queue.pop_front() {
            let Some(here) = out[k] else { continue };
            for &j in &graph.adjacency[part[k]] {
                let Some(&m) = inside.get(&j) else { continue };
                if out[m].is_none() {
                    let step = cell.mic_delta(raw(j), raw(part[k]));
                    out[m] = Some([here[0] + step[0], here[1] + step[1], here[2] + step[2]]);
                    queue.push_back(m);
                }
            }
        }
    }
    out.into_iter()
        .zip(part)
        .map(|(p, &i)| p.unwrap_or_else(|| raw(i)))
        .collect()
}

/// Whether the bonds of `part` lead from an atom to one of its own periodic images.
fn reaches_images(sys: &MolecularSystem, graph: &BondGraph, part: &[usize]) -> bool {
    let Some(cell) = sys.active_cell() else {
        return false;
    };
    let raw = |i: usize| sys.position(AtomId::new(i)).value();
    let inside: std::collections::HashMap<usize, usize> =
        part.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    let mut at: Vec<Option<[f64; 3]>> = vec![None; part.len()];
    for start in 0..part.len() {
        if at[start].is_some() {
            continue;
        }
        at[start] = Some(raw(part[start]));
        let mut queue = VecDeque::from([start]);
        while let Some(k) = queue.pop_front() {
            let Some(here) = at[k] else { continue };
            for &j in &graph.adjacency[part[k]] {
                let Some(&m) = inside.get(&j) else { continue };
                let step = cell.mic_delta(raw(j), raw(part[k]));
                let there = [here[0] + step[0], here[1] + step[1], here[2] + step[2]];
                match at[m] {
                    None => {
                        at[m] = Some(there);
                        queue.push_back(m);
                    }
                    Some(known) => {
                        let gap = (0..3).map(|c| (known[c] - there[c]).powi(2)).sum::<f64>();
                        if gap.sqrt() > IMAGE_MISMATCH_BOHR {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::bohr;
    use crate::domain::{positions, Angstrom, Element, PeriodicCell, SpinCharge};
    use crate::workflow::explore::identity::DEFAULT_BOND_SCALE;

    fn waters(n: usize, cell: Option<PeriodicCell>) -> MolecularSystem {
        let mut pos = Vec::new();
        let mut el = Vec::new();
        for k in 0..n {
            let x = 3.0 * k as f64;
            pos.extend([[x, 0.0, 0.117], [x, 0.757, -0.469], [x, -0.757, -0.469]]);
            el.extend([Element::O, Element::H, Element::H]);
        }
        MolecularSystem::new(
            el,
            &positions::<Angstrom>(&pos),
            SpinCharge::default(),
            cell,
        )
        .unwrap()
    }

    #[test]
    fn parts_from_lists_follow_multioptpy() {
        let sys = waters(3, None);
        let p = ComplexParts::from_lists(&sys, &["1-3".into()], &[], DEFAULT_BOND_SCALE).unwrap();
        assert_eq!(p.atoms, vec![vec![0, 1, 2], vec![3, 4, 5, 6, 7, 8]]);
        let p = ComplexParts::from_lists(&sys, &["1,2,3".into(), "4-6".into()], &[2], 1.2).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p.fixed, vec![false, true, false]);
        assert_eq!(p.movable(), vec![0, 2]);
        assert!(p.describe(&sys)[1].contains("OH2, atoms 4-6 (fixed)"));
        assert!(ComplexParts::from_lists(&sys, &["1-9".into()], &[], 1.2).is_err());
        assert!(ComplexParts::from_lists(&sys, &["1-3".into(), "3-4".into()], &[], 1.2).is_err());
        assert!(ComplexParts::from_lists(&sys, &["1-10".into()], &[], 1.2).is_err());
        assert!(ComplexParts::from_lists(&sys, &["1-3".into()], &[3], 1.2).is_err());
        let m = ComplexParts::molecules(&sys, &[], 1.2).unwrap();
        assert_eq!(m.atoms.len(), 3);
        assert_eq!(m.source, PartSource::Molecules);
        assert!(ComplexParts::molecules(&waters(1, None), &[], 1.2).is_err());
        assert_eq!(ranges(&[0, 1, 2, 6, 8, 9]), "1-3,7,9-10");
    }

    #[test]
    fn a_chain_through_the_cell_stays_and_a_split_molecule_is_made_whole() {
        // A chain of carbon atoms 1.5 Angstrom apart through a cell of 4.5 Angstrom, and a
        // water molecule cut by the boundary.
        let cell = PeriodicCell::orthorhombic(
            bohr(4.5 * 1.889_726_1),
            bohr(12.0 * 1.889_726_1),
            bohr(12.0 * 1.889_726_1),
        )
        .unwrap();
        let sys = MolecularSystem::new(
            vec![
                Element::C,
                Element::C,
                Element::C,
                Element::O,
                Element::H,
                Element::H,
            ],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.1, 5.0, 0.117],
                [0.1, 5.757, -0.469],
                [4.45, 4.5, -0.3],
            ]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap();
        let p = ComplexParts::molecules(&sys, &[], DEFAULT_BOND_SCALE).unwrap();
        assert_eq!(p.atoms, vec![vec![0, 1, 2], vec![3, 4, 5]]);
        assert_eq!(p.periodic, vec![true, false]);
        assert_eq!(p.movable(), vec![1]);
        let graph = BondGraph::of(&sys, DEFAULT_BOND_SCALE);
        let whole = whole_positions(&sys, &graph, &p.atoms[1]);
        // The last hydrogen comes back next to the oxygen.
        let d = (0..3)
            .map(|c| (whole[2][c] - whole[0][c]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(d < 1.1 * 1.889_726_1, "{d}");
    }
}
