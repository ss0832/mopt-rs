//! Bond graphs of structures: bonds from covalent radii, classes of equivalent atoms by colour
//! refinement, comparisons of two graphs (the same bonds in the same numbering, or the same
//! graph up to a renumbering of the atoms), graph distances, and molecules in periodic cells.
//!
//! The bonds are those of [`BondConnectivity`] (minimum-image distances within a scale of the
//! sum of the covalent radii). The classes start from the elements and are refined by the
//! classes of the neighbours until the partition is stable (the colour refinement of
//! Weisfeiler and Lehman); a class is named by the rank of its signature among the sorted
//! signatures, so two isomorphic graphs give their atoms the same class names. Isomorphism is
//! decided by a backtracking search that maps atoms only to atoms of the same class and keeps
//! the bonds among the atoms mapped so far.
//!
//! MultiOptPy: `Utils/bond_connectivity.py:BondConnectivity` (`bond_connect_matrix`,
//!  `bond_connect_table`) and the comparison of the bond tables in
//!  `Entrypoints/conformation_search.py:check_bond_connectivity_preserved`; the classes, the
//!  isomorphism, and the periodic molecules are new.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use crate::coord::connectivity::BondConnectivity;
use crate::domain::{AtomId, MolecularSystem, Position};
use crate::error::{MoptError, MoptResult};

/// MultiOptPy's scale of the sum of the covalent radii within which two atoms are bonded
/// (`BondConnectivity.covalent_radii_threshold`).
pub const MULTIOPTPY_BOND_SCALE: f64 = 1.1;

/// Largest number of partial mappings the isomorphism search tries before it gives up.
pub const ISOMORPHISM_BUDGET: usize = 2_000_000;

/// The bond graph of a structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MolecularGraph {
    atomic_numbers: Vec<u8>,
    neighbors: Vec<Vec<usize>>,
}

/// Bonds that differ between two graphs of the same atoms in the same numbering.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BondChanges {
    /// Bonds of the second graph that the first does not have (atoms from 0, `i < j`).
    pub formed: Vec<[usize; 2]>,
    /// Bonds of the first graph that the second does not have.
    pub broken: Vec<[usize; 2]>,
}

impl BondChanges {
    /// Whether no bond differs.
    pub fn is_empty(&self) -> bool {
        self.formed.is_empty() && self.broken.is_empty()
    }
}

impl MolecularGraph {
    /// Bonds of `sys` within `scale` times the sum of the covalent radii (minimum images).
    pub fn from_system(sys: &MolecularSystem, scale: f64) -> Self {
        let conn = BondConnectivity::from_system(sys, scale);
        let bonds: Vec<[usize; 2]> = conn
            .bonds
            .iter()
            .map(|[a, b]| [a.index(), b.index()])
            .collect();
        let numbers = sys.elements.iter().map(|e| e.atomic_number()).collect();
        Self::build(numbers, &bonds)
    }

    /// A graph of atoms with `atomic_numbers` and `bonds` (atoms from 0).
    pub fn from_bonds(atomic_numbers: Vec<u8>, bonds: &[[usize; 2]]) -> MoptResult<Self> {
        let n = atomic_numbers.len();
        if let Some(b) = bonds
            .iter()
            .find(|b| b[0] >= n || b[1] >= n || b[0] == b[1])
        {
            return Err(MoptError::config(format!(
                "bond {}-{} does not join two atoms of {n}",
                b[0] + 1,
                b[1] + 1
            )));
        }
        Ok(Self::build(atomic_numbers, bonds))
    }

    fn build(atomic_numbers: Vec<u8>, bonds: &[[usize; 2]]) -> Self {
        let mut neighbors = vec![Vec::new(); atomic_numbers.len()];
        for &[a, b] in bonds {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
        for list in &mut neighbors {
            list.sort_unstable();
            list.dedup();
        }
        Self {
            atomic_numbers,
            neighbors,
        }
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.atomic_numbers.len()
    }

    /// Atomic numbers of the atoms.
    pub fn atomic_numbers(&self) -> &[u8] {
        &self.atomic_numbers
    }

    /// Bonded neighbours of atom `i` (sorted).
    pub fn neighbors(&self, i: usize) -> &[usize] {
        &self.neighbors[i]
    }

    /// Whether atoms `i` and `j` are bonded.
    pub fn bonded(&self, i: usize, j: usize) -> bool {
        self.neighbors[i].binary_search(&j).is_ok()
    }

    /// Bonds `[i, j]` with `i < j`, sorted.
    pub fn bonds(&self) -> Vec<[usize; 2]> {
        let mut out = Vec::new();
        for (i, list) in self.neighbors.iter().enumerate() {
            out.extend(list.iter().filter(|&&j| j > i).map(|&j| [i, j]));
        }
        out
    }

    /// Bonds formed and broken from `self` to `other` (the same atoms, the same numbering).
    pub fn changes_to(&self, other: &Self) -> BondChanges {
        let (a, b) = (self.bonds(), other.bonds());
        BondChanges {
            formed: b.iter().filter(|x| !a.contains(x)).copied().collect(),
            broken: a.iter().filter(|x| !b.contains(x)).copied().collect(),
        }
    }

    /// Whether `other` has the same bonds in the same numbering.
    pub fn same_bonds(&self, other: &Self) -> bool {
        self == other
    }

    /// Connected components (bonded fragments), each sorted, in the order of their lowest atom.
    pub fn fragments(&self) -> Vec<Vec<usize>> {
        let n = self.n_atoms();
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            seen[start] = true;
            let mut queue = VecDeque::from([start]);
            let mut frag = Vec::new();
            while let Some(a) = queue.pop_front() {
                frag.push(a);
                for &b in &self.neighbors[a] {
                    if !seen[b] {
                        seen[b] = true;
                        queue.push_back(b);
                    }
                }
            }
            frag.sort_unstable();
            out.push(frag);
        }
        out
    }

    /// Number of bonds on the shortest path between every pair of atoms (`usize::MAX` between
    /// fragments).
    pub fn graph_distances(&self) -> Vec<Vec<usize>> {
        let n = self.n_atoms();
        (0..n)
            .map(|start| {
                let mut d = vec![usize::MAX; n];
                d[start] = 0;
                let mut queue = VecDeque::from([start]);
                while let Some(a) = queue.pop_front() {
                    for &b in &self.neighbors[a] {
                        if d[b] == usize::MAX {
                            d[b] = d[a] + 1;
                            queue.push_back(b);
                        }
                    }
                }
                d
            })
            .collect()
    }

    /// Classes of equivalent atoms by colour refinement: atoms of one class have the same
    /// element and, round after round, neighbours of the same classes. Isomorphic graphs give
    /// their atoms the same class names.
    pub fn classes(&self) -> Vec<usize> {
        let n = self.n_atoms();
        let mut color: Vec<usize> = self.atomic_numbers.iter().map(|z| *z as usize).collect();
        let mut count = distinct(&color);
        for _ in 0..n.max(1) {
            let signatures: Vec<(usize, Vec<usize>)> = (0..n)
                .map(|i| {
                    let mut around: Vec<usize> =
                        self.neighbors[i].iter().map(|&j| color[j]).collect();
                    around.sort_unstable();
                    (color[i], around)
                })
                .collect();
            let mut sorted = signatures.clone();
            sorted.sort();
            sorted.dedup();
            let next: Vec<usize> = signatures
                .iter()
                .map(|s| sorted.binary_search(s).unwrap_or(0))
                .collect();
            let next_count = sorted.len();
            color = next;
            if next_count == count {
                break;
            }
            count = next_count;
        }
        color
    }

    /// A renumbering `map` with `map[i]` the atom of `other` that atom `i` of `self` becomes,
    /// under which the bonds and elements agree; `None` when the graphs are not isomorphic (or
    /// the search exceeded [`ISOMORPHISM_BUDGET`] partial mappings, which colour refinement
    /// makes rare for molecules).
    pub fn isomorphism(&self, other: &Self) -> Option<Vec<usize>> {
        let n = self.n_atoms();
        if n != other.n_atoms() || self.bonds().len() != other.bonds().len() {
            return None;
        }
        if self.same_bonds(other) {
            return Some((0..n).collect());
        }
        let (ca, cb) = (self.classes(), other.classes());
        let mut sa = ca.clone();
        let mut sb = cb.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        if sa != sb {
            return None;
        }
        let size = |c: usize| ca.iter().filter(|&&x| x == c).count();
        // Atoms in breadth-first order from the atom of the rarest class of each fragment, so
        // that every atom after the first of a fragment has a mapped neighbour.
        let mut order = Vec::with_capacity(n);
        let mut placed = vec![false; n];
        for frag in self.fragments() {
            let Some(&start) = frag.iter().min_by_key(|&&a| (size(ca[a]), a)) else {
                continue;
            };
            placed[start] = true;
            let mut queue = VecDeque::from([start]);
            while let Some(a) = queue.pop_front() {
                order.push(a);
                for &b in &self.neighbors[a] {
                    if !placed[b] {
                        placed[b] = true;
                        queue.push_back(b);
                    }
                }
            }
        }
        let candidates = |depth: usize, map: &[Option<usize>], used: &[bool]| -> Vec<usize> {
            let u = order[depth];
            let mapped: Vec<usize> = self.neighbors[u].iter().filter_map(|&w| map[w]).collect();
            (0..n)
                .filter(|&v| {
                    !used[v]
                        && cb[v] == ca[u]
                        && other.neighbors[v].len() == self.neighbors[u].len()
                        && mapped.iter().all(|&m| other.bonded(m, v))
                        && other.neighbors[v].iter().filter(|&&x| used[x]).count() == mapped.len()
                })
                .collect()
        };
        let mut map: Vec<Option<usize>> = vec![None; n];
        let mut used = vec![false; n];
        let mut stack: Vec<(Vec<usize>, usize)> = vec![(candidates(0, &map, &used), 0)];
        let mut budget = ISOMORPHISM_BUDGET;
        loop {
            let depth = match stack.len() {
                0 => break,
                len => len - 1,
            };
            let u = order[depth];
            if let Some(v) = map[u].take() {
                used[v] = false;
            }
            let Some((options, next)) = stack.last_mut() else {
                break;
            };
            if *next >= options.len() {
                stack.pop();
                continue;
            }
            let v = options[*next];
            *next += 1;
            budget = budget.checked_sub(1)?;
            map[u] = Some(v);
            used[v] = true;
            if depth + 1 == n {
                return map.into_iter().collect();
            }
            let further = candidates(depth + 1, &map, &used);
            stack.push((further, 0));
        }
        None
    }

    /// Whether `other` is the same graph up to a renumbering of the atoms.
    pub fn is_isomorphic(&self, other: &Self) -> bool {
        self.isomorphism(other).is_some()
    }
}

fn distinct(values: &[usize]) -> usize {
    let mut v = values.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// Positions of the bonded fragments of `sys` made whole across the periodic boundaries, each
/// fragment moved to the periodic image whose centroid is nearest to that of the first: the
/// structure of molecules in a box as one cluster. Fails when the bonds of a fragment form an
/// infinite network through the boundaries (a crystal or a polymer), which has no such picture.
/// Structures without a periodic cell are returned as they are.
pub fn whole_molecules(
    sys: &MolecularSystem,
    graph: &MolecularGraph,
) -> MoptResult<MolecularSystem> {
    whole_molecules_at(sys, graph, FragmentAnchor::Centroid)
}

/// Reference point used to choose the relative periodic images of disconnected fragments.
#[derive(Clone, Copy)]
pub(crate) enum FragmentAnchor {
    Centroid,
    FirstAtom,
}

/// Unwrap bonded fragments and place each reference point near that of the first fragment.
pub(crate) fn whole_molecules_at(
    sys: &MolecularSystem,
    graph: &MolecularGraph,
    anchor: FragmentAnchor,
) -> MoptResult<MolecularSystem> {
    let Some(cell) = sys.active_cell() else {
        return Ok(sys.clone());
    };
    let n = sys.n_atoms();
    if graph.n_atoms() != n {
        return Err(MoptError::coord(
            "the bond graph does not match the structure",
        ));
    }
    let raw: Vec<[f64; 3]> = sys.atoms().map(|a| sys.position(a).value()).collect();
    let mut placed: Vec<Option<[f64; 3]>> = vec![None; n];
    let mut anchors = Vec::new();
    let mut members = Vec::new();
    for frag in graph.fragments() {
        let start = frag[0];
        placed[start] = Some(raw[start]);
        let mut queue = VecDeque::from([start]);
        while let Some(a) = queue.pop_front() {
            let pa = placed[a].unwrap_or(raw[a]);
            for &b in graph.neighbors(a) {
                let d = cell.mic_delta(raw[b], raw[a]);
                let pb = [pa[0] + d[0], pa[1] + d[1], pa[2] + d[2]];
                match placed[b] {
                    None => {
                        placed[b] = Some(pb);
                        queue.push_back(b);
                    }
                    Some(q) => {
                        let gap = (0..3).map(|k| (q[k] - pb[k]).abs()).fold(0.0, f64::max);
                        if gap > 1e-6 {
                            return Err(MoptError::unsupported(format!(
                                "the bonds of atom {} and its neighbours run through the periodic \
                                 boundaries into an infinite network (a crystal, a surface, or a \
                                 polymer): such a structure has no conformers",
                                b + 1
                            )));
                        }
                    }
                }
            }
        }
        let reference = match anchor {
            FragmentAnchor::FirstAtom => raw[start],
            FragmentAnchor::Centroid => std::array::from_fn(|k| {
                frag.iter()
                    .map(|&a| placed[a].unwrap_or(raw[a])[k] / frag.len() as f64)
                    .sum()
            }),
        };
        anchors.push(reference);
        members.push(frag);
    }
    let mut out = sys.clone();
    let first = anchors.first().copied().unwrap_or([0.0; 3]);
    for (c, frag) in anchors.iter().zip(&members) {
        let d = cell.mic_delta(*c, first);
        let shift = [
            first[0] + d[0] - c[0],
            first[1] + d[1] - c[1],
            first[2] + d[2] - c[2],
        ];
        for &a in frag {
            let p = placed[a].unwrap_or(raw[a]);
            out.set_position(
                AtomId::new(a),
                Position::new([p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]),
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::bohr;
    use crate::domain::{PeriodicCell, SpinCharge};
    use crate::io::xyz::parse_xyz_str;

    pub(crate) const BUTANE: &str = "14\nbutane\nC -1.9 0.4 0.0\nC -0.6 -0.4 0.0\n\
C 0.6 0.4 0.0\nC 1.9 -0.4 0.0\nH -2.0 1.0 0.9\nH -2.0 1.0 -0.9\nH -2.8 -0.3 0.0\n\
H -0.6 -1.0 0.9\nH -0.6 -1.0 -0.9\nH 0.6 1.0 0.9\nH 0.6 1.0 -0.9\nH 2.0 -1.0 0.9\n\
H 2.0 -1.0 -0.9\nH 2.8 0.3 0.0\n";

    fn butane() -> MolecularSystem {
        parse_xyz_str(BUTANE, SpinCharge::default(), None).unwrap()
    }

    #[test]
    fn butane_has_its_bonds_and_classes() {
        let g = MolecularGraph::from_system(&butane(), MULTIOPTPY_BOND_SCALE);
        assert_eq!(g.bonds().len(), 13);
        let c = g.classes();
        // Terminal carbons, inner carbons, methyl hydrogens, methylene hydrogens.
        assert_eq!(c[0], c[3]);
        assert_eq!(c[1], c[2]);
        assert_ne!(c[0], c[1]);
        assert!([4, 5, 6, 11, 12, 13].iter().all(|&i| c[i] == c[4]));
        assert!([7, 8, 9, 10].iter().all(|&i| c[i] == c[7]));
        assert_ne!(c[4], c[7]);
        let d = g.graph_distances();
        assert_eq!(d[0][3], 3);
        assert_eq!(d[4][13], 5);
        assert_eq!(g.fragments().len(), 1);
    }

    #[test]
    fn a_renumbered_graph_is_isomorphic_and_a_rearranged_one_is_not() {
        let g = MolecularGraph::from_system(&butane(), MULTIOPTPY_BOND_SCALE);
        // Renumber the atoms: atom i becomes perm[i].
        let perm: Vec<usize> = (0..14).map(|i| (i + 5) % 14).collect();
        let bonds: Vec<[usize; 2]> = g
            .bonds()
            .iter()
            .map(|[a, b]| [perm[*a], perm[*b]])
            .collect();
        let mut numbers = vec![0u8; 14];
        for (i, &p) in perm.iter().enumerate() {
            numbers[p] = g.atomic_numbers()[i];
        }
        let h = MolecularGraph::from_bonds(numbers.clone(), &bonds).unwrap();
        assert!(!g.same_bonds(&h));
        let map = g.isomorphism(&h).unwrap();
        for [a, b] in g.bonds() {
            assert!(h.bonded(map[a], map[b]));
        }
        assert_eq!(g.classes(), {
            let ch = h.classes();
            (0..14).map(|i| ch[map[i]]).collect::<Vec<_>>()
        });
        // Isobutane has the same atoms and a different graph.
        let mut iso = g.bonds();
        iso.retain(|b| *b != [2, 3]);
        iso.push([1, 3]);
        let other = MolecularGraph::from_bonds(g.atomic_numbers().to_vec(), &iso).unwrap();
        assert!(!g.is_isomorphic(&other));
        let changes = g.changes_to(&other);
        assert_eq!(changes.formed, vec![[1, 3]]);
        assert_eq!(changes.broken, vec![[2, 3]]);
        assert!(MolecularGraph::from_bonds(numbers, &[[0, 14]]).is_err());
    }

    #[test]
    fn molecules_in_a_box_are_made_whole_and_networks_are_refused() {
        let mut sys = butane();
        let cell = PeriodicCell::orthorhombic(bohr(30.0), bohr(30.0), bohr(30.0)).unwrap();
        sys.cell = Some(cell);
        // Move the last methyl group across the boundary.
        let mut shifted = sys.clone();
        for i in [3, 11, 12, 13] {
            let p = shifted.position(AtomId::new(i)).value();
            shifted.set_position(AtomId::new(i), Position::new([p[0] + 30.0, p[1], p[2]]));
        }
        let g = MolecularGraph::from_system(&shifted, MULTIOPTPY_BOND_SCALE);
        assert_eq!(g.bonds().len(), 13);
        let whole = whole_molecules(&shifted, &g).unwrap();
        let d =
            whole.position(AtomId::new(3)).value()[0] - whole.position(AtomId::new(2)).value()[0];
        assert!(d.abs() < 3.0, "{d}");
        // A chain of carbons through a short cell is an infinite network.
        let mut chain = parse_xyz_str(
            "3\n\nC 0.0 0.0 0.0\nC 1.45 0.0 0.0\nC 2.9 0.0 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        chain.cell = Some(
            PeriodicCell::orthorhombic(bohr(4.35 * 1.889_726), bohr(30.0), bohr(30.0)).unwrap(),
        );
        let g = MolecularGraph::from_system(&chain, MULTIOPTPY_BOND_SCALE);
        assert_eq!(g.bonds().len(), 3);
        assert!(whole_molecules(&chain, &g).is_err());
    }
}
