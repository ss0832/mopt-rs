//! A SMILES string of a structure from its coordinates, for grouping the structures of an
//! exploration by chemical species: the conformers of one species have one string, whatever
//! the numbering of their atoms.
//!
//! The string is written in four steps.
//!
//! 1. **Bonds.** The bond graph of the structure ([`BondGraph`]: bonds within `bond_scale`
//!    times the sums of the covalent radii), its connected parts being the molecules.
//! 2. **Bond orders** from the usual valences (H, halogens 1; B 3; C, Si 4; N, P 3 or 5; O 2;
//!    S 2, 4 or 6). Every atom below the smallest of its valences that its bonds allow lacks
//!    bonds; bonds between two such atoms are raised, a maximum matching of them at a time,
//!    until none is left (C=C, C#N, the Kekule form of an aromatic ring). After that N, P and S
//!    take two more bonds from terminal O or S atoms that still lack one (nitro, sulfone,
//!    sulfonate, phosphate). Elements without a valence here (metals, noble gases) keep single
//!    bonds.
//! 3. **Charges.** An atom with one bond more than its valence is a cation or anion of the
//!    usual kind (N, P with four bonds, O, S with three: +1; B, Al with four: -1). What is left
//!    of the charge of the structure goes to atoms that still lack a bond (negative: O, S, N,
//!    the halogens, C; positive: C, Si, H, N, P, S, O), the molecule of several by the same
//!    order. An atom that lacks bonds after that is written as a radical (in brackets, with
//!    its hydrogens).
//! 4. **Canonical form.** The heavy atoms of a molecule are ordered by the colours of the
//!    Weisfeiler-Lehman refinement of [`BondGraph`]; where atoms share a colour, each of them
//!    is tried as the first (individualization and refinement, as in the canonical labelling
//!    of graphs: McKay, Congr. Numer. 1981, 30, 45), and of the strings of all complete
//!    orders the smallest is taken. Steps 2 and 3 are done within every order, so that the
//!    choice between equivalent Kekule forms or equivalent charged atoms does not depend on
//!    the input. At most [`MAX_ORDERINGS`] orders are tried per molecule; a molecule with more
//!    (several hundred equivalent atoms) has a string that can depend on the numbering.
//!
//! Hydrogens bonded to one heavy atom are written with it; other hydrogens are atoms. The
//! molecules are joined by `.` in the order of their strings.
//!
//! **Tetrahedral centres (R and S).** An atom with four neighbours that the colours of the
//! bond graph tell apart (four different substituents) gets `@` or `@@` from the coordinates:
//! looking from the first of its neighbours in the order of the string towards the atom, the
//! other three follow anticlockwise (`@`) or clockwise (`@@`). Two enantiomers have different
//! strings, and a structure and its mirror image the same one when the molecule is not
//! chiral, because the smallest string of all orders is taken.
//!
//! **Double bonds (E and Z).** A double bond outside every ring whose two atoms each have one
//! other neighbour, or two that the colours of the bond graph tell apart, gets `/` and `\` on
//! one single bond at each of its atoms, from the coordinates: the two marked neighbours lie
//! on the same side of the double bond (their dihedral angle about it below 90 degrees) or on
//! opposite sides. As in SMILES, `a/b` says that `b` lies above `a` and `a\b` that it lies
//! below, so `F/C=C/F` is the E and `F/C=C\F` the Z isomer. The first mark of a chain of
//! conjugated double bonds is `/`. An atom with two double bonds (an allene, a ketene) has
//! none, nor has a double bond twisted to within 12 degrees of a right angle, nor one whose
//! marked neighbour would be a hydrogen written with its atom (an imine N-H).
//!
//! What the string does not have: the configuration of double bonds in rings (the E and Z
//! forms of a large ring share a string), centres whose substituents differ only in their own
//! configuration, centres of three neighbours and a lone pair (sulfoxides, phosphines), axial
//! and planar chirality, aromatic symbols (rings are in a Kekule form), and isotopes. It is a label of the
//! bond graph with plausible bond orders, made without a calculation of the electrons; where
//! the valence rules do not apply (transition states, metal complexes, unusual charges) it
//! still separates different bond graphs, and its bond orders are a guess.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use super::identity::{mix, BondGraph};
use crate::domain::MolecularSystem;

/// The largest number of complete orders of the heavy atoms of one molecule that are tried for
/// the canonical string.
pub const MAX_ORDERINGS: usize = 720;

/// The valences of the element `z` that the bond orders are fitted to, ascending; empty for an
/// element without one here.
fn valences(z: u8) -> &'static [u32] {
    match z {
        1 | 9 | 17 | 35 | 53 => &[1],
        5 | 13 => &[3],
        6 | 14 | 32 => &[4],
        7 | 15 | 33 => &[3, 5],
        8 => &[2],
        16 | 34 => &[2, 4, 6],
        _ => &[],
    }
}

/// The element of atomic number `z` can be written without brackets.
fn organic(z: u8) -> bool {
    matches!(z, 5 | 6 | 7 | 8 | 9 | 15 | 16 | 17 | 35 | 53)
}

/// Rank of the element `z` among the atoms that take a negative charge (lower first).
fn anion_rank(z: u8) -> u32 {
    match z {
        8 => 0,
        16 => 1,
        7 => 2,
        9 => 3,
        17 => 4,
        35 => 5,
        53 => 6,
        6 => 7,
        _ => 8,
    }
}

/// Rank of the element `z` among the atoms that take a positive charge (lower first).
fn cation_rank(z: u8) -> u32 {
    match z {
        6 => 0,
        14 => 1,
        1 => 2,
        7 => 3,
        15 => 4,
        16 => 5,
        8 => 6,
        _ => 7,
    }
}

/// The mark of a tetrahedral centre at `centre` whose four neighbours lie at `around`, in the
/// order they are written in: `@` when, looking from the first towards the centre, the other
/// three follow anticlockwise, `@@` when clockwise; `None` when the four lie in a plane. The
/// sign is that of the volume of the tetrahedron of the neighbours, which does not depend on
/// where the centre lies among them.
pub fn tetrahedral_mark(centre: [f64; 3], around: [[f64; 3]; 4]) -> Option<&'static str> {
    let from_first =
        |k: usize| -> [f64; 3] { std::array::from_fn(|c| around[k][c] - around[0][c]) };
    let (a, b, c) = (from_first(1), from_first(2), from_first(3));
    let volume = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    // Against the size of the neighbours' distances: a flat arrangement has no handedness.
    let size: f64 = around
        .iter()
        .map(|p| {
            (0..3)
                .map(|c| (p[c] - centre[c]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .product::<f64>()
        .powf(0.75);
    if !volume.is_finite() || volume.abs() <= 1e-3 * size.max(1e-12) {
        return None;
    }
    Some(if volume < 0.0 { "@" } else { "@@" })
}

/// The bond orders and charges of one molecule in one order of its atoms.
struct Assignment {
    /// The order of every bond, by its two atoms (the lower index first).
    orders: BTreeMap<(usize, usize), u32>,
    /// The formal charge of every atom of the molecule (by atom).
    charges: BTreeMap<usize, i32>,
    /// The atoms that lack bonds after the charges, with how many.
    lacking: Vec<(usize, u32)>,
}

/// The structure as the graph the strings are written from.
struct Graph<'a> {
    system: &'a MolecularSystem,
    bonds: &'a BondGraph,
    /// Whether an atom is a hydrogen written with its heavy atom.
    implicit: Vec<bool>,
    /// Whether the tetrahedral centres get their marks (`@`, `@@`).
    tetrahedral: bool,
    /// Whether the double bonds get their marks (`/`, `\`).
    double_bonds: bool,
}

/// The smallest size of the cosine of the dihedral angle of the marked neighbours about a
/// double bond for it to have a configuration (12 degrees from a right angle).
const TWIST_COSINE: f64 = 0.2;

impl Graph<'_> {
    fn z(&self, i: usize) -> u8 {
        self.system.elements[i].atomic_number()
    }

    fn degree(&self, i: usize) -> u32 {
        self.bonds.adjacency[i].len() as u32
    }

    /// The bond orders and charges of the molecule `atoms` with the charge `charge`, the atoms
    /// taken in the order of `rank` (lower first) wherever there is a choice.
    fn assign(&self, atoms: &[usize], rank: &dyn Fn(usize) -> usize, charge: i32) -> Assignment {
        let mut orders: BTreeMap<(usize, usize), u32> = BTreeMap::new();
        for &i in atoms {
            for &j in &self.bonds.adjacency[i] {
                if i < j {
                    orders.insert((i, j), 1);
                }
            }
        }
        let key = |i: usize, j: usize| (i.min(j), i.max(j));
        // How many bonds every atom lacks to the smallest valence its bonds allow.
        let mut need: BTreeMap<usize, u32> = atoms
            .iter()
            .map(|&i| {
                let d = self.degree(i);
                let target = valences(self.z(i)).iter().copied().find(|&v| v >= d);
                (i, target.map_or(0, |t| t - d))
            })
            .collect();
        let mut sorted: Vec<usize> = atoms.to_vec();
        sorted.sort_by_key(|&i| rank(i));
        // Bonds between two atoms that lack one: a maximum matching at a time.
        loop {
            let open: Vec<usize> = sorted.iter().copied().filter(|i| need[i] > 0).collect();
            let mut partner: BTreeMap<usize, usize> = BTreeMap::new();
            for &i in &open {
                if partner.contains_key(&i) {
                    continue;
                }
                let mut seen: Vec<usize> = Vec::new();
                self.augment(i, &need, &orders, rank, &mut partner, &mut seen);
            }
            if partner.is_empty() {
                break;
            }
            for (&i, &j) in &partner {
                if i < j {
                    if let Some(o) = orders.get_mut(&(i, j)) {
                        *o += 1;
                    }
                    for a in [i, j] {
                        if let Some(n) = need.get_mut(&a) {
                            *n -= 1;
                        }
                    }
                }
            }
        }
        // Two more bonds of N, P, S from terminal O or S atoms that lack one.
        let bond_sum = |i: usize, orders: &BTreeMap<(usize, usize), u32>| -> u32 {
            self.bonds.adjacency[i]
                .iter()
                .map(|&j| orders.get(&key(i, j)).copied().unwrap_or(1))
                .sum()
        };
        for &x in &sorted {
            let allowed = valences(self.z(x));
            if allowed.len() < 2 || need[&x] > 0 {
                continue;
            }
            let Some(&most) = allowed.last() else {
                continue;
            };
            let room = most.saturating_sub(bond_sum(x, &orders));
            let mut ends: Vec<usize> = self.bonds.adjacency[x]
                .iter()
                .copied()
                .filter(|&j| {
                    matches!(self.z(j), 8 | 16 | 34) && self.degree(j) == 1 && need[&j] == 1
                })
                .collect();
            ends.sort_by_key(|&j| rank(j));
            let raises = 2 * (room.min(ends.len() as u32) / 2);
            for &j in ends.iter().take(raises as usize) {
                if let Some(o) = orders.get_mut(&key(x, j)) {
                    *o += 1;
                }
                if let Some(n) = need.get_mut(&j) {
                    *n = 0;
                }
            }
        }
        // The cations and anions of the usual kind.
        let mut charges: BTreeMap<usize, i32> = atoms.iter().map(|&i| (i, 0)).collect();
        let mut left = charge;
        for &i in &sorted {
            let sum = bond_sum(i, &orders);
            let formal = match self.z(i) {
                7 | 15 | 33 if sum == 4 => 1,
                8 | 16 | 34 if sum == 3 => 1,
                5 | 13 if sum == 4 => -1,
                _ => 0,
            };
            if formal != 0 {
                charges.insert(i, formal);
                need.insert(i, 0);
                left -= formal;
            }
        }
        // The rest of the charge on atoms that lack a bond.
        while left != 0 {
            let site = sorted
                .iter()
                .copied()
                .filter(|i| need[i] > 0 && charges[i] == 0)
                .min_by_key(|&i| {
                    let by = if left < 0 {
                        anion_rank(self.z(i))
                    } else {
                        cation_rank(self.z(i))
                    };
                    (by, rank(i))
                });
            let Some(i) = site else {
                break;
            };
            let unit = left.signum();
            charges.insert(i, unit);
            if let Some(n) = need.get_mut(&i) {
                *n -= 1;
            }
            left -= unit;
        }
        let lacking = sorted
            .iter()
            .filter(|i| need[i] > 0)
            .map(|&i| (i, need[&i]))
            .collect();
        Assignment {
            orders,
            charges,
            lacking,
        }
    }

    /// An augmenting path of the matching `partner` from the atom `i` among the atoms that lack
    /// a bond (the neighbours tried in the order of `rank`); `true` when `i` got a partner.
    fn augment(
        &self,
        i: usize,
        need: &BTreeMap<usize, u32>,
        orders: &BTreeMap<(usize, usize), u32>,
        rank: &dyn Fn(usize) -> usize,
        partner: &mut BTreeMap<usize, usize>,
        seen: &mut Vec<usize>,
    ) -> bool {
        let mut next: Vec<usize> = self.bonds.adjacency[i]
            .iter()
            .copied()
            .filter(|j| {
                need.get(j).is_some_and(|&n| n > 0)
                    && orders.get(&(i.min(*j), i.max(*j))).is_some_and(|&o| o < 3)
            })
            .collect();
        next.sort_by_key(|&j| rank(j));
        for j in next {
            if seen.contains(&j) {
                continue;
            }
            seen.push(j);
            let free = match partner.get(&j).copied() {
                None => true,
                Some(k) => {
                    partner.remove(&k);
                    partner.remove(&j);
                    if self.augment(k, need, orders, rank, partner, seen) {
                        true
                    } else {
                        partner.insert(k, j);
                        partner.insert(j, k);
                        false
                    }
                }
            };
            if free {
                partner.insert(i, j);
                partner.insert(j, i);
                return true;
            }
        }
        false
    }

    /// The position of atom `i` (Bohr).
    fn position(&self, i: usize) -> [f64; 3] {
        std::array::from_fn(|c| self.system.coords_bohr()[(i, c)])
    }

    /// Whether atom `i` is a tetrahedral centre of four neighbours that the colours of the
    /// bond graph tell apart.
    fn stereo_centre(&self, i: usize) -> bool {
        let around = &self.bonds.adjacency[i];
        around.len() == 4
            && (0..4).all(|a| {
                (0..a).all(|b| self.bonds.colors[around[a]] != self.bonds.colors[around[b]])
            })
    }

    /// Whether the bond between `i` and `j` lies in a ring: `j` is reached from `i` without
    /// it.
    fn in_ring(&self, i: usize, j: usize) -> bool {
        let mut seen = vec![false; self.bonds.adjacency.len()];
        seen[i] = true;
        let mut todo: Vec<usize> = self.bonds.adjacency[i]
            .iter()
            .copied()
            .filter(|&k| k != j)
            .collect();
        while let Some(k) = todo.pop() {
            if k == j {
                return true;
            }
            if std::mem::replace(&mut seen[k], true) {
                continue;
            }
            todo.extend(
                self.bonds.adjacency[k]
                    .iter()
                    .copied()
                    .filter(|&n| !seen[n]),
            );
        }
        false
    }

    /// The neighbour of the atom `s` of a double bond to `t` that carries its mark: of its
    /// other neighbours (one, or two of different colours, all by single bonds) the first by
    /// `rank` that is written and joined to `s` in the tree of the string. `None` where the
    /// atom has no configuration or no such neighbour.
    fn marked_neighbour(
        &self,
        s: usize,
        t: usize,
        a: &Assignment,
        rank: &dyn Fn(usize) -> usize,
        parent: &BTreeMap<usize, Option<usize>>,
    ) -> Option<usize> {
        let others: Vec<usize> = self.bonds.adjacency[s]
            .iter()
            .copied()
            .filter(|&k| k != t)
            .collect();
        let single = |k: usize| a.orders.get(&(s.min(k), s.max(k))).copied().unwrap_or(1) == 1;
        let told_apart = match others[..] {
            [k] => single(k),
            [k, l] => single(k) && single(l) && self.bonds.colors[k] != self.bonds.colors[l],
            _ => false,
        };
        if !told_apart {
            return None;
        }
        let in_tree = |k: usize| {
            parent.get(&s).copied().flatten() == Some(k)
                || parent.get(&k).copied().flatten() == Some(s)
        };
        others
            .into_iter()
            .filter(|&k| !self.implicit[k] && in_tree(k))
            .min_by_key(|&k| rank(k))
    }

    /// The cosine of the dihedral angle of `p`, `i`, `j`, `q` about the bond from `i` to `j`;
    /// `None` where `p` or `q` lies on the line of the bond.
    fn dihedral_cosine(&self, p: usize, i: usize, j: usize, q: usize) -> Option<f64> {
        let (xp, xi, xj, xq) = (
            self.position(p),
            self.position(i),
            self.position(j),
            self.position(q),
        );
        let sub = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] { std::array::from_fn(|c| u[c] - v[c]) };
        let dot = |u: [f64; 3], v: [f64; 3]| -> f64 { (0..3).map(|c| u[c] * v[c]).sum() };
        let axis = sub(xj, xi);
        let length = dot(axis, axis);
        if length <= 1e-12 {
            return None;
        }
        // The parts of the two bonds across the axis.
        let across = |u: [f64; 3]| -> [f64; 3] {
            let along = dot(u, axis) / length;
            std::array::from_fn(|c| u[c] - along * axis[c])
        };
        let (u, v) = (across(sub(xp, xi)), across(sub(xq, xj)));
        let size = (dot(u, u) * dot(v, v)).sqrt();
        (size > 1e-8).then(|| dot(u, v) / size)
    }

    /// The marks `/` and `\` of the tree bonds (parent, child) next to the double bonds that
    /// have a configuration (see the module text), for the tree `parent` of the string.
    fn double_bond_marks(
        &self,
        nodes: &[usize],
        a: &Assignment,
        rank: &dyn Fn(usize) -> usize,
        parent: &BTreeMap<usize, Option<usize>>,
    ) -> BTreeMap<(usize, usize), char> {
        let mut marks: BTreeMap<(usize, usize), char> = BTreeMap::new();
        if !self.double_bonds {
            return marks;
        }
        // A marked bond as (parent, child), and whether the neighbour is the child.
        let bond_of = |s: usize, r: usize| -> ((usize, usize), bool) {
            if parent.get(&r).copied().flatten() == Some(s) {
                ((s, r), true)
            } else {
                ((r, s), false)
            }
        };
        // The double bonds of the tree with a configuration: their marked bonds and whether
        // the two marked neighbours lie on the same side.
        let mut bonds: Vec<[((usize, usize), bool); 2]> = Vec::new();
        let mut same: Vec<bool> = Vec::new();
        for &j in nodes {
            let Some(i) = parent.get(&j).copied().flatten() else {
                continue;
            };
            if a.orders.get(&(i.min(j), i.max(j))).copied().unwrap_or(1) != 2 || self.in_ring(i, j)
            {
                continue;
            }
            let (Some(ri), Some(rj)) = (
                self.marked_neighbour(i, j, a, rank, parent),
                self.marked_neighbour(j, i, a, rank, parent),
            ) else {
                continue;
            };
            let Some(cosine) = self.dihedral_cosine(ri, i, j, rj) else {
                continue;
            };
            if cosine.abs() < TWIST_COSINE {
                continue;
            }
            bonds.push([bond_of(i, ri), bond_of(j, rj)]);
            same.push(cosine > 0.0);
        }
        // `a/b`: b lies above a. Whether the neighbour lies above the atom of the double bond.
        let above = |mark: char, neighbour_is_child: bool| (mark == '/') == neighbour_is_child;
        let mark_for = |up: bool, neighbour_is_child: bool| {
            if up == neighbour_is_child {
                '/'
            } else {
                '\\'
            }
        };
        let mut done = vec![false; bonds.len()];
        while done.iter().any(|d| !d) {
            // A double bond next to a marked bond first (a chain of conjugated double bonds
            // shares the bonds between them); otherwise the first that is left.
            let next = (0..bonds.len())
                .filter(|&k| !done[k])
                .find(|&k| bonds[k].iter().any(|(b, _)| marks.contains_key(b)))
                .or_else(|| (0..bonds.len()).find(|&k| !done[k]));
            let Some(k) = next else {
                break;
            };
            done[k] = true;
            let [(b0, child0), (b1, child1)] = bonds[k];
            match (marks.get(&b0).copied(), marks.get(&b1).copied()) {
                (None, None) => {
                    marks.insert(b0, '/');
                    let up = above('/', child0);
                    marks.insert(b1, mark_for(up == same[k], child1));
                }
                (Some(m), None) => {
                    let up = above(m, child0);
                    marks.insert(b1, mark_for(up == same[k], child1));
                }
                (None, Some(m)) => {
                    let up = above(m, child1);
                    marks.insert(b0, mark_for(up == same[k], child0));
                }
                // Both marked already: only in a ring of marked bonds, which the double
                // bonds outside the rings do not form.
                (Some(_), Some(_)) => {}
            }
        }
        marks
    }

    /// The text of the atom `i` with the bonds `a`; `mark` is that of a tetrahedral centre.
    fn atom_text(&self, i: usize, a: &Assignment, mark: Option<&str>) -> String {
        let z = self.z(i);
        let symbol = self.system.elements[i].symbol();
        let hydrogens = self.bonds.adjacency[i]
            .iter()
            .filter(|&&j| self.implicit[j])
            .count() as u32;
        let charge = a.charges.get(&i).copied().unwrap_or(0);
        let sum: u32 = self.bonds.adjacency[i]
            .iter()
            .map(|&j| a.orders.get(&(i.min(j), i.max(j))).copied().unwrap_or(1))
            .sum();
        // Without brackets: an element of the organic subset whose hydrogens are those its
        // other bonds imply.
        let heavy = sum - hydrogens;
        let implied = valences(z)
            .iter()
            .copied()
            .find(|&v| v >= heavy)
            .map(|v| v - heavy);
        if organic(z) && charge == 0 && implied == Some(hydrogens) && mark.is_none() {
            return symbol.to_string();
        }
        let mut text = format!("[{symbol}{}", mark.unwrap_or(""));
        match hydrogens {
            0 => {}
            1 => text.push('H'),
            n => text.push_str(&format!("H{n}")),
        }
        match charge {
            0 => {}
            1 => text.push('+'),
            -1 => text.push('-'),
            c if c > 0 => text.push_str(&format!("+{c}")),
            c => text.push_str(&format!("{c}")),
        }
        text.push(']');
        text
    }

    /// The string of the molecule `atoms` in the order `rank` of its atoms with the bonds `a`.
    fn write(&self, atoms: &[usize], rank: &dyn Fn(usize) -> usize, a: &Assignment) -> String {
        let mut nodes: Vec<usize> = atoms
            .iter()
            .copied()
            .filter(|&i| !self.implicit[i])
            .collect();
        nodes.sort_by_key(|&i| rank(i));
        let Some(&root) = nodes.first() else {
            return String::new();
        };
        let neighbours = |i: usize| -> Vec<usize> {
            let mut n: Vec<usize> = self.bonds.adjacency[i]
                .iter()
                .copied()
                .filter(|&j| !self.implicit[j])
                .collect();
            n.sort_by_key(|&j| rank(j));
            n
        };
        // The spanning tree from the root and the bonds that close its rings.
        let mut parent: BTreeMap<usize, Option<usize>> = BTreeMap::new();
        let mut children: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        // The ring bonds of every atom: (the other atom, whether this atom opens the ring).
        let mut rings: BTreeMap<usize, Vec<(usize, bool)>> = BTreeMap::new();
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        parent.insert(root, None);
        while let Some(&(i, at)) = stack.last() {
            let n = neighbours(i);
            let Some(&j) = n.get(at) else {
                stack.pop();
                continue;
            };
            if let Some(top) = stack.last_mut() {
                top.1 += 1;
            }
            if parent.get(&i).copied().flatten() == Some(j) {
                continue;
            }
            if parent.contains_key(&j) {
                // A bond to an atom met before: a ring, opened at the atom that is still on
                // the stack (the ancestor), closed here. Seen from the ancestor later, it is
                // known.
                let known = rings
                    .get(&i)
                    .is_some_and(|r| r.iter().any(|(other, _)| *other == j));
                if !known {
                    rings.entry(j).or_default().push((i, true));
                    rings.entry(i).or_default().push((j, false));
                }
                continue;
            }
            parent.insert(j, Some(i));
            children.entry(i).or_default().push(j);
            stack.push((j, 0));
        }
        let marks = self.double_bond_marks(&nodes, a, rank, &parent);
        let bond = |i: usize, j: usize| -> &'static str {
            match a.orders.get(&(i.min(j), i.max(j))).copied().unwrap_or(1) {
                2 => "=",
                3 => "#",
                // (A marked bond is a bond of the tree, written from the parent `i`.)
                _ => match marks.get(&(i, j)) {
                    Some('/') => "/",
                    Some(_) => "\\",
                    None => "",
                },
            }
        };
        let digit = |n: usize| -> String {
            if n < 10 {
                n.to_string()
            } else {
                format!("%{n}")
            }
        };
        // Written with a stack of what is still to be written: an atom (after the bond from
        // its parent) or a closing parenthesis.
        enum Piece {
            Atom(usize),
            Text(&'static str),
        }
        let mut out = String::new();
        let mut numbers: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut todo: Vec<Piece> = vec![Piece::Atom(root)];
        while let Some(piece) = todo.pop() {
            let i = match piece {
                Piece::Text(t) => {
                    out.push_str(t);
                    continue;
                }
                Piece::Atom(i) => i,
            };
            if let Some(Some(p)) = parent.get(&i) {
                out.push_str(bond(*p, i));
            }
            // A tetrahedral centre: its neighbours in the order of the string (the atom
            // before it, its hydrogen, its ring bonds, its branches) decide its mark.
            let mark = if self.tetrahedral && self.stereo_centre(i) {
                let mut order: Vec<usize> = Vec::with_capacity(4);
                order.extend(parent.get(&i).copied().flatten());
                order.extend(
                    self.bonds.adjacency[i]
                        .iter()
                        .copied()
                        .filter(|&j| self.implicit[j]),
                );
                order.extend(
                    rings
                        .get(&i)
                        .map_or(&[][..], Vec::as_slice)
                        .iter()
                        .map(|(other, _)| *other),
                );
                order.extend(children.get(&i).map_or(&[][..], Vec::as_slice));
                match order[..] {
                    [n1, n2, n3, n4] => tetrahedral_mark(
                        self.position(i),
                        [n1, n2, n3, n4].map(|j| self.position(j)),
                    ),
                    _ => None,
                }
            } else {
                None
            };
            out.push_str(&self.atom_text(i, a, mark));
            for &(other, opens) in rings.get(&i).map_or(&[][..], Vec::as_slice) {
                if opens {
                    let n = numbers.len() + 1;
                    numbers.insert((i, other), n);
                    out.push_str(&digit(n));
                } else if let Some(&n) = numbers.get(&(other, i)) {
                    out.push_str(bond(other, i));
                    out.push_str(&digit(n));
                }
            }
            let kids = children.get(&i).map_or(&[][..], Vec::as_slice);
            // The last child follows without parentheses; the others are branches before it.
            for (k, &c) in kids.iter().enumerate().rev() {
                if k + 1 < kids.len() {
                    todo.push(Piece::Text(")"));
                    todo.push(Piece::Atom(c));
                    todo.push(Piece::Text("("));
                } else {
                    todo.push(Piece::Atom(c));
                }
            }
        }
        out
    }

    /// The complete orders of the atoms of the molecule `atoms` that are written (not the
    /// hydrogens written with a heavy atom): their colours refined, and every atom of a colour
    /// that several share tried as the first of them. Each is the atoms from the first to the
    /// last. At most [`MAX_ORDERINGS`].
    fn orderings(&self, atoms: &[usize]) -> Vec<Vec<usize>> {
        let nodes: Vec<usize> = atoms
            .iter()
            .copied()
            .filter(|&i| !self.implicit[i])
            .collect();
        let refine = |colors: &mut BTreeMap<usize, u64>| {
            let count = |c: &BTreeMap<usize, u64>| {
                let mut v: Vec<u64> = c.values().copied().collect();
                v.sort_unstable();
                v.dedup();
                v.len()
            };
            let mut classes = count(colors);
            for _ in 0..nodes.len() {
                let next: BTreeMap<usize, u64> = nodes
                    .iter()
                    .map(|&i| {
                        let mut n: Vec<u64> = self.bonds.adjacency[i]
                            .iter()
                            .filter_map(|j| colors.get(j).copied())
                            .collect();
                        n.sort_unstable();
                        (
                            i,
                            n.iter()
                                .fold(mix(colors[&i], n.len() as u64), |h, &c| mix(h, c)),
                        )
                    })
                    .collect();
                let found = count(&next);
                *colors = next;
                if found == classes {
                    break;
                }
                classes = found;
            }
        };
        let mut start: BTreeMap<usize, u64> =
            nodes.iter().map(|&i| (i, self.bonds.colors[i])).collect();
        refine(&mut start);
        let mut out: Vec<Vec<usize>> = Vec::new();
        let mut open: Vec<BTreeMap<usize, u64>> = vec![start];
        while let Some(colors) = open.pop() {
            if out.len() >= MAX_ORDERINGS {
                break;
            }
            // The colour shared by the fewest atoms (the smallest colour of those).
            let mut sizes: BTreeMap<u64, usize> = BTreeMap::new();
            for c in colors.values() {
                *sizes.entry(*c).or_default() += 1;
            }
            let shared = sizes
                .iter()
                .filter(|(_, &n)| n > 1)
                .min_by_key(|(&c, &n)| (n, c))
                .map(|(&c, _)| c);
            let Some(shared) = shared else {
                let mut order = nodes.clone();
                order.sort_by_key(|i| colors[i]);
                out.push(order);
                continue;
            };
            for &i in nodes.iter().filter(|i| colors[i] == shared) {
                let mut next = colors.clone();
                next.insert(i, mix(shared, 0x5eed_0fac_1a55));
                refine(&mut next);
                open.push(next);
            }
        }
        if out.is_empty() {
            out.push(nodes);
        }
        out
    }

    /// The canonical string of the molecule `atoms` with the charge `charge`.
    fn molecule(&self, atoms: &[usize], charge: i32) -> String {
        let mut best: Option<String> = None;
        for order in self.orderings(atoms) {
            let place: BTreeMap<usize, usize> =
                order.iter().enumerate().map(|(k, &i)| (i, k)).collect();
            // A hydrogen written with its heavy atom comes after every written atom.
            let rank = |i: usize| place.get(&i).copied().unwrap_or(order.len() + i);
            let a = self.assign(atoms, &rank, charge);
            let text = self.write(atoms, &rank, &a);
            if best.as_ref().is_none_or(|b| text < *b) {
                best = Some(text);
            }
        }
        best.unwrap_or_default()
    }
}

/// A number for the handedness of `system`: 0 when its string has no mark of a tetrahedral
/// centre (the structure and its mirror image then have one string), and a hash of the string
/// otherwise, which two enantiomers do not share (nor two diastereomers).
pub fn handedness(system: &MolecularSystem, bond_scale: f64) -> u64 {
    let text = smiles(system, bond_scale);
    if !text.contains('@') {
        return 0;
    }
    text.bytes()
        .fold(0x6a09_e667_f3bc_c908, |h, b| mix(h, u64::from(b)))
        .max(1)
}

/// A number for the configurations of the double bonds of `system`: 0 when its string has no
/// mark of a double bond, and otherwise a hash of the string written without the marks of
/// the tetrahedral centres, which the E and Z isomers do not share and which a structure and
/// its mirror image do.
pub fn double_bond_configuration(system: &MolecularSystem, bond_scale: f64) -> u64 {
    let text = smiles_with(system, bond_scale, false, true);
    if !text.contains(['/', '\\']) {
        return 0;
    }
    text.bytes()
        .fold(0xbb67_ae85_84ca_a73b, |h, b| mix(h, u64::from(b)))
        .max(1)
}

/// The SMILES string of `system` with the bonds of `bond_scale` (see the module text): its
/// molecules in the order of their strings, joined by `.`.
pub fn smiles(system: &MolecularSystem, bond_scale: f64) -> String {
    smiles_with(system, bond_scale, true, true)
}

/// [`smiles`] with or without the marks of the tetrahedral centres and of the double bonds.
fn smiles_with(
    system: &MolecularSystem,
    bond_scale: f64,
    tetrahedral: bool,
    double_bonds: bool,
) -> String {
    let bonds = BondGraph::of(system, bond_scale);
    let n = system.n_atoms();
    let implicit: Vec<bool> = (0..n)
        .map(|i| {
            system.elements[i].atomic_number() == 1
                && bonds.adjacency[i].len() == 1
                && system.elements[bonds.adjacency[i][0]].atomic_number() != 1
        })
        .collect();
    let graph = Graph {
        system,
        bonds: &bonds,
        implicit,
        tetrahedral,
        double_bonds,
    };
    // The charge of every molecule: the usual ions of each as they are, the rest of the charge
    // of the structure to the molecules with the atoms that take it first.
    let by_index = |i: usize| i;
    let total = system.spin_charge.charge;
    let mut charges: Vec<i32> = Vec::with_capacity(bonds.molecules.len());
    let mut sites: Vec<(usize, u8)> = Vec::new();
    let mut left = total;
    for (m, atoms) in bonds.molecules.iter().enumerate() {
        let a = graph.assign(atoms, &by_index, 0);
        let own: i32 = a.charges.values().sum();
        charges.push(own);
        left -= own;
        for (i, lacks) in a.lacking {
            for _ in 0..lacks {
                sites.push((m, graph.z(i)));
            }
        }
    }
    while left != 0 {
        let best = sites
            .iter()
            .enumerate()
            .min_by_key(|(_, &(m, z))| {
                let by = if left < 0 {
                    anion_rank(z)
                } else {
                    cation_rank(z)
                };
                (by, bonds.molecule_colors[m], m)
            })
            .map(|(k, _)| k);
        let Some(k) = best else {
            break;
        };
        let (m, _) = sites.remove(k);
        charges[m] += left.signum();
        left -= left.signum();
    }
    let mut parts: Vec<String> = bonds
        .molecules
        .iter()
        .zip(&charges)
        .map(|(atoms, &q)| graph.molecule(atoms, q))
        .collect();
    parts.sort();
    parts.join(".")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};
    use crate::workflow::explore::identity::DEFAULT_BOND_SCALE;

    /// A structure of `atoms` (symbol, x, y, z in Angstrom) with the charge `charge`.
    fn system(atoms: &[(&str, f64, f64, f64)], charge: i32) -> MolecularSystem {
        let elements: Vec<Element> = atoms
            .iter()
            .map(|a| Element::parse_token(a.0).unwrap())
            .collect();
        let coords: Vec<[f64; 3]> = atoms.iter().map(|a| [a.1, a.2, a.3]).collect();
        let electrons: i64 = elements
            .iter()
            .map(|e| i64::from(e.atomic_number()))
            .sum::<i64>()
            - i64::from(charge);
        let multiplicity = if electrons % 2 == 0 { 1 } else { 2 };
        MolecularSystem::new(
            elements,
            &positions::<Angstrom>(&coords),
            SpinCharge::checked(i64::from(charge), multiplicity).unwrap(),
            None,
        )
        .unwrap()
    }

    fn text(atoms: &[(&str, f64, f64, f64)], charge: i32) -> String {
        smiles(&system(atoms, charge), DEFAULT_BOND_SCALE)
    }

    /// The same atoms in another order.
    fn shuffled<'a>(
        atoms: &[(&'a str, f64, f64, f64)],
        seed: usize,
    ) -> Vec<(&'a str, f64, f64, f64)> {
        let n = atoms.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut state = seed as u64 * 2_654_435_761 + 1;
        for k in (1..n).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            order.swap(k, (state >> 33) as usize % (k + 1));
        }
        order.iter().map(|&i| atoms[i]).collect()
    }

    const WATER: [(&str, f64, f64, f64); 3] = [
        ("O", 0.0, 0.0, 0.0),
        ("H", 0.96, 0.0, 0.0),
        ("H", -0.24, 0.93, 0.0),
    ];
    const FORMALDEHYDE: [(&str, f64, f64, f64); 4] = [
        ("C", 0.0, 0.0, 0.0),
        ("O", 1.21, 0.0, 0.0),
        ("H", -0.55, 0.94, 0.0),
        ("H", -0.55, -0.94, 0.0),
    ];
    const HCN: [(&str, f64, f64, f64); 3] = [
        ("H", -1.07, 0.0, 0.0),
        ("C", 0.0, 0.0, 0.0),
        ("N", 1.16, 0.0, 0.0),
    ];

    /// Benzene in the plane.
    fn benzene() -> Vec<(&'static str, f64, f64, f64)> {
        let mut atoms = Vec::new();
        for k in 0..6 {
            let a = std::f64::consts::PI / 3.0 * k as f64;
            atoms.push(("C", 1.39 * a.cos(), 1.39 * a.sin(), 0.0));
        }
        for k in 0..6 {
            let a = std::f64::consts::PI / 3.0 * k as f64;
            atoms.push(("H", 2.48 * a.cos(), 2.48 * a.sin(), 0.0));
        }
        atoms
    }

    #[test]
    fn small_molecules_have_their_usual_strings() {
        assert_eq!(text(&WATER, 0), "O");
        assert_eq!(text(&FORMALDEHYDE, 0), "C=O");
        assert_eq!(text(&HCN, 0), "C#N");
        assert_eq!(
            text(&[("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)], 0),
            "[H][H]"
        );
        assert_eq!(
            text(&[("Ar", 0.0, 0.0, 0.0), ("Ar", 3.8, 0.0, 0.0)], 0),
            "[Ar].[Ar]"
        );
        // Carbon dioxide: both bonds double.
        assert_eq!(
            text(
                &[
                    ("O", -1.16, 0.0, 0.0),
                    ("C", 0.0, 0.0, 0.0),
                    ("O", 1.16, 0.0, 0.0)
                ],
                0
            ),
            "O=C=O"
        );
    }

    #[test]
    fn the_string_does_not_depend_on_the_numbering() {
        let ring = benzene();
        let first = text(&ring, 0);
        // A Kekule form of the ring: three double bonds, one ring closure.
        assert_eq!(first.matches('=').count(), 3, "{first}");
        assert_eq!(first.matches('C').count(), 6, "{first}");
        assert!(first.contains('1'), "{first}");
        for seed in 1..20 {
            assert_eq!(text(&shuffled(&ring, seed), 0), first, "seed {seed}");
        }
        for seed in 1..6 {
            assert_eq!(text(&shuffled(&FORMALDEHYDE, seed), 0), "C=O");
            assert_eq!(text(&shuffled(&HCN, seed), 0), "C#N");
        }
    }

    #[test]
    fn conformers_share_a_string_and_isomers_do_not() {
        // Hydrogen peroxide at two dihedral angles.
        let peroxide = |turn: f64| {
            vec![
                ("O", 0.0, 0.0, 0.0),
                ("O", 1.45, 0.0, 0.0),
                ("H", -0.3, 0.9, 0.0),
                ("H", 1.75, 0.9 * turn.cos(), 0.9 * turn.sin()),
            ]
        };
        assert_eq!(text(&peroxide(0.3), 0), "OO");
        assert_eq!(text(&peroxide(2.0), 0), text(&peroxide(0.3), 0));
        // HCN and HNC.
        let hnc = [
            ("H", -1.0, 0.0, 0.0),
            ("N", 0.0, 0.0, 0.0),
            ("C", 1.17, 0.0, 0.0),
        ];
        assert_ne!(text(&hnc, 0), text(&HCN, 0));
    }

    /// Bromochlorofluoromethane: looking from the fluorine, the hydrogen, the chlorine and
    /// the bromine follow anticlockwise (`F[C@H](Cl)Br`, the R enantiomer).
    const CHIRAL: [(&str, f64, f64, f64); 5] = [
        ("C", 0.0, 0.0, 0.0),
        ("F", 0.0, 0.0, 1.33),
        ("H", 1.03, 0.0, -0.36),
        ("Cl", -0.84, 1.46, -0.59),
        ("Br", -0.91, -1.58, -0.64),
    ];

    #[test]
    fn the_mark_of_a_tetrahedral_centre_follows_the_rule_of_the_string() {
        // The first neighbour towards the viewer, the other three below it at 0, 120 and 240
        // degrees: anticlockwise as seen from the first.
        let h = 1.0 / 3.0;
        let (s, c) = (0.75_f64.sqrt(), -0.5);
        let around = [[0.0, 0.0, 1.0], [1.0, 0.0, -h], [c, s, -h], [c, -s, -h]];
        assert_eq!(tetrahedral_mark([0.0; 3], around), Some("@"));
        // Two of them exchanged: clockwise. L-alanine, `N[C@@H](C)C(=O)O`, is this
        // arrangement of N, H, CH3 and COOH.
        let other = [around[0], around[1], around[3], around[2]];
        assert_eq!(tetrahedral_mark([0.0; 3], other), Some("@@"));
        // Four neighbours in a plane have no handedness.
        let flat = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        assert_eq!(tetrahedral_mark([0.0; 3], flat), None);
    }

    #[test]
    fn enantiomers_have_different_strings() {
        let right = text(&CHIRAL, 0);
        assert!(right.contains('@'), "{right}");
        // The mirror image is the other enantiomer: another string, the two marks exchanged.
        let mirror: Vec<(&str, f64, f64, f64)> =
            CHIRAL.iter().map(|a| (a.0, -a.1, a.2, a.3)).collect();
        let left = text(&mirror, 0);
        assert_ne!(left, right);
        assert_eq!(left.contains("@@"), !right.contains("@@"), "{left} {right}");
        // Neither depends on the numbering or on how the molecule is turned.
        for seed in 1..12 {
            assert_eq!(text(&shuffled(&CHIRAL, seed), 0), right, "seed {seed}");
            assert_eq!(text(&shuffled(&mirror, seed), 0), left, "seed {seed}");
        }
        let turned: Vec<(&str, f64, f64, f64)> =
            CHIRAL.iter().map(|a| (a.0, a.3, a.1, a.2)).collect();
        assert_eq!(text(&turned, 0), right);
        // A carbon with two hydrogens is no centre, and its mirror image has its string.
        let plain = [
            ("C", 0.0, 0.0, 0.0),
            ("F", 0.0, 0.0, 1.33),
            ("H", 1.03, 0.0, -0.36),
            ("Cl", -0.84, 1.46, -0.59),
            ("H", -0.51, -0.89, -0.36),
        ];
        let flipped: Vec<(&str, f64, f64, f64)> =
            plain.iter().map(|a| (a.0, -a.1, a.2, a.3)).collect();
        assert!(!text(&plain, 0).contains('@'));
        assert_eq!(text(&flipped, 0), text(&plain, 0));
    }

    /// Chloride beside (S)-1-chloro-1-fluoroethane, and the complex after the chloride has
    /// taken the place of the chlorine from behind (the identity SN2 reaction, with the
    /// inversion of the carbon): the two minima of the built-in GFN1-xTB, every atom with its
    /// number on both sides (the last atom is the chloride before, the one before it after).
    const BEFORE_THE_SUBSTITUTION: [(&str, f64, f64, f64); 9] = [
        ("C", 0.4588, 0.1734, 0.6082),
        ("H", -0.2795, 0.8939, 0.9726),
        ("F", 0.7406, -0.6881, 1.6278),
        ("C", -0.0336, -0.6066, -0.5815),
        ("H", 0.7246, -1.3108, -0.9233),
        ("H", -0.2882, 0.0853, -1.3808),
        ("H", -0.9343, -1.1464, -0.2920),
        ("Cl", 1.9731, 1.0402, 0.2311),
        ("Cl", -2.3614, 1.5593, -0.2620),
    ];
    const AFTER_THE_SUBSTITUTION: [(&str, f64, f64, f64); 9] = [
        ("C", -0.4588, 0.1734, 0.6081),
        ("H", 0.2793, 0.8941, 0.9724),
        ("F", -0.7405, -0.6880, 1.6279),
        ("C", 0.0338, -0.6068, -0.5814),
        ("H", -0.7243, -1.3111, -0.9232),
        ("H", 0.2884, 0.0850, -1.3808),
        ("H", 0.9344, -1.1464, -0.2916),
        ("Cl", 2.3610, 1.5599, -0.2622),
        ("Cl", -1.9732, 1.0399, 0.2308),
    ];

    #[test]
    fn a_substitution_with_inversion_gives_the_other_enantiomer() {
        // Before: the S molecule (from the hydrogen, Cl, F and CH3 follow clockwise, so with
        // the hydrogen behind they follow anticlockwise), and the chloride on its own.
        let before = text(&BEFORE_THE_SUBSTITUTION, -1);
        assert_eq!(before, "C[C@@H](F)Cl.[Cl-]");
        // After: the same atoms and bonds of the graph, the other handedness (R).
        let after = text(&AFTER_THE_SUBSTITUTION, -1);
        assert_eq!(after, "C[C@H](F)Cl.[Cl-]");
        // The product of the identity reaction is the mirror image of the reactant.
        let mirror: Vec<(&str, f64, f64, f64)> = BEFORE_THE_SUBSTITUTION
            .iter()
            .map(|a| (a.0, -a.1, a.2, a.3))
            .collect();
        assert_eq!(text(&mirror, -1), after);
        // Neither string depends on the numbering, and the two are two kinds of fragment.
        for seed in 1..10 {
            assert_eq!(
                text(&shuffled(&BEFORE_THE_SUBSTITUTION, seed), -1),
                before,
                "seed {seed}"
            );
            assert_eq!(
                text(&shuffled(&AFTER_THE_SUBSTITUTION, seed), -1),
                after,
                "seed {seed}"
            );
        }
        let hand =
            |atoms: &[(&str, f64, f64, f64)]| handedness(&system(atoms, -1), DEFAULT_BOND_SCALE);
        assert_ne!(hand(&BEFORE_THE_SUBSTITUTION), 0);
        assert_ne!(
            hand(&BEFORE_THE_SUBSTITUTION),
            hand(&AFTER_THE_SUBSTITUTION)
        );
        assert_eq!(hand(&mirror), hand(&AFTER_THE_SUBSTITUTION));
        // Without a centre the number is 0, for the structure and its mirror image.
        assert_eq!(handedness(&system(&WATER, 0), DEFAULT_BOND_SCALE), 0);
    }

    /// A substituted ethane: the first carbon with hydrogen, fluorine and chlorine at 0, 120
    /// and 240 degrees about the C-C bond, the second carbon with `second` at the same
    /// angles turned by `twist` (radians), both pointing away from the bond.
    fn ethane(second: [&'static str; 3], twist: f64) -> Vec<(&'static str, f64, f64, f64)> {
        let length = |s: &str| match s {
            "H" => 1.09,
            "F" => 1.38,
            "Cl" => 1.77,
            _ => 1.94,
        };
        let (t, r) = (1.0 / 3.0, (8.0_f64 / 9.0).sqrt());
        let mut atoms = vec![("C", 0.0, 0.0, 0.0), ("C", 1.54, 0.0, 0.0)];
        for (k, s) in ["H", "F", "Cl"].into_iter().enumerate() {
            let (a, l) = (2.0 * std::f64::consts::PI * k as f64 / 3.0, length(s));
            atoms.push((s, -t * l, r * l * a.cos(), r * l * a.sin()));
        }
        for (k, s) in second.into_iter().enumerate() {
            let (a, l) = (
                2.0 * std::f64::consts::PI * k as f64 / 3.0 + twist,
                length(s),
            );
            atoms.push((s, 1.54 + t * l, r * l * a.cos(), r * l * a.sin()));
        }
        atoms
    }

    /// The mirror image (through the plane y = 0).
    fn mirrored<'a>(atoms: &[(&'a str, f64, f64, f64)]) -> Vec<(&'a str, f64, f64, f64)> {
        atoms.iter().map(|a| (a.0, a.1, -a.2, a.3)).collect()
    }

    #[test]
    fn two_centres_with_the_same_substituents_have_a_meso_form() {
        // 1,2-dichloro-1,2-difluoroethane, CHFCl-CHFCl. With the same order of H, F and Cl
        // about the bond on both carbons, one carbon is the mirror image of the other (each
        // is seen from the other end): the meso form. With F and Cl exchanged on the second
        // carbon, both carbons have one handedness: one of the two enantiomers.
        let staggered = std::f64::consts::PI / 3.0;
        let meso = ethane(["H", "F", "Cl"], staggered);
        let chiral = ethane(["H", "Cl", "F"], staggered);
        let (m, c) = (text(&meso, 0), text(&chiral, 0));
        let other = text(&mirrored(&chiral), 0);
        assert_eq!(
            [m.as_str(), c.as_str(), other.as_str()],
            [
                "[C@@H](Cl)([C@H](Cl)F)F",
                "[C@@H](Cl)([C@@H](Cl)F)F",
                "[C@H](Cl)([C@H](Cl)F)F"
            ]
        );
        // Two marks in each string.
        for s in [&m, &c, &other] {
            assert_eq!(s.matches('@').count() - s.matches("@@").count(), 2, "{s}");
        }
        // The meso form is its own mirror image; the other two are each other's.
        assert_eq!(text(&mirrored(&meso), 0), m);
        assert_ne!(c, other);
        assert_eq!(text(&mirrored(&mirrored(&chiral)), 0), c);
        assert_ne!(m, c);
        assert_ne!(m, other);
        // Conformers (the other two staggered forms) and other numberings keep the strings.
        for turn in [3.0, 5.0] {
            assert_eq!(text(&ethane(["H", "F", "Cl"], turn * staggered), 0), m);
            assert_eq!(text(&ethane(["H", "Cl", "F"], turn * staggered), 0), c);
        }
        for seed in 1..12 {
            assert_eq!(text(&shuffled(&meso, seed), 0), m, "seed {seed}");
            assert_eq!(text(&shuffled(&chiral, seed), 0), c, "seed {seed}");
            assert_eq!(
                text(&shuffled(&mirrored(&chiral), seed), 0),
                other,
                "seed {seed}"
            );
        }
        // As fragments: the meso form and its mirror image are one, the enantiomers two.
        let hand =
            |atoms: &[(&str, f64, f64, f64)]| handedness(&system(atoms, 0), DEFAULT_BOND_SCALE);
        assert_eq!(hand(&meso), hand(&mirrored(&meso)));
        assert_ne!(hand(&chiral), hand(&mirrored(&chiral)));
        assert_ne!(hand(&meso), hand(&chiral));
    }

    #[test]
    fn two_different_centres_give_four_stereoisomers() {
        // 1-bromo-2-chloro-1,2-difluoroethane, CHFCl-CHFBr: no meso form; two pairs of
        // enantiomers, and the members of different pairs are diastereomers.
        let staggered = std::f64::consts::PI / 3.0;
        let first = ethane(["H", "F", "Br"], staggered);
        let second = ethane(["H", "Br", "F"], staggered);
        let all = [
            text(&first, 0),
            text(&mirrored(&first), 0),
            text(&second, 0),
            text(&mirrored(&second), 0),
        ];
        assert_eq!(
            all,
            [
                "[C@@H](F)([C@@H](Cl)F)Br",
                "[C@H](F)([C@H](Cl)F)Br",
                "[C@H](F)([C@@H](Cl)F)Br",
                "[C@@H](F)([C@H](Cl)F)Br"
            ]
        );
        for (i, a) in all.iter().enumerate() {
            assert_eq!(a.matches('@').count() - a.matches("@@").count(), 2, "{a}");
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
        for seed in 1..8 {
            assert_eq!(text(&shuffled(&first, seed), 0), all[0], "seed {seed}");
            assert_eq!(text(&shuffled(&second, seed), 0), all[2], "seed {seed}");
        }
        // The marks of a structure and of its mirror image are exchanged, centre by centre.
        let exchanged = |s: &str| s.replace("@@", "#").replace('@', "@@").replace('#', "@");
        assert_eq!(exchanged(&all[0]), all[1]);
        assert_eq!(exchanged(&all[2]), all[3]);
    }

    /// 1,2-dichlorocyclopropane: the chlorines on one side of the ring (`cis`) or on two.
    fn dichlorocyclopropane(cis: bool) -> Vec<(&'static str, f64, f64, f64)> {
        let radius = 0.872;
        let mut atoms = Vec::new();
        let mut hang = Vec::new();
        for k in 0..3 {
            let a = std::f64::consts::PI / 2.0 + 2.0 * std::f64::consts::PI * k as f64 / 3.0;
            let (x, y) = (radius * a.cos(), radius * a.sin());
            atoms.push(("C", x, y, 0.0));
            // Up and down: out of the ring and away from its middle.
            let up = match k {
                0 => "Cl",
                1 if cis => "Cl",
                _ => "H",
            };
            let down = match k {
                1 if !cis => "Cl",
                _ => "H",
            };
            for (s, side) in [(up, 1.0), (down, -1.0)] {
                let l = if s == "Cl" { 1.76 } else { 1.09 };
                hang.push((
                    s,
                    x + 0.54 * l * a.cos(),
                    y + 0.54 * l * a.sin(),
                    0.84 * l * side,
                ));
            }
        }
        atoms.extend(hang);
        atoms
    }

    #[test]
    fn a_ring_with_two_centres_has_a_meso_form_and_a_pair_of_enantiomers() {
        let (cis, trans) = (dichlorocyclopropane(true), dichlorocyclopropane(false));
        let (c, t) = (text(&cis, 0), text(&trans, 0));
        // cis has a mirror plane (meso): one string with its mirror image. trans has none.
        assert_eq!(text(&mirrored(&cis), 0), c);
        let other = text(&mirrored(&trans), 0);
        assert_eq!(
            [c.as_str(), t.as_str(), other.as_str()],
            [
                "C1[C@@H](Cl)[C@H]1Cl",
                "C1[C@H](Cl)[C@H]1Cl",
                "C1[C@@H](Cl)[C@@H]1Cl"
            ]
        );
        assert_ne!(other, t);
        assert_ne!(c, t);
        assert_ne!(c, other);
        for s in [&c, &t, &other] {
            assert_eq!(s.matches('@').count() - s.matches("@@").count(), 2, "{s}");
        }
        for seed in 1..12 {
            assert_eq!(text(&shuffled(&cis, seed), 0), c, "seed {seed}");
            assert_eq!(text(&shuffled(&trans, seed), 0), t, "seed {seed}");
            assert_eq!(
                text(&shuffled(&mirrored(&trans), seed), 0),
                other,
                "seed {seed}"
            );
        }
    }

    #[test]
    fn ions_and_separate_molecules() {
        // Chloride and bromomethane: the charge on the atom that lacks a bond.
        let sn2 = [
            ("Cl", -3.5, 0.0, 0.0),
            ("C", 0.0, 0.0, 0.0),
            ("Br", 1.95, 0.0, 0.0),
            ("H", -0.35, 1.03, 0.0),
            ("H", -0.35, -0.51, 0.89),
            ("H", -0.35, -0.51, -0.89),
        ];
        assert_eq!(text(&sn2, -1), "BrC.[Cl-]");
        // Ammonium: four bonds of nitrogen.
        let ammonium = [
            ("N", 0.0, 0.0, 0.0),
            ("H", 0.59, 0.59, 0.59),
            ("H", -0.59, -0.59, 0.59),
            ("H", -0.59, 0.59, -0.59),
            ("H", 0.59, -0.59, -0.59),
        ];
        assert_eq!(text(&ammonium, 1), "[NH4+]");
        // A methyl radical: the carbon lacks a bond and no charge is left for it.
        let methyl = [
            ("C", 0.0, 0.0, 0.0),
            ("H", 1.08, 0.0, 0.0),
            ("H", -0.54, 0.935, 0.0),
            ("H", -0.54, -0.935, 0.0),
        ];
        assert_eq!(text(&methyl, 0), "[CH3]");
    }

    /// 1,2-difluoroethene with the fluorines on opposite sides (E) or on the same side (Z).
    fn difluoroethene(same_side: bool) -> Vec<(&'static str, f64, f64, f64)> {
        let y = if same_side { 1.0 } else { -1.0 };
        vec![
            ("C", 0.0, 0.0, 0.0),
            ("C", 1.33, 0.0, 0.0),
            ("F", -0.70, 1.10, 0.0),
            ("H", -0.60, -0.90, 0.0),
            ("F", 2.03, 1.10 * y, 0.0),
            ("H", 1.93, -0.90 * y, 0.0),
        ]
    }

    /// N-methylethanimine CH3-CH=N-CH3: the two methyl groups on opposite sides (E) or on
    /// the same side (Z) of the double bond; the nitrogen has one neighbour besides it.
    fn imine(same_side: bool) -> Vec<(&'static str, f64, f64, f64)> {
        let y = if same_side { 1.0 } else { -1.0 };
        let mut atoms = vec![
            ("C", 0.0, 0.0, 0.0),
            ("N", 1.28, 0.0, 0.0),
            ("C", -0.80, 1.25, 0.0),
            ("H", -0.55, -0.93, 0.0),
            ("C", 2.05, 1.22 * y, 0.0),
        ];
        // The hydrogens of the two methyl groups.
        for (x, y0, side) in [(-0.80, 1.25, 1.0), (2.05, 1.22 * y, y)] {
            atoms.push(("H", x, y0 + 1.09 * side, 0.0));
            atoms.push(("H", x - 0.6, y0 - 0.3 * side, 0.89));
            atoms.push(("H", x - 0.6, y0 - 0.3 * side, -0.89));
        }
        atoms
    }

    #[test]
    fn e_and_z_isomers_have_different_strings() {
        let e = text(&difluoroethene(false), 0);
        let z = text(&difluoroethene(true), 0);
        eprintln!("difluoroethene: E {e}, Z {z}");
        assert_ne!(e, z);
        assert!(
            e.contains(['/', '\\']) && z.contains(['/', '\\']),
            "{e} {z}"
        );
        // Whatever the numbering of the atoms, and for the mirror image too (a double bond
        // has no handedness).
        for seed in 0..8 {
            assert_eq!(
                text(&shuffled(&difluoroethene(false), seed), 0),
                e,
                "{seed}"
            );
            assert_eq!(text(&shuffled(&difluoroethene(true), seed), 0), z, "{seed}");
        }
        assert_eq!(text(&mirrored(&difluoroethene(false)), 0), e);
        assert_eq!(text(&mirrored(&difluoroethene(true)), 0), z);
        // An atom of the double bond with one other neighbour: an imine.
        let (ie, iz) = (text(&imine(false), 0), text(&imine(true), 0));
        eprintln!("imine: E {ie}, Z {iz}");
        assert_ne!(ie, iz);
        for seed in 0..8 {
            assert_eq!(text(&shuffled(&imine(false), seed), 0), ie, "{seed}");
            assert_eq!(text(&shuffled(&imine(true), seed), 0), iz, "{seed}");
        }
        // The number of the configurations: two for the two isomers, the same for a structure
        // and its mirror image, and no handedness in it.
        let sides = |atoms: &[(&str, f64, f64, f64)]| {
            double_bond_configuration(&system(atoms, 0), DEFAULT_BOND_SCALE)
        };
        let (se, sz) = (sides(&difluoroethene(false)), sides(&difluoroethene(true)));
        assert!(se != 0 && sz != 0 && se != sz);
        assert_eq!(sides(&mirrored(&difluoroethene(false))), se);
        assert_eq!(sides(&shuffled(&difluoroethene(true), 3)), sz);
        assert_eq!(sides(&CHIRAL), 0);
    }

    #[test]
    fn double_bonds_without_two_configurations_have_no_mark() {
        // Two equal neighbours at one end: 1,1-difluoroethene, and formaldehyde.
        let geminal = [
            ("C", 0.0, 0.0, 0.0),
            ("C", 1.33, 0.0, 0.0),
            ("F", -0.70, 1.10, 0.0),
            ("F", -0.70, -1.10, 0.0),
            ("H", 1.90, 0.93, 0.0),
            ("H", 1.90, -0.93, 0.0),
        ];
        for atoms in [&geminal[..], &FORMALDEHYDE[..]] {
            let t = text(atoms, 0);
            assert!(!t.contains(['/', '\\']), "{t}");
            assert_eq!(
                double_bond_configuration(&system(atoms, 0), DEFAULT_BOND_SCALE),
                0
            );
        }
        // A double bond twisted to a right angle has none either.
        let twisted = [
            ("C", 0.0, 0.0, 0.0),
            ("C", 1.33, 0.0, 0.0),
            ("F", -0.70, 1.10, 0.0),
            ("H", -0.60, -0.90, 0.0),
            ("F", 2.03, 0.0, 1.10),
            ("H", 1.93, 0.0, -0.90),
        ];
        let t = text(&twisted, 0);
        assert!(!t.contains(['/', '\\']), "{t}");
        // A double bond in a ring: cyclopropene with a fluorine on each of its carbons.
        let ring = [
            ("C", 0.0, 0.65, 0.0),
            ("C", 0.0, -0.65, 0.0),
            ("C", 1.36, 0.0, 0.0),
            ("F", -0.80, 1.70, 0.0),
            ("F", -0.80, -1.70, 0.0),
            ("H", 1.95, 0.0, 0.91),
            ("H", 1.95, 0.0, -0.91),
        ];
        let t = text(&ring, 0);
        assert!(!t.contains(['/', '\\']), "{t}");
    }
}
