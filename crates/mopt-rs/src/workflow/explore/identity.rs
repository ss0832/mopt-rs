//! Whether two structures are the same local minimum, with equivalent atoms and identical
//! molecules free to exchange places (the duplicate removal of exploration workflows).
//!
//! Two structures are compared in three steps:
//!
//! 1. Their bond graphs (bonds within `bond_scale` times the sum of the covalent radii) must be
//!    the same up to a permutation that keeps the elements. Every atom gets a colour by
//!    Weisfeiler-Lehman refinement of its element over its bonds (Weisfeiler and Leman 1968,
//!    in the form of Shervashidze et al., J. Mach. Learn. Res. 2011, 12, 2539), every molecule
//!    (connected component) the multiset of the colours of its atoms; both structures must
//!    have the same colours of molecules.
//! 2. Atoms are matched. Identical molecules (molecules of one colour) are assigned to each
//!    other by the Hungarian method ([`crate::math::assignment`]) with the cost of the best
//!    matching of their atoms, and within each pair the atoms of one colour by the Hungarian
//!    method on squared distances. The superposition comes from the 24 proper rotations that
//!    map the principal axes of one structure onto those of the other (with turns about an
//!    axis whose moments are degenerate), each refined by alternating Kabsch fits and
//!    assignments until the assignment stays (the scheme of ArbAlign, Temelso et al., J. Chem.
//!    Inf. Model. 2017, 57, 1045, with molecules matched as units first). A matching must map
//!    bonds onto bonds.
//! 3. The RMSD of the best matching decides, together with the difference of the energies.
//!
//! Mirror images (`mirror_images`) count as the same structure when asked: an arrangement of
//! achiral molecules and its mirror image have the same energy and the same properties other
//! than optical ones. In a periodic cell rotations are not symmetries: the structures are
//! matched after the removal of their mean drift, with minimum-image distances, under every
//! point operation of the lattice (its rotations, and its reflections with `mirror_images`),
//! each from no translation and from the translations that put an atom of the rarest element
//! onto each atom of that element of the other structure: a crystal moved by a lattice
//! translation of a sublattice, or turned by an operation of its lattice, is the same crystal
//! (a vacancy that hopped to the next site; the hops of a vacancy to its neighbours). Structures
//! in
//! two different cells (a variable cell) are the same only where every component of their
//! lattice vectors agrees within the RMSD tolerance; the atoms of the second are then
//! compared at their fractional coordinates in the cell of the first.
//!
//! MultiOptPy: no counterpart (new in mopt-rs). MultiOptPy's conformation search compares the
//!  energies and the sorted interatomic distances (`Entrypoints/conformation_search.py:
//!  is_identical`), blind to elements and bonds; its orientation search keeps every sample.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::coord::alignment::proper_rotation_from_covariance;
use crate::coord::connectivity::BondConnectivity;
use crate::domain::{Bohr, Hartree, MolecularSystem, PeriodicCell, Quantity, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::assignment::linear_sum_assignment;
use crate::math::linalg::symmetric_eigen;

/// Bond graph of a structure with the colours of its atoms and molecules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BondGraph {
    /// Bonded neighbours of each atom.
    pub adjacency: Vec<Vec<usize>>,
    /// Colour of each atom: equal for atoms that the refinement cannot tell apart.
    pub colors: Vec<u64>,
    /// Molecules (connected components of the graph), atoms ascending.
    pub molecules: Vec<Vec<usize>>,
    /// Colour of each molecule (from the sorted colours of its atoms).
    pub molecule_colors: Vec<u64>,
}

/// Combination of two numbers into a hash (SplitMix64 finalizer): the same on every platform
/// and in every version, unlike the hashers of the standard library.
pub(super) fn mix(a: u64, b: u64) -> u64 {
    let mut z = a.rotate_left(29) ^ b.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn distinct(values: &[u64]) -> usize {
    let mut v = values.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

fn sorted(values: &[u64]) -> Vec<u64> {
    let mut v = values.to_vec();
    v.sort_unstable();
    v
}

impl BondGraph {
    /// The graph of `sys` with bonds within `bond_scale` times the sums of covalent radii
    /// (minimum-image distances in a cell).
    pub fn of(sys: &MolecularSystem, bond_scale: f64) -> Self {
        let conn = BondConnectivity::from_system(sys, bond_scale);
        let n = sys.n_atoms();
        let adjacency: Vec<Vec<usize>> = conn
            .adjacency
            .iter()
            .map(|nbrs| nbrs.iter().map(|a| a.index()).collect())
            .collect();
        let mut colors: Vec<u64> = sys
            .elements
            .iter()
            .map(|e| mix(0, u64::from(e.atomic_number())))
            .collect();
        // Refine until the partition of the atoms stops splitting (at most n rounds).
        let mut classes = distinct(&colors);
        for _ in 0..n {
            let next: Vec<u64> = (0..n)
                .map(|i| {
                    let nbrs = sorted(&adjacency[i].iter().map(|&j| colors[j]).collect::<Vec<_>>());
                    nbrs.iter()
                        .fold(mix(colors[i], nbrs.len() as u64), |h, &c| mix(h, c))
                })
                .collect();
            let count = distinct(&next);
            colors = next;
            if count == classes {
                break;
            }
            classes = count;
        }
        let mut molecules: Vec<Vec<usize>> = conn
            .fragments
            .iter()
            .map(|f| {
                let mut atoms: Vec<usize> = f.iter().map(|a| a.index()).collect();
                atoms.sort_unstable();
                atoms
            })
            .collect();
        molecules.sort();
        let molecule_colors = molecules
            .iter()
            .map(|m| {
                let c = sorted(&m.iter().map(|&i| colors[i]).collect::<Vec<_>>());
                c.iter().fold(mix(1, c.len() as u64), |h, &x| mix(h, x))
            })
            .collect();
        Self {
            adjacency,
            colors,
            molecules,
            molecule_colors,
        }
    }

    /// Whether the graphs can be the same up to a permutation of the atoms.
    pub fn same_as(&self, other: &Self) -> bool {
        self.colors.len() == other.colors.len()
            && sorted(&self.colors) == sorted(&other.colors)
            && sorted(&self.molecule_colors) == sorted(&other.molecule_colors)
    }

    /// This graph with every atom of `held` in a colour of its own (its number mixed into its
    /// colour), so that a matching maps each of them onto itself only: the atoms of hard
    /// constraints, whose numbers the constraints name. The colours of the molecules follow.
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs).
    #[must_use]
    pub fn holding(&self, held: &[usize]) -> Self {
        if held.is_empty() {
            return self.clone();
        }
        let mut out = self.clone();
        for &i in held {
            if let Some(c) = out.colors.get_mut(i) {
                *c = mix(*c, mix(0x6865_6c64, i as u64));
            }
        }
        out.molecule_colors = out
            .molecules
            .iter()
            .map(|m| {
                let c = sorted(&m.iter().map(|&i| out.colors[i]).collect::<Vec<_>>());
                c.iter().fold(mix(1, c.len() as u64), |h, &x| mix(h, x))
            })
            .collect();
        out
    }

    /// Whether atoms `i` and `j` are bonded.
    pub fn bonded(&self, i: usize, j: usize) -> bool {
        self.adjacency.get(i).is_some_and(|n| n.contains(&j))
    }

    /// Whether `permutation` (atom `i` here to atom `permutation[i]` of `other`) maps every
    /// bond onto a bond.
    pub fn keeps_bonds(&self, other: &Self, permutation: &[usize]) -> bool {
        self.adjacency.iter().enumerate().all(|(i, nbrs)| {
            nbrs.iter()
                .all(|&j| other.bonded(permutation[i], permutation[j]))
                && nbrs.len() == other.adjacency[permutation[i]].len()
        })
    }
}

/// The best matching of two structures.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// Atom `i` of the first structure is atom `permutation[i]` of the second.
    pub permutation: Vec<usize>,
    /// RMSD after the superposition.
    pub rmsd: Quantity<Bohr>,
    /// Whether the second structure was mirrored for it.
    pub mirrored: bool,
}

/// When two structures are the same local minimum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructureMatcher {
    /// Largest RMSD after the best matching and superposition.
    pub rmsd_tolerance: Quantity<Bohr>,
    /// Largest difference of the energies.
    pub energy_tolerance: Quantity<Hartree>,
    /// Bonds within this factor times the sum of the covalent radii.
    pub bond_scale: f64,
    /// Whether a structure and its mirror image are the same.
    pub mirror_images: bool,
}

/// Bond scale of the comparisons: 1.2 times the sums of the covalent radii of Pyykko and
/// Atsumi (the scale of `BondConnectivity` elsewhere in mopt-rs is 1.2 to 1.3); hydrogen bonds
/// (H...O 1.8 to 2.0 Angstrom against a limit of 1.16) stay outside it.
pub const DEFAULT_BOND_SCALE: f64 = 1.2;

impl Default for StructureMatcher {
    /// The tolerances measured for orientation searches (see running.md, `orientsearch`):
    /// 0.25 Angstrom and 1e-4 Hartree.
    fn default() -> Self {
        Self {
            rmsd_tolerance: Quantity::new(0.25 * Units::ANGSTROM_TO_BOHR),
            energy_tolerance: Quantity::new(1.0e-4),
            bond_scale: DEFAULT_BOND_SCALE,
            mirror_images: true,
        }
    }
}

/// Alternations of Kabsch fits and assignments per starting rotation.
const REFINEMENTS: usize = 8;
/// Relative difference of two principal moments below which they count as degenerate.
const DEGENERATE: f64 = 0.05;
/// Turns about the unique axis tried when two moments are degenerate.
const TURNS: usize = 12;

type Rot = [[f64; 3]; 3];

fn mat_mul(a: &Rot, b: &Rot) -> Rot {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

fn transpose(a: &Rot) -> Rot {
    std::array::from_fn(|r| std::array::from_fn(|c| a[c][r]))
}

fn det(a: &Rot) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn apply(r: &Rot, p: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2])
}

fn sq(d: [f64; 3]) -> f64 {
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Positions about the unweighted centroid, mirrored through the yz plane when asked.
fn centered(sys: &MolecularSystem, mirrored: bool) -> Vec<[f64; 3]> {
    let c = sys.centroid().value();
    sys.atoms()
        .map(|a| {
            let p = sub(sys.position(a).value(), c);
            if mirrored {
                [-p[0], p[1], p[2]]
            } else {
                p
            }
        })
        .collect()
}

/// Principal axes (columns, moments ascending) of the gyration tensor of centred points.
fn principal_axes(p: &[[f64; 3]]) -> MoptResult<([f64; 3], Rot)> {
    let g = Mat::from_fn(3, 3, |r, c| p.iter().map(|x| x[r] * x[c]).sum::<f64>());
    let (values, vectors) = symmetric_eigen(&g)?;
    Ok((
        std::array::from_fn(|k| values[k]),
        std::array::from_fn(|r| std::array::from_fn(|c| vectors[(r, c)])),
    ))
}

/// The proper rotations of the cube (signed permutation matrices with determinant `sign`).
fn signed_permutations(sign: f64) -> Vec<Rot> {
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut out = Vec::with_capacity(24);
    for order in orders {
        for signs in 0..8_u32 {
            let s: Rot = std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    if c == order[r] {
                        if signs & (1 << r) == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    } else {
                        0.0
                    }
                })
            });
            if det(&s) * sign > 0.0 {
                out.push(s);
            }
        }
    }
    out
}

fn turn_about(axis: usize, angle: f64) -> Rot {
    let (c, s) = (angle.cos(), angle.sin());
    let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut r: Rot = [[0.0; 3]; 3];
    r[axis][axis] = 1.0;
    r[i][i] = c;
    r[i][j] = -s;
    r[j][i] = s;
    r[j][j] = c;
    r
}

/// Rotations that take `pb` onto `pa` by mapping principal frames, with turns about a unique
/// axis when the other two moments are degenerate.
fn frame_rotations(pa: &[[f64; 3]], pb: &[[f64; 3]]) -> MoptResult<Vec<Rot>> {
    let (ma, ea) = principal_axes(pa)?;
    let (_, eb) = principal_axes(pb)?;
    let sign = det(&ea) * det(&eb);
    let scale = ma[2].abs().max(1e-12);
    let mut turns = vec![None];
    for (unique, (x, y)) in [(2, (0, 1)), (0, (1, 2))] {
        if (ma[x] - ma[y]).abs() < DEGENERATE * scale {
            turns.extend(
                (1..TURNS).map(|k| Some((unique, std::f64::consts::TAU * k as f64 / TURNS as f64))),
            );
        }
    }
    let mut out = Vec::new();
    for s in signed_permutations(sign) {
        for turn in &turns {
            let s = match turn {
                Some((axis, angle)) => mat_mul(&turn_about(*axis, *angle), &s),
                None => s,
            };
            out.push(mat_mul(&mat_mul(&ea, &s), &transpose(&eb)));
        }
    }
    Ok(out)
}

/// Kabsch rotation taking `pb[permutation[i]]` onto `pa[i]` (both centred).
fn kabsch(pa: &[[f64; 3]], pb: &[[f64; 3]], permutation: &[usize]) -> Rot {
    let mut cov = Mat::<f64>::zeros(3, 3);
    for (i, &j) in permutation.iter().enumerate() {
        for r in 0..3 {
            for c in 0..3 {
                cov[(r, c)] += pb[j][r] * pa[i][c];
            }
        }
    }
    let rot = proper_rotation_from_covariance(&cov);
    std::array::from_fn(|r| std::array::from_fn(|c| rot[(r, c)]))
}

/// The squared distance of a pair under a superposition.
type Distance<'a> = dyn Fn(usize, usize) -> f64 + 'a;

/// Matching of the atoms of molecule `ma` of `ga` with those of `mb` of `gb` by colours:
/// `(cost, pairs)`, `None` when their colours differ.
fn match_molecule(
    ga: &BondGraph,
    gb: &BondGraph,
    ma: &[usize],
    mb: &[usize],
    distance: &Distance<'_>,
) -> MoptResult<Option<(f64, Vec<(usize, usize)>)>> {
    let mut colors: Vec<u64> = ma.iter().map(|&i| ga.colors[i]).collect();
    colors.sort_unstable();
    colors.dedup();
    let (mut cost, mut pairs) = (0.0, Vec::with_capacity(ma.len()));
    for color in colors {
        let rows: Vec<usize> = ma
            .iter()
            .copied()
            .filter(|&i| ga.colors[i] == color)
            .collect();
        let cols: Vec<usize> = mb
            .iter()
            .copied()
            .filter(|&j| gb.colors[j] == color)
            .collect();
        if rows.len() != cols.len() {
            return Ok(None);
        }
        let m = Mat::from_fn(rows.len(), cols.len(), |r, c| distance(rows[r], cols[c]));
        let assigned = linear_sum_assignment(&m)?;
        for (r, &c) in assigned.iter().enumerate() {
            cost += m[(r, c)];
            pairs.push((rows[r], cols[c]));
        }
    }
    Ok(Some((cost, pairs)))
}

/// Assignment of all atoms (molecules as units first); `None` when the graphs disagree.
fn assign(
    ga: &BondGraph,
    gb: &BondGraph,
    distance: &Distance<'_>,
) -> MoptResult<Option<Vec<usize>>> {
    let n = ga.colors.len();
    let mut permutation = vec![usize::MAX; n];
    let mut kinds: Vec<u64> = ga.molecule_colors.clone();
    kinds.sort_unstable();
    kinds.dedup();
    for kind in kinds {
        let la: Vec<usize> = (0..ga.molecules.len())
            .filter(|&k| ga.molecule_colors[k] == kind)
            .collect();
        let lb: Vec<usize> = (0..gb.molecules.len())
            .filter(|&k| gb.molecule_colors[k] == kind)
            .collect();
        if la.len() != lb.len() {
            return Ok(None);
        }
        let mut table = Vec::with_capacity(la.len() * lb.len());
        for &a in &la {
            for &b in &lb {
                match match_molecule(ga, gb, &ga.molecules[a], &gb.molecules[b], distance)? {
                    Some(found) => table.push(found),
                    None => return Ok(None),
                }
            }
        }
        let cost = Mat::from_fn(la.len(), lb.len(), |r, c| table[r * lb.len() + c].0);
        for (r, &c) in linear_sum_assignment(&cost)?.iter().enumerate() {
            for &(i, j) in &table[r * lb.len() + c].1 {
                permutation[i] = j;
            }
        }
    }
    if permutation.contains(&usize::MAX) {
        return Err(MoptError::calc("structure matching left atoms unassigned"));
    }
    Ok(Some(permutation))
}

fn rmsd_of(distance: &Distance<'_>, permutation: &[usize]) -> f64 {
    let n = permutation.len().max(1) as f64;
    (permutation
        .iter()
        .enumerate()
        .map(|(i, &j)| distance(i, j))
        .sum::<f64>()
        / n)
        .sqrt()
}

impl StructureMatcher {
    /// The best matching of `b` onto `a` (`None` when their bond graphs differ).
    pub fn best_match(
        &self,
        a: &MolecularSystem,
        b: &MolecularSystem,
    ) -> MoptResult<Option<Match>> {
        let (ga, gb) = (
            BondGraph::of(a, self.bond_scale),
            BondGraph::of(b, self.bond_scale),
        );
        self.best_match_with(a, &ga, b, &gb)
    }

    /// [`Self::best_match`] with the bond graphs given (as a pool of minima keeps them).
    pub fn best_match_with(
        &self,
        a: &MolecularSystem,
        ga: &BondGraph,
        b: &MolecularSystem,
        gb: &BondGraph,
    ) -> MoptResult<Option<Match>> {
        if a.elements.len() != b.elements.len() || !ga.same_as(gb) {
            return Ok(None);
        }
        if a.n_atoms() == 0 {
            return Ok(Some(Match {
                permutation: Vec::new(),
                rmsd: Quantity::new(0.0),
                mirrored: false,
            }));
        }
        match (a.active_cell(), b.active_cell()) {
            // Two cells (a variable cell): a basis of the lattice of `a`, turned rigidly,
            // agrees with the lattice vectors of `b` within the RMSD tolerance (a crystal
            // turned as a whole, or with its vectors in another order, has another basis of
            // the same lattice), and the atoms of `b` are compared at their fractional
            // coordinates in that basis, in the cell of `a`.
            (Some(ca), Some(cb)) if lattice_difference(ca, cb) > SAME_LATTICE => {
                let bases = turned_bases(ca, cb, self.rmsd_tolerance.value());
                let mut best: Option<Match> = None;
                for (m, _) in bases.iter().take(BASES_TRIED) {
                    let mut strained = b.clone();
                    for atom in b.atoms() {
                        let f = cb.cart_to_frac(b.position(atom).value());
                        let in_a: [f64; 3] =
                            std::array::from_fn(|j| (0..3).map(|k| f[k] * m[k][j]).sum());
                        strained.set_position(
                            atom,
                            crate::domain::Position::new(ca.frac_to_cart(in_a)),
                        );
                    }
                    strained.cell = Some(ca.clone());
                    if let Some(found) =
                        periodic_match(ca, a, ga, &strained, gb, self.mirror_images)?
                    {
                        if best
                            .as_ref()
                            .is_none_or(|m| found.rmsd.value() < m.rmsd.value())
                        {
                            best = Some(found);
                        }
                    }
                    if best.as_ref().is_some_and(|m| m.rmsd.value() < EXACT_MATCH) {
                        break;
                    }
                }
                Ok(best)
            }
            (Some(cell), _) | (None, Some(cell)) => {
                periodic_match(cell, a, ga, b, gb, self.mirror_images)
            }
            (None, None) => self.isolated_match(a, ga, b, gb),
        }
    }

    fn isolated_match(
        &self,
        a: &MolecularSystem,
        ga: &BondGraph,
        b: &MolecularSystem,
        gb: &BondGraph,
    ) -> MoptResult<Option<Match>> {
        let pa = centered(a, false);
        let mut best: Option<Match> = None;
        let mirrors: &[bool] = if self.mirror_images {
            &[false, true]
        } else {
            &[false]
        };
        for &mirrored in mirrors {
            let pb = centered(b, mirrored);
            let identity: Vec<usize> = (0..pa.len()).collect();
            let mut starts = frame_rotations(&pa, &pb)?;
            starts.push(kabsch(&pa, &pb, &identity));
            for start in starts {
                let Some((permutation, rmsd)) = refine(ga, gb, &pa, &pb, start)? else {
                    continue;
                };
                if best.as_ref().is_none_or(|m| rmsd < m.rmsd.value()) {
                    best = Some(Match {
                        permutation,
                        rmsd: Quantity::new(rmsd),
                        mirrored,
                    });
                }
            }
        }
        Ok(best)
    }

    /// [`Self::best_match_with`] of structures that share a frame (atoms held in space by hard
    /// constraints): the atoms of `held` keep their numbers (each in a colour of its own,
    /// [`BondGraph::holding`]), and with `fixed_frame` an isolated pair is compared where it is,
    /// without superposition and without mirror images (in a cell, as
    /// [`Self::best_match_with`] compares, mirror images left out).
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs).
    pub fn best_match_holding(
        &self,
        a: &MolecularSystem,
        ga: &BondGraph,
        b: &MolecularSystem,
        gb: &BondGraph,
        held: &[usize],
        fixed_frame: bool,
    ) -> MoptResult<Option<Match>> {
        let (ga, gb) = (ga.holding(held), gb.holding(held));
        let unmirrored = Self {
            mirror_images: false,
            ..*self
        };
        if !fixed_frame || a.active_cell().is_some() || b.active_cell().is_some() {
            return unmirrored.best_match_with(a, &ga, b, &gb);
        }
        if a.elements.len() != b.elements.len() || !ga.same_as(&gb) {
            return Ok(None);
        }
        let pa: Vec<[f64; 3]> = a.atoms().map(|i| a.position(i).value()).collect();
        let pb: Vec<[f64; 3]> = b.atoms().map(|i| b.position(i).value()).collect();
        let distance = |i: usize, j: usize| sq(sub(pa[i], pb[j]));
        let Some(permutation) = assign(&ga, &gb, &distance)? else {
            return Ok(None);
        };
        if !ga.keeps_bonds(&gb, &permutation) {
            return Ok(None);
        }
        let rmsd = rmsd_of(&distance, &permutation);
        Ok(Some(Match {
            permutation,
            rmsd: Quantity::new(rmsd),
            mirrored: false,
        }))
    }

    /// Whether `(a, energy_a)` and `(b, energy_b)` are the same minimum.
    pub fn same(
        &self,
        a: &MolecularSystem,
        energy_a: Quantity<Hartree>,
        b: &MolecularSystem,
        energy_b: Quantity<Hartree>,
    ) -> MoptResult<bool> {
        if (energy_a.value() - energy_b.value()).abs() > self.energy_tolerance.value() {
            return Ok(false);
        }
        Ok(self
            .best_match(a, b)?
            .is_some_and(|m| m.rmsd.value() <= self.rmsd_tolerance.value()))
    }
}

/// Alternating assignments and Kabsch fits from the rotation `start`: the permutation and
/// its RMSD, `None` when no assignment keeps the bonds.
fn refine(
    ga: &BondGraph,
    gb: &BondGraph,
    pa: &[[f64; 3]],
    pb: &[[f64; 3]],
    start: Rot,
) -> MoptResult<Option<(Vec<usize>, f64)>> {
    let mut rot = start;
    let mut previous: Option<Vec<usize>> = None;
    for _ in 0..REFINEMENTS {
        let rotated: Vec<[f64; 3]> = pb.iter().map(|&p| apply(&rot, p)).collect();
        let distance = |i: usize, j: usize| sq(sub(pa[i], rotated[j]));
        let Some(permutation) = assign(ga, gb, &distance)? else {
            return Ok(None);
        };
        rot = kabsch(pa, pb, &permutation);
        if previous.as_ref() == Some(&permutation) {
            break;
        }
        previous = Some(permutation);
    }
    let Some(permutation) = previous else {
        return Ok(None);
    };
    if !ga.keeps_bonds(gb, &permutation) {
        return Ok(None);
    }
    let rotated: Vec<[f64; 3]> = pb.iter().map(|&p| apply(&rot, p)).collect();
    let distance = |i: usize, j: usize| sq(sub(pa[i], rotated[j]));
    let rmsd = rmsd_of(&distance, &permutation);
    Ok(Some((permutation, rmsd)))
}

/// Two lattices are the same below this largest difference of their vectors (Bohr).
const SAME_LATTICE: f64 = 1e-10;

/// Bases of the lattice of one cell tried against another cell (the best fits first): bases
/// that differ by an operation of the lattice are tried by [`periodic_match`] itself.
const BASES_TRIED: usize = 4;

/// The bases of the lattice of `ca` whose vectors, turned rigidly, are those of `cb`: integer
/// matrices `M` (entries -1, 0, 1, determinant that of `h_b` over that of `h_a`, so that the
/// turn is proper) with the basis vectors `sum_j M_kj a_j` within `tolerance` (Bohr, the
/// largest difference of a component) of `b_k` after the best rotation (Kabsch, on the three
/// vectors), the best first. A crystal turned as a whole has the identity among them, one with
/// its lattice vectors in another order a permutation. Point `f` (fractional in `cb`) is then
/// at `f M` in the basis of `ca`.
fn turned_bases(ca: &PeriodicCell, cb: &PeriodicCell, tolerance: f64) -> Vec<([[f64; 3]; 3], f64)> {
    let ha: [[f64; 3]; 3] = std::array::from_fn(|k| ca.lattice_vector(k).value());
    let hb: [[f64; 3]; 3] = std::array::from_fn(|k| cb.lattice_vector(k).value());
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let hand = det(&ha).signum() * det(&hb).signum();
    let metric = |h: &[[f64; 3]; 3]| -> [[f64; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|c| h[i][c] * h[j][c]).sum()))
    };
    let gb = metric(&hb);
    let longest = hb.iter().map(|v| sq(*v).sqrt()).fold(0.0_f64, f64::max);
    // A difference of the vectors below the tolerance changes the metric by at most this.
    let metric_bound = 2.0 * tolerance * longest + 3.0 * tolerance * tolerance;
    let mut out = Vec::new();
    for code in 0..19683_u32 {
        let mut m = [[0.0; 3]; 3];
        let mut rest = code;
        for row in &mut m {
            for x in row.iter_mut() {
                *x = f64::from((rest % 3) as i32 - 1);
                rest /= 3;
            }
        }
        if (det(&m) - hand).abs() > 1e-9 {
            continue;
        }
        let basis: [[f64; 3]; 3] = std::array::from_fn(|k| {
            std::array::from_fn(|c| (0..3).map(|j| m[k][j] * ha[j][c]).sum())
        });
        let g = metric(&basis);
        let off = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (g[i][j] - gb[i][j]).abs())
            .fold(0.0_f64, f64::max);
        if off > metric_bound {
            continue;
        }
        // The rotation that takes the basis onto the vectors of `b` (column vectors).
        let mut cov = faer::Mat::<f64>::zeros(3, 3);
        for (p, t) in basis.iter().zip(&hb) {
            for r in 0..3 {
                for c in 0..3 {
                    cov[(r, c)] += p[r] * t[c];
                }
            }
        }
        let rot = &crate::coord::alignment::proper_rotation_from_covariance(&cov);
        let difference = basis
            .iter()
            .zip(&hb)
            .flat_map(|(p, t)| {
                (0..3).map(move |c| ((0..3).map(|r| rot[(c, r)] * p[r]).sum::<f64>() - t[c]).abs())
            })
            .fold(0.0_f64, f64::max);
        if difference <= tolerance {
            out.push((m, difference));
        }
    }
    // Differences far below the tolerance are the noise of an optimized cell: a cubic cell
    // that was optimized is cubic only to its convergence, every operation of its lattice
    // fits about as well as the identity, and the operations of the lattice that
    // `periodic_match` tries are those of the cell within a much tighter tolerance (a
    // tetragonal cell then), which do not undo a wrong basis. Within a hundredth of the
    // tolerance the differences count as equal and the basis nearest the identity comes
    // first, so that two structures of nearly the same cell are compared in the same basis.
    let grain = 0.01 * tolerance;
    let rank = |(m, d): &([[f64; 3]; 3], f64)| -> (u64, usize) {
        let off_identity = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .filter(|&(i, j)| (m[i][j] - if i == j { 1.0 } else { 0.0 }).abs() > 0.5)
            .count();
        let coarse = if grain > 0.0 {
            (d / grain).floor().min(f64::from(u32::MAX)) as u64
        } else {
            0
        };
        (coarse, off_identity)
    };
    out.sort_by(|x, y| rank(x).cmp(&rank(y)).then(x.1.total_cmp(&y.1)));
    out
}

/// The largest difference of a component of the lattice vectors of two cells (Bohr).
fn lattice_difference(a: &PeriodicCell, b: &PeriodicCell) -> f64 {
    (0..3)
        .flat_map(|k| {
            let (u, v) = (a.lattice_vector(k).value(), b.lattice_vector(k).value());
            (0..3).map(move |c| (u[c] - v[c]).abs())
        })
        .fold(0.0, f64::max)
}

/// Translations tried as the start of a matching in a cell, beside none (the best by the cost
/// of [`translation_cost`]).
const TRANSLATIONS_TRIED: usize = 6;

/// The cost of the translation `shift` of `pb` onto `pa`: the sum over the atoms of `a` of the
/// squared minimum-image distance to the nearest atom of `b` of the same element, moved.
fn translation_cost(
    cell: &PeriodicCell,
    a: &MolecularSystem,
    pa: &[[f64; 3]],
    b: &MolecularSystem,
    pb: &[[f64; 3]],
    shift: [f64; 3],
) -> f64 {
    pa.iter()
        .enumerate()
        .map(|(i, p)| {
            pb.iter()
                .enumerate()
                .filter(|(j, _)| b.elements[*j] == a.elements[i])
                .map(|(_, q)| {
                    sq(cell.mic_delta(*p, [q[0] + shift[0], q[1] + shift[1], q[2] + shift[2]]))
                })
                .fold(f64::INFINITY, f64::min)
        })
        .sum()
}

/// Structures of at most this many atoms are compared under the point operations of their
/// lattice as well (each operation costs as many matchings as the translations tried).
const OPERATIONS_LARGEST: usize = 200;

/// A matching below this RMSD (Bohr) ends the search over operations and translations.
const EXACT_MATCH: f64 = 1e-6;

/// The point operations of the lattice of `cell` (periodic in three directions): the
/// orthogonal maps `R` that take every lattice vector to an integer combination of them,
/// `R a_i = sum_j M_ij a_j` with every `M_ij` in -1, 0, 1 (all of a reduced cell), the
/// identity first; with `mirror` the reflections too. Only the identity for a slab or a wire.
fn lattice_operations(cell: &PeriodicCell, mirror: bool) -> Vec<[[f64; 3]; 3]> {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut out = vec![identity];
    if !cell.periodic_axes.iter().all(|p| *p) {
        return out;
    }
    let a: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
    // Columns of `ht` are the lattice vectors; R = ht M^T ht^-1.
    let ht: [[f64; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| a[c][r]));
    let Some(inverse) = inverse3(&ht) else {
        return out;
    };
    for code in 0..19683_u32 {
        let mut m = [[0.0; 3]; 3];
        let mut rest = code;
        for row in &mut m {
            for x in row.iter_mut() {
                *x = f64::from((rest % 3) as i32 - 1);
                rest /= 3;
            }
        }
        // ht M^T
        let hm: [[f64; 3]; 3] = std::array::from_fn(|r| {
            std::array::from_fn(|c| (0..3).map(|k| ht[r][k] * m[c][k]).sum())
        });
        let rot: [[f64; 3]; 3] = std::array::from_fn(|r| {
            std::array::from_fn(|c| (0..3).map(|k| hm[r][k] * inverse[k][c]).sum())
        });
        // Orthogonal: R^T R = I.
        let orthogonal = (0..3).all(|i| {
            (0..3).all(|j| {
                let dot: f64 = (0..3).map(|k| rot[k][i] * rot[k][j]).sum();
                (dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-6
            })
        });
        if !orthogonal {
            continue;
        }
        let det = rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1])
            - rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0])
            + rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0]);
        if det < 0.0 && !mirror {
            continue;
        }
        let same = |x: &[[f64; 3]; 3], y: &[[f64; 3]; 3]| {
            (0..3).all(|i| (0..3).all(|j| (x[i][j] - y[i][j]).abs() < 1e-9))
        };
        if !out.iter().any(|o| same(o, &rot)) {
            out.push(rot);
        }
    }
    out
}

/// The inverse of a 3 x 3 matrix (`None` where it is singular).
fn inverse3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let cof = |r: usize, c: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]
    };
    let det: f64 = (0..3).map(|c| m[0][c] * cof(0, c)).sum();
    if det.abs() < 1e-300 {
        return None;
    }
    Some(std::array::from_fn(|r| {
        std::array::from_fn(|c| cof(c, r) / det)
    }))
}

/// Matching in a cell: assignments with minimum-image distances after the mean drift, under
/// every point operation of the lattice (structures of at most [`OPERATIONS_LARGEST`] atoms;
/// reflections with `mirror`), each from no translation and from the translations that put the
/// first atom of the rarest element of `a` onto each atom of that element of `b`: a crystal
/// moved by a lattice translation of a sublattice, or turned by an operation of its lattice,
/// is one structure (a vacancy that hopped to the next site; the hops to the twelve neighbours
/// of a vacancy in a face-centred cubic crystal). The lowest RMSD of them.
fn periodic_match(
    cell: &PeriodicCell,
    a: &MolecularSystem,
    ga: &BondGraph,
    b: &MolecularSystem,
    gb: &BondGraph,
    mirror: bool,
) -> MoptResult<Option<Match>> {
    let operations = if a.n_atoms() <= OPERATIONS_LARGEST {
        lattice_operations(cell, mirror)
    } else {
        vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    };
    let mut best: Option<Match> = None;
    for op in &operations {
        let pb: Vec<[f64; 3]> = b
            .atoms()
            .map(|i| {
                let p = b.position(i).value();
                std::array::from_fn(|r| (0..3).map(|c| op[r][c] * p[c]).sum())
            })
            .collect();
        let det = op[0][0] * (op[1][1] * op[2][2] - op[1][2] * op[2][1])
            - op[0][1] * (op[1][0] * op[2][2] - op[1][2] * op[2][0])
            + op[0][2] * (op[1][0] * op[2][1] - op[1][1] * op[2][0]);
        if let Some(mut m) = translated_match(cell, a, ga, b, gb, &pb)? {
            m.mirrored = det < 0.0;
            if best
                .as_ref()
                .is_none_or(|b| m.rmsd.value() < b.rmsd.value())
            {
                best = Some(m);
            }
        }
        if best.as_ref().is_some_and(|m| m.rmsd.value() < EXACT_MATCH) {
            break;
        }
    }
    Ok(best)
}

/// [`periodic_match`] for the positions `pb` of the atoms of `b` (turned by an operation):
/// from no translation and from the translations of the rarest element.
fn translated_match(
    cell: &PeriodicCell,
    a: &MolecularSystem,
    ga: &BondGraph,
    b: &MolecularSystem,
    gb: &BondGraph,
    pb: &[[f64; 3]],
) -> MoptResult<Option<Match>> {
    let pa: Vec<[f64; 3]> = a.atoms().map(|i| a.position(i).value()).collect();
    let mut starts = vec![[0.0; 3]];
    // The rarest element of `a` and its first atom.
    let mut counts: Vec<(usize, crate::domain::Element)> = Vec::new();
    for e in &a.elements {
        match counts.iter_mut().find(|(_, x)| x == e) {
            Some(c) => c.0 += 1,
            None => counts.push((1, *e)),
        }
    }
    if let Some(&(_, rare)) = counts.iter().min_by_key(|(n, _)| *n) {
        if let Some(anchor) = a.elements.iter().position(|e| *e == rare) {
            let mut candidates: Vec<(f64, [f64; 3])> = b
                .elements
                .iter()
                .enumerate()
                .filter(|(_, e)| **e == rare)
                .map(|(j, _)| {
                    let shift = cell.mic_delta(pb[j], pa[anchor]);
                    let shift = [-shift[0], -shift[1], -shift[2]];
                    (translation_cost(cell, a, &pa, b, pb, shift), shift)
                })
                .collect();
            candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
            starts.extend(candidates.iter().take(TRANSLATIONS_TRIED).map(|(_, t)| *t));
        }
    }
    let mut best: Option<Match> = None;
    for start in starts {
        if let Some(m) = periodic_match_from(cell, &pa, ga, pb, gb, start)? {
            if best
                .as_ref()
                .is_none_or(|b| m.rmsd.value() < b.rmsd.value())
            {
                best = Some(m);
            }
            if best.as_ref().is_some_and(|m| m.rmsd.value() < EXACT_MATCH) {
                break;
            }
        }
    }
    Ok(best)
}

/// [`periodic_match`] from the translation `start` of the atoms of `b`.
fn periodic_match_from(
    cell: &PeriodicCell,
    pa: &[[f64; 3]],
    ga: &BondGraph,
    pb: &[[f64; 3]],
    gb: &BondGraph,
    start: [f64; 3],
) -> MoptResult<Option<Match>> {
    let mut drift = start;
    let mut previous: Option<Vec<usize>> = None;
    for _ in 0..REFINEMENTS {
        let shift = drift;
        let distance = |i: usize, j: usize| {
            let q = [
                pb[j][0] + shift[0],
                pb[j][1] + shift[1],
                pb[j][2] + shift[2],
            ];
            sq(cell.mic_delta(pa[i], q))
        };
        let Some(permutation) = assign(ga, gb, &distance)? else {
            return Ok(None);
        };
        // The mean minimum-image displacement of the matched atoms, moved.
        let n = permutation.len() as f64;
        let mut mean = [0.0; 3];
        for (i, &j) in permutation.iter().enumerate() {
            let q = [
                pb[j][0] + shift[0],
                pb[j][1] + shift[1],
                pb[j][2] + shift[2],
            ];
            let d = cell.mic_delta(pa[i], q);
            for k in 0..3 {
                mean[k] += d[k] / n;
            }
        }
        // `mic_delta(p, q)` is `p - q` at its nearest image: the drift moves `b` onto `a`.
        drift = [shift[0] + mean[0], shift[1] + mean[1], shift[2] + mean[2]];
        if previous.as_ref() == Some(&permutation) {
            break;
        }
        previous = Some(permutation);
    }
    let Some(permutation) = previous else {
        return Ok(None);
    };
    if !ga.keeps_bonds(gb, &permutation) {
        return Ok(None);
    }
    let distance = |i: usize, j: usize| {
        let q = [
            pb[j][0] + drift[0],
            pb[j][1] + drift[1],
            pb[j][2] + drift[2],
        ];
        sq(cell.mic_delta(pa[i], q))
    };
    Ok(Some(Match {
        rmsd: Quantity::new(rmsd_of(&distance, &permutation)),
        permutation,
        mirrored: false,
    }))
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
