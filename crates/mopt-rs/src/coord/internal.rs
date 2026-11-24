//! Primitive internal coordinates (bonds, angles, linear-bend complements, dihedrals) and their
//! automatic generation from bond connectivity.
//!
//! Every primitive is a function of at most four atoms; its value and Wilson B-matrix row are
//! obtained by per-term automatic differentiation of the generic primitives in
//! [`crate::coord::primitives`], with minimum-image displacements for periodic systems.
//!
//! MultiOptPy: `Coordinate/redundant_coordinate.py` (`partial_stretch_B_matirx`,
//!  `partial_bend_B_matrix`, `partial_torsion_B_matrix`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::{Col, Mat};

use crate::coord::connectivity::BondConnectivity;
use crate::coord::primitives::{
    bond_angle_rad, bond_distance, dihedral_angle_rad, wrap_angle_diff_rad,
};
use crate::domain::{AtomId, MolecularSystem, PeriodicCell};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{local_value_grad, mic_delta_ad, scatter_gradient, Scalar};

/// Angles above this value (radians) are treated as linear.
pub const LINEAR_ANGLE_RAD: f64 = 175.0 * PI / 180.0;

/// Angles below this value (radians) are treated as folded: `a` and `c` lie on the same side of
/// `b` along one line, where the angle has no derivative (as at pi) and a dihedral through it
/// no value. It happens to an angle between a bond and a link between fragments when a
/// hydrogen bond along the link becomes linear (`O-H...O` with the link `O...O`): such angles
/// and the dihedrals through them are left out, and a set that holds one is generated anew
/// (new in mopt-rs).
pub const FOLDED_ANGLE_RAD: f64 = 5.0 * PI / 180.0;

/// Whether an angle (radians) can serve as a primitive or inside a dihedral.
fn regular_angle(theta: f64) -> bool {
    theta > FOLDED_ANGLE_RAD && theta < LINEAR_ANGLE_RAD
}

/// A linear bend whose angle has fallen below this value (radians) is an ordinary angle again
/// (well below [`LINEAR_ANGLE_RAD`], so that a set is not regenerated back and forth).
pub const BENT_AGAIN_ANGLE_RAD: f64 = 160.0 * PI / 180.0;

/// One primitive internal coordinate.
#[derive(Debug, Clone, PartialEq)]
pub enum Primitive {
    /// Distance between two atoms (Bohr).
    Bond([AtomId; 2]),
    /// Angle `a - b - c` (radians).
    Angle([AtomId; 3]),
    /// Component of the bend of a (near-)linear `a - b - c` along a fixed direction `w`
    /// perpendicular to the axis: `w . (u_ba + u_bc)`, which vanishes for a linear arrangement.
    LinearBend {
        /// Atoms `a, b, c`.
        atoms: [AtomId; 3],
        /// Fixed unit direction perpendicular to the `a - c` axis.
        direction: [f64; 3],
    },
    /// Dihedral `a - b - c - d` (radians, periodic).
    Dihedral([AtomId; 4]),
}

fn local_positions<const K: usize>(sys: &MolecularSystem, atoms: &[AtomId; K]) -> [[f64; 3]; K] {
    std::array::from_fn(|s| sys.position(atoms[s]).value())
}

fn linear_bend<T: Scalar>(p: &[[T; 3]; 3], w: [f64; 3], cell: Option<&PeriodicCell>) -> T {
    let ba = mic_delta_ad(p[0], p[1], cell);
    let bc = mic_delta_ad(p[2], p[1], cell);
    let na = T::norm3(ba);
    let nc = T::norm3(bc);
    let wv = [T::from_f64(w[0]), T::from_f64(w[1]), T::from_f64(w[2])];
    T::dot3(
        wv,
        [
            ba[0] / na + bc[0] / nc,
            ba[1] / na + bc[1] / nc,
            ba[2] / na + bc[2] / nc,
        ],
    )
}

impl Primitive {
    /// Name with atom numbers starting at 1: `R(1-2)` bond, `A(1-2-3)` angle, `L(1-2-3)` bend
    /// of a linear arrangement, `D(1-2-3-4)` dihedral.
    pub fn label(&self) -> String {
        let (letter, atoms): (&str, Vec<usize>) = match self {
            Self::Bond(a) => ("R", a.iter().map(|x| x.index()).collect()),
            Self::Angle(a) => ("A", a.iter().map(|x| x.index()).collect()),
            Self::LinearBend { atoms, .. } => ("L", atoms.iter().map(|x| x.index()).collect()),
            Self::Dihedral(a) => ("D", a.iter().map(|x| x.index()).collect()),
        };
        let numbers: Vec<String> = atoms.iter().map(|a| (a + 1).to_string()).collect();
        format!("{letter}({})", numbers.join("-"))
    }

    /// Whether the coordinate is an angle with period `2 pi`.
    pub fn is_periodic(&self) -> bool {
        matches!(self, Self::Dihedral(_))
    }

    /// Value and Wilson B-matrix row (length `3N`).
    pub fn value_and_gradient(&self, sys: &MolecularSystem) -> (f64, Col<f64>) {
        let cell = sys.active_cell();
        let mut row = Col::zeros(sys.dof());
        let value = match self {
            Self::Bond(a) => {
                let d = local_value_grad(&local_positions(sys, a), |p| {
                    bond_distance(p[0], p[1], cell)
                });
                scatter_gradient(&mut row, a, &d.gradient, 1.0);
                d.value
            }
            Self::Angle(a) => {
                let d = local_value_grad(&local_positions(sys, a), |p| {
                    bond_angle_rad(p[0], p[1], p[2], cell)
                });
                scatter_gradient(&mut row, a, &d.gradient, 1.0);
                d.value
            }
            Self::LinearBend { atoms, direction } => {
                let d = local_value_grad(&local_positions(sys, atoms), |p| {
                    linear_bend(p, *direction, cell)
                });
                scatter_gradient(&mut row, atoms, &d.gradient, 1.0);
                d.value
            }
            Self::Dihedral(a) => {
                let d = local_value_grad(&local_positions(sys, a), |p| {
                    dihedral_angle_rad(p[0], p[1], p[2], p[3], cell)
                });
                scatter_gradient(&mut row, a, &d.gradient, 1.0);
                d.value
            }
        };
        (value, row)
    }

    /// Value only.
    pub fn value(&self, sys: &MolecularSystem) -> f64 {
        let cell = sys.active_cell();
        match self {
            Self::Bond(a) => {
                let p = local_positions(sys, a);
                bond_distance(p[0], p[1], cell)
            }
            Self::Angle(a) => {
                let p = local_positions(sys, a);
                bond_angle_rad(p[0], p[1], p[2], cell)
            }
            Self::LinearBend { atoms, direction } => {
                linear_bend(&local_positions(sys, atoms), *direction, cell)
            }
            Self::Dihedral(a) => {
                let p = local_positions(sys, a);
                dihedral_angle_rad(p[0], p[1], p[2], p[3], cell)
            }
        }
    }

    /// `to - from`, wrapped into `(-pi, pi]` for periodic coordinates.
    pub fn difference(&self, to: f64, from: f64) -> f64 {
        if self.is_periodic() {
            wrap_angle_diff_rad(to - from)
        } else {
            to - from
        }
    }
}

/// Ordered set of primitive internal coordinates.
#[derive(Debug, Clone, Default)]
pub struct PrimitiveSet {
    /// Primitives in row order of the B-matrix.
    pub primitives: Vec<Primitive>,
}

/// Options for automatic primitive generation.
#[derive(Debug, Clone)]
pub struct PrimitiveOptions {
    /// Bond detection scale applied to the sum of covalent radii.
    pub bond_scale: f64,
    /// Connect disconnected fragments through their closest atom pair.
    pub connect_fragments: bool,
}

impl Default for PrimitiveOptions {
    fn default() -> Self {
        Self {
            bond_scale: 1.3,
            connect_fragments: true,
        }
    }
}

fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    (n > 1e-12).then(|| [v[0] / n, v[1] / n, v[2] / n])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Two unit directions perpendicular to `axis` and to each other.
fn perpendicular_pair(axis: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let ax = unit(axis)?;
    let trial = if ax[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let w1 = unit(cross(ax, trial))?;
    let w2 = unit(cross(ax, w1))?;
    Some((w1, w2))
}

/// Fragments are also linked when their closest atoms are within this factor of the distance
/// from either of them to its nearest fragment (the factor of the auxiliary interfragment
/// bonds of Bakken and Helgaker, J. Chem. Phys. 2002, 117, 9160).
pub const NEIGHBOR_LINK_FACTOR: f64 = 1.3;

/// Bonds including the pairs that connect the fragments: the closest atom pairs along a
/// minimum spanning tree of the fragments, which makes the set connected, and those of
/// fragments that are neighbours (within [`NEIGHBOR_LINK_FACTOR`] of the distance of either to
/// its nearest fragment), which keeps the contacts of a cluster as coordinates.
///
/// The number of links grows with the number of fragments, not with its square. Linking
/// every pair, as earlier versions did, gives `F (F - 1) / 2` links, and each link brings
/// angles and dihedrals with every other link at its atoms: tens of thousands of primitives
/// for twenty molecules, and a transformation matrix that no longer fits into memory.
fn bonds_with_fragment_links(
    sys: &MolecularSystem,
    conn: &BondConnectivity,
    opts: &PrimitiveOptions,
) -> Vec<[AtomId; 2]> {
    let mut bonds = conn.bonds.clone();
    let n_frag = conn.fragments.len();
    if !opts.connect_fragments || n_frag < 2 {
        return bonds;
    }
    // Closest atom pair of every pair of fragments (symmetric table, stored flat).
    let pair = |a: usize, b: usize| a.min(b) * n_frag + a.max(b);
    let mut closest = vec![None::<(f64, AtomId, AtomId)>; n_frag * n_frag];
    for (fa, atoms_a) in conn.fragments.iter().enumerate() {
        for (fb, atoms_b) in conn.fragments.iter().enumerate().skip(fa + 1) {
            let mut best: Option<(f64, AtomId, AtomId)> = None;
            for &i in atoms_a {
                for &j in atoms_b {
                    let d = sys.distance(i, j).value();
                    if best.is_none_or(|(bd, _, _)| d < bd) {
                        best = Some((d, i.min(j), i.max(j)));
                    }
                }
            }
            closest[pair(fa, fb)] = best;
        }
    }
    let dist = |a: usize, b: usize| closest[pair(a, b)].map_or(f64::INFINITY, |c| c.0);
    let mut linked = vec![false; n_frag * n_frag];
    // Prim's algorithm: `link[f]` is the fragment of the tree nearest to fragment `f`.
    let mut in_tree = vec![false; n_frag];
    in_tree[0] = true;
    let mut link: Vec<usize> = vec![0; n_frag];
    for _ in 1..n_frag {
        let next = (0..n_frag)
            .filter(|f| !in_tree[*f])
            .min_by(|a, b| dist(*a, link[*a]).total_cmp(&dist(*b, link[*b])));
        let Some(f) = next else {
            break;
        };
        linked[pair(f, link[f])] = true;
        in_tree[f] = true;
        for g in (0..n_frag).filter(|g| !in_tree[*g]) {
            if dist(g, f) < dist(g, link[g]) {
                link[g] = f;
            }
        }
    }
    // Neighbours.
    let nearest: Vec<f64> = (0..n_frag)
        .map(|f| {
            (0..n_frag)
                .filter(|g| *g != f)
                .map(|g| dist(f, g))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    for (fa, near_a) in nearest.iter().enumerate() {
        for (fb, near_b) in nearest.iter().enumerate().skip(fa + 1) {
            let neighbours = dist(fa, fb) <= NEIGHBOR_LINK_FACTOR * near_a.max(*near_b);
            if neighbours || linked[pair(fa, fb)] {
                if let Some((_, i, j)) = closest[pair(fa, fb)] {
                    bonds.push([i, j]);
                }
            }
        }
    }
    bonds
}

/// The maximal linear chains made of the collinear triples `triples` (`[a, b, c]` with `b`
/// between `a` and `c`): atom sequences `p0, ..., pk` in which every three consecutive atoms
/// are such a triple, in the order of their first triple, each chain once.
fn linear_chains(triples: &[[AtomId; 3]]) -> Vec<Vec<AtomId>> {
    // The triple that continues `p, q` beyond `q` (centre `q`, one end `p`), with its far end.
    let next = |p: AtomId, q: AtomId, chain: &[AtomId]| -> Option<AtomId> {
        triples.iter().find_map(|&[a, b, c]| {
            let far = if b != q {
                return None;
            } else if a == p {
                c
            } else if c == p {
                a
            } else {
                return None;
            };
            (!chain.contains(&far)).then_some(far)
        })
    };
    let mut chains: Vec<Vec<AtomId>> = Vec::new();
    for &[a, b, c] in triples {
        let mut chain = vec![a, b, c];
        while let Some(far) = next(chain[chain.len() - 2], chain[chain.len() - 1], &chain) {
            chain.push(far);
        }
        chain.reverse();
        while let Some(far) = next(chain[chain.len() - 2], chain[chain.len() - 1], &chain) {
            chain.push(far);
        }
        if chain[0] > chain[chain.len() - 1] {
            chain.reverse();
        }
        if !chains.contains(&chain) {
            chains.push(chain);
        }
    }
    chains
}

impl PrimitiveSet {
    /// Generate bonds, angles (linear-bend complements for near-linear angles), and dihedrals.
    ///
    /// Collinear atoms leave torsions without a value: a dihedral through a linear angle is not
    /// defined, so that the twist about a linear chain, such as that of the two ends of allene
    /// or of the two molecules of a linear hydrogen bond `O-H...O`, has no coordinate. As in
    /// the published schemes for linear chains (Bakken and Helgaker, J. Chem. Phys. 2002, 117,
    /// 9160; Wang and Song, J. Chem. Phys. 2016, 144, 214108, the dihedrals of geomeTRIC that
    /// skip the linear atoms), every maximal chain `p0 - ... - pk` of atoms joined by linear
    /// angles gets the dihedrals `x - p0 - pk - y` about its axis from the neighbours `x` of
    /// `p0` and `y` of `pk` off the line (in addition to the two linear-bend complements of
    /// every linear angle). Sets of structures without linear angles are unchanged.
    pub fn generate(sys: &MolecularSystem, opts: &PrimitiveOptions) -> MoptResult<Self> {
        Self::generate_with(sys, opts, true)
    }

    /// [`Self::generate`]; `linear_chains_too`: with the dihedrals about linear chains.
    fn generate_with(
        sys: &MolecularSystem,
        opts: &PrimitiveOptions,
        linear_chains_too: bool,
    ) -> MoptResult<Self> {
        let n = sys.n_atoms();
        if n < 2 {
            return Err(MoptError::coord(
                "internal coordinates need at least two atoms",
            ));
        }
        let conn = BondConnectivity::from_system(sys, opts.bond_scale);
        let bonds = bonds_with_fragment_links(sys, &conn, opts);
        let mut adjacency: Vec<Vec<AtomId>> = vec![Vec::new(); n];
        for &[i, j] in &bonds {
            adjacency[i.index()].push(j);
            adjacency[j.index()].push(i);
        }
        let mut primitives: Vec<Primitive> = bonds.iter().map(|&b| Primitive::Bond(b)).collect();
        let cell = sys.active_cell();
        let angle_of = |a: AtomId, b: AtomId, c: AtomId| {
            bond_angle_rad(
                sys.position(a).value(),
                sys.position(b).value(),
                sys.position(c).value(),
                cell,
            )
        };
        // The linear angles `[a, b, c]` (`b` in the middle).
        let mut collinear: Vec<[AtomId; 3]> = Vec::new();
        for (b, nbrs) in sys.atoms().zip(&adjacency) {
            for (ia, &a) in nbrs.iter().enumerate() {
                for &c in &nbrs[ia + 1..] {
                    let (a, c) = (a.min(c), a.max(c));
                    let theta = angle_of(a, b, c);
                    if regular_angle(theta) {
                        primitives.push(Primitive::Angle([a, b, c]));
                    } else if theta >= LINEAR_ANGLE_RAD {
                        collinear.push([a, b, c]);
                        let axis = sys.delta(c, a).value();
                        if let Some((w1, w2)) = perpendicular_pair(axis) {
                            for w in [w1, w2] {
                                primitives.push(Primitive::LinearBend {
                                    atoms: [a, b, c],
                                    direction: w,
                                });
                            }
                        }
                    }
                }
            }
        }
        for &[b, c] in &bonds {
            for &a in &adjacency[b.index()] {
                if a == c || !regular_angle(angle_of(a, b, c)) {
                    continue;
                }
                for &d in &adjacency[c.index()] {
                    if d == b || d == a || !regular_angle(angle_of(b, c, d)) {
                        continue;
                    }
                    primitives.push(Primitive::Dihedral([a, b, c, d]));
                }
            }
        }
        if linear_chains_too {
            // The dihedrals about the linear chains.
            for chain in linear_chains(&collinear) {
                let (first, last) = (chain[0], chain[chain.len() - 1]);
                let off = |end: AtomId, other: AtomId| -> Vec<AtomId> {
                    adjacency[end.index()]
                        .iter()
                        .copied()
                        .filter(|x| !chain.contains(x) && regular_angle(angle_of(*x, end, other)))
                        .collect()
                };
                for x in off(first, last) {
                    for y in off(last, first) {
                        let torsion = Primitive::Dihedral([x, first, last, y]);
                        let reversed = Primitive::Dihedral([y, last, first, x]);
                        if x != y
                            && !primitives.contains(&torsion)
                            && !primitives.contains(&reversed)
                        {
                            primitives.push(torsion);
                        }
                    }
                }
            }
        }
        Ok(Self { primitives })
    }

    /// Whether a generated set no longer suits `sys` and should be generated anew there: an
    /// angle, or an angle inside a dihedral, has become linear or folded
    /// ([`FOLDED_ANGLE_RAD`]; the angle has no derivative there and the dihedral no value),
    /// or a linear bend has bent back below
    /// [`BENT_AGAIN_ANGLE_RAD`] or its fixed direction has come within 30 degrees of the axis
    /// (it then no longer measures the bend).
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs); `Coordinate/redundant_coordinate.py`
    /// switches formulas within `1e-6` of pi and keeps its primitives for the whole run.
    pub fn needs_regeneration(&self, sys: &MolecularSystem) -> bool {
        let cell = sys.active_cell();
        let angle = |a: AtomId, b: AtomId, c: AtomId| {
            bond_angle_rad(
                sys.position(a).value(),
                sys.position(b).value(),
                sys.position(c).value(),
                cell,
            )
        };
        self.primitives.iter().any(|p| match p {
            Primitive::Bond(_) => false,
            Primitive::Angle([a, b, c]) => !regular_angle(angle(*a, *b, *c)),
            Primitive::Dihedral([a, b, c, d]) => {
                !regular_angle(angle(*a, *b, *c)) || !regular_angle(angle(*b, *c, *d))
            }
            Primitive::LinearBend { atoms, direction } => {
                let [a, b, c] = *atoms;
                let along = unit(sys.delta(c, a).value()).map_or(1.0, |ax| {
                    (ax[0] * direction[0] + ax[1] * direction[1] + ax[2] * direction[2]).abs()
                });
                angle(a, b, c) < BENT_AGAIN_ANGLE_RAD || along > 0.5
            }
        })
    }

    /// Number of primitives.
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Primitive values.
    pub fn values(&self, sys: &MolecularSystem) -> Col<f64> {
        Col::from_fn(self.len(), |k| self.primitives[k].value(sys))
    }

    /// Wilson B-matrix (`m x 3N`).
    pub fn wilson_b(&self, sys: &MolecularSystem) -> Mat<f64> {
        let mut b = Mat::zeros(self.len(), sys.dof());
        for (k, p) in self.primitives.iter().enumerate() {
            let (_, row) = p.value_and_gradient(sys);
            for c in 0..sys.dof() {
                b[(k, c)] = row[c];
            }
        }
        b
    }

    /// Primitive displacement `q(to) - q(from)` with periodic wrapping.
    pub fn difference(&self, to: &MolecularSystem, from: &MolecularSystem) -> Col<f64> {
        Col::from_fn(self.len(), |k| {
            let p = &self.primitives[k];
            p.difference(p.value(to), p.value(from))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom};
    use crate::domain::{Element, SpinCharge};

    fn sys(el: Vec<Element>, pos: &[[f64; 3]]) -> MolecularSystem {
        MolecularSystem::new(el, &positions::<Angstrom>(pos), SpinCharge::default(), None).unwrap()
    }

    /// HCN at the bend angle `degrees`.
    fn hcn(degrees: f64) -> MolecularSystem {
        let t = degrees.to_radians();
        sys(
            vec![Element::H, Element::C, Element::N],
            &[
                [1.07 * t.cos(), 1.07 * t.sin(), 0.0],
                [0.0, 0.0, 0.0],
                [1.16, 0.0, 0.0],
            ],
        )
    }

    /// Twenty-seven separated molecules are joined along a spanning tree and to their
    /// neighbours, not by one link for every pair, and the set stays small.
    #[test]
    fn fragments_are_linked_along_a_spanning_tree() {
        let mut elements = Vec::new();
        let mut rows = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let o = [
                        3.1 * i as f64,
                        3.0 * j as f64 + 0.2 * i as f64,
                        2.9 * k as f64,
                    ];
                    elements.extend([Element::O, Element::H, Element::H]);
                    rows.extend([
                        o,
                        [o[0], o[1] + 0.757, o[2] - 0.586],
                        [o[0], o[1] - 0.757, o[2] - 0.586],
                    ]);
                }
            }
        }
        let cluster = sys(elements, &rows);
        let set = PrimitiveSet::generate(&cluster, &PrimitiveOptions::default()).unwrap();
        let bonds = set
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Bond(_)))
            .count();
        // Far fewer links than one for every pair of molecules (351), and at least a tree.
        let links = bonds - 27 * 2;
        assert!((26..=27 * 6).contains(&links), "{links} links");
        assert!(set.len() < 40 * cluster.dof(), "{} primitives", set.len());
        // The links connect everything.
        let conn_bonds: Vec<[AtomId; 2]> = set
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::Bond(b) => Some(*b),
                _ => None,
            })
            .collect();
        let mut group: Vec<usize> = (0..cluster.n_atoms()).collect();
        for _ in 0..cluster.n_atoms() {
            for [a, b] in &conn_bonds {
                let m = group[a.index()].min(group[b.index()]);
                group[a.index()] = m;
                group[b.index()] = m;
            }
        }
        assert!(group.iter().all(|g| *g == 0));
    }

    #[test]
    fn sets_are_regenerated_at_linear_angles() {
        let opts = PrimitiveOptions::default();
        // Generated at 150 degrees: an ordinary angle, which stops being one at 175 degrees.
        let bent = PrimitiveSet::generate(&hcn(150.0), &opts).unwrap();
        assert!(bent
            .primitives
            .iter()
            .any(|p| matches!(p, Primitive::Angle(_))));
        assert!(!bent.needs_regeneration(&hcn(150.0)));
        assert!(!bent.needs_regeneration(&hcn(170.0)));
        assert!(bent.needs_regeneration(&hcn(176.0)));
        assert!(bent.needs_regeneration(&hcn(180.0)));
        // Generated there: two linear bends, kept down to 160 degrees.
        let linear = PrimitiveSet::generate(&hcn(180.0), &opts).unwrap();
        let bends = linear
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::LinearBend { .. }))
            .count();
        assert_eq!(bends, 2);
        assert!(!linear.needs_regeneration(&hcn(180.0)));
        assert!(!linear.needs_regeneration(&hcn(165.0)));
        assert!(linear.needs_regeneration(&hcn(150.0)));
        // A dihedral with a linear angle inside (H-C-C-H of a nearly linear acetylene-like chain).
        let chain = |a: f64| {
            let t = a.to_radians();
            sys(
                vec![Element::H, Element::C, Element::C, Element::H],
                &[
                    [-1.06 * t.cos().abs(), 1.06 * t.sin(), 0.0],
                    [0.0, 0.0, 0.0],
                    [1.2, 0.0, 0.0],
                    [1.2 + 0.6, 0.0, 0.9],
                ],
            )
        };
        let set = PrimitiveSet::generate(&chain(60.0), &opts).unwrap();
        assert!(set
            .primitives
            .iter()
            .any(|p| matches!(p, Primitive::Dihedral(_))));
        assert!(!set.needs_regeneration(&chain(60.0)));
        assert!(set.needs_regeneration(&chain(2.0)));
    }

    /// Two waters linked O...O where they start; when the hydrogen bond O-H...O becomes linear
    /// the angle H-O...O folds to zero (it has no derivative there, and the delocalized
    /// coordinates of a water dimer stalled with a Cartesian gradient of 4.9e-4 and steps of
    /// 1e-13 Bohr): the set asks to be generated anew, and a set generated there holds no
    /// folded angle and no dihedral through one.
    #[test]
    fn sets_are_regenerated_at_folded_angles() {
        let opts = PrimitiveOptions::default();
        let dimer = |rows: &[[f64; 3]]| {
            sys(
                vec![
                    Element::O,
                    Element::H,
                    Element::H,
                    Element::O,
                    Element::H,
                    Element::H,
                ],
                rows,
            )
        };
        let start = dimer(&[
            [0.00, 0.00, 0.12],
            [0.00, 0.76, -0.47],
            [0.00, -0.76, -0.47],
            [3.40, 0.50, 0.90],
            [3.10, 1.30, 1.35],
            [4.20, 0.30, 1.40],
        ]);
        let set = PrimitiveSet::generate(&start, &opts).unwrap();
        let (o1, o2) = (AtomId::new(0), AtomId::new(3));
        assert!(set.primitives.contains(&Primitive::Bond([o1, o2])));
        assert!(!set.needs_regeneration(&start));
        // O-H...O along x: H-O...O is 0 degrees (and 1 degree off the line still folded).
        let t = 104.5_f64.to_radians();
        let linear = |tilt: f64| {
            let s = tilt.to_radians();
            dimer(&[
                [0.0, 0.0, 0.0],
                [0.977 * s.cos(), 0.977 * s.sin(), 0.0],
                [0.971 * t.cos(), 0.971 * t.sin(), 0.0],
                [2.936, 0.0, 0.0],
                [3.2, 0.5, 0.8],
                [3.2, 0.5, -0.8],
            ])
        };
        for tilt in [0.0, 1.0] {
            assert!(set.needs_regeneration(&linear(tilt)), "{tilt}");
        }
        assert!(!set.needs_regeneration(&linear(20.0)));
        let anew = PrimitiveSet::generate(&linear(1.0), &opts).unwrap();
        assert!(!anew.needs_regeneration(&linear(1.0)));
        let angle = |a: AtomId, b: AtomId, c: AtomId| {
            let s = linear(1.0);
            bond_angle_rad(
                s.position(a).value(),
                s.position(b).value(),
                s.position(c).value(),
                None,
            )
        };
        for p in &anew.primitives {
            match p {
                Primitive::Angle([a, b, c]) => assert!(regular_angle(angle(*a, *b, *c))),
                Primitive::Dihedral([a, b, c, d]) => {
                    assert!(regular_angle(angle(*a, *b, *c)) && regular_angle(angle(*b, *c, *d)));
                }
                _ => {}
            }
        }
    }

    #[test]
    fn water_and_co2_primitives() {
        let h2o = sys(
            vec![Element::O, Element::H, Element::H],
            &[
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
            ],
        );
        let set = PrimitiveSet::generate(&h2o, &PrimitiveOptions::default()).unwrap();
        assert_eq!(set.len(), 3);
        let co2 = sys(
            vec![Element::O, Element::C, Element::O],
            &[[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]],
        );
        let set = PrimitiveSet::generate(&co2, &PrimitiveOptions::default()).unwrap();
        // 2 bonds + 2 linear-bend components = 3N - 5
        assert_eq!(set.len(), 4);
    }

    /// The rank of the Wilson B-matrix of `set` at `s`: the number of eigenvalues of `B^T B`
    /// above `1e-10` of the largest.
    fn rank(set: &PrimitiveSet, s: &MolecularSystem) -> usize {
        let b = set.wilson_b(s);
        let btb = b.transpose() * &b;
        let (values, _) = crate::math::linalg::symmetric_eigen(&btb).unwrap();
        let top = values.iter().copied().fold(0.0_f64, f64::max);
        values.iter().filter(|v| **v > 1e-10 * top).count()
    }

    /// Molecules with linear angles, and whether each is linear as a whole.
    fn with_linear_angles() -> Vec<(&'static str, MolecularSystem, bool)> {
        let t = 104.5_f64.to_radians();
        let tetra = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        let mut carbonyl = vec![[0.0; 3]];
        let mut carbonyl_el = vec![Element::parse_token("Ni").unwrap()];
        for d in tetra {
            let s = 1.0 / 3f64.sqrt();
            carbonyl.push([1.82 * s * d[0], 1.82 * s * d[1], 1.82 * s * d[2]]);
            carbonyl.push([2.97 * s * d[0], 2.97 * s * d[1], 2.97 * s * d[2]]);
            carbonyl_el.extend([Element::C, Element::O]);
        }
        vec![
            (
                // O-H...O along x: the link H...O makes a linear angle at the hydrogen.
                "water dimer",
                sys(
                    vec![
                        Element::O,
                        Element::H,
                        Element::H,
                        Element::O,
                        Element::H,
                        Element::H,
                    ],
                    &[
                        [0.0, 0.0, 0.0],
                        [0.96, 0.0, 0.0],
                        [0.96 * t.cos(), 0.96 * t.sin(), 0.0],
                        [2.9, 0.0, 0.0],
                        [3.2, 0.5, 0.8],
                        [3.2, 0.5, -0.8],
                    ],
                ),
                false,
            ),
            ("HCN", hcn(180.0), true),
            (
                "acetylene",
                sys(
                    vec![Element::H, Element::C, Element::C, Element::H],
                    &[
                        [-1.66, 0.0, 0.0],
                        [-0.6, 0.0, 0.0],
                        [0.6, 0.0, 0.0],
                        [1.66, 0.0, 0.0],
                    ],
                ),
                true,
            ),
            (
                // The two CH2 groups in perpendicular planes.
                "allene",
                sys(
                    vec![
                        Element::C,
                        Element::C,
                        Element::C,
                        Element::H,
                        Element::H,
                        Element::H,
                        Element::H,
                    ],
                    &[
                        [0.0, 0.0, 0.0],
                        [1.31, 0.0, 0.0],
                        [2.62, 0.0, 0.0],
                        [-0.55, 0.93, 0.0],
                        [-0.55, -0.93, 0.0],
                        [3.17, 0.0, 0.93],
                        [3.17, 0.0, -0.93],
                    ],
                ),
                false,
            ),
            ("Ni(CO)4", sys(carbonyl_el, &carbonyl), false),
            (
                // Water over the carbon of CO2, in the plane of the molecule.
                "CO2...H2O",
                sys(
                    vec![
                        Element::O,
                        Element::C,
                        Element::O,
                        Element::O,
                        Element::H,
                        Element::H,
                    ],
                    &[
                        [-1.16, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.16, 0.0, 0.0],
                        [0.0, 2.8, 0.0],
                        [0.76, 3.39, 0.0],
                        [-0.76, 3.39, 0.0],
                    ],
                ),
                false,
            ),
        ]
    }

    #[test]
    fn sets_with_linear_chains_span_the_internal_space() {
        let opts = PrimitiveOptions::default();
        let mut deficient = Vec::new();
        for (name, s, linear) in with_linear_angles() {
            let internal = s.dof() - if linear { 5 } else { 6 };
            let set = PrimitiveSet::generate(&s, &opts).unwrap();
            let without = PrimitiveSet::generate_with(&s, &opts, false).unwrap();
            let (with_rank, without_rank) = (rank(&set, &s), rank(&without, &s));
            if without_rank < internal {
                deficient.push(name);
            }
            assert_eq!(with_rank, internal, "{name}: {set:?}");
            assert!(!set.needs_regeneration(&s), "{name}");
        }
        // The water dimer (11 of 12 before) and allene (the twist of its ends) lacked a
        // coordinate without the dihedrals about the chains.
        assert_eq!(deficient, vec!["water dimer", "allene"]);
        // The dihedrals added for allene skip the middle carbon.
        let allene = &with_linear_angles()[3].1;
        let set = PrimitiveSet::generate(allene, &opts).unwrap();
        let (c1, c3) = (AtomId::new(0), AtomId::new(2));
        assert!(set
            .primitives
            .iter()
            .any(|p| matches!(p, Primitive::Dihedral([_, b, c, _]) if *b == c1 && *c == c3)));
    }

    #[test]
    fn sets_without_linear_angles_do_not_change() {
        let opts = PrimitiveOptions::default();
        let molecules = [
            sys(
                vec![Element::O, Element::H, Element::H],
                &[
                    [0.0, 0.0, 0.117],
                    [0.0, 0.757, -0.469],
                    [0.0, -0.757, -0.469],
                ],
            ),
            sys(
                vec![Element::H, Element::O, Element::O, Element::H],
                &[
                    [0.9, 0.3, 0.6],
                    [0.0, 0.7, 0.0],
                    [0.0, -0.7, 0.0],
                    [-0.8, -0.9, 0.5],
                ],
            ),
            // Ethane (staggered).
            sys(
                vec![
                    Element::C,
                    Element::C,
                    Element::H,
                    Element::H,
                    Element::H,
                    Element::H,
                    Element::H,
                    Element::H,
                ],
                &[
                    [0.0, 0.0, 0.765],
                    [0.0, 0.0, -0.765],
                    [1.02, 0.0, 1.16],
                    [-0.51, 0.88, 1.16],
                    [-0.51, -0.88, 1.16],
                    [-1.02, 0.0, -1.16],
                    [0.51, 0.88, -1.16],
                    [0.51, -0.88, -1.16],
                ],
            ),
            // Formaldehyde, ammonia, and a bent HCN.
            sys(
                vec![Element::C, Element::O, Element::H, Element::H],
                &[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.21],
                    [0.0, 0.94, -0.58],
                    [0.0, -0.94, -0.58],
                ],
            ),
            sys(
                vec![Element::N, Element::H, Element::H, Element::H],
                &[
                    [0.0, 0.0, 0.11],
                    [0.0, 0.94, -0.26],
                    [0.81, -0.47, -0.26],
                    [-0.81, -0.47, -0.26],
                ],
            ),
            hcn(150.0),
        ];
        for s in &molecules {
            let set = PrimitiveSet::generate(s, &opts).unwrap();
            let before = PrimitiveSet::generate_with(s, &opts, false).unwrap();
            assert_eq!(set.primitives, before.primitives);
        }
    }

    #[test]
    fn b_matrix_matches_finite_differences() {
        let h2o2 = sys(
            vec![Element::H, Element::O, Element::O, Element::H],
            &[
                [0.9, 0.3, 0.6],
                [0.0, 0.7, 0.0],
                [0.0, -0.7, 0.0],
                [-0.8, -0.9, 0.5],
            ],
        );
        let set = PrimitiveSet::generate(&h2o2, &PrimitiveOptions::default()).unwrap();
        assert!(set
            .primitives
            .iter()
            .any(|p| matches!(p, Primitive::Dihedral(_))));
        let b = set.wilson_b(&h2o2);
        let h = 1e-6;
        for c in 0..h2o2.dof() {
            let mut xp = h2o2.coords_flat_bohr();
            let mut xm = xp.clone();
            xp[c] += h;
            xm[c] -= h;
            let sp = h2o2.with_flat_coords_bohr(&xp).unwrap();
            let sm = h2o2.with_flat_coords_bohr(&xm).unwrap();
            let fd = set.difference(&sp, &sm);
            for k in 0..set.len() {
                assert!((b[(k, c)] - fd[k] / (2.0 * h)).abs() < 1e-7);
            }
        }
    }
}
