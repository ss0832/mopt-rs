//! Z-matrices along the bond graph: every atom by a bond length, a bond angle, and a dihedral
//! angle to atoms defined before it, and back to Cartesian coordinates.
//!
//! The atoms are taken in breadth-first order through the bonds from the heavy atom with the
//! most bonds; an atom is defined by its bond to the atom it was reached from, an angle with
//! a bonded atom of that one (or of the atom before it in the tree), and a dihedral with a
//! third atom chosen the same way, avoiding nearly collinear triples. Further fragments start
//! from the atom defined before that is nearest to them. Cartesian coordinates are rebuilt
//! row by row by the natural extension reference frame (NeRF) of Parsons et al. (J. Comput.
//! Chem. 2005, 26, 1063). Bonds that close rings are not rows of the Z-matrix, so rings stay
//! closed only when their dihedral angles change consistently.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use crate::coord::graph::MolecularGraph;
use crate::coord::primitives::{bond_angle_rad, dihedral_angle_rad, wrap_angle_diff_rad};
use crate::domain::{AtomId, MolecularSystem, Position};
use crate::error::{MoptError, MoptResult};

/// One row: an atom and the atoms it is defined by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZRow {
    /// The atom (from 0).
    pub atom: usize,
    /// The atom of the bond length.
    pub bond: Option<usize>,
    /// The third atom of the bond angle `atom - bond - angle`.
    pub angle: Option<usize>,
    /// The fourth atom of the dihedral `atom - bond - angle - dihedral`.
    pub dihedral: Option<usize>,
}

/// Values of a row: bond length (Bohr), bond angle and dihedral angle (rad).
pub type ZValues = [f64; 3];

/// A Z-matrix of a structure.
#[derive(Debug, Clone, PartialEq)]
pub struct ZMatrix {
    rows: Vec<ZRow>,
}

/// Bends closer than this to a straight line (rad) do not define a dihedral angle.
const COLLINEAR: f64 = 10.0 * std::f64::consts::PI / 180.0;

fn pos(sys: &MolecularSystem, i: usize) -> [f64; 3] {
    sys.position(AtomId::new(i)).value()
}

fn bent(sys: &MolecularSystem, a: usize, b: usize, c: usize) -> bool {
    let t = bond_angle_rad(pos(sys, a), pos(sys, b), pos(sys, c), None);
    t > COLLINEAR && t < std::f64::consts::PI - COLLINEAR
}

impl ZMatrix {
    /// The Z-matrix of `sys` along the bonds of `graph` (molecules in a cell made whole first).
    pub fn from_graph(graph: &MolecularGraph, sys: &MolecularSystem) -> MoptResult<Self> {
        let n = sys.n_atoms();
        if graph.n_atoms() != n {
            return Err(MoptError::coord(
                "the bond graph does not match the structure",
            ));
        }
        let heavy = |i: usize| graph.atomic_numbers()[i] > 1;
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut order = Vec::with_capacity(n);
        let mut placed = vec![false; n];
        let mut fragments = graph.fragments();
        // The largest fragment first.
        fragments.sort_by_key(|f| std::cmp::Reverse(f.len()));
        for frag in fragments {
            let Some(&root) = frag
                .iter()
                .max_by_key(|&&a| (heavy(a), graph.neighbors(a).len(), std::cmp::Reverse(a)))
            else {
                continue;
            };
            if !order.is_empty() {
                // Attach the fragment to the nearest atom defined before.
                parent[root] = order.iter().copied().min_by(|&p: &usize, &q: &usize| {
                    sys.distance(AtomId::new(p), AtomId::new(root))
                        .value()
                        .total_cmp(&sys.distance(AtomId::new(q), AtomId::new(root)).value())
                });
            }
            placed[root] = true;
            let mut queue = VecDeque::from([root]);
            while let Some(a) = queue.pop_front() {
                order.push(a);
                for &b in graph.neighbors(a) {
                    if !placed[b] {
                        placed[b] = true;
                        parent[b] = Some(a);
                        queue.push_back(b);
                    }
                }
            }
        }
        let mut rows = Vec::with_capacity(n);
        let mut defined: Vec<usize> = Vec::with_capacity(n);
        for &i in &order {
            let mut row = ZRow {
                atom: i,
                bond: None,
                angle: None,
                dihedral: None,
            };
            if let Some(&first) = defined.first() {
                let b = parent[i].unwrap_or(first);
                row.bond = Some(b);
                // Candidates in order of preference: the tree parent, bonded atoms, any atom.
                let around = |x: usize, skip: &[usize]| -> Vec<usize> {
                    let mut c: Vec<usize> = parent[x].into_iter().collect();
                    c.extend(graph.neighbors(x).iter().copied());
                    c.extend(defined.iter().copied());
                    c.retain(|y| !skip.contains(y) && defined.contains(y));
                    c.dedup();
                    c
                };
                if defined.len() >= 2 {
                    let a_options = around(b, &[i, b]);
                    let a = a_options
                        .iter()
                        .copied()
                        .find(|&a| bent(sys, i, b, a))
                        .or_else(|| a_options.first().copied());
                    row.angle = a;
                    if let (Some(a), true) = (a, defined.len() >= 3) {
                        let mut d_options = around(a, &[i, b, a]);
                        d_options.extend(around(b, &[i, b, a]));
                        row.dihedral = d_options
                            .iter()
                            .copied()
                            .find(|&d| bent(sys, b, a, d))
                            .or_else(|| d_options.first().copied());
                    }
                }
            }
            defined.push(i);
            rows.push(row);
        }
        Ok(Self { rows })
    }

    /// The rows in the order of definition.
    pub fn rows(&self) -> &[ZRow] {
        &self.rows
    }

    /// The values of every row at `sys` (unused entries 0).
    pub fn values(&self, sys: &MolecularSystem) -> Vec<ZValues> {
        self.rows
            .iter()
            .map(|r| {
                let x = pos(sys, r.atom);
                let mut v = [0.0; 3];
                if let Some(b) = r.bond {
                    let pb = pos(sys, b);
                    v[0] = libm::sqrt((0..3).map(|k| (x[k] - pb[k]).powi(2)).sum::<f64>());
                    if let Some(a) = r.angle {
                        v[1] = bond_angle_rad(x, pb, pos(sys, a), None);
                        if let Some(d) = r.dihedral {
                            v[2] = dihedral_angle_rad(x, pb, pos(sys, a), pos(sys, d), None);
                        }
                    }
                }
                v
            })
            .collect()
    }

    /// The structure of `template` (elements, charge, cell) with the atoms placed from
    /// `values`, row by row.
    pub fn to_system(
        &self,
        values: &[ZValues],
        template: &MolecularSystem,
    ) -> MoptResult<MolecularSystem> {
        if values.len() != self.rows.len() || template.n_atoms() != self.rows.len() {
            return Err(MoptError::coord(
                "the Z-matrix values do not match its rows",
            ));
        }
        let mut x: Vec<Option<[f64; 3]>> = vec![None; self.rows.len()];
        let get = |x: &[Option<[f64; 3]>], i: usize| {
            x[i].ok_or_else(|| MoptError::coord("a Z-matrix row refers to a later atom"))
        };
        for (row, v) in self.rows.iter().zip(values) {
            let p = match (row.bond, row.angle, row.dihedral) {
                (None, _, _) => [0.0; 3],
                (Some(b), None, _) => {
                    let pb = get(&x, b)?;
                    [pb[0] + v[0], pb[1], pb[2]]
                }
                (Some(b), Some(a), None) => {
                    let (pb, pa) = (get(&x, b)?, get(&x, a)?);
                    // In the plane of b, a, and the z axis direction.
                    let u = unit(sub(pa, pb))?;
                    let w = unit(perpendicular(u))?;
                    let (c, s) = (v[1].cos(), v[1].sin());
                    std::array::from_fn(|k| pb[k] + v[0] * (c * u[k] + s * w[k]))
                }
                (Some(b), Some(a), Some(d)) => {
                    nerf(get(&x, d)?, get(&x, a)?, get(&x, b)?, v[0], v[1], v[2])?
                }
            };
            x[row.atom] = Some(p);
        }
        let mut out = template.clone();
        for i in 0..x.len() {
            out.set_position(AtomId::new(i), Position::new(get(&x, i)?));
        }
        Ok(out)
    }
}

/// `reference + (to - from)` row by row, with the angles of the difference wrapped into
/// `(-pi, pi]`: the change from `from` to `to` carried over to `reference`.
pub fn transfer(reference: &[ZValues], from: &[ZValues], to: &[ZValues]) -> Vec<ZValues> {
    reference
        .iter()
        .zip(from.iter().zip(to))
        .map(|(r, (f, t))| {
            [
                r[0] + (t[0] - f[0]),
                r[1] + (t[1] - f[1]),
                r[2] + wrap_angle_diff_rad(t[2] - f[2]),
            ]
        })
        .collect()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: [f64; 3]) -> MoptResult<[f64; 3]> {
    let n = libm::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    if n < 1e-10 {
        return Err(MoptError::coord("two atoms of a Z-matrix row coincide"));
    }
    Ok([a[0] / n, a[1] / n, a[2] / n])
}

fn perpendicular(u: [f64; 3]) -> [f64; 3] {
    let trial = if u[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let c = cross(u, trial);
    cross(c, u)
}

/// The position at bond length `r` from `c`, angle `theta` with `b`, and dihedral `phi` with
/// `a` (atoms `a - b - c - new`).
fn nerf(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    r: f64,
    theta: f64,
    phi: f64,
) -> MoptResult<[f64; 3]> {
    let bc = unit(sub(c, b))?;
    let n = unit(cross(sub(b, a), bc))?;
    let m = cross(n, bc);
    // The sign of the last component follows `dihedral_angle_rad` (a positive angle turns
    // the new atom from `m` towards `-n`).
    let d2 = [
        -r * theta.cos(),
        r * theta.sin() * phi.cos(),
        -r * theta.sin() * phi.sin(),
    ];
    Ok(std::array::from_fn(|k| {
        c[k] + bc[k] * d2[0] + m[k] * d2[1] + n[k] * d2[2]
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coord::graph::MULTIOPTPY_BOND_SCALE;
    use crate::coord::invariant_rmsd::direct_rmsd;
    use crate::domain::SpinCharge;
    use crate::io::xyz::parse_xyz_str;

    const BUTANE: &str = "14\nbutane\nC -1.9 0.4 0.0\nC -0.6 -0.4 0.0\n\
C 0.6 0.4 0.0\nC 1.9 -0.4 0.0\nH -2.0 1.0 0.9\nH -2.0 1.0 -0.9\nH -2.8 -0.3 0.0\n\
H -0.6 -1.0 0.9\nH -0.6 -1.0 -0.9\nH 0.6 1.0 0.9\nH 0.6 1.0 -0.9\nH 2.0 -1.0 0.9\n\
H 2.0 -1.0 -0.9\nH 2.8 0.3 0.0\n";

    #[test]
    fn the_z_matrix_rebuilds_the_structure_and_carries_a_torsion_over() {
        let sys = parse_xyz_str(BUTANE, SpinCharge::default(), None).unwrap();
        let g = MolecularGraph::from_system(&sys, MULTIOPTPY_BOND_SCALE);
        let z = ZMatrix::from_graph(&g, &sys).unwrap();
        assert_eq!(z.rows().len(), 14);
        let v = z.values(&sys);
        let rebuilt = z.to_system(&v, &sys).unwrap();
        assert!(direct_rmsd(&rebuilt, &sys).unwrap() < 1e-8);
        // Turn the central dihedral by 120 degrees in a copy, then carry the change over.
        let row = z
            .rows()
            .iter()
            .position(|r| {
                let mut atoms = [Some(r.atom), r.bond, r.angle, r.dihedral];
                atoms.sort();
                atoms == [Some(0), Some(1), Some(2), Some(3)]
            })
            .unwrap();
        let mut turned = v.clone();
        for (k, r) in z.rows().iter().enumerate() {
            // Every atom defined by a dihedral about the central bond turns with it.
            let axis = matches!((r.bond, r.angle), (Some(1), Some(2)) | (Some(2), Some(1)));
            if axis {
                turned[k][2] += 2.0 * std::f64::consts::PI / 3.0;
            }
        }
        assert!(turned[row][2] != v[row][2]);
        let other = z.to_system(&turned, &sys).unwrap();
        let crossed = transfer(&v, &v, &turned);
        let again = z.to_system(&crossed, &sys).unwrap();
        assert!(direct_rmsd(&again, &other).unwrap() < 1e-8);
        let d = z.values(&other)[row][2] - v[row][2];
        assert!((wrap_angle_diff_rad(d) - 2.0 * std::f64::consts::PI / 3.0).abs() < 1e-8);
        // The bonds survive.
        let h = MolecularGraph::from_system(&other, MULTIOPTPY_BOND_SCALE);
        assert!(g.same_bonds(&h));
    }
}
