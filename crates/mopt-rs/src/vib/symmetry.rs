//! Point group and rotational symmetry number of a molecule.
//!
//! A symmetry operation is an isometry about the centre of mass that maps every atom onto an
//! atom of the same element. It is fixed by the images of two atoms `a` and `b` that are not
//! collinear with the centre. The atoms are sorted into shells of equal element and distance
//! from the centre; `a` is taken from the smallest shell and `b` from the smallest shell that
//! has an atom off the line through `a`. Every operation maps `a` into its shell and `b` into
//! its shell at an unchanged distance from the image of `a`, so the candidates are the pairs
//! of such images, each giving one rotation and one rotation-reflection. A candidate is
//! accepted when it maps all atoms onto atoms.
//!
//! The number of candidates is about the size of the smallest shell (one for a molecule
//! without symmetry), and testing one costs the number of atoms times a shell lookup, ended
//! at the first atom without an image. MultiOptPy tests rotations of every order up to
//! `max_n_fold`, reflections, and rotation-reflections about several thousand candidate axes
//! made from pairs of atoms.
//!
//! The rotations found are all the proper operations, and their number is the symmetry
//! number. The point group follows from the orders of the rotation axes and the presence of
//! the inversion, of mirror planes, and of other improper operations. Since every operation
//! is among the candidates, the group is complete for the tolerance in use; nothing is
//! sampled.
//!
//! What the tolerance decides: a structure from an optimization is symmetric only within the
//! accuracy of the optimization, and with a tolerance that is too wide, operations that hold
//! only roughly are accepted and the set is no longer a group. The operations found are
//! therefore checked for closure (as permutations of the atoms with their parity); when the
//! check fails the tolerance is halved and the search repeated, and the tolerance used is
//! reported.
//!
//! MultiOptPy: `Utils/symmetry_analyzer.py:SymmetryAnalyzer`,
//!  `Thermo/normal_mode_analyzer.py:MolecularVibrations.get_rotational_symmetry_number`
//!  (which returns 1 for the groups Th, O, and I).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, MolecularSystem};
use crate::math::projection::{shape, Shape};

type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

/// Point group of a molecule.
#[derive(Debug, Clone, PartialEq)]
pub struct PointGroup {
    /// Schoenflies symbol (`C2v`, `D6h`, `Td`, `C\u{221e}v`, ...; `Kh` for an atom).
    pub label: String,
    /// Rotational symmetry number: the number of proper rotations.
    pub symmetry_number: u32,
    /// Number of symmetry operations found (rotations and improper operations; the two
    /// reflections of a linear molecule's symbol are not counted).
    pub operations: usize,
    /// Tolerance (Bohr) at which the operations form a group: the one asked for, or a
    /// smaller one.
    pub tolerance: f64,
}

impl PointGroup {
    /// No symmetry (`C1`): what is assumed when the analysis is switched off.
    pub fn c1() -> Self {
        Self {
            label: "C1".to_string(),
            symmetry_number: 1,
            operations: 1,
            tolerance: 0.0,
        }
    }
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scaled(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn apply(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// Orthonormal frame with the first axis along `a` and the second in the plane of `a` and
/// `b`; `None` when `a` and `b` are collinear or `a` vanishes.
fn frame(a: V3, b: V3, tolerance: f64) -> Option<[V3; 3]> {
    let la = norm(a);
    if la <= tolerance {
        return None;
    }
    let e1 = scaled(a, 1.0 / la);
    let across = sub(b, scaled(e1, dot(b, e1)));
    let lb = norm(across);
    if lb <= tolerance {
        return None;
    }
    let e2 = scaled(across, 1.0 / lb);
    Some([e1, e2, cross(e1, e2)])
}

/// The matrix that takes the frame `from` to the frame `to` (`sum_k to_k from_k^T`).
fn between(from: &[V3; 3], to: &[V3; 3]) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for k in 0..3 {
        for (r, row) in m.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value += to[k][r] * from[k][c];
            }
        }
    }
    m
}

/// Atoms relative to the centre of mass, with their shells.
struct Shells {
    positions: Vec<V3>,
    /// Shell of every atom.
    of: Vec<usize>,
    /// Atoms of every shell.
    members: Vec<Vec<usize>>,
    tolerance: f64,
}

impl Shells {
    fn new(sys: &MolecularSystem, tolerance: f64) -> Self {
        let com = sys.center_of_mass().value();
        let positions: Vec<V3> = sys
            .atoms()
            .map(|a| sub(sys.position(a).value(), com))
            .collect();
        let mut keys: Vec<(Element, f64)> = Vec::new();
        let mut members: Vec<Vec<usize>> = Vec::new();
        let mut of = Vec::with_capacity(positions.len());
        for (i, p) in positions.iter().enumerate() {
            let (element, radius) = (sys.elements[i], norm(*p));
            let found = keys
                .iter()
                .position(|(e, r)| *e == element && (r - radius).abs() <= tolerance);
            let shell = match found {
                Some(k) => k,
                None => {
                    keys.push((element, radius));
                    members.push(Vec::new());
                    keys.len() - 1
                }
            };
            members[shell].push(i);
            of.push(shell);
        }
        Self {
            positions,
            of,
            members,
            tolerance,
        }
    }

    /// The atom every atom is mapped onto by `op`, or `None` when an atom has no image in
    /// its shell (small shells are tried first, so that a wrong operation fails early).
    fn images(&self, op: &M3, order: &[usize]) -> Option<Vec<usize>> {
        let mut images = vec![0; self.positions.len()];
        for &i in order {
            let image = apply(op, self.positions[i]);
            images[i] = self.members[self.of[i]]
                .iter()
                .copied()
                .find(|&j| norm(sub(image, self.positions[j])) <= self.tolerance)?;
        }
        Some(images)
    }
}

/// Whether the operations, given as permutations of the atoms with their parity (`true`:
/// improper), contain the product of every two of them.
fn closed(operations: &[(Vec<usize>, bool)]) -> bool {
    let known: std::collections::HashSet<(&[usize], bool)> = operations
        .iter()
        .map(|(images, improper)| (images.as_slice(), *improper))
        .collect();
    operations.iter().all(|(p, p_improper)| {
        operations.iter().all(|(q, q_improper)| {
            let product: Vec<usize> = q.iter().map(|&j| p[j]).collect();
            known.contains(&(product.as_slice(), p_improper != q_improper))
        })
    })
}

/// Axis of a proper rotation other than the identity.
fn rotation_axis(op: &M3) -> Option<V3> {
    // The antisymmetric part gives the axis, except for half turns.
    let mut axis = [
        op[2][1] - op[1][2],
        op[0][2] - op[2][0],
        op[1][0] - op[0][1],
    ];
    if norm(axis) < 1e-3 {
        // Half turn: R + 1 = 2 n n^T; take its largest column.
        let columns: [V3; 3] = std::array::from_fn(|c| {
            std::array::from_fn(|r| op[r][c] + if r == c { 1.0 } else { 0.0 })
        });
        axis = columns
            .into_iter()
            .max_by(|x, y| norm(*x).total_cmp(&norm(*y)))?;
    }
    let length = norm(axis);
    (length > 1e-6).then(|| scaled(axis, 1.0 / length))
}

/// Whether two unit vectors are parallel within about eight degrees (operations that hold
/// within the tolerance have axes that are slightly off).
fn parallel(a: V3, b: V3) -> bool {
    dot(a, b).abs() > 0.99
}

/// Order of a permutation: the least common multiple of its cycle lengths.
fn permutation_order(images: &[usize]) -> usize {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let mut seen = vec![false; images.len()];
    let mut order = 1;
    for start in 0..images.len() {
        let mut length = 0;
        let mut i = start;
        while !seen[i] {
            seen[i] = true;
            i = images[i];
            length += 1;
        }
        if length > 1 {
            order = order / gcd(order, length) * length;
        }
    }
    order
}

/// Point group of `sys`; atoms count as coinciding within `tolerance` (Bohr). When the
/// operations found with it do not form a group, the tolerance is halved (up to eight
/// times; see [`PointGroup::tolerance`]).
pub fn point_group(sys: &MolecularSystem, tolerance: f64) -> PointGroup {
    let mut tolerance = tolerance;
    let mut found = at_tolerance(sys, tolerance);
    for _ in 0..8 {
        if found.1 {
            break;
        }
        tolerance *= 0.5;
        found = at_tolerance(sys, tolerance);
    }
    found.0
}

/// The operations within `tolerance` as a point group, and whether they form a group.
fn at_tolerance(sys: &MolecularSystem, tolerance: f64) -> (PointGroup, bool) {
    let n = sys.n_atoms();
    if n <= 1 {
        return (
            PointGroup {
                label: "Kh".to_string(),
                symmetry_number: 1,
                operations: 1,
                tolerance,
            },
            true,
        );
    }
    let shells = Shells::new(sys, tolerance);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| shells.members[shells.of[i]].len());
    let inversion: M3 = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    let has_inversion = shells.images(&inversion, &order).is_some();

    // Reference atoms: `a` off the centre in the smallest shell, `b` off the line of `a`.
    let Some(&a) = order
        .iter()
        .find(|&&i| norm(shells.positions[i]) > tolerance)
    else {
        return (
            PointGroup {
                tolerance,
                ..PointGroup::c1()
            },
            true,
        );
    };
    // Whether the molecule is linear is the decision of the projection and the thermochemistry
    // (`math::projection::shape`), so that the point group, the rotor and the number of
    // vibrations agree. A molecule that is not linear by that test takes its reference atom
    // `b` off the line at any distance (down to exact collinearity) when none is off it by
    // the tolerance.
    let linear = matches!(shape(sys), Shape::Linear { .. });
    let reference = if linear {
        None
    } else {
        order
            .iter()
            .find_map(|&b| {
                frame(shells.positions[a], shells.positions[b], tolerance).map(|f| (b, f))
            })
            .or_else(|| {
                order.iter().find_map(|&b| {
                    frame(shells.positions[a], shells.positions[b], 0.0).map(|f| (b, f))
                })
            })
    };
    let Some((b, from)) = reference else {
        // All atoms on a line through the centre.
        let (label, symmetry_number) = if has_inversion {
            ("D\u{221e}h", 2)
        } else {
            ("C\u{221e}v", 1)
        };
        return (
            PointGroup {
                label: label.to_string(),
                symmetry_number,
                operations: symmetry_number as usize,
                tolerance,
            },
            true,
        );
    };

    let (pa, pb) = (shells.positions[a], shells.positions[b]);
    let separation = norm(sub(pa, pb));
    let mut rotations: Vec<M3> = Vec::new();
    let mut improper: Vec<M3> = Vec::new();
    let mut permutations: Vec<(Vec<usize>, bool)> = Vec::new();
    for &ia in &shells.members[shells.of[a]] {
        for &ib in &shells.members[shells.of[b]] {
            let (qa, qb) = (shells.positions[ia], shells.positions[ib]);
            if (norm(sub(qa, qb)) - separation).abs() > 2.0 * tolerance {
                continue;
            }
            let Some(to) = frame(qa, qb, tolerance) else {
                continue;
            };
            let proper = between(&from, &to);
            if let Some(images) = shells.images(&proper, &order) {
                rotations.push(proper);
                permutations.push((images, false));
            }
            let mirrored = [to[0], to[1], scaled(to[2], -1.0)];
            let other = between(&from, &mirrored);
            if let Some(images) = shells.images(&other, &order) {
                improper.push(other);
                permutations.push((images, true));
            }
        }
    }
    let symmetry_number = rotations.len().max(1);
    let operations = symmetry_number + improper.len();

    // Orders of the rotations as permutations of the atoms: the same as their orders as
    // rotations, since no rotation of a molecule that is not linear leaves every atom in
    // place. They do not depend on how well the operations hold.
    let orders: Vec<usize> = permutations
        .iter()
        .filter(|(_, improper)| !improper)
        .map(|(images, _)| permutation_order(images))
        .collect();
    let highest = orders.iter().copied().max().unwrap_or(1);
    let principal = rotations
        .iter()
        .zip(&orders)
        .filter(|(_, order)| **order == highest && highest > 1)
        .find_map(|(op, _)| rotation_axis(op));
    // Mirror planes: improper operations with a trace near 1 (the inversion has -3, the
    // rotation-reflections lie between -2 and 0).
    let mirrors: Vec<V3> = improper
        .iter()
        .filter(|op| (op[0][0] + op[1][1] + op[2][2] - 1.0).abs() < 0.5)
        .filter_map(|op| {
            // 1 - Q = 2 n n^T.
            let columns: [V3; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|r| if r == c { 1.0 } else { 0.0 } - op[r][c])
            });
            let normal = columns
                .into_iter()
                .max_by(|x, y| norm(*x).total_cmp(&norm(*y)))?;
            let length = norm(normal);
            (length > 1e-6).then(|| scaled(normal, 1.0 / length))
        })
        .collect();

    // The rotation groups T, O, and I have 12, 24, and 60 elements of order 3, 4, and 5 at
    // most; a dihedral or cyclic group of the same size has an element of higher order.
    let cubic = matches!((symmetry_number, highest), (12, 3) | (24, 4) | (60, 5));
    let label = if cubic {
        match (symmetry_number, has_inversion, mirrors.is_empty()) {
            (60, true, _) => "Ih".to_string(),
            (60, false, _) => "I".to_string(),
            (24, true, _) => "Oh".to_string(),
            (24, false, _) => "O".to_string(),
            (_, true, _) => "Th".to_string(),
            (_, false, false) => "Td".to_string(),
            (_, false, true) => "T".to_string(),
        }
    } else {
        match principal {
            None if has_inversion => "Ci".to_string(),
            None if !mirrors.is_empty() => "Cs".to_string(),
            None => "C1".to_string(),
            Some(axis) => {
                let order = highest;
                let horizontal = mirrors.iter().any(|m| parallel(*m, axis));
                if symmetry_number == 2 * order {
                    if horizontal {
                        format!("D{order}h")
                    } else if !mirrors.is_empty() {
                        format!("D{order}d")
                    } else {
                        format!("D{order}")
                    }
                } else if horizontal {
                    format!("C{order}h")
                } else if !mirrors.is_empty() {
                    format!("C{order}v")
                } else if !improper.is_empty() {
                    format!("S{}", 2 * order)
                } else {
                    format!("C{order}")
                }
            }
        }
    };
    (
        PointGroup {
            label,
            symmetry_number: symmetry_number as u32,
            operations,
            tolerance,
        },
        closed(&permutations),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, SpinCharge};

    fn molecule(rows: &[(&str, [f64; 3])]) -> MolecularSystem {
        MolecularSystem::new(
            rows.iter()
                .map(|r| Element::parse_token(r.0).unwrap())
                .collect(),
            &positions::<Angstrom>(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    fn group(rows: &[(&str, [f64; 3])]) -> (String, u32) {
        let g = point_group(&molecule(rows), 1e-3);
        (g.label, g.symmetry_number)
    }

    /// `n` atoms on a circle of radius `r` at height `z`, turned by `phase` degrees.
    fn ring(
        symbol: &'static str,
        n: usize,
        r: f64,
        z: f64,
        phase: f64,
    ) -> Vec<(&'static str, [f64; 3])> {
        (0..n)
            .map(|k| {
                let t = (360.0 * k as f64 / n as f64 + phase).to_radians();
                (symbol, [r * t.cos(), r * t.sin(), z])
            })
            .collect()
    }

    #[test]
    fn groups_and_symmetry_numbers_of_common_molecules() {
        let c = |s: &str, n: u32| (s.to_string(), n);
        // Linear.
        assert_eq!(
            group(&[("H", [0.0; 3]), ("Cl", [0.0, 0.0, 1.27])]),
            c("C\u{221e}v", 1)
        );
        assert_eq!(
            group(&[
                ("O", [0.0, 0.0, -1.16]),
                ("C", [0.0; 3]),
                ("O", [0.0, 0.0, 1.16])
            ]),
            c("D\u{221e}h", 2)
        );
        // Water, ammonia.
        assert_eq!(
            group(&[
                ("O", [0.0, 0.0, 0.117]),
                ("H", [0.0, 0.757, -0.469]),
                ("H", [0.0, -0.757, -0.469])
            ]),
            c("C2v", 2)
        );
        let mut ammonia = vec![("N", [0.0, 0.0, 0.3])];
        ammonia.extend(ring("H", 3, 0.94, -0.1, 0.0));
        assert_eq!(group(&ammonia), c("C3v", 3));
        // Benzene.
        let mut benzene = ring("C", 6, 1.39, 0.0, 0.0);
        benzene.extend(ring("H", 6, 2.48, 0.0, 0.0));
        assert_eq!(group(&benzene), c("D6h", 12));
        // Ethane, staggered and eclipsed.
        let ethane = |twist: f64| {
            let mut rows = vec![("C", [0.0, 0.0, 0.77]), ("C", [0.0, 0.0, -0.77])];
            rows.extend(ring("H", 3, 1.03, 1.16, 0.0));
            rows.extend(ring("H", 3, 1.03, -1.16, twist));
            rows
        };
        assert_eq!(group(&ethane(60.0)), c("D3d", 6));
        assert_eq!(group(&ethane(0.0)), c("D3h", 6));
        assert_eq!(group(&ethane(20.0)), c("D3", 6));
        // Methane and its distortions.
        let t = 0.629;
        let methane = vec![
            ("C", [0.0; 3]),
            ("H", [t, t, t]),
            ("H", [t, -t, -t]),
            ("H", [-t, t, -t]),
            ("H", [-t, -t, t]),
        ];
        assert_eq!(group(&methane), c("Td", 12));
        let mut chlorinated = methane.clone();
        chlorinated[1] = ("Cl", [1.03, 1.03, 1.03]);
        assert_eq!(group(&chlorinated), c("C3v", 3));
        // An octahedron and a cube.
        let mut sf6 = vec![("S", [0.0; 3])];
        for axis in 0..3 {
            for sign in [1.56, -1.56] {
                let mut p = [0.0; 3];
                p[axis] = sign;
                sf6.push(("F", p));
            }
        }
        assert_eq!(group(&sf6), c("Oh", 24));
        // Hydrogen peroxide (C2), trans-dichloroethene (C2h), a molecule without symmetry.
        assert_eq!(
            group(&[
                ("O", [0.0, 0.73, 0.0]),
                ("O", [0.0, -0.73, 0.0]),
                ("H", [0.8, 0.9, 0.5]),
                ("H", [-0.8, -0.9, 0.5]),
            ]),
            c("C2", 2)
        );
        assert_eq!(
            group(&[
                ("C", [0.66, 0.0, 0.0]),
                ("C", [-0.66, 0.0, 0.0]),
                ("Cl", [1.5, 1.4, 0.0]),
                ("Cl", [-1.5, -1.4, 0.0]),
                ("H", [1.2, -0.95, 0.0]),
                ("H", [-1.2, 0.95, 0.0]),
            ]),
            c("C2h", 2)
        );
        assert_eq!(
            group(&[
                ("C", [0.0; 3]),
                ("H", [1.0, 0.1, 0.0]),
                ("F", [-0.4, 1.2, 0.2]),
                ("Cl", [-0.5, -0.9, 1.3]),
                ("Br", [0.1, -0.6, -1.7])
            ]),
            c("C1", 1)
        );
        // Planar molecule without rotation axes, and one with an inversion centre only.
        assert_eq!(
            group(&[
                ("O", [0.0; 3]),
                ("H", [0.96, 0.0, 0.0]),
                ("Cl", [-0.5, 1.6, 0.0])
            ]),
            c("Cs", 1)
        );
        assert_eq!(
            point_group(&molecule(&[("Ar", [0.0; 3])]), 1e-3).label,
            "Kh"
        );
    }

    #[test]
    fn icosahedral_dihedral_and_improper_groups() {
        let c = |s: &str, n: u32| (s.to_string(), n);
        // Twelve atoms at the corners of an icosahedron around a thirteenth.
        let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
        let mut cluster = vec![("B", [0.0; 3])];
        for (a, b) in [(1.0, phi), (1.0, -phi), (-1.0, phi), (-1.0, -phi)] {
            cluster.extend([("H", [0.0, a, b]), ("H", [a, b, 0.0]), ("H", [b, 0.0, a])]);
        }
        assert_eq!(group(&cluster), c("Ih", 60));
        // Ethylene.
        let ethylene = [
            ("C", [0.0, 0.0, 0.667]),
            ("C", [0.0, 0.0, -0.667]),
            ("H", [0.0, 0.923, 1.238]),
            ("H", [0.0, -0.923, 1.238]),
            ("H", [0.0, 0.923, -1.238]),
            ("H", [0.0, -0.923, -1.238]),
        ];
        assert_eq!(group(&ethylene), c("D2h", 4));
        // Allene: the two CH2 planes are perpendicular.
        let allene = [
            ("C", [0.0; 3]),
            ("C", [0.0, 0.0, 1.31]),
            ("C", [0.0, 0.0, -1.31]),
            ("H", [0.93, 0.0, 1.87]),
            ("H", [-0.93, 0.0, 1.87]),
            ("H", [0.0, 0.93, -1.87]),
            ("H", [0.0, -0.93, -1.87]),
        ];
        assert_eq!(group(&allene), c("D2d", 4));
        // An inversion centre alone, and a four-fold rotation-reflection alone.
        let ci = [
            ("C", [0.7, 0.1, 0.0]),
            ("C", [-0.7, -0.1, 0.0]),
            ("F", [1.3, 1.2, 0.4]),
            ("F", [-1.3, -1.2, -0.4]),
            ("Cl", [1.5, -1.1, -0.9]),
            ("Cl", [-1.5, 1.1, 0.9]),
        ];
        assert_eq!(group(&ci), c("Ci", 1));
        // Four arms related by a four-fold rotation-reflection; the second atom of every
        // arm is off the plane through the first, which leaves no mirror plane and no
        // perpendicular two-fold axis.
        let s4 = [
            ("C", [0.0; 3]),
            ("F", [1.2, 0.3, 0.7]),
            ("F", [-0.3, 1.2, -0.7]),
            ("F", [-1.2, -0.3, 0.7]),
            ("F", [0.3, -1.2, -0.7]),
            ("H", [1.9, 0.9, 1.0]),
            ("H", [-0.9, 1.9, -1.0]),
            ("H", [-1.9, -0.9, 1.0]),
            ("H", [0.9, -1.9, -1.0]),
        ];
        assert_eq!(group(&s4), c("S4", 2));
        // The number of operations found is the order of the group.
        assert_eq!(point_group(&molecule(&ethylene), 1e-3).operations, 8);
        assert_eq!(point_group(&molecule(&cluster), 1e-3).operations, 120);
    }

    #[test]
    fn the_tolerance_decides_what_counts_as_symmetric() {
        let rows = [
            ("O", [0.0, 0.0, 0.117]),
            ("H", [0.0, 0.757, -0.469]),
            ("H", [0.0, -0.752, -0.469]),
        ];
        let sys = molecule(&rows);
        assert_eq!(point_group(&sys, 1e-4).label, "Cs");
        let loose = point_group(&sys, 2e-2);
        assert_eq!((loose.label.as_str(), loose.tolerance), ("C2v", 2e-2));
        let rows = [
            ("C", [0.0, 0.0, 0.0]),
            ("H", [1.09, 0.0, 0.0]),
            ("H", [-0.2, 1.05, 0.2]),
            ("H", [-0.5, -0.6, 0.9]),
            ("F", [-0.4, -0.5, -1.2]),
        ];
        let distorted = molecule(&rows);
        assert_eq!(point_group(&distorted, 1e-3).label, "C1");
        // A tolerance as wide as this takes the three hydrogens for images of each other:
        // the group of the idealized molecule, named from the orders of the operations.
        let wide = point_group(&distorted, 1.5);
        assert_eq!((wide.label.as_str(), wide.symmetry_number), ("C3v", 3));
        // Operations that do not combine into a group are recognized.
        let turn = (vec![1, 2, 0, 3], false);
        let swap = (vec![1, 0, 2, 3], true);
        let identity = (vec![0, 1, 2, 3], false);
        assert!(!closed(&[identity.clone(), turn.clone(), swap]));
        let back = (vec![2, 0, 1, 3], false);
        assert!(closed(&[identity, turn, back]));
        assert_eq!(permutation_order(&[1, 2, 0, 4, 3]), 6);
    }
}
