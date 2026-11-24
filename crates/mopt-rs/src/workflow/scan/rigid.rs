//! Structures of a rigid scan (single points without relaxation): the scanned coordinates set
//! to their values by moving the atoms on one side of each rigidly, as a scan of a Z-matrix
//! coordinate does.
//!
//! The bond graph of the input structure (covalent radii times 1.2) says what is on which side:
//!
//! - a distance `a-b`: the atoms on the side of `b` (reached from `b` without passing `a`) are
//!   translated along `a -> b`; of two groups (`fbond`), the side of the second group;
//! - an angle `a-b-c`: the side of `c` (reached without passing `b`) is turned about the
//!   normal of the plane `a, b, c` through `b`;
//! - a dihedral angle `a-b-c-d`: the side of `c` (reached without passing `b`) is turned about
//!   the axis `b -> c`.
//!
//! The side moved is the smaller of the two ends where both are possible. Where the cut does
//! not divide the structure (a ring, a coordinate inside one fragment that the graph keeps
//! together on both sides), and for the other kinds (Cartesian components, user functions),
//! the atoms are moved by the shortest displacement that sets the value (the Newton projection
//! of the constraints, [`crate::pes::constraint`]). After the moves of all coordinates the
//! Newton projection holds them all together.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `relaxedscan` always optimizes).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use crate::coord::connectivity::BondConnectivity;
use crate::domain::{AtomId, MolecularSystem, Position};
use crate::error::MoptResult;
use crate::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};

/// Scale of the covalent radii of the bond graph.
pub const BOND_SCALE: f64 = 1.2;
/// Passes over the coordinates (a move of one can change another).
const PASSES: usize = 3;
/// Largest residual (Bohr, rad) after which the moves stop.
const TOLERANCE: f64 = 1e-10;

/// The atoms reached from `from` in the graph without passing the atoms of `blocked`; `None`
/// when one of `blocked` is reached (the cut does not divide the graph).
fn side(bonds: &BondConnectivity, from: &[AtomId], blocked: &[AtomId]) -> Option<Vec<AtomId>> {
    let mut seen = vec![false; bonds.n_atoms];
    let mut queue: VecDeque<AtomId> = VecDeque::new();
    for &a in from {
        if !seen[a.index()] {
            seen[a.index()] = true;
            queue.push_back(a);
        }
    }
    let mut out = Vec::new();
    while let Some(u) = queue.pop_front() {
        out.push(u);
        for &v in bonds.neighbors(u) {
            if blocked.contains(&v) {
                if from.contains(&u) {
                    // The cut bond itself.
                    continue;
                }
                // Another way back to the fixed atoms: a ring.
                return None;
            }
            if !seen[v.index()] {
                seen[v.index()] = true;
                queue.push_back(v);
            }
        }
    }
    Some(out)
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

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(a);
    (n > 1e-12).then(|| [a[0] / n, a[1] / n, a[2] / n])
}

/// Rodrigues' rotation of `p` about the unit `axis` through `origin` by `angle`.
fn rotate(p: [f64; 3], origin: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let v = sub(p, origin);
    let (c, s) = (angle.cos(), angle.sin());
    let k_cross_v = cross(axis, v);
    let k_dot_v = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = origin[i] + v[i] * c + k_cross_v[i] * s + axis[i] * k_dot_v * (1.0 - c);
    }
    out
}

/// The residual `value - target` of one constraint at `sys` (periodic ones wrapped).
fn residual(sys: &MolecularSystem, constraint: &Constraint) -> MoptResult<f64> {
    let set = ConstraintSet::new(std::slice::from_ref(constraint), sys, None)?;
    Ok(set.residuals(sys)?[0])
}

fn centroid(sys: &MolecularSystem, atoms: &[AtomId]) -> [f64; 3] {
    let mut c = [0.0; 3];
    let origin = sys.position(atoms[0]).value();
    for &a in atoms {
        // Each atom at its image nearest to the first.
        let d = sys.delta(a, atoms[0]).value();
        for k in 0..3 {
            c[k] += (origin[k] + d[k]) / atoms.len() as f64;
        }
    }
    c
}

/// The atoms that move and how, for one coordinate: `(atoms, kind of move)`.
enum Move {
    Translate(Vec<AtomId>, [f64; 3]),
    Rotate(Vec<AtomId>, [f64; 3], [f64; 3]),
}

fn smaller(a: Option<Vec<AtomId>>, b: Option<Vec<AtomId>>) -> Option<(Vec<AtomId>, bool)> {
    match (a, b) {
        (Some(a), Some(b)) if b.len() < a.len() => Some((b, true)),
        (Some(a), _) => Some((a, false)),
        (None, Some(b)) => Some((b, true)),
        (None, None) => None,
    }
}

/// The move that changes `kind` at `sys`, and whether it acts on the first end (so that it
/// runs the other way).
fn plan(
    sys: &MolecularSystem,
    bonds: &BondConnectivity,
    kind: &ConstraintKind,
) -> Option<(Move, bool)> {
    let p = |a: AtomId| sys.position(a).value();
    match kind {
        ConstraintKind::Distance(a, b) => {
            let (moving, first) = smaller(side(bonds, b, a), side(bonds, a, b))?;
            let direction = unit(sub(centroid(sys, b), centroid(sys, a)))?;
            Some((Move::Translate(moving, direction), first))
        }
        ConstraintKind::Angle([a, b, c]) => {
            let (moving, first) = smaller(side(bonds, &[*c], &[*b]), side(bonds, &[*a], &[*b]))?;
            if moving.contains(b) || (moving.contains(a) && moving.contains(c)) {
                return None;
            }
            let (u, v) = (sys.delta(*a, *b).value(), sys.delta(*c, *b).value());
            let axis = unit(cross(u, v)).or_else(|| {
                // Three atoms on a line: any normal of the line.
                let trial = if u[0].abs() < 0.9 * norm(u) {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                unit(cross(u, trial))
            })?;
            Some((Move::Rotate(moving, p(*b), axis), first))
        }
        ConstraintKind::Dihedral([a, b, c, d]) => {
            let (moving, first) = smaller(side(bonds, &[*c], &[*b]), side(bonds, &[*b], &[*c]))?;
            if moving.contains(a) && moving.contains(d) {
                return None;
            }
            let axis = unit(sys.delta(*c, *b).value())?;
            Some((Move::Rotate(moving, p(*b), axis), first))
        }
        _ => None,
    }
}

fn apply(sys: &MolecularSystem, m: &Move, amount: f64) -> MolecularSystem {
    let mut out = sys.clone();
    match m {
        Move::Translate(atoms, direction) => {
            for &a in atoms {
                let q = sys.position(a).value();
                out.set_position(
                    a,
                    Position::new([
                        q[0] + amount * direction[0],
                        q[1] + amount * direction[1],
                        q[2] + amount * direction[2],
                    ]),
                );
            }
        }
        Move::Rotate(atoms, origin, axis) => {
            for &a in atoms {
                out.set_position(
                    a,
                    Position::new(rotate(sys.position(a).value(), *origin, *axis, amount)),
                );
            }
        }
    }
    out
}

/// `sys` moved rigidly onto one constraint: the move is linear in its amount for a distance
/// and an angle of rotation, so a secant on the residual finds it in a few tries.
fn move_one(
    sys: &MolecularSystem,
    bonds: &BondConnectivity,
    constraint: &Constraint,
) -> MoptResult<Option<MolecularSystem>> {
    let Some((m, first)) = plan(sys, bonds, &constraint.kind) else {
        return Ok(None);
    };
    let r0 = residual(sys, constraint)?;
    if r0.abs() < TOLERANCE {
        return Ok(Some(sys.clone()));
    }
    // The change of the coordinate per unit of the move, from a small trial.
    let sign = if first { -1.0 } else { 1.0 };
    let mut amount = -sign * r0;
    let mut moved = apply(sys, &m, amount);
    for _ in 0..20 {
        let r = residual(&moved, constraint)?;
        if r.abs() < TOLERANCE {
            return Ok(Some(moved));
        }
        let probe = 1e-6;
        let rp = residual(&apply(sys, &m, amount + probe), constraint)?;
        let slope = (rp - r) / probe;
        if slope.abs() < 1e-12 {
            return Ok(None);
        }
        amount -= r / slope;
        moved = apply(sys, &m, amount);
    }
    Ok((residual(&moved, constraint)?.abs() < 1e-8).then_some(moved))
}

/// `sys` with every constraint of `constraints` at its target, by rigid moves where the bond
/// graph `bonds` allows them and the Newton projection of all constraints after them.
pub fn set_rigidly(
    sys: &MolecularSystem,
    bonds: &BondConnectivity,
    constraints: &[Constraint],
) -> MoptResult<MolecularSystem> {
    let mut s = sys.clone();
    for _ in 0..PASSES {
        for c in constraints {
            if let Some(moved) = move_one(&s, bonds, c)? {
                s = moved;
            }
        }
    }
    let set = ConstraintSet::new(constraints, sys, None)?;
    set.restore(&s)
}

/// The bond graph of `sys` used to divide it.
pub fn bond_graph(sys: &MolecularSystem) -> BondConnectivity {
    BondConnectivity::from_system(sys, BOND_SCALE)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::domain::{Element, SpinCharge, Units};
    use faer::Col;

    /// Ethane (Angstrom), staggered.
    fn ethane() -> MolecularSystem {
        let a = [
            ("C", [0.0, 0.0, 0.0]),
            ("C", [1.53, 0.0, 0.0]),
            ("H", [-0.39, 1.03, 0.0]),
            ("H", [-0.39, -0.51, 0.89]),
            ("H", [-0.39, -0.51, -0.89]),
            ("H", [1.92, -1.03, 0.0]),
            ("H", [1.92, 0.51, 0.89]),
            ("H", [1.92, 0.51, -0.89]),
        ];
        let elements = a
            .iter()
            .map(|(s, _)| Element::parse_token(s).unwrap())
            .collect();
        let flat = Col::from_fn(24, |k| a[k / 3].1[k % 3] * Units::ANGSTROM_TO_BOHR);
        MolecularSystem::from_flat_bohr(elements, &flat, SpinCharge::default(), None).unwrap()
    }

    #[test]
    fn a_torsion_turns_a_methyl_group_rigidly() {
        let sys = ethane();
        let bonds = bond_graph(&sys);
        let torsion = Constraint {
            kind: ConstraintKind::Dihedral([
                AtomId::new(2),
                AtomId::new(0),
                AtomId::new(1),
                AtomId::new(5),
            ]),
            target: Some(60.0_f64.to_radians()),
            anchor: None,
        };
        let turned = set_rigidly(&sys, &bonds, std::slice::from_ref(&torsion)).unwrap();
        assert!(residual(&turned, &torsion).unwrap().abs() < 1e-9);
        // Every bond length and every angle within a methyl group is kept.
        for (a, b) in [(0, 1), (1, 5), (1, 6), (1, 7), (0, 2), (5, 6), (6, 7)] {
            let before = sys.distance(AtomId::new(a), AtomId::new(b)).value();
            let after = turned.distance(AtomId::new(a), AtomId::new(b)).value();
            assert!((before - after).abs() < 1e-9, "{a}-{b}: {before} {after}");
        }
        // A bond stretched: the smaller side (one hydrogen) moves along it.
        let stretch = Constraint {
            kind: ConstraintKind::Distance(vec![AtomId::new(0)], vec![AtomId::new(2)]),
            target: Some(1.4 * Units::ANGSTROM_TO_BOHR),
            anchor: None,
        };
        let longer = set_rigidly(&sys, &bonds, std::slice::from_ref(&stretch)).unwrap();
        assert!(residual(&longer, &stretch).unwrap().abs() < 1e-9);
        assert_eq!(
            longer.position(AtomId::new(0)).value(),
            sys.position(AtomId::new(0)).value()
        );
        // An angle opened at the first carbon.
        let angle = Constraint {
            kind: ConstraintKind::Angle([AtomId::new(2), AtomId::new(0), AtomId::new(1)]),
            target: Some(100.0_f64.to_radians()),
            anchor: None,
        };
        let bent = set_rigidly(&sys, &bonds, std::slice::from_ref(&angle)).unwrap();
        assert!(residual(&bent, &angle).unwrap().abs() < 1e-9);
        let cc = |s: &MolecularSystem| s.distance(AtomId::new(0), AtomId::new(1)).value();
        assert!((cc(&bent) - cc(&sys)).abs() < 1e-9);
    }
}
