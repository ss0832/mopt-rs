//! Covalent bond topology (`BondConnectivity`), Grimme fractional coordination numbers,
//! molecular fragments, and geometric run conditions (`ShapeCondition`).
//!
//! MultiOptPy: `Utils/bond_connectivity.py:BondConnectivity`,
//!  `Utils/calc_tools.py:calc_bond_matrix`, `fragment_check`, and the shape conditions (`-sc`) of
//!  `optimization.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use crate::coord::primitives::{bond_angle_rad, dihedral_angle_rad};
use crate::domain::{Angstrom, AtomId, Bohr, Degree, MolecularSystem, Quantity, Radian};
use crate::error::{MoptError, MoptResult};
use crate::parameters::{
    double_covalent_radius_bohr, sum_covalent_radii_bohr, triple_covalent_radius_bohr,
};

/// Bond order from the bond length (see [`BondConnectivity::classify_bond_order`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondOrderClass {
    /// Single bond.
    Single,
    /// Aromatic bond.
    Aromatic,
    /// Double bond.
    Double,
    /// Triple bond.
    Triple,
}

/// Covalent graph topology derived from interatomic (minimum-image) distances.
#[derive(Debug, Clone)]
pub struct BondConnectivity {
    /// Number of atoms.
    pub n_atoms: usize,
    /// Bonded neighbours of each atom (indexed by `AtomId::index`).
    pub adjacency: Vec<Vec<AtomId>>,
    /// Bonded pairs `i < j`.
    pub bonds: Vec<[AtomId; 2]>,
    /// Bond angles `a - center - c` with `a < c`.
    pub angles: Vec<[AtomId; 3]>,
    /// Proper dihedrals `a - b - c - d` (both bends below 175.4 degrees).
    pub dihedrals: Vec<[AtomId; 4]>,
    /// Out-of-plane quadruples `center, a, b, c` (atoms with three neighbours).
    pub out_of_planes: Vec<[AtomId; 4]>,
    /// Connected components of the bond graph.
    pub fragments: Vec<Vec<AtomId>>,
}

/// Bends above `pi - 0.08` rad are treated as collinear when enumerating dihedrals.
const COLLINEAR_RAD: f64 = std::f64::consts::PI - 0.08;

fn angle(sys: &MolecularSystem, a: AtomId, b: AtomId, c: AtomId) -> f64 {
    bond_angle_rad(
        sys.position(a).value(),
        sys.position(b).value(),
        sys.position(c).value(),
        sys.active_cell(),
    )
}

impl BondConnectivity {
    /// Bonds where `0.1 Bohr < r_ij <= scale * (r_cov(i) + r_cov(j))`; dummy atoms are skipped.
    pub fn from_system(sys: &MolecularSystem, scale: f64) -> Self {
        let n = sys.n_atoms();
        let mut adjacency: Vec<Vec<AtomId>> = vec![Vec::new(); n];
        let mut bonds = Vec::new();
        for i in sys.atoms() {
            if sys.element(i).is_dummy() {
                continue;
            }
            for j in sys.atoms().skip(i.index() + 1) {
                if sys.element(j).is_dummy() {
                    continue;
                }
                let dist = sys.distance(i, j).value();
                let cov_sum = sum_covalent_radii_bohr(sys.element(i), sys.element(j));
                if dist > 0.1 && dist <= scale * cov_sum {
                    adjacency[i.index()].push(j);
                    adjacency[j.index()].push(i);
                    bonds.push([i, j]);
                }
            }
        }

        let mut angles = Vec::new();
        for (b, nbrs) in sys.atoms().zip(&adjacency) {
            for (ia, &a) in nbrs.iter().enumerate() {
                for &c in &nbrs[ia + 1..] {
                    angles.push([a.min(c), b, a.max(c)]);
                }
            }
        }

        let mut dihedrals = Vec::new();
        for &[b, c] in &bonds {
            for &a in &adjacency[b.index()] {
                if a == c || angle(sys, a, b, c) > COLLINEAR_RAD {
                    continue;
                }
                for &d in &adjacency[c.index()] {
                    if d == b || d == a || angle(sys, b, c, d) > COLLINEAR_RAD {
                        continue;
                    }
                    dihedrals.push([a, b, c, d]);
                }
            }
        }

        let out_of_planes = sys
            .atoms()
            .zip(&adjacency)
            .filter(|(_, nbrs)| nbrs.len() == 3)
            .map(|(center, nbrs)| [center, nbrs[0], nbrs[1], nbrs[2]])
            .collect();

        let mut visited = vec![false; n];
        let mut fragments = Vec::new();
        for start in sys.atoms() {
            if visited[start.index()] {
                continue;
            }
            let mut frag = Vec::new();
            let mut queue = VecDeque::from([start]);
            visited[start.index()] = true;
            while let Some(u) = queue.pop_front() {
                frag.push(u);
                for &v in &adjacency[u.index()] {
                    if !visited[v.index()] {
                        visited[v.index()] = true;
                        queue.push_back(v);
                    }
                }
            }
            fragments.push(frag);
        }

        Self {
            n_atoms: n,
            adjacency,
            bonds,
            angles,
            dihedrals,
            out_of_planes,
            fragments,
        }
    }

    /// Neighbours of `atom`.
    pub fn neighbors(&self, atom: AtomId) -> &[AtomId] {
        &self.adjacency[atom.index()]
    }

    /// Approximate bond order `(Single, Aromatic, Double, Triple)` from the interatomic distance.
    pub fn classify_bond_order(
        &self,
        sys: &MolecularSystem,
        i: AtomId,
        j: AtomId,
    ) -> BondOrderClass {
        let d = sys.distance(i, j).value();
        let (ei, ej) = (sys.element(i), sys.element(j));
        let r1 = sum_covalent_radii_bohr(ei, ej);
        let r2 = double_covalent_radius_bohr(ei) + double_covalent_radius_bohr(ej);
        let r3 = triple_covalent_radius_bohr(ei) + triple_covalent_radius_bohr(ej);
        if d <= 0.5 * (r2 + r3) {
            BondOrderClass::Triple
        } else if d <= 0.5 * (r1 + r2) - 0.04 {
            BondOrderClass::Double
        } else if d <= 0.5 * (r1 + r2) + 0.05 {
            BondOrderClass::Aromatic
        } else {
            BondOrderClass::Single
        }
    }
}

/// Comparison of a [`ShapeCondition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeCmp {
    /// The value must stay greater than the threshold (`gt`).
    GreaterThan,
    /// The value must stay less than the threshold (`lt`).
    LessThan,
}

/// Monitored coordinate of a [`ShapeCondition`].
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeMetric {
    /// Interatomic distance.
    Bond {
        /// Atoms.
        atoms: [AtomId; 2],
        /// Threshold.
        threshold: Quantity<Angstrom>,
    },
    /// Bond angle.
    Angle {
        /// Atoms (`a - center - c`).
        atoms: [AtomId; 3],
        /// Threshold.
        threshold: Quantity<Degree>,
    },
    /// Signed dihedral angle in degrees, in MultiOptPy's sign convention (opposite to IUPAC).
    Dihedral {
        /// Atoms.
        atoms: [AtomId; 4],
        /// Threshold.
        threshold: Quantity<Degree>,
    },
}

/// Geometric run condition (`-sc`): the run continues while every condition holds and stops
/// as soon as one is violated (MultiOptPy `judge_shape_condition`).
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeCondition {
    /// Comparison.
    pub cmp: ShapeCmp,
    /// Monitored coordinate and threshold.
    pub metric: ShapeMetric,
}

impl ShapeCondition {
    /// Parse `-sc` arguments: triples `value gt|lt atoms`, where `atoms` is a comma-separated
    /// list of 2 (bond, Angstrom), 3 (angle, degrees), or 4 (dihedral, degrees) 1-based atom
    /// numbers, e.g. `["2.5", "gt", "2,3", "120", "lt", "1,2,3"]`.
    pub fn parse_cli(args: &[String]) -> MoptResult<Vec<Self>> {
        if !args.len().is_multiple_of(3) {
            return Err(MoptError::parse(format!(
                "-sc expects triples 'value gt|lt atoms', got {} arguments",
                args.len()
            )));
        }
        args.chunks(3)
            .map(|t| Self::parse_triple(&t[0], &t[1], &t[2]))
            .collect()
    }

    fn parse_triple(value: &str, cmp: &str, atoms: &str) -> MoptResult<Self> {
        let v = value
            .trim()
            .parse::<f64>()
            .map_err(|e| MoptError::parse(format!("invalid -sc threshold '{value}': {e}")))?;
        let cmp = match cmp.trim() {
            "gt" => ShapeCmp::GreaterThan,
            "lt" => ShapeCmp::LessThan,
            other => {
                return Err(MoptError::parse(format!(
                    "-sc condition must be 'gt' or 'lt', got '{other}'"
                )))
            }
        };
        let ids = atoms
            .split(',')
            .map(AtomId::parse_one_based)
            .collect::<MoptResult<Vec<_>>>()?;
        let metric = match ids[..] {
            [a, b] => ShapeMetric::Bond {
                atoms: [a, b],
                threshold: Quantity::new(v),
            },
            [a, b, c] => ShapeMetric::Angle {
                atoms: [a, b, c],
                threshold: Quantity::new(v),
            },
            [a, b, c, d] => ShapeMetric::Dihedral {
                atoms: [a, b, c, d],
                threshold: Quantity::new(v),
            },
            _ => {
                return Err(MoptError::parse(format!(
                    "-sc atoms must list 2, 3, or 4 atoms, got '{atoms}'"
                )))
            }
        };
        Ok(Self { cmp, metric })
    }

    /// Current value of the monitored coordinate (Angstrom or degrees) and its threshold.
    fn value_and_threshold(&self, sys: &MolecularSystem) -> MoptResult<(f64, f64)> {
        let n = sys.n_atoms();
        let p = |a: AtomId| -> MoptResult<[f64; 3]> { Ok(sys.position(a.check(n)?).value()) };
        let deg = |rad: f64| Quantity::<Radian>::new(rad).to::<Degree>().value();
        let cell = sys.active_cell();
        Ok(match &self.metric {
            ShapeMetric::Bond {
                atoms: [a, b],
                threshold,
            } => {
                let d = Quantity::<Bohr>::new(sys.distance(a.check(n)?, b.check(n)?).value());
                (d.to::<Angstrom>().value(), threshold.value())
            }
            ShapeMetric::Angle {
                atoms: [a, b, c],
                threshold,
            } => (
                deg(bond_angle_rad(p(*a)?, p(*b)?, p(*c)?, cell)),
                threshold.value(),
            ),
            ShapeMetric::Dihedral {
                atoms: [a, b, c, d],
                threshold,
            } => (
                // MultiOptPy measures dihedrals with the opposite sign of the IUPAC convention
                // used elsewhere in mopt-rs; -sc thresholds keep MultiOptPy's sign.
                -deg(dihedral_angle_rad(p(*a)?, p(*b)?, p(*c)?, p(*d)?, cell)),
                threshold.value(),
            ),
        })
    }

    /// Whether the condition holds for `sys`.
    pub fn holds(&self, sys: &MolecularSystem) -> MoptResult<bool> {
        let (v, t) = self.value_and_threshold(sys)?;
        Ok(match self.cmp {
            ShapeCmp::GreaterThan => v > t,
            ShapeCmp::LessThan => v < t,
        })
    }
}

/// Whether a run must stop because some condition is violated.
pub fn any_violated(conditions: &[ShapeCondition], sys: &MolecularSystem) -> MoptResult<bool> {
    for c in conditions {
        if !c.holds(sys)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{positions, Element, SpinCharge};

    fn h2o2() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::H, Element::O, Element::O, Element::H],
            &positions::<Angstrom>(&[
                [0.9, 0.3, 0.6],
                [0.0, 0.7, 0.0],
                [0.0, -0.7, 0.0],
                [-0.8, -0.9, 0.5],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    /// MultiOptPy's `calc_dihedral_angle_from_vec` (degrees).
    fn reference_dihedral(p: [[f64; 3]; 4]) -> f64 {
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let (v1, v2, v3) = (sub(p[0], p[1]), sub(p[1], p[2]), sub(p[2], p[3]));
        let (n1, n2) = (cross(v1, v2), cross(v2, v3));
        let (l1, l2) = (dot(n1, n1).sqrt(), dot(n2, n2).sqrt());
        let mut ang = (dot(n1, n2) / (l1 * l2)).acos().abs();
        let s = dot(cross(n1.map(|x| x / l1), n2.map(|x| x / l2)), v2).signum();
        if s != 0.0 {
            ang *= -s;
        }
        ang.to_degrees()
    }

    #[test]
    fn graph_and_fragments() {
        let c = BondConnectivity::from_system(&h2o2(), 1.25);
        assert_eq!(c.bonds.len(), 3);
        assert_eq!(c.dihedrals.len(), 1);
        assert_eq!(c.fragments.len(), 1);
        assert_eq!(c.neighbors(AtomId::new(1)).len(), 2);
    }

    #[test]
    fn shape_conditions_follow_multioptpy() {
        let sys = h2o2();
        let p: [[f64; 3]; 4] = std::array::from_fn(|i| sys.position(AtomId::new(i)).value());
        let ours = Quantity::<Radian>::new(dihedral_angle_rad(p[0], p[1], p[2], p[3], None))
            .to::<Degree>()
            .value();
        assert!(
            (ours + reference_dihedral(p)).abs() < 1e-9,
            "{ours} vs {}",
            reference_dihedral(p)
        );

        let args: Vec<String> = ["1.0", "gt", "2,3", "10", "lt", "1,2,3,4"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let conds = ShapeCondition::parse_cli(&args).unwrap();
        assert!(conds[0].holds(&sys).unwrap()); // O-O = 1.4 A > 1.0
        assert_eq!(conds[1].holds(&sys).unwrap(), reference_dihedral(p) < 10.0);
        let stop = ["1.5", "gt", "2,3"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        assert!(any_violated(&ShapeCondition::parse_cli(&stop).unwrap(), &sys).unwrap());
        assert!(ShapeCondition::parse_cli(&["1.0".into(), "gt".into()]).is_err());
        assert!(ShapeCondition::parse_cli(&["1.0".into(), "gt".into(), "0,2".into()]).is_err());
    }
}
