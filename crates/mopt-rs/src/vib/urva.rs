//! Unified reaction valley approach (Kraka and Cremer; the decomposition here follows Zou,
//! Sexton, Kraka, Freindorf, Cremer, J. Chem. Theory Comput. 2016, 12, 650): which internal
//! coordinates carry the direction of a reaction path and its curvature, and into which
//! phases the curvature divides the path.
//!
//! For an internal coordinate `q_n` with the row `b_n` of the Wilson B matrix, the unit
//! vector `u_n = M^(-1/2) b_n^T / (b_n M^-1 b_n^T)^(1/2)` is the direction in mass-weighted
//! coordinates along which `q_n` changes fastest. The components `t . u_n` of the unit
//! tangent and `k . u_n` of the curvature vector `k = dt/ds` tell how much the path moves
//! along that coordinate and how much the coordinate takes part where the path bends. The
//! `u_n` are neither orthogonal nor complete, so the components do not add up to the
//! vectors; they are read coordinate by coordinate. The sign of a curvature component tells
//! whether the bend supports (same sign as the direction component) or resists the change of
//! the coordinate along the path.
//!
//! The scalar curvature is small where the path runs straight (the electronic structure
//! changes steadily) and has peaks where it turns (bonds start to break or form). The
//! minima of the curvature divide the path into reaction phases, each with the chemical
//! event of its curvature peak.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use super::{PathPoint, PathProfile};
use crate::coord::internal::{Primitive, PrimitiveOptions, PrimitiveSet};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_dot;

/// Components of the direction and the curvature of a path along internal coordinates.
#[derive(Debug, Clone)]
pub struct InternalComponents {
    /// The internal coordinates.
    pub primitives: Vec<Primitive>,
    /// Their names: `R(1-2)` bond, `A(1-2-3)` angle, `L(1-2-3)` bend of a linear
    /// arrangement, `D(1-2-3-4)` dihedral, with atom numbers starting at 1.
    pub labels: Vec<String>,
    /// `t . u_n` at every point (`[point][coordinate]`; NaN where the coordinate is not
    /// defined, e.g. a dihedral with three atoms in line).
    pub direction: Vec<Vec<f64>>,
    /// `k . u_n` at every point (`1 / (sqrt(amu) Bohr)`).
    pub curvature: Vec<Vec<f64>>,
}

/// Kind, atoms in a canonical order, and name of a primitive.
fn identity(primitive: &Primitive) -> ((u8, Vec<usize>), String) {
    let (kind, letter, atoms): (u8, &str, Vec<usize>) = match primitive {
        Primitive::Bond(a) => (0, "R", a.iter().map(|x| x.index()).collect()),
        Primitive::Angle(a) => (1, "A", a.iter().map(|x| x.index()).collect()),
        Primitive::LinearBend { atoms, .. } => (2, "L", atoms.iter().map(|x| x.index()).collect()),
        Primitive::Dihedral(a) => (3, "D", a.iter().map(|x| x.index()).collect()),
    };
    // The same coordinate read from either end.
    let reversed: Vec<usize> = atoms.iter().rev().copied().collect();
    let atoms = atoms.min(reversed);
    let numbers: Vec<String> = atoms.iter().map(|a| (a + 1).to_string()).collect();
    let label = format!("{letter}({})", numbers.join("-"));
    ((kind, atoms), label)
}

/// Name of a primitive: `R(1-2)` bond, `A(1-2-3)` angle, `L(1-2-3)` bend of a linear
/// arrangement, `D(1-2-3-4)` dihedral, with atom numbers starting at 1.
pub(crate) fn primitive_label(primitive: &Primitive) -> String {
    identity(primitive).1
}

/// The internal coordinates of a path: bonds, angles, and dihedrals of every point taken
/// together (a bond that exists at some point of the path is followed along all of it),
/// bonds first. A bend of a linear arrangement is kept once per triple of atoms.
pub fn path_primitives(points: &[PathPoint]) -> MoptResult<Vec<Primitive>> {
    let mut found: BTreeMap<(u8, Vec<usize>), Primitive> = BTreeMap::new();
    for point in points {
        if point.system.n_atoms() < 2 {
            continue;
        }
        let set = PrimitiveSet::generate(&point.system, &PrimitiveOptions::default())?;
        for primitive in set.primitives {
            found.entry(identity(&primitive).0).or_insert(primitive);
        }
    }
    Ok(found.into_values().collect())
}

/// Components of the tangent and the curvature vector of `profile` (of `points`) along
/// `primitives`.
pub fn internal_components(
    points: &[PathPoint],
    profile: &PathProfile,
    primitives: &[Primitive],
) -> MoptResult<InternalComponents> {
    if points.len() != profile.arc_length.len() {
        return Err(MoptError::config(format!(
            "internal components: {} points, but the profile has {}",
            points.len(),
            profile.arc_length.len()
        )));
    }
    let mut out = InternalComponents {
        primitives: primitives.to_vec(),
        labels: primitives.iter().map(|p| identity(p).1).collect(),
        direction: Vec::with_capacity(points.len()),
        curvature: Vec::with_capacity(points.len()),
    };
    for (i, point) in points.iter().enumerate() {
        // The masses the profile is weighted with.
        let masses = if profile.masses.len() == point.system.n_atoms() {
            profile.masses.clone()
        } else {
            point.system.masses_amu()
        };
        let (mut direction, mut curvature) = (Vec::new(), Vec::new());
        for primitive in primitives {
            let (_, row) = primitive.value_and_gradient(&point.system);
            let weighted = faer::Col::from_fn(row.nrows(), |c| row[c] / masses[c / 3].sqrt());
            let length = col_dot(&weighted, &weighted).sqrt();
            if length > 1e-10 && length.is_finite() {
                direction.push(col_dot(&profile.tangent[i], &weighted) / length);
                curvature.push(col_dot(&profile.curvature_vector[i], &weighted) / length);
            } else {
                direction.push(f64::NAN);
                curvature.push(f64::NAN);
            }
        }
        out.direction.push(direction);
        out.curvature.push(curvature);
    }
    Ok(out)
}

/// A stretch of a path between two minima of its curvature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReactionPhase {
    /// First point.
    pub start: usize,
    /// Last point (the first of the next phase).
    pub end: usize,
    /// Point of the largest curvature in between.
    pub peak: usize,
}

/// The phases of a path with the scalar curvature `curvature` at its points: the path is
/// divided at the minima of the curvature. A minimum counts when the curvature rises on both
/// sides by more than `prominence` times the largest curvature of the path before it falls
/// below the minimum again (0: every minimum; a larger value merges the ripples of a path
/// with numerical noise).
pub fn reaction_phases(curvature: &[f64], prominence: f64) -> Vec<ReactionPhase> {
    let n = curvature.len();
    if n == 0 {
        return Vec::new();
    }
    let largest = curvature.iter().fold(0.0_f64, |m, c| m.max(*c));
    let rise = |i: usize, towards: &mut dyn Iterator<Item = usize>| -> f64 {
        let mut highest = curvature[i];
        for j in towards {
            if curvature[j] < curvature[i] {
                break;
            }
            highest = highest.max(curvature[j]);
        }
        highest - curvature[i]
    };
    let mut boundaries = vec![0];
    for i in 1..n.saturating_sub(1) {
        if curvature[i] < curvature[i - 1] && curvature[i] <= curvature[i + 1] {
            let depth = rise(i, &mut (0..i).rev()).min(rise(i, &mut (i + 1..n)));
            if depth > prominence * largest {
                boundaries.push(i);
            }
        }
    }
    if n > 1 {
        boundaries.push(n - 1);
    }
    let phase = |start: usize, end: usize| {
        let peak = (start..=end).fold(start, |best, i| {
            if curvature[i] > curvature[best] {
                i
            } else {
                best
            }
        });
        ReactionPhase { start, end, peak }
    };
    if boundaries.len() == 1 {
        return vec![phase(0, 0)];
    }
    boundaries.windows(2).map(|w| phase(w[0], w[1])).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use faer::{Col, Mat};

    use super::*;
    use crate::domain::{positions, Bohr, Element, MolecularSystem, SpinCharge};
    use crate::vib::{transverse_profile, Direction};

    /// A bent triatomic H-O-H whose first bond stretches along the path while the rest
    /// stays: the frames are in a common orientation with the centre of mass at rest.
    fn stretching_bond() -> Vec<PathPoint> {
        let frames: Vec<[[f64; 3]; 3]> = (0..5)
            .map(|k| {
                let r = 1.8 + 0.05 * k as f64;
                [[0.0, 0.0, 0.0], [r, 0.0, 0.0], [-0.45, 1.75, 0.0]]
            })
            .collect();
        let system = |p: &[[f64; 3]; 3]| {
            MolecularSystem::new(
                vec![Element::O, Element::H, Element::H],
                &positions::<Bohr>(p),
                SpinCharge::default(),
                None,
            )
            .unwrap()
        };
        frames
            .iter()
            .map(|p| PathPoint {
                system: system(p),
                // Only the first hydrogen moves, along the bond.
                tangent: Direction::Displacement(Col::from_fn(9, |i| f64::from(u8::from(i == 3)))),
                hessian: Mat::from_fn(9, 9, |i, j| f64::from(u8::from(i == j))),
            })
            .collect()
    }

    #[test]
    fn a_stretching_bond_carries_the_direction_of_the_path() {
        let points = stretching_bond();
        let primitives = path_primitives(&points).unwrap();
        // Two bonds and the angle between them.
        let labels: Vec<String> = primitives.iter().map(|p| identity(p).1).collect();
        assert_eq!(labels, vec!["R(1-2)", "R(1-3)", "A(2-1-3)"]);
        // In a fixed frame (the frames are not in the Eckart frame of each other).
        let profile = transverse_profile(&points, false).unwrap();
        let parts = internal_components(&points, &profile, &primitives).unwrap();
        let (m_o, m_h) = {
            let m = points[0].system.masses_amu();
            (m[0], m[1])
        };
        for row in &parts.direction {
            // The tangent is the motion of H2 along the bond; u of R(1-2) moves both atoms
            // against each other with the weights 1 / sqrt(m): t . u = sqrt(m_O / (m_O + m_H)).
            let expected = (m_o / (m_o + m_h)).sqrt();
            assert!((row[0] - expected).abs() < 1e-10, "{}", row[0]);
            // Neither the other bond nor the angle takes part: the hydrogen moves along its
            // own bond.
            assert!(row[1].abs() < 1e-10);
            assert!(row[2].abs() < 1e-10);
        }
        // A straight path has no curvature in any coordinate.
        assert!(parts
            .curvature
            .iter()
            .all(|row| row.iter().all(|c| c.abs() < 1e-10)));
        // Another number of points is refused.
        assert!(internal_components(&points[..3], &profile, &primitives).is_err());
    }

    #[test]
    fn phases_are_divided_at_the_minima_of_the_curvature() {
        // Two peaks with a minimum in between and a ripple on the second flank.
        let curvature = [0.0, 1.0, 3.0, 1.0, 0.5, 2.0, 5.0, 4.0, 4.1, 2.0, 0.1];
        let all = reaction_phases(&curvature, 0.0);
        assert_eq!(
            all,
            vec![
                ReactionPhase {
                    start: 0,
                    end: 4,
                    peak: 2
                },
                ReactionPhase {
                    start: 4,
                    end: 7,
                    peak: 6
                },
                ReactionPhase {
                    start: 7,
                    end: 10,
                    peak: 8
                },
            ]
        );
        // The ripple (0.1 deep, 2 % of the largest curvature) is merged at 5 %.
        let merged = reaction_phases(&curvature, 0.05);
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged[1],
            ReactionPhase {
                start: 4,
                end: 10,
                peak: 6
            }
        );
        // A path without a minimum is one phase; so is a single point.
        assert_eq!(reaction_phases(&[0.0, 1.0, 0.5], 0.0).len(), 1);
        assert_eq!(
            reaction_phases(&[2.0], 0.0),
            vec![ReactionPhase {
                start: 0,
                end: 0,
                peak: 0
            }]
        );
        assert!(reaction_phases(&[], 0.0).is_empty());
    }
}
