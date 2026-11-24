//! O1NumHess for a molecule: distances, model Hessian and starting directions.
//!
//! - Distance of two atoms: their separation minus their van der Waals radii (half the UFF
//!   van der Waals distances), the same for all nine pairs of their coordinates; minimum-image
//!   separations in periodic cells.
//! - Model Hessian: the Swart model as modified for O1NumHess.
//! - Starting directions: translations (the gradient does not change), rotations about the
//!   principal axes of the atoms with equal masses (the gradient turns with the molecule, so
//!   its derivative is `a x g_i` divided by the length of the rotation vector), and the
//!   breathing mode `r_i - centroid`, differentiated with points on both sides. Rotations are
//!   left out in periodic cells and for surfaces in a fixed frame, translations for the
//!   latter.
//!
//! MultiOptPy: `ModelHessian/o1numhess.py:O1NumHessCalculator.compute_hessian` (which builds
//!  the neighbour list from covalent radii and differentiates rotations numerically).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{InitialDirection, O1Problem};
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::math::linalg::{col_norm, symmetric_eigen};
use crate::model_hessian::{ModelHessian, SwartO1};
use crate::parameters::uff_vdw_distance_bohr;
use crate::pes::Calculator;

/// A principal moment below this (Bohr^2, equal masses) belongs to the axis of a linear
/// molecule (the reference's threshold).
const LINEAR_MOMENT: f64 = 1e-4;

/// Motions that leave the energy unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Invariance {
    /// Rigid translations.
    pub translation: bool,
    /// Rigid rotations.
    pub rotation: bool,
}

impl Invariance {
    /// The invariances of `calculator` for `sys`: none for a surface in a fixed frame, no
    /// rotations in a periodic cell.
    pub fn of(calculator: &dyn Calculator, sys: &MolecularSystem) -> Self {
        let rigid = calculator.is_rigid_body_invariant();
        Self {
            translation: rigid,
            rotation: rigid && !sys.has_pbc(),
        }
    }
}

/// Distances between the van der Waals spheres of the atoms, for every pair of coordinates
/// (`3N x 3N`, Bohr).
pub fn vdw_distance_matrix(sys: &MolecularSystem) -> Mat<f64> {
    let n = sys.n_atoms();
    let radius: Vec<f64> = sys
        .elements
        .iter()
        .map(|&e| 0.5 * uff_vdw_distance_bohr(e))
        .collect();
    let mut atoms = Mat::<f64>::zeros(n, n);
    for (i, a) in sys.atoms().enumerate() {
        for (j, b) in sys.atoms().enumerate() {
            let separation = if i == j {
                0.0
            } else {
                sys.distance(a, b).value()
            };
            atoms[(i, j)] = separation - radius[i] - radius[j];
        }
    }
    Mat::from_fn(3 * n, 3 * n, |p, q| atoms[(p / 3, q / 3)])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Stack `f(atom)` over the atoms.
fn stacked(n_atoms: usize, f: impl Fn(usize) -> [f64; 3]) -> Col<f64> {
    let rows: Vec<[f64; 3]> = (0..n_atoms).map(f).collect();
    Col::from_fn(3 * n_atoms, |k| rows[k / 3][k % 3])
}

/// The problem O1NumHess solves for `sys` with the gradient `gradient` there.
pub fn molecular_problem(
    sys: &MolecularSystem,
    gradient: &Col<f64>,
    invariance: Invariance,
) -> MoptResult<O1Problem> {
    let n = sys.n_atoms();
    let centroid = sys.centroid().value();
    let from_centroid: Vec<[f64; 3]> = sys
        .atoms()
        .map(|a| {
            let r = sys.position(a).value();
            [r[0] - centroid[0], r[1] - centroid[1], r[2] - centroid[2]]
        })
        .collect();
    let mut initial = Vec::new();
    if invariance.translation {
        for axis in 0..3 {
            initial.push(InitialDirection {
                vector: Col::from_fn(3 * n, |k| f64::from(u8::from(k % 3 == axis))),
                derivative: Some(Col::zeros(3 * n)),
                double_sided: false,
            });
        }
    }
    if invariance.rotation && n > 1 {
        let mut inertia = Mat::<f64>::zeros(3, 3);
        for d in &from_centroid {
            let square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            for a in 0..3 {
                inertia[(a, a)] += square;
                for b in 0..3 {
                    inertia[(a, b)] -= d[a] * d[b];
                }
            }
        }
        let (moments, axes) = symmetric_eigen(&inertia)?;
        for k in (0..3).filter(|&k| moments[k] >= LINEAR_MOMENT) {
            let axis = [axes[(0, k)], axes[(1, k)], axes[(2, k)]];
            let vector = stacked(n, |i| cross(axis, from_centroid[i]));
            let length = col_norm(&vector);
            let turned = stacked(n, |i| {
                cross(
                    axis,
                    [gradient[3 * i], gradient[3 * i + 1], gradient[3 * i + 2]],
                )
            });
            initial.push(InitialDirection {
                vector,
                derivative: Some(Col::from_fn(3 * n, |p| turned[p] / length)),
                double_sided: false,
            });
        }
    }
    let breathing = stacked(n, |i| from_centroid[i]);
    if col_norm(&breathing) > 1e-12 {
        initial.push(InitialDirection {
            vector: breathing,
            derivative: None,
            double_sided: true,
        });
    }
    Ok(O1Problem {
        x: sys.coords_flat_bohr(),
        gradient: gradient.clone(),
        distance: vdw_distance_matrix(sys),
        model: SwartO1::default().hessian(sys, gradient)?,
        initial,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Angstrom, Element, SpinCharge, Vector3};

    fn water() -> MolecularSystem {
        let el = ["O", "H", "H"].map(|s| Element::parse_token(s).unwrap());
        let xyz = [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ];
        let pos: Vec<_> = xyz.iter().map(|p| Vector3::<Angstrom>::new(*p)).collect();
        MolecularSystem::new(el.to_vec(), &pos, SpinCharge::default(), None).unwrap()
    }

    #[test]
    fn rotations_carry_the_derivative_of_a_turning_gradient() {
        let sys = water();
        let g = Col::from_fn(9, |k| {
            0.01 * (k as f64 + 1.0) * if k % 2 == 0 { 1.0 } else { -1.0 }
        });
        let both = Invariance {
            translation: true,
            rotation: true,
        };
        let problem = molecular_problem(&sys, &g, both).unwrap();
        // Three translations, three rotations, breathing.
        assert_eq!(problem.initial.len(), 7);
        assert!(problem.initial[6].double_sided && problem.initial[6].derivative.is_none());
        let x = sys.coords_flat_bohr();
        for rotation in &problem.initial[3..6] {
            let length = col_norm(&rotation.vector);
            let derivative = rotation.derivative.as_ref().unwrap();
            // The gradient turns like the positions: g . (dx/ds) and x . (dg/ds) are equal and
            // opposite for a vector field that follows the rotation.
            let work: f64 = (0..9).map(|k| g[k] * rotation.vector[k] / length).sum();
            let back: f64 = (0..9).map(|k| x[k] * derivative[k]).sum();
            let centroid_term: f64 = {
                let c = sys.centroid().value();
                (0..9).map(|k| c[k % 3] * derivative[k]).sum()
            };
            assert!((work + back - centroid_term).abs() < 1e-12);
        }
        // Atoms of a molecule are neighbours of each other.
        let d = vdw_distance_matrix(&sys);
        assert!((0..9).all(|p| (0..9).all(|q| d[(p, q)] < 1.0)));
        assert!((d[(0, 3)] - d[(2, 5)]).abs() < 1e-15);
    }
}
