//! Position and orientation of the structures of a chain.
//!
//! After every step MultiOptPy moves each structure to its centroid and turns structure
//! `k` onto structure `k + 1` (the rotation that minimizes the distance between the two,
//! Kabsch), going through the chain from the first image to the last but one. Structure
//! `k + 1` is turned onto `k + 2` afterwards, so the chain is not left aligned pair by
//! pair: every structure is aligned with where its neighbour was before that neighbour
//! was turned. The last structure is moved to its centroid only. This is kept as it is:
//! the distances along the chain, the limits of the steps, and the directions that are
//! removed from the gradients all depend on it.
//!
//! That is [`align_chain_in_sequence`], used under `Compatibility::MultiOptPy`. It has two
//! drawbacks. A structure is aligned with a neighbour that is turned right afterwards, so
//! that the chain keeps turning from iteration to iteration; and what is kept for an image
//! from one iteration to the next (its Hessian, its last gradient, its last step) stays
//! in the old orientation while the image is turned. For a small molecule that takes long
//! steps the turns are large (HCN to HNC, 0.5 Angstrom between the images: the free end
//! image, a minimization of HCN, kept forces of 0.1 to 0.4 Hartree/Bohr for 60
//! iterations). The default, [`align_chain_onto_previous`], centres every structure and
//! turns structure `k` onto structure `k - 1` as that one lies after its own turn, from
//! the second image on; a chain that is aligned stays as it is. It returns the motion of
//! every structure ([`RigidMotion`]), with which the relaxation turns what it keeps for
//! the images.
//!
//! Structures in a periodic cell are not turned, since the cell does not turn with them:
//! they are shifted along the periodic directions only ([`centre_chain_at`],
//! [`free_shifts`]). MultiOptPy has no chains in periodic cells.
//!
//! MultiOptPy: `neb.py:NEB._apply_constraints`, `Utils/calc_tools.py:kabsch_algorithm`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{centroid, BLOCK_ATOM};
use crate::coord::alignment::proper_rotation_from_covariance;
use crate::domain::PeriodicCell;

/// `point` moved to its centroid.
fn centred(point: &Col<f64>) -> Col<f64> {
    let c = centroid(point, BLOCK_ATOM);
    Col::from_fn(point.nrows(), |k| point[k] - c[k % BLOCK_ATOM])
}

/// The rotation that turns the centred structure `mobile` onto the centred structure
/// `reference`.
fn rotation_onto(mobile: &Col<f64>, reference: &Col<f64>) -> [[f64; 3]; 3] {
    let atoms = mobile.nrows() / BLOCK_ATOM;
    let mut covariance = Mat::<f64>::zeros(3, 3);
    for atom in 0..atoms {
        for r in 0..3 {
            for c in 0..3 {
                covariance[(r, c)] += mobile[3 * atom + r] * reference[3 * atom + c];
            }
        }
    }
    let rotation = proper_rotation_from_covariance(&covariance);
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, value) in row.iter_mut().enumerate() {
            *value = rotation[(r, c)];
        }
    }
    out
}

/// `vector` turned by `rotation`, atom by atom.
fn turned(rotation: &[[f64; 3]; 3], vector: &Col<f64>) -> Col<f64> {
    Col::from_fn(vector.nrows(), |k| {
        let (atom, r) = (k / 3, k % 3);
        (0..3).map(|c| rotation[r][c] * vector[3 * atom + c]).sum()
    })
}

/// The centred structure `mobile` turned onto the centred structure `reference`.
fn turned_onto(mobile: &Col<f64>, reference: &Col<f64>) -> Col<f64> {
    turned(&rotation_onto(mobile, reference), mobile)
}

/// A rigid motion of a structure: every atom `x` goes to `R (x - c)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidMotion {
    /// The rotation `R`.
    pub rotation: [[f64; 3]; 3],
    /// The point `c` that goes to the origin.
    pub centre: [f64; 3],
}

impl RigidMotion {
    /// The structure `point` after the motion.
    pub fn point(&self, point: &Col<f64>) -> Col<f64> {
        let shifted = Col::from_fn(point.nrows(), |k| point[k] - self.centre[k % BLOCK_ATOM]);
        turned(&self.rotation, &shifted)
    }

    /// A vector that belongs to the structure (a gradient, a force, a step) after the
    /// motion.
    pub fn vector(&self, vector: &Col<f64>) -> Col<f64> {
        turned(&self.rotation, vector)
    }

    /// A matrix that belongs to the structure (its Hessian) after the motion: `R H R^T`
    /// with the rotation of every atom.
    pub fn matrix(&self, matrix: &Mat<f64>) -> Mat<f64> {
        let n = matrix.nrows();
        let r = &self.rotation;
        // R H, then (R H) R^T, in blocks of three.
        let half = Mat::from_fn(n, n, |row, col| {
            let (atom, a) = (row / 3, row % 3);
            (0..3)
                .map(|k| r[a][k] * matrix[(3 * atom + k, col)])
                .sum::<f64>()
        });
        Mat::from_fn(n, n, |row, col| {
            let (atom, b) = (col / 3, col % 3);
            (0..3)
                .map(|k| half[(row, 3 * atom + k)] * r[b][k])
                .sum::<f64>()
        })
    }
}

/// The structures of `chain` (three coordinates per atom) centred, and from the second
/// one on turned onto the structure before them as it lies after its own turn. Returns
/// the motion of every structure.
pub fn align_chain_onto_previous(chain: &mut [Col<f64>]) -> Vec<RigidMotion> {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut motions = Vec::with_capacity(chain.len());
    for k in 0..chain.len() {
        let c = centroid(&chain[k], BLOCK_ATOM);
        let centre = [c[0], c[1], c[2]];
        let moved = centred(&chain[k]);
        let rotation = if k == 0 {
            identity
        } else {
            rotation_onto(&moved, &chain[k - 1])
        };
        chain[k] = turned(&rotation, &moved);
        motions.push(RigidMotion { rotation, centre });
    }
    motions
}

/// Orthonormal directions along which a shift of all atoms changes nothing in `cell`: its
/// periodic lattice vectors (the directions of rigid motion that the steps of periodic
/// structures leave out, `math::projection::build_tr_rot_basis`).
pub fn free_shifts(cell: &PeriodicCell) -> Vec<[f64; 3]> {
    let mut found: Vec<[f64; 3]> = Vec::with_capacity(3);
    for axis in (0..3).filter(|axis| cell.periodic_axes[*axis]) {
        let mut v = cell.lattice_vector(axis).value();
        for d in &found {
            let along: f64 = (0..3).map(|k| v[k] * d[k]).sum();
            for k in 0..3 {
                v[k] -= along * d[k];
            }
        }
        let norm = (0..3).map(|k| v[k] * v[k]).sum::<f64>().sqrt();
        if norm > 1e-10 {
            found.push([v[0] / norm, v[1] / norm, v[2] / norm]);
        }
    }
    found
}

/// The structures of `chain` (three coordinates per atom) shifted along the orthonormal
/// directions `along` so that the centroid of each is that of `centre` there, without a
/// turn: for structures in a periodic cell, where a shift of all atoms along a periodic
/// direction changes nothing and a turn does. What is kept for the images (forces, steps,
/// Hessians) is the same after a shift.
pub fn centre_chain_at(chain: &mut [Col<f64>], centre: [f64; 3], along: &[[f64; 3]]) {
    for image in chain.iter_mut() {
        let c = centroid(image, BLOCK_ATOM);
        let mut shift = [0.0; 3];
        for d in along {
            let size: f64 = (0..3).map(|k| (centre[k] - c[k]) * d[k]).sum();
            for k in 0..3 {
                shift[k] += size * d[k];
            }
        }
        for k in 0..image.nrows() {
            image[k] += shift[k % BLOCK_ATOM];
        }
    }
}

/// The structures of `chain` (three coordinates per atom) centred and turned as described
/// in the module documentation.
pub fn align_chain_in_sequence(chain: &mut [Col<f64>]) {
    for image in chain.iter_mut() {
        *image = centred(image);
    }
    for k in 0..chain.len().saturating_sub(1) {
        chain[k] = turned_onto(&chain[k], &chain[k + 1]);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    fn rotated_about_z(point: &Col<f64>, angle: f64) -> Col<f64> {
        let (c, s) = (angle.cos(), angle.sin());
        Col::from_fn(point.nrows(), |k| match k % 3 {
            0 => c * point[k] - s * point[k + 1],
            1 => s * point[k - 1] + c * point[k],
            _ => point[k],
        })
    }

    #[test]
    fn every_structure_is_turned_onto_its_neighbour_as_it_was() {
        // A bent triatomic, centred.
        let base = centred(&col(&[0.0, 0.0, 0.0, 1.8, 0.0, 0.0, -0.4, 1.7, 0.3]));
        let shift = |p: &Col<f64>, d: f64| Col::from_fn(p.nrows(), |k| p[k] + d);
        // The same structure three times, turned by different angles and moved away.
        let mut chain = vec![
            shift(&rotated_about_z(&base, 0.7), 3.0),
            shift(&rotated_about_z(&base, -0.4), -1.0),
            shift(&rotated_about_z(&base, 1.3), 0.5),
        ];
        align_chain_in_sequence(&mut chain);
        let close = |a: &Col<f64>, b: &Col<f64>| (a - b).iter().all(|d| d.abs() < 1e-12);
        // The first image lies on the second one as it was (-0.4), the second on the third
        // (1.3), and the third is centred only.
        assert!(close(&chain[0], &rotated_about_z(&base, -0.4)));
        assert!(close(&chain[1], &rotated_about_z(&base, 1.3)));
        assert!(close(&chain[2], &rotated_about_z(&base, 1.3)));
        for image in &chain {
            assert!(centroid(image, 3).iter().all(|c| c.abs() < 1e-14));
        }
    }

    #[test]
    fn an_aligned_chain_stays_and_the_motions_turn_what_belongs_to_the_images() {
        let base = centred(&col(&[0.0, 0.0, 0.0, 1.8, 0.0, 0.0, -0.4, 1.7, 0.3]));
        let shift = |p: &Col<f64>, d: f64| Col::from_fn(p.nrows(), |k| p[k] + d);
        let start = vec![
            shift(&rotated_about_z(&base, 0.7), 3.0),
            shift(&rotated_about_z(&base, -0.4), -1.0),
            shift(&rotated_about_z(&base, 1.3), 0.5),
        ];
        let mut chain = start.clone();
        let motions = align_chain_onto_previous(&mut chain);
        let close = |a: &Col<f64>, b: &Col<f64>| (a - b).iter().all(|d| d.abs() < 1e-12);
        // The first image is centred only; the others lie on it.
        for image in &chain {
            assert!(close(image, &rotated_about_z(&base, 0.7)));
        }
        // The motions are what was done to the images.
        for ((motion, before), after) in motions.iter().zip(&start).zip(&chain) {
            assert!(close(&motion.point(before), after));
        }
        assert_eq!(motions[0].rotation[0], [1.0, 0.0, 0.0]);
        // Once more: nothing moves.
        let aligned = chain.clone();
        let again = align_chain_onto_previous(&mut chain);
        for (a, b) in chain.iter().zip(&aligned) {
            assert!(close(a, b));
        }
        for motion in &again {
            for (r, row) in motion.rotation.iter().enumerate() {
                for (c, value) in row.iter().enumerate() {
                    let expected = if r == c { 1.0 } else { 0.0 };
                    assert!((value - expected).abs() < 1e-12);
                }
            }
        }
        // A gradient and a Hessian turn with their structure: g . d and d^T H d of a
        // displacement d do not change.
        let motion = &motions[1];
        let gradient = col(&[0.3, -0.2, 0.5, 0.1, 0.7, -0.4, 0.2, 0.2, 0.9]);
        let step = col(&[0.05, 0.01, -0.02, 0.03, -0.04, 0.02, 0.01, 0.06, -0.03]);
        let hessian = Mat::from_fn(9, 9, |r, c| 1.0 / (1.0 + (r as f64 - c as f64).abs()));
        let (g, d, h) = (
            motion.vector(&gradient),
            motion.vector(&step),
            motion.matrix(&hessian),
        );
        let dot =
            |a: &Col<f64>, b: &Col<f64>| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
        assert!((dot(&g, &d) - dot(&gradient, &step)).abs() < 1e-14);
        assert!((dot(&d, &(&h * &d)) - dot(&step, &(&hessian * &step))).abs() < 1e-14);
        for r in 0..9 {
            for c in 0..9 {
                assert!((h[(r, c)] - h[(c, r)]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn structures_in_a_cell_are_shifted_along_the_periodic_directions_only() {
        use crate::domain::{Bohr, Vector3};
        // A slab: periodic along a and b (not at right angles), not along c.
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Bohr>::new([8.0, 0.0, 0.0]),
            Vector3::<Bohr>::new([4.0, 7.0, 0.0]),
            Vector3::<Bohr>::new([0.0, 0.0, 30.0]),
            [true, true, false],
        )
        .unwrap();
        let along = free_shifts(&cell);
        assert_eq!(along.len(), 2);
        let dot = |a: &[f64; 3], b: &[f64; 3]| (0..3).map(|k| a[k] * b[k]).sum::<f64>();
        assert!((dot(&along[0], &along[0]) - 1.0).abs() < 1e-14);
        assert!((dot(&along[1], &along[1]) - 1.0).abs() < 1e-14);
        assert!(dot(&along[0], &along[1]).abs() < 1e-14);
        assert!(along.iter().all(|d| d[2].abs() < 1e-14));

        let first = col(&[0.0, 0.0, 0.0, 2.0, 0.0, 1.0]);
        let moved = col(&[0.5, -0.3, 0.4, 2.6, -0.2, 1.2]);
        let mut chain = vec![first.clone(), moved.clone()];
        centre_chain_at(&mut chain, [1.0, 0.0, 0.5], &along);
        // The first image is where it was; the second lost its shift within the plane
        // (0.55, -0.25) and kept the one across it, and its shape.
        for k in 0..6 {
            assert!((chain[0][k] - first[k]).abs() < 1e-14);
        }
        let expected = [-0.05, -0.05, 0.4, 2.05, 0.05, 1.2];
        for (k, value) in expected.iter().enumerate() {
            assert!((chain[1][k] - value).abs() < 1e-14, "{k}: {}", chain[1][k]);
        }
        // All three directions for a crystal.
        let crystal = PeriodicCell::from_lattice_vectors(
            Vector3::<Bohr>::new([8.0, 0.0, 0.0]),
            Vector3::<Bohr>::new([4.0, 7.0, 0.0]),
            Vector3::<Bohr>::new([1.0, 2.0, 9.0]),
            [true, true, true],
        )
        .unwrap();
        assert_eq!(free_shifts(&crystal).len(), 3);
    }
}
