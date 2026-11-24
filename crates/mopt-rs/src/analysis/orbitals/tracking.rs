//! Following molecular orbitals from one structure to a neighbouring one.
//!
//! The orbitals of two calculations come in the order of their energies and with
//! arbitrary signs, and orbitals of equal energy in an arbitrary rotation among
//! themselves. A difference of orbital coefficients or of orbital energies between two
//! structures means something only for orbitals that correspond to each other. They are
//! matched here by their overlaps `O = C0^T S C1` (`S`: the overlap of the atomic orbitals):
//!
//! 1. each orbital of the reference gets one of the other set so that the sum of `|O|`
//!    over the pairs is largest (a linear assignment);
//! 2. orbitals that are degenerate in the reference are rotated among their partners so
//!    that the partners come as close to them as a rotation allows (orthogonal
//!    Procrustes: `R = (M^T M)^(-1/2) M^T` for the block `M` of overlaps);
//! 3. the others get the sign that makes their overlap positive.
//!
//! The overlap of each pair after this is reported, as the cosine of the angle between
//! the two coefficient vectors in the metric `S`: a value well below 1 says that the
//! orbital has no clear counterpart (the structures are too far apart, or orbitals mix
//! strongly between them). The cosine is used because the orbitals of a displaced
//! structure are normalized with its own overlap matrix, not with `S`, which changes
//! `C0^T S C1` itself in first order of the displacement.
//!
//! MultiOptPy: no counterpart (new in mopt-rs). `WFAnalyzer/MO_analysis.py:NROAnalysis`
//! subtracts the coefficients of two calculations orbital by orbital in the order they come.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::domain::OrbitalSet;
use crate::error::{MoptError, MoptResult};
use crate::math::assignment::linear_sum_assignment;
use crate::math::linalg::symmetric_eigen;

/// Orbitals of the reference closer than this in energy (Hartree), and occupied alike, are
/// treated as degenerate.
pub const DEFAULT_DEGENERACY: f64 = 1e-5;

/// A set of orbitals brought into correspondence with a reference.
#[derive(Debug, Clone)]
pub struct Followed {
    /// The orbitals in the order, rotation, and sign of the reference.
    pub orbitals: OrbitalSet,
    /// For each orbital of the reference, the index of its partner in the set as it was
    /// given (within a degenerate group: the partner assigned before the rotation).
    pub source: Vec<usize>,
    /// For each orbital of the reference, the cosine between it and what follows it in the
    /// metric of the overlap matrix (1: the same orbital).
    pub overlaps: Vec<f64>,
}

impl Followed {
    /// The smallest of the overlaps.
    pub fn weakest(&self) -> f64 {
        self.overlaps.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// `C0^T S C1`.
pub fn orbital_overlaps(
    reference: &Mat<f64>,
    overlap: &Mat<f64>,
    other: &Mat<f64>,
) -> MoptResult<Mat<f64>> {
    let n = overlap.nrows();
    if overlap.ncols() != n || reference.nrows() != n || other.nrows() != n {
        return Err(MoptError::config(format!(
            "orbital coefficients of {} and {} atomic orbitals with an overlap matrix of {}x{}",
            reference.nrows(),
            other.nrows(),
            n,
            overlap.ncols()
        )));
    }
    Ok(reference.transpose() * overlap * other)
}

/// Groups of orbitals of `set` that are degenerate (see [`DEFAULT_DEGENERACY`]), as lists
/// of indices; single orbitals are groups of one. The orbitals need not be in the order
/// of their energies (a set that has been followed is in the order of its reference).
fn degenerate_groups(set: &OrbitalSet, degeneracy: f64) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..set.energies.len()).collect();
    order.sort_by(|&a, &b| set.energies[a].total_cmp(&set.energies[b]));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for k in order {
        let joined = groups.last().and_then(|g| g.last()).is_some_and(|&last| {
            (set.energies[k] - set.energies[last]).abs() < degeneracy
                && (set.occupations[k] - set.occupations[last]).abs() < 1e-8
        });
        match groups.last_mut() {
            Some(group) if joined => group.push(k),
            _ => groups.push(vec![k]),
        }
    }
    groups
}

/// `other` in the order, rotation, and sign of `reference` (both in the same atomic
/// orbitals, whose overlap at the reference structure is `overlap`).
pub fn follow(
    reference: &OrbitalSet,
    other: &OrbitalSet,
    overlap: &Mat<f64>,
    degeneracy: f64,
) -> MoptResult<Followed> {
    let n = reference.energies.len();
    if other.energies.len() != n {
        return Err(MoptError::config(format!(
            "{} orbitals cannot follow {n}",
            other.energies.len()
        )));
    }
    let o = orbital_overlaps(&reference.coefficients, overlap, &other.coefficients)?;
    if (0..n).any(|i| (0..n).any(|j| !o[(i, j)].is_finite())) {
        return Err(MoptError::calc("an overlap of two orbitals is not finite"));
    }
    let cost = Mat::from_fn(n, n, |i, j| -o[(i, j)].abs());
    let source = linear_sum_assignment(&cost)?;

    let n_ao = other.coefficients.nrows();
    let mut coefficients = Mat::<f64>::zeros(n_ao, n);
    let mut energies = vec![0.0; n];
    let mut occupations = vec![0.0; n];
    for group in degenerate_groups(reference, degeneracy) {
        let g = group.len();
        // The block of overlaps of the group with its partners, and the rotation of the
        // partners (a sign for a single orbital).
        let m = Mat::from_fn(g, g, |a, b| o[(group[a], source[group[b]])]);
        let rotation = if g == 1 {
            Mat::from_fn(1, 1, |_, _| if m[(0, 0)] < 0.0 { -1.0 } else { 1.0 })
        } else {
            procrustes(&m)?
        };
        for (b, &k) in group.iter().enumerate() {
            for (a, &member) in group.iter().enumerate() {
                let weight = rotation[(a, b)];
                let partner = source[member];
                for row in 0..n_ao {
                    coefficients[(row, k)] += weight * other.coefficients[(row, partner)];
                }
                // Expectation values of the rotated orbitals.
                energies[k] += weight * weight * other.energies[partner];
                occupations[k] += weight * weight * other.occupations[partner];
            }
        }
    }
    let aligned = orbital_overlaps(&reference.coefficients, overlap, &coefficients)?;
    let own = orbital_overlaps(&reference.coefficients, overlap, &reference.coefficients)?;
    let other_own = orbital_overlaps(&coefficients, overlap, &coefficients)?;
    let overlaps = (0..n)
        .map(|k| {
            let norms = own[(k, k)] * other_own[(k, k)];
            if norms > 0.0 {
                aligned[(k, k)] / libm::sqrt(norms)
            } else {
                0.0
            }
        })
        .collect();
    Ok(Followed {
        orbitals: OrbitalSet {
            coefficients,
            energies,
            occupations,
        },
        source,
        overlaps,
    })
}

/// The rotation `R` that brings `M R` closest to a symmetric positive matrix (the largest
/// trace): `R = V U^T` for `M = U S V^T`, that is `(M^T M)^(-1/2) M^T`. A block that does
/// not have full rank (partners that do not span the group) is not rotated: its orbitals
/// only get their signs.
fn procrustes(m: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let g = m.nrows();
    let (values, vectors) = symmetric_eigen(&(m.transpose() * m))?;
    let largest = values.iter().copied().fold(0.0, f64::max);
    if values.iter().any(|v| *v <= 1e-12 * largest) || largest <= 0.0 {
        return Ok(Mat::from_fn(g, g, |a, b| match (a == b, m[(a, b)] < 0.0) {
            (true, true) => -1.0,
            (true, false) => 1.0,
            _ => 0.0,
        }));
    }
    // (M^T M)^(-1/2) = V S^-1 V^T.
    let inverse_root = Mat::from_fn(g, g, |a, b| {
        (0..g)
            .map(|k| vectors[(a, k)] * vectors[(b, k)] / libm::sqrt(values[k]))
            .sum::<f64>()
    });
    Ok(inverse_root * m.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Orthonormal orbitals in five atomic orbitals with overlap 1.
    fn reference() -> OrbitalSet {
        // Columns of a rotation built from two plane rotations.
        let (a, b) = (0.4_f64, -0.9_f64);
        let r1 = Mat::from_fn(5, 5, |i, j| match (i, j) {
            (0, 0) | (1, 1) => a.cos(),
            (0, 1) => -a.sin(),
            (1, 0) => a.sin(),
            _ if i == j => 1.0,
            _ => 0.0,
        });
        let r2 = Mat::from_fn(5, 5, |i, j| match (i, j) {
            (2, 2) | (4, 4) => b.cos(),
            (2, 4) => -b.sin(),
            (4, 2) => b.sin(),
            _ if i == j => 1.0,
            _ => 0.0,
        });
        OrbitalSet {
            coefficients: &r1 * &r2,
            // Orbitals 1 and 2 are degenerate.
            energies: vec![-1.0, -0.5, -0.5, 0.2, 0.9],
            occupations: vec![1.0, 1.0, 1.0, 0.0, 0.0],
        }
    }

    fn column(m: &Mat<f64>, k: usize) -> Vec<f64> {
        (0..m.nrows()).map(|r| m[(r, k)]).collect()
    }

    #[test]
    fn order_and_signs_are_restored() {
        let set = reference();
        let unit = Mat::<f64>::identity(5, 5);
        // The same orbitals, shuffled, some with the other sign, and slightly changed.
        let order = [3, 0, 4, 1, 2];
        let signs = [1.0, -1.0, 1.0, -1.0, -1.0];
        let other = OrbitalSet {
            coefficients: Mat::from_fn(5, 5, |r, k| {
                signs[k] * set.coefficients[(r, order[k])] + 1e-4 * ((r * 5 + k) as f64).sin()
            }),
            energies: order.iter().map(|&k| set.energies[k] + 0.01).collect(),
            occupations: order.iter().map(|&k| set.occupations[k]).collect(),
        };
        let followed = follow(&set, &other, &unit, DEFAULT_DEGENERACY).unwrap();
        // Outside the degenerate pair, each orbital has its partner.
        assert_eq!(followed.source[0], 1);
        assert_eq!(followed.source[3], 0);
        assert_eq!(followed.source[4], 2);
        for k in 0..5 {
            assert!(
                followed.overlaps[k] > 0.999,
                "{k}: {}",
                followed.overlaps[k]
            );
            assert!((followed.orbitals.energies[k] - set.energies[k] - 0.01).abs() < 1e-9);
            assert!((followed.orbitals.occupations[k] - set.occupations[k]).abs() < 1e-9);
            for (found, expected) in column(&followed.orbitals.coefficients, k)
                .iter()
                .zip(column(&set.coefficients, k))
            {
                assert!((found - expected).abs() < 1e-3);
            }
        }
        assert!(followed.weakest() > 0.999);
    }

    #[test]
    fn degenerate_orbitals_are_rotated_back() {
        let set = reference();
        let unit = Mat::<f64>::identity(5, 5);
        // The degenerate pair comes back rotated by 30 degrees among itself.
        let phi = 30.0_f64.to_radians();
        let other = OrbitalSet {
            coefficients: Mat::from_fn(5, 5, |r, k| match k {
                1 => phi.cos() * set.coefficients[(r, 1)] + phi.sin() * set.coefficients[(r, 2)],
                2 => -phi.sin() * set.coefficients[(r, 1)] + phi.cos() * set.coefficients[(r, 2)],
                _ => set.coefficients[(r, k)],
            }),
            energies: vec![-1.0, -0.52, -0.48, 0.2, 0.9],
            occupations: set.occupations.clone(),
        };
        let followed = follow(&set, &other, &unit, DEFAULT_DEGENERACY).unwrap();
        for k in 0..5 {
            assert!((followed.overlaps[k] - 1.0).abs() < 1e-12, "{k}");
        }
        // The energies of the rotated pair are the expectation values.
        let mixed = phi.cos().powi(2) * -0.52 + phi.sin().powi(2) * -0.48;
        assert!((followed.orbitals.energies[1] - mixed).abs() < 1e-12);
        assert!(
            (followed.orbitals.energies[1] + followed.orbitals.energies[2] + 1.0).abs() < 1e-12
        );
        // Without the rotation (no degeneracy assumed) the pair keeps an overlap of
        // cos(30 degrees) only.
        let plain = follow(&set, &other, &unit, 0.0).unwrap();
        assert!((plain.overlaps[1] - phi.cos()).abs() < 1e-12);
    }

    #[test]
    fn a_basis_that_is_not_orthonormal_and_wrong_sizes() {
        let set = reference();
        // The same orbitals in atomic orbitals scaled by 2 and 0.5: C' = A^-1 C, S' = A^T A.
        let scale = [2.0, 0.5, 1.0, 1.5, 0.8];
        let s = Mat::from_fn(5, 5, |i, j| if i == j { scale[i] * scale[i] } else { 0.0 });
        let scaled = |c: &Mat<f64>| Mat::from_fn(5, 5, |r, k| c[(r, k)] / scale[r]);
        let a = OrbitalSet {
            coefficients: scaled(&set.coefficients),
            ..set.clone()
        };
        let b = OrbitalSet {
            coefficients: Mat::from_fn(5, 5, |r, k| -set.coefficients[(r, 4 - k)] / scale[r]),
            energies: set.energies.iter().rev().copied().collect(),
            occupations: set.occupations.iter().rev().copied().collect(),
        };
        let followed = follow(&a, &b, &s, DEFAULT_DEGENERACY).unwrap();
        for k in 0..5 {
            assert!((followed.overlaps[k] - 1.0).abs() < 1e-12);
            assert!((followed.orbitals.energies[k] - set.energies[k]).abs() < 1e-12);
        }
        let fewer = OrbitalSet {
            coefficients: Mat::zeros(5, 4),
            energies: vec![0.0; 4],
            occupations: vec![0.0; 4],
        };
        assert!(follow(&set, &fewer, &s, DEFAULT_DEGENERACY).is_err());
        assert!(follow(&set, &set, &Mat::<f64>::identity(4, 4), DEFAULT_DEGENERACY).is_err());
    }
}
