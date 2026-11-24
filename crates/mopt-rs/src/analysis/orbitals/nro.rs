//! Natural reaction orbitals: the pairs of an occupied and an empty orbital through which
//! the electrons respond to a displacement of the nuclei.
//!
//! S. Ebisawa, M. Hasebe, T. Tsutsumi, T. Tsuneda, T. Taketsugu, Phys. Chem. Chem. Phys.
//! 2022, 24, 3532. With the orbitals `C` at a structure and their first-order change
//! `C1 = C U` under a displacement along a unit vector (Eqs. (6) and (13) there,
//! `U = C^T S C1`), the block of `U` between empty and occupied orbitals is decomposed,
//! `U_vo = Z L Y^T` (Eq. (17)). The occupied orbitals `C_occ Y` and the empty ones `C_virt Z`
//! are the natural reaction orbitals (Eqs. (18) and (19)): under the displacement the
//! `i`-th occupied one takes up the `i`-th empty one at the rate of the singular value
//! `l_i` (Eq. (20)). The sum of the squared singular values measures how strongly the
//! electrons respond; along a reaction path it peaks near transition states.
//!
//! `U` is taken from orbitals at structures displaced by `+-step`, as the paper allows where
//! the coupled-perturbed equations are not solved: `U = C^T S (C(+) - C(-)) / (2 step)`,
//! after the displaced orbitals have been brought into correspondence with the reference
//! ([`super::tracking`]). For a displacement along a unit vector of mass-weighted
//! coordinates the singular values are in amu^(-1/2) Bohr^(-1).
//!
//! MultiOptPy: `WFAnalyzer/MO_analysis.py:NROAnalysis`, which decomposes the whole of `U`
//! (also the blocks within the occupied and within the empty orbitals, which do not change
//! the wavefunction), sums the singular values instead of their squares, and does not match
//! the orbitals of the displaced structures; it is not called from anywhere there.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::tracking::{follow, orbital_overlaps};
use crate::domain::OrbitalSet;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// The displacement of the Cartesian coordinates that is a step of length `step`
/// (amu^(1/2) Bohr) in mass-weighted coordinates along `direction`, a Cartesian vector (the
/// tangent of a path, a step of an optimization, a gradient): `step M^(-1/2) u` with the unit
/// vector `u` along `M^(1/2) direction`.
pub fn mass_weighted_displacement(
    masses_amu: &[f64],
    direction: &Col<f64>,
    step: f64,
) -> MoptResult<Col<f64>> {
    let n = direction.nrows();
    if n != 3 * masses_amu.len() {
        return Err(MoptError::config(format!(
            "a direction of {n} components for {} atoms",
            masses_amu.len()
        )));
    }
    if masses_amu.iter().any(|m| !(m.is_finite() && *m > 0.0)) {
        return Err(MoptError::config("masses must be positive"));
    }
    let norm = libm::sqrt(
        (0..n)
            .map(|k| masses_amu[k / 3] * direction[k] * direction[k])
            .sum::<f64>(),
    );
    if !(norm.is_finite() && norm > 0.0) {
        return Err(MoptError::config(
            "the direction of the displacement vanishes or is not finite",
        ));
    }
    // M^(-1/2) (M^(1/2) d / |M^(1/2) d|) = d / |M^(1/2) d|.
    Ok(Col::from_fn(n, |k| step * direction[k] / norm))
}

/// The first-order change of the orbitals in the basis of the orbitals themselves.
#[derive(Debug, Clone)]
pub struct Response {
    /// `U = C^T S dC/ds`: column `q` is the change of orbital `q`, row `p` its component
    /// along orbital `p`.
    pub u: Mat<f64>,
    /// For each orbital of the reference, the smaller of its overlaps with what followed
    /// it at the two displaced structures (see [`super::tracking::Followed::overlaps`]).
    pub overlaps: Vec<f64>,
}

impl Response {
    /// The smallest of the overlaps of the orbitals `indices`.
    pub fn weakest_overlap(&self, indices: &[usize]) -> f64 {
        indices
            .iter()
            .map(|&k| self.overlaps[k])
            .fold(f64::INFINITY, f64::min)
    }
}

/// `U` from the orbitals at the reference structure and at structures displaced by
/// `+step` and `-step` along one direction; `overlap` is that of the atomic orbitals at the
/// reference structure.
pub fn response_matrix(
    reference: &OrbitalSet,
    plus: &OrbitalSet,
    minus: &OrbitalSet,
    overlap: &Mat<f64>,
    step: f64,
    degeneracy: f64,
) -> MoptResult<Response> {
    if !(step.is_finite() && step > 0.0) {
        return Err(MoptError::config("the displacement must be positive"));
    }
    let forward = follow(reference, plus, overlap, degeneracy)?;
    let backward = follow(reference, minus, overlap, degeneracy)?;
    let difference =
        (&forward.orbitals.coefficients - &backward.orbitals.coefficients) * (0.5 / step);
    Ok(Response {
        u: orbital_overlaps(&reference.coefficients, overlap, &difference)?,
        overlaps: forward
            .overlaps
            .iter()
            .zip(&backward.overlaps)
            .map(|(a, b)| a.min(*b))
            .collect(),
    })
}

/// The natural reaction orbitals of one set of orbitals (one spin, or the doubly occupied
/// orbitals of a closed shell).
#[derive(Debug, Clone)]
pub struct NaturalReactionOrbitals {
    /// Singular values, largest first: one per pair.
    pub singular_values: Vec<f64>,
    /// Share of each pair in the sum of the squared singular values.
    pub shares: Vec<f64>,
    /// Sum of the squared singular values.
    pub sum_of_squares: f64,
    /// The occupied orbital of each pair in the atomic orbitals (column `i`: pair `i`).
    pub occupied: Mat<f64>,
    /// The empty orbital of each pair in the atomic orbitals.
    pub virtuals: Mat<f64>,
    /// The occupied orbital of each pair in the occupied orbitals of the calculation (row
    /// `k`: the `k`-th occupied one).
    pub occupied_weights: Mat<f64>,
    /// The empty orbital of each pair in the empty orbitals of the calculation.
    pub virtual_weights: Mat<f64>,
    /// Indices of the occupied orbitals of the calculation.
    pub occupied_indices: Vec<usize>,
    /// Indices of its empty orbitals.
    pub virtual_indices: Vec<usize>,
}

/// The natural reaction orbitals for the response `u` of the orbitals `reference`. Orbitals
/// filled to more than half of the highest occupation count as occupied. Each pair has
/// the sign that makes the largest coefficient of its occupied orbital positive.
pub fn natural_reaction_orbitals(
    reference: &OrbitalSet,
    u: &Mat<f64>,
) -> MoptResult<NaturalReactionOrbitals> {
    let n = reference.energies.len();
    if u.nrows() != n || u.ncols() != n {
        return Err(MoptError::config(format!(
            "a response matrix of {}x{} for {n} orbitals",
            u.nrows(),
            u.ncols()
        )));
    }
    let highest = reference.occupations.iter().copied().fold(0.0, f64::max);
    let occupied: Vec<usize> = (0..n)
        .filter(|&k| highest > 0.0 && reference.occupations[k] > 0.5 * highest)
        .collect();
    let virtuals: Vec<usize> = (0..n).filter(|k| !occupied.contains(k)).collect();
    if occupied.is_empty() || virtuals.is_empty() {
        return Err(MoptError::config(
            "natural reaction orbitals need occupied and empty orbitals",
        ));
    }
    let (n_occ, n_virt) = (occupied.len(), virtuals.len());
    let block = Mat::from_fn(n_virt, n_occ, |a, i| u[(virtuals[a], occupied[i])]);
    let pairs = n_occ.min(n_virt);
    // The decomposition from the smaller of the two products of the block with itself.
    let small_is_occupied = n_occ <= n_virt;
    let gram = if small_is_occupied {
        block.transpose() * &block
    } else {
        &block * block.transpose()
    };
    let (values, vectors) = symmetric_eigen(&gram)?;
    let size = gram.nrows();
    let order: Vec<usize> = (0..size).rev().take(pairs).collect();
    let singular_values: Vec<f64> = order
        .iter()
        .map(|&k| libm::sqrt(values[k].max(0.0)))
        .collect();
    let largest = singular_values.first().copied().unwrap_or(0.0);
    let mut y = Mat::<f64>::zeros(n_occ, pairs);
    let mut z = Mat::<f64>::zeros(n_virt, pairs);
    for (p, &k) in order.iter().enumerate() {
        let sigma = singular_values[p];
        let defined = sigma > 1e-12 * largest.max(f64::MIN_POSITIVE);
        if small_is_occupied {
            for i in 0..n_occ {
                y[(i, p)] = vectors[(i, k)];
            }
            if defined {
                for a in 0..n_virt {
                    z[(a, p)] = (0..n_occ)
                        .map(|i| block[(a, i)] * vectors[(i, k)])
                        .sum::<f64>()
                        / sigma;
                }
            }
        } else {
            for a in 0..n_virt {
                z[(a, p)] = vectors[(a, k)];
            }
            if defined {
                for i in 0..n_occ {
                    y[(i, p)] = (0..n_virt)
                        .map(|a| block[(a, i)] * vectors[(a, k)])
                        .sum::<f64>()
                        / sigma;
                }
            }
        }
    }
    let n_ao = reference.coefficients.nrows();
    let mut occupied_orbitals = Mat::from_fn(n_ao, pairs, |r, p| {
        (0..n_occ)
            .map(|i| reference.coefficients[(r, occupied[i])] * y[(i, p)])
            .sum::<f64>()
    });
    let mut virtual_orbitals = Mat::from_fn(n_ao, pairs, |r, p| {
        (0..n_virt)
            .map(|a| reference.coefficients[(r, virtuals[a])] * z[(a, p)])
            .sum::<f64>()
    });
    // A pair is defined up to a common sign of its two orbitals.
    for p in 0..pairs {
        let peak = (0..n_ao)
            .map(|r| occupied_orbitals[(r, p)])
            .fold(0.0_f64, |a, b| if b.abs() > a.abs() { b } else { a });
        if peak < 0.0 {
            for r in 0..n_ao {
                occupied_orbitals[(r, p)] = -occupied_orbitals[(r, p)];
                virtual_orbitals[(r, p)] = -virtual_orbitals[(r, p)];
            }
            for i in 0..n_occ {
                y[(i, p)] = -y[(i, p)];
            }
            for a in 0..n_virt {
                z[(a, p)] = -z[(a, p)];
            }
        }
    }
    let sum_of_squares: f64 = singular_values.iter().map(|s| s * s).sum();
    Ok(NaturalReactionOrbitals {
        shares: singular_values
            .iter()
            .map(|s| {
                if sum_of_squares > 0.0 {
                    s * s / sum_of_squares
                } else {
                    0.0
                }
            })
            .collect(),
        singular_values,
        sum_of_squares,
        occupied: occupied_orbitals,
        virtuals: virtual_orbitals,
        occupied_weights: y,
        virtual_weights: z,
        occupied_indices: occupied,
        virtual_indices: virtuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::orbitals::tracking::DEFAULT_DEGENERACY;

    /// A model of `n` orbitals: `H(x) = H0 + x H1` in orthonormal atomic orbitals.
    fn hamiltonian(x: f64) -> Mat<f64> {
        let h0 = [-2.0, -1.2, -0.7, 0.4, 1.1, 1.9];
        Mat::from_fn(6, 6, |i, j| {
            let coupling = 0.3 * (((i + 1) * (j + 2) + (j + 1) * (i + 2)) as f64).sin();
            (if i == j { h0[i] } else { 0.05 * coupling }) + x * coupling
        })
    }

    fn orbitals(x: f64, electrons: usize) -> OrbitalSet {
        let (values, vectors) = symmetric_eigen(&hamiltonian(x)).unwrap();
        OrbitalSet {
            coefficients: vectors,
            energies: values.iter().copied().collect(),
            occupations: (0..6)
                .map(|k| if k < electrons { 1.0 } else { 0.0 })
                .collect(),
        }
    }

    /// `U_vo` from first-order perturbation theory: `(C^T H1 C)_ai / (e_i - e_a)`.
    fn exact_block(set: &OrbitalSet, electrons: usize) -> Mat<f64> {
        let step = 1.0;
        let h1 = (hamiltonian(step) - hamiltonian(0.0)) * (1.0 / step);
        let coupled = set.coefficients.transpose() * h1 * &set.coefficients;
        Mat::from_fn(6 - electrons, electrons, |a, i| {
            let v = electrons + a;
            coupled[(v, i)] / (set.energies[i] - set.energies[v])
        })
    }

    #[test]
    fn the_response_is_that_of_perturbation_theory() {
        let unit = Mat::<f64>::identity(6, 6);
        let step = 1e-4;
        for electrons in [2, 3, 5] {
            let reference = orbitals(0.0, electrons);
            // The displaced orbitals come shuffled and with other signs, as from two
            // separate calculations.
            let scramble = |set: OrbitalSet, shift: usize| OrbitalSet {
                coefficients: Mat::from_fn(6, 6, |r, k| {
                    let source = (k + shift) % 6;
                    let sign = if source.is_multiple_of(2) { -1.0 } else { 1.0 };
                    sign * set.coefficients[(r, source)]
                }),
                energies: (0..6).map(|k| set.energies[(k + shift) % 6]).collect(),
                occupations: (0..6).map(|k| set.occupations[(k + shift) % 6]).collect(),
            };
            let plus = scramble(orbitals(step, electrons), 2);
            let minus = scramble(orbitals(-step, electrons), 5);
            let response =
                response_matrix(&reference, &plus, &minus, &unit, step, DEFAULT_DEGENERACY)
                    .unwrap();
            assert!(response.weakest_overlap(&[0, 1, 2, 3, 4, 5]) > 0.999_999);
            let exact = exact_block(&reference, electrons);
            for a in 0..6 - electrons {
                for i in 0..electrons {
                    let found = response.u[(electrons + a, i)];
                    assert!((found - exact[(a, i)]).abs() < 1e-6, "{electrons}: {a} {i}");
                }
            }
            // U is antisymmetric to first order (the orbitals stay orthonormal).
            assert!((response.u[(0, 5)] + response.u[(5, 0)]).abs() < 1e-6);

            let nro = natural_reaction_orbitals(&reference, &response.u).unwrap();
            let pairs = electrons.min(6 - electrons);
            assert_eq!(nro.singular_values.len(), pairs);
            let total: f64 = (0..6 - electrons)
                .flat_map(|a| (0..electrons).map(move |i| (a, i)))
                .map(|(a, i)| exact[(a, i)].powi(2))
                .sum();
            assert!((nro.sum_of_squares - total).abs() < 1e-6 * total.max(1.0));
            assert!((nro.shares.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(nro.singular_values.windows(2).all(|w| w[0] >= w[1]));
            // Z^T U_vo Y is the diagonal matrix of the singular values, and the orbitals of
            // the pairs are orthonormal.
            for p in 0..pairs {
                for q in 0..pairs {
                    let element: f64 = (0..6 - electrons)
                        .flat_map(|a| (0..electrons).map(move |i| (a, i)))
                        .map(|(a, i)| {
                            nro.virtual_weights[(a, p)]
                                * response.u[(electrons + a, i)]
                                * nro.occupied_weights[(i, q)]
                        })
                        .sum();
                    let expected = if p == q { nro.singular_values[p] } else { 0.0 };
                    assert!((element - expected).abs() < 1e-9, "{p} {q}");
                    let overlap: f64 = (0..6)
                        .map(|r| nro.occupied[(r, p)] * nro.occupied[(r, q)])
                        .sum();
                    assert!((overlap - if p == q { 1.0 } else { 0.0 }).abs() < 1e-9);
                }
                let mixed: f64 = (0..6)
                    .map(|r| nro.occupied[(r, p)] * nro.virtuals[(r, 0)])
                    .sum();
                assert!(mixed.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn a_step_in_mass_weighted_coordinates() {
        // Two atoms of masses 1 and 16 moving against each other along x.
        let masses = [1.0, 16.0];
        let direction = Col::from_fn(6, |k| [1.0, 0.0, 0.0, -1.0, 0.0, 0.0][k]);
        let dx = mass_weighted_displacement(&masses, &direction, 0.02).unwrap();
        // Its length in mass-weighted coordinates is the step.
        let length = (0..6)
            .map(|k| masses[k / 3] * dx[k] * dx[k])
            .sum::<f64>()
            .sqrt();
        assert!((length - 0.02).abs() < 1e-15);
        // The Cartesian direction is kept.
        assert!((dx[0] + dx[3]).abs() < 1e-18 && dx[0] > 0.0 && dx[1] == 0.0);
        assert!(mass_weighted_displacement(&masses, &Col::zeros(6), 0.02).is_err());
        assert!(mass_weighted_displacement(&masses, &Col::zeros(3), 0.02).is_err());
    }

    #[test]
    fn two_orbitals_and_wrong_input() {
        // One occupied and one empty orbital at energies -1 and 1, coupled by t x: the
        // singular value is |t| / 2.
        let t = 0.3;
        let at = |x: f64| {
            let h = Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => -1.0,
                (1, 1) => 1.0,
                _ => t * x,
            });
            let (values, vectors) = symmetric_eigen(&h).unwrap();
            OrbitalSet {
                coefficients: vectors,
                energies: values.iter().copied().collect(),
                occupations: vec![2.0, 0.0],
            }
        };
        let unit = Mat::<f64>::identity(2, 2);
        let response = response_matrix(
            &at(0.0),
            &at(1e-4),
            &at(-1e-4),
            &unit,
            1e-4,
            DEFAULT_DEGENERACY,
        )
        .unwrap();
        let nro = natural_reaction_orbitals(&at(0.0), &response.u).unwrap();
        assert!((nro.singular_values[0] - 0.5 * t).abs() < 1e-8);
        assert_eq!(nro.occupied_indices, vec![0]);
        assert_eq!(nro.virtual_indices, vec![1]);
        // The pair: the lower orbital with a positive largest coefficient, and the upper
        // one with the sign that the displacement mixes in.
        assert!((nro.occupied[(0, 0)].abs() - 1.0).abs() < 1e-12 && nro.occupied[(0, 0)] > 0.0);

        let full = OrbitalSet {
            occupations: vec![2.0, 2.0],
            ..at(0.0)
        };
        assert!(natural_reaction_orbitals(&full, &response.u).is_err());
        assert!(natural_reaction_orbitals(&at(0.0), &Mat::<f64>::zeros(3, 3)).is_err());
        assert!(response_matrix(
            &at(0.0),
            &at(1e-4),
            &at(-1e-4),
            &unit,
            0.0,
            DEFAULT_DEGENERACY
        )
        .is_err());
    }
}
