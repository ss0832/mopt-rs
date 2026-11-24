//! Vibrational self-consistent field (VSCF) on a quartic force field.
//!
//! The vibrational wave function is a product of one function (modal) per normal mode. Each
//! modal is an eigenfunction of the one-mode Hamiltonian in which the other modes enter
//! through the averages of their coordinates over their own modals:
//!
//! ```text
//! h_i = w_i / 2 (p_i^2 + q_i^2) + c_1 q_i + c_2 q_i^2 + c_3 q_i^3 + c_4 q_i^4
//! ```
//!
//! with coefficients from the cubic and quartic constants times `<q_j^p>` of the other
//! modes. The modals are expanded in the harmonic oscillator functions of their mode, and
//! the equations are iterated until the energy `<Psi| H |Psi>` is stationary. Unlike
//! perturbation theory this has no small denominators and treats the anharmonicity of every
//! single mode exactly; what it lacks is the correlation between the modes.
//!
//! A fundamental is the difference between the energy of the state with one quantum in the
//! mode, converged on its own, and that of the ground state.
//!
//! A quartic expansion need not be bounded from below: with a negative quartic constant, or
//! a cubic one that outweighs the quartic one, the potential falls below the minimum at
//! large displacements, and a large enough basis has functions that live there, with
//! energies below those of the vibrations. The modal of a mode is therefore not taken by
//! its position among the eigenvalues but as the eigenfunction that overlaps most with the
//! modal of the iteration before (the harmonic oscillator function at the start). When this
//! is not the function its number of quanta would give, the result says so.
//!
//! The force field has the cubic constants and the quartic ones with two equal indices; the
//! vibration-rotation coupling is not included.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::ForceField;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{symmetric_eigen, symmetrize};

/// Settings of the self-consistent field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VscfSettings {
    /// Harmonic oscillator functions per mode.
    pub basis: usize,
    /// Change of the energy (cm^-1) below which the iterations stop.
    pub tolerance: f64,
    /// Largest number of iterations for one state.
    pub max_iterations: usize,
}

impl Default for VscfSettings {
    fn default() -> Self {
        Self {
            basis: DEFAULT_BASIS,
            tolerance: 1e-8,
            max_iterations: 200,
        }
    }
}

/// Default number of harmonic oscillator functions per mode; see
/// `docs/source/migration_vibrations.md` ("Anharmonic vibrational analysis") for the measurement.
pub const DEFAULT_BASIS: usize = 16;

/// Result of the self-consistent field (cm^-1).
#[derive(Debug, Clone)]
pub struct Vscf {
    /// Harmonic wavenumbers.
    pub harmonic: Vec<f64>,
    /// Energy of the ground state.
    pub zero_point: f64,
    /// Fundamentals `1 <- 0` of every mode.
    pub fundamentals: Vec<f64>,
    /// Whether the ground state and every excited state converged.
    pub converged: bool,
    /// Whether the force field has states below those of the vibrations (it is not bounded
    /// from below within the basis), so that the modals had to be followed by their overlap.
    pub unbounded: bool,
}

/// One term of the potential: a coefficient times powers of the coordinates of different
/// modes, `(mode, power)`.
pub(crate) type Term = (f64, Vec<(usize, usize)>);

/// The anharmonic part of the potential of `field` as a sum of terms.
pub(crate) fn potential_terms(field: &ForceField) -> Vec<Term> {
    let m = field.modes();
    let mut terms: Vec<Term> = Vec::new();
    let mut add = |coefficient: f64, factors: Vec<(usize, usize)>| {
        if coefficient != 0.0 {
            terms.push((coefficient, factors));
        }
    };
    for i in 0..m {
        add(field.cubic(i, i, i) / 6.0, vec![(i, 3)]);
        add(field.quartic(i, i, i) / 24.0, vec![(i, 4)]);
        for j in 0..m {
            if j == i {
                continue;
            }
            add(field.cubic(i, i, j) / 2.0, vec![(i, 2), (j, 1)]);
            add(field.quartic(i, i, j) / 6.0, vec![(i, 3), (j, 1)]);
            if j > i {
                add(field.quartic(i, j, j) / 4.0, vec![(i, 2), (j, 2)]);
            }
            for k in j + 1..m {
                if k == i {
                    continue;
                }
                add(field.quartic(i, j, k) / 2.0, vec![(i, 2), (j, 1), (k, 1)]);
                if i < j {
                    add(field.cubic(i, j, k), vec![(i, 1), (j, 1), (k, 1)]);
                }
            }
        }
    }
    terms
}

/// The matrices of `q`, `q^2`, `q^3`, `q^4` in the first `n` harmonic oscillator functions
/// (index `p - 1`).
pub(crate) fn coordinate_powers(n: usize) -> [Mat<f64>; 4] {
    // Products in a larger basis, so that the truncation does not reach the block kept.
    let large = n + 4;
    let q = Mat::from_fn(large, large, |a, b| {
        if a + 1 == b {
            (b as f64 / 2.0).sqrt()
        } else if b + 1 == a {
            (a as f64 / 2.0).sqrt()
        } else {
            0.0
        }
    });
    let q2 = &q * &q;
    let q3 = &q2 * &q;
    let q4 = &q2 * &q2;
    let cut = |matrix: &Mat<f64>| Mat::from_fn(n, n, |a, b| matrix[(a, b)]);
    [cut(&q), cut(&q2), cut(&q3), cut(&q4)]
}

/// The state with the quanta `occupation`: its energy, whether it converged, and whether a
/// modal was not the eigenfunction that its number of quanta gives.
fn state(
    field: &ForceField,
    terms: &[Term],
    powers: &[Mat<f64>; 4],
    occupation: &[usize],
    settings: &VscfSettings,
) -> MoptResult<(f64, bool, bool)> {
    let (m, n) = (field.modes(), settings.basis);
    // Modals as coefficient vectors, starting from the harmonic oscillator functions.
    let mut modals: Vec<Vec<f64>> = occupation
        .iter()
        .map(|quanta| (0..n).map(|a| f64::from(u8::from(a == *quanta))).collect())
        .collect();
    let moments_of = |modal: &[f64]| -> [f64; 4] {
        std::array::from_fn(|p| {
            let mut sum = 0.0;
            for a in 0..n {
                for b in 0..n {
                    sum += modal[a] * powers[p][(a, b)] * modal[b];
                }
            }
            sum
        })
    };
    let harmonic_of = |mode: usize, modal: &[f64]| -> f64 {
        (0..n)
            .map(|a| field.harmonic[mode] * (a as f64 + 0.5) * modal[a] * modal[a])
            .sum()
    };
    let mut moments: Vec<[f64; 4]> = modals.iter().map(|c| moments_of(c)).collect();
    let energy_of = |modals: &[Vec<f64>], moments: &[[f64; 4]]| -> f64 {
        let own: f64 = (0..m).map(|i| harmonic_of(i, &modals[i])).sum();
        let coupled: f64 = terms
            .iter()
            .map(|(coefficient, factors)| {
                coefficient
                    * factors
                        .iter()
                        .map(|(mode, power)| moments[*mode][power - 1])
                        .product::<f64>()
            })
            .sum();
        own + coupled
    };
    let mut energy = energy_of(&modals, &moments);
    let mut followed = false;
    for _ in 0..settings.max_iterations {
        for i in 0..m {
            // The one-mode Hamiltonian of mode i in the field of the others.
            let mut h = Mat::from_fn(n, n, |a, b| {
                if a == b {
                    field.harmonic[i] * (a as f64 + 0.5)
                } else {
                    0.0
                }
            });
            for (coefficient, factors) in terms {
                let Some(&(_, power)) = factors.iter().find(|(mode, _)| *mode == i) else {
                    continue;
                };
                let others: f64 = factors
                    .iter()
                    .filter(|(mode, _)| *mode != i)
                    .map(|(mode, p)| moments[*mode][p - 1])
                    .product();
                let scale = coefficient * others;
                for a in 0..n {
                    for b in 0..n {
                        h[(a, b)] += scale * powers[power - 1][(a, b)];
                    }
                }
            }
            symmetrize(&mut h);
            let (_, vectors) = symmetric_eigen(&h)?;
            // The eigenfunction that continues the modal.
            let overlap = |k: usize| -> f64 {
                (0..n)
                    .map(|a| vectors[(a, k)] * modals[i][a])
                    .sum::<f64>()
                    .abs()
            };
            let chosen = (0..n).fold(
                0,
                |best, k| {
                    if overlap(k) > overlap(best) {
                        k
                    } else {
                        best
                    }
                },
            );
            followed |= chosen != occupation[i];
            modals[i] = (0..n).map(|a| vectors[(a, chosen)]).collect();
            moments[i] = moments_of(&modals[i]);
        }
        let next = energy_of(&modals, &moments);
        let change = (next - energy).abs();
        energy = next;
        if change < settings.tolerance {
            return Ok((energy, true, followed));
        }
    }
    Ok((energy, false, followed))
}

/// VSCF for the force field `field`: the ground state and the fundamentals.
pub fn vscf(field: &ForceField, settings: &VscfSettings) -> MoptResult<Vscf> {
    let m = field.modes();
    if settings.basis < 4 {
        return Err(MoptError::config(
            "VSCF needs four harmonic oscillator functions per mode at least",
        ));
    }
    let terms = potential_terms(field);
    let powers = coordinate_powers(settings.basis);
    let (zero_point, mut converged, mut unbounded) =
        state(field, &terms, &powers, &vec![0; m], settings)?;
    let mut fundamentals = Vec::with_capacity(m);
    for i in 0..m {
        let occupation: Vec<usize> = (0..m).map(|j| usize::from(j == i)).collect();
        let (energy, done, followed) = state(field, &terms, &powers, &occupation, settings)?;
        fundamentals.push(energy - zero_point);
        converged &= done;
        unbounded |= followed;
    }
    Ok(Vscf {
        harmonic: field.harmonic.clone(),
        zero_point,
        fundamentals,
        converged,
        unbounded,
    })
}

/// The lowest levels of the force field by diagonalization in the product basis of `basis`
/// harmonic oscillator functions per mode: the reference for the tests of the approximate
/// methods (its dimension is `basis^modes`).
#[cfg(test)]
pub(crate) fn variational_levels(field: &ForceField, basis: usize) -> Vec<f64> {
    let m = field.modes();
    let dimension = basis.pow(m as u32);
    let digits = |state: usize| -> Vec<usize> {
        let mut rest = state;
        (0..m)
            .map(|_| {
                let digit = rest % basis;
                rest /= basis;
                digit
            })
            .collect()
    };
    let states: Vec<Vec<usize>> = (0..dimension).map(digits).collect();
    let terms = potential_terms(field);
    let powers = coordinate_powers(basis);
    let mut h = Mat::<f64>::zeros(dimension, dimension);
    for (a, bra) in states.iter().enumerate() {
        h[(a, a)] = (0..m)
            .map(|i| field.harmonic[i] * (bra[i] as f64 + 0.5))
            .sum();
        for (b, ket) in states.iter().enumerate() {
            for (coefficient, factors) in &terms {
                // Modes without a factor must have the same quanta on both sides.
                let diagonal_elsewhere = (0..m)
                    .filter(|i| !factors.iter().any(|(mode, _)| mode == i))
                    .all(|i| bra[i] == ket[i]);
                if !diagonal_elsewhere {
                    continue;
                }
                let element: f64 = factors
                    .iter()
                    .map(|(mode, power)| powers[power - 1][(bra[*mode], ket[*mode])])
                    .product();
                h[(a, b)] += coefficient * element;
            }
        }
    }
    symmetrize(&mut h);
    #[allow(clippy::unwrap_used)]
    let (values, _) = symmetric_eigen(&h).unwrap();
    values.iter().copied().collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::vib::{vpt2, Vpt2Settings};

    /// The level above the ground state nearest to `expected`, and its distance.
    fn nearest(levels: &[f64], expected: f64) -> (f64, f64) {
        let ground = levels[0];
        levels[1..]
            .iter()
            .map(|e| e - ground)
            .map(|e| (e, (e - expected).abs()))
            .fold((f64::NAN, f64::INFINITY), |best, found| {
                if found.1 < best.1 {
                    found
                } else {
                    best
                }
            })
    }

    fn three_modes() -> ForceField {
        ForceField::from_constants(
            &[1000.0, 1700.0, 3100.0],
            &[
                (0, 0, 0, 60.0),
                (0, 0, 1, 40.0),
                (0, 1, 1, -50.0),
                (1, 1, 1, 70.0),
                (0, 1, 2, 45.0),
                (2, 2, 2, -80.0),
                (0, 0, 2, 30.0),
                (1, 1, 2, 35.0),
                (0, 2, 2, 25.0),
                (1, 2, 2, -20.0),
            ],
            &[
                (0, 0, 0, 10.0),
                (1, 1, 1, 12.0),
                (2, 2, 2, 15.0),
                (0, 1, 1, 6.0),
                (0, 2, 2, 5.0),
                (1, 2, 2, -4.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn perturbation_theory_agrees_with_the_diagonalization_for_a_weak_anharmonicity() {
        let field = three_modes();
        let levels = variational_levels(&field, 8);
        let pt = vpt2(&field, &Vpt2Settings::default());
        assert!(pt.resonances.is_empty(), "{:?}", pt.resonances);
        // The shifts are a few cm^-1; what second order leaves out is a few hundredths.
        let tolerance = 0.02;
        assert!(
            (pt.zero_point - levels[0]).abs() < tolerance,
            "zero point {} against {}",
            pt.zero_point,
            levels[0]
        );
        assert!((pt.zero_point - pt.zero_point_harmonic).abs() > 1.0);
        for i in 0..3 {
            let (found, distance) = nearest(&levels, pt.fundamentals[i]);
            assert!(
                distance < tolerance,
                "fundamental {i}: {} against {found}",
                pt.fundamentals[i]
            );
            assert!((pt.fundamentals[i] - field.harmonic[i]).abs() > 0.2);
        }
        for i in 0..2 {
            let (found, distance) = nearest(&levels, pt.overtones[i]);
            assert!(
                distance < 2.0 * tolerance,
                "overtone {i}: {} against {found}",
                pt.overtones[i]
            );
        }
        let (found, distance) = nearest(&levels, pt.combination(0, 1));
        assert!(
            distance < 2.0 * tolerance,
            "combination: {} against {found}",
            pt.combination(0, 1)
        );
    }

    #[test]
    fn a_morse_oscillator_has_its_anharmonic_constant() {
        // w = 2 D x^2, phi_111 = -6 D x^3, phi_1111 = 14 D x^4 give x_11 = -w^2 / (4 D), the
        // exact anharmonic constant of the Morse oscillator.
        let (depth, x) = (40_000.0_f64, 0.19_f64);
        let omega = 2.0 * depth * x * x;
        let field = ForceField::from_constants(
            &[omega],
            &[(0, 0, 0, -6.0 * depth * x.powi(3))],
            &[(0, 0, 0, 14.0 * depth * x.powi(4))],
        )
        .unwrap();
        let pt = vpt2(&field, &Vpt2Settings::default());
        let expected = -omega * omega / (4.0 * depth);
        assert!((pt.constants[(0, 0)] - expected).abs() < 1e-9 * expected.abs());
        assert!((pt.fundamentals[0] - (omega + 2.0 * expected)).abs() < 1e-9);
        // E_0 = w / 2 + x_11 / 4 for the Morse oscillator.
        assert!((pt.zero_point - (0.5 * omega + 0.25 * expected)).abs() < 1e-9);
    }

    #[test]
    fn a_fermi_resonance_is_taken_out_and_solved_as_a_pair_of_states() {
        // 2 w_0 = 2000 next to w_1 = 2010 with phi_001 = 60: second-order perturbation
        // theory would divide by the gap of 10 cm^-1.
        let field = ForceField::from_constants(&[1000.0, 2010.0], &[(0, 0, 1, 60.0)], &[]).unwrap();
        let pt = vpt2(&field, &Vpt2Settings::default());
        assert_eq!(pt.resonances.len(), 1);
        let resonance = pt.resonances[0];
        assert_eq!((resonance.pair, resonance.single), ((0, 0), 1));
        assert!((resonance.gap + 10.0).abs() < 1e-12);
        let levels = variational_levels(&field, 12);
        for level in [resonance.levels.0, resonance.levels.1] {
            let (found, distance) = nearest(&levels, level);
            assert!(distance < 0.3, "{level} against {found}");
        }
        // The pair is split by more than the gap of the harmonic levels.
        assert!(resonance.levels.1 - resonance.levels.0 > 30.0);
        // Without the test for resonances the constants explode.
        let blind = vpt2(
            &field,
            &Vpt2Settings {
                fermi_gap: 0.0,
                ..Vpt2Settings::default()
            },
        );
        assert!(blind.resonances.is_empty());
        assert!(blind.constants[(0, 0)].abs() > 10.0 * pt.constants[(0, 0)].abs());
    }

    #[test]
    fn the_self_consistent_field_is_exact_for_one_mode_and_an_upper_bound_for_more() {
        // One mode: the modal is the eigenfunction.
        let single =
            ForceField::from_constants(&[1500.0], &[(0, 0, 0, -150.0)], &[(0, 0, 0, 40.0)])
                .unwrap();
        let settings = VscfSettings {
            basis: 14,
            ..VscfSettings::default()
        };
        let scf = vscf(&single, &settings).unwrap();
        let levels = variational_levels(&single, 14);
        assert!(scf.converged);
        assert!((scf.zero_point - levels[0]).abs() < 1e-8);
        assert!((scf.fundamentals[0] - (levels[1] - levels[0])).abs() < 1e-8);
        // Three coupled modes: above the lowest level, and close to it and to the
        // fundamentals when the coupling is weak.
        let field = three_modes();
        let settings = VscfSettings {
            basis: 8,
            ..VscfSettings::default()
        };
        let scf = vscf(&field, &settings).unwrap();
        let levels = variational_levels(&field, 8);
        assert!(scf.converged);
        assert!(scf.zero_point >= levels[0] - 1e-9);
        assert!(scf.zero_point - levels[0] < 1.0);
        for i in 0..3 {
            let (found, distance) = nearest(&levels, scf.fundamentals[i]);
            assert!(
                distance < 1.5,
                "fundamental {i}: {} against {found}",
                scf.fundamentals[i]
            );
        }
        assert!(!scf.unbounded);
        assert!(vscf(
            &field,
            &VscfSettings {
                basis: 2,
                ..settings
            }
        )
        .is_err());
    }

    #[test]
    fn modals_are_followed_where_the_force_field_is_not_bounded() {
        // A negative quartic constant: the potential turns down beyond q = 7, and a basis
        // that reaches there has states below the vibrations. In a small basis the
        // vibrations are the lowest states; in the large one they are found by their
        // overlap, at nearly the same wavenumbers.
        let field = ForceField::from_constants(&[1500.0], &[], &[(0, 0, 0, -350.0)]).unwrap();
        let small = vscf(
            &field,
            &VscfSettings {
                basis: 8,
                ..VscfSettings::default()
            },
        )
        .unwrap();
        let large = vscf(
            &field,
            &VscfSettings {
                basis: 40,
                ..VscfSettings::default()
            },
        )
        .unwrap();
        assert!(!small.unbounded && large.unbounded);
        // First order: 3 phi / 96 (2 n^2 + 2 n + 1): -10.9 and -54.7 cm^-1.
        assert!((small.fundamentals[0] - (1500.0 - 43.75)).abs() < 5.0);
        assert!(
            (large.fundamentals[0] - small.fundamentals[0]).abs() < 5.0,
            "{} against {}",
            large.fundamentals[0],
            small.fundamentals[0]
        );
    }
}
