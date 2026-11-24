//! Second-order vibrational perturbation theory (VPT2) on a quartic force field.
//!
//! With the cubic terms of the potential to second order and the quartic ones to first, the
//! vibrational term values of an asymmetric top are
//!
//! ```text
//! G(n) = G_0 + sum_i w_i (n_i + 1/2) + sum_(i<=j) x_ij (n_i + 1/2) (n_j + 1/2)
//!
//! x_ii = phi_iiii / 16
//!      - 1/16 sum_k phi_iik^2 (2 / w_k + 1/2 / (2 w_i + w_k) - 1/2 / (2 w_i - w_k))
//! x_ij = phi_iijj / 4 - 1/4 sum_k phi_iik phi_jjk / w_k
//!      - 1/8 sum_k phi_ijk^2 (1 / (w_i + w_j + w_k) + 1 / (-w_i + w_j + w_k)
//!                             + 1 / (w_i - w_j + w_k) - 1 / (w_i + w_j - w_k))
//!      + sum_a B_a (zeta^a_ij)^2 (w_i / w_j + w_j / w_i)
//! ```
//!
//! (Nielsen; Mills, in Molecular Spectroscopy: Modern Research, 1972; the partial fractions
//! are those of the usual closed forms). Fundamentals are `w_i + 2 x_ii + 1/2 sum_(j != i)
//! x_ij`, overtones `2 w_i + 6 x_ii + sum_(j != i) x_ij`.
//!
//! **Fermi resonances.** A fraction whose denominator nearly vanishes (`2 w_i ~ w_k` or
//! `w_i + w_j ~ w_k`) is not a small perturbation. Such a term is found by the test of
//! Martin, Lee, Taylor, and Francois (J. Chem. Phys. 1995, 103, 2589): the frequency gap is
//! below a limit and the leading correction of the two-state problem beyond perturbation
//! theory, `phi^4 / (256 gap^3)` (`phi^4 / (64 gap^3)` for three different modes), exceeds a
//! threshold. Its fraction is left out (deperturbed VPT2), and the resonance is reported with
//! the two wavenumbers of the pair of states coupled by `phi / 4` (`phi / (2 sqrt 2)` for
//! three different modes) from the 2x2 problem of the deperturbed levels.
//!
//! **Zero-point energy.** The ground-state energy to the same order is computed directly,
//! not from `G_0` and the `x_ij`, so that it has no resonance denominators at all:
//!
//! ```text
//! E_0 = 1/2 sum_i w_i + sum_i phi_iiii / 32 + sum_(i<j) phi_iijj / 16
//!     - sum_i (phi_iii + sum_(j != i) phi_jji)^2 / (32 w_i) - sum_i phi_iii^2 / (144 w_i)
//!     - sum_(i != j) phi_iij^2 / (16 (2 w_i + w_j))
//!     - sum_(i<j<k) phi_ijk^2 / (8 (w_i + w_j + w_k))
//!     + 1/4 sum_(i<j) sum_a B_a (zeta^a_ij)^2 (w_i / w_j + w_j / w_i)
//! ```
//!
//! The terms of the rotational kinetic energy that do not depend on the vibrational state
//! (Watson's term and the constant part of the Coriolis term) are not included; they cancel
//! in every transition.
//!
//! The formulas are those of an asymmetric top. Degenerate vibrations (symmetric tops,
//! linear molecules) need their own treatment of the vibrational angular momentum; for a
//! linear molecule the Coriolis terms are left out and a note says so.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::ForceField;

/// Settings of the perturbation theory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vpt2Settings {
    /// Largest frequency gap (cm^-1) of a Fermi resonance (Martin et al.: 200).
    pub fermi_gap: f64,
    /// Smallest correction beyond perturbation theory (cm^-1) of a Fermi resonance
    /// (Martin et al.: 1).
    pub fermi_threshold: f64,
    /// Include the Coriolis terms.
    pub coriolis: bool,
}

impl Default for Vpt2Settings {
    fn default() -> Self {
        Self {
            fermi_gap: 200.0,
            fermi_threshold: 1.0,
            coriolis: true,
        }
    }
}

/// A Fermi resonance `w_i + w_j ~ w_k` (`i == j`: an overtone with a fundamental).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FermiResonance {
    /// The modes `i <= j` whose quanta add up.
    pub pair: (usize, usize),
    /// The mode `k` near their sum.
    pub single: usize,
    /// `w_i + w_j - w_k` (cm^-1).
    pub gap: f64,
    /// The cubic constant `phi_ijk` (cm^-1).
    pub constant: f64,
    /// The two wavenumbers of the coupled pair of states above the ground state (cm^-1).
    pub levels: (f64, f64),
}

/// Result of the perturbation theory (wavenumbers in cm^-1).
#[derive(Debug, Clone)]
pub struct Vpt2 {
    /// Harmonic wavenumbers.
    pub harmonic: Vec<f64>,
    /// Anharmonic constants `x_ij` (symmetric; resonant terms left out).
    pub constants: Mat<f64>,
    /// Fundamentals `1 <- 0` of every mode.
    pub fundamentals: Vec<f64>,
    /// First overtones `2 <- 0`.
    pub overtones: Vec<f64>,
    /// Harmonic zero-point energy.
    pub zero_point_harmonic: f64,
    /// Anharmonic zero-point energy.
    pub zero_point: f64,
    /// Fermi resonances found.
    pub resonances: Vec<FermiResonance>,
    /// Remarks on what was left out.
    pub notes: Vec<String>,
}

impl Vpt2 {
    /// Combination band `i + j <- 0` (`i != j`).
    pub fn combination(&self, i: usize, j: usize) -> f64 {
        let x = &self.constants;
        let rest: f64 = (0..self.harmonic.len())
            .filter(|k| *k != i && *k != j)
            .map(|k| x[(i, k)] + x[(j, k)])
            .sum();
        self.harmonic[i]
            + self.harmonic[j]
            + 2.0 * x[(i, i)]
            + 2.0 * x[(j, j)]
            + 2.0 * x[(i, j)]
            + 0.5 * rest
    }
}

/// Whether the term with the cubic constant `constant` and the denominator `gap` is a
/// Fermi resonance; `different` for three different modes.
fn resonant(constant: f64, gap: f64, different: bool, settings: &Vpt2Settings) -> bool {
    if gap.abs() > settings.fermi_gap {
        return false;
    }
    let weight = if different { 64.0 } else { 256.0 };
    // phi^4 / (weight |gap|^3) >= threshold, without dividing by a gap of zero.
    constant.powi(4) >= settings.fermi_threshold * weight * gap.abs().powi(3)
}

/// VPT2 for the force field `field`.
pub fn vpt2(field: &ForceField, settings: &Vpt2Settings) -> Vpt2 {
    let m = field.modes();
    let w = &field.harmonic;
    let mut notes = Vec::new();
    let mut resonances: Vec<FermiResonance> = Vec::new();
    // `1 / gap`, or zero and a record if the term (i + j ~ k) is resonant.
    let fraction = |i: usize, j: usize, k: usize, found: &mut Vec<FermiResonance>| -> f64 {
        let gap = w[i] + w[j] - w[k];
        let constant = field.cubic(i, j, k);
        if constant == 0.0 {
            return 0.0;
        }
        if resonant(constant, gap, i != j, settings) {
            let pair = (i.min(j), i.max(j));
            if !found.iter().any(|r| r.pair == pair && r.single == k) {
                found.push(FermiResonance {
                    pair,
                    single: k,
                    gap,
                    constant,
                    levels: (0.0, 0.0),
                });
            }
            0.0
        } else {
            1.0 / gap
        }
    };

    let rotation = settings.coriolis && field.rotational.iter().any(|b| *b != 0.0);
    let linear = field.rotational.iter().any(|b| b.is_infinite());
    if rotation && linear {
        notes.push(
            "Linear molecule: the Coriolis terms are left out (the formulas are those of an \
             asymmetric top)."
                .to_string(),
        );
    }
    let coriolis = |i: usize, j: usize| -> f64 {
        if !rotation || linear {
            return 0.0;
        }
        (0..3)
            .map(|a| {
                let zeta = field.coriolis[a][(i, j)];
                field.rotational[a] * zeta * zeta * (w[i] / w[j] + w[j] / w[i])
            })
            .sum()
    };

    let mut x = Mat::<f64>::zeros(m, m);
    for i in 0..m {
        let mut sum = 0.0;
        for k in 0..m {
            let phi = field.cubic(i, i, k);
            // 2 w_i ~ w_k: the term (i + i ~ k).
            sum += phi
                * phi
                * (2.0 / w[k] + 0.5 / (2.0 * w[i] + w[k])
                    - 0.5 * fraction(i, i, k, &mut resonances));
        }
        x[(i, i)] = field.quartic(i, i, i) / 16.0 - sum / 16.0;
    }
    for i in 0..m {
        for j in i + 1..m {
            let mut value = field.quartic(i, j, j) / 4.0;
            for k in 0..m {
                value -= field.cubic(i, i, k) * field.cubic(j, j, k) / (4.0 * w[k]);
                let phi = field.cubic(i, j, k);
                // The three fractions that can vanish: j + k ~ i, i + k ~ j, i + j ~ k.
                value -= phi * phi / 8.0
                    * (1.0 / (w[i] + w[j] + w[k])
                        + fraction(j, k, i, &mut resonances)
                        + fraction(i, k, j, &mut resonances)
                        - fraction(i, j, k, &mut resonances));
            }
            value += coriolis(i, j);
            x[(i, j)] = value;
            x[(j, i)] = value;
        }
    }

    let fundamentals: Vec<f64> = (0..m)
        .map(|i| {
            w[i] + 2.0 * x[(i, i)]
                + 0.5 * (0..m).filter(|j| *j != i).map(|j| x[(i, j)]).sum::<f64>()
        })
        .collect();
    let overtones: Vec<f64> = (0..m)
        .map(|i| {
            2.0 * w[i]
                + 6.0 * x[(i, i)]
                + (0..m).filter(|j| *j != i).map(|j| x[(i, j)]).sum::<f64>()
        })
        .collect();

    // The ground state directly.
    let zero_point_harmonic = 0.5 * w.iter().sum::<f64>();
    let mut zero_point = zero_point_harmonic;
    for i in 0..m {
        zero_point += field.quartic(i, i, i) / 32.0;
        let single: f64 = field.cubic(i, i, i)
            + (0..m)
                .filter(|j| *j != i)
                .map(|j| field.cubic(j, j, i))
                .sum::<f64>();
        zero_point -= single * single / (32.0 * w[i]);
        zero_point -= field.cubic(i, i, i).powi(2) / (144.0 * w[i]);
        for j in 0..m {
            if j != i {
                zero_point -= field.cubic(i, i, j).powi(2) / (16.0 * (2.0 * w[i] + w[j]));
            }
        }
        for j in i + 1..m {
            zero_point += field.quartic(i, j, j) / 16.0 + 0.25 * coriolis(i, j);
            for k in j + 1..m {
                zero_point -= field.cubic(i, j, k).powi(2) / (8.0 * (w[i] + w[j] + w[k]));
            }
        }
    }

    let mut result = Vpt2 {
        harmonic: w.clone(),
        constants: x,
        fundamentals,
        overtones,
        zero_point_harmonic,
        zero_point,
        resonances,
        notes,
    };
    // The coupled pairs of states from their deperturbed levels.
    let levels: Vec<(f64, f64)> = result
        .resonances
        .iter()
        .map(|r| {
            let (i, j) = r.pair;
            let (upper, coupling) = if i == j {
                (result.overtones[i], r.constant / 4.0)
            } else {
                (
                    result.combination(i, j),
                    r.constant / (2.0 * 2.0_f64.sqrt()),
                )
            };
            let lower = result.fundamentals[r.single];
            let mean = 0.5 * (upper + lower);
            let half = (0.25 * (upper - lower).powi(2) + coupling * coupling).sqrt();
            (mean - half, mean + half)
        })
        .collect();
    for (resonance, found) in result.resonances.iter_mut().zip(levels) {
        resonance.levels = found;
    }
    if !result.resonances.is_empty() {
        result.notes.push(
            "Fermi resonances: the resonant terms are left out of the anharmonic constants \
             (deperturbed); the coupled pairs of states are listed with the levels of their \
             2x2 problems."
                .to_string(),
        );
    }
    result
}
