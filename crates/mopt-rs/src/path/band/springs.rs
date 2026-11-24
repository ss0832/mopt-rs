//! Springs between the images of a chain, stiffer where the energy is high.
//!
//! The chains of this module keep their images apart by putting them back along the chain
//! ([`super::redistribute`]). The nudged elastic band does it with springs: an inner image
//! gets, besides the force of the surface across the chain, the force
//! `(k_i d_i - k_(i-1) d_(i-1)) t` along its unit tangent `t`, with `d_i` the distance to
//! the next image and `k_i` the constant of the spring between the images `i` and `i + 1`
//! (Henkelman, Jonsson, J. Chem. Phys. 2000, 113, 9978). With energy-weighted springs
//! (Asgeirsson, Birgisson, Bjornsson, Becker, Neese, Riplinger, Jonsson, J. Chem. Theory
//! Comput. 2021, 17, 4929) the constant grows with the energy of the higher of its two
//! images,
//!
//! ```text
//! k = k_lower                               for E <= E_ref
//! k = (1 - a) k_upper + a k_lower,          a = (E_max - E) / (E_max - E_ref)   otherwise
//! ```
//!
//! (`E_ref`: the higher of the two end images; `E_max`: the highest image), so that the
//! images lie closer together near the top of the barrier. The formula and the constants
//! (`0.1` and `0.01` Hartree/Bohr^2) are as later papers and the manual of ORCA, where the
//! method was published, give them; the text of the publication was not at hand.
//!
//! A quasi-Newton step across the chain knows nothing of the springs. The step that
//! brings the two springs of an image to balance, `(k_i d_i - k_(i-1) d_(i-1)) / (k_i +
//! k_(i-1))` along the tangent, is its counterpart along the chain
//! ([`SpringPull::newton`]). An image that climbs has no springs.
//!
//! MultiOptPy's `-ewbneb` differs in three ways, kept under
//! [`Compatibility::MultiOptPy`]: the constant belongs to an image, not to a spring, and
//! falls from the upper one at the energy of the lower end to the lower one at the top
//! (the other way round); what is added is the constant times the unit vector to the next
//! image minus that of the image before times the unit vector from the image before (no
//! distances, so nothing that pulls harder when stretched); and it is added to the
//! gradient, so that the force pushes an image away from the next one. Its constants are
//! `0.005` and `0.0001`.
//!
//! MultiOptPy: `MEP/pathopt_ewbneb_force.py:CaluculationEWBNEB`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{path_lengths, ChainKind, Compatibility};
use crate::math::linalg::col_norm;

/// Energy-weighted springs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyWeightedSprings {
    /// Constant of a spring at the highest image (Hartree/Bohr^2).
    pub upper: f64,
    /// Constant of a spring at or below the energy of the higher end (Hartree/Bohr^2).
    pub lower: f64,
}

impl Default for EnergyWeightedSprings {
    fn default() -> Self {
        Self {
            upper: 0.1,
            lower: 0.01,
        }
    }
}

/// What the springs do to one image.
#[derive(Debug, Clone)]
pub struct SpringPull {
    /// The force of its two springs along its tangent.
    pub force: Col<f64>,
    /// The step along the tangent that brings the two springs to balance.
    pub newton: Col<f64>,
}

impl EnergyWeightedSprings {
    /// MultiOptPy's constants.
    pub fn of_multioptpy() -> Self {
        Self {
            upper: 0.005,
            lower: 0.0001,
        }
    }

    /// The constants of the springs between the images `s` and `s + 1` of a chain with
    /// the energies `energies`.
    pub fn constants(&self, energies: &[f64]) -> Vec<f64> {
        let (Some(first), Some(last)) = (energies.first(), energies.last()) else {
            return Vec::new();
        };
        let reference = first.max(*last);
        let highest = energies.iter().fold(f64::NEG_INFINITY, |m, e| m.max(*e));
        energies
            .windows(2)
            .map(|pair| {
                let energy = pair[0].max(pair[1]);
                if energy <= reference || highest <= reference {
                    self.lower
                } else {
                    let a = ((highest - energy) / (highest - reference)).clamp(0.0, 1.0);
                    (1.0 - a) * self.upper + a * self.lower
                }
            })
            .collect()
    }

    /// The pull of the springs on every image of `chain` (nothing on the end images and
    /// on the images `climbing`); `tangents` are the directions of the chain at the
    /// images, of any length.
    pub fn pulls(
        &self,
        chain: &[Col<f64>],
        energies: &[f64],
        tangents: &[Col<f64>],
        climbing: &[usize],
        kind: ChainKind,
        compatibility: Compatibility,
    ) -> Vec<SpringPull> {
        let images = chain.len();
        let n = chain.first().map_or(0, Col::nrows);
        let none = || SpringPull {
            force: Col::zeros(n),
            newton: Col::zeros(n),
        };
        let mut pulls: Vec<SpringPull> = (0..images).map(|_| none()).collect();
        if images < 3 {
            return pulls;
        }
        if compatibility == Compatibility::MultiOptPy {
            // A constant per image, falling towards the top; unit vectors; the sign of a
            // term of the gradient.
            let reference = energies[0].min(energies[images - 1]);
            let highest = energies.iter().fold(f64::NEG_INFINITY, |m, e| m.max(*e));
            let span = (highest - reference).max(1e-8);
            let of_image: Vec<f64> = (0..images)
                .map(|i| {
                    if i == 0 || i == images - 1 || energies[i] == reference {
                        0.0
                    } else if energies[i] < reference {
                        self.lower
                    } else {
                        let a = (energies[i] - reference) / span;
                        self.upper * (1.0 - a) + a * self.lower
                    }
                })
                .collect();
            let unit = |v: Col<f64>| {
                let length = col_norm(&v);
                if length > 1e-8 {
                    v * (1.0 / length)
                } else {
                    Col::zeros(n)
                }
            };
            for i in 1..images - 1 {
                let forward = unit(&chain[i + 1] - &chain[i]);
                let backward = unit(&chain[i] - &chain[i - 1]);
                pulls[i].force = -(of_image[i] * forward - of_image[i - 1] * backward);
            }
            return pulls;
        }
        let constants = self.constants(energies);
        let lengths = path_lengths(chain, kind);
        for i in 1..images - 1 {
            if climbing.contains(&i) {
                continue;
            }
            let length = col_norm(&tangents[i]);
            if length < 1e-12 {
                continue;
            }
            let tangent = &tangents[i] * (1.0 / length);
            let (before, after) = (lengths[i] - lengths[i - 1], lengths[i + 1] - lengths[i]);
            let (k_before, k_after) = (constants[i - 1], constants[i]);
            let size = k_after * after - k_before * before;
            pulls[i].force = &tangent * size;
            let stiffness = k_after + k_before;
            if stiffness > 0.0 {
                pulls[i].newton = &tangent * (size / stiffness);
            }
        }
        pulls
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    #[test]
    fn the_springs_are_stiffest_at_the_top_and_pull_the_images_there_together() {
        let springs = EnergyWeightedSprings::default();
        // A barrier of 1 above the first image; the last image 0.2 above the first.
        let energies = [0.0, 0.3, 1.0, 0.6, 0.2];
        let k = springs.constants(&energies);
        assert_eq!(k.len(), 4);
        // The two springs at the highest image have the upper constant; the others fall
        // with the energy of their higher image, down to the lower one at the reference.
        assert!((k[1] - 0.1).abs() < 1e-15 && (k[2] - 0.1).abs() < 1e-15);
        let expected = 0.125 * 0.1 + 0.875 * 0.01;
        assert!((k[0] - expected).abs() < 1e-15, "{}", k[0]);
        assert!((k[3] - 0.055).abs() < 1e-15, "{}", k[3]);
        // A flat profile: the lower constant everywhere.
        assert!(springs
            .constants(&[0.5, 0.5, 0.5])
            .iter()
            .all(|c| (*c - 0.01).abs() < 1e-15));

        // Images at equal distances on a line: the image below the top is pulled towards
        // it (its upper spring is stiffer), the top image stays, and the step that
        // balances the springs has the size of the force over the two constants.
        let chain: Vec<Col<f64>> = (0..5).map(|i| col(&[f64::from(i), 0.0])).collect();
        let tangents: Vec<Col<f64>> = (0..5).map(|_| col(&[2.0, 0.0])).collect();
        let pulls = springs.pulls(
            &chain,
            &energies,
            &tangents,
            &[],
            ChainKind::Variables,
            Compatibility::Consistent,
        );
        assert!(
            pulls[0].force.iter().all(|f| *f == 0.0) && pulls[4].force.iter().all(|f| *f == 0.0)
        );
        assert!((pulls[1].force[0] - (0.1 - expected)).abs() < 1e-15);
        assert!(pulls[2].force[0].abs() < 1e-15);
        assert!((pulls[3].force[0] - (0.055 - 0.1)).abs() < 1e-15);
        assert!((pulls[1].newton[0] - (0.1 - expected) / (0.1 + expected)).abs() < 1e-15);
        // An image that climbs has no springs.
        let climbing = springs.pulls(
            &chain,
            &energies,
            &tangents,
            &[3],
            ChainKind::Variables,
            Compatibility::Consistent,
        );
        assert!(climbing[3].force.iter().all(|f| *f == 0.0));

        // MultiOptPy's form: constants per image that fall towards the top, unit vectors,
        // and the sign of a term of the gradient.
        let old = EnergyWeightedSprings::of_multioptpy();
        let pulls = old.pulls(
            &chain,
            &energies,
            &tangents,
            &[],
            ChainKind::Variables,
            Compatibility::MultiOptPy,
        );
        // Image 2 (the top): its own constant is the lower one, that of image 1 is
        // 0.005 * 0.7 + 0.3 * 0.0001.
        let of_one = 0.005 * 0.7 + 0.3 * 0.0001;
        assert!(
            (pulls[2].force[0] + (0.0001 - of_one)).abs() < 1e-15,
            "{}",
            pulls[2].force[0]
        );
    }
}
