//! The Zhu-Nakamura probability of a hop between two adiabatic states at a minimum of their
//! gap along a trajectory, from the straight two-state model of the crossing along the
//! hopping direction.
//!
//! The model is one-dimensional in mass-weighted coordinates (unit mass): the coupling `V`
//! (half the smallest gap), the diabatic forces `F1`, `F2` along the direction, given as
//! the length `|F2 - F1|` and the scalar product `F1 . F2` (whose sign is the type of the
//! crossing), and the crossing energy `E_X` (the mean of the two adiabatic energies at the
//! smallest gap). With the energy `E` of the motion along the direction (the potential
//! energy of the active state and the kinetic energy along the direction), the parameters
//! `a^2`, `b^2` of the review are those of eqs. 2.1-2.2 (Ishida, Nanbu, Nakamura 2017),
//! taken for the hop as Yu et al. do (eqs. 1-2, 4-5).
//!
//! - Sloped (Landau-Zener type, `F1 . F2 > 0`): classically allowed when `E` reaches the
//!   upper adiabatic energy at the smallest gap, `E >= E_X + V` (section 2.1.1); the
//!   probability of one passage is eq. 2.11,
//!   `p = exp(-(pi / 4a) (2 / (b^2 + sqrt(b^4 + 0.4 a^2 + 0.7)))^(1/2))`.
//! - Peaked (nonadiabatic-tunnelling type, `F1 . F2 < 0`): classically allowed above the
//!   bottom of the upper adiabatic energy, `b^2 >= 1` (section 2.1.2); `p` is eq. 2.26,
//!   `exp(-(pi / 4a) (2 / (b^2 + sqrt(b^4 - 0.72 + 0.62 a^1.43)))^(1/2))`.
//!
//! Both are the single-passage probabilities of the kinetics
//! ([`crate::kinetics::rate::zhu_nakamura::ZhuNakamuraCrossing::minus_ln_hop`],
//! [`crate::kinetics::rate::nonadiabatic_tunnelling::PeakedCrossing::minus_ln_hop`]), which
//! this module calls. Below those energies the transition is classically forbidden. For a
//! peaked crossing between the barrier top and the well bottom (`-1 < b^2 < 1`) and below
//! the top, the meaningful quantity is the overall transmission `P12` along the lower
//! state (eqs. 2.54-2.62, [`PeakedCrossing::ln_transmission`]); it is reported, and no hop
//! is made, since the upper state is closed there. For a sloped crossing below the upper
//! energy the transition would end at the turning point of the upper state away from the
//! point of the hop (eqs. 2.41-2.53); a hop at one point cannot conserve the energy there,
//! so none is made and the candidate is reported as forbidden.
//!
//! References: T. Ishida, S. Nanbu, H. Nakamura, Int. Rev. Phys. Chem. 2017, 36, 229 (the
//! authors' draft; eqs. 2.1-2.11, 2.26, 2.54-2.62, and section 3.1); L. Yu, C. Xu, Y. Lei,
//! C. Zhu, Z. Wen, Phys. Chem. Chem. Phys. 2014, 16, 25883 (eqs. 1-5); C. Zhu, H. Nakamura,
//! J. Chem. Phys. 1994, 101, 10630 (the theory; not read).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::Units;
use crate::kinetics::rate::nonadiabatic_tunnelling::PeakedCrossing;
use crate::kinetics::rate::zhu_nakamura::ZhuNakamuraCrossing;

/// The type of a crossing along the hopping direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingKind {
    /// The diabatic forces point the same way (Landau-Zener type).
    Sloped,
    /// The diabatic forces point opposite ways (nonadiabatic-tunnelling type).
    Peaked,
}

impl CrossingKind {
    /// The name in the log and the files.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sloped => "sloped",
            Self::Peaked => "peaked",
        }
    }
}

/// The straight two-state model of a crossing along the hopping direction, in
/// mass-weighted coordinates with unit (electron) mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingModel {
    /// The coupling, half the smallest adiabatic gap (Hartree).
    pub coupling: f64,
    /// `|F2 - F1|`, the length of the difference of the diabatic forces (Hartree per
    /// mass-weighted Bohr).
    pub force_difference: f64,
    /// `F1 . F2`, the scalar product of the diabatic forces (its sign is the type).
    pub force_product: f64,
    /// The crossing energy `E_X`, the mean of the two adiabatic energies at the smallest
    /// gap (Hartree).
    pub crossing_energy: f64,
}

/// The Zhu-Nakamura quantities of a hop at the energy of the motion along the direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HopProbability {
    /// The type of the crossing.
    pub kind: CrossingKind,
    /// `a^2`, the effective coupling (dimensionless).
    pub a2: f64,
    /// `b^2`, the effective collision energy (dimensionless).
    pub b2: f64,
    /// Whether the transition is classically allowed at this energy.
    pub allowed: bool,
    /// The probability that one passage changes the adiabatic state (allowed only).
    pub probability: Option<f64>,
    /// The overall transmission `P12` along the lower state of a peaked crossing below the
    /// bottom of the upper state (classically forbidden).
    pub transmission: Option<f64>,
}

/// One electron mass in amu: the mass of the mass-weighted coordinates.
fn unit_mass_amu() -> f64 {
    1.0 / Units::AMU_TO_AU_MASS
}

impl CrossingModel {
    /// The probability of a hop for the energy `energy` (Hartree) of the motion along the
    /// direction. `None` when the model has no direction (equal diabatic forces) or values
    /// that are not finite.
    pub fn probability(&self, energy: f64) -> Option<HopProbability> {
        let v = self.coupling.abs();
        let df = self.force_difference.abs();
        let finite = [v, df, self.force_product, self.crossing_energy, energy]
            .iter()
            .all(|x| x.is_finite());
        if !finite || df <= 0.0 {
            return None;
        }
        let kind = if self.force_product < 0.0 {
            CrossingKind::Peaked
        } else {
            CrossingKind::Sloped
        };
        let above = energy - self.crossing_energy;
        if v == 0.0 {
            // A true crossing at the smallest gap: the diabatic limit, a^2 -> infinity.
            let allowed = above >= 0.0;
            return Some(HopProbability {
                kind,
                a2: f64::INFINITY,
                b2: f64::INFINITY.copysign(above),
                allowed,
                probability: allowed.then_some(1.0),
                transmission: None,
            });
        }
        // A product of zero is the limit of a sloped crossing (b^2 -> infinity at a fixed
        // a^2 b^2, the Landau-Zener probability); a tiny one keeps the formulas finite.
        let tiny = 1e-12 * df * df;
        let product = match kind {
            CrossingKind::Sloped => self.force_product.max(tiny),
            CrossingKind::Peaked => self.force_product.min(-tiny),
        };
        let mass = unit_mass_amu();
        match kind {
            CrossingKind::Sloped => {
                let zn = ZhuNakamuraCrossing::from_diabatic(v, df, product, mass)?;
                let b2 = zn.b2(above);
                let allowed = above >= v;
                Some(HopProbability {
                    kind,
                    a2: zn.a2,
                    b2,
                    allowed,
                    probability: allowed.then(|| (-zn.minus_ln_hop(b2)).exp()),
                    transmission: None,
                })
            }
            CrossingKind::Peaked => {
                let zn = PeakedCrossing::from_diabatic(v, df, product, mass)?;
                let b2 = zn.b2(above);
                let allowed = b2 >= 1.0;
                Some(HopProbability {
                    kind,
                    a2: zn.a2,
                    b2,
                    allowed,
                    probability: allowed.then(|| (-zn.minus_ln_hop(b2)).exp()),
                    transmission: (!allowed).then(|| zn.ln_transmission(b2).exp()),
                })
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn the_probabilities_are_those_of_the_review_and_of_the_kinetics() {
        // F1 = 0.03, F2 = 0.01 (sloped) and F1 = 0.02, F2 = -0.01 (peaked) per mass-weighted
        // Bohr, V = 1e-3 Hartree, at energies above the crossing.
        for (f1, f2) in [(0.03_f64, 0.01_f64), (0.02, -0.01)] {
            let v = 1e-3;
            let model = CrossingModel {
                coupling: v,
                force_difference: (f2 - f1).abs(),
                force_product: f1 * f2,
                crossing_energy: -0.5,
            };
            for above in [0.0011, 0.003, 0.02, 0.2] {
                let hop = model.probability(-0.5 + above).unwrap();
                // Eqs. 2.1-2.2 with hbar = mu = 1.
                let f = (f1 * f2).abs().sqrt();
                let df = (f1 - f2).abs();
                let a2 = f * df / (16.0 * v.powi(3));
                let b2 = above * df / (2.0 * f * v);
                assert!((hop.a2 / a2 - 1.0).abs() < 1e-12, "{hop:?}");
                assert!((hop.b2 / b2 - 1.0).abs() < 1e-12, "{hop:?}");
                let a = a2.sqrt();
                let root = if f1 * f2 > 0.0 {
                    (b2 * b2 + 0.4 * a2 + 0.7).sqrt()
                } else {
                    (b2 * b2 - 0.72 + 0.62 * a.powf(1.43)).sqrt()
                };
                let p = (-(PI / (4.0 * a)) * (2.0 / (b2 + root)).sqrt()).exp();
                if hop.allowed {
                    assert!((hop.probability.unwrap() / p - 1.0).abs() < 1e-12);
                }
                // The kinetics module gives the same.
                let mass = unit_mass_amu();
                let kinetic = if f1 * f2 > 0.0 {
                    ZhuNakamuraCrossing::from_diabatic(v, df, f1 * f2, mass)
                        .unwrap()
                        .minus_ln_hop(b2)
                } else {
                    PeakedCrossing::from_diabatic(v, df, f1 * f2, mass)
                        .unwrap()
                        .minus_ln_hop(b2)
                };
                assert!((kinetic - (-p.ln())).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn what_is_classically_allowed() {
        let v = 2e-3;
        let sloped = CrossingModel {
            coupling: v,
            force_difference: 0.02,
            force_product: 3e-4,
            crossing_energy: 0.0,
        };
        // Up to the upper adiabatic energy at the smallest gap, E_X + V, a sloped crossing
        // is forbidden; above, allowed.
        assert!(!sloped.probability(0.9 * v).unwrap().allowed);
        assert!(sloped.probability(0.9 * v).unwrap().probability.is_none());
        assert!(sloped.probability(1.1 * v).unwrap().allowed);
        // A peaked crossing is allowed from the bottom of the upper well (b^2 = 1) up and
        // reports the transmission of the lower state below it.
        let peaked = CrossingModel {
            force_product: -0.5e-4,
            ..sloped
        };
        let low = peaked.probability(0.0).unwrap();
        assert_eq!(low.kind, CrossingKind::Peaked);
        assert!(!low.allowed && low.b2.abs() < 1e-12);
        let t = low.transmission.unwrap();
        assert!(t > 0.0 && t < 1.0, "{t}");
        let bottom = 1.0
            / PeakedCrossing::from_diabatic(v, 0.02, -0.5e-4, unit_mass_amu())
                .unwrap()
                .b2_per_hartree;
        assert!(peaked.probability(1.001 * bottom).unwrap().allowed);
        // Equal forces: no direction; a zero product: the sloped limit stays finite.
        let flat = CrossingModel {
            force_difference: 0.0,
            ..sloped
        };
        assert!(flat.probability(1.0).is_none());
        let zero = CrossingModel {
            force_product: 0.0,
            ..sloped
        };
        let p = zero.probability(0.01).unwrap();
        assert_eq!(p.kind, CrossingKind::Sloped);
        // The Landau-Zener limit exp(-pi / (4 sqrt(a^2 b^2))), a^2 b^2 = df^2 E / (32 V^4).
        let lz = (-PI / (4.0 * (0.02_f64.powi(2) * 0.01 / (32.0 * v.powi(4))).sqrt())).exp();
        assert!(
            (p.probability.unwrap() / lz - 1.0).abs() < 1e-4,
            "{p:?} {lz}"
        );
        // A true degeneracy hops with certainty above the crossing.
        let degenerate = CrossingModel {
            coupling: 0.0,
            ..sloped
        };
        assert_eq!(degenerate.probability(0.01).unwrap().probability, Some(1.0));
    }
}
