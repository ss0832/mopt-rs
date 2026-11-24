//! The Zhu-Nakamura probability of the passages through a crossing of two electronic states,
//! for a crossing of the Landau-Zener type (sloped: the two diabatic energies have slopes of
//! the same sign along the hopping coordinate). A peaked crossing (slopes of opposite signs,
//! the nonadiabatic-tunnelling type) is in [`super::nonadiabatic_tunnelling`], at every energy
//! including those below the crossing; the rate correction `zhu_nakamura` takes the type from
//! the sign of the product of the slopes.
//!
//! In atomic units (`hbar = 1`), with the coupling `V` of the two states, the slopes `F1`,
//! `F2` of their diabatic energies along the hopping coordinate, the reduced mass `mu` along
//! it, and the energy `E` above the crossing energy `E_X`:
//!
//! ```text
//! a^2 = sqrt(F1 F2) |F1 - F2| / (2 mu (2 V)^3),
//! b^2 = (E - E_X) |F1 - F2| / (2 sqrt(F1 F2) V),
//! p   = exp(-(pi / (4 a)) sqrt(2 / (b^2 + sqrt(b^4 + 0.4 a^2 + 0.7))))      (E >= E_X),
//! ```
//!
//! `p` being the probability that one passage changes the adiabatic state. It tends to the
//! Landau-Zener `exp(-pi / (4 a b))` for large `b` and, unlike it, stays finite at the
//! crossing energy. The same `a` and `b` follow from the adiabatic energies alone (Zhu and
//! Nakamura's form for potentials that are not straight lines): with the point `r0` of the
//! smallest gap, the crossing energy `E_X = (E1(r0) + E2(r0)) / 2`, and the points `t1`,
//! `t2` where the lower and the upper adiabatic energies reach `E_X`,
//!
//! ```text
//! d^2 = (E2(t1) - E1(t1)) (E2(t2) - E1(t2)) / (E2(r0) - E1(r0))^2,
//! a^2 = sqrt(d^2 - 1) / (mu (t2 - t1)^2 (E2(r0) - E1(r0))),
//! b^2 = sqrt(d^2 - 1) (E - E_X) / ((E2(r0) - E1(r0)) / 2),
//! ```
//!
//! which is the two-state model with `V = (E2(r0) - E1(r0)) / 2`,
//! `sqrt(F1 F2) = 2 V / (t2 - t1)`, and `|F1 - F2| = 2 sqrt(F1 F2) sqrt(d^2 - 1)`
//! ([`ZhuNakamuraCrossing::from_adiabatic`]).
//!
//! At a sloped crossing a passage in and the passage back out change the diabatic state
//! together when exactly one of them is nonadiabatic: `4 p (1 - p) sin^2 psi`, and
//! `2 p (1 - p)` with the phase `psi` averaged. The rate correction `zhu_nakamura` is the
//! thermal average of that over the energies above the crossing,
//! `int_0^inf 2 p (1 - p) exp(-E / k_B T) dE / k_B T`, like the Landau-Zener average of
//! [`super::crossing`] (the flux is counted from the crossing point).
//!
//! Covered for a sloped crossing: the energies above the crossing energy. Not built: those
//! below it (eqs. 2.41-2.53 of Ishida, Nanbu, Nakamura, Int. Rev. Phys. Chem. 2017, 36, 229,
//! with the phases `sigma` and `delta` from the actions of the adiabatic energies), which add
//! tunnelling below the crossing.
//!
//! References: C. Zhu, H. Nakamura, J. Chem. Phys. 1994, 101, 10630 and 1995, 102, 7448
//! (the theory; the texts were not read); the formulas above as given by A. O. Lykhin,
//! S. A. Varganov, Phys. Chem. Chem. Phys. 2020, 22, 5500 (eqs. 12-17) and by J. Lehto,
//! arXiv:1304.3331 (eq. 29), who quotes the appendices of H. Nakamura, Nonadiabatic
//! Transition: Concepts, Basic Theories and Applications (World Scientific, 2012). The two
//! forms of `a^2` and `b^2` were checked against each other for straight diabatic states.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::nonadiabatic_tunnelling::PeakedCrossing;
use super::{Diagnostic, RateCorrection, StepContext};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::conditions::Conditions;
use crate::kinetics::network::{Bottleneck, BottleneckKind, KineticNetwork};
use crate::kinetics::quadrature::ln_integral;

/// The two Zhu-Nakamura parameters of a sloped crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZhuNakamuraCrossing {
    /// `a^2` (dimensionless).
    pub a2: f64,
    /// `b^2 / (E - E_X)` (per Hartree).
    pub b2_per_hartree: f64,
}

impl ZhuNakamuraCrossing {
    /// From the two-state model: the coupling `coupling` (Hartree), the slope difference
    /// `slope_difference` = `|F1 - F2|` (Hartree/Bohr), the product of the slopes
    /// `slope_product` = `F1 F2` (Hartree^2/Bohr^2, positive for a sloped crossing), and the
    /// reduced mass `mass_amu` along the hopping coordinate. `None` for a peaked crossing (a
    /// negative product) and for values that are not finite and positive.
    pub fn from_diabatic(
        coupling: f64,
        slope_difference: f64,
        slope_product: f64,
        mass_amu: f64,
    ) -> Option<Self> {
        let v = coupling.abs();
        let df = slope_difference.abs();
        let ok = |x: f64| x.is_finite() && x > 0.0;
        if !(ok(v) && ok(df) && ok(slope_product) && ok(mass_amu)) {
            return None;
        }
        let mass = mass_amu * Units::AMU_TO_AU_MASS;
        let mean = slope_product.sqrt();
        Some(Self {
            a2: mean * df / (2.0 * mass * (2.0 * v).powi(3)),
            b2_per_hartree: df / (2.0 * mean * v),
        })
    }

    /// From the adiabatic energies `lower` and `upper` (Hartree) at the points `arc` of a
    /// path (positions along the hopping coordinate in Bohr, for the reduced mass
    /// `mass_amu`; in amu^(1/2) Bohr with `mass_amu` 1 for a mass-weighted path). `None`
    /// when the smallest gap is at an end of the path, or the lower and the upper energies
    /// do not both reach the crossing energy on opposite sides of it inside the path (a
    /// peaked crossing, where the lower energy has a barrier below the crossing energy, or
    /// a path too short).
    pub fn from_adiabatic(
        arc: &[f64],
        lower: &[f64],
        upper: &[f64],
        mass_amu: f64,
    ) -> Option<Self> {
        let model = adiabatic_model(arc, lower, upper)?;
        Self::from_diabatic(
            model.coupling,
            model.slope_difference,
            model.mean_slope * model.mean_slope,
            mass_amu,
        )
    }

    /// `b^2` at the energy `above` (Hartree) above the crossing.
    pub fn b2(&self, above: f64) -> f64 {
        self.b2_per_hartree * above
    }

    /// `-ln p` at `b^2 >= 0`, `p` being the probability that one passage changes the
    /// adiabatic state.
    pub fn minus_ln_hop(&self, b2: f64) -> f64 {
        let a = self.a2.sqrt();
        let inner = b2 + (b2 * b2 + 0.4 * self.a2 + 0.7).sqrt();
        PI / (4.0 * a) * (2.0 / inner).sqrt()
    }

    /// `ln` of the thermal average of `2 p (1 - p)` over the energies above the crossing at
    /// `kt` (Hartree).
    pub fn ln_average(&self, kt: f64) -> f64 {
        // E = kT s^2: int_0^inf 2 s exp(-s^2) 2 p (1 - p) ds; the tail beyond s = 9 is below
        // 1e-35.
        ln_integral(
            |s| {
                let x = self.minus_ln_hop(self.b2(kt * s * s));
                (4.0 * s).ln() - s * s - x + (-(-x).exp_m1()).ln()
            },
            0.0,
            9.0,
            200,
            16,
        )
    }
}

/// The straight two-state model of Zhu and Nakamura's adiabatic form (see the module text).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiabaticModel {
    /// Where the gap is smallest (the unit of `arc`).
    pub position: f64,
    /// The crossing energy `(E1(r0) + E2(r0)) / 2` (Hartree).
    pub energy: f64,
    /// The coupling, half the smallest gap (Hartree).
    pub coupling: f64,
    /// `sqrt(F1 F2) = 2 V / (t2 - t1)` with the sign of the slope of the mean energy.
    pub mean_slope: f64,
    /// `|F1 - F2| = 2 sqrt(F1 F2) sqrt(d^2 - 1)`.
    pub slope_difference: f64,
}

/// Where the piecewise linear `values` at `arc` reaches `level` going from the point `from`
/// towards `step` (+1 or -1): the first crossing of `level`, by linear interpolation.
pub(super) fn reach(
    arc: &[f64],
    values: &[f64],
    from: usize,
    step: isize,
    level: f64,
) -> Option<f64> {
    let mut k = from;
    loop {
        let next = k.checked_add_signed(step)?;
        if next >= values.len() {
            return None;
        }
        let (here, there) = (values[k] - level, values[next] - level);
        if here == 0.0 {
            return Some(arc[k]);
        }
        if here * there <= 0.0 {
            let t = here / (here - there);
            return Some(arc[k] + t * (arc[next] - arc[k]));
        }
        k = next;
    }
}

/// `values` at the position `x` by linear interpolation (`x` inside `arc`).
pub(super) fn at(arc: &[f64], values: &[f64], x: f64) -> f64 {
    let k = arc
        .windows(2)
        .position(|w| x >= w[0].min(w[1]) && x <= w[0].max(w[1]))
        .unwrap_or(0);
    let (x0, x1) = (arc[k], arc[k + 1]);
    if x1 == x0 {
        return values[k];
    }
    values[k] + (x - x0) / (x1 - x0) * (values[k + 1] - values[k])
}

/// The straight two-state model of a sloped crossing from the adiabatic energies `lower`
/// and `upper` at the points `arc` (increasing): the smallest gap at a point inside the
/// path, the lower energy reaching the crossing energy on one side and the upper on the
/// other. `None` otherwise.
pub fn adiabatic_model(arc: &[f64], lower: &[f64], upper: &[f64]) -> Option<AdiabaticModel> {
    let n = arc.len().min(lower.len()).min(upper.len());
    if n < 3 || arc.windows(2).any(|w| w[1] <= w[0]) {
        return None;
    }
    let gap = |k: usize| upper[k] - lower[k];
    let m = (0..n).min_by(|&a, &b| gap(a).total_cmp(&gap(b)))?;
    if m == 0 || m + 1 >= n || gap(m) <= 0.0 {
        return None;
    }
    let (arc, lower, upper) = (&arc[..n], &lower[..n], &upper[..n]);
    let energy = 0.5 * (lower[m] + upper[m]);
    let v = 0.5 * gap(m);
    // The lower energy rises to E_X on the side where the mean rises, the upper falls to it
    // on the other side.
    let rising = (lower[m + 1] + upper[m + 1]) >= (lower[m - 1] + upper[m - 1]);
    let (up, down) = if rising { (1, -1) } else { (-1, 1) };
    let t1 = reach(arc, lower, m, up, energy)?;
    let t2 = reach(arc, upper, m, down, energy)?;
    let width = (t1 - t2).abs();
    if width <= 0.0 {
        return None;
    }
    let gap_at = |x: f64| at(arc, upper, x) - at(arc, lower, x);
    let d2 = gap_at(t1) * gap_at(t2) / (gap(m) * gap(m));
    if d2.is_nan() || d2 <= 1.0 {
        return None;
    }
    let mean = 2.0 * v / width;
    Some(AdiabaticModel {
        position: arc[m],
        energy,
        coupling: v,
        mean_slope: if rising { mean } else { -mean },
        slope_difference: 2.0 * mean * (d2 - 1.0).sqrt(),
    })
}

/// The Zhu-Nakamura probability at crossings with a coupling: sloped above the crossing
/// energy, peaked at every energy from the higher minimum up.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZhuNakamura;

impl RateCorrection for ZhuNakamura {
    fn name(&self) -> &str {
        "zhu_nakamura"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        if b.kind != BottleneckKind::Crossing {
            return Ok(None);
        }
        let Some(h12) = b.coupling else {
            return Ok(None);
        };
        let (Some(df), Some(product), Some(mu)) =
            (b.gradient_difference, b.slope_product, b.reduced_mass)
        else {
            return Err(MoptError::config(format!(
                "{}: the Zhu-Nakamura probability needs the gradient difference, the product \
                 of the slopes of the two states, and the reduced mass along the hopping \
                 coordinate besides the coupling (annotations: gradient_difference, \
                 slope_product, reduced_mass_amu, or diabatic_crossings)",
                b.label
            )));
        };
        if product < 0.0 {
            let Some(zn) = PeakedCrossing::from_diabatic(h12, df, product, mu) else {
                return Err(MoptError::config(format!(
                    "{}: a peaked crossing needs a finite and positive coupling, gradient \
                     difference and reduced mass, and a gradient difference of at least \
                     2 sqrt(|slope_product|) (got {df:.3e} and {product:.3e})",
                    b.label
                )));
            };
            let lowest = step.barriers[0].min(step.barriers[1]);
            let avg = zn.average(c.kt(), lowest);
            return Ok(Some((
                avg.ln_kappa,
                vec![
                    Diagnostic::new("p_sh", avg.ln_kappa.exp()),
                    Diagnostic::new("zn_a2", zn.a2),
                    Diagnostic::new("peaked", 1.0),
                    Diagnostic::new("below_crossing", avg.below_crossing),
                ],
            )));
        }
        let Some(zn) = ZhuNakamuraCrossing::from_diabatic(h12, df, product, mu) else {
            return Err(MoptError::config(format!(
                "{}: the coupling, the gradient difference, the product of the slopes, and \
                 the reduced mass must be finite and positive",
                b.label
            )));
        };
        let ln_p = zn.ln_average(c.kt());
        Ok(Some((
            ln_p,
            vec![
                Diagnostic::new("p_sh", ln_p.exp()),
                Diagnostic::new("zn_a2", zn.a2),
                Diagnostic::new("peaked", 0.0),
            ],
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probability_tends_to_landau_zener_far_above_the_crossing() {
        let zn = ZhuNakamuraCrossing {
            a2: 0.25,
            b2_per_hartree: 1.0,
        };
        for b2 in [400.0, 2500.0] {
            let lz = PI / (4.0 * (zn.a2 * b2).sqrt());
            let ratio = zn.minus_ln_hop(b2) / lz;
            assert!((ratio - 1.0).abs() < 1e-3, "{b2} {ratio}");
        }
        // Finite at the crossing energy, where Landau-Zener diverges.
        let at_crossing = zn.minus_ln_hop(0.0);
        let want = PI / (4.0 * 0.5) * (2.0 / (0.4f64 * 0.25 + 0.7).sqrt()).sqrt();
        assert!((at_crossing - want).abs() < 1e-14);
    }

    #[test]
    fn the_adiabatic_form_gives_the_straight_two_state_model() {
        // V1 = -F1 x, V2 = -F2 x, coupling V: the adiabatic energies of a sloped crossing.
        let (f1, f2, v) = (0.03, 0.01, 0.002);
        let arc: Vec<f64> = (0..801).map(|k| -2.0 + 0.005 * k as f64).collect();
        let mean = 0.5 * (f1 + f2);
        let half = 0.5 * (f1 - f2);
        let root = |x: f64| ((half * x).powi(2) + v * v).sqrt();
        let lower: Vec<f64> = arc.iter().map(|&x| -mean * x - root(x)).collect();
        let upper: Vec<f64> = arc.iter().map(|&x| -mean * x + root(x)).collect();
        let model = adiabatic_model(&arc, &lower, &upper).unwrap();
        assert!((model.coupling - v).abs() < 1e-12);
        assert!(model.energy.abs() < 1e-12);
        let product = model.mean_slope * model.mean_slope;
        assert!((product / (f1 * f2) - 1.0).abs() < 1e-3, "{product}");
        assert!((model.slope_difference / (f1 - f2) - 1.0).abs() < 1e-3);
        assert!(model.mean_slope < 0.0);
        let from_path = ZhuNakamuraCrossing::from_adiabatic(&arc, &lower, &upper, 2.0).unwrap();
        let straight = ZhuNakamuraCrossing::from_diabatic(v, f1 - f2, f1 * f2, 2.0).unwrap();
        assert!((from_path.a2 / straight.a2 - 1.0).abs() < 2e-3);
        assert!((from_path.b2_per_hartree / straight.b2_per_hartree - 1.0).abs() < 2e-3);
        // A peaked crossing (slopes of opposite signs) has no model of this type.
        let lower: Vec<f64> = arc
            .iter()
            .map(|&x| -0.5 * (f1 - f2) * x - root_peaked(x, f1, f2, v))
            .collect();
        let upper: Vec<f64> = arc
            .iter()
            .map(|&x| -0.5 * (f1 - f2) * x + root_peaked(x, f1, f2, v))
            .collect();
        assert!(adiabatic_model(&arc, &lower, &upper).is_none());
    }

    /// The half gap of V1 = -F1 x, V2 = +F2 x with coupling V.
    fn root_peaked(x: f64, f1: f64, f2: f64, v: f64) -> f64 {
        ((0.5 * (f1 + f2) * x).powi(2) + v * v).sqrt()
    }

    #[test]
    fn the_thermal_average_tends_to_the_landau_zener_average_when_hot() {
        // Far above the crossing (b^4 >> a^2) the two agree: 2 P (1 - P) averaged with
        // P = exp(-pi / (4 a b)).
        let zn = ZhuNakamuraCrossing {
            a2: 0.04,
            b2_per_hartree: 1e5,
        };
        let kt = 0.01;
        let lz = ln_integral(
            |s| {
                let x = PI / (4.0 * (zn.a2 * zn.b2(kt * s * s)).sqrt());
                (4.0 * s).ln() - s * s - x + (-(-x).exp_m1()).ln()
            },
            0.0,
            9.0,
            200,
            16,
        );
        let ratio = (zn.ln_average(kt) - lz).exp();
        assert!((ratio - 1.0).abs() < 1e-2, "{ratio}");
        // The phase-averaged probability is at most 1/2.
        assert!(zn.ln_average(kt).exp() <= 0.5 + 1e-12);
    }
}
