//! The hopping direction and the two-state model of a gap minimum without nonadiabatic
//! couplings, and the change of the velocities that conserves the energy in a hop.
//!
//! Yu, Xu, Lei, Zhu, and Wen build the diabatic forces at the middle step `t2` of three
//! consecutive steps with the smallest gap from the adiabatic forces at the outer ones
//! (eqs. 7-8): the first diabatic state joins the upper adiabatic force at `t1` to the
//! lower one at `t3`, the second the lower at `t1` to the upper at `t3`,
//!
//! ```text
//! F1(t2) = (F_up(t1) + F_low(t3)) / 2,    F2(t2) = (F_low(t1) + F_up(t3)) / 2,
//! ```
//!
//! here as the linear connection in time over steps of equal length; the paper writes it
//! per Cartesian component as a connection in the coordinate, which is the same for the
//! straight motion of a short step and is not defined for a component that does not move.
//! In mass-weighted coordinates `F~ = M^(-1/2) F` the model of the crossing has
//! `|F2 - F1| = |F~2 - F~1|` and `F1 F2 = F~1 . F~2` (eqs. 4-5, with the sign kept for the
//! type of the crossing), the coupling `V = (U_up(t2) - U_low(t2)) / 2`, and the hopping
//! direction along `F~2 - F~1` (eqs. 9-10), the direction in which the probability of
//! eq. 11 is largest. Eq. 10 of the paper normalizes the vector of each nucleus; here the
//! whole mass-weighted vector is normalized, so that the kinetic energy along the direction
//! is that of the one coordinate of the model (eqs. 4-5 are sums over all nuclei too).
//!
//! The adiabatic forces at `t1` and `t3` are those of the diabatic states only away from
//! the coupling region. In the straight two-state model (constant diabatic forces and
//! coupling) the difference of the adiabatic gradients is `(D / Delta) grad D` with the
//! diabatic difference `D`, the gap `Delta = sqrt(D^2 + 4 V^2)`, and
//! `|D / Delta| = sqrt(1 - (2V / Delta)^2)`, so that eqs. 7-8 shorten `|F2 - F1|` by that
//! factor at the outer steps (by 11 % two couplings away, and to nothing as the steps
//! shrink inside the coupling region). [`ForceModel::TwoState`] divides the differences at
//! the outer steps by it, and takes `V` from the vertex of the parabola of the squared gap
//! through the three steps (exact for the straight model passed at a constant velocity),
//! where [`ForceModel::Published`] takes half the gap at `t2`; both keep the mean force of
//! eqs. 7-8. The factor is bounded below by [`MIN_UNMIXING`] for steps that all lie in the
//! coupling region.
//!
//! The direction replaces the nonadiabatic coupling vector of the review of Ishida, Nanbu,
//! and Nakamura (section 3.1) and its approximation by the eigenvector of the Hessian of
//! the gap (their eq. 3.5): it needs the gradients of the two states at two steps, where the
//! Hessian of the gap needs the Hessians of both states at the hop. For the excited states of
//! `gfn0xtb`, whose gradients are central differences of excitation energies (`6N` excited-
//! state calculations each), a Hessian by differences of those gradients costs `6N` gradients
//! (`36 N^2` excited-state calculations) per state and hop, against two extra gradients per
//! gap minimum for this direction.
//!
//! In a hop only the velocity component along the direction changes, keeping its sign
//! (eqs. 13-15): with the mass-weighted unit direction `u` (`sum m u^2 = 1`) and
//! `c = sum m v u`, `c' = sign(c) sqrt(c^2 - 2 dU)` for the change `dU` of the potential
//! energy, and `v' = v + (c' - c) u`; the kinetic energy changes by `-dU` exactly. When
//! `c^2 < 2 dU` the hop is frustrated.
//!
//! References: L. Yu, C. Xu, Y. Lei, C. Zhu, Z. Wen, Phys. Chem. Chem. Phys. 2014, 16,
//! 25883 (eqs. 4-15); T. Ishida, S. Nanbu, H. Nakamura, Int. Rev. Phys. Chem. 2017, 36, 229
//! (section 3.1, eqs. 3.1-3.5).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::probability::CrossingModel;

/// The adiabatic energies and gradients of the lower and the upper state at one step.
#[derive(Debug, Clone, Copy)]
pub struct StatePair<'a> {
    /// Energy of the lower state (Hartree).
    pub lower: f64,
    /// Energy of the upper state (Hartree).
    pub upper: f64,
    /// Gradient of the lower state (Hartree / Bohr).
    pub lower_gradient: &'a Col<f64>,
    /// Gradient of the upper state (Hartree / Bohr).
    pub upper_gradient: &'a Col<f64>,
}

/// How the diabatic forces and the coupling of a gap minimum are taken from the adiabatic
/// states (see the module text).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceModel {
    /// Eqs. 7-8 of Yu et al. and half the gap at the middle step.
    Published,
    /// The same with the mixing of the adiabatic states at the outer steps taken out by
    /// the straight two-state model, and the coupling from the parabola of the squared gap.
    TwoState,
}

impl ForceModel {
    /// The name in the settings and the log.
    pub fn name(self) -> &'static str {
        match self {
            Self::Published => "published",
            Self::TwoState => "two_state",
        }
    }
}

/// The smallest factor `sqrt(1 - (2V / Delta)^2)` of [`ForceModel::TwoState`] (a factor of
/// 20 at most on the difference of the forces at an outer step). On the two-state model of
/// `tests/md_surface_hopping.rs` the factor is 0.063 at 0.1 fs and 0.16 at 0.25 fs.
pub const MIN_UNMIXING: f64 = 0.05;

/// The two-state model of a gap minimum and its hopping direction.
#[derive(Debug, Clone)]
pub struct GapMinimum {
    /// The straight two-state model along the direction.
    pub model: CrossingModel,
    /// The direction as a change of the Cartesian velocities, of mass-weighted length one
    /// (`sum m u^2 = 1`): `M^(-1/2)` times the unit mass-weighted direction.
    pub direction: Col<f64>,
}

/// The model of a gap minimum at the middle of three steps of equal length, from the
/// states at the outer steps `first` and `last` (gradients) and the energies at the middle
/// one, for the inverse masses `inverse_masses` (zero for atoms held in place). `None` when
/// the diabatic forces are equal (no direction).
pub fn gap_minimum(
    first: StatePair<'_>,
    middle: (f64, f64),
    last: StatePair<'_>,
    inverse_masses: &[f64],
    forces: ForceModel,
) -> Option<GapMinimum> {
    let n = inverse_masses.len();
    if [
        first.lower_gradient,
        first.upper_gradient,
        last.lower_gradient,
        last.upper_gradient,
    ]
    .iter()
    .any(|g| g.nrows() != n)
    {
        return None;
    }
    let (low, up) = middle;
    let gaps = [
        (first.upper - first.lower).abs(),
        (up - low).abs(),
        (last.upper - last.lower).abs(),
    ];
    let (coupling, unmixed) = match forces {
        ForceModel::Published => (0.5 * gaps[1], [1.0, 1.0]),
        ForceModel::TwoState => {
            let smallest = parabola_minimum(gaps.map(|g| g * g)).sqrt();
            let factor = |gap: f64| {
                let ratio = if gap > 0.0 { smallest / gap } else { 1.0 };
                (1.0 - ratio * ratio).max(0.0).sqrt().max(MIN_UNMIXING)
            };
            (0.5 * smallest, [factor(gaps[0]), factor(gaps[2])])
        }
    };
    // Mass-weighted diabatic forces (F = -gradient): the mean of eqs. 7-8, and half the
    // difference F1 - F2 = ((F_up - F_low)(t1) / c1 + (F_low - F_up)(t3) / c3) / 2.
    let root: Vec<f64> = inverse_masses.iter().map(|w| w.max(0.0).sqrt()).collect();
    let mean = |i: usize| {
        -0.25
            * (first.upper_gradient[i]
                + first.lower_gradient[i]
                + last.upper_gradient[i]
                + last.lower_gradient[i])
    };
    let half = |i: usize| {
        -0.25
            * ((first.upper_gradient[i] - first.lower_gradient[i]) / unmixed[0]
                + (last.lower_gradient[i] - last.upper_gradient[i]) / unmixed[1])
    };
    let f1 = Col::<f64>::from_fn(n, |i| (mean(i) + half(i)) * root[i]);
    let f2 = Col::<f64>::from_fn(n, |i| (mean(i) - half(i)) * root[i]);
    let difference = Col::<f64>::from_fn(n, |i| f2[i] - f1[i]);
    let length = difference.norm_l2();
    let product: f64 = (0..n).map(|i| f1[i] * f2[i]).sum();
    if !(length.is_finite() && length > 0.0 && product.is_finite()) {
        return None;
    }
    let direction = Col::<f64>::from_fn(n, |i| difference[i] / length * root[i]);
    Some(GapMinimum {
        model: CrossingModel {
            coupling,
            force_difference: length,
            force_product: product,
            crossing_energy: 0.5 * (up + low),
        },
        direction,
    })
}

/// The smallest value of the parabola through `y` at three equally spaced points whose
/// middle one is the lowest, bounded to `[0, y[1]]`.
fn parabola_minimum(y: [f64; 3]) -> f64 {
    let curvature = 0.5 * (y[0] + y[2]) - y[1];
    let slope = 0.5 * (y[2] - y[0]);
    if !(curvature.is_finite() && curvature > 0.0) {
        return y[1].max(0.0);
    }
    (y[1] - slope * slope / (4.0 * curvature)).clamp(0.0, y[1].max(0.0))
}

/// `sum m v u`, the mass-weighted velocity component along `u` (`masses` per component).
pub fn component(v: &Col<f64>, u: &Col<f64>, masses: &[f64]) -> f64 {
    (0..v.nrows()).map(|i| masses[i] * v[i] * u[i]).sum()
}

/// `u` scaled to mass-weighted length one; `None` when it has none.
pub fn normalized(u: &Col<f64>, masses: &[f64]) -> Option<Col<f64>> {
    let length = component(u, u, masses).max(0.0).sqrt();
    (length.is_finite() && length > 0.0).then(|| Col::from_fn(u.nrows(), |i| u[i] / length))
}

/// Change the velocities `v` (for the masses `masses` per component) along the
/// mass-weighted unit direction `u` so that the kinetic energy takes up the change
/// `potential_change` (Hartree, new minus old) of the potential energy, keeping the sign
/// of the component along `u`. Returns the kinetic energy along `u` after the change, or
/// `None` (and `v` unchanged) when there is not enough of it (a frustrated hop).
pub fn adjust_velocities(
    v: &mut Col<f64>,
    u: &Col<f64>,
    masses: &[f64],
    potential_change: f64,
) -> Option<f64> {
    let c = component(v, u, masses);
    let left = c * c - 2.0 * potential_change;
    if !(left.is_finite() && left >= 0.0) {
        return None;
    }
    let new = left.sqrt().copysign(if c == 0.0 { 1.0 } else { c });
    for i in 0..v.nrows() {
        v[i] += (new - c) * u[i];
    }
    Some(0.5 * new * new)
}

/// Reverse the velocity component along the mass-weighted unit direction `u` (the
/// treatment of a frustrated hop that turns the motion along the direction back).
pub fn reverse_along(v: &mut Col<f64>, u: &Col<f64>, masses: &[f64]) {
    let c = component(v, u, masses);
    for i in 0..v.nrows() {
        v[i] -= 2.0 * c * u[i];
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn kinetic(v: &Col<f64>, masses: &[f64]) -> f64 {
        0.5 * component(v, v, masses)
    }

    #[test]
    fn the_velocities_take_up_the_change_of_the_potential_energy() {
        let masses = [1800.0, 1800.0, 1800.0, 22000.0, 22000.0, 22000.0];
        let v = Col::<f64>::from_fn(6, |i| 1e-3 * (i as f64 - 2.5));
        let raw = Col::<f64>::from_fn(6, |i| [0.3, -0.1, 0.2, -0.05, 0.4, 0.1][i]);
        let u = normalized(&raw, &masses).unwrap();
        assert!((component(&u, &u, &masses) - 1.0).abs() < 1e-14);
        let along = component(&v, &u, &masses);
        for change in [-0.01, 0.0, 0.2 * along * along] {
            let mut w = v.clone();
            let left = adjust_velocities(&mut w, &u, &masses, change).unwrap();
            assert!((kinetic(&w, &masses) - (kinetic(&v, &masses) - change)).abs() < 1e-15);
            assert!((left - (0.5 * along * along - change)).abs() < 1e-15);
            // The sign along u is kept, the perpendicular part untouched.
            let after = component(&w, &u, &masses);
            assert!(after * along > 0.0);
            let perpendicular = |x: &Col<f64>| {
                let c = component(x, &u, &masses);
                Col::<f64>::from_fn(6, |i| x[i] - c * u[i])
            };
            let (p0, p1) = (perpendicular(&v), perpendicular(&w));
            assert!((0..6).all(|i| (p0[i] - p1[i]).abs() < 1e-15));
        }
        // Not enough kinetic energy along u: frustrated, nothing changes.
        let mut w = v.clone();
        assert!(adjust_velocities(&mut w, &u, &masses, 0.51 * along * along).is_none());
        assert_eq!(w, v);
        reverse_along(&mut w, &u, &masses);
        assert!((component(&w, &u, &masses) + along).abs() < 1e-15);
        assert!((kinetic(&w, &masses) - kinetic(&v, &masses)).abs() < 1e-15);
    }

    #[test]
    fn three_steps_through_a_straight_crossing_give_its_model() {
        // One coordinate x (mass 1), V1 = -F1 x, V2 = -F2 x, coupling V, passed at the
        // velocity 1e-3 with the smallest gap at the middle step.
        let (f1, f2, v) = (0.03, 0.01, 0.002);
        let adiabats = |x: f64| {
            let mean = -0.5 * (f1 + f2) * x;
            let half = 0.5 * (f1 - f2) * x;
            let root = (half * half + v * v).sqrt();
            let slope = |s: f64| -0.5 * (f1 + f2) + s * 0.25 * (f1 - f2).powi(2) * x / root;
            (
                mean - root,
                mean + root,
                Col::<f64>::from_fn(1, |_| slope(-1.0)),
                Col::<f64>::from_fn(1, |_| slope(1.0)),
            )
        };
        let x = [-0.4, 0.0, 0.4];
        let at: Vec<_> = x.iter().map(|&x| adiabats(x)).collect();
        let pair = |k: usize| StatePair {
            lower: at[k].0,
            upper: at[k].1,
            lower_gradient: &at[k].2,
            upper_gradient: &at[k].3,
        };
        let model = |forces| gap_minimum(pair(0), (at[1].0, at[1].1), pair(2), &[1.0], forces);
        // Two couplings from the crossing the published forces are short by 11 %.
        let published = model(ForceModel::Published).unwrap();
        assert!((published.model.coupling - v).abs() < 1e-15);
        let short = published.model.force_difference / (f1 - f2);
        assert!(
            (short - (1.0 - 1.0 / 5.0_f64).sqrt()).abs() < 1e-12,
            "{short}"
        );
        // The two-state form gives the straight model.
        let m = model(ForceModel::TwoState).unwrap();
        assert!((m.model.coupling - v).abs() < 1e-15);
        assert!(m.model.crossing_energy.abs() < 1e-15);
        assert!((m.model.force_difference / (f1 - f2) - 1.0).abs() < 1e-12);
        assert!((m.model.force_product / (f1 * f2) - 1.0).abs() < 1e-12);
        assert_eq!(m.direction.nrows(), 1);
        assert!((m.direction[0].abs() - 1.0).abs() < 1e-15);
        // The same pair of states seen the other way (the gap passed backwards) gives the
        // same model.
        let back = gap_minimum(
            pair(2),
            (at[1].0, at[1].1),
            pair(0),
            &[1.0],
            ForceModel::TwoState,
        )
        .unwrap();
        assert!((back.model.force_difference - m.model.force_difference).abs() < 1e-15);
        assert!((back.model.force_product - m.model.force_product).abs() < 1e-15);
        // Steps off the smallest gap: the parabola of the squared gap finds the coupling.
        let x = [-0.25, 0.15, 0.55];
        let at: Vec<_> = x.iter().map(|&x| adiabats(x)).collect();
        let pair = |k: usize| StatePair {
            lower: at[k].0,
            upper: at[k].1,
            lower_gradient: &at[k].2,
            upper_gradient: &at[k].3,
        };
        let off = gap_minimum(
            pair(0),
            (at[1].0, at[1].1),
            pair(2),
            &[1.0],
            ForceModel::TwoState,
        )
        .unwrap();
        assert!(
            (off.model.coupling - v).abs() < 1e-12,
            "{}",
            off.model.coupling
        );
        assert!((off.model.force_difference / (f1 - f2) - 1.0).abs() < 1e-9);
    }
}
