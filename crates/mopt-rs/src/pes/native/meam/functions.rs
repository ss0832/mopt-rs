//! The one-dimensional functions of the modified embedded-atom method, written once over
//! [`Scalar`]: cutoff function, `G(Gamma)`, embedding energy, universal equation of state,
//! and the screened nuclear repulsion at short distances.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::super::switching::switch_off;
use crate::math::ad::Scalar;

/// Below `-GAMMA_SMOOTHING / (GAMMA_SMOOTHING + 1)` the square root of `1 + Gamma` is
/// continued by a power of `1 / Gamma` (value and slope continuous), so that it stays real
/// for negative weights `t`.
const GAMMA_SMOOTHING: f64 = 99.0;

/// `f_c(x) = [1 - (1 - x)^4]^2` between 0 and 1, 0 below and 1 above: the cutoff function
/// of both the distance and the screening. Value and slope are continuous; the second
/// derivative jumps from 0 to 32 at `x = 0`.
pub(super) fn smooth_step<T: Scalar>(x: T) -> T {
    let value = x.val();
    if value >= 1.0 {
        T::one()
    } else if value <= 0.0 {
        T::zero()
    } else {
        let rest = (T::one() - x).powi(4);
        (T::one() - rest).powi(2)
    }
}

/// An infinitely smooth step in place of [`smooth_step`]: 0 up to `x = 0`, 1 from `x = 1`,
/// 1/2 half way, every derivative continuous.
pub(super) fn infinitely_smooth_step<T: Scalar>(x: T) -> T {
    switch_off(T::one() - x, 0.0, 1.0)
}

/// `G(Gamma)`, the factor that the angular partial densities put on `rho^(0)`.
pub(super) fn density_factor<T: Scalar>(gamma: T, ibar: i32) -> T {
    match ibar {
        1 => (gamma * T::from_f64(0.5)).exp(),
        3 => {
            // 2 / (1 + exp(-Gamma)), written so that the exponential cannot overflow.
            if gamma.val() >= 0.0 {
                T::from_f64(2.0) / (T::one() + (-gamma).exp())
            } else {
                let e = gamma.exp();
                T::from_f64(2.0) * e / (T::one() + e)
            }
        }
        _ => {
            let switch = -GAMMA_SMOOTHING / (GAMMA_SMOOTHING + 1.0);
            if gamma.val() < switch {
                let ratio = T::from_f64(switch) / gamma;
                (ratio.powf(GAMMA_SMOOTHING) * T::from_f64(1.0 / (GAMMA_SMOOTHING + 1.0))).sqrt()
            } else {
                (T::one() + gamma).sqrt()
            }
        }
    }
}

/// Embedding energy `A E_c x ln(x)` at the relative background density `x`; zero where the
/// density vanishes (its limit) or is negative.
pub(super) fn embedding<T: Scalar>(scale: f64, x: T) -> T {
    if x.val() > 0.0 {
        x * x.ln() * T::from_f64(scale)
    } else {
        T::zero()
    }
}

/// Universal equation of state of Rose, Smith, Guinea, and Ferrante: the energy per atom
/// of the reference structure at the nearest-neighbour distance `r`,
/// `-E_c (1 + a) exp(-a)` with `a = alpha (r / r_e - 1)`.
pub(super) fn rose<T: Scalar>(ec: f64, re: f64, alpha: f64, r: T) -> T {
    let a = (r / T::from_f64(re) - T::one()) * T::from_f64(alpha);
    -(T::one() + a) * (-a).exp() * T::from_f64(ec)
}

/// Universal screened repulsion of two nuclei (Ziegler, Biersack, Littmark), in eV with
/// `r` in Angstrom.
pub(super) fn zbl<T: Scalar>(z1: f64, z2: f64, r: T) -> T {
    const C: [f64; 4] = [0.028171, 0.28022, 0.50986, 0.18175];
    const D: [f64; 4] = [0.20162, 0.40290, 0.94229, 3.1998];
    /// `e^2 / (4 pi eps_0)` in eV Angstrom.
    const COULOMB: f64 = 14.3997;
    let length = 0.4685 / (z1.powf(0.23) + z2.powf(0.23));
    let x = r / T::from_f64(length);
    let mut screening = T::zero();
    for (c, d) in C.iter().zip(D) {
        screening += (x * T::from_f64(-d)).exp() * T::from_f64(*c);
    }
    screening * T::from_f64(z1 * z2 * COULOMB) / r
}

/// Weight of the equation of state against the nuclear repulsion at the reduced distance
/// `a = alpha (r / r_e - 1)`: 1 from `a = -1` on, 0 up to `a = -3`.
pub(super) fn short_range_weight<T: Scalar>(a: T) -> T {
    smooth_step(T::one() + (a + T::one()) * T::from_f64(0.5))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::math::ad::HyperDual;

    fn derivatives(f: impl Fn(HyperDual) -> HyperDual, x: f64) -> (f64, f64, f64) {
        let y = f(HyperDual {
            f0: x,
            f1: 1.0,
            f2: 1.0,
            f12: 0.0,
        });
        (y.f0, y.f1, y.f12)
    }

    #[test]
    fn the_cutoff_function_joins_its_ends_with_continuous_slope() {
        let step = |x: f64| -> f64 { smooth_step(x) };
        assert_eq!(step(-0.1), 0.0);
        assert_eq!(step(1.2), 1.0);
        assert!((step(0.5) - (1.0 - 0.0625_f64).powi(2)).abs() < 1e-15);
        let (low, slope_low, _) = derivatives(smooth_step, 1e-9);
        let (high, slope_high, _) = derivatives(smooth_step, 1.0 - 1e-9);
        assert!(low < 1e-15 && slope_low < 1e-7);
        assert!((high - 1.0).abs() < 1e-15 && slope_high.abs() < 1e-15);
        // Its second derivative jumps to 32 at zero; that of the smooth step does not.
        assert!((derivatives(smooth_step, 1e-9).2 - 32.0).abs() < 1e-6);
        let smooth = |x: f64| derivatives(infinitely_smooth_step, x);
        assert_eq!(smooth(-0.1), (0.0, 0.0, 0.0));
        assert_eq!(smooth(1e-9), (0.0, 0.0, 0.0));
        assert_eq!(smooth(1.0 - 1e-9), (1.0, 0.0, 0.0));
        assert_eq!(smooth(1.2), (1.0, 0.0, 0.0));
        let (half, slope, curvature) = smooth(0.5);
        assert!(
            (half - 0.5).abs() < 1e-15 && (slope - 2.0).abs() < 1e-14 && curvature.abs() < 1e-13
        );
    }

    #[test]
    fn the_density_factor_is_continued_smoothly_below_minus_one() {
        let switch = -GAMMA_SMOOTHING / (GAMMA_SMOOTHING + 1.0);
        let below = derivatives(|g| density_factor(g, 0), switch - 1e-9);
        let above = derivatives(|g| density_factor(g, 0), switch + 1e-9);
        assert!((below.0 - above.0).abs() < 1e-7 && (below.1 - above.1).abs() < 1e-5);
        let factor = |gamma: f64, ibar: i32| -> f64 { density_factor(gamma, ibar) };
        // It stays real where 1 + Gamma is negative.
        assert!(factor(-3.0, 0).is_finite() && factor(-3.0, 0) > 0.0);
        assert!((factor(0.21, 4) - 1.1).abs() < 1e-14);
        assert!((factor(0.4, 1) - 0.2_f64.exp()).abs() < 1e-14);
        for gamma in [-800.0_f64, -0.3, 0.0, 0.7, 800.0] {
            let expected = 2.0 / (1.0 + f64::exp(-gamma));
            let found = factor(gamma, 3);
            assert!(found.is_finite() && (found - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn the_equation_of_state_has_its_minimum_at_the_reference() {
        let (energy, slope, curvature) = derivatives(|r| rose(3.54, 2.56, 5.1, r), 2.56);
        assert!((energy + 3.54).abs() < 1e-14 && slope.abs() < 1e-14);
        // E'' = E_c alpha^2 / r_e^2.
        assert!((curvature - 3.54 * 5.1 * 5.1 / (2.56 * 2.56)).abs() < 1e-12);
        assert_eq!(embedding(1.0, 0.0_f64), 0.0);
        assert_eq!(embedding(1.0, 1.0_f64), 0.0);
    }

    #[test]
    fn the_nuclear_repulsion_tends_to_the_bare_coulomb_law() {
        // The four coefficients add up to one: at short distance, Z1 Z2 e^2 / r.
        let r = 1e-6;
        let bare = 29.0 * 29.0 * 14.3997 / r;
        let found: f64 = zbl(29.0, 29.0, r);
        assert!((found / bare - 1.0).abs() < 1e-4);
        assert_eq!(short_range_weight(-3.0_f64), 0.0);
        assert_eq!(short_range_weight(-1.0_f64), 1.0);
    }
}
