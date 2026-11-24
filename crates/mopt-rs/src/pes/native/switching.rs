//! Switching function that takes an interaction to zero at its cutoff without a step in the
//! energy or in any of its derivatives.
//!
//! With `u = (r_off - r) / (r_off - r_on)` between `r_on` and `r_off`,
//!
//! `S(r) = f(u) / (f(u) + f(1 - u))`, `f(x) = exp(-1 / x)`,
//!
//! that is `S = 1 / (1 + exp(1 / u - 1 / (1 - u)))`; 1 below `r_on`, 0 above `r_off`. `f`
//! and all its derivatives vanish at zero, so `S` joins both constants with every
//! derivative continuous (infinitely smooth): a potential multiplied by it is as smooth as
//! the potential itself for every cutoff, which a neighbor search with a cutoff needs.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};
use crate::math::ad::Scalar;

/// Share of the cutoff over which interactions are switched off when no width is given.
pub const DEFAULT_SWITCH_SHARE: f64 = 0.1;

/// Beyond this size of the exponent the logistic function is 0 or 1 to all digits: the
/// switching function takes these values a little inside the ends of its range (within
/// 1/700 of it), where `1 / u` would otherwise overflow.
const SATURATED: f64 = 700.0;

/// `S(r)`: 1 up to `on`, 0 from `off`, infinitely smooth in between.
pub fn switch_off<T: Scalar>(r: T, on: f64, off: f64) -> T {
    let value = r.val();
    if value <= on {
        T::one()
    } else if value >= off {
        T::zero()
    } else {
        let u = (T::from_f64(off) - r) * T::from_f64(1.0 / (off - on));
        let exponent = T::one() / u - T::one() / (T::one() - u);
        // The logistic function of -exponent, written without overflow on either side.
        if exponent.val() > SATURATED {
            T::zero()
        } else if exponent.val() < -SATURATED {
            T::one()
        } else if exponent.val() >= 0.0 {
            let e = (-exponent).exp();
            e / (T::one() + e)
        } else {
            T::one() / (T::one() + exponent.exp())
        }
    }
}

/// Where the switching begins for a `cutoff` and the `width` of the range (the default
/// share of the cutoff without one); `None` for a width of zero (no switching).
pub fn switch_start(cutoff: f64, width: Option<f64>) -> MoptResult<Option<f64>> {
    let width = width.unwrap_or(DEFAULT_SWITCH_SHARE * cutoff);
    if !(width.is_finite() && width >= 0.0 && width < cutoff) {
        return Err(MoptError::config(format!(
            "the width of the switching range must lie between 0 and the cutoff {cutoff}, \
             got {width}"
        )));
    }
    Ok((width > 0.0).then_some(cutoff - width))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::math::ad::HyperDual;

    fn derivatives(r: f64) -> (f64, f64, f64) {
        let y = switch_off(
            HyperDual {
                f0: r,
                f1: 1.0,
                f2: 1.0,
                f12: 0.0,
            },
            3.0,
            4.0,
        );
        (y.f0, y.f1, y.f12)
    }

    #[test]
    fn value_slope_and_curvature_are_continuous_at_both_ends() {
        assert_eq!(derivatives(2.9), (1.0, 0.0, 0.0));
        assert_eq!(derivatives(4.1), (0.0, 0.0, 0.0));
        let (value, slope, curvature) = derivatives(3.0 + 1e-7);
        assert!((value - 1.0).abs() < 1e-15 && slope.abs() < 1e-12 && curvature.abs() < 1e-5);
        let (value, slope, curvature) = derivatives(4.0 - 1e-7);
        assert!(value.abs() < 1e-15 && slope.abs() < 1e-12 && curvature.abs() < 1e-5);
        // Half way: 1/2, with the steepest slope -2 per width.
        let (value, slope, curvature) = derivatives(3.5);
        assert!((value - 0.5).abs() < 1e-15 && (slope + 2.0).abs() < 1e-14);
        assert!(curvature.abs() < 1e-13);
    }

    #[test]
    fn every_derivative_dies_out_towards_the_ends() {
        // The third derivative (differences of the second) shrinks faster than a high power
        // of the distance to the ends, where a polynomial switch has a jump in some
        // derivative.
        for end in [3.0, 4.0] {
            let inward = if end == 3.0 { 1.0 } else { -1.0 };
            let mut previous = f64::INFINITY;
            for distance in [0.08, 0.04, 0.02, 0.01] {
                let h = 1e-4;
                let at = |x: f64| derivatives(end + inward * distance + x).2;
                let third = (at(h) - at(-h)) / (2.0 * h);
                let scaled = third.abs() / distance.powi(6);
                assert!(scaled < previous, "{end} {distance}: {scaled} {previous}");
                previous = scaled;
            }
        }
        // Finite everywhere, also next to the ends and in the saturated parts.
        for k in 0..=4000 {
            let r = 2.9995 + 1.001 * f64::from(k) / 4000.0;
            let (value, slope, curvature) = derivatives(r);
            assert!(
                value.is_finite() && slope.is_finite() && curvature.is_finite(),
                "{r}"
            );
            assert!((0.0..=1.0).contains(&value) && slope <= 0.0, "{r}");
        }
        for r in [3.0 + 1e-300, 3.0 + 1e-17, 4.0 - 1e-15, 4.0 - 1e-9] {
            let (value, slope, curvature) = derivatives(r);
            assert!(value.is_finite() && slope == 0.0 && curvature == 0.0, "{r}");
        }
    }

    #[test]
    fn the_start_of_the_range() {
        assert_eq!(switch_start(5.0, None).unwrap(), Some(4.5));
        assert_eq!(switch_start(5.0, Some(1.0)).unwrap(), Some(4.0));
        assert_eq!(switch_start(5.0, Some(0.0)).unwrap(), None);
        assert!(switch_start(5.0, Some(5.0)).is_err());
        assert!(switch_start(5.0, Some(-0.1)).is_err());
    }
}
