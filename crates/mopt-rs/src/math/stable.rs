//! Numerically stable forms of functions that overflow when written literally.
//!
//! All functions are generic over [`Scalar`], so automatic derivatives stay finite as well: the
//! branches select algebraically identical expressions by the value of the argument.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::math::ad::Scalar;

/// Largest argument of `exp` that does not overflow (`ln(f64::MAX)` rounded down).
pub const EXP_MAX: f64 = 709.0;

/// Logistic function `1 / (1 + exp(-x))` without overflow for any `x`.
pub fn logistic<T: Scalar>(x: T) -> T {
    if x.val() >= 0.0 {
        T::one() / (T::one() + (-x).exp())
    } else {
        let e = x.exp();
        e / (T::one() + e)
    }
}

/// Softplus `ln(1 + exp(x))` without overflow: `max(x, 0) + ln(1 + exp(-|x|))`.
pub fn softplus<T: Scalar>(x: T) -> T {
    if x.val() > 0.0 {
        x + (T::one() + (-x).exp()).ln()
    } else {
        (T::one() + x.exp()).ln()
    }
}

/// `exp(x - shift)` with the exponent clamped below so that it underflows to zero instead of
/// producing `0 * inf` in derivatives.
pub fn exp_shifted<T: Scalar>(x: T, shift: f64) -> T {
    let y = x - T::from_f64(shift);
    if y.val() < -EXP_MAX {
        T::zero()
    } else {
        y.exp()
    }
}

/// Normalized weights `exp(x_k) / sum_l exp(x_l)` (softmax over `x`), shifted by the largest
/// exponent so that the normalization can neither overflow nor underflow to zero.
pub fn normalized_exp<T: Scalar>(x: &[T]) -> Vec<T> {
    let m = x.iter().map(|v| v.val()).fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return x.iter().map(|_| T::zero()).collect();
    }
    let e: Vec<T> = x.iter().map(|&v| exp_shifted(v, m)).collect();
    let norm = e.iter().fold(T::zero(), |a, &b| a + b);
    e.into_iter().map(|v| v / norm).collect()
}

/// `ln(sum_k exp(x_k))` without overflow or underflow, shifted by the largest term; `-inf` for
/// an empty slice or when every term is `-inf`.
pub fn log_sum_exp(x: &[f64]) -> f64 {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

/// `ln(exp(a) + exp(b))` without overflow or underflow.
pub fn log_add_exp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if !hi.is_finite() || lo == f64::NEG_INFINITY {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// `ln(exp(a) - exp(b))` for `a >= b` without cancellation in the logarithm: `a + ln(1 -
/// exp(b - a))`, with `ln_1p(-exp(-d))` or `ln(-expm1(-d))` for `d = a - b` (whichever is
/// accurate there). `-inf` for `a == b`, NaN for `a < b`.
pub fn log_diff_exp(a: f64, b: f64) -> f64 {
    if b == f64::NEG_INFINITY {
        return a;
    }
    let d = a - b;
    if d.is_nan() || d < 0.0 {
        return f64::NAN;
    }
    if d == 0.0 {
        return f64::NEG_INFINITY;
    }
    if d > std::f64::consts::LN_2 {
        a + (-(-d).exp()).ln_1p()
    } else {
        a + (-(-d).exp_m1()).ln()
    }
}

/// `ln(sinh(x))` for `x > 0`: `x - ln 2 + ln(1 - exp(-2 x))`, accurate for small and large
/// `x` (NaN for `x <= 0`, where the logarithm is not real).
pub fn ln_sinh(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    x - std::f64::consts::LN_2 + (-(-2.0 * x).exp_m1()).ln()
}

/// `ln(cosh(x))`: `|x| - ln 2 + ln(1 + exp(-2 |x|))`, without overflow for any `x`.
pub fn ln_cosh(x: f64) -> f64 {
    let a = x.abs();
    a - std::f64::consts::LN_2 + (-2.0 * a).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ad::Dual;

    #[test]
    fn the_logarithmic_sums_are_exact_where_the_plain_forms_fail() {
        // Plain values where the plain forms are fine.
        let x = [0.3_f64, -1.2, 2.5];
        let plain = x.iter().map(|v| v.exp()).sum::<f64>().ln();
        assert!((log_sum_exp(&x) - plain).abs() < 1e-15);
        assert!((log_add_exp(0.3, -1.2) - (0.3_f64.exp() + (-1.2_f64).exp()).ln()).abs() < 1e-15);
        assert!((log_diff_exp(2.0, 1.0) - (2.0_f64.exp() - 1.0_f64.exp()).ln()).abs() < 1e-14);
        assert!((ln_sinh(0.7) - 0.7_f64.sinh().ln()).abs() < 1e-15);
        assert!((ln_cosh(-0.7) - 0.7_f64.cosh().ln()).abs() < 1e-15);
        // Arguments far beyond the range of exp.
        assert!((log_sum_exp(&[1000.0, 1000.0]) - (1000.0 + 2.0_f64.ln())).abs() < 1e-12);
        assert!(
            (log_sum_exp(&[-1e4, -1e4 - 1.0]) - (-1e4 + (-1.0_f64).exp().ln_1p())).abs() < 1e-9
        );
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY; 2]), f64::NEG_INFINITY);
        assert!((log_add_exp(-800.0, f64::NEG_INFINITY) + 800.0).abs() < 1e-12);
        assert!((ln_sinh(800.0) - (800.0 - 2.0_f64.ln())).abs() < 1e-12);
        assert!((ln_cosh(-800.0) - (800.0 - 2.0_f64.ln())).abs() < 1e-12);
        // Small differences and small arguments keep their relative accuracy.
        assert!((log_diff_exp(1e-12, 0.0) - (1e-12_f64).ln()).abs() < 1e-9);
        assert!((ln_sinh(1e-200) - (1e-200_f64).ln()).abs() < 1e-12);
        assert!((log_diff_exp(-700.0, -760.0) + 700.0).abs() < 1e-15);
        assert_eq!(log_diff_exp(3.0, 3.0), f64::NEG_INFINITY);
        assert!(log_diff_exp(1.0, 2.0).is_nan() && ln_sinh(0.0).is_nan());
    }

    #[test]
    fn extreme_arguments_stay_finite() {
        for x in [-1e5, -800.0, -30.0, 0.0, 30.0, 800.0, 1e5] {
            let d = Dual { re: x, eps: 1.0 };
            let (l, s) = (logistic(d), softplus(d));
            assert!(l.re.is_finite() && l.eps.is_finite(), "logistic {x}");
            assert!(s.re.is_finite() && s.eps.is_finite(), "softplus {x}");
            assert!((0.0..=1.0).contains(&l.re));
        }
        assert!((softplus(1000.0_f64) - 1000.0).abs() < 1e-12);
        assert!((logistic(0.3_f64) - 1.0 / (1.0 + (-0.3_f64).exp())).abs() < 1e-15);
        let w = normalized_exp(&[-1e6_f64, -1e6 - 1.0, -2e6]);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15 && w[2] == 0.0);
        let w = normalized_exp(&[0.1_f64, 0.3]);
        let want = 0.1_f64.exp() / (0.1_f64.exp() + 0.3_f64.exp());
        assert!((w[0] - want).abs() < 1e-15);
    }
}
