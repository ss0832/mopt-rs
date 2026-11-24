//! `Scalar` abstraction over `f64`, `Dual`, and `HyperDual`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// Generic real/hyper-dual scalar abstraction supporting differentiable elementary functions.
pub trait Scalar:
    Copy
    + Clone
    + std::fmt::Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
{
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Lift a constant.
    fn from_f64(v: f64) -> Self;
    /// Real (value) part.
    fn val(self) -> f64;

    /// Absolute value.
    fn abs(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Cube root.
    fn cbrt(self) -> Self;
    /// Exponential.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Real power `x^p`.
    fn powf(self, p: f64) -> Self;
    /// Integer power `x^n`.
    fn powi(self, n: i32) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Tangent.
    fn tan(self) -> Self;
    /// Arc sine.
    fn asin(self) -> Self;
    /// Arc cosine.
    fn acos(self) -> Self;
    /// Two-argument arc tangent `atan2(self, x)`.
    fn atan2(self, x: Self) -> Self;
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;

    /// Euclidean norm of a 3-vector.
    #[inline]
    fn norm3(v: [Self; 3]) -> Self {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// Dot product of 3-vectors.
    #[inline]
    fn dot3(a: [Self; 3], b: [Self; 3]) -> Self {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Cross product of 3-vectors.
    #[inline]
    fn cross3(a: [Self; 3], b: [Self; 3]) -> [Self; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
}

/// Angle (rad, in `[0, pi]`) between the 3-vectors `a` and `b`, as `atan2(|a x b|, a . b)`.
///
/// The arc cosine of the normalized dot product loses all accuracy near 0 and pi (its argument
/// rounds to `+-1`, where the derivative `-1 / sqrt(1 - u^2)` is infinite); this form keeps
/// the value and its derivatives accurate up to the collinear geometry itself, where the angle
/// is not differentiable (a cone) and the automatic derivatives of the norm are zero.
///
/// MultiOptPy: no counterpart (new in mopt-rs); it replaces `torch.acos` of the normalized
/// dot product in `Potential/*.py`.
pub fn angle_between<T: Scalar>(a: [T; 3], b: [T; 3]) -> T {
    T::norm3(T::cross3(a, b)).atan2(T::dot3(a, b))
}

impl Scalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn val(self) -> f64 {
        self
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn cbrt(self) -> Self {
        libm::cbrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        libm::pow(self, p)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        libm::pow(self, n as f64)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        libm::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        libm::asin(self.clamp(-1.0, 1.0))
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acos(self.clamp(-1.0, 1.0))
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        libm::atan2(self, x)
    }
    #[inline]
    fn sinh(self) -> Self {
        libm::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        libm::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanh(self)
    }
}
