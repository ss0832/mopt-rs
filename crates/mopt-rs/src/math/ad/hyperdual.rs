//! Hyper-dual numbers for exact second derivatives.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy differentiates with torch).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use super::scalar::Scalar;

/// Second-order Fike-Alonso hyper-dual number `f + f_1 e_1 + f_2 e_2 + f_12 e_1 e_2`
/// (`e_1^2 = e_2^2 = 0`, `e_1 e_2 != 0`), yielding exact 1st and 2nd derivatives free of
/// subtraction cancellation error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperDual {
    /// Value `f`.
    pub f0: f64,
    /// First derivative along `e_1`.
    pub f1: f64,
    /// First derivative along `e_2`.
    pub f2: f64,
    /// Mixed second derivative `e_1 e_2`.
    pub f12: f64,
}

impl HyperDual {
    /// Constant (all derivative parts zero).
    #[inline]
    pub const fn constant(f0: f64) -> Self {
        Self {
            f0,
            f1: 0.0,
            f2: 0.0,
            f12: 0.0,
        }
    }

    #[inline]
    fn chain(self, val: f64, d1: f64, d2: f64) -> Self {
        Self {
            f0: val,
            f1: d1 * self.f1,
            f2: d1 * self.f2,
            f12: d1 * self.f12 + d2 * self.f1 * self.f2,
        }
    }
}

impl PartialOrd for HyperDual {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.f0.partial_cmp(&other.f0)
    }
}

impl Add for HyperDual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            f0: self.f0 + rhs.f0,
            f1: self.f1 + rhs.f1,
            f2: self.f2 + rhs.f2,
            f12: self.f12 + rhs.f12,
        }
    }
}

impl AddAssign for HyperDual {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.f0 += rhs.f0;
        self.f1 += rhs.f1;
        self.f2 += rhs.f2;
        self.f12 += rhs.f12;
    }
}

impl Sub for HyperDual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            f0: self.f0 - rhs.f0,
            f1: self.f1 - rhs.f1,
            f2: self.f2 - rhs.f2,
            f12: self.f12 - rhs.f12,
        }
    }
}

impl SubAssign for HyperDual {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.f0 -= rhs.f0;
        self.f1 -= rhs.f1;
        self.f2 -= rhs.f2;
        self.f12 -= rhs.f12;
    }
}

impl Mul for HyperDual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            f0: self.f0 * rhs.f0,
            f1: self.f0 * rhs.f1 + self.f1 * rhs.f0,
            f2: self.f0 * rhs.f2 + self.f2 * rhs.f0,
            f12: self.f0 * rhs.f12 + self.f1 * rhs.f2 + self.f2 * rhs.f1 + self.f12 * rhs.f0,
        }
    }
}

impl Div for HyperDual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = 1.0 / rhs.f0;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let rhs_recip = Self {
            f0: inv,
            f1: -rhs.f1 * inv2,
            f2: -rhs.f2 * inv2,
            f12: -rhs.f12 * inv2 + 2.0 * rhs.f1 * rhs.f2 * inv3,
        };
        self * rhs_recip
    }
}

impl Neg for HyperDual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            f0: -self.f0,
            f1: -self.f1,
            f2: -self.f2,
            f12: -self.f12,
        }
    }
}

impl Scalar for HyperDual {
    #[inline]
    fn zero() -> Self {
        Self::constant(0.0)
    }
    #[inline]
    fn one() -> Self {
        Self::constant(1.0)
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }
    #[inline]
    fn val(self) -> f64 {
        self.f0
    }
    #[inline]
    fn abs(self) -> Self {
        let sgn = if self.f0 >= 0.0 { 1.0 } else { -1.0 };
        self.chain(libm::fabs(self.f0), sgn, 0.0)
    }
    #[inline]
    fn sqrt(self) -> Self {
        let r = libm::sqrt(self.f0.max(0.0));
        if r > 1e-30 {
            let d1 = 0.5 / r;
            let d2 = -0.25 / (self.f0 * r);
            self.chain(r, d1, d2)
        } else {
            Self::zero()
        }
    }
    #[inline]
    fn cbrt(self) -> Self {
        let r = libm::cbrt(self.f0);
        if r.abs() > 1e-30 {
            let d1 = 1.0 / (3.0 * r * r);
            let d2 = -2.0 / (9.0 * self.f0 * r * r);
            self.chain(r, d1, d2)
        } else {
            Self::zero()
        }
    }
    #[inline]
    fn exp(self) -> Self {
        let e = libm::exp(self.f0);
        self.chain(e, e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        let inv = 1.0 / self.f0;
        self.chain(libm::log(self.f0), inv, -inv * inv)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        if p == 0.0 {
            return Self::one();
        }
        let v = libm::pow(self.f0, p);
        let d1 = p * libm::pow(self.f0, p - 1.0);
        let d2 = p * (p - 1.0) * libm::pow(self.f0, p - 2.0);
        self.chain(v, d1, d2)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        match n {
            0 => Self::one(),
            1 => self,
            2 => self * self,
            3 => self * self * self,
            _ => self.powf(n as f64),
        }
    }
    #[inline]
    fn sin(self) -> Self {
        let s = libm::sin(self.f0);
        let c = libm::cos(self.f0);
        self.chain(s, c, -s)
    }
    #[inline]
    fn cos(self) -> Self {
        let s = libm::sin(self.f0);
        let c = libm::cos(self.f0);
        self.chain(c, -s, -c)
    }
    #[inline]
    fn tan(self) -> Self {
        let t = libm::tan(self.f0);
        let sec2 = 1.0 + t * t;
        self.chain(t, sec2, 2.0 * t * sec2)
    }
    #[inline]
    fn asin(self) -> Self {
        let x = self.f0.clamp(-1.0 + 1e-14, 1.0 - 1e-14);
        let one_minus_x2 = 1.0 - x * x;
        let d1 = 1.0 / libm::sqrt(one_minus_x2);
        let d2 = x / (one_minus_x2 * libm::sqrt(one_minus_x2));
        self.chain(libm::asin(self.f0.clamp(-1.0, 1.0)), d1, d2)
    }
    #[inline]
    fn acos(self) -> Self {
        let x = self.f0.clamp(-1.0 + 1e-14, 1.0 - 1e-14);
        let one_minus_x2 = 1.0 - x * x;
        let d1 = -1.0 / libm::sqrt(one_minus_x2);
        let d2 = -x / (one_minus_x2 * libm::sqrt(one_minus_x2));
        self.chain(libm::acos(self.f0.clamp(-1.0, 1.0)), d1, d2)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        let r2 = self * self + x * x;
        if r2.f0 <= 1e-30 {
            return Self::zero();
        }
        // d(atan2(y, x)) = (x dy - y dx) / (x^2 + y^2)
        let f0 = libm::atan2(self.f0, x.f0);
        let num_1 = x.f0 * self.f1 - self.f0 * x.f1;
        let num_2 = x.f0 * self.f2 - self.f0 * x.f2;
        let d_r2_2 = 2.0 * (self.f0 * self.f2 + x.f0 * x.f2);
        let inv = 1.0 / r2.f0;
        let f1 = num_1 * inv;
        let f2 = num_2 * inv;
        let d_num1_2 = (x.f2 * self.f1 + x.f0 * self.f12) - (self.f2 * x.f1 + self.f0 * x.f12);
        let f12 = (d_num1_2 * r2.f0 - num_1 * d_r2_2) * inv * inv;
        Self { f0, f1, f2, f12 }
    }
    #[inline]
    fn sinh(self) -> Self {
        let sh = libm::sinh(self.f0);
        let ch = libm::cosh(self.f0);
        self.chain(sh, ch, sh)
    }
    #[inline]
    fn cosh(self) -> Self {
        let sh = libm::sinh(self.f0);
        let ch = libm::cosh(self.f0);
        self.chain(ch, sh, ch)
    }
    #[inline]
    fn tanh(self) -> Self {
        let th = libm::tanh(self.f0);
        let sech2 = 1.0 - th * th;
        self.chain(th, sech2, -2.0 * th * sech2)
    }
}
