//! First-order forward-mode dual numbers.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy differentiates with torch).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use super::scalar::Scalar;

/// First-order forward-mode dual number `f + df * eps` (`eps^2 = 0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Real part `f`.
    pub re: f64,
    /// Derivative part `df`.
    pub eps: f64,
}

impl Dual {
    /// Constant (zero derivative).
    #[inline]
    pub const fn constant(re: f64) -> Self {
        Self { re, eps: 0.0 }
    }

    /// Independent variable (unit derivative).
    #[inline]
    pub const fn variable(re: f64) -> Self {
        Self { re, eps: 1.0 }
    }

    #[inline]
    fn chain(self, f_val: f64, df: f64) -> Self {
        Self {
            re: f_val,
            eps: self.eps * df,
        }
    }
}

impl PartialOrd for Dual {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            eps: self.eps + rhs.eps,
        }
    }
}

impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.eps += rhs.eps;
    }
}

impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            eps: self.eps - rhs.eps,
        }
    }
}

impl SubAssign for Dual {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.eps -= rhs.eps;
    }
}

impl Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re,
            eps: self.eps * rhs.re + self.re * rhs.eps,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = 1.0 / rhs.re;
        Self {
            re: self.re * inv,
            eps: (self.eps * rhs.re - self.re * rhs.eps) * inv * inv,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            eps: -self.eps,
        }
    }
}

impl Scalar for Dual {
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
        self.re
    }
    #[inline]
    fn abs(self) -> Self {
        let sgn = if self.re >= 0.0 { 1.0 } else { -1.0 };
        self.chain(libm::fabs(self.re), sgn)
    }
    #[inline]
    fn sqrt(self) -> Self {
        let r = libm::sqrt(self.re.max(0.0));
        let dr = if r > 1e-30 { 0.5 / r } else { 0.0 };
        self.chain(r, dr)
    }
    #[inline]
    fn cbrt(self) -> Self {
        let r = libm::cbrt(self.re);
        let dr = if r.abs() > 1e-30 {
            1.0 / (3.0 * r * r)
        } else {
            0.0
        };
        self.chain(r, dr)
    }
    #[inline]
    fn exp(self) -> Self {
        let e = libm::exp(self.re);
        self.chain(e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        self.chain(libm::log(self.re), 1.0 / self.re)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        if p == 0.0 {
            return Self::one();
        }
        let v = libm::pow(self.re, p);
        let dv = p * libm::pow(self.re, p - 1.0);
        self.chain(v, dv)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        match n {
            0 => Self::one(),
            1 => self,
            2 => self * self,
            _ => self.powf(n as f64),
        }
    }
    #[inline]
    fn sin(self) -> Self {
        self.chain(libm::sin(self.re), libm::cos(self.re))
    }
    #[inline]
    fn cos(self) -> Self {
        self.chain(libm::cos(self.re), -libm::sin(self.re))
    }
    #[inline]
    fn tan(self) -> Self {
        let c = libm::cos(self.re);
        self.chain(libm::tan(self.re), 1.0 / (c * c))
    }
    #[inline]
    fn asin(self) -> Self {
        let x = self.re.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
        self.chain(
            libm::asin(self.re.clamp(-1.0, 1.0)),
            1.0 / libm::sqrt(1.0 - x * x),
        )
    }
    #[inline]
    fn acos(self) -> Self {
        let x = self.re.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
        self.chain(
            libm::acos(self.re.clamp(-1.0, 1.0)),
            -1.0 / libm::sqrt(1.0 - x * x),
        )
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        let denom = self.re * self.re + x.re * x.re;
        let eps = if denom > 1e-30 {
            (x.re * self.eps - self.re * x.eps) / denom
        } else {
            0.0
        };
        Self {
            re: libm::atan2(self.re, x.re),
            eps,
        }
    }
    #[inline]
    fn sinh(self) -> Self {
        self.chain(libm::sinh(self.re), libm::cosh(self.re))
    }
    #[inline]
    fn cosh(self) -> Self {
        self.chain(libm::cosh(self.re), libm::sinh(self.re))
    }
    #[inline]
    fn tanh(self) -> Self {
        let t = libm::tanh(self.re);
        self.chain(t, 1.0 - t * t)
    }
}
