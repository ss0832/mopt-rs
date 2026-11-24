//! Unit-tagged physical quantities.
//!
//! A [`Quantity<U>`] is an `f64` measured in unit `U`; a [`Vector3<U>`] is a 3-vector in `U`.
//! Units belong to a dimension (length, energy, force, angle), and conversion with
//! [`Quantity::to`] only compiles between units of the same dimension, so Bohr cannot be mixed
//! up with Angstrom, or Hartree with eV or kcal/mol, without an explicit conversion. All types
//! are zero-cost wrappers.
//!
//! Internally `mopt-rs` works in Bohr and Hartree; other units appear only at I/O boundaries.
//!
//! MultiOptPy: `Parameters/unit_values.py:UnitValueLib` (typed instead of factors).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::cmp::Ordering;
use std::fmt;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use crate::domain::units::Units;

/// Physical dimension marker.
pub trait Dimension: Copy + fmt::Debug + Default + Send + Sync + 'static {}

/// Length.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LengthDim;
/// Energy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnergyDim;
/// Force (energy per length).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForceDim;
/// Plane angle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AngleDim;

impl Dimension for LengthDim {}
impl Dimension for EnergyDim {}
impl Dimension for ForceDim {}
impl Dimension for AngleDim {}

/// A unit of measurement.
pub trait Unit: Copy + fmt::Debug + Default + Send + Sync + 'static {
    /// Dimension of the unit.
    type Dim: Dimension;
    /// Symbol used for display.
    const SYMBOL: &'static str;
    /// Value of one unit in the internal unit of its dimension (Bohr, Hartree, Hartree/Bohr,
    /// radian).
    const TO_INTERNAL: f64;
}

macro_rules! unit {
    ($(#[$m:meta])* $name:ident, $dim:ty, $sym:expr, $factor:expr) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $name;
        impl Unit for $name {
            type Dim = $dim;
            const SYMBOL: &'static str = $sym;
            const TO_INTERNAL: f64 = $factor;
        }
    };
}

unit!(/// Bohr radius (internal length unit).
    Bohr, LengthDim, "Bohr", 1.0);
unit!(/// Angstrom.
    Angstrom, LengthDim, "Angstrom", Units::ANGSTROM_TO_BOHR);
unit!(/// Hartree (internal energy unit).
    Hartree, EnergyDim, "Hartree", 1.0);
unit!(/// Electron volt.
    ElectronVolt, EnergyDim, "eV", Units::EV_TO_HARTREE);
unit!(/// Kilocalorie per mole.
    KcalPerMol, EnergyDim, "kcal/mol", Units::KCAL_MOL_TO_HARTREE);
unit!(/// Kilojoule per mole.
    KjPerMol, EnergyDim, "kJ/mol", Units::KJ_MOL_TO_HARTREE);
unit!(/// Hartree per Bohr (internal force / gradient unit).
    HartreePerBohr, ForceDim, "Hartree/Bohr", 1.0);
unit!(/// Electron volt per Angstrom.
    EvPerAngstrom, ForceDim, "eV/Angstrom", Units::EV_TO_HARTREE / Units::ANGSTROM_TO_BOHR);
unit!(/// Radian (internal angle unit).
    Radian, AngleDim, "rad", 1.0);
unit!(/// Degree.
    Degree, AngleDim, "deg", Units::DEG_TO_RAD);

/// A scalar measured in unit `U`.
#[derive(Clone, Copy, Default)]
pub struct Quantity<U: Unit> {
    value: f64,
    unit: PhantomData<U>,
}

impl<U: Unit> Quantity<U> {
    /// `value` in unit `U`.
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self {
            value,
            unit: PhantomData,
        }
    }

    /// Zero.
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0)
    }

    /// The number in unit `U`.
    #[inline]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Convert to another unit of the same dimension.
    #[inline]
    pub fn to<V: Unit<Dim = U::Dim>>(self) -> Quantity<V> {
        Quantity::new(self.value * U::TO_INTERNAL / V::TO_INTERNAL)
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self::new(self.value.abs())
    }

    /// Smaller of two quantities.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self::new(self.value.min(other.value))
    }

    /// Larger of two quantities.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self::new(self.value.max(other.value))
    }

    /// Whether the value is finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.value.is_finite()
    }
}

impl<U: Unit> fmt::Debug for Quantity<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} {}", self.value, U::SYMBOL)
    }
}

impl<U: Unit> fmt::Display for Quantity<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)?;
        write!(f, " {}", U::SYMBOL)
    }
}

impl<U: Unit> PartialEq for Quantity<U> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<U: Unit> PartialOrd for Quantity<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<U: Unit> Add for Quantity<U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}

impl<U: Unit> Sub for Quantity<U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}

impl<U: Unit> AddAssign for Quantity<U> {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<U: Unit> SubAssign for Quantity<U> {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<U: Unit> Neg for Quantity<U> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}

impl<U: Unit> Mul<f64> for Quantity<U> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.value * rhs)
    }
}

impl<U: Unit> Mul<Quantity<U>> for f64 {
    type Output = Quantity<U>;
    fn mul(self, rhs: Quantity<U>) -> Quantity<U> {
        Quantity::new(self * rhs.value)
    }
}

impl<U: Unit> Div<f64> for Quantity<U> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.value / rhs)
    }
}

impl<U: Unit> Div for Quantity<U> {
    type Output = f64;
    fn div(self, rhs: Self) -> f64 {
        self.value / rhs.value
    }
}

impl<U: Unit> Sum for Quantity<U> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new(iter.map(|q| q.value).sum())
    }
}

/// A 3-vector measured in unit `U` (positions, displacements, lattice vectors).
#[derive(Clone, Copy, Default)]
pub struct Vector3<U: Unit> {
    xyz: [f64; 3],
    unit: PhantomData<U>,
}

impl<U: Unit> Vector3<U> {
    /// Components in unit `U`.
    #[inline]
    pub const fn new(xyz: [f64; 3]) -> Self {
        Self {
            xyz,
            unit: PhantomData,
        }
    }

    /// Components in unit `U`.
    #[inline]
    pub const fn value(self) -> [f64; 3] {
        self.xyz
    }

    /// Convert to another unit of the same dimension.
    #[inline]
    pub fn to<V: Unit<Dim = U::Dim>>(self) -> Vector3<V> {
        let f = U::TO_INTERNAL / V::TO_INTERNAL;
        Vector3::new(self.xyz.map(|c| c * f))
    }

    /// Euclidean norm.
    #[inline]
    pub fn norm(self) -> Quantity<U> {
        let [x, y, z] = self.xyz;
        Quantity::new(libm::sqrt(x * x + y * y + z * z))
    }
}

impl<U: Unit> fmt::Debug for Vector3<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} {}", self.xyz, U::SYMBOL)
    }
}

impl<U: Unit> PartialEq for Vector3<U> {
    fn eq(&self, other: &Self) -> bool {
        self.xyz == other.xyz
    }
}

impl<U: Unit> Add for Vector3<U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new([
            self.xyz[0] + rhs.xyz[0],
            self.xyz[1] + rhs.xyz[1],
            self.xyz[2] + rhs.xyz[2],
        ])
    }
}

impl<U: Unit> Sub for Vector3<U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new([
            self.xyz[0] - rhs.xyz[0],
            self.xyz[1] - rhs.xyz[1],
            self.xyz[2] - rhs.xyz[2],
        ])
    }
}

impl<U: Unit> Mul<f64> for Vector3<U> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.xyz.map(|c| c * rhs))
    }
}

/// Atomic position.
pub type Position<U> = Vector3<U>;

/// Lift raw `[f64; 3]` rows given in unit `U` into typed positions.
pub fn positions<U: Unit<Dim = LengthDim>>(rows: &[[f64; 3]]) -> Vec<Position<U>> {
    rows.iter().map(|r| Position::new(*r)).collect()
}

/// Shorthand constructors.
pub mod shorthand {
    use super::*;

    /// Length in Bohr.
    pub const fn bohr(v: f64) -> Quantity<Bohr> {
        Quantity::new(v)
    }
    /// Length in Angstrom.
    pub const fn angstrom(v: f64) -> Quantity<Angstrom> {
        Quantity::new(v)
    }
    /// Energy in Hartree.
    pub const fn hartree(v: f64) -> Quantity<Hartree> {
        Quantity::new(v)
    }
    /// Energy in eV.
    pub const fn ev(v: f64) -> Quantity<ElectronVolt> {
        Quantity::new(v)
    }
    /// Energy in kcal/mol.
    pub const fn kcal_per_mol(v: f64) -> Quantity<KcalPerMol> {
        Quantity::new(v)
    }
    /// Force in Hartree/Bohr.
    pub const fn hartree_per_bohr(v: f64) -> Quantity<HartreePerBohr> {
        Quantity::new(v)
    }
    /// Angle in degrees.
    pub const fn degrees(v: f64) -> Quantity<Degree> {
        Quantity::new(v)
    }
}

#[cfg(test)]
mod tests {
    use super::shorthand::*;
    use super::*;

    #[test]
    fn conversions_round_trip_within_a_dimension() {
        let a = angstrom(1.0);
        let b: Quantity<Bohr> = a.to();
        assert!((b.value() - 1.889_726_124_625_77).abs() < 1e-12);
        assert!((b.to::<Angstrom>().value() - 1.0).abs() < 1e-15);
        let e: Quantity<KcalPerMol> = hartree(1.0).to();
        assert!((e.value() - 627.509_474_063_055_8).abs() < 1e-10);
        assert!((ev(27.211_386_245_988).to::<Hartree>().value() - 1.0).abs() < 1e-12);
        assert!((degrees(180.0).to::<Radian>().value() - std::f64::consts::PI).abs() < 1e-15);
        let p = Position::<Angstrom>::new([1.0, 0.0, 0.0]).to::<Bohr>();
        assert!((p.norm().value() - b.value()).abs() < 1e-15);
        assert_eq!(format!("{}", bohr(2.0)), "2 Bohr");
        assert!(hartree(1.0) > hartree(0.5));
        assert_eq!(hartree(3.0) / hartree(1.5), 2.0);
    }
}
