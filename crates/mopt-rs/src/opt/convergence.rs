//! Convergence criteria (MultiOptPy `ConvergenceChecker`).
//!
//! Converged when the maximum and RMS force and the maximum and RMS displacement are all below
//! their thresholds. The displacement thresholds are relaxed by how far the forces are below
//! theirs. RMS values ignore components with `|v| <= 1e-10`, as in the reference.
//!
//! MultiOptPy: `optimization.py:ConvergenceChecker`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::domain::{Bohr, HartreePerBohr, Quantity};
use crate::math::linalg::col_max_abs;

/// Threshold set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceCriteria {
    /// Maximum force component.
    pub max_force: Quantity<HartreePerBohr>,
    /// RMS force.
    pub rms_force: Quantity<HartreePerBohr>,
    /// Maximum displacement component.
    pub max_displacement: Quantity<Bohr>,
    /// RMS displacement.
    pub rms_displacement: Quantity<Bohr>,
}

impl ConvergenceCriteria {
    /// Build from raw values in Hartree/Bohr and Bohr.
    pub const fn new(
        max_force: f64,
        rms_force: f64,
        max_displacement: f64,
        rms_displacement: f64,
    ) -> Self {
        Self {
            max_force: Quantity::new(max_force),
            rms_force: Quantity::new(rms_force),
            max_displacement: Quantity::new(max_displacement),
            rms_displacement: Quantity::new(rms_displacement),
        }
    }

    /// Default criteria.
    pub const NORMAL: Self = Self::new(0.0003, 0.0002, 0.0015, 0.0010);
    /// `-tcc`.
    pub const TIGHT: Self = Self::new(0.000015, 0.000010, 0.000060, 0.000040);
    /// `-lcc`.
    pub const LOOSE: Self = Self::new(0.0030, 0.0020, 0.0100, 0.0070);

    /// Criteria for the `-tcc` / `-lcc` flags; projection constraints quadruple the
    /// displacement thresholds (except for loose criteria, as in the reference).
    pub fn from_flags(tight: bool, loose: bool, projection_constraints: bool) -> Self {
        let (mut c, scalable) = match (tight, loose) {
            (true, false) => (Self::TIGHT, true),
            (false, true) => (Self::LOOSE, false),
            _ => (Self::NORMAL, true),
        };
        if projection_constraints && scalable {
            c.max_displacement = c.max_displacement * 4.0;
            c.rms_displacement = c.rms_displacement * 4.0;
        }
        c
    }
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// RMS over components with `|v| > 1e-10` (0 if none).
pub fn rms_nonzero(v: &Col<f64>) -> f64 {
    let (sum, n) = v
        .iter()
        .filter(|x| x.abs() > 1e-10)
        .fold((0.0, 0usize), |(s, n), x| (s + x * x, n + 1));
    if n == 0 {
        0.0
    } else {
        libm::sqrt(sum / n as f64)
    }
}

/// RMS over all components (0 for an empty vector).
pub fn rms_all(v: &Col<f64>) -> f64 {
    crate::math::linalg::col_rms(v)
}

/// Result of a convergence test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceReport {
    /// Maximum force component.
    pub max_force: Quantity<HartreePerBohr>,
    /// RMS force.
    pub rms_force: Quantity<HartreePerBohr>,
    /// Maximum displacement component.
    pub max_displacement: Quantity<Bohr>,
    /// RMS displacement.
    pub rms_displacement: Quantity<Bohr>,
    /// Effective maximum displacement threshold.
    pub max_displacement_threshold: Quantity<Bohr>,
    /// Effective RMS displacement threshold.
    pub rms_displacement_threshold: Quantity<Bohr>,
    /// All four criteria met.
    pub converged: bool,
}

impl ConvergenceCriteria {
    /// Test a Cartesian gradient (Hartree/Bohr) and displacement (Bohr).
    pub fn check(&self, gradient: &Col<f64>, displacement: &Col<f64>) -> ConvergenceReport {
        let max_force = Quantity::new(col_max_abs(gradient));
        let rms_force = Quantity::new(rms_nonzero(gradient));
        let max_displacement = Quantity::new(col_max_abs(displacement));
        let rms_displacement = Quantity::new(rms_nonzero(displacement));
        // The force margin (Hartree/Bohr) is added to the displacement threshold (Bohr) by
        // MultiOptPy; the numeric values are combined as in the reference.
        let max_th =
            self.max_displacement + Quantity::new((self.max_force - max_force).value().max(0.0));
        let rms_th =
            self.rms_displacement + Quantity::new((self.rms_force - rms_force).value().max(0.0));
        ConvergenceReport {
            max_force,
            rms_force,
            max_displacement,
            rms_displacement,
            max_displacement_threshold: max_th,
            rms_displacement_threshold: rms_th,
            converged: max_force < self.max_force
                && rms_force < self.rms_force
                && max_displacement < max_th
                && rms_displacement < rms_th,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_relax_with_small_forces() {
        let c = ConvergenceCriteria::NORMAL;
        let g = Col::from_fn(3, |i| [1e-5, 0.0, -1e-5][i]);
        let d = Col::from_fn(3, |i| [0.0016, 0.0004, 0.0][i]);
        let r = c.check(&g, &d);
        assert!((r.max_displacement_threshold.value() - (0.0015 + 0.0003 - 1e-5)).abs() < 1e-15);
        assert!((r.rms_force.value() - 1e-5).abs() < 1e-18);
        assert!(r.converged);
        assert_eq!(
            ConvergenceCriteria::from_flags(false, false, true)
                .max_displacement
                .value(),
            0.006
        );
    }
}
