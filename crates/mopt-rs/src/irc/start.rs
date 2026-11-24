//! The start of an IRC calculation: is the structure a transition state, and where do the
//! sides begin.
//!
//! The Hessian of the surface (with the bias potentials) is mass-weighted, the rigid-body
//! motion is projected out, and the eigenvalues below `-1e-8` are counted. With exactly one
//! and a gradient below the thresholds the structure is a transition state: the two sides
//! start at `q_TS +- L v` along the eigenvector `v` of that mode (`L`: 0.1 of the step
//! size). Otherwise the one path of a meta-IRC starts at `q - L g/|g|`. ModeKill starts at
//! the structure itself.
//!
//! Differences to MultiOptPy (`docs/source/migration_paths.md`): MultiOptPy adds the eigenvector
//! of the mass-weighted Hessian to the Cartesian coordinates without dividing by the square
//! roots of the masses, and for a meta-IRC it normalizes the Cartesian gradient before
//! dividing by them; both displacements are the mass-weighted ones here. It tests the
//! gradient without the bias potentials (a transition state found with a bias potential
//! was never taken for one) with the signed largest component, and with thresholds
//! (0.0004, 0.0001) of which the second is below the one of its optimizer (0.0002), so that
//! `optmain -irc` ran a meta-IRC from many converged transition states; here a structure
//! the optimization converged to passes ([`IrcSettings::accepting_converged`]). The sign of an eigenvector
//! is arbitrary; here the forward side is the one along which its largest component is
//! positive.
//!
//! MultiOptPy: `irc.py:IRC.saddle_check`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::convergence::convergence_check;
use super::integrator::unit;
use super::modekill::eigenvalue_to_wavenumber;
use super::settings::{IrcSettings, Sides};
use super::space::IrcSpace;
use super::types::{max_abs, rms, Branch, PointValue};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// What the calculation does from its starting structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartKind {
    /// A transition state: both sides along the transition vector.
    TransitionState,
    /// No transition state: one path down the gradient (meta-IRC).
    MetaIrc,
    /// ModeKill: remove imaginary modes of the structure.
    ModeKill,
}

impl StartKind {
    /// MultiOptPy's line for the log.
    pub fn message(self) -> &'static str {
        match self {
            Self::TransitionState => "Execute IRC",
            Self::MetaIrc => "Execute meta-IRC",
            Self::ModeKill => "Execute ModeKill",
        }
    }
}

/// The starting structure as the calculation found it.
#[derive(Debug, Clone)]
pub struct StartReport {
    /// What is done from it.
    pub kind: StartKind,
    /// Number of imaginary modes.
    pub imaginary: usize,
    /// Eigenvalues of the projected mass-weighted Hessian with magnitudes above `1e-7`,
    /// ascending (MultiOptPy prints these).
    pub eigenvalues: Vec<f64>,
    /// Eigenvalue of the transition vector (a transition state only).
    pub transition_eigenvalue: Option<f64>,
    /// Its wavenumber (cm^-1, negative).
    pub transition_wavenumber: Option<f64>,
    /// The transition vector: the unit eigenvector of that mode in mass-weighted
    /// coordinates, towards the forward side.
    pub transition_vector: Option<Col<f64>>,
    /// The Hessian of the surface without bias potentials at the structure (Cartesian,
    /// Hartree/Bohr^2), for the vibrations of a transition state.
    pub hessian: Mat<f64>,
    /// Whether the gradient is below the thresholds.
    pub stationary: bool,
    /// Largest absolute component of the gradient of the surface that is followed.
    pub max_gradient: f64,
    /// Its RMS.
    pub rms_gradient: f64,
    /// Energy without bias potentials.
    pub energy: f64,
    /// Energy with them.
    pub biased_energy: f64,
    /// RMS of the gradient without bias potentials.
    pub rms_raw_gradient: f64,
}

/// What the sides need of the starting structure.
pub(crate) struct Start {
    pub report: StartReport,
    /// Mass-weighted, projected Hessian without bias potentials.
    pub hessian: Mat<f64>,
    /// The same with them.
    pub total_hessian: Mat<f64>,
    /// The sides and their first points.
    pub sides: Vec<(Branch, Col<f64>)>,
}

/// `v` with the sign that makes its largest component positive.
fn oriented(v: Col<f64>) -> Col<f64> {
    let mut largest = 0.0_f64;
    for value in v.iter() {
        if value.abs() > largest.abs() {
            largest = *value;
        }
    }
    if largest < 0.0 {
        v * -1.0
    } else {
        v
    }
}

/// Examine the starting structure `x` with the surface `value` there (with its Hessian).
pub(crate) fn examine(
    space: &IrcSpace,
    settings: &IrcSettings,
    modekill: bool,
    x: &Col<f64>,
    value: &PointValue,
) -> MoptResult<Start> {
    let raw = value
        .hessian
        .as_ref()
        .ok_or_else(|| MoptError::calc("IRC: the Hessian of the starting structure is missing"))?;
    let hessian = space.prepared_hessian(raw, x)?;
    let total_hessian = match &value.bias_hessian {
        Some(b) => space.prepared_hessian(&(raw + b), x)?,
        None => hessian.clone(),
    };
    let (values, vectors) = symmetric_eigen(&total_hessian)?;
    let n = values.nrows();
    let imaginary: Vec<usize> = (0..n)
        .filter(|&k| values[k] < -settings.imaginary_threshold)
        .collect();
    let (max_force, rms_force) = settings
        .start_forces
        .unwrap_or((settings.max_force, settings.rms_force));
    let stationary = convergence_check(&value.gradient, max_force, rms_force);
    let kind = if modekill {
        StartKind::ModeKill
    } else if imaginary.len() == 1 && stationary {
        StartKind::TransitionState
    } else {
        StartKind::MetaIrc
    };
    let length = settings.first_displacement();
    let mut vector = None;
    let mut sides = Vec::new();
    match kind {
        StartKind::TransitionState => {
            let k = imaginary[0];
            let v = oriented(Col::from_fn(n, |i| vectors[(i, k)]));
            if settings.sides != Sides::Backward {
                sides.push((Branch::Forward, space.displaced(x, &(&v * length))));
            }
            if settings.sides != Sides::Forward {
                sides.push((Branch::Backward, space.displaced(x, &(&v * (-length)))));
            }
            vector = Some(v);
        }
        StartKind::MetaIrc => {
            let g = unit(&space.gradient_to_mw(&value.gradient));
            sides.push((Branch::Single, space.displaced(x, &(g * (-length)))));
        }
        StartKind::ModeKill => sides.push((Branch::Single, x.clone())),
    }
    let transition_eigenvalue = (kind == StartKind::TransitionState).then(|| values[imaginary[0]]);
    let report = StartReport {
        kind,
        imaginary: imaginary.len(),
        eigenvalues: values.iter().copied().filter(|v| v.abs() > 1e-7).collect(),
        transition_eigenvalue,
        transition_wavenumber: transition_eigenvalue.map(eigenvalue_to_wavenumber),
        transition_vector: vector,
        hessian: raw.clone(),
        stationary,
        max_gradient: max_abs(&value.gradient),
        rms_gradient: rms(&value.gradient),
        energy: value.raw_energy,
        biased_energy: value.energy,
        rms_raw_gradient: rms(&value.raw_gradient),
    };
    Ok(Start {
        report,
        hessian,
        total_hessian,
        sides,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::irc::types::norm;

    fn value(g: [f64; 2], h: [[f64; 2]; 2]) -> PointValue {
        PointValue::plain(
            0.0,
            Col::from_fn(2, |i| g[i]),
            Some(Mat::from_fn(2, 2, |i, j| h[i][j])),
        )
    }

    #[test]
    fn transition_state_meta_irc_and_modekill() {
        let masses = [4.0, 1.0];
        let space = IrcSpace::with_masses(&masses).unwrap();
        let settings = IrcSettings::default();
        let x = Col::from_fn(2, |i| [0.5, -0.5][i]);
        // In mass-weighted coordinates the Hessian is diag(-0.25, 2): the transition vector
        // is the first variable, and the first points lie 0.01 away in q, 0.005 in x.
        let ts = value([0.0, 0.0], [[-1.0, 0.0], [0.0, 2.0]]);
        let start = examine(&space, &settings, false, &x, &ts).unwrap();
        assert_eq!(start.report.kind, StartKind::TransitionState);
        assert_eq!(start.report.imaginary, 1);
        assert!((start.report.transition_eigenvalue.unwrap() + 0.25).abs() < 1e-14);
        assert_eq!(start.sides.len(), 2);
        let (branch, first) = &start.sides[0];
        assert_eq!(*branch, Branch::Forward);
        assert!((first[0] - 0.505).abs() < 1e-14 && (first[1] + 0.5).abs() < 1e-14);
        assert!((start.sides[1].1[0] - 0.495).abs() < 1e-14);
        // A gradient above the thresholds: one path down the mass-weighted gradient.
        let slope = value([0.02, 0.01], [[-1.0, 0.0], [0.0, 2.0]]);
        let start = examine(&space, &settings, false, &x, &slope).unwrap();
        assert_eq!(start.report.kind, StartKind::MetaIrc);
        assert_eq!(start.sides.len(), 1);
        let dq = space.to_mw(&(&start.sides[0].1 - &x));
        assert!((norm(&dq) - 0.01).abs() < 1e-14);
        // g_mw = (0.01, 0.01): the two components of dq are equal.
        assert!((dq[0] - dq[1]).abs() < 1e-14 && dq[0] < 0.0);
        // Two imaginary modes: no transition state either; ModeKill starts in place.
        let second = value([0.0, 0.0], [[-1.0, 0.0], [0.0, -2.0]]);
        assert_eq!(
            examine(&space, &settings, false, &x, &second)
                .unwrap()
                .report
                .kind,
            StartKind::MetaIrc
        );
        // The gradient that counts is the one of the surface that is followed: a transition
        // state found with a bias potential has a gradient without it.
        let mut biased = value([0.0, 0.0], [[-1.0, 0.0], [0.0, 2.0]]);
        biased.raw_gradient = Col::from_fn(2, |i| [0.05, 0.0][i]);
        assert_eq!(
            examine(&space, &settings, false, &x, &biased)
                .unwrap()
                .report
                .kind,
            StartKind::TransitionState
        );
        // A structure an optimization converged to with wider thresholds passes.
        let nearly = value([2.5e-4, 2.5e-4], [[-1.0, 0.0], [0.0, 2.0]]);
        assert_eq!(
            examine(&space, &settings, false, &x, &nearly)
                .unwrap()
                .report
                .kind,
            StartKind::MetaIrc
        );
        let accepting = settings.clone().accepting_converged(3e-4, 2.5e-4);
        assert_eq!(
            examine(&space, &accepting, false, &x, &nearly)
                .unwrap()
                .report
                .kind,
            StartKind::TransitionState
        );
        let kill = examine(&space, &settings, true, &x, &second).unwrap();
        assert_eq!(kill.report.kind, StartKind::ModeKill);
        assert_eq!(kill.sides[0].1, x);
    }
}
