//! What an IRC calculation asks for, is told, and reports.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's integrators call the calculator
//! and write their files themselves).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

/// What a point that is to be evaluated is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointRole {
    /// The starting structure (with its Hessian).
    TransitionState,
    /// A point of the path.
    Path,
    /// A point an integrator needs between two points of the path (the stages of RK4, the
    /// trial points of a minimization on a hypersphere).
    Auxiliary,
    /// The first point away from the transition state while it is tested and corrected.
    FirstStep,
    /// The last structure of ModeKill, for its Hessian.
    FinalHessian,
}

/// A point to evaluate.
#[derive(Debug, Clone)]
pub struct PointRequest {
    /// The variables (Cartesian coordinates in Bohr for a structure).
    pub x: Col<f64>,
    /// Whether the Hessian of the surface is needed.
    pub need_hessian: bool,
    /// What the point is for.
    pub role: PointRole,
    /// Number of the path point it belongs to (0 for the transition state).
    pub step: usize,
}

/// The surface at a point. The path follows `energy` and `gradient` (with the bias
/// potentials); `raw_*` are the values without them, as MultiOptPy reports both.
#[derive(Debug, Clone)]
pub struct PointValue {
    /// Energy of the surface that is followed.
    pub energy: f64,
    /// Its gradient with respect to the variables.
    pub gradient: Col<f64>,
    /// Energy without bias potentials.
    pub raw_energy: f64,
    /// Gradient without bias potentials.
    pub raw_gradient: Col<f64>,
    /// Hessian without bias potentials (when it was asked for).
    pub hessian: Option<Mat<f64>>,
    /// Hessian of the bias potentials (`None`: there are none).
    pub bias_hessian: Option<Mat<f64>>,
}

impl PointValue {
    /// A point of a surface without bias potentials.
    pub fn plain(energy: f64, gradient: Col<f64>, hessian: Option<Mat<f64>>) -> Self {
        Self {
            energy,
            raw_energy: energy,
            raw_gradient: gradient.clone(),
            gradient,
            hessian,
            bias_hessian: None,
        }
    }

    /// Whether every number is finite.
    pub fn is_finite(&self) -> bool {
        let matrix = |m: &Option<Mat<f64>>| {
            m.as_ref().is_none_or(|m| {
                (0..m.ncols()).all(|j| (0..m.nrows()).all(|i| m[(i, j)].is_finite()))
            })
        };
        self.energy.is_finite()
            && self.raw_energy.is_finite()
            && self.gradient.iter().all(|v| v.is_finite())
            && self.raw_gradient.iter().all(|v| v.is_finite())
            && matrix(&self.hessian)
            && matrix(&self.bias_hessian)
    }
}

/// A side of the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Branch {
    /// Along the transition vector (MultiOptPy's `irc_forward`).
    Forward,
    /// Against it (`irc_backward`).
    Backward,
    /// The one path of a start that is no transition state (meta-IRC) and of ModeKill.
    Single,
}

impl Branch {
    /// `forward`, `backward`, or `single`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
            Self::Single => "single",
        }
    }

    /// `+1` for forward and single, `-1` for backward.
    pub fn sign(self) -> f64 {
        if self == Self::Backward {
            -1.0
        } else {
            1.0
        }
    }
}

/// Why a side ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchEnd {
    /// The gradient is below the thresholds (a minimum, or a saddle point the path ran into).
    Converged,
    /// The energy change reversed its sign too many times in a row.
    Oscillation,
    /// The number of steps is used up.
    MaxSteps,
    /// The energy rises along the path (GS2: even with the smallest step; the other
    /// integrators: when they are set to end there): a minimum lies within the last step.
    EnergyRise,
    /// ModeKill: the modes to remove are no longer imaginary.
    ModesRemoved,
    /// ModeKill: there was no mode to remove.
    NothingToRemove,
}

impl BranchEnd {
    /// Whether the side reached what it was run for.
    pub fn arrived(self) -> bool {
        matches!(
            self,
            Self::Converged | Self::EnergyRise | Self::ModesRemoved | Self::NothingToRemove
        )
    }

    /// One line for the log.
    pub fn message(self) -> &'static str {
        match self {
            Self::Converged => "Convergence reached. (IRC)",
            Self::Oscillation => "Terminating IRC: Energy oscillated for 5 consecutive steps",
            Self::MaxSteps => "The number of IRC steps is used up.",
            Self::EnergyRise => {
                "The energy rises along the path: a minimum lies within the last step. (IRC)"
            }
            Self::ModesRemoved => "All targeted modes have been removed.",
            Self::NothingToRemove => "No imaginary modes to remove.",
        }
    }
}

/// Curvature of the path at a point, with its components along the modes of positive
/// curvature of the surface.
#[derive(Debug, Clone, PartialEq)]
pub struct PathCurvature {
    /// Norm of the change of the unit tangent per mass-weighted length.
    pub scalar: f64,
    /// Its components along the eigenvectors of the Hessian with eigenvalues above `1e-8`.
    pub couplings: Vec<f64>,
}

/// A point of the path as it is written to the files.
#[derive(Debug, Clone)]
pub struct StepRecord {
    /// MultiOptPy's step number of the point.
    pub step: usize,
    /// The variables.
    pub x: Col<f64>,
    /// Energy without bias potentials.
    pub energy: f64,
    /// Energy with them.
    pub biased_energy: f64,
    /// RMS of the gradient without bias potentials.
    pub rms_gradient: f64,
    /// RMS of the gradient with them.
    pub rms_biased_gradient: f64,
    /// Mass-weighted length of the path from the first point of the side to this one.
    pub arc_length: f64,
    /// Angle (degrees) at the previous point between the lines to its two neighbours.
    pub bending_angle: Option<f64>,
    /// Further columns of the table of the integrator (name, text).
    pub extras: Vec<(&'static str, String)>,
    /// Curvature of the path (Euler and RK4, as MultiOptPy writes it for them).
    pub curvature: Option<PathCurvature>,
    /// Whether the point has a row in the table (MultiOptPy's ModeKill leaves out the row
    /// of the structure at which no mode was left).
    pub in_table: bool,
    /// Lines for the log.
    pub notes: Vec<String>,
    /// The lowest curvature across the path where the point has an exact Hessian
    /// (mass-weighted, Hartree / (amu Bohr^2); `IrcSpace::curvature_across`): negative past a
    /// valley-ridge inflection, where the path may bifurcate.
    pub curvature_across: Option<f64>,
}

/// The largest absolute component of `v`.
/// What the step control of a side did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StepCounts {
    /// Points that were dropped because their energy was above that of the point before
    /// (their steps were taken again, shorter).
    pub dropped: usize,
    /// Points that were dropped because the gradient there pointed back along the step
    /// (the step went past the floor of the valley).
    pub overshot: usize,
    /// Times the length of the following steps was halved after a point that was kept
    /// (the path turned back, or the quadratic model was off).
    pub shortened: usize,
    /// Exact Hessians asked for after a step that raised the energy.
    pub refreshed: usize,
    /// Steps of the final approach.
    pub approach: usize,
}

pub(crate) fn max_abs(v: &Col<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// `sqrt(mean(v^2))`.
pub(crate) fn rms(v: &Col<f64>) -> f64 {
    if v.nrows() == 0 {
        return 0.0;
    }
    libm::sqrt(v.iter().map(|x| x * x).sum::<f64>() / v.nrows() as f64)
}

/// Euclidean norm.
pub(crate) fn norm(v: &Col<f64>) -> f64 {
    libm::sqrt(v.iter().map(|x| x * x).sum::<f64>())
}

/// Inner product.
pub(crate) fn dot(a: &Col<f64>, b: &Col<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `a + f * b`.
pub(crate) fn axpy(a: &Col<f64>, f: f64, b: &Col<f64>) -> Col<f64> {
    Col::from_fn(a.nrows(), |i| a[i] + f * b[i])
}
