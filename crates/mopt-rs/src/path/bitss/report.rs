//! What a BITSS search asks for and what it reports.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy keeps the two images in one optimizer state).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::function::Coefficients;
use crate::opt::convergence::ConvergenceReport;

/// Which point of a search an evaluation is asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointRole {
    /// The first image.
    First,
    /// The second image.
    Second,
    /// A point on the line from the first image to the second, at this share of the way.
    Between(f64),
}

/// A point to evaluate.
#[derive(Debug, Clone)]
pub struct BitssPoint {
    /// What the point is.
    pub role: PointRole,
    /// Its variables.
    pub x: Col<f64>,
    /// Whether its gradient is asked for (its energy always is).
    pub gradient: bool,
    /// Whether its Hessian is asked for.
    pub hessian: bool,
}

/// The surface at a point.
#[derive(Debug, Clone)]
pub struct PointValue {
    /// Energy.
    pub energy: f64,
    /// Gradient, when asked for.
    pub gradient: Option<Col<f64>>,
    /// Hessian, when asked for.
    pub hessian: Option<Mat<f64>>,
}

/// Why a search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitssEnd {
    /// The gradient at the midpoint of the images met the force thresholds.
    MidpointGradient,
    /// The images came closer than the smallest separation.
    Separation,
    /// The midpoint hardly moved during the last stage.
    MidpointShift,
    /// The images came to rest or close to each other, but their midpoint is not higher in
    /// energy than both: the pair has left the ridge and lies in one basin.
    NoBarrier,
    /// The number of stages was used up.
    StageLimit,
    /// The number of iterations was used up.
    IterationLimit,
}

impl BitssEnd {
    /// Whether the search ended on one of its criteria (not on a limit).
    pub fn converged(self) -> bool {
        !matches!(
            self,
            Self::NoBarrier | Self::StageLimit | Self::IterationLimit
        )
    }
}

/// Next action of a [`super::Bitss`] search.
#[derive(Debug, Clone)]
pub enum BitssAction {
    /// Evaluate these points and pass their values, in this order, to [`super::Bitss::tell`].
    Evaluate(Vec<BitssPoint>),
    /// The search has ended.
    Finished(BitssEnd),
}

/// The midpoint of the images as the last survey of the line between them found it.
#[derive(Debug, Clone)]
pub struct Midpoint {
    /// Its variables.
    pub x: Col<f64>,
    /// Its energy.
    pub energy: f64,
    /// Its gradient (frozen variables zeroed), when it was asked for: at the end of a
    /// stage.
    pub gradient: Option<Col<f64>>,
    /// Largest component of that gradient.
    pub max_force: Option<f64>,
    /// Root mean square of that gradient over the variables that are not frozen.
    pub rms_force: Option<f64>,
}

/// The step of the minimizer that followed an evaluation of the images.
#[derive(Debug, Clone)]
pub struct StepSummary {
    /// Convergence test of the minimization.
    pub convergence: ConvergenceReport,
    /// Length of the step in the `2n` variables.
    pub step_norm: f64,
    /// Trust radius.
    pub trust_radius: f64,
    /// Whether this ended the minimization of the stage.
    pub stage_converged: bool,
}

/// What one call of [`super::Bitss::tell`] found.
#[derive(Debug, Clone)]
pub struct BitssReport {
    /// Iterations of the minimizer so far, all stages together.
    pub iteration: usize,
    /// Stage (1 for the first reduction of the separation).
    pub stage: usize,
    /// Separation asked for at this stage.
    pub target_separation: f64,
    /// Separation of the images of this report.
    pub separation: f64,
    /// The two images this report is about.
    pub images: [Col<f64>; 2],
    /// Their energies.
    pub energies: [f64; 2],
    /// Their gradients.
    pub gradients: [Col<f64>; 2],
    /// The coefficients in force.
    pub coefficients: Coefficients,
    /// The BITSS function.
    pub function: f64,
    /// Its gradient in the `2n` variables (frozen variables zeroed).
    pub gradient: Col<f64>,
    /// The step of the minimizer (`None` when the search ended at this call).
    pub step: Option<StepSummary>,
    /// The midpoint, when the line between the images was surveyed at this call.
    pub midpoint: Option<Midpoint>,
    /// Whether this call began a new stage.
    pub new_stage: bool,
    /// Why the search ended, if it did at this call.
    pub end: Option<BitssEnd>,
    /// Notes of the minimizer.
    pub notes: Vec<String>,
}
