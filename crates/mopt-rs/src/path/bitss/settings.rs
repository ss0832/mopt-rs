//! Settings of a BITSS search and their defaults.
//!
//! MultiOptPy: `ModelFunction/binary_image_ts_search_model_function.py:BITSSModelFunction` (`f`, `alpha`, `beta`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};
use crate::opt::vector::VectorSettings;

/// Share by which the separation asked for shrinks from stage to stage (`f` of the paper).
pub const DEFAULT_REDUCTION: f64 = 0.5;
/// `alpha` of the paper: strength of the energy term.
pub const DEFAULT_ALPHA: f64 = 10.0;
/// `beta` of the paper: (inverse) strength of the distance term.
pub const DEFAULT_BETA: f64 = 0.1;
/// Iterations of a minimization after which the coefficients are set again (the paper's).
pub const DEFAULT_COEFFICIENT_INTERVAL: usize = 100;
/// Points on the line between the images whose energies enter the barrier estimate.
pub const DEFAULT_INTERPOLATION_POINTS: usize = 3;
/// Separation of the images below which the search ends. With a curvature `c` along the
/// line between the images the barrier estimate is about `|c| d^2 / 8`; at 0.01 and
/// `|c| = 0.05` that is `6e-7`, still well above the noise of energies converged to `1e-8`.
pub const DEFAULT_MIN_SEPARATION: f64 = 1e-2;
/// Stages after which the search gives up (`0.5^30` of the first separation).
pub const DEFAULT_MAX_STAGES: usize = 30;
/// Longest step of the pair of images as a share of the separation asked for.
pub const DEFAULT_STEP_SHARE: Option<f64> = Some(0.5);

/// Settings of a BITSS search.
#[derive(Debug, Clone)]
pub struct BitssSettings {
    /// `f`: the separation asked for is multiplied by `1 - f` at every stage.
    pub reduction: f64,
    /// `alpha` of `kappa_e`.
    pub alpha: f64,
    /// `beta` of `kappa_d`.
    pub beta: f64,
    /// Iterations of a minimization after which the coefficients are set again.
    pub coefficient_interval: usize,
    /// Points on the line between the images for the barrier estimate (the midpoint is
    /// always one of them).
    pub interpolation_points: usize,
    /// End when the gradient at the midpoint of the images meets the force thresholds of
    /// the minimizer.
    pub stop_on_midpoint_gradient: bool,
    /// End when the separation of the images after a minimization is at most this.
    pub min_separation: f64,
    /// End when the midpoint moved, during a stage, by less than this share of the
    /// reduction of the separation of that stage (`None`: not tested; the paper uses a
    /// tenth before handing over to an eigenvector-following search).
    pub midpoint_shift: Option<f64>,
    /// Stages after which the search gives up.
    pub max_stages: usize,
    /// No step of the pair of images is longer than this share of the separation asked for
    /// at its stage (`None`: only the trust radius of the minimizer limits it). A long step
    /// can carry an image over the ridge into the basin of the other.
    pub step_share: Option<f64>,
    /// The minimizer of the stages; its iteration limit counts all stages together.
    pub minimizer: VectorSettings,
}

impl Default for BitssSettings {
    fn default() -> Self {
        Self {
            reduction: DEFAULT_REDUCTION,
            alpha: DEFAULT_ALPHA,
            beta: DEFAULT_BETA,
            coefficient_interval: DEFAULT_COEFFICIENT_INTERVAL,
            interpolation_points: DEFAULT_INTERPOLATION_POINTS,
            stop_on_midpoint_gradient: true,
            min_separation: DEFAULT_MIN_SEPARATION,
            midpoint_shift: None,
            max_stages: DEFAULT_MAX_STAGES,
            step_share: DEFAULT_STEP_SHARE,
            minimizer: VectorSettings::default(),
        }
    }
}

impl BitssSettings {
    pub(super) fn validate(&self) -> MoptResult<()> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !(positive(self.reduction) && self.reduction < 1.0) {
            return Err(MoptError::config(
                "BITSS: the reduction of the separation must lie between 0 and 1",
            ));
        }
        if !positive(self.alpha) || !positive(self.beta) {
            return Err(MoptError::config("BITSS: alpha and beta must be positive"));
        }
        if self.coefficient_interval == 0 || self.interpolation_points == 0 || self.max_stages == 0
        {
            return Err(MoptError::config(
                "BITSS: the interval of the coefficients, the number of points between the \
                 images, and the number of stages must be positive",
            ));
        }
        if !(self.min_separation.is_finite() && self.min_separation >= 0.0) {
            return Err(MoptError::config(
                "BITSS: the smallest separation must not be negative",
            ));
        }
        if self.midpoint_shift.is_some_and(|s| !positive(s))
            || self.step_share.is_some_and(|s| !positive(s))
        {
            return Err(MoptError::config(
                "BITSS: the shares of the midpoint shift and of the step must be positive",
            ));
        }
        if self.minimizer.saddle_order != 0 {
            return Err(MoptError::config(
                "BITSS minimizes its function; a saddle order does not apply to it",
            ));
        }
        Ok(())
    }
}
