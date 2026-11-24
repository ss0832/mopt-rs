//! GEDIIS: the hybrid of gradient-based and energy-based DIIS for geometry optimization (Li and
//! Frisch, J. Chem. Theory Comput. 2006, 2, 835).
//!
//! Far from the minimum (RMS gradient above `1e-2`) the step of the base method is kept as it
//! is; below that, while the last step is still large (RMS above `2.5e-3`), the energy-based
//! interpolation (EDIIS) is used, which is robust because it never extrapolates; and near
//! convergence the gradient-based extrapolation (GDIIS on the steps of the base method), which
//! converges fastest. Saddle searches, for which the energy-based interpolation is not defined,
//! use the gradient-based extrapolation below the gradient threshold.
//!
//! The two thresholds and the order of the three stages are those of the implementation of
//! the method in pysisyphus (`RFOptimizer`: `gediis_thresh = 1e-2` on the RMS force,
//! `gdiis_thresh = 2.5e-3` on the RMS step, neither far from the minimum); the text of the
//! publication was not available for the comparison.
//!
//! MultiOptPy's `GEDIIS` runs its `GDIIS` and `EDIIS` classes side by side and mixes their
//! outputs with weights from success counters; both inputs have the problems described for
//! those classes (see `docs/source/migration_optimizers.md`).
//!
//! MultiOptPy: `Optimizer/gediis.py:GEDIIS`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::diis_core::DiisHistory;
use super::ediis::ediis_step;
use super::gdiis::gdiis_step;
use super::StepEnhancer;
use crate::error::MoptResult;
use crate::math::linalg::col_rms;
use crate::opt::step::StepInput;

/// Which extrapolation GEDIIS uses at a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GediisPhase {
    /// The step of the base method as it is.
    Base,
    /// Gradient-based DIIS.
    Gradient,
    /// Energy-based DIIS.
    Energy,
}

/// GEDIIS step enhancer.
#[derive(Debug, Clone)]
pub struct Gediis {
    history: DiisHistory,
    /// RMS gradient above which the step of the base method is kept.
    pub gradient_threshold: f64,
    /// RMS of the last step below which the gradient-based extrapolation is used.
    pub step_threshold: f64,
    last_phase: Option<GediisPhase>,
}

impl Gediis {
    /// GEDIIS over at most `max_points` points.
    pub fn new(max_points: usize) -> Self {
        Self {
            history: DiisHistory::new(max_points),
            gradient_threshold: 1e-2,
            step_threshold: 2.5e-3,
            last_phase: None,
        }
    }

    /// Phase for an RMS gradient and the RMS of the last step.
    pub fn phase(
        &self,
        rms_gradient: f64,
        rms_last_step: Option<f64>,
        saddle_order: usize,
    ) -> GediisPhase {
        let small_step = rms_last_step.is_some_and(|s| s < self.step_threshold);
        if rms_gradient > self.gradient_threshold {
            GediisPhase::Base
        } else if saddle_order > 0 || small_step {
            GediisPhase::Gradient
        } else {
            GediisPhase::Energy
        }
    }

    /// Phase used in the last call.
    pub fn last_phase(&self) -> Option<GediisPhase> {
        self.last_phase
    }
}

impl Default for Gediis {
    fn default() -> Self {
        Self::new(5)
    }
}

impl StepEnhancer for Gediis {
    fn name(&self) -> &str {
        "gediis"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        self.history.record(input, &step);
        let phase = self.phase(
            col_rms(input.gradient),
            input.previous.map(|p| col_rms(&p.step)),
            input.saddle_order,
        );
        self.last_phase = Some(phase);
        let points = self.history.points();
        if points.len() < 2 {
            return Ok(step);
        }
        let new = match phase {
            GediisPhase::Base => None,
            GediisPhase::Gradient => gdiis_step(points, &step),
            GediisPhase::Energy => ediis_step(points, &step),
        };
        Ok(new.unwrap_or(step))
    }

    fn reset(&mut self) {
        self.history.clear();
        self.last_phase = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_follow_the_gradient_and_the_last_step() {
        let g = Gediis::default();
        assert_eq!(g.phase(5e-2, Some(0.1), 0), GediisPhase::Base);
        assert_eq!(g.phase(5e-2, Some(1e-3), 1), GediisPhase::Base);
        assert_eq!(g.phase(5e-3, Some(0.1), 0), GediisPhase::Energy);
        assert_eq!(g.phase(5e-3, None, 0), GediisPhase::Energy);
        assert_eq!(g.phase(5e-3, Some(1e-3), 0), GediisPhase::Gradient);
        assert_eq!(g.phase(5e-3, Some(0.1), 1), GediisPhase::Gradient);
    }
}
