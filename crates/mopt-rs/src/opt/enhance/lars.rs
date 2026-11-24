//! Layer-wise adaptive rate scaling (You et al., arXiv:1708.03888), as used by MultiOptPy's
//! `FIRELARS` default: the step is scaled by `w / (||step|| + beta w)` with
//! `w = clip(||x||, 0, 10)` and `x` the Cartesian coordinates (Bohr).
//!
//! MultiOptPy: `Optimizer/lars.py:LARS`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::StepEnhancer;
use crate::error::MoptResult;
use crate::math::linalg::col_norm;
use crate::opt::step::StepInput;

/// LARS step scaling.
#[derive(Debug, Clone, Copy)]
pub struct Lars {
    /// Damping `beta`.
    pub beta: f64,
}

impl Default for Lars {
    fn default() -> Self {
        Self { beta: 0.6 }
    }
}

impl StepEnhancer for Lars {
    fn name(&self) -> &str {
        "lars"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let w = col_norm(input.cartesian).clamp(0.0, 10.0);
        let denom = col_norm(&step) + w * self.beta;
        let scale = if denom > 0.0 { w / denom } else { 0.0 };
        Ok(step * scale)
    }
}
