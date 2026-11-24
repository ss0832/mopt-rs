//! When a side of the path has arrived at a stationary point.
//!
//! MultiOptPy takes `abs(grad.max())` for the largest force: the largest component with its
//! sign, so that a gradient whose components are all negative or zero passes however large
//! they are. Here the largest absolute component is tested (`docs/source/migration_paths.md`).
//!
//! MultiOptPy: `IRC/converge_criteria.py:convergence_check`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::types::{max_abs, rms};

/// Whether the largest absolute component of `gradient` is below `max_force` and its RMS
/// below `rms_force`.
pub fn convergence_check(gradient: &Col<f64>, max_force: f64, rms_force: f64) -> bool {
    max_abs(gradient) < max_force && rms(gradient) < rms_force
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MultiOptPy's test as it is written.
    fn multioptpy(gradient: &Col<f64>, max_force: f64, rms_force: f64) -> bool {
        let largest = gradient.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        largest.abs() < max_force && rms(gradient) < rms_force
    }

    #[test]
    fn a_large_negative_component_is_not_converged() {
        // One component of -3e-4 among zeros: RMS 9.5e-5 passes, and MultiOptPy's largest
        // component is 0, although the largest force is 7.5 times the threshold of 4e-5.
        let g = Col::from_fn(10, |k| if k == 3 { -3e-4 } else { 0.0 });
        assert!(multioptpy(&g, 4e-5, 1e-4));
        assert!(!convergence_check(&g, 4e-5, 1e-4));
        // Where the largest component is positive the two agree.
        let g = Col::from_fn(10, |k| if k == 3 { 3e-5 } else { -1e-5 });
        assert!(multioptpy(&g, 4e-5, 1e-4));
        assert!(convergence_check(&g, 4e-5, 1e-4));
    }
}
