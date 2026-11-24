//! Differentiable minimum-image displacement.
//!
//! MultiOptPy: `Utils/pbc.py` (minimum-image displacements, here differentiable).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::scalar::Scalar;
use crate::domain::PeriodicCell;

/// Differentiable Minimum Image Convention displacement `r_i - r_j` over any `Scalar` (`f64`, `Dual`, `HyperDual`).
#[inline]
pub fn mic_delta_ad<T: Scalar>(r_i: [T; 3], r_j: [T; 3], cell: Option<&PeriodicCell>) -> [T; 3] {
    match cell {
        Some(c) if c.is_any_periodic() => {
            let shift = c.mic_lattice_shift_bohr(
                [r_i[0].val(), r_i[1].val(), r_i[2].val()],
                [r_j[0].val(), r_j[1].val(), r_j[2].val()],
            );
            [
                (r_i[0] - r_j[0]) + T::from_f64(shift[0]),
                (r_i[1] - r_j[1]) + T::from_f64(shift[1]),
                (r_i[2] - r_j[2]) + T::from_f64(shift[2]),
            ]
        }
        _ => [r_i[0] - r_j[0], r_i[1] - r_j[1], r_i[2] - r_j[2]],
    }
}
