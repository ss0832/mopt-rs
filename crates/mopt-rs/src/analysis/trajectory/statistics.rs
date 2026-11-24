//! Statistics of a time series with the error of its mean from block averages.
//!
//! The samples of a molecular dynamics run are correlated, so that the standard error
//! `s / sqrt(n)` of independent samples is too small. The blocking transformation of
//! Flyvbjerg and Petersen (J. Chem. Phys. 1989, 91, 461) averages neighbouring pairs over
//! and over; the standard error of the blocks grows until the blocks are longer than the
//! correlation time and then stays (within its own error `se / sqrt(2 (m - 1))` for `m`
//! blocks). The level taken is the first one with `B^3 > 2 n (se_B / se_1)^4` (`B` the
//! number of samples in a block), the criterion of Lee, Kim, Morales, Needs, and Rios
//! (Phys. Rev. E 2011, 83, 066706, as in their program pyblock). The statistical
//! inefficiency `g = (se_B / se_1)^2` is the number of steps between independent samples,
//! the integrated correlation time `(g - 1) / 2` steps.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// Fewest blocks a level may have to be taken when no level meets the criterion.
const FEWEST_BLOCKS: usize = 4;

/// One level of the blocking transformation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockLevel {
    /// Samples in a block.
    pub block_size: usize,
    /// Number of blocks.
    pub blocks: usize,
    /// Standard error of the mean from the blocks.
    pub error: f64,
    /// The error of that error, `error / sqrt(2 (blocks - 1))`.
    pub error_of_error: f64,
}

/// Statistics of a series.
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    /// Number of samples.
    pub count: usize,
    /// Mean.
    pub mean: f64,
    /// Standard deviation of the samples (with `n - 1`).
    pub deviation: f64,
    /// Smallest sample.
    pub min: f64,
    /// Largest sample.
    pub max: f64,
    /// Standard error of the mean as if the samples were independent.
    pub naive_error: f64,
    /// Standard error of the mean from the block averages.
    pub error: f64,
    /// Samples in a block of the level taken.
    pub block_size: usize,
    /// Statistical inefficiency `(error / naive_error)^2`.
    pub inefficiency: f64,
    /// Whether a level met the criterion; otherwise the series is too short for its
    /// correlation time and `error` is the largest one of the levels with at least four
    /// blocks (a lower bound).
    pub converged: bool,
    /// All levels.
    pub levels: Vec<BlockLevel>,
}

fn level_of(values: &[f64], block_size: usize) -> Option<BlockLevel> {
    let m = values.len();
    if m < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / m as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
    let error = libm::sqrt(variance / m as f64);
    Some(BlockLevel {
        block_size,
        blocks: m,
        error,
        error_of_error: error / libm::sqrt(2.0 * (m - 1) as f64),
    })
}

/// The statistics of `values` (at least two finite samples).
pub fn statistics(values: &[f64]) -> MoptResult<Statistics> {
    let n = values.len();
    if n < 2 {
        return Err(MoptError::config("statistics need at least two samples"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(MoptError::config("a sample is not finite"));
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut levels = Vec::new();
    let mut blocks = values.to_vec();
    let mut size = 1;
    while let Some(level) = level_of(&blocks, size) {
        levels.push(level);
        blocks = blocks
            .chunks_exact(2)
            .map(|p| 0.5 * (p[0] + p[1]))
            .collect();
        size *= 2;
    }
    let first = levels[0];
    let naive = first.error;
    let deviation = naive * libm::sqrt(n as f64);
    // The criterion of Lee et al.; a series that does not vary has every error zero.
    let chosen = if naive > 0.0 {
        levels.iter().position(|l| {
            (l.block_size as f64).powi(3) > 2.0 * n as f64 * (l.error / naive).powi(4)
        })
    } else {
        Some(0)
    };
    let (error, block_size, converged) = match chosen {
        Some(k) => (levels[k].error, levels[k].block_size, true),
        None => {
            let widest = levels
                .iter()
                .filter(|l| l.blocks >= FEWEST_BLOCKS)
                .max_by(|a, b| a.error.total_cmp(&b.error))
                .copied()
                .unwrap_or(first);
            (widest.error, widest.block_size, false)
        }
    };
    let inefficiency = if naive > 0.0 {
        (error / naive).powi(2)
    } else {
        1.0
    };
    Ok(Statistics {
        count: n,
        mean,
        deviation,
        min,
        max,
        naive_error: naive,
        error,
        block_size,
        inefficiency,
        converged,
        levels,
    })
}
