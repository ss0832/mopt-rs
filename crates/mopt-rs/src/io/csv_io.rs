//! Numeric CSV table reader and writer for optimization profiles, eigenvalues, IRC curvature, and RCMC.
//!
//! MultiOptPy: `optimization.py:RunLogger.log_dynamic_csv`, `save_energy_profiles`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fs;
use std::path::Path;

use crate::error::{MoptError, MoptResult};

/// Write a CSV table with `headers` and row-major `rows` to `path`.
pub fn write_csv_table(
    path: impl AsRef<Path>,
    headers: &[&str],
    rows: &[Vec<f64>],
) -> MoptResult<()> {
    let mut out = String::new();
    if !headers.is_empty() {
        out.push_str(&headers.join(","));
        out.push('\n');
    }
    for row in rows {
        let formatted: Vec<String> = row.iter().map(|v| format!("{v:.12}")).collect();
        out.push_str(&formatted.join(","));
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

/// Read numeric columns from a CSV file (automatically skipping an initial non-numeric header row).
pub fn read_csv_numeric_columns(
    path: impl AsRef<Path>,
) -> MoptResult<(Vec<String>, Vec<Vec<f64>>)> {
    let content = fs::read_to_string(path)?;
    let mut headers = Vec::new();
    let mut rows = Vec::new();

    for (line_idx, line) in content.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let cells: Vec<&str> = t.split(',').map(str::trim).collect();
        if line_idx == 0 && cells.iter().any(|c| c.parse::<f64>().is_err()) {
            headers = cells.into_iter().map(str::to_string).collect();
            continue;
        }
        let mut row = Vec::with_capacity(cells.len());
        for c in cells {
            let val = c.parse::<f64>().map_err(|e| {
                MoptError::parse(format!(
                    "Invalid numeric CSV cell '{c}' on line {}: {e}",
                    line_idx + 1
                ))
            })?;
            row.push(val);
        }
        rows.push(row);
    }

    Ok((headers, rows))
}
