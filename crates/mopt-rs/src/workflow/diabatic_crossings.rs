//! `diabatic_crossings.csv` of `freqmain -diabatize`: the crossings of two states that the
//! run found, one row each, in the form the kinetics reads (`diabatic_crossings` in the
//! annotations of `run_kinetics` and `run_rcmc`), so that a coupling need not be typed in.
//!
//! Columns: `number` (from 1), `kind`, `state_1`, `state_2` (the numbers of the states at
//! the first frame), `position` (amu^(1/2) Bohr), `energy_hartree`, `coupling_hartree`,
//! `slope_1`, `slope_2` (Hartree per amu^(1/2) Bohr), `slope_difference`, `slope_product`,
//! `reduced_mass_amu` (1, since the path is mass-weighted). The kinds:
//!
//! - `crossing`: two diabatic states cross (the coupling and the slopes of the diabatic
//!   potential matrix there);
//! - `gap`: the straight two-state model fitted to the smallest gap of two neighbouring
//!   adiabatic states;
//! - `zhu_nakamura`: Zhu and Nakamura's two-state model from the adiabatic energies: for a
//!   sloped crossing at the smallest gap and where they reach the crossing energy
//!   ([`crate::kinetics::rate::adiabatic_model`]), for a peaked one at the top of the lower
//!   energy and the bottom of the upper one ([`crate::kinetics::rate::peaked_model`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{MoptError, MoptResult};

/// The header of the file.
pub const HEADER: &str = "number,kind,state_1,state_2,position,energy_hartree,coupling_hartree,\
                          slope_1,slope_2,slope_difference,slope_product,reduced_mass_amu";

/// One crossing of the file.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossingRow {
    /// `crossing`, `gap` or `zhu_nakamura`.
    pub kind: String,
    /// The numbers of the two states at the first frame.
    pub states: [usize; 2],
    /// Where on the path (amu^(1/2) Bohr).
    pub position: f64,
    /// The crossing energy (Hartree).
    pub energy: f64,
    /// The coupling (Hartree).
    pub coupling: f64,
    /// The slopes of the two states (Hartree per amu^(1/2) Bohr).
    pub slopes: [f64; 2],
}

impl CrossingRow {
    /// `|slope_1 - slope_2|`.
    pub fn slope_difference(&self) -> f64 {
        (self.slopes[0] - self.slopes[1]).abs()
    }

    /// `slope_1 slope_2`: positive for a sloped crossing, negative for a peaked one.
    pub fn slope_product(&self) -> f64 {
        self.slopes[0] * self.slopes[1]
    }
}

/// The slopes `[F1, F2]` with `F1 F2 = mean^2`, `F1 - F2 = difference` (positive), and the
/// sign of `mean`.
pub fn slopes_of(mean: f64, difference: f64) -> [f64; 2] {
    let root = (difference * difference + 4.0 * mean * mean).sqrt();
    let first = 0.5 * (difference + mean.signum() * root);
    [first, first - difference]
}

/// Write `rows` to `path`.
pub fn write(path: &Path, rows: &[CrossingRow]) -> MoptResult<()> {
    // Shortest representations that read back to the same numbers.
    let mut text = format!("{HEADER}\n");
    for (k, r) in rows.iter().enumerate() {
        let _ = writeln!(
            text,
            "{},{},{},{},{},{},{:e},{:e},{:e},{:e},{:e},1",
            k + 1,
            r.kind,
            r.states[0],
            r.states[1],
            r.position,
            r.energy,
            r.coupling,
            r.slopes[0],
            r.slopes[1],
            r.slope_difference(),
            r.slope_product()
        );
    }
    std::fs::write(path, text).map_err(|e| MoptError::from(e).context(path.display()))
}

/// The rows of `path`.
pub fn read(path: &Path) -> MoptResult<Vec<CrossingRow>> {
    let text =
        std::fs::read_to_string(path).map_err(|e| MoptError::from(e).context(path.display()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| MoptError::parse(format!("{}: empty", path.display())))?
        .split(',')
        .map(str::trim)
        .collect();
    let col = |name: &str| {
        header.iter().position(|h| *h == name).ok_or_else(|| {
            MoptError::parse(format!(
                "{}: not a diabatic_crossings.csv of freqmain -diabatize (no column '{name}')",
                path.display()
            ))
        })
    };
    let kind = col("kind")?;
    let state_cols = [col("state_1")?, col("state_2")?];
    let position = col("position")?;
    let energy = col("energy_hartree")?;
    let coupling = col("coupling_hartree")?;
    let slope_cols = [col("slope_1")?, col("slope_2")?];
    let mut out = Vec::new();
    for line in lines {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        let bad = || MoptError::parse(format!("{}: bad line '{line}'", path.display()));
        let cell = |k: usize| cells.get(k).copied().ok_or_else(bad);
        let num = |k: usize| -> MoptResult<f64> { cell(k)?.parse::<f64>().map_err(|_| bad()) };
        let int = |k: usize| -> MoptResult<usize> { cell(k)?.parse::<usize>().map_err(|_| bad()) };
        out.push(CrossingRow {
            kind: cell(kind)?.to_string(),
            states: [int(state_cols[0])?, int(state_cols[1])?],
            position: num(position)?,
            energy: num(energy)?,
            coupling: num(coupling)?,
            slopes: [num(slope_cols[0])?, num(slope_cols[1])?],
        });
    }
    Ok(out)
}

/// The row `number` (from 1) of `path`; with `None`, the only row of the file.
pub fn row(path: &Path, number: Option<usize>) -> MoptResult<CrossingRow> {
    let rows = read(path)?;
    let listed = || {
        rows.iter()
            .enumerate()
            .map(|(k, r)| {
                format!(
                    "{} ({} of the states {} and {})",
                    k + 1,
                    r.kind,
                    r.states[0],
                    r.states[1]
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    match number {
        None if rows.len() == 1 => Ok(rows[0].clone()),
        None => Err(MoptError::config(format!(
            "{}: {} crossings; choose one with diabatic_crossing (from 1): {}",
            path.display(),
            rows.len(),
            listed()
        ))),
        Some(n) => rows.get(n.wrapping_sub(1)).cloned().ok_or_else(|| {
            MoptError::config(format!(
                "{}: no crossing {n} (the crossings: {})",
                path.display(),
                listed()
            ))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_read_back_as_written() {
        let dir = std::env::temp_dir().join(format!("mopt_crossings_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diabatic_crossings.csv");
        let rows = vec![
            CrossingRow {
                kind: "crossing".into(),
                states: [1, 2],
                position: 1.25,
                energy: -75.5,
                coupling: 1.5e-3,
                slopes: [0.02, -0.01],
            },
            CrossingRow {
                kind: "zhu_nakamura".into(),
                states: [1, 2],
                position: 1.3,
                energy: -75.49,
                coupling: 1.4e-3,
                slopes: slopes_of(-0.012, 0.03),
            },
        ];
        write(&path, &rows).unwrap();
        let back = read(&path).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].kind, "crossing");
        assert!((back[0].slope_product() + 2e-4).abs() < 1e-14);
        let zn = &back[1];
        assert!((zn.slope_product() - 0.012f64.powi(2)).abs() < 1e-12);
        assert!((zn.slope_difference() - 0.03).abs() < 1e-12);
        assert!(zn.slopes.iter().all(|s| *s < 0.0));
        assert!(row(&path, None).is_err());
        assert_eq!(row(&path, Some(2)).unwrap().kind, "zhu_nakamura");
        assert!(row(&path, Some(3)).is_err() && row(&path, Some(0)).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
