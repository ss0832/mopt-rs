//! The results of the points of a scan and their tables: the state of the scan that a restart
//! reads (`scan_state.json`), MultiOptPy's energy profile (`<stem>_scan_energy_profile_<date>.csv`),
//! and the table of every point (`scan_points.csv`).
//!
//! MultiOptPy: `Entrypoints/relaxed_scan.py:run_relaxedscan` (the energy profile: a header
//! with the kinds of the coordinates, one with their atoms and the units, then the values,
//! the energy, and the energy with the bias potentials of each point).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use super::grid::ScanGrid;
use crate::core_infra::json::JsonValue;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult, ResultExt};

/// File of the state of a scan in its directory.
pub const STATE_FILE: &str = "scan_state.json";
/// File of the table of all points.
pub const POINTS_FILE: &str = "scan_points.csv";

/// The result of one point of a scan.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanPoint {
    /// The grid point.
    pub point: usize,
    /// Energy of the surface without bias potentials (Hartree; NaN for a skipped point).
    pub energy: f64,
    /// Energy with them.
    pub biased: f64,
    /// Whether its optimization converged (true for single points).
    pub converged: bool,
    /// Iterations of its optimization (1 for single points).
    pub iterations: usize,
    /// The point its start structure came from (`None`: the input or a pre-image).
    pub start_from: Option<usize>,
    /// The structure, relative to the directory of the scan.
    pub file: String,
    /// A grid point of ReSPer coordinates beyond the references.
    pub extrapolated: bool,
    /// Left out (an extrapolated point with `resper_outside=skip`, or a failed one).
    pub skipped: bool,
    /// Its calculation failed (the constraints could not be restored, the calculator
    /// failed); the scan went on without it.
    pub failed: bool,
    /// Distance in the map between the point and its interpolated structure (rigid ReSPer).
    pub preimage_error: Option<f64>,
    /// Optimized again from a neighbour on a lower branch.
    pub repaired: bool,
}

fn number(v: f64) -> JsonValue {
    if v.is_finite() {
        JsonValue::Number(v)
    } else {
        JsonValue::Null
    }
}

impl ScanPoint {
    fn to_json(&self) -> JsonValue {
        let mut o = BTreeMap::new();
        o.insert("point".into(), JsonValue::Number(self.point as f64));
        o.insert("energy".into(), number(self.energy));
        o.insert("biased_energy".into(), number(self.biased));
        o.insert("converged".into(), JsonValue::Bool(self.converged));
        o.insert(
            "iterations".into(),
            JsonValue::Number(self.iterations as f64),
        );
        o.insert(
            "start_from".into(),
            self.start_from
                .map_or(JsonValue::Null, |p| JsonValue::Number(p as f64)),
        );
        o.insert("file".into(), JsonValue::String(self.file.clone()));
        o.insert("extrapolated".into(), JsonValue::Bool(self.extrapolated));
        o.insert("skipped".into(), JsonValue::Bool(self.skipped));
        o.insert("failed".into(), JsonValue::Bool(self.failed));
        o.insert(
            "preimage_error".into(),
            self.preimage_error.map_or(JsonValue::Null, number),
        );
        o.insert("repaired".into(), JsonValue::Bool(self.repaired));
        JsonValue::Object(o)
    }

    fn from_json(v: &JsonValue) -> MoptResult<Self> {
        let num = |key: &str| v.get(key).and_then(JsonValue::as_f64).unwrap_or(f64::NAN);
        let flag = |key: &str| v.get(key).and_then(JsonValue::as_bool).unwrap_or(false);
        Ok(Self {
            point: v
                .get("point")
                .and_then(JsonValue::as_usize)
                .ok_or_else(|| MoptError::parse("a point of the scan state has no number"))?,
            energy: num("energy"),
            biased: num("biased_energy"),
            converged: flag("converged"),
            iterations: v
                .get("iterations")
                .and_then(JsonValue::as_usize)
                .unwrap_or(0),
            start_from: v.get("start_from").and_then(JsonValue::as_usize),
            file: v
                .get("file")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_string(),
            extrapolated: flag("extrapolated"),
            skipped: flag("skipped"),
            failed: flag("failed"),
            preimage_error: v.get("preimage_error").and_then(JsonValue::as_f64),
            repaired: flag("repaired"),
        })
    }
}

/// What a restart needs: the points done, in the order they were done, and the name of the
/// energy profile.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScanState {
    /// Name of MultiOptPy's energy profile in the directory.
    pub profile: String,
    /// The points done, in order.
    pub done: Vec<ScanPoint>,
}

impl ScanState {
    /// Write to `dir/scan_state.json`.
    pub fn save(&self, dir: &Path) -> MoptResult<()> {
        let mut o = BTreeMap::new();
        o.insert("profile".into(), JsonValue::String(self.profile.clone()));
        o.insert(
            "done".into(),
            JsonValue::Array(self.done.iter().map(ScanPoint::to_json).collect()),
        );
        let text = JsonValue::Object(o).to_string_pretty() + "\n";
        // Written whole and then moved, so that an interruption leaves the old state.
        let tmp = dir.join(format!("{STATE_FILE}.tmp"));
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, dir.join(STATE_FILE))?;
        Ok(())
    }

    /// Read `dir/scan_state.json` (`None` when there is none).
    pub fn load(dir: &Path) -> MoptResult<Option<Self>> {
        let path = dir.join(STATE_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let doc = JsonValue::parse(&std::fs::read_to_string(&path)?).context(path.display())?;
        let done = doc
            .get("done")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .map(ScanPoint::from_json)
            .collect::<MoptResult<_>>()?;
        Ok(Some(Self {
            profile: doc
                .get("profile")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_string(),
            done,
        }))
    }

    /// The result of grid point `point`, if done.
    pub fn result(&self, point: usize) -> Option<&ScanPoint> {
        self.done.iter().rev().find(|p| p.point == point)
    }

    /// Record (or replace) the result of a point.
    pub fn record(&mut self, result: ScanPoint) {
        match self.done.iter().position(|p| p.point == result.point) {
            Some(k) => self.done[k] = result,
            None => self.done.push(result),
        }
    }

    /// The energies of all grid points (NaN where missing), the biased ones with `biased`.
    pub fn energies(&self, grid: &ScanGrid, biased: bool) -> Vec<f64> {
        let mut e = vec![f64::NAN; grid.len()];
        for p in &self.done {
            if !p.skipped && p.point < e.len() {
                e[p.point] = if biased { p.biased } else { p.energy };
            }
        }
        e
    }
}

/// MultiOptPy's energy profile: the kinds of the coordinates, their atoms (commas as `_`)
/// with the units of the energies, then per point in the order of the scan its values, energy,
/// and energy with the bias potentials (Hartree).
pub fn write_profile(
    path: &Path,
    kinds: &[String],
    atoms: &[String],
    grid: &ScanGrid,
    state: &ScanState,
) -> MoptResult<()> {
    let mut text = format!("{},energy,bias_energy\n", kinds.join(","));
    let atoms: Vec<String> = atoms.iter().map(|a| a.replace(',', "_")).collect();
    let _ = writeln!(text, "{},(hartree),(hartree)", atoms.join(","));
    for &point in grid.order() {
        let Some(p) = state.result(point).filter(|p| !p.skipped) else {
            continue;
        };
        let values: Vec<String> = grid.values(point).iter().map(|v| v.to_string()).collect();
        let _ = writeln!(text, "{},{},{}", values.join(","), p.energy, p.biased);
    }
    std::fs::write(path, text)?;
    Ok(())
}

/// The table of all points: grid indices, values, energies (Hartree, without and with the
/// bias potentials, and each in kcal/mol above the lowest point of its own surface:
/// `relative_bare_kcal_mol`, `relative_biased_kcal_mol`), how the point went, and its
/// structure.
pub fn write_points(path: &Path, grid: &ScanGrid, state: &ScanState) -> MoptResult<()> {
    let lowest_of = |energy: fn(&ScanPoint) -> f64| {
        state
            .done
            .iter()
            .filter(|p| !p.skipped && energy(p).is_finite())
            .map(energy)
            .fold(f64::INFINITY, f64::min)
    };
    // Each relative energy against the lowest point of its own surface: the bare one
    // (`energy_hartree`, the calculator alone) and the biased one (`biased_energy_hartree`,
    // the surface the points were minimized on, on which the candidates are found).
    let (lowest, lowest_biased) = (lowest_of(|p| p.energy), lowest_of(|p| p.biased));
    let relative = |e: f64, low: f64| {
        if e.is_finite() && low.is_finite() {
            format!("{:.6}", (e - low) * Units::HARTREE_TO_KCAL_MOL)
        } else {
            String::new()
        }
    };
    let mut text = String::from("point");
    for d in 0..grid.shape().len() {
        let _ = write!(text, ",index_{}", d + 1);
    }
    for axis in grid.axes() {
        let _ = write!(text, ",{}", axis.axis_title().replace(',', ";"));
    }
    text.push_str(
        ",energy_hartree,biased_energy_hartree,relative_bare_kcal_mol,relative_biased_kcal_mol,\
         converged,iterations,start_from,extrapolated,skipped,failed,preimage_error,repaired,file\n",
    );
    for point in 0..grid.len() {
        let Some(p) = state.result(point) else {
            continue;
        };
        let _ = write!(text, "{}", point + 1);
        for i in grid.indices(point) {
            let _ = write!(text, ",{}", i + 1);
        }
        for v in grid.values(point) {
            let _ = write!(text, ",{v:.10}");
        }
        let _ = writeln!(
            text,
            ",{:.12},{:.12},{},{},{},{},{},{},{},{},{},{},{}",
            p.energy,
            p.biased,
            relative(p.energy, lowest),
            relative(p.biased, lowest_biased),
            p.converged,
            p.iterations,
            p.start_from
                .map_or_else(String::new, |s| (s + 1).to_string()),
            p.extrapolated,
            p.skipped,
            p.failed,
            p.preimage_error
                .map_or_else(String::new, |e| format!("{e:.6e}")),
            p.repaired,
            p.file
        );
    }
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::workflow::scan::grid::ScanAxis;

    #[test]
    fn the_state_is_read_back_and_the_tables_are_written() {
        let dir = std::env::temp_dir().join(format!("mopt_scan_output_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let grid = ScanGrid::joint(vec![ScanAxis::linspace(
            "bond(1,2)",
            "Angstrom",
            1.0,
            2.0,
            3,
        )
        .unwrap()])
        .unwrap();
        let mut state = ScanState {
            profile: "x_scan_energy_profile_2026.csv".into(),
            done: Vec::new(),
        };
        for point in [0, 1] {
            state.record(ScanPoint {
                point,
                energy: -1.0 - 0.1 * point as f64,
                biased: -1.0,
                converged: true,
                iterations: 5,
                start_from: point.checked_sub(1),
                file: format!("x_RelaxedScan_{}.xyz", point + 1),
                extrapolated: false,
                skipped: false,
                failed: false,
                preimage_error: None,
                repaired: false,
            });
        }
        state.save(&dir).unwrap();
        let read = ScanState::load(&dir).unwrap().unwrap();
        assert_eq!(read, state);
        assert!(read.energies(&grid, false)[2].is_nan());
        write_profile(
            &dir.join("profile.csv"),
            &["bond".into()],
            &["1,2".into()],
            &grid,
            &read,
        )
        .unwrap();
        let profile = std::fs::read_to_string(dir.join("profile.csv")).unwrap();
        let lines: Vec<&str> = profile.lines().collect();
        assert_eq!(lines[0], "bond,energy,bias_energy");
        assert_eq!(lines[1], "1_2,(hartree),(hartree)");
        assert_eq!(lines.len(), 4);
        write_points(&dir.join(POINTS_FILE), &grid, &read).unwrap();
        let table = std::fs::read_to_string(dir.join(POINTS_FILE)).unwrap();
        // The first point is 0.1 Hartree above the second, the lowest.
        assert!(table.lines().nth(1).unwrap().contains(",62.75"), "{table}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
