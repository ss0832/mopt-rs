//! The texts of the files of an IRC calculation: frames of structures, rows of tables, and
//! the files that combine the two sides of a transition state (the path, the table, and
//! the energy profile).
//!
//! The combined path runs from the end of the forward side through the starting structure
//! to the end of the backward side; the combined table runs the other way round (backward
//! from its end with negative steps, step 0, forward), as in MultiOptPy.
//!
//! MultiOptPy: `irc.py:IRC` (`combine_xyz_files`, `combine_csv_data`) and
//! `IRC/lqa.py:LQA.save_xyz_structure`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;

use faer::Col;

use super::irc_output::PathLayout;
use crate::core_infra::log::python_float;
use crate::domain::{MolecularSystem, Units};
use crate::irc::StepRecord;

/// The columns every table of an IRC starts with.
pub(super) const TABLE_COLUMNS: [&str; 5] = [
    "Step",
    "Energy (Hartree)",
    "Bias Energy (Hartree)",
    "RMS Gradient",
    "RMS Bias Gradient",
];

/// A field of a CSV row (quoted when it holds a comma, as Python's `csv` does).
pub(super) fn field(text: &str) -> String {
    if text.contains(',') || text.contains('"') {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// One frame of a structure file (Angstrom), in MultiOptPy's format; a periodic structure
/// also carries its cell after the comment (extended xyz), so that the frames of a path keep
/// the system they belong to.
pub(super) fn structure_frame(atoms: &MolecularSystem, x: &Col<f64>, comment: &str) -> String {
    let comment = match atoms.cell.as_ref().filter(|c| c.is_any_periodic()) {
        Some(cell) => format!("{comment} {}", cell.to_extxyz_header())
            .trim()
            .to_string(),
        None => comment.to_string(),
    };
    let mut text = format!("{}\n{comment}\n", atoms.n_atoms());
    for (i, element) in atoms.elements.iter().enumerate() {
        let _ = writeln!(
            text,
            "{:<3} {:17.12} {:17.12} {:17.12}",
            element.symbol(),
            x[3 * i] * Units::BOHR_TO_ANGSTROM,
            x[3 * i + 1] * Units::BOHR_TO_ANGSTROM,
            x[3 * i + 2] * Units::BOHR_TO_ANGSTROM
        );
    }
    text
}

/// A row of a file of plain variables: a label and the values.
pub(super) fn variables_row(label: &str, x: &Col<f64>) -> String {
    let values: Vec<String> = x.iter().map(|v| python_float(*v)).collect();
    format!("{label},{}\n", values.join(","))
}

/// A point of the path in the layout of the calculation.
pub(super) fn point_text(layout: &PathLayout, x: &Col<f64>, label: &str, comment: &str) -> String {
    match layout {
        PathLayout::Atoms(atoms) => structure_frame(atoms, x, comment),
        PathLayout::Variables(_) => variables_row(label, x),
    }
}

/// The starting structure with its values.
pub(super) struct StartPoint<'a> {
    pub x: &'a Col<f64>,
    pub energy: f64,
    pub biased_energy: f64,
    pub rms_gradient: f64,
    pub rms_biased_gradient: f64,
}

/// One side: its points, the rows of its table, and the mass-weighted distance of its
/// first point from the start.
pub(super) struct SideData<'a> {
    pub records: &'a [StepRecord],
    pub rows: &'a [Vec<String>],
    pub first_length: f64,
}

/// The texts of the combined path and of the energy profile over the signed length of the
/// path (one row per frame of the path).
pub(super) fn combined_path(
    layout: &PathLayout,
    start: &StartPoint<'_>,
    forward: &SideData<'_>,
    backward: &SideData<'_>,
) -> (String, String) {
    let mut path = match layout {
        PathLayout::Atoms(_) => String::new(),
        PathLayout::Variables(names) => format!("Frame,{}\n", names.join(",")),
    };
    let mut profile = "Frame,Arc Length,Energy (Hartree),Bias Energy (Hartree)\n".to_string();
    let mut frame = 0_usize;
    let mut add = |x: &Col<f64>, s: f64, energy: f64, biased: f64, comment: &str| {
        path.push_str(&point_text(layout, x, &frame.to_string(), comment));
        let _ = writeln!(
            profile,
            "{frame},{},{},{}",
            python_float(s),
            python_float(energy),
            python_float(biased)
        );
        frame += 1;
    };
    for r in forward.records.iter().rev() {
        let s = forward.first_length + r.arc_length;
        add(&r.x, s, r.energy, r.biased_energy, "");
    }
    add(start.x, 0.0, start.energy, start.biased_energy, "TS");
    for r in backward.records {
        let s = -(backward.first_length + r.arc_length);
        add(&r.x, s, r.energy, r.biased_energy, "");
    }
    (path, profile)
}

/// The text of the combined table.
pub(super) fn combined_table(
    start: &StartPoint<'_>,
    forward: &SideData<'_>,
    backward: &SideData<'_>,
) -> String {
    let mut table = TABLE_COLUMNS.join(",") + "\n";
    for row in backward.rows.iter().rev() {
        // MultiOptPy: `-int(step)` and the first five columns.
        let sign = if row[0] == "0" { "" } else { "-" };
        let _ = writeln!(table, "{sign}{}", row[..5].join(","));
    }
    let _ = writeln!(
        table,
        "0,{},{},{},{}",
        python_float(start.energy),
        python_float(start.biased_energy),
        python_float(start.rms_gradient),
        python_float(start.rms_biased_gradient)
    );
    for row in forward.rows {
        let fields: Vec<String> = row.iter().map(|f| field(f)).collect();
        let _ = writeln!(table, "{}", fields.join(","));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(step: usize, x: f64, arc: f64, energy: f64) -> StepRecord {
        StepRecord {
            step,
            x: Col::from_fn(1, |_| x),
            energy,
            biased_energy: energy,
            rms_gradient: 0.0,
            rms_biased_gradient: 0.0,
            arc_length: arc,
            bending_angle: None,
            extras: Vec::new(),
            curvature: None,
            in_table: true,
            notes: Vec::new(),
            curvature_across: None,
        }
    }

    fn row(step: usize, energy: f64) -> Vec<String> {
        let e = python_float(energy);
        vec![
            step.to_string(),
            e.clone(),
            e,
            "0.0".into(),
            "0.0".into(),
            "extra,1".into(),
        ]
    }

    #[test]
    fn the_path_and_the_table_run_in_multioptpy_s_orders() {
        let layout = PathLayout::Variables(vec!["x".to_string()]);
        let x0 = Col::from_fn(1, |_| 0.0);
        let start = StartPoint {
            x: &x0,
            energy: 1.0,
            biased_energy: 1.0,
            rms_gradient: 0.0,
            rms_biased_gradient: 0.0,
        };
        let forward_records = [record(1, 0.1, 0.0, 0.9), record(2, 0.3, 0.2, 0.5)];
        let forward_rows = [row(1, 0.9), row(2, 0.5)];
        let backward_records = [record(0, -0.1, 0.0, 0.8), record(1, -0.4, 0.3, 0.2)];
        let backward_rows = [row(0, 0.8), row(1, 0.2)];
        let forward = SideData {
            records: &forward_records,
            rows: &forward_rows,
            first_length: 0.1,
        };
        let backward = SideData {
            records: &backward_records,
            rows: &backward_rows,
            first_length: 0.1,
        };
        let (path, profile) = combined_path(&layout, &start, &forward, &backward);
        assert_eq!(path, "Frame,x\n0,0.3\n1,0.1\n2,0.0\n3,-0.1\n4,-0.4\n");
        let lengths: Vec<&str> = profile
            .lines()
            .skip(1)
            .map(|l| l.split(',').nth(1).unwrap_or(""))
            .collect();
        assert_eq!(lengths[2], "0.0");
        assert!(lengths[0].starts_with("0.3") && lengths[4] == "-0.4");
        let table = combined_table(&start, &forward, &backward);
        let steps: Vec<&str> = table
            .lines()
            .skip(1)
            .map(|l| l.split(',').next().unwrap_or(""))
            .collect();
        // Backward from its end with negative steps (a step 0 of HPC stays 0), the start,
        // forward with all its columns (a field with a comma is quoted).
        assert_eq!(steps, vec!["-1", "0", "0", "1", "2"]);
        assert!(table
            .lines()
            .nth(1)
            .is_some_and(|l| l.split(',').count() == 5));
        assert!(table.ends_with(",\"extra,1\"\n"));
    }
}
