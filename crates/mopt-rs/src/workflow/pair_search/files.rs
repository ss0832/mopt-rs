//! The files of `ieipmain` in MultiOptPy's formats: the frames of the trajectories (every
//! iteration appends its structures to them, see [`crate::workflow::trajectories`];
//! MultiOptPy writes a directory `samples_<stem>_<k>/` per iteration), the indices of the
//! extrema of an energy list, the plots of `Graph.single_plot`, and plain trajectories and
//! tables.
//!
//! MultiOptPy: `fileio.py:FileIO` (`make_psi4_input_file`, `make_traj_file_for_DM`,
//! `argrelextrema_txt_save`), `Visualization/visualization.py:Graph.single_plot`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;

use faer::Col;

use crate::core_infra::viz::{PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::MoptResult;

/// The rows of the atoms of `sys` as MultiOptPy writes its inputs (Angstrom, 12 decimals).
fn atom_rows(sys: &MolecularSystem, gap: &str) -> String {
    crate::io::xyz::multioptpy_rows(sys, gap)
}

/// One frame of the trajectory of an image of iEIP in the layout of MultiOptPy's
/// `make_traj_file_for_DM` (the number of atoms, the comment, the atoms in Angstrom).
pub fn image_frame(sys: &MolecularSystem, comment: &str) -> String {
    format!(
        "{}\n{}\n{}",
        sys.n_atoms(),
        comment_line(sys, comment),
        atom_rows(sys, "  ")
    )
}

/// The second line of a frame: `comment`, and before it the lattice of a structure in a
/// periodic cell (extended XYZ), without which the frame could not be read as what it is
/// (a path of a search in a cell would be taken for a path of clusters).
pub fn comment_line(sys: &MolecularSystem, comment: &str) -> String {
    match sys.active_cell() {
        Some(cell) => format!(
            "{} Properties=species:S:1:pos:R:3 {comment}",
            cell.to_extxyz_header()
        ),
        None => comment.to_string(),
    }
}

/// One frame with its comment (Angstrom, 12 decimals; MultiOptPy's trajectories of its
/// other methods).
pub fn frame_text(sys: &MolecularSystem, comment: &str) -> String {
    let mut text = format!("{}\n{}\n", sys.n_atoms(), comment_line(sys, comment));
    for i in 0..sys.n_atoms() {
        let p = sys.position(AtomId::new(i)).value();
        let _ = writeln!(
            text,
            "{} {:.12} {:.12} {:.12}",
            sys.elements[i].symbol(),
            p[0] * Units::BOHR_TO_ANGSTROM,
            p[1] * Units::BOHR_TO_ANGSTROM,
            p[2] * Units::BOHR_TO_ANGSTROM
        );
    }
    text
}

/// Frames with comments in one file (see [`frame_text`]).
pub fn write_frames(path: &Path, frames: &[(MolecularSystem, String)]) -> MoptResult<()> {
    let text: String = frames
        .iter()
        .map(|(sys, comment)| frame_text(sys, comment))
        .collect();
    std::fs::write(path, text)?;
    Ok(())
}

/// Indices of the strict local maxima (`maxima`) or minima of `values`, the ends excluded
/// (`scipy.signal.argrelextrema` with `order=1`).
pub fn local_extrema(values: &[f64], maxima: bool) -> Vec<usize> {
    (1..values.len().saturating_sub(1))
        .filter(|&i| {
            let (a, b, c) = (values[i - 1], values[i], values[i + 1]);
            if maxima {
                b > a && b > c
            } else {
                b < a && b < c
            }
        })
        .collect()
}

/// `name.txt` with one index per line (MultiOptPy `argrelextrema_txt_save`).
pub fn write_indices(dir: &Path, name: &str, indices: &[usize]) -> MoptResult<()> {
    let text: String = indices.iter().map(|i| format!("{i}\n")).collect();
    std::fs::write(dir.join(format!("{name}.txt")), text)?;
    Ok(())
}

/// MultiOptPy's `Graph.single_plot`: `plot_<name>_<title>.png`.
pub fn single_plot(
    dir: &Path,
    name: &str,
    title: &str,
    y_label: &str,
    values: &[f64],
) -> MoptResult<()> {
    let xs: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
    crate::core_infra::viz::save_single_line_plot(
        dir.join(format!("plot_{name}_{title}.png")),
        title,
        "ITR. ",
        y_label,
        &xs,
        values,
    )
}

/// A figure of several series against the iteration, as PNG and SVG (`<name>.png`,
/// `<name>.svg`), drawn anew after every iteration.
pub fn progress_figure(
    dir: &Path,
    name: &str,
    title: &str,
    y_label: &str,
    series: &[(&str, &[f64])],
) -> MoptResult<()> {
    let colors = [Rgb::BLUE, Rgb::RED, Rgb::EMERALD, Rgb::AMBER];
    let mut figure = PlotFigure::new(title, "Iteration", y_label);
    for (k, (label, ys)) in series.iter().enumerate() {
        figure.add_series(PlotSeries {
            label: (*label).to_string(),
            xs: (0..ys.len()).map(|i| i as f64).collect(),
            ys: ys.to_vec(),
            color: colors[k % colors.len()],
            style: PlotStyle::LinePoints,
        });
    }
    figure.save(dir.join(format!("{name}.png")))?;
    figure.save(dir.join(format!("{name}.svg")))
}

/// A table of rows of numbers with a header line.
pub fn write_table(path: &Path, header: &str, rows: &[Vec<f64>]) -> MoptResult<()> {
    let mut text = format!("{header}\n");
    for row in rows {
        let line: Vec<String> = row.iter().map(|v| format!("{v:.12e}")).collect();
        text.push_str(&line.join(","));
        text.push('\n');
    }
    std::fs::write(path, text)?;
    Ok(())
}

/// The points of a search of plain variables as a table (one row per point).
pub fn write_points(path: &Path, header: &str, points: &[(f64, &Col<f64>)]) -> MoptResult<()> {
    let rows: Vec<Vec<f64>> = points
        .iter()
        .map(|(e, x)| std::iter::once(*e).chain(x.iter().copied()).collect())
        .collect();
    write_table(path, header, &rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrema_as_scipy() {
        let v = [0.0, 1.0, 0.5, 0.5, 2.0, 1.0, 3.0];
        assert_eq!(local_extrema(&v, true), vec![1, 4]);
        assert_eq!(local_extrema(&v, false), vec![5]);
        assert!(local_extrema(&[1.0, 2.0], true).is_empty());
    }
}
