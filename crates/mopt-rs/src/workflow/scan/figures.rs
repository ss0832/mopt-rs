//! Figures of a scan, as PNG and SVG with axis titles, legends, and colour bars: the energy
//! along a curve, the energy map of two coordinates (the grid interpolated bilinearly between
//! its points, missing points grey, the transition-state candidates and minima marked, a
//! reference path on top), and the curves of a map along its first coordinate.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `relaxedscan` writes no figure).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::candidates::{Candidate, CandidateKind};
use super::grid::{ScanAxis, ScanGrid};
use crate::core_infra::viz::color::Rgb;
use crate::core_infra::viz::figure::{ColorGrid, PlotFigure, PlotSeries, PlotStyle};
use crate::error::MoptResult;

/// Cells of the drawn map per step of the grid (at most [`MAX_CELLS`] along an axis).
const CELLS_PER_STEP: usize = 8;
/// Largest number of cells along an axis of a map.
const MAX_CELLS: usize = 400;
/// Largest number of curves drawn in one figure of the rows of a map.
const MAX_ROWS: usize = 12;

/// A line or points drawn on a figure (a reference path, for example).
#[derive(Debug, Clone)]
pub struct Overlay {
    /// Legend entry.
    pub label: String,
    /// Horizontal positions.
    pub xs: Vec<f64>,
    /// Vertical positions.
    pub ys: Vec<f64>,
    /// Color.
    pub color: Rgb,
    /// Line with points or points alone.
    pub style: PlotStyle,
}

/// Save `figure` as `<stem>.png` and `<stem>.svg` in `dir`.
pub fn save(figure: &PlotFigure, dir: &Path, stem: &str) -> MoptResult<()> {
    figure.save(dir.join(format!("{stem}.png")))?;
    figure.save(dir.join(format!("{stem}.svg")))
}

/// Where a candidate lies along `axis` (its grid value moved by its offset in steps).
fn at(axis: &ScanAxis, index: usize, offset: f64) -> f64 {
    let v = &axis.values;
    let step = if v.len() > 1 {
        (v[v.len() - 1] - v[0]) / (v.len() - 1) as f64
    } else {
        0.0
    };
    v[index] + offset * step
}

/// The candidates as points: transition-state candidates in magenta, minima in `minima` (black
/// on a curve, white on a map, where black is lost in the dark blue of the lowest energies).
fn markers(
    figure: &mut PlotFigure,
    candidates: &[Candidate],
    minima: Rgb,
    position: impl Fn(&Candidate) -> (f64, f64),
) {
    for (kind, label, color) in [
        (
            CandidateKind::Saddle,
            "transition-state candidates",
            Rgb::MAGENTA,
        ),
        (CandidateKind::Minimum, "minima", minima),
    ] {
        let (xs, ys): (Vec<f64>, Vec<f64>) = candidates
            .iter()
            .filter(|c| c.kind == kind)
            .map(&position)
            .unzip();
        if !xs.is_empty() {
            figure.add_series(PlotSeries {
                label: label.to_string(),
                xs,
                ys,
                color,
                style: PlotStyle::Scatter,
            });
        }
    }
}

/// The energy along a curve: `energies` (kcal/mol above the lowest point, in the order of the
/// points of the grid) against the first coordinate, with the energies with the bias
/// potentials when they differ, and the candidates at their grid points.
pub fn curve_figure(
    title: &str,
    grid: &ScanGrid,
    energies: &[f64],
    biased: Option<&[f64]>,
    candidates: &[Candidate],
) -> PlotFigure {
    let axis = &grid.axes()[0];
    let x_title = if grid.axes().len() > 1 {
        let others: Vec<String> = grid.axes()[1..].iter().map(ScanAxis::axis_title).collect();
        format!("{} (with {})", axis.axis_title(), others.join(", "))
    } else {
        axis.axis_title()
    };
    let mut figure = PlotFigure::new(title, x_title, "Energy (kcal/mol)");
    let xs: Vec<f64> = (0..grid.len()).map(|p| grid.values(p)[0]).collect();
    figure.add_series(PlotSeries {
        label: "energy".to_string(),
        xs: xs.clone(),
        ys: energies.to_vec(),
        color: Rgb::BLUE,
        style: PlotStyle::LinePoints,
    });
    if let Some(b) = biased {
        figure.add_series(PlotSeries {
            label: "energy with the bias potentials".to_string(),
            xs: xs.clone(),
            ys: b.to_vec(),
            color: Rgb::AMBER,
            style: PlotStyle::LinePoints,
        });
    }
    markers(&mut figure, candidates, Rgb::BLACK, |c| {
        (at(axis, c.point, c.offset[0]), energies[c.point])
    });
    figure
}

/// Bilinear interpolation of the grid of `values` (`nx` by `ny`, first index fastest) at the
/// fractional indices `(s, t)`; NaN where a corner is missing.
fn bilinear(values: &[f64], nx: usize, ny: usize, s: f64, t: f64) -> f64 {
    let i = (s.floor().max(0.0) as usize).min(nx.saturating_sub(2));
    let j = (t.floor().max(0.0) as usize).min(ny.saturating_sub(2));
    let (fs, ft) = (
        (s - i as f64).clamp(0.0, 1.0),
        (t - j as f64).clamp(0.0, 1.0),
    );
    let v = |a: usize, b: usize| values[(b.min(ny - 1)) * nx + a.min(nx - 1)];
    let (i1, j1) = ((i + 1).min(nx - 1), (j + 1).min(ny - 1));
    (1.0 - fs) * (1.0 - ft) * v(i, j)
        + fs * (1.0 - ft) * v(i1, j)
        + (1.0 - fs) * ft * v(i, j1)
        + fs * ft * v(i1, j1)
}

/// The energy map of a scan of two coordinates (`energies` in kcal/mol, one per grid point),
/// with the candidates and `overlays` (in the units of the two coordinates) on top.
pub fn map_figure(
    title: &str,
    grid: &ScanGrid,
    energies: &[f64],
    candidates: &[Candidate],
    overlays: &[Overlay],
) -> PlotFigure {
    let (ax, ay) = (&grid.axes()[0], &grid.axes()[1]);
    let (nx, ny) = (ax.values.len(), ay.values.len());
    let range = |a: &ScanAxis| {
        let (lo, hi) = a
            .values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| {
                (l.min(*v), h.max(*v))
            });
        if hi > lo {
            (lo, hi)
        } else {
            (lo - 0.5, hi + 0.5)
        }
    };
    let (rx, ry) = (range(ax), range(ay));
    let cells = |n: usize| ((n.max(2) - 1) * CELLS_PER_STEP).clamp(1, MAX_CELLS);
    let fraction = |a: &ScanAxis, v: f64| {
        let first = a.values[0];
        let last = a.values[a.values.len() - 1];
        if a.values.len() > 1 && last != first {
            (v - first) / (last - first) * (a.values.len() - 1) as f64
        } else {
            0.0
        }
    };
    let map = ColorGrid::sample(
        "energy (kcal/mol)",
        rx,
        ry,
        (cells(nx), cells(ny)),
        |x, y| {
            if nx < 2 || ny < 2 {
                return f64::NAN;
            }
            bilinear(energies, nx, ny, fraction(ax, x), fraction(ay, y))
        },
    );
    let mut figure = PlotFigure::new(title, ax.axis_title(), ay.axis_title());
    figure.set_color_grid(map);
    for o in overlays {
        figure.add_series(PlotSeries {
            label: o.label.clone(),
            xs: o.xs.clone(),
            ys: o.ys.clone(),
            color: o.color,
            style: o.style,
        });
    }
    markers(&mut figure, candidates, Rgb::WHITE, |c| {
        let idx = grid.indices(c.point);
        (at(ax, idx[0], c.offset[0]), at(ay, idx[1], c.offset[1]))
    });
    figure
}

/// The curves of a map along its first coordinate, one for each value of the second (at most
/// twelve, spread over the values).
pub fn rows_figure(title: &str, grid: &ScanGrid, energies: &[f64]) -> PlotFigure {
    let (ax, ay) = (&grid.axes()[0], &grid.axes()[1]);
    let (nx, ny) = (ax.values.len(), ay.values.len());
    let mut figure = PlotFigure::new(title, ax.axis_title(), "Energy (kcal/mol)");
    let rows: Vec<usize> = if ny <= MAX_ROWS {
        (0..ny).collect()
    } else {
        (0..MAX_ROWS)
            .map(|k| k * (ny - 1) / (MAX_ROWS - 1))
            .collect()
    };
    for (k, &j) in rows.iter().enumerate() {
        let t = if rows.len() > 1 {
            k as f64 / (rows.len() - 1) as f64
        } else {
            0.5
        };
        let color = Rgb::jet(0.1 + 0.8 * t);
        figure.add_series(PlotSeries {
            label: format!("{} = {:.4}", ay.label, ay.values[j]),
            xs: ax.values.clone(),
            ys: (0..nx).map(|i| energies[j * nx + i]).collect(),
            color: if color.is_light() { Rgb::SLATE } else { color },
            style: PlotStyle::LinePoints,
        });
    }
    figure
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::workflow::scan::grid::GridOrder;

    #[test]
    fn maps_and_curves_carry_their_labels() {
        let axes = vec![
            ScanAxis::linspace("dihedral(1,2,3,4)", "degree", 180.0, -180.0, 5).unwrap(),
            ScanAxis::linspace("bond(1,2)", "Angstrom", 1.0, 2.0, 4).unwrap(),
        ];
        let grid = ScanGrid::product(axes, GridOrder::Snake).unwrap();
        let mut energies: Vec<f64> = (0..grid.len()).map(|p| p as f64).collect();
        energies[3] = f64::NAN;
        let candidate = Candidate {
            point: 6,
            kind: CandidateKind::Saddle,
            energy: 6.0,
            offset: vec![0.2, -0.1],
            model_energy: 6.0,
        };
        let map = map_figure("Relaxed scan", &grid, &energies, &[candidate], &[]);
        let svg = map.to_svg_string();
        for text in [
            ">dihedral(1,2,3,4) (degree)<",
            ">bond(1,2) (Angstrom)<",
            ">energy (kcal/mol)<",
            ">transition-state candidates<",
        ] {
            assert!(svg.contains(text), "{text}");
        }
        // The axis that runs downwards is drawn upwards; the missing point is grey.
        assert_eq!(map.grid.as_ref().unwrap().x, (-180.0, 180.0));
        assert!(map
            .grid
            .as_ref()
            .unwrap()
            .colors()
            .contains(&Rgb::jet(f64::NAN)));
        let rows = rows_figure("Rows", &grid, &energies);
        assert_eq!(rows.series.len(), 4);
        let joint = ScanGrid::joint(vec![grid.axes()[1].clone()]).unwrap();
        let curve = curve_figure("Curve", &joint, &[0.0, 2.0, 1.0, 3.0], None, &[]);
        assert!(curve.to_svg_string().contains(">Energy (kcal/mol)<"));
    }
}
