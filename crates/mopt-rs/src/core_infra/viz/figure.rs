//! Figure model (series, labels, bounds) shared by the SVG and PNG renderers.
//!
//! MultiOptPy: `Visualization/visualization.py:Graph` (`single_plot`, `double_plot`, `stem_plot`)
//!  and `NEBVisualizer`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fs;
use std::path::Path;

use super::color::Rgb;
use super::raster::render_png;
use super::svg::render_svg;
use crate::error::MoptResult;

/// Visual style of a plotted series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotStyle {
    /// Connected polyline with circular markers.
    LinePoints,
    /// Unconnected scatter markers.
    Scatter,
    /// Vertical stems from `y = 0` to each data point.
    Stem,
}

/// Single 2D data series for scientific plotting.
#[derive(Debug, Clone)]
pub struct PlotSeries {
    /// Legend label.
    pub label: String,
    /// X values.
    pub xs: Vec<f64>,
    /// Y values (same length as `xs`).
    pub ys: Vec<f64>,
    /// Series color.
    pub color: Rgb,
    /// Rendering style.
    pub style: PlotStyle,
}

/// Points colored by a value (an energy along a path, for example), drawn with a color
/// bar at the right of the figure.
#[derive(Debug, Clone)]
pub struct ColoredPoints {
    /// What the value is (the label of the color bar).
    pub label: String,
    /// X values.
    pub xs: Vec<f64>,
    /// Y values (same length as `xs`).
    pub ys: Vec<f64>,
    /// The value of each point.
    pub values: Vec<f64>,
}

/// Smallest and largest finite value (`(0, 1)` when there is none).
fn finite_range(values: &[f64]) -> (f64, f64) {
    let finite = values.iter().copied().filter(|v| v.is_finite());
    let (lo, hi) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    if lo.is_finite() {
        (lo, hi)
    } else {
        (0.0, 1.0)
    }
}

/// The color of `value` on the scale from `lo` to `hi`.
fn scale_color(value: f64, (lo, hi): (f64, f64)) -> Rgb {
    if hi > lo {
        Rgb::jet((value - lo) / (hi - lo))
    } else {
        Rgb::jet(if value.is_finite() { 0.5 } else { f64::NAN })
    }
}

impl ColoredPoints {
    /// Smallest and largest finite value (`(0, 1)` when there is none).
    pub fn range(&self) -> (f64, f64) {
        finite_range(&self.values)
    }

    /// The color of point `k`.
    pub fn color(&self, k: usize) -> Rgb {
        let value = self.values.get(k).copied().unwrap_or(f64::NAN);
        scale_color(value, self.range())
    }
}

/// A function of two variables drawn as colored cells that fill the plot (a map of a
/// surface, for example), with a color bar at the right of the figure. The series of the
/// figure are drawn on top of it.
#[derive(Debug, Clone)]
pub struct ColorGrid {
    /// What the value is (the label of the color bar).
    pub label: String,
    /// Lower and upper edge of the cells along the horizontal axis.
    pub x: (f64, f64),
    /// Lower and upper edge of the cells along the vertical axis.
    pub y: (f64, f64),
    /// Cells along the horizontal axis.
    pub columns: usize,
    /// Cells along the vertical axis.
    pub rows: usize,
    /// The value of every cell, row by row from the lowest row, each row from the left.
    pub values: Vec<f64>,
}

impl ColorGrid {
    /// The grid of `value` at the centres of `columns` by `rows` cells.
    pub fn sample(
        label: impl Into<String>,
        x: (f64, f64),
        y: (f64, f64),
        (columns, rows): (usize, usize),
        value: impl Fn(f64, f64) -> f64,
    ) -> Self {
        let (columns, rows) = (columns.max(1), rows.max(1));
        let mut values = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            let at_y = y.0 + (y.1 - y.0) * (row as f64 + 0.5) / rows as f64;
            for column in 0..columns {
                let at_x = x.0 + (x.1 - x.0) * (column as f64 + 0.5) / columns as f64;
                values.push(value(at_x, at_y));
            }
        }
        Self {
            label: label.into(),
            x,
            y,
            columns,
            rows,
            values,
        }
    }

    /// Smallest and largest finite value (`(0, 1)` when there is none).
    pub fn range(&self) -> (f64, f64) {
        finite_range(&self.values)
    }

    /// The color of the cell in `column` and `row`.
    pub fn color(&self, column: usize, row: usize) -> Rgb {
        let value = self
            .values
            .get(row * self.columns + column)
            .copied()
            .unwrap_or(f64::NAN);
        scale_color(value, self.range())
    }

    /// The colors of all cells, in the order of the values.
    pub fn colors(&self) -> Vec<Rgb> {
        let range = self.range();
        self.values
            .iter()
            .map(|value| scale_color(*value, range))
            .collect()
    }

    /// The index of the cell that holds the point `(x, y)` (the nearest cell outside).
    pub fn cell_at(&self, x: f64, y: f64) -> usize {
        let cell = |v: f64, (lo, hi): (f64, f64), count: usize| {
            let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.0 };
            ((t * count as f64).floor().max(0.0) as usize).min(count.saturating_sub(1))
        };
        cell(y, self.y, self.rows) * self.columns + cell(x, self.x, self.columns)
    }

    /// The color of the cell that holds the point `(x, y)` (the nearest cell outside).
    pub fn color_at(&self, x: f64, y: f64) -> Rgb {
        let cell = self.cell_at(x, y);
        self.color(cell % self.columns, cell / self.columns)
    }
}

/// Figure specification for 2D scientific charts (energy profiles, NEB bands, IRC curvature, PCA/CMDS).
#[derive(Debug, Clone)]
pub struct PlotFigure {
    /// Figure title.
    pub title: String,
    /// X-axis label.
    pub x_label: String,
    /// Y-axis label.
    pub y_label: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Data series.
    pub series: Vec<PlotSeries>,
    /// Points colored by a value, with their color bar.
    pub colored: Option<ColoredPoints>,
    /// Colored cells under the series, with their color bar (which is then the one of
    /// the figure).
    pub grid: Option<ColorGrid>,
}

impl PlotFigure {
    /// Construct a new figure with default dimensions (`800 x 520`).
    pub fn new(
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            width: 800,
            height: 520,
            series: Vec::new(),
            colored: None,
            grid: None,
        }
    }

    /// Draw the colored cells of `grid` under the series, with a color bar. The plot then
    /// shows what the grid covers.
    pub fn set_color_grid(&mut self, grid: ColorGrid) -> &mut Self {
        self.grid = Some(grid);
        self
    }

    /// Draw `points` colored by their values, with a color bar.
    pub fn set_colored_points(&mut self, points: ColoredPoints) -> &mut Self {
        self.colored = Some(points);
        self
    }

    /// Append a data series to the figure.
    pub fn add_series(&mut self, series: PlotSeries) -> &mut Self {
        self.series.push(series);
        self
    }

    pub(crate) fn data_bounds(&self) -> (f64, f64, f64, f64) {
        if let Some(grid) = &self.grid {
            if grid.x.1 > grid.x.0 && grid.y.1 > grid.y.0 {
                return (grid.x.0, grid.x.1, grid.y.0, grid.y.1);
            }
        }
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for s in &self.series {
            for (&x, &y) in s.xs.iter().zip(s.ys.iter()) {
                if x.is_finite() && y.is_finite() {
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                    if s.style == PlotStyle::Stem {
                        y_min = y_min.min(0.0);
                        y_max = y_max.max(0.0);
                    }
                }
            }
        }
        if let Some(points) = &self.colored {
            for (&x, &y) in points.xs.iter().zip(points.ys.iter()) {
                if x.is_finite() && y.is_finite() {
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
        }
        if !x_min.is_finite() || !y_min.is_finite() {
            return (0.0, 1.0, 0.0, 1.0);
        }
        if (x_max - x_min).abs() < 1e-12 {
            x_min -= 0.5;
            x_max += 0.5;
        }
        if (y_max - y_min).abs() < 1e-12 {
            y_min -= 0.5;
            y_max += 0.5;
        } else {
            let pad = 0.06 * (y_max - y_min);
            y_min -= pad;
            y_max += pad;
        }
        (x_min, x_max, y_min, y_max)
    }

    /// Render as an SVG document.
    pub fn to_svg_string(&self) -> String {
        render_svg(self)
    }

    /// Render as PNG bytes.
    pub fn to_png_bytes(&self) -> Vec<u8> {
        render_png(self)
    }

    /// Save the figure to `path` (automatically selects `.svg` or `.png` by extension).
    pub fn save(&self, path: impl AsRef<Path>) -> MoptResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let is_svg = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("svg"));
        if is_svg {
            fs::write(path, render_svg(self))?;
        } else {
            fs::write(path, render_png(self))?;
        }
        Ok(())
    }
}

/// Convenience helper mirroring `Graph::single_plot` in MultiOptPy.
pub fn save_single_line_plot(
    png_path: impl AsRef<Path>,
    title: &str,
    x_label: &str,
    y_label: &str,
    xs: &[f64],
    ys: &[f64],
) -> MoptResult<()> {
    let mut fig = PlotFigure::new(title, x_label, y_label);
    fig.add_series(PlotSeries {
        label: title.to_string(),
        xs: xs.to_vec(),
        ys: ys.to_vec(),
        color: Rgb::BLUE,
        style: PlotStyle::LinePoints,
    });
    fig.save(png_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_infra::viz::png::PNG_SIGNATURE;

    #[test]
    fn svg_and_png_generation_produce_valid_headers() {
        let mut fig = PlotFigure::new("Energy Profile", "Step", "Energy (kcal/mol)");
        fig.add_series(PlotSeries {
            label: "Path".into(),
            xs: vec![0.0, 1.0, 2.0, 3.0],
            ys: vec![0.0, 12.5, 24.1, -3.2],
            color: Rgb::BLUE,
            style: PlotStyle::LinePoints,
        });
        let svg = fig.to_svg_string();
        assert!(svg.starts_with("<svg "));
        assert!(svg.contains("Energy Profile"));

        let png = fig.to_png_bytes();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        assert!(png.windows(4).any(|w| w == b"IHDR"));
        assert!(png.windows(4).any(|w| w == b"IDAT"));
        assert!(png.windows(4).any(|w| w == b"IEND"));
    }

    #[test]
    fn colored_points_get_their_colors_and_a_color_bar() {
        assert_eq!(Rgb::jet(0.0), Rgb(0, 0, 128));
        assert_eq!(Rgb::jet(0.5), Rgb(128, 255, 128));
        assert_eq!(Rgb::jet(1.0), Rgb(128, 0, 0));
        assert_eq!(Rgb::jet(7.0), Rgb::jet(1.0));
        let points = ColoredPoints {
            label: "energy (kcal/mol)".into(),
            xs: vec![0.0, 1.0, 2.0],
            ys: vec![0.0, 0.5, -0.5],
            values: vec![0.0, 10.0, 5.0],
        };
        assert_eq!(points.range(), (0.0, 10.0));
        assert_eq!(points.color(1), Rgb::jet(1.0));
        assert_eq!(points.color(2), Rgb::jet(0.5));
        let mut fig = PlotFigure::new("Map", "PCo1", "PCo2");
        fig.set_colored_points(points);
        // The points alone set the bounds.
        let (x_min, x_max, ..) = fig.data_bounds();
        assert!((x_min - 0.0).abs() < 1e-12 && (x_max - 2.0).abs() < 1e-12);
        let svg = fig.to_svg_string();
        assert!(svg.contains("energy (kcal/mol)"));
        assert!(svg.contains(&Rgb::jet(1.0).to_hex()));
        assert_eq!(svg.matches("<circle").count(), 3);
        let png = fig.to_png_bytes();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn a_colored_grid_fills_the_plot() {
        // x + 10 y at the centres of 4 by 2 cells over [0, 4] x [0, 2].
        let grid = ColorGrid::sample("height", (0.0, 4.0), (0.0, 2.0), (4, 2), |x, y| {
            x + 10.0 * y
        });
        assert_eq!(grid.values.len(), 8);
        assert_eq!(grid.values[0], 0.5 + 5.0);
        assert_eq!(grid.values[7], 3.5 + 15.0);
        assert_eq!(grid.range(), (5.5, 18.5));
        assert_eq!(grid.color(0, 0), Rgb::jet(0.0));
        assert_eq!(grid.color(3, 1), Rgb::jet(1.0));
        assert_eq!(grid.color_at(3.9, 1.9), Rgb::jet(1.0));
        assert_eq!(grid.color_at(-5.0, -5.0), Rgb::jet(0.0));
        assert_eq!(grid.color_at(9.0, 9.0), Rgb::jet(1.0));
        assert_eq!(grid.cell_at(2.5, 1.5), 6);
        assert_eq!(grid.colors()[6], grid.color(2, 1));
        let mut fig = PlotFigure::new("Surface", "x", "y");
        fig.set_color_grid(grid);
        fig.add_series(PlotSeries {
            label: "path".into(),
            xs: vec![1.0, 3.0],
            ys: vec![0.5, 1.5],
            color: Rgb::BLACK,
            style: PlotStyle::LinePoints,
        });
        // The plot shows what the grid covers, without a margin.
        assert_eq!(fig.data_bounds(), (0.0, 4.0, 0.0, 2.0));
        let svg = fig.to_svg_string();
        assert_eq!(svg.matches("<rect class=\"cell\"").count(), 8);
        assert!(svg.contains(">height<") && svg.contains(">path<"));
        let png = fig.to_png_bytes();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn figures_carry_their_title_labels_numbers_and_legend() {
        let mut fig = PlotFigure::new("Energy profile", "Image", "Energy (kcal/mol)");
        for (label, color) in [("first", Rgb::BLUE), ("last", Rgb::RED), ("", Rgb::AMBER)] {
            fig.add_series(PlotSeries {
                label: label.into(),
                xs: vec![0.0, 1.0, 2.0, 3.0],
                ys: vec![0.0, 12.5, 24.1, -3.2],
                color,
                style: PlotStyle::LinePoints,
            });
        }
        let svg = fig.to_svg_string();
        for text in [
            ">Energy profile<",
            ">Image<",
            ">Energy (kcal/mol)<",
            ">first<",
            ">last<",
            // Numbers of the axes: whole images, energies in steps of ten.
            ">3<",
            ">20<",
            ">10<",
        ] {
            assert!(svg.contains(text), "{text} is missing");
        }
        // The series without a label has no entry in the legend.
        assert_eq!(svg.matches("class=\"legend\"").count(), 2);
        assert!(!svg.contains(">0.5<"));

        // The PNG has the same text as pixels: dark pixels where the title is, and more
        // of them overall than the same figure without any text.
        let dark = |fig: &PlotFigure| {
            let (width, pixels) = super::super::raster::render_pixels(fig);
            let rows = pixels.len() / (3 * width);
            let is_dark = |x: usize, y: usize| pixels[3 * (y * width + x)] < 80;
            let count = |rows: usize| {
                (0..rows)
                    .flat_map(|y| (0..width).map(move |x| (x, y)))
                    .filter(|&(x, y)| is_dark(x, y))
                    .count()
            };
            (count(rows), count(26))
        };
        let (with_text, title) = dark(&fig);
        let mut bare = fig.clone();
        bare.title.clear();
        bare.x_label.clear();
        bare.y_label.clear();
        for series in &mut bare.series {
            series.label.clear();
        }
        let (without_labels, no_title) = dark(&bare);
        assert!(title > 100 && no_title == 0, "{title} {no_title}");
        assert!(
            with_text > without_labels + 500,
            "{with_text} {without_labels}"
        );
    }
}
