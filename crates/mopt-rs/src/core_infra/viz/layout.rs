//! Where the parts of a figure go, the same for the SVG and the PNG renderer: the plot
//! area, the ticks of the axes with their numbers, the entries of the legend, and the
//! color bar.
//!
//! A figure reads from top to bottom: the title, the legend (one entry per labelled
//! series, in rows above the plot so that it never covers data), the plot with the
//! numbers of the vertical axis and its label at the left, the numbers of the horizontal
//! axis and its label below. The color bar of colored points or of a colored grid stands
//! at the right with its numbers and its label.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy draws through matplotlib).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::figure::PlotFigure;

/// Size of the text of a renderer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TextMetrics {
    /// Width of a character.
    pub char_width: f64,
    /// Height of a line of text.
    pub height: f64,
}

impl TextMetrics {
    /// Width of `text`.
    pub(crate) fn width(&self, text: &str) -> f64 {
        self.char_width * text.chars().count() as f64
    }
}

/// A tick of an axis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Tick {
    /// Its value.
    pub value: f64,
    /// The number written at it.
    pub label: String,
}

/// An entry of the legend.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LegendEntry {
    /// Index of the series.
    pub series: usize,
    /// Left end of its sample.
    pub x: f64,
    /// Middle of its line.
    pub y: f64,
}

/// The color bar.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColorBar {
    /// Left edge.
    pub x: f64,
    /// Lowest and highest value.
    pub range: (f64, f64),
    /// Ticks between them.
    pub ticks: Vec<Tick>,
    /// What the values are.
    pub label: String,
    /// Horizontal middle of the label, which is written upwards.
    pub label_x: f64,
}

/// Width of the sample of a legend entry.
pub(crate) const LEGEND_SAMPLE: f64 = 28.0;
/// Width of the color bar.
pub(crate) const BAR_WIDTH: f64 = 18.0;
/// Space between parts.
const GAP: f64 = 8.0;
/// Margin of the figure.
const MARGIN: f64 = 10.0;

/// The layout of a figure.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    /// Left edge of the plot.
    pub left: f64,
    /// Top edge of the plot.
    pub top: f64,
    /// Width of the plot.
    pub width: f64,
    /// Height of the plot.
    pub height: f64,
    /// Lowest and highest value of the horizontal and of the vertical axis.
    pub bounds: (f64, f64, f64, f64),
    /// Ticks of the horizontal axis.
    pub x_ticks: Vec<Tick>,
    /// Ticks of the vertical axis.
    pub y_ticks: Vec<Tick>,
    /// Top of the title.
    pub title_top: f64,
    /// Entries of the legend.
    pub legend: Vec<LegendEntry>,
    /// The color bar, if the figure has colored points or a colored grid.
    pub bar: Option<ColorBar>,
    /// Top of the numbers of the horizontal axis.
    pub x_numbers_top: f64,
    /// Top of the label of the horizontal axis.
    pub x_label_top: f64,
    /// Horizontal middle of the label of the vertical axis, which is written upwards.
    pub y_label_x: f64,
}

impl Layout {
    /// The layout of `figure` for text of the size `text`.
    pub(crate) fn of(figure: &PlotFigure, text: TextMetrics) -> Self {
        let (w, h) = (f64::from(figure.width), f64::from(figure.height));
        let bounds = figure.data_bounds();
        let (x_min, x_max, y_min, y_max) = bounds;
        let y_ticks = nice_ticks(y_min, y_max, 6, false);
        let widest = |ticks: &[Tick]| {
            ticks
                .iter()
                .map(|t| text.width(&t.label))
                .fold(0.0_f64, f64::max)
        };
        let left = MARGIN + text.height + GAP + widest(&y_ticks) + GAP;

        let scale = figure
            .grid
            .as_ref()
            .map(|grid| (grid.range(), grid.label.clone()))
            .or_else(|| {
                figure
                    .colored
                    .as_ref()
                    .map(|points| (points.range(), points.label.clone()))
            });
        let (bar_ticks, bar_space) = match &scale {
            Some(((lo, hi), _)) => {
                let ticks = nice_ticks(*lo, *hi, 6, false);
                let space = 2.0 * GAP + BAR_WIDTH + 6.0 + widest(&ticks) + GAP + text.height;
                (ticks, space)
            }
            None => (Vec::new(), 0.0),
        };
        // Without a color bar the last number of the horizontal axis needs some room.
        let right = MARGIN + bar_space.max(4.0 * text.char_width);
        let width = (w - left - right).max(40.0);

        // The legend: rows above the plot, as many entries per row as fit.
        let labelled: Vec<usize> = figure
            .series
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.label.is_empty())
            .map(|(k, _)| k)
            .collect();
        let line = text.height + 6.0;
        let legend_top = MARGIN + text.height + GAP;
        let mut legend = Vec::with_capacity(labelled.len());
        let (mut x, mut row) = (left, 0usize);
        for &k in &labelled {
            let entry = LEGEND_SAMPLE + 6.0 + text.width(&figure.series[k].label) + 2.0 * GAP;
            if x > left && x + entry > left + width + GAP {
                x = left;
                row += 1;
            }
            legend.push(LegendEntry {
                series: k,
                x,
                y: legend_top + (row as f64 + 0.5) * line,
            });
            x += entry;
        }
        let rows = if labelled.is_empty() { 0 } else { row + 1 };
        let top = legend_top + rows as f64 * line + GAP;
        let bottom = MARGIN + text.height + GAP + text.height + GAP;
        let height = (h - top - bottom).max(40.0);

        // As many numbers along the horizontal axis as fit side by side.
        let whole = whole_numbers(figure);
        let mut x_ticks = nice_ticks(x_min, x_max, 8, whole);
        for target in (2..8).rev() {
            let needed: f64 = x_ticks
                .iter()
                .map(|t| text.width(&t.label) + 2.0 * text.char_width)
                .sum();
            if needed <= width {
                break;
            }
            x_ticks = nice_ticks(x_min, x_max, target, whole);
        }

        let bar = scale.map(|(range, label)| {
            let x = left + width + 2.0 * GAP;
            ColorBar {
                x,
                range,
                label_x: x + BAR_WIDTH + 6.0 + widest(&bar_ticks) + GAP + 0.5 * text.height,
                ticks: bar_ticks,
                label,
            }
        });
        Self {
            left,
            top,
            width,
            height,
            bounds,
            x_ticks,
            y_ticks,
            title_top: MARGIN,
            legend,
            bar,
            x_numbers_top: top + height + GAP,
            x_label_top: top + height + GAP + text.height + GAP,
            y_label_x: MARGIN + 0.5 * text.height,
        }
    }

    /// Horizontal position of the value `x`.
    pub(crate) fn x(&self, x: f64) -> f64 {
        let (lo, hi, ..) = self.bounds;
        self.left + (x - lo) / (hi - lo) * self.width
    }

    /// Vertical position of the value `y`.
    pub(crate) fn y(&self, y: f64) -> f64 {
        let (.., lo, hi) = self.bounds;
        self.top + self.height - (y - lo) / (hi - lo) * self.height
    }

    /// Vertical position of the value `v` on the color bar.
    pub(crate) fn bar_y(&self, v: f64) -> f64 {
        let (lo, hi) = self.bar.as_ref().map_or((0.0, 1.0), |bar| bar.range);
        let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
        self.top + self.height - t * self.height
    }
}

/// Whether every horizontal value of the series is a whole number (images, iterations):
/// the ticks are then whole numbers too.
fn whole_numbers(figure: &PlotFigure) -> bool {
    figure.colored.is_none()
        && figure.grid.is_none()
        && !figure.series.is_empty()
        && figure
            .series
            .iter()
            .flat_map(|s| s.xs.iter())
            .all(|x| !x.is_finite() || (x - x.round()).abs() < 1e-9)
}

/// Ticks at multiples of 1, 2, or 5 times a power of ten between `lo` and `hi`, about
/// `target` of them (`whole`: at whole numbers only).
pub(crate) fn nice_ticks(lo: f64, hi: f64, target: usize, whole: bool) -> Vec<Tick> {
    let span = hi - lo;
    if !span.is_finite() || span <= 0.0 {
        return vec![Tick {
            value: lo,
            label: number(lo, 0.0),
        }];
    }
    let raw = span / target.max(1) as f64;
    let power = power_of_ten(raw);
    // A little more than 1, 2, or 5 by rounding is 1, 2, or 5.
    let fraction = raw / power - 1e-9;
    let mut step = power
        * if fraction <= 1.0 {
            1.0
        } else if fraction <= 2.0 {
            2.0
        } else if fraction <= 5.0 {
            5.0
        } else {
            10.0
        };
    if whole {
        step = step.max(1.0);
    }
    let first = (lo / step - 1e-9).ceil() as i64;
    let last = (hi / step + 1e-9).floor() as i64;
    (first..=last)
        .map(|k| {
            let value = k as f64 * step;
            Tick {
                value,
                label: number(value, step),
            }
        })
        .collect()
}

/// The largest power of ten that is not larger than `value` (a positive number).
fn power_of_ten(value: f64) -> f64 {
    10.0_f64.powf((value.log10() + 1e-9).floor())
}

/// `value` written with the digits that ticks `step` apart need.
pub(crate) fn number(value: f64, step: f64) -> String {
    if !value.is_finite() {
        return "?".to_string();
    }
    let size = value.abs().max(step.abs());
    let text = if size == 0.0 {
        "0".to_string()
    } else if size >= 1e6 || (step > 0.0 && step < 1e-4) || (step <= 0.0 && size < 1e-4) {
        if value == 0.0 {
            "0".to_string()
        } else {
            // The digits of the multiple of the step's power of ten that the value is.
            let digits = if step > 0.0 {
                let multiple = value.abs() / power_of_ten(step);
                ((multiple.log10() + 1e-9).floor().max(0.0) as usize).min(6)
            } else {
                3
            };
            format!("{value:.digits$e}")
        }
    } else if step > 0.0 {
        let decimals = (-(step.log10() + 1e-9).floor()).clamp(0.0, 8.0) as usize;
        format!("{value:.decimals$}")
    } else {
        format!("{value:.4}")
    };
    // No "-0", "-0.00".
    if text.starts_with('-') && text[1..].chars().all(|c| c == '0' || c == '.') {
        text[1..].to_string()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::Rgb;
    use super::super::figure::{ColoredPoints, PlotSeries, PlotStyle};
    use super::*;

    fn labels(ticks: &[Tick]) -> Vec<&str> {
        ticks.iter().map(|t| t.label.as_str()).collect()
    }

    #[test]
    fn ticks_are_round_numbers_within_the_range() {
        assert_eq!(
            labels(&nice_ticks(0.0, 1.0, 5, false)),
            ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        );
        assert_eq!(
            labels(&nice_ticks(-0.3, 2.1, 6, false)),
            ["0.0", "0.5", "1.0", "1.5", "2.0"]
        );
        assert_eq!(
            labels(&nice_ticks(-7.3, -1.9, 6, false)),
            ["-7", "-6", "-5", "-4", "-3", "-2"]
        );
        // Whole numbers only: no halves between images 0 to 3.
        assert_eq!(labels(&nice_ticks(0.0, 3.0, 8, true)), ["0", "1", "2", "3"]);
        assert_eq!(
            labels(&nice_ticks(0.0, 3.0, 8, false)),
            ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]
        );
        // Large and small numbers in powers of ten; a range without a width has one tick.
        assert_eq!(
            labels(&nice_ticks(0.0, 4.0e6, 4, false)),
            ["0", "1e6", "2e6", "3e6", "4e6"]
        );
        assert_eq!(
            labels(&nice_ticks(1.0e-6, 3.0e-6, 2, false)),
            ["1e-6", "2e-6", "3e-6"]
        );
        assert_eq!(nice_ticks(2.0, 2.0, 5, false).len(), 1);
        assert_eq!(number(1.2e7, 2.0e6), "1.2e7");
        assert_eq!(number(-0.0001, 0.01), "0.00");
        assert_eq!(number(f64::NAN, 1.0), "?");
        assert_eq!(number(0.0, 0.0), "0");
        assert_eq!(number(1.5, 0.0), "1.5000");
    }

    #[test]
    fn the_parts_of_a_figure_do_not_overlap() {
        let text = TextMetrics {
            char_width: 12.0,
            height: 14.0,
        };
        let mut figure = PlotFigure::new("Title", "Iteration", "Energy");
        for k in 0..9 {
            figure.add_series(PlotSeries {
                label: format!("iteration {k}"),
                xs: vec![0.0, 1.0, 2.0],
                ys: vec![0.0, 1.0 + k as f64, 0.5],
                color: Rgb::BLUE,
                style: PlotStyle::LinePoints,
            });
        }
        let plain = Layout::of(&figure, text);
        // Nine entries of about 190 wide in a plot of about 690: three rows.
        let rows: Vec<f64> = plain.legend.iter().map(|e| e.y).collect();
        assert_eq!(plain.legend.len(), 9);
        assert!(rows[0] < rows[8] && rows[8] < plain.top);
        assert!(plain.title_top + text.height < rows[0]);
        assert!(plain.legend.iter().all(|e| e.x >= plain.left));
        assert!(plain.bar.is_none());
        // The axes: whole numbers below, the plot inside the figure.
        assert_eq!(labels(&plain.x_ticks), ["0", "1", "2"]);
        assert!(plain.left > 3.0 * text.char_width);
        assert!(plain.x_label_top + text.height <= 520.0);
        assert!((plain.x(0.0) - plain.left).abs() < 1e-9);
        assert!((plain.x(2.0) - plain.left - plain.width).abs() < 1e-9);
        assert!(plain.y(plain.bounds.3) < plain.y(plain.bounds.2));

        // Colored points bring a color bar at the right, inside the figure.
        figure.set_colored_points(ColoredPoints {
            label: "Energy (kcal/mol)".into(),
            xs: vec![0.0, 1.0],
            ys: vec![0.0, 1.0],
            values: vec![-3.0, 12.0],
        });
        let colored = Layout::of(&figure, text);
        let bar = colored.bar.clone().unwrap();
        assert!(bar.x > colored.left + colored.width);
        assert!(bar.label_x + 0.5 * text.height <= 800.0);
        assert!(colored.width < plain.width);
        assert_eq!(labels(&bar.ticks), ["0", "5", "10"]);
        assert!(colored.bar_y(12.0) < colored.bar_y(-3.0));
        assert!((colored.bar_y(-3.0) - colored.top - colored.height).abs() < 1e-9);
    }
}
