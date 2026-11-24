//! Vector (SVG) renderer for [`PlotFigure`]: the same parts in the same places as the PNG
//! renderer (`layout::Layout`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy draws through matplotlib).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;

use super::color::Rgb;
use super::figure::{PlotFigure, PlotStyle};
use super::layout::{Layout, TextMetrics, BAR_WIDTH, LEGEND_SAMPLE};

/// Size of the text: a sans-serif font of 12 pixels, whose characters are about 0.56 of
/// that wide on average.
const FONT_SIZE: f64 = 12.0;
const TEXT: TextMetrics = TextMetrics {
    char_width: 6.8,
    height: FONT_SIZE,
};
/// Length of a tick mark.
const TICK: f64 = 5.0;
/// From the top of a line of text to its baseline.
const BASELINE: f64 = 0.82 * FONT_SIZE;

/// Where a piece of text is anchored.
#[derive(Clone, Copy)]
enum Anchor {
    Start,
    Middle,
    End,
}

/// A piece of text with its top at `top`.
fn text(svg: &mut String, x: f64, top: f64, anchor: Anchor, content: &str, class: &str) {
    let anchor = match anchor {
        Anchor::Start => "start",
        Anchor::Middle => "middle",
        Anchor::End => "end",
    };
    let class = if class.is_empty() {
        String::new()
    } else {
        format!(r#" class="{class}""#)
    };
    let _ = writeln!(
        svg,
        r#"  <text{class} x="{x:.2}" y="{:.2}" text-anchor="{anchor}">{}</text>"#,
        top + BASELINE,
        escape_xml(content)
    );
}

/// A piece of text written upwards, centred on `(x, y)`.
fn text_upwards(svg: &mut String, x: f64, y: f64, content: &str) {
    let baseline = x + 0.5 * FONT_SIZE - (FONT_SIZE - BASELINE);
    let _ = writeln!(
        svg,
        r#"  <text x="{baseline:.2}" y="{y:.2}" text-anchor="middle" transform="rotate(-90 {baseline:.2} {y:.2})">{}</text>"#,
        escape_xml(content)
    );
}

fn line(svg: &mut String, from: (f64, f64), to: (f64, f64), color: Rgb, width: f64) {
    let _ = writeln!(
        svg,
        r#"  <line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="{width}"/>"#,
        from.0,
        from.1,
        to.0,
        to.1,
        color.to_hex()
    );
}

/// Render `fig` as a standalone SVG document.
pub fn render_svg(fig: &PlotFigure) -> String {
    let layout = Layout::of(fig, TEXT);
    let (left, top) = (layout.left, layout.top);
    let (right, bottom) = (left + layout.width, top + layout.height);
    let black = Rgb::BLACK.to_hex();

    let mut svg = String::with_capacity(4096);
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}" font-family="sans-serif" font-size="{FONT_SIZE}" fill="{black}">"#,
        fig.width, fig.height, fig.width, fig.height
    );
    let _ = writeln!(
        svg,
        r#"  <rect width="100%" height="100%" fill="{}"/>"#,
        Rgb::WHITE.to_hex()
    );

    // The colored cells, or the lines of the grid at the ticks.
    if let Some(grid) = &fig.grid {
        let cell_w = layout.width / grid.columns as f64;
        let cell_h = layout.height / grid.rows as f64;
        let colors = grid.colors();
        for row in 0..grid.rows {
            for column in 0..grid.columns {
                let _ = writeln!(
                    svg,
                    r#"  <rect class="cell" x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="{}" shape-rendering="crispEdges"/>"#,
                    left + column as f64 * cell_w,
                    bottom - (row as f64 + 1.0) * cell_h,
                    cell_w + 0.5,
                    cell_h + 0.5,
                    colors
                        .get(row * grid.columns + column)
                        .copied()
                        .unwrap_or(Rgb::jet(f64::NAN))
                        .to_hex()
                );
            }
        }
    } else {
        for tick in &layout.x_ticks {
            let gx = layout.x(tick.value);
            line(&mut svg, (gx, top), (gx, bottom), Rgb::GRID, 1.0);
        }
        for tick in &layout.y_ticks {
            let gy = layout.y(tick.value);
            line(&mut svg, (left, gy), (right, gy), Rgb::GRID, 1.0);
        }
    }

    // The frame, the tick marks, and the numbers of the axes.
    let _ = writeln!(
        svg,
        r#"  <rect x="{left:.2}" y="{top:.2}" width="{:.2}" height="{:.2}" fill="none" stroke="{black}" stroke-width="1.4"/>"#,
        layout.width, layout.height
    );
    for tick in &layout.x_ticks {
        let gx = layout.x(tick.value);
        line(&mut svg, (gx, bottom), (gx, bottom + TICK), Rgb::BLACK, 1.0);
        text(
            &mut svg,
            gx,
            layout.x_numbers_top,
            Anchor::Middle,
            &tick.label,
            "",
        );
    }
    for tick in &layout.y_ticks {
        let gy = layout.y(tick.value);
        line(&mut svg, (left - TICK, gy), (left, gy), Rgb::BLACK, 1.0);
        text(
            &mut svg,
            left - TICK - 3.0,
            gy - 0.5 * FONT_SIZE,
            Anchor::End,
            &tick.label,
            "",
        );
    }

    // The title and the labels of the axes.
    let _ = writeln!(
        svg,
        r#"  <text x="{:.2}" y="{:.2}" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#,
        f64::from(fig.width) * 0.5,
        layout.title_top + BASELINE,
        escape_xml(&fig.title)
    );
    text(
        &mut svg,
        0.5 * (left + right),
        layout.x_label_top,
        Anchor::Middle,
        &fig.x_label,
        "",
    );
    text_upwards(
        &mut svg,
        layout.y_label_x,
        0.5 * (top + bottom),
        &fig.y_label,
    );

    // The data, inside the frame.
    let _ = writeln!(
        svg,
        r#"  <clipPath id="plot"><rect x="{left:.2}" y="{top:.2}" width="{:.2}" height="{:.2}"/></clipPath>"#,
        layout.width, layout.height
    );
    svg.push_str("  <g clip-path=\"url(#plot)\">\n");
    let y_zero = layout.y(0.0_f64.clamp(layout.bounds.2, layout.bounds.3));
    for s in &fig.series {
        let color_hex = s.color.to_hex();
        if s.style == PlotStyle::LinePoints && s.xs.len() >= 2 {
            let mut pts = String::new();
            for (&x, &y) in s.xs.iter().zip(s.ys.iter()) {
                if x.is_finite() && y.is_finite() {
                    let _ = write!(pts, "{:.2},{:.2} ", layout.x(x), layout.y(y));
                }
            }
            let _ = writeln!(
                svg,
                r#"  <polyline fill="none" stroke="{color_hex}" stroke-width="2" points="{}"/>"#,
                pts.trim_end()
            );
        }
        for (&x, &y) in s.xs.iter().zip(s.ys.iter()) {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let (cx, cy) = (layout.x(x), layout.y(y));
            if s.style == PlotStyle::Stem {
                line(&mut svg, (cx, y_zero), (cx, cy), s.color, 2.0);
            }
            let _ = writeln!(
                svg,
                r#"  <circle cx="{cx:.2}" cy="{cy:.2}" r="3.5" fill="{color_hex}"/>"#
            );
        }
    }
    if let Some(points) = &fig.colored {
        for (k, (&x, &y)) in points.xs.iter().zip(points.ys.iter()).enumerate() {
            if x.is_finite() && y.is_finite() {
                let _ = writeln!(
                    svg,
                    r#"  <circle cx="{:.2}" cy="{:.2}" r="4" fill="{}"/>"#,
                    layout.x(x),
                    layout.y(y),
                    points.color(k).to_hex()
                );
            }
        }
    }
    svg.push_str("  </g>\n");

    // The color bar: a strip of the scale from the lowest value (bottom) to the highest,
    // its numbers, and what the values are.
    if let Some(bar) = &layout.bar {
        let steps = 64usize;
        let step_h = layout.height / steps as f64;
        for i in 0..steps {
            let t = (i as f64 + 0.5) / steps as f64;
            let _ = writeln!(
                svg,
                r#"  <rect x="{:.2}" y="{:.2}" width="{BAR_WIDTH}" height="{:.2}" fill="{}"/>"#,
                bar.x,
                bottom - (i as f64 + 1.0) * step_h,
                step_h + 0.5,
                Rgb::jet(t).to_hex()
            );
        }
        let _ = writeln!(
            svg,
            r#"  <rect x="{:.2}" y="{top:.2}" width="{BAR_WIDTH}" height="{:.2}" fill="none" stroke="{black}" stroke-width="1"/>"#,
            bar.x, layout.height
        );
        let edge = bar.x + BAR_WIDTH;
        for tick in &bar.ticks {
            let gy = layout.bar_y(tick.value);
            line(&mut svg, (edge, gy), (edge + 3.0, gy), Rgb::BLACK, 1.0);
            text(
                &mut svg,
                edge + 6.0,
                gy - 0.5 * FONT_SIZE,
                Anchor::Start,
                &tick.label,
                "",
            );
        }
        text_upwards(&mut svg, bar.label_x, 0.5 * (top + bottom), &bar.label);
    }

    // The legend.
    for entry in &layout.legend {
        let series = &fig.series[entry.series];
        let color_hex = series.color.to_hex();
        let (x0, x1, y) = (entry.x, entry.x + LEGEND_SAMPLE, entry.y);
        if series.color.is_light() {
            let _ = writeln!(
                svg,
                r#"  <rect x="{:.2}" y="{:.2}" width="{:.2}" height="12" fill="{}"/>"#,
                x0 - 2.0,
                y - 6.0,
                LEGEND_SAMPLE + 4.0,
                Rgb::SLATE.to_hex()
            );
        }
        match series.style {
            PlotStyle::LinePoints => line(&mut svg, (x0, y), (x1, y), series.color, 2.0),
            PlotStyle::Stem => {
                let middle = 0.5 * (x0 + x1);
                line(&mut svg, (middle, y + 5.0), (middle, y), series.color, 2.0);
            }
            PlotStyle::Scatter => {}
        }
        let _ = writeln!(
            svg,
            r#"  <circle cx="{:.2}" cy="{y:.2}" r="3.5" fill="{color_hex}"/>"#,
            0.5 * (x0 + x1)
        );
        text(
            &mut svg,
            x1 + 6.0,
            y - 0.5 * FONT_SIZE,
            Anchor::Start,
            &series.label,
            "legend",
        );
    }

    svg.push_str("</svg>\n");
    svg
}

fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
    out
}
