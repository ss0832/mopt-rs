//! Raster (PNG) renderer for [`PlotFigure`]: the same parts in the same places as the SVG
//! renderer (`layout::Layout`), the text drawn with the bitmap font of [`super::font`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy draws through matplotlib).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::color::Rgb;
use super::figure::{PlotFigure, PlotSeries, PlotStyle};
use super::font::{glyph_pixel, GLYPH_ADVANCE, GLYPH_COLUMNS, GLYPH_ROWS, GLYPH_ROWS_ABOVE};
use super::layout::{Layout, TextMetrics, BAR_WIDTH, LEGEND_SAMPLE};
use super::png::encode_rgb8;

/// Pixels per pixel of a glyph.
const TEXT_SCALE: i32 = 2;
/// Size of the text: its height is that of a capital (the parts of letters below the
/// line reach into the space between the lines).
const TEXT: TextMetrics = TextMetrics {
    char_width: (GLYPH_ADVANCE as i32 * TEXT_SCALE) as f64,
    height: (GLYPH_ROWS_ABOVE as i32 * TEXT_SCALE) as f64,
};
/// Length of a tick mark.
const TICK: i32 = 5;

/// Rasterize `fig` and encode it as PNG.
pub fn render_png(fig: &PlotFigure) -> Vec<u8> {
    let (width, pixels) = render_pixels(fig);
    encode_rgb8(
        width as u32,
        (pixels.len() / (3 * width.max(1))) as u32,
        &pixels,
    )
}

/// The pixels of `fig` (red, green, blue, row by row) and the width of a row.
pub(crate) fn render_pixels(fig: &PlotFigure) -> (usize, Vec<u8>) {
    let mut canvas = RasterCanvas::new(fig.width, fig.height, Rgb::WHITE);
    let layout = Layout::of(fig, TEXT);
    let px = |v: f64| v.round() as i32;
    let (x0, y0) = (px(layout.left), px(layout.top));
    let (x1, y1) = (
        px(layout.left + layout.width),
        px(layout.top + layout.height),
    );

    // The colored cells, or the lines of the grid at the ticks.
    if let Some(grid) = &fig.grid {
        let (x_min, x_max, y_min, y_max) = layout.bounds;
        let colors = grid.colors();
        for row in y0..=y1 {
            let y = y_max - (y_max - y_min) * f64::from(row - y0) / f64::from((y1 - y0).max(1));
            for column in x0..=x1 {
                let x =
                    x_min + (x_max - x_min) * f64::from(column - x0) / f64::from((x1 - x0).max(1));
                let color = colors.get(grid.cell_at(x, y)).copied();
                canvas.set_pixel(column, row, color.unwrap_or(Rgb::jet(f64::NAN)));
            }
        }
    } else {
        for tick in &layout.x_ticks {
            let gx = px(layout.x(tick.value));
            canvas.draw_line(gx, y0, gx, y1, Rgb::GRID);
        }
        for tick in &layout.y_ticks {
            let gy = px(layout.y(tick.value));
            canvas.draw_line(x0, gy, x1, gy, Rgb::GRID);
        }
    }

    // The frame, the tick marks, and the numbers of the axes.
    canvas.draw_frame(x0, y0, x1, y1);
    for tick in &layout.x_ticks {
        let gx = px(layout.x(tick.value));
        canvas.draw_line(gx, y1, gx, y1 + TICK, Rgb::BLACK);
        canvas.text_centered(gx, px(layout.x_numbers_top), &tick.label, Rgb::BLACK);
    }
    for tick in &layout.y_ticks {
        let gy = px(layout.y(tick.value));
        canvas.draw_line(x0 - TICK, gy, x0, gy, Rgb::BLACK);
        let top = gy - px(TEXT.height) / 2;
        canvas.text_right(x0 - TICK - 3, top, &tick.label, Rgb::BLACK);
    }

    // The title and the labels of the axes.
    canvas.text_centered(
        px(f64::from(fig.width) * 0.5),
        px(layout.title_top),
        &fig.title,
        Rgb::BLACK,
    );
    canvas.text_centered(
        (x0 + x1) / 2,
        px(layout.x_label_top),
        &fig.x_label,
        Rgb::BLACK,
    );
    canvas.text_upwards(
        px(layout.y_label_x),
        (y0 + y1) / 2,
        &fig.y_label,
        Rgb::BLACK,
    );

    // The data, inside the frame.
    canvas.clip = Some((x0, y0, x1, y1));
    let y_zero = px(layout.y(0.0_f64.clamp(layout.bounds.2, layout.bounds.3)));
    for s in &fig.series {
        let mut prev: Option<(i32, i32)> = None;
        for (&x, &y) in s.xs.iter().zip(s.ys.iter()) {
            if !x.is_finite() || !y.is_finite() {
                prev = None;
                continue;
            }
            let (at_x, at_y) = (px(layout.x(x)), px(layout.y(y)));
            if s.style == PlotStyle::LinePoints {
                if let Some((ox, oy)) = prev {
                    canvas.draw_thick_line(ox, oy, at_x, at_y, s.color);
                }
                prev = Some((at_x, at_y));
            } else if s.style == PlotStyle::Stem {
                canvas.draw_thick_line(at_x, y_zero, at_x, at_y, s.color);
            }
            canvas.fill_circle(at_x, at_y, 3, s.color);
        }
    }
    if let Some(points) = &fig.colored {
        for (k, (&x, &y)) in points.xs.iter().zip(points.ys.iter()).enumerate() {
            if x.is_finite() && y.is_finite() {
                canvas.fill_circle(px(layout.x(x)), px(layout.y(y)), 4, points.color(k));
            }
        }
    }
    canvas.clip = None;

    // The color bar: the scale from the lowest value (bottom) to the highest (top), its
    // numbers, and what the values are.
    if let Some(bar) = &layout.bar {
        let bx0 = px(bar.x);
        let bx1 = bx0 + px(BAR_WIDTH);
        let rows = (y1 - y0).max(1);
        for row in 0..=rows {
            let t = 1.0 - f64::from(row) / f64::from(rows);
            canvas.draw_line(bx0, y0 + row, bx1, y0 + row, Rgb::jet(t));
        }
        canvas.draw_frame(bx0, y0, bx1, y1);
        for tick in &bar.ticks {
            let gy = px(layout.bar_y(tick.value));
            canvas.draw_line(bx1, gy, bx1 + 3, gy, Rgb::BLACK);
            let top = gy - px(TEXT.height) / 2;
            canvas.text(bx1 + 6, top, &tick.label, Rgb::BLACK);
        }
        canvas.text_upwards(px(bar.label_x), (y0 + y1) / 2, &bar.label, Rgb::BLACK);
    }

    // The legend.
    for entry in &layout.legend {
        let series = &fig.series[entry.series];
        canvas.legend_sample(px(entry.x), px(entry.y), series);
        let top = px(entry.y) - px(TEXT.height) / 2;
        canvas.text(
            px(entry.x + LEGEND_SAMPLE) + 6,
            top,
            &series.label,
            Rgb::BLACK,
        );
    }

    (canvas.width as usize, canvas.pixels)
}

/// RGB8 pixel buffer with line, disc, and text primitives.
struct RasterCanvas {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    /// Pixels outside this rectangle (left, top, right, bottom) are not drawn.
    clip: Option<(i32, i32, i32, i32)>,
}

impl RasterCanvas {
    fn new(width: u32, height: u32, bg: Rgb) -> Self {
        let n = (width as usize) * (height as usize);
        let mut pixels = Vec::with_capacity(n * 3);
        for _ in 0..n {
            pixels.push(bg.0);
            pixels.push(bg.1);
            pixels.push(bg.2);
        }
        Self {
            width,
            height,
            pixels,
            clip: None,
        }
    }

    #[inline]
    fn set_pixel(&mut self, x: i32, y: i32, c: Rgb) {
        if let Some((left, top, right, bottom)) = self.clip {
            if x < left || x > right || y < top || y > bottom {
                return;
            }
        }
        if x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height {
            let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 3;
            self.pixels[idx] = c.0;
            self.pixels[idx + 1] = c.1;
            self.pixels[idx + 2] = c.2;
        }
    }

    fn draw_line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, c: Rgb) {
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.set_pixel(x0, y0, c);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    fn draw_thick_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Rgb) {
        self.draw_line(x0, y0, x1, y1, c);
        self.draw_line(x0 + 1, y0, x1 + 1, y1, c);
        self.draw_line(x0, y0 + 1, x1, y1 + 1, c);
    }

    fn draw_frame(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        self.draw_line(x0, y0, x1, y0, Rgb::BLACK);
        self.draw_line(x1, y0, x1, y1, Rgb::BLACK);
        self.draw_line(x1, y1, x0, y1, Rgb::BLACK);
        self.draw_line(x0, y1, x0, y0, Rgb::BLACK);
    }

    fn fill_rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Rgb) {
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.set_pixel(x, y, c);
            }
        }
    }

    fn fill_circle(&mut self, cx: i32, cy: i32, r: i32, c: Rgb) {
        let r2 = r * r;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r2 {
                    self.set_pixel(cx + dx, cy + dy, c);
                }
            }
        }
    }

    /// The sample of a series in the legend: a piece of its line with its marker, on a
    /// dark background if its color is light.
    fn legend_sample(&mut self, x: i32, y: i32, series: &PlotSeries) {
        let right = x + LEGEND_SAMPLE as i32;
        if series.color.is_light() {
            self.fill_rect(x - 2, y - 6, right + 2, y + 6, Rgb::SLATE);
        }
        match series.style {
            PlotStyle::LinePoints => self.draw_thick_line(x, y, right, y, series.color),
            PlotStyle::Stem => {
                self.draw_thick_line((x + right) / 2, y + 5, (x + right) / 2, y, series.color)
            }
            PlotStyle::Scatter => {}
        }
        self.fill_circle((x + right) / 2, y, 3, series.color);
    }

    /// Width of `text` in pixels.
    fn text_width(text: &str) -> i32 {
        let count = text.chars().count() as i32;
        (count * GLYPH_ADVANCE as i32 - i32::from(count > 0)) * TEXT_SCALE
    }

    /// `text` with its upper left corner at `(x, y)`.
    fn text(&mut self, x: i32, y: i32, text: &str, c: Rgb) {
        for (k, character) in text.chars().enumerate() {
            let left = x + k as i32 * GLYPH_ADVANCE as i32 * TEXT_SCALE;
            for row in 0..GLYPH_ROWS {
                for column in 0..GLYPH_COLUMNS {
                    if glyph_pixel(character, column, row) {
                        let (gx, gy) = (
                            left + column as i32 * TEXT_SCALE,
                            y + row as i32 * TEXT_SCALE,
                        );
                        self.fill_rect(gx, gy, gx + TEXT_SCALE - 1, gy + TEXT_SCALE - 1, c);
                    }
                }
            }
        }
    }

    /// `text` centred on `x` with its top at `y`.
    fn text_centered(&mut self, x: i32, y: i32, text: &str, c: Rgb) {
        self.text(x - Self::text_width(text) / 2, y, text, c);
    }

    /// `text` ending at `x` with its top at `y`.
    fn text_right(&mut self, x: i32, y: i32, text: &str, c: Rgb) {
        self.text(x - Self::text_width(text), y, text, c);
    }

    /// `text` written upwards (turned a quarter to the left), centred on `(x, y)`.
    fn text_upwards(&mut self, x: i32, y: i32, text: &str, c: Rgb) {
        let left = x - (GLYPH_ROWS_ABOVE as i32 * TEXT_SCALE) / 2;
        let bottom = y + Self::text_width(text) / 2;
        for (k, character) in text.chars().enumerate() {
            let start = bottom - k as i32 * GLYPH_ADVANCE as i32 * TEXT_SCALE;
            for row in 0..GLYPH_ROWS {
                for column in 0..GLYPH_COLUMNS {
                    if glyph_pixel(character, column, row) {
                        // The rows of the glyph run to the right, its columns upwards.
                        let gx = left + row as i32 * TEXT_SCALE;
                        let gy = start - (column as i32 + 1) * TEXT_SCALE;
                        self.fill_rect(gx, gy + 1, gx + TEXT_SCALE - 1, gy + TEXT_SCALE, c);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dark(canvas: &RasterCanvas) -> Vec<(i32, i32)> {
        let mut found = Vec::new();
        for y in 0..canvas.height as i32 {
            for x in 0..canvas.width as i32 {
                let idx = ((y as usize) * (canvas.width as usize) + (x as usize)) * 3;
                if canvas.pixels[idx] < 128 {
                    found.push((x, y));
                }
            }
        }
        found
    }

    #[test]
    fn text_is_drawn_where_it_is_asked_for() {
        // "T" at (10, 20): the bar of the T is the top two rows of ten pixels, the stem
        // the two columns in its middle down to the fourteenth row.
        let mut canvas = RasterCanvas::new(60, 60, Rgb::WHITE);
        canvas.text(10, 20, "T", Rgb::BLACK);
        let found = dark(&canvas);
        assert_eq!(found.len(), 2 * 10 + 2 * 12);
        assert!(found.contains(&(10, 20)) && found.contains(&(19, 21)));
        assert!(found.contains(&(14, 33)) && found.contains(&(15, 33)));
        assert!(!found.contains(&(10, 33)));
        assert_eq!(RasterCanvas::text_width("T"), 10);
        assert_eq!(RasterCanvas::text_width("TT"), 22);
        assert_eq!(RasterCanvas::text_width(""), 0);

        // Centred and right-aligned text ends up around and before the given position.
        let mut centred = RasterCanvas::new(60, 60, Rgb::WHITE);
        centred.text_centered(30, 5, "TT", Rgb::BLACK);
        let xs: Vec<i32> = dark(&centred).iter().map(|p| p.0).collect();
        assert_eq!((xs.iter().min(), xs.iter().max()), (Some(&19), Some(&40)));
        let mut right = RasterCanvas::new(60, 60, Rgb::WHITE);
        right.text_right(50, 5, "T", Rgb::BLACK);
        assert_eq!(dark(&right).iter().map(|p| p.0).max(), Some(49));

        // Upwards: the bar of the T is at the left, the stem runs to the right, and the
        // same number of pixels is set.
        let mut up = RasterCanvas::new(60, 60, Rgb::WHITE);
        up.text_upwards(30, 30, "T", Rgb::BLACK);
        let found = dark(&up);
        assert_eq!(found.len(), 2 * 10 + 2 * 12);
        let left = found.iter().map(|p| p.0).min().unwrap_or(0);
        let column: Vec<i32> = found.iter().filter(|p| p.0 == left).map(|p| p.1).collect();
        assert_eq!(column.len(), 10);
        assert_eq!(found.iter().map(|p| p.0).max(), Some(left + 13));

        // Nothing is drawn outside the clip rectangle or the canvas.
        let mut clipped = RasterCanvas::new(60, 60, Rgb::WHITE);
        clipped.clip = Some((0, 0, 14, 59));
        clipped.text(10, 20, "T", Rgb::BLACK);
        assert!(dark(&clipped).iter().all(|p| p.0 <= 14));
        clipped.clip = None;
        clipped.text(55, 55, "TTT", Rgb::BLACK);
        clipped.fill_circle(-3, -3, 5, Rgb::BLACK);
    }
}
