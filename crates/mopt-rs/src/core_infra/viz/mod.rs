//! Dependency-free scientific plotting: one figure model rendered to SVG or PNG.
//!
//! - [`figure`]: series, labels, bounds, and file output
//! - `layout`: where the title, the legend, the axes with their numbers, and the color
//!   bar go (shared by the renderers)
//! - [`svg`] / [`raster`]: renderers; [`font`]: the bitmap font of the raster renderer
//! - [`png`], [`deflate`], [`checksum`]: PNG container, zlib/DEFLATE, CRC-32/Adler-32
//!
//! MultiOptPy: `Visualization/visualization.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod checksum;
pub mod color;
pub mod deflate;
pub mod figure;
pub mod font;
pub(crate) mod layout;
pub mod png;
pub mod raster;
pub mod svg;

pub use color::Rgb;
pub use figure::{
    save_single_line_plot, ColorGrid, ColoredPoints, PlotFigure, PlotSeries, PlotStyle,
};
