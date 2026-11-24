//! RGB colors used by the plot renderers.
//!
//! MultiOptPy: `Visualization/visualization.py` (matplotlib colours).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

/// RGB color triplet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    /// Background.
    pub const WHITE: Self = Self(255, 255, 255);
    /// Text and frame.
    pub const BLACK: Self = Self(24, 28, 36);
    /// Grid lines.
    pub const GRID: Self = Self(226, 232, 240);
    /// Default series color.
    pub const BLUE: Self = Self(37, 99, 235);
    /// Accent color.
    pub const RED: Self = Self(220, 38, 38);
    /// Accent color.
    pub const EMERALD: Self = Self(5, 150, 105);
    /// Accent color.
    pub const AMBER: Self = Self(217, 119, 6);
    /// Accent color that stands out on a map in the colors of [`Self::jet`].
    pub const MAGENTA: Self = Self(192, 38, 211);
    /// Background of the legend sample of a light color.
    pub const SLATE: Self = Self(71, 85, 105);

    /// The color of `t` in `[0, 1]` on the blue-cyan-yellow-red scale known as "jet"
    /// (values outside are clamped; a value that is not a number is drawn grey).
    pub fn jet(t: f64) -> Self {
        if t.is_nan() {
            return Self(128, 128, 128);
        }
        let t = t.clamp(0.0, 1.0);
        let channel = |center: f64| {
            let v = (1.5 - (4.0 * t - center).abs()).clamp(0.0, 1.0);
            (255.0 * v).round() as u8
        };
        Self(channel(3.0), channel(2.0), channel(1.0))
    }

    /// Whether the color is too light to be seen on a white background.
    pub fn is_light(self) -> bool {
        // Perceived brightness (ITU-R BT.601 weights) above four fifths of white.
        299 * u32::from(self.0) + 587 * u32::from(self.1) + 114 * u32::from(self.2) > 204_000
    }

    /// `#rrggbb` notation.
    pub fn to_hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }
}
