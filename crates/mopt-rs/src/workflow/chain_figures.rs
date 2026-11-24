//! Figures of a chain relaxation, drawn anew after every iteration so that the relaxation
//! can be watched while it runs:
//!
//! - `energy_profile`: the energy of the images against the length of the chain, one line
//!   per iteration (blue for the first, red for the last; the legend names them);
//! - `image_forces`: the largest force component of every image in the last iteration;
//! - `convergence`: the largest force component of the inner images against the iteration;
//! - `chain_on_surface` (chains of two variables with a surface to draw): the chains of
//!   the iterations on a map of the surface (the first chain, the last one, and those
//!   between in three colors).
//!
//! Every figure has its title, the labels and numbers of its axes, a legend where it has
//! more than one kind of line, and the numbers of its color bar. Each is written as PNG
//! and SVG; with `keep_every_iteration` the PNG of every iteration is also kept under
//! `figures/` with its number.
//!
//! MultiOptPy: `Visualization/visualization.py:NEBVisualizer` (`-spng`: energy and force
//! plots of every iteration).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use faer::Col;

use super::chain::{ChainObserver, Flow};
use crate::core_infra::viz::color::Rgb;
use crate::core_infra::viz::figure::{ColorGrid, PlotFigure, PlotSeries, PlotStyle};
use crate::error::MoptResult;
use crate::path::band::machine::{ChainRelaxation, ChainReport};

/// Points per axis of the map of a surface unless given.
pub const MAP_POINTS: usize = 70;
/// Lines of earlier iterations kept in a figure (the first, the last, and evenly between).
const MAX_LINES: usize = 12;

/// A surface of two variables to draw under the chains.
pub struct SurfaceMap {
    /// Lower and upper limit of the first variable.
    pub x: (f64, f64),
    /// Lower and upper limit of the second variable.
    pub y: (f64, f64),
    /// Values above this are drawn at this value (`None`: no limit).
    pub ceiling: Option<f64>,
    /// Points per axis ([`MAP_POINTS`] is a good value).
    pub points: usize,
    /// The surface.
    pub value: Box<dyn Fn(f64, f64) -> f64>,
}

/// What is kept of an iteration for the figures.
struct Snapshot {
    lengths: Vec<f64>,
    energies: Vec<f64>,
    chain: Vec<Col<f64>>,
    inner_force: f64,
}

/// Observer that draws the figures.
pub struct ChainFigures {
    directory: PathBuf,
    keep_every_iteration: bool,
    energy_unit: (f64, String),
    /// The title of the figure of the convergence and the quantity on its axis.
    convergence_title: (String, String),
    map: Option<SurfaceMap>,
    /// The map of the surface, computed when it is first drawn.
    grid: Option<ColorGrid>,
    history: Vec<Snapshot>,
}

impl ChainFigures {
    /// Figures written into `directory`.
    pub fn new(directory: impl AsRef<Path>) -> Self {
        Self {
            directory: directory.as_ref().to_path_buf(),
            keep_every_iteration: false,
            energy_unit: (1.0, "Energy (Hartree)".to_string()),
            convergence_title: (
                "Largest force component of the inner images".to_string(),
                "force".to_string(),
            ),
            map: None,
            grid: None,
            history: Vec::new(),
        }
    }

    /// Also keep the figures of every iteration under `figures/`.
    pub fn keep_every_iteration(mut self, keep: bool) -> Self {
        self.keep_every_iteration = keep;
        self
    }

    /// Energies are multiplied by `factor` and the axis is called `label`.
    pub fn energy_axis(mut self, factor: f64, label: &str) -> Self {
        self.energy_unit = (factor, label.to_string());
        self
    }

    /// The title of the figure of the convergence and the quantity on its axis
    /// (`log10 of the QUANTITY`; `force` for chains).
    pub fn convergence_title(mut self, title: &str, quantity: &str) -> Self {
        self.convergence_title = (title.to_string(), quantity.to_string());
        self
    }

    /// Draw the chains of two variables on this surface.
    pub fn on_surface(mut self, map: SurfaceMap) -> Self {
        self.map = Some(map);
        self.grid = None;
        self
    }

    /// The map of the surface as colored cells.
    fn surface_grid(&self, map: &SurfaceMap) -> ColorGrid {
        let (factor, label) = &self.energy_unit;
        let per_axis = map.points.max(2);
        ColorGrid::sample(label.clone(), map.x, map.y, (per_axis, per_axis), |x, y| {
            let value = (map.value)(x, y);
            factor * map.ceiling.map_or(value, |top| value.min(top))
        })
    }

    /// The iterations whose lines are drawn: all of them up to [`MAX_LINES`], else the
    /// first, the last, and evenly spaced ones between.
    fn shown(&self) -> Vec<usize> {
        let n = self.history.len();
        if n <= MAX_LINES {
            return (0..n).collect();
        }
        let mut picked: Vec<usize> = (0..MAX_LINES)
            .map(|k| k * (n - 1) / (MAX_LINES - 1))
            .collect();
        picked.dedup();
        picked
    }

    fn shade(&self, iteration: usize) -> Rgb {
        let last = self.history.len().saturating_sub(1).max(1);
        Rgb::jet(iteration as f64 / last as f64)
    }

    fn save(&self, figure: &PlotFigure, name: &str, iteration: usize) -> MoptResult<()> {
        let png = self.directory.join(format!("{name}.png"));
        figure.save(&png)?;
        figure.save(self.directory.join(format!("{name}.svg")))?;
        if self.keep_every_iteration {
            let kept = self.directory.join("figures");
            std::fs::create_dir_all(&kept)?;
            std::fs::copy(&png, kept.join(format!("{name}_{iteration:04}.png")))?;
        }
        Ok(())
    }

    fn energy_profile(&self) -> PlotFigure {
        let (factor, label) = &self.energy_unit;
        let reference = self.history[0].energies[0];
        let mut figure = PlotFigure::new(
            format!(
                "Energy relative to the first image (iterations 0 to {})",
                self.history.len() - 1
            ),
            "Length along the chain",
            label.clone(),
        );
        for k in self.shown() {
            let snapshot = &self.history[k];
            figure.add_series(PlotSeries {
                label: format!("iteration {k}"),
                xs: snapshot.lengths.clone(),
                ys: snapshot
                    .energies
                    .iter()
                    .map(|e| factor * (e - reference))
                    .collect(),
                color: self.shade(k),
                style: PlotStyle::LinePoints,
            });
        }
        figure
    }

    fn image_forces(&self, report: &ChainReport) -> PlotFigure {
        let mut figure = PlotFigure::new(
            format!(
                "Largest force component of the images (iteration {})",
                report.iteration
            ),
            "Image",
            "Force (Hartree/Bohr)",
        );
        figure.add_series(PlotSeries {
            label: String::new(),
            xs: (0..report.force_max.len()).map(|i| i as f64).collect(),
            ys: report.force_max.clone(),
            color: Rgb::BLUE,
            style: PlotStyle::Stem,
        });
        figure
    }

    fn convergence(&self) -> PlotFigure {
        let mut figure = PlotFigure::new(
            self.convergence_title.0.clone(),
            "Iteration",
            format!("log10 of the {} (Hartree/Bohr)", self.convergence_title.1),
        );
        figure.add_series(PlotSeries {
            label: String::new(),
            xs: (0..self.history.len()).map(|k| k as f64).collect(),
            ys: self
                .history
                .iter()
                .map(|s| s.inner_force.max(1e-300).log10())
                .collect(),
            color: Rgb::RED,
            style: PlotStyle::LinePoints,
        });
        figure
    }

    /// Draws iteration `iteration` of a path that is not a relaxed chain: the energies
    /// along it at the lengths `lengths`, its points `chain`, and the number of its
    /// convergence (`largest`; the figures of the energy, of the convergence, and of a
    /// path of two variables on the surface).
    pub fn record(
        &mut self,
        iteration: usize,
        lengths: &[f64],
        energies: &[f64],
        chain: &[Col<f64>],
        largest: f64,
    ) -> MoptResult<()> {
        self.history.push(Snapshot {
            lengths: lengths.to_vec(),
            energies: energies.to_vec(),
            chain: chain.to_vec(),
            inner_force: largest,
        });
        self.save(&self.energy_profile(), "energy_profile", iteration)?;
        self.save(&self.convergence(), "convergence", iteration)?;
        if chain.iter().all(|p| p.nrows() == 2) {
            if let (None, Some(map)) = (&self.grid, &self.map) {
                self.grid = Some(self.surface_grid(map));
            }
            if let Some(grid) = &self.grid {
                self.save(&self.chain_on_surface(grid), "chain_on_surface", iteration)?;
            }
        }
        Ok(())
    }

    fn chain_on_surface(&self, grid: &ColorGrid) -> PlotFigure {
        let last = self.history.len() - 1;
        let mut figure = PlotFigure::new(
            format!("The chain on the surface (iterations 0 to {last})"),
            "x",
            "y",
        );
        figure.set_color_grid(grid.clone());
        // The first chain, those between (one entry in the legend for all), the last.
        let mut between_named = false;
        for k in self.shown() {
            let chain = &self.history[k].chain;
            let (label, color) = if k == last {
                (format!("iteration {k} (latest)"), Rgb::BLACK)
            } else if k == 0 {
                ("iteration 0 (start)".to_string(), Rgb::MAGENTA)
            } else if between_named {
                (String::new(), Rgb::WHITE)
            } else {
                between_named = true;
                ("iterations between".to_string(), Rgb::WHITE)
            };
            figure.add_series(PlotSeries {
                label,
                xs: chain.iter().map(|p| p[0]).collect(),
                ys: chain.iter().map(|p| p[1]).collect(),
                color,
                style: PlotStyle::LinePoints,
            });
        }
        figure
    }
}

impl ChainObserver for ChainFigures {
    fn after_iteration(
        &mut self,
        report: &ChainReport,
        _machine: &mut ChainRelaxation,
    ) -> MoptResult<Flow> {
        let images = report.force_max.len();
        let inner = if images > 2 {
            &report.force_max[1..images - 1]
        } else {
            &report.force_max[..]
        };
        self.record(
            report.iteration,
            &report.lengths,
            &report.energies,
            &report.chain,
            inner.iter().fold(0.0_f64, |m, v| m.max(*v)),
        )?;
        self.save(&self.image_forces(report), "image_forces", report.iteration)?;
        Ok(Flow::Continue)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::path::band::machine::{ChainRelaxation, ChainSettings};
    use crate::path::band::trust::FixedEnds;
    use crate::path::band::ChainKind;
    use crate::pes::objective::{MullerBrown2D, Objective};
    use crate::workflow::chain::{relax_chain, ObjectiveSurface};

    #[test]
    fn figures_are_written_after_every_iteration() {
        let dir = std::env::temp_dir().join(format!("mopt_chain_figures_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let surface = MullerBrown2D::default();
        let chain: Vec<Col<f64>> = (0..7)
            .map(|k| {
                let t = k as f64 / 6.0;
                Col::from_fn(2, |c| [-0.558 + t * 0.508, 1.442 - t * 0.975][c])
            })
            .collect();
        let settings = ChainSettings {
            kind: ChainKind::Variables,
            max_iterations: 14,
            ends: FixedEnds::from_code(3),
            ..ChainSettings::default()
        };
        let mut machine = ChainRelaxation::new(settings, chain).unwrap();
        let drawn = MullerBrown2D::default();
        let mut figures = ChainFigures::new(&dir)
            .keep_every_iteration(true)
            .energy_axis(1.0e3, "Energy (mHartree)")
            .on_surface(SurfaceMap {
                x: (-1.5, 1.2),
                y: (-0.3, 2.0),
                ceiling: Some(0.1),
                points: 8,
                value: Box::new(move |x, y| {
                    drawn
                        .evaluate(&Col::from_fn(2, |c| [x, y][c]), false)
                        .map_or(f64::NAN, |v| v.value)
                }),
            });
        relax_chain(
            &mut machine,
            &ObjectiveSurface(&surface),
            &mut [&mut figures],
        )
        .unwrap();
        for name in [
            "energy_profile",
            "image_forces",
            "convergence",
            "chain_on_surface",
        ] {
            for extension in ["png", "svg"] {
                let path = dir.join(format!("{name}.{extension}"));
                assert!(std::fs::metadata(&path).unwrap().len() > 500, "{path:?}");
            }
            assert!(dir
                .join("figures")
                .join(format!("{name}_0013.png"))
                .is_file());
        }
        let svg = std::fs::read_to_string(dir.join("energy_profile.svg")).unwrap();
        assert!(svg.contains("iterations 0 to 13"));
        // What the lines are, and what the axes show.
        for text in [
            ">iteration 0<",
            ">iteration 13<",
            ">Energy (mHartree)<",
            ">Length along the chain<",
        ] {
            assert!(svg.contains(text), "{text} is missing");
        }
        let svg = std::fs::read_to_string(dir.join("chain_on_surface.svg")).unwrap();
        for text in [
            ">iteration 0 (start)<",
            ">iterations between<",
            ">iteration 13 (latest)<",
            ">Energy (mHartree)<",
        ] {
            assert!(svg.contains(text), "{text} is missing");
        }
        assert_eq!(svg.matches("class=\"legend\"").count(), 3);
        assert_eq!(svg.matches("class=\"cell\"").count(), 64);
        let svg = std::fs::read_to_string(dir.join("convergence.svg")).unwrap();
        assert!(svg.contains(">Iteration<") && svg.contains("log10 of the force"));
        // More iterations than lines: the first and the last are among those drawn.
        assert_eq!(figures.shown().len(), MAX_LINES);
        assert_eq!(figures.shown()[0], 0);
        assert_eq!(figures.shown()[MAX_LINES - 1], 13);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
