//! Figures and tables of surface hopping (`md::hopping`): for one trajectory the energies of
//! all the states over time with the state the trajectory is on and its hops
//! (`plot_surface_hopping_.png`), and for several trajectories the share of them on every
//! state over time (`surface_hopping_populations.csv` and `.png`) from their
//! `<stem>_states.csv`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::core_infra::viz::{PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::md::hopping::HopOutcome;
use crate::md::MdStepReport;

/// Colors of the states in the figures, in the order of the states.
const STATE_COLORS: [Rgb; 6] = [
    Rgb::BLUE,
    Rgb::EMERALD,
    Rgb::AMBER,
    Rgb::MAGENTA,
    Rgb::SLATE,
    Rgb(14, 116, 144),
];

fn state_color(k: usize) -> Rgb {
    STATE_COLORS[k % STATE_COLORS.len()]
}

/// The states of one trajectory step by step, for its figure.
#[derive(Debug, Default, Clone)]
pub struct HoppingSeries {
    /// The states that take part.
    pub states: Vec<usize>,
    /// Times (fs).
    pub times: Vec<f64>,
    /// The active state at every time.
    pub active: Vec<usize>,
    /// The energies of the states at every time (Hartree; rows in the order of `states`).
    pub energies: Vec<Vec<f64>>,
    /// The hops made: time (fs) and the state hopped to.
    pub hops: Vec<(f64, usize)>,
}

impl HoppingSeries {
    /// Keep the states of a step (nothing for a run without surface hopping).
    pub fn push(&mut self, r: &MdStepReport) {
        let Some(active) = r.electronic_state else {
            return;
        };
        if self.states.is_empty() {
            self.states = r.state_energies.iter().map(|(s, _)| *s).collect();
        }
        self.times.push(r.time_fs());
        self.active.push(active);
        self.energies
            .push(r.state_energies.iter().map(|(_, e)| *e).collect());
        for h in r.hops.iter().filter(|h| h.outcome == HopOutcome::Hopped) {
            self.hops.push((r.time_fs(), h.to));
        }
    }

    /// Whether there is anything to draw.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The figure: the energies of the states (eV above the lowest of the start), the
    /// energy of the state the trajectory is on (about a hundred black points), and the hops
    /// (red stems at the energy hopped to). At most `points` points per line.
    pub fn figure(&self, points: usize) -> Option<PlotFigure> {
        let first = self.energies.first()?;
        let zero = first.iter().copied().fold(f64::INFINITY, f64::min);
        let ev = |e: f64| (e - zero) * Units::HARTREE_TO_EV;
        let stride = self.times.len().div_ceil(points.max(1)).max(1);
        let rows: Vec<usize> = (0..self.times.len()).step_by(stride).collect();
        let mut fig = PlotFigure::new(
            "Surface hopping: energies relative to the lowest state at the start",
            "time (fs)",
            "energy [eV]",
        );
        for (k, state) in self.states.iter().enumerate() {
            fig.add_series(PlotSeries {
                label: format!("state {state}"),
                xs: rows.iter().map(|&i| self.times[i]).collect(),
                ys: rows.iter().map(|&i| ev(self.energies[i][k])).collect(),
                color: state_color(k),
                style: PlotStyle::LinePoints,
            });
        }
        let on = |i: usize| {
            let k = self
                .states
                .iter()
                .position(|s| *s == self.active[i])
                .unwrap_or(0);
            ev(self.energies[i][k])
        };
        // About a hundred markers, so that the lines of the states stay visible under them.
        let marks: Vec<usize> = rows
            .iter()
            .copied()
            .step_by(rows.len().div_ceil(100).max(1))
            .collect();
        fig.add_series(PlotSeries {
            label: "trajectory".to_string(),
            xs: marks.iter().map(|&i| self.times[i]).collect(),
            ys: marks.iter().map(|&i| on(i)).collect(),
            color: Rgb::BLACK,
            style: PlotStyle::Scatter,
        });
        if !self.hops.is_empty() {
            let at: Vec<usize> = self
                .hops
                .iter()
                .filter_map(|(t, _)| self.times.iter().position(|x| x == t))
                .collect();
            fig.add_series(PlotSeries {
                label: "hops".to_string(),
                xs: at.iter().map(|&i| self.times[i]).collect(),
                ys: at.iter().map(|&i| on(i)).collect(),
                color: Rgb::RED,
                style: PlotStyle::Stem,
            });
        }
        Some(fig)
    }
}

/// The `<stem>_states.csv` of one trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct StatesTable {
    /// The states (from the header).
    pub states: Vec<usize>,
    /// Steps.
    pub steps: Vec<usize>,
    /// Times (fs).
    pub times: Vec<f64>,
    /// The active state of every step.
    pub active: Vec<usize>,
    /// The energies of the states at every step (Hartree).
    pub energies: Vec<Vec<f64>>,
}

/// Read a `<stem>_states.csv`.
pub fn read_states(path: &Path) -> MoptResult<StatesTable> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| MoptError::config(format!("{}: {e}", path.display())))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| MoptError::parse(format!("{}: empty", path.display())))?;
    let states = header
        .split(',')
        .skip(3)
        .map(|name| {
            name.trim_start_matches("energy_state_")
                .trim_end_matches("_hartree")
                .parse::<usize>()
                .map_err(|e| MoptError::parse(format!("{}: column {name}: {e}", path.display())))
        })
        .collect::<MoptResult<Vec<usize>>>()?;
    let mut table = StatesTable {
        states,
        steps: Vec::new(),
        times: Vec::new(),
        active: Vec::new(),
        energies: Vec::new(),
    };
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let bad = || MoptError::parse(format!("{}: row '{line}'", path.display()));
        if f.len() < 3 {
            return Err(bad());
        }
        table.steps.push(f[0].trim().parse().map_err(|_| bad())?);
        table.times.push(f[1].trim().parse().map_err(|_| bad())?);
        table.active.push(f[2].trim().parse().map_err(|_| bad())?);
        table.energies.push(
            f[3..]
                .iter()
                .map(|x| x.trim().parse::<f64>().map_err(|_| bad()))
                .collect::<MoptResult<Vec<f64>>>()?,
        );
    }
    Ok(table)
}

/// Draw the figure of one trajectory again from its `<stem>_states.csv` and the
/// `<stem>_hops.csv` beside it, into `out/plot_surface_hopping_.png` (for runs analyzed
/// afterwards, `mdmain RUN -analyze_only`). Returns the path of the figure.
pub fn redraw(states: &Path, out: &Path) -> MoptResult<Option<PathBuf>> {
    let table = read_states(states)?;
    let name = states.to_string_lossy();
    let hops_path = PathBuf::from(format!("{}_hops.csv", name.trim_end_matches("_states.csv")));
    let mut hops = Vec::new();
    if let Ok(text) = std::fs::read_to_string(&hops_path) {
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            if f.last().map(|o| o.trim()) != Some(HopOutcome::Hopped.name()) || f.len() < 5 {
                continue;
            }
            let (Ok(step), Ok(to)) = (f[2].trim().parse::<usize>(), f[4].trim().parse::<usize>())
            else {
                continue;
            };
            if let Some(k) = table.steps.iter().position(|s| *s == step) {
                hops.push((table.times[k], to));
            }
        }
    }
    let series = HoppingSeries {
        states: table.states,
        times: table.times,
        active: table.active,
        energies: table.energies,
        hops,
    };
    let Some(fig) = series.figure(4000) else {
        return Ok(None);
    };
    let path = out.join("plot_surface_hopping_.png");
    fig.save(&path)?;
    Ok(Some(path))
}

/// The share of the trajectories on every state at every step that all of them reached.
#[derive(Debug, Clone, PartialEq)]
pub struct Populations {
    /// The states.
    pub states: Vec<usize>,
    /// Times (fs).
    pub times: Vec<f64>,
    /// `shares[i][k]`: the share on state `states[k]` at `times[i]`.
    pub shares: Vec<Vec<f64>>,
    /// Number of trajectories.
    pub trajectories: usize,
}

/// The populations of the states over the trajectories `tables` (over the steps all of them
/// have, by step number).
pub fn populations(tables: &[StatesTable]) -> MoptResult<Populations> {
    let first = tables
        .first()
        .ok_or_else(|| MoptError::config("no trajectories for the populations"))?;
    if tables.iter().any(|t| t.states != first.states) {
        return Err(MoptError::config(
            "the trajectories have different states of surface hopping",
        ));
    }
    let common = tables.iter().map(|t| t.steps.len()).min().unwrap_or(0);
    let mut shares = Vec::with_capacity(common);
    for i in 0..common {
        let step = first.steps[i];
        let mut counts = vec![0usize; first.states.len()];
        for t in tables {
            if t.steps[i] != step {
                return Err(MoptError::config(
                    "the trajectories do not have the same steps",
                ));
            }
            if let Some(k) = first.states.iter().position(|s| *s == t.active[i]) {
                counts[k] += 1;
            }
        }
        shares.push(
            counts
                .iter()
                .map(|&c| c as f64 / tables.len() as f64)
                .collect(),
        );
    }
    Ok(Populations {
        states: first.states.clone(),
        times: first.times[..common].to_vec(),
        shares,
        trajectories: tables.len(),
    })
}

impl Populations {
    /// `surface_hopping_populations.csv` and `plot_surface_hopping_populations_.png` in
    /// `dir`; returns the paths.
    pub fn write(&self, dir: &Path) -> MoptResult<(PathBuf, PathBuf)> {
        let csv = dir.join("surface_hopping_populations.csv");
        let mut text = String::from("time_fs");
        for s in &self.states {
            let _ = write!(text, ",state_{s}");
        }
        text.push('\n');
        for (t, row) in self.times.iter().zip(&self.shares) {
            let _ = write!(text, "{t}");
            for x in row {
                let _ = write!(text, ",{x}");
            }
            text.push('\n');
        }
        std::fs::write(&csv, text)?;
        let mut fig = PlotFigure::new(
            format!(
                "Populations of the states ({} trajectories)",
                self.trajectories
            ),
            "time (fs)",
            "share of the trajectories",
        );
        for (k, s) in self.states.iter().enumerate() {
            fig.add_series(PlotSeries {
                label: format!("state {s}"),
                xs: self.times.clone(),
                ys: self.shares.iter().map(|row| row[k]).collect(),
                color: state_color(k),
                style: PlotStyle::LinePoints,
            });
        }
        let png = dir.join("plot_surface_hopping_populations_.png");
        fig.save(&png)?;
        Ok((csv, png))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_populations_count_the_trajectories_on_every_state() {
        let dir = std::env::temp_dir().join(format!("mopt_sh_pop_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut tables = Vec::new();
        for (k, active) in [[2, 2, 1], [2, 1, 1], [2, 2, 2], [2, 2, 1]]
            .iter()
            .enumerate()
        {
            let path = dir.join(format!("t{k}_states.csv"));
            let mut text = String::from(
                "step,time_fs,active_state,energy_state_1_hartree,energy_state_2_hartree\n",
            );
            for (i, a) in active.iter().enumerate() {
                text.push_str(&format!("{i},{},{a},-1.0,-0.9\n", 0.5 * i as f64));
            }
            std::fs::write(&path, text).unwrap();
            tables.push(read_states(&path).unwrap());
        }
        assert_eq!(tables[0].states, vec![1, 2]);
        let p = populations(&tables).unwrap();
        assert_eq!(p.shares[0], vec![0.0, 1.0]);
        assert_eq!(p.shares[1], vec![0.25, 0.75]);
        assert_eq!(p.shares[2], vec![0.75, 0.25]);
        let (csv, png) = p.write(&dir).unwrap();
        assert!(std::fs::read_to_string(csv)
            .unwrap()
            .starts_with("time_fs,state_1,state_2\n0,0,1\n"));
        assert!(std::fs::metadata(png).unwrap().len() > 1000);
        // The figure of one trajectory from its tables.
        std::fs::write(
            dir.join("t1_hops.csv"),
            "step_of_minimum,time_fs,decided_at_step,from_state,to_state,gap_ev,crossing,             allowed,a2,b2,kinetic_energy_along_ev,probability,transmission,random,outcome
             0,0,1,2,1,0.1,sloped,true,1,1,0.1,0.9,,0.5,hop
",
        )
        .unwrap();
        let figure = redraw(&dir.join("t1_states.csv"), &dir).unwrap().unwrap();
        assert!(std::fs::metadata(figure).unwrap().len() > 1000);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
