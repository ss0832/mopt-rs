//! The files of a finished scan: one trajectory per curve, the figures, and the stationary
//! points of the scanned surface with the structures of the transition-state candidates.
//!
//! - `<stem>_scan_trajectory.xyz` (a curve), or `<stem>_scan_curve_<j>.xyz` for every value
//!   `j` of the other coordinates of a map (the points along the first coordinate); `.csv`
//!   for plain variables;
//! - `relaxedscan_energy_profile` (a curve) or `relaxedscan_energy_map` and
//!   `relaxedscan_energy_curves` (a map of two coordinates), `.png` and `.svg`; with bias
//!   potentials also the energies with them (`_bias`);
//! - `stationary_points.csv` (kind, point, values, the position of the stationary point of the
//!   local model between the grid points, energies) and `<stem>_ts_candidate_<n>.xyz`, the
//!   structures of the candidates for `optmain -order 1`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::fmt::Write as _;

use super::relaxedscan::{structure_text, Scan};
use super::relaxedscan_points::Structure;
use super::relaxedscan_settings::TargetKind;
use crate::core_infra::viz::color::Rgb;
use crate::core_infra::viz::figure::{PlotFigure, PlotStyle};
use crate::domain::Units;
use crate::error::MoptResult;
use crate::workflow::scan::figures::{curve_figure, map_figure, rows_figure, save, Overlay};
use crate::workflow::scan::{
    stationary_points_periodic, Candidate, CandidateKind, GridKind, ScanState,
};

/// Energies in kcal/mol above the lowest finite one (NaN stays NaN).
pub(crate) fn relative(energies: &[f64]) -> Vec<f64> {
    let lowest = energies
        .iter()
        .copied()
        .filter(|e| e.is_finite())
        .fold(f64::INFINITY, f64::min);
    energies
        .iter()
        .map(|e| (e - lowest) * Units::HARTREE_TO_KCAL_MOL)
        .collect()
}

/// The structure file of saddle candidate `n` (1-based).
pub(crate) fn candidate_file(scan: &Scan<'_>, n: usize) -> String {
    format!(
        "{}_ts_candidate_{n}.{}",
        scan.stem,
        scan.surface.extension()
    )
}

fn trajectories(
    scan: &Scan<'_>,
    state: &ScanState,
    structures: &HashMap<usize, Structure>,
) -> MoptResult<()> {
    let grid = &scan.grid;
    let shape = grid.shape();
    let curves: Vec<Vec<usize>> = if shape.len() == 1 {
        vec![(0..grid.len()).collect()]
    } else {
        let n0 = shape[0];
        (0..grid.len() / n0)
            .map(|j| (0..n0).map(|i| j * n0 + i).collect())
            .collect()
    };
    for (j, points) in curves.iter().enumerate() {
        let name = if curves.len() == 1 {
            format!("{}_scan_trajectory", scan.stem)
        } else {
            format!("{}_scan_curve_{}", scan.stem, j + 1)
        };
        let mut text = String::new();
        let mut table = String::new();
        for &p in points {
            let (Some(s), Some(r)) = (structures.get(&p), state.result(p)) else {
                continue;
            };
            let comment = format!(
                "RelaxedScan_{} {} energy={:.12}",
                p + 1,
                grid.values(p)
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                r.energy
            );
            match s {
                Structure::Atoms(_) => {
                    text.push_str(&structure_text(&scan.surface, s, &comment, r.energy))
                }
                Structure::Variables(x) => {
                    let _ = write!(table, "{},{:.12e}", p + 1, r.energy);
                    for v in x.iter() {
                        let _ = write!(table, ",{v:.12e}");
                    }
                    table.push('\n');
                }
            }
        }
        if !text.is_empty() {
            std::fs::write(scan.dir.join(format!("{name}.xyz")), text)?;
        }
        if !table.is_empty() {
            if let super::relaxedscan::Surface::Plain { names, .. } = &scan.surface {
                let header = format!("point,value,{}\n", names.join(","));
                std::fs::write(scan.dir.join(format!("{name}.csv")), header + &table)?;
            }
        }
    }
    Ok(())
}

/// The reference path on a map whose axes are PCo1 and PCo2.
fn reference_overlay(scan: &Scan<'_>) -> Option<Overlay> {
    let refs = scan.resper.as_ref()?;
    let kinds: Vec<Option<usize>> = scan
        .config
        .scan
        .targets
        .iter()
        .map(|t| match t.kind {
            TargetKind::Resper(k) => Some(k),
            _ => None,
        })
        .collect();
    let coordinate = |k: usize| -> Vec<f64> {
        if k == 0 {
            refs.path.0.clone()
        } else {
            refs.path.1.clone()
        }
    };
    match kinds.as_slice() {
        [Some(a), Some(b), ..] => Some(Overlay {
            label: "reference path".into(),
            xs: coordinate(*a),
            ys: coordinate(*b),
            color: Rgb::BLACK,
            style: PlotStyle::LinePoints,
        }),
        _ => None,
    }
}

/// What [`write_all`] made: the stationary points and the figures (caption, figure).
pub(crate) struct Written {
    /// The stationary points.
    pub(crate) candidates: Vec<Candidate>,
    /// The figures with what they show.
    pub(crate) figures: Vec<(String, PlotFigure)>,
}

fn figures(
    scan: &Scan<'_>,
    state: &ScanState,
    candidates: &[Candidate],
) -> MoptResult<Vec<(String, PlotFigure)>> {
    let mut out = Vec::new();
    let grid = &scan.grid;
    let raw = relative(&state.energies(grid, false));
    let biased_raw = state.energies(grid, true);
    let has_bias = state
        .done
        .iter()
        .any(|p| p.energy.is_finite() && (p.biased - p.energy).abs() > 1e-12);
    let biased = relative(&biased_raw);
    let title = if scan.config.scan.relax {
        "Relaxed scan"
    } else {
        "Rigid scan (single points)"
    };
    match grid.shape().len() {
        1 => {
            let figure = curve_figure(
                title,
                grid,
                &raw,
                has_bias.then_some(biased.as_slice()),
                candidates,
            );
            save(&figure, &scan.dir, "relaxedscan_energy_profile")?;
            out.push((
                "The energy along the scan (kcal/mol above its lowest point).".to_string(),
                figure,
            ));
        }
        2 => {
            let mut overlays: Vec<Overlay> = reference_overlay(scan).into_iter().collect();
            // The smaller of the two sets is marked: the points beyond the references, or
            // those among them (a path covers a thin band of its map).
            let (beyond, among): (Vec<usize>, Vec<usize>) = state
                .done
                .iter()
                .filter(|p| !p.skipped)
                .map(|p| p.point)
                .partition(|&p| state.result(p).is_some_and(|r| r.extrapolated));
            let marked = if beyond.is_empty() {
                None
            } else if beyond.len() <= among.len() {
                Some(("beyond the references", beyond))
            } else {
                Some(("among the references", among))
            };
            if let Some((label, points)) = marked {
                let (xs, ys) = points
                    .iter()
                    .map(|&p| {
                        let v = grid.values(p);
                        (v[0], v[1])
                    })
                    .unzip();
                overlays.push(Overlay {
                    label: label.into(),
                    xs,
                    ys,
                    color: Rgb::SLATE,
                    style: PlotStyle::Scatter,
                });
            }
            let shown = if has_bias { &raw } else { &biased };
            let figure = map_figure(title, grid, shown, candidates, &overlays);
            save(&figure, &scan.dir, "relaxedscan_energy_map")?;
            out.push((
                "The energy map (kcal/mol above its lowest point; bilinear between the grid \
                 points, grey where a point is missing)."
                    .to_string(),
                figure,
            ));
            if has_bias {
                let figure = map_figure(
                    &format!("{title} with the bias potentials"),
                    grid,
                    &biased,
                    candidates,
                    &overlays,
                );
                save(&figure, &scan.dir, "relaxedscan_energy_map_bias")?;
                out.push(("The same with the bias potentials.".to_string(), figure));
            }
            let rows = rows_figure(title, grid, shown);
            save(&rows, &scan.dir, "relaxedscan_energy_curves")?;
            out.push((
                "The curves of the map along its first coordinate.".to_string(),
                rows,
            ));
        }
        n => scan.log.info(&format!(
            "No figures of a scan of {n} coordinates (the tables have every point)."
        )),
    }
    Ok(out)
}

/// Write the trajectories, figures, and stationary points of the finished scan.
pub(crate) fn write_all(
    scan: &Scan<'_>,
    state: &ScanState,
    structures: &HashMap<usize, Structure>,
) -> MoptResult<Written> {
    trajectories(scan, state, structures)?;
    let grid = &scan.grid;
    let energies = state.energies(grid, true);
    let candidates = if scan.config.scan.candidates {
        stationary_points_periodic(&grid.shape(), &energies, &scan.periodic_dims())?
    } else {
        Vec::new()
    };
    let figures = if scan.config.scan.figures {
        figures(scan, state, &candidates)?
    } else {
        Vec::new()
    };
    if !scan.config.scan.candidates {
        return Ok(Written {
            candidates,
            figures,
        });
    }
    let lowest = energies
        .iter()
        .copied()
        .filter(|e| e.is_finite())
        .fold(f64::INFINITY, f64::min);
    let mut table = String::from("kind,candidate,point");
    for axis in grid.axes() {
        let _ = write!(table, ",{}", axis.axis_title().replace(',', ";"));
    }
    for axis in grid.axes() {
        let _ = write!(table, ",model {}", axis.axis_title().replace(',', ";"));
    }
    table.push_str(",energy_hartree,relative_kcal_mol,model_relative_kcal_mol,file\n");
    let mut saddles = 0;
    for c in &candidates {
        let values = grid.values(c.point);
        let model: Vec<f64> = match grid.kind() {
            GridKind::Joint => {
                // Along a joint curve every coordinate moves by the same share of its step.
                grid.axes()
                    .iter()
                    .zip(&values)
                    .map(|(a, v)| v + c.offset[0] * step(&a.values))
                    .collect()
            }
            GridKind::Product => grid
                .axes()
                .iter()
                .zip(&values)
                .zip(&c.offset)
                .map(|((a, v), o)| v + o * step(&a.values))
                .collect(),
        };
        let (label, file) = if c.kind == CandidateKind::Saddle {
            saddles += 1;
            let file = candidate_file(scan, saddles);
            if let Some(r) = state.result(c.point) {
                std::fs::copy(scan.dir.join(&r.file), scan.dir.join(&file))?;
            }
            (saddles.to_string(), file)
        } else {
            (String::new(), String::new())
        };
        let _ = write!(table, "{},{label},{}", c.kind.name(), c.point + 1);
        for v in values.iter().chain(&model) {
            let _ = write!(table, ",{v:.8}");
        }
        let _ = writeln!(
            table,
            ",{:.12},{:.6},{:.6},{file}",
            c.energy,
            (c.energy - lowest) * Units::HARTREE_TO_KCAL_MOL,
            (c.model_energy - lowest) * Units::HARTREE_TO_KCAL_MOL
        );
        scan.log.info(&format!(
            "{}: point {} ({}), {:.4} kcal/mol above the lowest point{}",
            if c.kind == CandidateKind::Saddle {
                format!("Transition-state candidate {label}")
            } else {
                "Minimum".to_string()
            },
            c.point + 1,
            scan.describe_point(c.point),
            (c.energy - lowest) * Units::HARTREE_TO_KCAL_MOL,
            if file.is_empty() {
                String::new()
            } else {
                format!(" ({file})")
            }
        ));
    }
    std::fs::write(scan.dir.join("stationary_points.csv"), table)?;
    if saddles == 0 && grid.shape().len() == 1 {
        let highest = (0..energies.len())
            .filter(|&p| energies[p].is_finite())
            .max_by(|&a, &b| energies[a].total_cmp(&energies[b]));
        if let Some(p) = highest {
            scan.log.info(&format!(
                "No transition-state candidate: the highest point is point {} ({}){}.",
                p + 1,
                scan.describe_point(p),
                if p == 0 || p + 1 == energies.len() {
                    ", an end of the scan; a longer range may reach the maximum"
                } else {
                    ""
                }
            ));
        }
    }
    Ok(Written {
        candidates,
        figures,
    })
}

/// The step of evenly spaced `values`.
fn step(values: &[f64]) -> f64 {
    if values.len() > 1 {
        (values[values.len() - 1] - values[0]) / (values.len() - 1) as f64
    } else {
        0.0
    }
}
