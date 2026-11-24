//! The page of a scan: `relaxedscan_report.html` in its directory, one file that loads
//! nothing from elsewhere, with the figures of the scan (SVG), the stationary points and their
//! structures, the structures along the scan (played as frames), the table of the points, and
//! the settings. The structures are drawn by the viewer of the page of `run_autots`
//! ([`super::autots_report::VIEWER`], with its style [`super::autots_report::STYLE`]): turned
//! with the mouse or a finger, display styles, atom numbers, periodic cells, the light or dark
//! theme, and every structure can be saved as an xyz file.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::autots_report::{STYLE, VIEWER};
use super::relaxedscan::{RelaxedscanResult, Scan, Surface};
use super::relaxedscan_outputs::{candidate_file, relative, Written};
use crate::core_infra::json::JsonValue;
use crate::domain::Units;
use crate::error::MoptResult;
use crate::workflow::scan::{CandidateKind, GridKind, ScanState};

/// File of the page in the directory of the scan.
pub const REPORT_FILE: &str = "relaxedscan_report.html";
/// Frames of the structures along the scan kept in the page at most.
const MAX_FRAMES: usize = 400;

fn escape(text: &str) -> String {
    super::autots_report::escape(text)
}

/// The structures embedded in the page.
#[derive(Default)]
struct Models {
    entries: Vec<(String, String, String, bool)>,
}

impl Models {
    /// A viewer of `xyz` (animated when `animate`) with `caption`.
    fn viewer(&mut self, name: &str, xyz: String, animate: bool, caption: &str) -> String {
        let id = format!("m{}", self.entries.len());
        self.entries
            .push((id.clone(), name.to_string(), xyz, animate));
        format!(
            "<figure class=\"viewer-box\"><div class=\"viewer\" data-model=\"{id}\"></div>\
             <figcaption>{}</figcaption></figure>",
            escape(caption)
        )
    }

    fn json(&self) -> String {
        JsonValue::Object(
            self.entries
                .iter()
                .map(|(id, name, xyz, animate)| {
                    let mut m = BTreeMap::new();
                    m.insert("xyz".to_string(), JsonValue::String(xyz.clone()));
                    m.insert("animate".to_string(), JsonValue::Bool(*animate));
                    m.insert("name".to_string(), JsonValue::String(name.clone()));
                    (id.clone(), JsonValue::Object(m))
                })
                .collect(),
        )
        .to_string_compact()
        .replace("</", "<\\/")
    }
}

/// Write the page of `scan`.
pub(crate) fn write(
    scan: &Scan<'_>,
    state: &ScanState,
    written: &Written,
    result: &RelaxedscanResult,
) -> MoptResult<()> {
    let grid = &scan.grid;
    let atoms = matches!(scan.surface, Surface::Atoms { .. });
    let mut models = Models::default();
    let energies = state.energies(grid, true);
    let rel = relative(&energies);
    let kind = if scan.config.scan.relax {
        "Relaxed scan"
    } else {
        "Rigid scan"
    };

    let mut figures = String::new();
    for (caption, figure) in &written.figures {
        let mut f = figure.clone();
        f.width = 760;
        f.height = 460;
        let _ = write!(
            figures,
            "<figure><div class=\"chart\">{}</div><figcaption>{}</figcaption></figure>",
            f.to_svg_string(),
            escape(caption)
        );
    }
    if figures.is_empty() {
        figures.push_str(
            "<p class=\"note\">No figures (figures=off, or more than two coordinates).</p>",
        );
    }

    // Stationary points.
    let mut stationary =
        String::from("<div class=\"table-wrap\"><table><thead><tr><th>kind</th><th>point</th>");
    for axis in grid.axes() {
        let _ = write!(stationary, "<th>{}</th>", escape(&axis.axis_title()));
    }
    stationary.push_str("<th>kcal/mol</th></tr></thead><tbody>");
    let mut saddle = 0;
    let mut viewers = String::new();
    for c in &written.candidates {
        let _ = write!(
            stationary,
            "<tr><td>{}</td><td class=\"num\">{}</td>",
            c.kind.name(),
            c.point + 1
        );
        for v in grid.values(c.point) {
            let _ = write!(stationary, "<td class=\"num\">{v:.4}</td>");
        }
        let _ = write!(
            stationary,
            "<td class=\"num\">{:.3}</td></tr>",
            rel[c.point]
        );
        if c.kind == CandidateKind::Saddle {
            saddle += 1;
            let file = candidate_file(scan, saddle);
            if atoms {
                if let Ok(xyz) = std::fs::read_to_string(scan.dir.join(&file)) {
                    viewers.push_str(&models.viewer(
                        &file.replace(".xyz", ""),
                        xyz,
                        false,
                        &format!(
                            "Transition-state candidate {saddle}: point {} ({}).",
                            c.point + 1,
                            scan.describe_point(c.point)
                        ),
                    ));
                }
            }
        }
    }
    stationary.push_str("</tbody></table></div>");
    if written.candidates.is_empty() {
        stationary = "<p class=\"note\">No stationary points inside the scanned range.</p>".into();
    }
    let mut optimized = String::new();
    if !result.optimized.is_empty() {
        optimized.push_str(
            "<h3>Optimized candidates (optmain -order 1)</h3><div class=\"table-wrap\"><table>\
             <thead><tr><th>candidate</th><th>converged</th><th>iterations</th><th>kcal/mol</th>\
             <th>imaginary</th><th>same as</th></tr></thead><tbody>",
        );
        let lowest = energies
            .iter()
            .copied()
            .filter(|e| e.is_finite())
            .fold(f64::INFINITY, f64::min);
        for o in &result.optimized {
            let _ = write!(
                optimized,
                "<tr><td>{}</td><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{:.3}</td>\
                 <td class=\"num\">{}</td><td>{}</td></tr>",
                o.candidate,
                if o.converged {
                    "<span class=\"pill ok\">yes</span>"
                } else {
                    "<span class=\"pill bad\">no</span>"
                },
                o.iterations,
                (o.energy - lowest) * Units::HARTREE_TO_KCAL_MOL,
                o.imaginary_modes
                    .map_or_else(|| "—".into(), |m| m.to_string()),
                o.duplicate_of.map_or_else(String::new, |d| d.to_string())
            );
            if atoms {
                if let Ok(xyz) = std::fs::read_to_string(&o.file) {
                    viewers.push_str(&models.viewer(
                        &format!("{}_ts_{}", scan.stem, o.candidate),
                        xyz,
                        false,
                        &format!("Candidate {} after optmain -order 1.", o.candidate),
                    ));
                }
            }
        }
        optimized.push_str("</tbody></table></div>");
    }

    // The structures along the scan, in the order the points were taken.
    let mut along = String::new();
    if atoms {
        let mut frames = String::new();
        let order: Vec<usize> = grid
            .order()
            .iter()
            .copied()
            .filter(|p| state.result(*p).is_some_and(|r| !r.skipped))
            .collect();
        let step = order.len().div_ceil(MAX_FRAMES).max(1);
        for &p in order.iter().step_by(step) {
            if let Some(r) = state.result(p) {
                if let Ok(text) = std::fs::read_to_string(scan.dir.join(&r.file)) {
                    frames.push_str(&text);
                }
            }
        }
        if !frames.is_empty() {
            along = models.viewer(
                &format!("{}_scan", scan.stem),
                frames,
                true,
                &format!(
                    "The points in the order they were taken{} (played as frames).",
                    if grid.kind() == GridKind::Product && grid.axes().len() > 1 {
                        ", along the snake through the map"
                    } else {
                        ""
                    }
                ),
            );
        }
    }

    // The table of the points.
    let mut points = String::from("<div class=\"table-wrap\"><table><thead><tr><th>point</th>");
    for axis in grid.axes() {
        let _ = write!(points, "<th>{}</th>", escape(&axis.axis_title()));
    }
    points.push_str("<th>kcal/mol</th><th>status</th><th>iterations</th></tr></thead><tbody>");
    for (p, relative_energy) in rel.iter().enumerate() {
        let Some(r) = state.result(p) else {
            continue;
        };
        let _ = write!(points, "<tr><td class=\"num\">{}</td>", p + 1);
        for v in grid.values(p) {
            let _ = write!(points, "<td class=\"num\">{v:.4}</td>");
        }
        let status = if r.skipped {
            "<span class=\"pill\">skipped</span>"
        } else if r.extrapolated {
            "<span class=\"pill\">beyond the references</span>"
        } else if r.converged {
            "<span class=\"pill ok\">ok</span>"
        } else {
            "<span class=\"pill bad\">not converged</span>"
        };
        let _ = write!(
            points,
            "<td class=\"num\">{:.3}</td><td>{status}{}</td><td class=\"num\">{}</td></tr>",
            relative_energy,
            if r.repaired { " repaired" } else { "" },
            r.iterations
        );
    }
    points.push_str("</tbody></table></div>");
    let not_converged = state
        .done
        .iter()
        .filter(|p| !p.skipped && !p.converged)
        .count();
    let warning = if not_converged > 0 {
        format!(
            "<p class=\"warn\">{not_converged} point(s) did not converge; their energies are \
             those of the last step.</p>"
        )
    } else {
        String::new()
    };
    let references = if scan.resper.is_some() {
        "<li>Y. Tsutsumi, Y. Ono, T. Taketsugu, Chem. Commun. 2021, 57, 11734 (the reaction \
         space projector, ReSPer).</li><li>J. C. Gower, Biometrika 1968, 55, 582 (the \
         coordinates of a new structure on the map).</li>"
    } else {
        ""
    };
    let page_info = "{\"layers\":null}";
    let html = format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{kind} {name}</title>
<style>{style}</style>
</head>
<body>
<header class="topbar glass"><div class="topbar-inner">
<div class="brand">{kind} · {name}</div>
<nav aria-label="Sections"><a href="#energy">Energy</a><a href="#stationary">Stationary points</a><a href="#points">Points</a><a href="#settings">Settings</a></nav>
<div class="seg" role="group" aria-label="Theme"><button type="button" data-theme-choice="auto">Auto</button><button type="button" data-theme-choice="light">Light</button><button type="button" data-theme-choice="dark">Dark</button></div>
</div></header>
<main>
<div class="hero">
<h1>{kind} · {name}</h1>
<p>mopt-rs {version} · calculator: {calculator} · {count} point(s) · {coordinates}</p>
</div>
{warning}
<section class="card glass" id="energy"><h2>Energy</h2>{figures}</section>
<section class="card glass" id="stationary"><h2>Stationary points</h2><p class="note">From the discrete derivatives of the scanned surface: one negative curvature is a transition-state candidate (a start for optmain -order 1), none a minimum.</p>{stationary}{optimized}{viewers}</section>
<section class="card glass" id="points"><h2>Points</h2>{along}{points}</section>
<section class="card glass" id="settings"><h2>Settings</h2><details><summary>settings.json of the scan</summary><pre>{settings}</pre></details><ul>{references}</ul><p class="note">Directory: {directory}. The files of every point are in it; the structures of this page are embedded in it.</p></section>
</main>
<script type="application/json" id="models">{models}</script>
<script type="application/json" id="page-info">{page_info}</script>
{viewer}
</body>
</html>
"##,
        name = escape(&scan.stem),
        style = STYLE,
        viewer = VIEWER,
        version = crate::VERSION,
        calculator = escape(&scan.config.optmain.calculator),
        count = state.done.iter().filter(|p| !p.skipped).count(),
        coordinates = escape(
            &grid
                .axes()
                .iter()
                .map(|a| a.axis_title())
                .collect::<Vec<_>>()
                .join(" × ")
        ),
        settings = escape(&scan.config.settings_document),
        directory = escape(
            &scan
                .dir
                .file_name()
                .map_or_else(String::new, |f| f.to_string_lossy().into_owned())
        ),
        models = models.json(),
    );
    std::fs::write(scan.dir.join(REPORT_FILE), html)?;
    Ok(())
}
