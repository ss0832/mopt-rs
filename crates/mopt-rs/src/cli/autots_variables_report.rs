//! The page of `run_autots` on plain variables (`autots_report.html`): the summary of the
//! transition states and the evaluations of every step, the map of the variables with the path
//! of step 1, the relaxed chain of step 2, the candidates, the transition states, and the IRCs
//! with their ends (the surface itself behind them for two variables; for more, the two
//! variables in which the points spread most), and the values along every path.
//!
//! The figures are drawn by mopt-rs as SVG; the page loads nothing from elsewhere and has the
//! style and the themes of the AutoTS report.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;

use faer::Col;

use super::autots::{AutotsConfig, AutotsResult};
use super::autots_report::{escape, STYLE, VIEWER};
use super::autots_variables::{Point, VariablesRun};
use crate::error::MoptResult;
use crate::pes::objective::Objective;

/// Points on each side of the grid of the surface behind the map.
const GRID: usize = 56;

/// The viridis colour of `t` in [0, 1].
fn viridis(t: f64) -> String {
    const STOPS: [(f64, [f64; 3]); 5] = [
        (0.0, [68.0, 1.0, 84.0]),
        (0.25, [59.0, 82.0, 139.0]),
        (0.5, [33.0, 145.0, 140.0]),
        (0.75, [94.0, 201.0, 98.0]),
        (1.0, [253.0, 231.0, 37.0]),
    ];
    let t = t.clamp(0.0, 1.0);
    let k = STOPS.iter().rposition(|(s, _)| *s <= t).unwrap_or(0).min(3);
    let (a, ca) = STOPS[k];
    let (b, cb) = STOPS[k + 1];
    let u = if b > a { (t - a) / (b - a) } else { 0.0 };
    let c: Vec<u8> = (0..3)
        .map(|j| (ca[j] + u * (cb[j] - ca[j])).round().clamp(0.0, 255.0) as u8)
        .collect();
    format!("rgb({},{},{})", c[0], c[1], c[2])
}

/// The two variables of the map: the first two with two variables, otherwise the two in
/// which `points` spread most.
fn axes_of(n: usize, points: &[&Col<f64>]) -> (usize, usize) {
    if n <= 2 {
        return (0, n.saturating_sub(1).min(1));
    }
    let spread = |a: usize| {
        let m = points.iter().map(|p| p[a]).sum::<f64>() / points.len().max(1) as f64;
        points.iter().map(|p| (p[a] - m).powi(2)).sum::<f64>()
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| spread(b).total_cmp(&spread(a)));
    let (a, b) = (order[0], order[1]);
    (a.min(b), a.max(b))
}

/// The map: the paths and points of the run on two variables (the surface behind them for
/// two variables).
fn map_svg(run: &VariablesRun, objective: Option<&dyn Objective>) -> String {
    let n = run.names.len();
    let mut all: Vec<&Col<f64>> = Vec::new();
    for p in run
        .first_path
        .iter()
        .chain(&run.chain)
        .chain(&run.candidates)
        .chain(run.saddles.iter().map(|s| &s.0))
    {
        all.push(&p.x);
    }
    for (_, path, ends) in &run.ircs {
        all.extend(path.iter().chain(ends).map(|p| &p.x));
    }
    if all.is_empty() || n == 0 {
        return "<p class=\"note\">No point to draw.</p>".to_string();
    }
    if n == 1 {
        use super::autots_report::{chart, series};
        use crate::core_infra::viz::{color::Rgb, figure::PlotStyle};
        let curve = |label: &str, path: &[Point], color| {
            series(
                label,
                path.iter().map(|p| p.x[0]).collect(),
                path.iter().map(|p| p.value).collect(),
                color,
                PlotStyle::LinePoints,
            )
        };
        let mut curves = Vec::new();
        for (label, path, color) in [
            ("Step 1", &run.first_path, Rgb::SLATE),
            ("Step 2", &run.chain, Rgb::BLUE),
        ] {
            if !path.is_empty() {
                curves.push(curve(label, path, color));
            }
        }
        for (i, path, _) in &run.ircs {
            curves.push(curve(&format!("IRC of TS{}", i + 1), path, Rgb::EMERALD));
        }
        return chart(
            "Paths on the objective",
            &run.names[0],
            "Objective value",
            curves,
        );
    }
    let (ax, ay) = axes_of(n, &all);
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in &all {
        for (k, a) in [ax, ay].into_iter().enumerate() {
            lo[k] = lo[k].min(p[a]);
            hi[k] = hi[k].max(p[a]);
        }
    }
    // One scale on both axes, with a margin of a quarter of the larger span.
    let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-6) * 1.5;
    let mid = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let (x0, y0) = (mid[0] - 0.5 * span, mid[1] - 0.5 * span);
    let (w, h, pad) = (560.0, 560.0, 46.0);
    let sx = |x: f64| pad + (x - x0) / span * (w - 2.0 * pad);
    let sy = |y: f64| h - pad - (y - y0) / span * (h - 2.0 * pad);
    let mut svg = format!(
        "<svg viewBox=\"0 0 {w} {h}\" class=\"figure\" role=\"img\" aria-label=\"The paths of \
         the run on the variables {} and {}\" style=\"max-width:100%;height:auto\">",
        escape(&run.names[ax]),
        escape(&run.names[ay])
    );
    // The surface (two variables): values from the lowest to the highest point of the run.
    let values: Vec<f64> = run
        .first_path
        .iter()
        .chain(&run.chain)
        .chain(run.saddles.iter().map(|s| &s.0))
        .map(|p| p.value)
        .filter(|v| v.is_finite())
        .collect();
    let vmin = values.iter().copied().fold(f64::INFINITY, f64::min);
    let vmax = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if let (Some(o), true) = (objective, n == 2 && vmax > vmin) {
        let cell = span / GRID as f64;
        let top = vmax + 0.25 * (vmax - vmin);
        for j in 0..GRID {
            for i in 0..GRID {
                let (x, y) = (x0 + (i as f64 + 0.5) * cell, y0 + (j as f64 + 0.5) * cell);
                let Ok(v) = o.evaluate(&Col::from_fn(2, |c| if c == 0 { x } else { y }), false)
                else {
                    continue;
                };
                if !v.value.is_finite() || v.value > top {
                    continue;
                }
                let _ = write!(
                    svg,
                    "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\" \
                     fill-opacity=\"0.55\"/>",
                    sx(x - 0.5 * cell),
                    sy(y + 0.5 * cell),
                    sx(x + 0.5 * cell) - sx(x - 0.5 * cell) + 0.3,
                    sy(y - 0.5 * cell) - sy(y + 0.5 * cell) + 0.3,
                    viridis((v.value - vmin) / (top - vmin))
                );
            }
        }
    }
    // The frame and the ticks.
    let _ = write!(
        svg,
        "<rect x=\"{pad}\" y=\"{pad}\" width=\"{}\" height=\"{}\" fill=\"none\" \
         stroke=\"var(--muted)\" stroke-width=\"1\"/>",
        w - 2.0 * pad,
        h - 2.0 * pad
    );
    for k in 0..=4 {
        let t = k as f64 / 4.0;
        let (vx, vy) = (x0 + t * span, y0 + t * span);
        let _ = write!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" text-anchor=\"middle\" \
             fill=\"var(--muted)\">{vx:.3}</text><text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" \
             text-anchor=\"end\" fill=\"var(--muted)\">{vy:.3}</text>",
            sx(vx),
            h - pad + 16.0,
            pad - 6.0,
            sy(vy) + 4.0
        );
    }
    let _ = write!(
        svg,
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\" text-anchor=\"middle\" \
         fill=\"var(--fg)\">{}</text><text x=\"14\" y=\"{:.1}\" font-size=\"12\" \
         text-anchor=\"middle\" fill=\"var(--fg)\" transform=\"rotate(-90 14 {:.1})\">{}</text>",
        w / 2.0,
        h - 8.0,
        escape(&run.names[ax]),
        h / 2.0,
        h / 2.0,
        escape(&run.names[ay])
    );
    let line = |svg: &mut String, points: &[Point], style: &str| {
        if points.len() < 2 {
            return;
        }
        let d: Vec<String> = points
            .iter()
            .map(|p| format!("{:.2},{:.2}", sx(p.x[ax]), sy(p.x[ay])))
            .collect();
        let _ = write!(
            svg,
            "<polyline points=\"{}\" fill=\"none\" {style}/>",
            d.join(" ")
        );
    };
    line(
        &mut svg,
        &run.first_path,
        "stroke=\"var(--muted)\" stroke-width=\"1.5\" stroke-dasharray=\"5 4\"",
    );
    line(
        &mut svg,
        &run.chain,
        "stroke=\"var(--accent)\" stroke-width=\"2\"",
    );
    for (_, path, _) in &run.ircs {
        line(&mut svg, path, "stroke=\"var(--fg)\" stroke-width=\"1.5\"");
    }
    for p in &run.candidates {
        let (x, y) = (sx(p.x[ax]), sy(p.x[ay]));
        let _ = write!(
            svg,
            "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"8\" height=\"8\" fill=\"var(--s-amber)\" \
             stroke=\"var(--fg)\" stroke-width=\"0.6\"/>",
            x - 4.0,
            y - 4.0
        );
    }
    for (k, (p, converged, negative, same)) in run.saddles.iter().enumerate() {
        let (x, y) = (sx(p.x[ax]), sy(p.x[ay]));
        let ok = *converged && *negative == 1 && same.is_none();
        let _ = write!(
            svg,
            "<path d=\"M{x:.2},{:.2} L{:.2},{y:.2} L{x:.2},{:.2} L{:.2},{y:.2} Z\" \
             fill=\"{}\" stroke=\"var(--fg)\" stroke-width=\"0.8\"/><text x=\"{:.2}\" \
             y=\"{:.2}\" font-size=\"11\" fill=\"var(--fg)\">TS{}</text>",
            y - 7.0,
            x + 7.0,
            y + 7.0,
            x - 7.0,
            if ok { "var(--s-red)" } else { "var(--muted)" },
            x + 9.0,
            y - 6.0,
            k + 1
        );
    }
    for (_, _, ends) in &run.ircs {
        for p in ends {
            let _ = write!(
                svg,
                "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"5\" fill=\"var(--accent)\" \
                 stroke=\"var(--fg)\" stroke-width=\"0.8\"/>",
                sx(p.x[ax]),
                sy(p.x[ay])
            );
        }
    }
    svg.push_str("</svg>");
    let which = if n > 2 {
        format!(
            " The {} variables are shown on {} and {}, the two in which the points spread most.",
            n,
            escape(&run.names[ax]),
            escape(&run.names[ay])
        )
    } else {
        String::new()
    };
    format!(
        "<figure>{svg}<figcaption>The path of step 1 (dashed), the relaxed chain of step 2 \
         (solid, accent), the candidates (squares), the optimized candidates (diamonds; red when \
         a transition state, grey otherwise), and the IRCs (lines) with their ends (circles).{}\
         {which}</figcaption></figure>",
        if n == 2 && objective.is_some() {
            " Behind them the surface on a grid, in the colors of the viridis scale from the \
             lowest to a quarter above the highest point of the run (white above)."
        } else {
            ""
        }
    )
}

/// Value profile and a frame viewer using the native variables format for downloads.
fn path_view(
    name: &str,
    title: &str,
    frame_label: &str,
    names: &[String],
    points: &[Point],
) -> String {
    use super::autots_report::{chart, series};
    use crate::core_infra::viz::color::Rgb;
    use crate::core_infra::viz::figure::PlotStyle;
    if points.is_empty() {
        return "<p class=\"note\">No path recorded.</p>".to_string();
    }
    let profile = chart(
        title,
        "Point",
        "Objective value",
        vec![series(
            "Objective",
            (1..=points.len()).map(|i| i as f64).collect(),
            points.iter().map(|p| p.value).collect(),
            Rgb::BLUE,
            PlotStyle::LinePoints,
        )],
    );
    let frames = super::autots_variables::frames_text(names, points, frame_label);
    let download = super::autots_static::data_download(
        &format!("{name}.txt"),
        &frames,
        "Download path (.txt)",
        "dl",
        "text/plain",
    );
    format!(r#"{profile}<div class="variable-path"><div class="controls">{download}</div></div>"#)
}

fn point_table(names: &[String], points: &[Point], label: &str) -> String {
    if points.is_empty() {
        return String::new();
    }
    let mut out = format!(
        "<div class=\"table-wrap\"><table><thead><tr><th>{label}</th><th>Objective value</th>"
    );
    for name in names {
        let _ = write!(out, "<th>{}</th>", escape(name));
    }
    out.push_str("</tr></thead><tbody>");
    for (i, p) in points.iter().enumerate() {
        let _ = write!(out, "<tr><td>{}</td><td>{:.8}</td>", i + 1, p.value);
        for value in p.x.iter() {
            let _ = write!(out, "<td>{value:.6}</td>");
        }
        out.push_str("</tr>");
    }
    out.push_str("</tbody></table></div>");
    out
}

/// Write the page of a run on plain variables to `path`.
pub fn write(
    config: &AutotsConfig,
    result: &AutotsResult,
    run: &VariablesRun,
    objective: Option<&dyn Objective>,
    objective_name: &str,
    path: &Path,
) -> MoptResult<()> {
    let name = config
        .input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    let mut rows = String::new();
    let mut found = 0;
    for (k, ts) in result.transition_states.iter().enumerate() {
        let is_ts = ts.converged
            && ts.imaginary_modes == Some(1)
            && ts.same_as.is_none()
            && ts.meta_irc.is_none();
        if is_ts {
            found += 1;
        }
        // An end that is not a verified minimum is marked in its cell.
        let ends: Vec<String> = ts
            .ends
            .iter()
            .enumerate()
            .map(|(j, e)| match ts.end_checks.get(j) {
                Some(super::autots::EndCheck::NotVerified { .. }) => {
                    format!("{:.8} <span class=\"warn\">(end not verified)</span>", e.1)
                }
                _ => format!("{:.8}", e.1),
            })
            .collect();
        // The start of a meta-IRC is no transition state: no barriers from it.
        let barriers: Vec<String> = ts
            .ends
            .iter()
            .filter(|_| ts.meta_irc.is_none())
            .map(|e| format!("{:.8}", ts.energy - e.1))
            .collect();
        let label = if ts.meta_irc.is_some() {
            format!("meta-IRC start {}", k + 1)
        } else {
            format!("TS{}", k + 1)
        };
        let values = run
            .saddles
            .get(k)
            .map(|(p, ..)| {
                p.x.iter()
                    .zip(&run.names)
                    .map(|(v, n)| format!("{} = {v:.6}", escape(n)))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        let _ = write!(
            rows,
            "<tr><td>{label}</td><td>{values}</td><td>{:.8}</td><td>{}</td><td>{}</td><td>{}</td>\
             <td>{}</td><td>{}</td></tr>",
            ts.energy,
            if ts.converged { "yes" } else { "no" },
            ts.imaginary_modes
                .map_or_else(|| "—".to_string(), |n| n.to_string()),
            ts.same_as.map_or_else(String::new, |j| format!("TS{j}")),
            ends.join(", "),
            barriers.join(", ")
        );
    }
    let summary = format!(
        "<div class=\"table-wrap\"><table><thead><tr><th>candidate</th><th>variables</th>\
         <th>value</th><th>converged</th><th>negative curvatures</th><th>same as</th><th>IRC \
         ends</th><th>barriers from the ends</th></tr></thead><tbody>{rows}</tbody></table>\
         </div><p class=\"note\">A transition state converged with one negative curvature of the \
         Hessian of the objective. Values are those of the objective as they are.</p>"
    );
    let first_path = path_view(
        "step1_path",
        "Value along the first path",
        "step 1 point",
        &run.names,
        &run.first_path,
    );
    let chain = path_view(
        "step2_path",
        "Value along the relaxed path",
        "step 2 image",
        &run.names,
        &run.chain,
    );
    let bands = super::autots_report::band_history(
        &result.band_profiles,
        "Path relaxation",
        "Value relative to the first image of the initial path",
        1.0,
    );
    let guesses = point_table(&run.names, &run.candidates, "Candidate");
    let mut saddles = String::new();
    for (i, (point, converged, order, same)) in run.saddles.iter().enumerate() {
        let k = i + 1;
        let status = if !converged {
            "Unconverged"
        } else if *order == 1 {
            "First-order saddle"
        } else {
            "Stationary point"
        };
        let _ = write!(
            saddles,
            "<h3>Candidate {k} · {status}</h3><p>Negative curvatures: {order}</p>"
        );
        if let Some(previous) = same {
            let _ = write!(saddles, "<p>Same as TS{}.</p>", previous + 1);
        }
        if let Some(path) = run.optimizations.get(i) {
            saddles.push_str(&path_view(
                &format!("ts{k}_optimization"),
                "TS optimization",
                "Optimization point",
                &run.names,
                path,
            ));
        }
        saddles.push_str(&point_table(
            &run.names,
            std::slice::from_ref(point),
            "Final point",
        ));
    }
    if saddles.is_empty() {
        saddles.push_str("<p class=\"note\">No optimized candidate recorded.</p>");
    }
    let mut ircs = String::new();
    for (i, path, ends) in &run.ircs {
        let k = i + 1;
        let meta = result
            .transition_states
            .get(*i)
            .and_then(|t| t.meta_irc.as_ref());
        match meta {
            Some(notice) => {
                let _ = write!(
                    ircs,
                    "<h3>meta-IRC from the input (not a transition state)</h3>\
                     <p class=\"warn\">{}</p>",
                    escape(notice)
                );
            }
            None => {
                let _ = write!(ircs, "<h3>IRC of TS {k}</h3>");
            }
        }
        ircs.push_str(&path_view(
            &format!("ts{k}_irc"),
            if meta.is_some() {
                "Value along the meta-IRC"
            } else {
                "Value along the IRC"
            },
            if meta.is_some() {
                "meta-IRC point"
            } else {
                "IRC point"
            },
            &run.names,
            path,
        ));
        ircs.push_str(&point_table(&run.names, ends, "Endpoint"));
    }
    if ircs.is_empty() {
        ircs.push_str("<p class=\"note\">No IRC recorded.</p>");
    }
    // What the run warns of (ends of an IRC that are not verified minima, among others).
    for note in &result.notes {
        let _ = write!(ircs, "<p class=\"warn\">{}</p>", escape(note));
    }
    let mut levels = Vec::new();
    for (i, ts) in result.transition_states.iter().enumerate() {
        if let [a, b] = ts.ends.as_slice() {
            if ts.converged
                && ts.imaginary_modes == Some(1)
                && ts.same_as.is_none()
                && ts.meta_irc.is_none()
            {
                levels.push((
                    [a.1, ts.energy, b.1],
                    ["End 1".into(), format!("TS{}", i + 1), "End 2".into()],
                ));
            }
        }
    }
    let mut diagram = String::new();
    if !levels.is_empty() {
        let lowest = levels
            .iter()
            .flat_map(|(v, _)| v)
            .copied()
            .fold(f64::INFINITY, f64::min);
        for (values, _) in &mut levels {
            for value in values {
                *value -= lowest;
            }
        }
        diagram = format!("<h3>Objective values relative to the lowest point</h3><div class=\"table-wrap\">{}</div>",
            super::autots_report::levels(&levels, "IRC endpoints and transition states", 1e-9, 5));
    }
    let state = match (&result.error, &result.progress) {
        (Some(e), _) => format!(
            "<p class=\"warn\">The run stopped with an error; this page shows what was done \
             before it. {}</p>",
            escape(e)
        ),
        (None, Some(stage)) => format!(
            "<p class=\"warn\">The run had not ended when this page was written ({}).</p>",
            escape(stage)
        ),
        (None, None) => String::new(),
    };
    let html = format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoTS {name}</title>
<style>{style}
.variable-path > svg {{display:block;width:100%;height:min(32vw,270px);min-height:160px;background:var(--viewer-bg);border:1px solid var(--line);border-radius:16px}}
</style>
</head>
<body>
<header class="topbar glass"><div class="topbar-inner">
<div class="brand">AutoTS · {name}</div>
<nav aria-label="Sections"><a href="#summary">Summary</a><a href="#map">Map</a><a href="#step1">Step 1</a><a href="#step2">Step 2</a><a href="#step3">Step 3</a><a href="#step4">Step 4</a><a href="#settings">Settings</a><a href="#credits">Credits</a></nav>
<div class="seg" role="group" aria-label="Theme"><button type="button" data-theme-choice="auto">Auto</button><button type="button" data-theme-choice="light">Light</button><button type="button" data-theme-choice="dark">Dark</button></div>
</div></header>
<main>
<div class="hero">
<h1>AutoTS · {name}</h1>
<p>mopt-rs {version} · objective: {objective} ({nvars} plain variables: {vars}) · {found} transition state(s) from {candidates} candidate(s) · first path: {first}</p>
{state}
</div>
<section class="card glass" id="summary"><h2>Summary</h2>{summary}{diagram}{evaluations}</section>
<section class="card glass" id="map"><h2>Map</h2>{map}</section>
<section class="card glass" id="step1"><h2>Step 1 · First path ({first})</h2>{first_path}</section>
<section class="card glass" id="step2"><h2>Step 2 · Path ({method})</h2>{chain}{bands}{guesses}</section>
<section class="card glass" id="step3"><h2>Step 3 · Transition states</h2>{saddles}</section>
<section class="card glass" id="step4"><h2>Step 4 · IRC</h2>{ircs}</section>
<section class="card glass" id="settings"><h2>Settings</h2><details><summary>The settings of the steps</summary><pre>{settings}</pre></details>
<p class="note">Working directory: {work}. Its files: the paths as frames of variables (<code>*_path.txt</code>, <code>*_irc.txt</code>), the points as variables files (<code>*_ts_final_*.txt</code>, <code>*_end_*.txt</code>, read by <code>optmain -os {objective}</code>), <code>autots_summary.csv</code>, <code>autots.log</code>.</p></section>
<section class="card glass" id="credits"><h2>Rights and credits</h2><p>This report was generated by mopt-rs {version}, Copyright © 2023–2026 ss0832, free software under the GNU General Public License, version 3 or later (GPL-3.0-or-later), with no warranty. The figures, the style and the code embedded in this page are part of mopt-rs, under the same license; the color scale is the viridis map of matplotlib (CC0). The values and other results on this page are the output of the calculation that was run; mopt-rs claims no rights to them. The four steps follow the AutoTS workflow of MultiOptPy (ss0832, GPL-3.0).</p></section>
</main>
<script type="application/json" id="models">{{}}</script>
<script type="application/json" id="page-info">{{"layers":null}}</script>
{viewer}
<script>{frames_script}</script>
</body>
</html>
"##,
        style = STYLE,
        viewer = VIEWER,
        frames_script = include_str!("autots_variable_frames.js"),
        method = escape(
            result
                .path_stage
                .as_ref()
                .map_or("not run", |stage| stage.method.as_str())
        ),
        version = crate::VERSION,
        name = escape(&name),
        objective = escape(objective_name),
        nvars = run.names.len(),
        vars = escape(&run.names.join(", ")),
        candidates = result.candidates.len(),
        first = escape(&result.first_path_method),
        evaluations = super::autots_report::evaluations_table(result),
        map = map_svg(run, objective),
        settings = escape(&super::autots_settings::text(&result.effective_settings)),
        work = escape(&result.directory.display().to_string()),
    );
    std::fs::write(path, html)?;
    Ok(())
}
