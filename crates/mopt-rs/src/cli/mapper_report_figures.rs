//! The figures of the report of `run_mapper`, drawn as inline SVG by mopt-rs: the map of the
//! network (classical multidimensional scaling of the structures), the disconnectivity graph,
//! the energy diagram, and the energy along an IRC.
//!
//! Every node of a figure carries `data-node` (and every link `data-edge`), so that the script
//! of the page can select a node in all figures at once and mark a path; where scripts do not
//! run, the figures stand as they are, with the label and energy of every node in its title.
//! The colors of lines and text are classes of the page, so the figures follow its light and
//! dark themes; the energies are colored on one scale (dark blue low, yellow high) that reads in
//! both.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::mapper_report_data::{LinkKind, View};
use crate::core_infra::viz::layout::nice_ticks;
use crate::domain::Units;

/// The smallest span of an energy axis (kcal/mol): energies closer than this (conformers of
/// one species that are mirror images or copies by a symmetry) are drawn on an axis this wide,
/// whose labels differ.
const AXIS_SPAN_KCAL: f64 = 1e-3;

/// [`AXIS_SPAN_KCAL`] in the unit of `view`.
fn axis_span(view: &View<'_>) -> f64 {
    AXIS_SPAN_KCAL / Units::HARTREE_TO_KCAL_MOL * view.options.unit.factor()
}

/// `text` safe in HTML and SVG.
pub fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Stops of the energy scale (the viridis colours of matplotlib, a scale that reads in grey
/// and for most kinds of colour blindness).
const SCALE: [(f64, [f64; 3]); 5] = [
    (0.0, [68.0, 1.0, 84.0]),
    (0.25, [59.0, 82.0, 139.0]),
    (0.5, [33.0, 145.0, 140.0]),
    (0.75, [94.0, 201.0, 98.0]),
    (1.0, [253.0, 231.0, 37.0]),
];

/// The colour of `t` (0 low, 1 high) on the energy scale.
pub fn energy_color(t: f64) -> String {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let k = SCALE
        .windows(2)
        .position(|w| t <= w[1].0)
        .unwrap_or(SCALE.len() - 2);
    let ((t0, c0), (t1, c1)) = (SCALE[k], SCALE[k + 1]);
    let u = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    let c: [u8; 3] = std::array::from_fn(|i| (c0[i] + u * (c1[i] - c0[i])).round() as u8);
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// The gradient of the energy scale, `id` for `fill="url(#id)"`, from the bottom up.
fn scale_gradient(id: &str) -> String {
    let mut out = format!("<linearGradient id=\"{id}\" x1=\"0\" y1=\"1\" x2=\"0\" y2=\"0\">");
    for (t, _) in SCALE {
        let _ = write!(
            out,
            "<stop offset=\"{t}\" stop-color=\"{}\"/>",
            energy_color(t)
        );
    }
    out.push_str("</linearGradient>");
    out
}

/// A vertical colour bar of relative energies `0..=top` at `x`, from `y0` (top) to `y1`.
fn color_bar(x: f64, y0: f64, y1: f64, top: f64, unit: &str, id: &str) -> String {
    let mut out = format!(
        "<defs>{}</defs><rect x=\"{x:.1}\" y=\"{y0:.1}\" width=\"12\" height=\"{:.1}\" \
         fill=\"url(#{id})\" class=\"frame-line\"/>",
        scale_gradient(id),
        y1 - y0
    );
    let top = top.max(1e-12);
    for t in nice_ticks(0.0, top, 5, false) {
        let y = y1 - (y1 - y0) * t.value / top;
        let _ = write!(
            out,
            "<line x1=\"{:.1}\" x2=\"{:.1}\" y1=\"{y:.1}\" y2=\"{y:.1}\" class=\"frame-line\"/>\
             <text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\">{}</text>",
            x + 12.0,
            x + 16.0,
            x + 18.0,
            y + 3.5,
            escape(&t.label)
        );
    }
    let _ = write!(
        out,
        "<text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
        x + 6.0,
        y0 - 6.0,
        escape(unit)
    );
    out
}

/// The vertical axis of relative energies `lo..=hi` at `x` (labels left of it), mapping
/// with `y`, and the grid lines to `x_end`.
fn energy_axis(x: f64, x_end: f64, lo: f64, hi: f64, y: &dyn Fn(f64) -> f64, unit: &str) -> String {
    let mut out = String::new();
    for t in nice_ticks(lo, hi, 6, false) {
        let yy = y(t.value);
        let _ = write!(
            out,
            "<line x1=\"{x:.1}\" x2=\"{x_end:.1}\" y1=\"{yy:.1}\" y2=\"{yy:.1}\" class=\"grid\"/>\
             <text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{}</text>",
            x - 5.0,
            yy + 3.5,
            escape(&t.label)
        );
    }
    let _ = write!(
        out,
        "<text class=\"axis-title\" transform=\"translate(14 {:.1}) rotate(-90)\" \
         text-anchor=\"middle\">Relative energy ({})</text>",
        0.5 * (y(lo) + y(hi)),
        escape(unit)
    );
    out
}

/// The title of a node in the figures: label, energy, and for a minimum its population.
fn node_title(view: &View<'_>, id: usize) -> String {
    let mut text = format!(
        "{} · {} {}",
        view.label_of(id),
        view.fmt(view.rel_of(id)),
        view.options.unit.label()
    );
    if let Some(p) = view.population_of(id) {
        let _ = write!(text, " · population {:.3} %", 100.0 * p);
    }
    escape(&text)
}

/// A marker of a transition state (a diamond) or crossing point (a triangle) at `(x, y)`.
fn saddle_marker(view: &View<'_>, id: usize, x: f64, y: f64, size: f64, fill: &str) -> String {
    let d = if view.is_crossing(id) {
        format!(
            "M{x:.1},{:.1} L{:.1},{:.1} L{:.1},{:.1} Z",
            y - size,
            x + size,
            y + size * 0.8,
            x - size,
            y + size * 0.8
        )
    } else {
        format!(
            "M{x:.1},{:.1} L{:.1},{y:.1} L{x:.1},{:.1} L{:.1},{y:.1} Z",
            y - size,
            x + size,
            y + size,
            x - size
        )
    };
    format!(
        "<path d=\"{d}\" class=\"node ts\" fill=\"{fill}\" data-node=\"{id}\" tabindex=\"0\" \
         role=\"button\"><title>{}</title></path>",
        node_title(view, id)
    )
}

/// The map of the network: every structure at its place by classical multidimensional scaling
/// of the mass-weighted distances between the structures (ReSPer), minima as circles, transition
/// states as diamonds, each link drawn from its two minima through its saddle. Drawn for the
/// lowest `figure_minima` minima (the script of the page draws all).
pub fn network_map(view: &View<'_>) -> String {
    match &view.map {
        Ok(m) => scatter(view, &m.plane(0, 1), true),
        Err(why) => format!(
            "<p class=\"note\">No map of the network: {}.</p>",
            escape(why)
        ),
    }
}

/// The network as a graph: minima as circles placed by a force-directed layout, joined by a
/// line for every link, its transition state (a diamond) in the middle. Drawn for the lowest
/// `figure_minima` minima (the script of the page draws all).
pub fn network_graph(view: &View<'_>) -> String {
    if view.minima.is_empty() {
        return "<p class=\"note\">No minimum yet.</p>".to_string();
    }
    scatter(view, &view.layout, false)
}

/// The nodes of the figure part at `places` (by index among the nodes shown): with axes and
/// one scale for both (`map`), or as a graph.
fn scatter(view: &View<'_>, places: &[[f64; 2]], map: bool) -> String {
    let (minima, link_ids, saddles) = view.figure_part();
    let place = |id: usize| view.index(id).and_then(|i| places.get(i).copied());
    let (w, h) = (760.0, 480.0);
    let (left, right, top, bottom) = if map {
        (58.0, 92.0, 22.0, 46.0)
    } else {
        (14.0, 92.0, 22.0, 14.0)
    };
    let (pw, ph) = (w - left - right, h - top - bottom);
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in minima.iter().chain(&saddles).filter_map(|&id| place(id)) {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    if x0 > x1 {
        return "<p class=\"note\">Nothing to draw.</p>".to_string();
    }
    let span = (x1 - x0).max(y1 - y0).max(1e-6);
    let scale =
        (pw / ((x1 - x0) + 0.12 * span).max(1e-9)).min(ph / ((y1 - y0) + 0.12 * span).max(1e-9));
    let (cx, cy) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
    let px = |x: f64| left + 0.5 * pw + (x - cx) * scale;
    let py = |y: f64| top + 0.5 * ph - (y - cy) * scale;
    let at = |id: usize| place(id).map(|p| (px(p[0]), py(p[1])));
    let e_top = view.top().max(1e-12);
    let color = |id: usize| energy_color(view.rel_of(id) / e_top);
    let mut svg = format!(
        "<svg class=\"netfig\" viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\"{}\" \
         xmlns=\"http://www.w3.org/2000/svg\">",
        if map {
            "Map of the network: the structures by classical multidimensional scaling"
        } else {
            "The network as a graph of its minima and links"
        }
    );
    if map {
        svg.push_str(&map_axes(cx, cy, scale, [left, top, pw, ph], h, &px, &py));
    }
    for &k in &link_ids {
        let l = &view.links[k];
        let (Some(a), Some(s), Some(b)) = (at(l.ends[0]), at(l.node), at(l.ends[1])) else {
            continue;
        };
        let class = if l.kind == LinkKind::Crossing {
            "edge crossing"
        } else {
            "edge"
        };
        // Through the saddle (in the graph, parallel links of one pair bend apart).
        let _ = write!(
            svg,
            "<polyline class=\"{class}\" data-edge=\"{k}\" points=\"{:.1},{:.1} {:.1},{:.1} \
             {:.1},{:.1}\"/>",
            a.0, a.1, s.0, s.1, b.0, b.1
        );
    }
    for &id in &saddles {
        if let Some((x, y)) = at(id) {
            svg.push_str(&saddle_marker(
                view,
                id,
                x,
                y,
                if map { 4.5 } else { 3.5 },
                &color(id),
            ));
        }
    }
    // The saddle points of higher order: a square above what their descents reach, joined
    // to it by dashed lines (one kept without descents: beside the node whose job found it,
    // joined by a dotted line).
    let mut beside: std::collections::BTreeMap<Vec<usize>, f64> = std::collections::BTreeMap::new();
    for mark in view.higher_saddles() {
        let targets: Vec<(f64, f64)> = mark
            .to
            .iter()
            .filter_map(|&i| places.get(i).map(|p| (px(p[0]), py(p[1]))))
            .collect();
        if targets.is_empty() {
            continue;
        }
        let n = targets.len() as f64;
        let mut x = targets.iter().map(|t| t.0).sum::<f64>() / n;
        let mut y = targets.iter().map(|t| t.1).sum::<f64>() / n - 26.0;
        if !mark.descended {
            let nth = beside.entry(mark.to.clone()).or_insert(0.0);
            *nth += 1.0;
            x += 18.0 * *nth;
            y += 26.0 - 18.0 * *nth;
        }
        let dash = if mark.descended { "4 3" } else { "1 3" };
        for (tx, ty) in &targets {
            let _ = write!(
                svg,
                "<line x1=\"{x:.1}\" y1=\"{y:.1}\" x2=\"{tx:.1}\" y2=\"{ty:.1}\" \
                 class=\"edge\" stroke-dasharray=\"{dash}\"/>"
            );
        }
        let imaginary = if mark.imaginary.is_empty() {
            String::new()
        } else {
            let w: Vec<String> = mark
                .imaginary
                .iter()
                .map(|v| format!("{:.1}i", -v))
                .collect();
            format!(" · {} imaginary: {}", w.len(), w.join(", "))
        };
        let _ = write!(
            svg,
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"11\" height=\"11\" class=\"node\" \
             fill=\"{}\" data-node=\"{id}\"><title>{} · {} {}{}</title></rect><text \
             class=\"lbl\" x=\"{:.1}\" y=\"{:.1}\">{}</text>",
            x - 5.5,
            y - 5.5,
            energy_color(mark.energy / e_top),
            escape(&mark.tag),
            view.fmt(mark.energy),
            view.options.unit.label(),
            escape(&imaginary),
            x + 8.0,
            y - 6.0,
            escape(&mark.tag),
            id = mark.id
        );
    }
    let labels = minima.len() <= 40;
    for (i, &id) in minima.iter().enumerate() {
        let Some((x, y)) = at(id) else {
            continue;
        };
        if i == 0 {
            let _ = write!(
                svg,
                "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"10\" class=\"lowest\"/>"
            );
        }
        let _ = write!(
            svg,
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"6\" class=\"node eq\" fill=\"{}\" \
             data-node=\"{id}\" tabindex=\"0\" role=\"button\"><title>{}</title></circle>",
            color(id),
            node_title(view, id)
        );
        if labels {
            let _ = write!(
                svg,
                "<text class=\"lbl\" x=\"{:.1}\" y=\"{:.1}\">{}</text>",
                x + 8.0,
                y - 7.0,
                escape(&view.label_of(id))
            );
        }
    }
    svg.push_str(&color_bar(
        w - right + 30.0,
        top + 14.0,
        top + ph,
        e_top,
        view.options.unit.label(),
        if map { "map-scale" } else { "graph-scale" },
    ));
    svg.push_str("</svg>");
    svg
}

/// The lowest of the energies drawn in a figure (0 when there is none), where its axis
/// starts.
fn lowest_drawn(energies: impl Iterator<Item = f64>) -> f64 {
    let lo = energies.fold(f64::INFINITY, f64::min);
    if lo.is_finite() {
        lo
    } else {
        0.0
    }
}

/// The frame, grid, ticks and titles of the map (both coordinates on one scale).
fn map_axes(
    cx: f64,
    cy: f64,
    scale: f64,
    [left, top, pw, ph]: [f64; 4],
    h: f64,
    px: &dyn Fn(f64) -> f64,
    py: &dyn Fn(f64) -> f64,
) -> String {
    let mut svg =
        format!("<rect x=\"{left}\" y=\"{top}\" width=\"{pw}\" height=\"{ph}\" class=\"frame\"/>");
    let (dx0, dx1) = (cx - 0.5 * pw / scale, cx + 0.5 * pw / scale);
    for t in nice_ticks(dx0, dx1, 7, false) {
        let x = px(t.value);
        let _ = write!(
            svg,
            "<line x1=\"{x:.1}\" x2=\"{x:.1}\" y1=\"{top}\" y2=\"{:.1}\" class=\"grid\"/>\
             <text class=\"tick\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
            top + ph,
            top + ph + 14.0,
            escape(&t.label)
        );
    }
    let (dy0, dy1) = (cy - 0.5 * ph / scale, cy + 0.5 * ph / scale);
    for t in nice_ticks(dy0, dy1, 6, false) {
        let y = py(t.value);
        let _ = write!(
            svg,
            "<line x1=\"{left}\" x2=\"{:.1}\" y1=\"{y:.1}\" y2=\"{y:.1}\" class=\"grid\"/>\
             <text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{}</text>",
            left + pw,
            left - 5.0,
            y + 3.5,
            escape(&t.label)
        );
    }
    let _ = write!(
        svg,
        "<text class=\"axis-title\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">PCo1 \
         (Å amu<tspan baseline-shift=\"super\" font-size=\"8\">1/2</tspan>)</text>\
         <text class=\"axis-title\" transform=\"translate(14 {:.1}) rotate(-90)\" \
         text-anchor=\"middle\">PCo2 (Å amu<tspan baseline-shift=\"super\" \
         font-size=\"8\">1/2</tspan>)</text>",
        left + 0.5 * pw,
        h - 8.0,
        top + 0.5 * ph
    );
    svg
}

/// The disconnectivity graph (Becker and Karplus 1997): every minimum a line from its energy
/// up to where its basin merges with another, at the energy of the lowest transition state that
/// joins them; the basins that nothing joins reach the top.
pub fn disconnectivity_graph(view: &View<'_>) -> String {
    let basins = &view.drawn.basins;
    let n = basins.iter().filter(|b| b.minimum.is_some()).count();
    if n == 0 {
        return "<p class=\"note\">No minimum yet.</p>".to_string();
    }
    let mut parent: Vec<Option<usize>> = vec![None; basins.len()];
    for (p, b) in basins.iter().enumerate() {
        for &c in &b.children {
            if let Some(slot) = parent.get_mut(c) {
                *slot = Some(p);
            }
        }
    }
    let (w, h) = (760.0, 460.0);
    let (left, right, top, bottom) = (64.0, 18.0, 18.0, if n <= 60 { 64.0 } else { 26.0 });
    let (pw, ph) = (w - left - right, h - top - bottom);
    let rel: Vec<f64> = basins.iter().map(|b| view.rel(b.energy)).collect();
    // The axis starts at the lowest energy drawn: 0 for the whole network, and the lowest
    // conformer for the minima of one species, which lie far above 0 close together.
    let lo = lowest_drawn(rel.iter().copied());
    let hi = rel
        .iter()
        .copied()
        .fold(lo, f64::max)
        .max(lo + axis_span(view));
    let ceiling = hi + 0.08 * (hi - lo);
    let y = |e: f64| top + ph * (1.0 - (e - lo) / (ceiling - lo));
    let x = |b: f64| left + pw * (b + 0.5) / n as f64;
    let mut svg = format!(
        "<svg class=\"netfig\" viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\"Disconnectivity \
         graph of the minima\" xmlns=\"http://www.w3.org/2000/svg\">"
    );
    svg.push_str(&energy_axis(
        left,
        left + pw,
        lo,
        ceiling,
        &y,
        view.options.unit.label(),
    ));
    for (i, b) in basins.iter().enumerate() {
        let (xb, yb) = (x(b.x), y(rel[i]));
        let y_up = parent[i].map_or(top, |p| y(rel[p]));
        // The line of a minimum is marked with it; the line above a merge belongs to no link.
        let edge = b.minimum.map_or_else(String::new, |m| {
            format!(
                " data-node=\"{}\"",
                view.minima.get(m).copied().unwrap_or(usize::MAX)
            )
        });
        let _ = write!(
            svg,
            "<line class=\"dg-line\"{edge} x1=\"{xb:.1}\" x2=\"{xb:.1}\" y1=\"{yb:.1}\" \
             y2=\"{y_up:.1}\"/>"
        );
        if let (Some(&first), Some(&last)) = (b.children.first(), b.children.last()) {
            let _ = write!(
                svg,
                "<line class=\"dg-line\" data-edge=\"{}\" x1=\"{:.1}\" x2=\"{:.1}\" \
                 y1=\"{yb:.1}\" y2=\"{yb:.1}\"/>",
                b.connection.unwrap_or(usize::MAX),
                x(basins[first].x),
                x(basins[last].x)
            );
        }
    }
    // The merges (their transition states) and the minima over the lines.
    let e_top = view.top().max(1e-12);
    for (i, b) in basins.iter().enumerate() {
        let (xb, yb) = (x(b.x), y(rel[i]));
        if let Some(l) = b.connection.and_then(|k| view.links.get(k)) {
            let fill = energy_color(view.rel_of(l.node) / e_top);
            svg.push_str(&saddle_marker(view, l.node, xb, yb, 3.5, &fill));
        }
        if let Some(id) = b.minimum.and_then(|m| view.minima.get(m).copied()) {
            let _ = write!(
                svg,
                "<circle cx=\"{xb:.1}\" cy=\"{yb:.1}\" r=\"4\" class=\"node eq\" fill=\"{}\" \
                 data-node=\"{id}\" tabindex=\"0\" role=\"button\"><title>{}</title></circle>",
                energy_color(view.rel_of(id) / e_top),
                node_title(view, id)
            );
            if n <= 60 {
                let _ = write!(
                    svg,
                    "<text class=\"lbl\" transform=\"translate({:.1} {:.1}) rotate(-60)\" \
                     text-anchor=\"end\">{}</text>",
                    xb + 3.0,
                    top + ph + 10.0,
                    escape(&view.label_of(id))
                );
            }
        }
    }
    svg.push_str("</svg>");
    svg
}

/// The energy diagram: minima side by side at their energies (lowest first), every link above
/// the middle of its two minima, joined to both.
pub fn energy_diagram(view: &View<'_>) -> String {
    let (minima, link_ids, _) = view.figure_part();
    let n = minima.len();
    if n == 0 {
        return "<p class=\"note\">No minimum yet.</p>".to_string();
    }
    let (w, h) = (760.0, 400.0);
    let (left, right, top, bottom) = (64.0, 18.0, 18.0, if n <= 40 { 40.0 } else { 18.0 });
    let (pw, ph) = (w - left - right, h - top - bottom);
    let step = pw / n as f64;
    let x_of: BTreeMap<usize, f64> = minima
        .iter()
        .enumerate()
        .map(|(k, &id)| (id, left + step * (k as f64 + 0.5)))
        .collect();
    let drawn = || {
        minima
            .iter()
            .copied()
            .chain(link_ids.iter().map(|&k| view.links[k].node))
            .map(|id| view.rel_of(id))
    };
    let lo = lowest_drawn(drawn());
    let hi = drawn().fold(lo, f64::max).max(lo + axis_span(view));
    let ceiling = hi + 0.06 * (hi - lo);
    let y = |e: f64| top + ph * (1.0 - (e - lo) / (ceiling - lo));
    let mut svg = format!(
        "<svg class=\"netfig\" viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\"The network on \
         an energy scale\" xmlns=\"http://www.w3.org/2000/svg\">"
    );
    svg.push_str(&energy_axis(
        left,
        left + pw,
        lo,
        ceiling,
        &y,
        view.options.unit.label(),
    ));
    let e_top = view.top().max(1e-12);
    let half = (0.32 * step).min(16.0);
    for &k in &link_ids {
        let l = &view.links[k];
        let (Some(&x1), Some(&x2)) = (x_of.get(&l.ends[0]), x_of.get(&l.ends[1])) else {
            continue;
        };
        let xt = if l.ends[0] == l.ends[1] {
            x1 + 0.3 * step
        } else {
            0.5 * (x1 + x2)
        };
        let _ = write!(
            svg,
            "<polyline class=\"edge\" data-edge=\"{k}\" points=\"{x1:.1},{:.1} {xt:.1},{:.1} \
             {x2:.1},{:.1}\"/>",
            y(view.rel_of(l.ends[0])),
            y(view.rel_of(l.node)),
            y(view.rel_of(l.ends[1]))
        );
        let fill = energy_color(view.rel_of(l.node) / e_top);
        svg.push_str(&saddle_marker(
            view,
            l.node,
            xt,
            y(view.rel_of(l.node)),
            3.5,
            &fill,
        ));
    }
    for &id in &minima {
        let (x, yy) = (x_of.get(&id).copied().unwrap_or(left), y(view.rel_of(id)));
        let _ = write!(
            svg,
            "<line class=\"level\" data-node=\"{id}\" x1=\"{:.1}\" x2=\"{:.1}\" y1=\"{yy:.1}\" \
             y2=\"{yy:.1}\" tabindex=\"0\" role=\"button\"><title>{}</title></line>",
            x - half,
            x + half,
            node_title(view, id)
        );
        if n <= 40 {
            let _ = write!(
                svg,
                "<text class=\"lbl\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
                top + ph + 16.0,
                escape(&view.label_of(id))
            );
        }
    }
    svg.push_str("</svg>");
    svg
}

/// The energy along an IRC, `points` as `(arc length, relative energy)`, its ends named.
pub fn profile_chart(points: &[(f64, f64)], ends: [&str; 2], unit: &str) -> String {
    if points.len() < 2 {
        return "<p class=\"note\">No IRC profile.</p>".to_string();
    }
    let (w, h) = (520.0, 230.0);
    let (left, right, top, bottom) = (64.0, 14.0, 14.0, 40.0);
    let (pw, ph) = (w - left - right, h - top - bottom);
    let (s0, s1) = (points[0].0, points[points.len() - 1].0);
    let lo = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let hi = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let pad = 0.08 * (hi - lo).max(1e-9);
    let (lo, hi) = (lo - pad, hi + pad);
    let x = |s: f64| left + pw * (s - s0) / (s1 - s0).max(1e-12);
    let y = |e: f64| top + ph * (1.0 - (e - lo) / (hi - lo));
    let mut svg = format!(
        "<svg class=\"netfig profile\" viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\"Energy \
         along the IRC\" xmlns=\"http://www.w3.org/2000/svg\">"
    );
    svg.push_str(&energy_axis(left, left + pw, lo, hi, &y, unit));
    for t in nice_ticks(s0, s1, 6, false) {
        let xx = x(t.value);
        let _ = write!(
            svg,
            "<text class=\"tick\" x=\"{xx:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
            top + ph + 14.0,
            escape(&t.label)
        );
    }
    let _ = write!(
        svg,
        "<text class=\"axis-title\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">IRC arc \
         length (amu<tspan baseline-shift=\"super\" font-size=\"8\">1/2</tspan> Bohr)</text>",
        left + 0.5 * pw,
        h - 6.0
    );
    let mut line = String::new();
    for (s, e) in points {
        let _ = write!(line, "{:.1},{:.1} ", x(*s), y(*e));
    }
    let _ = write!(
        svg,
        "<polyline class=\"curve\" points=\"{}\"/>",
        line.trim_end()
    );
    for (k, (s, e)) in [points[0], points[points.len() - 1]]
        .into_iter()
        .enumerate()
    {
        let _ = write!(
            svg,
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3.5\" class=\"end\"/><text class=\"lbl\" \
             x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"{}\">{}</text>",
            x(s),
            y(e),
            x(s) + if k == 0 { 6.0 } else { -6.0 },
            y(e) - 6.0,
            if k == 0 { "start" } else { "end" },
            escape(ends[k])
        );
    }
    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_energy_scale_runs_from_dark_blue_to_yellow() {
        assert_eq!(energy_color(0.0), "#440154");
        assert_eq!(energy_color(1.0), "#fde725");
        assert_eq!(energy_color(0.5), "#21918c");
        assert_eq!(energy_color(f64::NAN), "#fde725");
        assert_eq!(energy_color(-2.0), "#440154");
    }
}
