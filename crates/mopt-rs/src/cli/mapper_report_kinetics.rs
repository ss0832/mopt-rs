//! The section "Kinetics" of the report of `run_mapper`, from the summary of the kinetics at
//! the end of the run (`kinetics/kinetics_summary.json`, [`super::mapper_kinetics`]): its
//! settings, the populations against time of the states that reach at least
//! [`SHOWN_SHARE`] (at most [`SHOWN_STATES`], on a logarithmic time axis), and the table of the
//! states at the time of interest.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;

use super::mapper_kinetics::{KINETICS_DIR, SUMMARY_FILE};
use super::mapper_report_figures::escape;
use crate::core_infra::json::JsonValue;
use crate::workflow::mapper::Network;

/// A state is drawn when its population reaches this share at some output time.
pub const SHOWN_SHARE: f64 = 0.01;

/// States drawn at most (those with the highest populations).
pub const SHOWN_STATES: usize = 8;

/// States listed in the table at most (the files have all).
const TABLE_STATES: usize = 30;

/// Colours of the curves (the palette of Okabe and Ito, readable in both themes).
const COLORS: [&str; SHOWN_STATES] = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999999", "#F0E442",
];

/// One state of the summary.
struct State {
    node: Option<usize>,
    label: String,
    populations: Vec<f64>,
}

fn states(summary: &JsonValue) -> Vec<State> {
    summary
        .get("states")
        .and_then(JsonValue::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|s| {
                    Some(State {
                        node: s.get("node").and_then(JsonValue::as_usize),
                        label: s.get("label")?.as_str()?.to_string(),
                        populations: s
                            .get("populations")?
                            .as_array()?
                            .iter()
                            .map(|p| p.as_f64().unwrap_or(f64::NAN))
                            .collect(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn share(p: f64) -> String {
    if !p.is_finite() {
        "—".to_string()
    } else if p >= 1e-3 {
        format!("{:.1} %", 100.0 * p)
    } else if p > 0.0 {
        format!("{:.1e} %", 100.0 * p)
    } else {
        "0 %".to_string()
    }
}

/// The populations of `shown` against the logarithm of `times`.
fn chart(times: &[f64], shown: &[&State]) -> String {
    if times.len() < 2 || shown.is_empty() {
        return String::new();
    }
    let (w, h) = (560.0, 250.0);
    let (left, right, top, bottom) = (54.0, 14.0, 12.0, 42.0);
    let (pw, ph) = (w - left - right, h - top - bottom);
    let lt: Vec<f64> = times.iter().map(|t| t.log10()).collect();
    let (l0, l1) = (lt[0], lt[lt.len() - 1]);
    let x = |l: f64| left + pw * (l - l0) / (l1 - l0).max(1e-12);
    let y = |p: f64| top + ph * (1.0 - p.clamp(0.0, 1.0));
    let mut svg = format!(
        "<svg class=\"netfig kinetics\" viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\
         \"Populations against time\" xmlns=\"http://www.w3.org/2000/svg\">"
    );
    for k in 0..=4 {
        let p = f64::from(k) * 0.25;
        let yy = y(p);
        let _ = write!(
            svg,
            "<line x1=\"{left:.1}\" x2=\"{:.1}\" y1=\"{yy:.1}\" y2=\"{yy:.1}\" class=\"grid\"/>\
             <text class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{:.0}</text>",
            left + pw,
            left - 5.0,
            yy + 3.5,
            100.0 * p
        );
    }
    let first = l0.ceil() as i32;
    let last = l1.floor() as i32;
    let step = ((last - first) / 6).max(1);
    let mut d = first;
    while d <= last {
        let xx = x(f64::from(d));
        let _ = write!(
            svg,
            "<text class=\"tick\" x=\"{xx:.1}\" y=\"{:.1}\" text-anchor=\"middle\">1e{d}</text>",
            top + ph + 14.0
        );
        d += step;
    }
    let _ = write!(
        svg,
        "<text class=\"axis-title\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">time (s)</text>\
         <text class=\"axis-title\" transform=\"translate(14 {:.1}) rotate(-90)\" \
         text-anchor=\"middle\">population (%)</text>",
        left + 0.5 * pw,
        h - 6.0,
        top + 0.5 * ph
    );
    for (k, s) in shown.iter().enumerate() {
        let mut line = String::new();
        for (l, p) in lt.iter().zip(&s.populations) {
            if p.is_finite() {
                let _ = write!(line, "{:.1},{:.1} ", x(*l), y(*p));
            }
        }
        let _ = write!(
            svg,
            "<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"2\" points=\"{}\"><title>{}\
             </title></polyline>",
            COLORS[k % COLORS.len()],
            line.trim_end(),
            escape(&s.label)
        );
    }
    svg.push_str("</svg>");
    let mut legend = String::from("<p class=\"note\">");
    for (k, s) in shown.iter().enumerate() {
        let _ = write!(
            legend,
            "<span style=\"color:{}\">■</span> {} ",
            COLORS[k % COLORS.len()],
            escape(&s.label)
        );
    }
    legend.push_str("</p>");
    format!("<figure>{svg}<figcaption>{legend}</figcaption></figure>")
}

/// The section of the summary `summary`, for the network `network` the page shows.
pub fn kinetics_section(summary: &JsonValue, network: &Network) -> String {
    let settings = summary.get("settings");
    let get = |key: &str| settings.and_then(|s| s.get(key));
    let words = |key: &str| {
        get(key)
            .map(|v| match v {
                JsonValue::String(s) => s.clone(),
                JsonValue::Array(items) => {
                    let list: Vec<String> = items
                        .iter()
                        .filter_map(|i| i.as_str().map(str::to_string))
                        .collect();
                    if list.is_empty() {
                        "none".to_string()
                    } else {
                        list.join(", ")
                    }
                }
                JsonValue::Number(n) => format!("{n}"),
                _ => "—".to_string(),
            })
            .unwrap_or_else(|| "—".to_string())
    };
    let start = get("start_node")
        .and_then(JsonValue::as_usize)
        .and_then(|id| network.node(id).ok().map(|n| n.label()))
        .unwrap_or_else(|| "—".to_string());
    let mut out = format!(
        "<section class=\"card glass\" id=\"kinetics\"><h2>Kinetics</h2><p class=\"note\">\
         Computed when the exploration ended: {} at {} K, the energies {} (clamp {}), the \
         solver {}, corrections {}; all population in {start} at the start, the time of \
         interest {} s.</p>",
        escape(&words("rate_theory")),
        escape(&words("temperature_K")),
        escape(&words("energy")),
        escape(&words("clamp")),
        escape(&words("solver")),
        escape(&words("corrections")),
        escape(&words("time_s")),
    );
    let jobs = summary.get("jobs").and_then(JsonValue::as_usize);
    if jobs.is_some_and(|j| j != network.jobs.len()) {
        let _ = write!(
            out,
            "<p class=\"warn\">This kinetics is that of the network after {} job(s); the \
             network shown has {}: it is computed again when the run ends.</p>",
            jobs.unwrap_or(0),
            network.jobs.len()
        );
    }
    if summary.get("computed").and_then(JsonValue::as_bool) != Some(true) {
        let reason = summary
            .get("reason")
            .and_then(JsonValue::as_str)
            .unwrap_or("not computed");
        let _ = write!(out, "<p class=\"warn\">{}.</p></section>", escape(reason));
        return out;
    }
    let times: Vec<f64> = summary
        .get("times_s")
        .and_then(JsonValue::as_array)
        .map(|t| t.iter().filter_map(JsonValue::as_f64).collect())
        .unwrap_or_default();
    let mut all = states(summary);
    let peak = |s: &State| {
        s.populations
            .iter()
            .copied()
            .filter(|p| p.is_finite())
            .fold(0.0, f64::max)
    };
    all.sort_by(|a, b| peak(b).total_cmp(&peak(a)));
    let shown: Vec<&State> = all
        .iter()
        .filter(|s| peak(s) >= SHOWN_SHARE)
        .take(SHOWN_STATES)
        .collect();
    out.push_str(&chart(&times, &shown));
    let last = |s: &State| s.populations.last().copied().unwrap_or(f64::NAN);
    let mut by_last: Vec<&State> = all.iter().collect();
    by_last.sort_by(|a, b| last(b).total_cmp(&last(a)));
    out.push_str(
        "<div class=\"table-wrap\"><table><thead><tr><th>state</th><th>population at the \
         time of interest</th><th>highest population</th></tr></thead><tbody>",
    );
    for s in by_last.iter().take(TABLE_STATES) {
        let data = s.node.map_or_else(String::new, |id| {
            format!(" data-node=\"{id}\" tabindex=\"0\"")
        });
        let _ = write!(
            out,
            "<tr{data}><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            escape(&s.label),
            share(last(s)),
            share(peak(s))
        );
    }
    out.push_str("</tbody></table></div>");
    let notes: Vec<String> = summary
        .get("notes")
        .and_then(JsonValue::as_array)
        .map(|n| {
            n.iter()
                .filter_map(|x| x.as_str().map(escape))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !notes.is_empty() {
        let _ = write!(
            out,
            "<details><summary>Notes of the kinetics ({})</summary><ul><li>{}</li></ul>\
             </details>",
            notes.len(),
            notes.join("</li><li>")
        );
    }
    let _ = write!(
        out,
        "<p class=\"note\">The files are in {KINETICS_DIR}/ of the run directory \
         (populations.csv, rates.csv, priority.csv, {SUMMARY_FILE}); <code>run_kinetics \
         RUN_DIR</code> computes the kinetics of the network with other settings.</p></section>"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(computed: bool, jobs: usize) -> JsonValue {
        let text = format!(
            r#"{{"format": "mopt-rs mapper kinetics", "version": 1, "computed": {computed},
               "reason": "the network has no transition state", "jobs": {jobs},
               "settings": {{"temperature_K": 300, "time_s": 1, "rate_theory": "eyring",
                            "energy": "electronic", "clamp": "consistent", "solver": "propagate",
                            "corrections": []}},
               "times_s": [1e-3, 1e-2, 1e-1, 1],
               "states": [
                 {{"node": 0, "label": "EQ0", "populations": [1.0, 0.9, 0.5, 0.2]}},
                 {{"node": 1, "label": "EQ1", "populations": [0.0, 0.1, 0.5, 0.795]}},
                 {{"node": 2, "label": "EQ2", "populations": [0.0, 0.0, 0.0, 0.005]}}
               ],
               "notes": ["a note"]}}"#
        );
        JsonValue::parse(&text).unwrap_or(JsonValue::Null)
    }

    #[test]
    fn the_section_draws_the_states_that_reach_a_percent() {
        let net = Network::default();
        let html = kinetics_section(&summary(true, 0), &net);
        assert!(html.contains("id=\"kinetics\""));
        // EQ0 and EQ1 are drawn; EQ2 (0.5 % at most) is only in the table.
        assert_eq!(html.matches("<polyline").count(), 2, "{html}");
        assert!(html.contains("<td>EQ2</td>"));
        // The table is ordered by the population at the time of interest.
        let (a, b) = (html.find("<td>EQ1</td>"), html.find("<td>EQ0</td>"));
        assert!(a < b);
        assert!(html.contains("79.5 %") && html.contains("a note"));
        assert!(!html.contains("class=\"warn\""));
    }

    #[test]
    fn a_kinetics_that_was_not_computed_or_is_stale_says_so() {
        let net = Network::default();
        let html = kinetics_section(&summary(false, 0), &net);
        assert!(html.contains("no transition state") && !html.contains("<svg"));
        let html = kinetics_section(&summary(true, 5), &net);
        assert!(html.contains("after 5 job(s)"), "{html}");
    }
}
