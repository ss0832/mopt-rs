//! The report of `run_automecp`: one HTML page, `automecp_report.html` in the run directory,
//! made like that of `run_autots` ([`super::autots_report`]: its style, its structure viewer,
//! its charts and its energy diagram) with the steps of a search for crossing points.
//!
//! - A summary: every crossing point (MECP or MECI) with its energy, the norm of the gradient
//!   difference, the minima of its two states and how far it lies above each; an energy
//!   diagram with each crossing point between its two minima; the table of the minima; the
//!   evaluations of the run.
//! - Step 1: for every push, the energies of the two states along the AFIR path, with the
//!   points that were taken as candidates, and the path itself (animated).
//! - Step 2: every candidate optimized to a crossing point without the force: the energies of
//!   the two states along the optimization, its end, and whether it converged; a candidate
//!   that did not is marked with a warning.
//! - Step 3: for every crossing point, the energy along the meta-IRC of each state and the
//!   whole way between its two minima through it (animated; every frame is named after its
//!   part and has the energy of the state it lies on), with the minima.
//! - The settings of the run and its files.
//!
//! The page is written from the files of the mapper run that `run_automecp` is (the network,
//! the trajectories and logs of its jobs in `mapper/jobs`, the paths in `mapper/paths`), so it
//! can be written again from a run directory. It is one file and loads nothing from elsewhere.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::autots_report::{
    chart, escape, kcal, level_diagram, logged_references, series, Viewers, LABEL_CONTROLS, STYLE,
    VIEWER,
};
use crate::core_infra::json::JsonValue;
use crate::core_infra::viz::color::Rgb;
use crate::core_infra::viz::figure::PlotStyle;
use crate::domain::Units;
use crate::error::MoptResult;
use crate::workflow::mapper::network::CrossingEdge;
use crate::workflow::mapper::{surface_name, JobStatus, Network, Node, NodeKind, SeamCheck};

/// The page, in the run directory.
pub const REPORT_FILE: &str = "automecp_report.html";

/// A frame of a trajectory of a job.
struct Frame {
    /// What it is a point of (the comment without the number of the point and its energies).
    stage: String,
    /// The energy the trajectory has for it (Hartree).
    energy: f64,
    /// The energies of the two states, on a surface of two states (Hartree).
    states: Option<[f64; 2]>,
    /// The frame as xyz text.
    text: String,
}

/// `text` without a number at its end after the word `word` (` iteration 12`).
fn without_count<'a>(text: &'a str, word: &str) -> &'a str {
    match text.rfind(word) {
        Some(at)
            if text[at + word.len()..]
                .trim()
                .chars()
                .all(|c| c.is_ascii_digit())
                && !text[at + word.len()..].trim().is_empty() =>
        {
            text[..at].trim_end()
        }
        _ => text,
    }
}

/// The frames of the xyz trajectory `path` (nothing when it cannot be read).
fn read_frames(path: &Path) -> Vec<Frame> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        if lines[at].trim().is_empty() {
            at += 1;
            continue;
        }
        let Ok(n) = lines[at].trim().parse::<usize>() else {
            break;
        };
        if at + 2 + n > lines.len() {
            break;
        }
        let comment = lines[at + 1];
        // `LABEL [states=E1,E2] energy=E`.
        let (head, energy) = match comment.rfind(" energy=") {
            Some(k) => (
                &comment[..k],
                comment[k + 8..].trim().parse::<f64>().unwrap_or(f64::NAN),
            ),
            None => (comment, f64::NAN),
        };
        let (label, states) = match head.rfind(" states=") {
            Some(k) => {
                let both: Vec<f64> = head[k + 8..]
                    .split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect();
                (
                    &head[..k],
                    match both.as_slice() {
                        [a, b] => Some([*a, *b]),
                        _ => None,
                    },
                )
            }
            None => (head, None),
        };
        let stage = without_count(without_count(label.trim(), " iteration "), " point ");
        out.push(Frame {
            stage: stage.to_string(),
            energy,
            states,
            text: lines[at..at + 2 + n].join("\n") + "\n",
        });
        at += 2 + n;
    }
    out
}

/// Read the stages in the recorded order, including runs with a single legacy XYZ file.
fn job_frames(path: &Path) -> Vec<Frame> {
    if path.file_name().and_then(|name| name.to_str()) != Some("trajectories.csv") {
        return read_frames(path);
    }
    let (Some(dir), Ok(index)) = (path.parent(), std::fs::read_to_string(path)) else {
        return Vec::new();
    };
    index
        .lines()
        .skip(1)
        .filter_map(|row| row.rsplit_once(','))
        .filter_map(|(_, filename)| {
            let file = Path::new(filename);
            (file.file_name()?.to_str()? == filename).then(|| dir.join(file))
        })
        .flat_map(|file| read_frames(&file))
        .collect()
}

/// The frames in runs of one stage: the name of each and its frames.
fn stages(frames: &[Frame]) -> Vec<(&str, &[Frame])> {
    let mut out = Vec::new();
    let mut start = 0;
    for k in 1..=frames.len() {
        if k == frames.len() || frames[k].stage != frames[start].stage {
            out.push((frames[start].stage.as_str(), &frames[start..k]));
            start = k;
        }
    }
    out
}

/// What the log of a job says of the stage `stage`: the text after `stage: `.
fn logged<'a>(log: &'a str, stage: &str) -> Option<&'a str> {
    log.lines().find_map(|l| {
        l.strip_prefix(stage)
            .and_then(|rest| rest.strip_prefix(": "))
    })
}

/// The number after `key` in `text`, up to the next comma or space.
fn number_after(text: &str, key: &str) -> Option<f64> {
    let rest = &text[text.find(key)? + key.len()..];
    rest.split([',', ' '])
        .find(|t| !t.is_empty())
        .and_then(|t| t.parse().ok())
}

/// The names of the two states of the stage `stage` (`... of PES1 and PES2 ...`): the first
/// is the state of the job, whose energy is the first of `states=E1,E2` in its frames.
fn states_of(stage: &str) -> [String; 2] {
    let named = stage.split_once(" of ").and_then(|(_, rest)| {
        let (a, rest) = rest.split_once(" and ")?;
        let b = rest.split_whitespace().next()?;
        (a.starts_with("PES") && b.starts_with("PES")).then(|| [a.to_string(), b.to_string()])
    });
    named.unwrap_or_else(|| ["state 1".to_string(), "state 2".to_string()])
}

/// The frames as one xyz text, at most `most` of them (the first, the last, and frames
/// between at equal steps).
fn xyz_of(frames: &[Frame], most: usize) -> String {
    if frames.len() <= most || most < 2 {
        return frames.iter().map(|f| f.text.as_str()).collect();
    }
    let last = frames.len() - 1;
    (0..most)
        .map(|k| frames[k * last / (most - 1)].text.as_str())
        .collect()
}

/// The most frames of one viewer.
const MAX_FRAMES: usize = 200;

/// The name of the state of `node` among the states of `network` (`PES2`).
fn surface_of(network: &Network, node: &Node) -> String {
    network
        .states
        .iter()
        .position(|s| *s == node.state)
        .map_or_else(String::new, surface_name)
}

/// The states of the run as text (`PES1: charge 0, multiplicity 1; PES2: ...`).
fn states_text(network: &Network) -> String {
    network
        .states
        .iter()
        .enumerate()
        .map(|(k, s)| {
            format!(
                "{}: charge {}, multiplicity {}{}",
                surface_name(k),
                s.charge,
                s.multiplicity,
                if s.excited_state > 0 {
                    format!(", excited state {}", s.excited_state)
                } else {
                    String::new()
                }
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The structure of `node` as an xyz frame.
fn node_xyz(node: &Node) -> String {
    super::mapper_report_data::xyz_of(node)
}

/// The crossing-point node whose energy is nearest to `energy` (Hartree), within `within`.
fn crossing_at(network: &Network, energy: f64, within: f64) -> Option<&Node> {
    network
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Crossing { .. }))
        .map(|n| ((n.energy - energy).abs(), n))
        .filter(|(d, _)| *d <= within)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, n)| n)
}

/// Whether `job` is a search for a crossing point from the minimum itself
/// (`crossing_direct`), not a push.
fn is_direct(job: &crate::workflow::mapper::network::JobRecord) -> bool {
    job.task.starts_with("crossing:")
}

/// The title of every job of `network` on the page (`Push 2`, `Direct search 1`, numbered
/// apart) and its name in the ids of the viewers (`push2`, `direct1`).
fn job_names(network: &Network) -> Vec<(String, String)> {
    let (mut pushes, mut directs) = (0, 0);
    network
        .jobs
        .iter()
        .map(|job| {
            if is_direct(job) {
                directs += 1;
                (
                    format!("Direct search {directs}"),
                    format!("direct{directs}"),
                )
            } else {
                pushes += 1;
                (format!("Push {pushes}"), format!("push{pushes}"))
            }
        })
        .collect()
}

/// The sides of the meta-IRC of the crossing edge `edge` as points of a chart: the label of
/// each side and its points (signed arc length, energy in kcal/mol from the crossing point),
/// with the crossing point itself at the origin.
fn meta_irc_points(network: &Network, edge: &CrossingEdge) -> Vec<(String, Vec<(f64, f64)>)> {
    let Some(x) = network.nodes.get(edge.crossing) else {
        return Vec::new();
    };
    // The side of each state of the crossing point: the first to the left. A side names its
    // surface (`PES2`), and a meta-IRC that gave no path on one state has one side only, so
    // the place in the list does not say which state it is on.
    let first = match &x.kind {
        NodeKind::Crossing { states, .. } => network
            .states
            .iter()
            .position(|s| s == &states[0])
            .map(surface_name),
        _ => None,
    };
    let mut out = Vec::new();
    for (k, side) in edge.irc.sides.iter().enumerate() {
        let on_first = match &first {
            Some(name) => side.name == *name,
            None => k == 0,
        };
        let sign = if on_first { -1.0 } else { 1.0 };
        let mut points: Vec<(f64, f64)> = side
            .arc_lengths
            .iter()
            .zip(&side.energies)
            .map(|(s, en)| (sign * s, Units::HARTREE_TO_KCAL_MOL * (en - x.energy)))
            .collect();
        points.push((0.0, 0.0));
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        out.push((format!("on {}", side.name), points));
    }
    out
}

/// Two crossing points are the same when their energies (the means of the two states) differ
/// by less than this (Hartree): only to name the node a candidate of step 2 converged to. The
/// mapper itself decides it by the structure; a converged candidate is one of its nodes.
const SAME_CROSSING_HARTREE: f64 = 1.0e-4;

/// Write the report of the run in `dir` (its mapper run in `mapper_dir`, with the network
/// `network`) and return its path. `name` is the name of the input, `ended` why the run
/// ended (`None`: it was stopped or has not ended).
pub fn write(
    dir: &Path,
    mapper_dir: &Path,
    name: &str,
    network: &Network,
    ended: Option<&str>,
) -> MoptResult<PathBuf> {
    let mut viewers = Viewers::new();
    let settings =
        std::fs::read_to_string(dir.join(super::automecp::MAPPER_CONFIG_FILE)).unwrap_or_default();
    let step1 = JsonValue::parse(&settings)
        .ok()
        .and_then(|d| d.as_object()?.get("step1_settings")?.as_object().cloned())
        .unwrap_or_default();
    // The calculator as the mapper run recorded it (resolved from all the options that name
    // one); without that record, the option of step1_settings that names it.
    let recorded =
        std::fs::read_to_string(mapper_dir.join(crate::workflow::mapper::store::NETWORK_FILE))
            .ok()
            .and_then(|t| JsonValue::parse(&t).ok())
            .and_then(|d| {
                d.get("run")?
                    .get("calculator")?
                    .as_str()
                    .map(str::to_string)
            });
    let calculator = recorded
        .or_else(|| {
            ["othersoft", "usextb", "usedxtb"].iter().find_map(|key| {
                step1
                    .get(*key)
                    .and_then(JsonValue::as_str)
                    .map(str::to_string)
            })
        })
        .unwrap_or_else(|| "not recorded".to_string());
    // On a variable cell the energies of the nodes are enthalpies.
    let on_cell = step1.get("variable_cell").is_some_and(|v| match v {
        JsonValue::Bool(b) => *b,
        JsonValue::String(s) => {
            ["true", "on", "yes", "1"].contains(&s.to_ascii_lowercase().as_str())
        }
        _ => false,
    });
    let quantity = if on_cell { "enthalpy" } else { "energy" };
    // The zero of the energies: the first input minimum.
    let start = network
        .inputs
        .first()
        .and_then(|i| i.node)
        .and_then(|id| network.nodes.get(id));
    let zero = start.map_or(0.0, |n| n.energy);
    let rel = |e: f64| Units::HARTREE_TO_KCAL_MOL * (e - zero);

    // Summary: the crossing points, each between its two minima.
    let mut summary = format!(
        "<table><thead><tr><th>crossing point</th><th>states</th><th>{quantity} (Hartree)</th>\
         <th>above the start</th><th>|Δg| (Hartree/Bohr)</th><th>gap (Hartree)</th>\
         <th>within the seam</th><th>minima</th>\
         <th>above minimum 1</th><th>above minimum 2</th></tr></thead><tbody>"
    );
    let mut levels: Vec<([f64; 3], [String; 3])> = Vec::new();
    let mut loose = 0;
    let mut saddles = Vec::new();
    for e in &network.crossing_edges {
        let Some(x) = network.nodes.get(e.crossing) else {
            continue;
        };
        let NodeKind::Crossing { states, .. } = &x.kind else {
            continue;
        };
        let pes = |s| {
            network
                .states
                .iter()
                .position(|t| t == s)
                .map_or_else(|| "?".to_string(), surface_name)
        };
        let end = |side: usize| e.ends[side].and_then(|id| network.nodes.get(id));
        let ends = [end(0), end(1)];
        if ends.iter().any(Option::is_none) {
            loose += 1;
        }
        let end_text = |side: usize| {
            ends[side].map_or_else(
                || "— (not found)".to_string(),
                |n| format!("{} ({})", n.label(), surface_of(network, n)),
            )
        };
        let seam = e
            .seam
            .as_ref()
            .map_or_else(|| "—".to_string(), SeamCheck::describe);
        if e.seam.as_ref().is_some_and(SeamCheck::is_saddle) && !saddles.contains(&x.label()) {
            saddles.push(x.label());
        }
        let _ = write!(
            summary,
            "<tr><td>{}</td><td>{}/{}</td><td class=\"num\">{:.8}</td><td class=\"num\">{:.2}</td>\
             <td class=\"num\">{}</td><td class=\"num\">{}</td><td>{}</td><td>{} → {}</td>\
             <td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            x.label(),
            pes(&states[0]),
            pes(&states[1]),
            x.energy,
            rel(x.energy),
            e.gradient_difference
                .map_or_else(|| "—".to_string(), |g| format!("{g:.4}")),
            e.gap
                .map_or_else(|| "—".to_string(), |g| format!("{g:.3e}")),
            escape(&seam),
            end_text(0),
            end_text(1),
            kcal(e.barriers[0].map(|b| Units::HARTREE_TO_KCAL_MOL * b)),
            kcal(e.barriers[1].map(|b| Units::HARTREE_TO_KCAL_MOL * b)),
        );
        if let [Some(a), Some(b)] = ends {
            levels.push((
                [rel(a.energy), rel(x.energy), rel(b.energy)],
                [
                    format!("{} · {}", a.label(), surface_of(network, a)),
                    x.label(),
                    format!("{} · {}", b.label(), surface_of(network, b)),
                ],
            ));
        }
    }
    let _ = write!(
        summary,
        "</tbody></table><p class=\"note\">{} above the start and above the minima in \
         kcal/mol; the {quantity} of a crossing point is the mean of its two states there{}. \
         The minima are those the meta-IRC of each state ends at (step 3).</p>",
        if on_cell { "Enthalpies" } else { "Energies" },
        if on_cell {
            " (the atoms and the cell optimized together, at the pressure of the run)"
        } else {
            ""
        }
    );
    // The count of the console and the summary: crossing-point nodes (the table has a line
    // per crossing edge).
    let found = network.crossing_points();
    if !saddles.is_empty() {
        let _ = write!(
            summary,
            "<p class=\"warn\">{} {} a saddle point on the seam by the analysis within the seam \
             (one or more imaginary modes): kept, but not a minimum-energy crossing point. To go \
             on, displace it along the imaginary mode within the seam and optimize it again \
             (freqmain -mf at the structure shows the mode).</p>",
            saddles.join(", "),
            if saddles.len() == 1 { "is" } else { "are" }
        );
    }
    if found == 0 {
        summary.push_str(
            "<p class=\"warn\">No crossing point was found: no candidate of step 2 converged \
             to one. The paths of step 1 and the candidates of step 2 are below.</p>",
        );
    }
    if loose > 0 {
        let _ = write!(
            summary,
            "<p class=\"warn\">{loose} crossing point(s) are not joined to a minimum on both \
             states: a meta-IRC gave no path, or its end did not converge to a minimum.</p>"
        );
    }
    let diagram = if levels.is_empty() {
        String::new()
    } else {
        format!(
            "<figure><div class=\"table-wrap\">{}</div><figcaption>Energies in kcal/mol above \
             the start{}. Each crossing point is drawn between the minimum of its first state \
             and that of its second.</figcaption></figure>",
            level_diagram(
                &levels,
                "Energy diagram of the crossing points and the minima of their two states"
            ),
            start.map_or_else(String::new, |n| format!(" ({})", n.label()))
        )
    };
    // The minima.
    let mut minima = format!(
        "<h3>Minima</h3><div class=\"table-wrap\"><table><thead><tr><th>minimum</th>\
         <th>state</th><th>{quantity} (Hartree)</th><th>kcal/mol</th><th>crossing points</th>\
         <th></th></tr></thead><tbody>"
    );
    for n in network.minima() {
        let joined: Vec<String> = network
            .crossing_edges
            .iter()
            .filter(|e| e.ends.contains(&Some(n.id)))
            .filter_map(|e| network.nodes.get(e.crossing).map(Node::label))
            .collect();
        let _ = write!(
            minima,
            "<tr><td>{}</td><td>{}</td><td class=\"num\">{:.8}</td><td class=\"num\">{:.2}</td>\
             <td>{}</td><td>{}</td></tr>",
            n.label(),
            surface_of(network, n),
            n.energy,
            rel(n.energy),
            if joined.is_empty() {
                "—".to_string()
            } else {
                joined.join(", ")
            },
            super::autots_static::download_link(
                &format!("{}.xyz", n.label()),
                &node_xyz(n),
                "Download .xyz",
                "dl"
            )
        );
    }
    minima.push_str("</tbody></table></div>");
    let e = network.evaluations;
    let evaluations = format!(
        "<p class=\"note\">Evaluations of the run: {} gradient calculation(s), {} exact Hessian \
         calculation(s){}; {}.</p>",
        e.gradients,
        e.hessians,
        if e.energies > 0 {
            format!(", {} energy calculation(s) alone", e.energies)
        } else {
            String::new()
        },
        if e.micro_relaxations > 0 {
            format!(
                "{} relaxation(s) of an ONIOM environment with {} microiteration(s)",
                e.micro_relaxations, e.micro_iterations
            )
        } else {
            "no microiterations (no ONIOM environment was relaxed)".to_string()
        }
    );

    // Steps 1 and 2 from the trajectories and logs of the jobs.
    let mut step1 = String::from(
        "<section class=\"card glass\" id=\"step1\"><h2>Step 1 · AFIR paths towards the \
         crossing</h2>",
    );
    if let Some(n) = start {
        let _ = write!(
            step1,
            "<p class=\"note\">The start is the input minimized on its own state: {} on {}, \
             {:.8} Hartree.</p>",
            n.label(),
            surface_of(network, n),
            n.energy
        );
    }
    let mut step2 = String::from(
        "<section class=\"card glass\" id=\"step2\"><h2>Step 2 · Crossing points</h2>\
         <p class=\"note\">Every candidate of step 1 optimized without the force on the \
         gradient projection of the two states.</p>",
    );
    let (mut candidates, mut unconverged) = (0, 0);
    let names = job_names(network);
    let pushes = network.jobs.iter().filter(|j| !is_direct(j)).count();
    let directs = network.jobs.len() - pushes;
    for (job, (title, slug)) in network.jobs.iter().zip(&names) {
        let frames = job
            .trajectory
            .as_deref()
            .map(|t| job_frames(&mapper_dir.join(t)))
            .unwrap_or_default();
        let log = std::fs::read_to_string(
            mapper_dir.join(crate::workflow::mapper::observers::job_log_file(job.id)),
        )
        .unwrap_or_default();
        let _ = write!(step1, "<h3>{title} · {}</h3>", escape(&job.description));
        if let JobStatus::Failed(error) = &job.status {
            let _ = write!(
                step1,
                "<p class=\"warn\">The job failed: {}</p>",
                escape(error)
            );
        }
        let all = stages(&frames);
        let refinements: Vec<(&str, &[Frame])> = all
            .iter()
            .filter(|(name, _)| name.starts_with("the MECP of") || name.starts_with("the MECI of"))
            .copied()
            .collect();
        if let Some((path_name, path)) = all
            .iter()
            .find(|(name, _)| name.starts_with("AFIR path on"))
            .copied()
        {
            let samples: Vec<_> = path
                .iter()
                .enumerate()
                .filter_map(|(i, frame)| {
                    frame
                        .states
                        .filter(|e| e.iter().all(|v| v.is_finite()))
                        .map(|e| (i, e))
                })
                .collect();
            let xs: Vec<f64> = samples.iter().map(|(i, _)| *i as f64).collect();
            let [first, second] = states_of(path_name);
            if !samples.is_empty() {
                let mut lines = vec![
                    series(
                        &first,
                        xs.clone(),
                        samples.iter().map(|(_, s)| rel(s[0])).collect(),
                        Rgb::BLUE,
                        PlotStyle::LinePoints,
                    ),
                    series(
                        &second,
                        xs.clone(),
                        samples.iter().map(|(_, s)| rel(s[1])).collect(),
                        Rgb::RED,
                        PlotStyle::LinePoints,
                    ),
                ];
                // The points taken as candidates: named in the stages of step 2.
                let mut at: Vec<usize> = refinements
                    .iter()
                    .filter_map(|(name, _)| {
                        if name.contains("from the end of the AFIR path") {
                            Some(path.len() - 1)
                        } else {
                            number_after(name, "at point ").map(|p| p as usize)
                        }
                    })
                    .filter(|&k| {
                        path.get(k)
                            .and_then(|p| p.states)
                            .is_some_and(|e| e.iter().all(|v| v.is_finite()))
                    })
                    .collect();
                at.sort_unstable();
                at.dedup();
                if !at.is_empty() {
                    lines.push(series(
                        "candidates (mean of the two states)",
                        at.iter().map(|&k| k as f64).collect(),
                        at.iter()
                            .filter_map(|&k| path[k].states.map(|s| rel(0.5 * (s[0] + s[1]))))
                            .collect(),
                        Rgb::EMERALD,
                        PlotStyle::Scatter,
                    ));
                }
                step1.push_str(&chart(
                    "Energies of the two states along the AFIR path",
                    "point of the path",
                    "kcal/mol (above the start)",
                    lines,
                ));
            }
            if samples.len() != path.len() {
                let _ = write!(step1, "<p class=\"warn\">Both state energies are available for {} of {} saved AFIR points.</p>", samples.len(), path.len());
            }
            let ended_text = logged(&log, path_name).map_or_else(String::new, |t| {
                format!(" {}.", escape(t.trim_end_matches('.')))
            });
            let _ = write!(
                step1,
                "<p class=\"note\">{}: {} points.{ended_text} The candidates are its end, when it \
             converged, and the local minima along it of the function that is low where the \
             gap is small.</p>",
                escape(path_name),
                path.len()
            );
            step1.push_str(&viewers.add_text(
                &format!("{slug}_afir_path"),
                xyz_of(path, MAX_FRAMES),
                true,
                false,
                &format!("The AFIR path of {} (animated).", title.to_lowercase()),
            ));
            if refinements.is_empty() {
                step1.push_str(
                    "<p class=\"warn\">This path gave no candidate: it did not converge, and the \
                 function that marks the seam has no minimum along it.</p>",
                );
            }
        } else if job.task.starts_with("crossing:") {
            step1.push_str("<p class=\"note\">Direct crossing search from the minimum. See Step 2 for the refinement.</p>");
        } else {
            step1.push_str("<p class=\"note\">No AFIR path of this push is in its files.</p>");
        }
        for (k, (name, points)) in refinements.iter().enumerate() {
            candidates += 1;
            let origin = name.split_once(" from ").map_or("", |(_, rest)| rest);
            let said = logged(&log, name).unwrap_or("");
            let converged = said.trim_end().ends_with(", converged");
            let last = points.last();
            let gap = number_after(said, "gap ")
                .or_else(|| last.and_then(|f| f.states).map(|s| s[0] - s[1]));
            let difference = number_after(said, "|g1 - g2| ");
            let node = last
                .and_then(|f| f.states)
                .filter(|_| converged)
                .and_then(|s| crossing_at(network, 0.5 * (s[0] + s[1]), SAME_CROSSING_HARTREE));
            let state = match (converged, node) {
                (true, Some(n)) => n.label(),
                (true, None) => "crossing point".to_string(),
                (false, _) => "not converged".to_string(),
            };
            let _ = write!(
                step2,
                "<h3>{title}, candidate {} · {state}</h3><p class=\"note\">From {}: {} \
                 iteration(s){}{}.</p>",
                k + 1,
                escape(origin),
                points.len(),
                gap.map_or_else(String::new, |g| format!(", gap {g:.3e} Hartree")),
                difference
                    .map_or_else(String::new, |d| format!(", |g1 − g2| {d:.3e} Hartree/Bohr")),
            );
            if !converged {
                unconverged += 1;
                step2.push_str(
                    "<p class=\"warn\">This optimization did not converge to a crossing point \
                     (the iteration limit, or a gap that stays open); its last structure is \
                     no crossing point and has no meta-IRC.</p>",
                );
            }
            let [first, second] = states_of(name);
            let both: Vec<[f64; 2]> = points.iter().filter_map(|f| f.states).collect();
            if both.len() == points.len() && points.len() > 1 {
                let xs: Vec<f64> = (0..points.len()).map(|i| i as f64).collect();
                step2.push_str(&chart(
                    "Energies of the two states along the optimization",
                    "iteration",
                    "kcal/mol (above the start)",
                    vec![
                        series(
                            &first,
                            xs.clone(),
                            both.iter().map(|s| rel(s[0])).collect(),
                            Rgb::BLUE,
                            PlotStyle::LinePoints,
                        ),
                        series(
                            &second,
                            xs,
                            both.iter().map(|s| rel(s[1])).collect(),
                            Rgb::RED,
                            PlotStyle::LinePoints,
                        ),
                    ],
                ));
            }
            step2.push_str(&viewers.add_text(
                &format!("{slug}_candidate{}", k + 1),
                xyz_of(points, MAX_FRAMES),
                false,
                false,
                &format!(
                    "The optimization of candidate {} of {} (the last frame is its end).",
                    k + 1,
                    title.to_lowercase()
                ),
            ));
        }
    }
    if network.jobs.is_empty() {
        step1.push_str("<p class=\"note\">No push has run yet.</p>");
    }
    if candidates == 0 {
        step2.push_str("<p class=\"note\">No candidate has been optimized yet.</p>");
    }
    step1.push_str("</section>");
    step2.push_str("</section>");

    // Step 3: the way between the two minima of every crossing point.
    let mut step3 = String::from(
        "<section class=\"card glass\" id=\"step3\"><h2>Step 3 · Meta-IRC and minima</h2>\
         <p class=\"note\">From every crossing point the path of steepest descent runs down \
         on each of its two states, and its end is minimized on that state.</p>",
    );
    if network.crossing_edges.is_empty() {
        step3.push_str("<p class=\"note\">No crossing point has a meta-IRC yet.</p>");
    }
    for e in &network.crossing_edges {
        let Some(x) = network.nodes.get(e.crossing) else {
            continue;
        };
        let end = |side: usize| e.ends[side].and_then(|id| network.nodes.get(id));
        let name = |side: usize| end(side).map_or_else(|| "—".to_string(), Node::label);
        let _ = write!(
            step3,
            "<h3>{} · {} ← → {}{}</h3>",
            x.label(),
            name(0),
            name(1),
            e.seam
                .as_ref()
                .filter(|s| s.is_saddle())
                .map_or_else(String::new, |s| format!(" · {}", escape(&s.describe())))
        );
        // The energy along the two sides, each on its own state, from the crossing point.
        let colors = [Rgb::BLUE, Rgb::RED];
        let mut lines = Vec::new();
        for (k, (label, points)) in meta_irc_points(network, e).into_iter().enumerate() {
            lines.push(series(
                &label,
                points.iter().map(|p| p.0).collect(),
                points.iter().map(|p| p.1).collect(),
                colors[k % 2],
                PlotStyle::LinePoints,
            ));
        }
        if !lines.is_empty() {
            step3.push_str(&chart(
                &format!("Energy along the meta-IRC of each state from {}", x.label()),
                "arc length (mass-weighted, Bohr amu^1/2; negative: the first state)",
                "kcal/mol (from the crossing point)",
                lines,
            ));
        }
        let path = e
            .irc
            .file
            .as_deref()
            .map(|f| read_frames(&mapper_dir.join(f)))
            .unwrap_or_default();
        if path.is_empty() {
            step3.push_str(
                "<p class=\"warn\">The file of this path is not in the run directory.</p>",
            );
        } else {
            // Thinned to the frames of the viewer; its comments name the parts.
            let text = xyz_of(&path, MAX_FRAMES);
            step3.push_str(&viewers.add_text(
                &format!("{}_path", x.label()),
                text,
                true,
                true,
                &format!(
                    "From {} through {} to {} (animated): the minimization and the meta-IRC of \
                     the first state backwards, the crossing point, the meta-IRC and the \
                     minimization of the second state.",
                    name(0),
                    x.label(),
                    name(1)
                ),
            ));
        }
        let mut list = String::from("<ul>");
        for side in 0..2 {
            match end(side) {
                Some(n) => {
                    let _ = write!(
                        list,
                        "<li>{} on {}: {:.8} Hartree ({:+.2} kcal/mol from the crossing point) \
                         {}</li>",
                        n.label(),
                        surface_of(network, n),
                        n.energy,
                        Units::HARTREE_TO_KCAL_MOL * (n.energy - x.energy),
                        super::autots_static::download_link(
                            &format!("{}.xyz", n.label()),
                            &node_xyz(n),
                            "Download .xyz",
                            "dl"
                        )
                    );
                }
                None => list.push_str(
                    "<li>No minimum on this side: the meta-IRC gave no path, or its end did \
                     not converge.</li>",
                ),
            }
        }
        let _ = write!(
            list,
            "<li>{}: {:.8} Hartree {}</li></ul>",
            x.label(),
            x.energy,
            super::autots_static::download_link(
                &format!("{}.xyz", x.label()),
                &node_xyz(x),
                "Download .xyz",
                "dl"
            )
        );
        step3.push_str(&list);
    }
    step3.push_str("</section>");

    // Warnings of the run.
    let mut warnings = String::new();
    for w in &network.warnings {
        let _ = write!(
            warnings,
            "<li>{}{}</li>",
            w.job
                .and_then(|j| names.get(j))
                .map_or_else(String::new, |(title, _)| format!("{title}: ")),
            escape(&w.message)
        );
    }
    let warnings = if warnings.is_empty() {
        String::new()
    } else {
        format!(
            "<details><summary>Warnings of the run ({})</summary><ul>{warnings}</ul></details>",
            network.warnings.len()
        )
    };

    let mut references = vec![
        "S. Maeda, K. Morokuma, J. Chem. Phys. 2010, 132, 241102 (AFIR)".to_string(),
        "M. J. Bearpark, M. A. Robb, H. B. Schlegel, Chem. Phys. Lett. 1994, 223, 269 \
         (gradient projection)"
            .to_string(),
        "S. Maeda, K. Ohno, K. Morokuma, J. Chem. Theory Comput. 2010, 6, 1538 (branching \
         plane updating)"
            .to_string(),
        "Y. Harabuchi, T. Taketsugu, S. Maeda, Chem. Phys. Lett. 2017, 674, 141 (AFIR on the \
         gradient projection)"
            .to_string(),
    ];
    logged_references(mapper_dir, 2, &mut references);
    let mut refs = String::new();
    for r in &references {
        let _ = write!(refs, "<li>{}</li>", escape(r));
    }
    let credits = format!(
        "<section class=\"card glass\" id=\"credits\"><h2>Rights and credits</h2>\
         <p>This report was generated by mopt-rs {version}, Copyright © 2023–2026 ss0832, free \
         software under the GNU General Public License, version 3 or later \
         (GPL-3.0-or-later), with no warranty. The structure viewer, the style and the code \
         embedded in this page are part of mopt-rs, under the same license.</p>\
         <p>The structures, energies and other results on this page are the output of the \
         calculation that was run; mopt-rs claims no rights to them.</p>\
         <p>Element colors follow the CPK convention (as in Jmol); bonds are drawn from the \
         single-bond covalent radii of P. Pyykkö and M. Atsumi, Chem. Eur. J. 2009, 15, 186; \
         space-filling spheres use van der Waals radii mostly from A. Bondi, J. Phys. Chem. \
         1964, 68, 441 (approximate values for some metals).</p>\
         <h3>Methods and calculators of this run</h3><ul>{refs}</ul></section>",
        version = crate::VERSION
    );

    let html = format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoMECP {name}</title>
<style>{style}</style>
</head>
<body>
<header class="topbar glass"><div class="topbar-inner">
<div class="brand">AutoMECP · {name}</div>
<nav aria-label="Sections"><a href="#summary">Summary</a><a href="#step1">AFIR paths</a><a href="#step2">Crossing points</a><a href="#step3">Meta-IRC</a><a href="#settings">Settings</a><a href="#credits">Credits</a></nav>
<div class="seg" role="group" aria-label="Theme"><button type="button" data-theme-choice="auto">Auto</button><button type="button" data-theme-choice="light">Light</button><button type="button" data-theme-choice="dark">Dark</button></div>
</div></header>
<main>
<div class="hero">
<h1>AutoMECP · {name}</h1>
<p>mopt-rs {version} · calculator: {calculator} · {found} crossing point(s) from {candidates} candidate(s) of {pushes} push(es){direct}{not_converged} · {states}</p>
{stopped}
<div class="toolrow"><span class="seg-label">Molecules</span><div class="seg" role="group" aria-label="Display style of every molecule"><button type="button" data-style-choice="ball">Ball and stick</button><button type="button" data-style-choice="stick">Sticks</button><button type="button" data-style-choice="spacefill">Space-filling</button><button type="button" data-style-choice="wire">Wireframe</button></div>{labels}</div>
</div>
<section class="card glass" id="summary"><h2>Summary</h2><div class="table-wrap">{summary}</div>{diagram}{minima}{evaluations}{warnings}</section>
{step1}
{step2}
{step3}
<section class="card glass" id="settings"><h2>Settings</h2><details><summary>The settings of the mapper run behind this search</summary><pre>{settings}</pre></details>
<p class="note">The run is a run of the reaction-network mapper on the two states: its network, the trajectories and logs of its jobs, its paths and its own report are in <code>mapper/</code> of the run directory, the crossing points as a table in <code>automecp_summary.csv</code>. The structures of this page are embedded in it.</p></section>
{credits}
</main>
<script type="application/json" id="models">{models}</script>
<script type="application/json" id="page-info">{{"layers":null}}</script>
{viewer}
</body>
</html>
"##,
        name = escape(name),
        version = crate::VERSION,
        style = STYLE,
        viewer = VIEWER,
        labels = LABEL_CONTROLS,
        calculator = escape(&calculator),
        states = escape(&states_text(network)),
        direct = if directs > 0 {
            format!(" and {directs} direct search(es) from the minimum")
        } else {
            String::new()
        },
        not_converged = if unconverged > 0 {
            format!(" · {unconverged} candidate(s) did not converge")
        } else {
            String::new()
        },
        stopped = match ended {
            Some(_) => String::new(),
            None => "<p class=\"warn\">The run had not ended when this page was written (it was \
                     stopped, or is still running): this is what it had done. \
                     <code>run_automecp --restart RUN_DIR</code> continues it.</p>"
                .to_string(),
        },
        settings = escape(&settings),
        models = viewers.models_json(),
    );
    let path = dir.join(REPORT_FILE);
    std::fs::write(&path, html)?;
    Ok(path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Element, MolecularSystem, SpinCharge};
    use crate::workflow::mapper::network::{
        CrossingEdge, CrossingKind, ElectronicState, IrcPath, IrcSide, JobRecord, Level, Origin,
    };
    use crate::workflow::mapper::Configuration;
    use faer::Col;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mopt_automecp_report_{name}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("mapper")).unwrap();
        dir
    }

    fn state(multiplicity: u32) -> ElectronicState {
        ElectronicState {
            charge: 0,
            multiplicity,
            excited_state: 0,
        }
    }

    fn node(kind: NodeKind, energy: f64, on: ElectronicState) -> Node {
        let sys = MolecularSystem::from_flat_bohr(
            vec![Element::H],
            &Col::zeros(3),
            SpinCharge::new(on.charge, on.multiplicity),
            None,
        )
        .unwrap();
        Node::new(
            kind,
            Level::Exploration,
            Configuration::Structure(sys),
            energy,
            on,
            Origin::default(),
        )
    }

    /// A network of a singlet and a triplet: the start on the singlet, a crossing point
    /// whose meta-IRC gave a path on the triplet only, and its end there.
    fn network() -> Network {
        let (s, t) = (state(1), state(3));
        let mut net = Network::new(vec![Element::H]);
        net.states = vec![s, t];
        let start = net.add_node(node(NodeKind::Minimum, 0.0, s)).unwrap();
        net.inputs
            .push(crate::workflow::mapper::network::InputRecord {
                file: "start.xyz".to_string(),
                frame: 0,
                node: Some(start),
                goal: false,
            });
        let crossing = net
            .add_node(node(
                NodeKind::Crossing {
                    kind: CrossingKind::Mecp,
                    states: [s, t],
                },
                0.01,
                s,
            ))
            .unwrap();
        let end = net.add_node(node(NodeKind::Minimum, 0.004, t)).unwrap();
        net.add_crossing_edge(CrossingEdge {
            id: 0,
            level: Level::Exploration,
            crossing,
            ends: [None, Some(end)],
            barriers: [None, Some(0.006)],
            gradient_difference: Some(0.01),
            gap: Some(2.0e-6),
            seam: Some(crate::workflow::mapper::SeamCheck::Checked(1)),
            irc: IrcPath {
                sides: vec![IrcSide {
                    name: "PES2".to_string(),
                    arc_lengths: vec![0.5, 1.0, 1.5],
                    energies: vec![0.008, 0.006, 0.004],
                    end: "converged".to_string(),
                    transverse: Vec::new(),
                }],
                file: None,
                possible_bifurcations: Vec::new(),
            },
            job: Some(0),
        })
        .unwrap();
        for (id, task) in [(0, "gpafir:1:afir:100:1:2"), (1, "crossing:1")] {
            net.jobs.push(JobRecord {
                id,
                node: start,
                method: "crossings".to_string(),
                task: task.to_string(),
                description: task.to_string(),
                status: JobStatus::Done,
                new_nodes: Vec::new(),
                new_edges: Vec::new(),
                found_new: false,
                found_lower: false,
                trajectory: None,
            });
        }
        net
    }

    fn page(dir: &Path, step1: &str, recorded: Option<&str>) -> String {
        std::fs::write(
            dir.join(super::super::automecp::MAPPER_CONFIG_FILE),
            format!("{{\"step1_settings\": {step1}}}"),
        )
        .unwrap();
        if let Some(name) = recorded {
            std::fs::write(
                dir.join("mapper")
                    .join(crate::workflow::mapper::store::NETWORK_FILE),
                format!("{{\"run\": {{\"calculator\": \"{name}\"}}}}"),
            )
            .unwrap();
        }
        let path = write(dir, &dir.join("mapper"), "start", &network(), Some("done")).unwrap();
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn the_page_names_the_calculator_the_run_used() {
        let dir = scratch("calculator");
        let text = page(&dir, r#"{"usedxtb": "GFN1-xTB"}"#, Some("dxtb:GFN1-xTB"));
        assert!(text.contains("calculator: dxtb:GFN1-xTB"), "{text}");
    }

    #[test]
    fn the_page_of_a_variable_cell_shows_enthalpies() {
        let dir = scratch("enthalpy");
        let text = page(&dir, r#"{"othersoft": "lj", "variable_cell": true}"#, None);
        assert!(text.contains("<th>enthalpy (Hartree)</th>"), "{text}");
        assert!(!text.contains("<th>energy (Hartree)</th>"), "{text}");
        let fixed = page(&dir, r#"{"othersoft": "lj"}"#, None);
        assert!(fixed.contains("<th>energy (Hartree)</th>"));
    }

    #[test]
    fn a_saddle_point_on_the_seam_is_named_so_with_its_gap() {
        // The crossing point of the network has one imaginary mode within the seam.
        let dir = scratch("seam_saddle");
        let text = page(&dir, r#"{"othersoft": "lj"}"#, None);
        assert!(
            text.contains("saddle point on the seam (1 imaginary mode(s))"),
            "{text}"
        );
        assert!(
            text.contains("not a minimum-energy crossing point"),
            "{text}"
        );
        assert!(text.contains("2.000e-6"), "the gap: {text}");
        let csv = super::super::automecp::summary_csv(&network());
        let line = csv.lines().nth(1).unwrap();
        assert!(
            line.ends_with(",2.000000e-6,1,\"saddle point on the seam (1 imaginary mode(s))\""),
            "{line}"
        );
    }

    #[test]
    fn a_direct_search_is_no_push() {
        let dir = scratch("pushes");
        let text = page(&dir, r#"{"othersoft": "lj"}"#, None);
        assert!(text.contains("of 1 push(es)"), "{text}");
    }

    #[test]
    fn a_side_of_the_meta_irc_is_drawn_on_the_side_of_its_state() {
        let net = network();
        let edge = &net.crossing_edges[0];
        // The one side is on the second state, PES2: drawn at positive arc lengths.
        let drawn = meta_irc_points(&net, edge);
        assert_eq!(drawn.len(), 1);
        assert_eq!(drawn[0].0, "on PES2");
        assert!(drawn[0].1.iter().all(|(s, _)| *s >= 0.0), "{drawn:?}");
    }
}
