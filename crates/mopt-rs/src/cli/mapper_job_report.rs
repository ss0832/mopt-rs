//! Render the saved stages of one mapper job with the AutoTS report components.
//!
//! MultiOptPy: Wrapper/mapper.py and the AutoTS workflow.
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use super::autots_report::{chart, escape, series, Viewers, LABEL_CONTROLS, STYLE, VIEWER};
use crate::core_infra::viz::{color::Rgb, figure::PlotStyle};
use crate::domain::Units;
use crate::error::MoptResult;
use crate::workflow::mapper::network::{JobStatus, Network};
use crate::workflow::mapper::observers::{frame_stage, stage_file};

fn publish(path: &Path, text: &str) -> MoptResult<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temp = path.with_extension(format!(
        "{}.{}.part",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&temp, text)?;
    fs::rename(temp, path)?;
    Ok(())
}

pub(super) fn index(root: &Path, ids: &[usize]) -> MoptResult<()> {
    let jobs = root.join("jobs");
    fs::create_dir_all(&jobs)?;
    let mut html = format!(
        r#"<!doctype html><meta charset="utf-8"><title>Mapper jobs</title><style>{STYLE}</style><main><a href="../mapper_report.html">Network</a><h1>Mapper jobs</h1><ul>"#
    );
    for id in ids {
        let _ = write!(
            html,
            r#"<li><a href="job_{id:06}/job_report.html">Job {id}</a></li>"#
        );
    }
    html.push_str("</ul></main>");
    publish(&jobs.join("index.html"), &html)
}

struct Stage {
    name: String,
    energies: Vec<f64>,
}

fn stages(path: &Path) -> MoptResult<Vec<Stage>> {
    let mut out: Vec<Stage> = Vec::new();
    for row in BufReader::new(File::open(path)?).lines().skip(1) {
        let row = row?;
        let Some((_, rest)) = row.split_once(',') else {
            continue;
        };
        let Some((label, energy)) = rest.rsplit_once(',') else {
            continue;
        };
        let Ok(energy) = energy.parse::<f64>() else {
            continue;
        };
        let label = label.trim_matches('"').replace("\"\"", "\"");
        let name = frame_stage(&label);
        match out.last_mut() {
            Some(stage) if stage.name == name => stage.energies.push(energy),
            _ => out.push(Stage {
                name: name.to_string(),
                energies: vec![energy],
            }),
        }
    }
    Ok(out)
}

/// Stream complete XYZ frames, retaining their cell headers and both ends of the preview.
fn preview(path: &Path, count: usize, limit: usize) -> std::io::Result<(String, usize)> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut text = String::new();
    let mut line = String::new();
    let mut shown = 0;
    for i in 0..count {
        line.clear();
        reader.read_line(&mut line)?;
        let Ok(atoms) = line.trim().parse::<usize>() else {
            break;
        };
        let stride = count
            .saturating_sub(1)
            .div_ceil(limit.min(4000 / atoms.max(1)).max(2) - 1)
            .max(1);
        let keep = i % stride == 0 || i + 1 == count;
        let start = text.len();
        if keep {
            text.push_str(&line);
        }
        for _ in 0..=atoms {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                text.truncate(start);
                return Ok((text, shown));
            }
            if keep {
                text.push_str(&line);
            }
        }
        shown += usize::from(keep);
    }
    Ok((text, shown))
}

fn profile(energies: &[f64], variables: bool, axis: &str, reference: f64, legend: &str) -> String {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let factor = if variables {
        1.0
    } else {
        Units::HARTREE_TO_KCAL_MOL
    };
    for (i, energy) in energies.iter().enumerate() {
        if i % energies.len().div_ceil(300).max(1) == 0 || i + 1 == energies.len() {
            xs.push((i + 1) as f64);
            ys.push((energy - reference) * factor);
        }
    }
    let unit = if variables {
        "Objective change"
    } else {
        "Relative energy (kcal/mol)"
    };
    chart(
        unit,
        axis,
        unit,
        vec![series(legend, xs, ys, Rgb::BLUE, PlotStyle::LinePoints)],
    )
}

fn result_section(network: &Network, id: usize) -> MoptResult<String> {
    let Some(job) = network.jobs.iter().find(|j| j.id == id) else {
        return Ok(String::new());
    };
    let mut text = format!(
        "<section><h2>Job result</h2><p>Starting node {} · {} · {} new TS connections</p>",
        escape(&network.node(job.node)?.label()),
        escape(&job.method),
        job.new_edges.len()
    );
    if !job.new_nodes.is_empty() {
        text.push_str("<table><thead><tr><th>New node</th><th>Energy (Ha)</th><th>Imaginary frequencies (cm⁻¹)</th></tr></thead><tbody>");
        for &id in &job.new_nodes {
            let node = network.node(id)?;
            let modes = node.frequencies.as_ref().map_or_else(
                || "Unavailable".to_string(),
                |freq| {
                    let negative: Vec<_> = freq
                        .iter()
                        .filter(|&&v| v < 0.0)
                        .map(|v| format!("{v:.2}"))
                        .collect();
                    if negative.is_empty() {
                        "None".to_string()
                    } else {
                        negative.join(", ")
                    }
                },
            );
            let _ = write!(
                text,
                "<tr><td>{}</td><td>{:.12}</td><td>{modes}</td></tr>",
                escape(&node.label()),
                node.energy
            );
        }
        text.push_str("</tbody></table>");
    }
    for edge in network.ts_edges.iter().filter(|edge| edge.job == Some(id)) {
        let ts = network.node(edge.ts)?;
        let [a, b] = edge.ends;
        let _ = write!(
            text,
            "<p>{} → {} → {}",
            escape(&network.node(a)?.label()),
            escape(&ts.label()),
            escape(&network.node(b)?.label())
        );
        if let Some(file) = &edge.irc.file {
            let _ = write!(
                text,
                r#" · <a href="../../{}">IRC trajectory</a>"#,
                escape(file)
            );
        }
        text.push_str("</p>");
    }
    for warning in network.warnings.iter().filter(|w| w.job == Some(id)) {
        let _ = write!(text, "<p>{}</p>", escape(&warning.message));
    }
    text.push_str("</section>");
    Ok(text)
}

pub(super) fn write(
    dir: &Path,
    id: usize,
    task: &str,
    status: &str,
    variables: bool,
    network: Option<&Network>,
) -> MoptResult<()> {
    let stages = stages(&dir.join("steps.csv"))?;
    let limit = (150 / stages.len().max(1)).clamp(2, 24);
    let mut viewers = Viewers::new();
    let mut html = format!(
        r#"<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Mapper job {id}</title><style>{STYLE}</style><main><nav><a href="../index.html">Jobs</a> · <a href="../../mapper_report.html">Network</a> · <a href="../../queue_priority.log">Queue priority</a></nav><h1>Mapper job {id}</h1><p>{} · {}</p><p><a href="steps.csv">Frame energies</a> · <a href="trajectories.csv">Stage trajectories</a></p><div class="toolrow">{LABEL_CONTROLS}</div>"#,
        escape(task),
        escape(status)
    );
    if let Some(network) = network {
        html.push_str(&result_section(network, id)?);
    }
    for (i, stage) in stages.iter().enumerate() {
        let file = stage_file(i + 1, &stage.name, variables);
        let name = escape(&stage.name);
        let count = stage.energies.len();
        let _ = write!(
            html,
            r#"<section><h2>{name}</h2><p>{count} frames · <a href="{file}">Complete trajectory</a></p>"#
        );
        html.push_str(&profile(
            &stage.energies,
            variables,
            "Evaluation",
            stage.energies[0],
            "From first evaluation",
        ));
        if !variables {
            if let Ok((xyz, shown)) = preview(&dir.join(&file), count, limit) {
                html.push_str(&viewers.add_text(
                    file.trim_end_matches(".xyz"),
                    xyz,
                    true,
                    true,
                    &format!("{} · Preview: {shown} of {count} saved frames", stage.name),
                ));
            }
        }
        html.push_str("</section>");
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let bands = entry.path();
        if !bands.is_dir() || !bands.join("energy_plot.csv").is_file() {
            continue;
        }
        let folder = escape(&entry.file_name().to_string_lossy());
        let _ = write!(
            html,
            r#"<section><h2>NEB bands</h2><p><a href="{folder}/energy_plot.csv">Band energies</a> · <a href="{folder}/path_length.csv">Path lengths</a></p>"#
        );
        let energies = fs::read_to_string(bands.join("energy_plot.csv"))?;
        let iterations = energies.lines().count();
        let mut reference = None;
        for (i, row) in energies.lines().enumerate() {
            let Ok(values) = row
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
            else {
                continue;
            };
            let _ = write!(html, "<details><summary>Iteration {i}</summary>");
            let zero = *reference.get_or_insert(values[0]);
            html.push_str(&profile(
                &values,
                variables,
                "Image",
                zero,
                "From first image of initial band",
            ));
            let file = format!("path_ITR_{i}_neb/neb_path.xyz");
            if !variables && bands.join(&file).is_file() {
                let _ = write!(html, r#"<a href="{folder}/{file}">Complete band</a>"#);
                if i + 1 == iterations {
                    let (xyz, shown) = preview(&bands.join(file), values.len(), limit)?;
                    html.push_str(&viewers.add_text(
                        "neb_band",
                        xyz,
                        true,
                        true,
                        &format!(
                            "Latest completed NEB band · Preview: {shown} of {} images",
                            values.len()
                        ),
                    ));
                }
            }
            html.push_str("</details>");
        }
        html.push_str("</section>");
    }
    let _ = write!(
        html,
        r#"</main><script type="application/json" id="models">{}</script><script type="application/json" id="page-info">{{}}</script>{VIEWER}"#,
        viewers.models_json()
    );
    publish(&dir.join("job_report.html"), &html)
}

pub(super) fn regenerate(root: &Path, network: &Network) -> MoptResult<()> {
    let jobs = root.join("jobs");
    if !jobs.is_dir() {
        return Ok(());
    }
    let mut ids = Vec::new();
    for entry in fs::read_dir(&jobs)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(id) = name
            .to_str()
            .and_then(|s| s.strip_prefix("job_"))
            .and_then(|s| s.parse::<usize>().ok())
        else {
            continue;
        };
        if !entry.path().join("steps.csv").is_file() {
            continue;
        }
        let (task, status) = match network.jobs.iter().find(|j| j.id == id) {
            Some(j) => (
                j.description.as_str(),
                match &j.status {
                    JobStatus::Done => "Finished",
                    JobStatus::Failed(reason) => reason.as_str(),
                },
            ),
            None => ("", "Last recorded progress"),
        };
        write(
            &entry.path(),
            id,
            task,
            status,
            network.variables.is_some(),
            Some(network),
        )?;
        ids.push(id);
    }
    ids.sort_unstable();
    if !ids.is_empty() {
        index(root, &ids)?;
    }
    Ok(())
}
