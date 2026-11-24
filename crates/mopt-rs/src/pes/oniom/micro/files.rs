//! The files of a run with microiterations: the frames of the core are written with the
//! whole system (the environment relaxed for each core structure).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::RelaxedEnvironment;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// The directories whose files are analyses (animations of modes: the vibrations and the
/// Koopman modes) rather than structures of the surface.
const ANALYSIS_DIRECTORIES: [&str; 2] = ["vibration_animation", "koopman_analysis"];

/// Replace, in every xyz file under `dir`, the frames of the core by the full structures of
/// `relaxed` (the comment lines kept). Files whose frames are not of the core are left alone.
/// A frame of a structure of the run takes the environment kept for it, or one relaxed for it
/// where none is kept. The frames of an analysis (`vibration_animation/`, `koopman_analysis/`: a core displaced
/// along a mode, not a structure the run evaluated) take the environment of the nearest kept
/// core as it is: relaxing it for every frame cost as many relaxations as frames (50 for the
/// Koopman modes of an optimization of chorismate mutase, half the time of the run).
pub fn write_full_structures(
    dir: &std::path::Path,
    relaxed: &RelaxedEnvironment,
) -> MoptResult<usize> {
    let n = relaxed.split.core.len();
    let mut rewritten = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "xyz") {
                continue;
            }
            let analysis = path
                .strip_prefix(dir)
                .unwrap_or(&path)
                .components()
                .any(|c| ANALYSIS_DIRECTORIES.iter().any(|a| c.as_os_str() == *a));
            let text = std::fs::read_to_string(&path)?;
            if let Some((full, replicas)) = expand_frames(&text, n, relaxed, analysis)? {
                std::fs::write(&path, full)?;
                rewritten += 1;
                if let Some(all) = replicas {
                    let stem = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    std::fs::write(path.with_file_name(format!("{stem}{REPLICAS_SUFFIX}")), all)?;
                }
            }
        }
    }
    Ok(rewritten)
}

/// With replicas of the environment, the whole structures of every replica of the frames of
/// `NAME.xyz` are written to `NAME_replicas.xyz` (one frame per replica and frame, each with
/// its replica, energy, and weight), and `NAME.xyz` holds the lowest replica, its comment line
/// starting with `micro_replica=R/K weight=W replica_temperature=T`. A run that is to go on
/// on the same surface takes the file with `micro_replica_start=files micro_replica_files=`.
pub const REPLICAS_SUFFIX: &str = "_replicas.xyz";

/// Write the frame `full` with the comment `comment` to `out`.
fn write_frame(out: &mut String, full: &MolecularSystem, comment: &str) {
    use std::fmt::Write as _;
    let y = full.coords_flat_bohr();
    let _ = writeln!(out, "{}\n{comment}", full.n_atoms());
    for (a, element) in full.elements.iter().enumerate() {
        let _ = writeln!(
            out,
            "{:<3} {:18.12} {:18.12} {:18.12}",
            element.symbol(),
            y[3 * a] * crate::domain::Units::BOHR_TO_ANGSTROM,
            y[3 * a + 1] * crate::domain::Units::BOHR_TO_ANGSTROM,
            y[3 * a + 2] * crate::domain::Units::BOHR_TO_ANGSTROM
        );
    }
}

/// `tag` in front of `comment`, unless the comment is the line of the charge and multiplicity
/// (two whole numbers), which keeps its meaning only alone.
fn tagged(tag: &str, comment: &str) -> String {
    let mut words = comment.split_whitespace();
    let state = words.next().is_some_and(|w| w.parse::<i64>().is_ok())
        && words.next().is_some_and(|w| w.parse::<i64>().is_ok())
        && words.next().is_none();
    if state {
        comment.to_string()
    } else {
        format!("{tag} {comment}").trim_end().to_string()
    }
}

/// The frames of `text` with the full structures (and, with replicas, the frames of every
/// replica), or `None` when a frame is not of `n` atoms; `analysis`: the frames of an
/// analysis, with the environment of the nearest kept core.
#[allow(clippy::type_complexity)]
fn expand_frames(
    text: &str,
    n: usize,
    relaxed: &RelaxedEnvironment,
    analysis: bool,
) -> MoptResult<Option<(String, Option<String>)>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::new();
    let mut all = String::new();
    let replicas = relaxed.replica_count() > 1 && !analysis;
    let mut k = 0;
    let mut frames = 0;
    while k < lines.len() {
        if lines[k].trim().is_empty() {
            k += 1;
            continue;
        }
        if lines[k].trim().parse::<usize>().ok() != Some(n) || k + 2 + n > lines.len() {
            return Ok(None);
        }
        let comment = lines[k + 1];
        if comment
            .split_whitespace()
            .any(|field| field == "oniom_layer=high")
        {
            return Ok(None);
        }
        let mut x = Col::<f64>::zeros(3 * n);
        for (i, line) in lines[k + 2..k + 2 + n].iter().enumerate() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 4 {
                return Ok(None);
            }
            for c in 0..3 {
                let v: f64 = f[1 + c]
                    .parse()
                    .map_err(|_| MoptError::parse(format!("xyz: '{line}'")))?;
                x[3 * i + c] = v * crate::domain::Units::ANGSTROM_TO_BOHR;
            }
        }
        let core = MolecularSystem::from_flat_bohr(
            relaxed
                .split
                .core
                .iter()
                .map(|&a| relaxed.template.elements[a])
                .collect(),
            &x,
            relaxed.template.spin_charge,
            relaxed.template.cell.clone(),
        )?;
        match relaxed.replica_systems(&core)?.filter(|_| replicas) {
            Some((lowest, systems)) => {
                let weight = systems.get(lowest).map_or(0.0, |s| s.2);
                let tag = relaxed.replica_tag(lowest, weight);
                if let Some((full, _, _)) = systems.get(lowest) {
                    write_frame(&mut out, full, &tagged(&tag, comment));
                }
                let count = systems.len();
                for (r, (full, energy, weight)) in systems.iter().enumerate() {
                    let tag = format!(
                        "micro_replica={}/{count} weight={weight:.4} replica_energy={energy:.10} \
                         frame={frames}",
                        r + 1
                    );
                    write_frame(&mut all, full, &tagged(&tag, comment));
                }
            }
            None => {
                let full = if analysis {
                    relaxed.nearest_full_system(&core)?
                } else {
                    relaxed.full_system(&core)?
                };
                write_frame(&mut out, &full, comment);
            }
        }
        frames += 1;
        k += 2 + n;
    }
    Ok((frames > 0).then_some((out, replicas.then_some(all))))
}
