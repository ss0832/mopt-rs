//! `run_mapper -merge RUN_DIR ...`: the networks of several runs written as one, in a run
//! directory of its own, without a calculation ([`crate::workflow::mapper::merge`]). The path
//! files of what was added are copied under their new names, the table `merge.csv` says what
//! every node of every run is in the merged network, and the files and the report of the
//! merged network are those of any run. The runs are only read.
//!
//! The merged directory is read like a run directory by `run_kinetics`, `-remap`,
//! `-single_points` and the report; it has no journal and is not continued by `--restart`.
//! The rules are in `docs/source/mapper_design.md` ("Networks of other runs").
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use super::mapper::{MapperConfig, MapperParts, MapperRun};
use super::mapper_remap::FirstNetwork;
use super::mapper_report::MapperReport;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::workflow::mapper::merge::merge_into;
use crate::workflow::mapper::peak::peak_path_file;
use crate::workflow::mapper::store::{write_whole, FileStore, NetworkStore};
use crate::workflow::mapper::{mapper_matcher, ExplorationEnd, MapperObserver};

/// The table of what the nodes of the runs are in the merged network, in the run directory.
pub const MERGE_TABLE: &str = "merge.csv";

/// The method named in the network file of a merged network.
pub const MERGE_METHOD: &str = "merge";

/// The name of a run in the table and the notes: the last part of what was given.
fn run_name(first: &FirstNetwork) -> String {
    first.given.file_name().map_or_else(
        || first.given.display().to_string(),
        |s| s.to_string_lossy().into_owned(),
    )
}

/// Merge the networks `config.merge` into the run directory of `config` under `base`.
pub(super) fn run(
    config: &MapperConfig,
    base: &Path,
    console: Option<Level>,
    parts: &MapperParts,
) -> MoptResult<MapperRun> {
    let s = &config.settings;
    let Some((first, rest)) = config.merge.split_first() else {
        return Err(MoptError::config("run_mapper: -merge names no run"));
    };
    let dir = base.join(&config.output_dir);
    let used = dir
        .read_dir()
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);
    if used {
        return Err(MoptError::config(format!(
            "run_mapper: the run directory {} exists; give another --output_dir for the merged \
             network",
            dir.display()
        )));
    }
    std::fs::create_dir_all(&dir).map_err(|e| MoptError::from(e).context(dir.display()))?;
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    let matcher = if first.network.variables.is_some() {
        crate::workflow::explore::identity::StructureMatcher {
            rmsd_tolerance: crate::domain::Quantity::new(s.variable_tolerance),
            ..mapper_matcher(s.rmsd_threshold, s.energy_tolerance, s.mirror)
        }
    } else {
        mapper_matcher(s.rmsd_threshold, s.energy_tolerance, s.mirror)
    };
    log.info(&format!(
        "Merge: the networks of {} run(s) as one, without a calculation; that they ran on the \
         same surface is taken on the word of the user. Two nodes are the same within {} \
         Angstrom and {} kcal/mol.",
        config.merge.len(),
        s.rmsd_threshold,
        s.energy_tolerance
    ));
    let mut network = first.network.clone();
    // The jobs of the first run keep their numbers; their trajectories stay in its directory.
    for j in &mut network.jobs {
        j.trajectory = None;
    }
    let mut table = String::from("run,node,merged_node,status\n");
    let name = run_name(first);
    for n in &first.network.nodes {
        let _ = writeln!(table, "\"{name}\",{},{},first", n.label(), n.label());
    }
    // The path files to copy: (the directory they are in, their name there, their new name).
    let mut copies: Vec<(&Path, String, String)> = Vec::new();
    let variables = first.network.variables.is_some();
    for e in &first.network.ts_edges {
        if let Some(f) = &e.irc.file {
            copies.push((&first.dir, f.clone(), f.clone()));
        }
    }
    for e in &first.network.crossing_edges {
        if let Some(f) = &e.irc.file {
            copies.push((&first.dir, f.clone(), f.clone()));
        }
    }
    for p in &first.network.path_peaks {
        let f = peak_path_file(p.id, variables);
        copies.push((&first.dir, f.clone(), f));
    }
    log.info(&format!(
        "{name}: {} node(s), {} TS edge(s), taken as they are.",
        first.network.nodes.len(),
        first.network.ts_edges.len()
    ));
    for other in rest {
        let name = run_name(other);
        let map = merge_into(&mut network, &other.network, &matcher, &name)?;
        for (old, node) in other.network.nodes.iter().enumerate() {
            let _ = writeln!(
                table,
                "\"{name}\",{},{},{}",
                node.label(),
                network.node(map.nodes[old])?.label(),
                if map.added[old] { "added" } else { "known" }
            );
        }
        for (from, to) in &map.paths {
            copies.push((&other.dir, from.clone(), to.clone()));
        }
        for &(old, new) in &map.peaks {
            copies.push((
                &other.dir,
                peak_path_file(old, variables),
                peak_path_file(new, variables),
            ));
        }
        log.info(&format!(
            "{name}: {} of its {} node(s) added ({} were known); {} TS edge(s), {} crossing \
             edge(s), {} descent(s), {} dissociation(s) and {} path peak(s) added.",
            map.added.iter().filter(|a| **a).count(),
            other.network.nodes.len(),
            map.added.iter().filter(|a| !**a).count(),
            map.edges[0],
            map.edges[1],
            map.edges[2],
            map.edges[3],
            map.peaks.len()
        ));
        for note in &map.notes {
            log.warn(note);
        }
    }
    let mut missing = 0;
    for (from, old, new) in &copies {
        let source = from.join(old);
        if !source.is_file() {
            missing += 1;
            continue;
        }
        let bytes =
            std::fs::read(&source).map_err(|e| MoptError::from(e).context(source.display()))?;
        write_whole(&dir.join(new), &bytes)?;
    }
    if missing > 0 {
        log.warn(&format!(
            "{missing} path file(s) of the runs were not found and are not in the merged \
             directory."
        ));
    }
    write_whole(&dir.join(MERGE_TABLE), table.as_bytes())?;
    let end = ExplorationEnd {
        reason: Some(format!(
            "the networks of {} run(s) are merged",
            config.merge.len()
        )),
        interrupted: false,
    };
    let mut run = BTreeMap::new();
    run.insert(
        "method".to_string(),
        JsonValue::String(MERGE_METHOD.to_string()),
    );
    run.insert(
        "merged".to_string(),
        JsonValue::Array(
            config
                .merge
                .iter()
                .map(|f| JsonValue::String(run_name(f)))
                .collect(),
        ),
    );
    run.insert(
        "calculator".to_string(),
        JsonValue::String(config.steps.biased.calculator.clone()),
    );
    run.insert(
        "minima_verified".to_string(),
        JsonValue::Bool(config.steps.verify_minima),
    );
    run.insert(
        "ended".to_string(),
        end.reason
            .clone()
            .map_or(JsonValue::Null, JsonValue::String),
    );
    run.insert("interrupted".to_string(), JsonValue::Bool(false));
    let run = JsonValue::Object(run);
    FileStore.write(&network, &run, &dir)?;
    for store in &s.stores {
        if !store.eq_ignore_ascii_case("files") {
            parts.stores.create(store)?.write(&network, &run, &dir)?;
        }
    }
    if s.html_report {
        let title = format!("{} (merge)", run_name(first));
        let mut report = MapperReport::new(
            &dir,
            &title,
            &config.steps.biased.calculator,
            &format!("{:#?}", config.settings),
            config.report.clone(),
        )
        .with_log(log.clone());
        report
            .on_finish(&network, &end)
            .context("the report of the merged network")?;
    }
    log.info(&format!(
        "The merged network has {} node(s) ({} minima), {} TS edge(s) and {} job(s); {MERGE_TABLE} \
         says what every node of every run is in it.",
        network.nodes.len(),
        network.minima().count(),
        network.ts_edges.len(),
        network.jobs.len()
    ));
    log.flush();
    Ok(MapperRun {
        directory: dir,
        network,
        interrupted: false,
        reason: end.reason,
    })
}
