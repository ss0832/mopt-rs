//! The `workflow` block of the settings of `run_autots` (MultiOptPy's AutoTS "v2"): which
//! steps run, how often, and with which settings each time.
//!
//! ```json
//! "workflow": [
//!   {"step": "step1", "repeat": 2, "repeat_settings": [
//!       {"param_override": {"manual_AFIR": ["50", "1", "16"]}},
//!       {"param_override": {"manual_AFIR": ["-300", "1", "14"], "NSTEP": 20}}]},
//!   {"step": "step2", "repeat": 2, "repeat_settings": [
//!       {"param_override": {"NSTEP": 10, "DMF": true}},
//!       {"param_override": {"NSTEP": 15, "QSM": true}}]},
//!   {"step": "step3"},
//!   {"step": "step4"}
//! ]
//! ```
//!
//! An entry names a step (`step1` .. `step4`), whether it runs (`enabled`, true unless
//! given), how often (`repeat`, 1 unless given), the block its settings come from
//! (`settings_key`, `<step>_settings` unless given), and for every run what is changed in
//! that block (`repeat_settings[k].param_override`; a list shorter than `repeat` uses its
//! last entry for the runs after it). Only the steps that are listed run. Without a
//! `workflow` block the four steps run once each (AutoTS "v1").
//!
//! What the runs of a step do with each other (`cli::autots`):
//!
//! - Step 1: every run starts from the last structure of the run before it, and the
//!   trajectories are joined to one first path.
//! - Step 2: every run starts from the relaxed path of the run before it, so that a path
//!   can be relaxed several times with different settings (`"mode": "sequential"`, the
//!   default), or every run starts from the first path (`"mode": "independent"`). The
//!   candidates are those of the last run, or those of all runs.
//! - Step 3: a later run takes up the candidates that have no transition state yet.
//! - Step 4 runs once.
//!
//! MultiOptPy: `Wrapper/autots.py:AutoTSWorkflow_v2` (`_validate_workflow_config`,
//! `_get_settings_for_repeat`, `merge_paths`). Differences are listed in
//! `docs/source/migration.md` ("AutoTS with a workflow block").
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::core_infra::json::JsonValue;
use crate::domain::{SpinCharge, Trajectory};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{read_xyz_trajectory_file, write_xyz_trajectory_file};

/// The settings of one run of a step.
pub type StepSettings = BTreeMap<String, JsonValue>;

/// The names of the steps in the settings file.
const STEPS: [&str; 4] = ["step1", "step2", "step3", "step4"];
/// Two structures are the same when no coordinate differs by more than this (Bohr).
const SAME_BOHR: f64 = 1.0e-6;

/// The `workflow` block: the runs of every step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Workflow {
    /// The settings of the runs of each step, in their order (none: the step does not run).
    pub runs: [Vec<StepSettings>; 4],
    /// Step 2: every run starts from the first path (`"mode": "independent"`) instead of
    /// from the path of the run before it.
    pub independent_paths: bool,
}

fn wrong(k: usize, what: impl std::fmt::Display) -> MoptError {
    MoptError::parse(format!("run_autots: workflow entry {}: {what}", k + 1))
}

impl Workflow {
    /// The `workflow` block of the settings `doc`, or `None` when there is none. `ready`
    /// is applied to every block of settings before the changes of a run go over it (the
    /// state of the input file, as for the blocks of the four steps).
    pub fn parse(doc: &JsonValue, ready: &dyn Fn(&mut StepSettings)) -> MoptResult<Option<Self>> {
        let Some(block) = doc.get("workflow") else {
            return Ok(None);
        };
        let JsonValue::Array(entries) = block else {
            return Err(MoptError::parse(
                "run_autots: 'workflow' must be a list of steps",
            ));
        };
        let mut out = Self::default();
        for (k, entry) in entries.iter().enumerate() {
            let JsonValue::Object(entry) = entry else {
                return Err(wrong(k, "must be an object"));
            };
            let name = entry
                .get("step")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| wrong(k, "has no 'step'"))?;
            let step = STEPS.iter().position(|s| *s == name).ok_or_else(|| {
                wrong(k, format!("'{name}' is not step1, step2, step3, or step4"))
            })?;
            let repeat = match entry.get("repeat") {
                None => 1,
                Some(JsonValue::Number(n)) if *n >= 1.0 && n.fract() == 0.0 => *n as usize,
                Some(_) => return Err(wrong(k, "'repeat' must be a whole number of 1 or more")),
            };
            let changes = match entry.get("repeat_settings") {
                None => &[][..],
                Some(JsonValue::Array(list)) => list.as_slice(),
                Some(_) => return Err(wrong(k, "'repeat_settings' must be a list")),
            };
            if changes.len() > repeat {
                return Err(wrong(
                    k,
                    format!(
                        "'repeat_settings' has {} entries for {repeat} run(s)",
                        changes.len()
                    ),
                ));
            }
            let default_key = format!("{name}_settings");
            let key = entry
                .get("settings_key")
                .and_then(JsonValue::as_str)
                .unwrap_or(&default_key);
            let Some(JsonValue::Object(base)) = doc.get(key) else {
                return Err(wrong(
                    k,
                    format!("the settings '{key}' are not in the file"),
                ));
            };
            let mut base = base.clone();
            ready(&mut base);
            if step == 1 {
                out.independent_paths = match entry.get("mode").and_then(JsonValue::as_str) {
                    None | Some("sequential") => false,
                    Some("independent") => true,
                    Some(other) => {
                        return Err(wrong(
                            k,
                            format!("'mode' is sequential or independent, not '{other}'"),
                        ))
                    }
                };
            }
            if matches!(entry.get("enabled"), Some(JsonValue::Bool(false))) {
                continue;
            }
            for run in 0..repeat {
                let mut settings = base.clone();
                // A list shorter than the number of runs: its last entry for the rest.
                if let Some(change) = changes.get(run).or(changes.last()) {
                    match change {
                        JsonValue::Object(change) => match change.get("param_override") {
                            None => {}
                            Some(JsonValue::Object(over)) => {
                                settings.extend(over.iter().map(|(a, b)| (a.clone(), b.clone())));
                            }
                            Some(_) => return Err(wrong(k, "'param_override' must be an object")),
                        },
                        _ => {
                            return Err(wrong(k, "an entry of 'repeat_settings' must be an object"))
                        }
                    }
                }
                out.runs[step].push(settings);
            }
        }
        Ok(Some(out))
    }
}

/// ` (run K of N)` when a step has several runs, nothing otherwise.
pub fn run_label(run: usize, runs: usize) -> String {
    if runs > 1 {
        format!(" (run {} of {runs})", run + 1)
    } else {
        String::new()
    }
}

/// The trajectories `parts` of runs that each start where the one before ended, as one
/// trajectory in `out`: the first structure of a later part is left out when it is the
/// last structure before it (MultiOptPy joins the files as they are, with that structure
/// twice). Returns the number of structures.
pub fn joined_trajectories(parts: &[PathBuf], out: &Path) -> MoptResult<usize> {
    let mut all = Trajectory { frames: Vec::new() };
    for part in parts {
        let read =
            read_xyz_trajectory_file(part, SpinCharge::default(), None).context(part.display())?;
        for frame in read.frames {
            let twice = all.frames.last().is_some_and(|last| {
                let (a, b) = (
                    last.system.coords_flat_bohr(),
                    frame.system.coords_flat_bohr(),
                );
                a.nrows() == b.nrows() && (0..a.nrows()).all(|i| (a[i] - b[i]).abs() < SAME_BOHR)
            });
            if !twice {
                all.frames.push(frame);
            }
        }
    }
    if all.frames.is_empty() {
        return Err(MoptError::calc(
            "run_autots: the first path has no structures",
        ));
    }
    let count = all.frames.len();
    write_xyz_trajectory_file(out, &all)?;
    Ok(count)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn settings(text: &str) -> JsonValue {
        JsonValue::parse(text).unwrap()
    }

    fn number(settings: &StepSettings, key: &str) -> Option<f64> {
        settings.get(key).and_then(JsonValue::as_f64)
    }

    #[test]
    fn the_runs_of_the_steps_with_their_changes() {
        let doc = settings(
            r#"{
              "step1_settings": {"NSTEP": 5, "manual_AFIR": ["100", "1", "2"]},
              "step2_settings": {"NSTEP": 15, "node_distance": 0.5},
              "rough": {"NSTEP": 3},
              "step3_settings": {"calc_exact_hess": 5},
              "step4_settings": {},
              "workflow": [
                {"step": "step1", "repeat": 2, "repeat_settings": [
                    {"param_override": {"manual_AFIR": ["50", "1", "3"]}},
                    {"param_override": {"NSTEP": 20}}]},
                {"step": "step2", "repeat": 3, "mode": "sequential", "repeat_settings": [
                    {"param_override": {"NSTEP": 10, "DMF": true}},
                    {"param_override": {"QSM": true}}]},
                {"step": "step2", "settings_key": "rough"},
                {"step": "step3"},
                {"step": "step4", "enabled": false}
              ]
            }"#,
        );
        let ready = |block: &mut StepSettings| {
            block.insert("electronic_charge".to_string(), JsonValue::Number(-1.0));
        };
        let workflow = Workflow::parse(&doc, &ready).unwrap().unwrap();
        // Step 1: the change of each run over the block.
        assert_eq!(workflow.runs[0].len(), 2);
        assert_eq!(number(&workflow.runs[0][0], "NSTEP"), Some(5.0));
        assert_eq!(number(&workflow.runs[0][1], "NSTEP"), Some(20.0));
        assert!(workflow.runs[0][0]["manual_AFIR"]
            .to_string_compact()
            .contains("50"));
        assert!(workflow.runs[0][1]["manual_AFIR"]
            .to_string_compact()
            .contains("100"));
        // Step 2: the last change again for the third run, then the entry with a block of
        // its own; what `ready` adds is in all of them.
        assert_eq!(workflow.runs[1].len(), 4);
        assert_eq!(number(&workflow.runs[1][0], "NSTEP"), Some(10.0));
        assert!(workflow.runs[1][0].contains_key("DMF"));
        for run in 1..3 {
            assert_eq!(number(&workflow.runs[1][run], "NSTEP"), Some(15.0));
            assert!(workflow.runs[1][run].contains_key("QSM"));
            assert!(!workflow.runs[1][run].contains_key("DMF"));
        }
        assert_eq!(number(&workflow.runs[1][3], "NSTEP"), Some(3.0));
        assert!(!workflow.runs[1][3].contains_key("node_distance"));
        assert!(workflow
            .runs
            .iter()
            .flatten()
            .all(|run| number(run, "electronic_charge") == Some(-1.0)));
        assert!(!workflow.independent_paths);
        // Step 3 once, step 4 not at all.
        assert_eq!(workflow.runs[2].len(), 1);
        assert!(workflow.runs[3].is_empty());
        assert_eq!(run_label(1, 3), " (run 2 of 3)");
        assert_eq!(run_label(0, 1), "");
    }

    #[test]
    fn no_block_and_what_is_wrong_in_one() {
        let ready = |_: &mut StepSettings| {};
        assert_eq!(Workflow::parse(&settings("{}"), &ready).unwrap(), None);
        let independent = settings(
            r#"{"step2_settings": {}, "workflow": [{"step": "step2", "mode": "independent"}]}"#,
        );
        assert!(
            Workflow::parse(&independent, &ready)
                .unwrap()
                .unwrap()
                .independent_paths
        );
        for wrong in [
            r#"{"workflow": {}}"#,
            r#"{"workflow": [3]}"#,
            r#"{"workflow": [{"repeat": 2}]}"#,
            r#"{"step5_settings": {}, "workflow": [{"step": "step5"}]}"#,
            r#"{"workflow": [{"step": "step1"}]}"#,
            r#"{"step1_settings": {}, "workflow": [{"step": "step1", "repeat": 0}]}"#,
            r#"{"step1_settings": {}, "workflow": [{"step": "step1", "repeat": 1.5}]}"#,
            r#"{"step1_settings": {}, "workflow": [{"step": "step1", "repeat_settings": [{}, {}]}]}"#,
            r#"{"step1_settings": {}, "workflow": [{"step": "step1", "repeat_settings": [3]}]}"#,
            r#"{"step1_settings": {}, "workflow": [{"step": "step1", "repeat_settings": [{"param_override": 3}]}]}"#,
            r#"{"step2_settings": {}, "workflow": [{"step": "step2", "mode": "parallel"}]}"#,
            r#"{"step1_settings": {}, "workflow": [{"step": "step1", "settings_key": "other"}]}"#,
        ] {
            assert!(
                Workflow::parse(&settings(wrong), &ready).is_err(),
                "{wrong}"
            );
        }
    }

    #[test]
    fn trajectories_are_joined_without_the_structure_they_share() {
        let dir = std::env::temp_dir().join(format!("mopt_autots_join_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let frame = |d: f64| format!("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 {d}\n");
        let first = dir.join("first.xyz");
        std::fs::write(&first, frame(0.7) + &frame(0.8) + &frame(0.9)).unwrap();
        let second = dir.join("second.xyz");
        std::fs::write(&second, frame(0.9) + &frame(1.0)).unwrap();
        let out = dir.join("joined.xyz");
        assert_eq!(joined_trajectories(&[first, second], &out).unwrap(), 4);
        let read = read_xyz_trajectory_file(&out, SpinCharge::default(), None).unwrap();
        assert_eq!(read.frames.len(), 4);
        assert!(joined_trajectories(&[], &out).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
