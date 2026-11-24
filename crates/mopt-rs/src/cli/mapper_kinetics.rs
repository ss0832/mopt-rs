//! The kinetics at the end of a run of `run_mapper` (`-map_opt kinetics=on`): when the
//! exploration ends (not when it is interrupted), the kinetics of the level-1 network runs with
//! the `kinetics_*` settings (those of `priority=kinetics`), and its files go to
//! `RUN_DIR/kinetics/`: the files of `run_kinetics` and [`SUMMARY_FILE`], which the report of
//! the run reads (`docs/source/mapper_design.md`, "Kinetic parameters from the mapper").
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::core_infra::json::JsonValue;
use crate::core_infra::log::Logger;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::KineticsParts;
use crate::workflow::kinetics::{kinetic_network, run_job, KineticsJob, KineticsRun, Report};
use crate::workflow::mapper::network::Level;
use crate::workflow::mapper::store::write_whole;
use crate::workflow::mapper::{ExplorationEnd, KineticChoice, MapperObserver, Network};

/// The directory of the kinetics in the run directory.
pub const KINETICS_DIR: &str = "kinetics";

/// The summary of the kinetics that the report reads, in [`KINETICS_DIR`].
pub const SUMMARY_FILE: &str = "kinetics_summary.json";

/// The format name of [`SUMMARY_FILE`].
pub const SUMMARY_FORMAT: &str = "mopt-rs mapper kinetics";

/// Decades of the output times below the time of interest: the times are
/// `10^-k kinetics_time` for `k` from this down to 0.
pub const OUTPUT_DECADES: i32 = 12;

/// The output times of a time of interest `time` (s), ascending, the last `time`.
pub fn output_times(time: f64) -> Vec<f64> {
    (0..=OUTPUT_DECADES)
        .rev()
        .map(|k| time * 10f64.powi(-k))
        .collect()
}

/// The kinetics at the end of a run, as an observer of its exploration.
pub struct EndKinetics {
    choice: KineticChoice,
    dir: Option<PathBuf>,
    log: Logger,
}

fn text(s: &str) -> JsonValue {
    JsonValue::String(s.to_string())
}

fn num(x: f64) -> JsonValue {
    if x.is_finite() {
        JsonValue::Number(x)
    } else {
        JsonValue::Null
    }
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

impl EndKinetics {
    /// The kinetics of the `kinetics_*` settings `options` (without the prefix) at the
    /// temperature of the run `temperature` (K); what is unknown is refused here.
    pub fn new(options: &BTreeMap<String, String>, temperature: f64) -> MoptResult<Self> {
        Ok(Self {
            choice: KineticChoice::new(temperature, options)?,
            dir: None,
            log: Logger::silent(),
        })
    }

    /// Write to the run directory `dir` and tell `log`.
    pub fn set_directory(&mut self, dir: &Path, log: Logger) {
        self.dir = Some(dir.to_path_buf());
        self.log = log;
    }

    /// The settings as the summary records them.
    fn settings_entry(&self, start: Option<usize>) -> JsonValue {
        let s = &self.choice.settings;
        object(vec![
            ("temperature_K", num(s.conditions.temperature)),
            ("time_s", num(self.choice.time)),
            (
                "start_node",
                start.map_or(JsonValue::Null, |id| JsonValue::Number(id as f64)),
            ),
            ("energy", text(s.conditions.energy.name())),
            ("clamp", text(s.conditions.clamp.name())),
            ("rate_theory", text(&s.rate_theory)),
            (
                "corrections",
                JsonValue::Array(s.corrections.iter().map(|c| text(c)).collect()),
            ),
            ("solver", text(&s.solver)),
            ("score", text(self.choice.score.name())),
        ])
    }

    /// The start node: the setting, or the node of the first input.
    fn start(&self, network: &Network) -> Option<usize> {
        self.choice
            .start
            .or_else(|| network.inputs.iter().find_map(|i| i.node))
    }

    /// Run the kinetics of `network` (whose file is in `dir`) into `dir/kinetics`; returns the
    /// summary that was written.
    pub fn run(&self, dir: &Path, network: &Network) -> MoptResult<JsonValue> {
        let out = dir.join(KINETICS_DIR);
        let start = self.start(network);
        let counts = |doc: &mut Vec<(&str, JsonValue)>| {
            doc.push(("jobs", JsonValue::Number(network.jobs.len() as f64)));
            doc.push(("nodes", JsonValue::Number(network.nodes.len() as f64)));
        };
        let mut doc = vec![
            ("format", text(SUMMARY_FORMAT)),
            ("version", JsonValue::Number(1.0)),
            ("settings", self.settings_entry(start)),
        ];
        counts(&mut doc);
        let k = kinetic_network(network, Level::Exploration);
        if k.bottlenecks.is_empty() || k.states.is_empty() {
            doc.push(("computed", JsonValue::Bool(false)));
            doc.push((
                "reason",
                text("the network has no transition state (or no minimum) yet: no kinetics"),
            ));
            let summary = object(doc);
            write_whole(
                &out.join(SUMMARY_FILE),
                (summary.to_string_pretty() + "\n").as_bytes(),
            )?;
            return Ok(summary);
        }
        let job = KineticsJob {
            network: dir.to_path_buf(),
            level: Level::Exploration,
            annotations: None,
            settings: self.choice.settings.clone(),
            times: output_times(self.choice.time),
            initial: start.map(|id| vec![(id, 1.0)]).unwrap_or_default(),
            validate: None,
            sensitivity: None,
            uncertainty: None,
            priority: Some(self.choice.score),
            lump_conformers: false,
            spectrum: false,
            output: Some(out.clone()),
            report: Report::Kinetics,
            settings_document: Some(self.settings_entry(start).to_string_pretty()),
            cli_annotations: None,
            extras: Default::default(),
        };
        let run = run_job(&job, &KineticsParts::default(), &self.log)?;
        doc.push(("computed", JsonValue::Bool(true)));
        doc.extend(results(&run));
        let summary = object(doc);
        write_whole(
            &out.join(SUMMARY_FILE),
            (summary.to_string_pretty() + "\n").as_bytes(),
        )?;
        Ok(summary)
    }
}

/// The times, the populations of every state, and the notes of `run`.
fn results(run: &KineticsRun) -> Vec<(&'static str, JsonValue)> {
    let rates = &run.prepared.rates;
    let sol = &run.solution;
    let states: Vec<JsonValue> = rates
        .states
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            object(vec![
                ("node", JsonValue::Number(run.network.states[k].id as f64)),
                ("label", text(&rates.labels[i])),
                (
                    "populations",
                    JsonValue::Array(
                        sol.populations
                            .iter()
                            .map(|p| num(p.get(i).copied().unwrap_or(f64::NAN)))
                            .collect(),
                    ),
                ),
            ])
        })
        .collect();
    let mut notes: Vec<JsonValue> = run.network.notes.iter().map(|n| text(n)).collect();
    notes.extend(rates.notes.iter().map(|n| text(n)));
    notes.extend(sol.notes.iter().map(|n| text(n)));
    vec![
        ("description", text(&rates.description)),
        ("solver_used", text(&sol.solver)),
        ("steps", JsonValue::Number(rates.steps.len() as f64)),
        (
            "times_s",
            JsonValue::Array(sol.times.iter().map(|&t| num(t)).collect()),
        ),
        ("states", JsonValue::Array(states)),
        ("notes", JsonValue::Array(notes)),
    ]
}

impl MapperObserver for EndKinetics {
    fn on_finish(&mut self, network: &Network, end: &ExplorationEnd) -> MoptResult<()> {
        if end.interrupted {
            return Ok(());
        }
        let Some(dir) = self.dir.clone() else {
            return Err(MoptError::config(
                "run_mapper: the kinetics at the end has no run directory",
            ));
        };
        let start = std::time::Instant::now();
        match self.run(&dir, network) {
            Ok(summary) => {
                if summary.get("computed").and_then(JsonValue::as_bool) == Some(true) {
                    self.log.info(&format!(
                        "Kinetics at the end ({:.2} s): {}; {}",
                        start.elapsed().as_secs_f64(),
                        final_line(&summary),
                        dir.join(KINETICS_DIR).display()
                    ));
                } else if let Some(reason) = summary.get("reason").and_then(JsonValue::as_str) {
                    self.log.info(&format!("Kinetics at the end: {reason}."));
                }
            }
            Err(e) => {
                // The run itself has ended well: its network stands, and the kinetics can be
                // computed afterwards with run_kinetics RUN_DIR.
                self.log.warn(&format!(
                    "The kinetics at the end could not be computed ({e}); run_kinetics {} \
                     computes it from the network file.",
                    dir.display()
                ));
                let doc = vec![
                    ("format", text(SUMMARY_FORMAT)),
                    ("version", JsonValue::Number(1.0)),
                    ("settings", self.settings_entry(self.start(network))),
                    ("jobs", JsonValue::Number(network.jobs.len() as f64)),
                    ("nodes", JsonValue::Number(network.nodes.len() as f64)),
                    ("computed", JsonValue::Bool(false)),
                    ("reason", text(&format!("the kinetics failed: {e}"))),
                ];
                write_whole(
                    &dir.join(KINETICS_DIR).join(SUMMARY_FILE),
                    (object(doc).to_string_pretty() + "\n").as_bytes(),
                )?;
            }
        }
        Ok(())
    }
}

/// The states with the largest populations at the time of interest, for the log.
pub fn final_line(summary: &JsonValue) -> String {
    let mut last: Vec<(String, f64)> = summary
        .get("states")
        .and_then(JsonValue::as_array)
        .map(|states| {
            states
                .iter()
                .filter_map(|s| {
                    let label = s.get("label")?.as_str()?.to_string();
                    let p = s.get("populations")?.as_array()?.last()?.as_f64()?;
                    Some((label, p))
                })
                .collect()
        })
        .unwrap_or_default();
    last.sort_by(|a, b| b.1.total_cmp(&a.1));
    let time = summary
        .get("settings")
        .and_then(|s| s.get("time_s"))
        .and_then(JsonValue::as_f64)
        .unwrap_or(f64::NAN);
    format!(
        "populations after {time:e} s: {}",
        last.iter()
            .take(4)
            .map(|(l, p)| format!("{l} {:.1} %", 100.0 * p))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The summary of the kinetics in the run directory `dir`, when there is one.
pub fn read_summary(dir: &Path) -> Option<JsonValue> {
    let content = std::fs::read_to_string(dir.join(KINETICS_DIR).join(SUMMARY_FILE)).ok()?;
    let doc = JsonValue::parse(&content).ok()?;
    (doc.get("format").and_then(JsonValue::as_str) == Some(SUMMARY_FORMAT)).then_some(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_output_times_end_at_the_time_of_interest() {
        let t = output_times(2.0);
        assert_eq!(t.len(), 13);
        assert!((t[0] - 2e-12).abs() < 1e-24);
        assert_eq!(t[12], 2.0);
        assert!(t.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn unknown_settings_are_refused_before_the_run() {
        let mut o = BTreeMap::new();
        o.insert("solver".to_string(), "nope".to_string());
        assert!(EndKinetics::new(&o, 300.0).is_err());
        o.clear();
        o.insert("time".to_string(), "1e-3".to_string());
        assert!(EndKinetics::new(&o, 300.0).is_ok());
    }
}
