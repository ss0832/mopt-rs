//! Tests of the settings of `run_mapper`: the command line over the file, the steps on one surface, and the refusals.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;
use crate::cli::option::parser;
use std::sync::atomic::{AtomicUsize, Ordering};

static CASE: AtomicUsize = AtomicUsize::new(0);

fn config(settings: &str, args: &[&str]) -> MoptResult<MapperConfig> {
    let dir = std::env::temp_dir().join(format!(
        "mopt_mapper_settings_{}_{}",
        std::process::id(),
        CASE.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("ar2.xyz");
    std::fs::write(&input, "2\n0 1\nAr 0 0 0\nAr 3.8 0 0\n").unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, settings).unwrap();
    let mut all = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    all.extend(args.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(all).unwrap();
    let out = MapperConfig::from_args(&parsed, None);
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
fn the_command_line_goes_over_the_file_and_the_steps_share_the_surface() {
    let c = config(
        r#"{"mapper_settings": {"temperature_K": 500, "max_iterations": 7,
            "include_negative_gamma": true, "priority": "tabu", "method_depth": 3},
            "top_n_candidates": 4, "work_dir": "x", "step2_settings": {},
            "step1_settings": {"othersoft": "lj", "manual_AFIR": ["100", "1", "2"]},
            "step4_settings": {"intrinsic_reaction_coordinates": ["0.5", "50", "lqa"]}}"#,
        &[
            "--temperature",
            "350",
            "--active_atoms",
            "1",
            "2",
            "-map_opt",
            "stop=nothing_new:4",
        ],
    )
    .unwrap();
    let s = &c.settings;
    assert_eq!(s.temperature, 350.0);
    assert_eq!(s.max_jobs, 7);
    assert!(s.afir.negative_gamma);
    assert_eq!(s.afir.active_atoms, Some(vec![0, 1]));
    assert_eq!(s.priority, "tabu");
    assert_eq!(s.stop, vec!["nothing_new:4".to_string()]);
    assert_eq!(s.top_n, 4);
    assert_eq!(s.method_extra.get("depth").map(String::as_str), Some("3"));
    // Steps 3 and 4 run on the calculator of step 1; the IRC of step 4.
    assert_eq!(c.steps.saddle.calculator, "lj");
    assert_eq!(c.steps.minimum.calculator, "lj");
    assert_eq!(c.steps.saddle.settings.saddle_order, 1);
    assert_eq!(c.steps.irc.max_steps, 50);
    assert!(c.steps.biased.biases.is_empty(), "manual_AFIR is the job's");
    assert!(c.notes.iter().any(|n| n.contains("work_dir")));
    // step2_settings are the settings of the NEB that relaxes the AFIR paths of afir_ts.
    assert!(!c.notes.iter().any(|n| n.contains("step2_settings")));
    assert!(c.notes.iter().any(|n| n.contains("manual_AFIR")));
    assert_eq!(c.output_dir, PathBuf::from("mapper_output"));
}

#[test]
fn the_jobs_at_the_same_time_come_from_the_options_of_both_programs() {
    use crate::workflow::mapper::{Schedule, ScheduleMode};
    let lj = r#""step1_settings": {"othersoft": "lj"}"#;
    let schedule =
        |settings: String, args: &[&str]| config(&settings, args).unwrap().settings.schedule;
    assert_eq!(schedule(format!("{{{lj}}}"), &[]), Schedule::default());
    // MultiOptPy's n_parallel and worker_timeout_s; the batch is as large as the workers.
    let s = schedule(
        format!(r#"{{"mapper_settings": {{"n_parallel": 3, "worker_timeout_s": 90}}, {lj}}}"#),
        &[],
    );
    assert_eq!((s.workers, s.batch, s.mode), (3, 3, ScheduleMode::Batch));
    assert_eq!(s.time_limit, Some(std::time::Duration::from_secs(90)));
    // workers over n_parallel, --max_workers over both; batch_size and schedule of mopt-rs.
    let s = schedule(
        format!(r#"{{"mapper_settings": {{"n_parallel": 3, "workers": 2}}, {lj}}}"#),
        &["-map_opt", "batch_size=6", "schedule=rolling"],
    );
    assert_eq!((s.workers, s.batch, s.mode), (2, 6, ScheduleMode::Rolling));
    let s = schedule(
        format!(r#"{{"mapper_settings": {{"workers": 2}}, {lj}}}"#),
        &["--max_workers", "4"],
    );
    assert_eq!((s.workers, s.batch), (4, 4));
    for bad in [
        &["-map_opt", "workers=0"][..],
        &["-map_opt", "batch_size=0"],
        &["-map_opt", "schedule=eager"],
        &["--max_workers", "0"],
    ] {
        assert!(config(&format!("{{{lj}}}"), bad).is_err(), "{bad:?}");
    }
}

#[test]
fn what_is_not_there_yet_is_refused() {
    let lj = r#""step1_settings": {"othersoft": "lj"}"#;
    // The kinetics at the end is accepted now.
    let on = format!(r#"{{"mapper_settings": {{"kinetics": "on"}}, {lj}}}"#);
    assert!(config(&on, &[]).is_ok_and(|c| c.settings.kinetics));
    let cases: Vec<(String, &[&str], bool)> = vec![
        (
            format!(r#"{{"mapper_settings": {{"use_rcmc": true, "priority": "tabu"}}, {lj}}}"#),
            &[],
            false,
        ),
        (
            format!(r#"{{"mapper_settings": {{"kinetics": "rcmc"}}, {lj}}}"#),
            &[],
            false,
        ),
        (
            format!(r#"{{"mapper_settings": {{"n_parallel": 0}}, {lj}}}"#),
            &[],
            false,
        ),
        (
            format!(r#"{{"mapper_settings": {{"worker_timeout_s": -9}}, {lj}}}"#),
            &[],
            false,
        ),
        (
            format!(r#"{{"mapper_settings": {{"refine": "on"}}, {lj}}}"#),
            &[],
            true,
        ),
        (
            format!(r#"{{"mapper_settings": {{"temperature": 1}}, {lj}}}"#),
            &[],
            false,
        ),
        (
            format!(r#"{{"mapper_settings": {{"max_pairs": "x"}}, {lj}}}"#),
            &[],
            false,
        ),
        (format!(r#"{{"other": 1, {lj}}}"#), &[], false),
        (format!(r#"{{{lj}}}"#), &["--dist_lower", "6"], false),
        // Crossing points are built: without the states they name, a mistake in the settings.
        (format!(r#"{{{lj}}}"#), &["-map_opt", "crossings=on"], false),
        (format!(r#"{{{lj}}}"#), &["-map_opt", "unknown=1"], false),
        (
            format!(r#"{{{lj}, "step3_settings": {{"othersoft": "emt"}}}}"#),
            &[],
            true,
        ),
        (
            format!(r#"{{{lj}, "step3_settings": {{"-kp": ["1.0", "1.5", "1,2"]}}}}"#),
            &[],
            true,
        ),
    ];
    for (settings, args, unsupported) in cases {
        let e = config(&settings, args).err().unwrap();
        assert_eq!(
            matches!(e, MoptError::Unsupported(_)),
            unsupported,
            "{settings} {args:?}: {e}"
        );
    }
}
