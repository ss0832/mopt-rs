//! `run_autots` on objectives of plain variables: the four steps on the Muller-Brown surface of
//! two variables (`mb2d`) from its minima A and B, whose path runs through the minimum C and
//! both saddle points, and on the Rosenbrock function of four variables between its two minima.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::autots::{run_with, AutotsConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::pes::model_objectives::ROSENBROCK_4_POINTS;
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_autots_vars_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Frames of variables: the number of variables, a comment, `name value` lines.
fn frames(names: &[&str], points: &[&[f64]]) -> String {
    let mut out = String::new();
    for (k, p) in points.iter().enumerate() {
        out.push_str(&format!("{}\npoint {}\n", names.len(), k + 1));
        for (n, v) in names.iter().zip(p.iter()) {
            out.push_str(&format!("{n} {v:.10}\n"));
        }
    }
    out
}

fn run(dir: &Path, input: &str, settings: &str) -> mopt_rs::cli::autots::AutotsResult {
    let path = dir.join("start.txt");
    std::fs::write(&path, input).unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, settings).unwrap();
    let args = [
        path.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--run_step4".to_string(),
    ];
    let p = parser("run_autots", RUN_AUTOTS).parse(&args).unwrap();
    let c = AutotsConfig::from_args(&p).unwrap();
    let result = run_with(&c, dir, None, &CalculatorRegistry::native()).unwrap();
    let page = std::fs::read_to_string(result.directory.join("autots_report.html")).unwrap();
    for stage in ["step1", "step2", "step3", "step4"] {
        assert!(page.contains(&format!("id=\"{stage}\"")), "{stage}");
    }
    let initial = result.band_profiles.first().unwrap();
    assert_eq!(initial.iteration, 0);
    assert_eq!(
        page.matches("<div data-band").count(),
        result.band_profiles.len()
    );
    assert!(page.contains("Value relative to the first image of the initial path"));
    let first = std::fs::read(result.afir_path.as_ref().unwrap()).unwrap();
    let encoded = mopt_rs::cli::autots_static::base64_bytes(&first);
    assert!(page.contains(&format!("data:text/plain;base64,{encoded}")));
    assert!(page.contains("ts1_optimization.txt") && page.contains("ts1_irc.txt"));
    result
}

fn near(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt()
}

/// The values of a variables file of the run.
fn values(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        .collect()
}

#[test]
fn muller_brown_from_a_to_b_gives_both_saddle_points_and_their_irc() {
    let dir = scratch("mb2d");
    let input = frames(&["x", "y"], &[&MB_MINIMA[0], &MB_MINIMA[1]]);
    let settings = r#"{
  "work_dir": "work",
  "top_n_candidates": 3,
  "step1_settings": {"othersoft": "mb2d"},
  "step2_settings": {"NSTEP": 100},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {"tight_convergence_criteria": true, "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]}
}"#;
    let r = run(&dir, &input, settings);
    let log = std::fs::read_to_string(r.directory.join("autots.log")).unwrap_or_default();
    let ts: Vec<Vec<f64>> = r
        .transition_states
        .iter()
        .filter(|t| t.converged && t.imaginary_modes == Some(1) && t.same_as.is_none())
        .map(|t| values(&t.file))
        .collect();
    for s in MB_SADDLES {
        assert!(
            ts.iter().any(|t| near(t, &s) < 1e-3),
            "saddle {s:?} not among {ts:?}\n{log}"
        );
    }
    // Every IRC ends at two of the three minima.
    for t in r.transition_states.iter().filter(|t| !t.ends.is_empty()) {
        assert_eq!(t.ends.len(), 2, "{log}");
        for (file, _) in &t.ends {
            let v = values(file);
            assert!(MB_MINIMA.iter().any(|m| near(&v, m) < 1e-3), "{v:?}\n{log}");
        }
    }
    // The steps counted their evaluations, the summary and the page are written.
    assert!(
        r.evaluations.iter().all(|e| e.gradients > 0),
        "{:?}",
        r.evaluations
    );
    assert!(log.contains("evaluations by step"));
    let csv = std::fs::read_to_string(r.directory.join("autots_summary.csv")).unwrap();
    assert!(csv.starts_with("ts,file,value,"));
    let page = std::fs::read_to_string(r.directory.join("autots_report.html")).unwrap();
    assert!(
        page.contains("<h2>Map</h2>") && page.contains("<rect") && page.contains("Evaluations")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn four_variables_of_rosenbrock_between_its_two_minima() {
    let dir = scratch("rosenbrock");
    let names = ["x1", "x2", "x3", "x4"];
    let input = frames(
        &names,
        &[&ROSENBROCK_4_POINTS[0].at, &ROSENBROCK_4_POINTS[1].at],
    );
    let settings = r#"{
  "work_dir": "work",
  "top_n_candidates": 2,
  "step1_settings": {"othersoft": "rosenbrock"},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {"tight_convergence_criteria": true, "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.02", "600", "lqa"]}
}"#;
    let r = run(&dir, &input, settings);
    let log = std::fs::read_to_string(r.directory.join("autots.log")).unwrap_or_default();
    let saddle = ROSENBROCK_4_POINTS[2].at;
    assert!(
        r.transition_states.iter().any(|t| t.converged
            && t.imaginary_modes == Some(1)
            && near(&values(&t.file), &saddle) < 1e-3),
        "{log}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
