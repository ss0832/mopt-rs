//! `run_autots --restart WORK_DIR`: a run made again in its directory takes what the run before
//! evaluated from the journals under it and computes only what is missing (the migration of
//! a vacancy in copper with the built-in EAM, steps 2 to 4).
//!
//! MultiOptPy: no counterpart (MultiOptPy cannot resume a run).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::autots::{run_with, AutotsConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::run_command_with;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::pes::registry::CalculatorRegistry;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mopt_autots_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Copper with a vacancy in a 2x2x2 cell of the fcc conventional cell (31 atoms), before and
/// after a nearest neighbour moves into it.
fn copper_vacancy(path: &Path) {
    let a = 3.615;
    let mut sites = Vec::new();
    for (i, j, k) in (0..2).flat_map(|i| (0..2).flat_map(move |j| (0..2).map(move |k| (i, j, k)))) {
        for b in [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ] {
            sites.push([
                a * (i as f64 + b[0]),
                a * (j as f64 + b[1]),
                a * (k as f64 + b[2]),
            ]);
        }
    }
    let start: Vec<[f64; 3]> = sites[1..].to_vec();
    let mut end = start.clone();
    end[0] = [0.0, 0.0, 0.0];
    let lattice = format!(
        "Lattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"",
        l = 2.0 * a
    );
    let mut text = String::new();
    for frame in [&start, &end] {
        text.push_str(&format!("{}\n{lattice}\n", frame.len()));
        for x in frame {
            text.push_str(&format!("Cu {:.6} {:.6} {:.6}\n", x[0], x[1], x[2]));
        }
    }
    std::fs::write(path, text).unwrap();
}

/// The journal files under `dir` whose directory name contains `part`.
fn journals_in(dir: &Path, part: &str, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            journals_in(&path, part, out);
        } else if path.file_name().is_some_and(|f| f == "evaluations.journal")
            && path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|f| f.to_string_lossy().contains(part))
        {
            out.push(path);
        }
    }
}

#[test]
fn a_run_made_again_takes_what_was_evaluated_and_computes_the_rest() {
    let d = dir("restart");
    std::fs::write(
        d.join("config.json"),
        r#"{
  "work_dir": "work",
  "top_n_candidates": 1,
  "step2_settings": {"othersoft": "eam", "NSTEP": 40},
  "step3_settings": {"othersoft": "eam", "calc_exact_hess": 5,
                     "tight_convergence_criteria": true},
  "step4_settings": {"othersoft": "eam", "calc_exact_hess": 10,
                     "tight_convergence_criteria": true,
                     "intrinsic_reaction_coordinates": ["0.3", "200", "lqa"]}
}"#,
    )
    .unwrap();
    let input = d.join("vacancy.xyz");
    copper_vacancy(&input);
    let all = [
        input.to_string_lossy().into_owned(),
        "-cfg".to_string(),
        d.join("config.json").to_string_lossy().into_owned(),
        "--skip_step1".to_string(),
        "--run_step4".to_string(),
    ];
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    let p = parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap();
    let c = AutotsConfig::from_args(&p).unwrap();
    let first_registry = CalculatorRegistry::native();
    let first = run_with(&c, &d, None, &first_registry).unwrap();
    let computed = first_registry.counts().gradients;
    assert!(computed > 0);
    let work = d.join("work");
    assert!(work.join("settings.json").is_file() && work.join("autots_config.json").is_file());
    let ts = &first.transition_states[0];
    assert!(ts.converged && ts.ends.len() == 2, "{ts:?}");

    // Made again: every evaluation comes from the journals, and the result is the same.
    let again = CalculatorRegistry::native();
    let code = run_command_with(
        "run_autots",
        &["--restart".to_string(), work.to_string_lossy().into_owned()],
        &again,
    );
    assert_eq!(code, 0);
    assert_eq!(again.counts().gradients, 0, "{:?}", again.counts());
    let summary = std::fs::read_to_string(work.join("autots_summary.csv")).unwrap();
    assert!(summary.contains(&format!("{:.12}", ts.energy)), "{summary}");

    // As if the run had been killed in the IRC of step 4: its journal and those of the runs
    // after it are gone; the run made again computes those and takes the rest.
    let mut lost = Vec::new();
    journals_in(&work, "IRC", &mut lost);
    assert!(!lost.is_empty());
    for file in &lost {
        std::fs::remove_file(file).unwrap();
    }
    let third = CalculatorRegistry::native();
    let code = run_command_with(
        "run_autots",
        &["--restart".to_string(), work.to_string_lossy().into_owned()],
        &third,
    );
    assert_eq!(code, 0);
    let redone = third.counts().gradients;
    assert!(redone > 0 && redone < computed, "{redone} of {computed}");
    let summary = std::fs::read_to_string(work.join("autots_summary.csv")).unwrap();
    for (_, energy) in &ts.ends {
        assert!(summary.contains(&format!("{energy:.12}")), "{summary}");
    }
    let _ = std::fs::remove_dir_all(&d);
}
