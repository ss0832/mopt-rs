//! The hard constraints of `run_autots` in every step: {frozen atoms, a distance, a centroid
//! distance of two fragments (`fbond`), an angle, a rigid group (`-gfix`), a user-defined
//! function (`-uc`)} x {an isolated cluster, a cluster in a fixed periodic cell}, given on the
//! command line as options shared by every step (merged into every step block). Every structure
//! the run writes (the AFIR path, the chains, the candidates, the transition states, the IRC and
//! its ends) holds the constraints at the values of the input within 1e-10 and the frozen atoms
//! where they are in the input; a step whose own constraints differ is reported by name; a
//! variable cell with constraints is refused by name.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use std::path::{Path, PathBuf};

use matrix_support::{check, frames, kinds, read, scratch, write, FROM_FILES, FROZEN_IN_FILES};
use mopt_rs::cli::autots::{run_with, AutotsConfig, AutotsResult};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::pes::registry::CalculatorRegistry;

const AR7: &str = "7\nAr7\nAr 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

const AR7_CELL: &str = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

/// AFIR between atoms 3 and 5 (the push of step 1), a short chain, one candidate, its IRC.
pub const SETTINGS: &str = r#"{
  "work_dir": "work",
  "top_n_candidates": 1,
  "run_step4": true,
  "step1_settings": {"othersoft": "lj", "manual_AFIR": ["5", "3", "5"], "NSTEP": 150},
  "step2_settings": {"othersoft": "lj", "NSTEP": 10},
  "step3_settings": {"othersoft": "lj", "NSTEP": 100, "calc_exact_hess": 5, "tight_convergence_criteria": true},
  "step4_settings": {"othersoft": "lj", "NSTEP": 200, "tight_convergence_criteria": true,
                     "intrinsic_reaction_coordinates": ["0.5", "60", "lqa"]}
}"#;

/// The settings of a run of `text` with `settings` and `extra` options in `dir/name`.
pub fn autots_config(
    dir: &Path,
    name: &str,
    text: &str,
    settings: &str,
    extra: &[String],
) -> AutotsConfig {
    let d = dir.join(name);
    std::fs::create_dir_all(&d).unwrap();
    let cfg = write(&d, "config.json", settings);
    let file = write(&d, "start.xyz", text);
    let mut all = vec![
        file.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    all.extend(extra.iter().cloned());
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    AutotsConfig::from_args(&parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap()).unwrap()
}

/// Every xyz file under `dir`, recursively.
fn xyz_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let p = entry.unwrap().path();
            // The animations of the normal modes show displacements along the modes, not
            // structures of the run.
            if p.is_dir() && !p.ends_with("vibration_animation") {
                todo.push(p);
            } else if p.extension().is_some_and(|e| e == "xyz") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn context(name: &str, text: &str) {
    context_with(name, text, SETTINGS, [1, 2, 6, 7], (&[1, 2, 6], &[4, 7]));
}

/// [`context`] with the settings `settings` and the constraints of [`kinds`] on `atoms` and
/// `fragments`.
fn context_with(
    name: &str,
    text: &str,
    settings: &str,
    atoms: [usize; 4],
    fragments: (&[usize], &[usize]),
) {
    let dir = scratch("atsmatrix", name);
    for (kind, options) in kinds(atoms, fragments) {
        let cell = format!("{name} x {kind}");
        let config = autots_config(&dir, kind, text, settings, &options);
        let input = read(&config.input);
        let base = config.input.parent().unwrap().to_path_buf();
        let result: AutotsResult = run_with(&config, &base, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{cell}: {e}"));
        // The constraints of every step are those of the command line.
        let third = config.runs(2);
        assert!(third[0].contains_key(match kind {
            "frozen" => "fix_atoms",
            "gfix" => "gradient_fix_atoms",
            "uc" => "user_constraint",
            _ => "projection_constrain",
        }));
        let constraints = mopt_rs::cli::autots::step_optmain_config(&third[0], &config.input)
            .unwrap()
            .settings
            .constraints;
        let files = xyz_files(&result.directory);
        eprintln!(
            "{cell}: {} files, {} transition state(s)",
            files.len(),
            result.transition_states.len()
        );
        for f in &files {
            let name = f
                .strip_prefix(&result.directory)
                .unwrap_or(f)
                .display()
                .to_string();
            check(
                &format!("{cell} ({name})"),
                &frames(f),
                &constraints,
                &input,
                FROM_FILES,
                FROZEN_IN_FILES,
                false,
            );
        }
        assert!(
            !result.notes.iter().any(|n| n.contains("same surface")),
            "{cell}: {:?}",
            result.notes
        );
        assert!(
            result.surface.iter().any(|s| s.starts_with("Constraints")),
            "{cell}"
        );
        let page = std::fs::read_to_string(result.directory.join("autots_report.html")).unwrap();
        assert!(page.contains("Constraints of the surface"), "{cell}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_autots_on_an_isolated_cluster() {
    context("isolated", AR7);
}

#[test]
fn constraints_of_autots_in_a_periodic_cell() {
    context("periodic", AR7_CELL);
}

#[test]
fn constraints_of_autots_with_oniom_microiterations() {
    // Atoms 1-4 are the core (its own Lennard-Jones layer), 5-7 the environment relaxed by
    // the microiterations of every step; the constraints on atoms of the core.
    let settings = SETTINGS.replace(
        "{\"othersoft\": \"lj\",",
        "{\"othersoft\": \"lj\", \"oniom_flag\": [\"1-4\", \"none\", \"lj\"],",
    );
    context_with("oniom", AR7, &settings, [1, 2, 3, 4], (&[1, 2], &[3, 4]));
}

#[test]
fn constraints_that_differ_between_the_steps_are_reported_by_name() {
    let dir = scratch("atsmatrix", "differ");
    let settings = SETTINGS.replace(
        "\"calc_exact_hess\": 5, \"tight_convergence_criteria\": true}",
        "\"calc_exact_hess\": 5, \"tight_convergence_criteria\": true, \"fix_atoms\": [\"2\"]}",
    );
    let options: Vec<String> = ["-fix", "1"].iter().map(|s| s.to_string()).collect();
    let config = autots_config(&dir, "differ", AR7, &settings, &options);
    assert!(config
        .notes
        .iter()
        .any(|n| n.contains("step3_settings sets 'fix_atoms'")));
    let base = config.input.parent().unwrap().to_path_buf();
    let result = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    assert!(
        result.notes.iter().any(
            |n| n.contains("Steps 2 and 3 do not run on the same surface")
                && n.contains("the constraints")
        ),
        "{:?}",
        result.notes
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_with_a_variable_cell_are_refused_by_the_irc() {
    let dir = scratch("atsmatrix", "vc");
    // The variable cell in the optimizations (steps 1, 3 and 4; nebmain has no -vc).
    let settings = SETTINGS
        .replace(
            "\"step1_settings\": {\"othersoft\": \"lj\",",
            "\"step1_settings\": {\"othersoft\": \"lj\", \"variable_cell\": true,",
        )
        .replace(
            "\"step3_settings\": {\"othersoft\": \"lj\",",
            "\"step3_settings\": {\"othersoft\": \"lj\", \"variable_cell\": true,",
        )
        .replace(
            "\"step4_settings\": {\"othersoft\": \"lj\",",
            "\"step4_settings\": {\"othersoft\": \"lj\", \"variable_cell\": true,",
        );
    // A constraint of the cell (-vc_fix): refused when the settings are read.
    {
        let d = dir.join("vc_fix");
        std::fs::create_dir_all(&d).unwrap();
        let cfg = write(&d, "config.json", &settings);
        let file = write(&d, "start.xyz", AR7_CELL);
        let all = [
            file.display().to_string(),
            "-cfg".to_string(),
            cfg.display().to_string(),
            "-vc_fix".to_string(),
            "a".to_string(),
        ];
        let refs: Vec<&str> = all.iter().map(String::as_str).collect();
        let e = AutotsConfig::from_args(&parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap())
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("-vc_fix") && e.contains("nebmain"), "{e}");
    }
    // Every kind of constraint.
    for (kind, options) in kinds([1, 2, 6, 7], (&[1, 2, 6], &[4, 7])) {
        let config = autots_config(&dir, kind, AR7_CELL, &settings, &options);
        let base = config.input.parent().unwrap().to_path_buf();
        let e = run_with(&config, &base, None, &CalculatorRegistry::native())
            .map(|r| r.error.unwrap_or_default())
            .unwrap_or_else(|e| e.to_string());
        // Refused by name: by the IRC of step 4 (optmain -vc -irc holds no constraints), or
        // before it by the chain of step 2, whose images keep the cell of the first one.
        assert!(
            e.contains("constrained -vc") || e.contains("a chain whose cell changes"),
            "{kind}: {e}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
