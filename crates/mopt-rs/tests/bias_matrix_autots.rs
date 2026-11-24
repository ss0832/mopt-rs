//! The static bias potentials of `run_autots` in every step: {a restraint (`-kp`), an
//! absolute-position bias (`-ubias` with `x()`, `y()`, `z()`), a wall (`-wwp`)} x {an isolated
//! cluster, a cluster in a fixed periodic cell}, given on the command line as options shared by
//! every step. Every transition state has the energy of the calculator alone and that with the
//! bias, consistent with a recomputation from its file; the steps run on one surface; a step
//! with another bias is reported by name, and `-ma` with a workflow block pushes every run of
//! step 1.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use std::path::Path;

use matrix_support::{bias_energy, read, scratch, write};
use mopt_rs::cli::autots::{run_with, AutotsConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::domain::{AtomId, EvalRequest, MolecularSystem};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;

const AR7: &str = "7\nAr7\nAr 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

const AR7_CELL: &str = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

const SETTINGS: &str = r#"{
  "work_dir": "work",
  "top_n_candidates": 1,
  "run_step4": true,
  "step1_settings": {"othersoft": "lj", "manual_AFIR": ["5", "3", "5"], "NSTEP": 150},
  "step2_settings": {"othersoft": "lj", "NSTEP": 10},
  "step3_settings": {"othersoft": "lj", "NSTEP": 100, "calc_exact_hess": 5, "tight_convergence_criteria": true},
  "step4_settings": {"othersoft": "lj", "NSTEP": 200, "tight_convergence_criteria": true,
                     "intrinsic_reaction_coordinates": ["0.5", "60", "lqa"]}
}"#;

fn biases(input: &MolecularSystem) -> Vec<(&'static str, String, Vec<String>)> {
    let p = input.position(AtomId::new(0)).value();
    let goal = [p[0] + 0.4, p[1] - 0.2, p[2] + 0.1];
    vec![
        (
            "restraint",
            "-kp".to_string(),
            vec!["0.002".into(), "3.90".into(), "1,2".into()],
        ),
        (
            "absolute",
            "-ubias".to_string(),
            vec![format!(
                "0.002 * ((x(1) - {:.10})^2 + (y(1) - {:.10})^2 + (z(1) - {:.10})^2)",
                goal[0], goal[1], goal[2]
            )],
        ),
        (
            "wall",
            "-wwp".to_string(),
            vec![
                "2.0".into(),
                "z".into(),
                "-10,-9,2.5,4.0".into(),
                "1-7".into(),
            ],
        ),
    ]
}

fn config(dir: &Path, name: &str, text: &str, settings: &str, extra: &[String]) -> AutotsConfig {
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

fn context(name: &str, text: &str) {
    context_with(name, text, SETTINGS);
}

/// [`context`] with the settings `settings` (on Lennard-Jones layers alone, an ONIOM energy is
/// the Lennard-Jones energy of the whole system).
fn context_with(name: &str, text: &str, settings: &str) {
    let dir = scratch("atsbias", name);
    let input = read(&write(&dir, "input.xyz", text));
    for (kind, flag, mut arguments) in biases(&input) {
        let cell = format!("{name} x {kind}");
        // With ONIOM microiterations a bias on every atom leaves no environment to relax (a
        // refusal of optmain by name): the wall holds the core.
        if name == "oniom" {
            for a in &mut arguments {
                if a == "1-7" {
                    *a = "1-4".to_string();
                }
            }
        }
        let mut extra = vec![flag.clone()];
        extra.extend(arguments.iter().cloned());
        let c = config(&dir, kind, text, settings, &extra);
        let base = c.input.parent().unwrap().to_path_buf();
        let result = run_with(&c, &base, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{cell}: {e}"));
        assert!(
            !result.notes.iter().any(|n| n.contains("same surface")),
            "{cell}: {:?}",
            result.notes
        );
        assert!(!result.transition_states.is_empty(), "{cell}");
        for ts in &result.transition_states {
            let sys = read(&ts.file);
            let lj = LennardJones
                .evaluate(&sys, &EvalRequest::ENERGY_GRAD)
                .unwrap()
                .energy
                .value();
            // The file has 12 decimals of Angstrom.
            let b = bias_energy(&flag, &arguments, &sys);
            // On Lennard-Jones layers alone the ONIOM energy is that of the whole system, so
            // the same checks hold with ONIOM microiterations.
            assert!(
                (ts.energy - lj).abs() < 1e-9,
                "{cell}: {} against {lj}",
                ts.energy
            );
            assert!(
                (ts.biased_energy - ts.energy - b).abs() < 1e-9,
                "{cell}: biased {} - bare {} against the bias {b}",
                ts.biased_energy,
                ts.energy
            );
        }
        assert!(
            result
                .surface
                .iter()
                .any(|s| s.starts_with("Bias potentials")),
            "{cell}"
        );
        let page = std::fs::read_to_string(result.directory.join("autots_report.html")).unwrap();
        assert!(page.contains("Bias potentials of the surface"), "{cell}");
        assert!(
            page.contains("with the bias potentials of the surface"),
            "{cell}: both energies"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn biases_of_autots_on_an_isolated_cluster() {
    context("isolated", AR7);
}

#[test]
fn biases_of_autots_in_a_periodic_cell() {
    context("periodic", AR7_CELL);
}

#[test]
fn biases_of_autots_with_oniom_microiterations() {
    // Atoms 1-4 are the core, 5-7 the environment relaxed by the microiterations of every
    // step; the bias of the surface acts on the whole system.
    let settings = SETTINGS.replace(
        "{\"othersoft\": \"lj\",",
        "{\"othersoft\": \"lj\", \"oniom_flag\": [\"1-4\", \"none\", \"lj\"],",
    );
    context_with("oniom", AR7, &settings);
}

#[test]
fn a_step_with_another_bias_is_reported_by_name() {
    let dir = scratch("atsbias", "differ");
    let settings = SETTINGS.replace(
        "\"calc_exact_hess\": 5, \"tight_convergence_criteria\": true}",
        "\"calc_exact_hess\": 5, \"tight_convergence_criteria\": true, \"keep_pot\": [\"0.004\", \"3.9\", \"1,2\"]}",
    );
    let extra: Vec<String> = ["-kp", "0.002", "3.9", "1,2"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let c = config(&dir, "differ", AR7, &settings, &extra);
    let base = c.input.parent().unwrap().to_path_buf();
    let result = run_with(&c, &base, None, &CalculatorRegistry::native()).unwrap();
    assert!(
        result.notes.iter().any(
            |n| n.contains("Steps 2 and 3 do not run on the same surface")
                && n.contains("the bias potentials")
        ),
        "{:?}",
        result.notes
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ma_overrides_the_push_of_every_run_of_step_1_of_a_workflow() {
    let dir = scratch("atsbias", "workflow");
    let settings = r#"{
      "work_dir": "work",
      "top_n_candidates": 1,
      "step1_settings": {"othersoft": "lj", "NSTEP": 60},
      "workflow": [{"step": "step1", "repeat": 1, "repeat_settings": [
          {"param_override": {"manual_AFIR": ["1", "3", "5"]}}]}]
    }"#;
    let extra: Vec<String> = ["-ma", "5", "3", "5"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let c = config(&dir, "workflow", AR7, settings, &extra);
    assert!(
        c.notes.iter().any(|n| n.contains("-ma replaces")),
        "{:?}",
        c.notes
    );
    let run = &c.runs(0)[0];
    assert_eq!(
        run.get("manual_AFIR").and_then(|v| v.as_string_vec()),
        Some(vec!["5".to_string(), "3".to_string(), "5".to_string()])
    );
    let _ = std::fs::remove_dir_all(&dir);
}
