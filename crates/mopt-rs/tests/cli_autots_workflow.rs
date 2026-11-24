//! `run_autots` with a `workflow` block (several runs of a step, MultiOptPy's AutoTS v2), with
//! a search between two structures as step 1, and with steps 3 and 4 on different surfaces:
//! HCN to HNC with the built-in GFN0-xTB (a growing string, then two relaxations of the path
//! with different settings, then the transition state), and two AFIR runs one after the
//! other on Cl- + CH3Br with the built-in GFN1-xTB.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::autots::{run_with, AutotsConfig, FirstPathMethod};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::domain::Units;
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

const KCAL: f64 = Units::HARTREE_TO_KCAL_MOL;
/// The saddle of HCN to HNC above HCN with GFN0-xTB (kcal/mol).
const BARRIER_KCAL: f64 = 68.285;

/// HCN and HNC, the reactant and the product.
const HCN_AND_HNC: &str =
    "3\nHCN\nC -0.025197409828 0.0 0.0\nN 1.115411466511 0.0 0.0\nH -1.090214056682 0.0 0.0\n\
3\nHNC\nC -1.099032916795 0.0 0.0\nN 0.054338186952 0.0 0.0\nH 1.044694729843 0.0 0.0\n";

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mopt_autots_wf_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn config(d: &Path, input: &str, structure: &str, json: &str, more: &[&str]) -> AutotsConfig {
    let cfg = d.join("config.json");
    std::fs::write(&cfg, json).unwrap();
    let file = d.join(input);
    std::fs::write(&file, structure).unwrap();
    let mut all: Vec<String> = vec![
        file.to_string_lossy().into_owned(),
        "-cfg".to_string(),
        cfg.to_string_lossy().into_owned(),
    ];
    all.extend(more.iter().map(|s| (*s).to_string()));
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    AutotsConfig::from_args(&parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap()).unwrap()
}

fn energy(path: &Path, calculator: &str) -> f64 {
    let sys = read_xyz_file(path, SpinCharge::default(), None).unwrap();
    CalculatorRegistry::native()
        .create(calculator, &CalculatorOptions::default(), &sys)
        .unwrap()
        .evaluate(&sys, &EvalRequest::ENERGY_ONLY)
        .unwrap()
        .energy
        .value()
}

fn frames(path: &Path) -> usize {
    read_xyz_trajectory_file(path, SpinCharge::default(), None)
        .unwrap()
        .frames
        .len()
}

#[test]
fn a_string_as_step_1_and_two_relaxations_of_the_path_one_after_the_other() {
    let d = scratch("hcn");
    let c = config(
        &d,
        "hcn.xyz",
        HCN_AND_HNC,
        r#"{
          "work_dir": "work",
          "top_n_candidates": 1,
          "step1_method": "GSM",
          "step1_settings": {"othersoft": "gfn0xtb", "ieip_options": ["figures=off"]},
          "step2_settings": {"othersoft": "gfn0xtb", "NSTEP": 40, "neb_options": ["figures=off"]},
          "step3_settings": {"othersoft": "gfn0xtb", "calc_exact_hess": 5,
                             "tight_convergence_criteria": true},
          "workflow": [
            {"step": "step1"},
            {"step": "step2", "repeat": 2, "repeat_settings": [
                {"param_override": {"NSTEP": 4, "use_model_hessian": "none"}},
                {"param_override": {"QSM": true}}]},
            {"step": "step3"}
          ]
        }"#,
        &[],
    );
    assert!(matches!(c.first_method, FirstPathMethod::Pair(m) if m.name == "gsm"));
    assert_eq!(c.runs(1).len(), 2);
    let result = run_with(&c, &d, None, &CalculatorRegistry::native()).unwrap();
    let work = &result.directory;

    // Step 1: the points of the string from HCN to HNC are the first path.
    assert!(result.first_path_method.contains("growing string method"));
    let first = work.join("hcn_step1_traj.xyz");
    assert_eq!(result.afir_path.as_deref(), Some(first.as_path()));
    assert!(frames(&first) >= 5, "{}", frames(&first));
    assert!(work.join("hcn_step1_gsm").join("reactant.xyz").is_file());

    // Both initial bands survive the second relaxation and the subsequent TS search.
    let profiles = &result.band_profiles;
    for run in [1, 2] {
        let saved: Vec<_> = profiles.iter().filter(|p| p.run == run).collect();
        assert_eq!(saved[0].iteration, 0);
        assert!(saved.len() > 1);
        assert!(saved.iter().all(|p| p.energies.len() >= 2));
    }
    let html = std::fs::read_to_string(work.join("autots_report.html")).unwrap();
    assert_eq!(html.matches("<div data-band").count(), profiles.len());
    assert!(html.contains("Run 1 · Iteration 0"));
    assert!(html.contains("Run 2 · Iteration 0"));
    assert!(html.contains("first image of the initial path"));
    assert!(html.contains("<h3>Evaluations</h3>") && html.contains("from the lowest point"));
    assert!(html.contains("<div class=\"controls\"><label>Saved band"));
    assert!(
        html.find("Energy profile of the path").unwrap()
            < html.find("<h3>Path relaxation").unwrap()
    );
    assert!(!html.contains("class=\"band-figures\""), "figures=off");

    // Step 2: the second run relaxes the path that the first run left.
    let read = |name: &str| std::fs::read(work.join(name)).unwrap();
    assert_eq!(
        read("hcn_step2_run1_init_path.xyz"),
        read("hcn_step1_traj.xyz")
    );
    assert_eq!(
        read("hcn_step2_run2_init_path.xyz"),
        read("hcn_step2_run1_final_path.xyz")
    );
    assert_ne!(
        read("hcn_step2_run2_final_path.xyz"),
        read("hcn_step2_run1_final_path.xyz")
    );
    let log = std::fs::read_to_string(work.join("autots.log")).unwrap();
    assert!(
        log.contains("AutoTS step 2 (run 1 of 2): the path (nebmain)."),
        "{log}"
    );
    assert!(
        log.contains("AutoTS step 2 (run 2 of 2): the path (nebmain)."),
        "{log}"
    );
    // Each run with its own settings: four iterations along the forces alone, then the
    // chain with the directions atom by atom until it rests or its iterations end.
    let settings_of = |run: usize| -> String {
        // The run directory of nebmain is next to its input, or in a directory of the day.
        let entries = |dir: &Path| -> Vec<PathBuf> {
            std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .collect()
        };
        let mut all = entries(work);
        for inner in all.clone() {
            all.extend(entries(&inner));
        }
        let wanted = format!("run{run}_init_path_NEB");
        let mut found: Vec<PathBuf> = all
            .into_iter()
            .filter(|p| {
                p.is_dir()
                    && p.file_name()
                        .is_some_and(|n| n.to_string_lossy().contains(&wanted))
            })
            .collect();
        assert_eq!(found.len(), 1, "{found:?}");
        std::fs::read_to_string(found.remove(0).join("settings.json")).unwrap()
    };
    let (one, two) = (settings_of(1), settings_of(2));
    assert!(one.contains("\"NSTEP\": \"4\""), "{one}");
    assert!(two.contains("\"NSTEP\": \"40\""), "{two}");
    assert!(one.contains("\"use_model_hessian\""), "{one}");
    assert!(two.contains("\"QSM\""), "{two}");
    assert!(!two.contains("\"use_model_hessian\""), "{two}");
    let stage = result.path_stage.as_ref().unwrap();
    assert!(stage.method.ends_with("(run 2 of 2)"), "{}", stage.method);
    // The candidates are those of the last run.
    assert_eq!(result.candidates.len(), 1);
    assert!(work
        .join("hcn_step2_run2_candidates")
        .join("hcn_s2_run2_guess_1.xyz")
        .is_file());

    // Step 3: the transition state of HCN to HNC, under the name of the workflow.
    assert_eq!(result.transition_states.len(), 1);
    let ts = &result.transition_states[0];
    assert!(ts.converged);
    assert_eq!(ts.imaginary_modes, Some(1));
    assert!(
        ts.file.ends_with("hcn_s3_run1_ts_final_1.xyz"),
        "{:?}",
        ts.file
    );
    let barrier =
        KCAL * (ts.energy - energy(&work.join("hcn_step1_gsm").join("reactant.xyz"), "gfn0xtb"));
    assert!((barrier - BARRIER_KCAL).abs() < 0.3, "{barrier}");
    // No step 4 in the workflow.
    assert!(ts.ends.is_empty() && result.notes.is_empty());
}

#[test]
fn two_runs_of_step_1_follow_each_other_and_only_listed_steps_run() {
    let d = scratch("sn2");
    // CH3Br with Cl- behind the carbon; the charge is in the file.
    let c = config(
        &d,
        "sn2.xyz",
        "6\n-1 1\nC 0.000 0.000 0.000\nH -0.360 1.010 0.050\nH -0.360 -0.480 0.880\n\
         H -0.360 -0.530 -0.900\nBr 1.960 0.020 -0.010\nCl -3.300 0.080 0.060\n",
        r#"{
          "work_dir": "work",
          "step1_settings": {"othersoft": "gfn1xtb", "NSTEP": 60},
          "step2_settings": {"othersoft": "gfn1xtb"},
          "workflow": [
            {"step": "step1", "repeat": 2, "repeat_settings": [
                {"param_override": {"manual_AFIR": ["40", "1", "6"]}},
                {"param_override": {"manual_AFIR": ["150", "1", "6"]}}]},
            {"step": "step2", "enabled": false}
          ]
        }"#,
        &[],
    );
    // The state of the input file is in the settings of every run.
    for run in c.runs(0) {
        assert_eq!(run["electronic_charge"].as_f64(), Some(-1.0));
    }
    assert!(c.runs(1).is_empty());
    let result = run_with(&c, &d, None, &CalculatorRegistry::native()).unwrap();
    let work = &result.directory;
    let (one, two) = (
        work.join("sn2_step1_run1_traj.xyz"),
        work.join("sn2_step1_run2_traj.xyz"),
    );
    // The second run starts where the first ended, and the first path is the two
    // trajectories with that structure once.
    let last = read_xyz_file(
        work.join("sn2_step1_run1_final.xyz"),
        SpinCharge::default(),
        None,
    )
    .unwrap()
    .coords_flat_bohr();
    let second = read_xyz_trajectory_file(&two, SpinCharge::default(), None).unwrap();
    let start = second.frames[0].system.coords_flat_bohr();
    assert!((0..last.nrows()).all(|i| (last[i] - start[i]).abs() < 1e-6));
    let joined = work.join("sn2_step1_combined_traj.xyz");
    assert_eq!(result.afir_path.as_deref(), Some(joined.as_path()));
    let together = frames(&one) + frames(&two);
    assert!(
        frames(&joined) == together || frames(&joined) == together - 1,
        "{} of {together}",
        frames(&joined)
    );
    assert!(frames(&two) > 1);
    // The stronger push of the second run brings the chloride nearer to the carbon.
    let distance = |x: &faer::Col<f64>| {
        ((x[0] - x[15]).powi(2) + (x[1] - x[16]).powi(2) + (x[2] - x[17]).powi(2)).sqrt()
    };
    let end = second.frames[second.frames.len() - 1]
        .system
        .coords_flat_bohr();
    assert!(
        distance(&end) < distance(&start),
        "{} {}",
        distance(&end),
        distance(&start)
    );
    // Step 2 is switched off: the run ends with the first path.
    assert!(result.path_stage.is_none() && result.transition_states.is_empty());
    let log = std::fs::read_to_string(work.join("autots.log")).unwrap();
    assert!(log.contains("AutoTS step 1 (run 2 of 2)"), "{log}");
    assert!(log.contains("step 2 is not in the workflow"), "{log}");
}

#[test]
fn steps_3_and_4_on_different_surfaces_are_warned_about() {
    let d = scratch("levels");
    let settings = |fourth: &str| {
        format!(
            r#"{{
              "work_dir": "work",
              "step1_settings": {{"othersoft": "gfn0xtb"}},
              "step2_settings": {{"othersoft": "gfn0xtb"}},
              "step3_settings": {{"othersoft": "gfn0xtb", "calc_exact_hess": 5}},
              "step4_settings": {{{fourth}, "calc_exact_hess": 10,
                                 "intrinsic_reaction_coordinates": ["0.3", "50", "lqa"],
                                 "step4b_opt_method": ["rsirfo_block_fsb"]}}
            }}"#
        )
    };
    // Step 1 has no bias potential, so the run stops there; what it warns about before
    // it starts is in the log.
    let run = |fourth: &str, more: &[&str]| -> String {
        let work = d.join("work");
        let _ = std::fs::remove_dir_all(&work);
        let c = config(&d, "hcn.xyz", HCN_AND_HNC, &settings(fourth), more);
        assert!(run_with(&c, &d, None, &CalculatorRegistry::native()).is_err());
        std::fs::read_to_string(work.join("autots.log")).unwrap()
    };
    let other = run(
        r#""othersoft": "gfn1xtb", "electronic_charge": 1, "spin_multiplicity": 2"#,
        &["--run_step4"],
    );
    assert!(
        other.contains("Steps 3 and 4 do not run on the same surface"),
        "{other}"
    );
    assert!(
        other.contains("the calculator (gfn0xtb against gfn1xtb)"),
        "{other}"
    );
    assert!(other.contains("charge and multiplicity"), "{other}");
    // The same surface with other settings of the optimization, and no step 4: no warning.
    let same = run(r#""othersoft": "gfn0xtb""#, &["--run_step4"]);
    assert!(!same.contains("do not run on the same surface"), "{same}");
    let unused = run(r#""othersoft": "gfn1xtb""#, &[]);
    assert!(
        !unused.contains("do not run on the same surface"),
        "{unused}"
    );
}

#[test]
fn every_search_between_two_structures_gives_a_first_path() {
    let d = scratch("pairs");
    for method in [
        "ieip", "2pshs", "gnt", "gsm", "fsm", "ds-afir", "dimer", "spm",
    ] {
        let _ = std::fs::remove_dir_all(d.join("work"));
        let c = config(
            &d,
            "hcn.xyz",
            HCN_AND_HNC,
            &format!(
                r#"{{
                  "work_dir": "work",
                  "step1_method": "{method}",
                  "step1_settings": {{"othersoft": "gfn0xtb", "ieip_options": ["figures=off"]}},
                  "step2_settings": {{"othersoft":"gfn0xtb", "NSTEP":1, "node_distance":0, "neb_options":["figures=off", "path=linear"]}},
                  "workflow": [{{"step":"step1"}}, {{"step":"step2"}}]
                }}"#
            ),
            &[],
        );
        let result = run_with(&c, &d, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{method}: {e}"));
        // The points of the search, from HCN to HNC.
        let path = result.afir_path.clone().unwrap();
        let read = read_xyz_trajectory_file(&path, SpinCharge::default(), None).unwrap();
        assert!(
            read.frames.len() >= 3,
            "{method}: {} points",
            read.frames.len()
        );
        let reference =
            read_xyz_trajectory_file(d.join("hcn.xyz"), SpinCharge::default(), None).unwrap();
        // Interatomic distances do not depend on how a method laid the structures.
        let distances = |sys: &mopt_rs::domain::MolecularSystem| -> [f64; 3] {
            let x = sys.coords_flat_bohr();
            let d = |i: usize, j: usize| {
                (0..3)
                    .map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            };
            [d(0, 1), d(0, 2), d(1, 2)]
        };
        let same = |a: [f64; 3], b: [f64; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-3);
        assert!(
            same(
                distances(&read.frames[0].system),
                distances(&reference.frames[0].system)
            ),
            "{method}: the path starts at the reactant"
        );
        assert!(
            same(
                distances(&read.frames[read.frames.len() - 1].system),
                distances(&reference.frames[1].system)
            ),
            "{method}: the path ends at the product"
        );
        // At iteration zero NEB has evaluated the handed-off band but has not moved
        // it. Check every image, including the joint and the reversed product side.
        let band_name = "hcn_step2_run1_init_path";
        let band_dir = std::fs::read_dir(&result.directory)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.is_dir() && p.file_name().unwrap().to_string_lossy().contains("_NEB_"))
            .expect("NEB run directory");
        let initial = band_dir
            .join(format!("path_ITR_0_{band_name}"))
            .join(format!("{band_name}_path.xyz"));
        let evaluated = read_xyz_trajectory_file(initial, SpinCharge::default(), None).unwrap();
        assert_eq!(evaluated.frames.len(), read.frames.len(), "{method}");
        for (i, (source, used)) in read.frames.iter().zip(&evaluated.frames).enumerate() {
            assert!(
                same(distances(&source.system), distances(&used.system)),
                "{method}: initial band image {i}"
            );
        }
        println!("PATH {method}: {} points", read.frames.len());
    }
}

#[test]
fn neb_evaluates_the_complete_step_1_path_on_its_first_iteration() {
    use mopt_rs::domain::{AtomId, MolecularSystem};

    let same_geometry = |a: &MolecularSystem, b: &MolecularSystem| {
        assert_eq!(a.elements, b.elements);
        for i in 0..a.n_atoms() {
            for j in 0..i {
                let bond = |s: &MolecularSystem| s.distance(AtomId::new(i), AtomId::new(j)).value();
                assert!((bond(a) - bond(b)).abs() < 1e-6, "atoms {i} and {j}");
            }
        }
    };
    // These two AutoTS methods go through optmain. The other pair methods go through
    // ieipmain and are covered by every_search_between_two_structures_gives_a_first_path.
    for (method, input, options) in [
        (
            "afir",
            "2\n0 1\nH 0 0 0\nH 1.5 0 0\n",
            r#""manual_AFIR": ["40", "1", "2"], "NSTEP": 12"#,
        ),
        ("bitss", HCN_AND_HNC, r#""NSTEP": 6"#),
    ] {
        let d = scratch(&format!("handoff_{method}"));
        let json = format!(
            r#"{{"work_dir": "work", "step1_method": "{method}",
                 "step1_settings": {{"othersoft": "gfn0xtb", {options}}},
                 "step2_settings": {{"othersoft": "gfn0xtb", "node_distance": 0,
                                    "NSTEP": 1, "neb_options": ["figures=off"]}},
                 "workflow": [{{"step": "step1"}}, {{"step": "step2"}}]}}"#
        );
        let c = config(&d, "input.xyz", input, &json, &[]);
        let run = run_with(&c, &d, None, &CalculatorRegistry::native()).unwrap();
        let read =
            |file: &Path| read_xyz_trajectory_file(file, SpinCharge::default(), None).unwrap();
        let first = read(run.afir_path.as_ref().unwrap());
        let input = read(&d.join("input.xyz"));
        let end = if method == "afir" {
            read_xyz_file(
                run.afir_directory.unwrap().join("input_optimized.xyz"),
                SpinCharge::default(),
                None,
            )
            .unwrap()
        } else {
            input.frames.last().unwrap().system.clone()
        };
        assert!(first.frames.len() > 2, "{method}");
        same_geometry(&first.frames[0].system, &input.frames[0].system);
        same_geometry(&first.frames.last().unwrap().system, &end);

        // Check the structures that nebmain actually evaluated, after parsing and alignment.
        // Turning off node redistribution lets every interior frame be checked in order.
        let band = run.path_stage.unwrap().path.unwrap();
        let band_name = band
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_suffix("_path")
            .unwrap();
        let initial = band
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join(format!("path_ITR_0_{band_name}"));
        let evaluated = read(&initial.join(band.file_name().unwrap()));
        assert_eq!(first.frames.len(), evaluated.frames.len(), "{method}");
        for (expected, actual) in first.frames.iter().zip(&evaluated.frames) {
            same_geometry(&expected.system, &actual.system);
        }
    }
}

#[test]
fn a_search_that_finds_nothing_is_reported_as_such() {
    // The local minimax method on the straight line of HCN and HNC: its first ray has no
    // maximum, and it ends without an estimate. Step 1 says that there is no path.
    let d = scratch("nothing");
    let c = config(
        &d,
        "hcn.xyz",
        HCN_AND_HNC,
        r#"{
          "work_dir": "work",
          "step1_method": "lmm",
          "step1_settings": {"othersoft": "gfn0xtb", "ieip_options": ["figures=off"]},
          "step2_settings": {"othersoft": "gfn0xtb"}
        }"#,
        &[],
    );
    let e = run_with(&c, &d, None, &CalculatorRegistry::native())
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("left no path between the two structures"), "{e}");
    assert!(e.contains("hcn_step1_lmm"), "{e}");
    // A name that is no search is refused when the settings are read.
    let cfg = d.join("other.json");
    std::fs::write(&cfg, r#"{"step1_method": "neb"}"#).unwrap();
    let input = d.join("hcn.xyz");
    let all = [
        input.to_string_lossy().into_owned(),
        "-cfg".to_string(),
        cfg.to_string_lossy().into_owned(),
    ];
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    let e = AutotsConfig::from_args(&parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("ieip, 2pshs, gnt, gsm, fsm, dsafir"), "{e}");
}
