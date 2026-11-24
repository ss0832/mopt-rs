//! `ieipmain` through the command-line layer: every method finds the saddle of HCN to HNC
//! with the built-in GFN0-xTB (confirmed by a search of order 1 from its estimate), the files
//! of MultiOptPy's iEIP on a Lennard-Jones pair, plain variables on the Muller-Brown surface,
//! restart from the journal, `-mf`, settings files, and what is refused.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::ieipmain::{config_from_cli, run_with, IeipmainResult};
use mopt_rs::cli::{optmain, run_command, EXIT_OK, EXIT_USAGE};
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::Units;
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::counting::EvaluationCounts;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

const HCN: &str =
    "3\nHCN\nC -0.025197409828 0.0 0.0\nN 1.115411466511 0.0 0.0\nH -1.090214056682 0.0 0.0\n";
const HNC: &str =
    "3\nHNC\nC -1.099032916795 0.0 0.0\nN 0.054338186952 0.0 0.0\nH 1.044694729843 0.0 0.0\n";
/// The saddle of HCN to HNC above HCN with GFN0-xTB (kcal/mol).
const BARRIER_KCAL: f64 = 68.285;

fn scratch(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("mopt_ieipmain_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn pair(base: &Path) -> String {
    let dir = base.join("hcn_pair");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("hcn_A.xyz"), HCN).unwrap();
    std::fs::write(dir.join("hnc_B.xyz"), HNC).unwrap();
    dir.to_string_lossy().into_owned()
}

fn run(base: &Path, args: &[&str]) -> IeipmainResult {
    run_counted(base, args).0
}

fn run_counted(base: &Path, args: &[&str]) -> (IeipmainResult, EvaluationCounts) {
    let config = config_from_cli(args).unwrap();
    let registry = CalculatorRegistry::native();
    let result = run_with(&config, base, None, &registry).unwrap();
    (result, registry.counts())
}

fn energy(path: &Path) -> f64 {
    let sys = read_xyz_file(path, SpinCharge::default(), None).unwrap();
    let calc = CalculatorRegistry::native()
        .create("gfn0xtb", &CalculatorOptions::default(), &sys)
        .unwrap();
    calc.evaluate(&sys, &EvalRequest::ENERGY_ONLY)
        .unwrap()
        .energy
        .value()
}

/// The barrier (kcal/mol above HCN) of the saddle a search of order 1 finds from `guess`.
fn refined_barrier(base: &Path, guess: &Path, reference: f64) -> (f64, usize) {
    let g = guess.to_string_lossy().into_owned();
    let config = optmain::OptmainConfig::from_args(
        &mopt_rs::cli::option::parser("optmain", mopt_rs::cli::spec::OPTMAIN)
            .parse([
                g.as_str(),
                "-os",
                "gfn0xtb",
                "-order",
                "1",
                "-fc",
                "1",
                "-ns",
                "60",
            ])
            .unwrap(),
    )
    .unwrap();
    let out = optmain::run_with(&config, base, None, &CalculatorRegistry::native()).unwrap();
    let job = out[0].summary.clone().unwrap();
    assert!(out[0].converged, "{}", guess.display());
    (
        (job.energy.value() - reference) * Units::HARTREE_TO_KCAL_MOL,
        job.iterations,
    )
}

#[test]
fn every_method_finds_the_saddle_of_hcn_to_hnc() {
    let base = scratch("hcn");
    let input = pair(&base);
    let single = base.join("hcn.xyz");
    std::fs::write(&single, HCN).unwrap();
    let reference = energy(&single);
    let single = single.to_string_lossy().into_owned();
    let off = ["-ieip_opt", "figures=off"];
    let cases: Vec<(&str, Vec<&str>)> = vec![
        ("iEIP", vec![input.as_str(), "-os", "gfn0xtb"]),
        (
            "dimer",
            vec![input.as_str(), "-os", "gfn0xtb", "-use_dimer"],
        ),
        ("2PSHS", vec![input.as_str(), "-os", "gfn0xtb", "-2pshs"]),
        ("GNT", vec![input.as_str(), "-os", "gfn0xtb", "-gnt"]),
        ("ADDF", vec![single.as_str(), "-os", "gfn0xtb", "-addf"]),
        ("SPM", vec![input.as_str(), "-os", "gfn0xtb", "-use_spm"]),
        ("GSM", vec![input.as_str(), "-os", "gfn0xtb", "-gsm"]),
        ("FSM", vec![input.as_str(), "-os", "gfn0xtb", "-fsm"]),
        ("DS-AFIR", vec![input.as_str(), "-os", "gfn0xtb", "-dsafir"]),
        // Other coordinates: iEIP takes its steps through them, the others move in them.
        (
            "iEIP -coord dic",
            vec![input.as_str(), "-os", "gfn0xtb", "-coord", "dic"],
        ),
        (
            "iEIP -coord tric-cart",
            vec![input.as_str(), "-os", "gfn0xtb", "-coord", "tric-cart"],
        ),
        (
            "dimer -coord dic",
            vec![
                input.as_str(),
                "-os",
                "gfn0xtb",
                "-use_dimer",
                "-coord",
                "dic",
            ],
        ),
        (
            "GSM -coord tric-cart",
            vec![
                input.as_str(),
                "-os",
                "gfn0xtb",
                "-gsm",
                "-coord",
                "tric-cart",
            ],
        ),
        (
            "2PSHS -coord tric-cart",
            vec![
                input.as_str(),
                "-os",
                "gfn0xtb",
                "-2pshs",
                "-coord",
                "tric-cart",
            ],
        ),
    ];
    for (name, mut args) in cases {
        args.extend_from_slice(&off);
        let (result, counts) = run_counted(&base, &args);
        assert!(result.converged, "{name}");
        assert!(!result.transition_states.is_empty(), "{name}");
        let found: Vec<(f64, usize)> = result
            .transition_states
            .iter()
            .map(|g| refined_barrier(&base, g, reference))
            .collect();
        eprintln!(
            "{name}: {} gradients, {} Hessians, {} energies; estimates refined to (barrier, \
             iterations) {found:?}",
            counts.gradients, counts.hessians, counts.energies
        );
        assert!(
            found.iter().any(|(b, _)| (b - BARRIER_KCAL).abs() < 0.01),
            "{name}: {found:?}"
        );
    }
}

fn write_ar7(dir: &Path) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/ieip_lj_ar7_default.json");
    let case = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let inputs = case.get("input_angstrom").unwrap().as_array().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    for (letter, frame) in ["A", "B"].iter().zip(inputs) {
        let mut text = String::from("7\n0 1\n");
        for atom in frame.as_array().unwrap() {
            let p: Vec<f64> = atom
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            text.push_str(&format!("Ar {:.12} {:.12} {:.12}\n", p[0], p[1], p[2]));
        }
        std::fs::write(dir.join(format!("ar7_{letter}.xyz")), text).unwrap();
    }
}

#[test]
fn ieip_writes_the_files_of_multioptpy() {
    let base = scratch("ar7");
    let dir = base.join("ar7");
    write_ar7(&dir);
    let d = dir.to_string_lossy().into_owned();
    let result = run(&base, &[&d, "-os", "lj", "-ns", "60"]);
    assert!(result.converged);
    let run_dir = &result.directory;
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/ieip_lj_ar7_default.json");
    let case = JsonValue::parse(&std::fs::read_to_string(golden).unwrap()).unwrap();
    // MultiOptPy's files (but the arrays of its bias potentials, raw_bias_*.npy).
    for name in case.get("files").unwrap().as_array().unwrap() {
        let name = name.as_str().unwrap();
        if name.ends_with(".npy") {
            continue;
        }
        assert!(run_dir.join(name).is_file(), "{name}");
    }
    let numbers = |name: &str| -> Vec<f64> {
        std::fs::read_to_string(run_dir.join(name))
            .unwrap()
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect()
    };
    let as_f64 = |key: &str| -> Vec<f64> {
        case.get(key)
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    assert_eq!(numbers("approx_TS.txt"), as_f64("approx_ts"));
    assert_eq!(numbers("approx_EQ.txt"), as_f64("approx_eq"));
    assert_eq!(numbers("local_min_grad.txt"), as_f64("local_min_grad"));
    let frames = std::fs::read_to_string(run_dir.join("ar7_A_traj.xyz")).unwrap();
    let count = frames.lines().filter(|l| l.starts_with("Frame ")).count();
    assert_eq!(
        count,
        case.get("trajectory_angstrom")
            .unwrap()
            .as_array()
            .unwrap()
            .len()
    );
    // The images had a trajectory each while the run went on; at its end they are the one
    // trajectory, and there is no directory per iteration.
    for entry in std::fs::read_dir(run_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(!entry.path().is_dir(), "{name}");
        assert!(!name.contains("_image_"), "{name}");
    }
    // Both images of every iteration are shown in Cartesian coordinates.
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    let iterations = log.matches("# ITR. ").count();
    assert!(iterations > 10, "{iterations}");
    assert_eq!(log.matches("\nImage 1\nAr ").count(), iterations);
    assert_eq!(log.matches("\nImage 2\nAr ").count(), iterations);
    for name in [
        "ieip_energies.png",
        "ieip_energies.svg",
        "ieip_energies.csv",
        "mopt.log",
        "settings.json",
        "ar7_A_ts_guess.xyz",
    ] {
        assert!(run_dir.join(name).is_file(), "{name}");
    }
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(log.contains("[[opt information]]") && log.contains("Converged!!!"));
    assert!(log.contains("Evaluations (this input):"));

    // Again from the journal of the run: nothing is calculated anew.
    let mut config = config_from_cli(&[&d, "-os", "lj", "-ns", "60"]).unwrap();
    config.restart_dir = Some(run_dir.clone());
    let again = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    assert!(again.converged);
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(log.contains("Restart: 57 evaluations taken from"), "{log}");

    // With the relaxation on hyperspheres.
    let micro = run(
        &base,
        &[
            &d,
            "-os",
            "lj",
            "-ns",
            "100",
            "-mi",
            "3",
            "-ieip_opt",
            "figures=off",
        ],
    );
    assert!(micro.converged);
}

fn variables(base: &Path, name: &str, p: [f64; 2]) -> String {
    let path = base.join(name);
    std::fs::write(&path, format!("x {}\ny {}\n", p[0], p[1])).unwrap();
    path.to_string_lossy().into_owned()
}

fn values(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
        .collect()
}

#[test]
fn plain_variables_on_the_muller_brown_surface() {
    let base = scratch("mb");
    let a = variables(&base, "b_min.txt", [0.623499, 0.028038]);
    let b = variables(&base, "c_min.txt", [-0.050011, 0.466694]);
    let saddle = [0.212487, 0.292988];
    // iEIP ends when the images are closer than 0.15: its midpoint is that close.
    for (extra, tolerance) in [
        (vec![], 0.15),
        (vec!["-2pshs", "-2pshs_step", "0.01"], 0.1),
        (vec!["-use_dimer", "-ieip_opt", "step=0.02"], 0.1),
        (vec!["-gnt", "-gnt_step", "0.02"], 0.1),
    ] {
        let mut args = vec![
            a.as_str(),
            b.as_str(),
            "-os",
            "mb2d",
            "-ieip_opt",
            "figures=off",
        ];
        args.extend(extra.iter().copied());
        let result = run(&base, &args);
        assert!(result.converged, "{extra:?}");
        let x = values(&result.transition_states[0]);
        let d = ((x[0] - saddle[0]).powi(2) + (x[1] - saddle[1]).powi(2)).sqrt();
        assert!(d < tolerance, "{extra:?}: {x:?}");
    }
}

#[test]
fn model_functions_run_optmain_at_the_first_structure() {
    let base = scratch("mf");
    let dir = base.join("ar7");
    write_ar7(&dir);
    let d = dir.to_string_lossy().into_owned();
    let result = run(&base, &[&d, "-os", "lj", "-mf", "seam", "-ns", "300"]);
    assert!(result.directory.to_string_lossy().contains("_OPT_"));
    assert!(result.converged);
}

#[test]
fn settings_and_refusals() {
    let base = scratch("refuse");
    let input = pair(&base);
    // -sqm1 and -sqm2 are not part of mopt-rs.
    assert_eq!(
        run_command(
            "ieipmain",
            &[input.clone(), "-os".into(), "lj".into(), "-sqm1".into()]
        ),
        EXIT_USAGE
    );
    assert_eq!(run_command("ieipmain", &["-h".to_string()]), EXIT_OK);
    // iEIP needs two structures; ADDF takes one.
    let single = base.join("one.xyz");
    std::fs::write(&single, HCN).unwrap();
    assert!(config_from_cli(&[single.to_str().unwrap(), "-os", "lj"]).is_err());
    assert!(config_from_cli(&[single.to_str().unwrap(), "-os", "lj", "-addf"]).is_ok());
    // The structures of a directory in the order of their letters, not of their names.
    let swapped = base.join("swapped");
    std::fs::create_dir_all(&swapped).unwrap();
    std::fs::write(swapped.join("hnc_A.xyz"), HNC).unwrap();
    std::fs::write(swapped.join("hcn_B.xyz"), HCN).unwrap();
    let config = config_from_cli(&[swapped.to_str().unwrap(), "-os", "lj"]).unwrap();
    let names: Vec<String> = config
        .inputs
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["hnc_A.xyz", "hcn_B.xyz"]);
    // -coord is refused where a method has no formulation in the coordinates (exit code 2).
    for extra in [
        vec!["-gsm", "-coord", "ric"],
        vec!["-coord", "dic", "-fix", "1"],
        vec!["-se_gsm", "ADD 1 2", "-coord", "dic"],
    ] {
        let mut args = vec![input.clone(), "-os".to_string(), "lj".to_string()];
        args.extend(extra.iter().map(|s| s.to_string()));
        assert_eq!(run_command("ieipmain", &args), EXIT_USAGE, "{extra:?}");
    }
    // The relaxation of the images needs structures.
    let a = variables(&base, "a.txt", [0.6, 0.0]);
    let b = variables(&base, "b.txt", [-0.05, 0.47]);
    let config = config_from_cli(&[&a, &b, "-os", "mb2d", "-mi", "2"]).unwrap();
    assert!(run_with(&config, &base, None, &CalculatorRegistry::native()).is_err());
    // A settings file of the command.
    let dumped = base.join("ieip.json");
    let code = run_command(
        "ieipmain",
        &[
            input,
            "-os".into(),
            "lj".into(),
            "-beta".into(),
            "2.0".into(),
            "--dump-config".into(),
            dumped.to_string_lossy().into_owned(),
        ],
    );
    assert_eq!(code, EXIT_OK);
    let text = std::fs::read_to_string(&dumped).unwrap();
    assert!(text.contains("\"BETA\"") && text.contains("2.0"), "{text}");
}

/// Ozone in the first and the second excited singlet state of the simplified TDA on
/// GFN0-xTB orbitals, as a directory of two structures: the first is the minimum of the
/// first state, the second comes from a few steps of a minimization in the second state.
fn ozone_in_two_states(base: &Path) -> String {
    let start = base.join("o3.xyz");
    std::fs::write(
        &start,
        "3\n\nO 0.0 0.0 0.0\nO 1.1 0.3 0.0\nO -1.0 0.5 0.1\n",
    )
    .unwrap();
    let optimized = |state: &str, steps: &str| -> PathBuf {
        let s = start.to_string_lossy().into_owned();
        let config = optmain::OptmainConfig::from_args(
            &mopt_rs::cli::option::parser("optmain", mopt_rs::cli::spec::OPTMAIN)
                .parse([s.as_str(), "-os", "gfn0xtb", "-es", state, "-ns", steps])
                .unwrap(),
        )
        .unwrap();
        let out = optmain::run_with(&config, base, None, &CalculatorRegistry::native()).unwrap();
        out[0].optimized.clone()
    };
    let dir = base.join("pair");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(optimized("1", "100"), dir.join("o3_A.xyz")).unwrap();
    std::fs::copy(optimized("2", "30"), dir.join("o3_B.xyz")).unwrap();
    dir.to_string_lossy().into_owned()
}

/// How far apart (Hartree) the first and the second excited state are at the structure.
fn gap_of_excited_states(structure: &Path) -> f64 {
    let sys = read_xyz_file(structure, SpinCharge::default(), None).unwrap();
    let calc = CalculatorRegistry::native()
        .create("gfn0xtb", &CalculatorOptions::default(), &sys)
        .unwrap();
    let state = |k: usize| {
        let request = EvalRequest {
            excited_state: k,
            ..EvalRequest::ENERGY_ONLY
        };
        calc.evaluate(&sys, &request).unwrap().energy.value()
    };
    (state(2) - state(1)).abs()
}

#[test]
fn dsafir_between_two_states_estimates_their_crossing_and_goes_on_to_it() {
    let base = scratch("two_states");
    let input = ozone_in_two_states(&base);

    let result = run(
        &base,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-dsafir",
            "-es",
            "1",
            "2",
            "-ieip_opt",
            "figures=off",
            "optimize=on",
        ],
    );
    assert!(result.converged);
    // Each end moved on the surface of its own state; the mean of the ends where they met
    // is the estimate of a crossing, not of a transition state.
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert!(log.contains("each end of DS-AFIR moves on the surface of its own state"));
    assert_eq!(result.transition_states.len(), 1);
    let name = result.transition_states[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(name, "o3_A_crossing_guess.xyz");
    assert!(log.contains("-es 1 -mf opt_meci 0 1 2"), "{log}");

    // The search for the conical intersection from the estimate: the two states meet there.
    assert_eq!(result.optimized.len(), 1);
    let found = &result.optimized[0];
    assert!(found.crossing && found.converged);
    let gap = gap_of_excited_states(&found.optimized);
    assert!(gap < 5.0e-4, "gap {gap}");
    assert!(found.directory.join("seam.txt").is_file());

    // From the crossing down both surfaces (meta-IRC): the path in the first state ends at
    // the minimum the first end started from.
    assert_eq!(found.descents.len(), 2);
    assert!(found.descents[0].1 < 0.05, "{:?}", found.descents);
    let table = std::fs::read_to_string(result.directory.join("optimized_estimates.csv")).unwrap();
    assert!(table
        .lines()
        .nth(1)
        .unwrap()
        .starts_with("1,crossing,true,"));

    // The paths can be left out, and the relaxation of the path needs one surface.
    let config = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-dsafir",
        "-es",
        "1",
        "2",
        "-ieip_opt",
        "optimize=on",
        "optimize_irc=off",
    ])
    .unwrap();
    assert!(config.extra.follow_up.on && !config.extra.follow_up.descend);
    let config = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-dsafir",
        "-es",
        "1",
        "2",
        "-ieip_opt",
        "dsafir_lup=3",
        "figures=off",
    ])
    .unwrap();
    let e = run_with(&config, &base, None, &CalculatorRegistry::native())
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("needs one surface"), "{e}");
}

#[test]
fn a_method_on_a_function_of_two_states_names_candidates_for_their_crossing() {
    let base = scratch("state_function");
    let input = ozone_in_two_states(&base);

    // -mf with a method of mopt-rs: the method runs on the function of the two states.
    let args = [
        input.as_str(),
        "-os",
        "gfn0xtb",
        "-dsafir",
        "-mf",
        "seam",
        "-es",
        "1",
        "2",
        "-ieip_opt",
        "figures=off",
        "optimize=on",
        "optimize_irc=off",
    ];
    let config = config_from_cli(&args).unwrap();
    assert_eq!(config.surface_function.as_deref(), Some("seam"));
    let result = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    println!("{log}");
    assert!(log.contains("Mode: Model Function (seam)"), "{log}");
    // One function for every point: not the search with an end on each surface.
    assert!(!log.contains("each end of DS-AFIR moves on the surface of its own state"));
    assert!(
        log.contains("State1") && log.contains("kcal/mol apart"),
        "{log}"
    );
    assert!(!log.contains("NaN"), "{log}");

    // The low point of the function along the path is the candidate for the crossing: the
    // states are closer there than at the first structure (the second is itself next to
    // the crossing here, and the candidate is a step from it), and the search for the
    // conical intersection from it finds one.
    assert!(log.contains("a candidate for a crossing"), "{log}");
    assert_eq!(result.transition_states.len(), 1);
    let candidate = &result.transition_states[0];
    assert!(
        candidate.ends_with("o3_A_crossing_guess.xyz"),
        "{candidate:?}"
    );
    let pair = Path::new(&input);
    let at_candidate = gap_of_excited_states(candidate);
    assert!(at_candidate < 0.5 * gap_of_excited_states(&pair.join("o3_A.xyz")));
    assert_eq!(result.optimized.len(), 1);
    let found = &result.optimized[0];
    assert!(found.crossing && found.converged);
    assert!(found.descents.is_empty());
    let gap = gap_of_excited_states(&found.optimized);
    assert!(gap < 5.0e-4, "gap {gap}");

    // Without a method of mopt-rs, -mf is MultiOptPy's mode as before.
    let config = config_from_cli(&[input.as_str(), "-os", "gfn0xtb", "-mf", "seam"]).unwrap();
    assert!(config.surface_function.is_none());
    // A projection for crossings is not a function a path can follow.
    let config = config_from_cli(&[
        input.as_str(),
        "-os",
        "gfn0xtb",
        "-dsafir",
        "-mf",
        "opt_meci",
        "-es",
        "1",
        "2",
        "-ieip_opt",
        "figures=off",
    ])
    .unwrap();
    let e = run_with(&config, &base, None, &CalculatorRegistry::native())
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("projects gradients"), "{e}");
}
