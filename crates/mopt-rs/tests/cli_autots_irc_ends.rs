//! The ends of the IRC of `run_autots` (step 4) are minima by the one criterion of an imaginary
//! mode (`opt::mode_kill::IMAGINARY_CURVATURE`, the mass-weighted Hessian with the rigid
//! motions projected out), by default: the transition state of the SN2 reaction OH- + CH3F
//! (`examples/python_autots`, GFN1-xTB), made symmetric under the plane z = 0 (Cs; a stationary
//! point to the 12 decimals of its file, as `optmain -order 1` from it shows), is the input of
//! `--skip_to_step4`. Its IRC keeps the symmetry, and the reactant side ends at the symmetric
//! stationary point of the ion-molecule complex, a saddle point with one imaginary mode of
//! -64.5 cm^-1 that a plain minimization keeps. The mode kill of the minimization of the ends
//! leaves it to the true minimum, 0.053 kcal/mol lower. (A whole run from a symmetric
//! reactant keeps the symmetry only approximately, and whether its IRC meets the saddle point
//! then depends on digits: the transition state is given here.)
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

use mopt_rs::cli::autots::{run_with, AutotsConfig, EndCheck};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::math::projection::build_tr_rot_basis;
use mopt_rs::opt::mode_kill::imaginary_modes;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

/// The transition state, symmetric under z = 0 (Angstrom).
const SYMMETRIC: &str = "7\n-1 1\nC 0.533620466219 0.088866913666 0.0\n\
F 2.083794081379 -0.003254136902 0.0\nH 0.407164649479 1.157848478140 0.0\n\
H 0.318386866476 -0.427667189069 -0.920609594030\n\
H 0.318386866476 -0.427667189069 0.920609594030\n\
O -1.654351555459 0.255263042194 0.0\nH -2.007001374571 -0.643389918961 0.0\n";

const SETTINGS: &str = r#"{
  "work_dir": "work",
  "skip_to_step4": true,
  "run_step4": true,
  "step4_settings": {"othersoft": "gfn1xtb", "opt_method": ["rsirfo_block_bofill"],
                     "calc_exact_hess": 10, "tight_convergence_criteria": true,
                     "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"],
                     "step4b_opt_method": ["rsirfo_block_fsb"]}
}"#;

#[test]
fn an_end_is_a_minimum_only_when_converged_without_imaginary_modes() {
    assert_eq!(EndCheck::of(true, true, Some(0)), EndCheck::Minimum);
    for (converged, modes) in [
        (true, Some(1)),
        (false, None),
        (true, None),
        (false, Some(0)),
    ] {
        let check = EndCheck::of(true, converged, modes);
        assert!(matches!(check, EndCheck::NotVerified { .. }), "{check:?}");
        assert!(check.describe().starts_with("end not verified"));
        assert_eq!(check.word(), "not_verified");
    }
    assert_eq!(EndCheck::of(false, true, Some(1)), EndCheck::NotChecked);
}

#[test]
fn every_end_of_the_irc_is_a_minimum_by_the_one_criterion() {
    let dir = std::env::temp_dir().join(format!("mopt_autots_ends_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("sn2.xyz");
    std::fs::write(&input, SYMMETRIC).unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, SETTINGS).unwrap();
    let args = [
        input.display().to_string(),
        "-cfg".into(),
        cfg.display().to_string(),
    ];
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let config =
        AutotsConfig::from_args(&parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap()).unwrap();
    let result = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    // The minimizations of the ends kill the imaginary modes by default.
    for (k, v) in result
        .effective_settings
        .iter()
        .filter(|(k, _)| k.contains("endpoint"))
    {
        assert!(v.contains("mode_kill: Some"), "{k} has no mode kill:\n{v}");
    }
    // The case is the one of the saddle point: an end of the IRC converged on it and was left
    // along its imaginary mode.
    let killed = std::fs::read_dir(&result.directory)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.to_string_lossy().contains("IRC_Endpoint"))
        .any(|p| {
            std::fs::read_to_string(p.join("mopt.log"))
                .unwrap_or_default()
                .contains("Mode kill: the converged structure has 1 imaginary mode(s)")
        });
    assert!(killed, "no end of the IRC was on a saddle point");
    // The command says so itself: every end it optimized is a verified minimum.
    for ts in result
        .transition_states
        .iter()
        .filter(|t| !t.ends.is_empty())
    {
        assert_eq!(
            ts.end_checks,
            vec![EndCheck::Minimum; ts.ends.len()],
            "{ts:?}"
        );
    }
    let summary = std::fs::read_to_string(result.directory.join("autots_summary.csv")).unwrap();
    assert!(summary.contains(",minimum,minimum"), "{summary}");
    let registry = CalculatorRegistry::native();
    let mut ends = 0;
    for ts in &result.transition_states {
        for (file, _) in &ts.ends {
            let sys = read_xyz_file(file, SpinCharge::new(-1, 1), None).unwrap();
            let calc = registry
                .create("gfn1xtb", &CalculatorOptions::default(), &sys)
                .unwrap();
            let h = calc
                .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
                .unwrap()
                .hessian_bohr
                .unwrap();
            let masses: Vec<f64> = sys.masses_amu().iter().flat_map(|&m| [m, m, m]).collect();
            let rigid = build_tr_rot_basis(&sys, false);
            let n = imaginary_modes(&h, Some(&masses), Some(&rigid)).unwrap();
            eprintln!("{}: {n} imaginary mode(s)", file.display());
            assert_eq!(n, 0, "{} is no minimum", file.display());
            ends += 1;
        }
    }
    assert!(ends >= 2, "the run gave no IRC ends");
    // --skip_to_step4: the start of the IRC judged the input by its exact Hessian.
    let log = std::fs::read_to_string(result.directory.join("autots.log")).unwrap();
    assert!(log.contains("the exact Hessian of the input"), "{log}");
    assert!(
        log.contains("is a first-order saddle point by its exact Hessian"),
        "{log}"
    );
    assert_eq!(result.transition_states[0].imaginary_modes, Some(1));
    assert!(result.transition_states[0].meta_irc.is_none());
    assert!(summary.contains(",irc\n"), "{summary}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The exact Hessians that the IRC run in `dir` computed (its count of evaluations).
fn irc_hessians(dir: &std::path::Path) -> usize {
    let text = std::fs::read_to_string(dir.join("mopt.log")).unwrap_or_default();
    let line = text
        .lines()
        .rev()
        .find(|l| l.contains("exact Hessian calculation"))
        .unwrap_or_else(|| panic!("no count in {}", dir.display()));
    line.split(',')
        .find(|s| s.contains("exact Hessian"))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{line}"))
}

/// The step 4 of a minimal run: the IRC and nothing else of the settings.
const MINIMAL: &str = r#"{
  "work_dir": "work",
  "skip_to_step4": true,
  "step4_settings": {"othersoft": "gfn1xtb", EXTRA
                     "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"]}
}"#;

/// A run of `--skip_to_step4` from `text` in `dir` with `extra` keys in `step4_settings`.
fn skip_to_step4(
    dir: &std::path::Path,
    text: &str,
    extra: &str,
) -> mopt_rs::cli::autots::AutotsResult {
    std::fs::create_dir_all(dir).unwrap();
    let input = dir.join("start.xyz");
    std::fs::write(&input, text).unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, MINIMAL.replace("EXTRA", extra)).unwrap();
    let args = [
        input.display().to_string(),
        "-cfg".into(),
        cfg.display().to_string(),
    ];
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let config =
        AutotsConfig::from_args(&parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap()).unwrap();
    let r = run_with(&config, dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(r.error.is_none(), "{:?}", r.error);
    r
}

#[test]
fn skip_to_step4_computes_the_exact_hessian_of_the_input_once() {
    let dir = std::env::temp_dir().join(format!("mopt_autots_skipts_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let r = skip_to_step4(&dir, SYMMETRIC, "");
    let ts = &r.transition_states[0];
    assert_eq!(ts.imaginary_modes, Some(1));
    assert!(ts.meta_irc.is_none());
    assert_eq!(ts.ends.len(), 2, "{ts:?}");
    // One exact Hessian, that of the input at the start of the IRC.
    assert_eq!(irc_hessians(ts.irc_directory.as_ref().unwrap()), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Water bent the wrong way round: linear, with stretched bonds (two imaginary modes, the
/// bends, and a gradient along the bonds).
const LINEAR_WATER: &str = "3\n0 1\nO 0.0 0.0 0.0\nH 0.0 0.0 1.05\nH 0.0 0.0 -1.05\n";
/// Water near its minimum but not at it: no imaginary mode, not a stationary point.
const NEAR_MINIMUM: &str =
    "3\n0 1\nO 0.0 0.0 0.1173\nH 0.0 0.7972 -0.4692\nH 0.0 -0.7572 -0.4692\n";

#[test]
fn skip_to_step4_from_a_point_that_is_no_first_order_saddle_is_a_meta_irc() {
    for (name, text, modes) in [
        ("linear", LINEAR_WATER, "2 imaginary modes"),
        ("near", NEAR_MINIMUM, "no imaginary mode"),
    ] {
        let dir =
            std::env::temp_dir().join(format!("mopt_autots_meta_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let r = skip_to_step4(&dir, text, "");
        let ts = &r.transition_states[0];
        let notice = ts
            .meta_irc
            .as_deref()
            .unwrap_or_else(|| panic!("{name}: {ts:?}"));
        assert!(
            notice.contains("is not a first-order saddle point")
                && notice.contains(modes)
                && notice.contains("a meta-IRC"),
            "{name}: {notice}"
        );
        assert!(r.notes.iter().any(|n| n == notice), "{name}");
        let log = std::fs::read_to_string(r.directory.join("autots.log")).unwrap();
        assert!(log.contains(notice), "{name}");
        let summary = std::fs::read_to_string(r.directory.join("autots_summary.csv")).unwrap();
        assert!(
            summary.trim_end().ends_with(",meta_irc"),
            "{name}: {summary}"
        );
        let page = std::fs::read_to_string(r.directory.join("autots_report.html")).unwrap();
        assert!(page.contains("meta-IRC from point 1"), "{name}");
        assert!(
            !page.contains("IRC of TS") && !page.contains("<h3>TS "),
            "{name}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn skip_to_step4_takes_a_hessian_handed_in() {
    let dir = std::env::temp_dir().join(format!("mopt_autots_skipih_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ts = dir.join("ts.xyz");
    std::fs::write(&ts, SYMMETRIC).unwrap();
    let sys = read_xyz_file(&ts, SpinCharge::new(-1, 1), None).unwrap();
    let h = CalculatorRegistry::native()
        .create("gfn1xtb", &CalculatorOptions::default(), &sys)
        .unwrap()
        .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    let file = dir.join("ts_hessian.npy");
    mopt_rs::io::npy::write_npy_matrix(&file, &h).unwrap();
    let r = skip_to_step4(
        &dir.join("run"),
        SYMMETRIC,
        &format!(" \"initial_hessian\": \"{}\",", file.display()),
    );
    let log = std::fs::read_to_string(r.directory.join("autots.log")).unwrap();
    assert!(log.contains("the Hessian handed in"), "{log}");
    let t = &r.transition_states[0];
    assert_eq!(t.imaginary_modes, Some(1));
    assert_eq!(irc_hessians(t.irc_directory.as_ref().unwrap()), 0);
    let _ = std::fs::remove_dir_all(&dir);
}
