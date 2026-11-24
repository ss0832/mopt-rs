//! `optmain` on plain variables through the command-line layer: `-os mb2d` and `-os expr:...`
//! with a variables file (text or JSON) as the input.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::cli::variables::{parse_variables, read_variables_file};
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_vars_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

fn near(v: &[f64], p: [f64; 2]) -> bool {
    ((v[0] - p[0]).powi(2) + (v[1] - p[1]).powi(2)).sqrt() < 2e-2
}

#[test]
fn muller_brown_minimum_and_run_directory() {
    let dir = scratch("min");
    let start = file(
        &dir,
        "start.txt",
        "# Muller-Brown, near minimum A\n-0.4\n1.3\n",
    );
    let cfg =
        config_from_cli(&[&start, "-os", "mb2d", "-opt", "rsirfo_bfgs", "-ns", "300"]).unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].converged && !res[0].interrupted);
    let run_dir = &res[0].directory;
    for f in [
        "input.txt",
        "settings.json",
        "mopt.log",
        "start_traj.csv",
        "start_optimized.txt",
        "value_plot_.png",
        "plot_gradient_.png",
    ] {
        assert!(run_dir.join(f).exists(), "{f}");
    }
    // No molecular output.
    assert!(!run_dir.join("start_traj.xyz").exists());
    let out = read_variables_file(&res[0].optimized).unwrap();
    assert_eq!(out.names, Some(vec!["x".to_string(), "y".to_string()]));
    assert!(near(&out.values, MB_MINIMA[0]), "{:?}", out.values);
    let csv = std::fs::read_to_string(run_dir.join("start_traj.csv")).unwrap();
    let mut lines = csv.lines();
    assert_eq!(
        lines.next().unwrap(),
        "iteration,value,max_gradient,rms_gradient,max_step,rms_step,trust_radius,x,y,d/dx,d/dy"
    );
    let first: Vec<f64> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    assert_eq!(first.len(), 11);
    assert!((first[7] + 0.4).abs() < 1e-12 && (first[8] - 1.3).abs() < 1e-12);
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(
        log.contains("optimization of 2 plain variables (x, y)"),
        "{log}"
    );
    assert!(log.contains("converged!!!"));
    // The optimized file is a valid input: it converges at once.
    let again = res[0].optimized.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[&again, "-os", "mb2d", "-opt", "rsirfo_bfgs"]).unwrap();
    let res2 = run(&cfg, &dir, None).unwrap();
    assert!(res2[0].converged);
    let csv2 = std::fs::read_to_string(res2[0].directory.join("start_optimized_traj.csv")).unwrap();
    assert!(csv2.lines().count() <= 3, "{csv2}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn muller_brown_saddles_with_rsirfo_and_ccqn() {
    let dir = scratch("ts");
    let start = file(&dir, "ts.txt", "x -0.8\ny 0.6\n");
    let cfg = config_from_cli(&[
        &start,
        "-os",
        "mb2d",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-ns",
        "300",
    ])
    .unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].converged);
    let out = read_variables_file(&res[0].optimized).unwrap();
    assert!(near(&out.values, MB_SADDLES[0]), "{:?}", out.values);

    // CCQN: the axis points from the start toward the product given as a variables file.
    let (c, b, ts) = (MB_MINIMA[2], MB_MINIMA[1], MB_SADDLES[1]);
    let begin = [ts[0] + 0.3 * (c[0] - ts[0]), ts[1] + 0.3 * (c[1] - ts[1])];
    let start = file(&dir, "ccqn.txt", &format!("{}\n{}\n", begin[0], begin[1]));
    let product = file(&dir, "product.txt", &format!("{}\n{}\n", b[0], b[1]));
    let cfg = config_from_cli(&[
        &start,
        "-os",
        "mb2d",
        "-opt",
        "ccqn",
        "-order",
        "1",
        "-fc",
        "5",
        "-ccqn_product",
        &product,
        "-ccqn_cosphi",
        "0.9",
        "-ns",
        "300",
    ])
    .unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].converged);
    let out = read_variables_file(&res[0].optimized).unwrap();
    assert!(near(&out.values, ts), "{:?}", out.values);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn expression_objective_with_json_variables() {
    let dir = scratch("expr");
    let start = file(
        &dir,
        "five.json",
        r#"{"variables": [0, 0, 0, 0, 0], "names": ["a", "b", "c", "d", "e"]}"#,
    );
    let cfg = config_from_cli(&[
        &start,
        "-os",
        "expr:(q1-1)^2 + 2*(q2+0.5)^2 + 3*(q3-0.25)^2 + (q4-q5)^2 + q5^2 + 0.1*q1*q2",
        "-opt",
        "rsirfo_bfgs",
        "-tcc",
    ])
    .unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].converged);
    assert!(res[0]
        .optimized
        .to_string_lossy()
        .ends_with("five_optimized.json"));
    let text = std::fs::read_to_string(&res[0].optimized).unwrap();
    let out = parse_variables(&text).unwrap();
    assert!(out.json);
    assert_eq!(out.names.unwrap()[4], "e");
    // Stationary point of the quadratic: 2(q1-1) + 0.1 q2 = 0, 4(q2+0.5) + 0.1 q1 = 0.
    let q2 = (-2.0 - 0.1) / (4.0 - 0.005);
    let q1 = 1.0 - 0.05 * q2;
    for (got, want) in out.values.iter().zip([q1, q2, 0.25, 0.0, 0.0]) {
        assert!((got - want).abs() < 1e-4, "{:?}", out.values);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn options_that_need_atoms_are_refused() {
    let dir = scratch("refuse");
    let start = file(&dir, "s.txt", "-0.4\n1.3\n");
    for extra in [
        vec!["-ma", "100", "1", "2"],
        vec!["-fix", "1"],
        vec!["-coord", "dic"],
        vec!["-modelhess", "lindh"],
        vec!["-elec", "1"],
        vec!["-opt", "ccqn", "-order", "1", "-ccqn_bonds", "1,2"],
    ] {
        let mut args = vec![&start[..], "-os", "mb2d"];
        args.extend(extra.iter().copied());
        let cfg = config_from_cli(&args).unwrap();
        let err = run(&cfg, &dir, None).unwrap_err().to_string();
        assert!(err.contains("needs atoms"), "{extra:?}: {err}");
    }
    // The number of variables must fit the objective, and expressions must not use more.
    let three = file(&dir, "three.txt", "0\n0\n0\n");
    let cfg = config_from_cli(&[&three, "-os", "mb2d"]).unwrap();
    assert!(run(&cfg, &dir, None).is_err());
    let cfg = config_from_cli(&[&start, "-os", "expr:q1^2 + q2^2 + q3^2"]).unwrap();
    assert!(run(&cfg, &dir, None).is_err());
    let cfg = config_from_cli(&[&start, "-os", "expr:dist(1,2)"]).unwrap();
    assert!(run(&cfg, &dir, None).is_err());
    // An xyz file is not a variables file.
    let xyz = file(&dir, "a.xyz", "1\n\nAr 0.0 0.0 0.0\n");
    let cfg = config_from_cli(&[&xyz, "-os", "mb2d"]).unwrap();
    assert!(run(&cfg, &dir, None).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stop_file_and_restart() {
    let dir = scratch("stop");
    let start = file(&dir, "s.txt", "-0.4\n1.3\n");
    let run_dir = dir.join("run");
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(run_dir.join("STOP"), "").unwrap();
    let mut cfg = config_from_cli(&[&start, "-os", "mb2d", "-opt", "rsirfo_bfgs"]).unwrap();
    cfg.restart_dir = Some(run_dir.clone());
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].interrupted && !res[0].converged);
    assert!(!run_dir.join("STOP").exists());
    let csv = std::fs::read_to_string(run_dir.join("s_traj.csv")).unwrap();
    assert_eq!(csv.lines().count(), 2);
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].converged && !res[0].interrupted);
    let out = read_variables_file(&res[0].optimized).unwrap();
    assert!(near(&out.values, MB_MINIMA[0]));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The same minimum search enables downhill HiSD only when the final check finds a saddle.
#[test]
fn minimum_mode_kill_starts_hisd_only_for_excess_negative_curvature() {
    let dir = scratch("downhill_hisd");
    for (start_x, enable, method, escape) in [
        (0.0_f64, false, None, false),
        (0.0, false, Some("hisd"), false),
        (0.0, true, None, true),
        (0.0, true, Some("classic"), true),
        (0.0, true, Some("hisd"), true),
        (1.0, true, None, false),
    ] {
        let base = dir.join(format!(
            "{start_x}_{enable}_{}",
            method.unwrap_or("default")
        ));
        std::fs::create_dir_all(&base).unwrap();
        let start = file(&base, "start.txt", &format!("{start_x}\n0.0\n"));
        let mut args = vec![
            start.as_str(),
            "-os",
            "expr:(q1^2-1)^2+q2^2",
            "-opt",
            "rsirfo_bfgs",
        ];
        if enable {
            args.push("-modekill");
        }
        if let Some(method) = method {
            args.extend(["-modekill_method", method]);
        }
        let result = run(&config_from_cli(&args).unwrap(), &base, None).unwrap();
        assert!(result[0].converged);
        let end = read_variables_file(&result[0].optimized).unwrap();
        let expected = if escape { 1.0 } else { start_x };
        assert!((end.values[0].abs() - expected).abs() < 1e-3, "{end:?}");
        let log = std::fs::read_to_string(result[0].directory.join("mopt.log")).unwrap();
        assert_eq!(
            log.contains("continuing with HiSD at order 0"),
            escape && method != Some("classic"),
            "{log}"
        );
    }
}

/// `-modekill`: a search for a first-order saddle that ends, by symmetry, at a saddle of order
/// 2 leaves it; the log says so.
#[test]
fn mode_kill_on_the_command_line() {
    let dir = scratch("modekill");
    let start = file(&dir, "start.txt", "0.3\n0.0\n0.4\n");
    let common = [
        start.as_str(),
        "-os",
        "expr:-0.5*q1^2 - 0.3*q2^2 + 0.25*q2^4 + q3^2",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1000",
    ];
    let res = run(&config_from_cli(&common).unwrap(), &dir, None).unwrap();
    let kept = read_variables_file(&res[0].optimized).unwrap();
    assert!(res[0].converged && kept.values[1].abs() < 1e-8);
    let mut with_kill = common.to_vec();
    with_kill.push("-modekill");
    for method in [None, Some("hisd")] {
        let mut args = with_kill.clone();
        if let Some(name) = method {
            args.extend(["-modekill_method", name]);
        }
        let res = run(&config_from_cli(&args).unwrap(), &dir, None).unwrap();
        let left = read_variables_file(&res[0].optimized).unwrap();
        assert!(res[0].converged);
        assert!(
            (left.values[1].abs() - 0.6_f64.sqrt()).abs() < 1e-3,
            "{:?}",
            left.values
        );
        assert!(left.values[0].abs() < 1e-3 && left.values[2].abs() < 1e-3);
        let log = std::fs::read_to_string(res[0].directory.join("mopt.log")).unwrap();
        assert!(
            log.contains("Mode kill: the converged point has 2 imaginary mode(s)"),
            "{log}"
        );
        assert_eq!(
            log.contains("continuing with HiSD at order 1"),
            method.is_some(),
            "{log}"
        );
    }
    with_kill.extend(["-modekill_method", "unknown"]);
    assert!(config_from_cli(&with_kill).is_err());
}
