//! `run_kinetics` with the models, solvers, corrections, and analyses beyond the canonical
//! model, on networks in MultiOptPy's format with annotations: `egme` (with its
//! phenomenological rate constants), `two_grain`, `semi_markov` with `kmc` and from a
//! trajectory (`--md-trajectory`), the corrections `kramers`, `grote_hynes`, `reactive_flux`,
//! `small_curvature`, measured `outcomes`, and `--refine`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use mopt_rs::domain::Units;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mopt_kinetics_models_{name}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const WATER: &str = "3\\n\\nO 0.0 0.0 0.1173\\nH 0.0 0.7572 -0.4692\\nH 0.0 -0.7572 -0.4692\\n";

/// A network of MultiOptPy's format: nodes `(id, energy)`, edges `(a, b, energy)`.
fn network(dir: &Path, nodes: &[f64], edges: &[(usize, usize, f64)]) -> PathBuf {
    let n: Vec<String> = nodes
        .iter()
        .enumerate()
        .map(|(k, e)| format!("{{\"node_id\": {k}, \"energy_hartree\": {e}}}"))
        .collect();
    let m: Vec<String> = edges
        .iter()
        .enumerate()
        .map(|(k, (a, b, e))| {
            format!(
                "{{\"edge_id\": {k}, \"node_id_1\": {a}, \"node_id_2\": {b}, \"ts_energy_hartree\": {e}}}"
            )
        })
        .collect();
    let file = dir.join("network.json");
    std::fs::write(
        &file,
        format!(
            "{{\"nodes\": [{}], \"edges\": [{}]}}",
            n.join(", "),
            m.join(", ")
        ),
    )
    .unwrap();
    file
}

fn run(args: &[String]) -> i32 {
    mopt_rs::cli::run_command("run_kinetics", args)
}

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// The populations of the last line of `populations.csv`.
fn last_populations(out: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(out.join("populations.csv")).unwrap();
    let last = text.lines().rfind(|l| !l.trim().is_empty()).unwrap();
    last.split(',')
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect()
}

#[test]
fn egme_and_two_grain_from_the_command_line() {
    let dir = scratch("egme");
    let net = network(&dir, &[0.0, 0.002], &[(0, 1, 0.04)]);
    let ann = dir.join("ann.json");
    std::fs::write(
        &ann,
        format!(
            "{{\"nodes\": {{\"0\": {{\"xyz\": \"{WATER}\", \"frequencies_cm1\": [1600, 3600, 3700]}}, \
             \"1\": {{\"xyz\": \"{WATER}\", \"frequencies_cm1\": [1500, 3500, 3650]}}}}, \
             \"edges\": {{\"TS0\": {{\"xyz\": \"{WATER}\", \"frequencies_cm1\": [1200, 2800, -1500]}}}}}}"
        ),
    )
    .unwrap();
    let out = dir.join("egme");
    let mut a = args(&[
        "--rate-theory",
        "rrkm",
        "--model",
        "egme",
        "--collision-model",
        "strong",
        "--collision-frequency",
        "1e12",
        "--egme-grain",
        "200",
        "-T",
        "1500",
        "--times",
        "1e-9",
        "1e-6",
        "--validate",
        "1e-8",
    ]);
    a.insert(0, net.display().to_string());
    a.extend([
        "--annotations".into(),
        ann.display().to_string(),
        "--output-dir".into(),
        out.display().to_string(),
    ]);
    assert_eq!(run(&a), 0);
    let p = last_populations(&out);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{p:?}");
    let phen = std::fs::read_to_string(out.join("phenomenological.csv")).unwrap();
    assert!(
        phen.contains("k_contraction_a_s-1") && phen.lines().count() >= 3,
        "{phen}"
    );
    // propagate on the grains against the matrix exponential at both times.
    let diag = std::fs::read_to_string(out.join("diagnostics.json")).unwrap();
    assert_eq!(diag.matches("\"passed\": true").count(), 2, "{diag}");
    // The model needs rrkm and refuses the derivatives.
    let mut bad = a.clone();
    bad.extend(["--sensitivity".into()]);
    assert_ne!(run(&bad), 0);
    // two_grain on the same network (no intermediate gets hot: the canonical populations).
    let out2 = dir.join("two");
    let mut b = args(&[
        "--rate-theory",
        "rrkm",
        "--model",
        "two_grain",
        "--cooling-rate",
        "1e12",
        "-T",
        "1500",
        "--times",
        "1e-6",
    ]);
    b.insert(0, net.display().to_string());
    b.extend([
        "--annotations".into(),
        ann.display().to_string(),
        "--output-dir".into(),
        out2.display().to_string(),
    ]);
    assert_eq!(run(&b), 0);
    let out3 = dir.join("canonical");
    let mut c = args(&["--rate-theory", "rrkm", "-T", "1500", "--times", "1e-6"]);
    c.insert(0, net.display().to_string());
    c.extend([
        "--annotations".into(),
        ann.display().to_string(),
        "--output-dir".into(),
        out3.display().to_string(),
    ]);
    assert_eq!(run(&c), 0);
    let (x, y) = (last_populations(&out2), last_populations(&out3));
    for (u, v) in x.iter().zip(&y) {
        assert!((u - v).abs() < 1e-10, "{x:?} {y:?}");
    }
}

#[test]
fn corrections_from_the_annotations() {
    let dir = scratch("corrections");
    let t = 298.15;
    let net = network(&dir, &[0.0, -0.005], &[(0, 1, 0.02)]);
    let base = |extra: &[&str], ann: &str, name: &str| -> String {
        let file = dir.join(format!("{name}.json"));
        std::fs::write(&file, ann).unwrap();
        let out = dir.join(name);
        let mut a = vec![net.display().to_string(), "-T".into(), format!("{t}")];
        a.extend(extra.iter().map(|s| s.to_string()));
        a.extend([
            "--annotations".into(),
            file.display().to_string(),
            "--output-dir".into(),
            out.display().to_string(),
        ]);
        assert_eq!(run(&a), 0, "{name}");
        std::fs::read_to_string(out.join("rates.csv")).unwrap()
    };
    // k from ln_k_ab (ten digits; the column k_ab has seven).
    let k_ab = |rates: &str| -> f64 {
        let line = rates.lines().find(|l| l.starts_with("TS0,")).unwrap();
        line.split(',')
            .nth(8)
            .unwrap()
            .parse::<f64>()
            .unwrap()
            .exp()
    };
    let plain = base(&[], "{}", "plain");
    // Kramers with gamma = 2 omega_b: kappa = sqrt(2) - 1.
    let omega = 2.0 * std::f64::consts::PI * Units::SPEED_OF_LIGHT_CM_S * 1000.0;
    let kramers = base(
        &[
            "--corrections",
            "kramers",
            "--friction",
            &format!("{}", 2.0 * omega),
        ],
        "{\"edges\": {\"TS0\": {\"imaginary_wavenumber_cm1\": -1000}}}",
        "kramers",
    );
    let ratio = k_ab(&kramers) / k_ab(&plain);
    assert!((ratio - (2f64.sqrt() - 1.0)).abs() < 1e-9, "{ratio}");
    assert!(kramers.contains("gamma_over_omega_b"), "{kramers}");
    // Grote-Hynes with an exponential memory of the same friction is between Kramers and 1.
    let gh = base(
        &["--corrections", "grote_hynes"],
        &format!(
            "{{\"edges\": {{\"TS0\": {{\"imaginary_wavenumber_cm1\": -1000, \"friction_s-1\": {}, \
             \"friction_memory_s\": 2e-14}}}}}}",
            2.0 * omega
        ),
        "grote_hynes",
    );
    let r = k_ab(&gh) / k_ab(&plain);
    assert!(r > 2f64.sqrt() - 1.0 && r < 1.0, "{r}");
    // A transmission coefficient from trajectories.
    let rf = base(
        &["--corrections", "reactive_flux"],
        "{\"edges\": {\"TS0\": {\"reactive_flux_kappa\": 0.4, \"reactive_flux_error\": 0.05}}}",
        "reactive_flux",
    );
    assert!((k_ab(&rf) / k_ab(&plain) - 0.4).abs() < 1e-9);
    // Small-curvature tunnelling from a path profile (a parabolic adiabatic barrier with
    // effective masses below 1 tunnels more than zero curvature).
    let mut csv =
        String::from("point,arc_length_sqrt_amu_bohr,adiabatic_energy_hartree,sct_mass_ratio\n");
    for k in 0..81 {
        let s = -2.0 + 0.05 * k as f64;
        csv.push_str(&format!("{k},{s},{},{}\n", -0.005 * s * s, 0.6));
    }
    std::fs::write(dir.join("path_profile.csv"), csv).unwrap();
    let sct = base(
        &["--corrections", "small_curvature"],
        "{\"edges\": {\"TS0\": {\"sct_profile_file\": \"path_profile.csv\"}}}",
        "sct",
    );
    let zct = base(
        &["--corrections", "zero_curvature"],
        "{\"edges\": {\"TS0\": {\"sct_profile_file\": \"path_profile.csv\"}}}",
        "zct",
    );
    assert!(k_ab(&sct) > k_ab(&zct) && k_ab(&zct) > k_ab(&plain));
    // The refused names say why.
    let mut a = vec![
        net.display().to_string(),
        "--corrections".into(),
        "marcus".into(),
    ];
    a.extend([
        "--output-dir".into(),
        dir.join("marcus").display().to_string(),
    ]);
    assert_ne!(run(&a), 0);
}

#[test]
fn measured_outcomes_and_refinement_tasks() {
    let dir = scratch("outcomes");
    // A -> B over TS0, B splits into C and D over TS1 (measured 0.8 : 0.2).
    let net = network(
        &dir,
        &[0.0, -0.01, -0.05, -0.05],
        &[(0, 1, 0.025), (1, 2, 0.0)],
    );
    let ann = dir.join("ann.json");
    std::fs::write(
        &ann,
        "{\"edges\": {\"TS1\": {\"outcomes\": {\"from\": \"1\", \"to\": {\"2\": 0.8, \"3\": 0.2}}, \
         \"outcome_errors\": {\"2\": 0.04, \"3\": 0.04}}}}",
    )
    .unwrap();
    let out = dir.join("out");
    let a = vec![
        net.display().to_string(),
        "-T".into(),
        "300".into(),
        "--times".into(),
        "1e-3".into(),
        "1".into(),
        "--annotations".into(),
        ann.display().to_string(),
        "--refine".into(),
        "2".into(),
        "--output-dir".into(),
        out.display().to_string(),
    ];
    assert_eq!(run(&a), 0);
    let p = last_populations(&out);
    assert!((p[2] / p[3] - 4.0).abs() < 1e-6, "{p:?}");
    let refine = std::fs::read_to_string(out.join("refinement.csv")).unwrap();
    let first = refine.lines().nth(1).unwrap();
    assert!(first.contains("branching,TS1"), "{refine}");
}

#[test]
fn semi_markov_with_kmc_and_from_a_trajectory() {
    let dir = scratch("semi_markov");
    let t = 300.0;
    let kt = Units::BOLTZMANN_HARTREE_PER_K * t;
    let kth = Units::BOLTZMANN_J_PER_K * t / Units::PLANCK_J_S;
    // Two states with rates 2 and 1 1/s.
    let ts = kt * (kth / 2.0).ln();
    let net = network(&dir, &[0.0, -kt * 2f64.ln()], &[(0, 1, ts)]);
    let ann = dir.join("ann.json");
    std::fs::write(&ann, "{\"nodes\": {\"0\": {\"residence_shape\": 4.0}}}").unwrap();
    let out = dir.join("kmc");
    let a = vec![
        net.display().to_string(),
        "-T".into(),
        format!("{t}"),
        "--model".into(),
        "semi_markov".into(),
        "--solver".into(),
        "kmc".into(),
        "--kmc-trajectories".into(),
        "20000".into(),
        "--times".into(),
        "10".into(),
        "--annotations".into(),
        ann.display().to_string(),
        "--output-dir".into(),
        out.display().to_string(),
    ];
    assert_eq!(run(&a), 0);
    let p = last_populations(&out);
    // The stationary populations 1/3, 2/3 within three standard errors.
    let se = (2.0f64 / 9.0 / 20000.0).sqrt();
    assert!((p[0] - 1.0 / 3.0).abs() < 3.0 * se, "{p:?}");
    assert!(std::fs::read_to_string(out.join("memory.csv"))
        .unwrap()
        .contains("k_memory_s-1"));
    // A trajectory of a hydrogen molecule that dissociates and comes back: residence times
    // from the frames.
    let bonded = "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
    let apart = "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 3.50\n";
    let mut traj = String::new();
    for (frames, s) in [
        (10, bonded),
        (4, apart),
        (6, bonded),
        (4, apart),
        (8, bonded),
        (4, apart),
        (6, bonded),
    ] {
        for _ in 0..frames {
            traj.push_str(s);
        }
    }
    std::fs::write(dir.join("traj.xyz"), traj).unwrap();
    let ann2 = dir.join("ann2.json");
    std::fs::write(
        &ann2,
        format!(
            "{{\"nodes\": {{\"0\": {{\"xyz\": \"{}\"}}, \"1\": {{\"xyz\": \"{}\"}}}}}}",
            bonded.replace('\n', "\\n"),
            apart.replace('\n', "\\n")
        ),
    )
    .unwrap();
    let out2 = dir.join("md");
    let b = vec![
        net.display().to_string(),
        "--model".into(),
        "semi_markov".into(),
        "--md-trajectory".into(),
        dir.join("traj.xyz").display().to_string(),
        "--md-time-step".into(),
        "1".into(),
        "--times".into(),
        "1e-12".into(),
        "--annotations".into(),
        ann2.display().to_string(),
        "--output-dir".into(),
        out2.display().to_string(),
    ];
    assert_eq!(run(&b), 0);
    let msm = std::fs::read_to_string(out2.join("msm.csv")).unwrap();
    // The completed visits: two of the bonded molecule (6 and 8 frames), three of the
    // separated atoms (4 frames each).
    let bonded_line = msm
        .lines()
        .find(|l| l.starts_with("EQ0,") || l.starts_with("0,"))
        .unwrap();
    let cells: Vec<&str> = bonded_line.split(',').collect();
    assert_eq!(cells[1], "2", "{msm}");
    let mean: f64 = cells[2].parse().unwrap();
    assert!((mean - 7e-15).abs() < 1e-24, "{mean}");
}

#[test]
fn mass_action_by_contraction_from_the_command_line() {
    // A + B <-> C fast (the association and the dissociation far faster than 1/t), C -> D
    // slow: the contraction of mass action against the stiff integration.
    let dir = scratch("contract");
    let net = network(&dir, &[0.0, -0.005, -0.01], &[(0, 1, 0.002), (1, 2, 0.015)]);
    let run_with = |solver: &str| -> (Vec<f64>, String) {
        let out = dir.join(solver);
        let mut a = vec![net.display().to_string()];
        a.extend(args(&[
            "--model",
            "mass_action",
            "--solver",
            solver,
            "--species",
            "B:0.0",
            "--partner",
            "TS0:B:EQ0",
            "--concentration",
            "B:1e-3",
            "--total-concentration",
            "1e-3",
            "--times",
            "1e-4",
            "1e-3",
            "--ode-tolerance",
            "1e-9",
        ]));
        a.extend(["--output-dir".into(), out.display().to_string()]);
        assert_eq!(run(&a), 0, "{solver}");
        let text = std::fs::read_to_string(out.join("concentrations.csv")).unwrap();
        let last = text.lines().rfind(|l| !l.trim().is_empty()).unwrap();
        let values = last
            .split(',')
            .skip(1)
            .map(|x| x.parse().unwrap())
            .collect();
        (
            values,
            std::fs::read_to_string(out.join("diagnostics.json")).unwrap(),
        )
    };
    let (contracted, diag) = run_with("contract");
    let (integrated, _) = run_with("rosenbrock");
    for (x, y) in contracted.iter().zip(&integrated) {
        assert!((x - y).abs() < 1e-5 * 1e-3, "{contracted:?} {integrated:?}");
    }
    assert!(
        diag.contains("\"eliminated\": 1") || diag.contains("\"eliminated\": 2"),
        "{diag}"
    );
}

#[test]
fn the_orders_of_the_steps_are_judged_automatically() {
    // A + B <-> C, C -> D with the states together at 1 mM: B at 1 M is a reservoir (the
    // canonical model, pseudo-first order), B at 1 mM is not (mass action).
    let dir = scratch("auto");
    let net = network(&dir, &[0.0, -0.005, -0.01], &[(0, 1, 0.002), (1, 2, 0.015)]);
    for (b, model) in [("B:1.0", "canonical"), ("B:1e-3", "mass_action")] {
        let out = dir.join(model);
        let mut a = vec![net.display().to_string()];
        a.extend(args(&[
            "--model",
            "auto",
            "--species",
            "B:0.0",
            "--partner",
            "TS0:B:EQ0",
            "--concentration",
            b,
            "--total-concentration",
            "1e-3",
            "--times",
            "1e-3",
        ]));
        a.extend(["--output-dir".into(), out.display().to_string()]);
        assert_eq!(run(&a), 0, "{model}");
        let diag = std::fs::read_to_string(out.join("diagnostics.json")).unwrap();
        assert!(diag.contains(&format!("\"model\": \"{model}\"")), "{diag}");
        assert!(diag.contains("auto:"), "{diag}");
    }
    // Pseudo-first-order windows agree with the stiff integration of mass action.
    let solve = |solver: &str| -> Vec<f64> {
        let out = dir.join(format!("ma_{solver}"));
        let mut a = vec![net.display().to_string()];
        a.extend(args(&[
            "--model",
            "mass_action",
            "--solver",
            solver,
            "--species",
            "B:0.0",
            "--partner",
            "TS0:B:EQ0",
            "--concentration",
            "B:2e-3",
            "--total-concentration",
            "1e-3",
            "--times",
            "1e-3",
            "--ode-tolerance",
            "1e-9",
            "--window-tolerance",
            "1e-4",
        ]));
        a.extend(["--output-dir".into(), out.display().to_string()]);
        assert_eq!(run(&a), 0, "{solver}");
        let text = std::fs::read_to_string(out.join("concentrations.csv")).unwrap();
        let last = text.lines().rfind(|l| !l.trim().is_empty()).unwrap();
        last.split(',')
            .skip(1)
            .map(|x| x.parse().unwrap())
            .collect()
    };
    let (w, r) = (solve("windows"), solve("rosenbrock"));
    for (x, y) in w.iter().zip(&r) {
        assert!((x - y).abs() < 1e-2 * 1e-3, "{w:?} {r:?}");
    }
}
