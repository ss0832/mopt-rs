//! `run_autots` from start to end with the built-in GFN1-xTB: the Diels-Alder reaction of
//! ethylene and s-cis-butadiene by AFIR (the two new C-C bonds pushed together; AFIR is a
//! push along a line, which suits an addition), the chain, the transition state with one
//! imaginary frequency, its IRC to the separated molecules and to cyclohexene, the summary and
//! the report; the SN2 reaction Cl- + CH3Br with step 1 by BITSS between its two ion-molecule
//! complexes; and the settings that step 1 and step 4 need.
//!
//! (AFIR on the SN2 reactions was tried first: with GFN1-xTB the end of the AFIR path of
//! Cl- + CH3Br relaxes back to the reactant complex, and F- + CH3Cl has no barrier, so in both
//! the two ends of the chain are the same minimum.)
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::autots::{run_with, AutotsConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::domain::Units;
use mopt_rs::pes::registry::CalculatorRegistry;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mopt_autots_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn config(d: &Path, json: &str, args: &[&str]) -> AutotsConfig {
    let cfg = d.join("config.json");
    std::fs::write(&cfg, json).unwrap();
    let input = d.join("sn2.xyz");
    // CH3Br with Cl- behind the carbon, opposite the bromine.
    std::fs::write(
        &input,
        "6\n-1 1\nC 0.000 0.000 0.000\nH -0.360 1.010 0.050\nH -0.360 -0.480 0.880\n\
         H -0.360 -0.530 -0.900\nBr 1.960 0.020 -0.010\nCl -3.300 0.080 0.060\n",
    )
    .unwrap();
    let mut all: Vec<String> = vec![
        input.to_string_lossy().into_owned(),
        "-cfg".to_string(),
        cfg.to_string_lossy().into_owned(),
    ];
    all.extend(args.iter().map(|s| (*s).to_string()));
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    let p = parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap();
    AutotsConfig::from_args(&p).unwrap()
}

const SETTINGS: &str = r#"{
  "work_dir": "work",
  "top_n_candidates": 2,
  "step1_settings": {"othersoft": "gfn1xtb", "electronic_charge": -1, "spin_multiplicity": 1},
  "step2_settings": {"othersoft": "gfn1xtb", "electronic_charge": -1, "spin_multiplicity": 1,
                     "NSTEP": 40},
  "step3_settings": {"othersoft": "gfn1xtb", "electronic_charge": -1, "spin_multiplicity": 1,
                     "calc_exact_hess": 5, "tight_convergence_criteria": true},
  "step4_settings": {"othersoft": "gfn1xtb", "electronic_charge": -1, "spin_multiplicity": 1,
                     "calc_exact_hess": 5, "tight_convergence_criteria": true,
                     "intrinsic_reaction_coordinates": ["0.3", "200", "lqa"]}
}"#;

/// C-Cl and C-Br (Angstrom) of a structure file of Cl- + CH3Br.
fn bonds(path: &Path) -> (f64, f64) {
    let text = std::fs::read_to_string(path).unwrap();
    let x: Vec<[f64; 3]> = text
        .lines()
        .skip(2)
        .take(6)
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .skip(1)
                .map(|s| s.parse().unwrap())
                .collect();
            [v[0], v[1], v[2]]
        })
        .collect();
    let d = |i: usize, j: usize| {
        ((x[i][0] - x[j][0]).powi(2) + (x[i][1] - x[j][1]).powi(2) + (x[i][2] - x[j][2]).powi(2))
            .sqrt()
    };
    (d(0, 5), d(0, 4))
}

/// Ethylene over s-cis-butadiene, 3.2 Angstrom apart (C1 and C4 of butadiene over C11 and C12).
const DIELS_ALDER: &str = "16\n0 1 ethylene + s-cis-butadiene\n\
C -1.4050 1.1605 0.0000\nC -0.7350 0.0000 0.0000\nC 0.7350 0.0000 0.0000\n\
C 1.4050 1.1605 0.0000\nH -2.4850 1.1605 0.0000\nH -0.8650 2.0958 0.0000\n\
H -1.3000 -0.9300 0.0000\nH 1.3000 -0.9300 0.0000\nH 0.8650 2.0958 0.0000\n\
H 2.4850 1.1605 0.0000\nC -0.6700 1.1600 3.1000\nC 0.6700 1.1600 3.1000\n\
H -1.2400 2.0900 3.1000\nH -1.2400 0.2300 3.1000\nH 1.2400 2.0900 3.1000\n\
H 1.2400 0.2300 3.1000\n";

const DIELS_ALDER_SETTINGS: &str = r#"{
  "work_dir": "work",
  "top_n_candidates": 1,
  "step1_settings": {"othersoft": "gfn1xtb"},
  "step2_settings": {"othersoft": "gfn1xtb", "NSTEP": 40},
  "step3_settings": {"othersoft": "gfn1xtb", "calc_exact_hess": 10,
                     "tight_convergence_criteria": true},
  "step4_settings": {"othersoft": "gfn1xtb", "calc_exact_hess": 10,
                     "tight_convergence_criteria": true,
                     "intrinsic_reaction_coordinates": ["0.3", "200", "lqa"]}
}"#;

/// The two forming bonds C1-C11 and C4-C12 (Angstrom) of a Diels-Alder structure file.
fn forming_bonds(path: &Path) -> (f64, f64) {
    let text = std::fs::read_to_string(path).unwrap();
    let x: Vec<[f64; 3]> = text
        .lines()
        .skip(2)
        .take(16)
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .skip(1)
                .map(|s| s.parse().unwrap())
                .collect();
            [v[0], v[1], v[2]]
        })
        .collect();
    let d = |i: usize, j: usize| {
        ((x[i][0] - x[j][0]).powi(2) + (x[i][1] - x[j][1]).powi(2) + (x[i][2] - x[j][2]).powi(2))
            .sqrt()
    };
    (d(0, 10), d(3, 11))
}

#[test]
fn afir_from_start_to_end() {
    let d = dir("diels_alder");
    std::fs::write(d.join("config.json"), DIELS_ALDER_SETTINGS).unwrap();
    let input = d.join("da.xyz");
    std::fs::write(&input, DIELS_ALDER).unwrap();
    let all = [
        input.to_string_lossy().into_owned(),
        "-cfg".to_string(),
        d.join("config.json").to_string_lossy().into_owned(),
        "-ma".to_string(),
        "100".to_string(),
        "1".to_string(),
        "11".to_string(),
        "100".to_string(),
        "4".to_string(),
        "12".to_string(),
        "--run_step4".to_string(),
    ];
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    let p = parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap();
    let c = AutotsConfig::from_args(&p).unwrap();
    let r = run_with(&c, &d, None, &CalculatorRegistry::native()).unwrap();
    assert!(r.afir_path.as_ref().unwrap().is_file());
    assert!(!r.candidates.is_empty());
    let ts: Vec<_> = r
        .transition_states
        .iter()
        .filter(|t| t.converged && t.imaginary_modes == Some(1) && t.same_as.is_none())
        .collect();
    assert!(!ts.is_empty(), "{:?}", r.transition_states);
    // One IRC end is the two molecules apart, the other cyclohexene; the transition state
    // forms both bonds at once.
    let found = ts.iter().any(|t| {
        if t.ends.len() != 2 {
            return false;
        }
        let (a, b) = forming_bonds(&t.file);
        let ends: Vec<(f64, f64)> = t.ends.iter().map(|e| forming_bonds(&e.0)).collect();
        println!(
            "TS {:.8} Hartree, bonds {a:.3} {b:.3}; ends: {ends:?}",
            t.energy
        );
        let apart = ends.iter().any(|&(x, y)| x > 2.8 && y > 2.8);
        let ring = ends.iter().any(|&(x, y)| x < 1.7 && y < 1.7);
        (1.9..2.6).contains(&a)
            && (1.9..2.6).contains(&b)
            && apart
            && ring
            && t.ends.iter().all(|e| e.1 < t.energy)
    });
    assert!(found, "{:?}", r.transition_states);
    let summary = std::fs::read_to_string(r.directory.join("autots_summary.csv")).unwrap();
    assert!(summary.lines().count() >= 2, "{summary}");
    assert!(r.directory.join("da_step1_traj.xyz").is_file());
    assert!(r.directory.join("da_ts_final_1.xyz").is_file());
    let report = std::fs::read_to_string(r.directory.join("autots_report.html")).unwrap();
    let band = r.path_stage.as_ref().unwrap().path.as_ref().unwrap();
    let band_dir = band.parent().unwrap().parent().unwrap();
    for name in ["energy_profile.svg", "image_forces.svg", "convergence.svg"] {
        let bytes = std::fs::read(band_dir.join(name)).unwrap();
        let encoded = mopt_rs::cli::autots_static::base64_bytes(&bytes);
        assert!(report.contains(&format!(
            "download=\"{name}\" href=\"data:image/svg+xml;base64,{encoded}\""
        )));
    }
    assert!(
        report.contains("Step 4") && report.contains("class=\"viewer\""),
        "report"
    );
    assert!(
        !report.contains("<script src="),
        "the report loads nothing from outside"
    );
}

#[test]
fn step_1_by_bitss_from_the_reactant_and_the_product() {
    let d = dir("bitss");
    let cfg = d.join("config.json");
    let json = SETTINGS.replacen("{\n", "{\n  \"step1_method\": \"bitss\",\n", 1);
    std::fs::write(&cfg, json).unwrap();
    // The two ion-molecule complexes: Cl- behind CH3Br, and ClCH3 with Br- behind it (the
    // methyl group inverted).
    let input = d.join("sn2.xyz");
    std::fs::write(
        &input,
        "6\n-1 1 reactant\nC 0.000 0.000 0.000\nH -0.360 1.010 0.050\nH -0.360 -0.480 0.880\n\
         H -0.360 -0.530 -0.900\nBr 1.960 0.020 -0.010\nCl -3.300 0.080 0.060\n\
         6\n-1 1 product\nC 0.000 0.000 0.000\nH 0.360 1.010 0.050\nH 0.360 -0.480 0.880\n\
         H 0.360 -0.530 -0.900\nBr 3.450 0.020 -0.010\nCl -1.800 0.080 0.060\n",
    )
    .unwrap();
    let all = [
        input.to_string_lossy().into_owned(),
        "-cfg".to_string(),
        cfg.to_string_lossy().into_owned(),
    ];
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    let p = parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap();
    let c = AutotsConfig::from_args(&p).unwrap();
    let r = run_with(&c, &d, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(r.first_path_method, "BITSS");
    // The first path is the trajectory of BITSS as it is: it starts at the reactant and ends
    // at the product.
    let path = r.afir_path.as_ref().unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let frames = text.lines().filter(|l| l.trim() == "6").count();
    assert!(frames > 2, "{frames} frames");
    let (ccl, cbr) = bonds(path);
    assert!(cbr < 2.2 && ccl > 2.6, "the first frame is the reactant");
    let ts: Vec<_> = r
        .transition_states
        .iter()
        .filter(|t| t.converged && t.imaginary_modes == Some(1))
        .collect();
    assert!(!ts.is_empty(), "{:?}", r.transition_states);
    let report = std::fs::read_to_string(r.directory.join("autots_report.html")).unwrap();
    assert!(report.contains("First path (BITSS)"), "report");
    assert!(report.contains("Energy during BITSS"), "report");
    assert!(!r.directory.join("autots_progress.html").exists());
    let effective = std::fs::read_to_string(r.directory.join("effective_settings.txt")).unwrap();
    for field in [
        "step1_method = BITSS",
        "Step 1 · BITSS (optmain)",
        "reduction: 0.5",
        "coefficient_interval: 100",
        "stop_on_midpoint_gradient: true",
        "Step 2 · NEB (nebmain)",
    ] {
        assert!(effective.contains(field), "{field}");
        assert!(report.contains(field), "missing from HTML: {field}");
    }
}

#[test]
fn step_1_needs_the_afir_settings_and_step_4_the_irc() {
    let d = dir("refused");
    let c = config(&d, SETTINGS, &[]);
    let err = run_with(&c, &d, None, &CalculatorRegistry::native())
        .unwrap_err()
        .to_string();
    assert!(err.contains("-ma"), "{err}");
    // The summary and the report are written also when a step fails or step 4 does not run.
    let report = std::fs::read_to_string(d.join("work").join("autots_report.html")).unwrap();
    // Refused while step 1 was set up: the page names the error.
    assert!(report.contains("Step 1 did not finish"), "report");
    assert!(report.contains("Step 4 did not run"), "report");
    assert!(d.join("work").join("autots_summary.csv").is_file());
    // The report is on by default; "html_report": false turns it off.
    let off = SETTINGS.replace(
        r#""work_dir": "work""#,
        r#""work_dir": "off", "html_report": false"#,
    );
    let c = config(&d, &off, &[]);
    assert!(!c.html_report);
    assert!(run_with(&c, &d, None, &CalculatorRegistry::native()).is_err());
    assert!(d.join("off").join("autots_summary.csv").is_file());
    assert!(!d.join("off").join("autots_report.html").exists());
    let no_irc = SETTINGS.replace(
        r#""intrinsic_reaction_coordinates": ["0.3", "200", "lqa"]"#,
        r#""NSTEP": 10"#,
    );
    let c = config(&d, &no_irc, &["--skip_to_step4"]);
    let err = run_with(&c, &d, None, &CalculatorRegistry::native())
        .unwrap_err()
        .to_string();
    assert!(err.contains("intrinsic_reaction_coordinates"), "{err}");
}

/// Copper with a vacancy in a 2x2x2 cell of the fcc conventional cell (31 atoms), before and
/// after the move of a nearest neighbour into the vacancy, as one xyz file of two frames.
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
    // The vacancy at the origin; the atom at (a/2, a/2, 0) moves into it.
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

#[test]
fn periodic_vacancy_migration_from_the_chain() {
    let d = dir("vacancy");
    std::fs::write(
        d.join("config.json"),
        r#"{
  "work_dir": "work",
  "top_n_candidates": 1,
  "step1_settings": {"othersoft": "eam"},
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
    let r = run_with(&c, &d, None, &CalculatorRegistry::native()).unwrap();
    // The vibrational check of a periodic cell counts the imaginary modes at the zone centre.
    let t = &r.transition_states[0];
    assert!(t.converged && t.imaginary_modes == Some(1), "{t:?}");
    // The ends of the IRC keep the cell and are minima below the transition state; the
    // barrier of the EAM of copper is about 0.53 eV.
    assert_eq!(t.ends.len(), 2, "{t:?}");
    for (file, energy) in &t.ends {
        let text = std::fs::read_to_string(file).unwrap();
        assert!(
            text.lines().nth(1).unwrap().contains("Lattice="),
            "{}",
            file.display()
        );
        let barrier_ev = (t.energy - energy) * Units::HARTREE_TO_EV;
        assert!((0.3..0.8).contains(&barrier_ev), "barrier {barrier_ev} eV");
    }
    // Every frame of the IRC path, combined and per side, carries the cell (31 atoms: one
    // frame is 33 lines, its comment the second).
    let irc = t.irc_directory.as_ref().unwrap();
    for file in [
        irc.join("irc_combined_path.xyz"),
        irc.join("irc_forward").join("irc_structures.xyz"),
        irc.join("irc_backward").join("irc_structures.xyz"),
    ] {
        let text = std::fs::read_to_string(&file).unwrap();
        let comments: Vec<&str> = text.lines().skip(1).step_by(33).collect();
        assert!(!comments.is_empty(), "{}", file.display());
        assert!(
            comments.iter().all(|c| c.contains("Lattice=")),
            "{}: {:?}",
            file.display(),
            &comments[..comments.len().min(3)]
        );
    }
    let report = std::fs::read_to_string(r.directory.join("autots_report.html")).unwrap();
    assert!(report.contains("periodic cell a = 7.230"), "report");
    assert!(
        !report.contains("class=\"warn\""),
        "no warning in a clean run"
    );
}
