//! The masses of a vibrational analysis as an explicit choice (`freqmain -freq_masses`): the
//! frequencies and the thermochemistry with the masses of the most abundant isotopes, the
//! choice named in the reports, and the analyses that weight with the standard atomic weights
//! refused by name; and `optmain -vc -freq`, the vibrations of the optimized structure in the
//! optimized cell with the cell held.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::freqmain::{config_from_cli, run_with};
use mopt_rs::cli::optmain;
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_freqmass_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

fn frequencies(dir: &Path) -> Vec<f64> {
    std::fs::read_to_string(dir.join("normal_modes.txt"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("Freq [cm^-1]"))
        .flat_map(|l| {
            l.split_whitespace()
                .skip(2)
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn the_masses_of_the_analysis_are_chosen_and_named() {
    let dir = scratch("choice");
    let water = file(
        &dir,
        "water.xyz",
        "3\n\nO 0.0 0.0 0.12\nH 0.0 0.76 -0.47\nH 0.0 -0.76 -0.47\n",
    );
    let run = |extra: &[&str]| {
        let mut args = vec![water.as_str(), "-os", "gfnff"];
        args.extend_from_slice(extra);
        let cfg = config_from_cli(&args).unwrap();
        run_with(&cfg, &dir, None, &CalculatorRegistry::native())
            .unwrap()
            .remove(0)
    };
    let standard = run(&[]);
    let isotope = run(&["-freq_masses", "isotope"]);
    let (a, b) = (
        frequencies(&standard.directory),
        frequencies(&isotope.directory),
    );
    assert_eq!((a.len(), b.len()), (3, 3));
    // Lighter isotopes (1H 1.00783 against 1.008, 16O 15.9949 against 15.999): every
    // frequency rises, by about the square root of the mass ratio of hydrogen.
    for (s, i) in a.iter().zip(&b) {
        let ratio = i / s;
        assert!(ratio > 1.0 && ratio < 1.0001, "{s} {i}");
    }
    let text = |run: &mopt_rs::cli::freqmain::FreqmainResult| {
        std::fs::read_to_string(run.directory.join("thermochemistry.txt")).unwrap()
    };
    assert!(text(&standard).contains("Masses: standard atomic weights\n"));
    assert!(text(&isotope).contains("Masses: most abundant isotopes\n"));
    let log = std::fs::read_to_string(isotope.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Masses: most abundant isotopes"));
    // The analyses that weight with the standard atomic weights are refused by name.
    let refused = config_from_cli(&[&water, "-freq_masses", "isotope", "-anharmonic"])
        .err()
        .unwrap()
        .to_string();
    assert!(refused.contains("-anharmonic"), "{refused}");
    assert!(config_from_cli(&[&water, "-freq_masses", "deuterium"]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A 2 x 2 x 2 fcc argon crystal (32 atoms) at 5.40 Angstrom, every atom shifted a little.
fn fcc() -> String {
    let a = 5.40;
    let mut rows = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let n = rows.len() as f64;
                    let shift = [
                        0.04 * (1.3 * n).sin(),
                        0.03 * (0.7 * n).cos(),
                        0.02 * (2.1 * n).sin(),
                    ];
                    let cell = [f64::from(i), f64::from(j), f64::from(k)];
                    rows.push(std::array::from_fn::<f64, 3, _>(|c| {
                        a * (cell[c] + f[c]) + shift[c]
                    }));
                }
            }
        }
    }
    let l = 2.0 * a;
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        rows.len()
    );
    for r in &rows {
        text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0], r[1], r[2]));
    }
    text
}

#[test]
fn freq_after_a_variable_cell_optimization_analyses_the_held_cell() {
    let dir = scratch("vc");
    let crystal = file(&dir, "fcc.xyz", &fcc());
    let run = |extra: &[&str]| {
        let mut args = vec![
            crystal.as_str(),
            "-os",
            "lj",
            "-vc",
            "-opt",
            "rsirfo_bfgs",
            "-ns",
            "400",
            "-freq",
        ];
        args.extend_from_slice(extra);
        let cfg = optmain::config_from_cli(&args).unwrap();
        optmain::run_with(&cfg, &dir, None, &CalculatorRegistry::native())
            .unwrap()
            .remove(0)
    };
    let out = run(&[]);
    assert!(out.converged);
    // The modes at the zone centre: 3N - 3 (the translations projected out).
    let w = frequencies(&out.directory);
    assert_eq!(w.len(), 3 * 32 - 3, "{w:?}");
    let summary = out.summary.unwrap();
    assert_eq!(summary.imaginary_modes, Some(0), "{w:?}");
    let note = std::fs::read_to_string(out.directory.join("frequency_cell.txt")).unwrap();
    assert!(note.contains("cell was held"));
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(log.contains("the cell held"));
    // With a frozen atom (it keeps its fractional coordinates in the run): the vibrations on
    // the surface of the constraint, 3N - 3 without the rigid translations.
    let out = run(&["-fix", "1"]);
    assert!(out.converged);
    let w = frequencies(&out.directory);
    assert_eq!(w.len(), 3 * 32 - 3, "{w:?}");
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(log.contains("constraint surface"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}
