//! `freqmain PATH.xyz -nro -roet`: natural reaction orbitals and orbital energies along
//! the frames of a path, with the orbitals of the built-in GFN0-xTB.
//!
//! MultiOptPy: `WFAnalyzer/MO_analysis.py:NROAnalysis`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::freqmain::{config_from_cli, run_with};
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_orbitals_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Ammonia with the nitrogen at the heights `heights` (Angstrom) above the plane of the
/// hydrogens: frames of an inversion.
fn inversion(dir: &Path, name: &str, heights: &[f64]) -> String {
    let mut text = String::new();
    for z in heights {
        text.push_str(&format!(
            "4\n\nN 0.0 0.0 {z}\nH 0.94 0.0 0.0\nH -0.47 0.814 0.0\nH -0.47 -0.814 0.0\n"
        ));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// The rows of a table as numbers, the spin column left out.
fn table(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .skip(1)
        .map(|line| {
            line.split(',')
                .filter_map(|cell| cell.parse::<f64>().ok())
                .collect()
        })
        .collect()
}

#[test]
fn reaction_orbitals_and_orbital_energies_along_an_inversion() {
    let dir = scratch("inversion");
    let heights: Vec<f64> = (0..9).map(|k| 0.4 - 0.1 * k as f64).collect();
    let input = inversion(&dir, "nh3.xyz", &heights);
    let cfg = config_from_cli(&[&input, "-os", "gfn0xtb", "-nro", "-roet"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let run_dir = &out[0].directory;
    let found = out[0].orbitals.as_ref().unwrap();
    assert!(out[0].analysis.is_none() && out[0].profile.is_none());
    assert_eq!(found.arc_lengths.len(), 9);
    let read = |name: &str| std::fs::read_to_string(run_dir.join(name)).unwrap();

    // Natural reaction orbitals: one closed-shell set per frame, the same on both sides
    // of the plane.
    let nro = read("nro.csv");
    assert!(nro.starts_with("frame,arc_length,spin,sum_of_squares,lambda_1,share_1,"));
    assert!(nro.lines().nth(1).unwrap().contains(",closed_shell,"));
    let rows = table(&nro);
    assert_eq!(rows.len(), 9);
    for k in 0..9 {
        let (here, mirror) = (&rows[k], &rows[8 - k]);
        // frame, arc length, sum of squares, first singular value, its share, ...
        assert!(here[2] > 0.0 && here[3] > 0.0);
        assert!((here[2] - mirror[2]).abs() < 1e-6 * here[2], "frame {k}");
        assert!((here[3] * here[3] / here[2] - here[4]).abs() < 1e-5);
        // The occupied orbitals are followed safely at this step.
        assert!(*here.last().unwrap() > 0.999);
    }
    let sets = found.reaction_orbitals.as_ref().unwrap();
    assert_eq!(sets[4].len(), 1);
    // Four occupied and six empty orbitals: four pairs.
    assert_eq!(sets[4][0].orbitals.singular_values.len(), 4);
    let pairs = read("nro_pairs.txt");
    assert!(pairs.contains("frame 4 (closed shell), sum of squares"));
    assert!(pairs.contains("  pair 1: singular value") && pairs.contains("    occupied: "));
    assert!(std::fs::read(run_dir.join("nro_sum_of_squares.png")).unwrap()[1..4] == *b"PNG");
    let peak = std::fs::read_dir(run_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .find(|name| name.starts_with("nro_orbitals_frame") && name.ends_with("_closed_shell.csv"))
        .unwrap();
    let orbitals = read(&peak);
    assert!(orbitals.starts_with("atomic_orbital,occupied_1,empty_1,"));
    // Ten atomic orbitals: 2s and 2p of nitrogen, 1s and 2s of each hydrogen.
    assert_eq!(orbitals.lines().count(), 1 + 10);

    // Orbital energies: followed through the plane, the same at mirror frames; the two
    // ends are mirror images, so no orbital has changed between them.
    let energies = table(&read("orbital_energies.csv"));
    assert_eq!(energies.len(), 9);
    assert_eq!(energies[0].len(), 2 + 10);
    for k in 0..9 {
        let pairs = energies[k].iter().zip(&energies[8 - k]);
        for (i, (here, mirror)) in pairs.enumerate().skip(2) {
            assert!((here - mirror).abs() < 1e-8, "{k} {i}");
        }
    }
    let report = read("reactive_orbitals.txt");
    assert!(report.contains("occupied reactive orbital (the most stabilized occupied one): none"));
    assert!(report.contains("unoccupied reactive orbital (the empty one that changes most): none"));
    // The derivatives: antisymmetric slopes, symmetric curvatures.
    let slopes = table(&read("1st_derivative_orbital_energy_plot.csv"));
    let curvatures = table(&read("2nd_derivative_orbital_energy_plot.csv"));
    for i in 2..12 {
        assert!((slopes[2][i] + slopes[6][i]).abs() < 1e-6, "{i}");
        assert!((curvatures[2][i] - curvatures[6][i]).abs() < 1e-6, "{i}");
        assert!(slopes[4][i].abs() < 1e-6);
    }
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(log.contains("Reference: S. Ebisawa, M. Hasebe, T. Tsutsumi, T. Tsuneda, T. Taketsugu"));
    assert!(log.contains("### Orbital energies along the path ###"));
}

#[test]
fn half_an_inversion_has_reactive_orbitals() {
    let dir = scratch("half");
    // From the pyramid to the plane.
    let heights: Vec<f64> = (0..5).map(|k| 0.4 - 0.1 * k as f64).collect();
    let input = inversion(&dir, "half.xyz", &heights);
    let cfg = config_from_cli(&[&input, "-os", "gfn0xtb", "-roet"]).unwrap();
    let out = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap();
    let run_dir = &out[0].directory;
    // Only what was asked for is written.
    assert!(!run_dir.join("nro.csv").exists());
    let found = out[0].orbitals.as_ref().unwrap();
    assert!(found.reaction_orbitals.is_none());
    let set = &found.energies.as_ref().unwrap()[0];
    assert_eq!(set.energies.len(), 5);
    assert!(set.weakest_overlaps.iter().all(|o| *o > 0.9));
    let report = std::fs::read_to_string(run_dir.join("reactive_orbitals.txt")).unwrap();
    let line = report
        .lines()
        .find(|l| l.starts_with("occupied reactive orbital"))
        .unwrap();
    assert!(
        line.contains(": orbital ") && line.contains("(change -"),
        "{line}"
    );
    assert!(
        report.contains("unoccupied reactive orbital (the empty one that changes most): orbital ")
    );
}

#[test]
fn what_cannot_be_analyzed() {
    let dir = scratch("errors");
    // A force field has no orbitals.
    let argon = dir.join("ar2.xyz");
    std::fs::write(
        &argon,
        "2\n\nAr 0 0 0\nAr 3.8 0 0\n2\n\nAr 0 0 0\nAr 3.9 0 0\n",
    )
    .unwrap();
    let input = argon.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[&input, "-os", "lj", "-nro"]).unwrap();
    let error = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(error.to_string().contains("molecular orbitals"), "{error}");
    // One structure is no path.
    let one = inversion(&dir, "one.xyz", &[0.38]);
    let cfg = config_from_cli(&[&one, "-os", "gfn0xtb", "-roet"]).unwrap();
    let error = run_with(&cfg, &dir, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(error.to_string().contains("two at least"), "{error}");
}
