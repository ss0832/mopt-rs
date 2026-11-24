//! `optmain -cmds -pca -km`: maps of the structures of a run and the dynamic modes of its
//! iterations.
//!
//! MultiOptPy: `PESAnalyzer/cmds_analysis.py`, `pca_analysis.py`, `koopman_analysis.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::optmain::{config_from_cli, run};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_pathmaps_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Five argon atoms away from their bipyramid.
fn cluster(dir: &Path) -> String {
    let atoms = [
        [0.0, 0.0, 0.0],
        [4.1, 0.2, 0.0],
        [1.9, 3.6, 0.3],
        [2.2, 1.0, 3.5],
        [2.0, 1.4, -3.9],
    ];
    let mut text = String::from("5\n\n");
    for p in atoms {
        text.push_str(&format!("Ar {} {} {}\n", p[0], p[1], p[2]));
    }
    let path = dir.join("ar5.xyz");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn rows(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter_map(|line| {
            let cells: Option<Vec<f64>> = line
                .split(',')
                .map(|c| c.trim().parse::<f64>().ok())
                .collect();
            cells.filter(|c| c.len() >= 3)
        })
        .collect()
}

#[test]
fn maps_and_modes_of_a_run() {
    let dir = scratch("run");
    let input = cluster(&dir);
    let cfg = config_from_cli(&[
        &input, "-os", "lj", "-opt", "fire", "-ns", "400", "-cmds", "-pca", "-km",
    ])
    .unwrap();
    let out = run(&cfg, &dir, None).unwrap();
    assert!(out[0].converged);
    let run_dir = &out[0].directory;
    let iterations = out[0].summary.as_ref().unwrap().iterations;
    assert!(iterations > 15, "{iterations}");
    let read = |name: &str| std::fs::read_to_string(run_dir.join(name)).unwrap();

    // Multidimensional scaling: the head of MultiOptPy's file and a row per structure.
    let cmds = read("cmds_analysis_result.csv");
    let lines: Vec<&str> = cmds.lines().collect();
    assert!(lines[0].starts_with("dimensional reproducibility:,"));
    assert!(lines[1].starts_with("Percentage contribution 1:,"));
    assert!(lines[2].starts_with("Percentage contribution 2:,"));
    assert!(lines[3].starts_with("itr.,   PCo1,     PCo2,"));
    let share = |line: &str| line.split(',').nth(1).unwrap().parse::<f64>().unwrap();
    let reproducibility = share(lines[0]);
    assert!(reproducibility > 0.5 && reproducibility <= 1.0 + 1e-12);
    assert!(share(lines[1]) >= share(lines[2]) && share(lines[1]) <= 1.0);
    let points = rows(&cmds);
    assert_eq!(points.len(), iterations);
    // The first structure lies on the negative side of the first coordinate, the energy
    // is counted from it, and the optimization lowers it.
    assert!(points[0][1] < 0.0 && points[0][3] == 0.0);
    assert!(points[iterations - 1][3] < 0.0);
    // Neighbouring structures are closer on the map than the two ends.
    let apart =
        |a: usize, b: usize| (points[a][1] - points[b][1]).hypot(points[a][2] - points[b][2]);
    assert!(apart(0, 1) < apart(0, iterations - 1));

    // Principal components: three sets of variables, each with its log, its table, and
    // its figures.
    for suffix in ["", "_redundant", "_z_matrix"] {
        let log = read(&format!("pca_analysis_result{suffix}.log"));
        assert!(log.contains("PCA analysis for approx. reaction path"));
        assert!(log.contains("basis 0\ncontribution: "));
        let table = rows(&read(&format!("pca_analysis_result{suffix}.csv")));
        assert_eq!(table.len(), iterations);
    }
    assert!(read("pca_analysis_result.log").contains("1 Ar - x : "));
    assert!(read("pca_analysis_result_redundant.log").contains("1 Ar - 2 Ar : "));
    assert!(read("pca_analysis_result_z_matrix.log").contains("4 Ar - 3 Ar - 2 Ar - 1 Ar : "));
    for figure in [
        "cmds_result_visualization_energy",
        "cmds_result_visualization_bias_energy",
        "pca_result_visualization_energy",
        "pca_result_visualization_bias_energy",
        "pca_result_visualization_redundant_energy",
        "pca_result_visualization_z_matrix_bias_energy",
    ] {
        let png = std::fs::read(run_dir.join(format!("{figure}.png"))).unwrap();
        assert_eq!(&png[1..4], b"PNG", "{figure}");
        let svg = read(&format!("{figure}.svg"));
        assert!(svg.contains("(kcal/mol)") && svg.matches("<circle").count() == iterations);
    }

    // Dynamic modes: every iteration is logged; the analysis starts when eleven are there.
    let koopman = run_dir.join("koopman_analysis");
    let coordinates = std::fs::read_to_string(koopman.join("coordinates.csv")).unwrap();
    assert!(coordinates.starts_with("iteration,x1,y1,z1,"));
    assert!(coordinates
        .lines()
        .next()
        .unwrap()
        .ends_with("grad_x5,grad_y5,grad_z5"));
    assert_eq!(coordinates.lines().count(), iterations + 1);
    let eigs = std::fs::read_to_string(koopman.join("koopman_eigs.csv")).unwrap();
    assert!(eigs.starts_with("iteration,rank,mode_0_real,mode_0_imag,mode_0_norm,"));
    assert_eq!(eigs.lines().count(), iterations - 10 + 1);
    let last: Vec<&str> = eigs.lines().last().unwrap().split(',').collect();
    assert_eq!(last[0].parse::<usize>().unwrap(), iterations - 1);
    let rank: usize = last[1].parse().unwrap();
    assert!((1..=10).contains(&rank));
    // The largest modulus first, and it is the root of the sum of the squares.
    let (re, im, norm): (f64, f64, f64) = (
        last[2].parse().unwrap(),
        last[3].parse().unwrap(),
        last[4].parse().unwrap(),
    );
    assert!((re.hypot(im) - norm).abs() < 1e-9);
    if rank > 1 {
        assert!(norm >= last[7].parse::<f64>().unwrap() - 1e-12);
    }
    let modes = std::fs::read_to_string(koopman.join("koopman_modes.log")).unwrap();
    assert!(modes.contains(&format!("Iteration {}:", iterations - 1)));
    assert!(modes.contains("Eigenvalue 0: Real=") && modes.contains("Mode vector (imag part):"));
    let animation = std::fs::read_to_string(koopman.join("koopman_mode_anim_0.xyz")).unwrap();
    assert_eq!(animation.lines().count(), 10 * 7);
    assert!(animation.contains("Mode 0 Frame 9"));
}

#[test]
fn a_run_of_two_structures_is_too_short_for_a_map() {
    let dir = scratch("short");
    let input = cluster(&dir);
    let cfg = config_from_cli(&[
        &input, "-os", "lj", "-opt", "fire", "-ns", "2", "-cmds", "-pca",
    ])
    .unwrap();
    let out = run(&cfg, &dir, None).unwrap();
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("There is not enough data to perform CMDS analysis..."));
    assert!(log.contains("There is not enough data to perform PCA analysis..."));
    assert!(!out[0].directory.join("cmds_analysis_result.csv").exists());
}
