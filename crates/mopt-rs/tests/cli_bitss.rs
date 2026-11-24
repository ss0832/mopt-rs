//! `optmain INPUT -mf bitss REFERENCE`: binary-image transition state search between two
//! structures.
//!
//! MultiOptPy: the BITSS branch of `optimization.py:ModelFunctionHandler`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::bitss::parse_settings;
use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::domain::SpinCharge;
use mopt_rs::domain::Units;
use mopt_rs::io::xyz::read_xyz_file;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_bitss_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn xyz(dir: &Path, name: &str, atoms: &[(&str, [f64; 3])]) -> String {
    let mut text = format!("{}\n\n", atoms.len());
    for (element, p) in atoms {
        text.push_str(&format!("{element} {} {} {}\n", p[0], p[1], p[2]));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// Positions in Angstrom of a structure file.
fn positions(path: &Path) -> Vec<[f64; 3]> {
    let sys = read_xyz_file(path, SpinCharge::default(), None).unwrap();
    sys.atoms()
        .map(|a| sys.position(a).value().map(|v| v * Units::BOHR_TO_ANGSTROM))
        .collect()
}

// Four copper atoms in a rhombus (two triangles with a common edge) and a fifth above it.
const SPACING: f64 = 2.55;
const HEIGHT: f64 = 2.08;

fn rhombus() -> [[f64; 3]; 4] {
    let rise = SPACING * 0.75_f64.sqrt();
    [
        [0.0, 0.0, 0.0],
        [SPACING, 0.0, 0.0],
        [0.5 * SPACING, rise, 0.0],
        [1.5 * SPACING, rise, 0.0],
    ]
}

fn with_adatom(adatom: [f64; 3]) -> Vec<(&'static str, [f64; 3])> {
    rhombus()
        .into_iter()
        .chain([adatom])
        .map(|p| ("Cu", p))
        .collect()
}

#[test]
fn an_adatom_crosses_the_edge_between_two_hollow_sites() {
    let dir = scratch("adatom");
    let base = rhombus();
    let centre = |k: [usize; 3]| -> [f64; 3] {
        [
            k.iter().map(|&i| base[i][0]).sum::<f64>() / 3.0,
            k.iter().map(|&i| base[i][1]).sum::<f64>() / 3.0,
            HEIGHT,
        ]
    };
    // The first image is off its hollow site; the search does not need minima.
    let first = centre([0, 1, 2]);
    let start = xyz(
        &dir,
        "adatom.xyz",
        &with_adatom([first[0] - 0.25, first[1] - 0.15, first[2] + 0.2]),
    );
    let reference = xyz(&dir, "other.xyz", &with_adatom(centre([1, 2, 3])));
    let cfg = config_from_cli(&[
        &start, "-os", "emt", "-opt", "lbfgs", "-fix", "1-4", "-mf", "bitss", &reference, "-ns",
        "2000",
    ])
    .unwrap();
    let out = run(&cfg, &dir, None).unwrap();
    assert!(out[0].converged && !out[0].interrupted);
    let run_dir = &out[0].directory;
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    for needle in [
        "Mode: Model Function Optimization (bitss)",
        "Reference: S. J. Avis, J. R. Panter, H. Kusumaatmaja",
        "BITSS Mode detected: Expanding state to 2N atoms.",
        "BITSS stage 1: separation asked for",
        "BITSS stage 2: separation asked for",
        "BITSS coefficients: barrier estimate",
        "BITSS ended: the forces at the midpoint of the images are below the thresholds.",
        "Transition state estimate file path:",
    ] {
        assert!(log.contains(needle), "{needle}\n{log}");
    }
    // The estimate: above the middle of the common edge (the two-fold axis of the rhombus).
    assert_eq!(out[0].optimized, run_dir.join("adatom_ts_guess.xyz"));
    let guess = positions(&out[0].optimized);
    let edge = [
        0.5 * (base[1][0] + base[2][0]),
        0.5 * (base[1][1] + base[2][1]),
    ];
    let off = (guess[4][0] - edge[0]).hypot(guess[4][1] - edge[1]);
    assert!(off < 0.02, "adatom at {:?}, edge at {edge:?}", guess[4]);
    assert!(guess[4][2] > 1.5 && guess[4][2] < 2.6);
    // The fixed atoms stayed, in the estimate and in both images.
    for name in [
        "adatom_ts_guess.xyz",
        "adatom_bitss_image_1.xyz",
        "adatom_bitss_image_2.xyz",
    ] {
        let p = positions(&run_dir.join(name));
        for (found, expected) in p.iter().zip(&base) {
            for k in 0..3 {
                assert!((found[k] - expected[k]).abs() < 1e-6, "{name}");
            }
        }
    }
    // MultiOptPy's trajectory: frames of one image (5 atoms), the first image forward and
    // then the second backward, so the last frame is the reference.
    let trajectory = std::fs::read_to_string(run_dir.join("adatom_traj.xyz")).unwrap();
    let lines: Vec<&str> = trajectory.lines().collect();
    assert_eq!(lines[0].trim(), "5");
    assert_eq!(lines[1], "BITSS_Step 0");
    assert_eq!(lines.len() % 7, 0);
    let frames = lines.len() / 7;
    assert!(frames >= 4 && frames.is_multiple_of(2), "{frames}");
    let last: Vec<f64> = lines[lines.len() - 1]
        .split_whitespace()
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect();
    let other = centre([1, 2, 3]);
    for k in 0..3 {
        assert!((last[k] - other[k]).abs() < 1e-6, "{last:?}");
    }
    // While the search ran, each image had a trajectory of its own; they are the one file
    // now. Every iteration shows the two images under their labels.
    assert!(!run_dir.join("adatom_image_1_traj.xyz").exists());
    assert!(!run_dir.join("adatom_image_2_traj.xyz").exists());
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    let iterations = log.matches("# ITR. ").count();
    assert!(iterations >= 2, "{iterations}");
    assert!(log.matches("\nState1\n").count() > iterations, "{log}");
    assert_eq!(
        log.matches("\nState1\n").count(),
        log.matches("\nState2\n").count()
    );
    // The pair as one structure of ten atoms in the usual file.
    let pair = positions(&run_dir.join("adatom_optimized.xyz"));
    assert_eq!(pair.len(), 10);
}

#[test]
fn ammonia_inverts_through_the_plane_of_its_hydrogens() {
    let dir = scratch("ammonia");
    let hydrogens = [
        [0.94, 0.0, -0.27],
        [-0.47, 0.814, -0.27],
        [-0.47, -0.814, -0.27],
    ];
    let start = xyz(
        &dir,
        "nh3.xyz",
        &[
            ("N", [0.0, 0.0, 0.11]),
            ("H", hydrogens[0]),
            ("H", hydrogens[1]),
            ("H", hydrogens[2]),
        ],
    );
    // The other pyramid, a little distorted and moved as a whole: the reference is laid
    // onto the input before the search.
    let shift = [0.7, -0.4, 0.3];
    let moved = |p: [f64; 3]| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
    let reference = xyz(
        &dir,
        "inverted.xyz",
        &[
            ("N", moved([0.0, 0.0, -0.65])),
            ("H", moved([0.97, 0.03, -0.27])),
            ("H", moved(hydrogens[1])),
            ("H", moved(hydrogens[2])),
        ],
    );
    let cfg = config_from_cli(&[
        &start,
        "-os",
        "gfnff",
        "-opt",
        "rsirfo_bfgs",
        "-mf",
        "bitss",
        &reference,
        "-ns",
        "2000",
    ])
    .unwrap();
    let out = run(&cfg, &dir, None).unwrap();
    assert!(out[0].converged);
    // Planar: the nitrogen lies in the plane of the three hydrogens.
    let p = positions(&out[0].optimized);
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (u, v) = (sub(p[2], p[1]), sub(p[3], p[1]));
    let normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let length = normal.iter().map(|c| c * c).sum::<f64>().sqrt();
    let w = sub(p[0], p[1]);
    let height = (w[0] * normal[0] + w[1] * normal[1] + w[2] * normal[2]) / length;
    assert!(
        height.abs() < 0.02,
        "nitrogen {height} Angstrom off the plane"
    );
    // The two images ended on either side of it.
    let side = |name: &str| {
        let q = positions(&out[0].directory.join(name));
        let (u, v) = (sub(q[2], q[1]), sub(q[3], q[1]));
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let w = sub(q[0], q[1]);
        w[0] * n[0] + w[1] * n[1] + w[2] * n[2]
    };
    assert!(side("nh3_bitss_image_1.xyz") * side("nh3_bitss_image_2.xyz") < 0.0);
}

#[test]
fn settings_and_wrong_input() {
    let s = parse_settings(&[
        "f=0.25".to_string(),
        "beta=0.02".to_string(),
        "points=5".to_string(),
        "step_share=none".to_string(),
        "midpoint_shift=0.1".to_string(),
        "midpoint_gradient=false".to_string(),
    ])
    .unwrap();
    assert!((s.reduction - 0.25).abs() < 1e-15 && (s.beta - 0.02).abs() < 1e-15);
    assert_eq!(s.interpolation_points, 5);
    assert_eq!(s.step_share, None);
    assert_eq!(s.midpoint_shift, Some(0.1));
    assert!(!s.stop_on_midpoint_gradient);
    assert!(parse_settings(&["gamma=1".to_string()]).is_err());
    assert!(parse_settings(&["f".to_string()]).is_err());

    let dir = scratch("errors");
    let start = xyz(
        &dir,
        "a.xyz",
        &[("Ar", [0.0, 0.0, 0.0]), ("Ar", [3.8, 0.0, 0.0])],
    );
    let other = xyz(
        &dir,
        "b.xyz",
        &[("Ar", [0.0, 0.0, 0.0]), ("Ne", [3.8, 0.0, 0.0])],
    );
    // No reference, another element, a constraint BITSS does not take.
    let no_reference = config_from_cli(&[&start, "-os", "lj", "-mf", "bitss"]).unwrap();
    assert!(run(&no_reference, &dir, None).is_err());
    let other_atoms = config_from_cli(&[&start, "-os", "lj", "-mf", "bitss", &other]).unwrap();
    assert!(run(&other_atoms, &dir, None).is_err());
}
