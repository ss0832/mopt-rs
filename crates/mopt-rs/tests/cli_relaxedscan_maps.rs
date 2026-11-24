//! `relaxedscan` on maps and ReSPer coordinates: the Muller-Brown surface as plain variables
//! (a map of both variables with its saddle points optimized, a relaxed curve), the repair of
//! a map whose rows lie on different branches, ReSPer scans of the Muller-Brown surface
//! (relaxed and rigid) and of a molecule.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `relaxedscan` moves its
//! coordinates together and has no ReSPer coordinates).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::relaxedscan::{config_from_cli, run_with};
use mopt_rs::domain::SpinCharge;
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::scan::CandidateKind;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_rscan_maps_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn a_map_of_the_muller_brown_surface_and_its_saddle_points() {
    let dir = scratch("mb");
    let input = file(&dir, "mb.txt", "x -0.5\ny 1.4\n");
    let mut config = config_from_cli(&[
        input.as_str(),
        "-os",
        "mb2d",
        "-scan",
        "var",
        "x",
        "-1.5,1.1",
        "var",
        "y",
        "-0.4,2.0",
        "-grid",
        "-nsample",
        "27,25",
        "-scan_opt",
        "ts_optimize=on",
    ])
    .unwrap();
    config.restart_dir = Some(dir.join("map"));
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(out.points.len(), 27 * 25);
    let saddles: Vec<_> = out
        .candidates
        .iter()
        .filter(|c| c.kind == CandidateKind::Saddle)
        .collect();
    let minima = out
        .candidates
        .iter()
        .filter(|c| c.kind == CandidateKind::Minimum)
        .count();
    assert_eq!(saddles.len(), 2, "{:?}", out.candidates);
    assert_eq!(minima, 3, "{:?}", out.candidates);
    // optmain -order 1 from each candidate goes to a saddle point of the surface.
    assert_eq!(out.optimized.len(), 2);
    for o in &out.optimized {
        assert!(o.converged);
        let text = std::fs::read_to_string(&o.file).unwrap();
        let v = mopt_rs::cli::variables::parse_variables(&text)
            .unwrap()
            .values;
        let nearest = MB_SADDLES
            .iter()
            .map(|s| (s[0] - v[0]).hypot(s[1] - v[1]))
            .fold(f64::INFINITY, f64::min);
        // Within what the default criteria resolve on the surface scaled by 1e-3.
        assert!(nearest < 2e-3, "{v:?}");
    }
    assert!(dir.join("map/relaxedscan_energy_map.png").is_file());
    assert!(dir.join("map/relaxedscan_energy_map.svg").is_file());
    assert!(dir.join("map/ts_optimizations.csv").is_file());

    // A relaxed curve: at every x the lowest y (MultiOptPy's joint form with one coordinate).
    let mut curve = config_from_cli(&[
        input.as_str(),
        "-os",
        "mb2d",
        "-scan",
        "var",
        "1",
        "-0.9,0.7",
        "-nsample",
        "9",
    ])
    .unwrap();
    curve.restart_dir = Some(dir.join("curve"));
    let out = run_with(&curve, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out.points.iter().all(|p| p.converged));
    // The point at x = -0.5 is near minimum A, whose y is 1.44 (the scan starts at y = 1.4).
    let text = std::fs::read_to_string(dir.join("curve/mb_RelaxedScan_3.txt")).unwrap();
    let v = mopt_rs::cli::variables::parse_variables(&text)
        .unwrap()
        .values;
    assert!(
        (v[0] + 0.5).abs() < 1e-10 && (v[1] - MB_MINIMA[0][1]).abs() < 0.05,
        "{v:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A path over the Muller-Brown surface: minimum A, the first saddle point, minimum C, the
/// second saddle point, minimum B, ten points per segment.
fn muller_brown_path() -> Vec<[f64; 2]> {
    let corners = [
        MB_MINIMA[0],
        MB_SADDLES[0],
        MB_MINIMA[2],
        MB_SADDLES[1],
        MB_MINIMA[1],
    ];
    let mut out = Vec::new();
    for w in corners.windows(2) {
        for k in 0..10 {
            let s = f64::from(k) / 10.0;
            out.push([
                w[0][0] + s * (w[1][0] - w[0][0]),
                w[0][1] + s * (w[1][1] - w[0][1]),
            ]);
        }
    }
    out.push(MB_MINIMA[1]);
    out
}

#[test]
fn resper_scans_of_the_muller_brown_surface() {
    use faer::Col;
    use mopt_rs::analysis::resper::ResperMap;
    let dir = scratch("resper_mb");
    let input = file(&dir, "mb.txt", "x -0.5\ny 1.4\n");
    let path = muller_brown_path();
    let text: String = path
        .iter()
        .map(|p| format!("{} {}\n", p[0], p[1]))
        .collect();
    let references = file(&dir, "path.txt", &text);
    let map =
        ResperMap::from_points(path.iter().map(|p| Col::from_fn(2, |i| p[i])).collect()).unwrap();
    // A relaxed scan along PCo1: every point holds its coordinate.
    let (lo, hi) = (
        map.reference_coordinate(0, 0),
        map.reference_coordinate(40, 0),
    );
    let range = format!("{},{}", lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo));
    let mut relaxed = config_from_cli(&[
        input.as_str(),
        "-os",
        "mb2d",
        "-resper",
        references.as_str(),
        "-scan",
        "resper",
        "1",
        range.as_str(),
        "-nsample",
        "9",
    ])
    .unwrap();
    relaxed.restart_dir = Some(dir.join("relaxed"));
    let out = run_with(&relaxed, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out.points.iter().all(|p| p.converged));
    for (k, p) in out.points.iter().enumerate() {
        let text = std::fs::read_to_string(dir.join("relaxed").join(&p.file)).unwrap();
        let v = mopt_rs::cli::variables::parse_variables(&text)
            .unwrap()
            .values;
        let (y, _) = map.project_point(&Col::from_fn(2, |i| v[i])).unwrap();
        let target = lo + 0.1 * (hi - lo) + 0.1 * (hi - lo) * k as f64;
        assert!((y[0] - target).abs() < 1e-8, "{k}: {} {target}", y[0]);
    }
    // A map of both coordinates around reference 20 (minimum C): the rigid scan puts the
    // reference itself at the centre, the relaxed one finds it from its neighbours.
    let (c1, c2) = (
        map.reference_coordinate(20, 0),
        map.reference_coordinate(20, 1),
    );
    let r1 = format!("{},{}", c1 - 0.05, c1 + 0.05);
    let r2 = format!("{},{}", c2 - 0.02, c2 + 0.02);
    let args = |extra: &[&'static str]| {
        let mut v: Vec<String> = [
            input.as_str(),
            "-os",
            "mb2d",
            "-resper",
            references.as_str(),
            "-scan",
            "resper",
            "1",
            r1.as_str(),
            "resper",
            "2",
            r2.as_str(),
            "-grid",
            "-nsample",
            "3",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    let rigid_args = args(&["-rigid"]);
    let mut rigid =
        config_from_cli(&rigid_args.iter().map(String::as_str).collect::<Vec<_>>()).unwrap();
    rigid.restart_dir = Some(dir.join("rigid"));
    let rigid = run_with(&rigid, &dir, None, &CalculatorRegistry::native()).unwrap();
    let plain_args = args(&[]);
    let mut relaxed =
        config_from_cli(&plain_args.iter().map(String::as_str).collect::<Vec<_>>()).unwrap();
    relaxed.restart_dir = Some(dir.join("map"));
    let relaxed = run_with(&relaxed, &dir, None, &CalculatorRegistry::native()).unwrap();
    let centre = |points: &[mopt_rs::workflow::scan::ScanPoint]| {
        points.iter().find(|p| p.point == 4).unwrap().energy
    };
    let at_reference =
        1e-3 * mopt_rs::pes::native::muller_brown::muller_brown(path[20][0], path[20][1]);
    assert!((centre(&rigid.points) - at_reference).abs() < 1e-9);
    assert!((centre(&relaxed.points) - at_reference).abs() < 1e-7);
    assert!(rigid.points.iter().all(|p| p.preimage_error.is_some()));
    assert!(dir.join("map/relaxedscan_energy_map.svg").is_file());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The surface of the repair test: a double well in z at about +-2, tilted by x.
fn tilted_wells(x: f64, y: f64, z: f64) -> f64 {
    (0.25 * z * z - 1.0).powi(2) + 1.25 * x * z + 0.1 * y * y
}

/// The lowest energy of `tilted_wells` over z at (x, y).
fn lowest_well(x: f64, y: f64) -> f64 {
    let mut best = f64::INFINITY;
    let mut z = -4.0;
    while z <= 4.0 {
        best = best.min(tilted_wells(x, y, z));
        z += 1e-4;
    }
    best
}

#[test]
fn the_repair_moves_points_to_the_lower_branch_and_the_tables_follow() {
    // z has two wells at about +-2; x tilts them, so that one disappears for |x| > 0.62.
    // Started at z = 2 and continued in snake order, the rows alternate between the wells
    // (each falls into the other well at its last x), so that neighbouring rows lie on
    // different branches; the repair moves the higher points to the lower well.
    let dir = scratch("jumps");
    let input = file(&dir, "wells.txt", "x -1\ny -1\nz 2\n");
    let mut config = config_from_cli(&[
        input.as_str(),
        "-os",
        "expr:(0.25*q3^2 - 1)^2 + 1.25*q1*q3 + 0.1*q2^2",
        "-scan",
        "var",
        "1",
        "-1,1",
        "var",
        "2",
        "-1,1",
        "-grid",
        "-nsample",
        "5,5",
    ])
    .unwrap();
    config.restart_dir = Some(dir.join("map"));
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(out.points.len(), 25);
    let log = std::fs::read_to_string(dir.join("map/mopt.log")).unwrap();
    assert!(log.contains("on different branches"), "{log}");
    assert!(out.points.iter().any(|p| p.repaired));
    // Every point off x = 0 (where both wells are equal) ends in the lower well.
    let value = |i: usize| -1.0 + 0.5 * i as f64;
    for p in &out.points {
        let (i, j) = (p.point % 5, p.point / 5);
        if i == 2 {
            continue;
        }
        let lowest = lowest_well(value(i), value(j));
        assert!((p.energy - lowest).abs() < 1e-5, "{p:?}: {lowest}");
    }
    // scan_points.csv holds the results after the repair.
    let table = std::fs::read_to_string(dir.join("map/scan_points.csv")).unwrap();
    let mut rows = table.lines();
    let header: Vec<&str> = rows.next().unwrap().split(',').collect();
    let column = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (point, energy, repaired) = (
        column("point"),
        column("energy_hartree"),
        column("repaired"),
    );
    let mut flagged = 0;
    for row in rows {
        let cells: Vec<&str> = row.split(',').collect();
        let k: usize = cells[point].parse().unwrap();
        let p = out.points.iter().find(|p| p.point + 1 == k).unwrap();
        let e: f64 = cells[energy].parse().unwrap();
        assert!((e - p.energy).abs() < 1e-11, "{row}");
        assert_eq!(cells[repaired] == "true", p.repaired, "{row}");
        flagged += usize::from(p.repaired);
    }
    assert!(flagged > 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_resper_scan_of_a_molecule_holds_its_coordinate() {
    use mopt_rs::analysis::resper::ResperMap;
    let dir = scratch("resper_hcn");
    // Hydrogen going round the CN bond from the carbon to the nitrogen.
    let mut frames = String::new();
    for k in 0..13 {
        let theta = std::f64::consts::PI * (1.0 - f64::from(k) / 12.0);
        frames.push_str(&format!(
            "3\nframe {k}\nC 0.0 0.0 0.0\nN 1.16 0.0 0.0\nH {:.8} {:.8} 0.0\n",
            0.58 + 1.62 * theta.cos(),
            1.62 * theta.sin()
        ));
    }
    let references = file(&dir, "path.xyz", &frames);
    let input = file(
        &dir,
        "hcn.xyz",
        "3\nHCN\nC 0.0 0.0 0.0\nN 1.16 0.0 0.0\nH -1.04 0.0 0.0\n",
    );
    let trajectory =
        mopt_rs::io::xyz::read_xyz_trajectory_file(&references, SpinCharge::default(), None)
            .unwrap();
    let map = ResperMap::from_structures(
        trajectory.frames.iter().map(|f| f.system.clone()).collect(),
        &[],
        true,
    )
    .unwrap();
    let (lo, hi) = (
        map.reference_coordinate(2, 0),
        map.reference_coordinate(10, 0),
    );
    let range = format!("{lo},{hi}");
    let mut config = config_from_cli(&[
        input.as_str(),
        "-os",
        "gfn0xtb",
        "-resper",
        references.as_str(),
        "-scan",
        "resper",
        "1",
        range.as_str(),
        "-nsample",
        "4",
    ])
    .unwrap();
    config.restart_dir = Some(dir.join("scan"));
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(out.points.len(), 4);
    let mut largest = 0.0_f64;
    for (k, p) in out.points.iter().enumerate() {
        assert!(p.converged, "point {k}");
        let s = read_xyz_file(dir.join("scan").join(&p.file), SpinCharge::default(), None).unwrap();
        let (y, _) = map.project_structure(&s).unwrap();
        let target = lo + (hi - lo) * k as f64 / 3.0;
        largest = largest.max((y[0] - target).abs());
        // A ReSPer coordinate is computed through a superposition (noise of about 1e-9): the
        // bound here is that, with the 10 decimals of the file, not the 1e-10 of the other
        // constraints.
        assert!((y[0] - target).abs() < 1e-6, "{k}: {} {target}", y[0]);
        // Where a restore left it above 1e-10, the log of the point says so, with the bound.
        let log = std::fs::read_to_string(
            dir.join("scan")
                .join(format!("point_{:03}", k + 1))
                .join("mopt.log"),
        )
        .unwrap_or_default();
        for line in log.lines().filter(|l| l.contains("Warning: constraint")) {
            assert!(
                line.contains("ReSPer PCo1") && line.contains("1e-10"),
                "{line}"
            );
        }
    }
    eprintln!("ReSPer scan: largest |PCo1 - target| from the files {largest:.2e}");
    let _ = std::fs::remove_dir_all(&dir);
}
