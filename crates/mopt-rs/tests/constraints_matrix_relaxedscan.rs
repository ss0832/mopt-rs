//! The user's constraints in `relaxedscan` in every kind of run: {frozen atoms, a distance, a
//! centroid distance (`fbond`), an angle, a rigid group (`-gfix`), a user-defined function} x
//! {a molecule, a periodic fixed cell, a periodic variable cell (`-vc`), ONIOM}. Every point
//! starts from its neighbour, yet every point the scan writes holds the user's constraints at
//! the values of the input (they are resolved once, at the input) and the scanned coordinate at
//! its value; frozen atoms stay where they are (their fractional coordinates with `-vc`). What
//! a variable cell cannot hold is refused with the reason, and each point's `settings.json`
//! says what it holds.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use matrix_support::{check, kinds, read, scratch, write, FROM_FILES, FROZEN_IN_FILES};
use mopt_rs::cli::constraints::projection_constraint;
use mopt_rs::cli::relaxedscan::{config_from_cli, run_with};
use mopt_rs::pes::registry::CalculatorRegistry;

const BUTANE: &str = "14\nn-butane all-anti\nC 0.000000 0.000000 0.000000\n\
C 1.268427 0.855565 0.000000\nC 2.536855 0.000000 0.000000\nC 3.805282 0.855565 0.000000\n\
H -0.875864 0.648816 0.000000\nH -0.013866 -0.629150 0.889988\nH -0.013866 -0.629150 -0.889988\n\
H 1.268427 1.484886 0.889975\nH 1.268427 1.484886 -0.889975\nH 2.536855 -0.629321 -0.889975\n\
H 2.536855 -0.629321 0.889975\nH 4.681147 0.206750 0.000000\nH 3.819149 1.484715 0.889988\n\
H 3.819149 1.484715 -0.889988\n";

const AR7: &str = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 1.9500 0.0000 0.0000\nAr 0.6026 1.8546 0.0300\nAr -1.5776 1.1462 0.0700\n\
Ar -1.5776 -1.1462 0.1100\nAr 0.6026 -1.8546 0.1500\nAr 0.0000 0.0000 2.1000\n\
Ar 0.2000 0.0000 -2.0000\n";

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

/// A 2 x 2 x 2 fcc argon crystal (32 atoms), every atom shifted a little.
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
                        0.07 * (1.3 * n).sin(),
                        0.06 * (0.7 * n).cos(),
                        0.05 * (2.1 * n).sin(),
                    ];
                    let cell = [f64::from(i), f64::from(j), f64::from(k)];
                    rows.push(std::array::from_fn::<f64, 3, _>(|c| {
                        a * (cell[c] + f[c]) + shift[c]
                    }));
                }
            }
        }
    }
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        rows.len(),
        l = 2.0 * a
    );
    for r in &rows {
        text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0], r[1], r[2]));
    }
    text
}

/// One context: the scan `scan` (KIND ATOMS RANGE) of the structure `text` with `calc`.
#[allow(clippy::too_many_arguments)]
fn context(
    name: &str,
    text: &str,
    calc: &[&str],
    scan: [&str; 3],
    atoms: [usize; 4],
    fragments: (&[usize], &[usize]),
    variable: bool,
) {
    let dir = scratch("rsmatrix", name);
    let file = write(&dir, "start.xyz", text);
    let input = read(&file);
    for (kind, options) in kinds(atoms, fragments) {
        let cell = format!("{name} x {kind}");
        let mut args: Vec<String> = vec![file.to_string_lossy().into_owned()];
        args.extend(calc.iter().map(|s| s.to_string()));
        args.push("-scan".into());
        args.extend(scan.iter().map(|s| s.to_string()));
        for s in [
            "-nsample",
            "3",
            "-scan_opt",
            "figures=off",
            "html=off",
            "candidates=off",
        ] {
            args.push(s.to_string());
        }
        if variable {
            args.push("-vc".into());
        }
        args.extend(options.iter().cloned());
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let mut config = config_from_cli(&argv).unwrap();
        config.restart_dir = Some(dir.join(kind));
        let constraints = config.optmain.settings.constraints.clone();
        let out = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{cell}: {e}"));
        let points: Vec<_> = out
            .points
            .iter()
            .filter(|p| !p.skipped)
            .map(|p| (p.point, read(&dir.join(kind).join(&p.file))))
            .collect();
        assert_eq!(points.len(), 3, "{cell}");
        let structures: Vec<_> = points.iter().map(|(_, s)| s.clone()).collect();
        let bound = if variable {
            matrix_support::FROM_VARIABLE_CELL_FILES
        } else {
            FROM_FILES
        };
        check(
            &cell,
            &structures,
            &constraints,
            &input,
            bound,
            FROZEN_IN_FILES,
            variable,
        );
        // The scanned coordinate at its value at every point.
        let (lo, hi) = scan[2].split_once(',').unwrap();
        let (lo, hi): (f64, f64) = (lo.parse().unwrap(), hi.parse().unwrap());
        for (k, s) in &points {
            let value = lo + (hi - lo) * *k as f64 / 2.0;
            let set = mopt_rs::pes::constraint::ConstraintSet::new(
                &[projection_constraint(scan[0], scan[1], Some(value)).unwrap()],
                s,
                None,
            )
            .unwrap();
            let r = set.residuals(s).unwrap()[0];
            assert!(r.abs() < bound, "{cell} point {k}: scanned {r}");
        }
        // Every point's settings hold the user's constraint and the scanned coordinate.
        let name0 = format!("point_{:03}", 1);
        let settings =
            std::fs::read_to_string(dir.join(kind).join(&name0).join("settings.json")).unwrap();
        assert!(settings.contains(scan[1]), "{cell}: {settings}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn user_constraints_in_a_scan_of_a_molecule() {
    context(
        "molecule",
        BUTANE,
        &["-os", "gfnff"],
        ["bond", "1,5", "1.06,1.14"],
        [2, 3, 4, 1],
        (&[2, 8, 9], &[3, 10, 11]),
        false,
    );
}

#[test]
fn user_constraints_in_a_scan_in_a_periodic_cell() {
    context(
        "periodic",
        AR7,
        &["-os", "lj"],
        ["bond", "6,7", "4.00,4.20"],
        [1, 2, 3, 4],
        (&[1, 2], &[3, 4]),
        false,
    );
}

#[test]
fn user_constraints_in_a_scan_in_a_variable_cell() {
    context(
        "variable_cell",
        &fcc(),
        &["-os", "lj", "-ns", "400"],
        ["bond", "1,2", "3.75,3.90"],
        [3, 4, 5, 6],
        (&[7, 8], &[9, 10]),
        true,
    );
    // What a variable cell cannot hold is refused with the reason.
    for (extra, expected) in [
        (vec!["-scan", "x", "1", "0.0,0.2"], "Cartesian component"),
        (
            vec!["-scan", "bond", "1,2", "3.75,3.90", "-pc", "rot", "1-4"],
            "orientation",
        ),
    ] {
        let mut args = vec!["fcc.xyz", "-os", "lj", "-vc"];
        args.extend(extra.iter().copied());
        let err = config_from_cli(&args).unwrap_err().to_string();
        assert!(
            err.contains("-vc") && err.contains(expected),
            "{extra:?}: {err}"
        );
    }
}

#[test]
fn user_constraints_in_a_scan_with_oniom() {
    context(
        "oniom",
        ETHANOL,
        &["-os", "gfnff", "-oniom", "2-4,8,9", "auto", "gfnff"],
        ["bond", "2,8", "1.08,1.14"],
        [2, 3, 4, 9],
        (&[2, 9], &[3, 4]),
        false,
    );
}

/// The parameters of the cell as scanned coordinates (`-scan cell NAME START,END` with `-vc`):
/// every point holds its value to the guarantee (read back from its file), and the energy-volume
/// curve with the atoms and the shape relaxed at every volume has its minimum at the volume of
/// the unconstrained variable-cell optimum.
#[test]
fn the_parameters_of_the_cell_as_scanned_coordinates() {
    use mopt_rs::cli::constraints::user_constraint;
    use mopt_rs::domain::Units;
    let dir = scratch("rsmatrix", "cell");
    let file = write(&dir, "start.xyz", &fcc());
    let name = file.to_string_lossy().into_owned();
    let ang3 = Units::ANGSTROM_TO_BOHR.powi(3);
    // The unconstrained optimum of the atoms and the cell.
    let argv = [name.as_str(), "-os", "lj", "-vc", "-tcc", "-ns", "600"];
    let free = mopt_rs::cli::optmain::config_from_cli(&argv).unwrap();
    let free = mopt_rs::cli::optmain::run_with(&free, &dir, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0);
    assert!(free.converged);
    let free = free.summary.unwrap();
    let v_star = free.system.active_cell().unwrap().volume_bohr3() / ang3;
    let scan = |kind: &str, range: String, samples: &str, sub: &str| {
        let args: Vec<String> = [
            name.as_str(),
            "-os",
            "lj",
            "-vc",
            "-tcc",
            "-ns",
            "600",
            "-scan",
            "cell",
            kind,
            range.as_str(),
            "-nsample",
            samples,
            "-scan_opt",
            "figures=off",
            "html=off",
            "candidates=off",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let mut config = config_from_cli(&argv).unwrap();
        config.restart_dir = Some(dir.join(sub));
        let out = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("-scan cell {kind}: {e}"));
        let (lo, hi) = range.split_once(',').unwrap();
        let (lo, hi): (f64, f64) = (lo.parse().unwrap(), hi.parse().unwrap());
        let n: usize = samples.parse().unwrap();
        let mut points = Vec::new();
        for p in out.points.iter().filter(|p| !p.skipped) {
            assert!(p.converged, "-scan cell {kind}: point {}", p.point);
            let s = read(&dir.join(sub).join(&p.file));
            let value = lo + (hi - lo) * p.point as f64 / (n - 1) as f64;
            let expr = match kind {
                "volume" => format!("volume() = {value}*ang^3"),
                "gamma" => format!("cell_gamma() = {value}*deg"),
                other => format!("cell_{other}() = {value}*ang"),
            };
            let set = mopt_rs::pes::constraint::ConstraintSet::new(
                &[user_constraint(&expr).unwrap()],
                &s,
                None,
            )
            .unwrap();
            let r = set.residuals(&s).unwrap()[0];
            eprintln!(
                "RESIDUAL scan cell {kind} point {}: {:.2e}",
                p.point,
                r.abs()
            );
            assert!(r.abs() <= 1e-10, "-scan cell {kind} point {}: {r}", p.point);
            points.push((value, p.energy));
        }
        assert_eq!(points.len(), n, "-scan cell {kind}");
        points
    };
    // The energy-volume curve around the optimum.
    let curve = scan(
        "volume",
        format!("{},{}", 0.96 * v_star, 1.04 * v_star),
        "5",
        "volume",
    );
    let lowest = (0..curve.len())
        .min_by(|&a, &b| curve[a].1.total_cmp(&curve[b].1))
        .unwrap();
    assert!(lowest > 0 && lowest < curve.len() - 1, "{curve:?}");
    let ((x0, y0), (x1, y1), (x2, y2)) = (curve[lowest - 1], curve[lowest], curve[lowest + 1]);
    // The vertex of the parabola through the three lowest points.
    let vertex = x1
        - 0.5 * ((x1 - x0).powi(2) * (y1 - y2) - (x1 - x2).powi(2) * (y1 - y0))
            / ((x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0));
    eprintln!("E-V minimum {vertex:.4} Angstrom^3, unconstrained optimum {v_star:.4}");
    assert!(
        (vertex - v_star).abs() < 5e-3 * v_star,
        "{vertex} vs {v_star}"
    );
    // The lowest point of the curve is the energy of the optimum (to the convergence of the
    // optimizations).
    let least = curve.iter().fold(f64::INFINITY, |m, (_, e)| m.min(*e));
    eprintln!(
        "E-V lowest {least:.10}, unconstrained optimum {:.10}",
        free.energy.value()
    );
    assert!((least - free.energy.value()).abs() < 1e-4, "{least}");
    // A length and an angle.
    let l = free.system.active_cell().unwrap().lattice_vector(0).value();
    let a = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt() * Units::BOHR_TO_ANGSTROM;
    scan("a", format!("{},{}", a - 0.1, a + 0.1), "3", "a");
    scan("gamma", "87.0,93.0".to_string(), "3", "gamma");
    let _ = std::fs::remove_dir_all(&dir);
}
