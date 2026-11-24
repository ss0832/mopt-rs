//! `-ubias` with functions of the cell: the expression `P * volume()` is the enthalpy term of
//! an external pressure `P`, so a variable-cell optimization with it ends where one with
//! `-vc_pressure P` ends (the stress of the term enters the cell gradient), and the term is
//! built with the lattice (`pes::bias::CellExpression`) instead of failing for want of it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use mopt_rs::cli::optmain;
use mopt_rs::domain::Units;
use mopt_rs::pes::registry::CalculatorRegistry;

/// A 2 x 2 x 2 fcc argon crystal at 5.30 Angstrom, every atom shifted a little.
fn fcc() -> String {
    let a = 5.30;
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
                    let shift = [0.03 * (1.3 * n).sin(), 0.02 * (0.7 * n).cos(), 0.0];
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
fn a_bias_on_the_volume_acts_as_a_pressure() {
    let dir = std::env::temp_dir().join(format!("mopt_ubias_cell_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("fcc.xyz");
    std::fs::write(&file, fcc()).unwrap();
    let file = file.to_string_lossy().into_owned();
    let gpa = 0.5;
    let per_bohr3 = gpa / Units::GPA_PER_AU_PRESSURE;
    let expression = format!("{per_bohr3:.12e} * volume()");
    let run = |extra: &[&str]| {
        let mut args = vec![
            file.as_str(),
            "-os",
            "lj",
            "-vc",
            "-opt",
            "rsirfo_bfgs",
            "-ns",
            "400",
        ];
        args.extend_from_slice(extra);
        let cfg = optmain::config_from_cli(&args).unwrap();
        let out = optmain::run_with(&cfg, &dir, None, &CalculatorRegistry::native())
            .unwrap()
            .remove(0);
        assert!(out.converged, "{extra:?}");
        out.summary
            .unwrap()
            .system
            .active_cell()
            .unwrap()
            .volume_bohr3()
    };
    let pressure = gpa.to_string();
    let with_pressure = run(&["-vc_pressure", &pressure]);
    let with_bias = run(&["-ubias", &expression]);
    let free = run(&[]);
    eprintln!("volumes: free {free}, -vc_pressure {with_pressure}, -ubias {with_bias}");
    assert!(with_pressure < free);
    assert!(
        (with_bias - with_pressure).abs() < 1e-4 * with_pressure,
        "{with_bias} against {with_pressure}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
