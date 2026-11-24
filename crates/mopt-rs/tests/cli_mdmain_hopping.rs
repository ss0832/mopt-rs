//! `mdmain` with Zhu-Nakamura surface hopping on the excited singlet states of the built-in
//! GFN0-xTB (sTDA): a short run of formaldehyde from its second excited state.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use mopt_rs::cli::mdmain::{config_from_cli, run_with};
use mopt_rs::pes::registry::CalculatorRegistry;

#[test]
fn a_run_on_the_excited_states_of_gfn0xtb_records_its_state_and_keeps_its_energy() {
    let base = std::env::temp_dir().join(format!("mopt_md_zn_gfn0_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let xyz = base.join("ch2o.xyz");
    std::fs::write(
        &xyz,
        "4\nformaldehyde\nC 0.000 0.000 0.000\nO 0.000 0.000 1.210\nH 0.000 0.940 -0.590\n\
         H 0.000 -0.940 -0.590\n",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let config = config_from_cli(&[
        &xyz,
        "-os",
        "gfn0xtb",
        "-es",
        "2",
        "-mt",
        "nve",
        "-time",
        "40",
        "-ts",
        "0.25fs",
        "-temp",
        "300",
        "-md_opt",
        "seed=5",
        "write_every=1",
        "hopping_states=1,2,3",
        "hopping_gap=3",
    ])
    .unwrap();
    let runs = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    let dir = &runs[0].directory;
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    assert!(log.contains("Surface hopping (Zhu-Nakamura"), "{log}");
    assert!(log.contains("Surface hopping: "), "{log}");
    let states = std::fs::read_to_string(dir.join("ch2o_states.csv")).unwrap();
    let rows: Vec<&str> = states.lines().collect();
    assert_eq!(rows.len(), 42);
    assert!(rows[0].ends_with("energy_state_3_hartree"));
    // Every row: the active state and three energies in ascending order.
    for row in &rows[1..] {
        let f: Vec<f64> = row.split(',').map(|x| x.parse().unwrap()).collect();
        assert!([1.0, 2.0, 3.0].contains(&f[2]));
        assert!(f[3] <= f[4] && f[4] <= f[5], "{row}");
    }
    let trajectory = std::fs::read_to_string(&runs[0].trajectory).unwrap();
    assert_eq!(trajectory.matches(" state=").count(), 41);
    // NVE on the excited states: the conserved quantity stays within the measured bound.
    let properties = std::fs::read_to_string(dir.join("md_properties.csv")).unwrap();
    let conserved: Vec<f64> = properties
        .lines()
        .skip(1)
        .map(|l| l.split(',').nth(6).unwrap().parse().unwrap())
        .collect();
    let largest = conserved
        .iter()
        .map(|c| (c - conserved[0]).abs())
        .fold(0.0, f64::max);
    assert!(largest < 2e-4, "{largest}");
    let _ = std::fs::remove_dir_all(&base);
}
