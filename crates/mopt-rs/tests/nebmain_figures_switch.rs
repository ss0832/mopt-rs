//! The figures of `nebmain` (`-spng`, `save_pict`): written by default, none with `-spng off`
//! or `"save_pict": false` in a settings file, the CSV data and the numbers the same in every
//! case; `-spng` alone keeps MultiOptPy's meaning (on).
//!
//! MultiOptPy: `neb.py` (the figures only with `save_pict`, off by default; on by default in
//! mopt-rs by the owner's choice).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with, NebmainConfig, NebmainResult};
use mopt_rs::cli::settings::{merge, parse_settings};
use mopt_rs::cli::spec::NEBMAIN;
use mopt_rs::pes::registry::CalculatorRegistry;

const FIGURES: [&str; 4] = [
    "energy_profile",
    "image_forces",
    "convergence",
    "chain_on_surface",
];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_nebfig_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A run with the options `extra`, and the settings file `settings` under them when given
/// (default < file < command line, as the commands merge them).
fn run(dir: &Path, extra: &[&str]) -> NebmainResult {
    run_with_settings(dir, extra, None)
}

fn run_with_settings(dir: &Path, extra: &[&str], settings: Option<&str>) -> NebmainResult {
    let start = dir.join("start.txt");
    let end = dir.join("end.txt");
    std::fs::write(&start, "x -0.558\ny 1.442\n").unwrap();
    std::fs::write(&end, "x 0.623\ny 0.028\n").unwrap();
    let (s, e) = (start.to_string_lossy(), end.to_string_lossy());
    let mut args = vec![&*s, &*e, "-os", "mb2d", "-p", "9", "-ns", "6"];
    args.extend_from_slice(extra);
    let config = match settings {
        None => config_from_cli(&args).unwrap(),
        Some(text) => {
            let cli = mopt_rs::cli::option::parser("nebmain", NEBMAIN)
                .parse(&args[..])
                .unwrap();
            NebmainConfig::from_args(&merge(parse_settings(text, NEBMAIN).unwrap(), cli)).unwrap()
        }
    };
    run_with(&config, dir, None, &CalculatorRegistry::native()).unwrap()
}

fn figures_in(dir: &Path) -> usize {
    FIGURES
        .iter()
        .flat_map(|f| ["png", "svg"].map(|x| dir.join(format!("{f}.{x}"))))
        .filter(|p| p.is_file())
        .count()
}

#[test]
fn figures_by_default_none_when_switched_off_and_the_same_numbers() {
    let on = run(&scratch("on"), &[]);
    let flag = run(&scratch("flag"), &["-spng"]);
    let flag_then_option = run(&scratch("flag_then"), &["-spng", "-ns", "6"]);
    let explicit_on = run(&scratch("explicit_on"), &["-spng", "on"]);
    let off = run(&scratch("off"), &["-spng", "off"]);
    let off_inline = run(&scratch("off_inline"), &["-spng=false"]);
    let file = run_with_settings(&scratch("settings"), &[], Some("{\"save_pict\": false}"));
    let file_on = run_with_settings(&scratch("settings_on"), &[], Some("{\"save_pict\": true}"));
    // The command line wins over the file.
    let file_then_flag = run_with_settings(
        &scratch("settings_flag"),
        &["-spng"],
        Some("{\"save_pict\": false}"),
    );
    for r in [
        &on,
        &flag,
        &flag_then_option,
        &explicit_on,
        &file_on,
        &file_then_flag,
    ] {
        assert_eq!(figures_in(&r.directory), 8, "{}", r.directory.display());
    }
    for r in [&off, &off_inline, &file] {
        assert_eq!(figures_in(&r.directory), 0, "{}", r.directory.display());
    }
    for r in [
        &flag,
        &flag_then_option,
        &explicit_on,
        &off,
        &off_inline,
        &file,
        &file_on,
        &file_then_flag,
    ] {
        // The data of the figures are written as before, and the chain is the same.
        assert_eq!(
            std::fs::read_to_string(r.directory.join("energy_plot.csv")).unwrap(),
            std::fs::read_to_string(on.directory.join("energy_plot.csv")).unwrap()
        );
        assert_eq!(r.iterations, on.iterations);
        let bits = |v: &[f64]| v.iter().map(|e| e.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&r.energies), bits(&on.energies));
    }
    // A word that is not a switch is refused with the option named.
    for bad in [
        &["a.txt", "b.txt", "-spng=maybe"][..],
        &["a.txt", "-spng", "b.txt", "c.txt"][..],
    ] {
        let message = match config_from_cli(bad) {
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        };
        assert!(message.contains("-spng"), "{bad:?}: '{message}'");
    }
}
