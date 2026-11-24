//! `nebmain` with the images between the structures made by a method other than the
//! straight line (`-neb_opt path=`, `-idpp`).
//!
//! MultiOptPy: `neb.py:NEB.make_geometry_list` (`-idpp`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with};
use mopt_rs::domain::Units;
use mopt_rs::path::band::machine::ChainEnd;
use mopt_rs::path::interp::FirstPath;
use mopt_rs::pes::registry::CalculatorRegistry;

const KCAL: f64 = Units::HARTREE_TO_KCAL_MOL;

/// HCN and HNC, both on the x axis: on the straight line between them the hydrogen passes
/// through the carbon and the nitrogen.
const ENDS: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_neb_path_{name}_{}", std::process::id()));
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
fn two_structures_on_a_line_give_the_path_of_the_hydrogen_round_the_bond() {
    let dir = scratch("hcn");
    let input = file(&dir, "ends.xyz", ENDS);
    let registry = CalculatorRegistry::native();

    // The settings: the straight line without anything, -idpp, and the names.
    let plain = config_from_cli(&[&input, "-os", "gfn0xtb"]).unwrap();
    assert_eq!(plain.first_path, None);
    assert_eq!(
        mopt_rs::cli::nebmain::DEFAULT_FIRST_PATH,
        FirstPath::Geodesic
    );
    let multioptpy =
        config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", "form=multioptpy"]).unwrap();
    assert_eq!(multioptpy.first_path, Some(FirstPath::Linear));
    let flag = config_from_cli(&[&input, "-os", "gfn0xtb", "-idpp"]).unwrap();
    assert_eq!(flag.first_path, Some(FirstPath::Idpp));
    let flag = config_from_cli(&[&input, "-os", "gfn0xtb", "-cfbenm"]).unwrap();
    assert_eq!(flag.first_path, Some(FirstPath::Cfbenm));
    assert!(config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", "path=curved"]).is_err());

    for name in ["sidpp", "geodesic", "lst", "cfbenm"] {
        let setting = format!("path={name}");
        let config = config_from_cli(&[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "100",
            "-neb_opt",
            &setting,
            "figures=off",
        ])
        .unwrap();
        assert_eq!(config.first_path.unwrap().name(), name);
        let result = run_with(&config, &dir, None, &registry).unwrap();
        let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
        assert!(
            log.contains(&format!("First images (-neb_opt path={name})")),
            "{log}"
        );
        assert_eq!(result.end, Some(ChainEnd::Converged), "{name}: {log}");
        // The transition state of the hydrogen shift, 68.29 kcal/mol above HCN, with the
        // ends relaxed to HCN and HNC (15.72 kcal/mol above it).
        let first = result.energies[0];
        let top = result
            .energies
            .iter()
            .fold(f64::NEG_INFINITY, |m, e| m.max(*e));
        let last = result.energies[result.energies.len() - 1];
        println!(
            "{name}: {} iterations, {} images, barrier {:.3}",
            result.iterations,
            result.energies.len(),
            KCAL * (top - first)
        );
        assert!(
            (KCAL * (top - first) - 68.285).abs() < 0.3,
            "{name}: {}",
            KCAL * (top - first)
        );
        assert!((KCAL * (last - first) - 15.72).abs() < 0.05, "{name}");
    }

    // Plain variables have no distances between atoms.
    let start = file(&dir, "start.txt", "x -0.558\ny 1.442\n");
    let end = file(&dir, "end.txt", "x 0.623\ny 0.028\n");
    let variables =
        config_from_cli(&[&start, &end, "-os", "mb2d", "-neb_opt", "path=idpp"]).unwrap();
    assert!(run_with(&variables, &dir, None, &registry).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
