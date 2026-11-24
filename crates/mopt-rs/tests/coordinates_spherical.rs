//! Spherical coordinates as coordinate systems: optimizations in `-coord spherical` and
//! `-coord hyperspherical` reach the minima of Cartesian coordinates; chains refuse them.
//!
//! MultiOptPy: `Coordinate/polar_coordinate.py` (used there inside iEIP only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::{nebmain, optmain};
use mopt_rs::coord::system::CoordinateRegistry;
use mopt_rs::pes::registry::CalculatorRegistry;

const WATER: &str = "3\nwater\nO 0.0 0.0 0.12\nH 0.0 0.76 -0.47\nH 0.0 -0.78 -0.45\n";
const ETHYLENE: &str = "6\nethylene\nC 0.0 0.0 0.67\nC 0.0 0.0 -0.66\nH 0.0 0.93 1.24\n\
H 0.0 -0.92 1.23\nH 0.0 0.92 -1.25\nH 0.05 -0.93 -1.22\n";
const HCN: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";
/// A centre of hyperspherical coordinates off the path of HCN -> HNC.
const CENTRE: &str = "3\ncentre\nC -0.5 -0.8 0.6\nN 1.7 -0.6 -0.5\nH 0.6 -3.5 0.4\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_spherical_{name}_{}", std::process::id()));
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
fn optimizations_in_spherical_coordinates_reach_the_cartesian_minima() {
    let dir = scratch("opt");
    let registry = CalculatorRegistry::native();
    for (name, text) in [("water", WATER), ("ethylene", ETHYLENE)] {
        let input = file(&dir, &format!("{name}.xyz"), text);
        let minimum = |coordinates: &str| {
            let config = optmain::config_from_cli(&[
                &input,
                "-os",
                "gfnff",
                "-coord",
                coordinates,
                "-ns",
                "300",
                "-tcc",
            ])
            .unwrap();
            let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
            let summary = out[0].summary.as_ref().unwrap();
            println!("{name} {coordinates}: {} iterations", summary.iterations);
            assert!(out[0].converged, "{name} {coordinates}");
            summary.energy.value()
        };
        let reference = minimum("cart");
        for coordinates in ["spherical", "hyperspherical", "polar"] {
            let found = minimum(coordinates);
            assert!(
                (found - reference).abs() < 1e-6,
                "{name} {coordinates}: {found} {reference}"
            );
        }
    }
    let names = CoordinateRegistry::default().names().join(" ");
    assert!(names.contains("spherical") && names.contains("hyperspherical"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_chain_refuses_spherical_coordinates() {
    // A chain of HCN -> HNC in hyperspherical coordinates about a structure off the path
    // still had forces of 1 Hartree/Bohr after 150 iterations: the lengths of the angles
    // change along it. Chains refuse them, whatever the centre.
    let dir = scratch("chain");
    let input = file(&dir, "hcn.xyz", HCN);
    let centre = file(&dir, "centre.xyz", CENTRE);
    let registry = CalculatorRegistry::native();
    let about = format!("hyperspherical:{centre}");
    for coordinates in ["spherical", "hyperspherical", "polar", about.as_str()] {
        let config =
            nebmain::config_from_cli(&[&input, "-os", "gfn0xtb", "-coord", coordinates]).unwrap();
        let refused = nebmain::run_with(&config, &dir, None, &registry);
        assert!(refused.is_err(), "{coordinates}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
