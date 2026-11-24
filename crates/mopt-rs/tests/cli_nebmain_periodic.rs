//! `nebmain` in a periodic cell: the hop of a vacancy in copper with the built-in
//! embedded-atom potential.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with};
use mopt_rs::cli::optmain;
use mopt_rs::domain::{SpinCharge, Units};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::path::band::machine::ChainEnd;
use mopt_rs::pes::registry::CalculatorRegistry;

const EV: f64 = Units::HARTREE_TO_EV;
/// Lattice constant of copper (Angstrom).
const A: f64 = 3.615;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_neb_pbc_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The 31 atoms of 2 x 2 x 2 cells of copper without the atom at the origin; the atom of
/// the site (A/2, A/2, 0) is the first and is put at `hopping`. All atoms are shifted by
/// `shift`.
fn copper_with_a_vacancy(hopping: [f64; 3], shift: [f64; 3]) -> String {
    let mut sites = vec![hopping];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let cell = [f64::from(i), f64::from(j), f64::from(k)];
                    let empty = cell == [0.0; 3] && f == [0.0, 0.0, 0.0];
                    let hops = cell == [0.0; 3] && f == [0.5, 0.5, 0.0];
                    if !empty && !hops {
                        sites.push([
                            A * (cell[0] + f[0]),
                            A * (cell[1] + f[1]),
                            A * (cell[2] + f[2]),
                        ]);
                    }
                }
            }
        }
    }
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        sites.len(),
        l = 2.0 * A
    );
    for s in &sites {
        text.push_str(&format!(
            "Cu {:.10} {:.10} {:.10}\n",
            s[0] + shift[0],
            s[1] + shift[1],
            s[2] + shift[2]
        ));
    }
    text
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn a_vacancy_of_copper_hops_to_the_next_site() {
    let dir = scratch("vacancy");
    // The neighbour of the empty site at the origin goes into it. The second structure
    // has that atom at the periodic image (2A, 2A, 0) of the origin, and all its atoms
    // shifted: the chain has to see a hop of A / sqrt(2), not a way through the cell.
    let start = file(
        &dir,
        "start.xyz",
        &copper_with_a_vacancy([0.5 * A, 0.5 * A, 0.0], [0.0; 3]),
    );
    let end = file(
        &dir,
        "end.xyz",
        &copper_with_a_vacancy([2.0 * A, 2.0 * A, 0.0], [0.3, -0.2, 0.1]),
    );
    let registry = CalculatorRegistry::native();
    let config = config_from_cli(&[
        &start,
        &end,
        "-os",
        "eam",
        "-ns",
        "150",
        "-neb_opt",
        "figures=off",
    ])
    .unwrap();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert_eq!(result.end, Some(ChainEnd::Converged), "{log}");
    assert!(log.contains("Periodic cell:"), "{log}");
    println!(
        "{} iterations, {} images",
        result.iterations,
        result.energies.len()
    );

    // The ends are the same structure but for the site that is empty.
    let images = result.energies.len();
    let (first, last) = (result.energies[0], result.energies[images - 1]);
    assert!((first - last).abs() < 1e-6, "{first} {last}");
    let candidates = result.candidates.clone().unwrap();
    assert_eq!(candidates.images.len(), 1, "{:?}", candidates.images);
    let top = result.energies[candidates.images[0]];
    let barrier = EV * (top - first);
    println!("barrier {barrier} eV");

    // The chain is as long as the hop, not as the way through the cell (the atoms beside
    // the empty site lean into it, so the hop is shorter than between the sites), the
    // files have the lattice, and the atom that hops is half way at the top.
    let path = read_xyz_trajectory_file(&candidates.path, SpinCharge::default(), None).unwrap();
    assert_eq!(path.frames.len(), images);
    let at = |frame: usize, k: usize| {
        path.frames[frame].system.coords_flat_bohr()[k] * Units::BOHR_TO_ANGSTROM
    };
    let cell = path.frames[0].system.active_cell().unwrap().clone();
    assert!((cell.h_bohr[0][0] * Units::BOHR_TO_ANGSTROM - 2.0 * A).abs() < 1e-9);
    let hop: f64 = (0..3)
        .map(|k| (at(images - 1, k) - at(0, k)).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((hop - A / 2.0_f64.sqrt()).abs() < 0.3, "{hop}");
    let saddle = candidates.images[0];
    let way: f64 = (0..3)
        .map(|k| (at(saddle, k) - at(0, k)).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((way - 0.5 * hop).abs() < 0.15, "{way} of {hop}");

    // A transition-state optimization from the highest image changes little: the image
    // is at the saddle of the hop.
    let image = candidates.files[0].to_string_lossy().into_owned();
    let ts = optmain::config_from_cli(&[
        &image,
        "-os",
        "eam",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "5",
        "-ns",
        "60",
    ])
    .unwrap();
    let out = optmain::run_with(&ts, &dir, None, &registry).unwrap();
    assert!(out[0].converged);
    let summary = out[0].summary.as_ref().unwrap();
    let saddle_barrier = EV * (summary.energy.value() - first);
    println!(
        "saddle {saddle_barrier} eV after {} iterations",
        summary.iterations
    );
    assert!(
        (barrier - saddle_barrier).abs() < 0.01,
        "{barrier} {saddle_barrier}"
    );
    // 0.53 eV in this cell of 31 atoms with this potential.
    assert!(
        saddle_barrier > 0.45 && saddle_barrier < 0.60,
        "{saddle_barrier}"
    );
    let _ = read_xyz_file(&candidates.files[0], SpinCharge::default(), None).unwrap();

    // A second structure in another cell is refused.
    let other = copper_with_a_vacancy([0.0; 3], [0.0; 3]).replace(
        &format!("Lattice=\"{}", 2.0 * A),
        &format!("Lattice=\"{}", 2.0 * A + 0.1),
    );
    let other = file(&dir, "other.xyz", &other);
    let refused = config_from_cli(&[&start, &other, "-os", "eam"]).unwrap();
    assert!(run_with(&refused, &dir, None, &registry).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
