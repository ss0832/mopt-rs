//! The ONIOM cell of the orientsearch constraint matrix with a bond constraint (a water of
//! GFN-FF in the high layer, another relaxed around it): it finds its minima, and the same ones
//! to the last bit in repeated runs on one and on four threads. Its samples once stood for
//! 1000 iterations with a step of zero: the gradient of the core kept a part along its rigid
//! motions (the residual of the relaxation of the environment) that the steps project out.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

use mopt_rs::cli::orientsearch::{config_from_cli, run_with};
use mopt_rs::pes::registry::CalculatorRegistry;

const WATERS: &str = "6\nwaters\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n\
O 4.0 0.0 0.1173\nH 4.0 0.7572 -0.4692\nH 4.0 -0.7572 -0.4692\n";

/// A fingerprint of the minima of one run: the bits of the sum of the coordinates of each.
fn minima(tag: &str, threads: usize) -> Vec<u64> {
    let dir = std::env::temp_dir().join(format!(
        "mopt_osoniom_{tag}_{threads}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("start.xyz");
    std::fs::write(&file, WATERS).unwrap();
    let path = file.to_string_lossy().into_owned();
    let args = [
        path.as_str(),
        "-os",
        "gfnff",
        "-oniom",
        "1-3",
        "none",
        "gfnff",
        "-nsample",
        "2",
        "-part",
        "1-3",
        "-orient_opt",
        "seed=5",
        "criteria=normal",
        "interaction=off",
        "-pc",
        "bond",
        "1,2",
    ];
    let config = config_from_cli(&args).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let out = pool
        .install(|| run_with(&config, &dir, None, &CalculatorRegistry::native()))
        .unwrap_or_else(|e| panic!("{tag} on {threads} thread(s): {e}"));
    assert!(
        !out.minima.is_empty(),
        "{tag} on {threads} thread(s): no minimum"
    );
    let energies = out
        .minima
        .iter()
        .map(|m| {
            m.system
                .coords_flat_bohr()
                .iter()
                .fold(0.0, |a, v| a + v)
                .to_bits()
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    energies
}

#[test]
fn the_oniom_bond_cell_finds_the_same_minima_on_one_and_four_threads() {
    let reference = minima("a", 1);
    assert_eq!(minima("b", 1), reference);
    assert_eq!(minima("c", 4), reference);
    assert_eq!(minima("d", 4), reference);
}
