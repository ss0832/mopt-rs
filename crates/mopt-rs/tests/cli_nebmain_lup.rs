//! `nebmain -lup`: the locally updated planes on their own, on the Muller-Brown surface (a
//! chain of plain variables whose saddles are known). The relaxed chain has no force across
//! the path left at its inner images, and it passes through both saddles; what the planes do
//! not have is refused by name.
//!
//! MultiOptPy: no counterpart (`-lup` is declared in MultiOptPy's `interface.py` without an
//! implementation; new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with};
use mopt_rs::path::band::machine::ChainEnd;
use mopt_rs::pes::native::muller_brown::MB_SADDLES;
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_nebmain_lup_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

/// The points of the last chain (`final_path.csv`-like tables are not assumed): the
/// images of the chain of the last iteration in its directory.
fn last_chain(dir: &Path) -> Vec<Vec<f64>> {
    let mut iterations: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.to_string_lossy().contains("path_ITR_"))
        .collect();
    iterations.sort_by_key(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| {
                n.trim_start_matches("path_ITR_")
                    .split('_')
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
            .unwrap_or(0)
    });
    let last = iterations.last().unwrap();
    let csv = std::fs::read_dir(last)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "csv") && p.to_string_lossy().contains("path"))
        .unwrap();
    std::fs::read_to_string(csv)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| l.split(',').skip(1).map(|v| v.parse().unwrap()).collect())
        .collect()
}

#[test]
fn the_planes_relax_the_muller_brown_chain_through_its_saddles() {
    let dir = scratch("mb");
    let start = file(&dir, "start.txt", "x -0.558\ny 1.442\n");
    let end = file(&dir, "end.txt", "x 0.623\ny 0.028\n");
    let config = config_from_cli(&[
        &start,
        &end,
        "-os",
        "mb2d",
        "-lup",
        "-p",
        "15",
        "-fe",
        "3",
        "-ns",
        "300",
        "-neb_opt",
        "tolerance=1e-6",
    ])
    .unwrap();
    // The force of the planes: no springs, no climbing image.
    assert!(config.chain.springs.is_none() && config.chain.climb_after.is_none());
    let result = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    // Converged: the largest force across the path of the inner images is below 1e-6.
    assert_eq!(
        result.end,
        Some(ChainEnd::Converged),
        "{}",
        result.iterations
    );
    // The path passes through both saddles: the polygon of the images comes near each.
    let chain = last_chain(&result.directory);
    for saddle in MB_SADDLES {
        let nearest = chain
            .windows(2)
            .map(|w| {
                let d = [w[1][0] - w[0][0], w[1][1] - w[0][1]];
                let o = [saddle[0] - w[0][0], saddle[1] - w[0][1]];
                let s = ((o[0] * d[0] + o[1] * d[1]) / (d[0] * d[0] + d[1] * d[1])).clamp(0.0, 1.0);
                ((o[0] - s * d[0]).powi(2) + (o[1] - s * d[1]).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(nearest < 2e-2, "saddle {saddle:?}: {nearest}");
    }
    // What the planes do not have is refused by name.
    for (extra, expected) in [
        (vec!["-ewbneb"], "-ewbneb"),
        (vec!["-cineb", "3"], "-cineb"),
        (vec!["-neb_opt", "projection=atom"], "whole images"),
        (vec!["-qsm"], "-qsm"),
    ] {
        let mut arguments = vec![start.as_str(), end.as_str(), "-os", "mb2d", "-lup"];
        arguments.extend(extra.iter().copied());
        let err = config_from_cli(&arguments).unwrap_err().to_string();
        assert!(err.contains(expected), "{extra:?}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
