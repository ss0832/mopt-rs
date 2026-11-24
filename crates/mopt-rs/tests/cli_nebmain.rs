//! `nebmain` through the command-line layer: chains of plain variables on the Muller-Brown
//! surface and chains of structures (HCN to HNC) with the built-in GFN0-xTB calculator.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with, run_with_observers, NebmainResult};
use mopt_rs::domain::{EvalRequest, SpinCharge, Units};
use mopt_rs::io::npy::read_npy_matrix;
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::path::band::images::{AddedImages, FailedImage};
use mopt_rs::path::band::machine::{ChainEnd, ChainRelaxation, ChainReport};
use mopt_rs::path::band::Compatibility;
use mopt_rs::pes::native::muller_brown::MB_SADDLES;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::chain::Flow;

const KCAL: f64 = Units::HARTREE_TO_KCAL_MOL;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_nebmain_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

fn run(base: &Path, arguments: &[&str]) -> NebmainResult {
    let config = config_from_cli(arguments).unwrap();
    run_with(&config, base, None, &CalculatorRegistry::native()).unwrap()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

/// The run directory of `base` (`<name>_NEB_<time>/`, next to the input) that was changed
/// last.
fn newest_run(base: &Path) -> PathBuf {
    std::fs::read_dir(base)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.to_string_lossy().contains("_NEB_"))
        .max_by_key(|run| std::fs::metadata(run).and_then(|m| m.modified()).ok())
        .unwrap()
}

/// Three rough structures from HCN over a bridged one to HNC.
const HCN: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

#[test]
fn default_band_limits_and_image_settings() {
    let dir = scratch("settings");
    let input = file(&dir, "path.xyz", HCN);
    let articles = config_from_cli(&[&input, "-os", "gfn0xtb"]).unwrap();
    assert_eq!(articles.chain.max_iterations, 30);
    let dmf = config_from_cli(&[&input, "-dmf"]).unwrap();
    assert_eq!(dmf.dmf.unwrap().max_iterations, 60);
    let limited = config_from_cli(&[&input, "-dmf", "-ns", "8"]).unwrap();
    assert_eq!(limited.dmf.unwrap().max_iterations, 8);
    // ... except for the Hessians: a model Hessian instead of `o1numhess`.
    assert_eq!(articles.model_hessian.as_deref(), Some("lindh2007d3"));
    assert_eq!(articles.chain.predicted_maxima_every, Some(2));
    assert_eq!(articles.chain.hessian_every, Some(50));
    // ... except that the run ends when the chain is relaxed.
    assert_eq!(articles.chain.force_tolerance, Some(3e-4));
    assert!(articles.chain.diis.is_none());
    assert_eq!(articles.spacing, None);
    assert!(articles.chain.rfo.is_some() && articles.chain.gathering.is_some());
    assert!(!articles.chain.ends.first && !articles.chain.ends.last);
    assert!(!articles.per_atom && articles.figures);
    assert_eq!(articles.chain.compatibility, Compatibility::Consistent);

    // MultiOptPy's defaults can be chosen, and its options are read.
    let plain = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-ns",
        "10",
        "-modelhess",
        "none",
        "-nd",
        "0",
        "-adpred",
        "0",
        "-fe",
        "2",
        "-cineb",
        "5",
        "-sd",
        "8",
        "-ad",
        "3",
        "-aconv",
        "-p",
        "4",
        "-fix",
        "1",
    ])
    .unwrap();
    assert_eq!(plain.chain.max_iterations, 10);
    assert!(plain.chain.rfo.is_none() && plain.chain.hessian_every.is_none());
    assert_eq!(plain.model_hessian, None);
    assert_eq!(plain.spacing, Some(0.0));
    assert_eq!(plain.chain.predicted_maxima_every, None);
    assert!(!plain.chain.ends.first && plain.chain.ends.last);
    assert_eq!(plain.chain.climb_after, Some(5));
    assert_eq!(plain.chain.steepest_from, Some(8));
    assert_eq!(plain.chain.equal_spacing_every, Some(3));
    assert!(plain.first_equal_spacing && plain.chain.hold_converged.is_some());
    assert_eq!(plain.inserted, 4);
    assert_eq!(plain.constraints.len(), 1);

    // Exact Hessians every 5 iterations; the share of the quasi-Newton step; -neb_opt.
    let exact = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-fc",
        "5",
        "-rrs",
        "0.3",
        "-neb_opt",
        "tolerance=1e-4",
        "gather=0",
        "pairs=1",
        "projection=atom",
        "form=multioptpy",
        "figures=off",
    ])
    .unwrap();
    assert!(exact.exact_hessians);
    assert_eq!(exact.chain.hessian_every, Some(5));
    let rfo = exact.chain.rfo.as_ref().unwrap();
    assert_eq!((rfo.ratio, rfo.pairs), (0.3, Some(1)));
    assert_eq!(exact.chain.force_tolerance, Some(1e-4));
    assert!(exact.chain.gathering.is_none() && exact.per_atom && !exact.figures);
    assert_eq!(exact.chain.compatibility, Compatibility::MultiOptPy);
    // MultiOptPy's form runs all its iterations unless a tolerance is given; the
    // extrapolation over the last chains is a setting.
    let legacy =
        config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", "form=multioptpy"]).unwrap();
    assert_eq!(legacy.chain.force_tolerance, None);
    let all = config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", "tolerance=0"]).unwrap();
    assert_eq!(all.chain.force_tolerance, None);
    let diis = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-neb_opt",
        "diis=4",
        "diis_start=1e-3",
    ])
    .unwrap();
    let settings = diis.chain.diis.unwrap();
    assert_eq!((settings.points, settings.start_force), (4, 1e-3));

    // What is not implemented yet says so; wrong values are refused.
    let refused = config_from_cli(&[&input, "-os", "gfn0xtb", "-dneb"]).unwrap_err();
    assert!(refused.to_string().contains("doubly nudged"), "{refused}");
    for bad in [
        vec!["-neb_opt", "speed=3"],
        vec!["-neb_opt", "projection=bond"],
        vec!["-rrs", "1.5"],
        vec!["-nd", "-1"],
        vec!["-ns", "many"],
    ] {
        let mut arguments = vec![input.as_str(), "-os", "gfn0xtb"];
        arguments.extend(bad.iter().copied());
        assert!(config_from_cli(&arguments).is_err(), "{bad:?}");
    }
    assert!(config_from_cli(&["-os", "gfn0xtb"]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_chain_of_variables_finds_both_saddles_of_the_muller_brown_surface() {
    let dir = scratch("mb");
    let start = file(&dir, "start.txt", "x -0.558\ny 1.442\n");
    let end = file(&dir, "end.txt", "x 0.623\ny 0.028\n");
    let result = run(
        &dir,
        &[
            &start,
            &end,
            "-os",
            "mb2d",
            "-p",
            "15",
            "-fe",
            "3",
            "-ns",
            "80",
            "-neb_opt",
            "tolerance=1e-6",
        ],
    );
    assert_eq!(
        result.end,
        Some(ChainEnd::Converged),
        "{}",
        result.iterations
    );
    assert!(result.iterations < 60, "{} iterations", result.iterations);
    assert!(!result.interrupted);
    assert_eq!(result.energies.len(), 17);
    // An image on each saddle: the candidates, with the energies of the saddles.
    let candidates = result.candidates.clone().unwrap();
    assert_eq!(candidates.images.len(), 2);
    let table = read(&candidates.path);
    let points: Vec<Vec<f64>> = table
        .lines()
        .skip(1)
        .map(|line| {
            line.split(',')
                .skip(1)
                .map(|v| v.parse().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(table.lines().next(), Some("image,x,y"));
    for (image, saddle) in candidates.images.iter().zip(MB_SADDLES) {
        let point = &points[*image];
        let off = ((point[0] - saddle[0]).powi(2) + (point[1] - saddle[1]).powi(2)).sqrt();
        assert!(off < 2e-3, "image {image} at {point:?}, saddle {saddle:?}");
    }
    // The fixed ends are where they were given.
    assert_eq!(points[0], vec![-0.558, 1.442]);
    assert_eq!(points[16], vec![0.623, 0.028]);

    // The files: MultiOptPy's tables with a line per iteration, the chain of every
    // iteration, the Hessians of the images, the figures, the log, and the settings.
    let run_dir = &result.directory;
    assert_eq!(
        read(&run_dir.join("energy_plot.csv")).lines().count(),
        result.iterations
    );
    assert!(run_dir
        .join("path_ITR_0_start")
        .join("start_path.csv")
        .is_file());
    assert!(run_dir
        .join("path_ITR_0_start")
        .join("approx_TS_node.txt")
        .is_file());
    let hessian = read_npy_matrix(run_dir.join("tmp_hessian_5.npy")).unwrap();
    assert_eq!((hessian.nrows(), hessian.ncols()), (2, 2));
    for figure in [
        "energy_profile",
        "image_forces",
        "convergence",
        "chain_on_surface",
    ] {
        for extension in ["png", "svg"] {
            assert!(
                run_dir.join(format!("{figure}.{extension}")).is_file(),
                "{figure}"
            );
        }
    }
    let log = read(&run_dir.join("mopt.log"));
    assert!(
        log.contains("Chain: 17 images, 80 iterations, ends fixed (-fe 3)"),
        "{log}"
    );
    assert!(log.contains("Candidate for a transition state: image"));
    assert!(log.contains("The chain is relaxed"));
    assert!(run_dir.join("settings.json").is_file() && run_dir.join("input.txt").is_file());

    // One variables file is no chain; other numbers of variables are refused.
    let alone = config_from_cli(&[&start, "-os", "mb2d"]).unwrap();
    assert!(run_with(&alone, &dir, None, &CalculatorRegistry::native()).is_err());
    let three = file(&dir, "three.txt", "1.0\n2.0\n3.0\n");
    let mixed = config_from_cli(&[&start, &three, "-os", "mb2d"]).unwrap();
    assert!(run_with(&mixed, &dir, None, &CalculatorRegistry::native()).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Distances C-N, C-H, and N-H (Angstrom) of a structure of C, N, H.
fn distances(x: &[f64]) -> (f64, f64, f64) {
    let d = |a: usize, b: usize| {
        (0..3)
            .map(|k| (x[3 * a + k] - x[3 * b + k]).powi(2))
            .sum::<f64>()
            .sqrt()
            * Units::BOHR_TO_ANGSTROM
    };
    (d(0, 1), d(0, 2), d(1, 2))
}

fn coordinates(path: &Path) -> Vec<f64> {
    let sys = read_xyz_file(path, SpinCharge::default(), None).unwrap();
    sys.coords_flat_bohr().iter().copied().collect()
}

#[test]
fn hcn_to_hnc_with_the_settings_of_the_articles() {
    let dir = scratch("hcn");
    let input = file(&dir, "hcn.xyz", HCN);
    // The settings of the articles (Hessians from gradients, images 0.5 Angstrom apart,
    // redistribution towards the predicted maxima), with a tolerance and more iterations.
    let result = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-modelhess",
            "o1numhess",
            "-ns",
            "60",
            "-neb_opt",
            "tolerance=1e-4",
        ],
    );
    assert_eq!(
        result.end,
        Some(ChainEnd::Converged),
        "{}",
        result.iterations
    );
    assert!(result.iterations < 40, "{} iterations", result.iterations);
    // The geodesic between the three structures is longer than the straight lines.
    assert_eq!(result.energies.len(), 9);
    // The free ends were minimized: HCN and, 15.72 kcal/mol above it, HNC.
    let first = result.energies[0];
    let reaction = KCAL * (result.energies[8] - first);
    assert!((reaction - 15.72).abs() < 0.02, "{reaction}");
    // One candidate: the transition state, 68.29 kcal/mol above HCN (the energy of the
    // transition state optimized with optmain from this image, see docs/source/irc.md),
    // with H bridging C and N.
    let candidates = result.candidates.clone().unwrap();
    assert_eq!(candidates.images.len(), 1);
    let barrier = KCAL * (result.energies[candidates.images[0]] - first);
    assert!((barrier - 68.285).abs() < 0.02, "{barrier}");
    let (cn, ch, nh) = distances(&coordinates(&candidates.files[0]));
    assert!(
        cn > 1.1 && cn < 1.3 && ch > 1.0 && ch < 1.4 && nh > 1.2 && nh < 1.6,
        "{cn} {ch} {nh}"
    );
    let last = read_xyz_trajectory_file(&candidates.path, SpinCharge::default(), None).unwrap();
    assert_eq!(last.frames.len(), 9);
    let run_dir = &result.directory;
    assert!(run_dir.join("evaluations.journal").is_file());
    assert_eq!(
        read_npy_matrix(run_dir.join("tmp_hessian_3.npy"))
            .unwrap()
            .nrows(),
        9
    );
    let log = read(&run_dir.join("mopt.log"));
    assert!(
        log.contains("3 structures read; first chain of 9 images, 0.5 Angstrom apart"),
        "{log}"
    );
    assert!(log.contains("from -modelhess o1numhess"));

    // Fixed ends (-fe 3) stay where the first chain has them; -fe 1 fixes the first only.
    let fixed = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "4",
            "-fe",
            "3",
            "-neb_opt",
            "figures=off",
        ],
    );
    let image = |result: &NebmainResult, iteration: usize, k: usize| {
        coordinates(
            &result
                .directory
                .join(format!("path_ITR_{iteration}_hcn"))
                .join(format!("hcn_{k}.xyz")),
        )
    };
    let same = |a: &[f64], b: &[f64]| {
        let (da, db) = (distances(a), distances(b));
        (da.0 - db.0).abs() < 1e-9 && (da.1 - db.1).abs() < 1e-9 && (da.2 - db.2).abs() < 1e-9
    };
    assert!(same(&image(&fixed, 0, 0), &image(&fixed, 3, 0)));
    assert!(same(&image(&fixed, 0, 8), &image(&fixed, 3, 8)));
    assert!(!fixed.directory.join("energy_profile.png").exists());
    let half = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "4",
            "-fe",
            "1",
            "-neb_opt",
            "figures=off",
        ],
    );
    assert!(same(&image(&half, 0, 0), &image(&half, 3, 0)));
    assert!(!same(&image(&half, 0, 8), &image(&half, 3, 8)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_chain_comes_to_rest_with_a_model_hessian_and_with_either_kind_of_step_alone() {
    let dir = scratch("steps");
    let input = file(&dir, "hcn.xyz", HCN);
    let barrier = |result: &NebmainResult| {
        let top = result.energies.iter().copied().fold(f64::MIN, f64::max);
        KCAL * (top - result.energies[0])
    };
    // A model Hessian instead of Hessians from gradients. Before the moves of the image
    // at the maximum were limited where they reverse, this chain went back and forth
    // between two places for ever (barriers of 68.29 and 68.05 kcal/mol in turn).
    let model = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "60",
            "-modelhess",
            "lindh2007d3",
            "-neb_opt",
            "tolerance=1e-4",
            "figures=off",
        ],
    );
    assert_eq!(model.end, Some(ChainEnd::Converged));
    assert!(model.iterations <= 40, "{}", model.iterations);
    assert!(
        (barrier(&model) - 68.285).abs() < 0.1,
        "{}",
        barrier(&model)
    );
    // Quasi-Newton steps alone (-rrs 1).
    let newton = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "60",
            "-rrs",
            "1",
            "-neb_opt",
            "tolerance=1e-4",
            "figures=off",
        ],
    );
    assert_eq!(newton.end, Some(ChainEnd::Converged));
    assert!(newton.iterations <= 40, "{}", newton.iterations);
    assert!(
        (barrier(&newton) - 68.285).abs() < 0.01,
        "{}",
        barrier(&newton)
    );
    // Steps along the forces alone (-modelhess none): slower, but the chain relaxes; after
    // 40 iterations the barrier is within 2 kcal/mol.
    let forces = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "40",
            "-modelhess",
            "none",
            "-neb_opt",
            "figures=off",
        ],
    );
    assert_eq!(forces.end, Some(ChainEnd::IterationLimit));
    assert!(
        (barrier(&forces) - 68.285).abs() < 2.0,
        "{}",
        barrier(&forces)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn images_are_added_and_the_whole_chain_takes_one_step() {
    let dir = scratch("adaptive");
    let input = file(&dir, "hcn.xyz", HCN);
    let barrier = |result: &NebmainResult| {
        let top = result.energies.iter().copied().fold(f64::MIN, f64::max);
        KCAL * (top - result.energies[0])
    };
    // The settings: -aneb N FREQ, -gqnt and the keys of -neb_opt.
    let config = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-aneb",
        "2",
        "7",
        "-gqnt",
        "-neb_opt",
        "failed=drop",
        "global_memory=8",
    ])
    .unwrap();
    assert_eq!(
        config.chain.added,
        Some(AddedImages {
            images: 2,
            every: 7
        })
    );
    assert_eq!(config.chain.on_failure, FailedImage::Drop);
    let global = config.chain.rfo.as_ref().unwrap().global.unwrap();
    assert_eq!(global.memory_mb, 8.0);
    let swept = config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", "sweep=on"]).unwrap();
    let rfo = swept.chain.rfo.as_ref().unwrap();
    assert!(rfo.sweep && rfo.global.is_none());
    assert_eq!(swept.chain.on_failure, FailedImage::Retry);
    assert_eq!(swept.chain.added, None);
    for wrong in [
        vec!["-aneb", "2"],
        vec!["-aneb", "0", "5"],
        vec!["-neb_opt", "failed=maybe"],
        vec!["-neb_opt", "global=perhaps"],
    ] {
        let mut arguments = vec![input.as_str(), "-os", "gfn0xtb"];
        arguments.extend(wrong);
        assert!(config_from_cli(&arguments).is_err(), "{arguments:?}");
    }

    // A coarse chain (images 1 Angstrom apart) with an image added on either side of the
    // maximum after every fifth iteration: the chain grows, and the barrier is that of
    // the fine chain.
    let adaptive = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "80",
            "-nd",
            "1.0",
            "-aneb",
            "1",
            "5",
            "-neb_opt",
            "figures=off",
        ],
    );
    assert_eq!(adaptive.end, Some(ChainEnd::Converged));
    let coarse = config_from_cli(&[&input, "-os", "gfn0xtb", "-nd", "1.0"]).unwrap();
    assert_eq!(coarse.chain.added, None);
    assert!(adaptive.energies.len() >= 6, "{}", adaptive.energies.len());
    assert!(
        (barrier(&adaptive) - 68.285).abs() < 0.05,
        "{}",
        barrier(&adaptive)
    );
    let log = read(&adaptive.directory.join("mopt.log"));
    assert!(log.contains("images were added around the maxima (-aneb)"));
    // The Hessians in the run directory have the numbers of the images of the last chain.
    for image in 0..adaptive.energies.len() {
        let hessian = adaptive.directory.join(format!("tmp_hessian_{image}.npy"));
        assert_eq!(read_npy_matrix(&hessian).unwrap().nrows(), 9, "{image}");
    }
    assert!(!adaptive
        .directory
        .join(format!("tmp_hessian_{}.npy", adaptive.energies.len()))
        .exists());

    // One quasi-Newton step for the whole chain (MultiOptPy's -gqnt, which it reads and
    // does not use), with a climbing image instead of the redistribution.
    let whole = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "60",
            "-gqnt",
            "-adpred",
            "0",
            "-cineb",
            "3",
            "-neb_opt",
            "figures=off",
        ],
    );
    assert_eq!(whole.end, Some(ChainEnd::Converged));
    assert!(whole.iterations <= 30, "{}", whole.iterations);
    assert!(
        (barrier(&whole) - 68.285).abs() < 0.01,
        "{}",
        barrier(&whole)
    );
    assert!(read(&whole.directory.join("mopt.log")).contains("one for the whole chain"));

    // Constraints kept image by image and added images go together: an added image holds
    // the values interpolated at its place along the chain
    // (tests/constraints_matrix_nebmain.rs checks the values).
    let config = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-pc",
        "bond",
        "1,2",
        "-aneb",
        "1",
        "2",
        "-ns",
        "4",
        "-neb_opt",
        "figures=off",
    ])
    .unwrap();
    let together = run_with(&config, &dir, None, &CalculatorRegistry::native());
    assert!(together.is_ok(), "{together:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_and_fixed_atoms_are_kept_in_every_image() {
    let dir = scratch("constraints");
    let input = file(&dir, "hcn.xyz", HCN);
    // The C-N distance of every image is held at its value in the first chain (-pc), and
    // the steps along the forces alone are taken (-modelhess none).
    let held = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "6",
            "-modelhess",
            "none",
            "-adpred",
            "0",
            "-pc",
            "bond",
            "1,2",
            "-neb_opt",
            "figures=off",
        ],
    );
    let image = |result: &NebmainResult, iteration: usize, k: usize| {
        coordinates(
            &result
                .directory
                .join(format!("path_ITR_{iteration}_hcn"))
                .join(format!("hcn_{k}.xyz")),
        )
    };
    for k in 0..8 {
        let (before, after) = (
            distances(&image(&held, 0, k)),
            distances(&image(&held, 5, k)),
        );
        assert!(
            (before.0 - after.0).abs() < 1e-6,
            "image {k}: {} {}",
            before.0,
            after.0
        );
    }
    // Something moved all the same.
    let (before, after) = (
        distances(&image(&held, 0, 2)),
        distances(&image(&held, 5, 2)),
    );
    assert!((before.1 - after.1).abs() > 1e-3);
    assert!(read(&held.directory.join("mopt.log")).contains("Constraints: 1 per image"));

    // -fix: the atoms stay at their coordinates of the first chain.
    let frozen = run(
        &dir,
        &[
            &input,
            "-os",
            "gfn0xtb",
            "-ns",
            "5",
            "-fix",
            "1,2",
            "-modelhess",
            "none",
            "-neb_opt",
            "figures=off",
        ],
    );
    for k in 0..8 {
        let (before, after) = (image(&frozen, 0, k), image(&frozen, 4, k));
        assert!(before[..6]
            .iter()
            .zip(&after[..6])
            .all(|(a, b)| (a - b).abs() < 1e-9));
    }
    let moved = (0..8).any(|k| {
        let (before, after) = (image(&frozen, 0, k), image(&frozen, 4, k));
        before[6..]
            .iter()
            .zip(&after[6..])
            .any(|(a, b)| (a - b).abs() > 1e-3)
    });
    assert!(moved);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_is_interrupted_by_an_observer_and_restarted() {
    let dir = scratch("restart");
    let input = file(&dir, "hcn.xyz", HCN);
    let arguments = [
        input.as_str(),
        "-os",
        "gfn0xtb",
        "-ns",
        "6",
        "-modelhess",
        "none",
        "-neb_opt",
        "figures=off",
    ];
    let whole = run(&dir, &arguments);
    assert_eq!(whole.end, Some(ChainEnd::IterationLimit));
    assert_eq!(whole.iterations, 6);

    // Extra processing after every iteration: here, the run is ended after the third.
    let mut seen = Vec::new();
    let mut stop_after_three = |report: &ChainReport, _: &mut ChainRelaxation| {
        seen.push(report.iteration);
        Ok(if report.iteration == 2 {
            Flow::Stop
        } else {
            Flow::Continue
        })
    };
    let config = config_from_cli(&arguments).unwrap();
    let registry = CalculatorRegistry::native();
    let part =
        run_with_observers(&config, &dir, None, &registry, &mut [&mut stop_after_three]).unwrap();
    assert_eq!((part.iterations, part.end), (3, None));
    assert_eq!(seen, vec![0, 1, 2]);

    // Continued in its run directory: the evaluations of the first three iterations come
    // from the journal, and the result is that of the uninterrupted run.
    let mut continued = config_from_cli(&arguments).unwrap();
    continued.restart_dir = Some(part.directory.clone());
    let rest = run_with(&continued, &dir, None, &registry).unwrap();
    assert_eq!(rest.iterations, 6);
    assert_eq!(rest.directory, part.directory);
    for (a, b) in rest.energies.iter().zip(&whole.energies) {
        assert!((a - b).abs() < 1e-12, "{a} {b}");
    }
    let log = read(&rest.directory.join("mopt.log"));
    assert!(log.contains("Restart: 27 evaluations taken from"), "{log}");
    assert_eq!(
        read(&rest.directory.join("energy_plot.csv"))
            .lines()
            .count(),
        6
    );

    // A STOP file in the run directory interrupts the run after the iteration at hand.
    let mut marked = false;
    let mut leave_stop_file = |report: &ChainReport, _: &mut ChainRelaxation| {
        if report.iteration == 1 && !marked {
            marked = true;
            let run_dir = newest_run(&dir);
            std::fs::write(run_dir.join("STOP"), "").unwrap();
        }
        Ok(Flow::Continue)
    };
    let stopped =
        run_with_observers(&config, &dir, None, &registry, &mut [&mut leave_stop_file]).unwrap();
    assert!(stopped.interrupted);
    assert_eq!(stopped.iterations, 2);
    assert!(read(&stopped.directory.join("mopt.log")).contains("Continue with: nebmain --restart"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_chain_on_a_model_function_of_two_states() {
    let dir = scratch("mf");
    let input = file(&dir, "hcn.xyz", HCN);
    // The seam function of the ground state and the first excited singlet (the simplified
    // Tamm-Dancoff states of the built-in GFN0-xTB), the three structures as the images
    // (-nd 0), the ends held, steps along the forces only.
    let arguments = [
        input.as_str(),
        "-os",
        "gfn0xtb",
        "-mf",
        "seam",
        "-nd",
        "0",
        "-fe",
        "3",
        "-ns",
        "3",
        "-modelhess",
        "none",
        "-neb_opt",
        "figures=off",
    ];
    let config = config_from_cli(&arguments).unwrap();
    assert_eq!(config.model_function, vec!["seam".to_string()]);
    let registry = CalculatorRegistry::native();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    assert_eq!(result.energies.len(), 3);

    // Every image has the value of the function, (E1 + E2) / 2 + (E1 - E2)^2 / 0.05, of the
    // two states calculated here one by one.
    let last = read_xyz_trajectory_file(
        result.directory.join("path_ITR_2_hcn").join("hcn_path.xyz"),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    assert_eq!(last.frames.len(), 3);
    let options = mopt_rs::pes::registry::CalculatorOptions::default();
    let mut gaps = Vec::new();
    for (frame, on_the_chain) in last.frames.iter().zip(&result.energies) {
        let calculator = registry.create("gfn0xtb", &options, &frame.system).unwrap();
        let state = |root: usize| {
            let request = EvalRequest {
                excited_state: root,
                ..EvalRequest::ENERGY_GRAD
            };
            calculator
                .evaluate(&frame.system, &request)
                .unwrap()
                .energy
                .value()
        };
        let (ground, excited) = (state(0), state(1));
        let function = 0.5 * (ground + excited) + (ground - excited).powi(2) / 0.05;
        // The files have the structures with 12 decimals in Angstrom.
        assert!(
            (function - on_the_chain).abs() < 1e-8,
            "{function} {on_the_chain}"
        );
        gaps.push(excited - ground);
    }
    assert!(gaps.iter().all(|gap| *gap > 0.01), "{gaps:?}");

    // The image between the held ends went down on the function.
    let energies = read(&result.directory.join("energy_plot.csv"));
    let middle = |line: &str| -> f64 { line.split(',').nth(1).unwrap().trim().parse().unwrap() };
    let rows: Vec<&str> = energies.lines().collect();
    assert_eq!(rows.len(), 3);
    assert!(middle(rows[2]) < middle(rows[0]), "{energies}");
    let log = read(&result.directory.join("mopt.log"));
    assert!(log.contains("Mode: Model Function (seam)"), "{log}");
    // Two states per image and iteration.
    assert!(
        log.contains("Evaluations (this run): 18 gradient calculation(s)"),
        "{log}"
    );

    // A name that is no function is refused before anything is calculated, and plain
    // variables have no electronic states.
    let unknown = config_from_cli(&[&input, "-os", "gfn0xtb", "-mf", "no_such_function"]).unwrap();
    assert!(run_with(&unknown, &dir, None, &registry).is_err());
    let start = file(&dir, "start.txt", "x -0.558\ny 1.442\n");
    let end = file(&dir, "end.txt", "x 0.623\ny 0.028\n");
    let variables = config_from_cli(&[&start, &end, "-os", "mb2d", "-mf", "seam"]).unwrap();
    assert!(run_with(&variables, &dir, None, &registry).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_candidates_of_a_chain_go_on_to_optmain() {
    let dir = scratch("optimize");
    let input = file(&dir, "hcn.xyz", HCN);
    let registry = CalculatorRegistry::native();
    // Nothing without the setting.
    let plain = config_from_cli(&[&input, "-os", "gfn0xtb"]).unwrap();
    assert!(plain.follow_up.is_none());
    assert!(config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", "optimize=some"]).is_err());

    // A chain that is stopped early (a force of 2e-3), and optmain from its images: the
    // ends to minima, the image at the maximum to the transition state.
    let config = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-ns",
        "60",
        "-neb_opt",
        "tolerance=2e-3",
        "optimize=on",
        "figures=off",
    ])
    .unwrap();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    assert_eq!(result.end, Some(ChainEnd::Converged));
    let log = read(&result.directory.join("mopt.log"));
    assert_eq!(result.optimized.len(), 3, "{log}");
    let (first, saddle, last) = (
        &result.optimized[0],
        &result.optimized[1],
        &result.optimized[2],
    );
    assert!(
        first.converged && saddle.converged && last.converged,
        "{log}"
    );
    assert!(!first.saddle && saddle.saddle && !last.saddle);
    assert_eq!((first.image, last.image), (0, result.energies.len() - 1));
    // HCN, the transition state 68.285 kcal/mol above it, HNC 15.72 kcal/mol above it
    // (the values of optmain and of the chain relaxed to 1e-4 in the test above).
    let barrier = KCAL * (saddle.energy - first.energy);
    let reaction = KCAL * (last.energy - first.energy);
    assert!((barrier - 68.285).abs() < 0.005, "{barrier}");
    assert!((reaction - 15.72).abs() < 0.02, "{reaction}");
    println!(
        "chain {} iterations; optimizations {} {} {}",
        result.iterations, first.iterations, saddle.iterations, last.iterations
    );
    // The transition state started from the Hessian of the chain.
    assert!(result.directory.join("first_hessian_3.npy").is_file() || saddle.image != 3);
    let table = read(&result.directory.join("optimized_images.csv"));
    assert_eq!(table.lines().count(), 4, "{table}");
    assert!(table.contains(",saddle,true,"), "{table}");
    // The transition state was analysed: one imaginary frequency.
    assert_eq!(saddle.imaginary_modes, Some(1), "{table}");
    assert!(first.imaginary_modes.is_none() && last.imaginary_modes.is_none());
    // The file of each optimized structure, relative to the run directory of the chain.
    let row = table.lines().find(|l| l.contains(",saddle,")).unwrap();
    let optimized = row.rsplit(',').next().unwrap();
    assert!(Path::new(optimized).is_relative(), "{row}");
    assert!(result.directory.join(optimized).is_file(), "{row}");
    assert!(log.contains("to a transition state: converged"), "{log}");
    assert!(
        log.contains("Evaluations (the optimizations after the chain)"),
        "{log}"
    );
    assert!(saddle.optimized.is_file());

    // Only the saddle, from the Hessian of the calculator.
    let config = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-ns",
        "60",
        "-neb_opt",
        "tolerance=2e-3",
        "optimize=saddles",
        "optimize_hessian=exact",
        "figures=off",
    ])
    .unwrap();
    let exact = run_with(&config, &dir, None, &registry).unwrap();
    assert_eq!(exact.optimized.len(), 1);
    assert!(exact.optimized[0].converged && exact.optimized[0].saddle);
    assert!((exact.optimized[0].energy - saddle.energy).abs() < 1e-6);
    println!(
        "saddle from the exact Hessian: {} iterations",
        exact.optimized[0].iterations
    );

    // Plain variables have no such step yet.
    let start = file(&dir, "start.txt", "x -0.558\ny 1.442\n");
    let end = file(&dir, "end.txt", "x 0.623\ny 0.028\n");
    let variables =
        config_from_cli(&[&start, &end, "-os", "mb2d", "-neb_opt", "optimize=on"]).unwrap();
    assert!(run_with(&variables, &dir, None, &registry).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
