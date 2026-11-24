//! `run_mapper` on objectives of plain variables: the Muller-Brown surface of two variables
//! (`mb2d`: three minima, two first-order saddle points) and the Rosenbrock function of four
//! variables (two minima, one saddle point between them), explored by ADDF, the default for
//! plain variables.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::mapper::{run_with, MapperConfig, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::pes::model_objectives::{ROSENBROCK_4_POINTS, WOLFE_QUAPP_POINTS};
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::mapper::{Network, NodeKind};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mapper_vars_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config(objective: &str, jobs: usize) -> String {
    format!(
        r#"{{
  "mapper_settings": {{"max_iterations": {jobs}, "rng_seed": 1}},
  "step1_settings": {{"othersoft": "{objective}", "NSTEP": 300}},
  "step3_settings": {{"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300}},
  "step4_settings": {{
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]
  }}
}}"#
    )
}

fn run(dir: &Path, start: &str, objective: &str, jobs: usize, extra: &[&str]) -> MapperRun {
    let input = dir.join("start.txt");
    std::fs::write(&input, start).unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, config(objective, jobs)).unwrap();
    let mut args = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    run_with(&c, dir, None, &CalculatorRegistry::native()).unwrap()
}

/// Every known point of `kind` has a node of that kind within `tolerance`.
fn found_all(net: &Network, kind: &NodeKind, known: &[&[f64]], tolerance: f64) -> Vec<String> {
    let mut missing = Vec::new();
    for p in known {
        let hit = net.of_kind(kind).any(|n| {
            let x = n.system.flat();
            x.nrows() == p.len()
                && x.iter()
                    .zip(p.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
                    < tolerance
        });
        if !hit {
            missing.push(format!("{p:?}"));
        }
    }
    missing
}

fn summary(net: &Network) -> String {
    net.nodes
        .iter()
        .map(|n| {
            format!(
                "{} {:?} {:.6}",
                n.label(),
                n.system.flat().iter().copied().collect::<Vec<_>>(),
                n.energy
            )
        })
        .chain(
            net.warnings
                .iter()
                .map(|w| format!("warning: {}", w.message)),
        )
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_muller_brown_network_has_its_three_minima_and_two_saddles() {
    let dir = scratch("mb2d");
    let run = run(&dir, "x -0.55\ny 1.44\n", "mb2d", 12, &[]);
    let net = &run.network;
    let minima: Vec<&[f64]> = MB_MINIMA.iter().map(|p| &p[..]).collect();
    let saddles: Vec<&[f64]> = MB_SADDLES.iter().map(|p| &p[..]).collect();
    let missing = found_all(net, &NodeKind::Minimum, &minima, 1e-3);
    assert!(missing.is_empty(), "minima {missing:?}\n{}", summary(net));
    let missing = found_all(net, &NodeKind::TransitionState, &saddles, 1e-3);
    assert!(missing.is_empty(), "saddles {missing:?}\n{}", summary(net));
    assert_eq!(net.minima().count(), 3, "{}", summary(net));
    assert_eq!(
        net.of_kind(&NodeKind::TransitionState).count(),
        2,
        "{}",
        summary(net)
    );
    // Every edge joins two different minima below its saddle point.
    for e in &net.ts_edges {
        assert_ne!(e.ends[0], e.ends[1]);
        assert!(e.barriers.iter().all(|b| *b > 0.0));
    }
    // The network keeps its variables; the files are variables files and frames of them.
    assert_eq!(
        net.variables.as_deref(),
        Some(&["x".to_string(), "y".to_string()][..])
    );
    let eq0 = std::fs::read_to_string(run.directory.join("structures/EQ000000.txt")).unwrap();
    assert!(eq0.starts_with("# EQ0 energy=") && eq0.contains("\nx ") && eq0.contains("\ny "));
    let doc = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    assert!(doc.contains("\"variables\""));
    assert!(run.directory.join("paths/TS000000_irc.txt").is_file());
    assert!(run.directory.join("mapper_report.html").is_file());
    // AFIR pushes need atoms: asking for them is refused with the reason.
    let other = scratch("mb2d_afir");
    let input = other.join("start.txt");
    std::fs::write(&input, "x -0.55\ny 1.44\n").unwrap();
    let cfg = other.join("config.json");
    std::fs::write(&cfg, config("mb2d", 2)).unwrap();
    let parsed = parser("run_mapper", RUN_MAPPER)
        .parse([
            input.display().to_string(),
            "-cfg".to_string(),
            cfg.display().to_string(),
            "-map_opt".to_string(),
            "method=afir_ts".to_string(),
        ])
        .unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    let err = run_with(&c, &other, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(err.to_string().contains("needs atoms"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&other);
}

#[test]
fn four_variables_of_rosenbrock_and_the_wolfe_quapp_surface() {
    let dir = scratch("rosenbrock");
    let first = run(&dir, "1.0\n1.0\n1.0\n1.0\n", "rosenbrock", 8, &[]);
    let net = &first.network;
    let minima: Vec<&[f64]> = ROSENBROCK_4_POINTS
        .iter()
        .filter(|p| p.index == 0)
        .map(|p| &p.at[..])
        .collect();
    let saddles: Vec<&[f64]> = ROSENBROCK_4_POINTS
        .iter()
        .filter(|p| p.index == 1)
        .map(|p| &p.at[..])
        .collect();
    let missing = found_all(net, &NodeKind::Minimum, &minima, 1e-3);
    assert!(missing.is_empty(), "minima {missing:?}\n{}", summary(net));
    let missing = found_all(net, &NodeKind::TransitionState, &saddles, 1e-3);
    assert!(missing.is_empty(), "saddles {missing:?}\n{}", summary(net));
    let _ = std::fs::remove_dir_all(&dir);

    let dir = scratch("wolfe_quapp");
    let second = run(&dir, "x -1.17\ny 1.48\n", "wolfe_quapp", 12, &[]);
    let net = &second.network;
    let minima: Vec<&[f64]> = WOLFE_QUAPP_POINTS
        .iter()
        .filter(|p| p.index == 0)
        .map(|p| &p.at[..])
        .collect();
    let missing = found_all(net, &NodeKind::Minimum, &minima, 1e-3);
    assert!(missing.is_empty(), "minima {missing:?}\n{}", summary(net));
    let _ = std::fs::remove_dir_all(&dir);
}
