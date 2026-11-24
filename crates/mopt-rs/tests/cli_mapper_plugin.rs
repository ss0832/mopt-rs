//! `run_mapper` with an exploration method of one's own that only proposes
//! (`workflow::mapper::plugin`): on the Muller-Brown surface a proposer gives, from the first
//! minimum, a guess near the saddle point next to it and a first path towards the middle
//! minimum; the mapper optimizes the guess to the saddle point, relaxes the path, and joins the
//! minima by the IRCs. Its methods are found by a prefix of the registry, as the Python package
//! finds the methods of Python classes (`method=python:MODULE:CLASS`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use faer::Col;
use mopt_rs::cli::mapper::{run_with_parts, MapperConfig, MapperParts};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::error::MoptResult;
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::mapper::plugin::{Proposal, ProposalMethod, Proposer};
use mopt_rs::workflow::mapper::{
    Configuration, ExplorationMethod, Job, MethodSettings, Network, Node, NodeKind, Task,
};

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mopt_mapper_plugin_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One task per minimum; from the first minimum a guess beside the saddle point at
/// (-0.82, 0.62) and a straight path to beside the middle minimum.
struct NearSaddle {
    name: String,
}

impl Proposer for NearSaddle {
    fn name(&self) -> &str {
        &self.name
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        Ok(vec![Task {
            key: format!("near_saddle:{}", node.id),
            description: "a guess and a path".to_string(),
            parameters: JsonValue::Object(BTreeMap::new()),
        }])
    }

    fn propose(&self, job: &Job) -> MoptResult<Proposal> {
        let Configuration::Variables(start) = &job.start else {
            return Ok(Proposal::default());
        };
        let at = |x: f64, y: f64| {
            Configuration::variables(Col::from_fn(2, |i| [x, y][i]), Arc::clone(&start.names))
        };
        if (start.values[1] - MB_MINIMA[0][1]).abs() > 0.1 {
            return Ok(Proposal {
                notes: vec!["not the first minimum: nothing to propose".to_string()],
                ..Proposal::default()
            });
        }
        let (sx, sy) = (MB_SADDLES[0][0] + 0.03, MB_SADDLES[0][1] - 0.02);
        let path = (0..=8)
            .map(|k| {
                let t = f64::from(k) / 8.0;
                at(
                    start.values[0] + t * (MB_MINIMA[2][0] - 0.05 - start.values[0]),
                    start.values[1] + t * (MB_MINIMA[2][1] + 0.05 - start.values[1]),
                )
            })
            .collect::<MoptResult<Vec<_>>>()?;
        Ok(Proposal {
            guesses: vec![at(sx, sy)?],
            paths: vec![path],
            notes: vec!["a guess and a path".to_string()],
        })
    }
}

#[test]
fn a_proposer_of_one_s_own_finds_the_saddle_points_it_points_to() {
    let dir = scratch("mb2d");
    let input = dir.join("start.txt");
    std::fs::write(&input, "x -0.55\ny 1.44\n").unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"max_iterations": 3, "rng_seed": 1, "method": "mine:near_saddle"},
  "step1_settings": {"othersoft": "mb2d", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let parsed = parser("run_mapper", RUN_MAPPER)
        .parse([
            input.display().to_string(),
            "-cfg".to_string(),
            cfg.display().to_string(),
        ])
        .unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    let mut parts = MapperParts::default();
    // Every name `mine:NAME` is a proposer of that name.
    parts.methods.register_prefix(
        "mine:",
        Arc::new(|name: &str, s: &MethodSettings| {
            Ok(Box::new(ProposalMethod::new(
                Box::new(NearSaddle {
                    name: name.to_string(),
                }),
                "neb",
                Vec::new(),
                s.top_n,
            )) as Box<dyn ExplorationMethod>)
        }),
    );
    let run = run_with_parts(
        &c,
        &dir,
        None,
        &CalculatorRegistry::native(),
        &parts,
        &mut [],
    )
    .unwrap();
    let net = &run.network;
    eprintln!(
        "proposer on mb2d: {} minima, {} saddle points, {} jobs, warnings {:?}",
        net.minima().count(),
        net.of_kind(&NodeKind::TransitionState).count(),
        net.jobs.len(),
        net.warnings
    );
    // The saddle point the guess points to, and the minima on both sides of it.
    let near = |n: &Node, p: &[f64; 2]| {
        let x = n.system.flat();
        ((x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2)).sqrt() < 1e-3
    };
    assert!(
        net.of_kind(&NodeKind::TransitionState)
            .any(|n| near(n, &MB_SADDLES[0])),
        "{:?}",
        net.warnings
    );
    assert!(net.minima().any(|n| near(n, &MB_MINIMA[2])));
    assert!(net
        .ts_edges
        .iter()
        .all(|e| e.ends[0] != e.ends[1] && e.barriers.iter().all(|b| *b > 0.0)));
    // A name that no prefix and no method takes is refused with the names there are.
    let mut bad = c.clone();
    bad.settings.method = "nobody:x".to_string();
    bad.output_dir = PathBuf::from("refused_run");
    let e = run_with_parts(
        &bad,
        &dir,
        None,
        &CalculatorRegistry::native(),
        &parts,
        &mut [],
    )
    .err()
    .unwrap()
    .to_string();
    assert!(e.contains("mine:"), "{e}");
}
