//! The kinetics at the end of a run of `run_mapper` (`-map_opt kinetics=on`) on the
//! Lennard-Jones cluster of seven argon atoms with the built-in `lj` calculator: its barriers
//! of about one epsilon (0.24 kcal/mol) are crossed in picoseconds at 300 K, so after the time
//! of interest (1 s) the populations of the minima joined to the start are their Boltzmann
//! weights `exp(-E_i / k_B T)` (Eyring rates of the electronic energies: the prefactors cancel
//! in the equilibrium constants).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mopt_rs::cli::mapper::{run_with, MapperConfig};
use mopt_rs::cli::mapper_kinetics::{KINETICS_DIR, SUMMARY_FILE};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::Network;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mopt_mapper_kinetics_{name}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// LJ7 near its lowest minimum (Bohr).
const AR7: [[f64; 3]; 7] = [
    [6.023853759546511, 0.0, -0.15549746403767273],
    [1.962006769746685, 5.809518090796886, -0.061011157764781325],
    [-4.854674003698997, 3.590479638369873, 0.03347514850811009],
    [-4.999838497421851, -3.5904796383698727, 0.12796145478100152],
    [1.6599768546465556, -5.809518090796886, 0.2224477610538929],
    [-0.08463505395523435, 0.0, 3.794030138169188],
    [0.2933101711363313, 0.0, -3.961405880709738],
];

const CONFIG: &str = r#"{
  "mapper_settings": {
    "afir_gamma_kJmol": 2.0,
    "dist_upper_ang": 7.0,
    "max_iterations": 3,
    "rng_seed": 42,
    "channels": false
  },
  "top_n_candidates": 2,
  "step1_settings": {"othersoft": "lj", "NSTEP": 300},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 200},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "calc_exact_hess": 1,
    "NSTEP": 400,
    "intrinsic_reaction_coordinates": ["0.5", "200", "lqa"]
  }
}"#;

fn input(dir: &Path) -> PathBuf {
    let mut text = String::from("7\n0 1\n");
    for p in AR7 {
        text.push_str(&format!(
            "Ar {:.10} {:.10} {:.10}\n",
            p[0] * Units::BOHR_TO_ANGSTROM,
            p[1] * Units::BOHR_TO_ANGSTROM,
            p[2] * Units::BOHR_TO_ANGSTROM
        ));
    }
    let path = dir.join("ar7.xyz");
    std::fs::write(&path, text).unwrap();
    path
}

fn config(dir: &Path, extra: &[&str]) -> MoptResult<MapperConfig> {
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, CONFIG).unwrap();
    let mut args = vec![
        input(dir).display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args)?;
    MapperConfig::from_args(&parsed, None)
}

fn summary(run_dir: &Path) -> JsonValue {
    let path = run_dir.join(KINETICS_DIR).join(SUMMARY_FILE);
    JsonValue::parse(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

/// The populations at the time of interest by node.
fn last_populations(s: &JsonValue) -> BTreeMap<usize, f64> {
    s.get("states")
        .and_then(JsonValue::as_array)
        .unwrap()
        .iter()
        .map(|st| {
            let node = st.get("node").and_then(JsonValue::as_usize).unwrap();
            let p = st.get("populations").and_then(JsonValue::as_array).unwrap();
            (node, p.last().and_then(JsonValue::as_f64).unwrap())
        })
        .collect()
}

/// The minima joined to `start` by transition-state edges.
fn component(net: &Network, start: usize) -> BTreeSet<usize> {
    let mut seen = BTreeSet::from([start]);
    loop {
        let before = seen.len();
        for e in &net.ts_edges {
            if seen.contains(&e.ends[0]) || seen.contains(&e.ends[1]) {
                seen.extend(e.ends);
            }
        }
        if seen.len() == before {
            return seen;
        }
    }
}

#[test]
fn the_kinetics_at_the_end_reaches_the_boltzmann_weights_of_lj7() {
    let dir = scratch("boltzmann");
    let c = config(&dir, &["--output_dir", "run", "-map_opt", "kinetics=on"]).unwrap();
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(!run.interrupted);
    let net = &run.network;
    assert!(
        !net.ts_edges.is_empty(),
        "the test needs a transition state"
    );
    let s = summary(&run.directory);
    assert_eq!(s.get("computed").and_then(JsonValue::as_bool), Some(true));
    assert_eq!(
        s.get("jobs").and_then(JsonValue::as_usize),
        Some(net.jobs.len())
    );
    let times = s.get("times_s").and_then(JsonValue::as_array).unwrap();
    assert_eq!(times.len(), 13);
    assert_eq!(times.last().and_then(JsonValue::as_f64), Some(1.0));
    // Every output time keeps the whole population.
    let states = s.get("states").and_then(JsonValue::as_array).unwrap();
    for k in 0..times.len() {
        let total: f64 = states
            .iter()
            .map(|st| {
                st.get("populations").and_then(JsonValue::as_array).unwrap()[k]
                    .as_f64()
                    .unwrap()
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-8, "time {k}: {total}");
    }
    // At 1 s: the Boltzmann weights of the minima joined to the start, nothing elsewhere.
    let start = net.inputs[0].node.unwrap();
    let joined = component(net, start);
    let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
    let e0 = joined
        .iter()
        .map(|&id| net.nodes[id].energy)
        .fold(f64::INFINITY, f64::min);
    let z: f64 = joined
        .iter()
        .map(|&id| (-(net.nodes[id].energy - e0) / kt).exp())
        .sum();
    let last = last_populations(&s);
    for (&id, &p) in &last {
        let want = if joined.contains(&id) {
            (-(net.nodes[id].energy - e0) / kt).exp() / z
        } else {
            0.0
        };
        assert!(
            (p - want).abs() < 1e-6,
            "node {id}: {p} against the Boltzmann weight {want}"
        );
    }
    // The files of run_kinetics and the section of the report.
    for f in ["populations.csv", "rates.csv", "priority.csv"] {
        assert!(run.directory.join(KINETICS_DIR).join(f).is_file(), "{f}");
    }
    let page = std::fs::read_to_string(run.directory.join("mapper_report.html")).unwrap();
    assert!(
        page.contains("id=\"kinetics\""),
        "the report shows the kinetics"
    );
    assert!(page.contains("Populations against time"));
    let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Kinetics at the end"), "{log}");
}

#[test]
fn kinetic_settings_without_a_use_are_refused() {
    let dir = scratch("refused");
    let c = config(&dir, &["-map_opt", "kinetics_time=1e-3"]).unwrap();
    let e = run_with(&c, &dir, None, &CalculatorRegistry::native())
        .err()
        .unwrap();
    assert!(e.to_string().contains("kinetics=on"), "{e}");
    // An unknown solver is refused before anything runs.
    let c = config(&dir, &["-map_opt", "kinetics=on", "kinetics_solver=nope"]).unwrap();
    assert!(run_with(&c, &dir, None, &CalculatorRegistry::native()).is_err());
    assert!(!dir
        .join("mapper_output")
        .join("reaction_network.json")
        .exists());
}

/// `lj` that writes the file `stop` at its evaluation `at` (counted from 0).
struct StopAt {
    count: Arc<AtomicUsize>,
    at: usize,
    stop: PathBuf,
}

impl Calculator for StopAt {
    fn name(&self) -> &str {
        "lj"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if self.count.fetch_add(1, Ordering::SeqCst) == self.at {
            std::fs::write(&self.stop, "").unwrap();
        }
        LennardJones.evaluate(sys, req)
    }
}

#[test]
fn an_interrupted_run_has_its_kinetics_after_the_restart() {
    let dir = scratch("restart");
    let native = CalculatorRegistry::native();
    let whole = run_with(
        &config(&dir, &["--output_dir", "whole", "-map_opt", "kinetics=on"]).unwrap(),
        &dir,
        None,
        &native,
    )
    .unwrap();
    let reference = last_populations(&summary(&whole.directory));
    // Stopped at several points of the run: no kinetics then, the same kinetics after the
    // restart.
    for at in [40, 260, 700] {
        let name = format!("stopped_{at}");
        let run_dir = dir.join(&name);
        let mut stopping = CalculatorRegistry::native();
        let count = Arc::new(AtomicUsize::new(0));
        let (c, stop) = (Arc::clone(&count), run_dir.join("STOP"));
        stopping.register("lj", move |_| {
            Ok(Box::new(StopAt {
                count: Arc::clone(&c),
                at,
                stop: stop.clone(),
            }) as Box<dyn Calculator>)
        });
        let stopped = run_with(
            &config(&dir, &["--output_dir", &name, "-map_opt", "kinetics=on"]).unwrap(),
            &dir,
            None,
            &stopping,
        )
        .unwrap();
        assert!(stopped.interrupted, "{at}: {:?}", stopped.reason);
        assert!(
            !run_dir.join(KINETICS_DIR).join(SUMMARY_FILE).exists(),
            "{at}: an interrupted run has no kinetics"
        );
        let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
        let again = MapperConfig::from_args(&settings, Some(&run_dir)).unwrap();
        let done = run_with(&again, &dir, None, &native).unwrap();
        assert!(!done.interrupted);
        let got = last_populations(&summary(&run_dir));
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            reference.keys().collect::<Vec<_>>(),
            "{at}"
        );
        for (id, p) in &got {
            assert!((p - reference[id]).abs() < 1e-9, "{at}: node {id}");
        }
    }
}
