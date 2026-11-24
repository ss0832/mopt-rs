//! `run_mapper -map_opt ts_conformers=on`: the conformers of the transition states of a network
//! are searched with the saddle direction held (the search of `confsearch -tsconf`), and every
//! saddle point found is a transition state of the network with its own IRC.
//!
//! The molecule is 2-ethyl-1,5-hexadiene on GFN0-xTB, as in the test of `confsearch -tsconf`:
//! the chair transition state of its Cope rearrangement with the ethyl group in its higher
//! position is given to the mapper as a guess by a method of the test, and the conformer
//! search turns the group to the transition state 0.78 kcal/mol lower.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faer::Col;
use mopt_rs::cli::mapper::{run_with_parts, MapperConfig, MapperParts, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Units};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::{
    follow_guesses, ExplorationMethod, Guess, Job, JobContext, JobOutcome, MethodSettings, Network,
    Node, NodeKind, Task, TS_CONFORMER_PREFIX,
};

/// A minimum next to the transition state below (GFN0-xTB): the diene with the bond between
/// the atoms 1 and 9 made and that between the atoms 3 and 11 broken.
const MINIMUM: &str = "22\n0 1\n\
C -0.663426 -1.079462 0.576825\nC 0.090718 0.101704 1.121908\nC -0.509402 1.250301 1.372131\n\
C 1.561973 -0.082430 1.357804\nH -1.638511 -1.146102 1.063460\nH -0.105662 -1.995706 0.770999\n\
H 0.033682 2.107044 1.751581\nH -1.571626 1.388028 1.213709\nC -0.901409 -0.956285 -0.937125\n\
C -1.789450 0.202077 -1.253840\nC -1.423822 1.217839 -2.006081\nH -2.774491 0.166191 -0.799715\n\
H -1.372977 -1.878938 -1.280624\nH 0.053350 -0.837308 -1.450276\nH -0.438158 1.250953 -2.454810\n\
H -2.077178 2.053836 -2.219406\nC 2.322035 -0.362054 0.064488\nH 1.710272 -0.925824 2.035756\n\
H 1.969880 0.817299 1.822529\nH 3.388652 -0.436747 0.270464\nH 2.153539 0.445281 -0.647442\n\
H 1.982010 -1.299698 -0.372332\n";

/// 2-ethyl-1,5-hexadiene at a chair transition state of its Cope rearrangement, the ethyl
/// group in its higher position (GFN0-xTB; the structure of `tests/cli_confsearch.rs`).
const ETHYL_COPE_TS: [[f64; 3]; 22] = [
    [-0.670891, -1.140282, 0.707891],
    [-0.018031, 0.069167, 1.021164],
    [-0.742897, 1.255670, 0.789091],
    [1.480696, 0.123450, 1.119992],
    [-1.695899, -1.228954, 1.051561],
    [-0.102712, -2.052829, 0.855736],
    [-0.236818, 2.186905, 1.022052],
    [-1.780232, 1.255681, 1.107114],
    [-0.966215, -1.090412, -1.055793],
    [-1.649737, 0.119372, -1.297169],
    [-0.963978, 1.310132, -0.976963],
    [-2.728728, 0.118383, -1.180740],
    [-1.511408, -2.009559, -1.240122],
    [0.048555, -1.134783, -1.433086],
    [0.065364, 1.362096, -1.316915],
    [-1.494970, 2.242987, -1.128999],
    [2.212228, -0.421121, -0.105863],
    [1.790507, -0.465669, 1.986976],
    [1.788829, 1.158068, 1.281889],
    [3.287975, -0.340669, 0.042135],
    [1.933329, 0.150209, -0.989993],
    [1.955035, -1.467843, -0.259956],
];

/// One task for the minimum of the input: the transition state above as a guess.
struct GivenTs;

impl ExplorationMethod for GivenTs {
    fn name(&self) -> &str {
        "given_ts"
    }

    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        let input = network.inputs.iter().any(|i| i.node == Some(node.id));
        Ok(if input && node.kind == NodeKind::Minimum {
            vec![Task {
                key: "given".to_string(),
                description: "the Cope transition state as a guess".to_string(),
                parameters: JsonValue::Null,
            }]
        } else {
            Vec::new()
        })
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let x = Col::from_fn(66, |k| {
            ETHYL_COPE_TS[k / 3][k % 3] * Units::ANGSTROM_TO_BOHR
        });
        let guess = Guess {
            system: job.start.with_flat(&x)?,
            origin: "the structure of the test".to_string(),
        };
        let mut outcome = JobOutcome::default();
        if !follow_guesses(job, ctx, &[guess], &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

/// The settings: an IRC of larger steps (its ends are minimized afterwards) and one AFIR sample
/// of the search, between the end of the ethyl group and a terminal carbon, keep the test short.
const CONFIG: &str = r#"{
  "mapper_settings": {"rng_seed": 1, "ts_conformers": true},
  "step1_settings": {"othersoft": "gfn0xtb", "NSTEP": 300},
  "step3_settings": {"NSTEP": 150},
  "step4_settings": {"NSTEP": 300, "intrinsic_reaction_coordinates": ["0.3", "100", "lqa"]},
  "ts_conformer_settings": {
    "max_samples": 1,
    "target_atoms": "1,17",
    "intrinsic_reaction_coordinates": ["0.3", "100", "lqa"],
    "confsearch_options": ["pairs=ratio"]
  }
}"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mopt_mapper_conformers_{name}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn parts() -> MapperParts {
    let mut parts = MapperParts::default();
    parts.methods.register(
        "given_ts",
        Arc::new(|_: &MethodSettings| Ok(Box::new(GivenTs) as Box<dyn ExplorationMethod>)),
    );
    parts
}

fn config(dir: &Path, out: &str, extra: &[&str]) -> MapperConfig {
    let input = dir.join("diene.xyz");
    std::fs::write(&input, MINIMUM).unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, CONFIG).unwrap();
    let mut args = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--max_iter".to_string(),
        "6".to_string(),
        "--output_dir".to_string(),
        out.to_string(),
        "-map_opt".to_string(),
        "method=given_ts".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

/// GFN0-xTB whose evaluations are counted; it writes the file `stop` at the evaluation `at`.
struct StopAt {
    inner: Box<dyn Calculator>,
    count: Arc<AtomicUsize>,
    at: usize,
    stop: PathBuf,
}

impl Calculator for StopAt {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn analytic_hessian(&self) -> bool {
        self.inner.analytic_hessian()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if self.count.fetch_add(1, Ordering::SeqCst) == self.at {
            std::fs::write(&self.stop, "").unwrap();
        }
        self.inner.evaluate(sys, req)
    }
}

/// The registry with `gfn0xtb` counted into `count`, the file `stop` written at `at`.
fn registry(count: &Arc<AtomicUsize>, at: usize, stop: PathBuf) -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    let count = Arc::clone(count);
    r.register("gfn0xtb", move |q| {
        let inner = CalculatorRegistry::native().create_uncounted(
            "gfn0xtb",
            &q.options.clone(),
            q.system,
        )?;
        Ok(Box::new(StopAt {
            inner,
            count: Arc::clone(&count),
            at,
            stop: stop.clone(),
        }) as Box<dyn Calculator>)
    });
    r
}

fn kcal(a: f64, b: f64) -> f64 {
    (a - b) * Units::HARTREE_TO_KCAL_MOL
}

fn check(run: &MapperRun, what: &str) -> f64 {
    let net = &run.network;
    // Two jobs: the guess, and the conformer search of the transition state it gave. The
    // transition state of the search is not searched again, and no other task is left.
    assert_eq!(net.jobs.len(), 2, "{what}: {:?}", net.jobs);
    assert_eq!(net.jobs[0].task, "given", "{what}");
    assert!(
        net.jobs[1].task.starts_with(TS_CONFORMER_PREFIX),
        "{what}: {}",
        net.jobs[1].task
    );
    // The two Cope saddles have a reaction mode near -476 cm^-1. A lower-energy
    // torsional saddle can also be reached at a shoulder of one of their IRCs.
    let cope: Vec<_> = net
        .ts_edges
        .iter()
        .filter(|e| e.imaginary_wavenumber.is_some_and(|w| w < -100.0))
        .collect();
    assert_eq!(cope.len(), 2, "{what}: {:?}", net.warnings);
    let ts = [&net.nodes[cope[0].ts], &net.nodes[cope[1].ts]];
    assert_eq!(ts[0].origin.job, Some(0), "{what}");
    assert_eq!(ts[1].origin.job, Some(1), "{what}");
    // The conformer of the search lies 0.78 kcal/mol below the transition state given.
    let gap = kcal(ts[0].energy, ts[1].energy);
    assert!(gap > 0.5 && gap < 1.2, "{what}: {gap}");
    assert_eq!(
        net.ts_edges.len(),
        net.of_kind(&NodeKind::TransitionState).count()
    );
    for e in &net.ts_edges {
        assert!(e.imaginary_wavenumber.is_some_and(|w| w < 0.0), "{what}");
        if !cope.iter().any(|c| c.ts == e.ts) {
            assert!(
                cope.iter().any(|c| c.ends.contains(&e.ts)),
                "{what}: unlinked saddle"
            );
        }
        let pairs = net.joined_by(e);
        assert!(
            !pairs.is_empty(),
            "{what}: the IRC has no verified products"
        );
        assert!(pairs
            .iter()
            .flatten()
            .all(|&n| net.nodes[n].kind == NodeKind::Minimum));
    }
    assert_ne!(net.joined_by(cope[0]), net.joined_by(cope[1]), "{what}");
    gap
}

#[test]
fn the_conformers_of_a_transition_state_are_transition_states_of_the_network() {
    let dir = scratch("cope");
    let count = Arc::new(AtomicUsize::new(0));
    let whole = run_with_parts(
        &config(&dir, "whole", &[]),
        &dir,
        None,
        &registry(&count, usize::MAX, dir.join("unused")),
        &parts(),
        &mut [],
    )
    .unwrap();
    assert!(!whole.interrupted);
    let gap = check(&whole, "whole");
    let total = count.load(Ordering::SeqCst);
    let log = std::fs::read_to_string(whole.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("the search of confsearch -tsconf") && log.contains("Reactive bonds held"),
        "{log}"
    );
    eprintln!(
        "ethyl Cope: {total} evaluations ({:?}); the conformer lies {gap:.3} kcal/mol lower",
        whole.network.evaluations
    );
    // A STOP during the conformer search (well into the second job), then the restart: the
    // network of the run that was not stopped.
    let jobs = std::fs::read_to_string(whole.directory.join("jobs").join("job_000001.log"))
        .unwrap_or_default();
    assert!(jobs.contains("Reaction coordinate held"), "{jobs}");
    let run_dir = dir.join("stopped");
    let at = total * 3 / 4;
    let first = run_with_parts(
        &config(&dir, "stopped", &[]),
        &dir,
        None,
        &registry(&Arc::new(AtomicUsize::new(0)), at, run_dir.join("STOP")),
        &parts(),
        &mut [],
    )
    .unwrap();
    assert!(first.interrupted, "a STOP at evaluation {at} of {total}");
    assert_eq!(first.network.jobs.len(), 1, "the search was in flight");
    let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
    let again = run_with_parts(
        &MapperConfig::from_args(&settings, Some(&run_dir)).unwrap(),
        &dir,
        None,
        &registry(
            &Arc::new(AtomicUsize::new(0)),
            usize::MAX,
            dir.join("unused"),
        ),
        &parts(),
        &mut [],
    )
    .unwrap();
    assert!(!again.interrupted);
    check(&again, "restart");
    assert_eq!(again.network.jobs, whole.network.jobs);
    assert_eq!(again.network.nodes.len(), whole.network.nodes.len());
    for (a, b) in again.network.nodes.iter().zip(&whole.network.nodes) {
        assert_eq!((a.kind, a.origin), (b.kind, b.origin), "{}", a.label());
        assert!(
            (a.energy - b.energy).abs() < 1e-9,
            "{}: {} against {}",
            a.label(),
            a.energy,
            b.energy
        );
    }
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(
        log.contains("evaluations taken from evaluations.journal"),
        "{log}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_settings_of_the_search_are_checked() {
    let dir = scratch("settings");
    let input = dir.join("diene.xyz");
    std::fs::write(&input, MINIMUM).unwrap();
    let from = |text: &str, extra: &[&str]| {
        let cfg = dir.join("config.json");
        std::fs::write(&cfg, text).unwrap();
        let mut args = vec![
            input.display().to_string(),
            "-cfg".to_string(),
            cfg.display().to_string(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
        MapperConfig::from_args(&parsed, None)
    };
    // The block is read with the surface of step 1 and the seed of the run; the IRC inside
    // the search is off.
    let c = from(CONFIG, &[]).unwrap();
    let search = c.steps.ts_conformers.as_deref().unwrap();
    assert_eq!(search.optmain.calculator, "gfn0xtb");
    assert_eq!(search.common.seed, 1);
    assert_eq!(search.afir.max_samples, 1);
    assert!(search.ts.is_some_and(|t| !t.check_reaction));
    // Without the setting the block is not used, and the notes say so.
    let off = from(
        &CONFIG.replace("\"ts_conformers\": true", "\"ts_conformers\": false"),
        &[],
    )
    .unwrap();
    assert!(off.steps.ts_conformers.is_none());
    assert!(
        off.notes
            .iter()
            .any(|n| n.contains("ts_conformer_settings")),
        "{:?}",
        off.notes
    );
    // Another surface in the block, several states, a remap.
    let other = CONFIG.replace(
        "\"max_samples\": 1,",
        "\"max_samples\": 1, \"othersoft\": \"gfn1xtb\",",
    );
    let e = from(&other, &[]).unwrap_err().to_string();
    assert!(e.contains("same surface"), "{e}");
    let e = from(CONFIG, &["-map_opt", "states=0,1;0,3"])
        .unwrap_err()
        .to_string();
    assert!(e.contains("several states"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}
