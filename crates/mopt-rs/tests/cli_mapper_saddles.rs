//! Saddle points of higher order in `run_mapper` (`higher_saddles=on`), on a surface of three
//! plain variables where the answer is known:
//!
//! `V = (x^2 - 1)^2 + cos(pi x) (y^2 + z^2) + y^4 + z^4 + 0.5 y^2 z^2`
//!
//! The transition state at the origin (V = 1) has its unstable mode along x, and its IRC runs
//! along y = z = 0 into the index-2 saddle points (+-1, 0, 0) (V = 0, curvatures -2 along y
//! and z). Below each lie four transition states (+-1, +-0.7071, 0) and (+-1, 0, +-0.7071)
//! (V = -0.25) and four minima (+-1, +-0.6325, +-0.6325) (V = -0.4): 8 minima, 9 transition
//! states, 2 saddle points of order 2 in all (Newton's method from a grid).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::Col;
use mopt_rs::cli::mapper::{run_with_parts, MapperConfig, MapperParts, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::error::MoptResult;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::mapper::configuration::Configuration;
use mopt_rs::workflow::mapper::{
    follow_guesses, ExplorationMethod, Guess, Job, JobContext, JobOutcome, MapperObserver,
    MethodSettings, Network, Node, NodeKind, Source, Task,
};

const SURFACE: &str = "(q1^2-1)^2+cos(3.141592653589793*q1)*(q2^2+q3^2)+q2^4+q3^4+0.5*q2^2*q3^2";

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mopt_mapper_saddles_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A method with one task per minimum: a transition-state guess at `at` (the origin for
/// `origin_guess`).
struct GuessAt {
    name: &'static str,
    at: [f64; 3],
}

impl ExplorationMethod for GuessAt {
    fn name(&self) -> &str {
        self.name
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        Ok(if node.kind == NodeKind::Minimum {
            vec![Task {
                key: "origin".to_string(),
                description: "a transition-state guess at the origin".to_string(),
                parameters: JsonValue::Null,
            }]
        } else {
            Vec::new()
        })
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let mut outcome = JobOutcome::default();
        let guess: Configuration = job.start.with_flat(&Col::from_fn(3, |i| self.at[i]))?;
        let guesses = [Guess {
            system: guess,
            origin: format!("{:?}", self.at),
        }];
        if !follow_guesses(job, ctx, &guesses, &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

fn parts() -> MapperParts {
    let mut parts = MapperParts::default();
    parts.methods.register(
        "origin_guess",
        Arc::new(|_: &MethodSettings| {
            Ok(Box::new(GuessAt {
                name: "origin_guess",
                at: [0.0, 0.0, 0.0],
            }) as Box<dyn ExplorationMethod>)
        }),
    );
    parts.methods.register(
        "saddle_guess",
        Arc::new(|_: &MethodSettings| {
            Ok(Box::new(GuessAt {
                name: "saddle_guess",
                at: [1.0, 0.0, 0.0],
            }) as Box<dyn ExplorationMethod>)
        }),
    );
    parts
}

fn config(dir: &Path, extra: &[&str]) -> MapperConfig {
    let start = dir.join("start.txt");
    std::fs::write(&start, "1.0\n0.6\n0.6\n").unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        format!(
            r#"{{
  "mapper_settings": {{"max_iterations": 1, "rng_seed": 1, "method": "origin_guess"}},
  "step1_settings": {{"othersoft": "expr:{SURFACE}", "NSTEP": 300}},
  "step3_settings": {{"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300}},
  "step4_settings": {{
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]
  }}
}}"#
        ),
    )
    .unwrap();
    let mut args = vec![
        start.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

fn near(net: &Network, kind: &NodeKind, p: [f64; 3]) -> Option<usize> {
    net.of_kind(kind)
        .find(|n| {
            let x = n.system.flat();
            (0..3).map(|i| (x[i] - p[i]).powi(2)).sum::<f64>().sqrt() < 1e-3
        })
        .map(|n| n.id)
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
        .chain(net.saddle_edges.iter().map(|e| {
            format!(
                "saddle edge {}: {} mode {} -> {:?}",
                e.id, e.saddle, e.mode, e.ends
            )
        }))
        .chain(
            net.warnings
                .iter()
                .map(|w| format!("warning: {}", w.message)),
        )
        .collect::<Vec<_>>()
        .join("\n")
}

/// The network the descent must give.
fn check_landscape(run: &MapperRun) {
    let net = &run.network;
    let s = summary(net);
    let sp2: Vec<usize> = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        .iter()
        .map(|p| {
            near(net, &NodeKind::HigherSaddle { order: 2 }, *p)
                .unwrap_or_else(|| panic!("2SP at {p:?}\n{s}"))
        })
        .collect();
    for id in &sp2 {
        assert!(net.nodes[*id].label().starts_with("2SP"), "{s}");
        assert!((net.nodes[*id].energy).abs() < 1e-8, "{s}");
    }
    // The transition state at the origin ends at the two saddle points of order 2.
    let ts0 = near(net, &NodeKind::TransitionState, [0.0, 0.0, 0.0]).expect("TS at the origin");
    let edge = net.ts_edges.iter().find(|e| e.ts == ts0).expect("its edge");
    let mut ends = edge.ends.to_vec();
    ends.sort_unstable();
    let mut want = sp2.clone();
    want.sort_unstable();
    assert_eq!(ends, want, "{s}");
    // Every transition state and minimum below them, each descent ending at transition states.
    let h = 0.5_f64.sqrt();
    let m = 0.4_f64.sqrt();
    for x in [1.0, -1.0] {
        for p in [[x, h, 0.0], [x, -h, 0.0], [x, 0.0, h], [x, 0.0, -h]] {
            assert!(
                near(net, &NodeKind::TransitionState, p).is_some(),
                "TS at {p:?}\n{s}"
            );
        }
        for p in [[x, m, m], [x, m, -m], [x, -m, m], [x, -m, -m]] {
            assert!(
                near(net, &NodeKind::Minimum, p).is_some(),
                "EQ at {p:?}\n{s}"
            );
        }
    }
    assert_eq!(net.minima().count(), 8, "{s}");
    assert_eq!(net.of_kind(&NodeKind::TransitionState).count(), 9, "{s}");
    assert_eq!(net.saddle_edges.len(), 4, "{s}");
    for e in &net.saddle_edges {
        for end in e.ends {
            let end = end.unwrap_or_else(|| panic!("a descent did not settle\n{s}"));
            assert_eq!(net.nodes[end].kind, NodeKind::TransitionState, "{s}");
        }
        // Plain variables: the wavenumber of unit masses, negative for the unstable mode.
        assert!(e.wavenumber.is_some_and(|w| w < 0.0), "{s}");
    }
}

#[test]
fn the_irc_of_the_origin_runs_into_two_saddle_points_of_order_two() {
    let dir = scratch("landscape");
    let c = config(&dir, &["-map_opt", "higher_saddles=on"]);
    let run = run_with_parts(
        &c,
        &dir,
        None,
        &CalculatorRegistry::native(),
        &parts(),
        &mut [],
    )
    .unwrap();
    check_landscape(&run);
    // The files keep the kind and the descents.
    let doc = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    assert!(doc.contains("\"2SP\"") && doc.contains("\"saddle_edges\""));
    assert!(run.directory.join("structures/2SP000000.txt").is_file());
    let back = mopt_rs::workflow::mapper::store::read_network(&run.directory)
        .unwrap()
        .unwrap()
        .0;
    assert_eq!(back.saddle_edges, run.network.saddle_edges);
    // The imaginary wavenumbers of the saddle points (unit masses for plain variables).
    for n in run.network.of_kind(&NodeKind::HigherSaddle { order: 2 }) {
        let w = n.frequencies.as_ref().expect("the wavenumbers of a 2SP");
        assert_eq!(w.iter().filter(|v| **v < 0.0).count(), 2, "{w:?}");
    }
    // Without the setting the saddle points the IRC ran into are kept (no descents), and the
    // ends of the IRC are minimized as before.
    let other = scratch("off");
    let c = config(&other, &[]);
    let off = run_with_parts(
        &c,
        &other,
        None,
        &CalculatorRegistry::native(),
        &parts(),
        &mut [],
    )
    .unwrap();
    assert_eq!(
        off.network
            .of_kind(&NodeKind::HigherSaddle { order: 2 })
            .count(),
        2,
        "{}",
        summary(&off.network)
    );
    assert!(off.network.saddle_edges.is_empty());
}

#[test]
fn a_search_that_ends_at_an_index_two_point_keeps_it_and_goes_on_to_a_transition_state() {
    // The guess (1, 0, 0) is the 2SP itself: the search for a transition state ends there at
    // once, the point is kept, and the mode kill displaces it along its second unstable mode
    // and optimizes at order 1 following the first, to a transition state of the plane x = 1.
    let dir = scratch("kill");
    let c = config(&dir, &["-map_opt", "method=saddle_guess"]);
    let run = run_with_parts(
        &c,
        &dir,
        None,
        &CalculatorRegistry::native(),
        &parts(),
        &mut [],
    )
    .unwrap();
    let net = &run.network;
    let s = summary(net);
    assert!(
        near(net, &NodeKind::HigherSaddle { order: 2 }, [1.0, 0.0, 0.0]).is_some(),
        "{s}"
    );
    assert!(net.saddle_edges.is_empty(), "{s}");
    let h = 0.5_f64.sqrt();
    let found = [[1.0, h, 0.0], [1.0, -h, 0.0], [1.0, 0.0, h], [1.0, 0.0, -h]]
        .iter()
        .filter(|p| near(net, &NodeKind::TransitionState, **p).is_some())
        .count();
    assert_eq!(found, 1, "{s}");
    assert_eq!(net.ts_edges.len(), 1, "{s}");
}

/// Writes the file `STOP` of the run directory at its point `at` (counted from 0).
struct StopAtFrame {
    count: usize,
    at: usize,
    stop: PathBuf,
}

impl MapperObserver for StopAtFrame {
    fn on_frame(
        &mut self,
        _source: &Source<'_>,
        _system: &Configuration,
        _energy: f64,
        _label: &str,
    ) -> MoptResult<()> {
        if self.count == self.at {
            std::fs::write(&self.stop, "").unwrap();
        }
        self.count += 1;
        Ok(())
    }
}

#[test]
fn a_run_stopped_during_the_descents_continues_to_the_same_network() {
    let dir = scratch("restart");
    let whole_dir = dir.join("whole");
    std::fs::create_dir_all(&whole_dir).unwrap();
    let c = config(&whole_dir, &["-map_opt", "higher_saddles=on"]);
    let whole = run_with_parts(
        &c,
        &whole_dir,
        None,
        &CalculatorRegistry::native(),
        &parts(),
        &mut [],
    )
    .unwrap();
    let mut interrupted = 0;
    for at in [5, 60, 200, 600] {
        let base = dir.join(format!("at_{at}"));
        std::fs::create_dir_all(&base).unwrap();
        let c = config(&base, &["-map_opt", "higher_saddles=on"]);
        let mut stopper = StopAtFrame {
            count: 0,
            at,
            stop: base.join("mapper_output").join("STOP"),
        };
        let stopped = run_with_parts(
            &c,
            &base,
            None,
            &CalculatorRegistry::native(),
            &parts(),
            &mut [&mut stopper],
        )
        .unwrap();
        if !stopped.interrupted {
            continue;
        }
        interrupted += 1;
        let run_dir = base.join("mapper_output");
        let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
        let again = MapperConfig::from_args(&settings, Some(&run_dir)).unwrap();
        let done = run_with_parts(
            &again,
            &base,
            None,
            &CalculatorRegistry::native(),
            &parts(),
            &mut [],
        )
        .unwrap();
        assert!(!done.interrupted, "{at}");
        let (a, b) = (&done.network, &whole.network);
        assert_eq!(a.nodes.len(), b.nodes.len(), "{at}\n{}", summary(a));
        for (x, y) in a.nodes.iter().zip(&b.nodes) {
            assert_eq!(x.kind, y.kind, "{at}");
            assert!((x.energy - y.energy).abs() < 1e-9, "{at}: {}", x.label());
        }
        assert_eq!(a.saddle_edges, b.saddle_edges, "{at}");
        assert_eq!(a.ts_edges.len(), b.ts_edges.len(), "{at}");
    }
    assert!(
        interrupted >= 3,
        "only {interrupted} of the stops came during the run"
    );
}

#[test]
fn the_upward_search_climbs_from_a_transition_state_to_a_saddle_point_of_order_two() {
    // ADDF from the minimum finds the transition states of the plane x = 1; the upward search
    // from one of them along its softest stable mode reaches the 2SP (1, 0, 0).
    let dir = scratch("upward");
    let c = config(
        &dir,
        &["-map_opt", "method=addf", "upward=on", "--max_iter", "14"],
    );
    let run = run_with_parts(
        &c,
        &dir,
        None,
        &CalculatorRegistry::native(),
        &parts(),
        &mut [],
    )
    .unwrap();
    let net = &run.network;
    let s = summary(net);
    let sp2 = near(net, &NodeKind::HigherSaddle { order: 2 }, [1.0, 0.0, 0.0])
        .unwrap_or_else(|| panic!("2SP at (1, 0, 0)\n{s}"));
    assert!(
        net.jobs.iter().any(|j| j.task.starts_with("upward:")),
        "{s}"
    );
    // Its two descents end at transition states of the plane x = 1.
    let descents: Vec<_> = net
        .saddle_edges
        .iter()
        .filter(|e| e.saddle == sp2)
        .collect();
    assert_eq!(descents.len(), 2, "{s}");
    for e in descents {
        for end in e.ends.iter().flatten() {
            assert_eq!(net.nodes[*end].kind, NodeKind::TransitionState, "{s}");
            assert!((net.nodes[*end].system.flat()[0] - 1.0).abs() < 1e-3, "{s}");
        }
    }
}

/// The surface with a term that breaks its mirror symmetry in q1: every stationary point is
/// at another place and energy, and there are as many of each kind.
const SECOND_SURFACE: &str =
    "(q1^2-1)^2+cos(3.141592653589793*q1)*(q2^2+q3^2)+q2^4+q3^4+0.5*q2^2*q3^2+0.05*(q1-0.3)^2";

/// The settings of a run that draws the network of `first` again on [`SECOND_SURFACE`], into
/// `output` under `dir`.
fn remap_config(
    dir: &Path,
    first: &Path,
    output: &str,
    extra: &[&str],
) -> MoptResult<MapperConfig> {
    second_config(dir, "-remap", first, output, extra)
}

/// The settings of a run on [`SECOND_SURFACE`] that takes the network of `first` with `how`
/// (`-remap`, `-single_points`), into `output` under `dir`.
fn second_config(
    dir: &Path,
    how: &str,
    first: &Path,
    output: &str,
    extra: &[&str],
) -> MoptResult<MapperConfig> {
    let cfg = dir.join("second.json");
    std::fs::write(
        &cfg,
        format!(
            r#"{{
  "mapper_settings": {{"rng_seed": 1}},
  "step1_settings": {{"othersoft": "expr:{SECOND_SURFACE}", "NSTEP": 300}},
  "step3_settings": {{"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300}},
  "step4_settings": {{
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.05", "400", "lqa"]
  }}
}}"#
        ),
    )
    .unwrap();
    let mut args = vec![
        how.to_string(),
        first.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        output.to_string(),
        "-map_opt".to_string(),
        "higher_saddles=on".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args)?;
    MapperConfig::from_args(&parsed, None)
}

/// The rows of `remap.csv` of a run directory: the item, its kind, how its job ended, and the
/// distance of the nearest node of its kind in the run.
fn remap_rows(dir: &Path) -> Vec<(String, String, String, String)> {
    let text = std::fs::read_to_string(dir.join("remap.csv")).unwrap();
    text.lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[0].to_string(),
                f[1].to_string(),
                f[6].to_string(),
                f[9].to_string(),
            )
        })
        .collect()
}

#[test]
fn the_network_is_drawn_again_on_another_surface() {
    let dir = scratch("remap");
    let native = CalculatorRegistry::native();
    let c = config(&dir, &["-map_opt", "higher_saddles=on"]);
    let first = run_with_parts(&c, &dir, None, &native, &parts(), &mut []).unwrap();
    check_landscape(&first);
    let before = std::fs::read(first.directory.join("reaction_network.json")).unwrap();
    // The second run: no input file and no method; its inputs are the minima of the first
    // network, its jobs the transition states and saddle points of order 2 of it.
    let c = remap_config(&dir, &first.directory, "second", &[]).unwrap();
    assert_eq!(c.settings.method, "remap");
    assert_eq!(c.settings.max_jobs, 0, "every item runs");
    let second = run_with_parts(&c, &dir, None, &native, &MapperParts::default(), &mut []).unwrap();
    let net = &second.network;
    let s = summary(net);
    assert!(!second.interrupted, "{s}");
    assert_eq!(net.inputs.len(), 8, "{s}");
    assert_eq!(net.minima().count(), 8, "{s}");
    assert_eq!(net.of_kind(&NodeKind::TransitionState).count(), 9, "{s}");
    assert_eq!(
        net.of_kind(&NodeKind::HigherSaddle { order: 2 }).count(),
        2,
        "{s}"
    );
    assert_eq!(net.saddle_edges.len(), 4, "{s}");
    // One job per transition state and saddle point of the first network, each a task of the
    // method.
    assert_eq!(net.jobs.len(), 11, "{s}");
    assert!(net.jobs.iter().all(|j| j.task.starts_with("remap:")), "{s}");
    assert!(net.jobs.iter().all(|j| j.method == "remap"), "{s}");
    // The stationary points are those of the second surface: near those of the first, at
    // other energies (the term added is 0.05 (q1 - 0.3)^2: about 0.0245 at q1 = 1 and 0.0845
    // at q1 = -1).
    for old in first.network.minima() {
        let p = old.system.flat();
        let new = net
            .minima()
            .find(|n| {
                let q = n.system.flat();
                (0..3).map(|i| (q[i] - p[i]).powi(2)).sum::<f64>().sqrt() < 0.1
            })
            .unwrap_or_else(|| panic!("no minimum near {}\n{s}", old.label()));
        let shift = new.energy - old.energy;
        assert!(shift > 0.01 && shift < 0.1, "{}: {shift}\n{s}", old.label());
    }
    let ts0 = net
        .of_kind(&NodeKind::TransitionState)
        .find(|n| {
            n.system.flat().iter().skip(1).all(|v| v.abs() < 1e-3) && n.system.flat()[0].abs() < 0.2
        })
        .unwrap_or_else(|| panic!("the transition state near the origin\n{s}"));
    assert!(ts0.system.flat()[0].abs() > 1e-4, "it has moved: {s}");
    // The table: every minimum minimized, every item with a job that did not fail. A job
    // says "known" when what it was asked for had been found before it (the IRC of the
    // first transition state runs into both saddle points of order 2, and their descents
    // reach other transition states), and every item has a node of its kind near it (the
    // points move by a few hundredths with the term of the second surface).
    let rows = remap_rows(&second.directory);
    assert_eq!(rows.len(), 8 + 11, "{rows:?}");
    assert_eq!(
        rows.iter()
            .filter(|r| r.1 == "EQ" && r.2 == "minimized")
            .count(),
        8
    );
    for r in rows.iter().filter(|r| r.1 != "EQ") {
        assert!(r.2 == "added" || r.2 == "known", "{r:?}");
    }
    for r in &rows {
        let d: f64 = r.3.parse().unwrap_or_else(|_| panic!("{r:?}"));
        assert!(d < 0.1, "{r:?}");
    }
    assert_eq!(rows.iter().filter(|r| r.1 == "TS").count(), 9);
    assert_eq!(rows.iter().filter(|r| r.1 == "saddle").count(), 2);
    assert!(
        rows.iter().filter(|r| r.2 == "added").count() >= 2,
        "{rows:?}"
    );
    // The network names where it comes from, and the first run is as it was.
    let doc = std::fs::read_to_string(second.directory.join("reaction_network.json")).unwrap();
    assert!(
        doc.contains("\"source\"") && doc.contains("\"remap\""),
        "{doc}"
    );
    assert_eq!(
        std::fs::read(first.directory.join("reaction_network.json")).unwrap(),
        before
    );
    assert!(!first.directory.join("remap.csv").exists());
    let log = std::fs::read_to_string(second.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Remap: the network of"), "{log}");
    assert!(log.contains("9 transition state(s), 0 crossing point(s), 2 saddle point(s)"));

    // Without a chain the transition states of the first network are the guesses: the same
    // network.
    let c = remap_config(&dir, &first.directory, "direct", &["method_path=none"]).unwrap();
    let direct = run_with_parts(&c, &dir, None, &native, &MapperParts::default(), &mut []).unwrap();
    let d = &direct.network;
    assert_eq!(d.minima().count(), 8, "{}", summary(d));
    assert_eq!(
        d.of_kind(&NodeKind::TransitionState).count(),
        9,
        "{}",
        summary(d)
    );
    assert!(
        d.evaluations.gradients < net.evaluations.gradients,
        "the chains cost gradients: {} against {}",
        d.evaluations.gradients,
        net.evaluations.gradients
    );

    // A run stopped on the way continues to the same network.
    let mut interrupted = 0;
    for at in [2, 40, 200, 600] {
        let output = format!("stopped_{at}");
        let c = remap_config(&dir, &first.directory, &output, &[]).unwrap();
        let run_dir = dir.join(&output);
        let mut stopper = StopAtFrame {
            count: 0,
            at,
            stop: run_dir.join("STOP"),
        };
        let stopped = run_with_parts(
            &c,
            &dir,
            None,
            &native,
            &MapperParts::default(),
            &mut [&mut stopper],
        )
        .unwrap();
        if !stopped.interrupted {
            continue;
        }
        interrupted += 1;
        let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
        let again = MapperConfig::from_args(&settings, Some(&run_dir)).unwrap();
        assert_eq!(again.settings.method, "remap");
        let done = run_with_parts(
            &again,
            &dir,
            None,
            &native,
            &MapperParts::default(),
            &mut [],
        )
        .unwrap();
        assert!(!done.interrupted, "{at}");
        let (a, b) = (&done.network, net);
        assert_eq!(a.nodes.len(), b.nodes.len(), "{at}\n{}", summary(a));
        for (x, y) in a.nodes.iter().zip(&b.nodes) {
            assert_eq!(x.kind, y.kind, "{at}");
            assert!((x.energy - y.energy).abs() < 1e-9, "{at}: {}", x.label());
        }
        assert_eq!(a.jobs.len(), b.jobs.len(), "{at}");
        assert_eq!(a.ts_edges.len(), b.ts_edges.len(), "{at}");
        assert_eq!(a.saddle_edges, b.saddle_edges, "{at}");
        assert_eq!(remap_rows(&run_dir), rows, "{at}");
    }
    assert!(
        interrupted >= 3,
        "only {interrupted} of the stops came during the run"
    );

    // What does not fit is refused: an input file with -remap, the method without it, a
    // directory without a network, and another method with -remap.
    let start = dir.join("start.txt");
    let cfg = dir.join("second.json");
    let parse = |args: &[&str]| {
        let parsed = parser("run_mapper", RUN_MAPPER)
            .parse(args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .unwrap();
        MapperConfig::from_args(&parsed, None)
            .map(|_| ())
            .map_err(|e| e.to_string())
    };
    let (first_dir, cfg, start) = (
        first.directory.display().to_string(),
        cfg.display().to_string(),
        start.display().to_string(),
    );
    let e = parse(&[&start, "-remap", &first_dir, "-cfg", &cfg]).unwrap_err();
    assert!(e.contains("no input file is given with it"), "{e}");
    let e = parse(&[&start, "-cfg", &cfg, "-map_opt", "method=remap"]).unwrap_err();
    assert!(e.contains("-remap RUN_DIR"), "{e}");
    let empty = dir.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let e = parse(&["-remap", &empty.display().to_string(), "-cfg", &cfg]).unwrap_err();
    assert!(e.contains("no network there"), "{e}");
    let e = parse(&[
        "-remap",
        &first_dir,
        "-cfg",
        &cfg,
        "-map_opt",
        "method=addf",
    ])
    .unwrap_err();
    assert!(e.contains("runs the method remap"), "{e}");
}

#[test]
fn the_energies_of_a_network_are_computed_on_another_surface() {
    let dir = scratch("points");
    let native = CalculatorRegistry::native();
    let c = config(&dir, &["-map_opt", "higher_saddles=on"]);
    let first = run_with_parts(&c, &dir, None, &native, &parts(), &mut []).unwrap();
    check_landscape(&first);
    let before = std::fs::read(first.directory.join("reaction_network.json")).unwrap();
    let c = second_config(&dir, "-single_points", &first.directory, "points", &[]).unwrap();
    assert_eq!(c.settings.method, "single_points");
    let second = run_with_parts(&c, &dir, None, &native, &MapperParts::default(), &mut []).unwrap();
    assert!(!second.interrupted);
    let (old, new) = (&first.network, &second.network);
    let s = summary(new);
    // The same nodes at the same places; the energy of each is that of the second surface
    // there, the first plus 0.05 (q1 - 0.3)^2.
    assert_eq!(new.nodes.len(), old.nodes.len(), "{s}");
    for (a, b) in old.nodes.iter().zip(&new.nodes) {
        assert_eq!((a.id, a.kind, a.label()), (b.id, b.kind, b.label()), "{s}");
        let (p, q) = (a.system.flat(), b.system.flat());
        assert!(p.iter().zip(q.iter()).all(|(x, y)| x == y), "{}", a.label());
        let term = 0.05 * (p[0] - 0.3).powi(2);
        assert!(
            (b.energy - a.energy - term).abs() < 1e-10,
            "{}: {} against {} + {term}",
            a.label(),
            b.energy,
            a.energy
        );
        assert_eq!(a.frequencies, b.frequencies, "{}", a.label());
    }
    // The edges keep their ends; their barriers are those of the new energies, and the
    // energies along the IRCs (of the first surface) are left out.
    assert_eq!(new.ts_edges.len(), old.ts_edges.len());
    for (a, b) in old.ts_edges.iter().zip(&new.ts_edges) {
        assert_eq!((a.ts, a.ends, &a.irc.file), (b.ts, b.ends, &b.irc.file));
        for k in 0..2 {
            let barrier = new.nodes[b.ts].energy - new.nodes[b.ends[k]].energy;
            assert!((b.barriers[k] - barrier).abs() < 1e-12, "{s}");
        }
        assert!(b.irc.sides.is_empty(), "{s}");
    }
    assert_eq!(new.saddle_edges, old.saddle_edges);
    assert_eq!(
        new.jobs, old.jobs,
        "the jobs are those that found the nodes"
    );
    // One evaluation per node, nothing else (an objective of plain variables gives its
    // gradient with every value, so its evaluations are counted with the gradients; a
    // calculator of structures is asked for the energy alone).
    let e = new.evaluations;
    assert_eq!(e.energies + e.gradients, old.nodes.len(), "{s}");
    assert_eq!(e.hessians, 0, "{s}");
    // The files of a run, the table of both energies, and the path files beside them.
    let run = &second.directory;
    let table = std::fs::read_to_string(run.join("single_points.csv")).unwrap();
    assert_eq!(table.lines().count(), 1 + old.nodes.len(), "{table}");
    assert!(
        table.lines().nth(1).unwrap().starts_with("node:0,EQ0,EQ,"),
        "{table}"
    );
    let doc = std::fs::read_to_string(run.join("reaction_network.json")).unwrap();
    assert!(doc.contains("\"single_points\""), "{doc}");
    assert!(run.join("mapper_report.html").is_file());
    assert!(run.join("structures/EQ000000.txt").is_file());
    assert!(run.join("structures/2SP000000.txt").is_file());
    let path = old.ts_edges[0].irc.file.as_deref().unwrap();
    assert_eq!(
        std::fs::read(run.join(path)).unwrap(),
        std::fs::read(first.directory.join(path)).unwrap()
    );
    let log = std::fs::read_to_string(run.join("mopt.log")).unwrap();
    assert!(log.contains("Single points: the energies of the"), "{log}");
    // The first run is as it was.
    assert_eq!(
        std::fs::read(first.directory.join("reaction_network.json")).unwrap(),
        before
    );
    assert!(!first.directory.join("single_points.csv").exists());
    // The network is read again as any network, and the page of the run is written from it.
    let back = mopt_rs::workflow::mapper::store::read_network(run)
        .unwrap()
        .unwrap()
        .0;
    assert_eq!(back.nodes.len(), new.nodes.len());
    // Several energies at a time: the same network.
    let c = second_config(
        &dir,
        "-single_points",
        &first.directory,
        "points_parallel",
        &["--max_workers", "3"],
    )
    .unwrap();
    let parallel =
        run_with_parts(&c, &dir, None, &native, &MapperParts::default(), &mut []).unwrap();
    for (a, b) in parallel.network.nodes.iter().zip(&new.nodes) {
        assert!((a.energy - b.energy).abs() < 1e-14, "{}", a.label());
    }
    // What does not fit is refused.
    let parse = |args: &[&str]| {
        let parsed = parser("run_mapper", RUN_MAPPER)
            .parse(args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .unwrap();
        MapperConfig::from_args(&parsed, None)
            .map(|_| ())
            .map_err(|e| e.to_string())
    };
    let (first_dir, cfg) = (
        first.directory.display().to_string(),
        dir.join("second.json").display().to_string(),
    );
    let e = parse(&[
        "-single_points",
        &first_dir,
        "-remap",
        &first_dir,
        "-cfg",
        &cfg,
    ])
    .unwrap_err();
    assert!(e.contains("one of the three"), "{e}");
    let e = parse(&[
        "-single_points",
        &first_dir,
        "-cfg",
        &cfg,
        "-map_opt",
        "method=addf",
    ])
    .unwrap_err();
    assert!(e.contains("explores nothing"), "{e}");
}
