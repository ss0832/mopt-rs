//! The hard constraints of `run_mapper` in every stage (`docs/source/mapper_design.md`,
//! "Constraints and bias potentials of the user"): {frozen atoms, a distance, a centroid
//! distance of two fragments (`fbond`), an angle, a rigid group (`-gfix`), a user-defined
//! function (`-uc`)} x {an isolated cluster, a cluster in a fixed periodic cell}, given on the
//! command line as options shared by every step. Every structure the run writes (the nodes, the
//! transition states, the IRC paths, the chains of the path relaxations, the trajectories of
//! the jobs), the nodes of the network file and of the SQLite store, and the nodes in memory
//! hold the constraints at the values of the first input within 1e-10, the frozen atoms
//! bit-identical in memory. A variable cell, the core of an ONIOM surface, an objective of plain
//! variables, and dissociation channels asked for are refused by name. A second input off the
//! constraints is put onto them; the network does not depend on the number of workers (batch
//! schedule) and a run stopped at two points continues to the same network.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use matrix_support::{check, frames, kinds, read, scratch, write, FROM_FILES, FROZEN_IN_FILES};
use mopt_rs::cli::mapper::{run_with, MapperConfig, MapperRun};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::settings::read_settings;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::constraint::Constraint;
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::network_json::from_json;
use mopt_rs::workflow::mapper::sqlite_store::read_sqlite;
use mopt_rs::workflow::mapper::Network;

/// Ar7 near its lowest minimum (Angstrom).
const AR7: &str = "7\nAr7\nAr 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

/// The same cluster in a cubic box of 30 Angstrom.
const AR7_CELL: &str = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

/// A small run on `lj`: three jobs of AFIR pushes, the path relaxed, a guess per job.
const SETTINGS: &str = r#"{
  "mapper_settings": {"afir_gamma_kJmol": 2.0, "dist_upper_ang": 7.0, "max_iterations": 3,
                      "rng_seed": 42},
  "top_n_candidates": 1,
  "step1_settings": {"othersoft": "lj", "NSTEP": 200},
  "step2_settings": {"NSTEP": 8},
  "step3_settings": {"NSTEP": 100, "calc_exact_hess": 5, "tight_convergence_criteria": true},
  "step4_settings": {"NSTEP": 200, "tight_convergence_criteria": true, "intrinsic_reaction_coordinates": ["0.5", "60", "lqa"]}
}"#;

/// The settings of a run of `input` with `settings` and `extra` options into `out`.
fn mapper_config(
    dir: &Path,
    input: &[&Path],
    settings: &str,
    out: &str,
    extra: &[String],
) -> MapperConfig {
    let cfg = write(dir, &format!("{out}.json"), settings);
    let mut args: Vec<String> = input.iter().map(|p| p.display().to_string()).collect();
    args.extend([
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        out.to_string(),
    ]);
    args.extend(extra.iter().cloned());
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

/// Every xyz file under `dir`, recursively.
fn xyz_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                todo.push(p);
            } else if p.extension().is_some_and(|e| e == "xyz") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// The structures of the nodes of `network`.
fn node_structures(network: &Network) -> Vec<MolecularSystem> {
    network
        .nodes
        .iter()
        .filter_map(|n| n.system.structure().cloned())
        .collect()
}

/// Every structure the run in `run` holds or wrote keeps `constraints` at their values in
/// `input`: in memory (frozen atoms bit-identical), in the network file and the SQLite store,
/// and in every xyz file of the run directory.
fn every_structure(
    cell: &str,
    run: &MapperRun,
    constraints: &[Constraint],
    input: &MolecularSystem,
) {
    let memory = node_structures(&run.network);
    check(
        &format!("{cell} (memory)"),
        &memory,
        constraints,
        input,
        1e-10,
        0.0,
        false,
    );
    let text = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
    let (file, _) = from_json(&JsonValue::parse(&text).unwrap()).unwrap();
    check(
        &format!("{cell} (network file)"),
        &node_structures(&file),
        constraints,
        input,
        1e-10,
        0.0,
        false,
    );
    let (store, _, _) = read_sqlite(&run.directory.join("reaction_network.sqlite")).unwrap();
    check(
        &format!("{cell} (SQLite)"),
        &node_structures(&store),
        constraints,
        input,
        1e-10,
        0.0,
        false,
    );
    let files = xyz_files(&run.directory);
    eprintln!("{cell}: {} files", files.len());
    for f in &files {
        let name = f
            .strip_prefix(&run.directory)
            .unwrap_or(f)
            .display()
            .to_string();
        check(
            &format!("{cell} ({name})"),
            &frames(f),
            constraints,
            input,
            FROM_FILES,
            FROZEN_IN_FILES,
            false,
        );
    }
}

fn context(name: &str, text: &str) {
    let dir = scratch("mapmatrix", name);
    let file = write(&dir, "start.xyz", text);
    let input = read(&file);
    for (kind, options) in kinds([1, 2, 3, 4], (&[1, 2, 3], &[4, 5, 6])) {
        let cell = format!("{name} x {kind}");
        let mut extra = options.clone();
        extra.extend(["-map_opt".to_string(), "store=sqlite".to_string()]);
        let config = mapper_config(&dir, &[&file], SETTINGS, kind, &extra);
        let constraints = config.steps.biased.settings.constraints.clone();
        assert!(!constraints.is_empty(), "{cell}");
        let run = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{cell}: {e}"));
        assert_eq!(run.network.jobs.len(), 3, "{cell}");
        let ts = run
            .network
            .nodes
            .iter()
            .filter(|n| n.kind.tag() == "TS")
            .count();
        eprintln!(
            "{cell}: {} nodes, {ts} transition states, {} edges",
            run.network.nodes.len(),
            run.network.ts_edges.len()
        );
        every_structure(&cell, &run, &constraints, &input);
        let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
        assert!(log.contains("held in every stage of every job"), "{cell}");
        let page = std::fs::read_to_string(run.directory.join("mapper_report.html")).unwrap();
        assert!(page.contains("Constraints (held in every stage"), "{cell}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_a_network_of_an_isolated_cluster() {
    context("isolated", AR7);
}

#[test]
fn constraints_of_a_network_in_a_periodic_cell() {
    context("periodic", AR7_CELL);
}

/// The error of a run of `settings` with `extra` options (made or run).
fn refused(dir: &Path, file: &Path, settings: &str, extra: &[&str]) -> String {
    let cfg = write(dir, "refused.json", settings);
    let mut args = vec![
        file.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        "refused".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    match MapperConfig::from_args(&parsed, None) {
        Err(e) => e.to_string(),
        Ok(c) => match run_with(&c, dir, None, &CalculatorRegistry::native()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("{extra:?} was taken"),
        },
    }
}

#[test]
fn what_cannot_hold_the_constraints_is_refused_by_name() {
    let dir = scratch("mapmatrix", "refused");
    let file = write(&dir, "start.xyz", AR7_CELL);
    // A variable cell: the IRC of the atoms and the cell holds no constraints.
    let vc = SETTINGS.replace(
        "\"othersoft\": \"lj\",",
        "\"othersoft\": \"lj\", \"variable_cell\": true,",
    );
    for (kind, options) in kinds([1, 2, 3, 4], (&[1, 2, 3], &[4, 5, 6])) {
        let o: Vec<&str> = options.iter().map(String::as_str).collect();
        let e = refused(&dir, &file, &vc, &o);
        assert!(e.contains("a variable cell"), "{kind}: {e}");
    }
    // A constraint of the cell (-vc_fix: constraints of the functions of the cell) likewise,
    // and without a variable cell it has nothing to hold.
    let e = refused(&dir, &file, &vc, &["-vc_fix", "a"]);
    assert!(e.contains("a variable cell"), "-vc_fix with -vc: {e}");
    let e = refused(&dir, &file, SETTINGS, &["-vc_fix", "a"]);
    assert!(e.contains("-vc_fix"), "-vc_fix without -vc: {e}");
    let isolated = write(&dir, "isolated.xyz", AR7);
    for (extra, why) in [
        (&["-fix", "1", "-map_opt", "channels=on"][..], "channels=on"),
        (
            &["-pc", "bond", "1,2", "-map_opt", "method=nanoreactor"],
            "nanoreactor",
        ),
        (
            &["-pc", "bond", "1,2", "-map_opt", "entrance_structures=2"],
            "entrance_structures",
        ),
        (
            &["-pc", "bond", "1,2", "-map_opt", "higher_saddles=on"],
            "higher_saddles",
        ),
        (
            &["-pc", "bond", "1,2", "-map_opt", "method_relax=dmf"],
            "method_relax=dmf",
        ),
        (&["-pc", "eigvec", "1"], "eigenvector constraint"),
        (
            &["-pc", "bond", "1,2", "-map_opt", "states=0,1;0,3"],
            "several electronic states",
        ),
    ] {
        let e = refused(&dir, &isolated, SETTINGS, extra);
        assert!(e.contains(why), "{extra:?}: {e}");
    }
    // The constraints of a step must be those of the run where the relaxation of a path holds
    // them.
    let step2 = SETTINGS.replace("\"NSTEP\": 8", "\"NSTEP\": 8, \"fix_atoms\": [\"2\"]");
    let e = refused(&dir, &isolated, &step2, &["-fix", "1"]);
    assert!(e.contains("step2_settings sets 'fix_atoms'"), "{e}");
    // An objective of plain variables.
    let points = write(&dir, "point.txt", "x 0.6\ny 0.0\n");
    let mb = SETTINGS.replace("\"othersoft\": \"lj\"", "\"othersoft\": \"mb2d\"");
    let e = refused(&dir, &points, &mb, &["-fix", "1"]);
    assert!(e.contains("objective of plain variables"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_on_the_core_of_an_oniom_surface_are_refused() {
    let dir = scratch("mapmatrix", "oniom");
    let file = write(
        &dir,
        "ethanol.xyz",
        "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n",
    );
    let oniom = SETTINGS.replace(
        "\"othersoft\": \"lj\",",
        "\"othersoft\": \"gfnff\", \"oniom_flag\": [\"2-4,8,9\", \"auto\", \"gfnff\"],",
    );
    for (kind, options) in kinds([2, 3, 4, 8], (&[2, 8], &[3, 4])) {
        let o: Vec<&str> = options.iter().map(String::as_str).collect();
        let e = refused(&dir, &file, &oniom, &o);
        // Frozen atoms of the environment are held by the surface (`tests/cli_mapper_oniom.rs`);
        // these are atoms of the core.
        let why = if kind == "frozen" {
            "-fix holds atoms of the core"
        } else {
            "on an ONIOM surface"
        };
        assert!(e.contains(why), "{kind}: {e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_second_input_is_put_onto_the_constraints_of_the_first() {
    let dir = scratch("mapmatrix", "inputs");
    let first = write(&dir, "first.xyz", AR7);
    // The second input: the first with atoms 1 and 2 moved apart.
    let second = write(
        &dir,
        "second.xyz",
        &AR7.replace("Ar 3.187637 0.000000", "Ar 3.287637 0.050000"),
    );
    let input = read(&first);
    let options: Vec<String> = ["-pc", "bond", "1,2", "-map_opt", "store=sqlite"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let config = mapper_config(&dir, &[&first, &second], SETTINGS, "two", &options);
    let constraints = config.steps.biased.settings.constraints.clone();
    let run = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    every_structure("two inputs", &run, &constraints, &input);
    let log = std::fs::read_to_string(run.directory.join("mopt.log")).unwrap();
    assert!(log.contains("was put onto the constraints"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
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

/// The networks `a` and `b` are the same: their nodes bit for bit.
fn same_network(cell: &str, a: &Network, b: &Network, exact: bool) {
    assert_eq!(a.nodes.len(), b.nodes.len(), "{cell}");
    assert_eq!(a.ts_edges.len(), b.ts_edges.len(), "{cell}");
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!(x.kind, y.kind, "{cell} {}", x.label());
        let (cx, cy) = (x.system.flat(), y.system.flat());
        if exact {
            assert_eq!(
                x.energy.to_bits(),
                y.energy.to_bits(),
                "{cell} {}",
                x.label()
            );
            assert!(
                cx.iter()
                    .zip(cy.iter())
                    .all(|(p, q)| p.to_bits() == q.to_bits()),
                "{cell}"
            );
        } else {
            assert!((x.energy - y.energy).abs() < 1e-12, "{cell} {}", x.label());
            assert!(
                cx.iter().zip(cy.iter()).all(|(p, q)| (p - q).abs() < 1e-7),
                "{cell}"
            );
        }
    }
}

#[test]
fn a_constrained_network_does_not_depend_on_the_workers_or_on_a_restart() {
    let dir = scratch("mapmatrix", "schedule");
    let file = write(&dir, "start.xyz", AR7);
    let input = read(&file);
    let with = |out: &str, more: &[&str]| {
        let mut extra: Vec<String> = [
            "-fix",
            "7",
            "-pc",
            "bond",
            "1,2",
            "-map_opt",
            "store=sqlite",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        extra.extend(more.iter().map(|s| s.to_string()));
        mapper_config(&dir, &[&file], SETTINGS, out, &extra)
    };
    let native = CalculatorRegistry::native();
    let one = with("one", &["workers=1", "batch_size=3"]);
    let constraints = one.steps.biased.settings.constraints.clone();
    let one = run_with(&one, &dir, None, &native).unwrap();
    let four = run_with(
        &with("four", &["workers=4", "batch_size=3"]),
        &dir,
        None,
        &native,
    )
    .unwrap();
    same_network("workers 1 and 4", &one.network, &four.network, true);
    every_structure("workers 4", &four, &constraints, &input);
    // Stopped by a STOP file at two points and continued from the run directory.
    for at in [150, 320] {
        let out = format!("stopped_{at}");
        let run_dir = dir.join(&out);
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
        let config = with(&out, &["workers=1", "batch_size=3"]);
        let stopped = run_with(&config, &dir, None, &stopping).unwrap();
        assert!(stopped.interrupted, "stopped at {at}");
        let settings = read_settings(&run_dir.join("settings.json"), RUN_MAPPER).unwrap();
        let again = MapperConfig::from_args(&settings, Some(&run_dir)).unwrap();
        let done = run_with(&again, &dir, None, &native).unwrap();
        assert!(!done.interrupted);
        same_network(
            &format!("restart at {at}"),
            &done.network,
            &one.network,
            false,
        );
        every_structure(&format!("restart at {at}"), &done, &constraints, &input);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
