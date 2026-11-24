//! The static bias potentials of `run_mapper` (`docs/source/mapper_design.md`, "Constraints and
//! bias potentials of the user"): {a restraint (`-kp`), an absolute-position bias (`-ubias` with
//! `x()`, `y()`, `z()`), a wall (`-wwp`)} x {an isolated cluster, a cluster in a fixed periodic
//! cell}, given on the command line as options shared by every step. Every node of the network
//! (in memory, in the network file, in `nodes.csv`) has the energy of the biased surface and the
//! energy of the calculator alone, consistent with a recomputation in the test: the bare energy
//! that of `lj` at the structure, the difference the energy of the bias there. A stateful bias
//! (metadynamics) and a bias on an ONIOM surface are refused by name; the network does not
//! depend on the number of workers.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use std::path::Path;

use matrix_support::{bias_energy, column, csv, read, scratch, write};
use mopt_rs::cli::mapper::{run_with, MapperConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{AtomId, EvalRequest, MolecularSystem};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::mapper::network_json::from_json;
use mopt_rs::workflow::mapper::Network;

const AR7: &str = "7\nAr7\nAr 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

const AR7_CELL: &str = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

const SETTINGS: &str = r#"{
  "mapper_settings": {"afir_gamma_kJmol": 2.0, "dist_upper_ang": 7.0, "max_iterations": 3,
                      "rng_seed": 42},
  "top_n_candidates": 1,
  "step1_settings": {"othersoft": "lj", "NSTEP": 200},
  "step2_settings": {"NSTEP": 8},
  "step3_settings": {"NSTEP": 100, "calc_exact_hess": 5, "tight_convergence_criteria": true},
  "step4_settings": {"NSTEP": 200, "tight_convergence_criteria": true, "intrinsic_reaction_coordinates": ["0.5", "60", "lqa"]}
}"#;

/// The biases of the matrix for `input`: (name, flag, arguments).
fn biases(input: &MolecularSystem) -> Vec<(&'static str, String, Vec<String>)> {
    let p = input.position(AtomId::new(0)).value();
    let goal = [p[0] + 0.4, p[1] - 0.2, p[2] + 0.1];
    vec![
        (
            "restraint",
            "-kp".to_string(),
            vec!["0.002".into(), "3.90".into(), "1,2".into()],
        ),
        (
            "absolute",
            "-ubias".to_string(),
            vec![format!(
                "0.002 * ((x(1) - {:.10})^2 + (y(1) - {:.10})^2 + (z(1) - {:.10})^2)",
                goal[0], goal[1], goal[2]
            )],
        ),
        (
            "wall",
            "-wwp".to_string(),
            vec![
                "2.0".into(),
                "z".into(),
                "-10,-9,2.5,4.0".into(),
                "1-7".into(),
            ],
        ),
    ]
}

fn config(dir: &Path, file: &Path, out: &str, extra: &[String]) -> MapperConfig {
    let cfg = write(dir, &format!("{out}.json"), SETTINGS);
    let mut args = vec![
        file.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        out.to_string(),
    ];
    args.extend(extra.iter().cloned());
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    MapperConfig::from_args(&parsed, None).unwrap()
}

/// Every node of `network` has the biased and the bare energy, consistent with `lj` and the
/// bias `flag arguments` recomputed at its structure.
fn consistent(cell: &str, network: &Network, flag: &str, arguments: &[String]) {
    assert!(!network.nodes.is_empty(), "{cell}");
    for n in &network.nodes {
        let sys = n.system.structure().unwrap();
        let bare = n
            .bare_energy
            .unwrap_or_else(|| panic!("{cell} {}: no bare energy", n.label()));
        let lj = LennardJones
            .evaluate(sys, &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .energy
            .value();
        assert!(
            (bare - lj).abs() < 1e-10,
            "{cell} {}: bare {bare}, lj {lj}",
            n.label()
        );
        let b = bias_energy(flag, arguments, sys);
        assert!(
            (n.energy - bare - b).abs() < 1e-10,
            "{cell} {}: biased {} - bare {bare} against the bias {b}",
            n.label(),
            n.energy
        );
    }
}

fn context(name: &str, text: &str) {
    let dir = scratch("mapbias", name);
    let file = write(&dir, "start.xyz", text);
    let input = read(&file);
    for (kind, flag, arguments) in biases(&input) {
        let cell = format!("{name} x {kind}");
        let mut extra = vec![flag.clone()];
        extra.extend(arguments.iter().cloned());
        let run = run_with(
            &config(&dir, &file, kind, &extra),
            &dir,
            None,
            &CalculatorRegistry::native(),
        )
        .unwrap_or_else(|e| panic!("{cell}: {e}"));
        consistent(&cell, &run.network, &flag, &arguments);
        let text = std::fs::read_to_string(run.directory.join("reaction_network.json")).unwrap();
        let (file_network, _) = from_json(&JsonValue::parse(&text).unwrap()).unwrap();
        consistent(&format!("{cell} (file)"), &file_network, &flag, &arguments);
        let (header, rows) = csv(&run.directory.join("nodes.csv"));
        let (e, e0) = (
            column(&header, "energy_hartree"),
            column(&header, "bare_energy_hartree"),
        );
        assert_eq!(rows.len(), run.network.nodes.len(), "{cell}");
        for (row, n) in rows.iter().zip(&run.network.nodes) {
            // The last column follows the quoted fields: take it from the end of the line.
            let last = row.last().unwrap().rsplit(',').next().unwrap();
            let bare: f64 = if e0 + 1 == header.len() {
                last.parse().unwrap()
            } else {
                row[e0].parse().unwrap()
            };
            assert!(
                (row[e].parse::<f64>().unwrap() - n.energy).abs() < 1e-11,
                "{cell}"
            );
            assert!((bare - n.bare_energy.unwrap()).abs() < 1e-11, "{cell}");
        }
        let page = std::fs::read_to_string(run.directory.join("mapper_report.html")).unwrap();
        assert!(page.contains("Static bias potentials"), "{cell}");
        assert!(page.contains("calculator alone"), "{cell}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn biases_of_a_network_of_an_isolated_cluster() {
    context("isolated", AR7);
}

#[test]
fn biases_of_a_network_in_a_periodic_cell() {
    context("periodic", AR7_CELL);
}

#[test]
fn biases_that_cannot_be_part_of_the_surface_are_refused() {
    let dir = scratch("mapbias", "refused");
    let file = write(&dir, "start.xyz", AR7);
    let cfg = write(&dir, "refused.json", SETTINGS);
    let run = |extra: &[&str], settings: &Path| {
        let mut args = vec![
            file.display().to_string(),
            "-cfg".to_string(),
            settings.display().to_string(),
            "--output_dir".to_string(),
            "refused".to_string(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
        match MapperConfig::from_args(&parsed, None) {
            Err(e) => e.to_string(),
            Ok(c) => match run_with(&c, &dir, None, &CalculatorRegistry::native()) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("{extra:?} was taken"),
            },
        }
    };
    // Metadynamics depends on its deposits, not on the structure alone.
    let e = run(&["-metad", "bond", "0.01", "0.05", "1,4"], &cfg);
    assert!(e.contains("depends on more than the structure"), "{e}");
    // An ONIOM surface.
    let ethanol = write(
        &dir,
        "ethanol.xyz",
        "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n",
    );
    let oniom = write(
        &dir,
        "oniom.json",
        &SETTINGS.replace(
            "\"othersoft\": \"lj\",",
            "\"othersoft\": \"gfnff\", \"oniom_flag\": [\"2-4,8,9\", \"auto\", \"gfnff\"],",
        ),
    );
    let mut args = vec![
        ethanol.display().to_string(),
        "-cfg".to_string(),
        oniom.display().to_string(),
        "--output_dir".to_string(),
        "oniom".to_string(),
    ];
    args.extend(["-kp", "0.002", "1.5", "2,3"].iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    let e = MapperConfig::from_args(&parsed, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("bias potentials on an ONIOM surface"), "{e}");
    // A bias in shared_settings and the same in a step: taken once; another in a step: refused
    // as another surface.
    let shared = write(
        &dir,
        "shared.json",
        &SETTINGS.replace(
            "\"top_n_candidates\": 1,",
            "\"top_n_candidates\": 1, \"shared_settings\": {\"keep_pot\": [\"0.002\", \"3.9\", \"1,2\"]},",
        ),
    );
    let mut args = vec![
        file.display().to_string(),
        "-cfg".to_string(),
        shared.display().to_string(),
        "--output_dir".to_string(),
        "shared".to_string(),
    ];
    args.extend(["--max_iter", "1"].iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(args).unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    assert_eq!(c.steps.biased.biases.len(), 1);
    assert_eq!(c.steps.saddle.biases.len(), 1);
    assert_eq!(c.steps.minimum.biases.len(), 1);
    let other = write(
        &dir,
        "other.json",
        &SETTINGS
            .replace(
                "\"top_n_candidates\": 1,",
                "\"top_n_candidates\": 1, \"shared_settings\": {\"keep_pot\": [\"0.002\", \"3.9\", \"1,2\"]},",
            )
            .replace("\"tight_convergence_criteria\": true}", "\"tight_convergence_criteria\": true, \"keep_pot\": [\"0.004\", \"3.9\", \"1,2\"]}"),
    );
    let e = run(&[], &other);
    assert!(e.contains("the bias potentials"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_biased_network_does_not_depend_on_the_workers() {
    let dir = scratch("mapbias", "workers");
    let file = write(&dir, "start.xyz", AR7);
    let input = read(&file);
    let (_, flag, arguments) = biases(&input).remove(2);
    let with = |out: &str, workers: &str| {
        let mut extra = vec![flag.clone()];
        extra.extend(arguments.iter().cloned());
        extra.extend(
            ["-map_opt", &format!("workers={workers}"), "batch_size=3"]
                .iter()
                .map(|s| s.to_string()),
        );
        run_with(
            &config(&dir, &file, out, &extra),
            &dir,
            None,
            &CalculatorRegistry::native(),
        )
        .unwrap()
    };
    let (one, four) = (with("one", "1"), with("four", "4"));
    assert_eq!(one.network.nodes.len(), four.network.nodes.len());
    for (x, y) in one.network.nodes.iter().zip(&four.network.nodes) {
        assert_eq!(x.energy.to_bits(), y.energy.to_bits(), "{}", x.label());
        assert_eq!(
            x.bare_energy.map(f64::to_bits),
            y.bare_energy.map(f64::to_bits)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
