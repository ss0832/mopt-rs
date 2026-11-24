//! One GFN-FF across the steps of a workflow: `-calc_opt topology=FILE` sets the topology of the
//! built-in GFN-FF up from the structure in `FILE`, so that a calculator made for a later step at
//! a structure with other bonds (a transition state, where the hydrogen of HCN bridges C and N)
//! gives the energy and gradient of the calculator of the run (whose topology came from its
//! first structure) at that structure to 1e-10, where a calculator made from the structure
//! itself does not; and `run_autots` passes the topology of its input to every step.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

use mopt_rs::cli::autots::AutotsConfig;
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

const REACTANT: &str =
    "3\nHCN\nH 0.0000 0.0000 -1.0650\nC 0.0000 0.0000 0.0000\nN 0.0000 0.0000 1.1560\n";
/// The hydrogen over the middle of the C-N bond: bonded to both in the topology of this
/// structure, to carbon alone in that of the reactant.
const BRIDGED: &str =
    "3\nHCN bridged\nH 1.0500 0.0000 0.5500\nC 0.0000 0.0000 0.0000\nN 0.0000 0.0000 1.1900\n";

#[test]
fn the_topology_of_a_reference_structure_gives_the_force_field_of_the_run() {
    let dir = std::env::temp_dir().join(format!("mopt_gfnff_topology_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (reactant_file, bridged_file) = (dir.join("reactant.xyz"), dir.join("bridged.xyz"));
    std::fs::write(&reactant_file, REACTANT).unwrap();
    std::fs::write(&bridged_file, BRIDGED).unwrap();
    let reactant = read_xyz_file(&reactant_file, SpinCharge::default(), None).unwrap();
    let bridged = read_xyz_file(&bridged_file, SpinCharge::default(), None).unwrap();
    let registry = CalculatorRegistry::native();
    let plain = CalculatorOptions::default();
    let mut referenced = CalculatorOptions::default();
    referenced
        .extra
        .push(("topology".to_string(), reactant_file.display().to_string()));
    // The calculator of the run: its topology from its first structure (the reactant), then
    // asked at the bridged structure.
    let run = registry.create("gfnff", &plain, &reactant).unwrap();
    let run = run.evaluate(&bridged, &EvalRequest::ENERGY_GRAD).unwrap();
    // A later command made at the bridged structure, with and without the reference.
    let follow = registry
        .create("gfnff", &referenced, &bridged)
        .unwrap()
        .evaluate(&bridged, &EvalRequest::ENERGY_GRAD)
        .unwrap();
    let own = registry
        .create("gfnff", &plain, &bridged)
        .unwrap()
        .evaluate(&bridged, &EvalRequest::ENERGY_GRAD)
        .unwrap();
    let largest = |a: &faer::Col<f64>, b: &faer::Col<f64>| {
        (0..a.nrows()).fold(0.0_f64, |m, i| m.max((a[i] - b[i]).abs()))
    };
    let (de, dg) = (
        (follow.energy.value() - run.energy.value()).abs(),
        largest(&follow.gradient_bohr, &run.gradient_bohr),
    );
    eprintln!(
        "with the reference: dE {de:.3e}, dg {dg:.3e}; with its own topology: dE {:.3e}, dg {:.3e}",
        (own.energy.value() - run.energy.value()).abs(),
        largest(&own.gradient_bohr, &run.gradient_bohr)
    );
    assert!(de < 1e-10 && dg < 1e-10, "{de} {dg}");
    assert!((own.energy.value() - run.energy.value()).abs() > 1e-6);
    // Refused: another molecule, and the reference together with refresh_topology.
    let mut both = referenced.clone();
    both.extra
        .push(("refresh_topology".to_string(), "true".to_string()));
    let e = registry
        .create("gfnff", &both, &bridged)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(e.contains("one of the two"), "{e}");
    let water = dir.join("water.xyz");
    std::fs::write(&water, "3\n\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n").unwrap();
    let mut other = CalculatorOptions::default();
    other
        .extra
        .push(("topology".to_string(), water.display().to_string()));
    let e = registry
        .create("gfnff", &other, &bridged)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(e.contains("other atoms"), "{e}");
    // run_autots gives every step the topology of its input.
    let cfg = dir.join("autots.json");
    std::fs::write(
        &cfg,
        r#"{"work_dir": "work",
            "step1_settings": {"othersoft": "gfnff", "manual_AFIR": ["100", "1", "3"]},
            "step3_settings": {"othersoft": "gfnff"}}"#,
    )
    .unwrap();
    let args = [
        reactant_file.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
    ];
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let config =
        AutotsConfig::from_args(&parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap()).unwrap();
    // The steps get the option when the run starts (the file is in its work directory); the
    // settings as given have none.
    assert!(!config.steps[2].contains_key("calc_option"));
    let _ = std::fs::remove_dir_all(&dir);
}
