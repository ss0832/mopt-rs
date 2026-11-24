//! Replicas of the environment of ONIOM microiterations (`-oniom_opt micro_replicas=K`) from
//! the command line, on a model whose environment has two minima by construction: one atom of
//! the core at `x` and one of the environment at `u + 6` Bohr along the axis, with
//! `E = C4 (x^2 - 1)^2 + A (u^2 - 1)^2 + B x u (3 - u^2) / 2` plus harmonic terms across the
//! axis. The environment has the minima `u = +1` and `u = -1` at every core, of the energies
//! `C4 (x^2 - 1)^2 +- B x`, which cross at `x = 0`. With both as replicas the surface of the
//! core is `C4 (x^2 - 1)^2 - kT ln(2 cosh(B x / kT))`: minima near `x = +-1` and a first-order
//! saddle point at `x = 0`, where the surface of either environment alone has none.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use faer::{Col, Mat};
use mopt_rs::cli::optmain;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity, SpinCharge};
use mopt_rs::error::MoptResult;
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::journal::JOURNAL_FILE;

const C4: f64 = 0.002;
const A: f64 = 0.01;
const B: f64 = 1e-3;
const K: f64 = 0.05;
const OFFSET: f64 = 6.0;

struct DoubleWell;

impl Calculator for DoubleWell {
    fn name(&self) -> &str {
        "double_well"
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let n = sys.dof();
        let p = sys.coords_flat_bohr();
        let mut g = Col::<f64>::zeros(n);
        let mut h = Mat::<f64>::zeros(n, n);
        let mut e = 0.0;
        if sys.n_atoms() == 2 {
            let (x, u) = (p[0], p[3] - OFFSET);
            let gu = u * (3.0 - u * u) / 2.0;
            let dgu = 1.5 * (1.0 - u * u);
            e = C4 * (x * x - 1.0).powi(2) + A * (u * u - 1.0).powi(2) + B * x * gu;
            g[0] = 4.0 * C4 * x * (x * x - 1.0) + B * gu;
            g[3] = 4.0 * A * u * (u * u - 1.0) + B * x * dgu;
            h[(0, 0)] = 4.0 * C4 * (3.0 * x * x - 1.0);
            h[(0, 3)] = B * dgu;
            h[(3, 0)] = B * dgu;
            h[(3, 3)] = 4.0 * A * (3.0 * u * u - 1.0) - 3.0 * B * x * u;
            for i in [1, 2, 4, 5] {
                e += 0.5 * K * p[i] * p[i];
                g[i] = K * p[i];
                h[(i, i)] = K;
            }
        }
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            out.hessian_bohr = Some(h);
        }
        Ok(out)
    }
}

fn registry() -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    r.register("double_well", |_| Ok(Box::new(DoubleWell)));
    r
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_replicas_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const BOHR: f64 = 0.529_177_210_544;

/// A whole structure (Angstrom in the file) with the core at `x` and the environment at `u`.
fn frame(x: f64, u: f64) -> String {
    format!(
        "2\n\nAr {:.10} {:.10} {:.10}\nAr {:.10} 0.0 0.0\n",
        x * BOHR,
        0.02 * BOHR,
        -0.01 * BOHR,
        (OFFSET + u) * BOHR
    )
}

/// The input at `x` (environment in the well `u = +1`) and the file of the two replicas.
fn inputs(dir: &Path, x: f64) -> (String, String) {
    let input = dir.join("start.xyz");
    std::fs::write(&input, frame(x, 1.0)).unwrap();
    let replicas = dir.join("start_replicas.xyz");
    std::fs::write(&replicas, format!("{}{}", frame(x, 1.0), frame(x, -1.0))).unwrap();
    (
        input.to_string_lossy().into_owned(),
        replicas.to_string_lossy().into_owned(),
    )
}

fn args<'a>(input: &'a str, extra: &[&'a str], options: &'a str) -> Vec<&'a str> {
    let mut a = vec![
        input,
        "-os",
        "double_well",
        "-oniom",
        "1",
        "none",
        "double_well",
        "-oniom_opt",
    ];
    a.extend(options.split_whitespace());
    a.extend(extra);
    a
}

fn core_x(path: &Path) -> f64 {
    read_xyz_file(path, SpinCharge::default(), None)
        .unwrap()
        .coords_flat_bohr()[0]
}

#[test]
fn a_minimization_and_a_saddle_search_with_two_replicas_converge_and_are_verified() {
    let dir = scratch("runs");
    let registry = registry();
    // The minimum near x = 1 with both replicas.
    let (input, replicas) = inputs(&dir, 0.8);
    let options = format!(
        "micro_replicas=2 micro_replica_start=files micro_replica_files={replicas} \
         micro_precondition=none micro_tolerance=1e-10"
    );
    let config = optmain::config_from_cli(&args(&input, &["-tcc"], &options)).unwrap();
    let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(out[0].converged);
    let x = core_x(&out[0].optimized);
    // f'(x) = B tanh(B x / kT) at the minimum: x a little above 1.
    let beta = 1.0 / (3.166_811_563e-6 * 298.15);
    let residual = 4.0 * C4 * x * (x * x - 1.0) - B * (beta * B * x).tanh();
    assert!(
        residual.abs() < 2e-5 && x > 1.0,
        "x = {x}, residual {residual}"
    );
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("ONIOM replicas (new): 2 replicas"), "{log}");
    assert!(log.contains("soft minimum of 2 replicas"), "{log}");
    // The structure file holds the lowest replica (u = -1 for x > 0) and says so; the file
    // beside it holds both.
    let written = std::fs::read_to_string(&out[0].optimized).unwrap();
    assert!(written.contains("micro_replica=2/2"), "{written}");
    let stem = out[0]
        .optimized
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let beside = out[0]
        .optimized
        .with_file_name(format!("{stem}_replicas.xyz"));
    let both = read_xyz_trajectory_file(&beside, SpinCharge::default(), None).unwrap();
    assert_eq!(both.frames.len(), 2);
    let u: Vec<f64> = both
        .frames
        .iter()
        .map(|f| f.system.coords_flat_bohr()[3] - OFFSET)
        .collect();
    assert!(
        (u[0] - 1.0).abs() < 1e-6 && (u[1] + 1.0).abs() < 1e-6,
        "{u:?}"
    );

    // The saddle point at x = 0, verified by the Hessian of a check of stability (both
    // replicas relaxed to the tolerance of the checks): one imaginary mode.
    let (input, replicas) = inputs(&dir, 0.15);
    let options = format!(
        "micro_replicas=2 micro_replica_start=files micro_replica_files={replicas} \
         micro_precondition=none micro_tolerance=1e-10"
    );
    let saddle_args = args(&input, &["-order", "1", "-fc", "5", "-tcc"], &options);
    let config = optmain::config_from_cli(&saddle_args).unwrap();
    let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(out[0].converged);
    let x = core_x(&out[0].optimized);
    assert!(x.abs() < 1e-4, "x = {x}");
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("is verified with the Hessian of a check of stability"),
        "{log}"
    );
    assert!(log.contains("every one of the 2 replicas"), "{log}");
    assert!(
        log.contains("Hessian of the soft minimum for a check of stability")
            || log.contains("soft minimum of 2 replicas"),
        "{log}"
    );
    // One environment alone (the input's, u = +1) has its saddle point where f' = -B instead.
    let one = "micro_precondition=none micro_tolerance=1e-10";
    let config =
        optmain::config_from_cli(&args(&input, &["-order", "1", "-fc", "5", "-tcc"], one)).unwrap();
    let alone = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(alone[0].converged);
    let x1 = core_x(&alone[0].optimized);
    assert!(
        (4.0 * C4 * x1 * (x1 * x1 - 1.0) + B).abs() < 2e-5,
        "x = {x1}"
    );
    // A run from the structure of the search with one environment says that it is not on
    // the surface of the replicas.
    let from = out[0].optimized.to_string_lossy().into_owned();
    let config = optmain::config_from_cli(&args(&from, &[], one)).unwrap();
    let later = optmain::run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(later[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("not on that surface"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_with_replicas_restarts_from_its_journal_to_the_same_result() {
    let dir = scratch("restart");
    let registry = registry();
    let (input, replicas) = inputs(&dir, 0.6);
    let options = format!(
        "micro_replicas=2 micro_replica_start=files micro_replica_files={replicas} \
         micro_precondition=none micro_tolerance=1e-10"
    );
    let a = args(&input, &["-opt", "rsirfo_bfgs", "-fc", "3"], &options);
    let config = optmain::config_from_cli(&a).unwrap();
    let full = optmain::run_with(&config, &dir, None, &registry).unwrap();
    assert!(full[0].converged);
    let reference = std::fs::read_to_string(&full[0].optimized).unwrap();
    let run_dir = full[0].directory.clone();
    let journal = run_dir.join(JOURNAL_FILE);
    // Killed after half of the evaluations.
    let (records, _) = mopt_rs::workflow::journal::read_journal(&journal).unwrap();
    let k = records.len() / 2;
    assert!(k >= 2, "{} records", records.len());
    let bytes = std::fs::read(&journal).unwrap();
    let mut cut = 8usize;
    for _ in 0..k {
        let len = u32::from_le_bytes(bytes[cut..cut + 4].try_into().unwrap()) as usize;
        cut += 8 + len;
    }
    std::fs::write(&journal, &bytes[..cut]).unwrap();
    std::fs::remove_file(&full[0].optimized).unwrap();
    let mut again = optmain::config_from_cli(&a).unwrap();
    again.restart_dir = Some(run_dir.clone());
    let resumed = optmain::run_with(&again, &dir, None, &registry).unwrap();
    assert!(resumed[0].converged);
    assert_eq!(resumed[0].directory, run_dir);
    assert_eq!(
        std::fs::read_to_string(&resumed[0].optimized).unwrap(),
        reference
    );
    let _ = std::fs::remove_dir_all(&dir);
}
