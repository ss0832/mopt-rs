//! Comparison with reference data produced by the original MultiOptPy (tests/golden, generated
//! by tools/golden/run_reference.py) on the Lennard-Jones surface.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod mop_units;

use mopt_rs::domain::{positions, Bohr};
use std::path::PathBuf;

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Element, EvalRequest, MolecularSystem, SpinCharge};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{Action, GeometryOptimizer, OptimizerSettings, Termination};

fn golden(name: &str) -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    JsonValue::parse(&text).unwrap()
}

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|x| match x.as_array() {
            Some(inner) => inner
                .iter()
                .map(|y| y.as_f64().unwrap())
                .collect::<Vec<_>>(),
            None => vec![x.as_f64().unwrap()],
        })
        .collect()
}

fn system(symbols: &JsonValue, coords_bohr: &[f64]) -> MolecularSystem {
    let el: Vec<Element> = symbols
        .as_string_vec()
        .unwrap()
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    let pos: Vec<[f64; 3]> = coords_bohr.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    MolecularSystem::new(el, &positions::<Bohr>(&pos), SpinCharge::default(), None).unwrap()
}

#[test]
fn single_point_matches_multioptpy() {
    let g = golden("single_point_lj.json");
    for p in g.get("points").unwrap().as_array().unwrap() {
        let sys = system(
            p.get("symbols").unwrap(),
            &floats(p.get("coords_bohr").unwrap()),
        );
        // MultiOptPy's numbers are in its own Bohr and Hartree (see `mop_units`).
        let out = mop_units::lennard_jones(&sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
        let e_ref = p.get("energy_hartree").unwrap().as_f64().unwrap();
        assert!((out.energy.value() - e_ref).abs() < 1e-14 * e_ref.abs().max(1.0));
        for (a, b) in out
            .gradient_bohr
            .iter()
            .zip(floats(p.get("gradient").unwrap()))
        {
            assert!((a - b).abs() < 1e-14);
        }
        // MultiOptPy's analytic LJ Hessian has the wrong sign on the V'(r)/r terms (it differs
        // from finite differences of its own gradient by 8.4e-5); mopt-rs' Hessian is checked
        // against finite differences in `pes::native::lennard_jones`. The reference Hessian is
        // therefore expected to differ.
        let h_ref = floats(p.get("hessian").unwrap());
        let h = out.hessian_bohr.unwrap();
        let n = sys.dof();
        let worst = (0..n * n).fold(0.0_f64, |m, k| m.max((h[(k / n, k % n)] - h_ref[k]).abs()));
        assert!(worst > 1e-6);
    }
}

struct Run {
    energies: Vec<f64>,
    coords: Vec<Col<f64>>,
    termination: Termination,
}

fn run_rust(file: &str) -> (JsonValue, Run) {
    let g = golden(file);
    let args: Vec<String> = g.get("arguments").unwrap().as_string_vec().unwrap();
    // The reference runs without -opt used MultiOptPy's default method.
    let mut settings = OptimizerSettings {
        methods: vec!["FIRELARS".to_string()],
        ..OptimizerSettings::default()
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-opt" => settings.methods = vec![it.next().unwrap().clone()],
            "-tcc" => settings.criteria = ConvergenceCriteria::TIGHT,
            _ => {}
        }
    }
    let start = system(
        g.get("symbols").unwrap(),
        &floats(g.get("initial_coords_bohr").unwrap()),
    );
    let mut opt = GeometryOptimizer::new(settings, start).unwrap();
    let mut run = Run {
        energies: Vec::new(),
        coords: Vec::new(),
        termination: Termination::MaxIterations,
    };
    loop {
        match opt.ask() {
            Action::Finished(t) => {
                run.termination = t;
                break;
            }
            Action::Evaluate(tasks) => {
                assert_eq!(tasks.len(), 1);
                let task = &tasks[0];
                let out = mop_units::lennard_jones(&task.system, &task.request).unwrap();
                run.energies.push(out.energy.value());
                run.coords.push(task.system.coords_flat_bohr());
                opt.tell_one(out).unwrap();
            }
        }
    }
    (g, run)
}

/// Index of the first frame whose coordinates differ by more than `tol`.
fn first_divergence(g: &JsonValue, run: &Run, tol: f64) -> Option<usize> {
    let frames = g.get("frames").unwrap().as_array().unwrap();
    for (k, f) in frames.iter().enumerate() {
        let Some(x) = run.coords.get(k) else {
            return Some(k);
        };
        let x_ref = floats(f.get("coords_bohr").unwrap());
        if x.iter().zip(&x_ref).any(|(a, b)| (a - b).abs() > tol) {
            return Some(k);
        }
    }
    (run.coords.len() != frames.len()).then_some(frames.len())
}

#[test]
fn firelars_trajectories_match_multioptpy() {
    for file in ["opt_lj_ar7_firelars.json", "opt_lj_ar7_firelars_tcc.json"] {
        let (g, run) = run_rust(file);
        let n_ref = g.get("iterations").unwrap().as_usize().unwrap();
        assert_eq!(first_divergence(&g, &run, 1e-8), None, "{file}");
        assert_eq!(run.energies.len(), n_ref, "{file}");
        assert_eq!(run.termination, Termination::Converged, "{file}");
        let e_ref = g.get("final_energy_hartree").unwrap().as_f64().unwrap();
        assert!((run.energies[n_ref - 1] - e_ref).abs() < 1e-12, "{file}");
    }
}

#[test]
fn rsirfo_runs_agree_with_multioptpy() {
    for file in [
        "opt_lj_ar7_rsirfo_bfgs.json",
        "opt_lj_ar7_rsirfo_fsb.json",
        "opt_lj_ar7_rsirfo_bofill.json",
    ] {
        let (g, run) = run_rust(file);
        let n_ref = g.get("iterations").unwrap().as_usize().unwrap();
        let div = first_divergence(&g, &run, 1e-8);
        eprintln!(
            "{file}: reference {n_ref} iterations, mopt-rs {}, first divergence {div:?}",
            run.energies.len()
        );
        assert_eq!(run.termination, Termination::Converged, "{file}");
        let e_ref = g.get("final_energy_hartree").unwrap().as_f64().unwrap();
        let e = *run.energies.last().unwrap();
        assert!(
            (e - e_ref).abs() < 1e-6 * e_ref.abs(),
            "{file}: {e} vs {e_ref}"
        );
    }
}
