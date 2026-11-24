//! The Rust interfaces as shown in `docs/source/running.md` ("From Rust"): every block of that
//! section is one step of this test, so the documentation stays correct.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::{Col, Mat};

use mopt_rs::cli::optmain::{config_from_cli, run_with};
use mopt_rs::cli::run_command_with;
use mopt_rs::domain::{
    positions, Angstrom, Element, EvalOutput, EvalRequest, MolecularSystem, Quantity, SpinCharge,
};
use mopt_rs::error::MoptResult;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::opt::vector::{VectorOptimizer, VectorSettings};
use mopt_rs::pes::objective::{MullerBrown2D, ObjectiveRegistry};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::pes::{evaluate_checked, Calculator};
use mopt_rs::workflow::journal::RunJournal;
use mopt_rs::workflow::{run_optimization, run_vector_optimization};

// --- Step 1: a program of your own behind the `Calculator` trait -------------------------------
// Calculators receive Cartesian structures (Bohr) and return the energy (Hartree), gradient
// (Hartree/Bohr), and, when requested, the Hessian (Hartree/Bohr^2). Any coordinate system,
// optimizer, bias, or constraint then works with it.

/// E = k/2 |x - x0|^2: a stand-in for an external program.
struct Harmonic {
    x0: Vec<f64>,
    k: f64,
}

impl Calculator for Harmonic {
    fn name(&self) -> &str {
        "harmonic"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        let n = x.nrows();
        let d = Col::from_fn(n, |i| x[i] - self.x0[i]);
        let energy = 0.5 * self.k * d.iter().map(|v| v * v).sum::<f64>();
        let mut out = EvalOutput::energy_gradient(Quantity::new(energy), d * faer::Scale(self.k));
        if req.need_hessian {
            out.hessian_bohr = Some(Mat::<f64>::identity(n, n) * faer::Scale(self.k));
        }
        Ok(out)
    }

    // The minimum is a fixed point in space, so rigid-body motion must not be removed.
    fn is_rigid_body_invariant(&self) -> bool {
        false
    }
}

fn water(shift: f64) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::O, Element::H, Element::H],
        &positions::<Angstrom>(&[
            [0.0, 0.0, 0.117 + shift],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469 - shift],
        ]),
        SpinCharge::new(0, 1),
        None,
    )
    .unwrap()
}

#[test]
fn rust_interfaces_as_documented() {
    let target = water(0.0);
    let x0: Vec<f64> = target.coords_flat_bohr().iter().copied().collect();
    let calc = Harmonic {
        x0: x0.clone(),
        k: 0.5,
    };

    // --- Step 2: evaluate and optimize with it ---------------------------------------------
    let start = water(0.08);
    let e = evaluate_checked(&calc, &start, &EvalRequest::ENERGY_GRAD).unwrap();
    assert!(e.energy.value() > 0.0);
    let settings = OptimizerSettings {
        methods: vec!["rsirfo_bfgs".to_string()],
        rigid_body_invariant: calc.is_rigid_body_invariant(),
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings.clone(), start.clone()).unwrap();
    let out = run_optimization(&mut opt, &calc, &mut []).unwrap();
    assert_eq!(out.termination, Some(Termination::Converged));
    let x = out.system.coords_flat_bohr();
    assert!(x.iter().zip(&x0).all(|(a, b)| (a - b).abs() < 1e-4));

    // --- Step 3: register it, so that every command finds it by name (`-os harmonic`) ------
    let mut registry = CalculatorRegistry::native();
    let x0_for_factory = x0.clone();
    registry.register("harmonic", move |req| {
        // `req.options` carries -func, -bs, -calc_opt ...; `req.argument` is the part after ':'.
        let k = req
            .options
            .extra
            .iter()
            .find(|(key, _)| key == "k")
            .map_or(Ok(0.5), |(_, v)| v.parse::<f64>())
            .map_err(|e| mopt_rs::error::MoptError::parse(format!("k: {e}")))?;
        Ok(Box::new(Harmonic {
            x0: x0_for_factory.clone(),
            k,
        }))
    });
    let by_name = registry
        .create("harmonic", &CalculatorOptions::default(), &start)
        .unwrap();
    assert_eq!(by_name.name(), "harmonic");
    let lj = registry
        .create("lj", &CalculatorOptions::default(), &start)
        .unwrap();
    assert_eq!(lj.name(), "lj");

    let base = std::env::temp_dir().join(format!("mopt_rust_api_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let xyz = base.join("water.xyz");
    let mut text = String::from("3\n\n");
    for (el, k) in ["O", "H", "H"].iter().zip(0..3) {
        let p = start.position(mopt_rs::domain::AtomId::new(k)).value();
        let a = p.map(|v| v * mopt_rs::domain::Units::BOHR_TO_ANGSTROM);
        text.push_str(&format!("{el} {:.10} {:.10} {:.10}\n", a[0], a[1], a[2]));
    }
    std::fs::write(&xyz, text).unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let config = config_from_cli(&[&xyz, "-os", "harmonic", "-calc_opt", "k=0.8"]).unwrap();
    let runs = run_with(&config, &base, None, &registry).unwrap();
    assert!(runs[0].converged);
    // The same through the command dispatcher (as the `optmain` binary does).
    let cwd = std::env::current_dir().unwrap();
    std::env::set_current_dir(&base).unwrap();
    let code = run_command_with(
        "optmain",
        &[xyz.clone(), "-os".into(), "harmonic".into()],
        &registry,
    );
    std::env::set_current_dir(cwd).unwrap();
    assert_eq!(code, 0);

    // --- Step 4: restartable evaluations in your own driver ---------------------------------
    let dir = base.join("my_run");
    std::fs::create_dir_all(&dir).unwrap();
    let journal = RunJournal::open(&dir).unwrap();
    let recorded = journal.wrap("main", &calc);
    let mut opt = GeometryOptimizer::new(settings.clone(), start.clone()).unwrap();
    let first = run_optimization(&mut opt, &recorded, &mut []).unwrap();
    drop(journal);
    let journal = RunJournal::open(&dir).unwrap(); // a restart
    let replayed = journal.wrap("main", &calc);
    let mut opt = GeometryOptimizer::new(settings, start).unwrap();
    let again = run_optimization(&mut opt, &replayed, &mut []).unwrap();
    assert_eq!(journal.replay_counts(), (first.iterations, 0));
    assert_eq!(again.iterations, first.iterations);

    // --- Step 5: plain variables (no atoms) -------------------------------------------------
    let settings = VectorSettings {
        methods: vec!["rsirfo_bfgs".into()],
        ..VectorSettings::default()
    };
    let mut opt = VectorOptimizer::new(settings, Col::from_fn(2, |i| [-0.4, 1.3][i])).unwrap();
    let outcome =
        run_vector_optimization(&mut opt, &MullerBrown2D::default(), &mut |_report| Ok(()))
            .unwrap();
    assert_eq!(outcome.termination, Some(Termination::Converged));
    assert!((outcome.x[0] + 0.558).abs() < 2e-2 && (outcome.x[1] - 1.442).abs() < 2e-2);
    let options = CalculatorOptions::default();
    let objective = ObjectiveRegistry::builtin()
        .create("expr:(q1-1)^2 + q2^2", &options)
        .unwrap();
    let v = objective
        .evaluate(&Col::from_fn(2, |i| [1.0, 0.0][i]), true)
        .unwrap();
    assert!(v.value.abs() < 1e-14 && v.hessian.is_some());
    let _ = std::fs::remove_dir_all(&base);
}
