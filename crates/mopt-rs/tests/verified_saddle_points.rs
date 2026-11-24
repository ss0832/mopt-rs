//! A saddle-point search judged by the Hessian of a check of stability
//! (`ModeKillSettings::checked_hessian`, which the searches on the relaxed surface of ONIOM
//! microiterations take): the converged structure is verified with such a Hessian even where
//! the converging iteration had one for its steps, a surplus imaginary mode is left and the
//! search goes on, and a structure that keeps it after the displacements does not end as
//! converged. A surface whose Hessian for the steps differs from that of the checks stands for
//! the relaxed surface (whose environment a check relaxes tighter).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

use faer::{Col, Mat};
use mopt_rs::domain::{Element, EvalOutput, EvalRequest, MolecularSystem, Quantity, SpinCharge};
use mopt_rs::error::MoptResult;
use mopt_rs::opt::mode_kill::ModeKillSettings;
use mopt_rs::opt::state_machine::{
    GeometryOptimizer, IterationReport, OptimizerSettings, Termination,
};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::{run_optimization, Observer};

/// `E = -a x^2 / 2 + b y^2 / 2 + c z^2 / 2` of one atom (Hartree, Bohr): a first-order saddle
/// at the origin. The Hessian of a check of stability has `check_y` along `y` in the place of
/// `b`, the one for the steps the true `b`.
struct TwoFaced {
    check_y: f64,
}

const A: f64 = 0.1;
const B: f64 = 0.04;
const C: f64 = 0.3;

impl Calculator for TwoFaced {
    fn name(&self) -> &str {
        "two_faced"
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x = sys.coords_flat_bohr();
        let energy = -0.5 * A * x[0] * x[0] + 0.5 * B * x[1] * x[1] + 0.5 * C * x[2] * x[2];
        let gradient = Col::from_fn(3, |i| [-A * x[0], B * x[1], C * x[2]][i]);
        let mut out = EvalOutput::energy_gradient(Quantity::new(energy), gradient);
        if req.need_hessian {
            let y = if req.for_stability { self.check_y } else { B };
            out.hessian_bohr = Some(Mat::from_fn(
                3,
                3,
                |i, j| {
                    if i == j {
                        [-A, y, C][i]
                    } else {
                        0.0
                    }
                },
            ));
        }
        Ok(out)
    }
}

#[derive(Default)]
struct Notes(Vec<String>);

impl Observer for Notes {
    fn on_iteration(&mut self, report: &IterationReport) -> MoptResult<()> {
        self.0.extend(report.notes.iter().cloned());
        Ok(())
    }
}

fn search(check_y: f64, checked: bool) -> (Option<Termination>, usize, Vec<String>, Col<f64>) {
    let start = MolecularSystem::from_flat_bohr(
        vec![Element::parse_token("Ar").unwrap()],
        &Col::from_fn(3, |i| [0.02, 0.03, -0.01][i]),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    let settings = OptimizerSettings {
        methods: vec!["rsirfo_bofill".to_string()],
        saddle_order: 1,
        exact_hessian_interval: Some(1),
        rigid_body_invariant: false,
        max_iterations: 60,
        mode_kill: Some(ModeKillSettings {
            max_rounds: 2,
            checked_hessian: checked,
            ..ModeKillSettings::default()
        }),
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, start).unwrap();
    let mut notes = Notes::default();
    let out = run_optimization(&mut opt, &TwoFaced { check_y }, &mut [&mut notes]).unwrap();
    let x = opt.current().coords_flat_bohr();
    (out.termination, out.iterations, notes.0, x)
}

#[test]
fn a_checked_hessian_with_a_surplus_mode_sends_the_search_on_and_then_stops_it() {
    // The check sees a second imaginary mode along y: the search leaves the structure along
    // it, converges again (the true surface has none), is checked again, and after the two
    // displacements it may take it ends without convergence.
    let (termination, _, notes, _) = search(-0.02, true);
    let left = notes
        .iter()
        .filter(|n| n.starts_with("Mode kill: the converged structure has 2 imaginary"))
        .count();
    assert_eq!(left, 2, "{notes:#?}");
    assert!(
        notes
            .iter()
            .any(|n| n.contains("not a stationary point of the order")),
        "{notes:#?}"
    );
    assert_eq!(termination, Some(Termination::MaxIterations));
    // Judged by the Hessian of the steps (the former behaviour), the same search ends at the
    // converging iteration, whose exact Hessian has one imaginary mode.
    let (termination, _, notes, _) = search(-0.02, false);
    assert_eq!(termination, Some(Termination::Converged), "{notes:#?}");
    assert!(!notes
        .iter()
        .any(|n| n.contains("imaginary mode(s), order 1")));
}

#[test]
fn a_checked_hessian_with_one_imaginary_mode_ends_the_search_at_the_checked_structure() {
    // The check has one imaginary mode and a soft curvature along y, along which its Newton
    // step from the converged structure exceeds the displacement threshold: the structure,
    // which met every criterion at the iteration before, ends the search where its forces
    // still meet theirs.
    let (termination, iterations, notes, x) = search(1e-5, true);
    assert_eq!(termination, Some(Termination::Converged), "{notes:#?}");
    assert!(
        notes
            .iter()
            .any(|n| n.starts_with("Converged; the Hessian is requested here")),
        "{notes:#?}"
    );
    assert!(iterations < 60);
    assert!(x.iter().all(|v| v.abs() < 1e-3), "{x:?}");
}
