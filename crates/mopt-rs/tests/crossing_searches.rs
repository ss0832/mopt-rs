//! Searches for crossing points of two states (`-mf opt_mesx`, `-mf opt_meci`) at their edges:
//! the check of the branching plane that runs out of corrections, the check under
//! constraints, and two states whose gradients are the same everywhere (no direction that
//! closes the gap).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use mopt_rs::domain::{positions, AtomId, Bohr, EvalOutput, EvalRequest, Quantity};
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::error::MoptResult;
use mopt_rs::math::ad::{eval_value_grad_hessian, HyperDual, Scalar};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::crossing_check::CrossingCheckSettings;
use mopt_rs::opt::state_machine::{Action, GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::pes::combine::{model_function, two_states, TaskSpec};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::CalculatorSet;

/// Two harmonic diabatic wells of the first atom, `V1 = k/2 |r - a|^2` and
/// `V2 = k/2 |r - b|^2 + c`, coupled by `lambda z`; the adiabatic states (by the root) meet
/// at a conical intersection on the line `x = 0.52`, `z = 0` for the values below. Every
/// further atom is tied to the first by a spring `s/2 |r_i - r_0 - (i, 0, 0)|^2` that both
/// states share. `parallel`: the second state is the first shifted by `c` (the same gradient
/// everywhere).
struct Wells {
    spring: f64,
    parallel: bool,
}

/// The diabatic wells of [`Wells`] by the multiplicity (singlet `V1`, triplet `V2`): two
/// states of different spin, which cross on the plane `x = 0.52` without coupling.
struct SpinWells;

impl Calculator for SpinWells {
    fn name(&self) -> &str {
        "spin_wells"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let triplet = sys.spin_charge.multiplicity == 3;
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let (e, g, h) = eval_value_grad_hessian(&x, None, |r: &[HyperDual]| {
            let (at, shift) = if triplet { (B, C) } else { (A, 0.0) };
            (0..3).fold(HyperDual::from_f64(shift), |acc, i| {
                let d = r[i] - HyperDual::from_f64(at[i]);
                acc + HyperDual::from_f64(0.5 * K) * d * d
            })
        });
        Ok(EvalOutput {
            energy: Quantity::new(e),
            gradient_bohr: g,
            hessian_bohr: req.need_hessian.then_some(h),
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }
}

const K: f64 = 1.0;
const A: [f64; 3] = [0.0, 0.0, 0.0];
const B: [f64; 3] = [1.0, 0.0, 0.0];
const C: f64 = 0.02;
const LAMBDA: f64 = 0.1;

impl Wells {
    fn energy<T: Scalar>(&self, root: usize, r: &[T]) -> T {
        let half_k = T::from_f64(0.5 * K);
        let sq = |p: [f64; 3]| {
            (0..3).fold(T::zero(), |acc, i| {
                let d = r[i] - T::from_f64(p[i]);
                acc + d * d
            })
        };
        let mut springs = T::zero();
        for atom in 1..r.len() / 3 {
            for i in 0..3 {
                let rest = if i == 0 { atom as f64 } else { 0.0 };
                let d = r[3 * atom + i] - r[i] - T::from_f64(rest);
                springs += T::from_f64(0.5 * self.spring) * d * d;
            }
        }
        let v1 = half_k * sq(A);
        if self.parallel {
            return springs + v1 + T::from_f64(if root == 0 { 0.0 } else { C });
        }
        let v2 = half_k * sq(B) + T::from_f64(C);
        let mean = (v1 + v2) * T::from_f64(0.5);
        let half_gap = (v1 - v2) * T::from_f64(0.5);
        let v12 = r[2] * T::from_f64(LAMBDA);
        let root_term = (half_gap * half_gap + v12 * v12 + T::from_f64(1e-16)).sqrt();
        springs
            + if root == 0 {
                mean - root_term
            } else {
                mean + root_term
            }
    }
}

impl Calculator for Wells {
    fn name(&self) -> &str {
        "wells"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let root = if sys.spin_charge.multiplicity == 3 {
            1
        } else {
            req.excited_state
        };
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let (e, g, h) = eval_value_grad_hessian(&x, None, |r: &[HyperDual]| self.energy(root, r));
        Ok(EvalOutput {
            energy: Quantity::new(e),
            gradient_bohr: g,
            hessian_bohr: req.need_hessian.then_some(h),
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }

    fn supports_excited_states(&self) -> bool {
        true
    }
}

fn structure(atoms: &[[f64; 3]]) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::H; atoms.len()],
        &positions::<Bohr>(atoms),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn settings() -> OptimizerSettings {
    OptimizerSettings {
        methods: vec!["rsirfo_bfgs".to_string()],
        max_iterations: 500,
        criteria: ConvergenceCriteria::TIGHT,
        rigid_body_invariant: false,
        dissociation_distance: None,
        ..OptimizerSettings::default()
    }
}

/// The surface of the conical intersection of the two roots.
fn meci() -> mopt_rs::pes::PesExpr {
    two_states(
        model_function("opt_meci", None, None).unwrap(),
        TaskSpec::whole("State2").excited_state(1),
    )
    .unwrap()
}

/// Run `opt` on `calc` to its end: the termination and every note of its iterations.
fn run(
    opt: &mut GeometryOptimizer,
    calc: &dyn Calculator,
) -> MoptResult<(Option<Termination>, Vec<String>)> {
    let mut notes = Vec::new();
    while let Action::Evaluate(tasks) = opt.ask() {
        let results = CalculatorSet::single(calc).evaluate(&tasks)?;
        let report = opt.tell(results)?;
        notes.extend(report.notes.iter().cloned());
    }
    Ok((opt.termination(), notes))
}

/// What a search cost: its termination, the evaluations of a state with a gradient, those
/// with an exact Hessian, and the end.
struct Cost {
    termination: Option<Termination>,
    gradients: usize,
    hessians: usize,
    end: MolecularSystem,
}

fn counted(mut opt: GeometryOptimizer, calc: &dyn Calculator) -> MoptResult<Cost> {
    let (mut gradients, mut hessians) = (0, 0);
    while let Action::Evaluate(tasks) = opt.ask() {
        gradients += tasks.len();
        hessians += tasks.iter().filter(|t| t.request.need_hessian).count();
        let results = CalculatorSet::single(calc).evaluate(&tasks)?;
        opt.tell(results)?;
    }
    Ok(Cost {
        termination: opt.termination(),
        gradients,
        hessians,
        end: opt.current().clone(),
    })
}

/// The settings of a crossing search that are compared for the defaults of `run_automecp`:
/// the method of the transition-state step of the mapper with its exact Hessians every five
/// iterations (what a crossing search took before), and the minimizers with and without them.
const COMPARED: [(&str, Option<usize>); 4] = [
    ("rsirfo_block_bofill", Some(5)),
    ("rsirfo_block_bofill", None),
    ("rsirfo_block_fsb", Some(5)),
    ("rsirfo_block_fsb", None),
];

#[test]
fn the_cost_of_crossing_searches_by_their_settings() {
    // MECI of the coupled wells and MESX of the spin wells, from three starts each, with the
    // normal and the tight criteria. Printed for docs/source/mapper.md ("Crossing points of
    // run_automecp"); the default must converge everywhere and cost no more than the old one.
    let starts = [[0.3, 0.2, 0.15], [0.7, -0.3, 0.25], [0.1, 0.5, -0.2]];
    let coupled = Wells {
        spring: 0.0,
        parallel: false,
    };
    let mut totals = [(0usize, 0usize, 0usize); COMPARED.len()];
    for (case, name) in [("MECI", "opt_meci"), ("MESX", "opt_mesx")] {
        for tight in [false, true] {
            for start in starts {
                for (k, (method, every)) in COMPARED.iter().enumerate() {
                    let mut s = settings();
                    s.methods = vec![method.to_string()];
                    s.exact_hessian_interval = *every;
                    s.criteria = if tight {
                        ConvergenceCriteria::TIGHT
                    } else {
                        ConvergenceCriteria::NORMAL
                    };
                    let second = if name == "opt_meci" {
                        TaskSpec::whole("State2").excited_state(1)
                    } else {
                        TaskSpec::whole("State2").spin_charge(SpinCharge::new(0, 3))
                    };
                    let surface =
                        two_states(model_function(name, None, None).unwrap(), second).unwrap();
                    let mut opt = GeometryOptimizer::new(s, structure(&[start])).unwrap();
                    opt.set_surface(surface).unwrap();
                    let calc: &dyn Calculator = if name == "opt_meci" {
                        &coupled
                    } else {
                        &SpinWells
                    };
                    let cost = counted(opt, calc).unwrap();
                    let p = position(&cost.end, 0);
                    // Both: x = 0.52 on the seam, and the mean of the wells lowest at y = 0
                    // (z = 0 on the seam of the coupled wells).
                    let target = [0.52, 0.0, 0.0];
                    let off = (0..3)
                        .map(|i| (p[i] - target[i]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let ok = cost.termination == Some(Termination::Converged) && off < 2e-2;
                    println!(
                        "{case} tight={tight} start={start:?} {method} every={every:?}: \
                         {:?}, {} gradients, {} Hessians, {off:.1e} from the point",
                        cost.termination, cost.gradients, cost.hessians
                    );
                    totals[k].0 += cost.gradients;
                    totals[k].1 += cost.hessians;
                    totals[k].2 += usize::from(!ok);
                }
            }
        }
    }
    for (k, (method, every)) in COMPARED.iter().enumerate() {
        println!(
            "total {method} every={every:?}: {} gradients, {} Hessians, {} failed",
            totals[k].0, totals[k].1, totals[k].2
        );
    }
    // The default of run_automecp (the last row) against the old one (the first).
    let (old, new) = (totals[0], totals[COMPARED.len() - 1]);
    assert_eq!(new.2, 0, "{totals:?}");
    assert!(new.0 <= old.0 && new.1 <= old.1, "{totals:?}");
}

fn position(sys: &MolecularSystem, atom: usize) -> [f64; 3] {
    sys.position(AtomId::new(atom)).value()
}

#[test]
fn a_meci_whose_check_runs_out_of_corrections_is_not_converged() {
    // From a start in the plane z = 0 the search never sees the coupling direction z and
    // stops off the minimum of the seam; the check finds the point not stationary. With no
    // correction allowed the search ends there, with its own reason and not as converged.
    let calc = Wells {
        spring: 0.0,
        parallel: false,
    };
    let mut s = settings();
    s.crossing_check = Some(CrossingCheckSettings {
        max_restarts: 0,
        ..CrossingCheckSettings::default()
    });
    let mut opt = GeometryOptimizer::new(s, structure(&[[0.3, 0.4, 0.0]])).unwrap();
    opt.set_surface(meci()).unwrap();
    let (termination, notes) = run(&mut opt, &calc).unwrap();
    assert_eq!(termination, Some(Termination::SeamCheckFailed), "{notes:?}");
    assert!(
        notes
            .iter()
            .any(|n| n.contains("not stationary on the seam") && n.contains("stops")),
        "{notes:?}"
    );
}

#[test]
fn the_check_of_the_branching_plane_holds_the_constraints() {
    // A frozen atom (`-fix`) tied to the first by a spring: its gradient is the force that
    // holds it, which is no part of the seam; the check must accept the converged point
    // without sending the search on. (The normal criteria: with the tight ones this search
    // stalls next to the cusp with or without the frozen atom, a matter of the steps, not of
    // the check.)
    let calc = Wells {
        spring: 0.2,
        parallel: false,
    };
    let mut s = settings();
    s.criteria = ConvergenceCriteria::NORMAL;
    s.constraints = vec![Constraint::at_start(ConstraintKind::Frozen(vec![
        AtomId::new(1),
    ]))];
    let mut opt =
        GeometryOptimizer::new(s, structure(&[[0.3, 0.4, 0.15], [1.3, 0.1, 0.0]])).unwrap();
    opt.set_surface(meci()).unwrap();
    let (termination, notes) = run(&mut opt, &calc).unwrap();
    assert_eq!(termination, Some(Termination::Converged), "{notes:?}");
    assert!(
        !notes
            .iter()
            .any(|n| n.contains("goes on with this branching plane")),
        "{notes:?}"
    );
    assert!(
        notes
            .iter()
            .any(|n| n.contains("second direction from gradients") && n.ends_with("converged.")),
        "the check did not run: {notes:?}"
    );
    let held = position(opt.current(), 1);
    assert!(
        (0..3).all(|i| (held[i] - [1.3, 0.1, 0.0][i]).abs() < 1e-9),
        "{held:?}"
    );
}

#[test]
fn states_with_the_same_gradient_everywhere_are_refused() {
    // The second state is the first shifted by C: their difference gradient is zero, and
    // no direction closes the gap. The search must not report its start as converged.
    let calc = Wells {
        spring: 0.0,
        parallel: true,
    };
    for (name, second) in [
        ("opt_meci", TaskSpec::whole("State2").excited_state(1)),
        (
            "opt_mesx",
            TaskSpec::whole("State2").spin_charge(SpinCharge::new(0, 3)),
        ),
    ] {
        let surface = two_states(model_function(name, None, None).unwrap(), second).unwrap();
        let mut opt = GeometryOptimizer::new(settings(), structure(&[[0.3, 0.4, 0.15]])).unwrap();
        opt.set_surface(surface).unwrap();
        match run(&mut opt, &calc) {
            Ok((termination, notes)) => {
                panic!("{name}: the search ended {termination:?} on parallel states: {notes:?}")
            }
            Err(e) => {
                let text = e.to_string();
                assert!(
                    text.contains("difference of the gradients") && text.contains("1e-8"),
                    "{name}: {text}"
                );
            }
        }
    }
}
