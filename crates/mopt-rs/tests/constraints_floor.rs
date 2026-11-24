//! Hard constraints hold to the rounding floor of double precision at every iteration of an
//! optimization: every structure the optimizer hands out for evaluation is checked through the
//! API (not from files) against the constraints resolved at the input. Frozen atoms and Cartesian
//! components are bit-identical to the input; distances, centroid distances, angles, dihedrals,
//! oriented groups, fixed directions, and user functions of distances are within `1e-10` (Bohr or
//! rad). The cost of the restore is measured on a run with ten constraints.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

use std::time::Instant;

use mopt_rs::cli::constraints::{resolve_for_run, user_constraint};
use mopt_rs::domain::{positions, Angstrom, AtomId, Element, MolecularSystem, SpinCharge};
use mopt_rs::domain::{PeriodicCell, Vector3};
use mopt_rs::error::MoptResult;
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{GeometryOptimizer, IterationReport, OptimizerSettings};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::workflow::{run_optimization, Observer};

/// Largest residual accepted in memory (Bohr or rad): the guarantee every hard constraint is
/// held to (`pes::constraint::GUARANTEE`); the restore reaches about 1e-14 here, printed.
const FLOOR: f64 = 1e-10;

fn id(i: usize) -> AtomId {
    AtomId::new(i)
}

/// Thirteen argon atoms: an icosahedron about a centre atom, distorted.
fn ar13(cell: Option<PeriodicCell>) -> MolecularSystem {
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    let mut v = vec![[0.0, 0.0, 0.0]];
    for (a, b) in [(1.0, phi), (-1.0, phi), (1.0, -phi), (-1.0, -phi)] {
        v.push([0.0, a, b]);
        v.push([a, b, 0.0]);
        v.push([b, 0.0, a]);
    }
    let scale = 3.75 / (1.0 + phi * phi).sqrt();
    let pos: Vec<[f64; 3]> = v
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let n = i as f64;
            std::array::from_fn(|c| {
                p[c] * scale
                    + 0.12 * (1.7 * n + 0.9 * c as f64).sin()
                    + if cell.is_some() { 6.0 } else { 0.0 }
            })
        })
        .collect();
    MolecularSystem::new(
        vec![Element::from_atomic_number(18).unwrap(); pos.len()],
        &positions::<Angstrom>(&pos),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

/// Ten constraints of every kind that keeps the frame (when `pinned`), or the six that are
/// invariant to rigid-body motion.
fn ten_constraints(start: &MolecularSystem, pinned: bool) -> Vec<Constraint> {
    let d612 = start.distance(id(6), id(12)).value();
    let mut c = vec![
        Constraint::at_start(ConstraintKind::Distance(vec![id(1)], vec![id(2)])),
        // A centroid distance between two fragments whose atoms stay free.
        Constraint::at_start(ConstraintKind::Distance(
            vec![id(1), id(2), id(3)],
            vec![id(7), id(8), id(9)],
        )),
        Constraint::at_start(ConstraintKind::Angle([id(4), id(0), id(5)])),
        Constraint::at_start(ConstraintKind::Dihedral([id(1), id(4), id(6), id(10)])),
        user_constraint("dist(3,7) - dist(4,8)").unwrap(),
        Constraint::with_target(
            ConstraintKind::Distance(vec![id(12)], vec![id(6)]),
            d612 + 0.2,
        ),
    ];
    if pinned {
        c.extend([
            Constraint::at_start(ConstraintKind::Frozen(vec![id(0)])),
            Constraint::at_start(ConstraintKind::Coordinate(id(11), 2)),
            Constraint::at_start(ConstraintKind::Orientation(vec![id(10), id(11), id(12)])),
            Constraint::at_start(ConstraintKind::AtomPair([id(5), id(8)])),
        ]);
    }
    c
}

/// Checks every structure the optimizer is about to have evaluated.
struct Check {
    set: ConstraintSet,
    input: MolecularSystem,
    frozen: Vec<usize>,
    coordinate: Option<(usize, usize)>,
    worst: f64,
    structures: usize,
}

impl Observer for Check {
    fn on_iteration(&mut self, _report: &IterationReport) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self, current: &MolecularSystem) -> bool {
        self.structures += 1;
        let r = self.set.residuals(current).unwrap();
        let worst = r.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        self.worst = self.worst.max(worst);
        assert!(worst <= FLOOR, "structure {}: {r:?}", self.structures);
        for &a in &self.frozen {
            assert_eq!(
                current.position(id(a)).value(),
                self.input.position(id(a)).value(),
                "frozen atom {a}, structure {}",
                self.structures
            );
        }
        if let Some((a, k)) = self.coordinate {
            assert_eq!(
                current.position(id(a)).value()[k].to_bits(),
                self.input.position(id(a)).value()[k].to_bits()
            );
        }
        false
    }
}

fn settings(method: &str, coords: &str, constraints: Vec<Constraint>) -> OptimizerSettings {
    OptimizerSettings {
        methods: vec![method.to_string()],
        max_iterations: 300,
        criteria: ConvergenceCriteria::default(),
        coordinates: coords.to_string(),
        constraints,
        ..OptimizerSettings::default()
    }
}

fn run(case: &str, method: &str, coords: &str, start: MolecularSystem, pinned: bool) {
    // Resolved once at the input, as the commands do; the optimizer starts on them.
    let constraints = resolve_for_run(&ten_constraints(&start, pinned), &start).unwrap();
    let set = ConstraintSet::new(&constraints, &start, None).unwrap();
    let on = set.restore(&start).unwrap();
    let mut check = Check {
        set,
        input: on.clone(),
        frozen: if pinned { vec![0] } else { Vec::new() },
        coordinate: pinned.then_some((11, 2)),
        worst: 0.0,
        structures: 0,
    };
    let t = Instant::now();
    let mut opt = GeometryOptimizer::new(settings(method, coords, constraints), on).unwrap();
    let out = run_optimization(&mut opt, &LennardJones, &mut [&mut check]).unwrap();
    let per = t.elapsed().as_secs_f64() / out.iterations.max(1) as f64 * 1e3;
    assert!(check.structures > 3, "{case}");
    // The final structure as well.
    let _ = check.should_stop(&out.system);
    println!(
        "{case}: {} structures, {} iterations, {per:.2} ms per iteration, largest residual {:.2e}",
        check.structures, out.iterations, check.worst
    );
}

#[test]
fn ten_constraints_hold_at_every_iteration_in_cartesian_coordinates() {
    run("cart rsirfo", "rsirfo_bfgs", "cart", ar13(None), true);
    run("cart fire", "fire", "cart", ar13(None), true);
}

#[test]
fn invariant_constraints_hold_at_every_iteration_with_superposition_and_internal_coordinates() {
    // Without frame-pinning constraints the optimizer superposes every new structure on the
    // previous one: the restore comes after it.
    run("cart superposed", "rsirfo_bfgs", "cart", ar13(None), false);
    run("dic", "rsirfo_bfgs", "dic", ar13(None), false);
}

#[test]
fn ten_constraints_hold_at_every_iteration_in_a_periodic_cell() {
    // Large enough that no constrained pair is closer through the boundary than within it
    // (the cluster is 7.5 Angstrom across).
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([16.5, 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.4, 16.0, 0.0]),
        Vector3::<Angstrom>::new([-0.3, 0.2, 16.8]),
        [true; 3],
    )
    .unwrap();
    run("periodic", "rsirfo_bfgs", "cart", ar13(Some(cell)), true);
}

/// The cost of a restore with ten constraints from a displacement of the size of a step
/// (0.05 Bohr per coordinate), averaged over many restores.
#[test]
fn cost_of_a_restore_with_ten_constraints() {
    let start = ar13(None);
    let set = ConstraintSet::new(&ten_constraints(&start, true), &start, None).unwrap();
    let on = set.restore(&start).unwrap();
    let x = on.coords_flat_bohr();
    let rounds = 200;
    let moved: Vec<MolecularSystem> = (0..rounds)
        .map(|k| {
            on.with_flat_coords_bohr(&faer::Col::from_fn(x.nrows(), |i| {
                x[i] + 0.05 * ((i as f64) * 1.3 + f64::from(k) * 0.7).sin()
            }))
            .unwrap()
        })
        .collect();
    let t = Instant::now();
    let mut worst = 0.0_f64;
    let mut steps = 0;
    for m in &moved {
        let back = set.restore_reported(m, None).unwrap();
        steps += back.newton_steps;
        let r = set.residuals(&back.system).unwrap();
        worst = worst.max(r.iter().fold(0.0_f64, |a, v| a.max(v.abs())));
        assert!(back.largest_floor_ratio <= 1.0 + 1e-12 || back.largest_residual <= FLOOR);
    }
    let per = t.elapsed().as_secs_f64() / f64::from(rounds) * 1e6;
    println!(
        "restore of 10 constraints: {per:.0} us per restore, {:.2} Newton steps per restore, \
         largest residual {worst:.2e}",
        steps as f64 / f64::from(rounds)
    );
    assert!(worst <= FLOOR, "{worst}");
}
