//! Constrained optimization (`-fix`, `-pc`): exact restoration, constraint derivatives, and
//! constrained minima that agree with stiff-penalty minima, in Cartesian and DIC coordinates.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;
use std::sync::Arc;

use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::domain::{positions, Angstrom, AtomId, Element, MolecularSystem, PeriodicCell};
use mopt_rs::domain::{SpinCharge, Units, Vector3};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::pes::bias::{AdBias, KeepDistance, Ramp};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::workflow::run_optimization;

const A2B: f64 = Units::ANGSTROM_TO_BOHR;

const AR7: [[f64; 3]; 7] = [
    [1.9500, 0.0000, 0.0000],
    [0.6026, 1.8546, 0.0300],
    [-1.5776, 1.1462, 0.0700],
    [-1.5776, -1.1462, 0.1100],
    [0.6026, -1.8546, 0.1500],
    [0.0000, 0.0000, 2.1000],
    [0.2000, 0.0000, -2.0000],
];

fn id(i: usize) -> AtomId {
    AtomId::new(i)
}

fn system(pos: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::from_atomic_number(18).unwrap(); pos.len()],
        &positions::<Angstrom>(pos),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

fn dist(s: &MolecularSystem, a: usize, b: usize) -> f64 {
    s.distance(id(a), id(b)).value()
}

fn sample_constraints() -> Vec<Constraint> {
    vec![
        Constraint {
            kind: ConstraintKind::Distance(vec![id(0)], vec![id(1)]),
            target: Some(3.0 * A2B),
            anchor: None,
        },
        Constraint {
            kind: ConstraintKind::Angle([id(0), id(5), id(2)]),
            target: Some(80f64.to_radians()),
            anchor: None,
        },
        Constraint {
            kind: ConstraintKind::Dihedral([id(0), id(1), id(2), id(3)]),
            target: Some(-30f64.to_radians()),
            anchor: None,
        },
        Constraint {
            kind: ConstraintKind::Distance(vec![id(3), id(4)], vec![id(6)]),
            target: None,
            anchor: None,
        },
        Constraint::at_start(ConstraintKind::Orientation(vec![id(3), id(4), id(6)])),
    ]
}

#[test]
fn restoration_meets_every_constraint_in_isolated_and_periodic_cells() {
    let skew = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([6.0, 0.0, 0.0]),
        Vector3::<Angstrom>::new([1.0, 6.5, 0.0]),
        Vector3::<Angstrom>::new([0.3, -0.4, 7.0]),
        [true, true, false],
    )
    .unwrap();
    for cell in [None, Some(skew)] {
        let start = system(&AR7, cell);
        let set = ConstraintSet::new(&sample_constraints(), &start, None).unwrap();
        // Distort and restore.
        let x = start.coords_flat_bohr();
        let noisy = start
            .with_flat_coords_bohr(&faer::Col::from_fn(x.nrows(), |i| {
                x[i] + 0.15 * ((i as f64) * 1.7).sin()
            }))
            .unwrap();
        let back = set.restore(&noisy).unwrap();
        let r = set.residuals(&back).unwrap();
        assert!(r.iter().all(|v| v.abs() < 1e-9), "{r:?}");
        assert!((dist(&back, 0, 1) - 3.0 * A2B).abs() < 1e-9);
    }
}

#[test]
fn constraint_gradients_match_finite_differences() {
    let start = system(&AR7, None);
    let set = ConstraintSet::new(&sample_constraints(), &start, None).unwrap();
    let zero = faer::Col::zeros(start.dof());
    let j = set.frame(&start, &zero).unwrap().jacobian;
    let x = start.coords_flat_bohr();
    let h = 1e-6;
    for k in 0..x.nrows() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[k] += h;
        xm[k] -= h;
        let rp = set
            .residuals(&start.with_flat_coords_bohr(&xp).unwrap())
            .unwrap();
        let rm = set
            .residuals(&start.with_flat_coords_bohr(&xm).unwrap())
            .unwrap();
        for c in 0..j.nrows() {
            let fd = (rp[c] - rm[c]) / (2.0 * h);
            assert!(
                (fd - j[(c, k)]).abs() < 1e-6,
                "row {c}, dof {k}: {fd} vs {}",
                j[(c, k)]
            );
        }
    }
}

fn settings(method: &str, coords: &str, constraints: Vec<Constraint>) -> OptimizerSettings {
    OptimizerSettings {
        methods: vec![method.to_string()],
        max_iterations: 2000,
        criteria: ConvergenceCriteria::TIGHT,
        coordinates: coords.to_string(),
        constraints,
        ..OptimizerSettings::default()
    }
}

#[test]
fn constrained_minima_agree_with_stiff_penalties() {
    let start = system(&AR7, None);
    let target = 3.2 * A2B;
    // Reference: a stiff harmonic restraint.
    let mut pen =
        GeometryOptimizer::new(settings("rsirfo_bfgs", "cart", vec![]), start.clone()).unwrap();
    pen.add_term(
        "kp",
        Arc::new(AdBias::new(KeepDistance {
            a: vec![id(0)],
            b: vec![id(5)],
            k: Ramp::constant(20.0),
            r0: Ramp::constant(target),
        })),
    )
    .unwrap();
    let reference = run_optimization(&mut pen, &LennardJones, &mut []).unwrap();
    assert_eq!(reference.termination, Some(Termination::Converged));
    let e_ref = LennardJones_energy(&reference.system);
    for (method, coords) in [
        ("rsirfo_bfgs", "cart"),
        ("crsirfo_bfgs", "cart"),
        ("rsirfo_bfgs", "dic"),
        ("fire", "cart"),
    ] {
        let c = vec![Constraint {
            kind: ConstraintKind::Distance(vec![id(0)], vec![id(5)]),
            target: Some(target),
            anchor: None,
        }];
        let mut opt = GeometryOptimizer::new(settings(method, coords, c), start.clone()).unwrap();
        let out = run_optimization(&mut opt, &LennardJones, &mut []).unwrap();
        let tag = format!("{method} in {coords}: {} iterations", out.iterations);
        assert_eq!(out.termination, Some(Termination::Converged), "{tag}");
        assert!((dist(&out.system, 0, 5) - target).abs() < 1e-8, "{tag}");
        let e = LennardJones_energy(&out.system);
        assert!((e - e_ref).abs() < 2e-6, "{tag}: {e} vs penalty {e_ref}");
    }
}

#[allow(non_snake_case)]
fn LennardJones_energy(s: &MolecularSystem) -> f64 {
    use mopt_rs::domain::EvalRequest;
    use mopt_rs::pes::Calculator;
    LennardJones
        .evaluate(s, &EvalRequest::ENERGY_GRAD)
        .unwrap()
        .energy
        .value()
}

#[test]
fn frozen_atoms_and_orientation_stay_put() {
    let start = system(&AR7, None);
    let c = vec![
        Constraint::at_start(ConstraintKind::Frozen(vec![id(0), id(1)])),
        Constraint::at_start(ConstraintKind::Coordinate(id(5), 2)),
        Constraint {
            kind: ConstraintKind::Dihedral([id(2), id(0), id(1), id(4)]),
            target: Some(PI - 0.3),
            anchor: None,
        },
    ];
    let mut opt =
        GeometryOptimizer::new(settings("rsirfo_bfgs", "cart", c), start.clone()).unwrap();
    let out = run_optimization(&mut opt, &LennardJones, &mut []).unwrap();
    assert_eq!(out.termination, Some(Termination::Converged));
    for a in [0, 1] {
        let (p, q) = (
            out.system.position(id(a)).value(),
            start.position(id(a)).value(),
        );
        assert!(
            (0..3).all(|k| (p[k] - q[k]).abs() < 1e-12),
            "atom {a} moved"
        );
    }
    let z = out.system.position(id(5)).value()[2];
    assert!((z - start.position(id(5)).value()[2]).abs() < 1e-12);
}

#[test]
fn constraints_from_the_command_line() {
    let dir = std::env::temp_dir().join(format!("mopt_pc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let xyz = dir.join("ar7.xyz");
    let mut text = String::from("7\n\n");
    for p in AR7 {
        text.push_str(&format!("Ar {} {} {}\n", p[0], p[1], p[2]));
    }
    std::fs::write(&xyz, text).unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let out = run(
        &config_from_cli(&[
            &xyz,
            "-os",
            "lj",
            "-opt",
            "crsirfo_bfgs",
            "-fix",
            "7",
            "-pc",
            "manual",
            "bond",
            "1,2",
            "3.4",
            "manual",
            "angle",
            "3,6,4",
            "75",
        ])
        .unwrap(),
        &dir,
        None,
    )
    .unwrap();
    assert!(out[0].converged);
    let text = std::fs::read_to_string(&out[0].optimized).unwrap();
    let rows: Vec<[f64; 3]> = text
        .lines()
        .skip(2)
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .skip(1)
                .map(|x| x.parse().unwrap())
                .collect();
            [v[0], v[1], v[2]]
        })
        .collect();
    let s = system(&rows, None);
    assert!((dist(&s, 0, 1) - 3.4 * A2B).abs() < 1e-5);
    assert!(
        (rows[6][2] - AR7[6][2]).abs() < 1e-6,
        "frozen atom 7 keeps its place"
    );
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-pc", "bond", "1"]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A Rust-defined constraint function: the difference of two distances.
#[derive(Debug, Clone)]
struct DistanceDifference;

impl mopt_rs::pes::bias::GeometricBias for DistanceDifference {
    fn name(&self) -> &str {
        "distance_difference"
    }
    fn atoms(&self) -> Vec<AtomId> {
        vec![id(0), id(1), id(2), id(5)]
    }
    fn energy<T: mopt_rs::math::ad::Scalar>(
        &self,
        g: &mopt_rs::pes::bias::Geom<'_, T>,
        _clock: mopt_rs::pes::bias::Clock,
    ) -> T {
        let d = |a: usize, b: usize| {
            let v = g.d_centroids(&[id(a)], &[id(b)]);
            T::dot3(v, v).sqrt()
        };
        d(0, 1) - d(2, 5)
    }
}

#[test]
fn user_defined_constraints_from_rust_terms_and_expressions() {
    use mopt_rs::cli::constraints::user_constraint;
    use mopt_rs::pes::constraint::ConstraintFunction;
    let start = system(&AR7, None);
    let rust = Constraint {
        kind: ConstraintKind::Function(ConstraintFunction::new(Arc::new(AdBias::new(
            DistanceDifference,
        )))),
        target: Some(0.0),
        anchor: None,
    };
    let expr = user_constraint("dist(1,2) = dist(3,6)").unwrap();
    for c in [rust, expr] {
        let set = ConstraintSet::new(std::slice::from_ref(&c), &start, None).unwrap();
        let mut opt =
            GeometryOptimizer::new(settings("rsirfo_bfgs", "cart", vec![c]), start.clone())
                .unwrap();
        let out = run_optimization(&mut opt, &LennardJones, &mut []).unwrap();
        assert_eq!(out.termination, Some(Termination::Converged));
        assert!((dist(&out.system, 0, 1) - dist(&out.system, 2, 5)).abs() < 1e-8);
        assert!(set.residuals(&out.system).unwrap()[0].abs() < 1e-8);
    }
    // Values with units, periods, and Cartesian functions.
    let angle = user_constraint("angle(1,6,3) = 80*deg").unwrap();
    assert!((angle.target.unwrap() - 80f64.to_radians()).abs() < 1e-14);
    let dihedral = user_constraint("dihedral(1,2,3,4) = pi; period = 2*pi").unwrap();
    let set = ConstraintSet::new(&[dihedral], &start, None).unwrap();
    let r = set.residuals(&start).unwrap()[0];
    assert!(r.abs() <= PI + 1e-12, "wrapped into half a period: {r}");
    let z = user_constraint("z(7)").unwrap();
    assert!(mopt_rs::pes::constraint::breaks_rigid_body(&[z]));
    for bad in [
        "1.5 * ang",
        "dist(1,2) = 1; period = -1",
        "dist(1,2); width = 2",
        "dist(1",
    ] {
        assert!(user_constraint(bad).is_err(), "{bad}");
    }
}

#[test]
fn user_constraints_on_the_command_line() {
    let dir = std::env::temp_dir().join(format!("mopt_uc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let xyz = dir.join("ar7.xyz");
    let mut text = String::from("7\n\n");
    for p in AR7 {
        text.push_str(&format!("Ar {} {} {}\n", p[0], p[1], p[2]));
    }
    std::fs::write(&xyz, text).unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let out = run(
        &config_from_cli(&[&xyz, "-os", "lj", "-uc", "dist(1,2) + dist(1,3) = 7.0*ang"]).unwrap(),
        &dir,
        None,
    )
    .unwrap();
    assert!(out[0].converged);
    let text = std::fs::read_to_string(&out[0].optimized).unwrap();
    let rows: Vec<[f64; 3]> = text
        .lines()
        .skip(2)
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .skip(1)
                .map(|x| x.parse().unwrap())
                .collect();
            [v[0], v[1], v[2]]
        })
        .collect();
    let s = system(&rows, None);
    assert!((dist(&s, 0, 1) + dist(&s, 0, 2) - 7.0 * A2B).abs() < 1e-5);
    let _ = std::fs::remove_dir_all(&dir);
}
