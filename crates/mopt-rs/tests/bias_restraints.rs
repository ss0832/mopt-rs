//! Geometric restraints and external fields: exact derivatives and lattice invariance in
//! isolated, 1D, 2D, and 3D (skewed) cells, plus known values.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use mopt_rs::domain::{positions, Angstrom, AtomId, Element, MolecularSystem, PeriodicCell};
use mopt_rs::domain::{SpinCharge, Units, Vector3};
use mopt_rs::pes::bias::{
    AdBias, AngleContinuation, BondRange, DihedralForm, Flux, GeometricBias, KeepAngle,
    KeepDihedral, KeepDistance, KeepOutOfPlane, MorseDistance, PullingForce, Ramp, Universal,
    VoidPoint,
};
use mopt_rs::pes::term_check::{check_term, energy_change_on_shift};
use mopt_rs::pes::{PesTerm, TermContext};

const A2B: f64 = Units::ANGSTROM_TO_BOHR;

fn ids(v: &[usize]) -> Vec<AtomId> {
    v.iter().map(|&i| AtomId::new(i)).collect()
}

fn id(i: usize) -> AtomId {
    AtomId::new(i)
}

fn system(pos: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::C; pos.len()],
        &positions::<Angstrom>(pos),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

fn cell(axes: [bool; 3]) -> PeriodicCell {
    PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([5.0, 0.0, 0.0]),
        Vector3::<Angstrom>::new([1.2, 5.5, 0.0]),
        Vector3::<Angstrom>::new([0.4, -0.6, 6.0]),
        axes,
    )
    .unwrap()
}

fn energy(t: &dyn PesTerm, sys: &MolecularSystem) -> f64 {
    let ctx = TermContext::new(0, sys, None);
    t.evaluate(sys, &ctx, false).unwrap().energy.value()
}

/// Atoms near the cell faces, so minimum images matter along every periodic axis.
const POS: [[f64; 3]; 8] = [
    [0.2, 0.3, 0.4],
    [1.1, 0.5, 5.6],
    [4.6, 5.2, 0.6],
    [4.4, 0.1, 5.1],
    [0.5, 5.0, 5.4],
    [2.4, 2.7, 3.0],
    [1.6, 1.1, 1.3],
    [3.1, 3.9, 2.2],
];

/// `check_term` in isolated, 1D, 2D, and 3D cells, and no wrapping along non-periodic axes.
fn check_all_cells(name: &str, t: &dyn PesTerm, moved: usize, tol: f64) {
    for axes in [
        None,
        Some([true, false, false]),
        Some([true, true, false]),
        Some([true, true, true]),
    ] {
        let sys = system(&POS, axes.map(cell));
        let c = check_term(t, &sys, 1e-5).unwrap();
        assert!(c.passes(tol), "{name} {axes:?}: {c:?}");
        let v = cell([true; 3]).lattice_vector(2).value();
        let de = energy_change_on_shift(t, &sys, id(moved), v).unwrap();
        let periodic_c = axes.is_some_and(|a| a[2]);
        assert_eq!(de.abs() < 1e-12, periodic_c, "{name} {axes:?}: {de}");
    }
}

fn ad<B: GeometricBias>(b: B) -> AdBias<B> {
    AdBias::new(b)
}

fn r(v: f64) -> Ramp {
    Ramp::constant(v)
}

#[test]
fn every_restraint_passes_the_term_checks_in_all_cells() {
    let deg = Units::DEG_TO_RAD;
    check_all_cells(
        "kp",
        &ad(KeepDistance {
            a: ids(&[0]),
            b: ids(&[1]),
            k: r(0.3),
            r0: r(1.5 * A2B),
        }),
        0,
        1e-6,
    );
    check_all_cells(
        "kpv2",
        &ad(KeepDistance {
            a: ids(&[0, 2]),
            b: ids(&[1, 3, 4]),
            k: r(0.3),
            r0: r(2.0 * A2B),
        }),
        0,
        1e-6,
    );
    check_all_cells(
        "akp",
        &ad(MorseDistance {
            pair: [id(0), id(3)],
            depth: 0.05,
            k: 0.3,
            r0: 1.4 * A2B,
        }),
        0,
        1e-6,
    );
    for continuation in [AngleContinuation::Series, AngleContinuation::Quadratic] {
        check_all_cells(
            "ka",
            &ad(KeepAngle {
                groups: [ids(&[0]), ids(&[5]), ids(&[2, 3])],
                k: r(0.2),
                theta0: r(100.0 * deg),
                continuation,
            }),
            0,
            1e-6,
        );
    }
    for form in [DihedralForm::Harmonic, DihedralForm::Cosine { n: 3.0 }] {
        check_all_cells(
            "kda",
            &ad(KeepDihedral {
                groups: [ids(&[0]), ids(&[6]), ids(&[7, 5]), ids(&[2])],
                k: r(0.1),
                phi0: r(40.0 * deg),
                form,
            }),
            0,
            1e-6,
        );
    }
    check_all_cells(
        "kopa",
        &ad(KeepOutOfPlane {
            groups: [ids(&[6]), ids(&[0]), ids(&[5]), ids(&[1])],
            k: r(0.1),
            chi0: r(20.0 * deg),
        }),
        0,
        1e-6,
    );
    check_all_cells(
        "vpp",
        &ad(VoidPoint {
            atom: id(0),
            point: [0.9 * A2B, 0.8 * A2B, 0.7 * A2B],
            k: 0.05,
            r0: 0.3 * A2B,
            order: 3.0,
        }),
        0,
        1e-6,
    );
    check_all_cells(
        "up",
        &ad(Universal {
            atoms: ids(&[0, 1, 2, 5]),
            c: 50.0 * Units::KJ_MOL_TO_HARTREE,
        }),
        0,
        1e-6,
    );
    check_all_cells(
        "fp",
        &ad(Flux {
            atoms: ids(&[0, 5]),
            k: [0.01, 0.02, 0.005],
            order: [2.0, 4.0, 1.0],
            point: [1.0, 1.5, 2.0],
        }),
        0,
        1e-6,
    );
    check_all_cells(
        "brp",
        &ad(BondRange {
            a: ids(&[0]),
            b: ids(&[5, 6]),
            upper_const: 3.0,
            lower_const: 3.0,
            upper: 2.5 * A2B,
            lower: 1.5 * A2B,
        }),
        0,
        1e-6,
    );
    check_all_cells(
        "lmefp",
        &ad(PullingForce {
            pairs: vec![[id(0), id(1)], [id(2), id(3)]],
            force: 1000.0 * Units::PN_TO_AU_FORCE,
        }),
        0,
        1e-6,
    );
}

#[test]
fn known_values() {
    let k = 0.4;
    // kp
    let sys = system(&[[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], None);
    let kp = ad(KeepDistance {
        a: ids(&[0]),
        b: ids(&[1]),
        k: r(k),
        r0: r(1.0 * A2B),
    });
    assert!((energy(&kp, &sys) - 0.5 * k * (0.3 * A2B).powi(2)).abs() < 1e-14);
    // akp
    let (de, r0) = (0.07, 1.0 * A2B);
    let akp = ad(MorseDistance {
        pair: [id(0), id(1)],
        depth: de,
        k,
        r0,
    });
    let x = 1.0 - (-(k / (2.0 * de)).sqrt() * 0.3 * A2B).exp();
    assert!((energy(&akp, &sys) - de * x * x).abs() < 1e-14);
    // ka: right angle restrained to 60 degrees.
    let sys = system(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], None);
    let ka = ad(KeepAngle {
        groups: [ids(&[0]), ids(&[1]), ids(&[2])],
        k: r(k),
        theta0: r(PI / 3.0),
        continuation: AngleContinuation::Series,
    });
    assert!((energy(&ka, &sys) - 0.5 * k * (PI / 6.0).powi(2)).abs() < 1e-13);
    // kda: the IUPAC dihedral of this chain is +90 degrees.
    let sys = system(
        &[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        None,
    );
    let kda = |phi0: f64| {
        ad(KeepDihedral {
            groups: [ids(&[0]), ids(&[1]), ids(&[2]), ids(&[3])],
            k: r(k),
            phi0: r(phi0),
            form: DihedralForm::Harmonic,
        })
    };
    assert!(energy(&kda(PI / 2.0), &sys).abs() < 1e-14);
    assert!((energy(&kda(-PI / 2.0), &sys) - 0.5 * k * PI * PI).abs() < 1e-12);
    // Wrapping: 170 vs -170 degrees differ by 20 degrees.
    let sys_w = system(
        &[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [
                (170f64).to_radians().cos(),
                (170f64).to_radians().sin(),
                1.0,
            ],
        ],
        None,
    );
    let e = energy(&kda(-170f64.to_radians()), &sys_w);
    assert!(
        (e - 0.5 * k * 20f64.to_radians().powi(2)).abs() < 1e-12,
        "{e}"
    );
    // kdac: 1 - cos(n phi - phi0) at phi = 90 degrees, n = 2, phi0 = 0 gives k.
    let kdac = ad(KeepDihedral {
        groups: [ids(&[0]), ids(&[1]), ids(&[2]), ids(&[3])],
        k: r(k),
        phi0: r(0.0),
        form: DihedralForm::Cosine { n: 2.0 },
    });
    assert!((energy(&kdac, &sys) - k).abs() < 1e-13);
    // kopa: 45 degrees above the plane.
    let sys = system(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        None,
    );
    let kopa = ad(KeepOutOfPlane {
        groups: [ids(&[0]), ids(&[1]), ids(&[2]), ids(&[3])],
        k: r(k),
        chi0: r(0.0),
    });
    assert!((energy(&kopa, &sys) - 0.5 * k * (PI / 4.0).powi(2)).abs() < 1e-13);
    // up: three atoms, c / 3 pairs * sum of distances to the centroid.
    let sys = system(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], None);
    let up = ad(Universal {
        atoms: ids(&[0, 1, 2]),
        c: 0.3,
    });
    let d = |p: [f64; 2]| ((p[0] - 1.0).powi(2) + (p[1] - 1.0).powi(2)).sqrt() * A2B;
    let want = 0.3 / 3.0 * (d([0.0, 0.0]) + d([3.0, 0.0]) + d([0.0, 3.0]));
    assert!((energy(&up, &sys) - want).abs() < 1e-13);
    // lmefpv2: -F r.
    let pull = ad(PullingForce {
        pairs: vec![[id(0), id(1)]],
        force: 0.01,
    });
    assert!((energy(&pull, &sys) + 0.01 * 3.0 * A2B).abs() < 1e-14);
}

#[test]
fn angle_restraints_are_smooth_at_linear_geometries() {
    // Restrained to 0 or pi, the energy is smooth through the linear geometry, where arccos is
    // singular: exact derivatives there come from the series.
    let bent =
        |x3: f64, eps: f64| system(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [x3, eps, 0.0]], None);
    for (sys, theta0) in [
        (bent(-1.0, 1e-5), PI),
        (bent(-1.0, 0.0), PI),
        (bent(1.3, 1e-5), 0.0),
        (bent(1.3, 0.0), 0.0),
    ] {
        for continuation in [AngleContinuation::Series, AngleContinuation::Quadratic] {
            let t = ad(KeepAngle {
                groups: [ids(&[0]), ids(&[1]), ids(&[2])],
                k: r(0.2),
                theta0: r(theta0),
                continuation,
            });
            let c = check_term(&t, &sys, 1e-5).unwrap();
            assert!(c.passes(1e-6), "{theta0} {continuation:?}: {c:?}");
        }
    }
    // Any other target: finite energy and gradient at the exactly linear geometry.
    for continuation in [AngleContinuation::Series, AngleContinuation::Quadratic] {
        let t = ad(KeepAngle {
            groups: [ids(&[0]), ids(&[1]), ids(&[2])],
            k: r(0.2),
            theta0: r(2.0),
            continuation,
        });
        let sys = bent(-1.0, 0.0);
        let ctx = TermContext::new(0, &sys, None);
        let out = t.evaluate(&sys, &ctx, true).unwrap();
        // The series is exact; the quadratic continuation in cos(theta) is an approximation
        // beyond the cut (theta(u) behaves like a square root at u = -1).
        let tol = match continuation {
            AngleContinuation::Series => 1e-9,
            AngleContinuation::Quadratic => 1e-3,
        };
        assert!((out.energy.value() - 0.1 * (PI - 2.0).powi(2)).abs() < tol);
        assert!(out.gradient_bohr.iter().all(|g| g.is_finite()));
    }
}

#[test]
fn ramps_follow_the_iteration() {
    let sys = system(&[[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], None);
    let kp = ad(KeepDistance {
        a: ids(&[0]),
        b: ids(&[1]),
        k: Ramp::parse("0.0,0.6").unwrap(),
        r0: r(1.0 * A2B),
    });
    let at = |it| {
        let ctx = TermContext::new(it, &sys, None);
        kp.evaluate(&sys, &ctx, false).unwrap().energy.value()
    };
    let full = 0.5 * 0.6 * (0.3 * A2B).powi(2);
    assert!(at(0).abs() < 1e-15);
    assert!((at(150) - 0.5 * full).abs() < 1e-14);
    assert!((at(900) - full).abs() < 1e-14);
}

#[test]
fn bond_range_stays_finite_far_outside_its_range() {
    let t = ad(BondRange {
        a: ids(&[0]),
        b: ids(&[1]),
        upper_const: 50.0,
        lower_const: 50.0,
        upper: 2.0 * A2B,
        lower: 1.0 * A2B,
    });
    for r in [0.01, 1.5, 40.0, 400.0] {
        let sys = system(&[[0.0, 0.0, 0.0], [r, 0.0, 0.0]], None);
        let ctx = TermContext::at_iteration(0, &sys);
        let out = t.evaluate(&sys, &ctx, true).unwrap();
        assert!(
            out.energy.value().is_finite() && out.gradient_bohr.iter().all(|g| g.is_finite()),
            "{r}"
        );
    }
    // Far outside: the energy grows linearly, `50 (r - 2 A)` in Bohr.
    let sys = system(&[[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]], None);
    let e = energy(&t, &sys);
    assert!((e - 50.0 * (40.0 - 2.0) * A2B).abs() < 1e-6 * e, "{e}");
}
