//! Repulsive, well, metadynamics, nanoreactor, ellipsoidal, and spacer-model biases: exact
//! derivatives and lattice invariance in isolated, 1D, 2D, and 3D (skewed) cells, plus their
//! defining properties.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use mopt_rs::domain::{positions, Angstrom, AtomId, Element, MolecularSystem, PeriodicCell};
use mopt_rs::domain::{SpinCharge, Units, Vector3};
use mopt_rs::pes::bias::{
    AdBias, AsymmetricEllipsoids, Clock, CollectiveVariable, ConeRepulsive, Ellipsoid,
    EllipsoidVariant, GeometricBias, LjCenter, LjGaussian, LjParameters, LjRepulsive, Metadynamics,
    MetadynamicsCv, MicroBias, NanoReactor, SpacerModel, Well, WellGeometry, WellPotential,
};
use mopt_rs::pes::term_check::{check_term, energy_change_on_shift};
use mopt_rs::pes::{PesTerm, TermContext};

const A2B: f64 = Units::ANGSTROM_TO_BOHR;
const KJ: f64 = Units::KJ_MOL_TO_HARTREE;

fn id(i: usize) -> AtomId {
    AtomId::new(i)
}

fn ids(v: &[usize]) -> Vec<AtomId> {
    v.iter().map(|&i| id(i)).collect()
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

fn cell(axes: [bool; 3], scale: f64) -> PeriodicCell {
    PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([5.0 * scale, 0.0, 0.0]),
        Vector3::<Angstrom>::new([1.2 * scale, 5.5 * scale, 0.0]),
        Vector3::<Angstrom>::new([0.4 * scale, -0.6 * scale, 6.0 * scale]),
        axes,
    )
    .unwrap()
}

fn ctx(sys: &MolecularSystem) -> TermContext<'_> {
    TermContext::at_iteration(0, sys)
}

fn energy(t: &dyn PesTerm, sys: &MolecularSystem) -> f64 {
    t.evaluate(sys, &ctx(sys), false).unwrap().energy.value()
}

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

const AXES: [Option<[bool; 3]>; 4] = [
    None,
    Some([true, false, false]),
    Some([true, true, false]),
    Some([true, true, true]),
];

fn check_all_cells(name: &str, t: &dyn PesTerm, pos: &[[f64; 3]], scale: f64, tol: f64) {
    check_all_cells_with_step(name, t, pos, scale, tol, 1e-5);
}

/// Terms with microiterations carry the inner convergence noise (~1e-11 in the gradient) into
/// finite differences, so they are checked with a larger step.
fn check_all_cells_with_step(
    name: &str,
    t: &dyn PesTerm,
    pos: &[[f64; 3]],
    scale: f64,
    tol: f64,
    step: f64,
) {
    for axes in AXES {
        let sys = system(pos, axes.map(|a| cell(a, scale)));
        let c = check_term(t, &sys, step).unwrap();
        assert!(c.passes(tol), "{name} {axes:?}: {c:?}");
        let v = cell([true; 3], scale).lattice_vector(2).value();
        let de = energy_change_on_shift(t, &sys, id(0), v).unwrap();
        let periodic_c = axes.is_some_and(|a| a[2]);
        assert_eq!(de.abs() < 1e-12, periodic_c, "{name} {axes:?}: {de}");
    }
}

fn ad<B: GeometricBias>(b: B) -> AdBias<B> {
    AdBias::new(b)
}

fn carbons(n: usize) -> Vec<Element> {
    vec![Element::C; n]
}

#[test]
fn repulsive_potentials_pass_the_term_checks() {
    let el = carbons(8);
    check_all_cells(
        "rp scale",
        &ad(LjRepulsive::new(
            &el,
            &ids(&[0, 1]),
            &ids(&[2, 3, 4]),
            LjParameters::Scale {
                well: 1.0,
                distance: 0.4,
            },
        )),
        &POS,
        1.0,
        1e-6,
    );
    check_all_cells(
        "rp value",
        &ad(LjRepulsive::new(
            &el,
            &ids(&[0]),
            &ids(&[5, 6]),
            LjParameters::Value {
                well: 1.0 * KJ,
                distance: 1.6 * A2B,
            },
        )),
        &POS,
        1.0,
        1e-6,
    );
    check_all_cells(
        "rpv2",
        &ad(LjCenter::new(
            &el,
            [id(6), id(0)],
            0.8 * A2B,
            &ids(&[5, 7]),
            LjParameters::Scale {
                well: 1.0,
                distance: 0.3,
            },
            [1.0, 1.0],
            [12.0, 6.0],
        )),
        &POS,
        1.0,
        1e-6,
    );
    check_all_cells(
        "rpg",
        &ad(LjGaussian {
            pairs: vec![(id(0), id(6)), (id(0), id(5))],
            lj_well: 1.0 * KJ,
            lj_distance: 1.5 * A2B,
            gauss_well: 2.0 * KJ,
            gauss_distance: 2.0 * A2B,
            gauss_range: 3.0 * A2B,
        }),
        &POS,
        1.0,
        1e-6,
    );
    check_all_cells(
        "cp",
        &ad(ConeRepulsive::new(
            &el,
            id(6),
            [id(5), id(7), id(1)],
            &ids(&[0, 2]),
            1.0 * KJ,
            0.5 * A2B,
            60f64.to_radians(),
        )),
        &POS,
        1.0,
        1e-6,
    );
}

#[test]
fn well_potentials_pass_the_term_checks() {
    let well = Well {
        wall: 10.0 * KJ,
        limits: [0.6 * A2B, 1.0 * A2B, 1.5 * A2B, 2.5 * A2B],
    };
    let geometries = [
        WellGeometry::Fragments {
            a: ids(&[0]),
            b: ids(&[5, 6]),
        },
        WellGeometry::Wall {
            axis: 2,
            targets: ids(&[0, 6]),
        },
        WellGeometry::Point {
            point: [1.0 * A2B, 1.0 * A2B, 1.0 * A2B],
            targets: ids(&[0, 5, 7]),
        },
        WellGeometry::Around {
            center: ids(&[5, 6]),
            targets: ids(&[0, 7]),
        },
    ];
    for geometry in geometries {
        let t = ad(WellPotential { well, geometry });
        check_all_cells(t.bias.name(), &t, &POS, 1.0, 1e-6);
    }
}

#[test]
fn the_well_is_twice_continuously_differentiable() {
    let w = Well {
        wall: 1.0,
        limits: [1.0, 2.0, 3.0, 4.5],
    };
    assert!(w.at(1.5 + 1.0) == 0.0);
    assert!((w.at(1.0) - 1.0).abs() < 1e-14 && (w.at(4.5) - 1.0).abs() < 1e-14);
    let f = |r: f64| w.at(r);
    for r0 in [1.0, 2.0, 3.0, 4.5] {
        let h = 1e-4;
        let d1 = |r: f64| (f(r + h) - f(r - h)) / (2.0 * h);
        let d2 = |r: f64| (f(r + h) - 2.0 * f(r) + f(r - h)) / (h * h);
        let (l, r) = (r0 - 1e-3, r0 + 1e-3);
        assert!((f(l) - f(r)).abs() < 1e-2, "value at {r0}");
        assert!((d1(l) - d1(r)).abs() < 1e-2, "slope at {r0}");
        assert!((d2(l) - d2(r)).abs() < 5e-2, "curvature at {r0}");
    }
}

#[test]
fn metadynamics_deposits_gaussians_on_commit() {
    let cvs = vec![
        MetadynamicsCv {
            cv: CollectiveVariable::Bond([id(0), id(6)]),
            height: 2.0 * KJ,
            width: 0.3 * A2B,
        },
        MetadynamicsCv {
            cv: CollectiveVariable::Angle([id(0), id(6), id(5)]),
            height: 1.0 * KJ,
            width: 10f64.to_radians(),
        },
        MetadynamicsCv {
            cv: CollectiveVariable::Dihedral([id(0), id(6), id(5), id(7)]),
            height: 1.0 * KJ,
            width: 10f64.to_radians(),
        },
        MetadynamicsCv {
            cv: CollectiveVariable::OutOfPlane([id(6), id(0), id(5), id(7)]),
            height: 1.0 * KJ,
            width: 10f64.to_radians(),
        },
    ];
    let t = ad(Metadynamics::new(cvs));
    let sys = system(&POS, None);
    assert!(energy(&t, &sys).abs() < 1e-15);
    t.commit(&sys, &ctx(&sys));
    // At the deposition point every Gaussian has its full height.
    assert!((energy(&t, &sys) - 5.0 * KJ).abs() < 1e-14);
    // A second deposit elsewhere, then the checks.
    let mut moved = POS;
    moved[0] = [0.0, 0.6, 0.7];
    t.commit(&system(&moved, None), &ctx(&sys));
    assert_eq!(t.bias.centers()[0].len(), 2);
    check_all_cells("metad", &t, &POS, 1.0, 1e-6);
}

#[test]
fn metadynamics_dihedral_gaussians_are_periodic() {
    let chain = |phi: f64| {
        system(
            &[
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [phi.cos(), phi.sin(), 1.0],
            ],
            None,
        )
    };
    let t = ad(Metadynamics::new(vec![MetadynamicsCv {
        cv: CollectiveVariable::Dihedral([id(0), id(1), id(2), id(3)]),
        height: 1.0,
        width: 0.2,
    }]));
    let s = chain(179f64.to_radians());
    t.commit(&s, &ctx(&s));
    // -179 degrees is 2 degrees away across the branch cut.
    let e = energy(&t, &chain(-179f64.to_radians()));
    let d = 2f64.to_radians();
    assert!((e - (-d * d / (2.0 * 0.04)).exp()).abs() < 1e-12, "{e}");
}

fn nanoreactor(atoms: usize) -> NanoReactor {
    let k = Units::KCAL_MOL_TO_HARTREE / (A2B * A2B);
    let ps = 1e-12 / Units::AU_TIME_TO_SEC;
    NanoReactor {
        atoms: (0..atoms).map(|i| (id(i), 12.0)).collect(),
        center: [0.0; 3],
        r1: 14.0 * A2B,
        k1: 1.0 * k,
        r2: 8.0 * A2B,
        k2: 0.5 * k,
        tau: 1.5 * ps,
        period: 2.0 * ps,
    }
}

#[test]
fn nanoreactor_confines_from_outside_like_the_paper() {
    let nr = nanoreactor(1);
    let ps = 1e-12 / Units::AU_TIME_TO_SEC;
    assert_eq!(nr.phase(0.0), 1.0);
    assert_eq!(nr.phase(1.4 * ps), 1.0);
    assert_eq!(nr.phase(1.6 * ps), 0.0);
    assert_eq!(nr.phase(2.1 * ps), 1.0);
    let t = ad(nr);
    let e_at = |r: f64, time_ps: f64| {
        let sys = system(&[[r, 0.0, 0.0]], None);
        let mut c = ctx(&sys);
        c.time_au = Some(time_ps * ps);
        t.evaluate(&sys, &c, false).unwrap().energy.value()
    };
    let k = Units::KCAL_MOL_TO_HARTREE / (A2B * A2B);
    // Inside both spheres: free in both phases (MultiOptPy pushed such atoms outward).
    assert_eq!(e_at(5.0, 0.0), 0.0);
    assert_eq!(e_at(5.0, 1.8), 0.0);
    // Between the spheres: free in the wide phase, pushed inward in the narrow phase.
    assert_eq!(e_at(10.0, 0.0), 0.0);
    let want = 0.5 * 12.0 * 0.5 * k * (2.0 * A2B).powi(2);
    assert!((e_at(10.0, 1.8) - want).abs() < 1e-14);
    // Outside: the wide wall.
    let want = 0.5 * 12.0 * 1.0 * k * (1.0 * A2B).powi(2);
    assert!((e_at(15.0, 0.0) - want).abs() < 1e-14);
    // Exact derivatives with atoms beyond the walls (cells large enough not to wrap them).
    let far: Vec<[f64; 3]> = POS.iter().map(|p| p.map(|x| 3.0 * x)).collect();
    let mut small = nanoreactor(8);
    small.r1 = 4.0 * A2B;
    small.r2 = 2.0 * A2B;
    check_all_cells("nrp", &ad(small), &far, 5.0, 1e-6);
}

fn ellipsoids(variant: EllipsoidVariant, two: bool) -> MicroBias<AsymmetricEllipsoids> {
    let e1 = Ellipsoid {
        root: id(0),
        axis_atom: id(1),
        eps: 2.0 * KJ,
        sigma: [
            1.2 * A2B,
            0.6 * A2B,
            0.7 * A2B,
            0.9 * A2B,
            0.8 * A2B,
            0.5 * A2B,
        ],
        distance: 1.0 * A2B,
        off_targets: vec![],
    };
    let mut list = vec![e1];
    if two {
        list.push(Ellipsoid {
            root: id(2),
            axis_atom: id(3),
            eps: 1.5 * KJ,
            sigma: [
                0.9 * A2B,
                0.5 * A2B,
                1.1 * A2B,
                0.6 * A2B,
                0.7 * A2B,
                0.6 * A2B,
            ],
            distance: 0.9 * A2B,
            off_targets: ids(&[0]),
        });
    }
    MicroBias::new(AsymmetricEllipsoids::new(&carbons(6), list, variant))
}

const ELL_POS: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.2, 1.1],
    [4.2, 0.3, 0.4],
    [5.1, 0.9, 0.2],
    [1.9, 1.7, 2.4],
    [-1.8, 2.1, 2.9],
];

#[test]
fn ellipsoidal_potentials_minimize_the_rotation_and_pass_the_term_checks() {
    for variant in [EllipsoidVariant::Gnb, EllipsoidVariant::Uff] {
        for two in [false, true] {
            let t = ellipsoids(variant, two);
            let sys = system(&ELL_POS, None);
            let e = energy(&t, &sys);
            // The relaxed angles are a minimum over a fine grid of rotations.
            let phi = t.auxiliary().unwrap();
            let atoms: Vec<AtomId> = (0..6).map(id).collect();
            let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
            let g = mopt_rs::pes::bias::Geom::new(&atoms, &x, &x, None);
            let mut grid_min = f64::INFINITY;
            for k in 0..360 {
                let mut trial = phi.clone();
                trial[0] = 2.0 * PI * k as f64 / 360.0;
                grid_min = grid_min.min(mopt_rs::pes::bias::AuxiliaryBias::energy(
                    &t.bias,
                    &g,
                    &trial,
                    Clock::at_iteration(0),
                ));
            }
            assert!(
                e <= grid_min + 1e-10,
                "{variant:?} {two}: {e} vs {grid_min}"
            );
            check_all_cells_with_step(PesTerm::name(&t), &t, &ELL_POS, 2.0, 1e-6, 1e-3);
        }
    }
}

#[test]
fn spacer_particles_relax_and_pass_the_term_checks() {
    let pos = [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [2.1, 1.3, 0.2]];
    let t = MicroBias::new(SpacerModel::new(
        &carbons(3),
        &ids(&[0, 1, 2]),
        1.0 * KJ,
        1.0 * A2B,
        2.0,
        3,
    ));
    let sys = system(&pos, None);
    let e = energy(&t, &sys);
    assert!(e.is_finite());
    assert_eq!(t.auxiliary().unwrap().len(), 9);
    check_all_cells_with_step("smp", &t, &pos, 4.0, 1e-6, 1e-3);
}
