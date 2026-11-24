//! DFT-D4: the reference energy of dftd4's test suite, EEQ charges, and the analytic gradient.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use mopt_rs::domain::{positions, Bohr, Element, MolecularSystem, SpinCharge};
use mopt_rs::pes::dispersion::D4;

/// mstore MB16-43 #01 (Bohr).
fn mb16_43_01(charge: i32) -> MolecularSystem {
    let sym = [
        "Na", "H", "O", "H", "F", "H", "H", "O", "N", "H", "H", "Cl", "B", "B", "N", "Al",
    ];
    let xyz = [
        [-1.85528263484662, 3.58670515364616, -2.41763729306344],
        [4.40178023537845, 0.02338844412653, -4.95457749372945],
        [-2.98706033463438, 4.76252065456814, 1.27043301573532],
        [0.79980886075526, 1.41103455609189, -5.04655321620119],
        [-4.20647469409936, 1.84275767548460, 4.55038084858449],
        [-3.54356121843970, -3.18835665176557, 1.46240021785588],
        [2.70032160109941, 1.06818452504054, -1.73234650374438],
        [3.73114088824361, -2.07001543363453, 2.23160937604731],
        [-1.75306819230397, 0.35951417150421, 1.05323406177129],
        [5.41755788583825, -1.57881830078929, 1.75394002750038],
        [-2.23462868255966, -2.13856505054269, 4.10922285746451],
        [1.01565866207568, -3.21952154552768, -3.36050963020778],
        [2.42119255723593, 0.26626435093114, -3.91862474360560],
        [-3.02526098819107, 2.53667889095925, 2.31664984740423],
        [-2.00438948664892, -2.29235136977220, 2.19782807357059],
        [1.12226554109716, -1.36942007032045, 0.48455055461782],
    ];
    let el: Vec<Element> = sym
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    MolecularSystem::new(
        el,
        &positions::<Bohr>(&xyz),
        SpinCharge::new(charge, 1),
        None,
    )
    .unwrap()
}

#[test]
fn pbe_d4_energy_matches_dftd4() {
    // dftd4 `test_pbed4_mb01` (s9 = 0, EEQ charges).
    let d4 = D4 { s9: 0.0, ..D4::PBE };
    let (e, _) = d4.energy_gradient(&mb16_43_01(0)).unwrap();
    let want = -1.8578752883363366e-2;
    assert!((e - want).abs() < 1e-11, "{e:.16e} vs {want:.16e}");
}

#[test]
fn eeq_charges_add_up_to_the_total_charge() {
    for charge in [0, 1, -2] {
        let q = D4::PBE.charges(&mb16_43_01(charge)).unwrap();
        let sum: f64 = q.iter().sum();
        assert!((sum - f64::from(charge)).abs() < 1e-10, "{charge}: {sum}");
    }
}

#[test]
fn gradient_matches_finite_differences() {
    for d4 in [D4 { s9: 0.0, ..D4::PBE }, D4::PBE] {
        let sys = mb16_43_01(0);
        let (_, g) = d4.energy_gradient(&sys).unwrap();
        let x = sys.coords_flat_bohr();
        let h = 1e-5;
        let mut worst = 0.0_f64;
        for k in 0..x.nrows() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += h;
            xm[k] -= h;
            let ep = d4
                .energy_gradient(&sys.with_flat_coords_bohr(&xp).unwrap())
                .unwrap()
                .0;
            let em = d4
                .energy_gradient(&sys.with_flat_coords_bohr(&xm).unwrap())
                .unwrap()
                .0;
            worst = worst.max(((ep - em) / (2.0 * h) - g[k]).abs());
        }
        assert!(worst < 1e-9, "s9 = {}: {worst}", d4.s9);
    }
}

#[test]
fn hessian_is_symmetric_and_periodic_cells_are_refused() {
    let out = D4::PBE.evaluate(&mb16_43_01(0), true).unwrap();
    let h = out.hessian_bohr.unwrap();
    let n = h.nrows();
    for i in 0..n {
        for j in 0..n {
            assert!((h[(i, j)] - h[(j, i)]).abs() < 1e-12);
        }
    }
    // Translation invariance.
    for axis in 0..3 {
        for i in 0..n {
            let s: f64 = (0..n).filter(|j| j % 3 == axis).map(|j| h[(i, j)]).sum();
            assert!(s.abs() < 1e-7, "{axis}: {s}");
        }
    }
    let cell = mopt_rs::domain::PeriodicCell::from_lattice_vectors(
        mopt_rs::domain::Vector3::<Bohr>::new([20.0, 0.0, 0.0]),
        mopt_rs::domain::Vector3::<Bohr>::new([0.0, 20.0, 0.0]),
        mopt_rs::domain::Vector3::<Bohr>::new([0.0, 0.0, 20.0]),
        [true, true, true],
    )
    .unwrap();
    let periodic = MolecularSystem::new(
        vec![Element::C, Element::O],
        &positions::<Bohr>(&[[0.0; 3], [2.1, 0.0, 0.0]]),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap();
    assert!(D4::PBE.evaluate(&periodic, false).is_err());
}

/// Geometries where the literal formulas overflow or divide zero by zero: nearly coincident
/// atoms, far-apart fragments, and coordination numbers far beyond every reference system.
fn extreme_systems() -> Vec<MolecularSystem> {
    let mk = |el: Element, pos: Vec<[f64; 3]>| {
        MolecularSystem::new(
            vec![el; pos.len()],
            &positions::<Bohr>(&pos),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    };
    let mut dense = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                dense.push([
                    0.9 * i as f64,
                    0.9 * j as f64,
                    0.9 * k as f64 + 0.01 * (i + j) as f64,
                ]);
            }
        }
    }
    vec![
        mk(
            Element::C,
            vec![[0.0; 3], [0.2, 0.0, 0.0], [0.1, 0.25, 0.0]],
        ),
        mk(
            Element::O,
            vec![[0.0; 3], [100.0, 0.0, 0.0], [0.0, 1.8, 0.4]],
        ),
        mk(Element::C, dense),
        mk(Element::H, vec![[0.0; 3], [1.4, 0.0, 0.0]]),
    ]
}

#[test]
fn extreme_geometries_stay_finite() {
    use mopt_rs::model_hessian::resolve;
    use mopt_rs::pes::dispersion::D3Bj;
    for sys in extreme_systems() {
        let d4 = D4::PBE.evaluate(&sys, true).unwrap();
        let d3 = D3Bj::PBE.evaluate(&sys, true).unwrap();
        for out in [&d4, &d3] {
            assert!(out.energy.value().is_finite());
            assert!(out.gradient_bohr.iter().all(|v| v.is_finite()));
            let h = out.hessian_bohr.as_ref().unwrap();
            assert!((0..h.nrows()).all(|i| (0..h.ncols()).all(|j| h[(i, j)].is_finite())));
        }
        let g = faer::Col::zeros(sys.dof());
        for name in [
            "lindh2007",
            "lindh2007d3",
            "lindh2007d4",
            "xtb_lindh_d2",
            "swartd4",
            "fischerd4",
        ] {
            let h = resolve(name).unwrap().hessian(&sys, &g).unwrap();
            assert!(
                (0..h.nrows()).all(|i| (0..h.ncols()).all(|j| h[(i, j)].is_finite())),
                "{name}"
            );
        }
    }
}
