//! DFT-D3(BJ): the reference energy of simple-dftd3's test suite, and exact derivatives.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use mopt_rs::domain::{positions, Angstrom, Bohr, Element, MolecularSystem, PeriodicCell};
use mopt_rs::domain::{SpinCharge, Vector3};
use mopt_rs::pes::dispersion::D3Bj;
use mopt_rs::pes::term_check::check_term;

fn el(symbols: &[&str]) -> Vec<Element> {
    symbols
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect()
}

/// mstore MB16-43 #01 (Bohr).
fn mb16_43_01() -> MolecularSystem {
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
    MolecularSystem::new(
        el(&sym),
        &positions::<Bohr>(&xyz),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

#[test]
fn pbe_d3bj_energy_matches_simple_dftd3() {
    // simple-dftd3 `test_pbed3bj_mb01` (s9 = 0).
    let e = D3Bj::PBE
        .evaluate(&mb16_43_01(), false)
        .unwrap()
        .energy
        .value();
    let want = -1.7882220155186028e-2;
    assert!((e - want).abs() < 1e-12, "{e:.16e} vs {want:.16e}");
}

#[test]
fn gradient_and_hessian_are_exact() {
    let sys = mb16_43_01();
    let c = check_term(&D3Bj::PBE, &sys, 1e-5).unwrap();
    assert!(
        c.gradient_error < 1e-9 && c.hessian_error < 1e-7 && c.asymmetry < 1e-12,
        "{c:?}"
    );
}

#[test]
fn periodic_cells_sum_images() {
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([4.2, 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.6, 4.5, 0.0]),
        Vector3::<Angstrom>::new([0.3, -0.2, 4.8]),
        [true, true, true],
    )
    .unwrap();
    let sys = MolecularSystem::new(
        el(&["C", "O", "H", "N"]),
        &positions::<Angstrom>(&[
            [0.1, 0.2, 0.3],
            [1.3, 0.1, 0.2],
            [0.0, 1.2, 4.4],
            [3.9, 3.8, 0.4],
        ]),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap();
    let c = check_term(&D3Bj::PBE, &sys, 1e-5).unwrap();
    assert!(c.passes(1e-7), "{c:?}");
}
