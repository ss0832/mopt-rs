//! The GFN1-xTB Hessian as a model Hessian (`-modelhess gfn1xtb`): it is the analytic Hessian
//! of the built-in GFN1-xTB with the charge and multiplicity of the structure, it refuses what
//! GFN1-xTB cannot do (periodic cells, elements without parameters), and minimizations and a
//! saddle search run with it on the surface of another calculator.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::{Col, Mat};
use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::domain::{positions, Angstrom, Element, MolecularSystem, PeriodicCell};
use mopt_rs::domain::{SpinCharge, Vector3};
use mopt_rs::math::linalg::symmetric_eigen;
use mopt_rs::model_hessian::resolve;
use mopt_rs::pes::native::Gfn1Xtb;

fn molecule(
    elements: Vec<Element>,
    angstrom: &[[f64; 3]],
    state: SpinCharge,
    cell: Option<PeriodicCell>,
) -> MolecularSystem {
    MolecularSystem::new(elements, &positions::<Angstrom>(angstrom), state, cell).unwrap()
}

/// Formaldehyde, a little away from its minimum.
fn formaldehyde(state: SpinCharge) -> MolecularSystem {
    molecule(
        vec![Element::C, Element::O, Element::H, Element::H],
        &[
            [0.0, 0.0, 0.0],
            [1.22, 0.03, 0.0],
            [-0.55, 0.95, 0.05],
            [-0.58, -0.92, -0.04],
        ],
        state,
        None,
    )
}

fn max_difference(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
    (0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |j| (i, j)))
        .fold(0.0_f64, |m, (i, j)| m.max((a[(i, j)] - b[(i, j)]).abs()))
}

fn model(sys: &MolecularSystem) -> Mat<f64> {
    resolve("gfn1xtb")
        .unwrap()
        .hessian(sys, &Col::zeros(sys.dof()))
        .unwrap()
}

#[test]
fn the_model_is_the_analytic_gfn1_hessian_of_the_structure() {
    let neutral = formaldehyde(SpinCharge::default());
    let cation = formaldehyde(SpinCharge::checked(1, 2).unwrap());
    let reference = Gfn1Xtb::default();
    assert!(reference.analytic_hessian);
    let mut hessians = Vec::new();
    for sys in [&neutral, &cation] {
        let h = model(sys);
        let want = reference.hessian(sys).unwrap();
        let diff = max_difference(&h, &want);
        assert!(diff < 1e-10, "model against GFN1-xTB: {diff}");
        // The analytic Hessian, not differences: it agrees with central differences of the
        // gradient to their accuracy.
        let mut by_differences = Gfn1Xtb::default();
        by_differences.analytic_hessian = false;
        let differences = by_differences.hessian(sys).unwrap();
        let diff = max_difference(&h, &differences);
        assert!(diff < 1e-4, "model against differences: {diff}");
        // Symmetric, invariant under translations, curved.
        assert!(max_difference(&h, &h.transpose().to_owned()) < 1e-10);
        let n = h.nrows();
        for axis in 0..3 {
            let t = Col::from_fn(n, |i| if i % 3 == axis { 1.0 } else { 0.0 });
            let ht = &h * &t;
            assert!(ht.iter().all(|v| v.abs() < 1e-6), "translation {axis}");
        }
        let (w, _) = symmetric_eigen(&h).unwrap();
        assert!(w[n - 1] > 0.1, "{w:?}");
        hessians.push(h);
    }
    // The charge and the multiplicity of the structure are those of the model.
    let change = max_difference(&hessians[0], &hessians[1]);
    assert!(
        change > 1e-2,
        "the cation has the Hessian of the neutral: {change}"
    );
    // Modifiers apply as to every model.
    let ts = resolve("gfn1xtb_ts")
        .unwrap()
        .hessian(&neutral, &Col::zeros(neutral.dof()))
        .unwrap();
    let (w, _) = symmetric_eigen(&ts).unwrap();
    assert_eq!(w.iter().filter(|&&x| x < -1e-6).count(), 1, "{w:?}");
    assert!(resolve("legacy_gfn1xtb").is_err());
}

#[test]
fn cells_and_elements_without_parameters_are_refused_with_the_reason() {
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([8.0, 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.0, 8.0, 0.0]),
        Vector3::<Angstrom>::new([0.0, 0.0, 8.0]),
        [true, true, true],
    )
    .unwrap();
    let crystal = molecule(
        vec![Element::O, Element::H, Element::H],
        &[
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ],
        SpinCharge::default(),
        Some(cell),
    );
    let model = resolve("gfn1xtb").unwrap();
    let e = model
        .hessian(&crystal, &Col::zeros(crystal.dof()))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("-modelhess gfn1xtb") && e.contains("isolated systems"),
        "{e}"
    );
    // Francium (Z = 87): beyond Rn, the last element of GFN1-xTB.
    let francium = molecule(
        vec![
            Element::from_atomic_number(87).unwrap(),
            Element::from_atomic_number(9).unwrap(),
            Element::from_atomic_number(9).unwrap(),
        ],
        &[[0.0, 0.0, 0.0], [1.98, 0.0, 0.0], [-1.98, 0.0, 0.0]],
        SpinCharge::default(),
        None,
    );
    let e = model
        .hessian(&francium, &Col::zeros(francium.dof()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("-modelhess gfn1xtb") && e.contains("Fr"), "{e}");
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mopt_modelhess_gfn1_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn optimizations_and_a_saddle_search_run_with_the_gfn1_model_on_other_surfaces() {
    let dir = scratch("runs");
    // A minimum on the GFN-FF surface and on the GFN0-xTB surface.
    let water = dir.join("water.xyz");
    std::fs::write(
        &water,
        "3\nwater\nO 0.0 0.0 0.12\nH 0.0 0.83 -0.43\nH 0.05 -0.70 -0.52\n",
    )
    .unwrap();
    let water = water.to_string_lossy().into_owned();
    for calculator in ["gfnff", "gfn0xtb"] {
        let out = run(
            &config_from_cli(&[
                &water,
                "-os",
                calculator,
                "-modelhess",
                "gfn1xtb",
                "-mfc",
                "5",
            ])
            .unwrap(),
            &dir,
            None,
        )
        .unwrap();
        assert!(out[0].converged, "{calculator}");
        let settings = std::fs::read_to_string(out[0].directory.join("settings.json")).unwrap();
        assert!(settings.contains("gfn1xtb"), "{calculator}: {settings}");
    }
    // The model is evaluated in the run: on an argon pair in a periodic cell, which GFN1-xTB
    // in mopt-rs does not take (every element the Lennard-Jones surface has, He to Rn, has
    // GFN1-xTB parameters), the run on the Lennard-Jones surface stops with the model's reason.
    let argon = dir.join("ar2_cell.xyz");
    std::fs::write(
        &argon,
        "2\nLattice=\"12.0 0.0 0.0 0.0 12.0 0.0 0.0 0.0 12.0\" pbc=\"T T T\"\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\n",
    )
    .unwrap();
    let argon = argon.to_string_lossy().into_owned();
    let refused = run(
        &config_from_cli(&[&argon, "-os", "lj", "-modelhess", "gfn1xtb"]).unwrap(),
        &dir,
        None,
    )
    .map(|out| out[0].converged);
    let e = format!("{refused:?}");
    assert!(
        e.contains("-modelhess gfn1xtb") && e.contains("isolated systems"),
        "{e}"
    );
    // The saddle of HCN to HNC on the GFN0-xTB surface, from the GFN1-xTB Hessian of the
    // bent start (which has one negative eigenvalue there).
    let hcn = dir.join("hcn_ts.xyz");
    std::fs::write(
        &hcn,
        "3\nHCN saddle guess\nC 0.0 0.0 0.0\nN 1.18 0.0 0.0\nH 0.62 1.08 0.0\n",
    )
    .unwrap();
    let hcn = hcn.to_string_lossy().into_owned();
    let out = run(
        &config_from_cli(&[
            &hcn,
            "-os",
            "gfn0xtb",
            "-opt",
            "rsirfo_bofill",
            "-order",
            "1",
            "-modelhess",
            "gfn1xtb",
            "-mfc",
            "3",
            "-ns",
            "200",
        ])
        .unwrap(),
        &dir,
        None,
    )
    .unwrap();
    assert!(out[0].converged, "the saddle search did not converge");
    // A saddle of the GFN0-xTB surface: one negative eigenvalue of its Hessian there.
    let text = std::fs::read_to_string(
        std::fs::read_dir(&out[0].directory)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("_optimized.xyz"))
            })
            .expect("the optimized structure"),
    )
    .unwrap();
    let sys = mopt_rs::io::xyz::parse_xyz_str(&text, SpinCharge::default(), None).unwrap();
    let h = mopt_rs::pes::native::Gfn0Xtb::default()
        .hessian(&sys)
        .unwrap();
    let modes = mopt_rs::vib::normal_modes(&sys, &h, true).unwrap();
    // A transition state by the one criterion of the crate (`IMAGINARY_CURVATURE`).
    assert_eq!(modes.imaginary_modes(), 1, "{:?}", modes.wavenumbers);
    let _ = std::fs::remove_dir_all(&dir);
}
