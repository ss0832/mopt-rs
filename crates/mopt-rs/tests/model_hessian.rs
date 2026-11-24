//! Model Hessians: reference values, xtb's D2 term, invariances, modifiers, and use in
//! optimizations.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use faer::{Col, Mat};
use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::domain::Units;
use mopt_rs::domain::{positions, Angstrom, Bohr, Element, MolecularSystem, PeriodicCell};
use mopt_rs::domain::{SpinCharge, Vector3};
use mopt_rs::math::linalg::symmetric_eigen;
use mopt_rs::model_hessian::{resolve, InternalModel, ModelHessian, ModelSpec};
use mopt_rs::parameters::covalent_radius_bohr;

fn sys(el: Vec<Element>, pos: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
    MolecularSystem::new(el, &positions::<Bohr>(pos), SpinCharge::default(), cell).unwrap()
}

fn h2(r: f64) -> MolecularSystem {
    sys(vec![Element::H; 2], &[[0.0, 0.0, 0.0], [r, 0.0, 0.0]], None)
}

fn hessian(name: &str, s: &MolecularSystem) -> Mat<f64> {
    resolve(name)
        .unwrap()
        .hessian(s, &Col::zeros(s.dof()))
        .unwrap()
}

/// The only non-zero eigenvalue of a two-atom stretch Hessian is `2 k`.
fn stretch_constant(h: &Mat<f64>) -> f64 {
    let (w, _) = symmetric_eigen(h).unwrap();
    w[w.nrows() - 1] / 2.0
}

#[test]
fn diatomic_force_constants_match_the_published_formulas() {
    let r = 1.5;
    // Lindh 1995: 0.45 exp(alpha (r_ref^2 - r^2)) with alpha = 1.0, r_ref = 1.35 for H-H.
    let k = stretch_constant(&hessian("lindh", &h2(r)));
    assert!(
        (k - 0.45 * (1.0f64 * (1.35f64 * 1.35 - r * r)).exp()).abs() < 1e-12,
        "{k}"
    );
    // Schlegel: 1.734 / (r - B)^3 with B = -0.2573 for H-H.
    let k = stretch_constant(&hessian("schlegel", &h2(r)));
    assert!((k - 1.734 / (r + 0.2573f64).powi(3)).abs() < 1e-12, "{k}");
    // Fischer: 0.3601 exp(-1.944 (r - r_cov)).
    let rc = 2.0 * covalent_radius_bohr(Element::H);
    let k = stretch_constant(&hessian("fischer", &h2(r)));
    assert!(
        (k - 0.3601 * (-1.944 * (r - rc)).exp()).abs() < 1e-12,
        "{k}"
    );
}

/// xtb's closed-form second derivatives of the damped D2 pair energy (`shared.f90`).
fn xtb_vdw_xx(rx: f64, ry: f64, rz: f64, c6: f64, s6: f64, r0: f64) -> f64 {
    let d = 20.0;
    let t1 = s6 * c6;
    let t2 = rx * rx;
    let t5 = rx * rx + ry * ry + rz * rz;
    let t6 = t5 * t5;
    let t7 = t6 * t6;
    let t10 = t5.sqrt();
    let t11 = 1.0 / r0;
    let t15 = (-d * (t10 * t11 - 1.0)).exp();
    let t16 = 1.0 + t15;
    let t17 = 1.0 / t16;
    let t24 = t16 * t16;
    let t25 = 1.0 / t24;
    let t29 = t11 * t15;
    let t33 = 1.0 / t7;
    let t44 = d * d / (r0 * r0);
    let t45 = t15 * t15;
    -48.0 * t1 / t7 / t5 * t17 * t2
        + 13.0 * t1 / t10 / t7 * t25 * t2 * d * t29
        + 6.0 * t1 * t33 * t17
        - 2.0 * t1 * t33 / t24 / t16 * t44 * t2 * t45
        - t1 / t10 / t6 / t5 * t25 * d * t29
        + t1 * t33 * t25 * t44 * t2 * t15
}

#[test]
fn d2_pair_term_is_xtbs() {
    // Two atoms far apart: only the D2 term and a negligible stretch remain. xtb subtracts its
    // vdw block on the diagonal atom blocks.
    let (a, b) = (Element::C, Element::from_atomic_number(18).unwrap());
    let d = [6.1, -2.3, 1.7];
    let s = sys(vec![a, b], &[[0.0; 3], d], None);
    let model = InternalModel::xtb_lindh_d2();
    let mut no_d2 = model.clone();
    no_d2.params.s6 = 0.0;
    let g = Col::zeros(6);
    let h = model.hessian(&s, &g).unwrap() - no_d2.hessian(&s, &g).unwrap();
    let c6 = (1.75f64 * 4.61).sqrt();
    let r0 = (1.32 + 1.45) * Units::ANGSTROM_TO_BOHR;
    let dx = [-d[0], -d[1], -d[2]];
    let want = -xtb_vdw_xx(dx[0], dx[1], dx[2], c6, 20.0, r0);
    assert!(
        (h[(0, 0)] - want).abs() < 1e-12 * want.abs().max(1.0),
        "{} vs {want}",
        h[(0, 0)]
    );
    assert!((h[(0, 3)] + want).abs() < 1e-12 * want.abs().max(1.0));
}

fn water() -> MolecularSystem {
    water_in(None)
}

fn water_in(cell: Option<PeriodicCell>) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::O, Element::H, Element::H, Element::C, Element::N],
        &positions::<Angstrom>(&[
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
            [1.9, 0.1, 0.3],
            [3.0, 0.15, 0.6],
        ]),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

const GENUINE: [&str; 15] = [
    "lindh",
    "lindh2007",
    "lindh2007d3",
    "schlegeld3",
    "fischerd3",
    "fischerd3old",
    "swartd2",
    "swartd3",
    "xtb_lindh",
    "xtb_lindh_d2",
    "xtb_swart",
    "swart",
    "schlegel",
    "fischer",
    "morse",
];

#[test]
fn models_are_symmetric_finite_and_translation_invariant() {
    let iso = water();
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([5.0, 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.8, 5.5, 0.0]),
        Vector3::<Angstrom>::new([0.0, 0.0, 30.0]),
        [true, true, false],
    )
    .unwrap();
    let slab = water_in(Some(cell));
    for s in [iso, slab] {
        for name in GENUINE {
            let h = hessian(name, &s);
            let n = h.nrows();
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        h[(i, j)].is_finite() && (h[(i, j)] - h[(j, i)]).abs() < 1e-12,
                        "{name}"
                    );
                }
            }
            for axis in 0..3 {
                let t = Col::from_fn(n, |i| if i % 3 == axis { 1.0 } else { 0.0 });
                let ht = &h * &t;
                assert!(
                    ht.iter().all(|v| v.abs() < 1e-10),
                    "{name}: translation {axis}"
                );
            }
            let (w, _) = symmetric_eigen(&h).unwrap();
            assert!(w[n - 1] > 1e-3, "{name}: no curvature");
        }
    }
}

#[test]
fn ts_and_clip_modifiers() {
    let s = water();
    let h = hessian("lindh_ts", &s);
    let (w, _) = symmetric_eigen(&h).unwrap();
    assert_eq!(w.iter().filter(|&&x| x < -1e-8).count(), 1);
    let h = hessian("schlegel_clip", &s);
    let (w, _) = symmetric_eigen(&h).unwrap();
    assert!(w.iter().all(|x| x.abs() <= 2.0));
    assert!(ModelSpec::parse("legacy_lindh").unwrap().legacy);
    assert!(resolve("lindh2007d4").is_ok());
    let d3 = hessian("schlegeld3", &s) - hessian("schlegel", &s);
    let want = mopt_rs::pes::dispersion::D3Bj::PBE
        .evaluate(&s, true)
        .unwrap()
        .hessian_bohr
        .unwrap();
    let diff = (0..d3.nrows())
        .flat_map(|i| (0..d3.ncols()).map(move |j| (i, j)))
        .fold(0.0_f64, |m, (i, j)| {
            m.max((d3[(i, j)] - want[(i, j)]).abs())
        });
    assert!(diff < 1e-12, "the D3 term is the D3(BJ) Hessian: {diff}");
}

#[test]
fn model_hessian_from_the_command_line() {
    let dir = std::env::temp_dir().join(format!("mopt_modelhess_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let xyz = dir.join("ar7.xyz");
    std::fs::write(
        &xyz,
        "7\n\nAr 1.95 0 0\nAr 0.6026 1.8546 0.03\nAr -1.5776 1.1462 0.07\nAr -1.5776 -1.1462 0.11\n\
         Ar 0.6026 -1.8546 0.15\nAr 0 0 2.1\nAr 0.2 0 -2.0\n",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    for model in ["swart", "xtb_swart", "morse"] {
        let out = run(
            &config_from_cli(&[
                &xyz,
                "-os",
                "lj",
                "-opt",
                "rsirfo_bfgs",
                "-modelhess",
                model,
                "-mfc",
                "5",
            ])
            .unwrap(),
            &dir,
            None,
        )
        .unwrap();
        assert!(out[0].converged, "{model}");
    }
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-modelhess", "nonsense"]).is_ok());
    let bad = run(
        &config_from_cli(&[&xyz, "-os", "lj", "-modelhess", "nonsense"]).unwrap(),
        &dir,
        None,
    );
    assert!(bad.is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
