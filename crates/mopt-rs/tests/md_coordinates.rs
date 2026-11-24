//! Molecular dynamics in generalized coordinates: the dynamics in delocalized internal
//! coordinates is that of Cartesian coordinates without rigid motion, the energy is
//! conserved in internal and in curvilinear coordinates that cover all motions, the
//! distributions at constant temperature agree with Cartesian ones, and `mdmain -coord`
//! integrates where it can, runs in Cartesian coordinates where it cannot, and refuses
//! `integrate=coordinates` then.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;
use std::path::PathBuf;

use faer::{Col, Mat};
use mopt_rs::analysis::trajectory::statistics;
use mopt_rs::cli::mdmain::{config_from_cli, run_with};
use mopt_rs::coord::system::{CoordinateRegistry, CoordinateSystem};
use mopt_rs::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::md::generalized::{GeneralizedDynamics, Span};
use mopt_rs::md::{MdSettings, MdSpace, MdStepReport, MolecularDynamics, RigidMotion};
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::workflow::md::{run_md, FnMdObserver, MoleculeMdSurface};

/// Spherical coordinates of every atom about a fixed point (all motions, curvilinear).
struct AtomSpherical([f64; 3]);

impl AtomSpherical {
    fn of(&self, p: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
        let d = [p[0] - self.0[0], p[1] - self.0[1], p[2] - self.0[2]];
        let rho2 = d[0] * d[0] + d[1] * d[1];
        let r2 = rho2 + d[2] * d[2];
        let (r, rho) = (r2.sqrt(), rho2.sqrt());
        (
            [r, (d[2] / r).acos(), d[1].atan2(d[0])],
            [
                [d[0] / r, d[1] / r, d[2] / r],
                [
                    d[0] * d[2] / (r2 * rho),
                    d[1] * d[2] / (r2 * rho),
                    -rho / r2,
                ],
                [-d[1] / rho2, d[0] / rho2, 0.0],
            ],
        )
    }
}

impl CoordinateSystem for AtomSpherical {
    fn name(&self) -> &str {
        "spherical_atoms"
    }
    fn initialize(&mut self, _sys: &MolecularSystem) -> MoptResult<()> {
        Ok(())
    }
    fn dim(&self, sys: &MolecularSystem) -> usize {
        3 * sys.n_atoms()
    }
    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        let x = sys.coords_flat_bohr();
        Ok(Col::from_fn(x.nrows(), |k| {
            let i = k / 3;
            self.of([x[3 * i], x[3 * i + 1], x[3 * i + 2]]).0[k % 3]
        }))
    }
    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        let x = sys.coords_flat_bohr();
        let n = x.nrows();
        let mut b = Mat::zeros(n, n);
        for i in 0..n / 3 {
            let (_, rows) = self.of([x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
            for (a, row) in rows.iter().enumerate() {
                for c in 0..3 {
                    b[(3 * i + a, 3 * i + c)] = row[c];
                }
            }
        }
        Ok(b)
    }
    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        let d = self.values(to)? - self.values(from)?;
        Ok(Col::from_fn(d.nrows(), |k| {
            if k % 3 == 2 {
                (d[k] + PI).rem_euclid(2.0 * PI) - PI
            } else {
                d[k]
            }
        }))
    }
}

/// Water near the GFN-FF minimum.
fn water() -> MolecularSystem {
    MolecularSystem::new(
        ["O", "H", "H"]
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect(),
        &positions::<Angstrom>(&[
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.468],
            [0.0, -0.757, -0.468],
        ]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn gfnff(sys: &MolecularSystem) -> Box<dyn Calculator> {
    CalculatorRegistry::native()
        .create("gfnff", &CalculatorOptions::default(), sys)
        .unwrap()
}

fn settings(dt_fs: f64, steps: usize, thermostat: &str) -> MdSettings {
    MdSettings {
        time_step: dt_fs * Units::FS_TO_AU_TIME,
        steps,
        temperature: 300.0,
        seed: 11,
        thermostat: thermostat.to_string(),
        rigid: RigidMotion::TranslationRotation,
        ..MdSettings::default()
    }
}

fn run(
    calc: &dyn Calculator,
    start: &MolecularSystem,
    coords: Option<Box<dyn CoordinateSystem>>,
    s: MdSettings,
) -> Vec<MdStepReport> {
    let surface = MoleculeMdSurface::with_terms(calc, start.clone(), Vec::new()).unwrap();
    let mut reports = Vec::new();
    {
        let mut keep = FnMdObserver::new(|r: &MdStepReport| {
            reports.push(r.clone());
            Ok(true)
        });
        match coords {
            None => {
                let space = MdSpace::atoms(&start.masses_amu()).unwrap();
                let mut md =
                    MolecularDynamics::new(s, space, start.coords_flat_bohr(), Some(start.clone()))
                        .unwrap();
                run_md(&mut md, &surface, &mut [&mut keep]).unwrap();
            }
            Some(c) => {
                let mut md = GeneralizedDynamics::new(s, c, start.clone(), true).unwrap();
                run_md(&mut md, &surface, &mut [&mut keep]).unwrap();
            }
        }
    }
    reports
}

fn distance(r: &MdStepReport, i: usize, j: usize) -> f64 {
    let d: f64 = (0..3)
        .map(|c| (r.x[3 * i + c] - r.x[3 * j + c]).powi(2))
        .sum();
    d.sqrt() * Units::BOHR_TO_ANGSTROM
}

fn deviation(reports: &[MdStepReport]) -> f64 {
    let c0 = reports[0].conserved;
    reports
        .iter()
        .map(|r| (r.conserved - c0).abs())
        .fold(0.0, f64::max)
}

fn dic() -> Box<dyn CoordinateSystem> {
    CoordinateRegistry::default().create("dic").unwrap()
}

#[test]
fn internal_coordinates_follow_the_cartesian_dynamics_without_rigid_motion() {
    let start = water();
    let calc = gfnff(&start);
    let md = GeneralizedDynamics::new(settings(0.1, 10, "none"), dic(), start.clone(), true);
    assert_eq!(md.unwrap().span(), Span::Internal(6));
    // The same starting state (the velocities of the same seed without rigid motion) and a
    // small time step: both integrators converge to the same trajectory.
    let cart = run(calc.as_ref(), &start, None, settings(0.05, 2000, "none"));
    let internal = run(
        calc.as_ref(),
        &start,
        Some(dic()),
        settings(0.05, 2000, "none"),
    );
    let worst = cart
        .iter()
        .zip(&internal)
        .map(|(a, b)| (distance(a, 0, 1) - distance(b, 0, 1)).abs())
        .fold(0.0, f64::max);
    eprintln!("100 fs: largest difference of r(O-H) {worst:.2e} ang.");
    assert!(worst < 1e-4, "{worst}");
    assert!((cart[0].kinetic - internal[0].kinetic).abs() < 1e-12);
    assert_eq!(internal[0].dof, 3.0);
}

#[test]
fn the_energy_is_conserved_in_internal_and_in_curvilinear_coordinates() {
    let start = water();
    let calc = gfnff(&start);
    let c = start.centroid().value();
    let origin = [c[0] - 11.0, c[1] - 13.0, c[2] - 9.0];
    let cart = deviation(&run(
        calc.as_ref(),
        &start,
        None,
        settings(0.5, 2000, "none"),
    ));
    let internal = deviation(&run(
        calc.as_ref(),
        &start,
        Some(dic()),
        settings(0.5, 2000, "none"),
    ));
    let spherical = deviation(&run(
        calc.as_ref(),
        &start,
        Some(Box::new(AtomSpherical(origin))),
        settings(0.5, 2000, "none"),
    ));
    let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
    eprintln!(
        "1 ps at 0.5 fs, largest deviation / kT: cartesian {:.2e}, dic {:.2e}, spherical {:.2e}",
        cart / kt,
        internal / kt,
        spherical / kt
    );
    for d in [cart, internal, spherical] {
        assert!(d < 0.05 * kt, "{d}");
    }
}

#[test]
fn bond_lengths_at_constant_temperature_agree_with_cartesian_dynamics() {
    let start = water();
    let calc = gfnff(&start);
    let mut means = Vec::new();
    for coords in [None, Some(dic())] {
        let reports = run(calc.as_ref(), &start, coords, settings(0.5, 24000, "bussi"));
        let r: Vec<f64> = reports[4000..].iter().map(|r| distance(r, 0, 1)).collect();
        let t: Vec<f64> = reports[4000..].iter().map(|r| r.temperature).collect();
        let (sr, st) = (statistics(&r).unwrap(), statistics(&t).unwrap());
        eprintln!(
            "r(O-H) {:.5} +- {:.5} (sd {:.4}), T {:.1} +- {:.1}",
            sr.mean, sr.error, sr.deviation, st.mean, st.error
        );
        assert!((st.mean - 300.0).abs() < 4.0 * st.error + 5.0);
        means.push(sr);
    }
    let gap = (means[0].mean - means[1].mean).abs();
    let error = (means[0].error.powi(2) + means[1].error.powi(2)).sqrt();
    assert!(gap < 4.0 * error, "{gap} against {error}");
    assert!((means[0].deviation / means[1].deviation - 1.0).abs() < 0.1);
}

const ETHANOL: &str = "9\n\nC 0.000 0.000 0.000\nC 1.515 0.000 0.000\nO 2.030 1.320 0.000\n\
H 2.990 1.280 0.000\nH -0.390 1.020 0.000\nH -0.390 -0.510 0.880\nH -0.390 -0.510 -0.880\n\
H 1.900 -0.520 0.880\nH 1.900 -0.520 -0.880\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mdcoord_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn mdmain_writes_the_coordinates_and_integrates_in_them_when_asked() {
    let base = scratch("cli");
    let path = base.join("water.xyz");
    std::fs::write(
        &path,
        "3\n\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.468\nH 0.0 -0.757 -0.468\n",
    )
    .unwrap();
    let p = path.to_string_lossy().into_owned();
    let common = [
        "-os", "gfnff", "-mt", "nve", "-time", "40", "-ts", "0.5fs", "-temp", "300",
    ];
    let args = |coord: &str, extra: &[&str]| -> Vec<String> {
        let mut a: Vec<String> = vec![p.clone()];
        a.extend(common.iter().map(|s| s.to_string()));
        a.extend(["-coord".to_string(), coord.to_string()]);
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    let go = |a: &[String]| {
        let refs: Vec<&str> = a.iter().map(String::as_str).collect();
        let config = config_from_cli(&refs).unwrap();
        run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap()
    };
    // By default: Cartesian dynamics, the coordinates written at the frames.
    let runs = go(&args("dic", &["-md_opt", "seed=1", "write_every=5"]));
    let log = std::fs::read_to_string(runs[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("Coordinates 'dic': the dynamics runs in Cartesian coordinates"));
    let csv = std::fs::read_to_string(runs[0].directory.join("water_coordinates.csv")).unwrap();
    assert_eq!(csv.lines().count(), 1 + 9, "header and a row every 5 steps");
    let in_dic = go(&args(
        "dic",
        &[
            "-md_opt",
            "seed=1",
            "write_every=5",
            "integrate=coordinates",
        ],
    ));
    let log = std::fs::read_to_string(in_dic[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("Molecular dynamics in the coordinates 'dic' (3 of them"));
    // The same start and almost the same trajectory.
    let read = |dir: &std::path::Path| std::fs::read_to_string(dir.join("water_coordinates.csv"));
    let (a, b) = (
        read(&runs[0].directory).unwrap(),
        read(&in_dic[0].directory).unwrap(),
    );
    let last = |t: &str| -> Vec<f64> {
        t.lines()
            .last()
            .unwrap()
            .split(',')
            .skip(2)
            .map(|v| v.parse().unwrap())
            .collect()
    };
    for (x, y) in last(&a).iter().zip(last(&b)) {
        assert!((x - y).abs() < 1e-3, "{x} {y}");
    }
    // The primitive internal coordinates of water (two bonds, one angle) are not redundant:
    // they are integrated in.
    let runs = go(&args(
        "ric",
        &["-md_opt", "seed=1", "integrate=coordinates"],
    ));
    let log = std::fs::read_to_string(runs[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("Molecular dynamics in the coordinates 'ric' (3 of them"));
    // Redundant coordinates (the primitive internal coordinates of ethanol: 33 for 21
    // internal motions): Cartesian dynamics, the coordinates written.
    let ethanol = base.join("ethanol.xyz");
    std::fs::write(&ethanol, ETHANOL).unwrap();
    let e = ethanol.to_string_lossy().into_owned();
    let with = |extra: &[&str]| -> Vec<String> {
        let mut a: Vec<String> = vec![e.clone()];
        a.extend(common.iter().map(|s| s.to_string()));
        a.extend(["-coord".to_string(), "ric".to_string()]);
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    let runs = go(&with(&["-md_opt", "seed=1"]));
    let log = std::fs::read_to_string(runs[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("Coordinates 'ric': the dynamics runs in Cartesian coordinates"),
        "{log}"
    );
    assert!(runs[0].directory.join("ethanol_coordinates.csv").exists());
    // Asked for where it cannot be done: refused with the reason, exit code 2.
    let refused = with(&["-md_opt", "integrate=coordinates"]);
    let refs: Vec<&str> = refused.iter().map(String::as_str).collect();
    let config = config_from_cli(&refs).unwrap();
    let err = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap_err();
    // An unsupported combination is a usage error (exit code 2).
    assert!(matches!(err, MoptError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("redundant"), "{err}");
    let held = args("dic", &["-fix", "1", "-md_opt", "integrate=coordinates"]);
    let refs: Vec<&str> = held.iter().map(String::as_str).collect();
    let config = config_from_cli(&refs).unwrap();
    let err = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap_err();
    assert!(err.to_string().contains("constraints"), "{err}");
    // By default the same run holds atom 1 in Cartesian dynamics.
    let held = args("dic", &["-fix", "1", "-md_opt", "seed=1"]);
    assert!(go(&held)[0]
        .directory
        .join("water_coordinates.csv")
        .exists());
    let _ = std::fs::remove_dir_all(&base);
}
