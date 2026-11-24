//! Tests of the trajectory analysis on series with known answers.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::*;
use crate::core_infra::rng::Rng;
use crate::domain::{positions, Angstrom, Element, MolecularSystem, PeriodicCell, Quantity};
use crate::domain::{SpinCharge, Units};

fn system(elements: &[Element], ang: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
    match MolecularSystem::new(
        elements.to_vec(),
        &positions::<Angstrom>(ang),
        SpinCharge::default(),
        cell,
    ) {
        Ok(s) => s,
        Err(e) => panic!("{e}"),
    }
}

fn cubic(a: f64) -> PeriodicCell {
    let l = Quantity::<Angstrom>::new(a);
    match PeriodicCell::orthorhombic(l, l, l) {
        Ok(c) => c,
        Err(e) => panic!("{e}"),
    }
}

fn ok<T>(r: crate::error::MoptResult<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    }
}

/// Face-centred cubic lattice of `k^3` conventional cells of side `a`.
fn fcc(a: f64, k: usize) -> Vec<[f64; 3]> {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut out = Vec::new();
    for x in 0..k {
        for y in 0..k {
            for z in 0..k {
                for b in basis {
                    out.push([
                        (x as f64 + b[0]) * a,
                        (y as f64 + b[1]) * a,
                        (z as f64 + b[2]) * a,
                    ]);
                }
            }
        }
    }
    out
}

#[test]
fn independent_samples_have_no_inefficiency_and_correlated_ones_have_theirs() {
    let mut rng = Rng::from_seed(3);
    let white: Vec<f64> = (0..8192).map(|_| rng.standard_normal()).collect();
    let s = ok(statistics(&white));
    assert!(s.converged);
    assert!((s.inefficiency - 1.0).abs() < 0.4, "{}", s.inefficiency);
    // AR(1) with phi = 0.9: g = (1 + phi) / (1 - phi) = 19.
    let mut x = 0.0;
    let ar: Vec<f64> = (0..65536)
        .map(|_| {
            x = 0.9 * x + rng.standard_normal();
            x
        })
        .collect();
    let s = ok(statistics(&ar));
    assert!(s.converged);
    assert!(
        (s.inefficiency - 19.0).abs() < 6.0,
        "inefficiency {}",
        s.inefficiency
    );
    assert!(s.error > 3.0 * s.naive_error);
    let flat = ok(statistics(&[2.0; 10]));
    assert_eq!((flat.mean, flat.error), (2.0, 0.0));
    assert!(statistics(&[1.0]).is_err());
}

#[test]
fn the_rdf_of_a_face_centred_cubic_crystal_has_its_shells() {
    let a = 3.6;
    let atoms = fcc(a, 3);
    let elements = vec![ok(Element::from_atomic_number(29)); atoms.len()];
    let s = system(&elements, &atoms, Some(cubic(3.0 * a)));
    let traj = ok(MdTrajectory::new(vec![s], vec![0.0]));
    let rdf = ok(radial_distribution(&traj, &RdfSettings::default()));
    assert!(rdf.periodic);
    assert_eq!(rdf.pairs.len(), 1);
    let pair = &rdf.pairs[0];
    let at = |r: f64| ((r / rdf.bin_width) as usize).min(rdf.r.len() - 1);
    // Shells: 12 at a / sqrt 2, 6 at a, 24 at a sqrt(3/2).
    let first = a / 2.0_f64.sqrt();
    assert!(pair.coordination[at(first - 0.05)] < 1e-9);
    assert!((pair.coordination[at(first + 0.05)] - 12.0).abs() < 1e-9);
    assert!((pair.coordination[at(a + 0.05)] - 18.0).abs() < 1e-9);
    assert!((pair.coordination[at(a * 1.5_f64.sqrt() + 0.05)] - 42.0).abs() < 1e-9);
    // The peak of g in the bin of the first shell: 12 / (rho dV), rho = (N - 1) / V.
    let rho = (atoms.len() - 1) as f64 / (3.0 * a).powi(3);
    let r = rdf.r[at(first)];
    let expected =
        12.0 / (rho * 4.0 / 3.0 * std::f64::consts::PI * ((r + 0.01).powi(3) - (r - 0.01).powi(3)));
    assert!((pair.g[at(first)] / expected - 1.0).abs() < 1e-6);
    // The limit of the minimum image.
    let settings = RdfSettings {
        max: Some(6.0),
        ..RdfSettings::default()
    };
    assert!(radial_distribution(&traj, &settings).is_err());
}

#[test]
fn a_random_walk_has_its_diffusion_coefficient() {
    let mut rng = Rng::from_seed(11);
    let (n, frames, sigma, dt) = (200, 1000, 0.05, 2.0);
    let mut p: Vec<[f64; 3]> = (0..n).map(|i| [i as f64 * 0.3, 0.0, 0.0]).collect();
    let elements = vec![ok(Element::from_atomic_number(18)); n];
    // In a cell of 5 Angstrom, so that the walkers cross its faces many times.
    let mut systems = Vec::new();
    let mut times = Vec::new();
    for t in 0..frames {
        let wrapped: Vec<[f64; 3]> = p
            .iter()
            .map(|q| std::array::from_fn(|c| q[c].rem_euclid(5.0)))
            .collect();
        systems.push(system(&elements, &wrapped, Some(cubic(5.0))));
        times.push(t as f64 * dt);
        for q in &mut p {
            for c in q.iter_mut() {
                *c += sigma * rng.standard_normal();
            }
        }
    }
    let traj = ok(MdTrajectory::new(systems, times));
    let msd = ok(mean_squared_displacement(
        &traj,
        &MsdSettings {
            remove_drift: false,
            ..MsdSettings::default()
        },
    ));
    // MSD = 3 sigma^2 m, D = sigma^2 / (2 dt) Angstrom^2 / fs.
    let expected = sigma * sigma / (2.0 * dt) * 1000.0;
    let d = msd.total_diffusion.coefficient;
    assert!((d / expected - 1.0).abs() < 0.15, "D {d} vs {expected}");
    assert!((msd.total[10] / (3.0 * sigma * sigma * 10.0) - 1.0).abs() < 0.1);
}

#[test]
fn oscillating_velocities_give_a_peak_at_their_frequency() {
    // Two atoms vibrating at 1000 cm^-1 (period 33.36 fs), frames 1 fs apart.
    let nu = 1000.0;
    let omega = 2.0 * std::f64::consts::PI * Units::SPEED_OF_LIGHT_CM_S * 1e-15 * nu;
    let frames = 4000;
    let elements = [Element::H, Element::H];
    let mut systems = Vec::new();
    let mut velocities = Vec::new();
    for t in 0..frames {
        let x = 0.1 * libm::sin(omega * t as f64);
        systems.push(system(
            &elements,
            &[[0.0, 0.0, -x], [0.0, 0.0, 1.0 + x]],
            None,
        ));
        let v = 0.001 * libm::cos(omega * t as f64);
        velocities.push(Col::from_fn(6, |k| match k {
            2 => -v,
            5 => v,
            _ => 0.0,
        }));
    }
    let times: Vec<f64> = (0..frames).map(|t| t as f64).collect();
    let traj = ok(ok(MdTrajectory::new(systems, times)).with_velocities(&velocities));
    let vacf = ok(velocity_autocorrelation(&traj, 0.5));
    assert!((vacf.total[0] - 1.0).abs() < 1e-12);
    let vdos = ok(vibrational_density_of_states(
        &vacf,
        &VdosSettings::default(),
    ));
    let found = peaks(&vdos.wavenumbers, &vdos.total, 0.5);
    assert!(!found.is_empty());
    assert!((found[0].0 - nu).abs() <= 2.0, "peak {:?}", found[0]);
    let integral: f64 =
        vdos.total.iter().sum::<f64>() * (vdos.wavenumbers[1] - vdos.wavenumbers[0]);
    assert!((integral / 6.0 - 1.0).abs() < 0.02, "integral {integral}");
    assert!(vdos.nyquist > 16000.0);
}

#[test]
fn measures_and_superposition_see_through_the_cell_and_rotations() {
    let elements = [Element::C, Element::C, Element::C, Element::C];
    // A chain with a dihedral of 90 degrees, split across the faces of a cell of 10.
    let ang = [
        [9.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 1.0, 1.0],
    ];
    let s = system(&elements, &ang, Some(cubic(10.0)));
    let traj = ok(MdTrajectory::new(vec![s.clone(), s], vec![0.0, 1.0]));
    let m = [
        ok(Measure::of(&[0, 1])),
        ok(Measure::of(&[0, 1, 2])),
        ok(Measure::of(&[0, 1, 2, 3])),
    ];
    let values = ok(geometry_series(&traj, &m));
    assert!((values[0][0] - 1.0).abs() < 1e-9);
    assert!((values[1][0] - 90.0).abs() < 1e-9);
    assert!((values[2][0].abs() - 90.0).abs() < 1e-9);
    assert_eq!(m[2].name(), "dihedral 1-2-3-4 (deg.)");
    assert!(geometry_series(&traj, &[Measure::Distance(0, 7)]).is_err());
    // A molecule rotated rigidly has no RMSD.
    let mol = [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.5, 1.1, 0.0],
        [2.0, 1.4, 0.9],
    ];
    let rotated: Vec<[f64; 3]> = mol.iter().map(|p| [-p[1], p[0], p[2] + 3.0]).collect();
    let traj = ok(MdTrajectory::new(
        vec![
            system(&elements, &mol, None),
            system(&elements, &rotated, None),
        ],
        vec![0.0, 1.0],
    ));
    let f = ok(fluctuations(&traj));
    assert!(f.rmsd[1] < 1e-9 && f.rmsf.iter().all(|r| *r < 1e-9));
}

#[test]
fn the_principal_component_of_a_stretch_is_the_stretch() {
    let elements = [Element::H, Element::H];
    let systems: Vec<MolecularSystem> = (0..20)
        .map(|t| {
            let x = 0.05 * libm::sin(0.7 * t as f64);
            system(&elements, &[[0.0, 0.0, -x], [0.0, 0.0, 0.74 + x]], None)
        })
        .collect();
    let times = (0..20).map(|t| t as f64).collect();
    let traj = ok(MdTrajectory::new(systems, times));
    let pca = ok(trajectory_pca(&traj, 3));
    assert!(pca.contributions[0] > 0.999);
    assert_eq!(pca.scores.nrows(), 20);
    // The frames before 5 fs are dropped; uneven frames cannot be correlated.
    assert_eq!(ok(traj.from_time(5.0)).len(), 15);
    let uneven = ok(MdTrajectory::new(
        traj.systems()[..3].to_vec(),
        vec![0.0, 1.0, 3.0],
    ));
    assert!(uneven.interval().is_err());
    assert_eq!(
        time_in_comment("Frame 3 time=1.500000 fs energy=-1.0"),
        Some(1.5)
    );
}
