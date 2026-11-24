//! Tests of the settings and the walls of the mapper method `nanoreactor`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;
use crate::domain::{positions, Angstrom, Element, SpinCharge};

fn extra(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn water() -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::O, Element::H, Element::H],
        &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

#[test]
fn settings_are_read_with_their_defaults_and_refused_when_wrong() {
    let s = NanoreactorSettings::parse(&BTreeMap::new(), 7).unwrap();
    assert_eq!(
        s,
        NanoreactorSettings {
            seed: 7,
            ..NanoreactorSettings::default()
        }
    );
    let s = NanoreactorSettings::parse(
        &extra(&[
            ("runs", "3"),
            ("detect", "quench"),
            ("path", "dmf"),
            ("path_images", "5"),
            ("r1", "6"),
        ]),
        1,
    )
    .unwrap();
    assert_eq!(s.runs, 3);
    assert_eq!(s.detect, Detection::Quench);
    assert_eq!(s.path, "dmf");
    assert_eq!(s.r1, Some(6.0));
    assert!(s.segment);
    // A method of two ends alone works without the frames of the trajectory.
    let s =
        NanoreactorSettings::parse(&extra(&[("path", "dsafir"), ("segment", "off")]), 1).unwrap();
    assert!(!s.segment);
    for bad in [
        &[("nonsense", "1")][..],
        &[("runs", "0")],
        &[("wide", "1.0")],
        &[("detect", "maybe")],
        &[("temperature", "-5")],
        // One-ended methods cannot join two minima.
        &[("path", "addf")],
        &[("path", "no_such_method")],
        &[("path_nonsense", "1")],
        // DS-AFIR takes no first path.
        &[("path", "dsafir")],
        &[("segment", "maybe")],
    ] {
        assert!(
            NanoreactorSettings::parse(&extra(bad), 1).is_err(),
            "{bad:?}"
        );
    }
}

#[test]
fn replicas_stand_on_a_geometric_ladder() {
    let s = NanoreactorSettings::parse(&extra(&[("replicas", "4"), ("temperature", "500")]), 1)
        .unwrap();
    let ladder = Nanoreactor::new(s).unwrap().ladder();
    let temperatures: Vec<f64> = ladder.iter().map(|r| r.0).collect();
    // From the temperature to twice it, each rung the same factor above the one below.
    assert!((temperatures[0] - 500.0).abs() < 1e-9 && (temperatures[3] - 1000.0).abs() < 1e-9);
    let factor = temperatures[1] / temperatures[0];
    assert!((temperatures[2] / temperatures[1] - factor).abs() < 1e-12);
    assert!(ladder.iter().all(|r| r.1 == 1.0));
    // Walls: one temperature, the narrow sphere from k2 to three times it.
    let s = NanoreactorSettings::parse(
        &extra(&[
            ("replicas", "3"),
            ("exchange", "walls"),
            ("replica_top", "3"),
        ]),
        1,
    )
    .unwrap();
    let ladder = Nanoreactor::new(s.clone()).unwrap().ladder();
    assert!(ladder.iter().all(|r| r.0 == s.temperature));
    assert!((ladder[1].1 - 3f64.sqrt()).abs() < 1e-12 && (ladder[2].1 - 3.0).abs() < 1e-12);
    for bad in [
        &[("replicas", "0")][..],
        &[("exchange", "pressure")],
        &[("replica_top", "1500"), ("temperature", "2000")],
        &[("exchange", "walls"), ("replica_top", "0.5")],
        &[("exchange_every", "0")],
    ] {
        assert!(
            NanoreactorSettings::parse(&extra(bad), 1).is_err(),
            "{bad:?}"
        );
    }
}

#[test]
fn the_walls_follow_the_size_of_the_minimum() {
    let sys = water();
    let method = Nanoreactor::new(NanoreactorSettings::default()).unwrap();
    let walls = method.walls(&sys);
    let (centre, radius) = centre_and_radius(&sys);
    assert_eq!(walls.center, centre);
    let a = Units::ANGSTROM_TO_BOHR;
    assert!((walls.r1 - (radius + 2.0 * a)).abs() < 1e-12);
    assert!((walls.r2 - NARROW_SHARE * radius).abs() < 1e-12);
    // The paper's share of the wide phase and the period of mopt-rs.
    let period = 250.0 * Units::FS_TO_AU_TIME;
    assert!((walls.period - period).abs() < 1e-9);
    assert!((walls.tau - 0.75 * period).abs() < 1e-9);
    // Force constants per amu in Hartree / Bohr^2.
    assert!((walls.k1 - Units::KCAL_MOL_TO_HARTREE / (a * a)).abs() < 1e-15);
    assert!((walls.k2 - 5.0 * walls.k1).abs() < 1e-15);
    assert_eq!(walls.atoms.len(), 3);
    // Radii given in Angstrom; the narrow sphere never lies outside the wide one.
    let given = Nanoreactor::new(NanoreactorSettings {
        r1: Some(3.0),
        r2: Some(5.0),
        ..NanoreactorSettings::default()
    })
    .unwrap()
    .walls(&sys);
    assert!((given.r1 - 3.0 * a).abs() < 1e-12);
    assert!((given.r2 - given.r1).abs() < 1e-12);
}

#[test]
fn bonds_choose_the_detection() {
    let method = Nanoreactor::new(NanoreactorSettings::default()).unwrap();
    assert!(method.uses_bonds(&water()));
    let argon = Element::from_atomic_number(18).unwrap();
    let pair = MolecularSystem::new(
        vec![argon, argon],
        &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    assert!(!method.uses_bonds(&pair));
}
