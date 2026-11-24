//! What the surface terms outside the bias potentials declare about themselves (frame,
//! history, copies), checked against a rigid motion of the structure: the dispersion
//! corrections, the umbrella restraints on collective variables, the terms of the core of
//! ONIOM microiterations, and the artificial force on the cell of the mapper. (The bias
//! potentials are in `term_contracts.rs`; the reaction coordinate of `confsearch` and the
//! clocked walls of the nanoreactor are private and delegate or are invariant by
//! construction.)
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::domain::{
    positions, Angstrom, AtomId, Element, MolecularSystem, PeriodicCell, Position, SpinCharge,
    Vector3,
};
use mopt_rs::md::{Cv, Umbrella};
use mopt_rs::pes::dispersion::{D3Bj, D4};
use mopt_rs::pes::oniom::micro::{CoreSplit, CoreTerm};
use mopt_rs::pes::term::fork_term;
use mopt_rs::pes::{Frame, History, PesTerm, TermContext};
use mopt_rs::workflow::mapper::afir::CellPush;

/// A water dimer without symmetry.
fn dimer() -> MolecularSystem {
    MolecularSystem::new(
        vec![
            Element::O,
            Element::H,
            Element::H,
            Element::O,
            Element::H,
            Element::H,
        ],
        &positions::<Angstrom>(&[
            [0.00, 0.00, 0.12],
            [0.00, 0.76, -0.47],
            [0.05, -0.75, -0.49],
            [2.81, 0.12, 0.31],
            [3.21, 0.85, -0.18],
            [3.25, -0.71, 0.02],
        ]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn moved(sys: &MolecularSystem) -> MolecularSystem {
    let (s, c) = 0.9_f64.sin_cos();
    let mut out = sys.clone();
    for a in sys.atoms() {
        let p = sys.position(a).value();
        out.set_position(
            a,
            Position::new([
                c * p[0] + s * p[2] - 0.4,
                p[1] + 1.7,
                -s * p[0] + c * p[2] + 0.9,
            ]),
        );
    }
    out
}

fn table(sys: &MolecularSystem) -> Vec<(&'static str, Arc<dyn PesTerm>, Frame, History)> {
    let (a, b, c, d) = (
        AtomId::new(0),
        AtomId::new(1),
        AtomId::new(3),
        AtomId::new(4),
    );
    let umbrella = |cv: Cv| -> Arc<dyn PesTerm> { Arc::new(Umbrella::new(cv, 1.0, 0.3).unwrap()) };
    let x: Vec<f64> = sys.coords_flat_bohr().iter().take(6).copied().collect();
    let shifted: Vec<f64> = x.iter().map(|v| v + 0.4).collect();
    let spring = parse_bias_option("-kp", &["0.1".into(), "1.0".into(), "1,4".into()])
        .unwrap()
        .remove(0)
        .build(sys)
        .unwrap();
    let ramped = parse_bias_option("-kp", &["0.1,0.3".into(), "1.0".into(), "1,4".into()])
        .unwrap()
        .remove(0)
        .build(sys)
        .unwrap();
    let split = CoreSplit {
        core: vec![0, 1, 2, 3],
        environment: vec![4, 5],
        fixed: vec![],
    };
    let core = |t: Arc<dyn PesTerm>| -> Arc<dyn PesTerm> {
        Arc::new(CoreTerm::new(t, sys, &split).unwrap())
    };
    use Frame::{Absolute, Invariant};
    use History::{IterationRamp, Static};
    vec![
        ("d3", Arc::new(D3Bj::PBE), Invariant, Static),
        ("d4", Arc::new(D4::PBE), Invariant, Static),
        (
            "umbrella distance",
            umbrella(Cv::distance(a, c)),
            Invariant,
            Static,
        ),
        (
            "umbrella angle",
            umbrella(Cv::angle(b, a, c)),
            Invariant,
            Static,
        ),
        (
            "umbrella dihedral",
            umbrella(Cv::dihedral(b, a, c, d)),
            Invariant,
            Static,
        ),
        (
            "umbrella coordination",
            umbrella(Cv::coordination(vec![a], vec![b, c], 2.0, 6, 12).unwrap()),
            Invariant,
            Static,
        ),
        (
            "umbrella expression",
            umbrella(Cv::expression("dist(1,4) - dist(1,2)", None).unwrap()),
            Invariant,
            Static,
        ),
        (
            "umbrella path",
            umbrella(Cv::path(vec![a, b], vec![x, shifted], 2.0, true).unwrap()),
            Absolute,
            Static,
        ),
        ("core spring", core(spring), Invariant, Static),
        ("core ramp", core(ramped), Invariant, IterationRamp),
    ]
}

#[test]
fn every_term_declares_its_frame_and_history_and_keeps_its_word() {
    let sys = dimer();
    let other = moved(&sys);
    let ctx = TermContext::at_iteration(3, &sys);
    for (name, term, frame, history) in table(&sys) {
        assert_eq!(term.frame(), frame, "{name}");
        assert_eq!(term.history(), history, "{name}");
        // The core terms take structures of the core.
        let (here, there) = if name.starts_with("core") {
            let keep = |s: &MolecularSystem| {
                MolecularSystem::new(
                    s.elements[..4].to_vec(),
                    &positions::<mopt_rs::domain::Bohr>(
                        &(0..4)
                            .map(|k| s.position(AtomId::new(k)).value())
                            .collect::<Vec<_>>(),
                    ),
                    SpinCharge::default(),
                    None,
                )
                .unwrap()
            };
            (keep(&sys), keep(&other))
        } else {
            (sys.clone(), other.clone())
        };
        let e0 = term.evaluate(&here, &ctx, false).unwrap().energy.value();
        let e1 = term.evaluate(&there, &ctx, false).unwrap().energy.value();
        match frame {
            Frame::Invariant => assert!(
                (e1 - e0).abs() < 1e-10 * (1.0 + e0.abs()),
                "{name}: {e0} -> {e1}"
            ),
            _ => assert!((e1 - e0).abs() > 1e-8, "{name}: {e0} -> {e1}"),
        }
        // Stateless terms are shared between jobs.
        if history != History::Stateful {
            assert!(fork_term(&term).is_ok(), "{name}");
        }
    }
}

#[test]
fn the_force_on_the_cell_depends_on_the_lattice_alone() {
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([6.0, 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.5, 6.2, 0.0]),
        Vector3::<Angstrom>::new([0.0, 0.3, 6.5]),
        [true; 3],
    )
    .unwrap();
    let sys = MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); 2],
        &positions::<Angstrom>(&[[0.5, 0.4, 0.3], [3.1, 2.9, 3.3]]),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap();
    let push = CellPush::new(100.0, "a").unwrap();
    assert_eq!(
        (push.frame(), push.history()),
        (Frame::Invariant, History::Static)
    );
    assert!(push.analytic_stress() && push.analytic_hessian());
    let ctx = TermContext::at_iteration(0, &sys);
    let mut shifted = sys.clone();
    for a in sys.atoms() {
        let p = sys.position(a).value();
        shifted.set_position(a, Position::new([p[0] + 0.7, p[1] - 0.2, p[2] + 0.1]));
    }
    let (e0, e1) = (
        push.evaluate(&sys, &ctx, false).unwrap().energy.value(),
        push.evaluate(&shifted, &ctx, false).unwrap().energy.value(),
    );
    assert_eq!(e0, e1);
}
