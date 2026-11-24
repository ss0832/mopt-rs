//! Tests of the network file: a network comes back as it was written, and other layouts are refused.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;
use crate::domain::{positions, Angstrom};

#[test]
fn a_network_comes_back_as_it_was_written() {
    let sys = |x: f64| {
        MolecularSystem::new(
            vec![
                Element::from_atomic_number(18).unwrap(),
                Element::from_atomic_number(18).unwrap(),
            ],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [x, 0.1 / 3.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    };
    let state = ElectronicState::of(&sys(3.8), 0);
    let mut net = Network::new(vec![
        Element::from_atomic_number(18).unwrap(),
        Element::from_atomic_number(18).unwrap(),
    ]);
    net.inputs.push(InputRecord {
        file: "a.xyz".into(),
        frame: 0,
        node: Some(0),
        goal: false,
    });
    for (x, kind) in [
        (3.8, NodeKind::Minimum),
        (4.5, NodeKind::TransitionState),
        (
            4.0,
            NodeKind::Crossing {
                kind: CrossingKind::Mecp,
                states: [
                    state,
                    ElectronicState {
                        multiplicity: 3,
                        ..state
                    },
                ],
            },
        ),
    ] {
        let mut n = Node::new(
            kind,
            Level::Exploration,
            sys(x).into(),
            -1.0 / 3.0,
            state,
            Origin {
                input: Some(0),
                job: None,
                from_node: None,
            },
        );
        n.frequencies = Some(vec![-12.5, 30.0]);
        n.index_wavenumbers = Some(vec![-42.7, -6.5]);
        net.add_node(n).unwrap();
    }
    net.add_ts_edge(TsEdge {
        id: 0,
        level: Level::Exploration,
        ts: 1,
        ends: [0, 0],
        barriers: [0.1, 0.2 / 7.0],
        imaginary_wavenumber: Some(-40.0),
        irc: IrcPath {
            sides: vec![IrcSide {
                name: "backward".into(),
                arc_lengths: vec![0.1, 0.2],
                energies: vec![-0.4, -0.5],
                end: "converged".into(),
                transverse: Vec::new(),
            }],
            file: Some("paths/x.xyz".into()),
            possible_bifurcations: vec![crate::workflow::mapper::PossibleBifurcation {
                side: "backward".into(),
                after: Some(0.1),
                at: 0.2,
                wavenumber: -120.5,
                confirmed_ends: Vec::new(),
                inflection: None,
            }],
        },
        job: Some(0),
    })
    .unwrap();
    net.add_crossing_edge(CrossingEdge {
        irc: crate::workflow::mapper::IrcPath::default(),
        id: 0,
        level: Level::Exploration,
        crossing: 2,
        ends: [Some(0), None],
        barriers: [Some(0.01), None],
        gradient_difference: Some(0.02),
        gap: Some(-2.5e-5),
        seam: Some(crate::workflow::mapper::SeamCheck::Checked(1)),
        job: None,
    })
    .unwrap();
    net.add_crossing_edge(CrossingEdge {
        irc: crate::workflow::mapper::IrcPath::default(),
        id: 0,
        level: Level::Exploration,
        crossing: 2,
        ends: [None, Some(0)],
        barriers: [None, Some(0.01)],
        gradient_difference: None,
        gap: None,
        seam: Some(crate::workflow::mapper::SeamCheck::NotChecked(
            "switched off, crossing_seam_check=off".into(),
        )),
        job: None,
    })
    .unwrap();
    net.jobs.push(JobRecord {
        id: 0,
        node: 0,
        method: "afir_ts".into(),
        task: "afir:+:1-2".into(),
        description: "AFIR".into(),
        status: JobStatus::Failed("no".into()),
        new_nodes: vec![1],
        new_edges: vec![0],
        found_new: true,
        found_lower: false,
        trajectory: None,
    });
    net.warnings.push(NetworkWarning {
        job: Some(0),
        message: "a warning".into(),
    });
    net.evaluations.gradients = 7;
    let doc = to_json(&net, JsonValue::Null);
    let text = doc.to_string_pretty();
    let (back, _) = from_json(&JsonValue::parse(&text).unwrap()).unwrap();
    assert_eq!(to_json(&back, JsonValue::Null).to_string_pretty(), text);
    let x = back.nodes[0].system.flat();
    let y = net.nodes[0].system.flat();
    assert!(x
        .iter()
        .zip(y.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!(back.nodes[2].kind, net.nodes[2].kind);
    assert_eq!(back.jobs, net.jobs);
    // The gap and the check within the seam of the crossing edges come back as they were.
    assert_eq!(back.crossing_edges, net.crossing_edges);
    // Another layout version is refused with what to do.
    let other = text.replace("\"version\": 1", "\"version\": 99");
    let e = from_json(&JsonValue::parse(&other).unwrap()).unwrap_err();
    assert!(e.to_string().contains("version 99"), "{e}");
}
