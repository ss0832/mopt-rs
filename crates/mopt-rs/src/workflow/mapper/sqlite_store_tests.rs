//! Tests of the SQLite store of the mapper: a network goes into the database and comes back
//! the same, and a database that lacks a table is read as far as it goes.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;
use crate::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
use crate::workflow::mapper::network::{
    ElectronicState, EvaluationTotals, InputRecord, IrcPath, IrcSide, JobRecord, JobStatus, Level,
    NetworkWarning, Node, NodeKind, OpenTsEdge, Origin, TsEdge,
};

fn network() -> Network {
    let argon = Element::from_atomic_number(18).unwrap();
    let mut net = Network::new(vec![argon; 3]);
    let system = |x: f64| {
        MolecularSystem::new(
            vec![argon; 3],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [x, 3.3, 0.1]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    };
    let node = |kind: NodeKind, x: f64, e: f64, job: Option<usize>| {
        let s = system(x);
        let state = ElectronicState::of(&s, 0);
        let mut n = Node::new(
            kind,
            Level::Exploration,
            s.into(),
            e,
            state,
            Origin {
                input: Some(0),
                job,
                from_node: job.map(|_| 0),
            },
        );
        n.frequencies = Some(vec![12.5, 30.25, 44.0]);
        n
    };
    net.inputs.push(InputRecord {
        file: "ar3.xyz".to_string(),
        frame: 0,
        node: Some(0),
        goal: false,
    });
    net.add_node(node(NodeKind::Minimum, 1.9, -0.0123456789012, None))
        .unwrap();
    net.add_node(node(NodeKind::Minimum, 2.7, -0.0101, Some(0)))
        .unwrap();
    let ts = net
        .add_node(node(NodeKind::TransitionState, 2.3, -0.0091, Some(0)))
        .unwrap();
    net.nodes[1].excluded = Some("its bonds are \"other\"".to_string());
    net.add_ts_edge(TsEdge {
        id: 0,
        level: Level::Exploration,
        ts,
        ends: [0, 1],
        barriers: [0.0032456789012, 0.001],
        imaginary_wavenumber: Some(-41.5),
        irc: IrcPath {
            sides: vec![
                IrcSide {
                    name: "backward".to_string(),
                    arc_lengths: vec![0.1, 0.35, 1.0],
                    energies: vec![-0.0092, -0.01, -0.0123],
                    end: "converged".to_string(),
                    transverse: Vec::new(),
                },
                IrcSide {
                    name: "forward".to_string(),
                    arc_lengths: vec![0.1, 0.8],
                    energies: vec![-0.00925, -0.0101],
                    end: "max_steps".to_string(),
                    transverse: Vec::new(),
                },
            ],
            file: Some("paths/TS000000_irc.xyz".to_string()),
            possible_bifurcations: vec![crate::workflow::mapper::PossibleBifurcation {
                side: "forward".to_string(),
                after: None,
                at: 0.8,
                wavenumber: -75.25,
                confirmed_ends: Vec::new(),
                inflection: None,
            }],
        },
        job: Some(0),
    })
    .unwrap();
    net.jobs.push(JobRecord {
        id: 0,
        node: 0,
        method: "afir_ts".to_string(),
        task: "afir:+2:1-3".to_string(),
        description: "AFIR push of atoms 1 and 3".to_string(),
        status: JobStatus::Done,
        new_nodes: vec![1, 2],
        new_edges: vec![0],
        found_new: true,
        found_lower: false,
        trajectory: Some("jobs/job_000000_traj.xyz".to_string()),
    });
    net.jobs.push(JobRecord {
        id: 1,
        node: 1,
        method: "afir_ts".to_string(),
        task: "afir:+2:2-3".to_string(),
        description: "AFIR push of atoms 2 and 3".to_string(),
        status: JobStatus::Failed("the search did not converge".to_string()),
        new_nodes: vec![],
        new_edges: vec![],
        found_new: false,
        found_lower: false,
        trajectory: None,
    });
    // A dissociation channel of the two fragments of the first minimum, and its edge.
    let atom = |z: f64| {
        MolecularSystem::new(
            vec![argon],
            &positions::<Angstrom>(&[[0.0, 0.0, z]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    };
    let pair = MolecularSystem::new(
        vec![argon; 2],
        &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    let f = net.add_fragment(crate::workflow::mapper::FragmentRecord {
        id: 0,
        key: 0x9e37_79b9_7f4a_7c15,
        formula: "Ar".to_string(),
        charge: 0,
        multiplicity: 1,
        level: Level::Exploration,
        system: atom(0.1),
        energy: -0.0003,
        frequencies: Some(Vec::new()),
    });
    let g = net.add_fragment(crate::workflow::mapper::FragmentRecord {
        id: 0,
        key: 7,
        formula: "Ar2".to_string(),
        charge: 0,
        multiplicity: 1,
        level: Level::Exploration,
        system: pair,
        energy: -0.0045,
        frequencies: Some(vec![30.5]),
    });
    let mut dc = node(NodeKind::Dissociated, 6.0, -0.0048, Some(0));
    dc.frequencies = None;
    dc.parts = vec![
        crate::workflow::mapper::ChannelPart {
            atoms: vec![0, 1],
            fragment: g,
        },
        crate::workflow::mapper::ChannelPart {
            atoms: vec![2],
            fragment: f,
        },
    ];
    let dc = net.add_node(dc).unwrap();
    net.add_dissociation_edge(crate::workflow::mapper::DissociationEdge {
        id: 0,
        level: Level::Exploration,
        minimum: 0,
        channel: dc,
        energy_difference: -0.0048 + 0.0123456789012,
        job: None,
        // The scan of confirm_dissociations: a curve with a hump, kept in the entry.
        scan: Some(crate::workflow::mapper::DissociationScan {
            distances: vec![5.0, 5.5, 6.0, 6.5],
            energies: vec![-0.0171, -0.0102, -0.0121, -0.0118],
            highest: Some(0.0015),
            barrier: true,
            separated: true,
        }),
    })
    .unwrap();
    net.warnings.push(NetworkWarning {
        job: Some(1),
        message: "the transition-state search did not converge".to_string(),
    });
    net.evaluations = EvaluationTotals {
        energies: 3,
        gradients: 120,
        hessians: 7,
        ..EvaluationTotals::default()
    };
    net
}

#[test]
fn a_network_goes_into_the_database_and_comes_back_the_same() {
    let dir = std::env::temp_dir().join(format!("mopt_mapper_sqlite_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let net = network();
    let mut run = BTreeMap::new();
    run.insert(
        "calculator".to_string(),
        JsonValue::String("lj".to_string()),
    );
    run.insert("interrupted".to_string(), JsonValue::Bool(false));
    let run = JsonValue::Object(run);
    SqliteStore.write(&net, &run, &dir).unwrap();
    // A copy for checks with SQLite itself (tests/test_sqlite_files.py).
    let _ = std::fs::copy(
        dir.join(SQLITE_FILE),
        std::env::temp_dir().join("mopt_mapper_unit.sqlite"),
    );
    let (back, back_run, notes) = read_sqlite(&dir.join(SQLITE_FILE)).unwrap();
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(back_run, run);
    assert_eq!(
        to_json(&back, run.clone()).to_string_pretty(),
        to_json(&net, run.clone()).to_string_pretty()
    );
    // The columns that can be queried.
    let db = Database::open(std::fs::read(dir.join(SQLITE_FILE)).unwrap()).unwrap();
    let nodes = db.rows("nodes").unwrap().unwrap();
    assert_eq!(nodes.len(), 4);
    assert_eq!(nodes[3].1[3], Value::Text("DC0".to_string()));
    assert_eq!(back.fragments.len(), 2);
    assert_eq!(back.fragments[0].key, 0x9e37_79b9_7f4a_7c15);
    assert_eq!(back.nodes[3].parts, net.nodes[3].parts);
    assert_eq!(db.rows("fragments").unwrap().unwrap().len(), 2);
    assert_eq!(db.rows("dissociation_edges").unwrap().unwrap().len(), 1);
    assert_eq!(nodes[2].1[3], Value::Text("TS0".to_string()));
    assert_eq!(nodes[0].1[5], Value::Real(-0.0123456789012));
    assert_eq!(nodes[0].1[14].as_blob().map(<[u8]>::len), Some(9 * 8));
    let points = db.rows("irc_points").unwrap().unwrap();
    assert_eq!(points.len(), 5);
    // A database without some tables is read as far as it goes.
    let some: Vec<_> = tables(&net, &run)
        .into_iter()
        .filter(|t| t.name != "jobs" && t.name != "irc_points")
        .collect();
    std::fs::write(
        dir.join("part.sqlite"),
        crate::core_infra::sqlite::write_database(&some).unwrap(),
    )
    .unwrap();
    let (part, _, notes) = read_sqlite(&dir.join("part.sqlite")).unwrap();
    assert_eq!(part.nodes.len(), 4);
    assert!(part.jobs.is_empty());
    assert!(notes.iter().any(|n| n.contains("'jobs'")), "{notes:?}");
    assert!(read_sqlite(&dir.join("none.sqlite")).is_err());
    std::fs::write(dir.join("text.sqlite"), "not a database").unwrap();
    assert!(read_sqlite(&dir.join("text.sqlite")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_transition_state_with_an_unknown_end_goes_into_the_database_and_back() {
    let dir = std::env::temp_dir().join(format!("mopt_mapper_sqlite_open_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut net = network();
    let mut ts = net.nodes[2].clone();
    ts.energy = -0.0085;
    let ts = net.add_node(ts).unwrap();
    net.add_open_ts_edge(OpenTsEdge {
        id: 0,
        level: Level::Exploration,
        ts,
        ends: [None, Some(1)],
        barriers: [None, Some(0.0016)],
        imaginary_wavenumber: Some(-20.5),
        irc: net.ts_edges[0].irc.clone(),
        job: Some(0),
    })
    .unwrap();
    let run = JsonValue::Null;
    SqliteStore.write(&net, &run, &dir).unwrap();
    let (back, _, notes) = read_sqlite(&dir.join(SQLITE_FILE)).unwrap();
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(back.open_ts_edges, net.open_ts_edges);
    assert_eq!(
        to_json(&back, run.clone()).to_string_pretty(),
        to_json(&net, run).to_string_pretty()
    );
    let db = Database::open(std::fs::read(dir.join(SQLITE_FILE)).unwrap()).unwrap();
    let rows = db.rows("open_ts_edges").unwrap().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].1[2], Value::Null);
    let _ = std::fs::remove_dir_all(&dir);
}
