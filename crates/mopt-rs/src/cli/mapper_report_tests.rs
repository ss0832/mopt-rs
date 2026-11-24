//! Tests of the report of `run_mapper`: the view of a small network (what is shown, the
//! populations, the lowest-barrier heights, the parts drawn without scripts), the page and its
//! data, the page written again from a run directory, and the landmark map of a larger network.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;
use crate::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge, Units};
use crate::workflow::mapper::network::{
    ElectronicState, IrcPath, IrcSide, Level, NodeKind, Origin, TsEdge,
};
use crate::workflow::mapper::network_json::to_json;

fn triangle(x: f64) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::from_atomic_number(18).unwrap(); 3],
        &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [x, 3.3, 0.1 * x]]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn node(kind: NodeKind, x: f64, kcal: f64) -> Node {
    let sys = triangle(x);
    let state = ElectronicState::of(&sys, 0);
    Node::new(
        kind,
        Level::Exploration,
        sys.into(),
        kcal * Units::KCAL_MOL_TO_HARTREE,
        state,
        Origin::default(),
    )
}

/// Minima 0 (0), 1 (2), 2 (5), 3 (30, joined to nothing); transition states 4 (between 0 and 1
/// at 10, with an IRC), 5 (1-2 at 8), 6 (0-2 at 20).
fn network() -> Network {
    let mut net = Network::new(vec![Element::from_atomic_number(18).unwrap(); 3]);
    for (x, e) in [(1.9, 0.0), (2.6, 2.0), (3.4, 5.0), (5.0, 30.0)] {
        net.add_node(node(NodeKind::Minimum, x, e)).unwrap();
    }
    let h = Units::KCAL_MOL_TO_HARTREE;
    for (x, e, a, b) in [(2.2, 10.0, 0, 1), (3.0, 8.0, 1, 2), (2.7, 20.0, 0, 2)] {
        let ts = net.add_node(node(NodeKind::TransitionState, x, e)).unwrap();
        let irc = if ts == 4 {
            IrcPath {
                sides: vec![
                    IrcSide {
                        name: "backward".to_string(),
                        arc_lengths: vec![0.1, 0.5, 1.0],
                        energies: vec![9.0 * h, 4.0 * h, 0.5 * h],
                        end: "converged".to_string(),
                        transverse: Vec::new(),
                    },
                    IrcSide {
                        name: "forward".to_string(),
                        arc_lengths: vec![0.1, 0.6],
                        energies: vec![9.5 * h, 2.5 * h],
                        end: "converged".to_string(),
                        transverse: Vec::new(),
                    },
                ],
                file: Some("paths/TS000004_irc.xyz".to_string()),
                possible_bifurcations: Vec::new(),
            }
        } else {
            IrcPath::default()
        };
        net.add_ts_edge(TsEdge {
            id: 0,
            level: Level::Exploration,
            ts,
            ends: [a, b],
            barriers: [(e - [0.0, 2.0, 5.0][a]) * h, (e - [0.0, 2.0, 5.0][b]) * h],
            imaginary_wavenumber: Some(-50.0),
            irc,
            job: None,
        })
        .unwrap();
    }
    net
}

#[test]
fn the_view_shows_the_minima_their_links_and_the_lowest_barriers() {
    let net = network();
    let view = View::new(&net, ReportOptions::default());
    assert_eq!(view.minima, vec![0, 1, 2, 3]);
    assert_eq!(view.saddles, vec![5, 4, 6]);
    assert_eq!(view.hidden, [0, 0, 0]);
    // The tree: 1-2 (8) and 0-1 (10); from 0, 2 is reached over 10 (not the 20 of 0-2).
    let reach: Vec<Option<f64>> = view.reach.iter().map(|r| r.map(|e| view.rel(e))).collect();
    assert!((reach[1].unwrap() - 10.0).abs() < 1e-9);
    assert!((reach[2].unwrap() - 10.0).abs() < 1e-9);
    assert_eq!(reach[3], None);
    // Populations of the energies, summing to one, the lowest the largest.
    let total: f64 = view.populations.iter().sum();
    assert!((total - 1.0).abs() < 1e-12 && view.populations[0] > view.populations[1]);
    // The IRC of TS4 from end 0 (negative arc lengths) through the TS (0) to end 1.
    let link = *view.link_of(4).unwrap();
    let p = view.profile(&link);
    assert_eq!(p.first().map(|q| q.0), Some(-1.0));
    assert!(p.contains(&(0.0, view.rel_of(4))));
    assert_eq!(view.profiles.len(), 1);
    // The graph has a place for every node shown; the map too (exact scaling).
    assert_eq!(view.layout.len(), 7);
    let map = view.map.as_ref().unwrap();
    assert_eq!(map.places.len(), 7);
    // Shares of the spread, largest first, at most one in all.
    assert!(map.shares.windows(2).all(|w| w[0] >= w[1] - 1e-12));
    assert!(map.shares.iter().sum::<f64>() <= 1.0 + 1e-9);
    // A window of 6 kcal/mol leaves out minimum 3 and every transition state.
    let window = ReportOptions::parse(&["energy_window=6".to_string()]).unwrap();
    let part = View::new(&net, window);
    assert_eq!((part.minima.len(), part.links.len()), (3, 0));
    assert_eq!(part.hidden, [1, 3, 0]);
    // Two minima drawn: the disconnectivity graph has two leaves, the merge of 0 and 1.
    let two = ReportOptions::parse(&["figure_minima=2".to_string()]).unwrap();
    let view = View::new(&net, two);
    let leaves = view
        .drawn
        .basins
        .iter()
        .filter(|b| b.minimum.is_some())
        .count();
    assert_eq!(leaves, 2);
    let (minima, links, saddles) = view.figure_part();
    assert_eq!((minima, saddles), (vec![0, 1], vec![4]));
    assert_eq!(links.len(), 1);
}

#[test]
fn the_page_holds_its_figures_tables_and_data() {
    let net = network();
    let view = View::new(&net, ReportOptions::default());
    let data = JsonValue::parse(&view.data()).unwrap();
    assert_eq!(data.get("minima").and_then(JsonValue::as_usize), Some(4));
    let ids = data.get("id").and_then(JsonValue::as_array).unwrap();
    assert_eq!(ids.len(), 7);
    let tree = data
        .get("edges")
        .and_then(|e| e.get("tree"))
        .and_then(JsonValue::as_array)
        .unwrap();
    let on: usize = tree.iter().filter_map(JsonValue::as_usize).sum();
    assert_eq!(on, 2);
    // An IRC path of the run directory is put in the page.
    let dir = std::env::temp_dir().join(format!("mopt_mapper_irc_{}", std::process::id()));
    std::fs::create_dir_all(dir.join("paths")).unwrap();
    std::fs::write(
        dir.join("paths/TS000004_irc.xyz"),
        "3\nback\nAr 0 0 0\nAr 3.8 0 0\nAr 1.9 3.3 0\n3\nts\nAr 0 0 0\nAr 3.8 0 0\nAr 2.2 3.3 0\n",
    )
    .unwrap();
    let mut view = view;
    view.load_ircs(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(view.ircs.len(), 1);
    assert!(view
        .data()
        .contains("\"irc_xyz\":{\"1\":\"3\\nback\\nAr 0.00000"));
    // Beyond the lines that the page holds as text, a path is packed: it is still in the page.
    let mut packed = View::new(
        &net,
        ReportOptions::parse(&["irc_lines=0".to_string()]).unwrap(),
    );
    std::fs::create_dir_all(dir.join("paths")).unwrap();
    std::fs::write(
        dir.join("paths/TS000004_irc.xyz"),
        "3\nback\nAr 0 0 0\nAr 3.8 0 0\nAr 1.9 3.3 0\n3\nts\nAr 0 0 0\nAr 3.8 0 0\nAr 2.2 3.3 0\n",
    )
    .unwrap();
    packed.load_ircs(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(packed.ircs.is_empty());
    assert_eq!(packed.packed_ircs.len(), 1);
    assert!(packed.data().contains("\"irc_z\":{\"1\":\""));
    // What is packed are the frames written compactly: they come back as the thinned text.
    let packed_text = packed.packed_ircs.values().next().unwrap();
    assert!(!packed_text.is_empty() && !packed_text.contains(' '));
    // The structures beyond those the page has as text are packed too, every one by default,
    // so that any node can be shown and saved; with no room for them they are left out.
    let few = View::new(
        &net,
        ReportOptions::parse(&["structures=2".to_string()]).unwrap(),
    );
    assert_eq!(few.structures.len(), 2);
    assert_eq!(
        few.structures.len() + few.packed_structures.len(),
        few.minima.len() + few.saddles.len()
    );
    assert!(few.data().contains("\"model_z\":{\""));
    assert!((0..net.nodes.len()).all(|id| few.has_structure(id)));
    let none_packed = View::new(
        &net,
        ReportOptions::parse(&[
            "structures=2".to_string(),
            "structure_packed_atoms=0".to_string(),
        ])
        .unwrap(),
    );
    assert!(none_packed.packed_structures.is_empty());
    assert!(!none_packed.data().contains("model_z"));
    // With no room for packed paths either, it is left out.
    let mut none = View::new(
        &net,
        ReportOptions::parse(&["irc_lines=0".to_string(), "irc_packed_lines=0".to_string()])
            .unwrap(),
    );
    none.load_ircs(&dir);
    assert!(none.ircs.is_empty() && none.packed_ircs.is_empty());
    let report = MapperReport::new(
        Path::new("."),
        "test",
        "lj",
        "settings",
        ReportOptions::default(),
    );
    let html = page(&report, &view, "The run ended.", true, &[], None);
    assert!(!html.contains("Missing information"));
    for part in [
        "data-view=\"map\"",
        "data-view=\"graph\"",
        "data-view=\"dg\"",
        "data-view=\"diagram\"",
        "data-node=\"4\"",
        "id=\"network-data\"",
        "window.moptViewer",
        "data-get=\"node:0\"",
        "data-get=\"irc:1\"",
        "data-view-choice=\"circle\"",
        "data-canvas=\"circle\"",
        "id=\"path-save\"",
        "id=\"path-play\"",
        "data-z=\"full\"",
        // Over the whole screen: the figures with the panel of the chosen node.
        ".explore.is-full #details",
    ] {
        assert!(html.contains(part), "{part}");
    }
    // The page can be moved on its own: what it holds is saved from it, and nothing in it
    // links to a file of the run directory.
    for link in ["href=\"structures/", "href=\"paths/", "href=\"jobs/"] {
        assert!(!html.contains(link), "{link}");
    }
    assert!(!html.contains("NaN") && !html.contains("=\"inf") && !html.contains("=\"-inf"));
    // The chemical species: the four minima are arrangements of three argon atoms that are
    // not bonded, one species of four conformers, and no transition state joins two species.
    let species = view.species();
    assert_eq!(species.len(), 1);
    assert_eq!(species[0].smiles, "[Ar].[Ar].[Ar]");
    assert_eq!(species[0].members, vec![0, 1, 2, 3]);
    for part in [
        "id=\"species\"",
        "<code>[Ar].[Ar].[Ar]</code>",
        "No transition state shown joins two species.",
        "1 species among the 4 minima shown; 3 transition state(s) shown join conformers",
        "<a href=\"#species\">Species</a>",
    ] {
        assert!(html.contains(part), "{part}");
    }
    assert_eq!(
        data.get("species")
            .and_then(JsonValue::as_array)
            .map(|a| a.len()),
        Some(1)
    );
    // Nothing in the data ends the script element it is in.
    let start = html.find("id=\"network-data\">").unwrap();
    let end = start + html[start..].find("</script>").unwrap();
    assert!(!html[start..end].contains("</"));
}

#[test]
fn the_page_is_written_again_from_a_run_directory() {
    let dir = std::env::temp_dir().join(format!("mopt_mapper_report_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let net = network();
    std::fs::write(
        dir.join(NETWORK_FILE),
        to_json(&net, JsonValue::empty_object()).to_string_pretty(),
    )
    .unwrap();
    std::fs::write(
        dir.join(RUN_SETTINGS_FILE),
        "{\"report_options\": [\"unit=kj\", \"temperature=500\"]}",
    )
    .unwrap();
    // The settings of the run, then those of the command line (a later key wins); a setting
    // that cannot be read is a note, not a failure.
    let (file, notes) = rewrite(
        &dir,
        &[
            "temperature=400".to_string(),
            "file=again.html".to_string(),
            "colour=red".to_string(),
        ],
    )
    .unwrap();
    assert_eq!(file, dir.join("again.html"));
    assert!(notes.iter().any(|n| n.contains("colour")), "{notes:?}");
    // The IRC path of TS0 is named but not in the directory; the calculator is not recorded.
    assert!(
        notes.iter().any(|n| n.contains("IRC path file")),
        "{notes:?}"
    );
    assert!(notes.iter().any(|n| n.contains("calculator")), "{notes:?}");
    let html = std::fs::read_to_string(&file).unwrap();
    assert!(html.contains("kJ/mol") && html.contains("at 400 K"));
    assert!(html.contains("had not ended") && html.contains("Missing information"));
    // Only what is not a directory fails.
    assert!(rewrite(&dir.join("missing"), &[]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_page_is_written_from_the_database_of_a_run() {
    use crate::workflow::mapper::sqlite_store::{SqliteStore, SQLITE_FILE};
    use crate::workflow::mapper::store::NetworkStore;
    let dir = std::env::temp_dir().join(format!("mopt_mapper_report_db_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let net = network();
    SqliteStore.write(&net, &JsonValue::Null, &dir).unwrap();
    let (file, notes) = rewrite(&dir.join(SQLITE_FILE), &["file=db.html".to_string()]).unwrap();
    assert_eq!(file, dir.join("db.html"));
    let html = std::fs::read_to_string(&file).unwrap();
    for part in ["data-view=\"graph\"", "EQ3", "TS0"] {
        assert!(html.contains(part), "{part}");
    }
    // What the database lacks is a note: the settings file and the IRC file are not there.
    assert!(
        notes.iter().any(|n| n.contains("settings.json")),
        "{notes:?}"
    );
    // A file that is not a database is a note too, and the structures beside it are shown.
    std::fs::write(dir.join("broken.sqlite"), "not a database").unwrap();
    let (_, notes) = rewrite(&dir.join("broken.sqlite"), &[]).unwrap();
    assert!(
        notes.iter().any(|n| n.contains("could not be read")),
        "{notes:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_damaged_run_directory_still_gives_a_page() {
    use crate::workflow::mapper::store::{network_from_files, FileStore, NetworkStore};
    let dir = std::env::temp_dir().join(format!("mopt_mapper_damaged_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let net = network();
    // The files of a run: structures, tables, network file.
    FileStore.write(&net, &JsonValue::Null, &dir).unwrap();
    // A node that cannot be read (EQ2) and a layout of another version: the rest is shown,
    // without the two transition states of EQ2.
    let mut doc = to_json(&net, JsonValue::empty_object());
    let m = doc.as_object_mut().unwrap();
    m.insert("version".to_string(), JsonValue::Number(99.0));
    if let Some(JsonValue::Array(nodes)) = m.get_mut("nodes") {
        if let Some(JsonValue::Object(n)) = nodes.get_mut(2) {
            n.remove("coords_bohr");
        }
    }
    std::fs::write(dir.join(NETWORK_FILE), doc.to_string_pretty()).unwrap();
    let (file, notes) = rewrite(&dir, &[]).unwrap();
    assert!(notes.iter().any(|n| n.contains("node 2")), "{notes:?}");
    assert!(
        notes.iter().any(|n| n.contains("layout version 99")),
        "{notes:?}"
    );
    let left_out = notes
        .iter()
        .filter(|n| n.contains("transition-state edge"))
        .count();
    assert_eq!(left_out, 2, "{notes:?}");
    let html = std::fs::read_to_string(&file).unwrap();
    assert!(html.contains("Missing information") && html.contains("TS0") && html.contains("EQ3"));
    // A network file that is not JSON: the structures and ts_edges.csv of the run.
    std::fs::write(dir.join(NETWORK_FILE), "{ not json").unwrap();
    let (_, notes) = rewrite(&dir, &[]).unwrap();
    assert!(
        notes.iter().any(|n| n.contains("could not be read")),
        "{notes:?}"
    );
    let (from_files, file_notes) = network_from_files(&dir);
    assert!(file_notes.is_empty(), "{file_notes:?}");
    assert_eq!((from_files.nodes.len(), from_files.ts_edges.len()), (7, 3));
    assert_eq!(from_files.nodes[4].label(), "TS0");
    // An empty directory: a page that says what is missing.
    let empty = dir.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let (page_file, notes) = rewrite(&empty, &[]).unwrap();
    assert!(page_file.is_file());
    assert!(
        notes.iter().any(|n| n.contains("No structure")),
        "{notes:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_saddle_point_of_higher_order_is_shown_in_the_panel() {
    // A saddle point of order 2 above the network: one of its modes descends to two transition
    // states, the other to a transition state and to a descent that did not settle.
    let mut net = network();
    let mut saddle = node(NodeKind::HigherSaddle { order: 2 }, 2.4, 35.0);
    saddle.frequencies = Some(vec![30.0]);
    saddle.index_wavenumbers = Some(vec![-412.5, -98.3]);
    let sp = net.add_node(saddle).unwrap();
    for (mode, ends) in [(0, [Some(4), Some(5)]), (1, [Some(6), None])] {
        net.add_saddle_edge(crate::workflow::mapper::network::SaddleEdge {
            id: 0,
            level: Level::Exploration,
            saddle: sp,
            mode,
            wavenumber: None,
            ends,
            job: None,
        })
        .unwrap();
    }
    let view = View::new(&net, ReportOptions::default());
    let marks = view.higher_saddles();
    assert_eq!(marks.len(), 1);
    assert_eq!(marks[0].order, 2);
    assert_eq!(
        marks[0].descents,
        vec![(0, [Some(4), Some(5)]), (1, [Some(6), None])]
    );
    // Its structure is in the page, as those of the nodes of the figures.
    assert!(view.structures.contains_key(&sp));
    let data = view.data();
    for part in [
        "\"order\":[2]",
        "\"job\":[-1]",
        "\"desc\":[[[0,4,5],[1,6,-1]]]",
        "\"names\":{\"4\":\"TS",
        "\"imag\":[[-412.5,-98.3]]",
    ] {
        assert!(data.contains(part), "{part} in {data}");
    }
    let report = MapperReport::new(
        Path::new("."),
        "test",
        "lj",
        "settings",
        ReportOptions::default(),
    );
    let html = page(&report, &view, "The run ended.", true, &[], None);
    // The row of its table saves its structure and opens its panel.
    assert!(html.contains(&format!("<tr data-node=\"{sp}\" tabindex=\"0\">")));
    assert!(html.contains(&format!("data-get=\"node:{sp}\"")));
    assert!(html.contains("function higher(hk, jump)"));
}

#[test]
fn a_large_network_is_mapped_by_landmarks() {
    // 60 minima in a chain joined by 59 transition states; landmark scaling above 20.
    let mut net = Network::new(vec![Element::from_atomic_number(18).unwrap(); 3]);
    for k in 0..60 {
        net.add_node(node(
            NodeKind::Minimum,
            1.0 + 0.05 * k as f64,
            0.1 * k as f64,
        ))
        .unwrap();
    }
    let h = Units::KCAL_MOL_TO_HARTREE;
    for k in 0..59 {
        let ts = net
            .add_node(node(
                NodeKind::TransitionState,
                1.025 + 0.05 * k as f64,
                0.1 * k as f64 + 1.0,
            ))
            .unwrap();
        net.add_ts_edge(TsEdge {
            id: 0,
            level: Level::Exploration,
            ts,
            ends: [k, k + 1],
            barriers: [1.0 * h, 0.9 * h],
            imaginary_wavenumber: None,
            irc: IrcPath::default(),
            job: None,
        })
        .unwrap();
    }
    let options =
        ReportOptions::parse(&["map_exact=20".to_string(), "map_landmarks=10".to_string()])
            .unwrap();
    let view = View::new(&net, options);
    assert!(view.map_method.starts_with("landmark"));
    let map = view.map.as_ref().unwrap().plane(0, 1);
    assert_eq!(map.len(), 119);
    // The chain is a line: its ends are the farthest apart on the map.
    let d = |a: usize, b: usize| {
        ((map[a][0] - map[b][0]).powi(2) + (map[a][1] - map[b][1]).powi(2)).sqrt()
    };
    assert!(d(0, 59) > d(0, 30) && d(0, 30) > d(0, 5));
    // The lowest-barrier height of the far end: the highest transition state on the way.
    let last = view.reach[59].map(|e| view.rel(e)).unwrap();
    assert!((last - (0.1 * 58.0 + 1.0)).abs() < 1e-9);
}

#[test]
fn the_species_are_drawn_as_a_graph() {
    // Water, its conformer with another angle, and hydrogen beside an oxygen atom: two
    // species (three minima), joined by one transition state; another transition state joins
    // the two conformers of water.
    let elements = vec![
        Element::parse_token("O").unwrap(),
        Element::parse_token("H").unwrap(),
        Element::parse_token("H").unwrap(),
    ];
    let structure = |xyz: [[f64; 3]; 3]| -> crate::workflow::mapper::Configuration {
        MolecularSystem::new(
            elements.clone(),
            &positions::<Angstrom>(&xyz),
            SpinCharge::default(),
            None,
        )
        .unwrap()
        .into()
    };
    let water = |half_angle: f64| {
        let (s, c) = (0.96 * half_angle.sin(), 0.96 * half_angle.cos());
        structure([[0.0, 0.0, 0.0], [s, c, 0.0], [-s, c, 0.0]])
    };
    let apart = structure([[0.0, 0.0, 0.0], [3.0, 0.37, 0.0], [3.0, -0.37, 0.0]]);
    let between = structure([[0.0, 0.0, 0.0], [1.6, 0.45, 0.0], [1.6, -0.45, 0.0]]);
    let h = Units::KCAL_MOL_TO_HARTREE;
    let mut net = Network::new(elements.clone());
    let mut add = |kind: NodeKind, system: crate::workflow::mapper::Configuration, kcal: f64| {
        let state = ElectronicState {
            charge: 0,
            multiplicity: 1,
            excited_state: 0,
        };
        net.add_node(Node::new(
            kind,
            Level::Exploration,
            system,
            kcal * h,
            state,
            Origin::default(),
        ))
        .unwrap()
    };
    let a = add(NodeKind::Minimum, water(0.91), 0.0);
    let b = add(NodeKind::Minimum, water(0.80), 1.0);
    let c = add(NodeKind::Minimum, apart, 40.0);
    let ts_ab = add(NodeKind::TransitionState, water(0.86), 3.0);
    let ts_ac = add(NodeKind::TransitionState, between, 90.0);
    for (ts, ends, top) in [(ts_ab, [a, b], 3.0), (ts_ac, [a, c], 90.0)] {
        let low = |k: usize| [0.0, 1.0, 40.0][k];
        net.add_ts_edge(TsEdge {
            id: 0,
            level: Level::Exploration,
            ts,
            ends,
            barriers: [(top - low(ends[0])) * h, (top - low(ends[1])) * h],
            imaginary_wavenumber: Some(-500.0),
            irc: IrcPath::default(),
            job: None,
        })
        .unwrap();
    }
    let view = View::new(&net, ReportOptions::default());
    let species = view.species();
    assert_eq!(
        species
            .iter()
            .map(|s| (s.smiles.as_str(), s.members.len()))
            .collect::<Vec<_>>(),
        vec![("O", 2), ("[H][H].[O]", 1)]
    );
    let report = MapperReport::new(
        Path::new("."),
        "test",
        "lj",
        "settings",
        ReportOptions::default(),
    );
    let html = page(&report, &view, "The run ended.", true, &[], None);
    for part in [
        "Graph of the species",
        "class=\"species-graph\"",
        // A node is dragged, and the edge of its two species follows.
        "data-s=\"0\" tabindex=\"0\" class=\"species-node\"",
        "data-a=\"0\" data-b=\"1\" tabindex=\"0\" class=\"species-edge\"",
        "Drag a node to move it (its edges follow), drag elsewhere to move the view",
        "svg.querySelectorAll('g.species-edge')",
        "<title>S1: O (2 conformer(s), the lowest EQ0 at 0.00)</title>",
        "<title>S2: [H][H].[O] (1 conformer(s), the lowest EQ2 at 40.00)</title>",
        // The edge of the two species: their one transition state, at 90 kcal/mol.
        "<title>TS1: the lowest of 1 transition state(s) between S1 and S2</title>",
        "2 species among the 3 minima shown; 1 transition state(s) shown join conformers",
    ] {
        assert!(html.contains(part), "{part}");
    }
    assert_eq!(html.matches("class=\"species-node\"").count(), 2);
    assert_eq!(html.matches("class=\"species-edge\"").count(), 1);
    // The disconnectivity graph and the energy diagram of each species beside those of the
    // whole network: figures of the two conformers of water, a line for the species of one
    // minimum; and the two forms of the circle.
    assert_eq!(
        html.matches("<figure data-species-figure=\"-1\">").count(),
        2
    );
    assert_eq!(
        html.matches("<figure data-species-figure=\"0\" hidden>")
            .count(),
        2
    );
    assert_eq!(
        html.matches("<p class=\"note\" data-species-figure=\"1\" hidden>")
            .count(),
        2
    );
    assert!(html.contains("S1 · <code>O</code>: 2 conformer(s)"));
    // The two conformers of water 1e-7 kcal/mol apart: the axes of their figures span 0.001
    // kcal/mol, and no two labels of a tick are the same.
    let mut close = net.clone();
    if let Some(n) = close.nodes.get_mut(b) {
        n.energy = 1e-7 * Units::KCAL_MOL_TO_HARTREE;
    }
    let only: std::collections::HashSet<usize> = [a, b].into_iter().collect();
    let pair = View::of(&close, ReportOptions::default(), Some(&only));
    for svg in [disconnectivity_graph(&pair), energy_diagram(&pair)] {
        let labels: Vec<&str> = svg
            .split("class=\"tick\"")
            .skip(1)
            .filter_map(|t| t.split('>').nth(1)?.split('<').next())
            .collect();
        let distinct: std::collections::HashSet<&&str> = labels.iter().collect();
        assert!(
            labels.len() >= 2 && distinct.len() == labels.len(),
            "{labels:?}"
        );
    }
    assert!(html.contains("data-circle-mode=\"one\""));
    assert!(html.contains("data-circle-mode=\"species\""));
    // The view of one species has its minima and what joins them alone, with the energies
    // of the whole network.
    let only: std::collections::HashSet<usize> = [b, c].into_iter().collect();
    let part = View::of(&net, ReportOptions::default(), Some(&only));
    assert_eq!(part.minima, vec![b, c]);
    assert!(part.links.is_empty());
    assert_eq!(part.low, view.low);
}

#[test]
fn the_settings_on_the_page_have_no_directories() {
    let text = r#"{"INPUT": ["/data/runs/start.xyz", "rel/start.xyz"],
        "config_file": "C:/work/config.json",
        "mapper_options": ["goal=/data/runs/goal.xyz", "rng_seed=1", "stop=a/b"],
        "step1_settings": {"NSTEP": 300, "othersoft": "gfn0xtb"}}"#;
    let shown = settings_for_page(text);
    assert!(
        !shown.contains("/data") && !shown.contains("C:/work"),
        "{shown}"
    );
    for kept in [
        "\"start.xyz\"",
        "\"rel/start.xyz\"",
        "\"config.json\"",
        "\"goal=goal.xyz\"",
        "\"rng_seed=1\"",
        "\"stop=a/b\"",
        "300",
        "gfn0xtb",
    ] {
        assert!(shown.contains(kept), "{kept}: {shown}");
    }
    // Text that is not JSON is shown as it is.
    assert_eq!(settings_for_page("not json"), "not json");
}
