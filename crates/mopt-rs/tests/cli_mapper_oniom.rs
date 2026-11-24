//! `run_mapper` on an ONIOM surface with the built-in engines: vinyl alcohol (GFN1-xTB) with two
//! water molecules hydrogen-bonded to it (GFN-FF, relaxed by the microiterations), the AFIR push
//! between the CH2 carbon and the hydrogen of the hydroxy group. The first barrier on that path
//! is the rotation of the hydroxy group (syn to anti); its transition state and both ends of its
//! IRC converge with the water relaxed at every point, and the end on the side of the start is
//! the starting minimum again, at its energy. The network holds the cores; the files of the
//! nodes and the IRC paths hold the whole system; the network file keeps the whole structures;
//! the settings that are not taken on an ONIOM surface are refused by name.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::mapper::{run_with, MapperConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_MAPPER;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::mapper::{Configuration, NodeKind};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mapper_oniom_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Vinyl alcohol (atoms 1 to 7, syn; the hydrogen of the hydroxy group atom 4) and two water
/// molecules, one donating a hydrogen bond to its oxygen, one near the CH2 group (Angstrom).
const INPUT: &str = "13
0 1
C    0.0000    0.0000    0.0000
C    1.3300    0.0000    0.0000
O    2.0800    1.1400    0.0000
H    1.5500    1.9500    0.0000
H   -0.5400    0.9300    0.0000
H   -0.5400   -0.9300    0.0000
H    1.8800   -0.9300    0.0000
O    4.9500    1.3000    0.0000
H    3.9935    1.2467    0.0000
H    5.1379    2.2394    0.0000
O   -3.0000   -1.7000    1.3000
H   -2.2201   -1.2581    0.9620
H   -2.9106   -1.6493    2.2525
";

/// The dihedral angle of four points (degrees).
fn dihedral(p: [[f64; 3]; 4]) -> f64 {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let b0 = sub(p[0], p[1]);
    let b1 = sub(p[2], p[1]);
    let n = dot(b1, b1).sqrt();
    let b1 = [b1[0] / n, b1[1] / n, b1[2] / n];
    let b2 = sub(p[3], p[2]);
    let v = sub(b0, b1.map(|c| dot(b0, b1) * c));
    let w = sub(b2, b1.map(|c| dot(b2, b1) * c));
    dot(cross(b1, v), w).atan2(dot(v, w)).to_degrees()
}

/// The dihedral angle C1-C2-O3-H4 of a core structure (degrees).
fn hydroxy_dihedral(core: &mopt_rs::domain::MolecularSystem) -> f64 {
    let x = core.coords_flat_bohr();
    let at = |a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
    dihedral([at(0), at(1), at(2), at(3)])
}

fn settings(extra: &str) -> String {
    format!(
        r#"{{
  "mapper_settings": {{
    "afir_gamma_kJmol": 400.0,
    "dist_upper_ang": 4.0,
    "max_iterations": 2,
    "rng_seed": 1,
    "active_atoms": [4, 1]{extra}
  }},
  "top_n_candidates": 1,
  "step1_settings": {{"othersoft": "gfn1xtb", "NSTEP": 300, "oniom_flag": ["1-7", "auto", "gfnff"]}},
  "step3_settings": {{"othersoft": "gfn1xtb", "NSTEP": 200, "calc_exact_hess": 5, "oniom_flag": ["1-7", "auto", "gfnff"]}},
  "step4_settings": {{
    "othersoft": "gfn1xtb",
    "NSTEP": 400,
    "oniom_flag": ["1-7", "auto", "gfnff"],
    "intrinsic_reaction_coordinates": ["0.1", "300", "lqa"]
  }}
}}"#
    )
}

fn config(dir: &Path, extra: &str, args: &[&str]) -> MoptResultConfig {
    let input = dir.join("enol_water2.xyz");
    std::fs::write(&input, INPUT).unwrap();
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, settings(extra)).unwrap();
    let mut all = vec![
        input.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        "run".to_string(),
    ];
    all.extend(args.iter().map(|s| s.to_string()));
    let parsed = parser("run_mapper", RUN_MAPPER).parse(all).unwrap();
    MapperConfig::from_args(&parsed, None).map_err(|e| e.to_string())
}

type MoptResultConfig = Result<MapperConfig, String>;

/// The number of atoms of the first frame of an xyz file.
fn atoms_of(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn a_network_of_cores_with_the_whole_system_in_its_files() {
    let dir = scratch("enol");
    let mut c = config(&dir, "", &[]).unwrap();
    // Follow the unstable endpoint mode to the anti minimum for this two-ended reference.
    c.steps.minimum.settings.mode_kill = Some(Default::default());
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let net = &run.network;
    eprintln!(
        "vinyl alcohol with two waters (ONIOM): {} minima, {} transition states, {} with an unknown \
         end, {} jobs, {} warnings, {} gradients",
        net.minima().count(),
        net.ts_edges.len(),
        net.open_ts_edges.len(),
        net.jobs.len(),
        net.warnings.len(),
        net.evaluations.gradients
    );
    // The network holds the cores (7 atoms) and the elements of the whole system (13).
    assert_eq!(net.elements.len(), 7);
    assert_eq!(net.whole_elements.as_ref().map(Vec::len), Some(13));
    assert!(net.minima().count() >= 2, "{:?}", net.warnings);
    assert!(!net.ts_edges.is_empty(), "{:?}", net.warnings);
    // The transition state from the start is the rotation of the hydroxy group: the dihedral
    // angle is near 90 degrees there, and its IRC goes back to the start (syn) at the energy
    // of the start and on to the anti form.
    let start = &net.nodes[0];
    let edge = net
        .ts_edges
        .iter()
        .find(|e| e.ends.contains(&0))
        .unwrap_or_else(|| panic!("no transition state from the start: {:?}", net.warnings));
    let turn = hydroxy_dihedral(net.nodes[edge.ts].system.structure().unwrap()).abs();
    assert!((60.0..120.0).contains(&turn), "{turn}");
    let other = &net.nodes[if edge.ends[0] == 0 {
        edge.ends[1]
    } else {
        edge.ends[0]
    }];
    let ends: Vec<f64> = edge
        .ends
        .iter()
        .map(|&k| hydroxy_dihedral(net.nodes[k].system.structure().unwrap()).abs())
        .collect();
    assert!(
        ends.iter().any(|d| *d < 30.0) && ends.iter().any(|d| *d > 150.0),
        "{ends:?}"
    );
    assert!(
        other.energy > start.energy,
        "{} {}",
        other.energy,
        start.energy
    );
    // The IRC end on the side of the start relaxed its water as the start did.
    assert!(start.environments.is_empty(), "{:?}", start.environments);
    // Every minimum and transition state has its whole structure, and its file holds it.
    for n in net
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Minimum | NodeKind::TransitionState))
    {
        let whole = n
            .whole
            .as_ref()
            .and_then(Configuration::structure)
            .unwrap_or_else(|| panic!("{} has no whole structure", n.label()));
        assert_eq!(whole.n_atoms(), 13);
        // The core of the whole structure is the structure of the node.
        let core = n.system.structure().unwrap();
        for k in 0..7 {
            let (a, b) = (whole.coords_flat_bohr(), core.coords_flat_bohr());
            for d in 0..3 {
                assert!((a[3 * k + d] - b[3 * k + d]).abs() < 1e-9, "{}", n.label());
            }
        }
        let file = run
            .directory
            .join(mopt_rs::workflow::mapper::store::structure_file(n));
        assert_eq!(atoms_of(&file), 13, "{}", file.display());
    }
    // The IRC path files hold the whole system.
    for e in &net.ts_edges {
        let file = run.directory.join(e.irc.file.as_deref().unwrap());
        assert_eq!(atoms_of(&file), 13, "{}", file.display());
    }
    // The network file keeps the whole structures.
    let (back, _) = mopt_rs::workflow::mapper::store::read_network(&run.directory)
        .unwrap()
        .unwrap();
    assert_eq!(back.whole_elements, net.whole_elements);
    for (a, b) in back.nodes.iter().zip(&net.nodes) {
        assert_eq!(
            a.whole.as_ref().map(Configuration::flat).map(|x| x.nrows()),
            b.whole.as_ref().map(Configuration::flat).map(|x| x.nrows())
        );
    }
}

/// With a longer NEB the mapper reaches the rotation saddle of the hydroxy group again. Its
/// anti end is either minimized to the anti minimum (a TS edge between the two minima) or, when
/// the minimization from the end of the IRC stops on a shoulder with an imaginary mode, left
/// open: such an end never becomes a minimum, the transition state is kept with its syn end
/// only, and the warning says why. Which of the two happens is decided by digits far below any
/// tolerance: with the same input both occurred, the open end with the CODATA 2014 constants
/// and the anti minimum (EQ1, dihedral -177.8 degrees, 1.96 kcal/mol above the syn start, TS
/// at -90.8 degrees) with CODATA 2018, whose Bohr radius differs by 4e-10 of itself. Both
/// outcomes are checked in full; the open edge itself is checked on a model surface in
/// `workflow::mapper::driver_tests` (an unconverged end, ends with two imaginary modes).
#[test]
fn the_anti_end_of_the_longer_neb_is_a_verified_minimum_or_stays_open() {
    let dir = scratch("enol_open");
    let c = config(&dir, "", &["-map_opt", "method_relax_iterations=100"]).unwrap();
    let run = run_with(&c, &dir, None, &CalculatorRegistry::native()).unwrap();
    let (net, _) = mopt_rs::workflow::mapper::store::read_network(&run.directory)
        .unwrap()
        .unwrap();
    let dihedral_of = |k: usize| hydroxy_dihedral(net.nodes[k].system.structure().unwrap()).abs();
    // The start is the syn minimum; every minimum is syn or anti.
    assert!(dihedral_of(0) < 30.0, "{}", dihedral_of(0));
    for (k, n) in net.nodes.iter().enumerate() {
        if n.kind == NodeKind::Minimum {
            let d = dihedral_of(k);
            assert!(!(30.0..=150.0).contains(&d), "{}: {d}", n.label());
        }
    }
    // One transition state, the rotation, with the start at one end.
    assert_eq!(
        net.ts_edges.len() + net.open_ts_edges.len(),
        1,
        "{:?}",
        net.warnings
    );
    let irc_file = if let [edge] = &net.open_ts_edges[..] {
        assert_eq!(net.minima().count(), 1);
        assert_eq!(edge.ends, [Some(0), None]);
        assert!(edge.barriers[0].is_some_and(|b| b > 0.0));
        assert!(edge.barriers[1].is_none());
        assert!(net
            .warnings
            .iter()
            .any(|w| w.message.contains("TS–TS link could not be resolved")));
        let turn = dihedral_of(edge.ts);
        assert!((60.0..120.0).contains(&turn), "{turn}");
        edge.irc.file.clone()
    } else {
        let edge = &net.ts_edges[0];
        assert_eq!(net.minima().count(), 2);
        assert!(edge.ends.contains(&0), "{:?}", edge.ends);
        let other = if edge.ends[0] == 0 {
            edge.ends[1]
        } else {
            edge.ends[0]
        };
        assert_eq!(net.nodes[other].kind, NodeKind::Minimum);
        assert!(dihedral_of(other) > 150.0, "{}", dihedral_of(other));
        assert!(net.nodes[other].energy > net.nodes[0].energy);
        let turn = dihedral_of(edge.ts);
        assert!((60.0..120.0).contains(&turn), "{turn}");
        assert!(net.nodes[edge.ts].energy > net.nodes[other].energy);
        edge.irc.file.clone()
    };
    // The IRC path file holds the whole system.
    assert_eq!(
        atoms_of(&run.directory.join(irc_file.as_deref().unwrap())),
        13
    );
}

#[test]
fn what_an_oniom_network_does_not_take_is_refused_by_name() {
    let dir = scratch("refused");
    let registry = CalculatorRegistry::native();
    for (extra, args, said) in [
        (
            ", \"identity_scope\": \"all\"",
            &[][..],
            "identity_scope=all",
        ),
        (", \"method\": \"nanoreactor\"", &[][..], "nanoreactor"),
        (
            ", \"entrance_structures\": 3",
            &[][..],
            "entrance_structures",
        ),
        (", \"ts_conformers\": true", &[][..], "ts_conformers"),
    ] {
        let message = match config(&dir, extra, args) {
            Err(e) => e,
            Ok(c) => match run_with(&c, &dir, None, &registry) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("{extra} ran"),
            },
        };
        assert!(message.contains(said), "{extra}: {message}");
    }
    // identity_scope without ONIOM.
    let plain = settings(", \"identity_scope\": \"high\"")
        .replace(", \"oniom_flag\": [\"1-7\", \"auto\", \"gfnff\"]", "")
        .replace("\"oniom_flag\": [\"1-7\", \"auto\", \"gfnff\"],", "");
    assert!(!plain.contains("oniom_flag"), "{plain}");
    let cfg = dir.join("plain.json");
    std::fs::write(&cfg, plain).unwrap();
    let input = dir.join("enol_water2.xyz");
    let parsed = parser("run_mapper", RUN_MAPPER)
        .parse([
            input.display().to_string(),
            "-cfg".to_string(),
            cfg.display().to_string(),
            "--output_dir".to_string(),
            "plain".to_string(),
        ])
        .unwrap();
    let c = MapperConfig::from_args(&parsed, None).unwrap();
    let e = run_with(&c, &dir, None, &registry).unwrap_err().to_string();
    assert!(e.contains("identity_scope is a setting of"), "{e}");

    // -fix in the steps: atoms of the environment are held by the surface and leave the
    // settings of the steps; an atom of the core is refused. Without ONIOM, -fix is a hard
    // constraint of every step (`docs/source/mapper_design.md`).
    let with_fix = |atoms: &str| {
        settings("").replace(
            "\"oniom_flag\": [\"1-7\", \"auto\", \"gfnff\"]",
            &format!(
                "\"oniom_flag\": [\"1-7\", \"auto\", \"gfnff\"], \"fix_atoms\": [\"{atoms}\"]"
            ),
        )
    };
    let parse = |text: String, name: &str| {
        let cfg = dir.join(format!("{name}.json"));
        std::fs::write(&cfg, text).unwrap();
        let parsed = parser("run_mapper", RUN_MAPPER)
            .parse([
                input.display().to_string(),
                "-cfg".to_string(),
                cfg.display().to_string(),
                "--output_dir".to_string(),
                name.to_string(),
            ])
            .unwrap();
        MapperConfig::from_args(&parsed, None)
    };
    let environment = parse(with_fix("11-13"), "fixed_environment").unwrap();
    assert_eq!(environment.steps.held.len(), 1);
    assert!(environment.steps.biased.settings.constraints.is_empty());
    assert!(environment.steps.saddle.settings.constraints.is_empty());
    let core = parse(with_fix("3"), "fixed_core").unwrap();
    let e = run_with(&core, &dir, None, &registry)
        .unwrap_err()
        .to_string();
    assert!(e.contains("-fix holds atoms of the core"), "{e}");
    let without = with_fix("11-13")
        .replace(", \"oniom_flag\": [\"1-7\", \"auto\", \"gfnff\"]", "")
        .replace("\"oniom_flag\": [\"1-7\", \"auto\", \"gfnff\"],", "");
    let plain = parse(without, "fixed_plain").unwrap();
    assert!(plain.steps.held.is_empty());
    for c in [
        &plain.steps.biased,
        &plain.steps.saddle,
        &plain.steps.minimum,
    ] {
        assert_eq!(c.settings.constraints.len(), 1);
    }
}
