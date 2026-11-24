//! `run_automecp`: the translation of its settings into those of the mapper run (the root of
//! the first state, the settings the command sets itself), a search that finds no crossing
//! point, and a restart with other arguments. The model is that of `cli_mapper_states.rs`
//! (three atoms, the second surface for the triplet and for the first excited singlet).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use faer::{Col, Mat};
use mopt_rs::cli::automecp::{self, AutomecpConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOMECP;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::Units;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity};
use mopt_rs::error::MoptResult;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;

const R1: f64 = 2.0;
const R2: f64 = 5.5;
const A: f64 = 0.004;
const RC: f64 = 3.5;
const KC: f64 = 0.02;
const D: f64 = 0.004;
const K: f64 = 0.05;
const AS: [f64; 3] = [2.5, 3.0, 2.0];
const RC3: f64 = 4.6;
const D3: f64 = 0.012;
const C: f64 = 4.0;

/// `f_s(r)` and its first two derivatives.
fn curve(s: usize, r: f64) -> (f64, f64, f64) {
    if s == 0 {
        let (p, q) = (r - R1, r - R2);
        (
            A * p * p * q * q,
            2.0 * A * p * q * (p + q),
            2.0 * A * (q * q + 4.0 * p * q + p * p),
        )
    } else if s == 1 {
        let p = r - RC;
        (KC * p * p + D, 2.0 * KC * p, 2.0 * KC)
    } else {
        let p = r - RC3;
        (KC * p * p + D3, 2.0 * KC * p, 2.0 * KC)
    }
}

/// The three-atom model: the surface by the multiplicity (3: PES2) and the root.
struct TwoSurfaces;

fn pair(
    x: &Col<f64>,
    (i, j): (usize, usize),
    term: &dyn Fn(f64) -> (f64, f64, f64),
    e: &mut f64,
    g: &mut Col<f64>,
    h: &mut Mat<f64>,
) {
    let d: Vec<f64> = (0..3).map(|k| x[3 * j + k] - x[3 * i + k]).collect();
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let u: Vec<f64> = d.iter().map(|v| v / r).collect();
    let (v, dv, d2v) = term(r);
    *e += v;
    for k in 0..3 {
        g[3 * i + k] -= dv * u[k];
        g[3 * j + k] += dv * u[k];
    }
    for a in 0..3 {
        for b in 0..3 {
            let delta = f64::from(u8::from(a == b));
            let block = d2v * u[a] * u[b] + dv / r * (delta - u[a] * u[b]);
            h[(3 * i + a, 3 * i + b)] += block;
            h[(3 * j + a, 3 * j + b)] += block;
            h[(3 * i + a, 3 * j + b)] -= block;
            h[(3 * j + a, 3 * i + b)] -= block;
        }
    }
}

impl Calculator for TwoSurfaces {
    fn name(&self) -> &str {
        "two_surfaces"
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let s = (usize::from(sys.spin_charge.multiplicity == 3) + req.excited_state).min(2);
        let x = sys.coords_flat_bohr();
        let mut e = 0.0;
        let mut g = Col::<f64>::zeros(9);
        let mut h = Mat::<f64>::zeros(9, 9);
        let harmonic = |at: f64| move |r: f64| (K * (r - at).powi(2), 2.0 * K * (r - at), 2.0 * K);
        pair(&x, (0, 1), &|r| curve(s, r), &mut e, &mut g, &mut h);
        pair(&x, (1, 2), &harmonic(AS[s]), &mut e, &mut g, &mut h);
        pair(&x, (0, 2), &harmonic(C), &mut e, &mut g, &mut h);
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            out.hessian_bohr = Some(h);
        }
        Ok(out)
    }
}

fn registry() -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    r.register("two_surfaces", |_| Ok(Box::new(TwoSurfaces)));
    r
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_automecp_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The minimum of PES1 at R1 as an xyz file in `dir`.
fn start_file(dir: &Path) -> PathBuf {
    let start = dir.join("start.xyz");
    let x3 = (R1 * R1 + C * C - AS[0] * AS[0]) / (2.0 * R1);
    let y3 = (C * C - x3 * x3).max(0.0).sqrt();
    let x = [0.0, 0.0, 0.0, R1, 0.0, 0.0, x3, y3, 0.0];
    let mut text = String::from("3\nthe minimum of PES1 at R1\n");
    for a in 0..3 {
        text.push_str(&format!(
            "He {:.10} {:.10} {:.10}\n",
            x[3 * a] * Units::BOHR_TO_ANGSTROM,
            x[3 * a + 1] * Units::BOHR_TO_ANGSTROM,
            x[3 * a + 2] * Units::BOHR_TO_ANGSTROM
        ));
    }
    std::fs::write(&start, text).unwrap();
    start
}

/// The command line of `run_automecp` in `dir` with the settings file `settings` and `more`.
fn args(dir: &Path, settings: &str, more: &[&str]) -> Vec<String> {
    let cfg = dir.join("automecp.json");
    std::fs::write(&cfg, settings).unwrap();
    let mut out = vec![
        start_file(dir).display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "--output_dir".to_string(),
        dir.join("automecp_output").display().to_string(),
    ];
    out.extend(more.iter().map(|s| s.to_string()));
    out
}

fn config(args: &[String]) -> MoptResult<AutomecpConfig> {
    AutomecpConfig::from_args(&parser("run_automecp", RUN_AUTOMECP).parse(args.to_vec())?)
}

const SETTINGS: &str = r#"{
  "step1_settings": {"othersoft": "two_surfaces", "NSTEP": 300},
  "step2_settings": {"tight_convergence_criteria": true, "NSTEP": 300},
  "step3_settings": {"NSTEP": 300, "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]}
}"#;

/// The settings of the mapper run that `document` and the command line `more` give.
fn translated(dir: &Path, document: &str, more: &[&str]) -> MoptResult<JsonValue> {
    let c = config(&args(dir, document, more))?;
    automecp::mapper_document(&c, &JsonValue::parse(document)?)
}

fn mapper_setting(doc: &JsonValue, key: &str) -> Option<String> {
    doc.get("mapper_settings")?
        .get(key)?
        .as_str()
        .map(str::to_string)
}

#[test]
fn the_first_state_can_be_an_excited_root() {
    let dir = scratch("first_root");
    // An S1/S2 conical intersection: the first state is the first excited singlet, from
    // excited_state of step1_settings, which leaves the block (the states of the mapper run
    // say which surface every step is on).
    let with_root = r#"{
      "step1_settings": {"othersoft": "two_surfaces", "excited_state": 1},
      "state2": [0, 1, 2], "manual_AFIR": [100, "1", "2"]
    }"#;
    let doc = translated(&dir, with_root, &[]).unwrap();
    assert_eq!(
        mapper_setting(&doc, "states").as_deref(),
        Some("0,1,1;0,1,2")
    );
    assert!(doc
        .get("step1_settings")
        .and_then(|s| s.get("excited_state"))
        .is_none());
    // The multiplicity given by its unpaired electrons.
    let unpaired = r#"{
      "step1_settings": {"othersoft": "two_surfaces", "unpaired_electrons": 2},
      "state2": [0, 1], "manual_AFIR": [100, "1", "2"]
    }"#;
    let doc = translated(&dir, unpaired, &[]).unwrap();
    assert_eq!(mapper_setting(&doc, "states").as_deref(), Some("0,3;0,1"));
    // The same root from the command line.
    let doc = translated(
        &dir,
        SETTINGS,
        &[
            "-ma", "100", "1", "2", "-root1", "1", "-state2", "0", "1", "2",
        ],
    )
    .unwrap();
    assert_eq!(
        mapper_setting(&doc, "states").as_deref(),
        Some("0,1,1;0,1,2")
    );
    // Two different roots for the first state are refused.
    let e = translated(&dir, with_root, &["-root1", "2"]).err().unwrap();
    assert!(e.to_string().contains("excited_state"), "{e}");
    // Without either, the first state is the ground state of its multiplicity.
    let doc = translated(
        &dir,
        SETTINGS,
        &["-ma", "100", "1", "2", "-state2", "0", "3"],
    )
    .unwrap();
    assert_eq!(mapper_setting(&doc, "states").as_deref(), Some("0,1;0,3"));
}

#[test]
fn the_settings_that_the_command_makes_are_not_overwritten_silently() {
    let dir = scratch("own_settings");
    let base = ["-ma", "100", "1", "2", "-state2", "0", "3"];
    for (key, value) in [
        ("states", "0,1;0,1,1"),
        ("crossing_pushes", "50:1:3"),
        ("crossing_from", "minima"),
        ("ts_states", "all"),
    ] {
        let document = format!(
            r#"{{"step1_settings": {{"othersoft": "two_surfaces"}},
                "mapper_settings": {{"{key}": "{value}"}}}}"#
        );
        let e = translated(&dir, &document, &base).err();
        assert!(
            e.as_ref().is_some_and(|e| e.to_string().contains(key)),
            "{key}: {e:?}"
        );
        // The same through -map_opt.
        let mut more = base.to_vec();
        let item = format!("{key}={value}");
        more.extend(["-map_opt", item.as_str()]);
        let e = translated(&dir, SETTINGS, &more).err();
        assert!(
            e.as_ref().is_some_and(|e| e.to_string().contains(key)),
            "-map_opt {key}: {e:?}"
        );
    }
    // A value equal to the one the command makes is no conflict.
    let same = r#"{"step1_settings": {"othersoft": "two_surfaces"},
                   "mapper_settings": {"ts_states": "none"}}"#;
    assert!(translated(&dir, same, &base).is_ok());
    // A kind of path in mapper_settings that the command line contradicts.
    let afir = r#"{"step1_settings": {"othersoft": "two_surfaces"},
                   "mapper_settings": {"crossing_afir": "model"}}"#;
    let mut more = base.to_vec();
    more.extend(["-afir", "gp"]);
    let e = translated(&dir, afir, &more).err();
    assert!(
        e.as_ref()
            .is_some_and(|e| e.to_string().contains("crossing_afir")),
        "{e:?}"
    );
}

#[test]
fn every_crossing_point_is_checked_within_the_seam_unless_switched_off() {
    // The crossing points of the model lie at minima of the seam: none has an imaginary mode
    // there. The check costs the Hessians of both states at each; switched off, it is said
    // not to be checked and those Hessians are not computed.
    let mut runs = Vec::new();
    for (name, more) in [
        ("seam_on", Vec::<&str>::new()),
        ("seam_off", vec!["-map_opt", "crossing_seam_check=off"]),
    ] {
        let dir = scratch(name);
        let mut given = vec!["-ma", "100", "1", "2", "-state2", "0", "3"];
        given.extend(more);
        let c = config(&args(&dir, SETTINGS, &given)).unwrap();
        let registry = registry();
        let before = registry.counts();
        let run = automecp::run_with(&c, &dir, None, &registry).unwrap();
        let hessians = registry.counts().since(&before).hessians;
        let net = &run.mapper.network;
        assert!(!net.crossing_edges.is_empty());
        for e in &net.crossing_edges {
            // The gap in the order of the states, within crossing_gap.
            let gap = e.gap.unwrap();
            assert!(gap.abs() <= 1e-3, "{gap}");
            match (name, &e.seam) {
                ("seam_on", Some(mopt_rs::workflow::mapper::SeamCheck::Checked(0))) => {}
                ("seam_off", Some(mopt_rs::workflow::mapper::SeamCheck::NotChecked(why))) => {
                    assert!(why.contains("switched off"), "{why}")
                }
                (_, other) => panic!("{name}: {other:?}"),
            }
        }
        let summary = std::fs::read_to_string(&run.summary).unwrap();
        let line = summary.lines().nth(1).unwrap();
        assert!(
            line.ends_with(if name == "seam_on" {
                ",0,\"minimum on the seam\""
            } else {
                "crossing_seam_check=off)\""
            }),
            "{line}"
        );
        let page = std::fs::read_to_string(&run.report).unwrap();
        assert!(!page.contains("saddle point on the seam ("), "{name}");
        runs.push((net.crossing_points(), hessians));
        // The network files keep it (JSON; the database holds the same entries).
        let json =
            std::fs::read_to_string(run.mapper.directory.join("reaction_network.json")).unwrap();
        assert!(json.contains("\"seam_check\"") && json.contains("\"gap_hartree\""));
    }
    let ((points, on), (_, off)) = (runs[0], runs[1]);
    assert!(
        on >= off + 2 * points,
        "{on} against {off} for {points} point(s)"
    );
}

#[test]
fn a_search_without_a_crossing_point_fails_with_its_files_written() {
    let dir = scratch("none_found");
    // Two iterations of the refinement cannot converge: no crossing point.
    let settings = r#"{
      "step1_settings": {"othersoft": "two_surfaces", "NSTEP": 300},
      "step2_settings": {"NSTEP": 2},
      "step3_settings": {"NSTEP": 300, "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]}
    }"#;
    let given = args(
        &dir,
        settings,
        &["-ma", "100", "1", "2", "-state2", "0", "3"],
    );
    let code = mopt_rs::cli::run_command_with("run_automecp", &given, &registry());
    assert_ne!(code, mopt_rs::cli::EXIT_OK);
    let out = dir.join("automecp_output");
    let summary = std::fs::read_to_string(out.join(automecp::SUMMARY_FILE)).unwrap();
    assert_eq!(summary.lines().count(), 1, "{summary}");
    let page = std::fs::read_to_string(out.join("automecp_report.html")).unwrap();
    assert!(page.contains("No crossing point was found"));
    // A restart with other arguments says which are taken from the run it continues.
    let mut again = config(&args(
        &dir,
        settings,
        &[
            "-ma", "150", "1", "3", "-state2", "0", "3", "-afir", "model",
        ],
    ))
    .unwrap();
    again.restart_dir = Some(out.clone());
    let notes = automecp::restart_notes(&again, &out).unwrap();
    for option in ["-ma", "-afir"] {
        assert!(
            notes.iter().any(|n| n.contains(option)),
            "{option}: {notes:?}"
        );
    }
    assert!(!notes.iter().any(|n| n.contains("-state2")), "{notes:?}");
}
