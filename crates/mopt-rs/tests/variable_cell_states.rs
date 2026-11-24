//! Two electronic states in a periodic cell: crossing points with the cell fixed and with the
//! cell optimized together with the atoms (`optmain -mf ... [-vc]`), and the optimization of an
//! excited state with the cell (`optmain -es 1 -vc`).
//!
//! The model has two atoms in a cell. The energy of state `s` is
//! `K sum_k (|a_k| - L_s)^2 + A sum_{i<j} cos^2(a_i, a_j) + KB (r - R_s)^2 + C_s` with the
//! lattice vectors `a_k` and the distance `r` of the atoms, plus `KB |d_perp|^2` for the
//! component of the pair displacement perpendicular to the first lattice vector. This selects
//! an orientation in the crystal and vanishes along the reference path. The calculator gives no stress, so
//! that the stress of every state comes from differences of its own energy. With
//! `u = |a| - 5.5` and `v = r - 2.25`, the gap is `E_1 - E_2 = 6 K u + KB v + 0.002` and the
//! mean is `3 K u^2 + KB v^2` up to a constant, so that the crossing point of the least mean
//! (the gradient projection) has `u = l`, `v = l / 2` with `l = -0.002 / (6 K + KB / 2)`, the
//! cell cubic; the quadratic penalty `mean + (E_1 - E_2)^2 / alpha` has its least at
//! `u = -2 d / alpha`, `v = -d / alpha` with `d = 0.002 / (1 + (12 K + KB) / alpha)`; and in a
//! fixed cubic cell of edge `a` the seam is the distance `v = -(6 K (a - 5.5) + 0.002) / KB`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use faer::{Col, Mat};
use mopt_rs::cli::optmain;
use mopt_rs::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::Calculator;

const K: f64 = 0.01;
const A: f64 = 0.05;
const KB: f64 = 0.02;
const L: [f64; 2] = [5.0, 6.0];
const R: [f64; 2] = [2.0, 2.5];
const C: [f64; 2] = [0.0, -0.002];

/// Two states of two atoms in a cell (the first in the singlet ground state; the second the
/// triplet or the first excited state).
struct TwoCells;

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Calculator for TwoCells {
    fn name(&self) -> &str {
        "two_cells"
    }

    fn supports_pbc(&self) -> bool {
        true
    }

    fn analytic_hessian(&self) -> bool {
        true
    }

    fn supports_excited_states(&self) -> bool {
        true
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let s = usize::from(sys.spin_charge.multiplicity == 3 || req.excited_state == 1);
        let cell = sys
            .active_cell()
            .ok_or_else(|| MoptError::config("two_cells needs a periodic cell"))?;
        let a: [[f64; 3]; 3] = std::array::from_fn(|k| cell.lattice_vector(k).value());
        let mut e = C[s];
        for v in &a {
            e += K * (norm(v) - L[s]).powi(2);
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            let c = dot(&a[i], &a[j]) / (norm(&a[i]) * norm(&a[j]));
            e += A * c * c;
        }
        let x = sys.coords_flat_bohr();
        let d = [x[3] - x[0], x[4] - x[1], x[5] - x[2]];
        let r = norm(&d);
        let axis = a[0].map(|v| v / norm(&a[0]));
        let parallel = dot(&d, &axis);
        let perpendicular = std::array::from_fn(|i| d[i] - parallel * axis[i]);
        e += KB * ((r - R[s]).powi(2) + dot(&perpendicular, &perpendicular));
        let f = 2.0 * KB * (r - R[s]) / r;
        let mut g = Col::<f64>::zeros(6);
        for c in 0..3 {
            g[3 + c] = f * d[c] + 2.0 * KB * perpendicular[c];
            g[c] = -g[3 + c];
        }
        let mut out = EvalOutput::energy_gradient(Quantity::new(e), g);
        if req.need_hessian {
            // Only the pair potential varies with the Cartesian coordinates at fixed lattice.
            let radial = 2.0 * KB * R[s] / r.powi(3);
            out.hessian_bohr = Some(Mat::from_fn(6, 6, |i, j| {
                let (row, col) = (i % 3, j % 3);
                let diagonal = if row == col { f + 2.0 * KB } else { 0.0 };
                let block = diagonal + radial * d[row] * d[col] - 2.0 * KB * axis[row] * axis[col];
                if i / 3 == j / 3 {
                    block
                } else {
                    -block
                }
            }));
        }
        Ok(out)
    }
}

fn registry() -> CalculatorRegistry {
    let mut r = CalculatorRegistry::native();
    r.register("two_cells", |_| Ok(Box::new(TwoCells)));
    r
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_vc_states_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn mode_kill_leaves_a_lattice_saddle_and_rejects_a_return() {
    use mopt_rs::cli::mapper::MapperConfig;
    use mopt_rs::cli::mapper_jobs::{CommandJobs, MapperStop, RunOutput};
    use mopt_rs::cli::option::parser;
    use mopt_rs::cli::spec::RUN_MAPPER;
    use mopt_rs::core_infra::log::Logger;
    use mopt_rs::opt::mode_kill::ModeKillSettings;
    use mopt_rs::workflow::mapper::{Configuration, FrameSink, MapperJobs};

    // Both instabilities are strains. All fixed-cell atomic curvatures are zero.
    struct DoubleWell;
    impl Calculator for DoubleWell {
        fn name(&self) -> &str {
            "strain_double_well"
        }
        fn supports_pbc(&self) -> bool {
            true
        }
        fn analytic_hessian(&self) -> bool {
            true
        }
        fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
            let cell = sys.active_cell().unwrap();
            let [a, b, c] = std::array::from_fn(|i| norm(&cell.lattice_vector(i).value()) - 5.0);
            let e = -0.05 * a * a + 0.01 * (b * b - 0.25).powi(2) + 0.03 * c * c;
            let mut out = EvalOutput::energy_gradient(Quantity::new(e), Col::zeros(sys.dof()));
            if req.need_hessian {
                out.hessian_bohr = Some(Mat::zeros(sys.dof(), sys.dof()));
            }
            Ok(out)
        }
    }
    struct Frames;
    impl FrameSink for Frames {
        fn push(&mut self, _: &Configuration, _: f64, _: &str) -> MoptResult<()> {
            Ok(())
        }
    }

    let dir = scratch("mode_kill");
    let input = start_file(&dir, 5.0, 2.0);
    let cfg = dir.join("config.json");
    std::fs::write(&cfg, r#"{"step1_settings":{"othersoft":"lj","variable_cell":true},"step3_settings":{"calc_exact_hess":1}}"#).unwrap();
    let args = parser("run_mapper", RUN_MAPPER)
        .parse(vec![
            input.display().to_string(),
            "-cfg".into(),
            cfg.display().to_string(),
        ])
        .unwrap();
    let mut config = MapperConfig::from_args(&args, None).unwrap();
    let point = Configuration::Structure(
        mopt_rs::io::xyz::read_xyz_file(&input, Default::default(), None).unwrap(),
    );
    let registry = CalculatorRegistry::native();
    for (step, rounds, should_escape) in [(1e-10, 2, false), (0.1, 10, true)] {
        config.steps.saddle.settings.mode_kill = Some(ModeKillSettings {
            displacement: step,
            max_rounds: rounds,
            ..Default::default()
        });
        let jobs = CommandJobs::new(
            &config.steps,
            &DoubleWell,
            MapperStop::new(&dir, Logger::silent()),
            RunOutput {
                both: Logger::silent(),
                screen: Logger::silent(),
                file: None,
            },
            &registry,
            Default::default(),
        );
        let modes = jobs.modes(&point, false).unwrap().unwrap();
        assert_eq!(modes.negative, 2);
        assert!(modes
            .wavenumbers
            .as_ref()
            .unwrap()
            .iter()
            .all(|w| w.abs() < 1e-6));
        assert_eq!(modes.unstable.len(), 2);
        let atomic = point.structure().unwrap().dof();
        for mode in &modes.unstable {
            assert!(mode.wavenumber.unwrap() < 0.0);
            assert!(mode.displacement.nrows() > atomic);
            assert!(mode
                .displacement
                .iter()
                .take(atomic)
                .all(|v| v.abs() < 1e-8));
        }
        assert_eq!(modes.modes.iter().filter(|m| m.curvature < 0.0).count(), 2);
        let found = jobs
            .mode_kill_saddle(&point, &modes, "strain mode kill", &mut Frames)
            .unwrap();
        assert_eq!(found.is_some(), should_escape);
        if let Some(saddle) = found {
            assert_eq!(saddle.modes.unwrap().negative, 1);
            let cell = saddle
                .optimized
                .system
                .structure()
                .unwrap()
                .active_cell()
                .unwrap();
            assert!((norm(&cell.lattice_vector(0).value()) - 5.0).abs() < 0.01);
            assert!(((norm(&cell.lattice_vector(1).value()) - 5.0).abs() - 0.5).abs() < 0.02);
        }
    }
}

/// A cubic cell of edge `edge` Bohr with the atoms `distance` Bohr apart along x.
fn start_file(dir: &Path, edge: f64, distance: f64) -> PathBuf {
    let a = edge * Units::BOHR_TO_ANGSTROM;
    let r = distance * Units::BOHR_TO_ANGSTROM;
    let path = dir.join("pair.xyz");
    std::fs::write(
        &path,
        format!(
            "2\nLattice=\"{a} 0 0 0 {a} 0 0 0 {a}\" pbc=\"T T T\"\nAr 0.3 0.4 0.5\nAr {} 0.4 0.5\n",
            0.3 + r
        ),
    )
    .unwrap();
    path
}

/// The optimized structure of `optmain` on the model with the options `extra`.
fn optimize(dir: &Path, input: &Path, extra: &[&str]) -> MolecularSystem {
    let input = input.to_string_lossy().into_owned();
    let mut args = vec![input.as_str(), "-os", "two_cells", "-tcc", "-ns", "400"];
    args.extend_from_slice(extra);
    let parsed = mopt_rs::cli::option::parser("optmain", mopt_rs::cli::spec::OPTMAIN)
        .parse(args)
        .unwrap();
    let config = optmain::OptmainConfig::from_args(&parsed).unwrap();
    let out = optmain::run_with(&config, dir, None, &registry()).unwrap();
    assert!(out[0].converged, "{extra:?} did not converge");
    out[0].summary.clone().unwrap().system
}

/// The lengths of the lattice vectors and the distance of the atoms (Bohr).
fn shape(sys: &MolecularSystem) -> ([f64; 3], f64) {
    let cell = sys.active_cell().unwrap();
    let lengths = std::array::from_fn(|k| norm(&cell.lattice_vector(k).value()));
    let x = sys.coords_flat_bohr();
    let r = norm(&[x[3] - x[0], x[4] - x[1], x[5] - x[2]]);
    (lengths, r)
}

/// The energies of the two states (singlet and triplet) at `sys`.
fn gap(sys: &MolecularSystem) -> f64 {
    let mut triplet = sys.clone();
    triplet.spin_charge.multiplicity = 3;
    let e = |s: &MolecularSystem| {
        TwoCells
            .evaluate(s, &EvalRequest::ENERGY_ONLY)
            .unwrap()
            .energy
            .value()
    };
    e(sys) - e(&triplet)
}

fn assert_shape(sys: &MolecularSystem, edge: f64, distance: f64, tolerance: f64, what: &str) {
    let (lengths, r) = shape(sys);
    for l in lengths {
        assert!(
            (l - edge).abs() < tolerance,
            "{what}: lattice {lengths:?}, expected {edge}"
        );
    }
    assert!(
        (r - distance).abs() < tolerance,
        "{what}: distance {r}, expected {distance}"
    );
}

#[test]
fn the_gradient_projection_finds_the_crossing_of_the_atoms_and_the_cell() {
    let dir = scratch("mesx");
    let input = start_file(&dir, 5.3, 2.3);
    let sys = optimize(&dir, &input, &["-vc", "-mf", "opt_mesx", "0", "3"]);
    let l = -0.002 / (6.0 * K + 0.5 * KB);
    assert_shape(&sys, 5.5 + l, 2.25 + 0.5 * l, 2e-3, "opt_mesx -vc");
    assert!(gap(&sys).abs() < 1e-5, "gap {}", gap(&sys));
}

#[test]
fn the_seam_penalty_moves_the_cell_by_the_combined_stress() {
    let dir = scratch("seam");
    let input = start_file(&dir, 5.3, 2.3);
    let sys = optimize(&dir, &input, &["-vc", "-mf", "seam", "0", "3"]);
    let alpha = 0.05;
    let d = 0.002 / (1.0 + (12.0 * K + KB) / alpha);
    assert_shape(
        &sys,
        5.5 - 2.0 * d / alpha,
        2.25 - d / alpha,
        2e-3,
        "seam -vc",
    );
    assert!(
        (gap(&sys) - d).abs() < 1e-4,
        "gap {}, expected {d}",
        gap(&sys)
    );
}

#[test]
fn the_gradient_projection_finds_the_crossing_in_a_fixed_periodic_cell() {
    let dir = scratch("fixed");
    let edge = 5.4;
    let input = start_file(&dir, edge, 2.3);
    let sys = optimize(&dir, &input, &["-mf", "opt_mesx", "0", "3"]);
    let (lengths, r) = shape(&sys);
    for l in lengths {
        assert!((l - edge).abs() < 1e-9, "the fixed cell moved: {lengths:?}");
    }
    let v = -(6.0 * K * (edge - 5.5) + 0.002) / KB;
    assert!(
        (r - (2.25 + v)).abs() < 2e-3,
        "distance {r}, expected {}",
        2.25 + v
    );
    assert!(gap(&sys).abs() < 1e-5, "gap {}", gap(&sys));
}

/// `run_mapper` on the two spin states with a variable cell: the minimum of each state with its
/// own cell, and the crossing point of the atoms and the cell, from the gradient projection
/// of the minimum itself (`crossing_direct=on`) and from the AFIR paths on the gradient
/// projection (`crossing_afir=gp`, the default), whose candidates are refined there too.
#[test]
fn the_mapper_finds_the_crossing_of_two_spin_states_with_the_cell() {
    use mopt_rs::cli::mapper::{run_with_parts, MapperConfig, MapperParts};
    use mopt_rs::workflow::mapper::{CrossingKind, NodeKind};

    let dir = scratch("mapper");
    let start = start_file(&dir, L[0], R[0]);
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"max_iterations": 8, "rng_seed": 1, "channels": false,
                      "afir_gamma_kJmol": 50.0, "dist_lower_ang": 0.5, "dist_upper_ang": 3.0},
  "step1_settings": {"othersoft": "two_cells", "NSTEP": 300, "variable_cell": true,
                     "tight_convergence_criteria": true},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let args = vec![
        start.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "-map_opt".to_string(),
        "states=0,1;0,3".to_string(),
        "crossing_direct=on".to_string(),
    ];
    let parsed = mopt_rs::cli::option::parser("run_mapper", mopt_rs::cli::spec::RUN_MAPPER)
        .parse(args)
        .unwrap();
    let config = MapperConfig::from_args(&parsed, None).unwrap();
    let run = run_with_parts(
        &config,
        &dir,
        None,
        &registry(),
        &MapperParts::default(),
        &mut [],
    )
    .unwrap();
    let net = &run.network;
    let summary = net
        .nodes
        .iter()
        .map(|n| {
            let sys = n.system.structure().unwrap();
            let (lengths, r) = shape(sys);
            format!(
                "{} {:?} E {:.6} a {:.4} {:.4} {:.4} r {r:.4}",
                n.label(),
                n.kind,
                n.energy,
                lengths[0],
                lengths[1],
                lengths[2]
            )
        })
        .collect::<Vec<String>>()
        .join("\n");
    assert_eq!(net.states.len(), 2, "{summary}");
    let (pes1, pes2) = (net.states[0], net.states[1]);
    assert_eq!(pes2.multiplicity, 3, "{summary}");
    // The minimum of each state with its own cell.
    for (state, s) in [(pes1, 0), (pes2, 1)] {
        let found = net.minima().any(|n| {
            let (lengths, r) = shape(n.system.structure().unwrap());
            n.state == state
                && lengths.iter().all(|l| (l - L[s]).abs() < 3e-3)
                && (r - R[s]).abs() < 3e-3
        });
        assert!(found, "the minimum of state {s} with its cell\n{summary}");
    }
    // Every crossing point is the one of the atoms and the cell, at the mean enthalpy there.
    let kind = NodeKind::Crossing {
        kind: CrossingKind::Mecp,
        states: [pes1, pes2],
    };
    let crossings: Vec<_> = net.of_kind(&kind).collect();
    assert!(!crossings.is_empty(), "no crossing point\n{summary}");
    let l = -0.002 / (6.0 * K + 0.5 * KB);
    let mean = 3.0 * K * (l * l + 0.25) + KB * (0.25 * l * l + 0.0625) - 0.001;
    for x in &crossings {
        let sys = x.system.structure().unwrap();
        let (lengths, r) = shape(sys);
        assert!(
            lengths.iter().all(|a| (a - (5.5 + l)).abs() < 3e-3)
                && (r - (2.25 + 0.5 * l)).abs() < 3e-3,
            "{}: not the crossing of the atoms and the cell\n{summary}",
            x.label()
        );
        assert!(
            (x.energy - mean).abs() < 1e-5,
            "{}: E {} against {mean}\n{summary}",
            x.label(),
            x.energy
        );
    }
    // One crossing point, however many routes reached it.
    assert_eq!(crossings.len(), 1, "{summary}");
}

/// The gap of the ground state and the first excited state (the same spin) at `sys`.
fn gap_of_roots(sys: &MolecularSystem) -> f64 {
    let e = |root: usize| {
        TwoCells
            .evaluate(
                sys,
                &EvalRequest {
                    excited_state: root,
                    ..EvalRequest::ENERGY_ONLY
                },
            )
            .unwrap()
            .energy
            .value()
    };
    e(0) - e(1)
}

#[test]
fn the_gradient_projection_of_a_conical_intersection_reaches_the_seam_with_the_cell() {
    // The ground state and the first excited state of one spin (`-mf opt_meci`: the second
    // state is the next root). The model has no coupling of the two, so that the second
    // direction of the branching plane (its updating from the mean gradient) leaves a point
    // of the seam rather than its minimum: the end is on the seam, the cell moved with it.
    let dir = scratch("meci");
    let input = start_file(&dir, 5.3, 2.3);
    let sys = optimize(&dir, &input, &["-vc", "-mf", "opt_meci"]);
    assert!(
        gap_of_roots(&sys).abs() < 1e-5,
        "gap {}",
        gap_of_roots(&sys)
    );
    let (lengths, _) = shape(&sys);
    assert!(
        lengths.iter().all(|l| (l - 5.3).abs() > 1e-3),
        "the cell did not move: {lengths:?}"
    );
}

/// `run_mapper` on the two roots of one spin with a variable cell: the minimum of each with
/// its own cell, and conical intersections on the seam of the atoms and the cell (the
/// gradient projection of `-mf opt_meci` from the minima and on the AFIR paths).
#[test]
fn the_mapper_finds_conical_intersections_of_two_roots_with_the_cell() {
    use mopt_rs::cli::mapper::{run_with_parts, MapperConfig, MapperParts};
    use mopt_rs::workflow::mapper::{CrossingKind, NodeKind};

    let dir = scratch("mapper_meci");
    let start = start_file(&dir, L[0], R[0]);
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"max_iterations": 8, "rng_seed": 1, "channels": false,
                      "afir_gamma_kJmol": 50.0, "dist_lower_ang": 0.5, "dist_upper_ang": 3.0},
  "step1_settings": {"othersoft": "two_cells", "NSTEP": 300, "variable_cell": true,
                     "tight_convergence_criteria": true},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let args = vec![
        start.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "-map_opt".to_string(),
        "states=0,1;0,1,1".to_string(),
        "crossing_direct=on".to_string(),
    ];
    let parsed = mopt_rs::cli::option::parser("run_mapper", mopt_rs::cli::spec::RUN_MAPPER)
        .parse(args)
        .unwrap();
    let config = MapperConfig::from_args(&parsed, None).unwrap();
    let run = run_with_parts(
        &config,
        &dir,
        None,
        &registry(),
        &MapperParts::default(),
        &mut [],
    )
    .unwrap();
    let net = &run.network;
    let summary = net
        .nodes
        .iter()
        .map(|n| {
            let (lengths, r) = shape(n.system.structure().unwrap());
            format!(
                "{} E {:.6} a {:.4} {:.4} {:.4} r {r:.4}",
                n.label(),
                n.energy,
                lengths[0],
                lengths[1],
                lengths[2]
            )
        })
        .collect::<Vec<String>>()
        .join("\n");
    assert_eq!(net.states.len(), 2, "{summary}");
    let (s0, s1) = (net.states[0], net.states[1]);
    assert_eq!((s1.multiplicity, s1.excited_state), (1, 1), "{summary}");
    for (state, s) in [(s0, 0), (s1, 1)] {
        let found = net.minima().any(|n| {
            let (lengths, r) = shape(n.system.structure().unwrap());
            n.state == state
                && lengths.iter().all(|l| (l - L[s]).abs() < 3e-3)
                && (r - R[s]).abs() < 3e-3
        });
        assert!(found, "the minimum of root {s} with its cell\n{summary}");
    }
    let kind = NodeKind::Crossing {
        kind: CrossingKind::Meci,
        states: [s0, s1],
    };
    let crossings: Vec<_> = net.of_kind(&kind).collect();
    assert!(!crossings.is_empty(), "no conical intersection\n{summary}");
    for x in &crossings {
        let sys = x.system.structure().unwrap();
        assert!(
            gap_of_roots(sys).abs() < 1e-4,
            "{}: gap {}\n{summary}",
            x.label(),
            gap_of_roots(sys)
        );
    }
}

#[test]
fn the_gradient_projection_finds_the_conical_intersection_in_a_fixed_periodic_cell() {
    // In a fixed cubic cell of edge `a` the seam of the two roots is the distance
    // `v = -(6 K (a - 5.5) + 0.002) / KB` for a pair aligned with the first lattice vector,
    // so that the conical intersection is that distance.
    let dir = scratch("meci_fixed");
    let edge = 5.4;
    let input = start_file(&dir, edge, 2.3);
    let sys = optimize(&dir, &input, &["-mf", "opt_meci"]);
    let (lengths, r) = shape(&sys);
    for l in lengths {
        assert!((l - edge).abs() < 1e-9, "the fixed cell moved: {lengths:?}");
    }
    let v = -(6.0 * K * (edge - 5.5) + 0.002) / KB;
    assert!(
        (r - (2.25 + v)).abs() < 2e-3,
        "distance {r}, expected {}",
        2.25 + v
    );
    assert!(
        gap_of_roots(&sys).abs() < 1e-5,
        "gap {}",
        gap_of_roots(&sys)
    );
}

/// `run_mapper` on the two roots of one spin in a fixed periodic cell: the conical
/// intersection of the gradient projection (from the minima and on the AFIR paths) is the one
/// distance of the seam in that cell, one node however many routes reach it.
#[test]
fn the_mapper_finds_the_conical_intersection_of_two_roots_in_a_fixed_cell() {
    use mopt_rs::cli::mapper::{run_with_parts, MapperConfig, MapperParts};
    use mopt_rs::workflow::mapper::{CrossingKind, NodeKind};

    let dir = scratch("mapper_meci_fixed");
    let edge = 5.5;
    let start = start_file(&dir, edge, R[0]);
    let cfg = dir.join("config.json");
    std::fs::write(
        &cfg,
        r#"{
  "mapper_settings": {"max_iterations": 8, "rng_seed": 1, "channels": false,
                      "afir_gamma_kJmol": 50.0, "dist_lower_ang": 0.5, "dist_upper_ang": 3.0},
  "step1_settings": {"othersoft": "two_cells", "NSTEP": 300, "tight_convergence_criteria": true},
  "step3_settings": {"tight_convergence_criteria": true, "calc_exact_hess": 1, "NSTEP": 300},
  "step4_settings": {
    "tight_convergence_criteria": true,
    "NSTEP": 300,
    "intrinsic_reaction_coordinates": ["0.1", "400", "lqa"]
  }
}"#,
    )
    .unwrap();
    let args = vec![
        start.display().to_string(),
        "-cfg".to_string(),
        cfg.display().to_string(),
        "-map_opt".to_string(),
        "states=0,1;0,1,1".to_string(),
        "crossing_direct=on".to_string(),
    ];
    let parsed = mopt_rs::cli::option::parser("run_mapper", mopt_rs::cli::spec::RUN_MAPPER)
        .parse(args)
        .unwrap();
    let config = MapperConfig::from_args(&parsed, None).unwrap();
    let run = run_with_parts(
        &config,
        &dir,
        None,
        &registry(),
        &MapperParts::default(),
        &mut [],
    )
    .unwrap();
    let net = &run.network;
    let summary = net
        .nodes
        .iter()
        .map(|n| {
            let (lengths, r) = shape(n.system.structure().unwrap());
            format!(
                "{} E {:.6} a {:.4} {:.4} {:.4} r {r:.4}",
                n.label(),
                n.energy,
                lengths[0],
                lengths[1],
                lengths[2]
            )
        })
        .collect::<Vec<String>>()
        .join("\n");
    assert_eq!(net.states.len(), 2, "{summary}");
    let (s0, s1) = (net.states[0], net.states[1]);
    let kind = NodeKind::Crossing {
        kind: CrossingKind::Meci,
        states: [s0, s1],
    };
    let crossings: Vec<_> = net.of_kind(&kind).collect();
    assert_eq!(crossings.len(), 1, "{summary}");
    let sys = crossings[0].system.structure().unwrap();
    let (lengths, r) = shape(sys);
    for l in lengths {
        assert!((l - edge).abs() < 1e-9, "the fixed cell moved: {lengths:?}");
    }
    let v = -(6.0 * K * (edge - 5.5) + 0.002) / KB;
    assert!(
        (r - (2.25 + v)).abs() < 3e-3,
        "distance {r}, expected {}\n{summary}",
        2.25 + v
    );
    assert!(gap_of_roots(sys).abs() < 1e-4, "{summary}");
}

#[test]
fn an_excited_state_moves_the_cell_by_its_own_stress() {
    let dir = scratch("excited");
    let input = start_file(&dir, 5.3, 2.3);
    let sys = optimize(&dir, &input, &["-vc", "-es", "1"]);
    assert_shape(&sys, L[1], R[1], 2e-3, "-es 1 -vc");
}
