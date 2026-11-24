//! The stream of a run on the console and in the log: MultiOptPy's layout, and `<S^2>` where
//! the calculator tells it.
//!
//! MultiOptPy: `optimization.py` (the `print` calls of `Optimize.run`), `fileio.py:
//! FileIO.print_geometry_list`, `optimizer.py:CalculateMoveVector.calc_move_vector`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use mopt_rs::cli::optmain;
use mopt_rs::core_infra::log::{Level, Logger};
use mopt_rs::domain::Units;
use mopt_rs::domain::{
    positions, Angstrom, Element, EvalOutput, EvalRequest, MolecularSystem, OrbitalData, SpinCharge,
};
use mopt_rs::error::MoptResult;
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::observers::ProgressLog;
use mopt_rs::workflow::run_optimization;

const AR4: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [3.9, 0.1, 0.0],
    [1.8, 3.4, 0.2],
    [2.0, 1.2, 3.3],
];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_console_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_log_of_a_run_follows_the_stream_of_multioptpy() {
    let dir = scratch("stream");
    let xyz = dir.join("ar4.xyz");
    let mut text = String::from("4\n\n");
    for p in AR4 {
        text.push_str(&format!("Ar {} {} {}\n", p[0], p[1], p[2]));
    }
    std::fs::write(&xyz, text).unwrap();
    let input = xyz.to_string_lossy().into_owned();
    let cfg = optmain::config_from_cli(&[&input, "-os", "lj", "-opt", "rsirfo_bfgs", "-fc", "1"])
        .unwrap();
    let out = optmain::run(&cfg, &dir, None).unwrap();
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    let position = |needle: &str| {
        log.find(needle)
            .unwrap_or_else(|| panic!("{needle}\n{log}"))
    };
    // The lines of MultiOptPy, in its order.
    let order = [
        "********************************",
        "Use Lennard-Jones cluster potential.",
        // The structure before the first iteration and again under its heading, in Angstrom.
        "Ar      0.000000000000      0.000000000000      0.000000000000",
        "# ITR. 0",
        "==================================================================================",
        "trust radii (unit. Bohr):",
        "step  radii (unit. Bohr):",
        "EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: 6):",
        "caluculation results (unit a.u.):",
        "                         Value                     Threshold ",
        "ENERGY                 :",
        "Maximum  Force         :",
        "RMS      Displacement  :",
        "BIAS ENERGY SHIFT      :",
        "# ITR. 1",
        "converged!!!",
        "Symmetry:",
        "Trial of geometry optimization (",
        "All calculations were completed.",
        "Optimized structure file path:",
        "Trajectory file path:",
    ];
    let mut last = 0;
    for needle in order {
        let at = position(needle);
        assert!(at >= last, "'{needle}' is out of order\n{log}");
        last = at;
    }
    // Six eigenvalues of the Hessian of four atoms, twelve characters each, on one line.
    let after = &log[position("EIGENVALUES")..];
    let values: Vec<f64> = after
        .lines()
        .nth(1)
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(values.len(), 6);
    assert!(values.iter().all(|v| *v > 0.0), "{values:?}");
    // The structure is printed at every iteration and once more at the end.
    let iterations = log.matches("# ITR. ").count();
    let first_atoms = log.lines().filter(|l| l.starts_with("Ar   ")).count();
    assert_eq!(first_atoms, 4 * (iterations + 2));
    assert!(!log.contains("<S**2>"));
    // Numbers that MultiOptPy does not format look as Python prints them, and Cartesian
    // coordinates are not repeated in another system.
    let radius = log
        .lines()
        .find_map(|l| l.strip_prefix("trust radii (unit. Bohr):  "))
        .unwrap();
    assert_eq!(radius, "0.5");
    assert!(!log.contains("COORDINATES ("));
}

#[test]
fn internal_coordinates_are_shown_beside_the_cartesian_structure() {
    let dir = scratch("internal");
    let xyz = dir.join("water.xyz");
    std::fs::write(
        &xyz,
        "3\n\nO 0.0 0.0 0.1\nH 0.0 0.78 -0.48\nH 0.0 -0.76 -0.5\n",
    )
    .unwrap();
    let input = xyz.to_string_lossy().into_owned();
    let run = |coordinates: &str| {
        let cfg = optmain::config_from_cli(&[
            &input,
            "-os",
            "gfnff",
            "-opt",
            "rsirfo_bfgs",
            "-coord",
            coordinates,
        ])
        .unwrap();
        let out = optmain::run(&cfg, &dir, None).unwrap();
        std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap()
    };
    // Redundant internal coordinates: two bonds and the angle, named, in Bohr and radian,
    // after the structure in Angstrom under every heading.
    let log = run("ric");
    let iterations = log.matches("# ITR. ").count();
    assert_eq!(
        log.matches("COORDINATES (ric, NUMBER OF VALUES: 3, unit a.u.):")
            .count(),
        iterations
    );
    let block = &log[log.find("COORDINATES (").unwrap()..];
    let rows: Vec<(&str, f64)> = block
        .lines()
        .skip(1)
        .take(3)
        .map(|l| {
            let (label, value) = l.split_once(':').unwrap();
            (label.trim(), value.trim().parse().unwrap())
        })
        .collect();
    assert_eq!(rows[0].0, "R(1-2)");
    assert_eq!(rows[1].0, "R(1-3)");
    assert!(rows[2].0.starts_with("A("), "{rows:?}");
    // 0.97 Angstrom is 1.83 Bohr; the angle is about 1.8 rad.
    assert!(
        (rows[0].1 - 1.83).abs() < 0.1 && (rows[2].1 - 1.8).abs() < 0.3,
        "{rows:?}"
    );
    let water = log.find("O   ").unwrap();
    assert!(water < log.find("COORDINATES (").unwrap());
    // Combinations of internal coordinates are numbered.
    let log = run("dic");
    assert!(
        log.contains("COORDINATES (dic, NUMBER OF VALUES: 3, unit a.u.):"),
        "{log}"
    );
    assert!(log.contains("q1                     :") && log.contains("q3                     :"));
}

/// Lennard-Jones that claims a spin-contaminated doublet.
struct Radical;

impl Calculator for Radical {
    fn name(&self) -> &str {
        "radical"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let mut out = LennardJones.evaluate(sys, req)?;
        out.orbitals = Some(OrbitalData {
            spin_square: Some(0.7812),
            ..OrbitalData::default()
        });
        Ok(out)
    }
}

#[test]
fn the_spin_expectation_value_is_shown_at_every_iteration() {
    let dir = scratch("spin");
    let sys = MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); 4],
        &positions::<Angstrom>(&AR4),
        SpinCharge {
            charge: 0,
            multiplicity: 2,
        },
        None,
    )
    .unwrap();
    let settings = OptimizerSettings {
        methods: vec!["fire".to_string()],
        max_iterations: 3,
        criteria: ConvergenceCriteria::TIGHT,
        ..OptimizerSettings::default()
    };
    let log = Logger::silent();
    log.add_file(dir.join("run.log"), Level::Debug).unwrap();
    let mut progress = ProgressLog::new(log.clone(), settings.criteria);
    let mut opt = GeometryOptimizer::new(settings.clone(), sys.clone()).unwrap();
    // The report of the first iteration carries it.
    let tasks = match opt.ask() {
        mopt_rs::opt::state_machine::Action::Evaluate(tasks) => tasks,
        _ => panic!("no evaluation asked for"),
    };
    let results = tasks
        .iter()
        .map(|t| Radical.evaluate(&t.system, &t.request).unwrap())
        .collect();
    let report = opt.tell(results).unwrap();
    assert_eq!(report.spin_squares.len(), 1);
    assert_eq!(report.spin_squares[0].value, 0.7812);
    assert_eq!(report.spin_squares[0].exact, 0.75);
    // And the log of a run shows it under every heading.
    let mut opt = GeometryOptimizer::new(settings, sys).unwrap();
    run_optimization(&mut opt, &Radical, &mut [&mut progress]).unwrap();
    log.flush();
    let text = std::fs::read_to_string(dir.join("run.log")).unwrap();
    let shown = text
        .matches("<S**2>                 :  0.781200000000  (S(S+1) = 0.7500;")
        .count();
    assert!(
        shown >= 2 && shown == text.matches("# ITR. ").count(),
        "{text}"
    );
}

#[test]
fn step_factor_fragments_geometry_information_and_the_early_stop() {
    let dir = scratch("small");
    let xyz = dir.join("ar4.xyz");
    let mut text = String::from("4\n\n");
    for p in AR4 {
        text.push_str(&format!(
            "Ar {} {} {}\n",
            p[0] * 1.1,
            p[1] * 1.1,
            p[2] * 1.1
        ));
    }
    std::fs::write(&xyz, text).unwrap();
    let input = xyz.to_string_lossy().into_owned();
    let run = |options: &[&str]| {
        let mut args = vec![input.as_str(), "-os", "lj"];
        args.extend_from_slice(options);
        let cfg = optmain::config_from_cli(&args)?;
        let out = optmain::run(&cfg, &dir, None)?;
        let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
        Ok::<_, mopt_rs::error::MoptError>((out[0].directory.clone(), log))
    };
    let step_of = |log: &str| -> f64 {
        log.lines()
            .find_map(|l| l.strip_prefix("step  radii (unit. Bohr):  "))
            .unwrap()
            .parse()
            .unwrap()
    };
    // -d: the first step of steepest descent is the gradient times the factor.
    let (_, unit) = run(&["-opt", "gradientdescent", "-ns", "2"]).unwrap();
    let (_, tenth) = run(&["-opt", "gradientdescent", "-ns", "2", "-d", "0.1"]).unwrap();
    assert!((step_of(&tenth) / step_of(&unit) - 0.1).abs() < 1e-9);
    // A method without a step factor says so; a factor that is no number is refused.
    let (_, log) = run(&["-opt", "fire", "-ns", "2", "-d", "0.1"]).unwrap();
    assert!(log.contains("-d: fire has no step factor"), "{log}");
    assert!(run(&["-opt", "gradientdescent", "-d", "fast"]).is_err());
    // -of: atoms 1 and 2 get the same gradient and so the same steps: their distance stays.
    let (directory, _) = run(&[
        "-opt",
        "gradientdescent",
        "-ns",
        "5",
        "-of",
        "1,2",
        "-gi",
        "1,2,3,4",
    ])
    .unwrap();
    let table = std::fs::read_to_string(directory.join("geometry_info.csv")).unwrap();
    let rows: Vec<&str> = table.lines().collect();
    assert_eq!(rows.len(), 6);
    // Six distances, twelve angles, twelve dihedral angles of four atoms, each once.
    let names: Vec<&str> = rows[0].split(',').collect();
    assert_eq!(names.len(), 6 + 12 + 12);
    assert_eq!(names[0], "Distance (1-2)  [ang.]");
    assert!(names[6].starts_with("Bond_angle (") && names[18].starts_with("Dihedral_angle ("));
    let first = |row: &str| -> f64 { row.split(',').next().unwrap().parse().unwrap() };
    assert!((first(rows[1]) - first(rows[5])).abs() < 1e-9);
    let third = |row: &str| -> f64 { row.split(',').nth(5).unwrap().parse().unwrap() };
    assert!((third(rows[1]) - third(rows[5])).abs() > 1e-6);
    assert!(directory.join("plot_orthogonality_1.png").exists());
    assert!(run(&["-opt", "gradientdescent", "-of", "1,9"]).is_err());
    // -negeigval: a search for a saddle point from a minimum-like structure stops at once
    // (the stretched cluster above has negative curvatures; this one has none).
    let compact = dir.join("compact.xyz");
    let mut text = String::from(
        "4

",
    );
    for p in AR4 {
        text.push_str(&format!(
            "Ar {} {} {}
",
            p[0], p[1], p[2]
        ));
    }
    std::fs::write(&compact, text).unwrap();
    let input = compact.to_string_lossy().into_owned();
    let run = |options: &[&str]| {
        let mut args = vec![input.as_str(), "-os", "lj"];
        args.extend_from_slice(options);
        let cfg = optmain::config_from_cli(&args)?;
        let out = optmain::run(&cfg, &dir, None)?;
        let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
        Ok::<_, mopt_rs::error::MoptError>((out[0].directory.clone(), log))
    };
    let (directory, log) = run(&[
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-negeigval",
    ])
    .unwrap();
    assert!(
        log.contains("No negative eigenvalues detected while saddle_order > 0."),
        "{log}"
    );
    assert!(directory
        .join("no_negative_eigenvalues_detected.txt")
        .exists());
    assert_eq!(log.matches("# ITR. ").count(), 1);
    let (directory, log) = run(&[
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-ns",
        "3",
    ])
    .unwrap();
    assert!(!directory
        .join("no_negative_eigenvalues_detected.txt")
        .exists());
    assert_eq!(log.matches("# ITR. ").count(), 3);
    // The messages of the inner solver of RS-I-RFO are in the log (debug level).
    for needle in [
        "RS-I-RFO Iteration 0",
        "Gradient norm: ",
        "Condition number of Hessian: ",
        "Found 0 negative eigenvalues (target for saddle order: 1)",
        "Initial step with alpha=1.000000 has norm=",
        "Final norm(step)=",
        "Predicted energy change: ",
    ] {
        assert!(log.contains(needle), "{needle}\n{log}");
    }
    assert!(log.find("RS-I-RFO Iteration 0").unwrap() < log.find("trust radii").unwrap());
    assert!(
        optmain::config_from_cli(&[&input, "-os", "lj", "-debug"])
            .unwrap()
            .debug
    );
}

#[test]
fn references_wildcards_and_the_results_of_all_jobs() {
    let dir = scratch("jobs");
    for (name, scale) in [("a1.xyz", 1.0), ("a2.xyz", 1.02), ("b.xyz", 1.0)] {
        let mut text = String::from("4\n\n");
        for p in AR4 {
            text.push_str(&format!(
                "Ar {} {} {}\n",
                p[0] * scale,
                p[1] * scale,
                p[2] * scale
            ));
        }
        std::fs::write(dir.join(name), text).unwrap();
    }
    // A name with a wildcard stands for the files that match, in order.
    let pattern = format!("{}/a*.xyz", dir.display());
    let cfg = optmain::config_from_cli(&[&pattern, "-os", "lj", "-opt", "rsirfo_bfgs"]).unwrap();
    assert_eq!(cfg.inputs.len(), 2);
    let out = optmain::run(&cfg, &dir, None).unwrap();
    assert_eq!(out.len(), 2);
    for (run, name) in out.iter().zip(["a1.xyz", "a2.xyz"]) {
        let summary = run.summary.as_ref().unwrap();
        assert!(summary.input.ends_with(name), "{:?}", summary.input);
        assert!(run.converged && summary.iterations > 0);
        assert!(summary.energy.value() < 0.0);
        assert_eq!(summary.energy.value(), summary.biased_energy.value());
        assert_eq!(summary.system.n_atoms(), 4);
        assert!(summary.symmetry.is_some() && summary.trajectory.exists());
    }
    // The papers of the potential are named once at the start; the last job closes the run.
    let logs: Vec<String> = out
        .iter()
        .map(|run| std::fs::read_to_string(run.directory.join("mopt.log")).unwrap())
        .collect();
    assert!(logs[0].contains("Reference: A. K. Rappe"), "{}", logs[0]);
    assert!(logs[0].find("Reference: ").unwrap() < logs[0].find("# ITR. 0").unwrap());
    assert!(!logs[0].contains("All calculations were completed."));
    assert!(logs[1].contains("All calculations were completed."));
    let nothing = format!("{}/z*.xyz", dir.display());
    assert!(optmain::config_from_cli(&[&nothing, "-os", "lj"]).is_err());
}

/// A cell of four atoms of `symbol` on fcc sites with lattice vectors `a * rows`.
fn fcc_file(
    dir: &std::path::Path,
    name: &str,
    symbol: &str,
    a: f64,
    rows: [[f64; 3]; 3],
) -> String {
    let lattice: Vec<String> = rows
        .iter()
        .flat_map(|row| row.iter().map(|c| format!("{}", c * a)))
        .collect();
    let mut text = format!("4\nLattice=\"{}\" pbc=\"T T T\"\n", lattice.join(" "));
    for f in [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ] {
        let p: Vec<f64> = (0..3)
            .map(|c| a * (f[0] * rows[0][c] + f[1] * rows[1][c] + f[2] * rows[2][c]))
            .collect();
        text.push_str(&format!("{symbol} {} {} {}\n", p[0], p[1], p[2]));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn the_cell_is_optimized_together_with_the_atoms() {
    use mopt_rs::pes::stress::full_cell;
    let dir = scratch("cell");
    // Copper too large and sheared: the cell relaxes to the fcc lattice of the potential,
    // whose nearest-neighbour distance is 2.556162 Angstrom.
    let sheared = [[1.0, 0.0, 0.0], [0.04, 1.03, 0.0], [-0.03, 0.05, 0.98]];
    let input = fcc_file(&dir, "cu.xyz", "Cu", 3.75, sheared);
    let run = |options: &[&str]| {
        let mut args = vec![input.as_str(), "-os", "eam", "-vc"];
        args.extend_from_slice(options);
        let cfg = optmain::config_from_cli(&args)?;
        let out = optmain::run(&cfg, &dir, None)?;
        let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
        Ok::<_, mopt_rs::error::MoptError>((out, log))
    };
    let volume_per_atom = |sys: &MolecularSystem| -> f64 {
        full_cell(sys).unwrap().volume_bohr3() * Units::BOHR_TO_ANGSTROM.powi(3) / 4.0
    };
    let expected = 2.556162_f64.powi(3) / 2.0_f64.sqrt();
    let (out, log) = run(&["-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    assert!(out[0].converged, "{log}");
    let summary = out[0].summary.as_ref().unwrap();
    assert!((volume_per_atom(&summary.system) / expected - 1.0).abs() < 1e-3);
    // The cohesive energy of copper in this potential, 3.54 eV per atom.
    let per_atom = summary.energy.value() * Units::HARTREE_TO_EV / 4.0;
    assert!((per_atom + 3.54).abs() < 1e-3, "{per_atom}");
    // Without pressure the enthalpy is the energy.
    assert_eq!(summary.energy.value(), summary.biased_energy.value());
    // The log shows the lattice, the volume, the stress, and the pressure at every iteration.
    let iterations = log.matches("# ITR. ").count();
    // The initial geometry, every iteration and the final geometry carry the current lattice.
    assert_eq!(
        log.matches("LATTICE VECTORS (unit ang.):").count(),
        iterations + 2
    );
    assert_eq!(log.matches("pbc=\"T T T\"").count(), iterations + 2);
    for needle in [
        "VOLUME (unit a.u.)",
        "STRESS (unit a.u.; xx yy zz yz xz xy):",
        "PRESSURE               :",
    ] {
        assert_eq!(log.matches(needle).count(), iterations, "{needle}");
    }
    assert!(log.contains("Reference: M. Gubler"));
    // The optimized file carries the new cell.
    let optimized = std::fs::read_to_string(&out[0].optimized).unwrap();
    assert!(optimized.contains("Lattice="), "{optimized}");
    // Other methods reach the same crystal.
    for method in ["lbfgs", "fire"] {
        let (other, log) = run(&["-opt", method, "-tcc", "-ns", "3000"]).unwrap();
        assert!(other[0].converged, "{method}\n{log}");
        let found = volume_per_atom(&other[0].summary.as_ref().unwrap().system);
        assert!((found / expected - 1.0).abs() < 1e-3, "{method}: {found}");
    }
    // Under pressure the cell is smaller, the pressure of the crystal is the external one,
    // and the enthalpy lies above the energy by P V.
    let (pressed, log) = run(&["-opt", "rsirfo_bfgs", "-tcc", "-vc_pressure", "20"]).unwrap();
    assert!(pressed[0].converged, "{log}");
    let s = pressed[0].summary.as_ref().unwrap();
    assert!(volume_per_atom(&s.system) < 0.95 * expected);
    let line = log
        .lines()
        .rev()
        .find(|l| l.starts_with("PRESSURE               :"))
        .unwrap();
    let gigapascal: f64 = line
        .split('(')
        .nth(1)
        .unwrap()
        .trim_end_matches(" GPa)")
        .parse()
        .unwrap();
    assert!((gigapascal - 20.0).abs() < 0.05, "{line}");
    let work = s.biased_energy.value() - s.energy.value();
    let volume = full_cell(&s.system).unwrap().volume_bohr3();
    assert!((work / volume * Units::GPA_PER_AU_PRESSURE - 20.0).abs() < 1e-3);
    // With the Hessian of the atoms and the cell (differences of the gradient) the same
    // crystal is reached in fewer iterations, and the log shows its eigenvalues.
    let plain = out[0].summary.as_ref().unwrap().iterations;
    let (exact, log) = run(&["-opt", "rsirfo_bfgs", "-tcc", "-fc", "1"]).unwrap();
    assert!(exact[0].converged, "{log}");
    let s = exact[0].summary.as_ref().unwrap();
    assert!((volume_per_atom(&s.system) / expected - 1.0).abs() < 1e-3);
    assert!(s.iterations < plain, "{} against {plain}", s.iterations);
    assert!(log.contains("EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES:"));
    // A cluster has no cell to optimize.
    let cluster = dir.join("cluster.xyz");
    std::fs::write(&cluster, "2\n\nCu 0.0 0.0 0.0\nCu 2.5 0.0 0.0\n").unwrap();
    let name = cluster.to_string_lossy().into_owned();
    let cfg = optmain::config_from_cli(&[&name, "-os", "eam", "-vc"]).unwrap();
    assert!(optmain::run(&cfg, &dir, None).is_err());
}

/// `reps` copies of the fcc cell of `symbol` along the three axes, the lattice vectors
/// `a * rows` per cell, the atoms off their sites by up to `shake` Angstrom.
fn supercell_file(
    dir: &std::path::Path,
    name: &str,
    symbol: &str,
    a: f64,
    rows: [[f64; 3]; 3],
    reps: [usize; 3],
    shake: f64,
) -> String {
    let lattice: Vec<String> = rows
        .iter()
        .zip(reps)
        .flat_map(|(row, n)| row.iter().map(move |c| format!("{}", c * a * n as f64)))
        .collect();
    let mut atoms = Vec::new();
    for i in 0..reps[0] {
        for j in 0..reps[1] {
            for k in 0..reps[2] {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let s = [f[0] + i as f64, f[1] + j as f64, f[2] + k as f64];
                    let m = atoms.len();
                    let p: Vec<f64> = (0..3)
                        .map(|c| {
                            a * (s[0] * rows[0][c] + s[1] * rows[1][c] + s[2] * rows[2][c])
                                + shake * ((((m * 7 + c * 3 + 1) % 5) as f64) - 2.0) / 2.0
                        })
                        .collect();
                    atoms.push(format!("{symbol} {} {} {}\n", p[0], p[1], p[2]));
                }
            }
        }
    }
    let text = format!(
        "{}\nLattice=\"{}\" pbc=\"T T T\"\n{}",
        atoms.len(),
        lattice.join(" "),
        atoms.concat()
    );
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// The iterations of variable-cell optimizations against the weight of the cell: one
/// crystal, one method, and the weights around the default on every run (they converge);
/// seven crystals, four methods, and five weights with `MOPT_FULL_MEASUREMENTS` set
/// (`cargo test -p mopt-rs --test console_stream iterations_of_variable_cell -- --nocapture`,
/// minutes).
#[test]
fn iterations_of_variable_cell_optimizations() {
    let full = std::env::var_os("MOPT_FULL_MEASUREMENTS").is_some();
    let dir = scratch("cellweights");
    let sheared = [[1.0, 0.0, 0.0], [0.04, 1.03, 0.0], [-0.03, 0.05, 0.98]];
    let cases = [
        ("Cu", "eam", 3.75, [1, 1, 1], 0.0, "cu.xyz"),
        ("Fe", "meam", 3.75, [1, 1, 1], 0.0, "fe_fcc.xyz"),
        ("Al", "eam", 4.2, [1, 1, 1], 0.0, "al.xyz"),
        ("Cu", "eam", 3.75, [1, 1, 4], 0.1, "cu_long.xyz"),
        ("Cu", "eam", 3.75, [2, 2, 1], 0.1, "cu_flat.xyz"),
        ("Cu", "eam", 3.75, [2, 2, 2], 0.1, "cu_32.xyz"),
        ("Al", "eam", 4.2, [1, 1, 6], 0.1, "al_long.xyz"),
    ];
    let methods: &[&str] = if full {
        &["rsirfo_bfgs", "rsprfo_bfgs", "lbfgs", "fire"]
    } else {
        &["lbfgs"]
    };
    let weights: &[&str] = if full {
        &["0.25", "0.5", "1", "2", "4"]
    } else {
        &["1", "2"]
    };
    let count = if full { cases.len() } else { 1 };
    for (symbol, potential, a, reps, shake, name) in cases.into_iter().take(count) {
        let input = supercell_file(&dir, name, symbol, a, sheared, reps, shake);
        for method in methods {
            let mut row = format!("{name} {potential} {method}:");
            for weight in weights {
                let cfg = optmain::config_from_cli(&[
                    &input,
                    "-os",
                    potential,
                    "-vc",
                    "-opt",
                    method,
                    "-tcc",
                    "-ns",
                    "3000",
                    "-vc_weight",
                    weight,
                ])
                .unwrap();
                let out = optmain::run(&cfg, &dir, None).unwrap();
                let s = out[0].summary.as_ref().unwrap();
                assert!(full || out[0].converged, "{name} {method} w={weight}");
                row.push_str(&format!(
                    "  w={weight}: {}{}",
                    s.iterations,
                    if out[0].converged {
                        ""
                    } else {
                        " (not converged)"
                    }
                ));
            }
            println!("{row}");
        }
    }
}

#[test]
fn forces_on_the_cell_and_bias_potentials_with_a_variable_cell() {
    use mopt_rs::pes::stress::full_cell;
    let dir = scratch("cellforce");
    let cubic = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let input = fcc_file(&dir, "cu.xyz", "Cu", 3.62, cubic);
    let run = |options: &[&str]| {
        let mut args = vec![input.as_str(), "-os", "eam", "-vc", "-opt", "lbfgs", "-tcc"];
        args.extend_from_slice(options);
        let cfg = optmain::config_from_cli(&args)?;
        let out = optmain::run(&cfg, &dir, None)?;
        let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
        assert!(out[0].converged, "{options:?}\n{log}");
        Ok::<_, mopt_rs::error::MoptError>((out[0].summary.clone().unwrap(), log))
    };
    let volume = |sys: &MolecularSystem| full_cell(sys).unwrap().volume_bohr3();
    let (free, _) = run(&[]).unwrap();
    // A positive collision energy on the three lattice vectors compresses the cell, a
    // negative one expands it.
    let (pressed, log) = run(&["-vc_afir", "300", "a,b,c"]).unwrap();
    let ratio = volume(&pressed.system) / volume(&free.system);
    assert!(ratio > 0.8 && ratio < 0.97, "{ratio}");
    assert!(log.contains("cell_AFIR_0"), "{log}");
    // The function minimized contains the energy of the force.
    assert!(pressed.biased_energy.value() > pressed.energy.value());
    let (widened, _) = run(&["-vc_afir", "-300", "a,b,c"]).unwrap();
    assert!(volume(&widened.system) / volume(&free.system) > 1.03);
    // Between the tips of a and b: the angle between the two vectors closes.
    let (sheared, _) = run(&["-vc_afir", "300", "ab"]).unwrap();
    let cell = full_cell(&sheared.system).unwrap();
    let (a, b) = (
        cell.lattice_vector(0).value(),
        cell.lattice_vector(1).value(),
    );
    let cosine = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
        / ((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
            * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt());
    assert!(cosine > 0.02, "{cosine}");
    // A bias potential on atoms works with the cell: atoms 1 and 2 pushed together.
    let distance = |sys: &MolecularSystem| {
        sys.distance(
            mopt_rs::domain::AtomId::new(0),
            mopt_rs::domain::AtomId::new(1),
        )
        .value()
    };
    let (pushed, log) = run(&["-ma", "150", "1", "2"]).unwrap();
    assert!(distance(&pushed.system) < distance(&free.system) - 0.05);
    assert!(log.contains("AFIR_pot_0"), "{log}");
    // What cannot be: a list that does not come in pairs, a distance that is none.
    assert!(optmain::config_from_cli(&[&input, "-os", "eam", "-vc", "-vc_afir", "300"]).is_err());
    assert!(
        optmain::config_from_cli(&[&input, "-os", "eam", "-vc", "-vc_afir", "300", "ad"]).is_err()
    );
}

#[test]
fn slabs_relax_their_two_lattice_vectors_and_wires_their_one() {
    let dir = scratch("slab");
    // A layer of copper (one atom per cell, a triangular net) stretched and sheared, with
    // the third vector of the cell standing for the vacuum; and a chain along z.
    let d = 2.7;
    let layer = dir.join("layer.xyz");
    std::fs::write(
        &layer,
        format!(
            "1\nLattice=\"{} 0.0 0.0 {} {} 0.0 0.0 0.0 30.0\" pbc=\"T T F\"\nCu 0.0 0.0 0.0\n",
            d,
            0.47 * d,
            0.9 * d
        ),
    )
    .unwrap();
    let chain = dir.join("chain.xyz");
    std::fs::write(
        &chain,
        "1\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 2.8\" pbc=\"F F T\"\nCu 0.0 0.0 0.0\n",
    )
    .unwrap();
    let run = |file: &std::path::Path, extra: &[&str]| {
        let name = file.to_string_lossy().into_owned();
        let mut args = vec![
            name.as_str(),
            "-os",
            "eam",
            "-vc",
            "-opt",
            "rsirfo_bfgs",
            "-tcc",
        ];
        args.extend_from_slice(extra);
        let cfg = optmain::config_from_cli(&args)?;
        let out = optmain::run(&cfg, &dir, None)?;
        let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
        Ok::<_, mopt_rs::error::MoptError>((out, log))
    };
    let lengths = |sys: &MolecularSystem| -> [f64; 3] {
        let cell = sys.active_cell().unwrap();
        std::array::from_fn(|k| {
            let v = cell.lattice_vector(k).value();
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() * Units::BOHR_TO_ANGSTROM
        })
    };
    let (out, log) = run(&layer, &[]).unwrap();
    assert!(out[0].converged, "{log}");
    let sys = &out[0].summary.as_ref().unwrap().system;
    let found = lengths(sys);
    // A triangular net: both vectors of one length, 60 or 120 degrees apart; the vacuum
    // vector is untouched.
    assert!((found[0] - found[1]).abs() < 1e-3, "{found:?}");
    assert!((found[2] - 30.0).abs() < 1e-12);
    let cell = sys.active_cell().unwrap();
    let (a, b) = (
        cell.lattice_vector(0).value(),
        cell.lattice_vector(1).value(),
    );
    let cosine = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
        / (found[0] * found[1] / Units::BOHR_TO_ANGSTROM.powi(2));
    assert!((cosine.abs() - 0.5).abs() < 1e-3, "{cosine}");
    assert!(found[0] > 2.2 && found[0] < 2.7, "{found:?}");
    assert!(log.contains("a slab or wire has no pressure"));
    let (out, log) = run(&chain, &[]).unwrap();
    assert!(out[0].converged, "{log}");
    let found = lengths(&out[0].summary.as_ref().unwrap().system);
    assert!((found[0] - 30.0).abs() < 1e-12 && (found[1] - 30.0).abs() < 1e-12);
    assert!(found[2] > 2.0 && found[2] < 2.7, "{found:?}");
    // No pressure on a slab.
    assert!(run(&layer, &["-vc_pressure", "5"]).is_err());
}

#[test]
fn a_saddle_point_of_the_atoms_and_the_cell_together() {
    use mopt_rs::pes::stress::stress_of;
    let dir = scratch("cellsaddle");
    // Copper with one atom missing from a 2 x 2 x 2 cell of 32 sites, and a neighbour of
    // the empty site half way into it: by symmetry a stationary point of the hop, with the
    // cell a little too large.
    let a = 3.66;
    let mut sites = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    sites.push([
                        a * (f64::from(i) + f[0]),
                        a * (f64::from(j) + f[1]),
                        a * (f64::from(k) + f[2]),
                    ]);
                }
            }
        }
    }
    // The site at the origin is empty; its neighbour at (a/2, a/2, 0) sits half way.
    let hopping = sites
        .iter()
        .position(|s| *s == [0.5 * a, 0.5 * a, 0.0])
        .unwrap();
    sites[hopping] = [0.25 * a + 0.02, 0.25 * a - 0.01, 0.015];
    sites.remove(0);
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        sites.len(),
        l = 2.0 * a
    );
    for s in &sites {
        text.push_str(&format!("Cu {} {} {}\n", s[0], s[1], s[2]));
    }
    let file = dir.join("vacancy.xyz");
    std::fs::write(&file, text).unwrap();
    let name = file.to_string_lossy().into_owned();
    let mut found = Vec::new();
    for method in ["rsirfo_bofill", "rsprfo_bofill"] {
        let cfg = optmain::config_from_cli(&[
            &name, "-os", "eam", "-vc", "-opt", method, "-order", "1", "-fc", "8", "-ns", "60",
        ])
        .unwrap();
        let out = optmain::run(&cfg, &dir, None).unwrap();
        let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
        assert!(out[0].converged, "{log}");
        let summary = out[0].summary.as_ref().unwrap();
        // The last Hessian shown has one negative eigenvalue: a saddle point of first order.
        let block = &log[log.rfind("EIGENVALUES (NORMAL COORDINATE").unwrap()..];
        let count: usize = block
            .lines()
            .next()
            .unwrap()
            .trim_end_matches("):")
            .rsplit(' ')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let values: Vec<f64> = block
            .lines()
            .skip(1)
            .flat_map(|l| l.split_whitespace().map(|v| v.parse::<f64>()))
            .take_while(Result::is_ok)
            .map(Result::unwrap)
            .take(count)
            .collect();
        let negative = values.iter().filter(|v| **v < -1e-6).count();
        assert_eq!(negative, 1, "{values:?}");
        // The cell has relaxed with the atoms: no stress is left.
        let calc = mopt_rs::pes::native::Eam::zhou2004(&summary.system.elements).unwrap();
        let stress = stress_of(&calc, &summary.system).unwrap();
        let largest = stress.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(largest < 2e-6, "{stress:?}");
        println!("{method}: {} iterations", summary.iterations);
        found.push(summary.clone());
    }
    // Both methods end at the same saddle point.
    let (first, second) = (&found[0], &found[1]);
    assert!((first.energy.value() - second.energy.value()).abs() < 1e-6);
    let summary = first;
    // The barrier of the hop of a vacancy in this potential is of the order of 0.7 eV.
    let cell = summary.system.active_cell().unwrap().volume_bohr3();
    assert!(cell > 0.0 && summary.iterations > 1);
}
