//! End-to-end `optmain` runs through the command-line layer.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::cli::{main_from_argv, run_command, EXIT_ERROR, EXIT_OK, EXIT_USAGE};
use mopt_rs::domain::Units;

const AR7: &str = "7
argon cluster
Ar   1.9500   0.0000   0.0000
Ar   0.6026   1.8546   0.0300
Ar  -1.5776   1.1462   0.0700
Ar  -1.5776  -1.1462   0.1100
Ar   0.6026  -1.8546   0.1500
Ar   0.0000   0.0000   2.1000
Ar   0.2000   0.0000  -2.0000
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_cli_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn input(dir: &Path) -> String {
    let p = dir.join("ar7.xyz");
    std::fs::write(&p, AR7).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn optmain_writes_multioptpy_run_directory() {
    let dir = scratch("files");
    let xyz = input(&dir);
    let cfg = config_from_cli(&[&xyz, "-os", "lj", "-opt", "rsirfo_bfgs", "-ns", "300"]).unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert_eq!(res.len(), 1);
    let run_dir = &res[0].directory;
    assert!(res[0].converged);
    // <date>/<stem>_OPT_ASE_<timestamp>/ under the base directory.
    let name = run_dir.file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.contains("ar7_OPT_ASE_"), "{name}");
    for f in [
        "input.txt",
        "mopt.log",
        "energy_profile.csv",
        "gradient_profile.csv",
        "bias_gradient_profile.csv",
        "energy_profile_kcalmol.csv",
        "approx_TS.txt",
        "approx_EQ.txt",
        "local_min_grad.txt",
        "energy_plot_.png",
        "plot_gradient_.png",
        "plot_bias_gradient_.png",
        "ar7_traj.xyz",
        "ar7_optimized.xyz",
    ] {
        assert!(run_dir.join(f).exists(), "missing {f}");
    }
    // Every iteration is a frame of the trajectory (appended while the run goes on); there
    // is no directory per iteration.
    let trajectory = std::fs::read_to_string(run_dir.join("ar7_traj.xyz")).unwrap();
    let frames: Vec<&str> = trajectory
        .lines()
        .filter(|l| l.starts_with("Frame "))
        .collect();
    let rows = std::fs::read_to_string(run_dir.join("energy_profile.csv"))
        .unwrap()
        .lines()
        .count()
        - 1;
    assert_eq!(frames.len(), rows);
    assert_eq!(frames[0], "Frame 0");
    assert_eq!(frames[rows - 1], format!("Frame {}", rows - 1));
    assert_eq!(trajectory.lines().count(), 9 * rows);
    let directories: Vec<String> = std::fs::read_dir(run_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(directories.is_empty(), "{directories:?}");
    let csv = std::fs::read_to_string(run_dir.join("energy_profile.csv")).unwrap();
    assert!(csv.starts_with("iter,bias,effective,raw\n"));
    let opt = std::fs::read_to_string(run_dir.join("ar7_optimized.xyz")).unwrap();
    assert!(opt.starts_with("7\nOptimizedStructure\nAr"));
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(log.contains("converged!!!"), "{log}");
    assert!(log.contains("Maximum  Force"));
    // The log says which calculator was used and where it comes from.
    assert!(log.contains("Calculator: lj: built into mopt-rs"), "{log}");
    assert!(log.contains("software_path.conf: not found"), "{log}");
    // The run ends in the basin of the LJ7 global minimum (-16.505 epsilon, UFF argon); with
    // MultiOptPy's default criteria the tiny argon forces stop it about 0.01 epsilon early.
    let last = csv.lines().last().unwrap();
    let e: f64 = last.split(',').nth(3).unwrap().parse().unwrap();
    let eps = 0.237 * Units::KCAL_MOL_TO_HARTREE;
    assert!((e / eps + 16.505).abs() < 5e-2, "{}", e / eps);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn coordinate_systems_model_functions_and_ccqn_from_the_command_line() {
    let dir = scratch("variants");
    let xyz = input(&dir);
    for args in [
        vec![
            &xyz[..],
            "-os",
            "lj",
            "-opt",
            "rsirfo_bfgs",
            "-coord",
            "dic",
        ],
        vec![
            &xyz[..],
            "-os",
            "lj",
            "-opt",
            "rsirfo_bfgs",
            "-coord",
            "ric",
            "-tcc",
        ],
        vec![&xyz[..], "-os", "lj", "-mf", "seam", "0", "1"],
        vec![
            &xyz[..],
            "-os",
            "lj",
            "-opt",
            "ccqn",
            "-order",
            "1",
            "-fc",
            "5",
            "-ccqn_bonds",
            "1,2",
            "-ccqn_cosphi",
            "0.5",
            "-coord",
            "dic",
            "-ns",
            "300",
        ],
    ] {
        let cfg = config_from_cli(&args).unwrap();
        let res = run(&cfg, &dir, None).unwrap();
        assert!(res[0].converged, "{args:?}");
        assert!(res[0].optimized.exists());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn manual_afir_pushes_fragments_together() {
    let dir = scratch("afir");
    let xyz = input(&dir);
    let dist = |path: &std::path::Path| {
        let text = std::fs::read_to_string(path).unwrap();
        let rows: Vec<[f64; 3]> = text
            .lines()
            .skip(2)
            .take(2)
            .map(|l| {
                let v: Vec<f64> = l
                    .split_whitespace()
                    .skip(1)
                    .map(|x| x.parse().unwrap())
                    .collect();
                [v[0], v[1], v[2]]
            })
            .collect();
        (0..3)
            .map(|c| (rows[0][c] - rows[1][c]).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let plain = run(
        &config_from_cli(&[&xyz, "-os", "lj", "-opt", "rsirfo_bfgs"]).unwrap(),
        &dir,
        None,
    )
    .unwrap();
    let pulled = run(
        &config_from_cli(&[
            &xyz,
            "-os",
            "lj",
            "-opt",
            "rsirfo_bfgs",
            "-ma",
            "50",
            "1",
            "2",
        ])
        .unwrap(),
        &dir,
        None,
    )
    .unwrap();
    assert!(pulled[0].converged);
    let (d0, d1) = (dist(&plain[0].optimized), dist(&pulled[0].optimized));
    assert!(d1 < d0 - 0.1, "pushing together: {d0} -> {d1} Angstrom");
    let csv = std::fs::read_to_string(pulled[0].directory.join("energy_profile.csv")).unwrap();
    let last: Vec<f64> = csv
        .lines()
        .last()
        .unwrap()
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    assert!(
        (last[1] - last[3]).abs() > 1e-6,
        "the bias energy is reported separately"
    );
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-ma", "100", "1"]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn afir_in_a_slab_cell_acts_across_the_periodic_boundary() {
    // Two argon atoms 2.5 Angstrom apart through the x boundary of a 2D-periodic cell
    // (periodic in x and y, open in z; the cell is large compared with the minimum-image LJ
    // interaction).
    let dir = scratch("afir2d");
    let xyz = dir.join("ar2.xyz");
    std::fs::write(
        &xyz,
        "2
Lattice=\"20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 30.0\" pbc=\"T T F\"          Properties=species:S:1:pos:R:3
Ar 1.0 10.0 15.0
Ar 18.5 10.0 15.0
",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let mic_x = |path: &std::path::Path| {
        let t = std::fs::read_to_string(path).unwrap();
        let x: Vec<f64> = t
            .lines()
            .skip(2)
            .take(2)
            .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
            .collect();
        let d = (x[0] - x[1]).abs() % 20.0;
        d.min(20.0 - d)
    };
    let plain = run(
        &config_from_cli(&[&xyz, "-os", "lj", "-opt", "rsirfo_bfgs"]).unwrap(),
        &dir,
        None,
    )
    .unwrap();
    let pushed = run(
        &config_from_cli(&[
            &xyz,
            "-os",
            "lj",
            "-opt",
            "rsirfo_bfgs",
            "-ma",
            "30",
            "1",
            "2",
        ])
        .unwrap(),
        &dir,
        None,
    )
    .unwrap();
    for r in [&plain[0], &pushed[0]] {
        let log = std::fs::read_to_string(r.directory.join("mopt.log")).unwrap();
        let mut lines = log.lines();
        assert!(lines.any(|line| line.starts_with("Ar")));
        assert!(lines.next().unwrap().starts_with("Ar"));
        assert_eq!(lines.next(), Some(""));
        assert_eq!(lines.next(), Some("LATTICE VECTORS (unit ang.):"));
        for (label, v) in [
            ("a", [20.0, 0.0, 0.0]),
            ("b", [0.0, 20.0, 0.0]),
            ("c", [0.0, 0.0, 30.0]),
        ] {
            assert_eq!(
                lines.next().unwrap(),
                format!(
                    "{label:2}   {:>17.12}   {:>17.12}   {:>17.12}",
                    v[0], v[1], v[2]
                )
            );
        }
        assert_eq!(lines.next(), Some("pbc=\"T T F\""));
        assert!(lines.any(|line| line == "Hessian update: bfgs"));
        assert!(lines.any(|line| line.starts_with("step  radii (unit. Bohr):")));
        assert_eq!(
            lines.next(),
            Some("EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: 6):")
        );
        let values: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(values.len(), 6);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
        let tail: Vec<&str> = log.lines().rev().take(25).collect();
        assert!(
            r.converged,
            "{}",
            tail.into_iter().rev().collect::<Vec<_>>().join(
                "
"
            )
        );
    }
    let (d0, d1) = (mic_x(&plain[0].optimized), mic_x(&pushed[0].optimized));
    assert!(
        d1 < d0 - 0.1,
        "minimum-image distance {d0} -> {d1} Angstrom"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn software_path_conf_is_reported() {
    let dir = scratch("spc");
    let xyz = input(&dir);
    let conf = dir.join("software_path.conf");
    std::fs::write(
        &conf,
        "# paths
orca::/opt/orca
",
    )
    .unwrap();
    let conf = conf.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[&xyz, "-os", "lj", "-ns", "3", "-osp", &conf]).unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    let log = std::fs::read_to_string(res[0].directory.join("mopt.log")).unwrap();
    assert!(
        log.contains(&format!("software_path.conf: read {conf}; entries orca")),
        "{log}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unimplemented_options_and_missing_axes_are_rejected() {
    let dir = scratch("reject");
    let xyz = input(&dir);
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-xtb", "GFN2-xTB"]).is_err());
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-irc", "10"]).is_err());
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-opt", "ccqn", "-order", "1"]).is_err());
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-unknown_flag"]).is_err());
    // Default-valued options that are not implemented yet are accepted.
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-gi", "1", "-fc", "-1"]).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dispatcher_exit_codes() {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(run_command("optmain", &s(&["-h"])), EXIT_OK);
    assert_eq!(run_command("nebmain", &s(&["-h"])), EXIT_OK);
    // An input file that is not there is an error of the run; an unknown option is one of
    // the usage.
    assert_eq!(run_command("mdmain", &s(&["-h"])), EXIT_OK);
    assert_eq!(run_command("mdmain", &s(&["a.xyz"])), EXIT_ERROR);
    assert_eq!(
        run_command("mdmain", &s(&["a.xyz", "-unknown_flag"])),
        EXIT_USAGE
    );
    assert_eq!(run_command("no_such_command", &[]), EXIT_USAGE);
    assert_eq!(main_from_argv(&s(&["mopt", "--version"])), EXIT_OK);
    assert_eq!(main_from_argv(&s(&["mopt"])), EXIT_USAGE);
    assert_eq!(main_from_argv(&s(&["mopt", "run_rcmc", "-h"])), EXIT_OK);
}

#[test]
fn bias_options_from_the_command_line() {
    let dir = scratch("bias");
    let xyz = input(&dir);
    // A distance restraint holds atoms 1 and 2 at 5 Angstrom; a flat-bottomed well keeps
    // atom 6 near the centroid of atoms 1-5; a one-ellipsoid repulsion runs its microiterations.
    let results = run(
        &config_from_cli(&[
            &xyz,
            "-os",
            "lj",
            "-opt",
            "rsirfo_bfgs",
            "-kp",
            "0.5",
            "5.0",
            "1,2",
            "-awp",
            "10",
            "1-5",
            "0.1,0.2,1.5,3.0",
            "6",
            "-aerp",
            "1.0",
            "0.5,0.5,0.6,0.4,0.5,0.5",
            "1.0",
            "7,6",
            "1-5",
        ])
        .unwrap(),
        &dir,
        None,
    )
    .unwrap();
    assert!(results[0].converged);
    let text = std::fs::read_to_string(&results[0].optimized).unwrap();
    let rows: Vec<Vec<f64>> = text
        .lines()
        .skip(2)
        .map(|l| {
            l.split_whitespace()
                .skip(1)
                .map(|x| x.parse().unwrap())
                .collect()
        })
        .collect();
    let d = (0..3)
        .map(|c| (rows[0][c] - rows[1][c]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((d - 5.0).abs() < 0.5, "restrained distance {d}");
    // Malformed groups are usage errors.
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-kp", "0.5", "5.0"]).is_err());
    assert!(config_from_cli(&[&xyz, "-os", "lj", "-wwp", "10", "w", "1,2,3,4", "1"]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// `-freq`: vibrational analysis and thermochemistry of the optimized structure, with the
/// files of MultiOptPy, and `symmetry.txt` with the point group.
#[test]
fn frequency_analysis_after_the_optimization() {
    let dir = scratch("freq");
    let p = dir.join("water.xyz");
    std::fs::write(
        &p,
        "3

O 0.0 0.0 0.12
H 0.0 0.80 -0.47
H 0.0 -0.70 -0.50
",
    )
    .unwrap();
    let xyz = p.to_string_lossy().into_owned();
    let cfg = config_from_cli(&[
        &xyz,
        "-os",
        "gfnff",
        "-opt",
        "rsirfo_bfgs",
        "-tcc",
        "-freq",
        "-temp",
        "350",
        "-freq_corr",
        "truhlar",
    ])
    .unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert!(res[0].converged);
    let run_dir = &res[0].directory;
    let thermo = std::fs::read_to_string(run_dir.join("thermochemistry.txt")).unwrap();
    assert!(
        thermo.starts_with(
            "Point group: C2v
Temperature 350.0000 [K]
"
        ),
        "{thermo}"
    );
    assert!(
        thermo.contains(
            "Symmetry number 2
"
        ),
        "{thermo}"
    );
    let modes = std::fs::read_to_string(run_dir.join("normal_modes.txt")).unwrap();
    // Three vibrations: the bend and the two stretches, all real.
    let frequencies: Vec<f64> = modes
        .lines()
        .find(|l| l.starts_with("Freq [cm^-1]"))
        .unwrap()
        .split_whitespace()
        .skip(2)
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(frequencies.len(), 3);
    assert!(
        frequencies[0] > 1000.0 && frequencies[2] > 3000.0,
        "{frequencies:?}"
    );
    let movies: Vec<_> = std::fs::read_dir(run_dir.join("vibration_animation"))
        .unwrap()
        .collect();
    assert_eq!(movies.len(), 3);
    assert_eq!(
        std::fs::read_to_string(run_dir.join("symmetry.txt")).unwrap(),
        "Symmetry of final structure: C2v"
    );
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(log.contains("Performing vibrational analysis...") && log.contains("Symmetry: C2v"));
    // Without -freq no analysis; with -nosymm the group is not determined.
    let cfg = config_from_cli(&[&xyz, "-os", "gfnff", "-opt", "rsirfo_bfgs", "-nosymm"]).unwrap();
    let res = run(&cfg, &dir, None).unwrap();
    assert!(!res[0].directory.join("thermochemistry.txt").exists());
    assert_eq!(
        std::fs::read_to_string(res[0].directory.join("symmetry.txt")).unwrap(),
        "Symmetry of final structure: C1"
    );
}

#[test]
fn the_method_without_opt_follows_the_order() {
    let dir = scratch("default_method");
    let xyz = dir.join("ar3.xyz");
    std::fs::write(
        &xyz,
        "3\n\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\nAr 1.8 3.5 0.0\n",
    )
    .unwrap();
    let xyz = xyz.to_string_lossy().into_owned();
    let methods = |arguments: &[&str]| {
        let mut all = vec![xyz.as_str(), "-os", "lj"];
        all.extend_from_slice(arguments);
        config_from_cli(&all).unwrap().settings.methods
    };
    assert_eq!(methods(&[]), vec!["rsirfo_block_fsb".to_string()]);
    assert_eq!(
        methods(&["-order", "1"]),
        vec!["rsirfo_block_bofill".to_string()]
    );
    assert_eq!(methods(&["-opt", "FIRELARS"]), vec!["FIRELARS".to_string()]);
    // The optimizers of the library choose in the same way when no method is named.
    assert!(mopt_rs::opt::state_machine::OptimizerSettings::default()
        .methods
        .is_empty());
    assert_eq!(
        mopt_rs::opt::registry::default_methods(2),
        vec!["rsirfo_block_bofill".to_string()]
    );
    // A run without -opt converges to the triangle of the three atoms.
    let config = config_from_cli(&[&xyz, "-os", "lj"]).unwrap();
    let out = run(&config, &dir, None).unwrap();
    assert!(out[0].converged);
    let settings = std::fs::read_to_string(out[0].directory.join("input.txt")).unwrap();
    assert!(settings.contains("rsirfo_block_fsb"), "{settings}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_starts_from_the_hessian_of_a_file() {
    use mopt_rs::domain::{EvalRequest, SpinCharge};
    use mopt_rs::io::npy::write_npy_matrix;
    use mopt_rs::io::xyz::read_xyz_file;
    use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

    let dir = scratch("first_hessian");
    let xyz = input(&dir);
    // The Hessian of the calculator at the start, in a NumPy file and as the numbers of
    // its lower triangle.
    let start = read_xyz_file(&xyz, SpinCharge::default(), None).unwrap();
    let calculator = CalculatorRegistry::native()
        .create("lj", &CalculatorOptions::default(), &start)
        .unwrap();
    let hessian = calculator
        .evaluate(&start, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    let npy = dir.join("start.npy");
    write_npy_matrix(&npy, &hessian).unwrap();
    let mut triangle = String::from("# Hartree/Bohr^2\n");
    for r in 0..hessian.nrows() {
        for c in 0..=r {
            triangle.push_str(&format!("{:.17e} ", hessian[(r, c)]));
        }
        triangle.push('\n');
    }
    let text = dir.join("start.txt");
    std::fs::write(&text, triangle).unwrap();

    let go = |more: &[&str]| {
        let mut arguments = vec![
            xyz.as_str(),
            "-os",
            "lj",
            "-opt",
            "rsirfo_bfgs",
            "-ns",
            "300",
        ];
        arguments.extend_from_slice(more);
        let out = run(&config_from_cli(&arguments).unwrap(), &dir, None).unwrap();
        assert!(out[0].converged);
        let summary = out[0].summary.clone().unwrap();
        let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
        (summary.iterations, summary.energy.value(), log)
    };
    // The run with the Hessian of the calculator at its first point only, and the runs
    // that read that Hessian: the same iterations to the same structure.
    let exact = go(&["-fc", "100000"]);
    let from_npy = go(&["-ih", &npy.to_string_lossy()]);
    let from_text = go(&["-ih", &text.to_string_lossy()]);
    assert_eq!(from_npy.0, exact.0);
    assert_eq!(from_text.0, exact.0);
    assert!((from_npy.1 - exact.1).abs() < 1e-12 && (from_text.1 - exact.1).abs() < 1e-12);
    assert!(
        from_npy.2.contains("the first Hessian is given"),
        "{}",
        from_npy.2
    );
    // Without a Hessian the run is another one.
    let plain = go(&[]);
    assert_ne!(plain.0, exact.0);
    println!(
        "iterations: {} with the Hessian, {} without",
        exact.0, plain.0
    );

    // A Hessian of another number of atoms, a file that is none, and surfaces that have
    // no single Hessian are refused.
    let small = dir.join("small.txt");
    std::fs::write(&small, "1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n").unwrap();
    for refused in [
        vec!["-ih", small.to_str().unwrap()],
        vec!["-ih", xyz.as_str()],
        vec!["-ih", npy.to_str().unwrap(), "-mf", "seam", "0", "1"],
    ] {
        let mut arguments = vec![xyz.as_str(), "-os", "lj"];
        arguments.extend_from_slice(&refused);
        let config = config_from_cli(&arguments).unwrap();
        assert!(run(&config, &dir, None).is_err(), "{refused:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The largest trust radius of a search on a gradient projection of two states with RS-I-RFO is
/// 0.1 Angstrom unless `-tr` gives one; other searches keep the default of their kind.
#[test]
fn a_search_on_a_gradient_projection_has_a_smaller_largest_trust_radius() {
    let max = |arguments: &[&str]| {
        config_from_cli(arguments)
            .unwrap()
            .settings
            .trust
            .max
            .map(|m| m.value())
    };
    let projection = 0.1 * Units::ANGSTROM_TO_BOHR;
    for name in ["opt_mesx", "opt_meci"] {
        let found = max(&["a.xyz", "-mf", name, "0", "3"]).unwrap();
        assert!((found - projection).abs() < 1e-6, "{name}: {found}");
    }
    // Given: kept as it is (in Bohr).
    assert_eq!(
        max(&["a.xyz", "-mf", "opt_mesx", "0", "3", "-tr", "0.3"]),
        Some(0.3)
    );
    // Another optimizer, another model function, one surface: the default of the optimizer.
    assert_eq!(
        max(&["a.xyz", "-mf", "opt_mesx", "0", "3", "-opt", "fire"]),
        None
    );
    assert_eq!(max(&["a.xyz", "-mf", "seam", "0", "3"]), None);
    assert_eq!(max(&["a.xyz"]), None);
    // What a step on the projection is judged by for the trust radius: the mean energy
    // unless the function of the projected gradient is asked for.
    let merit = |arguments: &[&str]| {
        config_from_cli(arguments)
            .map(|c| c.settings.projection_trust)
            .map_err(|e| e.to_string())
    };
    assert_eq!(merit(&["a.xyz", "-mf", "opt_meci"]), Ok(false));
    assert_eq!(
        merit(&["a.xyz", "-mf", "opt_meci", "-projection_trust", "merit"]),
        Ok(true)
    );
    assert!(merit(&["a.xyz", "-projection_trust", "other"]).is_err());
}
