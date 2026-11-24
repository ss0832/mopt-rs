//! Whole commands with faer's default parallelism against the sequential one that the
//! commands set ([`mopt_rs::core_infra::parallelism`]), on 1 and 8 threads, and the setup
//! and the gradient of GFN-FF on a large structure.
//!
//! ```text
//! cargo run --release -p mopt-rs --example bench_commands -- [CASE ...]
//! cargo run --release -p mopt-rs --example bench_commands -- gfnff FILE.xyz
//! cargo run --release -p mopt-rs --example bench_commands -- verlet
//! cargo run --release -p mopt-rs --example bench_commands -- log
//! ```
//!
//! The cases (all of them without arguments): `opt` (RS-I-RFO with an exact Hessian at every
//! step, taxol with GFN-FF), `dic` (the same in delocalized internal coordinates), `neb` (a
//! band of 12 images of HCN to HNC with GFN0-xTB), `freq` (343 argon atoms), `kinetics` (the
//! propagation of a network of 300 states), `md` (125 argon atoms with a constraint), `ljopt`
//! (a long Lennard-Jones optimization, for the cost of the log) and `ljmd` (100000 steps of
//! Lennard-Jones dynamics), `eam_md` (dynamics of Cu864 and of a slab with the EAM). Each is
//! run in a pool of 1 and of 8 threads with `-core` set alike, once with faer's default and
//! once sequential; the wall times are printed. `verlet` times the EAM with kept Verlet lists
//! against a fresh neighbour search, `log` a log file flushed per line against per message.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::time::Instant;

use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::error::{MoptError, MoptResult};

fn failure(e: impl std::fmt::Display) -> MoptError {
    MoptError::config(e.to_string())
}

/// A molecule of the golden GFN-FF file as XYZ text.
fn golden_xyz(name: &str) -> MoptResult<String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/gfnff.json");
    let doc = JsonValue::parse(&std::fs::read_to_string(path).map_err(failure)?)?;
    let case = doc
        .get("cases")
        .and_then(JsonValue::as_array)
        .and_then(|c| {
            c.iter()
                .find(|c| c.get("molecule").and_then(JsonValue::as_str) == Some(name))
        })
        .ok_or_else(|| MoptError::config(format!("no molecule {name}")))?;
    let symbols = case
        .get("symbols")
        .and_then(JsonValue::as_string_vec)
        .ok_or_else(|| MoptError::config("symbols"))?;
    let x: Vec<f64> = case
        .get("positions_bohr")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| MoptError::config("positions"))?
        .iter()
        .filter_map(JsonValue::as_f64)
        .collect();
    let a = mopt_rs::domain::Units::BOHR_TO_ANGSTROM;
    let mut text = format!("{}\n{name}\n", symbols.len());
    for (k, s) in symbols.iter().enumerate() {
        text.push_str(&format!(
            "{s} {:.8} {:.8} {:.8}\n",
            x[3 * k] * a,
            x[3 * k + 1] * a,
            x[3 * k + 2] * a
        ));
    }
    Ok(text)
}

/// An argon cluster of `m^3` atoms (Angstrom).
fn argon(m: usize) -> String {
    let mut text = format!("{}\nargon\n", m * m * m);
    let mut k = 0usize;
    for i in 0..m {
        for j in 0..m {
            for l in 0..m {
                k += 1;
                let jitter = 0.12 * ((k as f64) * 0.731).sin();
                text.push_str(&format!(
                    "Ar {:.6} {:.6} {:.6}\n",
                    i as f64 * 3.75 + jitter,
                    j as f64 * 3.75 - 0.7 * jitter,
                    l as f64 * 3.75 + 0.4 * jitter
                ));
            }
        }
    }
    text
}

/// Copper fcc, `m^3` conventional cells, periodic in all three directions or (`slab`) in two
/// with vacuum above (Angstrom).
fn copper(m: usize, slab: bool) -> String {
    let a = 3.615;
    let mut atoms = Vec::new();
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                for b in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    atoms.push([
                        a * (i as f64 + b[0]),
                        a * (j as f64 + b[1]),
                        a * (k as f64 + b[2]),
                    ]);
                }
            }
        }
    }
    let l = a * m as f64;
    let (c, pbc) = if slab {
        (l + 15.0, "T T F")
    } else {
        (l, "T T T")
    };
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {c}\" pbc=\"{pbc}\"\n",
        atoms.len()
    );
    for x in atoms {
        text.push_str(&format!("Cu {:.6} {:.6} {:.6}\n", x[0], x[1], x[2]));
    }
    text
}

const HCN_PATH: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

/// The command and the input of a case, as arguments of `mopt` (`INPUT` stands for the input
/// file), and the input text.
fn case(name: &str) -> MoptResult<(Vec<&'static str>, String)> {
    Ok(match name {
        "opt" => (
            vec!["optmain", "INPUT", "-os", "gfnff", "-fc", "1", "-ns", "8"],
            golden_xyz("taxol")?,
        ),
        "dic" => (
            vec![
                "optmain", "INPUT", "-os", "gfnff", "-ns", "8", "-coord", "dic",
            ],
            golden_xyz("taxol")?,
        ),
        "neb" => (
            vec!["nebmain", "INPUT", "-os", "gfn0xtb", "-ns", "8", "-p", "5"],
            HCN_PATH.to_string(),
        ),
        "freq" => (vec!["freqmain", "INPUT", "-os", "lj"], argon(7)),
        "md" => (
            vec![
                "mdmain", "INPUT", "-os", "lj", "-mt", "nve", "-time", "400", "-ts", "2fs",
                "-temp", "40", "-pc", "bond", "1,2", "-md_opt", "seed=3",
            ],
            argon(5),
        ),
        "ljopt" => (
            vec![
                "optmain", "INPUT", "-os", "lj", "-opt", "fire", "-ns", "20000",
            ],
            argon(4),
        ),
        "ljmd" => (
            vec![
                "mdmain", "INPUT", "-os", "lj", "-mt", "nve", "-time", "100000", "-ts", "1fs",
                "-temp", "20", "-md_opt", "seed=3",
            ],
            argon(2),
        ),
        "eam_md" => (
            vec![
                "mdmain", "INPUT", "-os", "eam", "-mt", "nve", "-time", "200", "-ts", "2fs",
                "-temp", "300", "-md_opt", "seed=3",
            ],
            copper(6, false),
        ),
        "eam_slab" => (
            vec![
                "mdmain", "INPUT", "-os", "eam", "-mt", "nve", "-time", "200", "-ts", "2fs",
                "-temp", "300", "-md_opt", "seed=3",
            ],
            copper(5, true),
        ),
        other => return Err(MoptError::config(format!("unknown case {other}"))),
    })
}

/// Run `args` with `threads` threads in a fresh directory under `root`; the wall time.
fn run(root: &Path, args: &[&str], input: &str, threads: usize) -> MoptResult<f64> {
    let dir = root.join(format!("run_{threads}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(failure)?;
    let file = dir.join("input.xyz");
    std::fs::write(&file, input).map_err(failure)?;
    let file = file.to_string_lossy().into_owned();
    let mut argv: Vec<String> = vec!["mopt".to_string()];
    argv.extend(args.iter().map(|a| {
        if *a == "INPUT" {
            file.clone()
        } else {
            a.to_string()
        }
    }));
    argv.push("-core".to_string());
    argv.push(threads.to_string());
    std::env::set_current_dir(&dir).map_err(failure)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(failure)?;
    let start = Instant::now();
    let code = pool.install(|| mopt_rs::cli::main_from_argv(&argv));
    let time = start.elapsed().as_secs_f64();
    std::env::set_current_dir(root).map_err(failure)?;
    let _ = std::fs::remove_dir_all(&dir);
    if code != 0 {
        return Err(MoptError::calc(format!("{args:?}: exit code {code}")));
    }
    Ok(time)
}

/// The propagation of a random network of `n` states (a few bottlenecks per state).
fn kinetics(n: usize) -> MoptResult<f64> {
    use mopt_rs::core_infra::rng::Rng;
    use mopt_rs::kinetics::network::{Bottleneck, KineticNetwork, KineticState};
    use mopt_rs::kinetics::rate::{Eyring, RateSet};
    use mopt_rs::kinetics::solve::{Closure, Evolution, KineticSolver, Propagate};
    use mopt_rs::kinetics::Conditions;
    let mut rng = Rng::from_seed(5);
    let states = (0..n)
        .map(|k| KineticState::with_energy(k, &format!("EQ{k}"), 0.01 * rng.next_f64()))
        .collect();
    let mut bottlenecks = Vec::new();
    for a in 1..n {
        for _ in 0..3 {
            let b = (rng.next_f64() * a as f64) as usize;
            let id = bottlenecks.len();
            bottlenecks.push(Bottleneck::saddle(
                id,
                &format!("TS{id}"),
                [a, b],
                0.02 + 0.02 * rng.next_f64(),
            ));
        }
    }
    let net = KineticNetwork {
        states,
        bottlenecks,
        ..KineticNetwork::default()
    };
    let conditions = Conditions {
        temperature: 400.0,
        ..Conditions::default()
    };
    let start = Instant::now();
    let rates = RateSet::assemble(&net, &Eyring, &[], &conditions)?;
    let evolution = Evolution::Linear(rates.generator()?);
    let mut p0 = vec![0.0; n];
    p0[0] = 1.0;
    let mut solver = Propagate::new(1e-8, Closure::A)?;
    let times: Vec<f64> = (0..20).map(|k| 10f64.powi(k - 12)).collect();
    solver.solve(&evolution, &p0, &times, &mut rng)?;
    Ok(start.elapsed().as_secs_f64())
}

/// The EAM along 200 small random steps of Cu864 and of a copper slab: one calculator that
/// keeps its Verlet lists against a new calculator (a fresh neighbour search) at every step.
fn verlet() -> MoptResult<()> {
    use mopt_rs::domain::{EvalRequest, SpinCharge};
    use mopt_rs::pes::calculator::Calculator;
    for (name, slab) in [("cu864", false), ("slab500", true)] {
        let dir = std::env::temp_dir().join(format!("mopt_bench_verlet_{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(failure)?;
        let file = dir.join("cu.xyz");
        std::fs::write(&file, copper(if slab { 5 } else { 6 }, slab)).map_err(failure)?;
        let start_sys = mopt_rs::io::xyz::read_xyz_file(&file, SpinCharge::default(), None)?;
        let _ = std::fs::remove_dir_all(&dir);
        for kept in [false, true] {
            let mut rng = mopt_rs::core_infra::rng::Rng::from_seed(3);
            let mut sys = start_sys.clone();
            let calc = mopt_rs::pes::native::Eam::zhou2004(&sys.elements)?;
            let start = Instant::now();
            for _ in 0..200 {
                let mut x = sys.coords_flat_bohr();
                for k in 0..x.nrows() {
                    x[k] += 0.02 * (rng.next_f64() - 0.5);
                }
                sys = sys.with_flat_coords_bohr(&x)?;
                if kept {
                    calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD)?;
                } else {
                    mopt_rs::pes::native::Eam::zhou2004(&sys.elements)?
                        .evaluate(&sys, &EvalRequest::ENERGY_GRAD)?;
                }
            }
            println!(
                "eam {name} ({} atoms), 200 gradients, {}: {:.3} s",
                sys.n_atoms(),
                if kept { "kept lists" } else { "fresh search" },
                start.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

/// 20000 messages of 30 lines to a log file: one call per message (a flush per message) against
/// one call per line (a flush per line, as the file log did before).
fn log_flushes() -> MoptResult<()> {
    use mopt_rs::core_infra::log::{Level, Logger};
    let dir = std::env::temp_dir().join(format!("mopt_bench_log_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(failure)?;
    let message: String = (0..30)
        .map(|k| format!("Ar {k} 0.123456789012 1.23456789012 -2.3456789012\n"))
        .collect();
    for per_line in [true, false] {
        let path = dir.join(format!("log_{per_line}.log"));
        let log = Logger::silent();
        log.add_file(&path, Level::Info)?;
        let start = Instant::now();
        for _ in 0..20000 {
            if per_line {
                for line in message.lines() {
                    log.info(line);
                }
            } else {
                log.info(&message);
            }
        }
        println!(
            "log of 600000 lines, {}: {:.3} s",
            if per_line {
                "a flush per line"
            } else {
                "a flush per message"
            },
            start.elapsed().as_secs_f64()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// The setup and one gradient of GFN-FF for the structure in `file`.
fn gfnff(file: &str) -> MoptResult<()> {
    use mopt_rs::domain::{EvalRequest, SpinCharge};
    use mopt_rs::pes::calculator::Calculator;
    let sys = mopt_rs::io::xyz::read_xyz_file(Path::new(file), SpinCharge::default(), None)?;
    for threads in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(failure)?;
        pool.install(|| -> MoptResult<()> {
            let start = Instant::now();
            let topology = mopt_rs::pes::native::gfnff::GfnffTopology::new(&sys)?;
            let setup = start.elapsed().as_secs_f64();
            drop(topology);
            let calc = mopt_rs::pes::native::Gfnff::starting_from(&sys);
            calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD)?;
            let start = Instant::now();
            calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD)?;
            let gradient = start.elapsed().as_secs_f64();
            println!(
                "gfnff {} atoms, {threads} threads: setup {setup:.2} s, energy and gradient {gradient:.2} s",
                sys.n_atoms()
            );
            Ok(())
        })?;
    }
    Ok(())
}

fn main() -> MoptResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("verlet") => return verlet(),
        Some("log") => return log_flushes(),
        _ => {}
    }
    if args.first().map(String::as_str) == Some("gfnff") {
        let file = args
            .get(1)
            .ok_or_else(|| MoptError::config("gfnff FILE.xyz"))?;
        return gfnff(file);
    }
    let all = [
        "opt", "dic", "neb", "freq", "kinetics", "md", "ljopt", "ljmd", "eam_md", "eam_slab",
    ];
    let names: Vec<&str> = if args.is_empty() {
        all.to_vec()
    } else {
        args.iter().map(String::as_str).collect()
    };
    let root: PathBuf = std::env::temp_dir().join("mopt_bench_commands");
    std::fs::create_dir_all(&root).map_err(failure)?;
    // The commands set the sequential parallelism once; the runs below set what they measure
    // after that.
    mopt_rs::core_infra::parallelism::deterministic_linear_algebra();
    for name in names {
        for mode in ["default", "sequential"] {
            for threads in [1, 8] {
                // faer's default takes the threads of the pool of the call: the pool of the
                // command, of `-core` threads (it cannot be set back as it was).
                faer::set_global_parallelism(if mode == "default" {
                    faer::Par::rayon(threads)
                } else {
                    faer::Par::Seq
                });
                let time = if name == "kinetics" {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .map_err(failure)?;
                    pool.install(|| kinetics(300))?
                } else {
                    let (args, input) = case(name)?;
                    run(&root, &args, &input, threads)?
                };
                println!("{name:9} {mode:10} {threads} threads: {time:8.3} s");
            }
        }
    }
    Ok(())
}
