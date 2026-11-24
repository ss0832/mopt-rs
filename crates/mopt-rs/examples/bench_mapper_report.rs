//! The time and size of the report of `run_mapper` for large networks: a made-up network of
//! `MINIMA` minima of 12 argon atoms (structures and energies from a fixed pseudo-random
//! sequence) joined by a chain and further random links, each through a transition state, is
//! written as the page, and the time of the view (map, layout, disconnectivity graph) and of the
//! whole page and the size of the file are printed.
//!
//! Usage: `cargo run --release --example bench_mapper_report [MINIMA ...] [-- OUTPUT_DIR]`
//! (default 1000 5000 20000; the pages go to the system's temporary directory).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;
use std::time::Instant;

use mopt_rs::cli::mapper_report::{MapperReport, ReportOptions};
use mopt_rs::cli::mapper_report_data::View;
use mopt_rs::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::workflow::mapper::network::{
    ElectronicState, IrcPath, Level, Network, Node, NodeKind, Origin, TsEdge,
};

/// A fixed pseudo-random sequence in [0, 1) (64-bit linear congruential).
struct Sequence(u64);

impl Sequence {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

const ATOMS: usize = 12;

fn structure(seq: &mut Sequence, base: &[[f64; 3]], spread: f64) -> MoptResult<MolecularSystem> {
    let points: Vec<[f64; 3]> = base
        .iter()
        .map(|p| {
            [
                p[0] + spread * (seq.next() - 0.5),
                p[1] + spread * (seq.next() - 0.5),
                p[2] + spread * (seq.next() - 0.5),
            ]
        })
        .collect();
    MolecularSystem::new(
        vec![Element::from_atomic_number(18)?; ATOMS],
        &positions::<Angstrom>(&points),
        SpinCharge::default(),
        None,
    )
}

fn node(kind: NodeKind, system: MolecularSystem, kcal: f64) -> Node {
    let state = ElectronicState::of(&system, 0);
    Node::new(
        kind,
        Level::Exploration,
        system.into(),
        kcal * Units::KCAL_MOL_TO_HARTREE,
        state,
        Origin::default(),
    )
}

fn network(minima: usize) -> MoptResult<Network> {
    let mut seq = Sequence(42);
    let base: Vec<[f64; 3]> = (0..ATOMS)
        .map(|k| {
            [
                3.8 * (k % 3) as f64,
                3.8 * ((k / 3) % 2) as f64,
                3.8 * (k / 6) as f64,
            ]
        })
        .collect();
    let mut net = Network::new(vec![Element::from_atomic_number(18)?; ATOMS]);
    let mut energies = Vec::with_capacity(minima);
    for _ in 0..minima {
        let e = 20.0 * seq.next();
        energies.push(e);
        net.add_node(node(NodeKind::Minimum, structure(&mut seq, &base, 1.5)?, e))?;
    }
    let h = Units::KCAL_MOL_TO_HARTREE;
    let links = minima + minima / 3;
    for k in 0..links {
        let (a, b) = if k + 1 < minima {
            (k, k + 1)
        } else {
            let a = (seq.next() * minima as f64) as usize % minima;
            (a, (seq.next() * minima as f64) as usize % minima)
        };
        let top = energies[a].max(energies[b]) + 1.0 + 10.0 * seq.next();
        let ts = net.add_node(node(
            NodeKind::TransitionState,
            structure(&mut seq, &base, 1.5)?,
            top,
        ))?;
        net.add_ts_edge(TsEdge {
            id: 0,
            level: Level::Exploration,
            ts,
            ends: [a, b],
            barriers: [(top - energies[a]) * h, (top - energies[b]) * h],
            imaginary_wavenumber: Some(-100.0),
            irc: IrcPath::default(),
            job: None,
        })?;
    }
    Ok(net)
}

fn main() -> MoptResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (sizes, dir) = match args.iter().position(|a| a == "--") {
        Some(k) => (&args[..k], args.get(k + 1).map(PathBuf::from)),
        None => (&args[..], None),
    };
    let sizes: Vec<usize> = if sizes.is_empty() {
        vec![1000, 5000, 20000]
    } else {
        sizes
            .iter()
            .map(|s| {
                s.parse()
                    .map_err(|_| MoptError::parse(format!("not a number of minima: {s}")))
            })
            .collect::<MoptResult<_>>()?
    };
    let dir = dir.unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir)?;
    println!("minima  nodes   links   view (s)  page (s)  size (MB)  map");
    for minima in sizes {
        let net = network(minima)?;
        let options = ReportOptions::parse(&[format!("file=bench_mapper_{minima}.html")])?;
        let start = Instant::now();
        let view = View::new(&net, options.clone());
        let view_time = start.elapsed().as_secs_f64();
        let method = view.map_method.clone();
        let links = view.links.len();
        drop(view);
        let mut report = MapperReport::new(&dir, "bench", "made up", "", options);
        let start = Instant::now();
        report.write(&net, "The run ended.", true)?;
        let page_time = start.elapsed().as_secs_f64();
        let size = std::fs::metadata(report.path())?.len() as f64 / 1e6;
        println!(
            "{minima:6}  {:6}  {links:6}  {view_time:8.2}  {page_time:8.2}  {size:9.2}  {method}",
            net.nodes.len()
        );
    }
    Ok(())
}
