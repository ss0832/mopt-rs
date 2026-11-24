//! A chain of structures relaxed with a built-in calculator, with its figures: the
//! structures of a file (a few points along a reaction, as a scan or a run with an
//! artificial force leaves them) become a chain of images a given distance apart, which
//! is relaxed the way `nebmain` is used in the articles of MultiOptPy's author.
//!
//! ```text
//! cargo run --release -p mopt-rs --example chain_molecule -- \
//!     STRUCTURES.xyz OUTPUT_DIRECTORY [CALCULATOR] [KEY=VALUE ...]
//! ```
//!
//! `CALCULATOR` is a name of the built-in calculators (`gfn0xtb` by default). Settings:
//! `spacing=A` (distance of the images in Angstrom, 0.5), `iterations=N`, `predicted=N` and
//! `equal=N` (redistribution every N iterations, 0: never), `climb=N` (0: never),
//! `ratio=R` (share of the quasi-Newton step), `pairs=N` (pairs of the Hessian updates, 0:
//! as many as the block updates keep), `hessian=N` (Hessians of the images every N
//! iterations), `fe=CODE` (fixed ends: 1 the first, 2 the last, 3 both), `tolerance=F`,
//! `gather=W` (images gathered around the predicted maxima, W + 1 times as dense there),
//! `multioptpy=1` (MultiOptPy's forms of the steps), `atom=1` (the component of the
//! gradient along the direction of every atom is removed, MultiOptPy's rule, not along
//! the one direction of the whole structure), `verbose=1` (energies, forces, and lengths
//! along the chain of all images in every iteration).
//!
//! The last chain is written to `path.xyz` in the output directory.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;
use mopt_rs::domain::{SpinCharge, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::io::xyz::{format_xyz_system, read_xyz_trajectory_file};
use mopt_rs::path::band::machine::{ChainRelaxation, ChainReport, ChainSettings};
use mopt_rs::path::band::redistribute::{by_spacing, Gathering};
use mopt_rs::path::band::trust::FixedEnds;
use mopt_rs::path::band::{ChainKind, Compatibility};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::workflow::chain::{relax_chain, CalculatorSurface, Flow};
use mopt_rs::workflow::chain_figures::ChainFigures;

/// Hartree in kcal/mol.
const KCAL: f64 = Units::HARTREE_TO_KCAL_MOL;

fn main() -> MoptResult<()> {
    let usage =
        "usage: chain_molecule STRUCTURES.xyz OUTPUT_DIRECTORY [CALCULATOR] [KEY=VALUE ...]";
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (input, directory) = match (arguments.first(), arguments.get(1)) {
        (Some(input), Some(directory)) => (input.clone(), directory.clone()),
        _ => return Err(MoptError::config(usage)),
    };
    let named = arguments.get(2).filter(|a| !a.contains('='));
    let calculator_name = named.map_or("gfn0xtb", String::as_str);
    let options = &arguments[2 + usize::from(named.is_some())..];

    let mut settings = ChainSettings {
        kind: ChainKind::Atoms,
        max_iterations: 60,
        force_tolerance: Some(1e-4),
        ..ChainSettings::of_the_articles()
    };
    let mut spacing = 0.5;
    let mut per_atom = false;
    let mut verbose = false;
    for option in options {
        let (key, value) = option
            .split_once('=')
            .ok_or_else(|| MoptError::config(format!("{option}: expected KEY=VALUE")))?;
        let number: f64 = value
            .parse()
            .map_err(|_| MoptError::config(format!("{option}: {value} is not a number")))?;
        let every = (number > 0.0).then_some(number as usize);
        match key {
            "spacing" => spacing = number,
            "atom" => per_atom = number > 0.0,
            "verbose" => verbose = number > 0.0,
            "iterations" => settings.max_iterations = number as usize,
            "predicted" => settings.predicted_maxima_every = every,
            "equal" => settings.equal_spacing_every = every,
            "climb" => settings.climb_after = every,
            "hessian" => settings.hessian_every = every,
            "fe" => settings.ends = FixedEnds::from_code(number as i64),
            "tolerance" => settings.force_tolerance = (number > 0.0).then_some(number),
            "ratio" => {
                if let Some(rfo) = settings.rfo.as_mut() {
                    rfo.ratio = number;
                }
            }
            "gather" => {
                settings.gathering = (number > 0.0).then_some(Gathering {
                    weight: number,
                    width: 1.0,
                });
            }
            "pairs" => {
                if let Some(rfo) = settings.rfo.as_mut() {
                    rfo.pairs = every;
                }
            }
            "multioptpy" => {
                if number > 0.0 {
                    settings.compatibility = Compatibility::MultiOptPy;
                }
            }
            _ => return Err(MoptError::config(format!("{key}: no such setting"))),
        }
    }

    let frames = read_xyz_trajectory_file(&input, SpinCharge::default(), None)?.frames;
    let template = frames
        .first()
        .map(|frame| frame.system.clone())
        .ok_or_else(|| MoptError::config(format!("{input}: no structure")))?;
    let points: Vec<Col<f64>> = frames
        .iter()
        .map(|frame| frame.system.coords_flat_bohr())
        .collect();
    let chain = by_spacing(&points, spacing * Units::ANGSTROM_TO_BOHR, ChainKind::Atoms);
    if per_atom {
        // One direction along the chain per atom (MultiOptPy's rule) instead of the one
        // of the whole structure.
        settings.block = Some(3);
    }
    println!(
        "{} structures read, {} images {spacing} Angstrom apart, calculator {calculator_name}",
        frames.len(),
        chain.len()
    );

    let calculator = CalculatorRegistry::native().create(
        calculator_name,
        &CalculatorOptions::default(),
        &template,
    )?;
    let surface = CalculatorSurface::new(calculator.as_ref(), template.clone());
    let mut machine = ChainRelaxation::new(settings, chain)?.with_structure(template.clone())?;
    let mut figures = ChainFigures::new(&directory)
        .keep_every_iteration(true)
        .energy_axis(KCAL, "Energy (kcal/mol)");
    let mut progress = |report: &ChainReport, _: &mut ChainRelaxation| {
        let first = report.energies[0];
        let top = report
            .energies
            .iter()
            .fold(f64::NEG_INFINITY, |m, e| m.max(*e));
        let (worst, force) = report
            .force_max
            .iter()
            .enumerate()
            .fold((0, 0.0_f64), |m, (i, f)| if *f > m.1 { (i, *f) } else { m });
        println!(
            "iteration {:3}  barrier {:8.3} kcal/mol  largest force {:9.2e} (image {:2})  maxima {:?}  redistributed {}",
            report.iteration,
            KCAL * (top - first),
            force,
            worst,
            report.maxima,
            report.redistributed
        );
        if verbose {
            let energies: Vec<String> = report
                .energies
                .iter()
                .map(|e| format!("{:7.1}", KCAL * (e - first)))
                .collect();
            let forces: Vec<String> = report
                .force_max
                .iter()
                .map(|f| format!("{f:7.0e}"))
                .collect();
            let lengths: Vec<String> = report.lengths.iter().map(|l| format!("{l:7.3}")).collect();
            println!("    energies {}", energies.join(" "));
            println!("    forces   {}", forces.join(" "));
            println!("    lengths  {}", lengths.join(" "));
        }
        Ok(Flow::Continue)
    };
    let outcome = relax_chain(&mut machine, &surface, &mut [&mut figures, &mut progress])?;
    println!(
        "{} iterations, ended with {:?}; figures in {directory}",
        outcome.iterations, outcome.end
    );
    if let Some(last) = outcome.last {
        let mut text = String::new();
        for (i, (point, energy)) in last.chain.iter().zip(&last.energies).enumerate() {
            println!(
                "image {i:2}  E {:14.8}  relative {:8.3} kcal/mol  force {:9.2e}",
                energy,
                KCAL * (energy - last.energies[0]),
                last.force_max[i]
            );
            text.push_str(&format_xyz_system(
                &template.with_flat_coords_bohr(point)?,
                None,
            ));
        }
        let path = std::path::Path::new(&directory).join("path.xyz");
        std::fs::write(&path, text)?;
    }
    Ok(())
}
