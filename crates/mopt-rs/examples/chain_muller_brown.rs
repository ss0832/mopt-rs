//! A chain of images relaxed on the Muller-Brown surface, with its figures: the chain
//! starts as a straight line between the two outer minima and ends on the path over the
//! two saddles, one image climbing to the higher one.
//!
//! ```text
//! cargo run --release -p mopt-rs --example chain_muller_brown -- \
//!     OUTPUT_DIRECTORY [forces|articles] [KEY=VALUE ...]
//! ```
//!
//! With `forces` (the default) the images move along the forces of the chain alone, with
//! equal spacing every tenth iteration and climbing images; with `articles` the chain is
//! relaxed the way `nebmain` is used in the articles of MultiOptPy's author (Hessians of
//! the images, half a quasi-Newton step, images put towards the predicted maxima every
//! second iteration). Settings to try on top of either: `iterations=N`, `predicted=N`,
//! `equal=N`, `climb=N` (0: never), `ratio=R` (share of the quasi-Newton step),
//! `gather=W` (images gathered around the predicted
//! maxima, W + 1 times as dense there), `multioptpy=1` (MultiOptPy's forms of the steps), `trace=IMAGE` (a line about that image
//! in every iteration).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::path::band::machine::{ChainRelaxation, ChainReport, ChainSettings};
use mopt_rs::path::band::redistribute::Gathering;
use mopt_rs::path::band::trust::FixedEnds;
use mopt_rs::path::band::{ChainKind, Compatibility};
use mopt_rs::pes::objective::{MullerBrown2D, Objective};
use mopt_rs::workflow::chain::{relax_chain, Flow, ObjectiveSurface};
use mopt_rs::workflow::chain_figures::{ChainFigures, SurfaceMap, MAP_POINTS};

/// The two outer minima of the surface.
const START: [f64; 2] = [-0.558, 1.442];
const END: [f64; 2] = [0.623, 0.028];
const IMAGES: usize = 17;

fn main() -> MoptResult<()> {
    let directory = std::env::args()
        .nth(1)
        .ok_or_else(|| MoptError::config("usage: chain_muller_brown OUTPUT_DIRECTORY"))?;
    let surface = MullerBrown2D::default();
    let chain: Vec<Col<f64>> = (0..IMAGES)
        .map(|k| {
            let t = k as f64 / (IMAGES - 1) as f64;
            Col::from_fn(2, |c| START[c] + t * (END[c] - START[c]))
        })
        .collect();
    let articles = std::env::args().nth(2).as_deref() == Some("articles");
    let mut settings = if articles {
        ChainSettings {
            kind: ChainKind::Variables,
            max_iterations: 100,
            ends: FixedEnds::from_code(3),
            force_tolerance: Some(1e-6),
            ..ChainSettings::of_the_articles()
        }
    } else {
        ChainSettings {
            kind: ChainKind::Variables,
            max_iterations: 300,
            ends: FixedEnds::from_code(3),
            equal_spacing_every: Some(10),
            climb_after: Some(60),
            force_tolerance: Some(1e-6),
            ..ChainSettings::default()
        }
    };
    let mut trace: Option<usize> = None;
    for option in std::env::args().skip(3) {
        let (key, value) = option
            .split_once('=')
            .ok_or_else(|| MoptError::config(format!("{option}: expected KEY=VALUE")))?;
        let number: f64 = value
            .parse()
            .map_err(|_| MoptError::config(format!("{option}: {value} is not a number")))?;
        let every = (number > 0.0).then_some(number as usize);
        match key {
            "iterations" => settings.max_iterations = number as usize,
            "predicted" => settings.predicted_maxima_every = every,
            "equal" => settings.equal_spacing_every = every,
            "climb" => settings.climb_after = every,
            "ratio" => {
                if let Some(rfo) = settings.rfo.as_mut() {
                    rfo.ratio = number;
                }
            }
            "multioptpy" => {
                if number > 0.0 {
                    settings.compatibility = Compatibility::MultiOptPy;
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
            "trace" => trace = Some(number as usize),
            _ => return Err(MoptError::config(format!("{key}: no such setting"))),
        }
    }
    let mut machine = ChainRelaxation::new(settings, chain)?;
    let drawn = MullerBrown2D::default();
    let mut figures = ChainFigures::new(&directory)
        .keep_every_iteration(true)
        .energy_axis(1.0e3, "Energy (mHartree)")
        .on_surface(SurfaceMap {
            x: (-1.6, 1.1),
            y: (-0.4, 2.0),
            ceiling: Some(0.05),
            points: MAP_POINTS,
            value: Box::new(move |x, y| {
                drawn
                    .evaluate(&Col::from_fn(2, |c| [x, y][c]), false)
                    .map_or(f64::NAN, |found| found.value)
            }),
        });
    // Extra processing after every iteration: here, a line on the screen.
    let mut progress = |report: &ChainReport, _: &mut ChainRelaxation| {
        let top = report
            .energies
            .iter()
            .fold(f64::NEG_INFINITY, |m, e| m.max(*e));
        let images = report.force_max.len();
        let (worst, force) =
            report.force_max[1..images - 1]
                .iter()
                .enumerate()
                .fold(
                    (0, 0.0_f64),
                    |m, (i, f)| if *f > m.1 { (i + 1, *f) } else { m },
                );
        println!(
            "iteration {:3}  highest energy {:10.6}  largest force {:9.2e} (image {:2})  climbing {:?}",
            report.iteration, top, force, worst, report.climbing
        );
        if let Some(i) = trace.filter(|&i| i < images) {
            println!(
                "    image {i}: x {:9.5} y {:9.5} E {:10.6} force {:9.2e} dt {:6.3} to neighbours {:?}",
                report.chain[i][0],
                report.chain[i][1],
                report.energies[i],
                report.force_max[i],
                report.time_step,
                (report.to_previous.get(i), report.to_next.get(i)),
            );
        }
        Ok(Flow::Continue)
    };
    let outcome = relax_chain(
        &mut machine,
        &ObjectiveSurface(&surface),
        &mut [&mut figures, &mut progress],
    )?;
    println!(
        "{} iterations, ended with {:?}; figures in {directory}",
        outcome.iterations, outcome.end
    );
    if let Some(last) = outcome.last {
        for (i, (point, energy)) in last.chain.iter().zip(&last.energies).enumerate() {
            println!(
                "image {i:2}  x {:9.5}  y {:9.5}  E {:10.6}",
                point[0], point[1], energy
            );
        }
    }
    Ok(())
}
