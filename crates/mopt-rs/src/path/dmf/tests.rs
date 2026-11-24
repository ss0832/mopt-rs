//! The direct MaxFlux method on the Müller–Brown surface.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::Col;

use super::*;
use crate::math::linalg::col_norm;
use crate::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use crate::pes::objective::{MullerBrown2D, Objective};

fn col(values: &[f64]) -> Col<f64> {
    Col::from_fn(values.len(), |k| values[k])
}

/// Runs the method on `objective` to its end; the last report and the number of
/// evaluations.
fn run(machine: &mut DirectMaxFlux, objective: &dyn Objective) -> (DmfEnd, DmfReport, usize) {
    let mut evaluations = 0;
    loop {
        match machine.ask().unwrap() {
            DmfAction::Finished(end) => {
                return (end, machine.report().unwrap().clone(), evaluations);
            }
            DmfAction::Evaluate(points) => {
                evaluations += points.len();
                let values = points
                    .iter()
                    .map(|point| {
                        let found = objective.evaluate(&point.x, false).unwrap();
                        (found.value, found.gradient)
                    })
                    .collect();
                machine.tell(values).unwrap();
                if let Some(r) = machine.report() {
                    if std::env::var("DMF_TRACE").is_ok() {
                        println!(
                            "  {:3} K {:.8} g {:.3e} len {:.4} uneven {:.3} refit {} t {:?}",
                            r.iteration,
                            r.objective,
                            r.largest_gradient,
                            r.length,
                            r.unevenness,
                            r.refitted,
                            r.t
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn the_top_of_the_path_comes_to_the_saddle_between_two_minima() {
    let objective = MullerBrown2D::default();
    // The minima at (-0.558, 1.442) and (-0.050, 0.467), with the saddle at
    // (-0.822, 0.624) between them, from a straight line.
    let chain = [col(&MB_MINIMA[0]), col(&MB_MINIMA[2])];
    let saddle = col(&MB_SADDLES[0]);
    let energy_at_the_saddle = objective.evaluate(&saddle, false).unwrap().value;
    // Settings that are measured; the default is checked below.
    let variants = [
        (
            "default (3 points to the top, 1 guard on each side)",
            DmfSettings::default(),
        ),
        (
            "no guards (as PyDMF)",
            DmfSettings {
                guards: 0,
                ..DmfSettings::default()
            },
        ),
        (
            "2 guards on each side",
            DmfSettings {
                guards: 2,
                ..DmfSettings::default()
            },
        ),
        (
            "5 points to the top (PyDMF's default), no guards",
            DmfSettings {
                points: 5,
                guards: 0,
                ..DmfSettings::default()
            },
        ),
        (
            "7 points that stay at equal shares",
            DmfSettings {
                moves: None,
                ..DmfSettings::default()
            },
        ),
    ];
    let mut default = None;
    for (name, settings) in variants {
        let is_default = settings == DmfSettings::default();
        let mut machine = DirectMaxFlux::new(settings, &chain).unwrap();
        let (end, report, evaluations) = run(&mut machine, &objective);
        // The highest point of the path itself, followed in fine steps.
        let (t_top, e_top) = (0..=400)
            .map(|k| {
                let t = f64::from(k) / 400.0;
                let x = machine.path().at(t);
                (t, objective.evaluate(&x, false).unwrap().value)
            })
            .fold(
                (0.0, f64::NEG_INFINITY),
                |m, (t, e)| {
                    if e > m.1 {
                        (t, e)
                    } else {
                        m
                    }
                },
            );
        let top = machine.path().at(t_top);
        println!(
            "{name}: {end:?} after {} iterations, {evaluations} evaluations; estimate {:.4} \
             from the saddle, {:+.6} above it; the path itself highest {:.4} from the saddle, \
             {:+.6} above it",
            report.iteration + 1,
            col_norm(&(&report.highest - &saddle)),
            report.e_max - energy_at_the_saddle,
            col_norm(&(&top - &saddle)),
            e_top - energy_at_the_saddle,
        );
        if is_default {
            default = Some((end, report, evaluations, e_top, top));
        }
    }
    let Some((end, report, evaluations, e_top, top)) = default else {
        panic!("the default was not run")
    };
    // The gradient below the tolerance, or the top no longer coming down.
    assert!(
        matches!(end, DmfEnd::Converged | DmfEnd::Settled),
        "{end:?}"
    );
    // Five points and the two ends in the first iteration, five in every other.
    assert_eq!(evaluations, 2 + 5 * (report.iteration + 1));
    // The estimate of the transition state is near the saddle and not below it (a path
    // through the saddle has its top there; any other path is higher), and the path
    // itself does not climb higher between the points.
    let off = col_norm(&(&report.highest - &saddle));
    assert!(off < 0.05, "{off}");
    assert!(
        report.e_max > energy_at_the_saddle - 1e-4,
        "{}",
        report.e_max
    );
    assert!(
        report.e_max < energy_at_the_saddle + 2e-3,
        "{}",
        report.e_max
    );
    assert!(e_top < energy_at_the_saddle + 2e-3, "{e_top}");
    assert!(col_norm(&(&top - &saddle)) < 0.05);
    // The points gathered around the top, a guard on each side.
    let inner = &report.t[2..report.t.len() - 2];
    assert!(
        inner.iter().all(|t| (t - report.t_max).abs() < 0.25),
        "{inner:?}"
    );
}

#[test]
fn the_machine_refuses_what_it_cannot_do() {
    let chain = [col(&[0.0, 0.0]), col(&[1.0, 0.0])];
    let none = DmfSettings {
        points: 0,
        ..DmfSettings::default()
    };
    assert!(DirectMaxFlux::new(none, &chain).is_err());
    let cold = DmfSettings {
        beta: 0.0,
        ..DmfSettings::default()
    };
    assert!(DirectMaxFlux::new(cold, &chain).is_err());
    let same = [col(&[0.0, 0.0]), col(&[0.0, 0.0])];
    assert!(DirectMaxFlux::new(DmfSettings::default(), &same).is_err());
    let mut machine = DirectMaxFlux::new(DmfSettings::default(), &chain).unwrap();
    let DmfAction::Evaluate(points) = machine.ask().unwrap() else {
        panic!("the first question is a set of points")
    };
    assert_eq!(points.len(), 7);
    assert_eq!(points[0].t, 0.0);
    assert_eq!(points[6].t, 1.0);
    // Asked again before it was told, or told the wrong number of points.
    assert!(machine.ask().is_err());
    assert!(machine.tell(vec![(0.0, col(&[0.0, 0.0]))]).is_err());
}
