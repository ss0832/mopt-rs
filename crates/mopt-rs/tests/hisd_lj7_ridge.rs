//! The bands on the ridge between two transition states of the seven-atom Lennard-Jones
//! cluster (built-in `lj`), and high-index saddle dynamics from the same information: from
//! index-2 saddles found by upward searches, the transition states below each (downward
//! along plus and minus each unstable mode), then for every pair of them
//!
//! - the Hessian form on the chain relaxation of `nebmain` (exact Hessians of the images in
//!   every iteration, quasi-Newton steps of the image function reversed along the transverse
//!   mode, `path::band::ridge::hessian_ridge_settings`),
//! - the moving-frame band from gradients alone (`RidgeBand`),
//! - the fixed-direction band on the mode-maximized surface (`FixedRidgeBand`, for the pairs
//!   whose modes are alike),
//! - an index-2 search of high-index saddle dynamics started at the middle of the band's
//!   first chain with the interpolated mode and the direction of the chain.
//!
//! A method succeeds on a pair when its point, refined at order 2, is the saddle the pair came
//! from. Pairs from the same unstable mode (`+u_a`, `-u_a`) lie side by side on one ridge
//! (their imaginary modes are both about the other unstable mode); pairs from different modes
//! share a minimum and start from a band over a corner.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod hisd_common;

use std::collections::BTreeMap;

use faer::Col;
use hisd_common::{bipyramid, parallel, refine, search, settings, Reduced, Stationary};
use mopt_rs::math::linalg::col_dot;
use mopt_rs::path::band::machine::{ChainAction, ChainRelaxation, ImageValue};
use mopt_rs::path::band::ridge::{
    cornered_band, hessian_ridge_settings, straight_band, RidgeBand, RidgeBandSettings,
};
use mopt_rs::path::band::ridge_fixed::{FixedRidgeBand, FixedRidgeSettings};
use mopt_rs::path::band::ChainKind;
use mopt_rs::path::hisd::landscape::{downward_starts, upward_starts, SearchStart};
use mopt_rs::path::hisd::subspace::gram_schmidt;
use mopt_rs::path::points::SearchAction;

fn distinct(points: Vec<Stationary>, order: usize) -> Vec<Stationary> {
    let mut out: Vec<Stationary> = Vec::new();
    for p in points {
        if p.index == order && p.near_zero == 0 && !out.iter().any(|q| q.same(&p)) {
            out.push(p);
        }
    }
    out
}

/// Upward from every point of `from` along its `modes` softest stable modes.
fn upward(lj: &Reduced, from: &[Stationary], modes: usize) -> Vec<Stationary> {
    let starts: Vec<SearchStart> = from
        .iter()
        .flat_map(|p| {
            let m = p.index;
            let stable = &p.modes[m..(m + modes).min(p.modes.len())];
            upward_starts(&p.x, &p.modes[..m], stable, 0.05).unwrap()
        })
        .collect();
    let order = from.first().map_or(1, |p| p.index + 1);
    distinct(
        parallel(&starts, &|s| search(lj, s, &settings))
            .into_iter()
            .filter_map(|o| o.found)
            .collect(),
        order,
    )
}

/// A pair of transition states below an index-2 saddle.
struct Pair {
    saddle: usize,
    first: Stationary,
    last: Stationary,
    adjacent: bool,
}

impl Pair {
    fn modes(&self) -> [Col<f64>; 2] {
        [self.first.modes[0].clone(), self.last.modes[0].clone()]
    }

    /// The first chain: straight, or over the corner for a pair that shares a minimum.
    fn chain(&self) -> Vec<Col<f64>> {
        let [a, b] = self.modes();
        if self.adjacent {
            cornered_band(&self.first.x, &self.last.x, &a, &b, 7)
                .unwrap_or_else(|_| straight_band(&self.first.x, &self.last.x, 7))
        } else {
            straight_band(&self.first.x, &self.last.x, 7)
        }
    }
}

/// What a method did on one pair.
struct Attempt {
    hit: bool,
    other: bool,
    gradients: usize,
    hessians: usize,
    end: String,
}

#[derive(Default)]
struct Tally {
    attempts: usize,
    successes: usize,
    other: usize,
    gradients: usize,
    hessians: usize,
    ends: BTreeMap<String, usize>,
}

#[test]
fn ridge_bands_and_hisd_between_lj7_transition_states() {
    let lj = Reduced::new();
    let mut hessians = 0;
    let minimum = refine(&lj, &bipyramid(), None, &mut hessians).unwrap();
    let transition_states = upward(&lj, &[minimum], 15);
    let saddles = upward(&lj, &transition_states, 3);
    println!(
        "{} transition states, {} index-2 saddles above them",
        transition_states.len(),
        saddles.len()
    );
    assert!(saddles.len() >= 5);
    // The transition states below every saddle.
    let starts: Vec<SearchStart> = saddles
        .iter()
        .flat_map(|p| downward_starts(&p.x, &p.modes[..2], 0.05, 0).unwrap())
        .collect();
    let below = parallel(&starts, &|s| search(&lj, s, &settings));
    let mut pairs = Vec::new();
    for (k, _) in saddles.iter().enumerate() {
        // Starts 0, 1: +-u_1; 2, 3: +-u_2 (the order of `downward_starts`).
        let found: Vec<Option<Stationary>> = (0..4)
            .map(|j| {
                below[4 * k + j]
                    .found
                    .clone()
                    .filter(|p| p.index == 1 && p.near_zero == 0)
            })
            .collect();
        for (a, b, adjacent) in [
            (0, 1, false),
            (2, 3, false),
            (0, 2, true),
            (0, 3, true),
            (1, 2, true),
            (1, 3, true),
        ] {
            if let (Some(first), Some(last)) = (&found[a], &found[b]) {
                if !first.same(last) {
                    pairs.push(Pair {
                        saddle: k,
                        first: first.clone(),
                        last: last.clone(),
                        adjacent,
                    });
                }
            }
        }
    }
    let judge = |pair: &Pair, top: Option<Col<f64>>, gradients, hessians, end: String| {
        let mut h = 0;
        let refined = top.and_then(|x| refine(&lj, &x, Some(2), &mut h).ok());
        let target = &saddles[pair.saddle];
        Attempt {
            hit: refined.as_ref().is_some_and(|r| r.same(target)),
            other: refined
                .as_ref()
                .is_some_and(|r| !r.same(target) && r.index == 2 && r.near_zero == 0),
            gradients,
            hessians,
            end,
        }
    };
    type Method<'a> = Box<dyn Fn(&Pair) -> Option<Attempt> + Sync + 'a>;
    let methods: Vec<(&str, Method<'_>)> = vec![
        (
            "Hessian form (chain relaxation of nebmain)",
            Box::new(|pair: &Pair| {
                let mut settings = hessian_ridge_settings(pair.modes(), ChainKind::Atoms);
                settings.force_tolerance = Some(1e-3);
                // Tuned in reduced units (spring constants in epsilon/sigma^2, steps in
                // sigma): stronger springs, shorter steps, and a smaller share of the
                // quasi-Newton step than the defaults of nebmain found the most saddles.
                settings.springs = Some(mopt_rs::path::band::springs::EnergyWeightedSprings {
                    upper: 1.0,
                    lower: 0.1,
                });
                if let Some(rfo) = settings.rfo.as_mut() {
                    rfo.inner_step = 0.03;
                    rfo.ratio = 0.3;
                }
                let mut band = ChainRelaxation::new(settings, pair.chain())
                    .and_then(|b| b.with_structure(lj.template.clone()))
                    .unwrap();
                let (mut gradients, mut hessians, mut last) = (0, 0, None);
                while let ChainAction::Evaluate(request) = band.ask() {
                    let values: Vec<ImageValue> = request
                        .images
                        .iter()
                        .map(|x| {
                            let (energy, gradient, hessian) = lj.eval(x, request.need_hessian);
                            gradients += 1;
                            hessians += usize::from(request.need_hessian);
                            ImageValue {
                                energy,
                                gradient,
                                hessian,
                            }
                        })
                        .collect();
                    last = Some(band.tell(values).unwrap());
                }
                let report = last?;
                let top = (1..report.energies.len() - 1)
                    .max_by(|a, b| report.energies[*a].total_cmp(&report.energies[*b]))?;
                Some(judge(
                    pair,
                    Some(report.chain[top].clone()),
                    gradients,
                    hessians,
                    format!("{:?}", band.end()),
                ))
            }),
        ),
        (
            "moving frame from gradients (RidgeBand)",
            Box::new(|pair: &Pair| {
                let settings = RidgeBandSettings {
                    spring: 1.0,
                    tolerance: 1e-3,
                    max_iterations: 600,
                    ..RidgeBandSettings::default()
                };
                let mut band = RidgeBand::new(settings, pair.chain(), pair.modes())
                    .and_then(|b| b.with_structure(lj.template.clone(), true, false))
                    .unwrap();
                let (mut gradients, mut end) = (0, None);
                while let SearchAction::Evaluate(points) = band.ask() {
                    let r = band.tell(lj.values(&points)).unwrap();
                    gradients = r.gradients;
                    end = r.end;
                }
                let top = band.chain().get(band.highest()).cloned();
                Some(judge(pair, top, gradients, 0, format!("{end:?}")))
            }),
        ),
        (
            "fixed direction (FixedRidgeBand)",
            Box::new(|pair: &Pair| {
                if pair.adjacent {
                    return None;
                }
                let mut settings = FixedRidgeSettings::default();
                settings.chain.force_tolerance = Some(1e-3);
                settings.chain.max_iterations = 300;
                let [a, b] = pair.modes();
                let band = FixedRidgeBand::new(settings, &pair.first.x, &pair.last.x, &a, &b, 7)
                    .and_then(|b| b.with_structure(lj.template.clone(), true, false));
                let Ok(mut band) = band else {
                    return Some(Attempt {
                        hit: false,
                        other: false,
                        gradients: 0,
                        hessians: 0,
                        end: "modes apart".to_string(),
                    });
                };
                while let SearchAction::Evaluate(points) = band.ask() {
                    band.tell(lj.values(&points)).unwrap();
                }
                let gradients = band.gradient_evaluations();
                Some(judge(
                    pair,
                    band.saddle_point(),
                    gradients,
                    0,
                    format!("{:?}", band.end()),
                ))
            }),
        ),
        (
            "HiSD from the middle of the first chain",
            Box::new(|pair: &Pair| {
                let chain = pair.chain();
                let middle = chain[chain.len() / 2].clone();
                let [a, b] = pair.modes();
                let sign = if col_dot(&a, &b) < 0.0 { -1.0 } else { 1.0 };
                let tangent = &chain[chain.len() / 2 + 1] - &chain[chain.len() / 2 - 1];
                let directions = gram_schmidt(&[&a + &b * sign, tangent]);
                let start = SearchStart {
                    point: middle,
                    directions,
                    order: 2,
                    along: Col::zeros(a.nrows()),
                    sign: 1.0,
                };
                let o = search(&lj, &start, &settings);
                let top = o.found.as_ref().map(|p| p.x.clone());
                Some(judge(
                    pair,
                    top,
                    o.gradients,
                    o.hessians,
                    format!("{:?}", o.end),
                ))
            }),
        ),
    ];
    let mut tallies: BTreeMap<String, Tally> = BTreeMap::new();
    for (name, method) in &methods {
        let attempts = parallel(&pairs, &|pair: &Pair| method(pair));
        for (pair, attempt) in pairs.iter().zip(attempts) {
            let Some(a) = attempt else { continue };
            let kind = if pair.adjacent {
                "adjacent"
            } else {
                "side by side"
            };
            let t = tallies.entry(format!("{name}, {kind}")).or_default();
            t.attempts += 1;
            t.successes += usize::from(a.hit);
            t.other += usize::from(a.other);
            t.gradients += a.gradients;
            t.hessians += a.hessians;
            *t.ends.entry(a.end).or_default() += 1;
        }
    }
    println!("| method, pairs | pairs | at the saddle of the pair | at another index-2 saddle | gradients per pair | Hessians per pair | ends |");
    println!("|---|---|---|---|---|---|---|");
    for (name, t) in &tallies {
        println!(
            "| {name} | {} | {} | {} | {} | {} | {:?} |",
            t.attempts,
            t.successes,
            t.other,
            t.gradients / t.attempts.max(1),
            t.hessians / t.attempts.max(1),
            t.ends
        );
    }
    // What is measured (docs/source/high_index_saddles.md), with some room.
    for (name, share) in [
        (
            "Hessian form (chain relaxation of nebmain), side by side",
            0.5,
        ),
        ("moving frame from gradients (RidgeBand), side by side", 0.8),
        ("HiSD from the middle of the first chain, side by side", 0.8),
    ] {
        let t = &tallies[name];
        assert!(t.attempts >= 5, "{name}");
        assert!(
            t.successes as f64 >= share * t.attempts as f64,
            "{name}: {}/{}",
            t.successes,
            t.attempts
        );
    }
}
