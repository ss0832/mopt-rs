//! The settings of high-index saddle dynamics compared on the seven-atom Lennard-Jones
//! cluster (built-in `lj`): upward searches from every transition state to index 2 and from
//! index-2 saddles to index 3, each refined at its order and checked by the Hessian. Per
//! setting: the share of searches that end at a stationary point of the index sought, the
//! share of those that end at another index, the duplicates, the distinct points found, and the
//! gradients per search (the differences of gradients included).
//!
//! The numbers are in `docs/source/high_index_saddles.md`; this test keeps the defaults
//! among the settings that find the most.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod hisd_common;

use hisd_common::{bipyramid, parallel, refine, search, settings, Reduced, Stationary};
use mopt_rs::path::hisd::landscape::{downward_starts, upward_starts, SearchStart};
use mopt_rs::path::hisd::{Curvature, DirectionUpdate, HandOff, HisdSettings, PointFlow};

fn distinct(points: impl IntoIterator<Item = Stationary>, order: usize) -> Vec<Stationary> {
    let mut out: Vec<Stationary> = Vec::new();
    for p in points {
        if p.index == order && p.near_zero == 0 && !out.iter().any(|q| q.same(&p)) {
            out.push(p);
        }
    }
    out
}

/// Upward starts from `from` along their `modes` softest stable modes.
fn upward(from: &[Stationary], modes: usize) -> Vec<SearchStart> {
    from.iter()
        .flat_map(|p| {
            let m = p.index;
            let stable = &p.modes[m..(m + modes).min(p.modes.len())];
            upward_starts(&p.x, &p.modes[..m], stable, 0.05).unwrap()
        })
        .collect()
}

/// Every transition state and minimum reachable from the global minimum by upward searches
/// from minima and downward searches from transition states (with the default settings).
fn transition_states(lj: &Reduced) -> (Vec<Stationary>, Vec<Stationary>) {
    let mut h = 0;
    let mut minima = vec![refine(lj, &bipyramid(), None, &mut h).unwrap()];
    let mut ts: Vec<Stationary> = Vec::new();
    loop {
        let starts = upward(&minima, 15);
        let found = parallel(&starts, &|s| search(lj, s, &settings));
        let before = ts.len();
        for p in distinct(found.into_iter().filter_map(|o| o.found), 1) {
            if !ts.iter().any(|q| q.same(&p)) {
                ts.push(p);
            }
        }
        let starts: Vec<SearchStart> = ts
            .iter()
            .flat_map(|p| downward_starts(&p.x, &p.modes[..1], 0.05, 0).unwrap())
            .collect();
        let found = parallel(&starts, &|s| search(lj, s, &settings));
        let mut grew = ts.len() > before;
        for p in distinct(found.into_iter().filter_map(|o| o.found), 0) {
            if !minima.iter().any(|q| q.same(&p)) {
                minima.push(p);
                grew = true;
            }
        }
        if !grew {
            return (minima, ts);
        }
    }
}

struct Row {
    name: &'static str,
    searches: usize,
    sought: usize,
    other_index: usize,
    failed: usize,
    distinct: usize,
    gradients: usize,
    hessians: usize,
}

fn measure(
    lj: &Reduced,
    name: &'static str,
    starts: &[SearchStart],
    make: &(dyn Fn(usize) -> HisdSettings + Sync),
) -> (Row, Vec<Stationary>) {
    let outcomes = parallel(starts, &|s| search(lj, s, make));
    let mut row = Row {
        name,
        searches: starts.len(),
        sought: 0,
        other_index: 0,
        failed: 0,
        distinct: 0,
        gradients: 0,
        hessians: 0,
    };
    let mut found = Vec::new();
    for (s, o) in starts.iter().zip(outcomes) {
        row.gradients += o.gradients;
        row.hessians += o.hessians;
        match o.found {
            Some(p) if p.index == s.order && p.near_zero == 0 => {
                row.sought += 1;
                found.push(p);
            }
            Some(_) => row.other_index += 1,
            None => row.failed += 1,
        }
    }
    let order = starts.first().map_or(0, |s| s.order);
    let unique = distinct(found, order);
    row.distinct = unique.len();
    (row, unique)
}

#[test]
fn settings_of_high_index_saddle_dynamics_on_lj7() {
    let lj = Reduced::new();
    let (minima, ts) = transition_states(&lj);
    println!("{} minima, {} transition states", minima.len(), ts.len());
    assert_eq!((minima.len(), ts.len()), (4, 12), "Doye and Massen 2005");
    let to_two = upward(&ts, 6);
    // The index-2 saddles of the default, for the searches to index 3.
    let (_, saddles) = measure(&lj, "default", &to_two, &settings);
    let to_three = upward(&saddles, 3);
    type Make = Box<dyn Fn(usize) -> HisdSettings + Sync>;
    let with = |f: fn(&mut HisdSettings)| -> Make {
        Box::new(move |k| {
            let mut s = settings(k);
            f(&mut s);
            s
        })
    };
    let configurations: Vec<(&'static str, Make)> = vec![
        ("HiOSD: euler, forward (default)", with(|_| {})),
        ("euler, central", with(|s| s.curvature = Curvature::Central)),
        ("euler, forward, guard", with(|s| s.guard = true)),
        ("oja, forward", with(|s| s.update = DirectionUpdate::OjaQr)),
        (
            "lobpcg, forward",
            with(|s| s.update = DirectionUpdate::Lobpcg),
        ),
        (
            "ggad, forward",
            with(|s| {
                s.update = DirectionUpdate::Ggad;
                s.first_along_gradient = true;
            }),
        ),
        (
            "eigenvectors, exact Hessian",
            with(|s| {
                s.update = DirectionUpdate::Eigenvectors;
                s.curvature = Curvature::Hessian;
            }),
        ),
        (
            "iHiSD upward",
            with(|s| {
                s.flow = PointFlow::Improved {
                    alpha0: 1e-7,
                    rate: 0.5,
                    upward: true,
                }
            }),
        ),
        (
            "hand-off at RMS 3e-3",
            with(|s| {
                s.hand_off = Some(HandOff {
                    rms_gradient: 3e-3,
                    projector_change: 0.1,
                })
            }),
        ),
        ("dimer length 1e-2", with(|s| s.dimer_length = 1e-2)),
    ];
    let mut best = [0usize; 2];
    let mut default_found = [0usize; 2];
    println!("| setting | index | searches | of the index sought | of another index | failed | distinct | gradients per search | Hessians per search |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for (name, make) in &configurations {
        for (k, starts) in [&to_two, &to_three].into_iter().enumerate() {
            let (row, _) = measure(&lj, name, starts, make.as_ref());
            println!(
                "| {} | {} | {} | {} ({:.0} %) | {} | {} | {} | {} | {:.1} |",
                row.name,
                k + 2,
                row.searches,
                row.sought,
                100.0 * row.sought as f64 / row.searches.max(1) as f64,
                row.other_index,
                row.failed,
                row.distinct,
                row.gradients / row.searches.max(1),
                row.hessians as f64 / row.searches.max(1) as f64,
            );
            best[k] = best[k].max(row.distinct);
            if name.contains("default") {
                default_found[k] = row.distinct;
            }
        }
    }
    // The default finds at least 80 % of the distinct points the best setting finds.
    for k in 0..2 {
        assert!(
            5 * default_found[k] >= 4 * best[k],
            "{default_found:?} {best:?}"
        );
    }
}
