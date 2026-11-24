//! The solution landscape of the seven-atom Lennard-Jones cluster: every stationary point of
//! index 0 to 10 found by the upward and downward searches of high-index saddle dynamics
//! (Yin, Yu, Zhang, Sci. China Math. 2021; HiOSD of Yin, Zhang, Zhang, SIAM J. Sci. Comput.
//! 2019) on the built-in `lj` calculator, counted by index and compared with the complete
//! counts of J. P. K. Doye and D. J. Wales, J. Chem. Phys. 2002, 116, 3777 (Table 4: 4, 12,
//! 44, 98, 168, 190, 168, 101, 45, 11, 1). The minima and transition states agree with
//! J. P. K. Doye and C. P. Massen, J. Chem. Phys. 2005, 122, 084105 (Table 1: 4 and 12).
//!
//! From every point found, in rounds until no new point appears: downward searches along plus
//! and minus every unstable mode (and a fan of four directions in the plane of the two of an
//! index-2 saddle), upward searches along plus and minus each of the six softest stable modes,
//! and four random combinations of modes each way (seeded by the energy of the point). Every
//! search ends with Newton steps on the exact Hessian at its order (without the rigid-body
//! motions).
//!
//! Stationary points are counted as geometrically distinct structures (permutations,
//! rotations and the inversion do not make a new one, the convention of those counts): two
//! are the same when their energies, sorted interatomic distances, and curvature spectra agree.
//! The index is the number of curvatures below -1e-6; a point with a curvature of magnitude
//! 1e-6 or less has an undecided index (one point: four negative curvatures and one of
//! magnitude 2.7e-7, whose sign depends on where along its flat direction it is refined), and
//! the counts are compared with that in mind.
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
use hisd_common::{bipyramid, parallel, refine, search, settings, Reduced, Stationary, N};
use mopt_rs::core_infra::rng::Rng;
use mopt_rs::path::hisd::landscape::{downward_starts, upward_starts, SearchStart};
use mopt_rs::path::hisd::subspace::gram_schmidt;

/// Doye and Wales 2002, Table 4, LJ7, index 0 to 10.
const REFERENCE: [usize; 11] = [4, 12, 44, 98, 168, 190, 168, 101, 45, 11, 1];
/// Stable modes of the upward searches (the softest).
const UPWARD: usize = 6;
/// Random combinations of modes per point, each way.
const RANDOM: usize = 4;

/// The starts of the searches from a stationary point.
fn starts_from(p: &Stationary, displacement: f64) -> Vec<SearchStart> {
    let m = p.index;
    let unstable: Vec<Col<f64>> = p.modes[..m].to_vec();
    let stable: Vec<Col<f64>> = p.modes[m..].to_vec();
    let mut out = Vec::new();
    if m >= 1 {
        let fan = if m == 2 { 4 } else { 0 };
        out.extend(downward_starts(&p.x, &unstable, displacement, fan).unwrap());
    }
    let up = UPWARD.min(stable.len());
    out.extend(upward_starts(&p.x, &unstable, &stable[..up], displacement).unwrap());
    let mut rng = Rng::from_seed(p.energy.to_bits());
    for _ in 0..RANDOM {
        if m >= 2 {
            let mut w = Col::<f64>::zeros(N);
            for u in &unstable {
                w += u * rng.standard_normal();
            }
            let mut with_w = vec![w];
            with_w.extend(unstable.iter().cloned());
            let basis = gram_schmidt(&with_w);
            let along = basis[0].clone();
            let kept: Vec<Col<f64>> = basis.into_iter().skip(1).collect();
            for sign in [1.0, -1.0] {
                out.push(SearchStart {
                    point: &p.x + &along * (sign * displacement),
                    directions: kept.clone(),
                    order: m - 1,
                    along: along.clone(),
                    sign,
                });
            }
        }
        let soft = stable.len().min(4);
        if soft > 0 {
            let mut w = Col::<f64>::zeros(N);
            for v in &stable[..soft] {
                w += v * rng.standard_normal();
            }
            out.extend(upward_starts(&p.x, &unstable, &[w], displacement).unwrap());
        }
    }
    out
}

#[test]
fn every_stationary_point_of_lj7_by_index() {
    let lj = Reduced::new();
    let mut hessians = 0;
    let first = refine(&lj, &bipyramid(), None, &mut hessians).unwrap();
    assert_eq!((first.index, first.near_zero), (0, 0));
    assert!((first.energy + 16.505384).abs() < 1e-5, "{}", first.energy);
    let mut known: Vec<Stationary> = vec![first.clone()];
    let mut undecided: Vec<Stationary> = Vec::new();
    let mut level = vec![first];
    let (mut searches, mut converged, mut wrong, mut duplicates, mut gradients) = (0, 0, 0, 0, 0);
    let mut ends = BTreeMap::<String, usize>::new();
    let mut failures = BTreeMap::<String, usize>::new();
    let mut rounds = 0;
    while !level.is_empty() {
        rounds += 1;
        let starts: Vec<SearchStart> = level.iter().flat_map(|p| starts_from(p, 0.05)).collect();
        let outcomes = parallel(&starts, &|s| search(&lj, s, &settings));
        let mut next = Vec::new();
        for (start, outcome) in starts.iter().zip(outcomes) {
            searches += 1;
            gradients += outcome.gradients;
            hessians += outcome.hessians;
            *ends
                .entry(outcome.end.map_or("none".to_string(), |e| format!("{e:?}")))
                .or_default() += 1;
            if let Some(f) = outcome.failure {
                *failures.entry(format!("{f:?}")).or_default() += 1;
            }
            let Some(found) = outcome.found else {
                continue;
            };
            converged += 1;
            if found.near_zero > 0 {
                // Along its flat direction such a point is not located (its copies differ by
                // more than 1e-2 in their distances): the same energy and the same numbers of
                // negative and vanishing curvatures make it the same.
                let alike = |k: &Stationary| {
                    (k.energy - found.energy).abs() < 1e-6
                        && (k.index, k.near_zero) == (found.index, found.near_zero)
                };
                if !undecided.iter().any(alike) {
                    undecided.push(found);
                }
                continue;
            }
            if found.index != start.order {
                wrong += 1;
            }
            if known.iter().any(|k| k.same(&found)) {
                duplicates += 1;
                continue;
            }
            known.push(found.clone());
            next.push(found);
        }
        level = next;
    }
    let mut counts = [0usize; 11];
    for k in &known {
        assert!(k.index <= 10, "index {}", k.index);
        counts[k.index] += 1;
    }
    println!(
        "LJ7: {rounds} rounds, {searches} searches, {converged} converged, {wrong} of another \
         index, {duplicates} duplicates, {gradients} gradients, {hessians} Hessians; ends \
         {ends:?}; failures {failures:?}"
    );
    println!("counts by index: {counts:?}");
    println!("reference:       {REFERENCE:?}");
    for d in &undecided {
        println!(
            "undecided: energy {:.7}, {} negative curvatures, {} of magnitude at most 1e-6",
            d.energy, d.index, d.near_zero
        );
    }
    // The saddle points of index 2 and 3 found by GGAD (Quapp, Bofill 2014). Their point of
    // "index 4" at -13.708 is of index 3 by the Hessian here.
    for (index, energy) in [(2, -14.723), (3, -14.348), (3, -13.708)] {
        assert!(
            known
                .iter()
                .any(|k| k.index == index && (k.energy - energy).abs() < 1e-3),
            "index {index} at {energy}"
        );
    }
    // No count exceeds the reference (every point found is a distinct stationary point of its
    // index), and the minima, transition states and saddle points of index 2 and 3 are complete
    // once the undecided points are given an index they can have.
    //
    // The searches do not reach every point of the higher indices in every run, and which
    // points they miss is decided by digits far below any tolerance (the seeds of the random
    // combinations are the bits of the energies; the Bohr radius of CODATA 2018 instead of
    // 2014, 4e-10 of itself, changed the outcome). The single point of index 10 (energy
    // -6.169) was found only by a search for index 9 from a point of index 8 that converged to
    // index 10 instead; the upward searches from all 11 points of index 9 along each of their
    // stable modes (with displacements of 0.02 to 0.2, up to 3000 steps, and the cluster held
    // together up to distances of 2, 2.5 or 3) end with the cluster apart or leaving the index,
    // and none reaches it. One point of index 6 (-9.4014259) is found only along a random
    // combination of soft modes. Over six seeds of the random combinations, with four or eight
    // combinations per point, 0 to 4 of the 842 points were missing, all of index 4 or higher.
    // The run here is checked to miss at most 5.
    for index in 0..=10 {
        let possible = undecided
            .iter()
            .filter(|d| d.index <= index && index <= d.index + d.near_zero)
            .count();
        assert!(
            counts[index] <= REFERENCE[index],
            "index {index}: {} found, {} in the reference",
            counts[index],
            REFERENCE[index]
        );
        if index <= 3 {
            assert!(
                REFERENCE[index] <= counts[index] + possible,
                "index {index}: {} found, {possible} undecided, {} in the reference",
                counts[index],
                REFERENCE[index]
            );
        }
    }
    let total = REFERENCE.iter().sum::<usize>();
    let found = known.len() + undecided.len();
    assert!(found <= total, "{found} of {total}");
    assert!(total - found <= 5, "{found} of {total}");
}
