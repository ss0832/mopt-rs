//! Tests of the high-index saddle searches on surfaces with known saddle points of index 2
//! and 3.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::landscape::{downward_starts, upward_starts};
use super::*;
use crate::math::linalg::symmetric_eigen;

/// A surface of plain variables: energy, gradient, Hessian.
pub(crate) type Surface = dyn Fn(&Col<f64>) -> (f64, Col<f64>, Mat<f64>);

/// Run a search to its end; the number of evaluations.
pub(crate) fn run(search: &mut HighIndexSaddleSearch, surface: &Surface) -> usize {
    let mut count = 0;
    while let SearchAction::Evaluate(points) = search.ask() {
        count += points.len();
        let values = points
            .iter()
            .map(|p| {
                let (energy, gradient, hessian) = surface(&p.x);
                PointValue {
                    energy,
                    gradient: Some(gradient),
                    hessian: p.hessian.then_some(hessian),
                }
            })
            .collect();
        search.tell(values).unwrap();
    }
    count
}

/// An orthogonal 4 x 4 matrix (a product of plane rotations).
fn rotation() -> Mat<f64> {
    let mut r = Mat::<f64>::identity(4, 4);
    for (i, j, t) in [
        (0, 1, 0.4),
        (1, 2, -0.7),
        (2, 3, 0.3),
        (0, 3, 1.1),
        (1, 3, 0.5),
    ] {
        let (c, s) = (libm::cos(t), libm::sin(t));
        let g = Mat::<f64>::from_fn(4, 4, |a, b| match (a, b) {
            _ if a == i && b == i => c,
            _ if a == j && b == j => c,
            _ if a == i && b == j => -s,
            _ if a == j && b == i => s,
            _ if a == b => 1.0,
            _ => 0.0,
        });
        r = &g * &r;
    }
    r
}

/// `E = sum_i a_i (y_i^2 - 1)^2` with `y = R x`: stationary points at `y_i` in
/// `{-1, 0, 1}`, of index the number of zeros (81 points; 24 of index 2, 8 of index 3).
pub(crate) fn rotated_quartic() -> (Box<Surface>, Mat<f64>) {
    let r = rotation();
    let a = [1.0, 1.7, 2.3, 3.1];
    let rr = r.clone();
    let surface = move |x: &Col<f64>| {
        let y = &rr * x;
        let energy = (0..4).map(|i| a[i] * (y[i] * y[i] - 1.0).powi(2)).sum();
        let gy = Col::from_fn(4, |i| 4.0 * a[i] * y[i] * (y[i] * y[i] - 1.0));
        let hy = Mat::from_fn(4, 4, |i, j| {
            if i == j {
                a[i] * (12.0 * y[i] * y[i] - 4.0)
            } else {
                0.0
            }
        });
        (energy, rr.transpose() * gy, rr.transpose() * hy * &rr)
    };
    (Box::new(surface), r)
}

/// The point of `rotated_quartic` with `y = target`.
fn quartic_point(r: &Mat<f64>, target: [f64; 4]) -> Col<f64> {
    r.transpose() * Col::from_fn(4, |i| target[i])
}

/// The test function of Yin, Zhang, Zhang (2019), as Chu and Lin (2025) restate it:
/// `B_k = B - sum_(i<=k) s_i atan(x_i - x*_i)^2 + sum_(i>k) s_i atan(x_i - x*_i)^2`, `B` the
/// sum of squares of Biggs' exponential fit; `x* = (1, 10, 1, 5, 4, 3)` is a saddle point of
/// index `k`.
pub(crate) fn biggs(k: usize) -> Box<Surface> {
    const S: [f64; 6] = [4.0, 8.0, 16.0, 8.0, 4.0, 2.0];
    const STAR: [f64; 6] = [1.0, 10.0, 1.0, 5.0, 4.0, 3.0];
    let gradient_at = move |x: &Col<f64>| {
        let mut e = 0.0;
        let mut g = Col::<f64>::zeros(6);
        for i in 1..=6 {
            let t = i as f64 / 10.0;
            let y = libm::exp(-t) - 5.0 * libm::exp(-10.0 * t) + 3.0 * libm::exp(-4.0 * t);
            let (e1, e2, e5) = (
                libm::exp(-t * x[0]),
                libm::exp(-t * x[1]),
                libm::exp(-t * x[4]),
            );
            let r = x[2] * e1 - x[3] * e2 + x[5] * e5 - y;
            let dr = [-t * x[2] * e1, t * x[3] * e2, e1, -e2, -t * x[5] * e5, e5];
            e += r * r;
            for j in 0..6 {
                g[j] += 2.0 * r * dr[j];
            }
        }
        for j in 0..6 {
            let sign = if j < k { -1.0 } else { 1.0 };
            let d = x[j] - STAR[j];
            let a = libm::atan(d);
            e += sign * S[j] * a * a;
            g[j] += sign * S[j] * 2.0 * a / (1.0 + d * d);
        }
        (e, g)
    };
    Box::new(move |x: &Col<f64>| {
        let (e, g) = gradient_at(x);
        // Central differences of the analytic gradient.
        let h = 1e-5;
        let mut hess = Mat::<f64>::zeros(6, 6);
        for j in 0..6 {
            let mut p = x.clone();
            let mut m = x.clone();
            p[j] += h;
            m[j] -= h;
            let d = (gradient_at(&p).1 - gradient_at(&m).1) * (0.5 / h);
            for i in 0..6 {
                hess[(i, j)] = d[i];
            }
        }
        let hess = (&hess + hess.transpose()) * 0.5;
        (e, g, hess)
    })
}

/// `f = x^2 - y^2 - z^2 + x^3 - 3 x y^2 - 3 x z^2 - 3 y z^2` (Chu, Lin 2025, eq. 5.3): an
/// index-2 saddle at the origin, an index-1 saddle at `(-2/3, 0, 0)`.
pub(crate) fn cubic() -> Box<Surface> {
    Box::new(|p: &Col<f64>| {
        let (x, y, z) = (p[0], p[1], p[2]);
        let e =
            x * x - y * y - z * z + x * x * x - 3.0 * x * y * y - 3.0 * x * z * z - 3.0 * y * z * z;
        let g = Col::from_fn(3, |i| {
            [
                2.0 * x + 3.0 * x * x - 3.0 * y * y - 3.0 * z * z,
                -2.0 * y - 6.0 * x * y - 3.0 * z * z,
                -2.0 * z - 6.0 * x * z - 6.0 * y * z,
            ][i]
        });
        let h = Mat::from_fn(3, 3, |i, j| {
            [
                [2.0 + 6.0 * x, -6.0 * y, -6.0 * z],
                [-6.0 * y, -2.0 - 6.0 * x, -6.0 * z],
                [-6.0 * z, -6.0 * z, -2.0 - 6.0 * x - 6.0 * y],
            ][i][j]
        });
        (e, g, h)
    })
}

fn index_at(surface: &Surface, x: &Col<f64>) -> usize {
    let (_, _, h) = surface(x);
    symmetric_eigen(&h)
        .unwrap()
        .0
        .iter()
        .filter(|v| **v < 0.0)
        .count()
}

fn tight(mut s: HisdSettings) -> HisdSettings {
    s.criteria = crate::opt::convergence::ConvergenceCriteria::new(1e-8, 1e-8, 1.0, 1.0);
    s.step.max_step = 0.1;
    s.max_iterations = 3000;
    s
}

#[test]
fn every_update_and_curvature_converges_to_the_index_two_saddle_of_the_cubic() {
    let f = cubic();
    let start = Col::from_fn(3, |i| [0.08, -0.05, 0.06][i]);
    let directions = [
        Col::from_fn(3, |i| [0.3, 1.0, 0.2][i]),
        Col::from_fn(3, |i| [-0.2, 0.1, 1.0][i]),
    ];
    for (update, curvature) in [
        (DirectionUpdate::HisdEuler, Curvature::Forward),
        (DirectionUpdate::HisdEuler, Curvature::Central),
        (DirectionUpdate::OjaQr, Curvature::Forward),
        (DirectionUpdate::Lobpcg, Curvature::Forward),
        (DirectionUpdate::Ggad, Curvature::Forward),
        (DirectionUpdate::Eigenvectors, Curvature::Hessian),
        (DirectionUpdate::HisdEuler, Curvature::Hessian),
    ] {
        let mut s = tight(HisdSettings::hiosd(2));
        s.update = update;
        s.curvature = curvature;
        let mut search = HighIndexSaddleSearch::new(s, &start, &directions).unwrap();
        run(&mut search, f.as_ref());
        assert_eq!(
            search.end(),
            Some(HisdEnd::Converged),
            "{update:?} {curvature:?}"
        );
        assert!(
            col_norm(search.point()) < 1e-6,
            "{update:?} {:?}",
            search.point()
        );
        assert_eq!(index_at(f.as_ref(), search.point()), 2);
        assert!(search.curvatures()[..2].iter().all(|c| *c < -1.0));
    }
    // The index-1 saddle with one direction.
    let start = Col::from_fn(3, |i| [-0.6, 0.05, -0.04][i]);
    let mut search = HighIndexSaddleSearch::new(
        tight(HisdSettings::hiosd(1)),
        &start,
        &[Col::from_fn(3, |i| [1.0, 0.2, 0.0][i])],
    )
    .unwrap();
    run(&mut search, f.as_ref());
    assert_eq!(search.end(), Some(HisdEnd::Converged));
    assert!((search.point()[0] + 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn index_two_and_three_saddles_of_the_rotated_quartic_with_a_guard() {
    let (f, r) = rotated_quartic();
    let mut rng = Rng::from_seed(7);
    for (target, k) in [
        ([0.0, 0.0, 1.0, -1.0], 2),
        ([1.0, 0.0, 0.0, 1.0], 2),
        ([0.0, 0.0, 0.0, 1.0], 3),
        ([0.0, -1.0, 0.0, 0.0], 3),
    ] {
        let exact = quartic_point(&r, target);
        let start = Col::from_fn(4, |i| exact[i] + 0.15 * rng.standard_normal());
        let directions: Vec<Col<f64>> = (0..k)
            .map(|_| Col::from_fn(4, |_| rng.standard_normal()))
            .collect();
        let mut s = tight(HisdSettings::hiosd(k));
        s.guard = true;
        let mut search = HighIndexSaddleSearch::new(s, &start, &directions).unwrap();
        run(&mut search, f.as_ref());
        assert_eq!(search.end(), Some(HisdEnd::Converged), "{target:?}");
        assert!(col_norm(&(search.point() - &exact)) < 1e-6, "{target:?}");
        assert_eq!(index_at(f.as_ref(), search.point()), k);
        // The guard follows the (k+1)-th curvature, positive here.
        assert!(search.curvatures()[k] > 0.0);
    }
}

#[test]
fn the_six_dimensional_test_function_of_hiosd() {
    for k in [2, 3] {
        let f = biggs(k);
        let star = Col::from_fn(6, |i| [1.0, 10.0, 1.0, 5.0, 4.0, 3.0][i]);
        assert_eq!(index_at(f.as_ref(), &star), k);
        let mut rng = Rng::from_seed(11 + k as u64);
        let start = Col::from_fn(6, |i| star[i] + 0.1 * rng.standard_normal());
        let directions: Vec<Col<f64>> = (0..k)
            .map(|j| Col::from_fn(6, |i| if i == j { 1.0 } else { 0.0 }))
            .collect();
        let mut search =
            HighIndexSaddleSearch::new(tight(HisdSettings::hiosd(k)), &start, &directions).unwrap();
        run(&mut search, f.as_ref());
        assert_eq!(search.end(), Some(HisdEnd::Converged), "k = {k}");
        assert!(
            col_norm(&(search.point() - &star)) < 1e-5,
            "{:?}",
            search.point()
        );
    }
}

#[test]
fn ggad_starts_along_the_gradient_and_ihisd_starts_as_a_descent() {
    let (f, r) = rotated_quartic();
    let exact = quartic_point(&r, [0.0, 0.0, 1.0, 1.0]);
    let start = &exact + Col::from_fn(4, |i| [0.1, -0.08, 0.05, 0.02][i]);
    let directions = [
        Col::from_fn(4, |i| [1.0, 0.0, 0.0, 0.0][i]),
        Col::from_fn(4, |i| [0.0, 1.0, 0.0, 0.0][i]),
    ];
    let mut ggad =
        HighIndexSaddleSearch::new(tight(HisdSettings::ggad(2)), &start, &directions).unwrap();
    run(&mut ggad, f.as_ref());
    assert_eq!(ggad.end(), Some(HisdEnd::Converged));
    assert!(col_norm(&(ggad.point() - &exact)) < 1e-6);
    let mut s = tight(HisdSettings::hiosd(2));
    s.flow = PointFlow::Improved {
        alpha0: 1e-7,
        rate: 0.5,
        upward: false,
    };
    let mut improved = HighIndexSaddleSearch::new(s, &start, &directions).unwrap();
    run(&mut improved, f.as_ref());
    assert_eq!(improved.end(), Some(HisdEnd::Converged));
    assert!(col_norm(&(improved.point() - &exact)) < 1e-6);
}

#[test]
fn hand_off_and_wrong_index_ends() {
    let (f, r) = rotated_quartic();
    let exact = quartic_point(&r, [0.0, 1.0, 0.0, -1.0]);
    let start = &exact + Col::from_fn(4, |i| [0.1, 0.05, -0.1, 0.03][i]);
    let directions = [
        Col::from_fn(4, |i| [1.0, 0.2, 0.0, 0.0][i]),
        Col::from_fn(4, |i| [0.0, 0.3, 1.0, 0.0][i]),
    ];
    let mut s = tight(HisdSettings::hiosd(2));
    s.hand_off = Some(HandOff {
        rms_gradient: 0.05,
        projector_change: 0.05,
    });
    let mut search = HighIndexSaddleSearch::new(s, &start, &directions).unwrap();
    run(&mut search, f.as_ref());
    assert_eq!(search.end(), Some(HisdEnd::HandOff));
    assert!(col_norm(&(search.point() - &exact)) < 0.1);
    let handed = search.iterations();
    // A search of index 2 that starts at a transition state stops there: a stationary point
    // of another index.
    let ts = quartic_point(&r, [1.0, 0.0, -1.0, 1.0]);
    let mut search =
        HighIndexSaddleSearch::new(tight(HisdSettings::hiosd(2)), &ts, &directions).unwrap();
    run(&mut search, f.as_ref());
    assert_eq!(search.end(), Some(HisdEnd::WrongIndex));
    assert!(handed > 0);
}

#[test]
fn upward_from_a_transition_state_and_downward_from_an_index_two_saddle() {
    let (f, r) = rotated_quartic();
    // From the transition state y = (1, 0, 1, 1) along its lowest positive mode.
    let ts = quartic_point(&r, [1.0, 0.0, 1.0, 1.0]);
    let (_, _, h) = f(&ts);
    let (values, vectors) = symmetric_eigen(&h).unwrap();
    assert!(values[0] < 0.0 && values[1] > 0.0);
    let starts = upward_starts(
        &ts,
        &[vectors.col(0).to_owned()],
        &[vectors.col(1).to_owned()],
        0.1,
    )
    .unwrap();
    for start in &starts {
        let mut search = HighIndexSaddleSearch::new(
            tight(HisdSettings::hiosd(start.order)),
            &start.point,
            &start.directions,
        )
        .unwrap();
        run(&mut search, f.as_ref());
        assert_eq!(search.end(), Some(HisdEnd::Converged));
        assert_eq!(index_at(f.as_ref(), search.point()), 2);
    }
    // From the index-2 saddle y = (0, 0, 1, 1) down to its four transition states.
    let top = quartic_point(&r, [0.0, 0.0, 1.0, 1.0]);
    let (_, _, h) = f(&top);
    let (_, vectors) = symmetric_eigen(&h).unwrap();
    let unstable = [vectors.col(0).to_owned(), vectors.col(1).to_owned()];
    let mut found: Vec<Col<f64>> = Vec::new();
    for start in downward_starts(&top, &unstable, 0.1, 0).unwrap() {
        let mut search = HighIndexSaddleSearch::new(
            tight(HisdSettings::hiosd(start.order)),
            &start.point,
            &start.directions,
        )
        .unwrap();
        run(&mut search, f.as_ref());
        assert_eq!(search.end(), Some(HisdEnd::Converged));
        assert_eq!(index_at(f.as_ref(), search.point()), 1);
        if found.iter().all(|p| col_norm(&(p - search.point())) > 1e-4) {
            found.push(search.point().clone());
        }
    }
    assert_eq!(found.len(), 4);
}

#[test]
fn bad_settings_and_starts_are_refused() {
    let x = Col::<f64>::zeros(3);
    let d = [Col::from_fn(3, |i| [1.0, 0.0, 0.0][i])];
    assert!(HighIndexSaddleSearch::new(HisdSettings::hiosd(1), &x, &d).is_ok());
    // Every reflected direction must be given.
    assert!(HighIndexSaddleSearch::new(HisdSettings::hiosd(2), &x, &d).is_err());
    assert!(HighIndexSaddleSearch::new(HisdSettings::hiosd(0), &x, &d).is_err());
    let mut s = HisdSettings::hiosd(1);
    s.update = DirectionUpdate::Eigenvectors;
    assert!(HighIndexSaddleSearch::new(s, &x, &d).is_err());
    let mut s = HisdSettings::hiosd(1);
    s.guard = true;
    let search = HighIndexSaddleSearch::new(s, &x, &d).unwrap();
    assert_eq!(search.directions().len(), 2);
    assert!(HighIndexSaddleSearch::new(HisdSettings::hiosd(4), &x, &d).is_err());
    assert!(DirectionUpdate::from_name("lobpcg").is_ok());
    assert!(DirectionUpdate::from_name("other").is_err());
}
