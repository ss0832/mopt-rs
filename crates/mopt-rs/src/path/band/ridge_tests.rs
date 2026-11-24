//! Tests of the bands on the ridge between two transition states.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::*;
use crate::math::linalg::symmetric_eigen;
use faer::Mat;

/// `E = sum_i a_i (y_i^2 - 1)^2` with `y = R x` (four variables, `R` orthogonal):
/// stationary points at `y_i` in `{-1, 0, 1}`, of index the number of zeros.
pub(crate) fn quartic() -> (impl Fn(&Col<f64>) -> PointValue, Mat<f64>) {
    let mut r = Mat::<f64>::identity(4, 4);
    for (i, j, t) in [(0, 1, 0.4), (1, 2, -0.7), (2, 3, 0.3), (0, 3, 1.1)] {
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
    // The in-band curvature at the index-2 saddle (-4 a_1 = -6.8 along y_1) is below the
    // transverse one (-4 a_0 = -4) for the pair joined along y_1.
    let a = [1.0, 1.7, 2.3, 3.1];
    let rr = r.clone();
    let f = move |x: &Col<f64>| {
        let y = &rr * x;
        let gy = Col::from_fn(4, |i| 4.0 * a[i] * y[i] * (y[i] * y[i] - 1.0));
        let hy = Mat::from_fn(4, 4, |i, j| {
            if i == j {
                a[i] * (12.0 * y[i] * y[i] - 4.0)
            } else {
                0.0
            }
        });
        PointValue {
            energy: (0..4).map(|i| a[i] * (y[i] * y[i] - 1.0).powi(2)).sum(),
            gradient: Some(rr.transpose() * gy),
            hessian: Some(rr.transpose() * hy * &rr),
        }
    };
    (f, r)
}

pub(crate) fn at(r: &Mat<f64>, y: [f64; 4]) -> Col<f64> {
    r.transpose() * Col::from_fn(4, |i| y[i])
}

/// The imaginary mode of a transition state.
pub(crate) fn mode(f: &impl Fn(&Col<f64>) -> PointValue, x: &Col<f64>) -> Col<f64> {
    let h = f(x).hessian.unwrap();
    let (values, vectors) = symmetric_eigen(&h).unwrap();
    assert!(values[0] < 0.0 && values[1] > 0.0);
    vectors.col(0).to_owned()
}

fn index(f: &impl Fn(&Col<f64>) -> PointValue, x: &Col<f64>) -> usize {
    let h = f(x).hessian.unwrap();
    symmetric_eigen(&h)
        .unwrap()
        .0
        .iter()
        .filter(|v| **v < 0.0)
        .count()
}

fn run(band: &mut RidgeBand, f: &impl Fn(&Col<f64>) -> PointValue) -> RidgeBandReport {
    let mut last = None;
    while let SearchAction::Evaluate(points) = band.ask() {
        let values = points
            .iter()
            .map(|p| {
                let mut v = f(&p.x);
                if !p.hessian {
                    v.hessian = None;
                }
                v
            })
            .collect();
        last = Some(band.tell(values).unwrap());
    }
    last.unwrap()
}

#[test]
fn the_highest_image_climbs_to_the_index_two_saddle_between_opposite_and_adjacent_pairs() {
    let (f, r) = quartic();
    let saddle = at(&r, [0.0, 0.0, 1.0, 1.0]);
    for (first, second, adjacent) in [
        // Opposite, joined along y_0 (transverse y_1) and along y_1 (transverse y_0, the
        // shallower of the two negative curvatures of the saddle).
        ([1.0, 0.0, 1.0, 1.0], [-1.0, 0.0, 1.0, 1.0], false),
        ([0.0, 1.0, 1.0, 1.0], [0.0, -1.0, 1.0, 1.0], false),
        // Adjacent: they share the minimum (1, 1, 1, 1); the band turns at the saddle.
        ([1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], true),
    ] {
        {
            let transverse = TransverseMode::Lowest;
            let spring_force = RidgeSpring::Full;
            let (t1, t2) = (at(&r, first), at(&r, second));
            let modes = [mode(&f, &t1), mode(&f, &t2)];
            let settings = RidgeBandSettings {
                transverse,
                spring: 1.0,
                tolerance: 1e-6,
                max_iterations: 3000,
                ..RidgeBandSettings::default()
            };
            let mut chain = if adjacent {
                cornered_band(&t1, &t2, &modes[0], &modes[1], 7).unwrap()
            } else {
                straight_band(&t1, &t2, 7)
            };
            // Off the ridge at the start.
            let mut rng = crate::core_infra::rng::Rng::from_seed(3);
            for image in chain.iter_mut().take(8).skip(1) {
                *image += Col::from_fn(4, |_| 0.1 * rng.standard_normal());
            }
            let mut band = RidgeBand::new(settings, chain, modes).unwrap();
            let report = run(&mut band, &f);
            assert_eq!(
                report.end,
                Some(RidgeBandEnd::Converged),
                "{first:?} {second:?} {spring_force:?}: iteration {}, largest {:.3e}, climbing \
                 {:?}, energies {:?}, curvatures {:?}",
                report.iteration,
                report.largest_force,
                report.climbing,
                report.energies,
                report.curvatures
            );
            let top = &band.chain()[band.highest()];
            assert!(
                col_norm(&(top - &saddle)) < 1e-5,
                "{first:?} {second:?} {spring_force:?}: {top:?}"
            );
            assert_eq!(index(&f, top), 2);
            assert_eq!(band.saddle_directions().len(), 2);
            assert!(report.climbing_force.is_some_and(|f| f < 1e-6));
            println!(
                "{first:?} {second:?} {transverse:?} {spring_force:?}: {} iterations, {} \
                 gradients, band {:.1e}",
                report.iteration, report.gradients, report.largest_force
            );
        }
    }
    // The straight band between the adjacent pair has no maximum across it near the ends.
    let (t1, t2) = (at(&r, [1.0, 0.0, 1.0, 1.0]), at(&r, [0.0, 1.0, 1.0, 1.0]));
    let modes = [mode(&f, &t1), mode(&f, &t2)];
    let settings = RidgeBandSettings {
        spring_force: RidgeSpring::Tangential,
        spring: 1.0,
        ..RidgeBandSettings::default()
    };
    let mut band = RidgeBand::new(settings, straight_band(&t1, &t2, 7), modes).unwrap();
    assert!(matches!(
        run(&mut band, &f).end,
        Some(RidgeBandEnd::RidgeModeLost(_))
    ));
}

#[test]
fn exact_hessians_and_the_continued_mode() {
    let (f, r) = quartic();
    let saddle = at(&r, [0.0, 0.0, -1.0, 1.0]);
    let (t1, t2) = (
        at(&r, [1.0, 0.0, -1.0, 1.0]),
        at(&r, [-1.0, 0.0, -1.0, 1.0]),
    );
    let modes = [mode(&f, &t1), mode(&f, &t2)];
    for (curvature, transverse) in [
        (Curvature::Hessian, TransverseMode::Lowest),
        (Curvature::Forward, TransverseMode::Continued),
        (Curvature::Central, TransverseMode::Lowest),
    ] {
        let settings = RidgeBandSettings {
            curvature,
            transverse,
            spring: 1.0,
            tolerance: 1e-6,
            max_iterations: 3000,
            ..RidgeBandSettings::default()
        };
        let mut band = RidgeBand::new(settings, straight_band(&t1, &t2, 5), modes.clone()).unwrap();
        let report = run(&mut band, &f);
        assert_eq!(report.end, Some(RidgeBandEnd::Converged), "{curvature:?}");
        assert!(col_norm(&(&band.chain()[band.highest()] - &saddle)) < 1e-5);
        if curvature == Curvature::Hessian {
            assert_eq!(report.gradients, report.hessians + 7);
        }
    }
}

#[test]
fn transition_states_in_series_and_a_band_off_any_ridge_are_reported() {
    let (f, r) = quartic();
    let t1 = at(&r, [1.0, 0.0, 1.0, 1.0]);
    let e1 = mode(&f, &t1);
    // The band leaves T1 along its imaginary mode.
    let t2 = &t1 + &e1 * 2.0;
    let mut band = RidgeBand::new(
        RidgeBandSettings::default(),
        straight_band(&t1, &t2, 5),
        [e1.clone(), e1.clone()],
    )
    .unwrap();
    assert_eq!(run(&mut band, &f).end, Some(RidgeBandEnd::Series));
    // Between two minima (over the transition state between them) nothing is a maximum
    // across the band.
    let (m1, m2) = (at(&r, [1.0, 1.0, 1.0, 1.0]), at(&r, [1.0, 1.0, 1.0, -1.0]));
    let settings = RidgeBandSettings {
        lost_after: 5,
        series_overlap: 1.1,
        ..RidgeBandSettings::default()
    };
    let across = at(&r, [1.0, 0.0, 0.0, 0.0]);
    let mut band = RidgeBand::new(
        settings,
        straight_band(&m1, &m2, 5),
        [across.clone(), across],
    )
    .unwrap();
    let end = run(&mut band, &f).end;
    assert!(
        matches!(end, Some(RidgeBandEnd::RidgeModeLost(_))),
        "{end:?}"
    );
    assert!(RidgeBand::new(
        RidgeBandSettings::default(),
        vec![t1.clone(), t2],
        [Col::zeros(4), Col::zeros(4)]
    )
    .is_err());
}

#[test]
fn the_hessian_form_on_the_chain_relaxation_of_nebmain() {
    use super::super::machine::{ChainAction, ChainRelaxation, ImageValue};
    let (f, r) = quartic();
    let saddle = at(&r, [0.0, 0.0, 1.0, 1.0]);
    for (first, second, adjacent) in [
        ([1.0, 0.0, 1.0, 1.0], [-1.0, 0.0, 1.0, 1.0], false),
        ([0.0, 1.0, 1.0, 1.0], [0.0, -1.0, 1.0, 1.0], false),
        ([1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], true),
    ] {
        let (t1, t2) = (at(&r, first), at(&r, second));
        let modes = [mode(&f, &t1), mode(&f, &t2)];
        let mut chain = if adjacent {
            cornered_band(&t1, &t2, &modes[0], &modes[1], 7).unwrap()
        } else {
            straight_band(&t1, &t2, 7)
        };
        let mut rng = crate::core_infra::rng::Rng::from_seed(3);
        for image in chain.iter_mut().take(8).skip(1) {
            *image += Col::from_fn(4, |_| 0.1 * rng.standard_normal());
        }
        let mut settings = hessian_ridge_settings(modes, super::super::ChainKind::Variables);
        settings.force_tolerance = Some(1e-6);
        let mut band = ChainRelaxation::new(settings, chain).unwrap();
        let (mut gradients, mut hessians, mut last) = (0, 0, None);
        while let ChainAction::Evaluate(request) = band.ask() {
            let values = request
                .images
                .iter()
                .map(|x| {
                    let v = f(x);
                    gradients += 1;
                    hessians += usize::from(request.need_hessian);
                    ImageValue {
                        energy: v.energy,
                        gradient: v.gradient.unwrap(),
                        hessian: if request.need_hessian {
                            v.hessian
                        } else {
                            None
                        },
                    }
                })
                .collect();
            last = Some(band.tell(values).unwrap());
        }
        let report = last.unwrap();
        assert_eq!(
            band.end(),
            Some(super::super::machine::ChainEnd::Converged),
            "{first:?} {second:?}: {:?}",
            report.energies
        );
        let top = report.climbing.first().copied().unwrap();
        assert!(
            col_norm(&(&report.chain[top] - &saddle)) < 1e-5,
            "{:?}",
            report.chain[top]
        );
        assert!(report.ridge_curvatures[1..8].iter().all(|c| *c < 0.0));
        println!(
            "{first:?} {second:?}: {} iterations, {gradients} gradients, {hessians} Hessians",
            report.iteration
        );
    }
}
