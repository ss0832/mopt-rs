//! Tests of the state machine of a chain relaxation (`machine.rs`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::*;
use crate::math::linalg::col_norm;
use crate::path::band::images::{AddedImages, FailedImage};

fn col(values: &[f64]) -> Col<f64> {
    Col::from_fn(values.len(), |k| values[k])
}

/// Scale of the test surface: forces of the size met in molecules for images this
/// far apart.
const SCALE: f64 = 0.1;

/// A valley along the parabola `y = x^2` with a barrier at `x = 0`:
/// `E = SCALE ((y - x^2)^2 + 0.3 exp(-4 x^2))`.
fn valley(p: &Col<f64>) -> ImageValue {
    let (x, y) = (p[0], p[1]);
    let off = y - x * x;
    let bump = 0.3 * (-4.0 * x * x).exp();
    ImageValue::new(
        SCALE * (off * off + bump),
        col(&[SCALE * (-4.0 * x * off - 8.0 * x * bump), SCALE * 2.0 * off]),
    )
}

fn straight(n: usize) -> Vec<Col<f64>> {
    (0..n)
        .map(|k| {
            let x = -1.0 + 2.0 * k as f64 / (n - 1) as f64;
            // From (-1, 1) to (1, 1) in a straight line: off the valley in between.
            col(&[x, 1.0])
        })
        .collect()
}

fn relax(settings: ChainSettings, chain: Vec<Col<f64>>) -> (ChainRelaxation, Vec<ChainReport>) {
    let mut machine = ChainRelaxation::new(settings, chain).unwrap();
    let mut reports = Vec::new();
    while let ChainAction::Evaluate(request) = machine.ask() {
        reports.push(
            machine
                .tell(request.images.iter().map(valley).collect())
                .unwrap(),
        );
    }
    (machine, reports)
}

#[test]
fn a_straight_chain_relaxes_into_the_valley() {
    let settings = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 600,
        ends: FixedEnds::from_code(3),
        equal_spacing_every: Some(5),
        force_tolerance: Some(1e-6),
        ..ChainSettings::default()
    };
    let (machine, reports) = relax(settings, straight(9));
    let last = &reports[reports.len() - 1];
    assert_eq!(machine.end(), Some(ChainEnd::Converged));
    assert!(reports.len() < 600, "{} iterations", reports.len());
    // The images came down from the straight line to the path of steepest descent
    // from the barrier, which runs a little below the floor y = x^2 of the valley.
    for image in &last.chain[1..8] {
        let floor = image[0] * image[0];
        assert!(
            image[1] < floor + 1e-3 && image[1] > floor - 0.2,
            "{image:?}"
        );
    }
    // The highest image is near the barrier (0.3 at x = 0) but not on it: nothing puts
    // an image on the saddle, and the limits of the steps treat the two neighbours of
    // an image differently, so that a symmetric chain does not stay symmetric.
    assert_eq!(last.maxima.len(), 1);
    let top = last.maxima[0];
    assert!(last.chain[top][0].abs() < 0.2);
    let barrier = 0.3 * SCALE;
    assert!(last.energies[top] > 0.8 * barrier && last.energies[top] <= barrier + 1e-9);
    // The ends did not move, and the first report is the straight chain.
    assert_eq!(last.chain[0], col(&[-1.0, 1.0]));
    assert_eq!(reports[0].chain, straight(9));
    assert!(reports[0].force_max[4] > 0.1 * SCALE);
    assert!(reports.iter().any(|r| r.redistributed));
}

/// The valley with a third variable that is held at 0 (a fixed component) and pulled
/// towards 2 by the surface: its gradient is a force on a fixed coordinate, which no step
/// can lower.
fn pulled(p: &Col<f64>) -> ImageValue {
    let plane = valley(&col(&[p[0], p[1]]));
    let z = p[2];
    ImageValue::new(
        plane.energy + SCALE * 0.5 * (z - 2.0) * (z - 2.0),
        col(&[plane.gradient[0], plane.gradient[1], SCALE * (z - 2.0)]),
    )
}

#[test]
fn a_force_on_a_fixed_component_does_not_keep_the_chain_from_converging() {
    let settings = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 600,
        ends: FixedEnds::from_code(3),
        equal_spacing_every: Some(5),
        force_tolerance: Some(1e-6),
        fixed_components: vec![2],
        diis: Some(crate::path::band::diis::ChainDiisSettings::default()),
        ..ChainSettings::default()
    };
    let chain: Vec<Col<f64>> = straight(9)
        .into_iter()
        .map(|p| col(&[p[0], p[1], 0.0]))
        .collect();
    let mut machine = ChainRelaxation::new(settings.clone(), chain).unwrap();
    let mut reports = Vec::new();
    while let ChainAction::Evaluate(request) = machine.ask() {
        reports.push(
            machine
                .tell(request.images.iter().map(pulled).collect())
                .unwrap(),
        );
    }
    assert_eq!(
        machine.end(),
        Some(ChainEnd::Converged),
        "{}",
        reports.len()
    );
    // The same chain without the fixed variable: every iteration is the same, so nothing
    // derived from the forces (the test of the end, the extrapolation, the files of the
    // forces) sees the pull on the fixed component.
    let plain = ChainSettings {
        fixed_components: Vec::new(),
        ..settings
    };
    let (plain_machine, without) = relax(plain, straight(9));
    assert_eq!(plain_machine.end(), Some(ChainEnd::Converged));
    assert!(reports.iter().all(|r| r.chain.iter().all(|p| p[2] == 0.0)));
    // The first iterations agree to the rounding of sums over another number of
    // components (later ones drift apart by that, as any two runs of a FIRE rule do).
    for (a, b) in reports.iter().zip(&without).take(20) {
        assert!((a.largest_force - b.largest_force).abs() < 1e-10);
        for k in 0..9 {
            assert!((a.force_max[k] - b.force_max[k]).abs() < 1e-10);
            assert!((a.force_rms[k] - b.force_rms[k]).abs() < 1e-10);
            assert!((a.gradient_rms[k] - b.gradient_rms[k]).abs() < 1e-10);
            assert!((a.chain[k][0] - b.chain[k][0]).abs() < 1e-10);
        }
    }
    let last = &reports[reports.len() - 1];
    assert!(last.largest_force < 1e-6);
}

#[test]
fn a_climbing_image_reaches_the_saddle() {
    let settings = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 800,
        ends: FixedEnds::from_code(3),
        equal_spacing_every: Some(5),
        climb_after: Some(10),
        force_tolerance: Some(1e-7),
        ..ChainSettings::default()
    };
    let (machine, reports) = relax(settings, straight(9));
    assert_eq!(
        machine.end(),
        Some(ChainEnd::Converged),
        "{}",
        reports.len()
    );
    let last = &reports[reports.len() - 1];
    assert_eq!(last.climbing.len(), 1);
    let top = last.climbing[0];
    // The saddle of the surface: (0, 0) with the energy 0.3.
    assert!(last.chain[top][0].abs() < 1e-4 && last.chain[top][1].abs() < 1e-4);
    assert!((last.energies[top] - 0.3 * SCALE).abs() < 1e-7);
    assert!(reports[10].climbing.is_empty() && !reports[11].climbing.is_empty());
}

/// The Hessian of [`valley`].
fn valley_hessian(p: &Col<f64>) -> Mat<f64> {
    let (x, y) = (p[0], p[1]);
    let bell = (-4.0 * x * x).exp();
    let xx = -4.0 * (y - x * x) + 8.0 * x * x - 2.4 * bell + 19.2 * x * x * bell;
    let values = [
        [SCALE * xx, -4.0 * SCALE * x],
        [-4.0 * SCALE * x, 2.0 * SCALE],
    ];
    Mat::from_fn(2, 2, |r, c| values[r][c])
}

fn relax_with_hessians(
    settings: ChainSettings,
    chain: Vec<Col<f64>>,
) -> (ChainRelaxation, Vec<ChainReport>) {
    let mut machine = ChainRelaxation::new(settings, chain).unwrap();
    let mut reports = Vec::new();
    while let ChainAction::Evaluate(request) = machine.ask() {
        let values = request
            .images
            .iter()
            .map(|image| {
                let mut value = valley(image);
                if request.need_hessian {
                    value.hessian = Some(valley_hessian(image));
                }
                value
            })
            .collect();
        reports.push(machine.tell(values).unwrap());
    }
    (machine, reports)
}

#[test]
fn the_way_of_the_articles_converges_with_steps_across_the_chain() {
    let articles = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 150,
        ends: FixedEnds::from_code(3),
        force_tolerance: Some(1e-6),
        ..ChainSettings::of_the_articles()
    };
    let (machine, reports) = relax_with_hessians(articles.clone(), straight(9));
    assert_eq!(
        machine.end(),
        Some(ChainEnd::Converged),
        "{}",
        reports.len()
    );
    // Far fewer iterations than with the steps along the forces alone (see
    // a_straight_chain_relaxes_into_the_valley), with redistribution every second one.
    assert!(reports.len() < 60, "{} iterations", reports.len());
    assert!(reports.iter().filter(|r| r.redistributed).count() > 5);
    assert!(reports[0].chain == straight(9));
    // The image at the maximum of the profile was put on the predicted maximum: the
    // saddle at (0, 0) with the energy 0.3.
    let last = &reports[reports.len() - 1];
    assert_eq!(last.maxima.len(), 1);
    let top = last.maxima[0];
    assert!(last.chain[top][0].abs() < 1e-3 && last.chain[top][1].abs() < 1e-3);
    assert!((last.energies[top] - 0.3 * SCALE).abs() < 1e-6);

    // MultiOptPy's form (a minimization of every image with its full gradient, images
    // put back on straight lines) keeps moving: it does not get below the tolerance.
    let legacy = ChainSettings {
        compatibility: Compatibility::MultiOptPy,
        ..articles.clone()
    };
    let (machine, reports) = relax_with_hessians(legacy, straight(9));
    assert_eq!(machine.end(), Some(ChainEnd::IterationLimit));
    let inner = |report: &ChainReport| {
        report.force_max[1..8]
            .iter()
            .fold(0.0_f64, |m, f| m.max(*f))
    };
    assert!(inner(&reports[reports.len() - 1]) > 1e-4);

    // With a climbing image and equal spacing instead of the predicted maxima.
    let climbing = ChainSettings {
        predicted_maxima_every: None,
        equal_spacing_every: Some(2),
        climb_after: Some(5),
        ..articles
    };
    let (machine, reports) = relax_with_hessians(climbing, straight(9));
    assert_eq!(
        machine.end(),
        Some(ChainEnd::Converged),
        "{}",
        reports.len()
    );
    assert!(reports.len() < 80, "{} iterations", reports.len());
    let last = &reports[reports.len() - 1];
    assert_eq!(last.climbing.len(), 1);
    let top = last.climbing[0];
    assert!(last.chain[top][0].abs() < 1e-4 && last.chain[top][1].abs() < 1e-4);
}

/// Two wells at `(-1, 0)` and `(1, 0)` with a saddle between them:
/// `E = SCALE ((x^2 - 1)^2 + 2 y^2)`.
fn wells(p: &Col<f64>, hessian: bool) -> ImageValue {
    let (x, y) = (p[0], p[1]);
    let mut value = ImageValue::new(
        SCALE * ((x * x - 1.0).powi(2) + 2.0 * y * y),
        col(&[SCALE * 4.0 * x * (x * x - 1.0), SCALE * 4.0 * y]),
    );
    if hessian {
        let values = [[SCALE * (12.0 * x * x - 4.0), 0.0], [0.0, SCALE * 4.0]];
        value.hessian = Some(Mat::from_fn(2, 2, |r, c| values[r][c]));
    }
    value
}

#[test]
fn end_images_are_minimized_unless_they_are_fixed() {
    // The ends start beside the wells.
    let (first, last) = ([-0.8, 0.3], [0.9, -0.2]);
    let chain: Vec<Col<f64>> = (0..7)
        .map(|k| {
            let t = k as f64 / 6.0;
            Col::from_fn(2, |c| first[c] + t * (last[c] - first[c]))
        })
        .collect();
    let run = |settings: ChainSettings| {
        let mut machine = ChainRelaxation::new(settings, chain.clone()).unwrap();
        let mut iterations = 0;
        while let ChainAction::Evaluate(request) = machine.ask() {
            let values = request
                .images
                .iter()
                .map(|image| wells(image, request.need_hessian))
                .collect();
            machine.tell(values).unwrap();
            iterations += 1;
        }
        (machine, iterations)
    };
    let near = |point: &Col<f64>, target: [f64; 2], tolerance: f64| {
        (point[0] - target[0]).abs() < tolerance && (point[1] - target[1]).abs() < tolerance
    };
    let along_forces = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 600,
        equal_spacing_every: Some(5),
        force_tolerance: Some(1e-6),
        ..ChainSettings::default()
    };
    let with_hessians = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 200,
        force_tolerance: Some(1e-6),
        ..ChainSettings::of_the_articles()
    };
    for (name, settings) in [("forces", along_forces), ("hessians", with_hessians)] {
        // Free ends (the default, MultiOptPy's `-fe 0`): both go to their wells, and the
        // relaxation is not over before they are there.
        let (machine, iterations) = run(settings.clone());
        assert_eq!(
            machine.end(),
            Some(ChainEnd::Converged),
            "{name}: {iterations}"
        );
        let found = machine.chain();
        assert!(near(&found[0], [-1.0, 0.0], 1e-4), "{name}: {:?}", found[0]);
        assert!(near(&found[6], [1.0, 0.0], 1e-4), "{name}: {:?}", found[6]);
        // Inner images lie on the line between the wells.
        assert!(found[1..6].iter().all(|p| p[1].abs() < 1e-4), "{name}");

        // `-fe 1`: the first image stays, the last one is minimized; `-fe 2` the other
        // way round; `-fe 3`: both stay.
        for (code, first_stays, last_stays) in [(1, true, false), (2, false, true), (3, true, true)]
        {
            let fixed = ChainSettings {
                ends: FixedEnds::from_code(code),
                ..settings.clone()
            };
            let (machine, iterations) = run(fixed);
            assert_eq!(
                machine.end(),
                Some(ChainEnd::Converged),
                "{name}, -fe {code}: {iterations}"
            );
            let found = machine.chain();
            if first_stays {
                assert_eq!(found[0], chain[0], "{name}, -fe {code}");
            } else {
                assert!(near(&found[0], [-1.0, 0.0], 1e-4), "{name}, -fe {code}");
            }
            if last_stays {
                assert_eq!(found[6], chain[6], "{name}, -fe {code}");
            } else {
                assert!(near(&found[6], [1.0, 0.0], 1e-4), "{name}, -fe {code}");
            }
        }
    }
}

#[test]
fn iteration_limit_fixed_components_and_wrong_calls() {
    let settings = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 3,
        fixed_components: vec![0],
        ..ChainSettings::default()
    };
    let (machine, reports) = relax(settings.clone(), straight(5));
    assert_eq!(reports.len(), 3);
    assert_eq!(machine.end(), Some(ChainEnd::IterationLimit));
    // x is fixed in every image; y moved towards the valley.
    for (image, start) in machine.chain().iter().zip(straight(5)) {
        assert_eq!(image[0], start[0]);
    }
    assert!(machine.chain()[2][1] < 1.0);

    let mut machine = ChainRelaxation::new(settings.clone(), straight(5)).unwrap();
    assert!(machine.tell(Vec::new()).is_err());
    let ChainAction::Evaluate(request) = machine.ask() else {
        panic!("an evaluation is asked for first");
    };
    assert!(!request.need_hessian);
    let images = request.images;
    assert!(machine.set_chain(straight(5)).is_err());
    assert!(machine.tell(vec![valley(&images[0])]).is_err());
    let mut values: Vec<ImageValue> = images.iter().map(valley).collect();
    values[1].energy = f64::NAN;
    assert!(machine.tell(values).is_err());
    machine.tell(images.iter().map(valley).collect()).unwrap();
    // Between iterations the caller may put another chain in, also a longer one.
    machine.set_chain(straight(7)).unwrap();
    assert_eq!(machine.chain().len(), 7);
    assert!(machine
        .set_chain(vec![col(&[0.0; 3]), col(&[1.0; 3])])
        .is_err());
    assert!(ChainRelaxation::new(settings, vec![col(&[0.0, 1.0])]).is_err());
    let bad = ChainSettings {
        fixed_components: vec![5],
        kind: ChainKind::Variables,
        ..ChainSettings::default()
    };
    assert!(ChainRelaxation::new(bad, straight(3)).is_err());
}

#[test]
fn images_are_added_around_the_maximum_and_keep_what_belongs_to_them() {
    // A coarse chain of five images, with two images added on either side of the
    // maximum after every sixth iteration (MultiOptPy's -aneb 2 6).
    let settings = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 120,
        ends: FixedEnds::from_code(3),
        // Every addition disturbs the chain a little: the run ends between two of them.
        force_tolerance: Some(1e-4),
        added: Some(AddedImages {
            images: 2,
            every: 6,
        }),
        ..ChainSettings::of_the_articles()
    };
    let (machine, reports) = relax_with_hessians(settings.clone(), straight(5));
    assert_eq!(
        machine.end(),
        Some(ChainEnd::Converged),
        "{}",
        reports.len()
    );
    // The first addition follows iteration 6, the later ones seven iterations apart.
    let grown: Vec<usize> = reports
        .iter()
        .filter(|r| r.added > 0)
        .map(|r| r.iteration)
        .collect();
    assert_eq!(&grown[..2], &[6, 13]);
    assert!(reports.iter().all(|r| r.added == 0 || r.added == 4));
    assert_eq!(reports[6].chain.len(), 5);
    assert_eq!(reports[7].chain.len(), 9);
    assert_eq!(reports[14].chain.len(), 13);
    // The ends stayed, and an image sits on the saddle at the end.
    let last = &reports[reports.len() - 1];
    assert_eq!(last.chain[0], straight(5)[0]);
    assert_eq!(last.chain[last.chain.len() - 1], straight(5)[4]);
    let top = last.maxima[0];
    assert!(last.chain[top][0].abs() < 2e-3 && last.chain[top][1].abs() < 2e-3);
    // Without the setting the chain keeps its images.
    let without = ChainSettings {
        added: None,
        ..settings
    };
    let (_, plain) = relax_with_hessians(without, straight(5));
    assert!(plain.iter().all(|r| r.added == 0 && r.chain.len() == 5));
}

#[test]
fn an_image_is_put_nearer_or_taken_out_between_ask_and_tell() {
    let settings = ChainSettings {
        kind: ChainKind::Variables,
        max_iterations: 40,
        ends: FixedEnds::from_code(3),
        ..ChainSettings::of_the_articles()
    };
    let mut machine = ChainRelaxation::new(settings, straight(7)).unwrap();
    assert_eq!(machine.on_failure(), FailedImage::Retry);
    let values = |request: &ChainRequest| -> Vec<ImageValue> {
        request
            .images
            .iter()
            .map(|image| {
                let mut value = valley(image);
                if request.need_hessian {
                    value.hessian = Some(valley_hessian(image));
                }
                value
            })
            .collect()
    };
    let asked = |machine: &mut ChainRelaxation| match machine.ask() {
        ChainAction::Evaluate(request) => request,
        ChainAction::Finished(end) => unreachable!("{end:?}"),
    };
    // Nothing to go back to in the first iteration, and nothing outside ask and tell.
    assert!(machine.retried_point(2, 0).is_none());
    assert!(machine.drop_images(&[2]).is_err());
    let request = asked(&mut machine);
    assert!(machine.retried_point(2, 0).is_none());
    let first = request.images.clone();
    machine.tell(values(&request)).unwrap();
    // Second iteration: image 2 at half and a quarter of its step, then where it was.
    let request = asked(&mut machine);
    let moved = &request.images[2] - &first[2];
    for (attempt, share) in [0.5, 0.25, 0.0].into_iter().enumerate() {
        let nearer = machine.retried_point(2, attempt).unwrap();
        assert!(col_norm(&(&nearer - &first[2] - share * &moved)) < 1e-14);
    }
    assert!(machine.retried_point(2, 3).is_none());
    let nearer = machine.retried_point(2, 0).unwrap();
    machine.replace_image(2, nearer.clone()).unwrap();
    assert_eq!(machine.chain()[2], nearer);
    // Images 3 and 5 are taken out: five are left, and the relaxation goes on with them.
    machine.drop_images(&[5, 3]).unwrap();
    assert_eq!(machine.chain().len(), 5);
    assert_eq!(machine.chain()[3], request.images[4]);
    let shorter = asked(&mut machine);
    assert_eq!(shorter.images.len(), 5);
    let report = machine.tell(values(&shorter)).unwrap();
    assert_eq!(report.energies.len(), 5);
    while let ChainAction::Evaluate(request) = machine.ask() {
        machine.tell(values(&request)).unwrap();
    }
    assert_eq!(machine.chain().len(), 5);
    // Fewer than two images cannot be left.
    let plain = ChainSettings {
        kind: ChainKind::Variables,
        ..ChainSettings::default()
    };
    let mut two = ChainRelaxation::new(plain.clone(), straight(3)).unwrap();
    let _ = two.ask();
    assert!(two.drop_images(&[0, 1]).is_err());
    assert!(two.drop_images(&[7]).is_err());
    // MultiOptPy's form leaves a failed image out without trying again.
    let legacy = ChainRelaxation::new(
        ChainSettings {
            compatibility: Compatibility::MultiOptPy,
            ..plain
        },
        straight(3),
    )
    .unwrap();
    assert_eq!(legacy.on_failure(), FailedImage::Drop);
}
