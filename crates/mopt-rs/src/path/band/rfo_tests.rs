//! Tests of the quasi-Newton steps of the images of a chain (`rfo.rs`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::*;

fn col(values: &[f64]) -> Col<f64> {
    Col::from_fn(values.len(), |k| values[k])
}

/// `E = sum k_c x_c^2 / 2` with the curvatures 1 and 4.
fn bowl(p: &Col<f64>) -> (f64, Col<f64>) {
    (
        0.5 * p[0] * p[0] + 2.0 * p[1] * p[1],
        col(&[p[0], 4.0 * p[1]]),
    )
}

#[test]
fn steps_go_down_the_gradient_with_the_given_hessians_and_limits() {
    let chain = vec![col(&[0.02, 0.01]), col(&[0.5, 0.3]), col(&[1.0, 0.01])];
    let (energies, gradients): (Vec<f64>, Vec<Col<f64>>) = chain.iter().map(bowl).unzip();
    let removed = vec![Col::zeros(2); 3];
    let along = AlongChain {
        removed: &removed,
        directions: &removed,
        climbing: &[],
    };
    let mut rfo = ChainRfo::new(ChainRfoSettings::default(), 3, 2, None, legacy()).unwrap();
    let exact = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, 4.0][i] } else { 0.0 });
    for i in 0..3 {
        rfo.set_hessian(i, exact.clone()).unwrap();
    }
    let steps = rfo.steps(&chain, &energies, &gradients, along).unwrap();
    // Near its minimum the first image takes almost the Newton step to the origin.
    assert!((steps[0][0] + 0.02).abs() < 2e-3 && (steps[0][1] + 0.01).abs() < 2e-3);
    // Far from it the step of an inner image is cut to 0.1, of an end image to 0.2.
    assert!((col_norm(&steps[1]) - 0.1).abs() < 1e-12);
    assert!((col_norm(&steps[2]) - 0.2).abs() < 1e-12);
    for (step, gradient) in steps.iter().zip(&gradients) {
        assert!(crate::math::linalg::col_dot(step, gradient) < 0.0);
    }
    // The direction is that of the unrestricted step: the same in every iteration at
    // the same point, however often it is asked for.
    let again = rfo.steps(&chain, &energies, &gradients, along).unwrap();
    for (a, b) in steps.iter().zip(&again) {
        assert!((a - b).iter().all(|d| d.abs() < 1e-14));
    }
    assert_eq!(rfo.len(), 3);
    assert!(!rfo.is_empty());
}

#[test]
fn the_curvature_along_the_chain_is_added_and_wrong_input_is_refused() {
    // Images along x on E = 0.5 x^2 + 2 y^2; the direction given for the middle image
    // is (-1, 0), a unit vector, so that gamma is the curvature along x, 1.
    let chain = vec![col(&[-0.4, 0.0]), col(&[0.0, 0.0]), col(&[0.5, 0.0])];
    let (energies, gradients): (Vec<f64>, Vec<Col<f64>>) = chain.iter().map(bowl).unzip();
    let removed = vec![Col::zeros(2), col(&[1.0, 0.0]), Col::zeros(2)];
    let along = AlongChain {
        removed: &removed,
        directions: &removed,
        climbing: &[],
    };
    let mut rfo = ChainRfo::new(ChainRfoSettings::default(), 3, 2, None, legacy()).unwrap();
    rfo.steps(&chain, &energies, &gradients, along).unwrap();
    let h = rfo.hessian(1).unwrap();
    assert!((h[(0, 0)] - 2.0).abs() < 1e-9 && (h[(1, 1)] - 1.0).abs() < 1e-12);
    // Once more: added again on top (and no update, the image did not move).
    rfo.steps(&chain, &energies, &gradients, along).unwrap();
    assert!((rfo.hessian(1).unwrap()[(0, 0)] - 3.0).abs() < 1e-9);
    assert_eq!(rfo.hessian(0).unwrap()[(0, 0)], 1.0);

    assert!(rfo.set_hessian(5, Mat::identity(2, 2)).is_err());
    assert!(rfo.set_hessian(0, Mat::identity(3, 3)).is_err());
    assert!(rfo
        .steps(&chain[..2], &energies, &gradients, along)
        .is_err());
    assert!(ChainRfo::new(ChainRfoSettings::default(), 1, 2, None, legacy()).is_err());
    let unknown = ChainRfoSettings {
        inner_update: "no_such_update".to_string(),
        ..ChainRfoSettings::default()
    };
    // An unknown name falls back to the default update of the registry or is refused,
    // as the registry decides; either way the minimizers can be built or not at all.
    let _ = ChainRfo::new(unknown, 3, 2, None, legacy());
}

#[test]
fn the_consistent_steps_go_across_the_chain_and_a_climbing_image_goes_up_along_it() {
    // A chain along x on the saddle surface E = -0.5 x^2 + 2 y^2, a little off y = 0.
    let saddle = |p: &Col<f64>| {
        (
            -0.5 * p[0] * p[0] + 2.0 * p[1] * p[1],
            col(&[-p[0], 4.0 * p[1]]),
        )
    };
    let chain = vec![col(&[-0.5, 0.03]), col(&[0.04, 0.03]), col(&[0.5, 0.03])];
    let (energies, gradients): (Vec<f64>, Vec<Col<f64>>) = chain.iter().map(saddle).unzip();
    let rule = UpwindProjection {
        block: 2,
        climb_after: None,
        compatibility: Compatibility::Consistent,
    };
    let directions = vec![Col::zeros(2), col(&[0.7, 0.0]), Col::zeros(2)];
    let removed = vec![Col::zeros(2), col(&[-0.04, 0.0]), Col::zeros(2)];
    let exact = Mat::from_fn(2, 2, |i, j| if i == j { [-1.0, 4.0][i] } else { 0.0 });
    let run = |climbing: &[usize]| {
        let mut rfo = ChainRfo::new(ChainRfoSettings::default(), 3, 2, None, rule).unwrap();
        rfo.set_hessian(1, exact.clone()).unwrap();
        let along = AlongChain {
            removed: &removed,
            directions: &directions,
            climbing,
        };
        let steps = rfo.steps(&chain, &energies, &gradients, along).unwrap();
        // Nothing is added to the Hessian of the surface.
        assert_eq!(rfo.hessian(1).unwrap()[(0, 0)], -1.0);
        steps[1].clone()
    };
    // Not climbing: no step along the chain, almost the Newton step to y = 0 across.
    let across = run(&[]);
    assert_eq!(across[0], 0.0);
    assert!((across[1] + 0.03).abs() < 2e-3, "{across:?}");
    // Climbing: also up along the chain to the saddle at x = 0.
    let climbing = run(&[1]);
    assert!((climbing[0] + 0.04).abs() < 4e-3, "{climbing:?}");
    assert!((climbing[1] + 0.03).abs() < 2e-3, "{climbing:?}");

    // Blocks: every block loses its own direction.
    let frame = Along::new(&col(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]), 3);
    assert_eq!((frame.units.nrows(), frame.units.ncols()), (9, 2));
    assert_eq!(
        (frame.starts.clone(), frame.lengths.clone()),
        (vec![0, 6], vec![2.0, 3.0])
    );
    let part = frame.part(&col(&[1.0; 9]));
    assert_eq!(part, col(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    let h = Mat::from_fn(9, 9, |r, c| 1.0 / (1.0 + r as f64 + c as f64));
    let model = frame.hessian(&h, false, &[]);
    for k in 0..9 {
        let expected = if k == 0 { 1.0 } else { 0.0 };
        assert!((model[(0, k)] - expected).abs() < 1e-15);
        assert!((model[(k, 7)] - if k == 7 { 1.0 } else { 0.0 }).abs() < 1e-15);
    }
    assert!((model[(1, 2)] - h[(1, 2)]).abs() < 1e-15);
    // The turning of the direction: slopes 4 and -6 along the two directions that
    // are 2 and 3 long give 2 in the first block and nothing in the third (uphill),
    // or nothing and 2 (downhill); the middle block has no direction and gets nothing.
    let gradient = col(&[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -6.0, 1.0]);
    assert_eq!(frame.stiffening(&gradient, true), vec![2.0, 0.0]);
    assert_eq!(frame.stiffening(&gradient, false), vec![0.0, 2.0]);
    let stiffer = frame.hessian(&h, false, &[2.0, 0.0]);
    assert!((stiffer[(1, 1)] - model[(1, 1)] - 2.0).abs() < 1e-15);
    assert!((stiffer[(0, 0)] - 1.0).abs() < 1e-15);
    assert!((stiffer[(4, 4)] - model[(4, 4)]).abs() < 1e-15);
    assert!((stiffer[(8, 8)] - model[(8, 8)]).abs() < 1e-15);
}

#[test]
fn the_slope_along_the_chain_stiffens_the_model_across_it() {
    // A plane that rises along x, E = 0.3 x: no curvature at all, but an image that
    // steps across the chain turns its direction to the neighbour uphill, 0.5 away,
    // and gets a gradient of 0.3 / 0.5 per unit step. With that in the model the
    // step against a gradient across the chain is finite.
    let chain = vec![col(&[-0.5, 0.0]), col(&[0.0, 0.0]), col(&[0.5, 0.0])];
    let energies = vec![-0.15, 0.0, 0.15];
    let gradients = vec![col(&[0.3, 0.0]), col(&[0.3, 0.02]), col(&[0.3, 0.0])];
    let rule = UpwindProjection {
        block: 2,
        climb_after: None,
        compatibility: Compatibility::Consistent,
    };
    let directions = vec![Col::zeros(2), col(&[0.5, 0.0]), Col::zeros(2)];
    let removed = vec![Col::zeros(2), col(&[0.3, 0.0]), Col::zeros(2)];
    let along = AlongChain {
        removed: &removed,
        directions: &directions,
        climbing: &[],
    };
    let mut rfo = ChainRfo::new(ChainRfoSettings::default(), 3, 2, None, rule).unwrap();
    rfo.set_hessian(1, Mat::zeros(2, 2)).unwrap();
    let steps = rfo.steps(&chain, &energies, &gradients, along).unwrap();
    // Newton: -0.02 / 0.6; the RFO step is a little shorter.
    assert_eq!(steps[1][0], 0.0);
    assert!(
        steps[1][1] < -0.03 && steps[1][1] > -0.02 / 0.6,
        "{:?}",
        steps[1]
    );
}

#[test]
fn hessians_kept_in_files_give_the_same_steps() {
    let chain = vec![col(&[0.02, 0.01]), col(&[0.5, 0.3]), col(&[1.0, 0.01])];
    let (energies, gradients): (Vec<f64>, Vec<Col<f64>>) = chain.iter().map(bowl).unzip();
    let zeros = vec![Col::zeros(2); 3];
    let along = AlongChain {
        removed: &zeros,
        directions: &zeros,
        climbing: &[],
    };
    let exact = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, 4.0][i] } else { 0.3 });
    let dir = std::env::temp_dir().join(format!("mopt_chain_hessians_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let run = |directory: Option<std::path::PathBuf>| {
        let settings = ChainRfoSettings {
            directory,
            ..ChainRfoSettings::default()
        };
        let mut rfo = ChainRfo::new(settings, 3, 2, None, legacy()).unwrap();
        for i in 0..3 {
            rfo.set_hessian(i, exact.clone()).unwrap();
        }
        let first = rfo.steps(&chain, &energies, &gradients, along).unwrap();
        // A second iteration at other points, so that the Hessians are updated.
        let moved: Vec<Col<f64>> = chain.iter().zip(&first).map(|(x, d)| x + d).collect();
        let (e, g): (Vec<f64>, Vec<Col<f64>>) = moved.iter().map(bowl).unzip();
        let second = rfo.steps(&moved, &e, &g, along).unwrap();
        (first, second, rfo.hessian(1).unwrap())
    };
    let in_memory = run(None);
    let in_files = run(Some(dir.clone()));
    for (a, b) in in_memory
        .0
        .iter()
        .chain(&in_memory.1)
        .zip(in_files.0.iter().chain(&in_files.1))
    {
        assert_eq!(a, b);
    }
    assert_eq!(in_memory.2, in_files.2);
    // The files are MultiOptPy's: one matrix per image, readable as .npy.
    for i in 0..3 {
        let kept = read_npy_matrix(dir.join(format!("tmp_hessian_{i}.npy"))).unwrap();
        assert_eq!((kept.nrows(), kept.ncols()), (2, 2));
    }
    assert_eq!(
        read_npy_matrix(dir.join("tmp_hessian_1.npy")).unwrap(),
        in_files.2
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn legacy() -> UpwindProjection {
    UpwindProjection {
        block: 2,
        climb_after: None,
        compatibility: Compatibility::MultiOptPy,
    }
}
