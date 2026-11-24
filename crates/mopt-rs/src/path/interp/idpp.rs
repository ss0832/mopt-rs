//! Chains relaxed on the image dependent pair potential.
//!
//! Smidstrup, Pedersen, Stokbro, and Jonsson (J. Chem. Phys. 2014, 140, 214106): image `l`
//! of `N` has the target distances `d(first) + (l - 1) / (N - 1) (d(last) - d(first))`
//! and the function `S_l = sum_{i<j} d_ij^-4 (d_ij - target_ij)^2` in the place of an
//! energy ([`super::pairs::idpp_value`]); a chain that starts on the straight line is
//! relaxed on these functions like a chain on an energy surface.
//!
//! [`idpp_images`] does that with the relaxation of [`crate::path::band`]: the ends stay,
//! the images move along the forces across the chain, and nothing moves them along it.
//! The publication has springs between the images to keep them apart evenly; here the
//! images keep the places along the chain they have on the straight line, and whoever
//! wants them evenly spaced puts them so afterwards (`nebmain` does, with `-nd`).
//!
//! [`sequential_idpp_images`] is the sequential variant of Schmerwitz, Asgeirsson, and
//! Jonsson (J. Chem. Theory Comput. 2024, 20, 155; arXiv:2310.04531), which does not start
//! from the straight line, where atoms can pass through each other and bonds break that
//! need not: the images are put one after the other, starting at both ends, each a step
//! from the last image of its side along the chain, and all images placed so far are
//! relaxed after every addition. Differences from the publication: the two sides take
//! turns (there an image is added where the last one has converged); the step is the
//! gap between the two sides divided evenly among the images still to come (there the
//! length of the path over the number of images); and there are no springs, as above.
//!
//! The tolerance is that of the second publication: a largest force component of
//! `0.01 / Angstrom^3`.
//!
//! MultiOptPy: `Potential/idpp.py:IDPP.opt_path` (`-idpp`). There every image is minimized
//! on its own function without regard to its neighbours, in steps of at most 0.01, with a
//! limited-memory update whose differences of gradients have the wrong sign for one of
//! the two gradients; only the function is taken over.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::pairs::{idpp_value, interpolated_distances, pair_distances};
use super::{nudged, on_the_line, Interpolated, PathFrame, PathSettings};
use crate::error::MoptResult;
use crate::math::linalg::col_norm;
use crate::path::band::machine::{
    ChainAction, ChainEnd, ChainRelaxation, ChainSettings, ImageValue,
};
use crate::path::band::trust::FixedEnds;

/// Largest force component at which a relaxation ends: `0.01 / Angstrom^3` in `1 / Bohr^3`.
pub const FORCE_TOLERANCE: f64 = 0.01 * 0.148_184_71;

/// The settings of a relaxation between two structures that stay.
pub(super) fn relaxation(frame: &PathFrame, settings: &PathSettings) -> ChainSettings {
    ChainSettings {
        max_iterations: settings.max_iterations.max(1),
        ends: FixedEnds {
            first: true,
            last: true,
        },
        fixed_components: frame.fixed_components.clone(),
        // The functions depend on distances only: nothing shifts or turns the images.
        align: false,
        force_tolerance: Some(settings.force_tolerance),
        ..ChainSettings::default()
    }
}

/// `chain` relaxed with the values and gradients of `value(point, image)`. Returns the
/// chain, the iterations, and whether the forces came below the tolerance.
pub(super) fn relaxed(
    settings: ChainSettings,
    chain: Vec<Col<f64>>,
    value: impl Fn(&Col<f64>, usize) -> (f64, Col<f64>),
) -> MoptResult<(Vec<Col<f64>>, usize, bool)> {
    let mut machine = ChainRelaxation::new(settings, chain)?;
    let mut iterations = 0;
    loop {
        match machine.ask() {
            ChainAction::Finished(end) => {
                return Ok((
                    machine.chain().to_vec(),
                    iterations,
                    end == ChainEnd::Converged,
                ));
            }
            ChainAction::Evaluate(request) => {
                let values = request
                    .images
                    .iter()
                    .enumerate()
                    .map(|(image, point)| {
                        let (function, gradient) = value(point, image);
                        ImageValue::new(function, gradient)
                    })
                    .collect();
                machine.tell(values)?;
                iterations += 1;
            }
        }
    }
}

/// `count` images between `first` and `last`: the straight line relaxed on the image
/// dependent pair potential.
pub fn idpp_images(
    first: &Col<f64>,
    last: &Col<f64>,
    count: usize,
    frame: &PathFrame,
    settings: &PathSettings,
) -> MoptResult<Interpolated> {
    if count == 0 {
        return Ok(Interpolated {
            images: Vec::new(),
            iterations: 0,
            converged: true,
        });
    }
    let cell = frame.cell.as_ref();
    let (from, to) = (pair_distances(first, cell), pair_distances(last, cell));
    let targets: Vec<Vec<f64>> = (0..count + 2)
        .map(|l| interpolated_distances(&from, &to, l as f64 / (count + 1) as f64))
        .collect();
    let mut chain = Vec::with_capacity(count + 2);
    chain.push(first.clone());
    chain.extend(
        on_the_line(first, last, count)
            .iter()
            .map(|image| nudged(image, frame)),
    );
    chain.push(last.clone());
    let (chain, iterations, converged) =
        relaxed(relaxation(frame, settings), chain, |point, image| {
            idpp_value(point, &targets[image], cell)
        })?;
    Ok(Interpolated {
        images: chain[1..=count].to_vec(),
        iterations,
        converged,
    })
}

/// `v` with the length one (unchanged when it has none).
fn unit(v: Col<f64>) -> Col<f64> {
    let norm = col_norm(&v);
    if norm > 1e-12 {
        v * (1.0 / norm)
    } else {
        v
    }
}

/// `count` images between `first` and `last`, put one after the other from both ends and
/// relaxed on the image dependent pair potential after every addition.
pub fn sequential_idpp_images(
    first: &Col<f64>,
    last: &Col<f64>,
    count: usize,
    frame: &PathFrame,
    settings: &PathSettings,
) -> MoptResult<Interpolated> {
    let cell = frame.cell.as_ref();
    let (from, to) = (pair_distances(first, cell), pair_distances(last, cell));
    let total = count + 2;
    let targets: Vec<Vec<f64>> = (0..total)
        .map(|l| interpolated_distances(&from, &to, l as f64 / (total - 1) as f64))
        .collect();
    // The images of the side of the first structure in order, and those of the side of
    // the last structure from the last structure inwards.
    let mut near_first = vec![first.clone()];
    let mut near_last = vec![last.clone()];
    let (mut iterations, mut converged) = (0, true);
    let mut on_the_first_side = true;
    while near_first.len() + near_last.len() < total {
        let missing = total - near_first.len() - near_last.len();
        let (side, other) = if on_the_first_side {
            (&near_first, &near_last)
        } else {
            (&near_last, &near_first)
        };
        let frontier = &side[side.len() - 1];
        let across = &other[other.len() - 1] - frontier;
        let step = col_norm(&across) / (missing + 1) as f64;
        // Along the chain at the frontier: towards the other side, and on from the image
        // before where there is one.
        let towards = unit(across);
        let direction = match side.len().checked_sub(2) {
            Some(before) => unit(unit(frontier - &side[before]) + &towards),
            None => towards,
        };
        let mut placed = nudged(&(frontier + step * direction), frame);
        for &fixed in &frame.fixed_components {
            placed[fixed] = frontier[fixed];
        }
        if on_the_first_side {
            near_first.push(placed);
        } else {
            near_last.push(placed);
        }
        on_the_first_side = !on_the_first_side;

        // All images placed so far, each with the targets of its place in the whole chain.
        let on_first = near_first.len();
        let chain: Vec<Col<f64>> = near_first
            .iter()
            .chain(near_last.iter().rev())
            .cloned()
            .collect();
        let placed_images = chain.len();
        let place = |image: usize| {
            if image < on_first {
                image
            } else {
                total - (placed_images - image)
            }
        };
        let (chain, taken, at_rest) =
            relaxed(relaxation(frame, settings), chain, |point, image| {
                idpp_value(point, &targets[place(image)], cell)
            })?;
        iterations += taken;
        converged &= at_rest;
        near_first = chain[..on_first].to_vec();
        near_last = chain[on_first..].iter().rev().cloned().collect();
    }
    let mut images: Vec<Col<f64>> = near_first.into_iter().skip(1).collect();
    images.extend(
        near_last
            .into_iter()
            .rev()
            .take(total - 1 - images.len() - 1),
    );
    Ok(Interpolated {
        images,
        iterations,
        converged,
    })
}
