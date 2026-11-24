//! Linear synchronous transit: images whose interatomic distances are, as nearly as a
//! structure can have them, those interpolated linearly between two structures.
//!
//! Halgren and Lipscomb (Chem. Phys. Lett. 1977, 49, 225): the image at the share `f` of
//! the way minimizes
//!
//! ```text
//! sum_{a<b} (r_ab - r_ab(f))^2 / r_ab(f)^4 + 1e-6 sum_a |x_a - x_a(f)|^2
//! ```
//!
//! with `r_ab(f)` and `x_a(f)` the distances and the positions interpolated linearly (the
//! formula as later papers quote it, in atomic units; the original was not at hand). Every
//! image is found on its own, from the straight line, with the L-BFGS method of
//! [`crate::opt::vector`]. Nothing ties the images to each other, so the path they form can
//! have a jump where the least-squares fit changes from one solution to another; the image
//! dependent pair potential ([`super::idpp`]) relaxes a whole chain instead.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::pairs::{interpolated_distances, lst_value, pair_distances};
use super::{nudged, Interpolated, PathFrame, PathSettings};
use crate::error::MoptResult;
use crate::opt::state_machine::Termination;
use crate::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};

/// Weight of the distance from the linearly interpolated structure (atomic units).
pub const CARTESIAN_WEIGHT: f64 = 1e-6;

/// `count` images of the linear synchronous transit between `first` and `last`.
pub fn lst_images(
    first: &Col<f64>,
    last: &Col<f64>,
    count: usize,
    frame: &PathFrame,
    settings: &PathSettings,
) -> MoptResult<Interpolated> {
    let cell = frame.cell.as_ref();
    let (from, to) = (pair_distances(first, cell), pair_distances(last, cell));
    let mut out = Interpolated {
        images: Vec::with_capacity(count),
        iterations: 0,
        converged: true,
    };
    for k in 1..=count {
        let share = k as f64 / (count + 1) as f64;
        let targets = interpolated_distances(&from, &to, share);
        let line = first + share * (last - first);
        let mut optimizer = VectorOptimizer::new(
            VectorSettings {
                methods: vec!["lbfgs".to_string()],
                max_iterations: settings.max_iterations.max(1),
                ..VectorSettings::default()
            },
            nudged(&line, frame),
        )?;
        loop {
            match optimizer.ask() {
                VectorAction::Finished(end) => {
                    out.converged &= end == Termination::Converged;
                    break;
                }
                VectorAction::Evaluate { x, .. } => {
                    let (value, mut gradient) =
                        lst_value(&x, &targets, &line, CARTESIAN_WEIGHT, cell);
                    for &fixed in &frame.fixed_components {
                        gradient[fixed] = 0.0;
                    }
                    optimizer.tell(value, gradient, None)?;
                }
            }
        }
        out.iterations += optimizer.iterations();
        out.images.push(optimizer.current().clone());
    }
    Ok(out)
}
