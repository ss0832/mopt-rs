//! How far the images of a chain may move in one step.
//!
//! An inner image may not pass its neighbours. With `r1`, `r2` half the distances to the
//! previous and the next image and `c1`, `c2` the overlaps of the displacement `d` with
//! the directions to them:
//!
//! - towards one neighbour and away from the other (`c1 c2 < 0`): `d` is shortened to
//!   `r1` if it points to the previous image, to `r2` if it points to the next one;
//! - away from both (`c1 < 0`, `c2 < 0`): shortened to the larger of `r1` and `r2`
//!   (MultiOptPy takes `d` as it is, without any limit: an image that climbs away from
//!   its neighbours can then run off; that form is kept under
//!   [`Compatibility::MultiOptPy`]);
//! - otherwise (towards both, or at a right angle to one): shortened to `r1` if longer
//!   than that, else to `r2` if longer than that;
//! - against the force (`F . d < 0`, or no force): the image is held.
//!
//! A free end image moves at most 0.5 Bohr, a fixed one not at all.
//!
//! MultiOptPy: `Optimizer/trust_radius_neb.py:TR_NEB`,
//! `Optimizer/gradientdescent_neb.py:SteepestDescentOptimizer`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::Compatibility;
use crate::math::linalg::{col_dot, col_norm};

/// Largest step of a free end image (Bohr).
pub const END_STEP: f64 = 0.5;
/// Factor on the force in a steepest-descent step.
pub const STEEPEST_FACTOR: f64 = 0.5;

/// Which end images stay where they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FixedEnds {
    /// The first image.
    pub first: bool,
    /// The last image.
    pub last: bool,
}

impl FixedEnds {
    /// MultiOptPy's `-fe`: 1 the first image, 2 the last one, 3 (or more) both.
    pub fn from_code(code: i64) -> Self {
        Self {
            first: code == 1 || code >= 3,
            last: code >= 2,
        }
    }
}

/// When an image counts as converged (`-aconv`): it is not moved any more.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageCriteria {
    /// Largest component of the force.
    pub max_force: f64,
    /// Root mean square of the force.
    pub rms_force: f64,
    /// Largest component of the displacement.
    pub max_displacement: f64,
    /// Root mean square of the displacement.
    pub rms_displacement: f64,
}

impl ImageCriteria {
    /// MultiOptPy's thresholds of `-aconv`.
    pub fn of_multioptpy() -> Self {
        Self::default()
    }
}

impl Default for ImageCriteria {
    fn default() -> Self {
        Self {
            max_force: 0.00045,
            rms_force: 0.00030,
            max_displacement: 0.0018,
            rms_displacement: 0.0012,
        }
    }
}

/// Limited steps of a chain and the distances between its images.
#[derive(Debug, Clone)]
pub struct LimitedSteps {
    /// Displacement of every image.
    pub steps: Vec<Col<f64>>,
    /// Distance of every inner image to the previous one.
    pub to_previous: Vec<f64>,
    /// Distance of every inner image to the next one.
    pub to_next: Vec<f64>,
    /// Inner images that were held because their displacement was against their force.
    pub held: Vec<usize>,
}

fn shortened(step: &Col<f64>, limit: f64) -> Col<f64> {
    let length = col_norm(step);
    if length > 0.0 {
        (limit / length) * step
    } else {
        step.clone()
    }
}

fn end_step(step: &Col<f64>, fixed: bool) -> Col<f64> {
    let length = col_norm(step);
    if fixed || length < 1e-15 {
        Col::zeros(step.nrows())
    } else {
        shortened(step, END_STEP.min(length))
    }
}

/// The displacements `proposed` of `chain` with its `forces`, limited as described in the
/// module documentation.
pub fn limit_steps(
    chain: &[Col<f64>],
    forces: &[Col<f64>],
    proposed: &[Col<f64>],
    ends: FixedEnds,
    compatibility: Compatibility,
) -> LimitedSteps {
    limit_steps_of(chain, forces, proposed, ends, compatibility, true)
}

/// [`limit_steps`]; with `hold` off an image whose step goes against its force is not
/// held (steps that are found for several images together count on the moves of the
/// neighbours, and may go against the force an image has while the neighbours stand).
pub fn limit_steps_of(
    chain: &[Col<f64>],
    forces: &[Col<f64>],
    proposed: &[Col<f64>],
    ends: FixedEnds,
    compatibility: Compatibility,
    hold: bool,
) -> LimitedSteps {
    let images = chain.len();
    let mut out = LimitedSteps {
        steps: Vec::with_capacity(images),
        to_previous: Vec::new(),
        to_next: Vec::new(),
        held: Vec::new(),
    };
    for i in 0..images {
        if i == 0 || i == images - 1 {
            let fixed = if i == 0 { ends.first } else { ends.last };
            out.steps.push(end_step(&proposed[i], fixed));
            continue;
        }
        let step = &proposed[i];
        let back = &chain[i - 1] - &chain[i];
        let ahead = &chain[i + 1] - &chain[i];
        let (d_back, d_ahead) = (col_norm(&back), col_norm(&ahead));
        out.to_previous.push(d_back);
        out.to_next.push(d_ahead);
        let length = col_norm(step);
        if length == 0.0 {
            out.steps.push(step.clone());
            continue;
        }
        let along_force = col_dot(&forces[i], step);
        if hold && (along_force < 0.0 || col_norm(&forces[i]) == 0.0) {
            out.held.push(i);
            out.steps.push(Col::zeros(step.nrows()));
            continue;
        }
        let (c_back, c_ahead) = (col_dot(&back, step), col_dot(&ahead, step));
        let (r_back, r_ahead) = (0.5 * d_back, 0.5 * d_ahead);
        let limited = if (c_back > 0.0 && c_ahead < 0.0) || (c_back < 0.0 && c_ahead > 0.0) {
            if length > r_back && c_back > 0.0 {
                shortened(step, r_back)
            } else if length > r_ahead && c_ahead > 0.0 {
                shortened(step, r_ahead)
            } else {
                step.clone()
            }
        } else if c_back < 0.0 && c_ahead < 0.0 {
            let limit = r_back.max(r_ahead);
            if compatibility == Compatibility::Consistent && length > limit {
                shortened(step, limit)
            } else {
                step.clone()
            }
        } else if length > r_back {
            shortened(step, r_back)
        } else if length > r_ahead {
            shortened(step, r_ahead)
        } else {
            step.clone()
        };
        out.steps.push(limited);
    }
    out
}

/// A steepest-descent step of the chain: [`STEEPEST_FACTOR`] times the force, an inner
/// image at most half the distance to its previous neighbour or, if it is shorter than
/// that, to its next one; free end images without a limit.
pub fn steepest_steps(chain: &[Col<f64>], forces: &[Col<f64>], ends: FixedEnds) -> Vec<Col<f64>> {
    let images = chain.len();
    (0..images)
        .map(|i| {
            let step = STEEPEST_FACTOR * &forces[i];
            if i == 0 || i == images - 1 {
                let fixed = if i == 0 { ends.first } else { ends.last };
                return if fixed {
                    Col::zeros(step.nrows())
                } else {
                    step
                };
            }
            let r_back = 0.5 * col_norm(&(&chain[i] - &chain[i - 1]));
            let r_ahead = 0.5 * col_norm(&(&chain[i] - &chain[i + 1]));
            let length = col_norm(&step);
            if length > r_back {
                shortened(&step, r_back)
            } else if length > r_ahead {
                shortened(&step, r_ahead)
            } else {
                step
            }
        })
        .collect()
}

/// Inner images whose force and step are below `criteria`; their steps are set to zero.
///
/// MultiOptPy compares the largest signed components (a large negative component passes);
/// the absolute values are compared here.
pub fn hold_converged(
    forces: &[Col<f64>],
    steps: &mut [Col<f64>],
    criteria: &ImageCriteria,
) -> Vec<usize> {
    let rms = |v: &Col<f64>| (col_dot(v, v) / v.nrows().max(1) as f64).sqrt();
    let largest = |v: &Col<f64>| v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let mut converged = Vec::new();
    for i in 1..forces.len().saturating_sub(1) {
        if largest(&forces[i]) < criteria.max_force
            && rms(&forces[i]) < criteria.rms_force
            && largest(&steps[i]) < criteria.max_displacement
            && rms(&steps[i]) < criteria.rms_displacement
        {
            steps[i] = Col::zeros(steps[i].nrows());
            converged.push(i);
        }
    }
    converged
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    #[test]
    fn images_do_not_pass_their_neighbours() {
        // Images along x at 0, 1, 3, 4.
        let chain: Vec<Col<f64>> = [0.0, 1.0, 3.0, 4.0]
            .iter()
            .map(|&x| col(&[x, 0.0]))
            .collect();
        let forces = vec![col(&[2.0, 0.0]); 4];
        // Image 1 towards image 2 (two away): at most 1. Image 2 the same way: at most 0.5.
        let steps = vec![col(&[2.0, 0.0]); 4];
        let plain = Compatibility::Consistent;
        let found = limit_steps(&chain, &forces, &steps, FixedEnds::default(), plain);
        assert_eq!(found.steps[0], col(&[0.5, 0.0]));
        assert_eq!(found.steps[1], col(&[1.0, 0.0]));
        assert_eq!(found.steps[2], col(&[0.5, 0.0]));
        assert_eq!(found.to_previous, vec![1.0, 2.0]);
        assert_eq!(found.to_next, vec![2.0, 1.0]);
        // At a right angle to the chain: the rule of the last case, half the distance to
        // the previous image. Away from both neighbours: as it is.
        let sideways = vec![col(&[0.0, 5.0]); 4];
        let found = limit_steps(&chain, &sideways, &sideways, FixedEnds::from_code(3), plain);
        assert_eq!(found.steps[1], col(&[0.0, 0.5]));
        // Away from both neighbours: half the larger distance to them, or (MultiOptPy) any
        // length.
        let bent = vec![col(&[0.0, 0.0]), col(&[1.0, -1.0]), col(&[4.0, 3.0])];
        let away = vec![col(&[0.0, -9.0]); 3];
        let found = limit_steps(&bent, &away, &away, FixedEnds::default(), plain);
        assert!((found.steps[1][1] + 2.5).abs() < 1e-14 && found.steps[1][0] == 0.0);
        let legacy = Compatibility::MultiOptPy;
        let found = limit_steps(&bent, &away, &away, FixedEnds::default(), legacy);
        assert_eq!(found.steps[1], col(&[0.0, -9.0]));
        let found = limit_steps(&chain, &sideways, &sideways, FixedEnds::from_code(3), plain);
        assert_eq!(found.steps[0], col(&[0.0, 0.0]));
        assert_eq!(found.steps[3], col(&[0.0, 0.0]));
        let back = vec![col(&[-0.1, 0.0]); 4];
        let against = limit_steps(&chain, &forces, &back, FixedEnds::default(), plain);
        assert_eq!(against.held, vec![1, 2]);
        assert_eq!(against.steps[1], col(&[0.0, 0.0]));
        assert_eq!(
            FixedEnds::from_code(1),
            FixedEnds {
                first: true,
                last: false
            }
        );
        assert_eq!(
            FixedEnds::from_code(2),
            FixedEnds {
                first: false,
                last: true
            }
        );
        assert_eq!(FixedEnds::from_code(0), FixedEnds::default());
    }

    #[test]
    fn steepest_descent_and_converged_images() {
        let chain: Vec<Col<f64>> = [0.0, 1.0, 3.0].iter().map(|&x| col(&[x, 0.0])).collect();
        let forces = vec![col(&[4.0, 0.0]), col(&[4.0, 0.0]), col(&[0.0, 1e-5])];
        let steps = steepest_steps(
            &chain,
            &forces,
            FixedEnds {
                first: false,
                last: true,
            },
        );
        assert_eq!(steps[0], col(&[2.0, 0.0]));
        assert_eq!(steps[1], col(&[0.5, 0.0]));
        assert_eq!(steps[2], col(&[0.0, 0.0]));
        let small = vec![col(&[0.0; 2]), col(&[-1e-5, 1e-5]), col(&[1.0; 2])];
        let mut moves = vec![col(&[1.0; 2]), col(&[1e-5, -1e-4]), col(&[1.0; 2])];
        let held = hold_converged(&small, &mut moves, &ImageCriteria::default());
        assert_eq!(held, vec![1]);
        assert_eq!(moves[1], col(&[0.0; 2]));
        assert_eq!(moves[0], col(&[1.0; 2]));
        let large = vec![col(&[0.0; 2]), col(&[-1.0, 0.0]), col(&[1.0; 2])];
        assert!(hold_converged(&large, &mut moves, &ImageCriteria::default()).is_empty());
    }
}
