//! Tangents of a chain that are carried outwards from its highest image (the tangents of
//! the quadratic string forms, `-qsmv2`).
//!
//! The rule is that of Ayala and Schlegel (J. Chem. Phys. 1997, 107, 375) as MultiOptPy
//! (`MEP/pathopt_qsmv2_force.py`) and ase_qsm (`force.py`) implement it; the paper itself
//! was not at hand. The tangent at the highest image comes from its two neighbours,
//! `t = (q_next - q) / |q_next - q|^2 - (q_prev - q) / |q_prev - q|^2`. From there the
//! tangent of every further image follows from the image above it, `q_up` with the tangent
//! `t_up`, and the chord `c = q - q_up`: by the arc through both points that leaves `q_up`
//! along `t_up` while the chord makes an angle of at most 45 degrees with `t_up`,
//!
//! ```text
//! r = |c|^2 / (2 t_up . c),    t = (c - r t_up) / r,
//! ```
//!
//! and by a parabola beyond, `t = n - tan(theta - 45 deg) (t_up - n)` with the angle
//! `theta` between chord and `t_up` and the unit vector `n` of the part of the chord across
//! `t_up`. The arc gives the exact tangent of a circular chain.
//!
//! **Difference from the two programs.** Both carry the tangent of the highest image, which
//! points to the next image, to both sides. Towards the first image the chord points the
//! other way, so the angle is near 180 degrees and the parabola is used even for a nearly
//! straight chain; its tangent agrees with the arc to first order in the bend only. Here
//! the tangent is carried down each side in the direction of that side (pointing away from
//! the highest image), with the same two formulas, and turned back to point along the
//! chain at the end: both sides are treated alike, and the arc is exact on both.
//!
//! The rule assumes one maximum of the energies along the chain. With several, the caller
//! uses the tangents towards the higher neighbour (as ase_qsm's default does).
//!
//! MultiOptPy: `MEP/pathopt_qsmv2_force.py:CaluculationQSMv2._calculate_all_tangents`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::math::linalg::{col_dot, col_norm};

/// Lengths below which a direction is taken as zero.
const TINY: f64 = 1.0e-10;

fn unit(v: Col<f64>) -> Col<f64> {
    let norm = col_norm(&v);
    if norm > TINY {
        v / norm
    } else {
        v
    }
}

/// The unit tangent at `q` from the image above it, `q_up`, whose unit tangent `t_up`
/// points from `q_up` towards `q`'s side.
fn carried(q: &Col<f64>, q_up: &Col<f64>, t_up: &Col<f64>) -> Col<f64> {
    let chord = q - q_up;
    let length = col_norm(&chord);
    if length < TINY {
        return t_up.clone();
    }
    let along = col_dot(&chord, t_up);
    let cosine = (along / length).clamp(-1.0, 1.0);
    if cosine >= std::f64::consts::FRAC_1_SQRT_2 {
        // The arc; `along` is at least |c| / sqrt(2) here.
        let r = length * length / (2.0 * along);
        unit((&chord - r * t_up) / r)
    } else {
        let across = unit(&chord - along * t_up);
        let tangent = (cosine.acos() - std::f64::consts::FRAC_PI_4).tan();
        unit(&across - tangent * (t_up - &across))
    }
}

/// The directions of `chain` at its inner images (unit vectors pointing to the next
/// image; zero vectors at the two ends), carried outwards from the image of the highest
/// energy. `None` for a chain of fewer than three images.
pub fn carried_from_the_top(chain: &[Col<f64>], energies: &[f64]) -> Option<Vec<Col<f64>>> {
    let images = chain.len();
    if images < 3 || energies.len() != images {
        return None;
    }
    let n = chain[0].nrows();
    let top = (1..images - 1).max_by(|a, b| energies[*a].total_cmp(&energies[*b]))?;
    let mut tangents = vec![Col::<f64>::zeros(n); images];
    let to_next = &chain[top + 1] - &chain[top];
    let to_previous = &chain[top - 1] - &chain[top];
    let next_squared = col_dot(&to_next, &to_next).max(TINY);
    let previous_squared = col_dot(&to_previous, &to_previous).max(TINY);
    tangents[top] = unit(to_next / next_squared - to_previous / previous_squared);
    // Down the side of the last image: the tangents point away from the top as they are.
    for i in top + 1..images - 1 {
        tangents[i] = carried(&chain[i], &chain[i - 1], &tangents[i - 1]);
    }
    // Down the side of the first image: carried pointing away from the top, kept pointing
    // along the chain.
    let mut away = -&tangents[top];
    for i in (1..top).rev() {
        away = carried(&chain[i], &chain[i + 1], &away);
        tangents[i] = -&away;
    }
    Some(tangents)
}

/// `tangents` (unit vectors, as [`carried_from_the_top`] makes them) with the length of the
/// distance from each inner image to its higher neighbour (the mean of the two distances
/// at an image that is higher than both): the directions of a chain carry that distance,
/// from which the quasi-Newton steps take how much the force of an image changes when its
/// neighbour uphill moves.
pub fn with_the_distance_uphill(
    mut tangents: Vec<Col<f64>>,
    chain: &[Col<f64>],
    energies: &[f64],
) -> Vec<Col<f64>> {
    let images = chain.len();
    for i in 1..images.saturating_sub(1) {
        let to = |j: usize| col_norm(&(&chain[j] - &chain[i]));
        let (before, here, after) = (energies[i - 1], energies[i], energies[i + 1]);
        let distance = if after > here && after >= before {
            to(i + 1)
        } else if before > here {
            to(i - 1)
        } else {
            0.5 * (to(i + 1) + to(i - 1))
        };
        tangents[i] *= distance;
    }
    tangents
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64) -> Col<f64> {
        Col::from_fn(2, |k| if k == 0 { x } else { y })
    }

    #[test]
    fn the_tangents_of_a_circular_chain_are_exact_on_both_sides() {
        // Images on the unit circle at unequal angles, the highest one in the middle.
        let angles = [-1.1_f64, -0.7, -0.45, -0.2, 0.0, 0.15, 0.5, 0.8, 1.2];
        let chain: Vec<Col<f64>> = angles.iter().map(|a| point(a.cos(), a.sin())).collect();
        let energies: Vec<f64> = angles.iter().map(|a| -a * a).collect();
        let tangents = carried_from_the_top(&chain, &energies).unwrap();
        assert_eq!(col_norm(&tangents[0]), 0.0);
        assert_eq!(col_norm(&tangents[8]), 0.0);
        for (i, a) in angles.iter().enumerate().skip(1).take(7) {
            let exact = point(-a.sin(), a.cos());
            let error = col_norm(&(&tangents[i] - &exact));
            // The tangent of the top image from unequal neighbours is not exact; the
            // carried ones are exact given that one.
            if i == 4 {
                assert!(error < 0.05, "top: {error}");
            }
            assert!((col_norm(&tangents[i]) - 1.0).abs() < 1e-12);
            assert!(col_dot(&tangents[i], &exact) > 0.99, "image {i}");
        }
        // With the exact tangent at the top the arc reproduces every other one.
        let exact_top = point(0.0, 1.0);
        let mut t = exact_top.clone();
        for i in 5..8 {
            t = carried(&chain[i], &chain[i - 1], &t);
            let exact = point(-angles[i].sin(), angles[i].cos());
            assert!(col_norm(&(&t - &exact)) < 1e-12, "image {i}");
        }
        let mut away = -&exact_top;
        for i in (1..4).rev() {
            away = carried(&chain[i], &chain[i + 1], &away);
            let exact = point(-angles[i].sin(), angles[i].cos());
            assert!(col_norm(&(-&away - &exact)) < 1e-12, "image {i}");
        }
    }

    #[test]
    fn a_straight_chain_a_sharp_bend_and_short_chains() {
        let straight: Vec<Col<f64>> = (0..5).map(|k| point(f64::from(k), 0.0)).collect();
        let energies = [0.0, 1.0, 3.0, 1.0, 0.0];
        let tangents = carried_from_the_top(&straight, &energies).unwrap();
        for t in &tangents[1..4] {
            assert!(col_norm(&(t - &point(1.0, 0.0))) < 1e-12);
        }
        // A bend of more than 45 degrees between chord and tangent: the parabola, a unit
        // vector that turns towards the chord.
        let bent = [
            point(-1.0, 0.0),
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(1.2, 1.0),
            point(1.2, 2.0),
        ];
        let tangents = carried_from_the_top(&bent, &[0.0, 3.0, 2.0, 1.0, 0.0]).unwrap();
        assert!((col_norm(&tangents[3]) - 1.0).abs() < 1e-12);
        assert!(
            tangents[3][1] > 0.5 && tangents[3][0] > -0.5,
            "{:?}",
            tangents[3]
        );
        assert!(carried_from_the_top(&bent[..2], &[0.0, 1.0]).is_none());
        // The lengths of the directions: the distance to the higher neighbour.
        let energies = [0.0, 3.0, 2.0, 1.0, 0.0];
        let scaled = with_the_distance_uphill(tangents, &bent, &energies);
        assert!(
            (col_norm(&scaled[1]) - 1.0).abs() < 1e-12,
            "the mean of 1 and 1"
        );
        assert!((col_norm(&scaled[2]) - 1.0).abs() < 1e-12, "to image 1");
        let to_two = (0.2_f64 * 0.2 + 1.0).sqrt();
        assert!((col_norm(&scaled[3]) - to_two).abs() < 1e-12, "to image 2");
        assert_eq!(col_norm(&scaled[0]), 0.0);
    }
}
