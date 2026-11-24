//! A structure for a point of a ReSPer map (the pre-image problem of the projection
//! [`super`]), for rigid scans of ReSPer coordinates.
//!
//! The references are superposed on each other in their order (each mass-weighted onto the
//! one before it, so that a path keeps one orientation) and a structure for the point `t`
//! of the map is interpolated between the references next to it there:
//!
//! - in one coordinate, linearly between the two consecutive references whose coordinates
//!   enclose `t` (the shortest such pair where a path folds back on itself);
//! - in two coordinates, barycentrically over the smallest triangle of the nearest
//!   references that contains `t`; where none does (a path is a thin band in its map), along
//!   the segment of two consecutive references nearest to `t`;
//! - beyond the references, the nearest reference itself.
//!
//! The coordinates of the interpolated structure (Gower's formula) are compared with `t`: the
//! distance is the error of the pre-image. With the correction (the default of the scans) the
//! structure is then moved onto `t` by Newton steps in the metric of the map (the shortest
//! mass-weighted way), and the length of that correction is reported. A point whose nearest
//! reference is further than `outside_factor` times the usual spacing of the references in the
//! map, or that lies beyond their range (one coordinate) or their convex hull (two), is
//! extrapolated.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::ResperMap;
use crate::coord::alignment::kabsch_align_bohr;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// Nearest references whose triangles are tried (two coordinates).
const NEIGHBOURS: usize = 8;
/// Newton steps of the correction.
const CORRECTION_STEPS: usize = 50;
/// Largest remaining distance (in the units of the map) of a corrected pre-image.
const CORRECTION_TOLERANCE: f64 = 1e-8;

/// Settings of the pre-image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreimageSettings {
    /// Move the interpolated structure onto the point.
    pub correct: bool,
    /// Points whose nearest reference is further than this many usual spacings of the
    /// references are extrapolated.
    pub outside_factor: f64,
}

impl Default for PreimageSettings {
    fn default() -> Self {
        Self {
            correct: true,
            outside_factor: 2.0,
        }
    }
}

/// A structure (flat positions in Bohr, or a point of plain variables) for a point of the
/// map.
#[derive(Debug, Clone)]
pub struct Preimage {
    /// The structure.
    pub x: Col<f64>,
    /// The references it was interpolated from, with their weights.
    pub weights: Vec<(usize, f64)>,
    /// Distance in the map between the point and the coordinates of the interpolated
    /// structure (before the correction).
    pub error: f64,
    /// The same after the correction (equal to `error` without it).
    pub remaining: f64,
    /// Length of the correction in the metric of the map (Angstrom amu^(1/2), or the unit of
    /// the variables).
    pub correction: f64,
    /// Distance in the map from the point to the nearest reference.
    pub nearest: f64,
    /// Whether the point lies beyond the references.
    pub extrapolated: bool,
}

/// The references superposed in their order, as flat vectors (Bohr or variables).
pub fn aligned_references(map: &ResperMap) -> MoptResult<Vec<Col<f64>>> {
    if !map.of_structures() {
        return Ok(map.reference_points().to_vec());
    }
    let frames = map.reference_structures();
    let mut out = Vec::with_capacity(frames.len());
    let mut previous = frames[0].clone();
    for frame in frames {
        let (aligned, _) = kabsch_align_bohr(frame, &previous, true)?;
        out.push(aligned.coords_flat_bohr());
        previous = aligned;
    }
    Ok(out)
}

/// The map coordinates `components` of reference `i`.
fn point_of(map: &ResperMap, i: usize, components: &[usize]) -> Vec<f64> {
    components
        .iter()
        .map(|&k| map.reference_coordinate(i, k))
        .collect()
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt()
}

/// The median distance of a reference to its nearest neighbour in the map.
pub fn usual_spacing(map: &ResperMap, components: &[usize]) -> f64 {
    let n = map.len();
    let points: Vec<Vec<f64>> = (0..n).map(|i| point_of(map, i, components)).collect();
    let mut nearest: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| distance(&points[i], &points[j]))
                .fold(f64::INFINITY, f64::min)
        })
        .filter(|d| d.is_finite())
        .collect();
    nearest.sort_by(f64::total_cmp);
    nearest.get(nearest.len() / 2).copied().unwrap_or(0.0)
}

/// Barycentric coordinates of `t` in the triangle `a, b, c` (`None` when it is degenerate).
fn barycentric(t: &[f64], a: &[f64], b: &[f64], c: &[f64]) -> Option<[f64; 3]> {
    let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    let scale = distance(a, b).max(distance(b, c)).max(distance(a, c));
    if det.abs() <= 1e-12 * scale * scale || scale == 0.0 {
        return None;
    }
    let l1 = ((b[1] - c[1]) * (t[0] - c[0]) + (c[0] - b[0]) * (t[1] - c[1])) / det;
    let l2 = ((c[1] - a[1]) * (t[0] - c[0]) + (a[0] - c[0]) * (t[1] - c[1])) / det;
    Some([l1, l2, 1.0 - l1 - l2])
}

/// Whether `t` lies in the convex hull of `points` (two coordinates).
fn inside_hull(t: &[f64], points: &[Vec<f64>]) -> bool {
    // Andrew's monotone chain, then the side of every edge.
    let mut p: Vec<[f64; 2]> = points.iter().map(|v| [v[0], v[1]]).collect();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return false;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let ordered: Vec<[f64; 2]> = if pass == 0 {
            p.clone()
        } else {
            p.iter().rev().copied().collect()
        };
        for q in ordered {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0
            {
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    if hull.len() < 3 {
        return false;
    }
    let point = [t[0], t[1]];
    (0..hull.len()).all(|k| cross(hull[k], hull[(k + 1) % hull.len()], point) >= -1e-12)
}

/// Interpolation weights for `t` and whether it lies beyond the range or hull of the
/// references.
fn weights(map: &ResperMap, components: &[usize], t: &[f64]) -> (Vec<(usize, f64)>, bool) {
    let n = map.len();
    let points: Vec<Vec<f64>> = (0..n).map(|i| point_of(map, i, components)).collect();
    let nearest = (0..n)
        .min_by(|&a, &b| distance(&points[a], t).total_cmp(&distance(&points[b], t)))
        .unwrap_or(0);
    if components.len() == 1 {
        let (lo, hi) = points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| {
                (l.min(p[0]), h.max(p[0]))
            });
        let beyond = t[0] < lo || t[0] > hi;
        // Consecutive references that enclose t; the shortest pair in the whole map.
        let best = (0..n.saturating_sub(1))
            .filter(|&i| {
                let (a, b) = (points[i][0], points[i + 1][0]);
                a.min(b) <= t[0] && t[0] <= a.max(b) && a != b
            })
            .min_by(|&i, &j| {
                let full = |k: usize| {
                    let all: Vec<usize> = (0..map.components()).collect();
                    distance(&point_of(map, k, &all), &point_of(map, k + 1, &all))
                };
                full(i).total_cmp(&full(j))
            });
        return match best {
            Some(i) => {
                let (a, b) = (points[i][0], points[i + 1][0]);
                let s = (t[0] - a) / (b - a);
                (vec![(i, 1.0 - s), (i + 1, s)], beyond)
            }
            None => (vec![(nearest, 1.0)], true),
        };
    }
    let beyond = !inside_hull(t, &points);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| distance(&points[a], t).total_cmp(&distance(&points[b], t)));
    let near: Vec<usize> = order.into_iter().take(NEIGHBOURS).collect();
    let mut best: Option<(f64, [usize; 3], [f64; 3])> = None;
    for (x, &a) in near.iter().enumerate() {
        for (y, &b) in near.iter().enumerate().skip(x + 1) {
            for &c in near.iter().skip(y + 1) {
                let Some(l) = barycentric(t, &points[a], &points[b], &points[c]) else {
                    continue;
                };
                if l.iter().any(|v| *v < -1e-9) {
                    continue;
                }
                let perimeter = distance(&points[a], &points[b])
                    + distance(&points[b], &points[c])
                    + distance(&points[a], &points[c]);
                if best.is_none_or(|(p, _, _)| perimeter < p) {
                    best = Some((perimeter, [a, b, c], l));
                }
            }
        }
    }
    if let Some((_, idx, l)) = best {
        return (idx.iter().copied().zip(l).collect(), beyond);
    }
    // No triangle around t: along the nearest segment of consecutive references.
    let segment = (0..n.saturating_sub(1))
        .map(|i| {
            let (a, b) = (&points[i], &points[i + 1]);
            let ab: Vec<f64> = (0..2).map(|k| b[k] - a[k]).collect();
            let len2 = ab[0] * ab[0] + ab[1] * ab[1];
            let s = if len2 > 0.0 {
                (((t[0] - a[0]) * ab[0] + (t[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let foot = [a[0] + s * ab[0], a[1] + s * ab[1]];
            (distance(&foot, t), i, s)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0));
    match segment {
        Some((_, i, s)) => (vec![(i, 1.0 - s), (i + 1, s)], beyond),
        None => (vec![(nearest, 1.0)], true),
    }
}

/// A structure for the point `target` of the coordinates `components` (0-based) of `map`.
pub fn preimage(
    map: &ResperMap,
    aligned: &[Col<f64>],
    components: &[usize],
    target: &[f64],
    settings: &PreimageSettings,
) -> MoptResult<Preimage> {
    if components.is_empty() || components.len() > 2 || components.len() != target.len() {
        return Err(MoptError::config(
            "ReSPer pre-image: one or two coordinates with a value each",
        ));
    }
    if let Some(k) = components.iter().find(|&&k| k >= map.components()) {
        return Err(MoptError::config(format!(
            "ReSPer: the map has {} coordinate(s); there is no PCo{}",
            map.components(),
            k + 1
        )));
    }
    if aligned.len() != map.len() {
        return Err(MoptError::config(
            "ReSPer pre-image: one superposed structure per reference",
        ));
    }
    let (w, beyond) = weights(map, components, target);
    let dof = aligned[0].nrows();
    let mut x = Col::<f64>::zeros(dof);
    for &(i, weight) in &w {
        x += &aligned[i] * faer::Scale(weight);
    }
    let landed = |x: &Col<f64>| -> MoptResult<(Vec<f64>, Vec<Col<f64>>)> {
        let (values, gradients) = map.project_flat(x)?;
        Ok((
            components
                .iter()
                .zip(target)
                .map(|(&k, t)| values[k] - t)
                .collect(),
            components.iter().map(|&k| gradients[k].clone()).collect(),
        ))
    };
    let norm = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let (mut residual, mut gradients) = landed(&x)?;
    let error = norm(&residual);
    let start = x.clone();
    if settings.correct {
        // Newton steps dx = -W J^T (J W J^T)^+ r with W the inverse masses of the metric.
        let metric = map.metric_weights(dof);
        let inverse: Vec<f64> = metric
            .iter()
            .map(|m| if *m > 0.0 { 1.0 / m } else { 0.0 })
            .collect();
        for _ in 0..CORRECTION_STEPS {
            if norm(&residual) < CORRECTION_TOLERANCE {
                break;
            }
            let m = components.len();
            let j = Mat::from_fn(m, dof, |r, c| gradients[r][c]);
            let jw = Mat::from_fn(m, dof, |r, c| j[(r, c)] * inverse[c]);
            let gram = &jw * j.transpose();
            let (values, vectors) = symmetric_eigen(&gram)?;
            let largest = values.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            let r = Col::from_fn(m, |k| residual[k]);
            let mut solved = Col::<f64>::zeros(m);
            for k in (0..m).filter(|&k| values[k] > 1e-12 * largest.max(1e-300)) {
                let along: f64 = (0..m).map(|a| vectors[(a, k)] * r[a]).sum();
                for a in 0..m {
                    solved[a] += vectors[(a, k)] * along / values[k];
                }
            }
            let dx = jw.transpose() * &solved;
            x -= &dx;
            (residual, gradients) = landed(&x)?;
        }
    }
    let remaining = norm(&residual);
    let metric = map.metric_weights(dof);
    let correction = (0..dof)
        .map(|k| metric[k] * (x[k] - start[k]).powi(2))
        .sum::<f64>()
        .sqrt()
        * if map.of_structures() {
            crate::domain::Units::BOHR_TO_ANGSTROM
        } else {
            1.0
        };
    let points: Vec<Vec<f64>> = (0..map.len())
        .map(|i| point_of(map, i, components))
        .collect();
    let nearest = points
        .iter()
        .map(|p| distance(p, target))
        .fold(f64::INFINITY, f64::min);
    let spacing = usual_spacing(map, components);
    let far = spacing > 0.0 && nearest > settings.outside_factor * spacing;
    Ok(Preimage {
        x,
        weights: w,
        error,
        remaining,
        correction,
        nearest,
        extrapolated: beyond || far,
    })
}
