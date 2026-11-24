//! Displacement directions of O1NumHess: neighbour lists from a distance matrix and directions
//! that span every neighbourhood with as few vectors as possible.
//!
//! A new direction is put together from local modes: in each neighbourhood that the existing
//! directions do not span yet, the model Hessian restricted to it is projected onto the
//! complement of the existing directions (restricted to the same indices) and its eigenvector
//! of the largest eigenvalue is taken. The local modes are averaged into one vector, each with
//! the sign that gives the larger result, and the vector is orthogonalized against the existing
//! directions. One direction therefore serves all neighbourhoods at once, and the number of
//! directions is the size of the largest neighbourhood rather than the number of variables.
//!
//! The averaging can cancel the part of a direction that a neighbourhood still lacks, so a
//! neighbourhood is not always spanned completely (seen for chains of identical
//! neighbourhoods). The symmetry of the Hessian makes up for it: for an anharmonic banded
//! function the error stayed within twice that of full finite differences
//! (`tests/o1numhess.rs`), so no further directions are added.
//!
//! Safeguards beyond the reference implementation: the orthogonalization is done twice, and a
//! seed that loses eight digits in it is taken as dependent (the reference tests the norm
//! against `1e-15` only).
//!
//! MultiOptPy: `ModelHessian/o1numhess.py:O1NumHessCalculator._build_neighbor_list`,
//!  `_generate_displacement_directions_corrected` (MultiOptPy's cutoffs and eigenvector choice
//!  differ from the published method; this file follows the publication and its reference
//!  implementation).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;

use faer::{Col, Mat};
use rayon::prelude::*;

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen, symmetrize};

/// Distances and norms closer than this count as equal (the reference's `eps`).
const TIE: f64 = 1e-8;
/// A seed shorter than this after orthogonalization ends the generation (the reference's
/// `eps2`).
const VANISHED: f64 = 1e-15;
/// A seed that keeps less than this fraction of its length in the orthogonalization is
/// dependent on the existing directions.
const DEPENDENT: f64 = 1e-8;

/// Neighbour lists: `j` is a neighbour of `i` when `distance[(i, j)] < dmax`. Parts that are
/// not connected are joined along a minimum spanning tree of their closest index pairs (all
/// pairs within `1e-8` of the closest one).
pub(crate) fn neighbor_lists(distance: &Mat<f64>, dmax: f64) -> Vec<Vec<usize>> {
    let n = distance.nrows();
    let mut lists: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&j| distance[(i, j)] < dmax).collect())
        .collect();
    let (part, count) = connected_parts(&lists);
    if count < 2 {
        return lists;
    }
    // Closest distance between every two parts.
    let mut closest = vec![f64::INFINITY; count * count];
    for i in 0..n {
        for j in 0..n {
            let slot = &mut closest[part[i] * count + part[j]];
            if part[i] != part[j] && distance[(i, j)] < *slot {
                *slot = distance[(i, j)];
            }
        }
    }
    // Prim's algorithm over the parts.
    let mut in_tree = vec![false; count];
    let mut key: Vec<f64> = (0..count).map(|b| closest[b]).collect();
    let mut parent = vec![0usize; count];
    in_tree[0] = true;
    let mut edges = Vec::with_capacity(count - 1);
    for _ in 1..count {
        let next = (0..count)
            .filter(|&b| !in_tree[b])
            .min_by(|&a, &b| key[a].total_cmp(&key[b]));
        let Some(b) = next else { break };
        in_tree[b] = true;
        edges.push((parent[b], b));
        for c in 0..count {
            if !in_tree[c] && closest[b * count + c] < key[c] {
                key[c] = closest[b * count + c];
                parent[c] = b;
            }
        }
    }
    for (a, b) in edges {
        let limit = closest[a * count + b] + TIE;
        for i in (0..n).filter(|&i| part[i] == a) {
            for j in (0..n).filter(|&j| part[j] == b) {
                if distance[(i, j)] <= limit {
                    lists[i].push(j);
                    lists[j].push(i);
                }
            }
        }
    }
    for list in &mut lists {
        list.sort_unstable();
        list.dedup();
    }
    lists
}

/// The connected part of every index and the number of parts.
fn connected_parts(lists: &[Vec<usize>]) -> (Vec<usize>, usize) {
    let n = lists.len();
    let mut part = vec![usize::MAX; n];
    let mut count = 0;
    for start in 0..n {
        if part[start] != usize::MAX {
            continue;
        }
        part[start] = count;
        let mut stack = vec![start];
        while let Some(i) = stack.pop() {
            for &j in &lists[i] {
                if part[j] == usize::MAX {
                    part[j] = count;
                    stack.push(j);
                }
            }
        }
        count += 1;
    }
    (part, count)
}

/// Orthonormal basis of the column space of `a` (singular vectors above
/// `eps * max(rows, cols) * s_max`, as `scipy.linalg.orth`).
fn range_basis(a: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let svd = a
        .thin_svd()
        .map_err(|e| MoptError::linalg(format!("SVD failed: {e:?}")))?;
    let (u, s) = (svd.U(), svd.S());
    let k = s.dim();
    let largest = (0..k).fold(0.0_f64, |m, i| m.max(s[i]));
    let cutoff = f64::EPSILON * a.nrows().max(a.ncols()) as f64 * largest;
    let keep: Vec<usize> = (0..k).filter(|&i| s[i] > cutoff).collect();
    Ok(Mat::from_fn(a.nrows(), keep.len(), |r, c| u[(r, keep[c])]))
}

/// Eigenvector of the largest eigenvalue of the model Hessian on the indices `nb`, projected
/// onto the complement of the directions restricted to `nb`.
fn local_mode(model: &Mat<f64>, nb: &[usize], dirs: &[Col<f64>]) -> MoptResult<Col<f64>> {
    let k = nb.len();
    let sub = Mat::from_fn(k, k, |a, b| model[(nb[a], nb[b])]);
    let mut projected = if dirs.is_empty() {
        sub
    } else {
        let restricted = Mat::from_fn(k, dirs.len(), |r, c| dirs[c][nb[r]]);
        let q = range_basis(&restricted)?;
        // P S P with P = 1 - Q Q^T.
        let left = &sub - &q * (q.transpose() * &sub);
        &left - (&left * &q) * q.transpose()
    };
    symmetrize(&mut projected);
    let (_, vectors) = symmetric_eigen(&projected)?;
    Ok(Col::from_fn(k, |r| vectors[(r, k - 1)]))
}

/// Average the local mode `loc` (on the indices `nb`) into `seed` with the sign that gives
/// the longer vector; `cover` counts the neighbourhoods each index has received.
fn merge(seed: &mut Col<f64>, cover: &mut [usize], nb: &[usize], loc: &Col<f64>) {
    let (mut plus, mut minus) = (0.0_f64, 0.0_f64);
    for (a, &idx) in nb.iter().enumerate() {
        let c = cover[idx] as f64;
        plus += ((c * seed[idx] + loc[a]) / (c + 1.0)).powi(2);
        minus += ((c * seed[idx] - loc[a]) / (c + 1.0)).powi(2);
    }
    let (plus, minus) = (plus.sqrt(), minus.sqrt());
    let sign = if (plus - minus).abs() > TIE {
        if plus > minus {
            1.0
        } else {
            -1.0
        }
    } else {
        // A tie: the largest component of the local mode is made positive.
        let largest = loc
            .iter()
            .fold(0.0_f64, |m, &v| if v.abs() > m.abs() { v } else { m });
        if largest > 0.0 {
            1.0
        } else {
            -1.0
        }
    };
    for (a, &idx) in nb.iter().enumerate() {
        let c = cover[idx] as f64;
        seed[idx] = (c * seed[idx] + sign * loc[a]) / (c + 1.0);
        cover[idx] += 1;
    }
}

/// Remove from `v` its components along the orthonormal `dirs` (twice) and return the norm
/// that is left.
pub(crate) fn orthogonalize(v: &mut Col<f64>, dirs: &[Col<f64>]) -> f64 {
    for _pass in 0..2 {
        for d in dirs {
            let p = col_dot(d, v);
            for i in 0..v.nrows() {
                v[i] -= p * d[i];
            }
        }
    }
    col_norm(v)
}

/// Add to the orthonormal `dirs` the directions that make them span every neighbourhood of
/// `lists`, guided by the model Hessian.
pub(crate) fn generate(
    lists: &[Vec<usize>],
    model: &Mat<f64>,
    dirs: &mut Vec<Col<f64>>,
) -> MoptResult<()> {
    let n = model.nrows();
    // Indices with the same neighbourhood (the three coordinates of an atom) share the local
    // mode.
    let mut index: HashMap<&[usize], usize> = HashMap::new();
    let mut groups: Vec<&[usize]> = Vec::new();
    let group_of: Vec<usize> = lists
        .iter()
        .map(|list| {
            *index.entry(list.as_slice()).or_insert_with(|| {
                groups.push(list.as_slice());
                groups.len() - 1
            })
        })
        .collect();
    while dirs.len() < n {
        let count = dirs.len();
        let active: Vec<usize> = (0..groups.len())
            .filter(|&g| groups[g].len() > count)
            .collect();
        if active.is_empty() {
            break;
        }
        let modes: Vec<MoptResult<Col<f64>>> = active
            .par_iter()
            .map(|&g| local_mode(model, groups[g], dirs))
            .collect();
        let mut local: Vec<Option<Col<f64>>> = vec![None; groups.len()];
        for (&g, mode) in active.iter().zip(modes) {
            local[g] = Some(mode?);
        }
        let mut seed = Col::<f64>::zeros(n);
        let mut cover = vec![0usize; n];
        for j in 0..n {
            if let Some(loc) = &local[group_of[j]] {
                merge(&mut seed, &mut cover, &lists[j], loc);
            }
        }
        let before = col_norm(&seed);
        let after = orthogonalize(&mut seed, dirs);
        if after < VANISHED || after < DEPENDENT * before {
            break;
        }
        dirs.push(Col::from_fn(n, |i| seed[i] / after));
    }
    Ok(())
}

/// Make the first component of `d` above `0.5 / sqrt(n)` positive; whether `d` was reversed.
pub(crate) fn fix_sign(d: &mut Col<f64>) -> bool {
    let n = d.nrows();
    let threshold = 0.5 / (n as f64).sqrt();
    let reverse = d
        .iter()
        .find(|v| v.abs() > threshold)
        .is_some_and(|&v| v < 0.0);
    if reverse {
        for i in 0..n {
            d[i] = -d[i];
        }
    }
    reverse
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn banded(n: usize) -> Mat<f64> {
        Mat::from_fn(n, n, |i, j| (i as f64 - j as f64).abs())
    }

    #[test]
    fn separate_parts_are_joined_by_their_closest_pairs() {
        // Two groups of three indices, far apart.
        let d = Mat::from_fn(6, 6, |i, j| {
            if (i < 3) == (j < 3) {
                0.5
            } else {
                10.0 + (i as f64 - 2.0).abs() + (j as f64 - 3.0).abs()
            }
        });
        let lists = neighbor_lists(&d, 1.0);
        assert_eq!(lists[0], vec![0, 1, 2]);
        assert_eq!(lists[2], vec![0, 1, 2, 3]);
        assert_eq!(lists[3], vec![2, 3, 4, 5]);
        let (_, count) = connected_parts(&lists);
        assert_eq!(count, 1);
    }

    #[test]
    fn as_many_orthonormal_directions_as_the_largest_neighbourhood() {
        let n = 30;
        let lists = neighbor_lists(&banded(n), 2.5);
        let model = Mat::from_fn(n, n, |i, j| {
            if i == j {
                2.0 + 0.1 * i as f64
            } else {
                -1.0 / (1.0 + (i as f64 - j as f64).abs())
            }
        });
        let mut dirs = Vec::new();
        generate(&lists, &model, &mut dirs).unwrap();
        // Five neighbours at most, so five directions for thirty variables.
        assert_eq!(dirs.len(), 5);
        for (a, da) in dirs.iter().enumerate() {
            for (b, db) in dirs.iter().enumerate() {
                let expect = f64::from(u8::from(a == b));
                assert!((col_dot(da, db) - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn signs_follow_the_first_large_component() {
        let mut d = Col::from_fn(4, |i| [0.1, -0.7, 0.7, 0.1][i]);
        assert!(fix_sign(&mut d));
        assert!(d[1] > 0.0 && d[2] < 0.0);
        assert!(!fix_sign(&mut d));
    }
}
