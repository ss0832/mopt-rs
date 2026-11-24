//! Stationary points of a scanned surface: transition-state candidates (one negative
//! curvature: the maxima of a curve, the saddle points of a map) and minima.
//!
//! At every interior point of the grid the surface is modelled by the quadratic of its
//! discrete derivatives in grid steps: `g_d = (E(+d) - E(-d)) / 2`, `H_dd = E(+d) - 2 E +
//! E(-d)`, `H_de = (E(+d,+e) - E(+d,-e) - E(-d,+e) + E(-d,-e)) / 4`. The point holds a
//! stationary point when that of its model, `delta = -H^-1 g`, lies within half a step of it in
//! every direction; the number of negative eigenvalues of `H` says which kind (none: a
//! minimum; one: a transition-state candidate; more: a higher saddle or a maximum of a map,
//! which are not listed). On a curve this is exactly "higher (lower) than both neighbours".
//! Points on the edge of the grid and points next to a missing energy are not judged: the
//! highest point of a curve at its end says that the range of the scan ends before the
//! maximum. Where the surface is far from quadratic over a step, neighbouring points can each
//! place the same stationary point within their half step; candidates of one kind whose model
//! stationary points lie within one step of each other are one, held by the point nearest to
//! it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; `optimization.py` writes `approx_TS.txt`, the
//! local maxima of the energies of an optimization, with `scipy.signal.argrelextrema`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// What a stationary point of the scanned surface is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    /// No negative curvature.
    Minimum,
    /// One negative curvature (a maximum of a curve, a saddle point of a map).
    Saddle,
}

impl CandidateKind {
    /// `minimum` or `saddle`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Minimum => "minimum",
            Self::Saddle => "saddle",
        }
    }
}

/// A stationary point found on the grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The grid point that holds it.
    pub point: usize,
    /// Its kind.
    pub kind: CandidateKind,
    /// Energy of the grid point.
    pub energy: f64,
    /// Where the stationary point of the local model lies, in grid steps from the point
    /// (one per dimension, each within half a step).
    pub offset: Vec<f64>,
    /// Energy of the local model there.
    pub model_energy: f64,
}

/// Index of the point at `idx` on a grid of `shape` (first dimension fastest).
fn flat(idx: &[usize], shape: &[usize]) -> usize {
    let mut f = 0;
    for d in (0..shape.len()).rev() {
        f = f * shape[d] + idx[d];
    }
    f
}

/// The stationary points of the surface `energies` on a grid of `shape` (first dimension
/// fastest; missing energies are `NaN`), lowest energy first within each kind.
pub fn stationary_points(shape: &[usize], energies: &[f64]) -> MoptResult<Vec<Candidate>> {
    stationary_points_periodic(shape, energies, &vec![false; shape.len()])
}

/// [`stationary_points`] where the dimensions marked in `periodic` cover one whole period
/// with the last point equal to the first (a dihedral angle from -180 to 180 degrees): their
/// neighbours wrap around, so that the edges are judged as well, and the last point is the
/// first.
pub fn stationary_points_periodic(
    shape: &[usize],
    energies: &[f64],
    periodic: &[bool],
) -> MoptResult<Vec<Candidate>> {
    let total: usize = shape.iter().product();
    if energies.len() != total || periodic.len() != shape.len() {
        return Err(MoptError::config(format!(
            "{} energies for a grid of {total} points",
            energies.len()
        )));
    }
    let dims = shape.len();
    let mut found = Vec::new();
    if shape
        .iter()
        .zip(periodic)
        .any(|(&n, &p)| n < 3 || (p && n < 4))
    {
        return Ok(found);
    }
    let indices = |point: usize| -> Vec<usize> {
        let mut rest = point;
        shape
            .iter()
            .map(|&n| {
                let i = rest % n;
                rest /= n;
                i
            })
            .collect()
    };
    // The first and the last point of a periodic dimension are one point that the scan
    // reached twice: each copy takes the lower energy, and a candidate there is the copy
    // with it (whose structure is the better one).
    let (energies, lowest_copy) = merge_periodic_copies(shape, energies, periodic, indices);
    let energies = energies.as_slice();
    for point in 0..total {
        let idx = indices(point);
        // An edge of an open dimension is not judged; the last point of a periodic one is
        // the first.
        let skip = idx.iter().zip(shape).zip(periodic).any(|((&i, &n), &p)| {
            if p {
                i + 1 == n
            } else {
                i == 0 || i + 1 == n
            }
        });
        if skip {
            continue;
        }
        let at = |moves: &[(usize, i64)]| -> f64 {
            let mut j = idx.clone();
            for &(d, s) in moves {
                let moved = j[d] as i64 + s;
                j[d] = if periodic[d] {
                    // Distinct points 0..n-2; the point n-1 is the point 0.
                    moved.rem_euclid(shape[d] as i64 - 1) as usize
                } else {
                    moved as usize
                };
            }
            energies[flat(&j, shape)]
        };
        let e = energies[point];
        let mut g = vec![0.0; dims];
        let mut h = Mat::<f64>::zeros(dims, dims);
        let mut finite = e.is_finite();
        for d in 0..dims {
            let (plus, minus) = (at(&[(d, 1)]), at(&[(d, -1)]));
            g[d] = 0.5 * (plus - minus);
            h[(d, d)] = plus - 2.0 * e + minus;
            for f in d + 1..dims {
                let mixed = 0.25
                    * (at(&[(d, 1), (f, 1)]) - at(&[(d, 1), (f, -1)]) - at(&[(d, -1), (f, 1)])
                        + at(&[(d, -1), (f, -1)]));
                h[(d, f)] = mixed;
                h[(f, d)] = mixed;
            }
            finite &= g[d].is_finite() && (0..dims).all(|f| h[(d, f)].is_finite());
        }
        if !finite {
            continue;
        }
        let (values, vectors) = symmetric_eigen(&h)?;
        let scale = values.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if scale == 0.0 || values.iter().any(|v| v.abs() <= 1e-12 * scale) {
            continue;
        }
        let mut delta = vec![0.0; dims];
        for k in 0..dims {
            let along: f64 = (0..dims).map(|d| vectors[(d, k)] * g[d]).sum();
            for d in 0..dims {
                delta[d] -= vectors[(d, k)] * along / values[k];
            }
        }
        if delta.iter().any(|v| v.abs() > 0.5 + 1e-9) {
            continue;
        }
        let kind = match values.iter().filter(|v| **v < 0.0).count() {
            0 => CandidateKind::Minimum,
            1 => CandidateKind::Saddle,
            _ => continue,
        };
        let model_energy = e + 0.5 * g.iter().zip(&delta).map(|(a, b)| a * b).sum::<f64>();
        found.push(Candidate {
            point: lowest_copy[point],
            kind,
            energy: e,
            offset: delta,
            model_energy,
        });
    }
    // One stationary point between grid points can be found from several of them (the local
    // models of a surface that is not quadratic differ): of candidates of one kind whose
    // model stationary points lie within one step of each other, the point nearest to its
    // own is kept.
    let reach = |c: &Candidate| c.offset.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let place = |c: &Candidate| -> Vec<f64> {
        let mut rest = c.point;
        shape
            .iter()
            .zip(&c.offset)
            .map(|(&n, o)| {
                let i = rest % n;
                rest /= n;
                i as f64 + o
            })
            .collect()
    };
    found.sort_by(|a, b| reach(a).total_cmp(&reach(b)));
    let mut kept: Vec<Candidate> = Vec::new();
    for c in found {
        let here = place(&c);
        let same = kept.iter().any(|k| {
            k.kind == c.kind
                && place(k)
                    .iter()
                    .zip(&here)
                    .zip(shape.iter().zip(periodic))
                    .all(|((a, b), (&n, &p))| {
                        let d = a - b;
                        let d = if p {
                            let period = (n - 1) as f64;
                            d - period * (d / period).round()
                        } else {
                            d
                        };
                        d.abs() < 1.0
                    })
        });
        if !same {
            kept.push(c);
        }
    }
    kept.sort_by(|a, b| {
        (a.kind == CandidateKind::Saddle)
            .cmp(&(b.kind == CandidateKind::Saddle))
            .then(a.energy.total_cmp(&b.energy))
    });
    Ok(kept)
}

/// The energies with every copy of a point of the periodic dimensions (index 0 and the last)
/// at the lowest energy of its copies, and for every point the copy that has it.
fn merge_periodic_copies(
    shape: &[usize],
    energies: &[f64],
    periodic: &[bool],
    indices: impl Fn(usize) -> Vec<usize>,
) -> (Vec<f64>, Vec<usize>) {
    let mut merged = energies.to_vec();
    let mut lowest: Vec<usize> = (0..energies.len()).collect();
    if !periodic.iter().any(|p| *p) {
        return (merged, lowest);
    }
    for (point, slot) in merged.iter_mut().enumerate() {
        let idx = indices(point);
        let mut copies = vec![idx.clone()];
        for d in 0..shape.len() {
            if periodic[d] && (idx[d] == 0 || idx[d] + 1 == shape[d]) {
                let other = if idx[d] == 0 { shape[d] - 1 } else { 0 };
                let more: Vec<Vec<usize>> = copies
                    .iter()
                    .map(|c| {
                        let mut c = c.clone();
                        c[d] = other;
                        c
                    })
                    .collect();
                copies.extend(more);
            }
        }
        let best = copies
            .iter()
            .map(|c| flat(c, shape))
            .filter(|&q| energies[q].is_finite())
            .min_by(|&a, &b| energies[a].total_cmp(&energies[b]));
        if let Some(b) = best {
            *slot = energies[b];
            lowest[point] = b;
        }
    }
    (merged, lowest)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn maxima_of_a_curve_and_saddles_of_a_map() {
        // A double well along a line: minima at -1 and 1, the maximum at 0.
        let xs: Vec<f64> = (0..21).map(|k| -1.5 + 0.15 * k as f64).collect();
        let e: Vec<f64> = xs.iter().map(|x| (x * x - 1.0).powi(2)).collect();
        let found = stationary_points(&[21], &e).unwrap();
        let saddles: Vec<&Candidate> = found
            .iter()
            .filter(|c| c.kind == CandidateKind::Saddle)
            .collect();
        assert_eq!(saddles.len(), 1);
        assert_eq!(saddles[0].point, 10);
        assert_eq!(
            found
                .iter()
                .filter(|c| c.kind == CandidateKind::Minimum)
                .count(),
            2
        );
        // A saddle of x^2 - y^2 + 0.3 x y shifted off the grid, on a 9 x 7 map.
        let (nx, ny) = (9, 7);
        let map: Vec<f64> = (0..nx * ny)
            .map(|p| {
                let (x, y) = ((p % nx) as f64 - 4.3, (p / nx) as f64 - 2.8);
                x * x - y * y + 0.3 * x * y
            })
            .collect();
        let found = stationary_points(&[nx, ny], &map).unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].kind, CandidateKind::Saddle);
        assert_eq!(found[0].point, 4 + 3 * nx);
        // The model of a quadratic is the quadratic: the offset is exact.
        let exact = [4.3 - 4.0, 2.8 - 3.0];
        for (d, v) in found[0].offset.iter().zip(exact) {
            assert!((d - v).abs() < 1e-10, "{:?}", found[0].offset);
        }
        // Missing energies are not judged.
        let mut holes = map.clone();
        holes[4 + 2 * nx] = f64::NAN;
        assert!(stationary_points(&[nx, ny], &holes).unwrap().is_empty());
        assert!(stationary_points(&[3, 3], &[0.0; 8]).is_err());
        // A threefold torsion from -180 to 180 degrees in 13 points: the minimum at the ends
        // (180 = -180) is found when the dimension is periodic, not when it is open.
        let torsion: Vec<f64> = (0..13)
            .map(|k| 1.0 + (3.0 * (-180.0 + 30.0 * k as f64).to_radians()).cos())
            .collect();
        let open = stationary_points(&[13], &torsion).unwrap();
        let wrapped = stationary_points_periodic(&[13], &torsion, &[true]).unwrap();
        let count = |c: &[Candidate], k: CandidateKind| c.iter().filter(|x| x.kind == k).count();
        assert_eq!(count(&open, CandidateKind::Minimum), 2);
        assert_eq!(count(&wrapped, CandidateKind::Minimum), 3);
        assert_eq!(count(&wrapped, CandidateKind::Saddle), 3);
        assert!(wrapped.iter().any(|c| c.point == 0));
        // The first point stuck higher than the last (the same angle): the lower copy counts.
        let mut stuck = torsion.clone();
        stuck[0] += 3.0;
        let found = stationary_points_periodic(&[13], &stuck, &[true]).unwrap();
        assert_eq!(count(&found, CandidateKind::Minimum), 3);
        assert!(found
            .iter()
            .any(|c| c.point == 12 && c.kind == CandidateKind::Minimum));
    }
}
