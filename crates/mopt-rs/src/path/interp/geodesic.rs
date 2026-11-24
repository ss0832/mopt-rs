//! Geodesic interpolation: the images of the shortest path in a space of scaled
//! interatomic distances.
//!
//! Zhu, Thompson, and Martinez (J. Chem. Phys. 2019, 150, 164103) measure a path with the
//! coordinates `w = exp(alpha (1 - r / r_e)) + beta r_e / r` of the pairs of atoms that
//! come near each other (`r_e`: the sum of the covalent radii), in which a bond that
//! stretches counts for much and two distant atoms for little, and take the shortest path
//! between two structures. This follows their program (geodesic-interpolate, MIT licence,
//! `third_party/GEODESIC_INTERPOLATE.md`), which MultiOptPy has a copy of:
//!
//! 1. images are added one at a time in the largest gap, each the structure whose
//!    coordinates `w` are nearest the mean of those of its two neighbours;
//! 2. all images together are moved so that the sum of the squared lengths of the half
//!    segments (image to the midpoint of two images, in `w`) is smallest, with a weak
//!    restraint to where they were.
//!
//! Both are least-squares problems. The program hands their Jacobian to SciPy as a dense
//! matrix of (images x pairs) rows and (images x coordinates) columns, which is what makes
//! it slow and large for more than some tens of atoms. Here the Jacobian is never formed:
//! a row has the six derivatives of one distance in at most three structures, so the
//! products with it and its transpose are sums over the pairs, and
//! [`super::least_squares`] needs nothing else. Memory and the work of a product grow
//! with images times pairs, the pairs with the number of atoms.
//!
//! Further differences: the two starts of the search for a new image are shifted by a
//! fixed pattern instead of random numbers (the same input gives the same path); the
//! better of the two is chosen by the length of the two segments in `w` plus its distance
//! from the neighbours, without the sub-division of the program; the structures are not
//! turned onto each other; distances in a periodic cell are those of the nearest images;
//! fixed coordinates stay.
//!
//! MultiOptPy: `Interpolation/geodesic_interpolation.py` (`-adg`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeSet;

use faer::Col;

use super::least_squares::{levenberg_marquardt, LeastSquaresSettings, Linearization};
use super::{Interpolated, PathFrame, PathSettings, NUDGE};
use crate::domain::{PeriodicCell, Units};
use crate::error::MoptResult;
use crate::parameters::covalent::covalent_radius_bohr;

/// `alpha` of the coordinates in which the path is measured.
pub const ALPHA: f64 = 1.7;
/// `alpha` in the search for a new image (wider: distant pairs count for more).
pub const ALPHA_NEW_IMAGE: f64 = 0.7;
/// `beta` of the coordinates.
pub const BETA: f64 = 0.01;
/// Pairs nearer than this in any image are coordinates (Angstrom).
pub const CUTOFF_ANGSTROM: f64 = 3.0;
/// The same in the search for a new image (Angstrom; one more while the list is made).
pub const CUTOFF_NEW_IMAGE_ANGSTROM: f64 = 4.0;
/// Atoms nearer than this are bonded; atoms two bonds apart are coordinates (Angstrom).
pub const BONDED_ANGSTROM: f64 = 1.8;
/// Every atom is in at least this many coordinates.
pub const NEIGHBOURS: usize = 4;
/// `r_e` of a pair without elements (Angstrom).
pub const PLAIN_PAIR_ANGSTROM: f64 = 2.0;
/// Weight of the distance from where the images were, per Angstrom, when the whole path
/// is moved.
pub const FRICTION: f64 = 1e-3;
/// The same in the search for a new image, times `1 / sqrt(atoms)`.
pub const FRICTION_NEW_IMAGE: f64 = 0.1;
/// Tolerance of the least-squares problems (gradient and relative decrease).
pub const TOLERANCE: f64 = 1e-4;
/// Most rounds in which new contacts are added to the coordinates of a new image.
const MOST_ROUNDS: usize = 6;

/// The pairs of atoms that are coordinates, and their `r_e` (Bohr).
#[derive(Debug, Clone)]
struct Pairs {
    list: Vec<(usize, usize)>,
    re: Vec<f64>,
}

/// The vector from atom `j` to atom `i` (Bohr).
fn between(x: &[f64], i: usize, j: usize, cell: Option<&PeriodicCell>) -> [f64; 3] {
    let (p, q) = (
        [x[3 * i], x[3 * i + 1], x[3 * i + 2]],
        [x[3 * j], x[3 * j + 1], x[3 * j + 2]],
    );
    match cell {
        Some(cell) => cell.mic_delta(p, q),
        None => [p[0] - q[0], p[1] - q[1], p[2] - q[2]],
    }
}

fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The coordinates of the structures `snapshots`: the pairs of `enforce`, every pair that
/// is nearer than `cutoff` (Bohr) or within two bonds in one of them, and for an atom
/// with fewer than `neighbours` pairs its nearest atoms in the last structure.
fn pair_list(
    snapshots: &[&[f64]],
    frame: &PathFrame,
    cutoff: f64,
    neighbours: usize,
    enforce: &BTreeSet<(usize, usize)>,
) -> Pairs {
    let cell = frame.cell.as_ref();
    let atoms = snapshots.first().map_or(0, |x| x.len() / 3);
    let bonded = BONDED_ANGSTROM * Units::ANGSTROM_TO_BOHR;
    let mut set = enforce.clone();
    for x in snapshots {
        let mut bonds: Vec<Vec<usize>> = vec![Vec::new(); atoms];
        for i in 0..atoms {
            for j in i + 1..atoms {
                let r = norm3(&between(x, i, j, cell));
                if r < cutoff {
                    set.insert((i, j));
                }
                if r < bonded {
                    bonds[i].push(j);
                    bonds[j].push(i);
                }
            }
        }
        for (middle, around) in bonds.iter().enumerate() {
            for &a in around {
                set.insert((a.min(middle), a.max(middle)));
                for &b in around {
                    if a < b {
                        set.insert((a, b));
                    }
                }
            }
        }
    }
    let neighbours = neighbours.min(atoms.saturating_sub(1));
    if let (true, Some(last)) = (neighbours > 0, snapshots.last()) {
        let mut count = vec![0usize; atoms];
        for (i, j) in &set {
            count[*i] += 1;
            count[*j] += 1;
        }
        for atom in 0..atoms {
            if count[atom] >= neighbours {
                continue;
            }
            let mut others: Vec<(f64, usize)> = (0..atoms)
                .filter(|other| *other != atom)
                .map(|other| (norm3(&between(last, atom, other, cell)), other))
                .collect();
            others.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (_, other) in others.into_iter().take(neighbours) {
                if set.insert((atom.min(other), atom.max(other))) {
                    count[atom] += 1;
                    count[other] += 1;
                }
            }
        }
    }
    let list: Vec<(usize, usize)> = set.into_iter().collect();
    let radius = |atom: usize| frame.elements.get(atom).map(|e| covalent_radius_bohr(*e));
    let re = list
        .iter()
        .map(|(i, j)| match (radius(*i), radius(*j)) {
            (Some(a), Some(b)) => a + b,
            _ => PLAIN_PAIR_ANGSTROM * Units::ANGSTROM_TO_BOHR,
        })
        .collect();
    Pairs { list, re }
}

/// The coordinates `w` of a structure and what their derivatives need: `dw/dr` and the
/// unit vector of every pair.
struct Scaled {
    w: Vec<f64>,
    slope: Vec<f64>,
    unit: Vec<[f64; 3]>,
}

fn scaled(x: &[f64], pairs: &Pairs, alpha: f64, cell: Option<&PeriodicCell>) -> Scaled {
    let n = pairs.list.len();
    let mut out = Scaled {
        w: Vec::with_capacity(n),
        slope: Vec::with_capacity(n),
        unit: Vec::with_capacity(n),
    };
    for (&(i, j), &re) in pairs.list.iter().zip(&pairs.re) {
        let v = between(x, i, j, cell);
        let r = norm3(&v).max(1e-12);
        let decay = (alpha * (1.0 - r / re)).exp();
        let tail = BETA * re / r;
        out.w.push(decay + tail);
        out.slope.push(-alpha / re * decay - tail / r);
        out.unit.push([v[0] / r, v[1] / r, v[2] / r]);
    }
    out
}

impl Scaled {
    /// The change of `w` with the change `v` of the coordinates of the structure.
    fn forward(&self, pairs: &Pairs, v: &[f64]) -> Vec<f64> {
        pairs
            .list
            .iter()
            .enumerate()
            .map(|(p, &(i, j))| {
                let u = &self.unit[p];
                self.slope[p]
                    * (0..3)
                        .map(|k| u[k] * (v[3 * i + k] - v[3 * j + k]))
                        .sum::<f64>()
            })
            .collect()
    }

    /// `factor` times the transpose of [`Self::forward`] applied to `y`, added to `out`.
    fn backward(&self, pairs: &Pairs, y: &[f64], factor: f64, out: &mut [f64]) {
        for (p, &(i, j)) in pairs.list.iter().enumerate() {
            let size = factor * y[p] * self.slope[p];
            let u = &self.unit[p];
            for k in 0..3 {
                out[3 * i + k] += size * u[k];
                out[3 * j + k] -= size * u[k];
            }
        }
    }
}

/// One structure with given coordinates `w` to come near, and a restraint to `start`.
struct NewImage<'a> {
    pairs: &'a Pairs,
    fixed: &'a [usize],
    here: Scaled,
    friction: f64,
    r: Vec<f64>,
}

impl Linearization for NewImage<'_> {
    fn residuals(&self) -> &[f64] {
        &self.r
    }

    fn forward(&self, v: &[f64]) -> Vec<f64> {
        let mut v = v.to_vec();
        for &k in self.fixed {
            v[k] = 0.0;
        }
        let mut out = self.here.forward(self.pairs, &v);
        out.extend(v.iter().map(|value| self.friction * value));
        out
    }

    fn backward(&self, u: &[f64]) -> Vec<f64> {
        let pairs = self.pairs.list.len();
        let mut out: Vec<f64> = u[pairs..].iter().map(|v| self.friction * v).collect();
        self.here.backward(self.pairs, &u[..pairs], 1.0, &mut out);
        for &k in self.fixed {
            out[k] = 0.0;
        }
        out
    }
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// The structure between `a` and `b` whose coordinates `w` are nearest the mean of
/// theirs. Returns it with the least-squares iterations and whether they converged.
fn new_image(a: &[f64], b: &[f64], frame: &PathFrame, most: usize) -> (Vec<f64>, usize, bool) {
    let cell = frame.cell.as_ref();
    let atoms = a.len() / 3;
    let to_bohr = Units::ANGSTROM_TO_BOHR;
    // The restraint is per Angstrom in the publication; the coordinates here are in Bohr.
    let friction = FRICTION_NEW_IMAGE / (atoms.max(1) as f64).sqrt() * Units::BOHR_TO_ANGSTROM;
    let settings = LeastSquaresSettings {
        max_iterations: most.max(1),
        gradient_tolerance: TOLERANCE,
        function_tolerance: TOLERANCE,
        ..LeastSquaresSettings::default()
    };
    let mut enforce = BTreeSet::new();
    let mut seen: Vec<Vec<f64>> = vec![a.to_vec(), b.to_vec()];
    let (mut iterations, mut converged) = (0, true);
    let mut best: Option<(f64, Vec<f64>)> = None;
    for _ in 0..MOST_ROUNDS {
        let snapshots: Vec<&[f64]> = seen.iter().map(Vec::as_slice).collect();
        let pairs = pair_list(
            &snapshots,
            frame,
            (CUTOFF_NEW_IMAGE_ANGSTROM + 1.0) * to_bohr,
            NEIGHBOURS,
            &enforce,
        );
        let (wa, wb) = (
            scaled(a, &pairs, ALPHA_NEW_IMAGE, cell),
            scaled(b, &pairs, ALPHA_NEW_IMAGE, cell),
        );
        let target: Vec<f64> = wa.w.iter().zip(&wb.w).map(|(p, q)| 0.5 * (p + q)).collect();
        let (la, lb) = (
            scaled(a, &pairs, ALPHA, cell),
            scaled(b, &pairs, ALPHA, cell),
        );
        best = None;
        let mut new_contacts = false;
        for share in [0.02, 0.98] {
            // Near one of the two structures, where no atoms lie on each other, shifted a
            // little by a fixed pattern so that no symmetry holds the search.
            let start: Vec<f64> = (0..a.len())
                .map(|k| {
                    let wave = (0.618_033_988_75 * (k + 1) as f64).fract();
                    let shift = if frame.fixed_components.contains(&k) {
                        0.0
                    } else {
                        NUDGE * wave
                    };
                    share * a[k] + (1.0 - share) * b[k] + shift
                })
                .collect();
            let found = levenberg_marquardt(
                |x: &[f64]| {
                    let here = scaled(x, &pairs, ALPHA_NEW_IMAGE, cell);
                    let mut r: Vec<f64> = here.w.iter().zip(&target).map(|(w, t)| w - t).collect();
                    r.extend(x.iter().zip(&start).map(|(p, q)| friction * (p - q)));
                    NewImage {
                        pairs: &pairs,
                        fixed: &frame.fixed_components,
                        here,
                        friction,
                        r,
                    }
                },
                start.clone(),
                &settings,
            );
            iterations += found.iterations;
            converged &= found.converged;
            let x = found.x;
            // Atoms that came near each other on the way are coordinates too.
            let mut with_new: Vec<&[f64]> = seen.iter().map(Vec::as_slice).collect();
            with_new.push(&x);
            let now = pair_list(
                &with_new,
                frame,
                CUTOFF_NEW_IMAGE_ANGSTROM * to_bohr,
                0,
                &BTreeSet::new(),
            );
            let known: BTreeSet<(usize, usize)> = pairs.list.iter().copied().collect();
            let extra: Vec<(usize, usize)> = now
                .list
                .into_iter()
                .filter(|pair| !known.contains(pair))
                .collect();
            if !extra.is_empty() {
                enforce.extend(extra);
                seen.push(x);
                new_contacts = true;
                break;
            }
            let here = scaled(&x, &pairs, ALPHA, cell);
            let leg = |end: &Scaled| distance(&here.w, &end.w);
            let rms = |end: &[f64]| {
                distance(&x, end) / (atoms.max(1) as f64).sqrt() * Units::BOHR_TO_ANGSTROM
            };
            let length = leg(&la) + leg(&lb) + rms(a).max(rms(b));
            if best.as_ref().is_none_or(|(shortest, _)| length < *shortest) {
                best = Some((length, x));
            }
        }
        if !new_contacts {
            break;
        }
    }
    let image = match best {
        Some((_, x)) => x,
        // New contacts in every round: the last structure found is used.
        None => seen.pop().unwrap_or_else(|| a.to_vec()),
    };
    (image, iterations, converged)
}

/// All images of a path at once: the half segments in `w` and the restraint.
struct WholePath<'a> {
    pairs: &'a Pairs,
    fixed: &'a [usize],
    coordinates: usize,
    images: Vec<Scaled>,
    middles: Vec<Scaled>,
    friction: f64,
    r: Vec<f64>,
}

impl WholePath<'_> {
    fn inner(&self) -> usize {
        self.images.len() - 2
    }
}

impl Linearization for WholePath<'_> {
    fn residuals(&self) -> &[f64] {
        &self.r
    }

    fn forward(&self, v: &[f64]) -> Vec<f64> {
        let (n, pairs, dof) = (self.images.len(), self.pairs.list.len(), self.coordinates);
        let mut v = v.to_vec();
        for image in 0..self.inner() {
            for &k in self.fixed {
                v[image * dof + k] = 0.0;
            }
        }
        // The change of every structure (none of the two ends).
        let zero = vec![0.0; dof];
        let moved = |image: usize| -> &[f64] {
            if image == 0 || image + 1 == n {
                &zero
            } else {
                &v[(image - 1) * dof..image * dof]
            }
        };
        let of_image: Vec<Vec<f64>> = (0..n)
            .map(|k| self.images[k].forward(self.pairs, moved(k)))
            .collect();
        let mut out = vec![0.0; 2 * (n - 1) * pairs];
        for s in 0..n - 1 {
            let half: Vec<f64> = moved(s)
                .iter()
                .zip(moved(s + 1))
                .map(|(p, q)| 0.5 * (p + q))
                .collect();
            let middle = self.middles[s].forward(self.pairs, &half);
            for p in 0..pairs {
                out[s * pairs + p] = middle[p] - of_image[s][p];
                out[(n - 1 + s) * pairs + p] = of_image[s + 1][p] - middle[p];
            }
        }
        out.extend(v.iter().map(|value| self.friction * value));
        out
    }

    fn backward(&self, u: &[f64]) -> Vec<f64> {
        let (n, pairs, dof) = (self.images.len(), self.pairs.list.len(), self.coordinates);
        let left = |s: usize| &u[s * pairs..(s + 1) * pairs];
        let right = |s: usize| &u[(n - 1 + s) * pairs..(n + s) * pairs];
        let mut out: Vec<f64> = u[2 * (n - 1) * pairs..]
            .iter()
            .map(|v| self.friction * v)
            .collect();
        let mut of_structure = vec![0.0; dof];
        for s in 0..n - 1 {
            // Through the midpoint to the two structures of the segment.
            let through: Vec<f64> = left(s).iter().zip(right(s)).map(|(l, r)| l - r).collect();
            of_structure.iter_mut().for_each(|v| *v = 0.0);
            self.middles[s].backward(self.pairs, &through, 0.5, &mut of_structure);
            for image in [s, s + 1] {
                if image > 0 && image + 1 < n {
                    for k in 0..dof {
                        out[(image - 1) * dof + k] += of_structure[k];
                    }
                }
            }
        }
        for image in 1..n - 1 {
            let direct: Vec<f64> = right(image - 1)
                .iter()
                .zip(left(image))
                .map(|(r, l)| r - l)
                .collect();
            let target = &mut out[(image - 1) * dof..image * dof];
            self.images[image].backward(self.pairs, &direct, 1.0, target);
        }
        for image in 0..self.inner() {
            for &k in self.fixed {
                out[image * dof + k] = 0.0;
            }
        }
        out
    }
}

/// `count` images between `first` and `last` on the geodesic.
pub fn geodesic_images(
    first: &Col<f64>,
    last: &Col<f64>,
    count: usize,
    frame: &PathFrame,
    settings: &PathSettings,
) -> MoptResult<Interpolated> {
    let cell = frame.cell.as_ref();
    let dof = first.nrows();
    let flat = |x: &Col<f64>| -> Vec<f64> { x.iter().copied().collect() };
    let mut path = vec![flat(first), flat(last)];
    let (mut iterations, mut converged) = (0, true);
    // Images are added in the largest gap until there are enough.
    while path.len() < count + 2 {
        let gap = (0..path.len() - 1)
            .max_by(|a, b| {
                distance(&path[*a], &path[*a + 1]).total_cmp(&distance(&path[*b], &path[*b + 1]))
            })
            .unwrap_or(0);
        let (image, taken, found) =
            new_image(&path[gap], &path[gap + 1], frame, settings.max_iterations);
        iterations += taken;
        converged &= found;
        path.insert(gap + 1, image);
    }
    if count == 0 {
        return Ok(Interpolated {
            images: Vec::new(),
            iterations,
            converged,
        });
    }
    // The whole path is moved to the shortest one.
    let snapshots: Vec<&[f64]> = path.iter().map(Vec::as_slice).collect();
    let pairs = pair_list(
        &snapshots,
        frame,
        CUTOFF_ANGSTROM * Units::ANGSTROM_TO_BOHR,
        NEIGHBOURS,
        &BTreeSet::new(),
    );
    let friction = FRICTION * Units::BOHR_TO_ANGSTROM;
    let n = path.len();
    let start: Vec<f64> = path[1..n - 1].concat();
    let found = levenberg_marquardt(
        |x: &[f64]| {
            let structure = |k: usize| -> &[f64] {
                if k == 0 {
                    &path[0]
                } else if k + 1 == n {
                    &path[n - 1]
                } else {
                    &x[(k - 1) * dof..k * dof]
                }
            };
            let images: Vec<Scaled> = (0..n)
                .map(|k| scaled(structure(k), &pairs, ALPHA, cell))
                .collect();
            let middles: Vec<Scaled> = (0..n - 1)
                .map(|s| {
                    let middle: Vec<f64> = structure(s)
                        .iter()
                        .zip(structure(s + 1))
                        .map(|(p, q)| 0.5 * (p + q))
                        .collect();
                    scaled(&middle, &pairs, ALPHA, cell)
                })
                .collect();
            let count = pairs.list.len();
            let mut r = vec![0.0; 2 * (n - 1) * count];
            for s in 0..n - 1 {
                for p in 0..count {
                    r[s * count + p] = middles[s].w[p] - images[s].w[p];
                    r[(n - 1 + s) * count + p] = images[s + 1].w[p] - middles[s].w[p];
                }
            }
            r.extend(x.iter().zip(&start).map(|(p, q)| friction * (p - q)));
            WholePath {
                pairs: &pairs,
                fixed: &frame.fixed_components,
                coordinates: dof,
                images,
                middles,
                friction,
                r,
            }
        },
        start.clone(),
        &LeastSquaresSettings {
            max_iterations: settings.max_iterations.clamp(1, 50),
            gradient_tolerance: TOLERANCE,
            function_tolerance: TOLERANCE,
            soft_l1: true,
            ..LeastSquaresSettings::default()
        },
    );
    iterations += found.iterations;
    converged &= found.converged;
    let images = (0..count)
        .map(|k| Col::from_fn(dof, |c| found.x[k * dof + c]))
        .collect();
    Ok(Interpolated {
        images,
        iterations,
        converged,
    })
}
