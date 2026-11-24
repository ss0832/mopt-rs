//! Putting the images of a chain back along the chain.
//!
//! The chain has no springs, so its images are kept apart by placing them anew on a
//! curve through the current images ([`Curve`]):
//!
//! - [`equal_spacing`] (`-ad`): the same number of images at equal distances;
//! - [`by_spacing`] (`-nd`): images a given distance apart (their number follows from the
//!   length of the chain), used on the chain that is read in;
//! - [`towards_predicted_maxima`] (`-adpred`): where the energy profile has a maximum at an
//!   image, the position of the maximum along the chain is predicted from a polynomial
//!   through the energies, slopes, and curvatures of the three images around it, that
//!   image is put there, and the others are spread evenly between the ends and these
//!   images, or, with a [`Gathering`], more densely around these images: the images are
//!   placed at equal amounts of the density `1 + w exp(-(s - p)^2 / (2 sigma^2))` along
//!   the chain, `p` being the nearest maximum. Maxima that lie close together share one
//!   dense region (the density is that of the nearest maximum, not the sum over all), so
//!   that a path with several barriers in a row does not take all images from the rest.
//!
//! MultiOptPy places the images on the straight lines from image to image. Where the path
//! bends, a point on such a line is off the path by about `k h^2 / 8` (`k`: curvature of
//! the path, `h`: distance of the images), so that every redistribution pushes the images
//! off the path that the relaxation has just found, and a chain that is redistributed
//! every few iterations does not come to rest (Muller-Brown surface, 17 images: the
//! forces stay at a third of their first values). The default therefore places the
//! images on a cubic curve through the images ([`Curve::Smooth`]; ase_qsm has the same
//! choice), which leaves images that are where they should be exactly where they are.
//!
//! MultiOptPy: `Interpolation/linear_interpolation.py:distribute_geometry`,
//! `distribute_geometry_by_length`, `distribute_geometry_by_predicted_energy`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::peaks::PeakMoves;
use super::{path_lengths, strict_maxima, ChainKind, Compatibility};
use crate::domain::Units;
use crate::math::linalg::{col_dot, col_norm};
use crate::math::roots::polynomial_real_roots;

/// Length below which a chain counts as a point (MultiOptPy's value).
const NO_LENGTH: f64 = 1e-8;
/// Distance (Angstrom) of the last new image to the end of the chain below which the end
/// is not added once more.
const END_TOLERANCE_ANGSTROM: f64 = 1e-4;
/// What an anchor is moved beyond the previous one when their order is wrong.
const ANCHOR_NUDGE: f64 = 1e-6;
/// Coefficients of the fitted polynomial below this share of the largest are dropped
/// from its top before its roots are searched.
const NEGLIGIBLE_COEFFICIENT: f64 = 1e-12;
/// Second derivative (in the scaled variable of the fit) below which a root is a maximum.
const MAXIMUM_CURVATURE: f64 = -1e-5;

/// How the points between the images of a chain are found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Curve {
    /// Straight lines from image to image (MultiOptPy).
    Polyline,
    /// A cubic curve through the images: in every interval the cubic with the positions
    /// of its two images and the slopes there, a slope being that of the parabola through
    /// an image and its two neighbours (parameter: the length along the straight lines).
    Smooth,
}

impl Curve {
    /// The curve of a form of the chain relaxation.
    pub fn of(compatibility: Compatibility) -> Self {
        match compatibility {
            Compatibility::Consistent => Self::Smooth,
            Compatibility::MultiOptPy => Self::Polyline,
        }
    }
}

/// Parameter interval below which two images count as one point.
const NO_SPAN: f64 = 1e-12;

/// The cubic curve of [`Curve::Smooth`] through `chain` at the increasing parameters `at`.
struct CubicThrough<'a> {
    chain: &'a [Col<f64>],
    at: &'a [f64],
    slopes: Vec<Col<f64>>,
}

impl<'a> CubicThrough<'a> {
    fn new(chain: &'a [Col<f64>], at: &'a [f64]) -> Self {
        let last = chain.len() - 1;
        let n = chain[0].nrows();
        let span = |j: usize| at[j + 1] - at[j];
        let chord = |j: usize| {
            if span(j) > NO_SPAN {
                (1.0 / span(j)) * (&chain[j + 1] - &chain[j])
            } else {
                Col::<f64>::zeros(n)
            }
        };
        // Slope of the parabola through three images at its first, middle, or last one.
        let slopes = (0..=last)
            .map(|i| {
                if last == 0 {
                    Col::zeros(n)
                } else if last == 1 {
                    chord(0)
                } else if i == 0 {
                    let (h0, h1) = (span(0), span(1));
                    if h0 + h1 > NO_SPAN {
                        ((2.0 * h0 + h1) / (h0 + h1)) * chord(0) - (h0 / (h0 + h1)) * chord(1)
                    } else {
                        chord(0)
                    }
                } else if i == last {
                    let (h0, h1) = (span(last - 2), span(last - 1));
                    if h0 + h1 > NO_SPAN {
                        ((2.0 * h1 + h0) / (h0 + h1)) * chord(last - 1)
                            - (h1 / (h0 + h1)) * chord(last - 2)
                    } else {
                        chord(last - 1)
                    }
                } else {
                    let (h0, h1) = (span(i - 1), span(i));
                    if h0 + h1 > NO_SPAN {
                        (h0 / (h0 + h1)) * chord(i) + (h1 / (h0 + h1)) * chord(i - 1)
                    } else {
                        Col::zeros(n)
                    }
                }
            })
            .collect();
        Self { chain, at, slopes }
    }

    /// The point at the parameter `target` (the ends beyond the ends).
    fn point(&self, target: f64) -> Col<f64> {
        let last = self.chain.len() - 1;
        if target <= self.at[0] {
            return self.chain[0].clone();
        }
        if target >= self.at[last] {
            return self.chain[last].clone();
        }
        // The last interval that starts at or before the target.
        let j = (0..last).rev().find(|&j| self.at[j] <= target).unwrap_or(0);
        let h = self.at[j + 1] - self.at[j];
        if h <= NO_SPAN {
            return self.chain[j].clone();
        }
        let u = (target - self.at[j]) / h;
        let (u2, u3) = (u * u, u * u * u);
        (2.0 * u3 - 3.0 * u2 + 1.0) * &self.chain[j]
            + (h * (u3 - 2.0 * u2 + u)) * &self.slopes[j]
            + (3.0 * u2 - 2.0 * u3) * &self.chain[j + 1]
            + (h * (u3 - u2)) * &self.slopes[j + 1]
    }
}

/// The point at `distance` along the polyline `chain` with the cumulative `lengths`.
fn point_at(chain: &[Col<f64>], lengths: &[f64], distance: f64) -> Option<Col<f64>> {
    (0..chain.len() - 1).find_map(|j| {
        (lengths[j] <= distance && distance <= lengths[j + 1]).then(|| {
            let span = lengths[j + 1] - lengths[j];
            let t = if span > 0.0 {
                (distance - lengths[j]) / span
            } else {
                0.0
            };
            &chain[j] + t * (&chain[j + 1] - &chain[j])
        })
    })
}

/// The same number of images at equal distances along the chain (measured along the
/// straight lines from image to image); the ends stay.
pub fn equal_spacing(chain: &[Col<f64>], kind: ChainKind, curve: Curve) -> Vec<Col<f64>> {
    equal_spacing_between(chain, kind, &[], curve)
}

/// [`equal_spacing`] between the ends and the images `anchors`, which stay where they
/// are: a climbing image is not taken off its saddle by spreading the chain evenly.
pub fn equal_spacing_between(
    chain: &[Col<f64>],
    kind: ChainKind,
    anchors: &[usize],
    curve: Curve,
) -> Vec<Col<f64>> {
    let last = chain.len() - 1;
    let lengths = path_lengths(chain, kind);
    let smooth = (curve == Curve::Smooth).then(|| CubicThrough::new(chain, &lengths));
    let mut marks: Vec<usize> = anchors
        .iter()
        .copied()
        .filter(|&i| i > 0 && i < last)
        .collect();
    marks.sort_unstable();
    marks.dedup();
    marks.insert(0, 0);
    marks.push(last);
    let mut out = chain.to_vec();
    for pair in marks.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        let total = lengths[to] - lengths[from];
        if total < NO_LENGTH || to - from < 2 {
            continue;
        }
        let spacing = total / (to - from) as f64;
        for (offset, image) in out[from + 1..to].iter_mut().enumerate() {
            let distance = lengths[from] + (offset + 1) as f64 * spacing;
            *image = match &smooth {
                Some(cubic) => cubic.point(distance),
                None => point_at(chain, &lengths, distance).unwrap_or_else(|| chain[to].clone()),
            };
        }
    }
    out
}

/// Images `spacing` apart along the chain, from the first image on, and the last image
/// (unless the last new image lies on it).
pub fn by_spacing(chain: &[Col<f64>], spacing: f64, kind: ChainKind) -> Vec<Col<f64>> {
    let lengths = path_lengths(chain, kind);
    let total = lengths[chain.len() - 1];
    if total < NO_LENGTH || spacing.is_nan() || spacing <= 0.0 {
        return vec![chain[0].clone()];
    }
    let last = &chain[chain.len() - 1];
    let mut out = vec![chain[0].clone()];
    for i in 1..=(total / spacing).floor() as usize {
        let distance = i as f64 * spacing;
        if distance >= total {
            break;
        }
        if let Some(point) = point_at(chain, &lengths, distance) {
            out.push(point);
        }
    }
    let tolerance = END_TOLERANCE_ANGSTROM * Units::ANGSTROM_TO_BOHR;
    if out.len() > 1
        && out
            .last()
            .is_some_and(|point| col_norm(&(point - last)) <= tolerance)
    {
        out.pop();
    }
    out.push(last.clone());
    out
}

/// Images on the Bezier curve whose control points are the images of `chain` (Bernstein
/// polynomials of the degree of the number of images less one), at the parameters `s / L`
/// for the arc lengths `s = 0, spacing, 2 spacing, ...` of the chain of length `L` and at
/// its end (MultiOptPy's `-ndb`). The curve starts and ends at the first and the last
/// image and passes near the others, not through them, and the new images are not
/// `spacing` apart on it; the distance only sets how many there are. The curve is evaluated
/// by the steps of de Casteljau, weighted means of neighbouring points (the binomial
/// coefficients of a long chain times the powers of the parameter lose every digit).
///
/// MultiOptPy: `Interpolation/binomial_interpolation.py:distribute_geometry_by_length_bernstein`.
pub fn bezier_by_spacing(chain: &[Col<f64>], spacing: f64, kind: ChainKind) -> Vec<Col<f64>> {
    let lengths = path_lengths(chain, kind);
    let total = lengths[chain.len() - 1];
    if total < NO_LENGTH || spacing.is_nan() || spacing <= 0.0 {
        return chain.to_vec();
    }
    let mut parameters: Vec<f64> = (0..)
        .map(|i| f64::from(i) * spacing)
        .take_while(|s| *s < total)
        .map(|s| s / total)
        .collect();
    parameters.push(1.0);
    parameters
        .into_iter()
        .map(|t| {
            let mut points = chain.to_vec();
            for level in 1..points.len() {
                for i in 0..points.len() - level {
                    points[i] = (1.0 - t) * &points[i] + t * &points[i + 1];
                }
            }
            points.swap_remove(0)
        })
        .collect()
}

/// Position of the maximum of the polynomial through three points, relative to the middle
/// one: through the values `e` and the slopes `g` at the arc lengths `s` (fifth degree), or
/// also through the second derivatives `c` (eighth degree). `None` without a maximum
/// between 1.1 times the outer points.
fn fitted_maximum(s: [f64; 3], e: [f64; 3], g: [f64; 3], c: Option<[f64; 3]>) -> Option<f64> {
    let scale = s[0].abs().max(s[2].abs());
    if scale < 1e-12 {
        return None;
    }
    let x = s.map(|v| v / scale);
    let degree = if c.is_some() { 8 } else { 5 };
    let size = degree + 1;
    let mut a = Mat::<f64>::zeros(size, size);
    let mut b = Mat::<f64>::zeros(size, 1);
    let mut row = 0;
    for i in 0..3 {
        for k in 0..size {
            a[(row, k)] = x[i].powi(k as i32);
            if k >= 1 {
                a[(row + 1, k)] = k as f64 * x[i].powi(k as i32 - 1);
            }
        }
        b[(row, 0)] = e[i];
        b[(row + 1, 0)] = g[i] * scale;
        row += 2;
        if let Some(curvatures) = c {
            for k in 2..size {
                a[(row, k)] = (k * (k - 1)) as f64 * x[i].powi(k as i32 - 2);
            }
            b[(row, 0)] = curvatures[i] * scale * scale;
            row += 1;
        }
    }
    let lu = a.partial_piv_lu();
    let solved = faer::linalg::solvers::Solve::solve(&lu, b.as_ref());
    let coefficients: Vec<f64> = (0..size).map(|k| solved[(k, 0)]).collect();
    if coefficients.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut derivative: Vec<f64> = (1..size).map(|k| k as f64 * coefficients[k]).collect();
    // Highest coefficients that are rounding noise would put huge numbers into the
    // companion matrix of the root search and spoil the roots that matter.
    let largest = derivative.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    while derivative
        .last()
        .is_some_and(|v| v.abs() <= NEGLIGIBLE_COEFFICIENT * largest)
    {
        derivative.pop();
    }
    let value = |r: f64| coefficients.iter().rev().fold(0.0, |acc, c| acc * r + c);
    let second = |r: f64| {
        (2..size)
            .map(|k| (k * (k - 1)) as f64 * coefficients[k] * r.powi(k as i32 - 2))
            .sum::<f64>()
    };
    polynomial_real_roots(&derivative)
        .ok()?
        .into_iter()
        .filter(|&r| 1.1 * x[0] <= r && r <= 1.1 * x[2] && second(r) < MAXIMUM_CURVATURE)
        .max_by(|p, q| value(*p).total_cmp(&value(*q)))
        .map(|r| r * scale)
}

/// How the images are gathered around the predicted maxima of the energy profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gathering {
    /// How much denser the images are at a maximum than far from it, minus one (1: twice
    /// as dense).
    pub weight: f64,
    /// Width of the dense region (the standard deviation of its bell), in units of the
    /// average distance of the images.
    pub width: f64,
}

/// Intervals of the numerical integration of the density between two anchors.
const DENSITY_INTERVALS: usize = 256;

/// The positions of `count - 1` images between the anchors at `from` and `to`: evenly
/// spread, or at equal amounts of the density around `peaks` (with the width `sigma`).
fn positions_between(
    (from, to): (f64, f64),
    count: usize,
    density: Option<(&[f64], f64, f64)>,
) -> Vec<f64> {
    let even = |offset: usize| from + (to - from) * offset as f64 / count.max(1) as f64;
    let Some((peaks, sigma, weight)) = density.filter(|(peaks, sigma, weight)| {
        !peaks.is_empty() && *sigma > 0.0 && *weight > 0.0 && to > from
    }) else {
        return (1..count).map(even).collect();
    };
    let rho = |s: f64| {
        let nearest = peaks
            .iter()
            .map(|p| (s - p).abs())
            .fold(f64::INFINITY, f64::min);
        1.0 + weight * (-0.5 * (nearest / sigma).powi(2)).exp()
    };
    // The amount of density up to every point of a grid, then its inverse.
    let step = (to - from) / DENSITY_INTERVALS as f64;
    let mut amount = vec![0.0; DENSITY_INTERVALS + 1];
    for k in 1..=DENSITY_INTERVALS {
        let (a, b) = (from + (k - 1) as f64 * step, from + k as f64 * step);
        amount[k] = amount[k - 1] + 0.5 * (rho(a) + rho(b)) * step;
    }
    let total = amount[DENSITY_INTERVALS];
    (1..count)
        .map(|offset| {
            let wanted = total * offset as f64 / count as f64;
            let k = amount
                .partition_point(|a| *a < wanted)
                .clamp(1, DENSITY_INTERVALS);
            let span = amount[k] - amount[k - 1];
            let t = if span > 0.0 {
                (wanted - amount[k - 1]) / span
            } else {
                0.0
            };
            from + ((k - 1) as f64 + t) * step
        })
        .collect()
}

/// Linear interpolation of the points `chain` given at the increasing positions `at`.
fn interpolated(chain: &[Col<f64>], at: &[f64], target: f64) -> Col<f64> {
    let last = chain.len() - 1;
    if target <= at[0] {
        return chain[0].clone();
    }
    if target >= at[last] {
        return chain[last].clone();
    }
    // The last interval that starts at or before the target.
    let j = (0..last).rev().find(|&j| at[j] <= target).unwrap_or(0);
    let span = at[j + 1] - at[j];
    let t = if span > 0.0 {
        (target - at[j]) / span
    } else {
        0.0
    };
    &chain[j] + t * (&chain[j + 1] - &chain[j])
}

/// The images of `chain` placed anew so that an image at a maximum of the profile sits
/// where the maximum is predicted (see the module documentation). `energies` and
/// `gradients` are those of the images; MultiOptPy passes the chain after its step with
/// the values computed before it, image by image.
///
/// Under [`Compatibility::MultiOptPy`] the arc length is in Angstrom while the gradients
/// are per Bohr, as in MultiOptPy, which scales the slopes and curvatures of the fit by
/// the Bohr radius in Angstrom, and the images are placed on the straight lines from
/// image to image; the default has both in Bohr and places the images on the curve of
/// [`Curve::Smooth`].
pub fn towards_predicted_maxima(
    chain: &[Col<f64>],
    energies: &[f64],
    gradients: &[Col<f64>],
    compatibility: Compatibility,
) -> Vec<Col<f64>> {
    gathered_at_predicted_maxima(chain, energies, gradients, compatibility, None, None)
}

/// [`towards_predicted_maxima`] with the images between the ends and the maxima gathered
/// around the maxima (`None`: spread evenly), and with the moves of the images at the
/// maxima limited where they reverse (`moves`; `None`: as the prediction says).
pub fn gathered_at_predicted_maxima(
    chain: &[Col<f64>],
    energies: &[f64],
    gradients: &[Col<f64>],
    compatibility: Compatibility,
    gathering: Option<Gathering>,
    mut moves: Option<&mut PeakMoves>,
) -> Vec<Col<f64>> {
    let images = chain.len();
    let unit = match compatibility {
        Compatibility::Consistent => 1.0,
        Compatibility::MultiOptPy => Units::BOHR_TO_ANGSTROM,
    };
    // Arc length without moving the images to their centroids.
    let mut s = vec![0.0; images];
    for i in 1..images {
        s[i] = s[i - 1] + unit * col_norm(&(&chain[i] - &chain[i - 1]));
    }
    let total = s[images - 1];
    if total < 1e-12 || images < 3 || energies.len() != images || gradients.len() != images {
        return chain.to_vec();
    }
    // Slope of the energy along the chain (central differences for the tangents) and its
    // derivative along the chain.
    let tangent = |i: usize| {
        let d = match i {
            0 => &chain[1] - &chain[0],
            i if i == images - 1 => &chain[i] - &chain[i - 1],
            i => &chain[i + 1] - &chain[i - 1],
        };
        let norm = (unit * col_norm(&d)).max(1e-12);
        (unit / norm) * d
    };
    let slope: Vec<f64> = (0..images)
        .map(|i| col_dot(&gradients[i], &tangent(i)))
        .collect();
    let mut curvature = vec![0.0; images];
    for k in 1..images - 1 {
        let span = s[k + 1] - s[k - 1];
        if span > 1e-10 {
            curvature[k] = (slope[k + 1] - slope[k - 1]) / span;
        }
    }
    curvature[0] = curvature[1];
    curvature[images - 1] = curvature[images - 2];

    let mut anchors = vec![(0_usize, 0.0)];
    let maxima = strict_maxima(energies);
    if let Some(moves) = moves.as_deref_mut() {
        moves.keep_only(images, &maxima);
    }
    for i in maxima {
        let around = [s[i - 1] - s[i], 0.0, s[i + 1] - s[i]];
        let e = [energies[i - 1], energies[i], energies[i + 1]];
        let g = [slope[i - 1], slope[i], slope[i + 1]];
        let c = [curvature[i - 1], curvature[i], curvature[i + 1]];
        let found =
            fitted_maximum(around, e, g, Some(c)).or_else(|| fitted_maximum(around, e, g, None));
        if let Some(local) = found {
            let local = match moves.as_deref_mut() {
                Some(moves) => moves.limited(i, local),
                None => local,
            };
            // The order of the anchors along the chain is kept.
            let position = s[i] + local;
            let previous = anchors[anchors.len() - 1].1;
            anchors.push((
                i,
                if position <= previous {
                    previous + ANCHOR_NUDGE
                } else {
                    position
                },
            ));
        }
    }
    let previous = anchors[anchors.len() - 1].1;
    anchors.push((
        images - 1,
        if total <= previous {
            previous + ANCHOR_NUDGE
        } else {
            total
        },
    ));
    // The maxima are the anchors between the ends.
    let peaks: Vec<f64> = anchors[1..anchors.len() - 1]
        .iter()
        .map(|(_, position)| *position)
        .collect();
    let density = gathering.map(|g| {
        let spacing = total / (images - 1) as f64;
        (peaks.as_slice(), g.width * spacing, g.weight)
    });
    let mut target = vec![0.0; images];
    for pair in anchors.windows(2) {
        let ((first, from), (last, to)) = (pair[0], pair[1]);
        target[first] = from;
        target[last] = to;
        let between = positions_between((from, to), (last - first).max(1), density);
        for (value, position) in target[first + 1..last].iter_mut().zip(between) {
            *value = position;
        }
    }
    let smooth = (Curve::of(compatibility) == Curve::Smooth).then(|| CubicThrough::new(chain, &s));
    let mut out: Vec<Col<f64>> = target
        .iter()
        .map(|&position| match &smooth {
            Some(cubic) => cubic.point(position),
            None => interpolated(chain, &s, position),
        })
        .collect();
    out[0] = chain[0].clone();
    out[images - 1] = chain[images - 1].clone();
    out
}

/// Where an image of a chain with added images comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSource {
    /// An image of the chain before, by its index there.
    Old(usize),
    /// A new image; `like` is the image of the chain before that is nearest to it along
    /// the chain.
    New {
        /// See the variant.
        like: usize,
    },
}

/// `chain` with `added` images on either side of every image of `maxima` (inner images),
/// at equal distances between that image and its neighbour, on the curve `curve` through
/// the images; and where every image of the new chain comes from.
///
/// MultiOptPy: `neb.py:NEB._exec_adaptive_neb` (points on the straight lines between the
/// images, [`Curve::Polyline`]).
pub fn with_images_around(
    chain: &[Col<f64>],
    maxima: &[usize],
    added: usize,
    curve: Curve,
) -> (Vec<Col<f64>>, Vec<ImageSource>) {
    let images = chain.len();
    let mut at = vec![0.0; images];
    for i in 1..images {
        at[i] = at[i - 1] + col_norm(&(&chain[i] - &chain[i - 1]));
    }
    let smooth = (curve == Curve::Smooth && images > 1).then(|| CubicThrough::new(chain, &at));
    let point = |position: f64| match &smooth {
        Some(cubic) => cubic.point(position),
        None => interpolated(chain, &at, position),
    };
    let mut out = Vec::with_capacity(images + 2 * added * maxima.len());
    let mut sources = Vec::with_capacity(out.capacity());
    for i in 0..images {
        let around = added > 0 && i > 0 && i + 1 < images && maxima.contains(&i);
        if around {
            for j in 0..added {
                let share = (j + 1) as f64 / (added + 1) as f64;
                out.push(point(at[i - 1] + share * (at[i] - at[i - 1])));
                sources.push(ImageSource::New {
                    like: if share < 0.5 { i - 1 } else { i },
                });
            }
        }
        out.push(chain[i].clone());
        sources.push(ImageSource::Old(i));
        if around {
            for j in 0..added {
                let share = (j + 1) as f64 / (added + 1) as f64;
                out.push(point(at[i] + share * (at[i + 1] - at[i])));
                sources.push(ImageSource::New {
                    like: if share < 0.5 { i } else { i + 1 },
                });
            }
        }
    }
    (out, sources)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "redistribute_tests.rs"]
mod tests;
