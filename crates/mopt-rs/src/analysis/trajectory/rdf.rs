//! The radial distribution function of every pair of elements.
//!
//! In a cell periodic along all three axes, `g_ab(r) = V <n_ab(r)> / (N_a N_b' dV(r))` with
//! `n_ab(r)` the pairs of an atom of `a` and one of `b` in the shell from `r` to `r + dr`
//! (`N_b' = N_b`, or `N_a - 1` for `a = b` with each pair counted from both sides),
//! `dV(r)` the volume of the shell, and `V` the volume of the cell of each frame (so that a
//! cell that changes, as at constant pressure, is right). The distances are minimum images,
//! which reach half the smallest width of the cell. Without such a cell there is no density
//! to divide by: `g` is then the distribution of the distances of the pairs, normalized to
//! one over `r` (per Angstrom). The running coordination number `N_ab(r)` (atoms of `b`
//! within `r` of an atom of `a`) is given either way.
//!
//! The default width of a bin, 0.02 Angstrom, is that of GROMACS (`gmx rdf -bin 0.002`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use rayon::prelude::*;

use super::{MdTrajectory, B2A};
use crate::domain::Element;
use crate::error::{MoptError, MoptResult};

/// Default width of a bin (Angstrom).
pub const DEFAULT_BIN_ANGSTROM: f64 = 0.02;

/// Settings of [`radial_distribution`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdfSettings {
    /// Width of a bin (Angstrom).
    pub bin_width: f64,
    /// Largest distance (Angstrom); `None`: half the smallest width of the cell, or the
    /// largest distance of two atoms without a cell.
    pub max: Option<f64>,
}

impl Default for RdfSettings {
    fn default() -> Self {
        Self {
            bin_width: DEFAULT_BIN_ANGSTROM,
            max: None,
        }
    }
}

/// The function of one pair of elements.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfPair {
    /// Element of the central atoms.
    pub first: Element,
    /// Element of the neighbours.
    pub second: Element,
    /// `g(r)` at the centres of the bins.
    pub g: Vec<f64>,
    /// Atoms of `second` within the upper edge of each bin of an atom of `first`.
    pub coordination: Vec<f64>,
}

/// Radial distribution functions.
#[derive(Debug, Clone, PartialEq)]
pub struct Rdf {
    /// Centres of the bins (Angstrom).
    pub r: Vec<f64>,
    /// Width of a bin (Angstrom).
    pub bin_width: f64,
    /// The pairs of elements (each unordered pair once, in the order of first appearance).
    pub pairs: Vec<RdfPair>,
    /// Whether `g` is normalized by the density (a cell periodic along every axis).
    pub periodic: bool,
    /// Frames averaged.
    pub frames: usize,
}

/// The largest distance a minimum image reaches in every frame (Angstrom).
fn minimum_image_limit(traj: &MdTrajectory) -> f64 {
    traj.systems()
        .iter()
        .filter_map(|s| s.active_cell())
        .map(|c| {
            let w = c.perpendicular_widths_bohr();
            0.5 * w.iter().copied().fold(f64::INFINITY, f64::min) * B2A
        })
        .fold(f64::INFINITY, f64::min)
}

fn largest_distance(traj: &MdTrajectory) -> f64 {
    let n = traj.n_atoms();
    (0..traj.len())
        .into_par_iter()
        .map(|t| {
            let mut m = 0.0_f64;
            for i in 0..n {
                for j in i + 1..n {
                    m = m.max(norm(traj.delta(t, i, j)));
                }
            }
            m * B2A
        })
        .reduce(|| 0.0, f64::max)
}

fn norm(d: [f64; 3]) -> f64 {
    libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
}

/// The largest distance of the functions (Angstrom).
fn reach(traj: &MdTrajectory, s: &RdfSettings, periodic: bool) -> MoptResult<f64> {
    let limit = if periodic {
        minimum_image_limit(traj)
    } else {
        f64::INFINITY
    };
    Ok(match s.max {
        Some(m) if m > limit + 1e-9 => {
            return Err(MoptError::config(format!(
                "with the minimum image the distances reach {limit:.4} Angstrom (half the \
                 smallest width of the cell), not {m}"
            )));
        }
        Some(m) if m > 0.0 => m,
        Some(m) => {
            return Err(MoptError::config(format!(
                "the largest distance must be positive, not {m}"
            )))
        }
        None if periodic => limit,
        None => largest_distance(traj) + s.bin_width,
    })
}

/// Index of the unordered pair of elements `a`, `b` among `ns` elements.
fn pair_index(ns: usize, a: usize, b: usize) -> usize {
    let (a, b) = (a.min(b), a.max(b));
    a * ns - a * (a + 1) / 2 + b
}

/// Per frame: the pairs of each pair of elements in each bin, and the volume (Angstrom^3).
fn counts(
    traj: &MdTrajectory,
    kind: &[usize],
    ns: usize,
    bins: usize,
    width: f64,
    max: f64,
) -> Vec<(Vec<f64>, f64)> {
    let n = kind.len();
    (0..traj.len())
        .into_par_iter()
        .map(|t| {
            let mut counts = vec![0.0; ns * (ns + 1) / 2 * bins];
            for i in 0..n {
                for j in i + 1..n {
                    let r = norm(traj.delta(t, i, j)) * B2A;
                    let b = (r / width) as usize;
                    if r < max && b < bins {
                        counts[pair_index(ns, kind[i], kind[j]) * bins + b] += 1.0;
                    }
                }
            }
            let volume = traj.systems()[t]
                .active_cell()
                .map_or(0.0, |c| c.volume_bohr3() * B2A.powi(3));
            (counts, volume)
        })
        .collect()
}

/// `g` and the running coordination number of the pair `p` from the counts of the frames;
/// `partners` ordered pairs, `same` for two atoms of one element, `na` central atoms.
fn pair_function(
    per_frame: &[(Vec<f64>, f64)],
    p: usize,
    (bins, width): (usize, f64),
    (partners, na, same): (f64, f64, bool),
    periodic: bool,
) -> (Vec<f64>, Vec<f64>) {
    let frames = per_frame.len() as f64;
    let mut g = vec![0.0; bins];
    let mut mean_counts = vec![0.0; bins];
    for (counts, volume) in per_frame {
        for k in 0..bins {
            let c = counts[p * bins + k];
            // An unordered pair of the same element counts around both of its atoms.
            let ordered = if same { 2.0 * c } else { c };
            mean_counts[k] += ordered / frames;
            if periodic {
                let (lo, hi) = (k as f64 * width, (k + 1) as f64 * width);
                let shell = 4.0 / 3.0 * PI * (hi.powi(3) - lo.powi(3));
                g[k] += volume * ordered / (partners * shell) / frames;
            }
        }
    }
    if !periodic {
        let total: f64 = mean_counts.iter().sum();
        for k in 0..bins {
            g[k] = if total > 0.0 {
                mean_counts[k] / (total * width)
            } else {
                0.0
            };
        }
    }
    let mut running = 0.0;
    let coordination = mean_counts
        .iter()
        .map(|c| {
            running += c / na;
            running
        })
        .collect();
    (g, coordination)
}

/// The radial distribution functions of `traj` (at least two atoms).
pub fn radial_distribution(traj: &MdTrajectory, s: &RdfSettings) -> MoptResult<Rdf> {
    let n = traj.n_atoms();
    if n < 2 {
        return Err(MoptError::config(
            "a radial distribution needs at least two atoms",
        ));
    }
    if !(s.bin_width > 0.0 && s.bin_width.is_finite()) {
        return Err(MoptError::config("the width of a bin must be positive"));
    }
    let periodic = traj.is_periodic();
    let max = reach(traj, s, periodic)?;
    let bins = ((max / s.bin_width).floor() as usize).max(1);
    let species = traj.species();
    let mut kind = vec![0; n];
    for (c, (_, atoms)) in species.iter().enumerate() {
        for &i in atoms {
            kind[i] = c;
        }
    }
    let ns = species.len();
    let per_frame = counts(traj, &kind, ns, bins, s.bin_width, max);
    let mut pairs = Vec::new();
    for a in 0..ns {
        for b in a..ns {
            let (na, nb) = (species[a].1.len() as f64, species[b].1.len() as f64);
            let partners = if a == b { na * (na - 1.0) } else { na * nb };
            if partners <= 0.0 {
                continue;
            }
            let (g, coordination) = pair_function(
                &per_frame,
                pair_index(ns, a, b),
                (bins, s.bin_width),
                (partners, na, a == b),
                periodic,
            );
            pairs.push(RdfPair {
                first: species[a].0,
                second: species[b].0,
                g,
                coordination,
            });
        }
    }
    Ok(Rdf {
        r: (0..bins).map(|k| (k as f64 + 0.5) * s.bin_width).collect(),
        bin_width: s.bin_width,
        pairs,
        periodic,
        frames: per_frame.len(),
    })
}
