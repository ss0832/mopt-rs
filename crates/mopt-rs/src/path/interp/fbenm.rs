//! The flat-bottom elastic network model and its correlated form: a function of a
//! structure that is zero where the structure is plausible on the way between two
//! structures and rises where it is not, and chains relaxed on it.
//!
//! Koda and Saito, J. Chem. Theory Comput. 2024, 20, 7176 (FB-ENM) and 2025, 21, 3513
//! (CFB-ENM). The definitions are those of the authors' program (PyDMF, `src/dmf/fbenm.py`,
//! MIT licence, `third_party/PYDMF.md`), which was read for them; the texts of the papers
//! were not accessible.
//!
//! **FB-ENM.** Every pair of atoms has a range of distances `[d_min, d_max]` and costs
//! `(d - d_min)^2 / (0.2 d_min)^2` below it and `(d - d_max)^2 / (0.2 d_max)^2` above it.
//! For a pair that is bonded or two bonds apart in one of the two structures (bonded:
//! nearer than 1.25 times the sum of the covalent radii), or in one plane with it, the
//! range of that structure is its distance there; for any other pair it reaches from the
//! smaller of its distance and the sum of the van der Waals radii to twice the largest
//! distance in the structure. The range of the model is the union of the ranges of the two
//! structures: a bond that is in both keeps its length, a bond that is made or broken may
//! be anything between, and atoms that are not bonded do not collide.
//!
//! **Planes.** Four atoms bonded in a row or three atoms bonded to a fourth that lie in a
//! plane in both structures (0.05 Angstrom) keep all their distances; such groups that
//! share three atoms are one plane.
//!
//! **Correlated term.** An atom that loses one bond and gains one (`i-j` broken, `i-k`
//! made, `j` and `k` not otherwise two bonds apart) should not be far from both partners
//! at once: with `x = d - 1.1 d_bond` of the two pairs, `P = (x1 x2 - g1 g2) /
//! (h1 h2 - g1 g2)` (`g`, `h`: the same differences at `1.5` and `1.6 d_bond`) costs
//! `sqrt(P^2 + 0.05^2) - 0.05` where `x1`, `x2`, and `P` are positive, so the new bond
//! forms as the old one breaks.
//!
//! **The path.** The authors find the path with their direct MaxFlux method, which
//! shortens the path inside the flat bottom. That method is not part of mopt-rs yet; here
//! the straight line is relaxed across the path on the model with the relaxation of
//! [`crate::path::band`] (as [`super::idpp`]), which takes the images out of the places
//! that cost something and leaves the rest of the line as it is. The sequential switching
//! on of the images of the program has no counterpart either.
//!
//! Differences in the model: a bond length `d_bond` is taken in the structure in which
//! the bond exists (the program takes the distances of the last structure for both);
//! elements without a van der Waals radius of Alvarez get no such limit; distances in a
//! periodic cell are those of the nearest images.
//!
//! MultiOptPy: `Potential/idpp.py:CFB_ENM` (`-cfbenm`): the correlated term alone, each
//! image minimized on its own; the flat-bottom terms are not there.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeSet;

use faer::{Col, Mat};

use super::idpp::{relaxation, relaxed};
use super::{nudged, on_the_line, Interpolated, PathFrame, PathSettings};
use crate::domain::{PeriodicCell, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;
use crate::parameters::covalent::covalent_radius_bohr;
use crate::parameters::vdw::alvarez_vdw_radius_bohr;

/// Atoms nearer than this times the sum of their covalent radii are bonded.
pub const BOND_SCALE: f64 = 1.25;
/// Width of the walls of the flat bottom as a share of the limit.
pub const WALL: f64 = 0.2;
/// The three distances of the correlated term as multiples of the bond length.
pub const CORRELATED: [f64; 3] = [1.10, 1.50, 1.60];
/// Rounding of the correlated term.
pub const ROUNDING: f64 = 0.05;
/// Four atoms are in a plane when they are this near to one (Angstrom, root mean square).
pub const PLANE_ANGSTROM: f64 = 0.05;
/// Three atoms count as on one line within this many degrees of 180.
pub const LINE_DEGREES: f64 = 10.0;
/// The model is relaxed times this factor, with which the steps of the relaxation fit
/// its walls; the tolerance is that of the scaled forces.
const SCALE: f64 = 0.02;
/// Largest scaled force component at which the relaxation ends (1/Bohr).
pub const FORCE_TOLERANCE: f64 = 1e-4;

/// The vector from atom `j` to atom `i`.
fn between(x: &Col<f64>, i: usize, j: usize, cell: Option<&PeriodicCell>) -> [f64; 3] {
    let at = |a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
    match cell {
        Some(cell) => cell.mic_delta(at(i), at(j)),
        None => {
            let (p, q) = (at(i), at(j));
            [p[0] - q[0], p[1] - q[1], p[2] - q[2]]
        }
    }
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A square table of booleans or distances of all pairs of atoms.
struct Table<T> {
    atoms: usize,
    values: Vec<T>,
}

impl<T: Copy> Table<T> {
    fn new(atoms: usize, value: T) -> Self {
        Self {
            atoms,
            values: vec![value; atoms * atoms],
        }
    }

    fn get(&self, i: usize, j: usize) -> T {
        self.values[i * self.atoms + j]
    }

    fn set_both(&mut self, i: usize, j: usize, value: T) {
        self.values[i * self.atoms + j] = value;
        self.values[j * self.atoms + i] = value;
    }
}

/// One structure as the model sees it: distances and bonds.
struct Seen {
    distance: Table<f64>,
    bonded: Table<bool>,
    neighbours: Vec<Vec<usize>>,
}

fn seen(x: &Col<f64>, radii: &[f64], cell: Option<&PeriodicCell>) -> Seen {
    let atoms = radii.len();
    let mut out = Seen {
        distance: Table::new(atoms, 0.0),
        bonded: Table::new(atoms, false),
        neighbours: vec![Vec::new(); atoms],
    };
    for i in 0..atoms {
        for j in i + 1..atoms {
            let d = norm(&between(x, i, j, cell));
            out.distance.set_both(i, j, d);
            if d < BOND_SCALE * (radii[i] + radii[j]) {
                out.bonded.set_both(i, j, true);
                out.neighbours[i].push(j);
                out.neighbours[j].push(i);
            }
        }
    }
    out
}

/// The positions of four atoms, the later three by their nearest images of the first.
fn four(x: &Col<f64>, atoms: &[usize; 4], cell: Option<&PeriodicCell>) -> [[f64; 3]; 4] {
    let first = [x[3 * atoms[0]], x[3 * atoms[0] + 1], x[3 * atoms[0] + 2]];
    let mut out = [first; 4];
    for k in 1..4 {
        let v = between(x, atoms[k], atoms[0], cell);
        out[k] = [first[0] + v[0], first[1] + v[1], first[2] + v[2]];
    }
    out
}

/// Root mean square distance of four points from the plane nearest to them.
fn off_the_plane(p: &[[f64; 3]; 4]) -> MoptResult<f64> {
    let centre: Vec<f64> = (0..3)
        .map(|k| p.iter().map(|q| q[k]).sum::<f64>() / 4.0)
        .collect();
    let spread = Mat::from_fn(3, 3, |r, c| {
        p.iter()
            .map(|q| (q[r] - centre[r]) * (q[c] - centre[c]))
            .sum::<f64>()
    });
    let (values, _) = symmetric_eigen(&spread)?;
    let smallest = values.iter().fold(f64::INFINITY, |m, v| m.min(*v));
    Ok((smallest.max(0.0) / 4.0).sqrt())
}

/// Whether the angle at `b` is further than [`LINE_DEGREES`] from a straight one.
fn bent(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    let (u, v) = (
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
        [c[0] - b[0], c[1] - b[1], c[2] - b[2]],
    );
    let cosine = (dot(&u, &v) / (norm(&u) * norm(&v)).max(1e-300)).clamp(-1.0, 1.0);
    180.0 - cosine.acos().to_degrees() > LINE_DEGREES
}

/// Whether the dihedral angle of four points in a row is nearer 0 than 180 degrees.
fn cis(p: &[[f64; 3]; 4]) -> bool {
    let b = |k: usize| {
        [
            p[k + 1][0] - p[k][0],
            p[k + 1][1] - p[k][1],
            p[k + 1][2] - p[k][2],
        ]
    };
    dot(&cross(&b(0), &b(1)), &cross(&b(1), &b(2))) >= 0.0
}

/// The groups of atoms that lie in one plane in every structure.
fn planes(
    structures: &[&Col<f64>],
    views: &[Seen],
    cell: Option<&PeriodicCell>,
) -> MoptResult<Vec<BTreeSet<usize>>> {
    let limit = PLANE_ANGSTROM * Units::ANGSTROM_TO_BOHR;
    let first = &views[0];
    let atoms = first.neighbours.len();
    // Four atoms bonded in a row (each row once), with whether they are cis.
    let mut rows: Vec<([usize; 4], bool)> = Vec::new();
    for a in 0..atoms {
        for &b in &first.neighbours[a] {
            for &c in first.neighbours[b].iter().filter(|c| **c != a) {
                for &d in first.neighbours[c].iter().filter(|d| **d != a && **d != b) {
                    if a < d {
                        let p = four(structures[0], &[a, b, c, d], cell);
                        if off_the_plane(&p)? < limit
                            && bent(&p[0], &p[1], &p[2])
                            && bent(&p[1], &p[2], &p[3])
                        {
                            rows.push(([a, b, c, d], cis(&p)));
                        }
                    }
                }
            }
        }
    }
    // An atom with three of its neighbours.
    let mut stars: Vec<[usize; 4]> = Vec::new();
    for (centre, around) in first.neighbours.iter().enumerate() {
        for (i, &a) in around.iter().enumerate() {
            for (j, &b) in around.iter().enumerate().skip(i + 1) {
                for &c in around.iter().skip(j + 1) {
                    let group = [centre, a, b, c];
                    if off_the_plane(&four(structures[0], &group, cell))? < limit {
                        stars.push(group);
                    }
                }
            }
        }
    }
    // The same in the other structures, with the same bonds.
    for (x, view) in structures.iter().zip(views).skip(1) {
        let mut kept = Vec::with_capacity(rows.len());
        for (row, was_cis) in rows {
            let p = four(x, &row, cell);
            let joined = (0..3).all(|k| view.bonded.get(row[k], row[k + 1]));
            if joined
                && off_the_plane(&p)? < limit
                && bent(&p[0], &p[1], &p[2])
                && bent(&p[1], &p[2], &p[3])
                && cis(&p) == was_cis
            {
                kept.push((row, was_cis));
            }
        }
        rows = kept;
        let mut kept = Vec::with_capacity(stars.len());
        for star in stars {
            let joined = (1..4).all(|k| view.bonded.get(star[0], star[k]));
            if joined && off_the_plane(&four(x, &star, cell))? < limit {
                kept.push(star);
            }
        }
        stars = kept;
    }
    // Groups that share three atoms are one plane.
    let mut groups: Vec<BTreeSet<usize>> = rows
        .iter()
        .map(|(row, _)| row.iter().copied().collect())
        .chain(stars.iter().map(|star| star.iter().copied().collect()))
        .collect();
    let mut found = Vec::new();
    while let Some(mut plane) = groups.pop() {
        loop {
            let (joining, rest): (Vec<_>, Vec<_>) = groups
                .into_iter()
                .partition(|group| group.intersection(&plane).count() >= 3);
            groups = rest;
            if joining.is_empty() {
                break;
            }
            for group in joining {
                plane.extend(group);
            }
        }
        found.push(plane);
    }
    Ok(found)
}

/// One correlated pair of pairs: a bond that breaks and a bond that forms.
#[derive(Debug, Clone, Copy)]
struct Correlated {
    broken: (usize, usize),
    formed: (usize, usize),
    /// `1.1 d_bond` of the two pairs.
    start: [f64; 2],
    /// `(1.5 - 1.1) d_bond` of the two pairs.
    g: [f64; 2],
    /// `(1.6 - 1.1) d_bond` of the two pairs.
    h: [f64; 2],
}

/// The model of the way between structures.
pub struct FlatBottomNetwork {
    atoms: usize,
    lower: Table<f64>,
    upper: Table<f64>,
    correlated: Vec<Correlated>,
    cell: Option<PeriodicCell>,
}

impl FlatBottomNetwork {
    /// The model of `structures` (the first and the last decide which bonds break and
    /// form); with the correlated term when `correlated`. `frame` has the elements.
    pub fn new(structures: &[&Col<f64>], frame: &PathFrame, correlated: bool) -> MoptResult<Self> {
        let atoms = frame.elements.len();
        if structures.len() < 2 || structures.iter().any(|x| x.nrows() != 3 * atoms) {
            return Err(MoptError::config(format!(
                "the flat-bottom elastic network needs the elements of the atoms and two \
                 structures of them ({} elements, {} structures)",
                atoms,
                structures.len()
            )));
        }
        let cell = frame.cell.as_ref();
        let radii: Vec<f64> = frame
            .elements
            .iter()
            .map(|e| covalent_radius_bohr(*e))
            .collect();
        let contact: Vec<Option<f64>> = frame
            .elements
            .iter()
            .map(|e| alvarez_vdw_radius_bohr(*e))
            .collect();
        let views: Vec<Seen> = structures.iter().map(|x| seen(x, &radii, cell)).collect();
        let mut in_a_plane = Table::new(atoms, false);
        for plane in planes(structures, &views, cell)? {
            for &i in &plane {
                for &j in &plane {
                    in_a_plane.set_both(i, j, true);
                }
            }
        }
        let mut lower = Table::new(atoms, f64::INFINITY);
        let mut upper = Table::new(atoms, 0.0_f64);
        for view in &views {
            let widest = view.distance.values.iter().fold(0.0_f64, |m, d| m.max(*d));
            for i in 0..atoms {
                for j in i + 1..atoms {
                    let d = view.distance.get(i, j);
                    let near = view.bonded.get(i, j)
                        || view.neighbours[i]
                            .iter()
                            .any(|middle| view.bonded.get(*middle, j))
                        || in_a_plane.get(i, j);
                    let (low, high) = if near {
                        (d, d)
                    } else {
                        let touching = match (contact[i], contact[j]) {
                            (Some(a), Some(b)) => a + b,
                            _ => f64::INFINITY,
                        };
                        (d.min(touching), 2.0 * widest)
                    };
                    lower.set_both(i, j, lower.get(i, j).min(low));
                    upper.set_both(i, j, upper.get(i, j).max(high));
                }
            }
        }
        let mut terms = Vec::new();
        if correlated {
            let (first, last) = (&views[0], &views[views.len() - 1]);
            let kept = |i: usize, j: usize| first.bonded.get(i, j) && last.bonded.get(i, j);
            // The length of a bond where it exists.
            let bond = |i: usize, j: usize| {
                views
                    .iter()
                    .filter(|view| view.bonded.get(i, j))
                    .map(|view| view.distance.get(i, j))
                    .fold(0.0_f64, f64::max)
            };
            for pivot in 0..atoms {
                let broken: Vec<usize> = (0..atoms)
                    .filter(|j| first.bonded.get(pivot, *j) && !last.bonded.get(pivot, *j))
                    .collect();
                let formed: Vec<usize> = (0..atoms)
                    .filter(|k| last.bonded.get(pivot, *k) && !first.bonded.get(pivot, *k))
                    .collect();
                // An atom that changes one partner for one other.
                let (&[j], &[k]) = (broken.as_slice(), formed.as_slice()) else {
                    continue;
                };
                // Not where the two partners stay two bonds apart (a ring of four).
                if (0..atoms).any(|middle| kept(j, middle) && kept(middle, k)) {
                    continue;
                }
                let lengths = [bond(pivot, j), bond(pivot, k)];
                terms.push(Correlated {
                    broken: (pivot, j),
                    formed: (pivot, k),
                    start: lengths.map(|d| CORRELATED[0] * d),
                    g: lengths.map(|d| (CORRELATED[1] - CORRELATED[0]) * d),
                    h: lengths.map(|d| (CORRELATED[2] - CORRELATED[0]) * d),
                });
            }
        }
        Ok(Self {
            atoms,
            lower,
            upper,
            correlated: terms,
            cell: frame.cell.clone(),
        })
    }

    /// The number of correlated pairs of pairs.
    pub fn correlated_terms(&self) -> usize {
        self.correlated.len()
    }

    /// The limits of the distance of the atoms `i` and `j` (Bohr).
    pub fn limits(&self, i: usize, j: usize) -> (f64, f64) {
        (self.lower.get(i, j), self.upper.get(i, j))
    }

    /// The value of the model at `x` and its gradient.
    pub fn value(&self, x: &Col<f64>) -> (f64, Col<f64>) {
        let cell = self.cell.as_ref();
        let mut value = 0.0;
        let mut gradient = Col::<f64>::zeros(x.nrows());
        let mut pull = |i: usize, j: usize, v: &[f64; 3], d: f64, slope: f64| {
            for k in 0..3 {
                let along = slope * v[k] / d;
                gradient[3 * i + k] += along;
                gradient[3 * j + k] -= along;
            }
        };
        for i in 0..self.atoms {
            for j in i + 1..self.atoms {
                let v = between(x, i, j, cell);
                let d = norm(&v).max(1e-12);
                let (low, high) = (self.lower.get(i, j), self.upper.get(i, j));
                if d < low {
                    let width = WALL * low;
                    value += ((d - low) / width).powi(2);
                    pull(i, j, &v, d, 2.0 * (d - low) / (width * width));
                } else if d > high {
                    let width = WALL * high;
                    value += ((d - high) / width).powi(2);
                    pull(i, j, &v, d, 2.0 * (d - high) / (width * width));
                }
            }
        }
        for term in &self.correlated {
            let (v1, v2) = (
                between(x, term.broken.0, term.broken.1, cell),
                between(x, term.formed.0, term.formed.1, cell),
            );
            let (d1, d2) = (norm(&v1).max(1e-12), norm(&v2).max(1e-12));
            let (x1, x2) = (d1 - term.start[0], d2 - term.start[1]);
            let both = term.g[0] * term.g[1];
            let scale = term.h[0] * term.h[1] - both;
            let product = x1 * x2 - both;
            if x1 <= 0.0 || x2 <= 0.0 || product <= 0.0 || scale <= 0.0 {
                continue;
            }
            let p = product / scale;
            let root = (p * p + ROUNDING * ROUNDING).sqrt();
            // sqrt(p^2 + e^2) - e = p^2 / (sqrt(p^2 + e^2) + e): no cancellation.
            value += p * p / (root + ROUNDING);
            let slope = p / root / scale;
            pull(term.broken.0, term.broken.1, &v1, d1, slope * x2);
            pull(term.formed.0, term.formed.1, &v2, d2, slope * x1);
        }
        (value, gradient)
    }
}

/// `count` images between `first` and `last`: the straight line relaxed on the
/// flat-bottom elastic network of the two structures (with the correlated term when
/// `correlated`).
pub fn fbenm_images(
    first: &Col<f64>,
    last: &Col<f64>,
    count: usize,
    frame: &PathFrame,
    settings: &PathSettings,
    correlated: bool,
) -> MoptResult<Interpolated> {
    let model = FlatBottomNetwork::new(&[first, last], frame, correlated)?;
    if count == 0 {
        return Ok(Interpolated {
            images: Vec::new(),
            iterations: 0,
            converged: true,
        });
    }
    let mut chain = Vec::with_capacity(count + 2);
    chain.push(first.clone());
    chain.extend(
        on_the_line(first, last, count)
            .iter()
            .map(|image| nudged(image, frame)),
    );
    chain.push(last.clone());
    let on_the_model = PathSettings {
        force_tolerance: FORCE_TOLERANCE,
        ..*settings
    };
    let (chain, iterations, converged) =
        relaxed(relaxation(frame, &on_the_model), chain, |point, _| {
            let (value, gradient) = model.value(point);
            (SCALE * value, gradient * SCALE)
        })?;
    Ok(Interpolated {
        images: chain[1..=count].to_vec(),
        iterations,
        converged,
    })
}
