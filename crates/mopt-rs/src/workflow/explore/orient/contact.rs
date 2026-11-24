//! Contacts between the parts of a complex: van der Waals radii, how close two groups of
//! atoms are, the point along a line at which a moving group first touches the atoms already
//! placed, and the free side of a slab in a cell.
//!
//! Closeness is the contact ratio `min |r_i - r_j| / (R_i + R_j)` over the pairs of the two
//! groups, with the van der Waals radii `R` of Alvarez (Dalton Trans. 2013, 42, 8617; half the
//! UFF distance where that table has none) and minimum-image distances in a cell.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy places the parts at a fixed
//!  distance between their centres (`Entrypoints/orientation_search.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, PeriodicCell, Units};
use crate::parameters::forcefield::uff_vdw_distance_bohr;
use crate::parameters::vdw::alvarez_vdw_radius_bohr;

/// Van der Waals radius (Bohr) of `el`.
pub fn vdw_radius_bohr(el: Element) -> f64 {
    alvarez_vdw_radius_bohr(el).unwrap_or_else(|| 0.5 * uff_vdw_distance_bohr(el))
}

/// Step (Bohr) of the scan along a line before the contact is bisected.
const SCAN_STEP_BOHR: f64 = 0.25;
/// Bisections of the contact (the interval shrinks to 0.25 / 2^40 Bohr).
const BISECTIONS: usize = 40;

pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Mean of points.
pub(crate) fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len().max(1) as f64;
    points
        .iter()
        .fold([0.0; 3], |c, p| add(c, scale(*p, 1.0 / n)))
}

/// Atoms placed so far.
#[derive(Debug, Clone)]
pub struct Placed<'a> {
    /// Positions (Bohr).
    pub positions: Vec<[f64; 3]>,
    /// Van der Waals radii (Bohr).
    pub radii: Vec<f64>,
    /// The cell of minimum-image distances.
    pub cell: Option<&'a PeriodicCell>,
}

impl<'a> Placed<'a> {
    /// Nothing placed yet.
    pub fn new(cell: Option<&'a PeriodicCell>) -> Self {
        Self {
            positions: Vec::new(),
            radii: Vec::new(),
            cell,
        }
    }

    /// Add a group of atoms.
    pub fn extend(&mut self, positions: &[[f64; 3]], radii: &[f64]) {
        self.positions.extend_from_slice(positions);
        self.radii.extend_from_slice(radii);
    }

    /// Whether nothing is placed.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Smallest contact ratio between the placed atoms and `group` (infinite when nothing is
    /// placed).
    pub fn contact_ratio(&self, group: &[[f64; 3]], radii: &[f64]) -> f64 {
        let mut best = f64::INFINITY;
        for (p, rp) in self.positions.iter().zip(&self.radii) {
            for (q, rq) in group.iter().zip(radii) {
                let d = match self.cell {
                    Some(cell) => cell.mic_distance(*p, *q),
                    None => norm([p[0] - q[0], p[1] - q[1], p[2] - q[2]]),
                };
                best = best.min(d / (rp + rq));
            }
        }
        best
    }

    /// Centroid of the placed atoms (in a cell: of the positions as they are).
    pub fn centroid(&self) -> [f64; 3] {
        centroid(&self.positions)
    }

    /// Largest distance of a placed atom from `center` plus its radius.
    pub fn reach(&self, center: [f64; 3]) -> f64 {
        self.positions
            .iter()
            .zip(&self.radii)
            .map(|(p, r)| norm([p[0] - center[0], p[1] - center[1], p[2] - center[2]]) + r)
            .fold(0.0, f64::max)
    }
}

/// Largest distance of a point of a group about its own origin plus its radius.
pub fn group_reach(group: &[[f64; 3]], radii: &[f64]) -> f64 {
    group
        .iter()
        .zip(radii)
        .map(|(p, r)| norm(*p) + r)
        .fold(0.0, f64::max)
}

/// `group` (positions about its own origin) put at `at`.
pub fn moved(group: &[[f64; 3]], at: [f64; 3]) -> Vec<[f64; 3]> {
    group.iter().map(|p| add(*p, at)).collect()
}

/// The largest `t` in `[t_min, t_max]` at which `group` put at `start + t direction` comes
/// to the contact ratio `ratio` with the placed atoms, approached from `t_max` (where it must
/// be farther). `None` when it is already closer at `t_max`; `t_min` when it never comes that
/// close.
pub fn first_contact(
    placed: &Placed<'_>,
    group: &[[f64; 3]],
    radii: &[f64],
    start: [f64; 3],
    direction: [f64; 3],
    (t_min, t_max): (f64, f64),
    ratio: f64,
) -> Option<f64> {
    let gap = |t: f64| {
        placed.contact_ratio(&moved(group, add(start, scale(direction, t))), radii) - ratio
    };
    if gap(t_max) <= 0.0 {
        return None;
    }
    let mut outer = t_max;
    let mut inner = t_max;
    loop {
        if inner <= t_min {
            return Some(t_min);
        }
        let next = (inner - SCAN_STEP_BOHR).max(t_min);
        if gap(next) <= 0.0 {
            inner = next;
            break;
        }
        outer = next;
        inner = next;
    }
    // The contact lies in (inner, outer]: `gap(inner) <= 0 < gap(outer)`.
    for _ in 0..BISECTIONS {
        let mid = 0.5 * (inner + outer);
        if gap(mid) <= 0.0 {
            inner = mid;
        } else {
            outer = mid;
        }
    }
    Some(outer)
}

/// The free side of atoms in a cell: the lattice direction across the widest empty layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Slab {
    /// Unit normal of the layer (Bohr direction), pointing from the surface into the gap.
    pub normal: [f64; 3],
    /// The lattice axis across the layer.
    pub axis: usize,
    /// Height (along the normal) of the surface below the gap.
    pub top: f64,
    /// Thickness of the empty layer (infinite along a direction that is not periodic).
    pub gap: f64,
}

/// Smallest empty layer (Angstrom) taken as the free side of a slab: two contact distances of
/// heavy atoms (2 x 3.5 Angstrom) and room for a small molecule.
pub const SLAB_GAP_ANGSTROM: f64 = 8.0;

/// The widest empty layer of `positions` in `cell` across one of its lattice planes, when it
/// is at least [`SLAB_GAP_ANGSTROM`] thick.
pub fn slab_of(cell: &PeriodicCell, positions: &[[f64; 3]]) -> Option<Slab> {
    if positions.is_empty() {
        return None;
    }
    let widths = cell.perpendicular_widths_bohr();
    let mut best: Option<Slab> = None;
    for (axis, &period) in widths.iter().enumerate() {
        let row = cell.h_inv_bohr[axis];
        let length = norm(row);
        if length <= 0.0 {
            continue;
        }
        let normal = scale(row, 1.0 / length);
        let mut heights: Vec<f64> = positions.iter().map(|p| dot(normal, *p)).collect();
        let (top, gap) = if cell.periodic_axes[axis] {
            for h in &mut heights {
                *h = h.rem_euclid(period);
            }
            heights.sort_by(f64::total_cmp);
            // The circular gaps between consecutive heights.
            let mut widest = (
                heights[heights.len() - 1],
                heights[0] + period - heights[heights.len() - 1],
            );
            for w in heights.windows(2) {
                if w[1] - w[0] > widest.1 {
                    widest = (w[0], w[1] - w[0]);
                }
            }
            widest
        } else {
            (
                heights.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                f64::INFINITY,
            )
        };
        if gap >= SLAB_GAP_ANGSTROM * Units::ANGSTROM_TO_BOHR
            && best.is_none_or(|b: Slab| gap > b.gap)
        {
            best = Some(Slab {
                normal,
                axis,
                top,
                gap,
            });
        }
    }
    best
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::bohr;

    #[test]
    fn a_group_stops_at_the_contact_distance() {
        let mut placed = Placed::new(None);
        placed.extend(&[[0.0; 3]], &[2.0]);
        let t = first_contact(
            &placed,
            &[[0.0; 3]],
            &[1.0],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            (0.0, 20.0),
            1.1,
        )
        .unwrap();
        assert!((t - 3.3).abs() < 1e-9, "{t}");
        // Already too close at the start of the approach: nothing.
        assert!(first_contact(
            &placed,
            &[[0.0; 3]],
            &[1.0],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            (0.0, 1.0),
            1.1
        )
        .is_none());
        assert!((placed.contact_ratio(&[[6.0, 0.0, 0.0]], &[1.0]) - 2.0).abs() < 1e-12);
        assert!((group_reach(&[[3.0, 4.0, 0.0]], &[1.0]) - 6.0).abs() < 1e-12);
        assert!(vdw_radius_bohr(Element::O) > 2.5 && vdw_radius_bohr(Element::H) > 2.0);
    }

    #[test]
    fn the_free_side_of_a_slab_is_found_in_a_periodic_cell() {
        let a = 5.0;
        let cell = PeriodicCell::orthorhombic(bohr(a), bohr(a), bohr(40.0)).unwrap();
        // Two layers at z = 2 and z = 6 Bohr: the gap is from 6 up to 42 (= 2 + 40).
        let atoms = [[0.0, 0.0, 2.0], [2.5, 2.5, 6.0]];
        let slab = slab_of(&cell, &atoms).unwrap();
        assert_eq!(slab.axis, 2);
        assert!((slab.top - 6.0).abs() < 1e-12 && (slab.gap - 36.0).abs() < 1e-12);
        assert!((slab.normal[2] - 1.0).abs() < 1e-12);
        // A full box has no free side.
        let full: Vec<[f64; 3]> = (0..10).map(|k| [0.0, 0.0, 4.0 * k as f64]).collect();
        assert!(slab_of(&cell, &full).is_none());
    }
}
