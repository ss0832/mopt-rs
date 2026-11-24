//! Flat-bottomed well potentials (MultiOptPy `-wp`, `-wwp`, `-vpwp`, `-awp`).
//!
//! A distance `r` is free between `b` and `c` and walled outside: with
//! `x = 0.5 (r - a) / (b - a) + 0.5` below `b` (and the mirrored map above `c`), the wall is
//! `W (2 - 20 x^3 + 30 x^4 - 12 x^5)` between `a` and `b` (and between `c` and `d`) and continues
//! linearly as `W (2.875 - 3.75 x)` beyond `a` (and `d`). The pieces join with continuous first and
//! second derivatives; `W` is the wall energy reached at `a` and `d`.
//!
//! MultiOptPy: `Potential/switching_potential.py` (`WellPotential`, `WellPotentialWall`,
//!  `WellPotentialVP`, `WellPotentialAround`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::engine::{norm, Clock, Geom, GeometricBias};
use crate::domain::AtomId;
use crate::math::ad::Scalar;
use crate::pes::term::{Frame, History};

/// Wall energy and limits `a < b < c < d` (Bohr).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Well {
    /// Wall energy `W` (Hartree).
    pub wall: f64,
    /// Limits `[a, b, c, d]` (Bohr).
    pub limits: [f64; 4],
}

impl Well {
    /// Energy at distance `r`.
    pub fn at<T: Scalar>(&self, r: T) -> T {
        let [a, b, c, d] = self.limits;
        let rv = r.val();
        let w = T::from_f64(self.wall);
        let smooth = |x: T| {
            let x3 = x * x * x;
            T::from_f64(2.0) - T::from_f64(20.0) * x3 + T::from_f64(30.0) * x3 * x
                - T::from_f64(12.0) * x3 * x * x
        };
        let linear = |x: T| T::from_f64(2.875) - T::from_f64(3.75) * x;
        // Maps a -> 0.5, b -> 1 and d -> 0.5, c -> 1 (MultiOptPy's slopes and intercepts).
        let short = || r * T::from_f64(0.5 / (b - a)) + T::from_f64(1.0 - 0.5 * b / (b - a));
        let long = || r * T::from_f64(0.5 / (c - d)) + T::from_f64(1.0 - 0.5 * c / (c - d));
        if rv <= a {
            w * linear(short())
        } else if rv <= b {
            w * smooth(short())
        } else if rv < c {
            T::zero()
        } else if rv < d {
            w * smooth(long())
        } else {
            w * linear(long())
        }
    }

    /// Whether the limits are increasing.
    pub fn is_valid(&self) -> bool {
        let [a, b, c, d] = self.limits;
        a < b && b < c && c < d
    }
}

/// Where the well is measured from.
#[derive(Debug, Clone, PartialEq)]
pub enum WellGeometry {
    /// Distance between the centroids of two fragments (MultiOptPy `-wp`).
    Fragments {
        /// Fragment 1.
        a: Vec<AtomId>,
        /// Fragment 2.
        b: Vec<AtomId>,
    },
    /// `|x_k|` of each target along Cartesian axis `k` (a slab between two walls; MultiOptPy
    /// `-wwp`). In periodic cells the coordinate is taken from the minimum image of the origin.
    Wall {
        /// Axis (0, 1, 2 for x, y, z).
        axis: usize,
        /// Targets.
        targets: Vec<AtomId>,
    },
    /// Distance of each target from a fixed point (a spherical shell; MultiOptPy `-vpwp`).
    Point {
        /// Point (Bohr).
        point: [f64; 3],
        /// Targets.
        targets: Vec<AtomId>,
    },
    /// Distance of each target from the centroid of a fragment (MultiOptPy `-awp`).
    Around {
        /// Fragment defining the center.
        center: Vec<AtomId>,
        /// Targets.
        targets: Vec<AtomId>,
    },
}

/// A well potential.
#[derive(Debug, Clone)]
pub struct WellPotential {
    /// Shape.
    pub well: Well,
    /// Measured distance(s).
    pub geometry: WellGeometry,
}

impl GeometricBias for WellPotential {
    fn name(&self) -> &str {
        match self.geometry {
            WellGeometry::Fragments { .. } => "well_pot",
            WellGeometry::Wall { .. } => "wall_well_pot",
            WellGeometry::Point { .. } => "void_point_well_pot",
            WellGeometry::Around { .. } => "around_well_pot",
        }
    }
    fn atoms(&self) -> Vec<AtomId> {
        match &self.geometry {
            WellGeometry::Fragments { a, b } => a.iter().chain(b).copied().collect(),
            WellGeometry::Wall { targets, .. } | WellGeometry::Point { targets, .. } => {
                targets.clone()
            }
            WellGeometry::Around { center, targets } => {
                center.iter().chain(targets).copied().collect()
            }
        }
    }
    fn energy<T: Scalar>(&self, g: &Geom<'_, T>, _clock: Clock) -> T {
        let w = &self.well;
        match &self.geometry {
            WellGeometry::Fragments { a, b } => w.at(norm(g.d_centroids(a, b))),
            WellGeometry::Wall { axis, targets } => targets.iter().fold(T::zero(), |e, &t| {
                let x = g.d_point(t, [0.0; 3])[*axis];
                e + w.at(if x.val() < 0.0 { T::zero() - x } else { x })
            }),
            WellGeometry::Point { point, targets } => targets
                .iter()
                .fold(T::zero(), |e, &t| e + w.at(norm(g.d_point(t, *point)))),
            WellGeometry::Around { center, targets } => {
                let c = g.centroid(center);
                targets
                    .iter()
                    .fold(T::zero(), |e, &t| e + w.at(norm(g.d_to(t, c))))
            }
        }
    }
    /// Distances between atoms or centroids are invariant; a wall along a Cartesian axis and a
    /// shell around a fixed point are fixed in space.
    fn frame(&self) -> Frame {
        match self.geometry {
            WellGeometry::Fragments { .. } | WellGeometry::Around { .. } => Frame::Invariant,
            WellGeometry::Wall { .. } | WellGeometry::Point { .. } => Frame::Absolute,
        }
    }
    fn history(&self) -> History {
        History::Static
    }
}
