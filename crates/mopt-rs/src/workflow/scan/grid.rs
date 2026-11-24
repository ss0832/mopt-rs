//! The points of a scan: the values of the scanned coordinates, the order in which the points
//! are taken, and the point each one starts from.
//!
//! Two kinds of scans:
//!
//! - joint (MultiOptPy's `relaxedscan`): all coordinates move together, point `k` has the
//!   `k`-th value of every coordinate (one curve, whatever the number of coordinates);
//! - product (`-grid`): every combination of the values, a map of `n_1 x n_2 x ...` points.
//!   Point indices run with the first coordinate fastest.
//!
//! Order and continuation. A relaxed scan starts every point from an optimized neighbour, so
//! that it follows one valley; a map whose points start from far away jumps between valleys.
//! The points of a product scan are taken in snake (boustrophedon) order by default: the
//! first coordinate runs forwards, then backwards at the next value of the second, and so on
//! (a reflected order in more dimensions), so that consecutive points are always neighbours
//! on the grid. Each point starts from the done point nearest to it on the grid (in steps),
//! the most recent one among equally near ones: in snake order that is the point before it.
//! The raster order (`order=raster`: every row in the same direction) starts each row from
//! the first point of the row before.
//!
//! MultiOptPy: `Entrypoints/relaxed_scan.py:run_relaxedscan` (the values `np.linspace(start,
//! end, nsample)` moved together); the product grid and its order are new in mopt-rs.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// One scanned coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanAxis {
    /// Its name in files and figures, such as `bond(1,2)`.
    pub label: String,
    /// Unit of its values (`Angstrom`, `degree`, ...).
    pub unit: String,
    /// Its values in that unit.
    pub values: Vec<f64>,
}

impl ScanAxis {
    /// `count` evenly spaced values from `start` to `end` (both included, as `numpy.linspace`).
    pub fn linspace(
        label: impl Into<String>,
        unit: impl Into<String>,
        start: f64,
        end: f64,
        count: usize,
    ) -> MoptResult<Self> {
        if count == 0 {
            return Err(MoptError::config("a scan needs at least one point"));
        }
        if !(start.is_finite() && end.is_finite()) {
            return Err(MoptError::config("the range of a scan must be finite"));
        }
        let values = (0..count)
            .map(|k| {
                if count == 1 {
                    start
                } else {
                    start + (end - start) * k as f64 / (count - 1) as f64
                }
            })
            .collect();
        Ok(Self {
            label: label.into(),
            unit: unit.into(),
            values,
        })
    }

    /// The label with the unit, for the axes of figures.
    pub fn axis_title(&self) -> String {
        if self.unit.is_empty() {
            self.label.clone()
        } else {
            format!("{} ({})", self.label, self.unit)
        }
    }
}

/// How the coordinates combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridKind {
    /// All move together (one curve).
    Joint,
    /// Every combination of their values.
    Product,
}

/// The order of the points of a product scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridOrder {
    /// Boustrophedon: consecutive points are neighbours.
    Snake,
    /// Every row in the same direction.
    Raster,
}

impl GridOrder {
    /// `snake` or `raster`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "snake" | "boustrophedon" => Ok(Self::Snake),
            "raster" => Ok(Self::Raster),
            other => Err(MoptError::parse(format!(
                "order must be snake or raster, got '{other}'"
            ))),
        }
    }
}

/// The points of a scan.
#[derive(Debug, Clone)]
pub struct ScanGrid {
    axes: Vec<ScanAxis>,
    kind: GridKind,
    /// Points in the order they are taken.
    order: Vec<usize>,
}

impl ScanGrid {
    /// All axes moved together; they must have the same number of values.
    pub fn joint(axes: Vec<ScanAxis>) -> MoptResult<Self> {
        let n = axes.first().map_or(0, |a| a.values.len());
        if axes.is_empty() || axes.iter().any(|a| a.values.len() != n) {
            return Err(MoptError::config(
                "the coordinates of a joint scan need the same number of points",
            ));
        }
        Ok(Self {
            axes,
            kind: GridKind::Joint,
            order: (0..n).collect(),
        })
    }

    /// Every combination of the values of `axes`, taken in `order`.
    pub fn product(axes: Vec<ScanAxis>, order: GridOrder) -> MoptResult<Self> {
        if axes.is_empty() || axes.iter().any(|a| a.values.is_empty()) {
            return Err(MoptError::config("a scan needs at least one point"));
        }
        let sizes: Vec<usize> = axes.iter().map(|a| a.values.len()).collect();
        let total: usize = sizes.iter().product();
        // An odometer: the first dimension steps; at its end it turns back (snake) or starts
        // again (raster) while the next dimension steps, and so on.
        let mut points = Vec::with_capacity(total);
        let mut index = vec![0usize; sizes.len()];
        let mut direction = vec![1i64; sizes.len()];
        points.push(flat_index(&index, &sizes));
        'points: loop {
            let mut d = 0;
            loop {
                if d == sizes.len() {
                    break 'points;
                }
                let next = index[d] as i64 + direction[d];
                if next >= 0 && (next as usize) < sizes[d] {
                    index[d] = next as usize;
                    break;
                }
                match order {
                    GridOrder::Snake => direction[d] = -direction[d],
                    GridOrder::Raster => index[d] = 0,
                }
                d += 1;
            }
            points.push(flat_index(&index, &sizes));
        }
        Ok(Self {
            axes,
            kind: GridKind::Product,
            order: points,
        })
    }

    /// The axes.
    pub fn axes(&self) -> &[ScanAxis] {
        &self.axes
    }

    /// Joint or product.
    pub fn kind(&self) -> GridKind {
        self.kind
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether there are no points (never for a grid that was built).
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Dimensions of the grid of energies: one for a joint scan, one per axis otherwise.
    pub fn shape(&self) -> Vec<usize> {
        match self.kind {
            GridKind::Joint => vec![self.len()],
            GridKind::Product => self.axes.iter().map(|a| a.values.len()).collect(),
        }
    }

    /// The points in the order they are taken.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Grid indices of `point` (one per dimension of [`Self::shape`]).
    pub fn indices(&self, point: usize) -> Vec<usize> {
        let mut rest = point;
        self.shape()
            .iter()
            .map(|&n| {
                let i = rest % n;
                rest /= n;
                i
            })
            .collect()
    }

    /// The point at grid indices `indices`.
    pub fn point_at(&self, indices: &[usize]) -> usize {
        flat_index(indices, &self.shape())
    }

    /// The value of every axis at `point`.
    pub fn values(&self, point: usize) -> Vec<f64> {
        match self.kind {
            GridKind::Joint => self.axes.iter().map(|a| a.values[point]).collect(),
            GridKind::Product => self
                .indices(point)
                .iter()
                .zip(&self.axes)
                .map(|(&i, a)| a.values[i])
                .collect(),
        }
    }

    /// The points one step from `point` along one dimension.
    pub fn neighbours(&self, point: usize) -> Vec<usize> {
        let shape = self.shape();
        let idx = self.indices(point);
        let mut out = Vec::new();
        for d in 0..shape.len() {
            for step in [-1i64, 1] {
                let moved = idx[d] as i64 + step;
                if moved >= 0 && (moved as usize) < shape[d] {
                    let mut next = idx.clone();
                    next[d] = moved as usize;
                    out.push(flat_index(&next, &shape));
                }
            }
        }
        out
    }

    /// Steps on the grid between two points.
    pub fn steps(&self, a: usize, b: usize) -> usize {
        self.indices(a)
            .iter()
            .zip(self.indices(b))
            .map(|(&x, y)| x.abs_diff(y))
            .sum()
    }

    /// The point `point` starts from: of the points in `done` (in the order they were
    /// done), the nearest on the grid, the latest among equally near ones; `None` for the
    /// first point.
    pub fn start_of(&self, point: usize, done: &[usize]) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (rank, &p) in done.iter().enumerate() {
            let s = self.steps(point, p);
            if best.is_none_or(|(steps, r)| s < steps || (s == steps && rank > r)) {
                best = Some((s, rank));
            }
        }
        best.map(|(_, rank)| done[rank])
    }
}

/// Index of the point with grid indices `idx` (first dimension fastest).
fn flat_index(idx: &[usize], sizes: &[usize]) -> usize {
    let mut flat = 0;
    for d in (0..sizes.len()).rev() {
        flat = flat * sizes[d] + idx[d];
    }
    flat
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn axis(n: usize) -> ScanAxis {
        ScanAxis::linspace("q", "", 0.0, 1.0, n).unwrap()
    }

    #[test]
    fn values_are_those_of_linspace() {
        let a = ScanAxis::linspace("bond(1,2)", "Angstrom", 1.3, 2.6, 8).unwrap();
        assert_eq!(a.values.len(), 8);
        assert!((a.values[7] - 2.6).abs() < 1e-15 && (a.values[0] - 1.3).abs() < 1e-15);
        assert!((a.values[1] - (1.3 + 1.3 / 7.0)).abs() < 1e-14);
        assert_eq!(a.axis_title(), "bond(1,2) (Angstrom)");
        assert!(ScanAxis::linspace("q", "", 0.0, 1.0, 0).is_err());
        let joint = ScanGrid::joint(vec![axis(4), axis(4)]).unwrap();
        assert_eq!(joint.shape(), vec![4]);
        assert_eq!(joint.values(2).len(), 2);
        assert!(ScanGrid::joint(vec![axis(4), axis(3)]).is_err());
    }

    #[test]
    fn snake_order_moves_between_neighbours() {
        let grid = ScanGrid::product(vec![axis(4), axis(3), axis(2)], GridOrder::Snake).unwrap();
        assert_eq!(grid.len(), 24);
        let order = grid.order();
        let mut seen = order.to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..24).collect::<Vec<_>>());
        for pair in order.windows(2) {
            assert_eq!(grid.steps(pair[0], pair[1]), 1, "{pair:?}");
        }
        // Each point starts from the one before it.
        for k in 1..order.len() {
            assert_eq!(grid.start_of(order[k], &order[..k]), Some(order[k - 1]));
        }
        assert_eq!(grid.start_of(order[0], &[]), None);
        // Raster: a row starts from the first point of the row before.
        let raster = ScanGrid::product(vec![axis(3), axis(2)], GridOrder::Raster).unwrap();
        assert_eq!(raster.order(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(raster.start_of(3, &[0, 1, 2]), Some(0));
        assert_eq!(raster.indices(4), vec![1, 1]);
        assert_eq!(raster.point_at(&[2, 1]), 5);
        let mut n = raster.neighbours(4);
        n.sort_unstable();
        assert_eq!(n, vec![1, 3, 5]);
        assert!(GridOrder::parse("zigzag").is_err());
    }
}
