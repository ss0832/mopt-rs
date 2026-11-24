//! The lattice translations of the periodic GFN-FF: xtb's translation sets, its sums of
//! translations, and its wrapping of positions into the cell.
//!
//! As xtb 6.7.1 computes them (`getTransVec`, `filliTrSum`, `fTrSum`, `init_n` in
//! `src/gfnff/neighbor.f90`; `shift_back_abc` in `src/pbc_tools.f90`; LGPL-3.0-or-later),
//! three-dimensional cells only. A translation is an integer vector `n` (the image of an atom
//! is `x + L n`) instead of xtb's index into its table; the order of the table is kept:
//! - the cells in shell order (index 1 the home cell, then each shell `s = max |n_k|` with
//!   `n_z` slowest and `n_x` fastest), so that the 27 cells around the home cell come first
//!   in a fixed order;
//! - the set of a cutoff keeps every cell of the first shell and a cell of shell `s >= 2`
//!   when `|L n| (1 - 1/s) <= cutoff`, up to the shell `ceil(cutoff / |a_min|)` of the
//!   shortest lattice vector;
//! - the sum of two translations of a set is defined when both and the sum lie within the
//!   shells of xtb's table of sums (at most four shells, fewer for large cells) and the sum is
//!   in the set ([`Translations::sum`]); otherwise xtb drops the term that needs it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;

use super::neighbors::V3;
use crate::error::{MoptError, MoptResult};

/// A lattice translation in units of the lattice vectors.
pub type Cell = [i32; 3];

/// The home cell.
pub const HOME: Cell = [0, 0, 0];

/// Cutoff of the largest translation set (Bohr), which also sizes xtb's table of sums.
pub const LARGEST_CUTOFF: f64 = 60.0;

/// Shells of xtb's table of sums at most (its first 729 cells).
const SUM_SHELLS: i32 = 4;

/// Translation tables beyond this many cells are refused, as xtb does ("Implementation is
/// not adjusted for such small cells").
const LARGEST_TABLE: usize = 24_389;

/// The lattice vectors of a three-dimensional cell (Bohr).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lattice {
    /// `a[k]` is the k-th lattice vector.
    pub a: [V3; 3],
}

impl Lattice {
    /// `L n`.
    pub fn translation(&self, n: Cell) -> V3 {
        let [a, b, c] = self.a;
        [0, 1, 2].map(|k| f64::from(n[0]) * a[k] + f64::from(n[1]) * b[k] + f64::from(n[2]) * c[k])
    }

    /// The image `x + L n`.
    pub fn image(&self, x: V3, n: Cell) -> V3 {
        let t = self.translation(n);
        [x[0] + t[0], x[1] + t[1], x[2] + t[2]]
    }

    /// Length of the shortest lattice vector.
    pub fn shortest(&self) -> f64 {
        self.a
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .fold(f64::INFINITY, f64::min)
    }

    /// The inverse of the matrix with the lattice vectors as columns.
    pub fn inverse(&self) -> MoptResult<[[f64; 3]; 3]> {
        let m = [0, 1, 2].map(|r| [0, 1, 2].map(|c| self.a[c][r]));
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det.abs() < 1e-12 {
            return Err(MoptError::calc(
                "GFN-FF: the lattice vectors are linearly dependent",
            ));
        }
        let cof = |r: usize, c: usize| {
            let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
            let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
            m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]
        };
        Ok([0, 1, 2].map(|r| [0, 1, 2].map(|c| cof(c, r) / det)))
    }

    /// The positions wrapped into the cell as xtb wraps them: fractional coordinates moved
    /// into `[0, 1)` (with xtb's tolerance of 1e-14 at both ends), then back to Cartesian.
    pub fn wrap(&self, x: &[V3]) -> MoptResult<Vec<V3>> {
        let inv = self.inverse()?;
        Ok(x.iter()
            .map(|p| {
                let f = [0, 1, 2].map(|r| inv[r][0] * p[0] + inv[r][1] * p[1] + inv[r][2] * p[2]);
                let f = f.map(shift_back);
                let [a, b, c] = self.a;
                [0, 1, 2].map(|k| f[0] * a[k] + f[1] * b[k] + f[2] * c[k])
            })
            .collect())
    }
}

/// xtb's `shift_back_abc`: a fractional coordinate into `[0, 1)`, tolerance 1e-14.
fn shift_back(value: f64) -> f64 {
    const EPS: f64 = 1.0e-14;
    let mut out = value;
    if value < -EPS {
        out = value + (-value).ceil();
    }
    if value > 1.0 + EPS {
        out = value - value.floor();
    }
    if (value - 1.0).abs() < EPS {
        out = value - 1.0;
    }
    out
}

/// The cells of shells `0..=shells` in xtb's order.
pub fn shell_order(shells: i32) -> Vec<Cell> {
    let mut out = vec![HOME];
    for s in 1..=shells {
        for z in -s..=s {
            for y in -s..=s {
                for x in -s..=s {
                    if x.abs() == s || y.abs() == s || z.abs() == s {
                        out.push([x, y, z]);
                    }
                }
            }
        }
    }
    out
}

/// The shell of a cell, `max |n_k|`.
pub fn shell(n: Cell) -> i32 {
    n[0].abs().max(n[1].abs()).max(n[2].abs())
}

/// The 27 cells of the first shell in xtb's order.
pub fn central_cells() -> Vec<Cell> {
    shell_order(1)
}

/// Whether `n` is one of the 27 cells around the home cell.
pub fn is_central(n: Cell) -> bool {
    shell(n) <= 1
}

/// The translations of one cutoff.
#[derive(Debug, Clone)]
pub struct Translations {
    /// The kept cells in xtb's order (the home cell first).
    pub cells: Vec<Cell>,
    position: HashMap<Cell, usize>,
    /// Shells of the table of sums.
    sum_shells: i32,
}

impl Translations {
    /// The home cell alone (a molecule).
    pub fn molecule() -> Self {
        Self {
            cells: vec![HOME],
            position: HashMap::from([(HOME, 0)]),
            sum_shells: 0,
        }
    }

    /// The set of `cutoff` (Bohr) in `lattice`.
    pub fn new(lattice: &Lattice, cutoff: f64) -> MoptResult<Self> {
        let shortest = lattice.shortest();
        if shortest.is_nan() || shortest <= 0.0 {
            return Err(MoptError::calc("GFN-FF: a lattice vector is zero"));
        }
        let shells_of = |c: f64| (c / shortest).ceil().max(1.0) as i32;
        let shells = shells_of(cutoff);
        let table = (2 * shells as usize + 1).pow(3);
        if table >= LARGEST_TABLE {
            return Err(MoptError::unsupported(format!(
                "GFN-FF: a cell this small (shortest lattice vector {shortest:.3} Bohr) needs \
                 more than {LARGEST_TABLE} translations, as in xtb"
            )));
        }
        let mut cells = Vec::new();
        for n in shell_order(shells) {
            let s = shell(n);
            if s >= 2 {
                let t = lattice.translation(n);
                let length = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                if length * (1.0 - 1.0 / f64::from(s)) > cutoff {
                    continue;
                }
            }
            cells.push(n);
        }
        let position = cells.iter().enumerate().map(|(k, &n)| (n, k)).collect();
        Ok(Self {
            cells,
            position,
            sum_shells: shells_of(LARGEST_CUTOFF).min(SUM_SHELLS),
        })
    }

    /// Whether `n` is in the set.
    pub fn contains(&self, n: Cell) -> bool {
        self.position.contains_key(&n)
    }

    /// The position of `n` in the set (xtb's index minus one).
    pub fn index(&self, n: Cell) -> Option<usize> {
        self.position.get(&n).copied()
    }

    /// xtb's `fTrSum`: `a + b` when `a` and `b` are in the set, all three lie within the
    /// shells of the table of sums, and the sum is in the set; `None` otherwise.
    pub fn sum(&self, a: Cell, b: Cell) -> Option<Cell> {
        if !self.contains(a) || !self.contains(b) {
            return None;
        }
        let m = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let within = |n: Cell| shell(n) <= self.sum_shells;
        (within(a) && within(b) && within(m) && self.contains(m)).then_some(m)
    }
}

/// A neighbour: an atom and the cell of its image.
pub type Nb = (usize, Cell);

/// The atoms where the terms see them: the positions and, in a periodic cell, the lattice.
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    /// Positions (Bohr).
    pub x: &'a [V3],
    /// The lattice of a periodic cell.
    pub lattice: Option<&'a Lattice>,
}

impl<'a> Frame<'a> {
    /// An isolated molecule.
    pub fn molecule(x: &'a [V3]) -> Self {
        Self { x, lattice: None }
    }

    /// Number of atoms.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether there are no atoms.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The image of atom `j` in cell `n` (the atom itself for a molecule).
    pub fn at(&self, j: usize, n: Cell) -> V3 {
        match self.lattice {
            Some(l) if n != HOME => l.image(self.x[j], n),
            _ => self.x[j],
        }
    }

    /// The cells in which bonds are looked for: the 27 around the home cell, or the home cell
    /// of a molecule.
    pub fn bond_cells(&self) -> Vec<Cell> {
        if self.lattice.is_some() {
            central_cells()
        } else {
            vec![HOME]
        }
    }
}

/// `-n`.
pub fn negate(n: Cell) -> Cell {
    [-n[0], -n[1], -n[2]]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_first_cells_are_in_the_order_of_xtb() {
        let c = central_cells();
        assert_eq!(c.len(), 27);
        // xtb's indices 2, 3, 6, 14, 15, 23, 27 (1-based).
        assert_eq!(c[0], [0, 0, 0]);
        assert_eq!(c[1], [-1, -1, -1]);
        assert_eq!(c[2], [0, -1, -1]);
        assert_eq!(c[5], [0, 0, -1]);
        assert_eq!(c[13], [-1, 0, 0]);
        assert_eq!(c[14], [1, 0, 0]);
        assert_eq!(c[22], [0, 0, 1]);
        assert_eq!(c[26], [1, 1, 1]);
        assert_eq!(shell_order(4).len(), 729);
    }

    #[test]
    fn sets_keep_the_central_cells_and_the_near_outer_ones() {
        // A cube of 10 Bohr: shell 3 for 22.36 Bohr; shell 2 kept when |T| / 2 <= 22.36.
        let lattice = Lattice {
            a: [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        };
        let t = Translations::new(&lattice, 500.0_f64.sqrt()).unwrap();
        assert!(t.cells[..27].iter().all(|&n| is_central(n)));
        assert!(t.contains([2, 2, 2])); // |T| = 34.6, 17.3 <= 22.36
        assert!(t.contains([3, 0, 0])); // |T| = 30, 20 <= 22.36
        assert!(!t.contains([3, 3, 0])); // |T| = 42.4, 28.3 > 22.36
                                         // Sums within the set and the table.
        assert_eq!(t.sum([1, 0, 0], [1, 0, 0]), Some([2, 0, 0]));
        assert_eq!(t.sum([3, 0, 0], [1, 0, 0]), None);
        // A large cell: the table of sums has the first shell only.
        let big = Lattice {
            a: [[90.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 0.0, 90.0]],
        };
        let t = Translations::new(&big, 60.0).unwrap();
        assert_eq!(t.cells.len(), 27);
        assert_eq!(t.sum([1, 0, 0], [1, 0, 0]), None);
        assert_eq!(t.sum([1, 0, 0], [-1, 0, 0]), Some(HOME));
    }

    #[test]
    fn positions_wrap_into_the_cell() {
        let lattice = Lattice {
            a: [[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [0.0, 1.0, 8.0]],
        };
        let x = [[-1.0, 3.0, 4.0], [25.0, -2.0, 17.0]];
        let w = lattice.wrap(&x).unwrap();
        let inv = lattice.inverse().unwrap();
        for (p, q) in x.iter().zip(&w) {
            let f = [0, 1, 2].map(|r| inv[r][0] * q[0] + inv[r][1] * q[1] + inv[r][2] * q[2]);
            assert!(f.iter().all(|v| (0.0..1.0).contains(v)), "{f:?}");
            // The same point up to a lattice translation.
            let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
            let n = [0, 1, 2].map(|r| inv[r][0] * d[0] + inv[r][1] * d[1] + inv[r][2] * d[2]);
            assert!(n.iter().all(|v| (v - v.round()).abs() < 1e-12), "{n:?}");
        }
    }
}
