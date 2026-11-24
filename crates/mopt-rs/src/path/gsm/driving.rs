//! The driving coordinates of the single-ended growing string: the changes a reaction is
//! asked to make (bonds formed and broken, angles and torsions brought to values, or
//! variables of plain objectives brought to values), the direction that makes them, and
//! when they are made.
//!
//! P. M. Zimmerman, J. Comput. Chem. 2015, 36, 601 (`ADD`, `BREAK`, `ANGLE`, `TORSION`). The
//! paper steps along them in delocalized internal coordinates; here the direction is the
//! sum of the unit Cartesian gradients of the coordinates that are not made yet, each with
//! the sign that brings it towards its target, and a bond counts as formed below 1.1 and as
//! broken beyond 2.0 times the sum of the covalent radii (Pyykko).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::domain::Element;
use crate::error::{MoptError, MoptResult};
use crate::parameters::covalent::covalent_radius_bohr;

/// A bond counts as formed below this multiple of the sum of the covalent radii.
pub const FORMED: f64 = 1.1;
/// A bond counts as broken beyond this multiple of the sum of the covalent radii.
pub const BROKEN: f64 = 2.0;
/// An angle or torsion counts as made within this many degrees of its target.
pub const ANGLE_TOLERANCE_DEG: f64 = 5.0;

/// One change asked for (atoms and variables from 0).
#[derive(Debug, Clone, PartialEq)]
pub enum DrivingCoordinate {
    /// Form the bond between two atoms.
    Add(usize, usize),
    /// Break the bond between two atoms.
    Break(usize, usize),
    /// Bring the angle `i-j-k` to the value (degrees).
    Angle(usize, usize, usize, f64),
    /// Bring the torsion `i-j-k-l` to the value (degrees).
    Torsion(usize, usize, usize, usize, f64),
    /// Bring a variable of a plain objective to the value.
    Variable(usize, f64),
}

impl DrivingCoordinate {
    /// Parse `ADD i j`, `BREAK i j`, `ANGLE i j k deg`, `TORSION i j k l deg`, `VAR k value`
    /// (atoms and variables counted from 1).
    pub fn parse(text: &str) -> MoptResult<Self> {
        let words: Vec<&str> = text
            .split([' ', ',', ':'])
            .filter(|w| !w.is_empty())
            .collect();
        let bad = || {
            MoptError::parse(format!(
                "driving coordinate '{text}': ADD i j, BREAK i j, ANGLE i j k degrees, \
                 TORSION i j k l degrees, or VAR k value (counted from 1)"
            ))
        };
        let index = |w: &str| -> MoptResult<usize> {
            w.parse::<usize>()
                .ok()
                .filter(|v| *v >= 1)
                .map(|v| v - 1)
                .ok_or_else(bad)
        };
        let number = |w: &str| -> MoptResult<f64> {
            w.parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(bad)
        };
        let kind = words.first().ok_or_else(bad)?.to_ascii_uppercase();
        match (kind.as_str(), words.len()) {
            ("ADD", 3) => Ok(Self::Add(index(words[1])?, index(words[2])?)),
            ("BREAK", 3) => Ok(Self::Break(index(words[1])?, index(words[2])?)),
            ("ANGLE", 5) => Ok(Self::Angle(
                index(words[1])?,
                index(words[2])?,
                index(words[3])?,
                number(words[4])?,
            )),
            ("TORSION", 6) => Ok(Self::Torsion(
                index(words[1])?,
                index(words[2])?,
                index(words[3])?,
                index(words[4])?,
                number(words[5])?,
            )),
            ("VAR", 3) => Ok(Self::Variable(index(words[1])?, number(words[2])?)),
            _ => Err(bad()),
        }
    }

    /// Whether it names atoms (and not a variable).
    pub fn names_atoms(&self) -> bool {
        !matches!(self, Self::Variable(..))
    }

    fn atoms(&self) -> Vec<usize> {
        match self {
            Self::Add(i, j) | Self::Break(i, j) => vec![*i, *j],
            Self::Angle(i, j, k, _) => vec![*i, *j, *k],
            Self::Torsion(i, j, k, l, _) => vec![*i, *j, *k, *l],
            Self::Variable(..) => Vec::new(),
        }
    }

    /// Check the indices against `n` variables (structures: `elements` given).
    pub fn check(&self, n: usize, elements: Option<&[Element]>) -> MoptResult<()> {
        let ok = match (self, elements) {
            (Self::Variable(k, _), _) => *k < n,
            (_, Some(e)) => {
                let atoms = self.atoms();
                let mut sorted = atoms.clone();
                sorted.sort_unstable();
                sorted.dedup();
                sorted.len() == atoms.len() && atoms.iter().all(|a| *a < e.len())
            }
            (_, None) => false,
        };
        if ok {
            Ok(())
        } else {
            Err(MoptError::config(format!(
                "driving coordinate {self:?}: its atoms or variable are out of range (or the \
                 surface has no atoms)"
            )))
        }
    }

    /// Its value at `x` (Bohr; angles in radians) and its gradient.
    pub(crate) fn value(&self, x: &Col<f64>) -> (f64, Col<f64>) {
        let n = x.nrows();
        let p = |a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
        let mut g = Col::<f64>::zeros(n);
        match self {
            Self::Add(i, j) | Self::Break(i, j) => {
                let (a, b) = (p(*i), p(*j));
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                let r = libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                for c in 0..3 {
                    let u = if r > 0.0 { d[c] / r } else { 0.0 };
                    g[3 * i + c] = u;
                    g[3 * j + c] = -u;
                }
                (r, g)
            }
            Self::Angle(i, j, k, _) => numerical(x, &[*i, *j, *k], |q| angle(&q[0], &q[1], &q[2])),
            Self::Torsion(i, j, k, l, _) => numerical(x, &[*i, *j, *k, *l], |q| {
                torsion(&q[0], &q[1], &q[2], &q[3])
            }),
            Self::Variable(k, _) => {
                g[*k] = 1.0;
                (x[*k], g)
            }
        }
    }

    /// The signed distance to the target at `x` (positive: the coordinate must grow) and
    /// whether the change is made.
    fn remaining(&self, x: &Col<f64>, elements: Option<&[Element]>) -> (f64, bool) {
        let (v, _) = self.value(x);
        let radii = |i: usize, j: usize| {
            elements.map_or(0.0, |e| {
                covalent_radius_bohr(e[i]) + covalent_radius_bohr(e[j])
            })
        };
        match self {
            Self::Add(i, j) => {
                let target = FORMED * radii(*i, *j);
                (target - v, v <= target)
            }
            Self::Break(i, j) => {
                let target = BROKEN * radii(*i, *j);
                (target - v, v >= target)
            }
            Self::Angle(_, _, _, deg) => {
                let left = deg.to_radians() - v;
                (left, left.abs().to_degrees() <= ANGLE_TOLERANCE_DEG)
            }
            Self::Torsion(_, _, _, _, deg) => {
                let mut left = deg.to_radians() - v;
                left -= std::f64::consts::TAU * libm::round(left / std::f64::consts::TAU);
                (left, left.abs().to_degrees() <= ANGLE_TOLERANCE_DEG)
            }
            Self::Variable(_, target) => {
                let left = target - v;
                (left, left.abs() <= 1e-3 * target.abs().max(1.0))
            }
        }
    }
}

/// The value of `f` of the positions of `atoms` and its gradient by central differences.
fn numerical(x: &Col<f64>, atoms: &[usize], f: impl Fn(&[[f64; 3]]) -> f64) -> (f64, Col<f64>) {
    let positions = |x: &Col<f64>| -> Vec<[f64; 3]> {
        atoms
            .iter()
            .map(|&a| [x[3 * a], x[3 * a + 1], x[3 * a + 2]])
            .collect()
    };
    let value = f(&positions(x));
    let mut g = Col::<f64>::zeros(x.nrows());
    let h = 1e-5;
    for &a in atoms {
        for c in 0..3 {
            let mut up = x.clone();
            up[3 * a + c] += h;
            let mut down = x.clone();
            down[3 * a + c] -= h;
            let mut d = f(&positions(&up)) - f(&positions(&down));
            // Torsions jump by 2 pi at +-pi.
            d -= std::f64::consts::TAU * libm::round(d / std::f64::consts::TAU);
            g[3 * a + c] = d / (2.0 * h);
        }
    }
    (value, g)
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

fn angle(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let (u, v) = (sub(a, b), sub(c, b));
    libm::atan2(libm::sqrt(dot(&cross(&u, &v), &cross(&u, &v))), dot(&u, &v))
}

fn torsion(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    let (b1, b2, b3) = (sub(b, a), sub(c, b), sub(d, c));
    let (n1, n2) = (cross(&b1, &b2), cross(&b2, &b3));
    let m = cross(&n1, &b2);
    let b2n = libm::sqrt(dot(&b2, &b2)).max(1e-300);
    libm::atan2(dot(&m, &n2) / b2n, dot(&n1, &n2))
}

/// The driving coordinates of a search.
#[derive(Debug, Clone, PartialEq)]
pub struct Driving {
    /// The changes asked for.
    pub coordinates: Vec<DrivingCoordinate>,
    /// The elements of the atoms (structures; `None` for plain variables).
    pub elements: Option<Vec<Element>>,
}

impl Driving {
    /// Check the coordinates against points of `n` variables.
    pub fn check(&self, n: usize) -> MoptResult<()> {
        if self.coordinates.is_empty() {
            return Err(MoptError::config(
                "the single-ended growing string needs driving coordinates",
            ));
        }
        for c in &self.coordinates {
            c.check(n, self.elements.as_deref())?;
        }
        Ok(())
    }

    /// The direction (not normalized; zero when every change is made) at `x` and whether
    /// every change is made.
    pub fn direction(&self, x: &Col<f64>) -> (Col<f64>, bool) {
        let mut d = Col::<f64>::zeros(x.nrows());
        let mut all = true;
        for c in &self.coordinates {
            let (left, made) = c.remaining(x, self.elements.as_deref());
            if made {
                continue;
            }
            all = false;
            let (_, g) = c.value(x);
            let size = crate::math::linalg::col_norm(&g);
            if size > 0.0 {
                d += &g * (left.signum() / size);
            }
        }
        (d, all)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parsed_and_checked() {
        assert_eq!(
            DrivingCoordinate::parse("ADD 1 3").unwrap(),
            DrivingCoordinate::Add(0, 2)
        );
        assert_eq!(
            DrivingCoordinate::parse("torsion 1 2 3 4 180").unwrap(),
            DrivingCoordinate::Torsion(0, 1, 2, 3, 180.0)
        );
        assert!(DrivingCoordinate::parse("ADD 0 1").is_err());
        assert!(DrivingCoordinate::parse("TURN 1 2").is_err());
    }

    #[test]
    fn the_direction_forms_a_bond_and_stops_when_it_is_formed() {
        let elements = Some(vec![Element::H, Element::H, Element::H]);
        let driving = Driving {
            coordinates: vec![DrivingCoordinate::Add(0, 2)],
            elements,
        };
        let x = Col::from_fn(9, |k| [0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 4.0, 0.0, 0.0][k]);
        let (d, done) = driving.direction(&x);
        assert!(!done);
        // Atom 1 moves towards atom 3 and atom 3 towards atom 1.
        assert!(d[0] > 0.0 && d[6] < 0.0 && d[3] == 0.0);
        let near = Col::from_fn(9, |k| [0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 1.0, 0.0, 0.0][k]);
        assert!(driving.direction(&near).1);
    }

    #[test]
    fn angle_gradients_match_their_values() {
        let c = DrivingCoordinate::Angle(0, 1, 2, 90.0);
        let x = Col::from_fn(9, |k| [1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.3][k]);
        let (v, g) = c.value(&x);
        let mut y = x.clone();
        y[7] += 1e-6;
        let (w, _) = c.value(&y);
        assert!(((w - v) / 1e-6 - g[7]).abs() < 1e-5);
    }
}
