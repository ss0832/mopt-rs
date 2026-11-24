//! Spherical coordinates as coordinate systems for optimizations, chains, and paths, like
//! the internal coordinates (`-coord spherical`, `-coord hyperspherical[:FILE]`).
//!
//! - [`Spherical`] (`spherical`): every atom by its distance from the centroid of the
//!   structure and two angles, `(r_i, theta_i, phi_i)` ([`super::super::hyperspherical`] with
//!   `n = 3`). Every atom has its own polar axis, chosen when the system is set up so that
//!   the atom starts on the equator (`theta = pi/2`, `phi = 0`), far from the poles where
//!   the angles are not defined; an atom closer to the centroid than 0.2 Bohr is described
//!   by its Cartesian displacement from the centroid instead. The coordinates do not change
//!   under translations (`3N` coordinates of rank `3N - 3`; the shared transformation treats
//!   them as redundant). The system asks to be set up anew when an atom comes within 5
//!   degrees of a pole or within 0.1 Bohr of the centroid.
//! - [`Hyperspherical`] (`hyperspherical`, also `polar`): the whole structure as one point
//!   of `3N` dimensions relative to a centre (the centroid of the structure, or a given
//!   structure with `hyperspherical:FILE`), by its distance from the centre and `3N - 1`
//!   angles: the coordinates of MultiOptPy's `polar_coordinate.py`. An orthogonal map, set up
//!   with the system, turns the starting displacement onto the direction `(1, ..., 1)`, at
//!   which no partial norm vanishes; the system asks to be set up anew when an angle comes
//!   within 1e-3 rad of a pole. With a given centre the radius is the distance from that
//!   structure (a hypersphere around it is a surface of constant radius).
//!
//! Periodic cells have no centroid that the coordinates could refer to; the systems refuse
//! them.
//!
//! MultiOptPy: `Coordinate/polar_coordinate.py` (as a coordinate system: new in mopt-rs;
//! MultiOptPy uses those coordinates inside `OtherMethod/elastic_image_pair.py` only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::CoordinateSystem;
use crate::coord::hyperspherical::{
    last_angle_difference, partial_norms, to_hyperspherical, wilson,
};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// Atoms nearer to the centroid than this (Bohr) are described by Cartesian displacements.
const CENTRAL: f64 = 0.2;
/// ... and the system is set up anew when a spherical atom comes this near (Bohr).
const TOO_CENTRAL: f64 = 0.1;
/// Sine of the angle from a pole at which [`Spherical`] is set up anew (5 degrees).
const NEAR_A_POLE: f64 = 0.087_155_742_747_658_17;
/// Angle from a pole (rad) at which [`Hyperspherical`] is set up anew.
const NEAR_A_HYPERPOLE: f64 = 1e-3;

fn refuse_cells(sys: &MolecularSystem, name: &str) -> MoptResult<()> {
    if sys.has_pbc() {
        return Err(MoptError::coord(format!(
            "{name} coordinates refer to the centroid of an isolated structure; a periodic \
             cell has none"
        )));
    }
    Ok(())
}

fn centroid(sys: &MolecularSystem) -> [f64; 3] {
    let n = sys.n_atoms().max(1) as f64;
    let mut c = [0.0; 3];
    for atom in sys.atoms() {
        let p = sys.position(atom).value();
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    c
}

fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v.map(|c| c / n)
}

/// A frame (rows) in which the direction `u` is the second axis: `F u = (0, |u|, 0)`.
fn frame_along(u: [f64; 3]) -> [[f64; 3]; 3] {
    let e2 = unit(u);
    // The Cartesian axis least along u, made perpendicular to it.
    let axis = (0..3)
        .min_by(|&a, &b| e2[a].abs().total_cmp(&e2[b].abs()))
        .unwrap_or(0);
    let mut seed = [0.0; 3];
    seed[axis] = 1.0;
    let e1 = unit(cross(e2, seed));
    let e3 = cross(e1, e2);
    [e1, e2, e3]
}

fn apply(f: &[[f64; 3]; 3], u: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|r| (0..3).map(|c| f[r][c] * u[c]).sum())
}

/// How an atom of [`Spherical`] is described.
#[derive(Debug, Clone)]
enum AtomFrame {
    /// `(r, theta, phi)` in the frame (rows).
    Polar([[f64; 3]; 3]),
    /// Cartesian displacement from the centroid.
    Central,
}

/// Every atom by its distance from the centroid and two angles.
#[derive(Debug, Clone, Default)]
pub struct Spherical {
    frames: Vec<AtomFrame>,
}

impl Spherical {
    fn displacements(sys: &MolecularSystem) -> Vec<[f64; 3]> {
        let c = centroid(sys);
        sys.atoms()
            .map(|a| {
                let p = sys.position(a).value();
                [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
            })
            .collect()
    }

    fn check(&self, sys: &MolecularSystem) -> MoptResult<()> {
        if self.frames.len() != sys.n_atoms() {
            return Err(MoptError::coord(
                "spherical coordinates were set up for another number of atoms",
            ));
        }
        Ok(())
    }
}

impl CoordinateSystem for Spherical {
    fn name(&self) -> &str {
        "spherical"
    }

    fn initialize(&mut self, sys: &MolecularSystem) -> MoptResult<()> {
        refuse_cells(sys, "spherical")?;
        self.frames = Self::displacements(sys)
            .into_iter()
            .map(|u| {
                let r = libm::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                if r < CENTRAL {
                    AtomFrame::Central
                } else {
                    AtomFrame::Polar(frame_along(u))
                }
            })
            .collect();
        Ok(())
    }

    fn dim(&self, sys: &MolecularSystem) -> usize {
        sys.dof()
    }

    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        self.check(sys)?;
        let mut q = Col::zeros(sys.dof());
        for (i, (u, frame)) in Self::displacements(sys)
            .iter()
            .zip(&self.frames)
            .enumerate()
        {
            let v = match frame {
                AtomFrame::Polar(f) => to_hyperspherical(&apply(f, *u))?,
                AtomFrame::Central => u.to_vec(),
            };
            for k in 0..3 {
                q[3 * i + k] = v[k];
            }
        }
        Ok(q)
    }

    fn labels(&self) -> Option<Vec<String>> {
        let mut out = Vec::with_capacity(3 * self.frames.len());
        for (i, frame) in self.frames.iter().enumerate() {
            let names: [&str; 3] = match frame {
                AtomFrame::Polar(_) => ["r", "theta", "phi"],
                AtomFrame::Central => ["dx", "dy", "dz"],
            };
            out.extend(names.iter().map(|n| format!("{n}({})", i + 1)));
        }
        Some(out)
    }

    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        self.check(sys)?;
        let n = sys.dof();
        let atoms = sys.n_atoms();
        let share = 1.0 / atoms.max(1) as f64;
        let mut b = Mat::zeros(n, n);
        for (i, (u, frame)) in Self::displacements(sys)
            .iter()
            .zip(&self.frames)
            .enumerate()
        {
            // dq_i/du_i (3 x 3).
            let w: [[f64; 3]; 3] = match frame {
                AtomFrame::Polar(f) => {
                    let d = wilson(&apply(f, *u))?;
                    std::array::from_fn(|r| {
                        std::array::from_fn(|c| (0..3).map(|k| d[(r, k)] * f[k][c]).sum())
                    })
                }
                AtomFrame::Central => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            };
            // du_i/dx_j = (delta_ij - 1/N) 1.
            for j in 0..atoms {
                let factor = if i == j { 1.0 - share } else { -share };
                for r in 0..3 {
                    for c in 0..3 {
                        b[(3 * i + r, 3 * j + c)] = w[r][c] * factor;
                    }
                }
            }
        }
        Ok(b)
    }

    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        let (qt, qf) = (self.values(to)?, self.values(from)?);
        let mut d = &qt - &qf;
        for (i, frame) in self.frames.iter().enumerate() {
            if let AtomFrame::Polar(_) = frame {
                d[3 * i + 2] = last_angle_difference(qt[3 * i + 2], qf[3 * i + 2]);
            }
        }
        Ok(d)
    }

    fn needs_reinitialization(&self, sys: &MolecularSystem) -> bool {
        if self.frames.len() != sys.n_atoms() || sys.has_pbc() {
            return true;
        }
        Self::displacements(sys)
            .iter()
            .zip(&self.frames)
            .any(|(u, frame)| match frame {
                AtomFrame::Polar(f) => {
                    let v = apply(f, *u);
                    let r = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    r < TOO_CENTRAL || libm::hypot(v[1], v[2]) < NEAR_A_POLE * r
                }
                AtomFrame::Central => false,
            })
    }
}

/// The centre of [`Hyperspherical`].
#[derive(Debug, Clone)]
pub enum Centre {
    /// The centroid of the structure itself, for every atom (no translations in the
    /// coordinates).
    Centroid,
    /// A given structure (Bohr, `3N`).
    Structure(Col<f64>),
}

/// The whole structure as one point relative to a centre, by its distance and `3N - 1`
/// angles.
#[derive(Debug, Clone)]
pub struct Hyperspherical {
    centre: Centre,
    /// Orthogonal map (Householder reflection) applied to the displacement first.
    turn: Option<Col<f64>>,
}

impl Hyperspherical {
    /// Relative to `centre`.
    pub fn new(centre: Centre) -> Self {
        Self { centre, turn: None }
    }

    /// The displacement of `sys` from the centre.
    fn displacement_of(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        let x = sys.coords_flat_bohr();
        match &self.centre {
            Centre::Centroid => {
                let c = centroid(sys);
                Ok(Col::from_fn(x.nrows(), |k| x[k] - c[k % 3]))
            }
            Centre::Structure(c) => {
                if c.nrows() != x.nrows() {
                    return Err(MoptError::coord(format!(
                        "the centre of the hyperspherical coordinates has {} coordinates, the \
                         structure {}",
                        c.nrows(),
                        x.nrows()
                    )));
                }
                Ok(&x - c)
            }
        }
    }

    /// `H v` with the Householder vector `w` (`H = 1 - 2 w w^T`, `|w| = 1`).
    fn turned(&self, v: &Col<f64>) -> Vec<f64> {
        match &self.turn {
            Some(w) => {
                let along: f64 = w.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (0..v.nrows()).map(|k| v[k] - 2.0 * along * w[k]).collect()
            }
            None => v.iter().copied().collect(),
        }
    }
}

impl CoordinateSystem for Hyperspherical {
    fn name(&self) -> &str {
        "hyperspherical"
    }

    fn initialize(&mut self, sys: &MolecularSystem) -> MoptResult<()> {
        refuse_cells(sys, "hyperspherical")?;
        self.turn = None;
        let v = self.displacement_of(sys)?;
        let n = v.nrows();
        let length = v.norm_l2();
        if length < 1e-6 {
            return Err(MoptError::coord(
                "hyperspherical coordinates: the structure is at the centre (its radius is zero)",
            ));
        }
        // The reflection that takes v / |v| to (1, ..., 1) / sqrt(n).
        let target = 1.0 / libm::sqrt(n as f64);
        let w = Col::from_fn(n, |k| v[k] / length - target);
        let size = w.norm_l2();
        if size > 1e-12 {
            self.turn = Some(w * faer::Scale(1.0 / size));
        }
        Ok(())
    }

    fn dim(&self, sys: &MolecularSystem) -> usize {
        sys.dof()
    }

    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        let q = to_hyperspherical(&self.turned(&self.displacement_of(sys)?))?;
        Ok(Col::from_fn(q.len(), |k| q[k]))
    }

    fn labels(&self) -> Option<Vec<String>> {
        None
    }

    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        let n = sys.dof();
        let v = self.turned(&self.displacement_of(sys)?);
        let d = wilson(&v)?;
        // dq/dx = (dq/dv) H (dv/dx); H is symmetric.
        let mut b = match &self.turn {
            Some(w) => {
                let dw = &d * w;
                let mut out = d.clone();
                for r in 0..n {
                    for c in 0..n {
                        out[(r, c)] -= 2.0 * dw[r] * w[c];
                    }
                }
                out
            }
            None => d,
        };
        if let Centre::Centroid = self.centre {
            // dv/dx = 1 - (1/N) per axis: subtract the mean over the atoms of each axis.
            let atoms = sys.n_atoms().max(1);
            for r in 0..n {
                for axis in 0..3 {
                    let mean: f64 =
                        (0..atoms).map(|j| b[(r, 3 * j + axis)]).sum::<f64>() / atoms as f64;
                    for j in 0..atoms {
                        b[(r, 3 * j + axis)] -= mean;
                    }
                }
            }
        }
        Ok(b)
    }

    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        let (qt, qf) = (self.values(to)?, self.values(from)?);
        let mut d = &qt - &qf;
        let last = d.nrows() - 1;
        d[last] = last_angle_difference(qt[last], qf[last]);
        Ok(d)
    }

    fn needs_reinitialization(&self, sys: &MolecularSystem) -> bool {
        let Ok(v) = self.displacement_of(sys) else {
            return true;
        };
        let turned = self.turned(&v);
        let rho = partial_norms(&turned);
        let n = turned.len();
        if n < 2 || rho[0] < 1e-6 {
            return true;
        }
        // An angle q_a = atan2(rho_a, x_(a-1)) near 0 or pi: rho_a small against rho_(a-1);
        // the last plane small against the radius.
        (1..n - 1).any(|a| rho[a] < NEAR_A_HYPERPOLE * rho[a - 1])
            || rho[n - 2] < NEAR_A_HYPERPOLE * rho[0]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    use crate::domain::{Element, SpinCharge};

    fn ethane_like() -> MolecularSystem {
        let x = [
            0.0, 0.0, 1.45, 0.0, 0.0, -1.45, 1.9, 0.3, 2.2, -1.2, 1.5, 2.1, -0.6, -1.8, 2.3, 1.7,
            -0.4, -2.1, -1.1, -1.4, -2.3, -0.5, 1.9, -2.2,
        ];
        let elements: Vec<Element> = ["C", "C", "H", "H", "H", "H", "H", "H"]
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect();
        MolecularSystem::from_flat_bohr(
            elements,
            &Col::from_fn(24, |k| x[k]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    fn check_b(system: &mut dyn CoordinateSystem, sys: &MolecularSystem) {
        system.initialize(sys).unwrap();
        let b = system.wilson_b(sys).unwrap();
        let x = sys.coords_flat_bohr();
        let h = 1e-6;
        for k in 0..x.nrows() {
            let mut up = x.clone();
            up[k] += h;
            let mut down = x.clone();
            down[k] -= h;
            let (su, sd) = (
                sys.with_flat_coords_bohr(&up).unwrap(),
                sys.with_flat_coords_bohr(&down).unwrap(),
            );
            let d = system.displacement(&su, &sd).unwrap();
            for r in 0..x.nrows() {
                assert!(
                    (d[r] / (2.0 * h) - b[(r, k)]).abs() < 1e-6,
                    "{} {r} {k}: {} {}",
                    system.name(),
                    d[r] / (2.0 * h),
                    b[(r, k)]
                );
            }
        }
        // No change under a translation.
        let moved = Col::from_fn(x.nrows(), |k| x[k] + [0.3, -0.2, 0.1][k % 3]);
        let d = system
            .displacement(&sys.with_flat_coords_bohr(&moved).unwrap(), sys)
            .unwrap();
        assert!(d.iter().all(|v| v.abs() < 1e-12), "{}", system.name());
    }

    #[test]
    fn wilson_matrices_against_differences_and_no_translations() {
        let sys = ethane_like();
        check_b(&mut Spherical::default(), &sys);
        check_b(&mut Hyperspherical::new(Centre::Centroid), &sys);
        // A given centre: a translation does change the coordinates there.
        let centre = Col::from_fn(24, |k| sys.coords_flat_bohr()[k] + 0.4 * ((k as f64).sin()));
        let mut given = Hyperspherical::new(Centre::Structure(centre.clone()));
        given.initialize(&sys).unwrap();
        let b = given.wilson_b(&sys).unwrap();
        assert_eq!(b.nrows(), 24);
        // The radius is the distance from the centre.
        let q = given.values(&sys).unwrap();
        assert!((q[0] - (sys.coords_flat_bohr() - &centre).norm_l2()).abs() < 1e-12);
    }

    #[test]
    fn atoms_start_on_the_equator_and_the_system_is_set_up_anew_near_a_pole() {
        let sys = ethane_like();
        let mut s = Spherical::default();
        s.initialize(&sys).unwrap();
        let q = s.values(&sys).unwrap();
        for i in 0..8 {
            assert!((q[3 * i + 1] - PI / 2.0).abs() < 1e-12);
            assert!(q[3 * i + 2].abs() < 1e-12 || (q[3 * i + 2] - 2.0 * PI).abs() < 1e-12);
        }
        assert!(!s.needs_reinitialization(&sys));
        // Turn the third atom by 88 degrees about an axis perpendicular to its radius and
        // to the plane it starts in: it comes near the pole of its frame.
        let mut x = sys.coords_flat_bohr();
        let c = centroid(&sys);
        let u = [x[6] - c[0], x[7] - c[1], x[8] - c[2]];
        let r = libm::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        if let AtomFrame::Polar(f) = &s.frames[2] {
            // Along the first axis of its frame: theta -> 0; the fourth atom moves the
            // other way, so that the centroid stays.
            for k in 0..3 {
                let to = c[k] + r * (0.999 * f[0][k] + 0.04 * f[1][k]);
                let shift = to - x[6 + k];
                x[6 + k] = to;
                x[9 + k] -= shift;
            }
        }
        assert!(s.needs_reinitialization(&sys.with_flat_coords_bohr(&x).unwrap()));
        let mut hyper = Hyperspherical::new(Centre::Centroid);
        hyper.initialize(&sys).unwrap();
        assert!(!hyper.needs_reinitialization(&sys));
        // At the start every angle but the last is away from the poles.
        let q = hyper.values(&sys).unwrap();
        for a in 1..23 {
            assert!(q[a] > 0.1 && q[a] < PI - 0.1, "{a} {}", q[a]);
        }
    }
}
