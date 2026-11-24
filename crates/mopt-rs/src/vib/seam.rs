//! Vibrations within the seam of two crossing states, at a minimum-energy crossing point.
//!
//! A crossing point found by gradient projection is a stationary point of the mean energy
//! `E_m = (E_1 + E_2) / 2` on the seam `E_1 = E_2`. With the multiplier `l` of that
//! condition, `grad E_m = l (g_1 - g_2)`, the energy along the seam is to second order
//!
//! ```text
//! 1/2 t^T W t,   W = H_m - l (H_1 - H_2) = (1/2 - l) H_1 + (1/2 + l) H_2
//! ```
//!
//! for displacements `t` within the seam, that is orthogonal to the difference gradient
//! (the seam bends, which is what the term with `H_1 - H_2` accounts for). The vibrations of
//! `W` in the space orthogonal to the branching plane and to the rigid-body motions tell
//! whether the point is a minimum of the seam (all real) or a saddle point on it.
//!
//! **Crossing of states that do not interact** (different spin or symmetry; one direction in
//! the branching plane): the two surfaces are smooth and the result is exact.
//!
//! **Conical intersection** (two directions): the adiabatic surfaces have a cusp, and their
//! Hessians near it contain a term of the size `1 / gap` that lives in the branching plane.
//! It cancels in `H_m` but not in `H_1 - H_2`, where it dominates, and its range is the
//! branching plane, so that projecting the branching plane out removes it. The second
//! direction of the plane is either given ([`SecondDirection::Given`]: from the gradients
//! at displaced structures, see [`crate::opt::crossing_check`], which is the reliable
//! source) or taken from the Hessians themselves ([`SecondDirection::FromHessians`]: the
//! eigenvector of `H_1 - H_2`, within the space orthogonal to the difference gradient, with
//! the eigenvalue of the largest magnitude; exact for analytic Hessians at a small gap, but
//! Hessians made of finite differences that reach across the cusp give a direction that is
//! off by several degrees). The overlap of the two is reported. The curvature of the
//! coupling within the seam is not known and is left out, so the frequencies of a conical
//! intersection are an estimate.
//!
//! Thermodynamic functions are not given: a crossing point is not a stationary point of a
//! surface on which a molecule is in equilibrium.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{transverse_modes, Direction, NormalModes};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, col_rms, symmetric_eigen, symmetrize};

/// Where the second direction of the branching plane of a conical intersection comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondDirection {
    /// The second vector of the plane given (made orthogonal to the difference gradient).
    Given,
    /// The eigenvector of the difference of the Hessians with the largest eigenvalue in
    /// magnitude.
    FromHessians,
}

/// The vibrations within a crossing seam.
#[derive(Debug, Clone)]
pub struct SeamAnalysis {
    /// `E_1 - E_2` (Hartree).
    pub gap: f64,
    /// `(E_1 + E_2) / 2` (Hartree).
    pub mean_energy: f64,
    /// Length of the difference gradient `g_1 - g_2` (Hartree/Bohr).
    pub difference_gradient: f64,
    /// Unit vectors of the branching plane: along the difference gradient, and for a
    /// conical intersection the second direction.
    pub plane: Vec<Col<f64>>,
    /// Multiplier `l` of the seam condition: `grad E_m = l (g_1 - g_2)` within the branching
    /// plane. `|l| < 1/2`: the gradients of the two states point to opposite sides (a peaked
    /// crossing); beyond: to the same side (a sloped one).
    pub multiplier: f64,
    /// Component of the mean gradient along the second direction (Hartree/Bohr).
    pub coupling_component: Option<f64>,
    /// RMS of the mean gradient outside the branching plane (Hartree/Bohr): zero at a
    /// stationary point of the seam.
    pub seam_gradient: f64,
    /// For a conical intersection: the eigenvalue of `H_1 - H_2` of the largest magnitude
    /// and the next one (Hartree/Bohr^2), and the overlap of its eigenvector with the second
    /// direction given.
    pub coupling_curvature: Option<(f64, f64, f64)>,
    /// `W`, the Hessian of the mean energy on the seam before projection (Hartree/Bohr^2).
    pub hessian: Mat<f64>,
    /// The vibrations within the seam.
    pub modes: NormalModes,
    /// Remarks on the validity of the result.
    pub notes: Vec<String>,
}

/// Vibrations within the seam of two states at `sys` from their energies (Hartree),
/// gradients (Hartree/Bohr), and Hessians (Hartree/Bohr^2). `plane` has the directions of
/// the branching plane: one (the difference gradient) for states that do not interact, two
/// for a conical intersection, of which the second is used as `second` says. With
/// `project_rigid`, translations and rotations are removed as well.
pub fn seam_analysis(
    sys: &MolecularSystem,
    energies: [f64; 2],
    gradients: [&Col<f64>; 2],
    hessians: [&Mat<f64>; 2],
    plane: &[Col<f64>],
    second: SecondDirection,
    project_rigid: bool,
) -> MoptResult<SeamAnalysis> {
    let n = sys.dof();
    if gradients.iter().any(|g| g.nrows() != n)
        || hessians.iter().any(|h| h.nrows() != n || h.ncols() != n)
        || plane.iter().any(|p| p.nrows() != n)
    {
        return Err(MoptError::config(format!(
            "seam analysis: gradients, Hessians, and branching plane must have {n} coordinates"
        )));
    }
    if plane.is_empty() || plane.len() > 2 {
        return Err(MoptError::config(
            "seam analysis: the branching plane has one or two directions",
        ));
    }
    let difference = gradients[0] - gradients[1];
    let length = col_norm(&difference);
    if !(length > 1e-10 && length.is_finite()) {
        return Err(MoptError::calc(
            "seam analysis: the gradients of the two states are the same, so that the seam \
             has no direction across it",
        ));
    }
    let x = Col::from_fn(n, |i| difference[i] / length);
    let mean_gradient = Col::from_fn(n, |i| 0.5 * (gradients[0][i] + gradients[1][i]));
    let mut delta = hessians[0] - hessians[1];
    symmetrize(&mut delta);
    let mut notes = Vec::new();

    // The second direction of a conical intersection, from the difference of the Hessians.
    let mut directions = vec![x.clone()];
    let mut coupling_curvature = None;
    if plane.len() == 2 {
        // P (H_1 - H_2) P with P = 1 - x x^T.
        let applied = &delta * &x;
        let both = col_dot(&x, &applied);
        let mut projected = Mat::from_fn(n, n, |i, j| {
            delta[(i, j)] - x[i] * applied[j] - applied[i] * x[j] + x[i] * both * x[j]
        });
        symmetrize(&mut projected);
        let (values, vectors) = symmetric_eigen(&projected)?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| values[*b].abs().total_cmp(&values[*a].abs()));
        let first = order[0];
        let raw = Col::from_fn(n, |i| vectors[(i, first)]);
        let along = col_dot(&raw, &x);
        let orthogonal = Col::from_fn(n, |i| raw[i] - along * x[i]);
        let size = col_norm(&orthogonal);
        if size.is_nan() || size <= 1e-8 {
            return Err(MoptError::calc(
                "seam analysis: the Hessians of the two states do not differ outside the \
                 difference gradient, so that the second direction of the branching plane \
                 cannot be found",
            ));
        }
        let from_hessians = Col::from_fn(n, |i| orthogonal[i] / size);
        // The second direction given, made orthogonal to x.
        let given = &plane[1];
        let given_along = col_dot(given, &x);
        let given_rest = Col::from_fn(n, |i| given[i] - given_along * x[i]);
        let given_size = col_norm(&given_rest);
        let given = (given_size > 1e-8).then(|| Col::from_fn(n, |i| given_rest[i] / given_size));
        let overlap = given
            .as_ref()
            .map_or(f64::NAN, |g| col_dot(g, &from_hessians).abs());
        let next = order.get(1).map_or(0.0, |k| values[*k]);
        coupling_curvature = Some((values[first], next, overlap));
        let y = match (second, given) {
            (SecondDirection::Given, Some(given)) => {
                notes.push(
                    "Conical intersection: the second direction of the branching plane is \
                     the one given. The curvature of the coupling within the seam is left \
                     out; the frequencies are an estimate."
                        .to_string(),
                );
                given
            }
            (SecondDirection::Given, None) => {
                return Err(MoptError::config(
                    "seam analysis: the second direction given is along the difference \
                     gradient",
                ))
            }
            (SecondDirection::FromHessians, _) => {
                notes.push(
                    "Conical intersection: the second direction of the branching plane is \
                     the eigenvector of the difference of the two Hessians with the largest \
                     eigenvalue in magnitude. The curvature of the coupling within the seam \
                     is left out; the frequencies are an estimate."
                        .to_string(),
                );
                from_hessians
            }
        };
        directions.push(y);
    }

    let multiplier = col_dot(&mean_gradient, &x) / length;
    let coupling_component = directions.get(1).map(|y| col_dot(&mean_gradient, y));
    let mut outside = mean_gradient.clone();
    for direction in &directions {
        let along = col_dot(&outside, direction);
        for i in 0..n {
            outside[i] -= along * direction[i];
        }
    }
    let mut hessian = Mat::from_fn(n, n, |i, j| {
        0.5 * (hessians[0][(i, j)] + hessians[1][(i, j)]) - multiplier * delta[(i, j)]
    });
    symmetrize(&mut hessian);
    let removed: Vec<Direction> = directions
        .iter()
        .map(|d| Direction::Gradient(d.clone()))
        .collect();
    let modes = transverse_modes(sys, &hessian, project_rigid, &removed)?;
    Ok(SeamAnalysis {
        gap: energies[0] - energies[1],
        mean_energy: 0.5 * (energies[0] + energies[1]),
        difference_gradient: length,
        plane: directions,
        multiplier,
        coupling_component,
        seam_gradient: col_rms(&outside),
        coupling_curvature,
        hessian,
        modes,
        notes,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};

    fn atom(r: [f64; 3]) -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::H],
            &positions::<Bohr>(&[r]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    /// Two harmonic wells of different stiffness: `V_1 = k_1 / 2 |r|^2`,
    /// `V_2 = k_2 / 2 |r - (d, 0, 0)|^2 + c`. The seam is a sphere about a point of the x
    /// axis, and its lowest point lies on that axis.
    struct Wells {
        k: [f64; 2],
        d: f64,
        c: f64,
    }

    impl Wells {
        fn energies(&self, r: [f64; 3]) -> [f64; 2] {
            let square = |p: [f64; 3]| p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            [
                0.5 * self.k[0] * square(r),
                0.5 * self.k[1] * square([r[0] - self.d, r[1], r[2]]) + self.c,
            ]
        }

        fn gradients(&self, r: [f64; 3]) -> [Col<f64>; 2] {
            [
                Col::from_fn(3, |i| self.k[0] * r[i]),
                Col::from_fn(3, |i| {
                    self.k[1] * (r[i] - if i == 0 { self.d } else { 0.0 })
                }),
            ]
        }

        fn hessians(&self) -> [Mat<f64>; 2] {
            [
                Mat::from_fn(3, 3, |i, j| if i == j { self.k[0] } else { 0.0 }),
                Mat::from_fn(3, 3, |i, j| if i == j { self.k[1] } else { 0.0 }),
            ]
        }

        /// The point of the seam with the given `y` (and `z = 0`) next to `x`, by Newton
        /// steps along x.
        fn on_seam(&self, mut x: f64, y: f64) -> [f64; 3] {
            for _ in 0..50 {
                let r = [x, y, 0.0];
                let [e1, e2] = self.energies(r);
                let slope = self.k[0] * x - self.k[1] * (x - self.d);
                x -= (e1 - e2) / slope;
            }
            [x, y, 0.0]
        }
    }

    #[test]
    fn the_curvature_of_a_bent_seam_is_that_of_the_energy_along_it() {
        let wells = Wells {
            k: [1.0, 0.4],
            d: 1.5,
            c: 0.05,
        };
        let point = wells.on_seam(0.6, 0.0);
        let sys = atom(point);
        let [g1, g2] = wells.gradients(point);
        let [h1, h2] = wells.hessians();
        let plane = vec![Col::from_fn(3, |i| f64::from(u8::from(i == 0)))];
        let found = seam_analysis(
            &sys,
            wells.energies(point),
            [&g1, &g2],
            [&h1, &h2],
            &plane,
            SecondDirection::Given,
            false,
        )
        .unwrap();
        assert!(found.gap.abs() < 1e-12);
        assert!(found.seam_gradient < 1e-12);
        assert!(found.notes.is_empty() && found.coupling_curvature.is_none());
        // Two vibrations within the seam, both with the curvature (1/2 - l) k_1 + (1/2 + l) k_2.
        let mass = sys.masses_amu()[0];
        let expected =
            (0.5 - found.multiplier) * wells.k[0] + (0.5 + found.multiplier) * wells.k[1];
        assert_eq!(found.modes.curvatures.len(), 2);
        for curvature in &found.modes.curvatures {
            assert!((curvature * mass - expected).abs() < 1e-12);
        }
        // The same from the mean energy along the seam itself: E(y) = E(0) + w y^2 / 2.
        let step = 1e-3;
        let moved = wells.on_seam(point[0], step);
        let mean = |r: [f64; 3]| 0.5 * (wells.energies(r)[0] + wells.energies(r)[1]);
        let numeric = 2.0 * (mean(moved) - mean(point)) / (step * step);
        assert!(
            (numeric - expected).abs() < 1e-5,
            "{numeric} against {expected}"
        );
        // It is not the curvature of the mean energy at fixed x: the seam bends.
        assert!((expected - 0.5 * (wells.k[0] + wells.k[1])).abs() > 0.05);
    }

    #[test]
    fn the_second_direction_of_a_conical_intersection_comes_from_the_hessians() {
        // Diabatic wells k / 2 |r - a|^2 and k / 2 |r - b|^2 + c with a, b on the x axis,
        // coupled by v = 0.1 z: the adiabatic states meet on the line x = x0, z = 0. Next to
        // it (at z = 1e-4) the Hessians of the adiabatic states differ by a large term along
        // z, and the seam has one vibration, along y, with the force constant k.
        let (k, coupling, x0, z) = (1.0, 0.1, 0.52, 1e-4);
        // u = (V_1 - V_2) / 2 = k (x - x0) / 2 for wells one Bohr apart, v = coupling z.
        let (u, v): (f64, f64) = (0.0, coupling * z);
        let root = f64::hypot(u, v);
        let gu = [0.5 * k, 0.0, 0.0];
        let gv = [0.0, 0.0, coupling];
        // grad root = (u gu + v gv) / root; Hessian of root = (gu gu^T + gv gv^T - gr gr^T) / root.
        let gr: [f64; 3] = std::array::from_fn(|i| (u * gu[i] + v * gv[i]) / root);
        let cusp = Mat::from_fn(3, 3, |i, j| {
            (gu[i] * gu[j] + gv[i] * gv[j] - gr[i] * gr[j]) / root
        });
        let mean_gradient = [k * (x0 - 0.5), 0.0, k * z];
        let g1 = Col::from_fn(3, |i| mean_gradient[i] - gr[i]);
        let g2 = Col::from_fn(3, |i| mean_gradient[i] + gr[i]);
        let h1 = Mat::from_fn(3, 3, |i, j| k * f64::from(u8::from(i == j)) - cusp[(i, j)]);
        let h2 = Mat::from_fn(3, 3, |i, j| k * f64::from(u8::from(i == j)) + cusp[(i, j)]);
        let sys = atom([x0, 0.0, z]);
        // The optimizer's plane: the difference gradient and a second direction that is off
        // by a few degrees.
        let plane = vec![
            Col::from_fn(3, |i| f64::from(u8::from(i == 2))),
            Col::from_fn(3, |i| [0.99, 0.14, 0.0][i]),
        ];
        let from_hessians = SecondDirection::FromHessians;
        let found = seam_analysis(
            &sys,
            [-root, root],
            [&g1, &g2],
            [&h1, &h2],
            &plane,
            from_hessians,
            false,
        )
        .unwrap();
        // The difference gradient is along z here (u = 0), the second direction along x.
        assert!(found.plane[0][2].abs() > 0.999);
        assert!(found.plane[1][0].abs() > 0.999_999);
        let (largest, next, overlap) = found.coupling_curvature.unwrap();
        assert!(largest.abs() > 1e3 && next.abs() < 1e-6 * largest.abs());
        assert!((overlap - 0.99 / (0.99_f64.powi(2) + 0.14_f64.powi(2)).sqrt()).abs() < 1e-6);
        // One vibration, along y, with the force constant k, although the Hessians of the
        // states have elements of the size 1 / gap.
        let mass = sys.masses_amu()[0];
        assert_eq!(found.modes.curvatures.len(), 1);
        assert!((found.modes.curvatures[0] * mass - k).abs() < 1e-8);
        assert_eq!(found.notes.len(), 1);
        // With the right second direction given, the same; with the one that is off, the
        // large term is not removed and the curvature is far off.
        let right = vec![plane[0].clone(), found.plane[1].clone()];
        let given = SecondDirection::Given;
        let same = seam_analysis(
            &sys,
            [-root, root],
            [&g1, &g2],
            [&h1, &h2],
            &right,
            given,
            false,
        )
        .unwrap();
        assert!((same.modes.curvatures[0] * mass - k).abs() < 1e-8);
        let off = seam_analysis(
            &sys,
            [-root, root],
            [&g1, &g2],
            [&h1, &h2],
            &plane,
            given,
            false,
        )
        .unwrap();
        assert!((off.modes.curvatures[0] * mass - k).abs() > 0.1);
        // What is refused.
        let e = [0.0, 0.0];
        assert!(seam_analysis(&sys, e, [&g1, &g1], [&h1, &h2], &plane, given, false).is_err());
        assert!(seam_analysis(&sys, e, [&g1, &g2], [&h1, &h2], &[], given, false).is_err());
        let along = vec![plane[0].clone(), plane[0].clone()];
        assert!(seam_analysis(&sys, e, [&g1, &g2], [&h1, &h2], &along, given, false).is_err());
    }
}
