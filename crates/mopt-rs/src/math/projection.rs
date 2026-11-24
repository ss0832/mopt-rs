//! Rigid-body (translation / rotation) mode basis and projectors.
//!
//! Isolated systems carry 3 translations and 2 (linear) or 3 rotations. Whether a structure is
//! linear is decided in one place, [`shape`] (see [`LINEAR_MOMENT_RATIO`]); the projection,
//! the rotor of the thermochemistry, the count of vibrations (`3N - 5` or `3N - 6`) and the
//! point group (`D∞h` / `C∞v`) all take that answer.
//!
//! Periodic systems (a cell with at least one periodic axis) carry all three translations: a
//! rigid translation along a periodic axis maps the crystal onto itself, and one along a
//! non-periodic axis (the normal of a slab, the directions across a wire) moves the whole
//! structure in its vacuum. A wire (one periodic axis) also carries the rotation about its
//! axis, which maps the periodic images onto each other; the rotations of a slab or of a
//! crystal turn the lattice and are no symmetries of the periodic structure. Whether the
//! surface is invariant under these motions at all (no frozen atoms, no frame-fixing
//! constraints, no frame-dependent terms) is the caller's question
//! (`pes::term::frame_invariant`, `ConstraintSet::breaks_rigid_body`): this module only says
//! which motions are rigid for a structure.
//!
//! MultiOptPy: `Utils/calc_tools.py:Calculationtools.project_out_hess_tr_and_rot`,
//!  `project_out_hess_tr_and_rot_for_coord`, `project_optional_vector_for_grad`,
//!  `project_optional_vector_for_hess`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::AtomId;
use faer::{Col, Mat};

use crate::domain::{MolecularSystem, PeriodicCell};
use crate::math::linalg::{orthonormalize_columns, symmetric_eigen, symmetrize};

/// A structure is linear when the smallest principal moment of its atoms (unit weights, about
/// their centroid) is at most this fraction of the largest.
///
/// With unit weights the smallest moment of a nearly linear structure is `sum_i d_i^2` (the
/// squared distances of the atoms from their best line) and the largest is `sum_i z_i^2`
/// (their spread along it), so the test reads `d_rms <= 1e-3 z_rms`. Derivation: an optimized
/// structure is accurate to its displacement criterion, an RMS displacement of `1e-3` Bohr
/// (`opt::convergence::ConvergenceCriteria::NORMAL`); the smallest linear structures with three
/// atoms or more have `z_rms` of about 1.4 Bohr (linear H3) to 2 Bohr (CO2, HCN), so a linear
/// structure that carries that noise has a ratio below `(1e-3 / 1.4)^2 = 5e-7`. A symmetric
/// triatomic bent by `alpha` at its centre has the ratio `alpha^2 / 12`: the threshold `1e-6`
/// is a bend of `3.5e-3` rad (0.2 degrees), and a bend of one degree (`2.5e-5`) is far above
/// it. Structures with a real bend of a degree or more are therefore never linear, and the
/// noise of a converged linear structure is. The test does not depend on the masses, so the
/// answer is the same for every choice of masses (`vib::MassChoice`). Measurements:
/// `docs/source/vibrations.md`.
pub const LINEAR_MOMENT_RATIO: f64 = 1e-6;

/// The rigid-body shape of a structure: what its rotations are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// One atom, or atoms that all coincide: no rotations.
    Point,
    /// Atoms on a line (within [`LINEAR_MOMENT_RATIO`]) along the unit vector `axis`: two
    /// rotations.
    Linear {
        /// Direction of the line.
        axis: [f64; 3],
    },
    /// Any other structure: three rotations.
    Nonlinear,
}

/// Principal moments (ascending) and axes (columns) of the atoms of `sys` with unit weights
/// about their centroid; `None` for no atoms or coordinates that are not finite.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn geometric_moments(sys: &MolecularSystem) -> Option<([f64; 3], [[f64; 3]; 3])> {
    let n = sys.n_atoms();
    if n == 0 {
        return None;
    }
    let points: Vec<[f64; 3]> = sys.atoms().map(|a| sys.position(a).value()).collect();
    let centre: [f64; 3] =
        std::array::from_fn(|c| points.iter().map(|p| p[c]).sum::<f64>() / n as f64);
    let mut inertia = Mat::<f64>::zeros(3, 3);
    for p in &points {
        let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
        let square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for i in 0..3 {
            inertia[(i, i)] += square;
            for j in 0..3 {
                inertia[(i, j)] -= d[i] * d[j];
            }
        }
    }
    if (0..3).any(|i| (0..3).any(|j| !inertia[(i, j)].is_finite())) {
        return None;
    }
    let (values, vectors) = symmetric_eigen(&inertia).ok()?;
    Some((
        std::array::from_fn(|k| values[k]),
        std::array::from_fn(|k| std::array::from_fn(|c| vectors[(c, k)])),
    ))
}

/// The rigid-body shape of `sys` as an isolated structure (the one linearity test of the
/// crate, [`LINEAR_MOMENT_RATIO`]).
///
/// MultiOptPy: `Thermo/normal_mode_analyzer.py:MolecularVibrations._get_rotor_type` (a
/// rotational constant above a threshold); `Utils/calc_tools.py:Calculationtools.
/// project_out_hess_tr_and_rot` (a vector norm); here one relative test for both.
pub fn shape(sys: &MolecularSystem) -> Shape {
    if sys.n_atoms() <= 1 {
        return Shape::Point;
    }
    let Some((moments, axes)) = geometric_moments(sys) else {
        return Shape::Nonlinear;
    };
    if moments[2] <= 0.0 {
        Shape::Point
    } else if moments[0] <= LINEAR_MOMENT_RATIO * moments[2] {
        Shape::Linear { axis: axes[0] }
    } else {
        Shape::Nonlinear
    }
}

/// Whether `sys` is linear ([`shape`]).
pub fn is_linear(sys: &MolecularSystem) -> bool {
    matches!(shape(sys), Shape::Linear { .. })
}

/// Orthonormal `3N x k` basis `D` of the rigid-body modes of `sys`.
///
/// With `mass_weighted`, modes are weighted by `sqrt(m_i)` with the standard atomic weights
/// (for mass-weighted gradients, vibrational analysis, and IRC; other masses:
/// [`build_tr_rot_basis_with_masses`]).
pub fn build_tr_rot_basis(sys: &MolecularSystem, mass_weighted: bool) -> Mat<f64> {
    let weights: Vec<f64> = sys
        .elements
        .iter()
        .map(|e| {
            if mass_weighted {
                libm::sqrt(e.mass_amu().max(1e-6))
            } else {
                1.0
            }
        })
        .collect();
    rigid_basis(sys, &weights)
}

/// [`build_tr_rot_basis`] mass-weighted with the masses `masses_amu` of the atoms (amu; for
/// instance those of the most abundant isotopes). The rotations are about the centre of mass of
/// the structure, whose masses do not change the space the modes span.
pub fn build_tr_rot_basis_with_masses(sys: &MolecularSystem, masses_amu: &[f64]) -> Mat<f64> {
    let weights: Vec<f64> = (0..sys.n_atoms())
        .map(|i| libm::sqrt(masses_amu.get(i).copied().unwrap_or(1.0).max(1e-6)))
        .collect();
    rigid_basis(sys, &weights)
}

/// The rigid-body modes of `sys` with the weights `weights` of the atoms.
fn rigid_basis(sys: &MolecularSystem, weights: &[f64]) -> Mat<f64> {
    let n = sys.n_atoms();
    let dof = 3 * n;
    if n == 0 {
        return Mat::zeros(0, 0);
    }
    if let Some(cell) = sys.active_cell() {
        return periodic_modes(sys, cell, weights);
    }

    let mut cols: Vec<Col<f64>> = Vec::with_capacity(6);
    cols.extend(cartesian_translations(weights));
    match shape(sys) {
        Shape::Point => {}
        Shape::Linear { axis } => {
            // The two rotations about axes across the line (the rotation about the line
            // itself moves the atoms only by their noise off the line).
            let (u, v) = perpendicular_pair(axis);
            cols.push(rotation(sys, weights, u));
            cols.push(rotation(sys, weights, v));
        }
        Shape::Nonlinear => {
            for axis in 0..3 {
                let e: [f64; 3] = std::array::from_fn(|c| f64::from(u8::from(c == axis)));
                cols.push(rotation(sys, weights, e));
            }
        }
    }
    orthonormalize_columns(dof, &cols, 1e-7)
}

/// The three Cartesian translations with the weights `weights`.
fn cartesian_translations(weights: &[f64]) -> Vec<Col<f64>> {
    let dof = 3 * weights.len();
    (0..3)
        .map(|c| Col::from_fn(dof, |k| if k % 3 == c { weights[k / 3] } else { 0.0 }))
        .collect()
}

/// The infinitesimal rotation `axis x (r_i - r_com)` about the centre of mass, weighted.
fn rotation(sys: &MolecularSystem, weights: &[f64], axis: [f64; 3]) -> Col<f64> {
    let n = sys.n_atoms();
    let com = sys.center_of_mass().value();
    let mut col = Col::zeros(3 * n);
    for i in 0..n {
        let p = sys.position(AtomId::new(i)).value();
        let r = [p[0] - com[0], p[1] - com[1], p[2] - com[2]];
        let cross = [
            axis[1] * r[2] - axis[2] * r[1],
            axis[2] * r[0] - axis[0] * r[2],
            axis[0] * r[1] - axis[1] * r[0],
        ];
        for c in 0..3 {
            col[3 * i + c] = weights[i] * cross[c];
        }
    }
    col
}

/// Two unit vectors that make an orthonormal frame with the unit vector `axis`.
fn perpendicular_pair(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Start from the Cartesian axis least parallel to `axis`.
    let k = (0..3)
        .min_by(|&a, &b| axis[a].abs().total_cmp(&axis[b].abs()))
        .unwrap_or(0);
    let e: [f64; 3] = std::array::from_fn(|c| f64::from(u8::from(c == k)));
    let along = e[0] * axis[0] + e[1] * axis[1] + e[2] * axis[2];
    let u0: [f64; 3] = std::array::from_fn(|c| e[c] - along * axis[c]);
    let length = libm::sqrt(u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]).max(1e-300);
    let u: [f64; 3] = std::array::from_fn(|c| u0[c] / length);
    let v = [
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
    ];
    (u, v)
}

/// The rigid-body modes of a periodic structure: the translations (along the periodic lattice
/// vectors first, then the Cartesian ones, which complete them to all three directions for a
/// slab or a wire), and for a wire the rotation about its axis.
fn periodic_modes(sys: &MolecularSystem, cell: &PeriodicCell, weights: &[f64]) -> Mat<f64> {
    let dof = 3 * weights.len();
    let periodic: Vec<usize> = (0..3).filter(|&ax| cell.periodic_axes[ax]).collect();
    let mut cols: Vec<Col<f64>> = periodic
        .iter()
        .map(|&ax| {
            let lv = cell.lattice_vector(ax).value();
            let norm = libm::sqrt(lv[0] * lv[0] + lv[1] * lv[1] + lv[2] * lv[2]).max(1e-14);
            Col::from_fn(dof, |k| weights[k / 3] * lv[k % 3] / norm)
        })
        .collect();
    if periodic.len() < 3 {
        cols.extend(cartesian_translations(weights));
    }
    if let [ax] = periodic[..] {
        let lv = cell.lattice_vector(ax).value();
        let norm = libm::sqrt(lv[0] * lv[0] + lv[1] * lv[1] + lv[2] * lv[2]).max(1e-14);
        // About the line through the centre of mass along the wire: the components of the
        // positions along the axis (the only ones a wrap into the cell changes) do not enter.
        cols.push(rotation(
            sys,
            weights,
            [lv[0] / norm, lv[1] / norm, lv[2] / norm],
        ));
    }
    orthonormalize_columns(dof, &cols, 1e-8)
}

/// The rigid-body motions of `sys` (Cartesian displacements, unit weights; those of
/// [`build_tr_rot_basis`]) that leave the constraints with the gradients `rows` (Cartesian,
/// one per scalar constraint) unchanged to first order: the null space of the rigid basis
/// restricted to them, as orthonormal displacements. With frozen atoms (unit rows on their
/// coordinates) these are the rotations about one frozen atom, the rotation about the line
/// through two, and nothing with three atoms off one line; they are zero modes of a
/// rigid-body-invariant surface held by such constraints.
///
/// MultiOptPy: no counterpart (new in mopt-rs; first written for the frozen atoms of the
/// two-structure methods, `cli::ieipmain_methods`).
pub fn rigid_motions_keeping(sys: &MolecularSystem, rows: &[Col<f64>]) -> Vec<Col<f64>> {
    let r = build_tr_rot_basis(sys, false);
    let (n, k) = (r.nrows(), r.ncols());
    if k == 0 {
        return Vec::new();
    }
    // The rows as unit vectors, so that the tolerance does not depend on their scale.
    let units: Vec<Col<f64>> = rows
        .iter()
        .filter(|row| row.nrows() == n)
        .filter_map(|row| {
            let length = crate::math::linalg::col_norm(row);
            (length > 0.0 && length.is_finite()).then(|| Col::from_fn(n, |c| row[c] / length))
        })
        .collect();
    if units.is_empty() {
        return (0..k).map(|j| Col::from_fn(n, |i| r[(i, j)])).collect();
    }
    let a = Mat::from_fn(units.len(), k, |i, j| {
        (0..n).map(|c| units[i][c] * r[(c, j)]).sum::<f64>()
    });
    let Ok((w, v)) = symmetric_eigen(&(a.transpose() * &a)) else {
        return Vec::new();
    };
    let largest = (0..k).fold(0.0_f64, |m, i| m.max(w[i].abs())).max(1.0);
    (0..k)
        .filter(|&i| w[i].abs() < 1e-10 * largest)
        .map(|i| &r * Col::from_fn(k, |j| v[(j, i)]))
        .collect()
}

/// Apply `P = I - D D^T` to a vector, where `D` is an orthonormal basis.
pub fn project_out_basis_vec(d: &Mat<f64>, v: &Col<f64>) -> Col<f64> {
    if d.ncols() == 0 {
        return v.clone();
    }
    let coeff = d.transpose() * v;
    v - d * coeff
}

/// Apply `P H P`, `P = I - D D^T`, to a symmetric matrix and re-symmetrize.
pub fn project_out_basis_mat(d: &Mat<f64>, h: &Mat<f64>) -> Mat<f64> {
    if d.ncols() == 0 {
        return h.clone();
    }
    let m = h - d * (d.transpose() * h);
    let mut out = &m - (&m * d) * d.transpose();
    symmetrize(&mut out);
    out
}

/// Remove rigid-body components from a `3N` vector.
pub fn project_out_tr_rot_vec(
    sys: &MolecularSystem,
    v: &Col<f64>,
    mass_weighted: bool,
) -> Col<f64> {
    project_out_basis_vec(&build_tr_rot_basis(sys, mass_weighted), v)
}

/// Remove rigid-body components from a `3N x 3N` symmetric matrix.
pub fn project_out_tr_rot_hessian(
    sys: &MolecularSystem,
    hess: &Mat<f64>,
    mass_weighted: bool,
) -> Mat<f64> {
    project_out_basis_mat(&build_tr_rot_basis(sys, mass_weighted), hess)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::quantity::shorthand::angstrom;
    use crate::domain::{positions, Angstrom};
    use crate::domain::{Element, SpinCharge};
    use crate::math::linalg::col_dot;

    #[test]
    fn rigid_motions_keeping_fixed_variables_are_those_of_the_former_ieip_copy() {
        // The copy `cli::ieipmain_methods` had: the rows of the rigid basis at the fixed
        // variables.
        let former = |sys: &MolecularSystem, frozen: &[usize]| -> Vec<Col<f64>> {
            let r = build_tr_rot_basis(sys, false);
            let k = r.ncols();
            let a = Mat::from_fn(frozen.len(), k, |i, j| r[(frozen[i], j)]);
            let (w, v) = symmetric_eigen(&(a.transpose() * &a)).unwrap();
            let largest = (0..k).fold(0.0_f64, |m, i| m.max(w[i].abs())).max(1e-300);
            (0..k)
                .filter(|&i| w[i].abs() < 1e-10 * largest.max(1.0))
                .map(|i| &r * Col::from_fn(k, |j| v[(j, i)]))
                .collect()
        };
        let sys = h2o();
        let n = sys.dof();
        for frozen in [vec![0, 1, 2], vec![0, 1, 2, 3, 4, 5], vec![4], vec![1, 7]] {
            let rows: Vec<Col<f64>> = frozen
                .iter()
                .map(|&f| Col::from_fn(n, |i| if i == f { 1.0 } else { 0.0 }))
                .collect();
            let new = rigid_motions_keeping(&sys, &rows);
            let old = former(&sys, &frozen);
            assert_eq!(new.len(), old.len(), "{frozen:?}");
            for (a, b) in new.iter().zip(&old) {
                for i in 0..n {
                    assert_eq!(a[i].to_bits(), b[i].to_bits(), "{frozen:?} {i}");
                }
            }
        }
    }
    fn h2o() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::O, Element::H, Element::H],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn mode_counts_isolated_linear_and_periodic() {
        assert_eq!(build_tr_rot_basis(&h2o(), true).ncols(), 6);
        let co2 = MolecularSystem::new(
            vec![Element::O, Element::C, Element::O],
            &positions::<Angstrom>(&[[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        assert_eq!(build_tr_rot_basis(&co2, true).ncols(), 5);
        let mut pbc = h2o();
        pbc.cell = Some(
            PeriodicCell::orthorhombic(angstrom(10.0), angstrom(10.0), angstrom(10.0)).unwrap(),
        );
        assert_eq!(build_tr_rot_basis(&pbc, true).ncols(), 3);
        // A slab: all three translations (the one along its normal moves it in its vacuum).
        pbc.cell = Some(PeriodicCell::parse_cli_spec("10,10,10:TTF").unwrap());
        assert_eq!(build_tr_rot_basis(&pbc, false).ncols(), 3);
        // A wire: the translations and the rotation about its axis.
        pbc.cell = Some(PeriodicCell::parse_cli_spec("10,10,10:TFF").unwrap());
        assert_eq!(build_tr_rot_basis(&pbc, true).ncols(), 4);
        assert!(is_linear(&co2) && !is_linear(&h2o()));
    }

    #[test]
    fn projected_vector_is_orthogonal_to_modes() {
        let sys = h2o();
        let d = build_tr_rot_basis(&sys, false);
        let v = Col::from_fn(9, |k| (k as f64 * 0.37).sin());
        let p = project_out_basis_vec(&d, &v);
        for c in 0..d.ncols() {
            let dc = Col::from_fn(9, |r| d[(r, c)]);
            assert!(col_dot(&dc, &p).abs() < 1e-12);
        }
        let h = Mat::from_fn(9, 9, |i, j| 1.0 / (1.0 + i as f64 + j as f64));
        let hp = project_out_tr_rot_hessian(&sys, &h, false);
        let hd = &hp * &d;
        assert!((0..9).all(|i| (0..d.ncols()).all(|c| hd[(i, c)].abs() < 1e-12)));
    }
}
