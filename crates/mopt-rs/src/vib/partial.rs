//! Partial Hessian analysis: the vibrations of chosen atoms.
//!
//! With the atoms `A` chosen and `B` the rest, the Hessian of `A` alone is taken in one of
//! three ways and analysed as the Hessian of the molecule made of the atoms `A`:
//!
//! - [`PartialKind::Block`]: the block `H_AA` (the atoms `B` stay where they are), with the
//!   translations and rotations of `A` projected out;
//! - [`PartialKind::Effective`]: the effective Hessian `H_AA - H_AB H_BB^+ H_BA` (the Schur
//!   complement: the atoms `B` follow `A` so that no force acts on them), with the
//!   translations and rotations of `A` projected out;
//! - [`PartialKind::Frozen`]: the block `H_AA` without a projection (the atoms `B` are a
//!   fixed frame; the usual partial Hessian vibrational analysis).
//!
//! `H_BB^+` is the inverse of `H_BB` on the space in which the atoms `B` have a restoring
//! force. With one atom chosen, or atoms on a line, the rest can still turn about them as a
//! rigid body; these rotations are projected out of `H_BB` (in a computed Hessian their
//! curvature is not zero but as small as the residual gradient allows, so a threshold alone
//! would not find them). Directions left with an eigenvalue below a relative threshold
//! (exact zeros, as in model Hessians) are left out as well.
//!
//! The effective Hessian has the translations and rotations of `A` as exact zero modes when
//! the full Hessian has those of the whole molecule. For two atoms a single vibration is
//! left, and its force constant is the relaxed force constant of the distance between them
//! ([`relaxed_stretch`]): the reciprocal of the compliance constant, which is the force
//! constant of the local stretching mode of Konkoli and Cremer (Int. J. Quantum Chem. 1998,
//! 67, 1), as used for local-mode measures of bond strength such as the local Tolman
//! electronic parameter.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{orthonormalize_columns, symmetric_eigen, symmetrize};
use crate::math::projection::project_out_basis_mat;

/// How the Hessian of the chosen atoms is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialKind {
    /// The block of the chosen atoms; their translations and rotations are projected out.
    Block,
    /// The effective Hessian with the other atoms relaxed; translations and rotations of
    /// the chosen atoms are projected out.
    Effective,
    /// The block of the chosen atoms in the fixed frame of the others; nothing is projected.
    Frozen,
}

impl PartialKind {
    /// From its name: `block`, `effective`, or `frozen`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "block" => Ok(Self::Block),
            "effective" => Ok(Self::Effective),
            "frozen" => Ok(Self::Frozen),
            other => Err(MoptError::config(format!(
                "partial Hessian is block, effective, or frozen, got '{other}'"
            ))),
        }
    }

    /// Name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Effective => "effective",
            Self::Frozen => "frozen",
        }
    }

    /// Whether the translations and rotations of the chosen atoms are projected out.
    pub fn projects_rigid(self) -> bool {
        !matches!(self, Self::Frozen)
    }
}

/// Relative size of an eigenvalue of `H_BB` (after the free rotations are projected out)
/// below which its direction is left out of the effective Hessian: a guard against exact
/// zeros. See `docs/source/migration_vibrations.md` ("Partial Hessian analysis") for the eigenvalues
/// met in a computed Hessian.
pub const DEFAULT_FREE_DIRECTION: f64 = 1e-10;

/// Rigid rotations of the atoms `outside` about the atoms `inside` that leave those in
/// place, as vectors over the coordinates of `outside`: about each axis through a single
/// atom, about the line through atoms on a line, none otherwise (and none in a cell).
fn free_rotations(sys: &MolecularSystem, inside: &[usize], outside: &[usize]) -> Vec<Col<f64>> {
    if sys.has_pbc() || outside.is_empty() {
        return Vec::new();
    }
    let x = sys.coords_flat_bohr();
    let at = |a: usize| -> [f64; 3] { std::array::from_fn(|c| x[3 * a + c]) };
    let origin = at(inside[0]);
    let from_origin = |a: usize| -> [f64; 3] {
        let r = at(a);
        std::array::from_fn(|c| r[c] - origin[c])
    };
    let cross = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let norm = |u: [f64; 3]| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    // The atom of `inside` farthest from the first gives the line, if all are on one.
    let far = inside
        .iter()
        .map(|a| from_origin(*a))
        .fold(
            [0.0; 3],
            |best, d| if norm(d) > norm(best) { d } else { best },
        );
    let reach = norm(far);
    let axes: Vec<[f64; 3]> = if reach < 1e-8 {
        vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    } else {
        let on_line = inside
            .iter()
            .all(|a| norm(cross(far, from_origin(*a))) < 1e-6 * reach * reach);
        if !on_line {
            return Vec::new();
        }
        vec![std::array::from_fn(|c| far[c] / reach)]
    };
    axes.into_iter()
        .map(|axis| {
            let moved: Vec<[f64; 3]> = outside
                .iter()
                .map(|a| cross(axis, from_origin(*a)))
                .collect();
            Col::from_fn(3 * outside.len(), |i| moved[i / 3][i % 3])
        })
        .collect()
}

/// The Hessian of chosen atoms.
#[derive(Debug, Clone)]
pub struct PartialHessian {
    /// How it was taken.
    pub kind: PartialKind,
    /// The chosen atoms (indices into the full structure, starting at 0), in the order
    /// given.
    pub atoms: Vec<usize>,
    /// The chosen atoms as a structure of their own.
    pub system: MolecularSystem,
    /// Their Hessian (Hartree/Bohr^2).
    pub hessian: Mat<f64>,
    /// Number of directions of the other atoms left out of an effective Hessian.
    pub free_directions: usize,
}

/// The Hessian of the atoms `atoms` (indices starting at 0) of `sys` from its Cartesian
/// Hessian `hessian` (Hartree/Bohr^2). `free_direction` is the relative threshold on the
/// eigenvalues of the block of the other atoms ([`DEFAULT_FREE_DIRECTION`]).
pub fn partial_hessian(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    atoms: &[usize],
    kind: PartialKind,
    free_direction: f64,
) -> MoptResult<PartialHessian> {
    let total = sys.n_atoms();
    if hessian.nrows() != 3 * total || hessian.ncols() != 3 * total {
        return Err(MoptError::config(format!(
            "partial Hessian: the Hessian is {}x{}, expected {n}x{n}",
            hessian.nrows(),
            hessian.ncols(),
            n = 3 * total
        )));
    }
    if atoms.is_empty() {
        return Err(MoptError::config("partial Hessian: no atoms chosen"));
    }
    let mut chosen = vec![false; total];
    for &a in atoms {
        if a >= total {
            return Err(MoptError::config(format!(
                "partial Hessian: atom {} of {total}",
                a + 1
            )));
        }
        if std::mem::replace(&mut chosen[a], true) {
            return Err(MoptError::config(format!(
                "partial Hessian: atom {} is chosen twice",
                a + 1
            )));
        }
    }
    let inside: Vec<usize> = atoms
        .iter()
        .flat_map(|a| (0..3).map(move |c| 3 * a + c))
        .collect();
    let other_atoms: Vec<usize> = (0..total).filter(|a| !chosen[*a]).collect();
    let outside: Vec<usize> = other_atoms
        .iter()
        .flat_map(|a| (0..3).map(move |c| 3 * a + c))
        .collect();
    let (k, rest) = (inside.len(), outside.len());
    let mut block = Mat::from_fn(k, k, |i, j| hessian[(inside[i], inside[j])]);
    let mut free_directions = 0;
    if kind == PartialKind::Effective && rest > 0 {
        let mut others = Mat::from_fn(rest, rest, |i, j| hessian[(outside[i], outside[j])]);
        symmetrize(&mut others);
        let rotations =
            orthonormalize_columns(rest, &free_rotations(sys, atoms, &other_atoms), 1e-8);
        let others = project_out_basis_mat(&rotations, &others);
        let coupling = Mat::from_fn(rest, k, |i, j| hessian[(outside[i], inside[j])]);
        let (values, vectors) = symmetric_eigen(&others)?;
        let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        // H_AB H_BB^+ H_BA = sum_m (V_m^T H_BA)^T (V_m^T H_BA) / lambda_m.
        let projected = vectors.transpose() * &coupling;
        for mode in 0..rest {
            if values[mode].abs() <= free_direction * largest {
                free_directions += 1;
                continue;
            }
            for i in 0..k {
                for j in 0..k {
                    block[(i, j)] -= projected[(mode, i)] * projected[(mode, j)] / values[mode];
                }
            }
        }
    }
    symmetrize(&mut block);
    let x = sys.coords_flat_bohr();
    let flat = Col::from_fn(k, |i| x[inside[i]]);
    let elements = atoms.iter().map(|a| sys.elements[*a]).collect();
    let system =
        MolecularSystem::from_flat_bohr(elements, &flat, sys.spin_charge, sys.cell.clone())?;
    Ok(PartialHessian {
        kind,
        atoms: atoms.to_vec(),
        system,
        hessian: block,
        free_directions,
    })
}

/// For two chosen atoms: the force constant of their distance in the partial Hessian
/// (Hartree/Bohr^2 and mdyne/Angstrom), `v^T H v` for the two atoms moving apart along
/// their axis by half a unit each. With an effective Hessian this is the relaxed (local
/// mode) stretching force constant. `None` for another number of atoms, and for a block
/// Hessian, which is not invariant to translations so that the value would depend on how
/// the change of the distance is divided between the atoms.
pub fn relaxed_stretch(partial: &PartialHessian) -> Option<(f64, f64)> {
    if partial.kind != PartialKind::Effective || partial.atoms.len() != 2 {
        return None;
    }
    let x = partial.system.coords_flat_bohr();
    let d: [f64; 3] = std::array::from_fn(|c| x[3 + c] - x[c]);
    let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if length.is_nan() || length <= 0.0 {
        return None;
    }
    let v = Col::from_fn(6, |i| {
        let sign = if i < 3 { -0.5 } else { 0.5 };
        sign * d[i % 3] / length
    });
    let applied = &partial.hessian * &v;
    let constant: f64 = (0..6).map(|i| v[i] * applied[i]).sum();
    // Hartree/Bohr^2 in N/m, and 1 mdyne/Angstrom = 100 N/m.
    Some((constant, constant * Units::HARTREE_BOHR2_TO_MDYNE_ANGSTROM))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};
    use crate::vib::{curvature_to_hz, normal_modes};

    /// Three atoms on a line with the springs `k1` (1-2) and `k2` (2-3) along it.
    fn chain(k1: f64, k2: f64) -> (MolecularSystem, Mat<f64>) {
        let sys = MolecularSystem::new(
            vec![Element::O, Element::C, Element::O],
            &positions::<Bohr>(&[[0.0, 0.0, -2.2], [0.0, 0.0, 0.0], [0.0, 0.0, 2.2]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let mut h = Mat::<f64>::zeros(9, 9);
        for (a, b, k) in [(0, 1, k1), (1, 2, k2)] {
            let (i, j) = (3 * a + 2, 3 * b + 2);
            h[(i, i)] += k;
            h[(j, j)] += k;
            h[(i, j)] -= k;
            h[(j, i)] -= k;
        }
        (sys, h)
    }

    #[test]
    fn the_effective_hessian_of_the_ends_of_a_chain_is_that_of_springs_in_series() {
        let (k1, k2) = (0.9, 0.6);
        let (sys, h) = chain(k1, k2);
        let partial = partial_hessian(
            &sys,
            &h,
            &[0, 2],
            PartialKind::Effective,
            DEFAULT_FREE_DIRECTION,
        )
        .unwrap();
        // The middle atom is free across the line: two directions without a restoring force.
        assert_eq!(partial.free_directions, 2);
        assert_eq!(partial.system.n_atoms(), 2);
        let series = k1 * k2 / (k1 + k2);
        let (constant, mdyne) = relaxed_stretch(&partial).unwrap();
        assert!((constant - series).abs() < 1e-12, "{constant}");
        // 1 Hartree/Bohr^2 = 15.5689 mdyne/Angstrom.
        assert!((mdyne - series * 15.5689).abs() < 1e-4);
        // One vibration of the two atoms with that force constant and their reduced mass.
        let modes = normal_modes(&partial.system, &partial.hessian, true).unwrap();
        assert_eq!(modes.vibrations(1.0, true).len(), 1);
        let mass = sys.masses_amu()[0];
        let expected =
            (series / (0.5 * mass)).sqrt() * curvature_to_hz() / Units::SPEED_OF_LIGHT_M_S * 1e-2;
        let found = modes.wavenumbers[modes.vibrations(1.0, true)[0]];
        assert!((found - expected).abs() < 1e-8 * expected, "{found}");
    }

    #[test]
    fn blocks_keep_the_other_atoms_in_place() {
        let (k1, k2) = (0.9, 0.6);
        let (sys, h) = chain(k1, k2);
        // The middle atom alone between its fixed neighbours: both springs act on it.
        let frozen = partial_hessian(&sys, &h, &[1], PartialKind::Frozen, 1e-6).unwrap();
        assert!((frozen.hessian[(2, 2)] - (k1 + k2)).abs() < 1e-14);
        let modes = normal_modes(&frozen.system, &frozen.hessian, false).unwrap();
        let mass = sys.masses_amu()[1];
        let expected =
            ((k1 + k2) / mass).sqrt() * curvature_to_hz() / Units::SPEED_OF_LIGHT_M_S * 1e-2;
        assert!((modes.wavenumbers[2] - expected).abs() < 1e-8 * expected);
        // The block of two neighbours in the order given, and no relaxed constant for it.
        let block = partial_hessian(&sys, &h, &[1, 0], PartialKind::Block, 1e-6).unwrap();
        assert_eq!(block.atoms, vec![1, 0]);
        assert!((block.hessian[(2, 2)] - (k1 + k2)).abs() < 1e-14);
        assert!((block.hessian[(5, 5)] - k1).abs() < 1e-14);
        assert!((block.hessian[(2, 5)] + k1).abs() < 1e-14);
        assert!(relaxed_stretch(&block).is_none());
        assert_eq!(block.free_directions, 0);
        // All atoms: the Hessian itself, in either way.
        for kind in [PartialKind::Block, PartialKind::Effective] {
            let all = partial_hessian(&sys, &h, &[0, 1, 2], kind, 1e-6).unwrap();
            assert!((0..9).all(|i| (0..9).all(|j| all.hessian[(i, j)] == h[(i, j)])));
        }
        // What is refused.
        assert!(partial_hessian(&sys, &h, &[], PartialKind::Block, 1e-6).is_err());
        assert!(partial_hessian(&sys, &h, &[0, 0], PartialKind::Block, 1e-6).is_err());
        assert!(partial_hessian(&sys, &h, &[3], PartialKind::Block, 1e-6).is_err());
        assert!(PartialKind::parse("Effective").is_ok() && PartialKind::parse("half").is_err());
    }
}
