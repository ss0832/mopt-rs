//! Normal modes of a molecule from its Cartesian Hessian.
//!
//! The Hessian is mass-weighted, `M^(-1/2) H M^(-1/2)` (masses in amu), translations and
//! rotations are projected out, and the eigenvalues `k` give the harmonic wavenumbers
//! `sqrt(k) / (2 pi c)`. A negative eigenvalue is an imaginary frequency and is reported as a
//! negative wavenumber. The Cartesian displacements of a mode are the eigenvector divided by
//! `sqrt(m)`; the reduced mass is the inverse of their squared norm.
//!
//! Further directions can be projected out together with the rigid-body motions
//! ([`normal_modes_without`]): with the gradient at a point of a reaction path, the remaining
//! modes are the generalized normal modes orthogonal to the path (Miller, Handy, Adams, J.
//! Chem. Phys. 1980, 72, 99), whose frequencies variational transition-state theory needs.
//! In mass-weighted coordinates a gradient `g` points along `M^(-1/2) g` and a displacement
//! `dx` along `M^(1/2) dx`.
//!
//! MultiOptPy: `Thermo/normal_mode_analyzer.py:MolecularVibrations.analyze_normal_modes`,
//!  `Utils/calc_tools.py:Calculationtools.project_out_hess_tr_and_rot`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::curvature_to_hz;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::orthonormalize_columns;
use crate::math::linalg::{symmetric_eigen, symmetrize};
use crate::math::projection::{build_tr_rot_basis_with_masses, project_out_basis_mat};

/// Normal modes, in the order of the eigenvalues (imaginary frequencies first; the modes of
/// the projected translations and rotations have wavenumbers of zero).
#[derive(Debug, Clone)]
pub struct NormalModes {
    /// Wavenumbers in cm^-1; negative for imaginary frequencies.
    pub wavenumbers: Vec<f64>,
    /// Eigenvalues of the mass-weighted Hessian (Hartree / (Bohr^2 amu)).
    pub curvatures: Vec<f64>,
    /// Cartesian displacements of each mode (eigenvector over `sqrt(m)`, length `3N`).
    pub displacements: Vec<Col<f64>>,
    /// Reduced masses (amu).
    pub reduced_masses: Vec<f64>,
    /// Force constants (mdyne / Angstrom).
    pub force_constants: Vec<f64>,
    /// Characteristic temperatures `h nu / k` (K); zero for imaginary frequencies.
    pub temperatures: Vec<f64>,
}

impl NormalModes {
    /// Indices of the vibrations: modes whose wavenumber exceeds `cutoff` (cm^-1) in
    /// magnitude (`imaginary`: with those of imaginary frequency), ascending.
    pub fn vibrations(&self, cutoff: f64, imaginary: bool) -> Vec<usize> {
        (0..self.wavenumbers.len())
            .filter(|&k| {
                let w = self.wavenumbers[k];
                w > cutoff || (imaginary && w < -cutoff)
            })
            .collect()
    }

    /// Number of imaginary modes by the one criterion of the crate: a mass-weighted curvature
    /// below `-opt::mode_kill::IMAGINARY_CURVATURE` (about -0.5 cm^-1). The projected
    /// rigid-body motions and directions have curvatures of zero and are never counted.
    pub fn imaginary_modes(&self) -> usize {
        self.curvatures
            .iter()
            .filter(|k| is_imaginary_curvature(**k))
            .count()
    }
}

/// Whether a mass-weighted curvature (Hartree / (Bohr^2 amu)) is that of an imaginary mode: the
/// one criterion of the crate, `opt::mode_kill::IMAGINARY_CURVATURE`.
pub fn is_imaginary_curvature(curvature: f64) -> bool {
    curvature < -crate::opt::mode_kill::IMAGINARY_CURVATURE
}

/// The number of imaginary modes among wavenumbers (cm^-1, negative for imaginary
/// frequencies) of this module, by [`is_imaginary_curvature`]: the wavenumber of the curvature
/// `-IMAGINARY_CURVATURE` is the bound.
pub fn imaginary_wavenumbers(wavenumbers: &[f64]) -> usize {
    let bound = crate::opt::mode_kill::IMAGINARY_CURVATURE.sqrt() * curvature_to_hz()
        / Units::SPEED_OF_LIGHT_M_S
        * 1e-2;
    wavenumbers.iter().filter(|w| **w < -bound).count()
}

/// A direction to leave out of a vibrational analysis.
#[derive(Debug, Clone)]
pub enum Direction {
    /// Along a Cartesian gradient (Hartree/Bohr): the tangent of the reaction path through
    /// the point.
    Gradient(Col<f64>),
    /// Along a Cartesian displacement (Bohr).
    Displacement(Col<f64>),
}

/// Normal modes of `sys` with the Cartesian Hessian `hessian` (Hartree/Bohr^2). With
/// `project`, translations and rotations are removed (translations only in a periodic cell).
pub fn normal_modes(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    project: bool,
) -> MoptResult<NormalModes> {
    normal_modes_without(sys, hessian, project, &[])
}

/// [`normal_modes`] with `directions` projected out as well; their modes have wavenumbers
/// of zero, like the rigid-body motions.
pub fn normal_modes_without(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    project: bool,
    directions: &[Direction],
) -> MoptResult<NormalModes> {
    normal_modes_with_masses(sys, hessian, project, directions, &sys.masses_amu())
}

/// [`normal_modes_without`] with the masses `masses_amu` of the atoms (amu) instead of the
/// standard atomic weights (`vib::MassChoice`); the rigid-body motions are weighted with the
/// same masses.
pub fn normal_modes_with_masses(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    project: bool,
    directions: &[Direction],
    masses_amu: &[f64],
) -> MoptResult<NormalModes> {
    let n = sys.dof();
    let (mut weighted, removed, inv_sqrt) =
        weighted_problem(sys, hessian, project, directions, masses_amu)?;
    if removed.ncols() > 0 {
        weighted = project_out_basis_mat(&removed, &weighted);
    }
    symmetrize(&mut weighted);
    let (values, vectors) = symmetric_eigen(&weighted)?;
    Ok(assemble(
        (0..n).map(|k| (values[k], Col::from_fn(n, |i| vectors[(i, k)]))),
        &inv_sqrt,
    ))
}

/// Only the modes that are left when the rigid-body motions (with `project`) and
/// `directions` are projected out: the Hessian is diagonalized within the complement of
/// those directions. Unlike in [`normal_modes_without`], a remaining mode whose curvature
/// happens to vanish is not confused with a projected one.
pub fn transverse_modes(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    project: bool,
    directions: &[Direction],
) -> MoptResult<NormalModes> {
    transverse_modes_with_masses(sys, hessian, project, directions, &sys.masses_amu())
}

/// [`transverse_modes`] with the masses `masses_amu` of the atoms (amu).
pub fn transverse_modes_with_masses(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    project: bool,
    directions: &[Direction],
    masses_amu: &[f64],
) -> MoptResult<NormalModes> {
    let n = sys.dof();
    let (weighted, removed, inv_sqrt) =
        weighted_problem(sys, hessian, project, directions, masses_amu)?;
    let taken = removed.ncols();
    // An orthonormal basis of the complement: the unit vectors after the removed directions.
    let mut columns: Vec<Col<f64>> = (0..taken)
        .map(|k| Col::from_fn(n, |i| removed[(i, k)]))
        .collect();
    columns.extend((0..n).map(|k| Col::from_fn(n, |i| f64::from(u8::from(i == k)))));
    let full = orthonormalize_columns(n, &columns, 1e-8);
    let rest = full.ncols() - taken;
    let basis = Mat::from_fn(n, rest, |i, k| full[(i, taken + k)]);
    let mut reduced = basis.transpose() * &weighted * &basis;
    symmetrize(&mut reduced);
    let (values, vectors) = symmetric_eigen(&reduced)?;
    Ok(assemble(
        (0..rest).map(|k| {
            let y = Col::from_fn(rest, |i| vectors[(i, k)]);
            (values[k], &basis * &y)
        }),
        &inv_sqrt,
    ))
}

/// The mass-weighted Hessian, an orthonormal basis of the directions to remove (mass-
/// weighted; columns), and `1 / sqrt(m)` of every coordinate.
fn weighted_problem(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    project: bool,
    directions: &[Direction],
    masses: &[f64],
) -> MoptResult<(Mat<f64>, Mat<f64>, Vec<f64>)> {
    let n = sys.dof();
    if hessian.nrows() != n || hessian.ncols() != n {
        return Err(MoptError::config(format!(
            "normal modes: the Hessian is {}x{}, expected {n}x{n}",
            hessian.nrows(),
            hessian.ncols()
        )));
    }
    if masses.len() != sys.n_atoms() || masses.iter().any(|m| !(*m > 0.0 && m.is_finite())) {
        return Err(MoptError::config(format!(
            "normal modes: {} positive masses expected, got {}",
            sys.n_atoms(),
            masses.len()
        )));
    }
    let inv_sqrt: Vec<f64> = (0..n).map(|k| 1.0 / masses[k / 3].sqrt()).collect();
    let weighted = Mat::from_fn(n, n, |i, j| hessian[(i, j)] * inv_sqrt[i] * inv_sqrt[j]);
    let mut removed: Vec<Col<f64>> = Vec::new();
    if project {
        let rigid = build_tr_rot_basis_with_masses(sys, masses);
        removed.extend((0..rigid.ncols()).map(|k| Col::from_fn(n, |i| rigid[(i, k)])));
    }
    for direction in directions {
        let (vector, gradient) = match direction {
            Direction::Gradient(g) => (g, true),
            Direction::Displacement(dx) => (dx, false),
        };
        if vector.nrows() != n {
            return Err(MoptError::config(format!(
                "normal modes: a direction has {} components, expected {n}",
                vector.nrows()
            )));
        }
        removed.push(Col::from_fn(n, |i| {
            if gradient {
                vector[i] * inv_sqrt[i]
            } else {
                vector[i] / inv_sqrt[i]
            }
        }));
    }
    // Directions that vanish or depend on the others (a gradient of zero at a stationary
    // point) are dropped.
    let basis = orthonormalize_columns(n, &removed, 1e-8);
    Ok((weighted, basis, inv_sqrt))
}

/// Normal modes from curvatures and mass-weighted eigenvectors.
fn assemble(pairs: impl Iterator<Item = (f64, Col<f64>)>, inv_sqrt: &[f64]) -> NormalModes {
    let n = inv_sqrt.len();
    let to_hz = curvature_to_hz();
    let to_mdyne_per_angstrom = Units::HARTREE_BOHR2_TO_MDYNE_ANGSTROM;
    let mut modes = NormalModes {
        wavenumbers: Vec::new(),
        curvatures: Vec::new(),
        displacements: Vec::new(),
        reduced_masses: Vec::new(),
        force_constants: Vec::new(),
        temperatures: Vec::new(),
    };
    for (curvature, vector) in pairs {
        let hz = curvature.abs().sqrt() * to_hz;
        let displacement = Col::from_fn(n, |i| vector[i] * inv_sqrt[i]);
        let reduced = 1.0 / displacement.iter().map(|v| v * v).sum::<f64>();
        modes
            .wavenumbers
            .push(curvature.signum() * hz / Units::SPEED_OF_LIGHT_M_S * 1e-2);
        modes.curvatures.push(curvature);
        modes
            .force_constants
            .push(reduced * curvature * to_mdyne_per_angstrom);
        modes.reduced_masses.push(reduced);
        modes.temperatures.push(if curvature > 0.0 {
            hz * Units::PLANCK_J_S / Units::BOLTZMANN_J_PER_K
        } else {
            0.0
        });
        modes.displacements.push(displacement);
    }
    modes
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};

    #[test]
    fn a_diatomic_molecule_has_the_frequency_of_its_reduced_mass() {
        // H-Cl with a force constant of 0.3 Hartree/Bohr^2 along the bond.
        let el = vec![Element::H, Element::parse_token("Cl").unwrap()];
        let sys = MolecularSystem::new(
            el,
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [0.0, 0.0, 1.275]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let k = 0.3;
        let mut h = Mat::<f64>::zeros(6, 6);
        for (a, b, v) in [(2, 2, k), (5, 5, k), (2, 5, -k), (5, 2, -k)] {
            h[(a, b)] = v;
        }
        let modes = normal_modes(&sys, &h, true).unwrap();
        let m = sys.masses_amu();
        let mu = m[0] * m[1] / (m[0] + m[1]);
        let expected = (k / mu).sqrt() * curvature_to_hz() / Units::SPEED_OF_LIGHT_M_S * 1e-2;
        // Five modes of zero frequency (the molecule is linear) and the stretch.
        assert_eq!(modes.vibrations(1.0, true), vec![5]);
        assert!((modes.wavenumbers[5] - expected).abs() < 1e-8 * expected);
        // The reduced mass of normalized Cartesian displacements (not m1 m2 / M).
        let cartesian = (m[0] + m[1]) * m[0] * m[1] / (m[0] * m[0] + m[1] * m[1]);
        assert!((modes.reduced_masses[5] - cartesian).abs() < 1e-10);
        assert!((modes.temperatures[5] - expected * Units::HC_OVER_KB_CM_K).abs() < 1e-6);
        // A maximum along the bond: an imaginary frequency, shown as negative.
        let modes = normal_modes(&sys, &(&h * -1.0), true).unwrap();
        assert!((modes.wavenumbers[0] + expected).abs() < 1e-8 * expected);
        assert_eq!(modes.imaginary_modes(), 1);
        assert_eq!(modes.temperatures[0], 0.0);
        assert!(normal_modes(&sys, &Mat::zeros(3, 3), true).is_err());
    }

    #[test]
    fn a_projected_direction_leaves_the_modes_orthogonal_to_it() {
        // A bent triatomic with springs along its three distances: three vibrations.
        let sys = MolecularSystem::new(
            vec![Element::O, Element::H, Element::H],
            &positions::<Angstrom>(&[
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let x = sys.coords_flat_bohr();
        let mut h = Mat::<f64>::zeros(9, 9);
        for (i, j, k) in [(0, 1, 0.5), (0, 2, 0.5), (1, 2, 0.1)] {
            let d: Vec<f64> = (0..3).map(|a| x[3 * i + a] - x[3 * j + a]).collect();
            let r2: f64 = d.iter().map(|v| v * v).sum();
            for a in 0..3 {
                for b in 0..3 {
                    let v = k * d[a] * d[b] / r2;
                    h[(3 * i + a, 3 * i + b)] += v;
                    h[(3 * j + a, 3 * j + b)] += v;
                    h[(3 * i + a, 3 * j + b)] -= v;
                    h[(3 * j + a, 3 * i + b)] -= v;
                }
            }
        }
        let all = normal_modes(&sys, &h, true).unwrap();
        assert_eq!(all.vibrations(1.0, true).len(), 3);
        // Without the highest mode (given as a displacement): the other two, unchanged.
        let highest = all.displacements[8].clone();
        let two =
            normal_modes_without(&sys, &h, true, &[Direction::Displacement(highest)]).unwrap();
        let kept = two.vibrations(1.0, true);
        assert_eq!(kept.len(), 2);
        assert!((two.wavenumbers[kept[0]] - all.wavenumbers[6]).abs() < 1e-6);
        assert!((two.wavenumbers[kept[1]] - all.wavenumbers[7]).abs() < 1e-6);
        // A gradient along that mode is M times its displacement: the same projection.
        let masses = sys.masses_amu();
        let gradient = Col::from_fn(9, |i| masses[i / 3] * all.displacements[8][i]);
        let same = normal_modes_without(&sys, &h, true, &[Direction::Gradient(gradient)]).unwrap();
        // The vibrations agree; the other modes are the projected rigid motions and the removed
        // direction, zero up to rounding noise (about 1e-4 cm^-1, which moves with the last bits
        // of the coordinates).
        assert_eq!(same.vibrations(1.0, true), kept);
        for (k, (a, b)) in same.wavenumbers.iter().zip(&two.wavenumbers).enumerate() {
            if kept.contains(&k) {
                assert!((a - b).abs() < 1e-6, "{k}: {a} {b}");
            } else {
                assert!(a.abs() < 1e-3 && b.abs() < 1e-3, "{k}: {a} {b}");
            }
        }
        // A general direction mixes the modes: two vibrations remain, between the old ones.
        let general = Col::from_fn(9, |i| ((i * 7 + 3) as f64).sin());
        let mixed = normal_modes_without(&sys, &h, true, &[Direction::Gradient(general)]).unwrap();
        let kept = mixed.vibrations(1.0, true);
        assert_eq!(kept.len(), 2);
        assert!(mixed.wavenumbers[kept[1]] <= all.wavenumbers[8] + 1e-6);
        assert!(mixed.wavenumbers[kept[0]] >= all.wavenumbers[6] - 1e-6);
        // Only the remaining modes, without the projected ones.
        let direct = transverse_modes(
            &sys,
            &h,
            true,
            &[Direction::Gradient(Col::from_fn(9, |i| {
                ((i * 7 + 3) as f64).sin()
            }))],
        )
        .unwrap();
        assert_eq!(direct.wavenumbers.len(), 2);
        assert!((direct.wavenumbers[0] - mixed.wavenumbers[kept[0]]).abs() < 1e-6);
        assert!((direct.wavenumbers[1] - mixed.wavenumbers[kept[1]]).abs() < 1e-6);
        // A gradient of zero (a stationary point) projects nothing.
        let none =
            normal_modes_without(&sys, &h, true, &[Direction::Gradient(Col::zeros(9))]).unwrap();
        assert_eq!(none.vibrations(1.0, true).len(), 3);
    }
}
