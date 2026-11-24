//! Local vibrational modes of internal coordinates (Konkoli and Cremer, Int. J. Quantum
//! Chem. 1998, 67, 1; Zou, Kalescky, Kraka, Cremer, J. Chem. Phys. 2012, 137, 084114).
//!
//! The local mode of an internal coordinate `q_n` is the motion that follows when `q_n` is
//! changed and all other atoms relax: the vibration of that coordinate decoupled from the
//! rest of the molecule. With the normal modes `l_m` (force constants `K_m`) and the row
//! `b_n` of the Wilson B matrix, `d_nm = b_n l_m`:
//!
//! ```text
//! a_n   = K^-1 d_n^T / (d_n K^-1 d_n^T)          local mode in normal coordinates
//! k_n^a = (d_n K^-1 d_n^T)^-1                     local force constant
//! G_nn  = b_n M^-1 b_n^T                          reciprocal of the local mode mass
//! w_n^a = (k_n^a G_nn)^(1/2)                      local mode frequency
//! ```
//!
//! `d_n K^-1 d_n^T` is the compliance constant of `q_n`, so `k_n^a` is the relaxed force
//! constant: for a bond it is what the effective Hessian of its two atoms gives
//! ([`super::relaxed_stretch`]). Unlike a normal-mode frequency, it measures the strength of
//! one bond (or the stiffness of one angle) and can be compared between molecules; the
//! local force constant of a metal-ligand or C-O stretch is the local-mode form of the
//! Tolman electronic parameter.
//!
//! `S_nm = d_nm^2 k_n^a / K_m` is the share of normal mode `m` in the local mode of `q_n`
//! (the shares of one coordinate add up to 1). For a complete, non-redundant set of
//! `3N - 6` coordinates, the same numbers normalized over the coordinates for each normal
//! mode are the decomposition of the normal modes into local modes.
//!
//! The formulas hold at a stationary point; the definition is that of a minimum. With an
//! imaginary frequency the compliance has a negative part, and the result says so. A
//! vibration without a restoring force makes the compliance infinite and is refused.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::urva::primitive_label;
use super::{curvature_to_hz, transverse_modes};
use crate::coord::internal::Primitive;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_dot;

/// The local mode of one internal coordinate.
#[derive(Debug, Clone)]
pub struct LocalMode {
    /// The coordinate.
    pub primitive: Primitive,
    /// Its name: `R(1-2)`, `A(1-2-3)`, `L(1-2-3)`, `D(1-2-3-4)`.
    pub label: String,
    /// Its value (Bohr or radians).
    pub value: f64,
    /// Local force constant (Hartree/Bohr^2 for a distance, Hartree/rad^2 for an angle);
    /// NaN where the coordinate is not defined.
    pub force_constant: f64,
    /// `G_nn`, the reciprocal of the local mode mass (1/amu for a distance, 1/(amu Bohr^2)
    /// for an angle).
    pub inverse_mass: f64,
    /// Local mode wavenumber (cm^-1; negative for a negative force constant).
    pub wavenumber: f64,
    /// Share of every normal mode in this local mode (`S_nm`; adds up to 1).
    pub composition: Vec<f64>,
}

/// Local modes of chosen internal coordinates.
#[derive(Debug, Clone)]
pub struct LocalModes {
    /// One per coordinate, in the order given.
    pub modes: Vec<LocalMode>,
    /// Wavenumbers of the normal modes (cm^-1), in the order of `composition`.
    pub normal_wavenumbers: Vec<f64>,
    /// Remarks on the validity of the result.
    pub notes: Vec<String>,
}

/// Local modes of the internal coordinates `primitives` of `sys` with the Cartesian Hessian
/// `hessian` (Hartree/Bohr^2). With `project_rigid`, translations and rotations are removed
/// from the normal modes (not for a surface in a fixed frame).
pub fn local_modes(
    sys: &MolecularSystem,
    hessian: &Mat<f64>,
    project_rigid: bool,
    primitives: &[Primitive],
) -> MoptResult<LocalModes> {
    let n = sys.dof();
    // The vibrations alone: mass-weighted unit vectors and curvatures.
    let normal = transverse_modes(sys, hessian, project_rigid, &[])?;
    let masses = sys.masses_amu();
    let largest = normal
        .curvatures
        .iter()
        .fold(0.0_f64, |m, c| m.max(c.abs()));
    if normal.curvatures.is_empty() {
        return Err(MoptError::calc("local modes: no vibration"));
    }
    if let Some(soft) = normal
        .curvatures
        .iter()
        .position(|c| c.abs() <= 1e-10 * largest)
    {
        return Err(MoptError::calc(format!(
            "local modes: the vibration at {:.4} cm^-1 has no restoring force, so that every \
             coordinate it moves has an infinite compliance",
            normal.wavenumbers[soft]
        )));
    }
    let mut notes = Vec::new();
    let imaginary = normal.imaginary_modes();
    if imaginary > 0 {
        notes.push(format!(
            "The structure has {imaginary} imaginary frequencies: local modes are defined for \
             a minimum, and the compliance of a coordinate that takes part in such a mode has \
             a negative contribution."
        ));
    }
    // l_m / |l_m| in mass-weighted form: the displacement of mode m is v_m / sqrt(m).
    let vectors: Vec<Col<f64>> = normal
        .displacements
        .iter()
        .map(|d| {
            let weighted = Col::from_fn(n, |c| d[c] * masses[c / 3].sqrt());
            let length = col_dot(&weighted, &weighted).sqrt();
            Col::from_fn(n, |c| weighted[c] / length)
        })
        .collect();
    let to_wavenumber = curvature_to_hz() / Units::SPEED_OF_LIGHT_M_S * 1e-2;
    let mut modes = Vec::with_capacity(primitives.len());
    for primitive in primitives {
        let (value, row) = primitive.value_and_gradient(sys);
        // b_n M^(-1/2): then d_nm = (b_n M^(-1/2)) . v_m for unit vectors v_m, and K_m is
        // the mass-weighted curvature.
        let weighted = Col::from_fn(n, |c| row[c] / masses[c / 3].sqrt());
        let inverse_mass = col_dot(&weighted, &weighted);
        let label = primitive_label(primitive);
        if !(inverse_mass > 1e-20 && inverse_mass.is_finite()) {
            notes.push(format!("{label} is not defined at this structure."));
            modes.push(LocalMode {
                primitive: primitive.clone(),
                label,
                value,
                force_constant: f64::NAN,
                inverse_mass: f64::NAN,
                wavenumber: f64::NAN,
                composition: vec![f64::NAN; vectors.len()],
            });
            continue;
        }
        let parts: Vec<f64> = vectors
            .iter()
            .zip(&normal.curvatures)
            .map(|(v, curvature)| col_dot(&weighted, v).powi(2) / curvature)
            .collect();
        let compliance: f64 = parts.iter().sum();
        let force_constant = 1.0 / compliance;
        let curvature = force_constant * inverse_mass;
        modes.push(LocalMode {
            primitive: primitive.clone(),
            label,
            value,
            force_constant,
            inverse_mass,
            wavenumber: curvature.signum() * curvature.abs().sqrt() * to_wavenumber,
            composition: parts.iter().map(|p| p * force_constant).collect(),
        });
    }
    Ok(LocalModes {
        modes,
        normal_wavenumbers: normal.wavenumbers,
        notes,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, AtomId, Bohr, Element, SpinCharge};
    use crate::math::linalg::inverse_small;
    use crate::vib::{partial_hessian, relaxed_stretch, PartialKind, DEFAULT_FREE_DIRECTION};

    fn water() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::O, Element::H, Element::H],
            &positions::<Bohr>(&[[0.0, 0.0, 0.22], [0.0, 1.45, -0.9], [0.0, -1.45, -0.9]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    fn coordinates() -> Vec<Primitive> {
        let atom = AtomId::new;
        vec![
            Primitive::Bond([atom(0), atom(1)]),
            Primitive::Bond([atom(0), atom(2)]),
            Primitive::Angle([atom(1), atom(0), atom(2)]),
        ]
    }

    /// The Cartesian Hessian `B^T F B` of a force field that is quadratic in the internal
    /// coordinates, at its minimum.
    fn hessian(sys: &MolecularSystem, force_constants: &Mat<f64>) -> Mat<f64> {
        let rows: Vec<Col<f64>> = coordinates()
            .iter()
            .map(|p| p.value_and_gradient(sys).1)
            .collect();
        Mat::from_fn(9, 9, |i, j| {
            let mut sum = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    sum += rows[a][i] * force_constants[(a, b)] * rows[b][j];
                }
            }
            sum
        })
    }

    #[test]
    fn local_force_constants_are_the_reciprocal_compliances_of_the_force_field() {
        let sys = water();
        // Two stretches coupled to each other and to the bend.
        let f = Mat::from_fn(3, 3, |i, j| {
            [[0.52, -0.06, 0.03], [-0.06, 0.52, 0.03], [0.03, 0.03, 0.16]][i][j]
        });
        let found = local_modes(&sys, &hessian(&sys, &f), true, &coordinates()).unwrap();
        let compliance = inverse_small(&f).unwrap();
        assert!(found.notes.is_empty());
        assert_eq!(found.normal_wavenumbers.len(), 3);
        let to_wavenumber = curvature_to_hz() / Units::SPEED_OF_LIGHT_M_S * 1e-2;
        for (i, mode) in found.modes.iter().enumerate() {
            // For a complete set of coordinates b_n f^+ b_n^T is (F^-1)_nn: the local force
            // constant is not the diagonal force constant when the coordinates are coupled.
            let expected = 1.0 / compliance[(i, i)];
            assert!(
                (mode.force_constant - expected).abs() < 1e-10,
                "{}: {} against {expected}",
                mode.label,
                mode.force_constant
            );
            assert!((mode.force_constant - f[(i, i)]).abs() > 1e-3);
            let wavenumber = (expected * mode.inverse_mass).sqrt() * to_wavenumber;
            assert!((mode.wavenumber - wavenumber).abs() < 1e-8 * wavenumber);
            assert!((mode.composition.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let labels: Vec<&str> = found.modes.iter().map(|m| m.label.as_str()).collect();
        assert_eq!(labels, vec!["R(1-2)", "R(1-3)", "A(2-1-3)"]);
        // The reciprocal mass of a bond is 1 / m_O + 1 / m_H.
        let m = sys.masses_amu();
        assert!((found.modes[0].inverse_mass - (1.0 / m[0] + 1.0 / m[1])).abs() < 1e-12);
        // The two bonds are equivalent, and their local stretch lies between the symmetric
        // and the antisymmetric stretch of the molecule.
        assert!((found.modes[0].wavenumber - found.modes[1].wavenumber).abs() < 1e-8);
        let (low, high) = (found.normal_wavenumbers[1], found.normal_wavenumbers[2]);
        assert!(found.modes[0].wavenumber > low.min(high));
        assert!(found.modes[0].wavenumber < low.max(high));
        // The effective Hessian of the two atoms of a bond gives the same force constant.
        let partial = partial_hessian(
            &sys,
            &hessian(&sys, &f),
            &[0, 1],
            PartialKind::Effective,
            DEFAULT_FREE_DIRECTION,
        )
        .unwrap();
        let (relaxed, _) = relaxed_stretch(&partial).unwrap();
        assert!((relaxed - found.modes[0].force_constant).abs() < 1e-9);
        // Without coupling the local force constants are those of the force field.
        let diagonal = Mat::from_fn(3, 3, |i, j| if i == j { f[(i, i)] } else { 0.0 });
        let plain = local_modes(&sys, &hessian(&sys, &diagonal), true, &coordinates()).unwrap();
        for (i, mode) in plain.modes.iter().enumerate() {
            assert!((mode.force_constant - f[(i, i)]).abs() < 1e-10);
        }
    }

    #[test]
    fn what_is_not_a_minimum_is_said_or_refused() {
        let sys = water();
        // A bend without a restoring force.
        let flat = Mat::from_fn(3, 3, |i, j| if i == j && i < 2 { 0.5 } else { 0.0 });
        assert!(local_modes(&sys, &hessian(&sys, &flat), true, &coordinates()).is_err());
        // A bend with a negative force constant: a note, and a negative local force
        // constant for the angle.
        let saddle = Mat::from_fn(3, 3, |i, j| if i == j { [0.5, 0.5, -0.1][i] } else { 0.0 });
        let found = local_modes(&sys, &hessian(&sys, &saddle), true, &coordinates()).unwrap();
        assert_eq!(found.notes.len(), 1);
        assert!(found.modes[2].force_constant < 0.0 && found.modes[2].wavenumber < 0.0);
        assert!((found.modes[0].force_constant - 0.5).abs() < 1e-10);
    }
}
