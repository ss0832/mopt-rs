//! Quartic force field in normal coordinates from Hessians at displaced structures.
//!
//! In the dimensionless normal coordinates `q_i = (w_i / hbar)^(1/2) Q_i` of a minimum the
//! potential is expanded as
//!
//! ```text
//! V / hc = 1/2 sum_i w_i q_i^2 + 1/6 sum_ijk phi_ijk q_i q_j q_k
//!        + 1/24 sum_ijkl phi_ijkl q_i q_j q_k q_l
//! ```
//!
//! with the harmonic wavenumbers `w_i` and the force constants `phi` in cm^-1. The structure
//! is displaced along every normal mode by `+-step` in `q_i`, and the Hessians there, written
//! in the normal coordinates of the reference, give by central differences
//!
//! ```text
//! phi_ijk  = (H_jk(+i) - H_jk(-i)) / (2 step)
//! phi_iijk = (H_jk(+i) + H_jk(-i) - 2 H_jk(0)) / step^2
//! ```
//!
//! that is all cubic constants and the quartic ones with two equal indices (the
//! semi-diagonal quartic force field that second-order perturbation theory needs), from
//! `2 M` Hessians for `M` vibrations. A cubic constant is obtained from each of its indices
//! and `phi_iijj` from both modes; the estimates are averaged.
//!
//! [`Qff`] is a state machine: `ask` gives the next displaced structure, `tell` takes its
//! Hessian, so the Hessians can come from any calculator (analytic, finite differences,
//! O1NumHess) and from a journal on a restart.
//!
//! For the vibration-rotation terms the force field carries the Coriolis constants
//! `zeta^a_ij = sum_atoms (l_i x l_j)_a` about the principal axes of inertia and the
//! rotational constants about them.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::normal_modes;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen};

/// Hartree in cm^-1.
pub(crate) fn hartree_in_wavenumbers() -> f64 {
    Units::HARTREE_TO_WAVENUMBER
}

/// Atomic mass unit in electron masses, from the constants of this module
/// (`m_e = hbar^2 / (E_h a_0^2)`).
pub(crate) fn amu_in_electron_masses() -> f64 {
    let hbar = Units::PLANCK_J_S / (2.0 * std::f64::consts::PI);
    Units::AMU_TO_KG * Units::HARTREE_TO_JOULE * Units::BOHR_TO_METER * Units::BOHR_TO_METER
        / (hbar * hbar)
}

/// A quartic force field in dimensionless normal coordinates (cm^-1).
#[derive(Debug, Clone)]
pub struct ForceField {
    /// Harmonic wavenumbers, ascending.
    pub harmonic: Vec<f64>,
    /// `phi_ijk` at `(i M + j) M + k`.
    cubic: Vec<f64>,
    /// `phi_iijk` at `(i M + j) M + k`.
    quartic: Vec<f64>,
    /// Coriolis constants `zeta^a_ij` about the three principal axes (empty matrices
    /// without rotation).
    pub coriolis: [Mat<f64>; 3],
    /// Rotational constants about those axes (cm^-1; infinite about the axis of a linear
    /// molecule, zero without rotation).
    pub rotational: [f64; 3],
}

impl ForceField {
    /// A force field with the harmonic wavenumbers `harmonic` and the listed constants:
    /// `cubic` as `(i, j, k, phi_ijk)` and `quartic` as `(i, j, k, phi_iijk)`, each given
    /// once in any order of its indices. Without rotation.
    pub fn from_constants(
        harmonic: &[f64],
        cubic: &[(usize, usize, usize, f64)],
        quartic: &[(usize, usize, usize, f64)],
    ) -> MoptResult<Self> {
        let m = harmonic.len();
        let mut field = Self {
            harmonic: harmonic.to_vec(),
            cubic: vec![0.0; m * m * m],
            quartic: vec![0.0; m * m * m],
            coriolis: [Mat::zeros(m, m), Mat::zeros(m, m), Mat::zeros(m, m)],
            rotational: [0.0; 3],
        };
        if harmonic.iter().any(|w| !(*w > 0.0 && w.is_finite())) {
            return Err(MoptError::config(
                "a force field needs positive harmonic wavenumbers",
            ));
        }
        for &(i, j, k, value) in cubic {
            if i >= m || j >= m || k >= m {
                return Err(MoptError::config(format!(
                    "cubic constant ({i}, {j}, {k}) for {m} modes"
                )));
            }
            for (a, b, c) in [
                (i, j, k),
                (i, k, j),
                (j, i, k),
                (j, k, i),
                (k, i, j),
                (k, j, i),
            ] {
                field.cubic[(a * m + b) * m + c] = value;
            }
        }
        for &(i, j, k, value) in quartic {
            if i >= m || j >= m || k >= m {
                return Err(MoptError::config(format!(
                    "quartic constant ({i}, {i}, {j}, {k}) for {m} modes"
                )));
            }
            field.quartic[(i * m + j) * m + k] = value;
            field.quartic[(i * m + k) * m + j] = value;
            // phi_iijj is phi_jjii.
            if j == k {
                field.quartic[(j * m + i) * m + i] = value;
            }
        }
        Ok(field)
    }

    /// Number of vibrations.
    pub fn modes(&self) -> usize {
        self.harmonic.len()
    }

    /// `phi_ijk`.
    pub fn cubic(&self, i: usize, j: usize, k: usize) -> f64 {
        let m = self.modes();
        self.cubic[(i * m + j) * m + k]
    }

    /// `phi_iijk`.
    pub fn quartic(&self, i: usize, j: usize, k: usize) -> f64 {
        let m = self.modes();
        self.quartic[(i * m + j) * m + k]
    }
}

/// Settings of the force field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QffSettings {
    /// Displacement along every mode in its dimensionless normal coordinate.
    pub step: f64,
    /// Modes at or below this wavenumber (cm^-1) are not vibrations.
    pub lowest: f64,
}

impl Default for QffSettings {
    fn default() -> Self {
        Self {
            step: DEFAULT_STEP,
            lowest: 10.0,
        }
    }
}

/// Default displacement in the dimensionless normal coordinates; see
/// `docs/source/migration_vibrations.md` ("Anharmonic vibrational analysis") for the measurement.
pub const DEFAULT_STEP: f64 = 0.05;

/// The displaced Hessians of a quartic force field, one request at a time.
#[derive(Debug, Clone)]
pub struct Qff {
    reference: MolecularSystem,
    settings: QffSettings,
    /// Mass-weighted unit vectors of the vibrations.
    vectors: Vec<Col<f64>>,
    /// Harmonic wavenumbers of the vibrations (cm^-1).
    harmonic: Vec<f64>,
    /// `(hbar / w_i)^(1/2)` in `sqrt(amu) Bohr`: the unit of `q_i` in mass-weighted length.
    unit: Vec<f64>,
    /// `1 / sqrt(m)` of every Cartesian coordinate.
    inv_sqrt_mass: Vec<f64>,
    /// Hessian of the reference in the normal coordinates (Hartree / (amu Bohr^2)).
    center: Mat<f64>,
    /// Hessians told so far, in the normal coordinates: `+` and `-` of mode 0, of mode 1, ...
    displaced: Vec<Mat<f64>>,
    rigid: bool,
}

impl Qff {
    /// A force field around the minimum `sys` with the Cartesian Hessian `hessian`
    /// (Hartree/Bohr^2); with `project_rigid`, translations and rotations are projected out
    /// of the modes (not for a surface in a fixed frame).
    pub fn new(
        sys: &MolecularSystem,
        hessian: &Mat<f64>,
        project_rigid: bool,
        settings: QffSettings,
    ) -> MoptResult<Self> {
        if !(settings.step > 0.0 && settings.step.is_finite()) {
            return Err(MoptError::config("force field: the step must be positive"));
        }
        let modes = normal_modes(sys, hessian, project_rigid)?;
        if modes.imaginary_modes() > 0 {
            return Err(MoptError::calc(format!(
                "force field: the structure has {} imaginary frequencies; the anharmonic \
                 analysis needs a minimum",
                modes.imaginary_modes()
            )));
        }
        let chosen = modes.vibrations(settings.lowest, false);
        if chosen.is_empty() {
            return Err(MoptError::calc("force field: no vibration"));
        }
        let n = sys.dof();
        let masses = sys.masses_amu();
        let inv_sqrt_mass: Vec<f64> = (0..n).map(|c| 1.0 / masses[c / 3].sqrt()).collect();
        let vectors: Vec<Col<f64>> = chosen
            .iter()
            .map(|&k| {
                let v = Col::from_fn(n, |c| modes.displacements[k][c] / inv_sqrt_mass[c]);
                let length = col_norm(&v);
                Col::from_fn(n, |c| v[c] / length)
            })
            .collect();
        let amu = amu_in_electron_masses();
        let unit = chosen
            .iter()
            .map(|&k| {
                // w in atomic units; (hbar / w)^(1/2) in sqrt(m_e) Bohr.
                let omega = (modes.curvatures[k] / amu).sqrt();
                1.0 / (omega * amu).sqrt()
            })
            .collect();
        let mut qff = Self {
            reference: sys.clone(),
            settings,
            harmonic: chosen.iter().map(|&k| modes.wavenumbers[k]).collect(),
            vectors,
            unit,
            inv_sqrt_mass,
            center: Mat::zeros(0, 0),
            displaced: Vec::new(),
            rigid: project_rigid,
        };
        qff.center = qff.in_normal_coordinates(hessian)?;
        Ok(qff)
    }

    /// Number of vibrations.
    pub fn modes(&self) -> usize {
        self.vectors.len()
    }

    /// Number of displaced Hessians needed in all.
    pub fn total(&self) -> usize {
        2 * self.modes()
    }

    /// Number of displaced Hessians told so far.
    pub fn done(&self) -> usize {
        self.displaced.len()
    }

    /// A Cartesian Hessian in the normal coordinates of the reference.
    fn in_normal_coordinates(&self, hessian: &Mat<f64>) -> MoptResult<Mat<f64>> {
        let n = self.reference.dof();
        if hessian.nrows() != n || hessian.ncols() != n {
            return Err(MoptError::config(format!(
                "force field: the Hessian is {}x{}, expected {n}x{n}",
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        let m = self.modes();
        let applied: Vec<Col<f64>> = self
            .vectors
            .iter()
            .map(|v| {
                let scaled = Col::from_fn(n, |c| v[c] * self.inv_sqrt_mass[c]);
                let product = hessian * &scaled;
                Col::from_fn(n, |c| product[c] * self.inv_sqrt_mass[c])
            })
            .collect();
        Ok(Mat::from_fn(m, m, |j, k| {
            0.5 * (col_dot(&self.vectors[j], &applied[k]) + col_dot(&self.vectors[k], &applied[j]))
        }))
    }

    /// The next structure whose Hessian is needed, or `None` when all are told.
    pub fn ask(&self) -> MoptResult<Option<MolecularSystem>> {
        let at = self.displaced.len();
        if at >= self.total() {
            return Ok(None);
        }
        let (mode, sign) = (at / 2, if at.is_multiple_of(2) { 1.0 } else { -1.0 });
        let amplitude = sign * self.settings.step * self.unit[mode];
        let x = self.reference.coords_flat_bohr();
        let moved = Col::from_fn(x.nrows(), |c| {
            x[c] + amplitude * self.vectors[mode][c] * self.inv_sqrt_mass[c]
        });
        Ok(Some(self.reference.with_flat_coords_bohr(&moved)?))
    }

    /// The Cartesian Hessian (Hartree/Bohr^2) of the structure last asked for.
    pub fn tell(&mut self, hessian: &Mat<f64>) -> MoptResult<()> {
        if self.displaced.len() >= self.total() {
            return Err(MoptError::config(
                "force field: all displaced Hessians are told already",
            ));
        }
        let projected = self.in_normal_coordinates(hessian)?;
        self.displaced.push(projected);
        Ok(())
    }

    /// The force field, once all Hessians are told.
    pub fn force_field(&self) -> MoptResult<ForceField> {
        let m = self.modes();
        if self.displaced.len() < self.total() {
            return Err(MoptError::config(format!(
                "force field: {} of {} displaced Hessians are told",
                self.displaced.len(),
                self.total()
            )));
        }
        let to_wavenumbers = hartree_in_wavenumbers();
        let step = self.settings.step;
        // Derivatives with respect to q_i of the Hessian in the dimensionless coordinates:
        // first[i](j, k) is the estimate of phi_ijk from the displacements along i, and
        // second[i](j, k) that of phi_iijk.
        let scaled = |h: &Mat<f64>, j: usize, k: usize| {
            h[(j, k)] * self.unit[j] * self.unit[k] * to_wavenumbers
        };
        let first: Vec<Mat<f64>> = (0..m)
            .map(|i| {
                let (plus, minus) = (&self.displaced[2 * i], &self.displaced[2 * i + 1]);
                Mat::from_fn(m, m, |j, k| {
                    (scaled(plus, j, k) - scaled(minus, j, k)) / (2.0 * step)
                })
            })
            .collect();
        let second: Vec<Mat<f64>> = (0..m)
            .map(|i| {
                let (plus, minus) = (&self.displaced[2 * i], &self.displaced[2 * i + 1]);
                Mat::from_fn(m, m, |j, k| {
                    (scaled(plus, j, k) + scaled(minus, j, k) - 2.0 * scaled(&self.center, j, k))
                        / (step * step)
                })
            })
            .collect();
        let mut field = ForceField {
            harmonic: self.harmonic.clone(),
            cubic: vec![0.0; m * m * m],
            quartic: vec![0.0; m * m * m],
            coriolis: [Mat::zeros(m, m), Mat::zeros(m, m), Mat::zeros(m, m)],
            rotational: [0.0; 3],
        };
        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    field.cubic[(i * m + j) * m + k] =
                        (first[i][(j, k)] + first[j][(i, k)] + first[k][(i, j)]) / 3.0;
                    field.quartic[(i * m + j) * m + k] = if j == k {
                        0.5 * (second[i][(j, j)] + second[j][(i, i)])
                    } else {
                        second[i][(j, k)]
                    };
                }
            }
        }
        if self.rigid && !self.reference.has_pbc() {
            self.rotation(&mut field)?;
        }
        Ok(field)
    }

    /// Rotational constants about the principal axes and the Coriolis constants.
    fn rotation(&self, field: &mut ForceField) -> MoptResult<()> {
        let sys = &self.reference;
        let masses = sys.masses_amu();
        let com = sys.center_of_mass().value();
        let x = sys.coords_flat_bohr();
        let mut inertia = Mat::<f64>::zeros(3, 3);
        for (a, mass) in masses.iter().enumerate() {
            let d: [f64; 3] = std::array::from_fn(|c| x[3 * a + c] - com[c]);
            let square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            for i in 0..3 {
                inertia[(i, i)] += mass * square;
                for j in 0..3 {
                    inertia[(i, j)] -= mass * d[i] * d[j];
                }
            }
        }
        let (moments, axes) = symmetric_eigen(&inertia)?;
        let largest = moments[2].max(f64::MIN_POSITIVE);
        for a in 0..3 {
            // h / (8 pi^2 c I) in cm^-1; no moment about the axis of a linear molecule.
            field.rotational[a] = if moments[a] > 1e-10 * largest {
                let moment =
                    moments[a] * Units::AMU_TO_KG * Units::BOHR_TO_METER * Units::BOHR_TO_METER;
                Units::PLANCK_J_S
                    / (8.0
                        * std::f64::consts::PI.powi(2)
                        * Units::SPEED_OF_LIGHT_M_S
                        * 100.0
                        * moment)
            } else {
                f64::INFINITY
            };
        }
        let m = self.modes();
        // The modes atom by atom in the principal axes.
        let in_axes: Vec<Vec<[f64; 3]>> = self
            .vectors
            .iter()
            .map(|v| {
                (0..masses.len())
                    .map(|a| {
                        std::array::from_fn(|axis| {
                            (0..3).map(|c| axes[(c, axis)] * v[3 * a + c]).sum::<f64>()
                        })
                    })
                    .collect()
            })
            .collect();
        for axis in 0..3 {
            let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
            field.coriolis[axis] = Mat::from_fn(m, m, |i, j| {
                in_axes[i]
                    .iter()
                    .zip(&in_axes[j])
                    .map(|(u, v)| u[b] * v[c] - u[c] * v[b])
                    .sum::<f64>()
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};

    /// Two atoms bound by the Morse potential `D (1 - exp(-a (r - r0)))^2`.
    struct Morse {
        depth: f64,
        width: f64,
        length: f64,
    }

    impl Morse {
        fn hessian(&self, sys: &MolecularSystem) -> Mat<f64> {
            let x = sys.coords_flat_bohr();
            let d: [f64; 3] = std::array::from_fn(|c| x[3 + c] - x[c]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let e = (-self.width * (r - self.length)).exp();
            let first = 2.0 * self.depth * self.width * e * (1.0 - e);
            let second = 2.0 * self.depth * self.width * self.width * e * (2.0 * e - 1.0);
            // d2V/dx dx' for a pair potential: second u u^T + first / r (1 - u u^T).
            let block = Mat::from_fn(3, 3, |i, j| {
                let uu = d[i] * d[j] / (r * r);
                second * uu + first / r * (f64::from(u8::from(i == j)) - uu)
            });
            Mat::from_fn(6, 6, |i, j| {
                let sign = if (i < 3) == (j < 3) { 1.0 } else { -1.0 };
                sign * block[(i % 3, j % 3)]
            })
        }
    }

    #[test]
    fn the_force_field_of_a_morse_oscillator() {
        let morse = Morse {
            depth: 0.17,
            width: 1.0,
            length: 2.4,
        };
        let sys = MolecularSystem::new(
            vec![Element::H, Element::parse_token("Cl").unwrap()],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let mut qff = Qff::new(
            &sys,
            &morse.hessian(&sys),
            true,
            QffSettings {
                step: 0.05,
                lowest: 10.0,
            },
        )
        .unwrap();
        assert_eq!((qff.modes(), qff.total()), (1, 2));
        assert!(qff.force_field().is_err());
        while let Some(structure) = qff.ask().unwrap() {
            qff.tell(&morse.hessian(&structure)).unwrap();
        }
        assert!(qff.tell(&morse.hessian(&sys)).is_err());
        let field = qff.force_field().unwrap();
        // In atomic units: w = a sqrt(2 D / mu); with x = a (hbar / (mu w))^(1/2) the
        // derivatives of the Morse potential give phi_111 = -6 D x^3 and phi_1111 = 14 D x^4
        // in sign-free form (the sign of the cubic constant follows that of the mode).
        let m = sys.masses_amu();
        let mu = m[0] * m[1] / (m[0] + m[1]) * amu_in_electron_masses();
        let omega = morse.width * (2.0 * morse.depth / mu).sqrt();
        let x = morse.width / (mu * omega).sqrt();
        let to_wavenumbers = hartree_in_wavenumbers();
        assert!((field.harmonic[0] - omega * to_wavenumbers).abs() < 1e-6 * field.harmonic[0]);
        let cubic = 6.0 * morse.depth * x.powi(3) * to_wavenumbers;
        let quartic = 14.0 * morse.depth * x.powi(4) * to_wavenumbers;
        assert!(
            (field.cubic(0, 0, 0).abs() - cubic).abs() < 1e-3 * cubic,
            "{} against {cubic}",
            field.cubic(0, 0, 0)
        );
        assert!(
            (field.quartic(0, 0, 0) - quartic).abs() < 1e-3 * quartic,
            "{} against {quartic}",
            field.quartic(0, 0, 0)
        );
        // Perturbation theory on this force field gives the anharmonic constant of the
        // Morse oscillator, -w^2 / (4 D).
        let pt = crate::vib::vpt2(&field, &crate::vib::Vpt2Settings::default());
        let expected = -omega * omega / (4.0 * morse.depth) * to_wavenumbers;
        assert!(
            (pt.constants[(0, 0)] - expected).abs() < 5e-3 * expected.abs(),
            "{} against {expected}",
            pt.constants[(0, 0)]
        );
        assert!(
            (pt.fundamentals[0] - (field.harmonic[0] + 2.0 * pt.constants[(0, 0)])).abs() < 1e-9
        );
        assert_eq!(pt.notes.len(), 1);
        // Beyond the inflection point of the potential the curvature is negative: not a
        // minimum.
        let stretched = sys
            .with_flat_coords_bohr(&Col::from_fn(6, |i| if i == 5 { 3.4 } else { 0.0 }))
            .unwrap();
        assert!(Qff::new(
            &stretched,
            &morse.hessian(&stretched),
            true,
            QffSettings::default()
        )
        .is_err());
        // A linear molecule: no moment about its axis, and one mode has no Coriolis
        // coupling with itself.
        assert!(field.rotational.iter().any(|b| b.is_infinite()));
        assert!(field.coriolis.iter().all(|z| z[(0, 0)].abs() < 1e-12));
    }

    #[test]
    fn constants_are_stored_for_all_orders_of_their_indices() {
        let field = ForceField::from_constants(
            &[1000.0, 2000.0, 3000.0],
            &[(0, 1, 2, 5.0), (1, 1, 0, -3.0)],
            &[(0, 1, 1, 7.0), (2, 0, 1, 2.0), (1, 1, 2, 4.0)],
        )
        .unwrap();
        assert_eq!(field.cubic(2, 0, 1), 5.0);
        assert_eq!(field.cubic(1, 0, 1), -3.0);
        // phi_0011 = phi_1100; phi_2201; phi_1112 = phi_1121.
        assert_eq!((field.quartic(0, 1, 1), field.quartic(1, 0, 0)), (7.0, 7.0));
        assert_eq!((field.quartic(2, 0, 1), field.quartic(2, 1, 0)), (2.0, 2.0));
        assert_eq!((field.quartic(1, 1, 2), field.quartic(1, 2, 1)), (4.0, 4.0));
        assert!(ForceField::from_constants(&[1000.0], &[(0, 0, 1, 1.0)], &[]).is_err());
        assert!(ForceField::from_constants(&[-5.0], &[], &[]).is_err());
    }
}
