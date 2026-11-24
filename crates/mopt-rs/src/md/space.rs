//! The variables of a molecular dynamics run: their masses, which of them move, and the
//! motions of the whole system that the dynamics conserve.
//!
//! Positions are in Bohr, velocities in Bohr per atomic unit of time, masses in electron
//! masses (atomic units throughout); atoms carry their atomic weights, plain variables a
//! mass of one each. A degree of freedom with infinite mass (an atom of `-fix`) has inverse
//! mass zero: forces, thermostats, and constraints never move it.
//!
//! MultiOptPy: `MD/thermostat.py:Thermostat` (`masses`, `calc_tot_kinetic_energy`,
//! `init_purtubation`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::core_infra::rng::Rng;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// Motions of the whole system that the dynamics conserve and that are taken out of the
/// starting velocities and out of the count of degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigidMotion {
    /// None (external fields, frozen atoms, stochastic thermostats, plain variables).
    None,
    /// The total momentum (periodic cells, and molecules in a fixed frame).
    Translation,
    /// The total momentum and angular momentum (isolated molecules).
    TranslationRotation,
}

impl RigidMotion {
    /// Parse `none`, `translation`, `full`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Ok(Self::None),
            "translation" | "momentum" => Ok(Self::Translation),
            "full" | "translation_rotation" | "rotation" => Ok(Self::TranslationRotation),
            other => Err(MoptError::parse(format!(
                "rigid motion '{other}': none, translation, or full"
            ))),
        }
    }

    /// Name for the log.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Translation => "translation",
            Self::TranslationRotation => "translation and rotation",
        }
    }
}

/// The variables of a run.
#[derive(Debug, Clone)]
pub struct MdSpace {
    masses: Vec<f64>,
    inverse_masses: Vec<f64>,
    /// Variables per particle: 3 for atoms, 1 for plain variables.
    group: usize,
}

impl MdSpace {
    /// Atoms with masses in atomic mass units (three variables each).
    pub fn atoms(masses_amu: &[f64]) -> MoptResult<Self> {
        let masses: Vec<f64> = masses_amu
            .iter()
            .flat_map(|m| [m * Units::AMU_TO_AU_MASS; 3])
            .collect();
        Self::from_masses(masses, 3)
    }

    /// `n` plain variables with a mass of one each.
    pub fn variables(n: usize) -> MoptResult<Self> {
        Self::from_masses(vec![1.0; n], 1)
    }

    /// Plain variables with the given masses.
    pub fn variables_with_masses(masses: &[f64]) -> MoptResult<Self> {
        Self::from_masses(masses.to_vec(), 1)
    }

    fn from_masses(masses: Vec<f64>, group: usize) -> MoptResult<Self> {
        if masses.is_empty() {
            return Err(MoptError::config("molecular dynamics needs variables"));
        }
        if let Some(m) = masses.iter().find(|m| !(m.is_finite() && **m > 0.0)) {
            return Err(MoptError::config(format!(
                "masses must be positive and finite, got {m}"
            )));
        }
        let inverse_masses = masses.iter().map(|m| 1.0 / m).collect();
        Ok(Self {
            masses,
            inverse_masses,
            group,
        })
    }

    /// Hold the variables `dofs` (inverse mass zero: they never move).
    pub fn freeze(&mut self, dofs: &[usize]) -> MoptResult<()> {
        for &k in dofs {
            if k >= self.masses.len() {
                return Err(MoptError::config(format!(
                    "variable {} does not exist ({} variables)",
                    k + 1,
                    self.masses.len()
                )));
            }
            self.inverse_masses[k] = 0.0;
        }
        Ok(())
    }

    /// Number of variables.
    pub fn dimension(&self) -> usize {
        self.masses.len()
    }

    /// Variables per particle (3 for atoms).
    pub fn group(&self) -> usize {
        self.group
    }

    /// Number of particles.
    pub fn particles(&self) -> usize {
        self.masses.len() / self.group
    }

    /// Whether the variables are the coordinates of atoms.
    pub fn is_atomic(&self) -> bool {
        self.group == 3
    }

    /// Masses (electron masses for atoms).
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Inverse masses (zero for held variables).
    pub fn inverse_masses(&self) -> &[f64] {
        &self.inverse_masses
    }

    /// Whether variable `k` moves.
    pub fn moves(&self, k: usize) -> bool {
        self.inverse_masses[k] > 0.0
    }

    /// Number of variables that move.
    pub fn mobile(&self) -> usize {
        self.inverse_masses.iter().filter(|w| **w > 0.0).count()
    }

    /// Whether any variable is held.
    pub fn has_held(&self) -> bool {
        self.mobile() < self.dimension()
    }

    /// Kinetic energy `sum m v^2 / 2` (Hartree).
    pub fn kinetic_energy(&self, v: &Col<f64>) -> f64 {
        0.5 * self
            .masses
            .iter()
            .zip(v.iter())
            .zip(&self.inverse_masses)
            .filter(|(_, w)| **w > 0.0)
            .map(|((m, x), _)| m * x * x)
            .sum::<f64>()
    }

    /// `sum m v v^T` over the atoms (Hartree), the kinetic part of the pressure tensor
    /// times the volume; zero for plain variables.
    pub fn kinetic_tensor(&self, v: &Col<f64>) -> [[f64; 3]; 3] {
        let mut t = [[0.0; 3]; 3];
        if !self.is_atomic() {
            return t;
        }
        for i in 0..self.particles() {
            let m = self.masses[3 * i];
            if self.inverse_masses[3 * i] == 0.0 {
                continue;
            }
            for a in 0..3 {
                for b in 0..3 {
                    t[a][b] += m * v[3 * i + a] * v[3 * i + b];
                }
            }
        }
        t
    }

    /// Velocities drawn from the Maxwell-Boltzmann distribution at `temperature` (K) for
    /// the variables that move.
    pub fn maxwell_boltzmann(&self, temperature: f64, rng: &mut Rng) -> Col<f64> {
        let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature.max(0.0);
        Col::from_fn(self.dimension(), |k| {
            let w = self.inverse_masses[k];
            if w > 0.0 {
                libm::sqrt(kt * w) * rng.standard_normal()
            } else {
                0.0
            }
        })
    }

    /// Number of conserved rigid motions `rigid` of the structure at `x` (3 translations,
    /// and 3 rotations, 2 for a linear arrangement); 0 for plain variables.
    pub fn rigid_count(&self, x: &Col<f64>, rigid: RigidMotion) -> usize {
        if !self.is_atomic() {
            return 0;
        }
        match rigid {
            RigidMotion::None => 0,
            RigidMotion::Translation => 3.min(self.mobile()),
            RigidMotion::TranslationRotation => {
                let rotations = self.inertia(x).map_or(0, |(values, _)| {
                    let top = values.iter().fold(0.0_f64, |a, v| a.max(*v));
                    values
                        .iter()
                        .filter(|v| **v > 1e-8 * top.max(1e-300))
                        .count()
                });
                (3 + rotations).min(self.mobile())
            }
        }
    }

    /// Centre of mass of the atoms at `x`.
    fn centre(&self, x: &Col<f64>) -> [f64; 3] {
        let mut c = [0.0; 3];
        let mut total = 0.0;
        for i in 0..self.particles() {
            let m = self.masses[3 * i];
            total += m;
            for a in 0..3 {
                c[a] += m * x[3 * i + a];
            }
        }
        c.map(|v| v / total)
    }

    /// Eigenvalues and eigenvectors of the inertia tensor about the centre of mass.
    fn inertia(&self, x: &Col<f64>) -> MoptResult<(Col<f64>, Mat<f64>)> {
        let c = self.centre(x);
        let mut i3 = Mat::<f64>::zeros(3, 3);
        for i in 0..self.particles() {
            let m = self.masses[3 * i];
            let r = [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]];
            let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            for a in 0..3 {
                for b in 0..3 {
                    let d = if a == b { r2 } else { 0.0 };
                    i3[(a, b)] += m * (d - r[a] * r[b]);
                }
            }
        }
        symmetric_eigen(&i3)
    }

    /// `v` without the motion of the centre of mass and, for `TranslationRotation`, without
    /// the rotation of the whole structure at `x` (nothing is taken out while atoms are
    /// held or for plain variables).
    pub fn remove_rigid_motion(
        &self,
        x: &Col<f64>,
        v: &mut Col<f64>,
        rigid: RigidMotion,
    ) -> MoptResult<()> {
        if !self.is_atomic() || rigid == RigidMotion::None || self.has_held() {
            return Ok(());
        }
        let n = self.particles();
        let total: f64 = (0..n).map(|i| self.masses[3 * i]).sum();
        let mut p = [0.0; 3];
        for i in 0..n {
            for a in 0..3 {
                p[a] += self.masses[3 * i] * v[3 * i + a];
            }
        }
        for i in 0..n {
            for a in 0..3 {
                v[3 * i + a] -= p[a] / total;
            }
        }
        if rigid != RigidMotion::TranslationRotation {
            return Ok(());
        }
        let c = self.centre(x);
        let rel = |i: usize| [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]];
        let mut l = [0.0; 3];
        for i in 0..n {
            let (r, m) = (rel(i), self.masses[3 * i]);
            let w = [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
            l[0] += m * (r[1] * w[2] - r[2] * w[1]);
            l[1] += m * (r[2] * w[0] - r[0] * w[2]);
            l[2] += m * (r[0] * w[1] - r[1] * w[0]);
        }
        // omega = I^+ L, over the principal axes with a moment of inertia.
        let (values, axes) = self.inertia(x)?;
        let top = values.iter().fold(0.0_f64, |a, v| a.max(*v));
        let mut omega = [0.0; 3];
        for k in 0..3 {
            if values[k] <= 1e-8 * top.max(1e-300) {
                continue;
            }
            let along = (0..3).map(|a| axes[(a, k)] * l[a]).sum::<f64>() / values[k];
            for a in 0..3 {
                omega[a] += along * axes[(a, k)];
            }
        }
        for i in 0..n {
            let r = rel(i);
            v[3 * i] -= omega[1] * r[2] - omega[2] * r[1];
            v[3 * i + 1] -= omega[2] * r[0] - omega[0] * r[2];
            v[3 * i + 2] -= omega[0] * r[1] - omega[1] * r[0];
        }
        Ok(())
    }
}

/// Temperature (K) of the kinetic energy `kinetic` spread over `dof` degrees of freedom.
pub fn temperature_of(kinetic: f64, dof: f64) -> f64 {
    if dof > 0.0 {
        2.0 * kinetic / (dof * Units::BOLTZMANN_HARTREE_PER_K)
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn rigid_motion_is_taken_out_and_counted() {
        let space = MdSpace::atoms(&[12.0, 16.0, 1.0]).unwrap();
        let x = Col::from_fn(9, |k| [0.0, 0.0, 0.0, 2.1, 0.0, 0.3, -1.0, 1.7, 0.2][k]);
        let mut rng = Rng::from_seed(3);
        let mut v = space.maxwell_boltzmann(300.0, &mut rng);
        space
            .remove_rigid_motion(&x, &mut v, RigidMotion::TranslationRotation)
            .unwrap();
        let mut p = [0.0; 3];
        let mut l = [0.0; 3];
        let c = space.centre(&x);
        for i in 0..3 {
            let m = space.masses()[3 * i];
            let r = [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]];
            for a in 0..3 {
                p[a] += m * v[3 * i + a];
            }
            l[0] += m * (r[1] * v[3 * i + 2] - r[2] * v[3 * i + 1]);
            l[1] += m * (r[2] * v[3 * i] - r[0] * v[3 * i + 2]);
            l[2] += m * (r[0] * v[3 * i + 1] - r[1] * v[3 * i]);
        }
        assert!(p.iter().chain(&l).all(|q| q.abs() < 1e-12), "{p:?} {l:?}");
        assert_eq!(space.rigid_count(&x, RigidMotion::TranslationRotation), 6);
        // A linear molecule has two rotations.
        let line = Col::from_fn(9, |k| if k % 3 == 0 { k as f64 } else { 0.0 });
        assert_eq!(
            space.rigid_count(&line, RigidMotion::TranslationRotation),
            5
        );
        assert_eq!(space.rigid_count(&line, RigidMotion::Translation), 3);
        assert_eq!(
            MdSpace::variables(2)
                .unwrap()
                .rigid_count(&x, RigidMotion::Translation),
            0
        );
    }

    #[test]
    fn held_variables_do_not_move_and_carry_no_kinetic_energy() {
        let mut space = MdSpace::atoms(&[1.0, 1.0]).unwrap();
        space.freeze(&[0, 1, 2]).unwrap();
        let mut rng = Rng::from_seed(5);
        let v = space.maxwell_boltzmann(500.0, &mut rng);
        assert_eq!((v[0], v[1], v[2]), (0.0, 0.0, 0.0));
        assert!(v[3] != 0.0);
        assert_eq!(space.mobile(), 3);
        assert!(space.freeze(&[6]).is_err());
        let temperature = temperature_of(space.kinetic_energy(&v), 3.0);
        assert!(temperature > 0.0);
    }
}
