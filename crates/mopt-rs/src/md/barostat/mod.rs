//! Barostats (extension point): the cell of a periodic structure follows the pressure.
//!
//! The pressure tensor is `P = (sum m v v^T) / V - sigma`, with the stress `sigma` of the
//! surface in the convention of [`crate::pes::stress`] (Hartree / Bohr^3, negative under
//! compression, as in ASE). A [`Barostat`] acts on its own variables at the barostat stages
//! of the integrator, may change how the velocities follow the forces (the coupling of the
//! Martyna-Tobias-Klein equations), and moves the positions together with the cell in the
//! drift. Built in: Berendsen's weak coupling (isotropic, or anisotropic with the cell
//! deforming under the full pressure tensor), the stochastic cell rescaling of Bernetti
//! and Bussi (isotropic) and its fully anisotropic form of Del Tatto, Raiteri, Bernetti, and
//! Bussi, and the Martyna-Tobias-Klein barostat with a Nose-Hoover chain of its own
//! (isotropic, or with a fully flexible cell).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy reads `-press` without using it).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod berendsen;
pub mod crescale_aniso;
pub mod mtk;
pub mod mtk_aniso;

use faer::Col;

pub use berendsen::{BerendsenBarostat, CellRescale};
pub use crescale_aniso::CellRescaleAniso;
pub use mtk::Mtk;
pub use mtk_aniso::MtkAniso;

use super::space::MdSpace;
use crate::core_infra::rng::Rng;
use crate::domain::{Bohr, PeriodicCell, Units, Vector3};
use crate::error::{MoptError, MoptResult};
use crate::pes::stress::Stress;

/// What a barostat sees when it acts.
pub struct BarostatContext<'a> {
    /// The variables and their masses.
    pub space: &'a MdSpace,
    /// Degrees of freedom of the temperature.
    pub dof: f64,
    /// Target temperature (K).
    pub temperature: f64,
    /// Target pressure (Hartree / Bohr^3).
    pub pressure: f64,
    /// Volume of the cell (Bohr^3).
    pub volume: f64,
    /// `sum m v v^T` (Hartree).
    pub kinetic: [[f64; 3]; 3],
    /// Stress of the surface at the current positions.
    pub stress: Stress,
    /// The random numbers of the run.
    pub rng: &'a mut Rng,
}

impl BarostatContext<'_> {
    /// The instantaneous pressure tensor `K / V - sigma`.
    pub fn pressure_tensor(&self) -> [[f64; 3]; 3] {
        pressure_tensor(&self.kinetic, &self.stress, self.volume)
    }

    /// The instantaneous pressure (a third of the trace of the tensor).
    pub fn instantaneous_pressure(&self) -> f64 {
        let p = self.pressure_tensor();
        (p[0][0] + p[1][1] + p[2][2]) / 3.0
    }

    /// `k_B T` of the target (Hartree).
    pub fn kt(&self) -> f64 {
        Units::BOLTZMANN_HARTREE_PER_K * self.temperature
    }
}

/// `K / V - sigma` for `K = sum m v v^T`.
pub fn pressure_tensor(kinetic: &[[f64; 3]; 3], stress: &Stress, volume: f64) -> [[f64; 3]; 3] {
    std::array::from_fn(|a| std::array::from_fn(|b| kinetic[a][b] / volume - stress[a][b]))
}

/// A barostat.
pub trait Barostat: Send {
    /// Registered name.
    fn name(&self) -> &str;

    /// Act on the barostat's variables for `dt` at the current positions and velocities.
    fn apply(&mut self, dt: f64, v: &mut Col<f64>, ctx: &mut BarostatContext<'_>)
        -> MoptResult<()>;

    /// The velocities follow the accelerations `a` for the time `h` (default: `v += a h`).
    fn kick(&self, h: f64, v: &mut Col<f64>, a: &Col<f64>) {
        for (x, acc) in v.iter_mut().zip(a.iter()) {
            *x += h * acc;
        }
    }

    /// The positions follow the velocities for the time `h`, and the cell moves with them
    /// (default: `x += v h` in a fixed cell).
    fn drift(
        &mut self,
        h: f64,
        x: &mut Col<f64>,
        v: &mut Col<f64>,
        cell: &mut PeriodicCell,
    ) -> MoptResult<()> {
        let _ = cell;
        for (p, w) in x.iter_mut().zip(v.iter()) {
            *p += h * w;
        }
        Ok(())
    }

    /// The energy of the barostat for the conserved quantity: `P V` and that of its own
    /// variables.
    fn energy(&self, volume: f64, pressure: f64, temperature: f64) -> f64 {
        let _ = temperature;
        pressure * volume
    }

    /// Whether the conserved quantity of the run is conserved with this barostat.
    fn has_conserved_quantity(&self) -> bool {
        false
    }

    /// The state of its variables for the log.
    fn notes(&self) -> Vec<String> {
        Vec::new()
    }

    /// The paper of the method.
    fn reference(&self) -> Option<&'static str> {
        None
    }
}

/// `x` and the cell scaled by the matrix `mu` (every position and lattice vector `r`
/// becomes `mu r`).
pub fn scale_positions_and_cell(
    mu: &[[f64; 3]; 3],
    x: &mut Col<f64>,
    cell: &mut PeriodicCell,
) -> MoptResult<()> {
    let apply = |r: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|a| (0..3).map(|b| mu[a][b] * r[b]).sum::<f64>())
    };
    for i in 0..x.nrows() / 3 {
        let r = apply([x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
        for a in 0..3 {
            x[3 * i + a] = r[a];
        }
    }
    let vectors: [[f64; 3]; 3] = std::array::from_fn(|k| apply(cell.lattice_vector(k).value()));
    let axes = cell.periodic_axes;
    *cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Bohr>::new(vectors[0]),
        Vector3::<Bohr>::new(vectors[1]),
        Vector3::<Bohr>::new(vectors[2]),
        axes,
    )?;
    Ok(())
}

/// `sinh(z) / z` without cancellation near zero.
pub fn sinhc(z: f64) -> f64 {
    if z.abs() < 1e-3 {
        let z2 = z * z;
        1.0 + z2 / 6.0 * (1.0 + z2 / 20.0)
    } else {
        libm::sinh(z) / z
    }
}

/// Settings of the built-in barostats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarostatSettings {
    /// Coupling time (atomic units): the relaxation time of Berendsen and the cell
    /// rescaling, the period of the motion of the cell of Martyna-Tobias-Klein.
    pub tau: f64,
    /// Isothermal compressibility assumed by Berendsen and the cell rescaling (Bohr^3 /
    /// Hartree).
    pub compressibility: f64,
    /// Links of the Nose-Hoover chain on the cell (Martyna-Tobias-Klein).
    pub chain: usize,
}

/// Coupling time of the barostats without `tau_p` (fs; see `docs/source/molecular_dynamics.md`).
pub const DEFAULT_TAU_FS: f64 = 1000.0;
/// Compressibility assumed without `compressibility` (1/GPa; see `docs/source/molecular_dynamics.md`).
pub const DEFAULT_COMPRESSIBILITY_PER_GPA: f64 = 0.01;

/// Pressure: one atomic unit (Hartree / Bohr^3) in GPa (the variable-cell optimizer's function).
pub use crate::opt::cell::gigapascal_per_atomic_unit;

impl Default for BarostatSettings {
    fn default() -> Self {
        Self {
            tau: DEFAULT_TAU_FS * Units::FS_TO_AU_TIME,
            compressibility: DEFAULT_COMPRESSIBILITY_PER_GPA * gigapascal_per_atomic_unit(),
            chain: 3,
        }
    }
}

type BarostatFactory =
    Box<dyn Fn(&BarostatSettings) -> MoptResult<Box<dyn Barostat>> + Send + Sync>;

/// Barostats by name.
pub struct BarostatRegistry {
    entries: Vec<(String, BarostatFactory)>,
}

impl BarostatRegistry {
    /// `berendsen`, `berendsen_aniso`, `crescale`, `crescale_aniso`, `mtk`, `mtk_aniso`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register("berendsen", |s| {
            Ok(Box::new(BerendsenBarostat::new(
                s.tau,
                s.compressibility,
                false,
            )?))
        });
        r.register("berendsen_aniso", |s| {
            Ok(Box::new(BerendsenBarostat::new(
                s.tau,
                s.compressibility,
                true,
            )?))
        });
        r.register("crescale", |s| {
            Ok(Box::new(CellRescale::new(s.tau, s.compressibility)?))
        });
        r.register("crescale_aniso", |s| {
            Ok(Box::new(CellRescaleAniso::new(s.tau, s.compressibility)?))
        });
        r.register("mtk", |s| Ok(Box::new(Mtk::new(s.tau, s.chain)?)));
        r.register("mtk_aniso", |s| {
            Ok(Box::new(MtkAniso::new(s.tau, s.chain)?))
        });
        r
    }

    /// Add (or replace) a barostat.
    pub fn register(
        &mut self,
        name: &str,
        make: impl Fn(&BarostatSettings) -> MoptResult<Box<dyn Barostat>> + Send + Sync + 'static,
    ) {
        self.entries.retain(|(n, _)| n != name);
        self.entries.push((name.to_string(), Box::new(make)));
    }

    /// Names.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|(n, _)| n.clone()).collect()
    }

    /// The barostat `name` with `settings`.
    pub fn create(&self, name: &str, settings: &BarostatSettings) -> MoptResult<Box<dyn Barostat>> {
        let key = match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "c_rescale" | "cellrescale" | "bussi" => "crescale".to_string(),
            "martyna_tobias_klein" | "mttk" => "mtk".to_string(),
            "anisotropic_berendsen" => "berendsen_aniso".to_string(),
            "anisotropic_crescale" | "crescale_anisotropic" => "crescale_aniso".to_string(),
            "anisotropic_mtk" | "mtk_flexible" | "mttk_aniso" => "mtk_aniso".to_string(),
            other => other.to_string(),
        };
        let make = self
            .entries
            .iter()
            .find(|(n, _)| *n == key)
            .map(|(_, m)| m)
            .ok_or_else(|| {
                MoptError::config(format!(
                    "unknown barostat '{name}' (known: {})",
                    self.names().join(", ")
                ))
            })?;
        make(settings)
    }
}

impl Default for BarostatRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn scaling_the_cell_scales_the_volume_and_the_positions() {
        let mut cell = PeriodicCell::orthorhombic(
            crate::domain::Quantity::<Bohr>::new(10.0),
            crate::domain::Quantity::new(12.0),
            crate::domain::Quantity::new(14.0),
        )
        .unwrap();
        let mut x = Col::from_fn(6, |k| k as f64);
        let mu = [[1.01, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.01]];
        let v0 = cell.volume_bohr3();
        scale_positions_and_cell(&mu, &mut x, &mut cell).unwrap();
        assert!((cell.volume_bohr3() / v0 - 1.01_f64.powi(3)).abs() < 1e-12);
        assert!((x[5] - 5.05).abs() < 1e-12);
        for z in [0.0, 1e-5, 0.3, -2.0] {
            let reference = if z == 0.0 { 1.0 } else { libm::sinh(z) / z };
            assert!((sinhc(z) - reference).abs() < 1e-14 * reference.max(1.0));
        }
        // 1 atomic unit of pressure is 29421 GPa.
        assert!((gigapascal_per_atomic_unit() - 29421.0).abs() < 1.0);
        let r = BarostatRegistry::builtin();
        for name in [
            "berendsen",
            "berendsen_aniso",
            "crescale",
            "c-rescale",
            "mtk",
        ] {
            assert!(
                r.create(name, &BarostatSettings::default()).is_ok(),
                "{name}"
            );
        }
        assert!(r
            .create("parrinello_rahman", &BarostatSettings::default())
            .is_err());
    }
}
