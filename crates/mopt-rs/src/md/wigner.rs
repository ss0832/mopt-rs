//! Initial conditions from the harmonic Wigner distribution of the normal modes at a
//! minimum or a transition state, for trajectories from a transition state or the initial
//! conditions of photodynamics.
//!
//! The Wigner function of a harmonic oscillator of frequency `omega` in thermal equilibrium
//! at `T` is a Gaussian in the mass-weighted normal coordinate `Q` and its momentum `P`
//! (Hillery, O'Connell, Scully, and Wigner, Phys. Rep. 1984, 106, 121, sec. 2.7; atomic
//! units, `hbar = 1`):
//!
//! ```text
//! W(Q, P) ~ exp(-tanh(omega / 2 k_B T) (P^2 / omega + omega Q^2))
//! <Q^2> = coth(omega / 2 k_B T) / (2 omega),   <P^2> = omega coth(omega / 2 k_B T) / 2
//! ```
//!
//! so that the mean energy of a mode, `(omega / 2) coth(omega / 2 k_B T)`, is that of the
//! quantum oscillator (the zero-point energy `omega / 2` at `T = 0`, the ground state;
//! `k_B T` in the classical limit). Every vibration is drawn on its own and the structure is
//! `x = x0 + M^-1/2 sum_k L_k Q_k`, the velocities `v = M^-1/2 sum_k L_k P_k`; translations
//! and rotations get nothing.
//!
//! Modes that need a decision of their own:
//! - imaginary modes (a transition state): no displacement along them; the momentum is
//!   zero, drawn from the Boltzmann distribution `N(0, k_B T)`, or that with a chosen sign
//!   (forward or backward along the mode);
//! - modes below a wavenumber cutoff (soft modes whose quantum width is large): the Wigner
//!   distribution like the others, the classical Boltzmann distribution
//!   (`<Q^2> = k_B T / omega^2`, `<P^2> = k_B T`), or nothing.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::core_infra::rng::Rng;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::vib::NormalModes;

/// What modes below the cutoff get.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowModes {
    /// The Wigner distribution, as the other modes.
    Wigner,
    /// The classical Boltzmann distribution at the temperature.
    Classical,
    /// No displacement and no momentum.
    Zero,
}

/// What imaginary modes get (never a displacement).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImaginaryModes {
    /// No momentum.
    Zero,
    /// A momentum from `N(0, k_B T)`.
    Boltzmann,
    /// Its magnitude, along the mode.
    Forward,
    /// Its magnitude, against the mode.
    Backward,
}

/// Settings of the sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WignerSettings {
    /// Temperature (K); 0: the vibrational ground state.
    pub temperature: f64,
    /// Modes with wavenumbers below this (cm^-1) are low modes.
    pub low_cutoff: f64,
    /// Treatment of the low modes.
    pub low_modes: LowModes,
    /// Treatment of imaginary modes.
    pub imaginary: ImaginaryModes,
}

impl Default for WignerSettings {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            low_cutoff: 0.0,
            low_modes: LowModes::Wigner,
            imaginary: ImaginaryModes::Boltzmann,
        }
    }
}

/// How a mode was treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Treatment {
    /// Wigner distribution.
    Wigner,
    /// Classical Boltzmann distribution.
    Classical,
    /// Nothing.
    Zero,
    /// Imaginary: a momentum only.
    Imaginary,
}

/// A structure and velocities drawn from the distribution.
#[derive(Debug, Clone)]
pub struct WignerSample {
    /// Positions (Bohr).
    pub x: Col<f64>,
    /// Velocities (Bohr per atomic unit of time).
    pub v: Col<f64>,
    /// Energy of every mode, `P^2/2 + omega^2 Q^2/2` (Hartree).
    pub energies: Vec<f64>,
}

/// `coth(z)` for `z > 0`, with its limits.
fn coth(z: f64) -> f64 {
    if z > 20.0 {
        1.0
    } else if z < 1e-8 {
        1.0 / z
    } else {
        1.0 / libm::tanh(z)
    }
}

/// Angular frequency (atomic units) of a curvature of the mass-weighted Hessian
/// (Hartree / (Bohr^2 amu)); negative for an imaginary frequency (its magnitude).
pub fn angular_frequency(curvature: f64) -> f64 {
    let k = curvature / Units::AMU_TO_AU_MASS;
    if k >= 0.0 {
        libm::sqrt(k)
    } else {
        -libm::sqrt(-k)
    }
}

/// Mean energy of a harmonic mode of angular frequency `omega` (atomic units) in the
/// Wigner (quantum) distribution at `temperature` (K): `(omega/2) coth(omega / 2 k_B T)`.
pub fn mean_energy(omega: f64, temperature: f64) -> f64 {
    let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature.max(0.0);
    if kt <= 0.0 {
        return 0.5 * omega;
    }
    0.5 * omega * coth(omega / (2.0 * kt))
}

/// The treatment of mode `k` of `modes` under `s`.
pub fn treatment(modes: &NormalModes, k: usize, s: &WignerSettings) -> Treatment {
    let nu = modes.wavenumbers[k];
    if nu < 0.0 {
        Treatment::Imaginary
    } else if nu < s.low_cutoff {
        match s.low_modes {
            LowModes::Wigner => Treatment::Wigner,
            LowModes::Classical => Treatment::Classical,
            LowModes::Zero => Treatment::Zero,
        }
    } else {
        Treatment::Wigner
    }
}

/// Variances of `Q` and `P` (atomic units) of a mode.
fn variances(t: Treatment, omega: f64, kt: f64) -> (f64, f64) {
    match t {
        Treatment::Wigner if kt > 0.0 => {
            let c = coth(omega / (2.0 * kt));
            (c / (2.0 * omega), 0.5 * omega * c)
        }
        Treatment::Wigner => (0.5 / omega, 0.5 * omega),
        Treatment::Classical => (kt / (omega * omega), kt),
        Treatment::Zero | Treatment::Imaginary => (0.0, 0.0),
    }
}

/// One sample about the structure `x0` (Bohr) with the vibrations `modes` (only the
/// vibrations: those of `vib::normal_modes::transverse_modes` with the rigid motions
/// projected out).
pub fn sample(
    x0: &Col<f64>,
    modes: &NormalModes,
    s: &WignerSettings,
    rng: &mut Rng,
) -> MoptResult<WignerSample> {
    let n = x0.nrows();
    if modes.displacements.iter().any(|d| d.nrows() != n) {
        return Err(MoptError::config(
            "the normal modes do not belong to this structure",
        ));
    }
    let kt = Units::BOLTZMANN_HARTREE_PER_K * s.temperature.max(0.0);
    // Q and P are in atomic units (sqrt(m_e) Bohr); the displacements of the modes are
    // per sqrt(amu) Bohr.
    let to_cartesian = 1.0 / libm::sqrt(Units::AMU_TO_AU_MASS);
    let mut x = x0.clone();
    let mut v = Col::<f64>::zeros(n);
    let mut energies = Vec::with_capacity(modes.curvatures.len());
    for (k, (&curvature, d)) in modes
        .curvatures
        .iter()
        .zip(&modes.displacements)
        .enumerate()
    {
        let t = treatment(modes, k, s);
        let omega = angular_frequency(curvature);
        let (q, p) = match t {
            Treatment::Imaginary => {
                let magnitude = libm::sqrt(kt) * rng.standard_normal();
                let p = match s.imaginary {
                    ImaginaryModes::Zero => 0.0,
                    ImaginaryModes::Boltzmann => magnitude,
                    ImaginaryModes::Forward => magnitude.abs(),
                    ImaginaryModes::Backward => -magnitude.abs(),
                };
                (0.0, p)
            }
            _ if omega <= 0.0 => (0.0, 0.0),
            _ => {
                let (sq, sp) = variances(t, omega, kt);
                (
                    libm::sqrt(sq) * rng.standard_normal(),
                    libm::sqrt(sp) * rng.standard_normal(),
                )
            }
        };
        for i in 0..n {
            x[i] += d[i] * q * to_cartesian;
            v[i] += d[i] * p * to_cartesian;
        }
        energies.push(0.5 * p * p + 0.5 * omega.max(0.0).powi(2) * q * q);
    }
    Ok(WignerSample { x, v, energies })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
    use crate::vib::transverse_modes;
    use faer::Mat;

    /// HCl with a harmonic bond (force constant 0.32 Hartree/Bohr^2, about 2990 cm^-1).
    fn hcl() -> (MolecularSystem, Mat<f64>) {
        let sys = MolecularSystem::new(
            vec![Element::H, Element::parse_token("Cl").unwrap()],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [0.0, 0.0, 1.27]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let k = 0.32;
        let h = Mat::from_fn(6, 6, |i, j| {
            if i % 3 == 2 && j % 3 == 2 {
                if i / 3 == j / 3 {
                    k
                } else {
                    -k
                }
            } else {
                0.0
            }
        });
        (sys, h)
    }

    #[test]
    fn the_sampled_energies_are_those_of_the_quantum_oscillator() {
        let (sys, h) = hcl();
        let modes = transverse_modes(&sys, &h, true, &[]).unwrap();
        let k = modes.wavenumbers.iter().position(|w| *w > 100.0).unwrap();
        let omega = angular_frequency(modes.curvatures[k]);
        let x0 = sys.coords_flat_bohr();
        let masses: Vec<f64> = sys
            .masses_amu()
            .iter()
            .flat_map(|m| [m * Units::AMU_TO_AU_MASS; 3])
            .collect();
        let mut rng = Rng::from_seed(5);
        for temperature in [0.0, 3000.0, 30000.0] {
            let s = WignerSettings {
                temperature,
                ..WignerSettings::default()
            };
            let count = 40000;
            let mut total = 0.0;
            for _ in 0..count {
                let w = sample(&x0, &modes, &s, &mut rng).unwrap();
                // The energy of the sample from the Cartesian Hessian and masses.
                let dx = &w.x - &x0;
                let potential: f64 = 0.5
                    * (0..6)
                        .flat_map(|i| (0..6).map(move |j| (i, j)))
                        .map(|(i, j)| dx[i] * h[(i, j)] * dx[j])
                        .sum::<f64>();
                let kinetic: f64 = 0.5 * (0..6).map(|i| masses[i] * w.v[i] * w.v[i]).sum::<f64>();
                let e: f64 = w.energies.iter().sum();
                assert!((potential + kinetic - e).abs() < 1e-9 * e.max(1e-12));
                total += e;
            }
            let mean = total / count as f64;
            let expected = mean_energy(omega, temperature);
            // The energy of a mode is exponentially distributed beyond its minimum: the
            // standard error of the mean is below expected / sqrt(count).
            assert!(
                (mean / expected - 1.0).abs() < 4.0 / (count as f64).sqrt(),
                "T {temperature}: {mean} against {expected}"
            );
        }
        // The ground state at T = 0 holds the zero-point energy.
        assert!((mean_energy(omega, 0.0) - 0.5 * omega).abs() < 1e-15);
        // The classical limit.
        let hot = 1e7;
        let kt = Units::BOLTZMANN_HARTREE_PER_K * hot;
        assert!((mean_energy(omega, hot) / kt - 1.0).abs() < 1e-3);
    }

    #[test]
    fn imaginary_and_low_modes_get_what_was_chosen() {
        let (sys, h) = hcl();
        // The same bond as a barrier: an imaginary mode.
        let barrier = Mat::from_fn(6, 6, |i, j| -h[(i, j)]);
        let modes = transverse_modes(&sys, &barrier, true, &[]).unwrap();
        assert!(modes.wavenumbers.iter().any(|w| *w < -100.0));
        let x0 = sys.coords_flat_bohr();
        let mut rng = Rng::from_seed(9);
        for (imaginary, sign) in [
            (ImaginaryModes::Forward, 1.0),
            (ImaginaryModes::Backward, -1.0),
        ] {
            let s = WignerSettings {
                temperature: 300.0,
                imaginary,
                ..WignerSettings::default()
            };
            let w = sample(&x0, &modes, &s, &mut rng).unwrap();
            assert!((&w.x - &x0).iter().all(|d| d.abs() < 1e-15));
            // H moves along +z with the mode's displacement (sign of the mode vector).
            let k = modes.wavenumbers.iter().position(|w| *w < -100.0).unwrap();
            let along: f64 = (0..6).map(|i| modes.displacements[k][i] * w.v[i]).sum();
            assert!(sign * along > 0.0);
        }
        // A cutoff above the frequency makes the mode a low mode; zero leaves it at rest.
        let modes = transverse_modes(&sys, &h, true, &[]).unwrap();
        let s = WignerSettings {
            temperature: 300.0,
            low_cutoff: 1e5,
            low_modes: LowModes::Zero,
            ..WignerSettings::default()
        };
        let w = sample(&x0, &modes, &s, &mut rng).unwrap();
        assert!(w.energies.iter().all(|e| *e == 0.0));
    }
}
