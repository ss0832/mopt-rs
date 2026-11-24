//! The rate constant of a barrierless association `A + B -> AB`: the reverse of the
//! barrierless dissociation of a complex into a dissociation channel of the mapper.
//!
//! - In the gas (the default) the capture of hard spheres that stick on contact, the
//!   collision rate `k_c = pi R^2 sqrt(8 k_B T / (pi mu))` with the contact distance
//!   `R = r_A + r_B` and the reduced mass `mu`.
//! - In a solution (a viscosity `eta` is given) Collins and Kimball's encounter by diffusion
//!   and capture at contact in series, `1 / k = 1 / k_D + 1 / k_c`, with Smoluchowski's
//!   `k_D = 4 pi R (D_A + D_B)` and the Stokes-Einstein coefficients `D = k_B T / (6 pi eta r)`
//!   (stick boundary); for equal radii `k_D = 8 R T / (3 eta)` whatever the radius.
//!
//! The radius `r` of a fragment is that of the sphere of its van der Waals volume (the union
//! of the spheres of the radii of Alvarez, on a grid of 0.1 Angstrom). The rates are in
//! L/mol/s. The step from the channel has the pseudo-first-order rate constant `k c°` at the
//! standard concentration (the free energy of the channel is that of its fragments at `c°`),
//! and the dissociation follows from detailed balance; [`association_top`] writes this as the
//! free energy of a bottleneck that follows the side of the channel.
//!
//! References: M. v. Smoluchowski, Z. Phys. Chem. 1917, 92, 129; F. C. Collins, G. E.
//! Kimball, J. Colloid Sci. 1949, 4, 425 (the texts were not read here: these are the forms
//! of the textbooks).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::kinetics::conditions::Conditions;
use crate::parameters::vdw::alvarez_vdw_radius_bohr;

/// Spacing of the grid of the van der Waals volume (Bohr): 0.1 Angstrom.
const GRID: f64 = 0.1 * Units::ANGSTROM_TO_BOHR;

/// Two fragments that associate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncounterPair {
    /// Their radii (Bohr).
    pub radii: [f64; 2],
    /// Their masses (amu).
    pub masses: [f64; 2],
}

/// A barrierless step between a complex and the channel of its two molecules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssociationStep {
    /// Which of the two states of the step is the channel (0 or 1).
    pub channel_side: usize,
    /// The two fragments.
    pub pair: EncounterPair,
}

impl EncounterPair {
    /// The pair of the structures `a` and `b`.
    pub fn of(a: &MolecularSystem, b: &MolecularSystem) -> Self {
        let mass = |s: &MolecularSystem| s.masses_amu().iter().sum::<f64>();
        Self {
            radii: [volume_radius(a), volume_radius(b)],
            masses: [mass(a), mass(b)],
        }
    }

    fn contact_m(&self) -> f64 {
        (self.radii[0] + self.radii[1]) * Units::BOHR_TO_METER
    }
}

/// The radius (Bohr) of the sphere of the van der Waals volume of `system`: the union of the
/// spheres of the radii of Alvarez (2 Angstrom for an element without one).
pub fn volume_radius(system: &MolecularSystem) -> f64 {
    let x = system.coords_flat_bohr();
    let n = system.n_atoms();
    let r: Vec<f64> = system
        .elements
        .iter()
        .map(|e| alvarez_vdw_radius_bohr(*e).unwrap_or(2.0 * Units::ANGSTROM_TO_BOHR))
        .collect();
    let lo: Vec<f64> = (0..3)
        .map(|k| {
            (0..n)
                .map(|i| x[3 * i + k] - r[i])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let hi: Vec<f64> = (0..3)
        .map(|k| {
            (0..n)
                .map(|i| x[3 * i + k] + r[i])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let steps: Vec<usize> = (0..3)
        .map(|k| ((hi[k] - lo[k]) / GRID).ceil().max(1.0) as usize)
        .collect();
    let mut inside = 0usize;
    for a in 0..steps[0] {
        let px = lo[0] + (a as f64 + 0.5) * GRID;
        for b in 0..steps[1] {
            let py = lo[1] + (b as f64 + 0.5) * GRID;
            for c in 0..steps[2] {
                let pz = lo[2] + (c as f64 + 0.5) * GRID;
                if (0..n).any(|i| {
                    let d = (px - x[3 * i]).powi(2)
                        + (py - x[3 * i + 1]).powi(2)
                        + (pz - x[3 * i + 2]).powi(2);
                    d <= r[i] * r[i]
                }) {
                    inside += 1;
                }
            }
        }
    }
    let volume = inside as f64 * GRID.powi(3);
    (3.0 * volume / (4.0 * PI)).cbrt()
}

/// The capture rate (L/mol/s) of the hard spheres of `pair` at `temperature` (K).
pub fn capture_rate(pair: &EncounterPair, temperature: f64) -> f64 {
    let mu = pair.masses[0] * pair.masses[1] / (pair.masses[0] + pair.masses[1]) * Units::AMU_TO_KG;
    let speed = (8.0 * Units::BOLTZMANN_J_PER_K * temperature / (PI * mu)).sqrt();
    let r = pair.contact_m();
    PI * r * r * speed * Units::AVOGADRO * 1.0e3
}

/// Smoluchowski's diffusion rate (L/mol/s) of `pair` at `temperature` (K) in a solvent of
/// `viscosity` (Pa s), with the Stokes-Einstein coefficients.
pub fn diffusion_rate(pair: &EncounterPair, temperature: f64, viscosity: f64) -> f64 {
    let kt = Units::BOLTZMANN_J_PER_K * temperature;
    let d = |r: f64| kt / (6.0 * PI * viscosity * r * Units::BOHR_TO_METER);
    4.0 * PI * pair.contact_m() * (d(pair.radii[0]) + d(pair.radii[1])) * Units::AVOGADRO * 1.0e3
}

/// The association rate (L/mol/s): the capture in the gas (`viscosity` `None`), Collins and
/// Kimball's in a solution.
pub fn association_rate(pair: &EncounterPair, temperature: f64, viscosity: Option<f64>) -> f64 {
    let capture = capture_rate(pair, temperature);
    match viscosity {
        None => capture,
        Some(eta) => {
            let diffusion = diffusion_rate(pair, temperature, eta);
            capture * diffusion / (capture + diffusion)
        }
    }
}

/// The free energy (Hartree) of the bottleneck of `step` whose sides have the free energies
/// `ends` under `c`, for the prefactor `ln_pre` (`ln(1/s)`) of the rate theory: from the side
/// of the channel the rate constant is `k c°` (the association rate at the standard
/// concentration), and from the complex it follows by detailed balance. Also the derivatives
/// of that energy with respect to the two sides.
pub fn association_top(
    step: &AssociationStep,
    ends: [f64; 2],
    c: &Conditions,
    ln_pre: f64,
    label: &str,
) -> MoptResult<(f64, [f64; 2])> {
    let side = step.channel_side;
    let p = &step.pair;
    let ok = |x: f64| x.is_finite() && x > 0.0;
    if side > 1 || !p.radii.iter().chain(&p.masses).all(|x| ok(*x)) {
        return Err(MoptError::config(format!(
            "{label}: the association needs two fragments with positive radii and masses"
        )));
    }
    if c.viscosity.is_some_and(|eta| !ok(eta)) {
        return Err(MoptError::config("the viscosity must be positive (Pa s)"));
    }
    let k = association_rate(p, c.temperature, c.viscosity) * c.standard_concentration();
    let top = ends[side] + c.kt() * (ln_pre - k.ln());
    let mut slopes = [0.0; 2];
    slopes[side] = 1.0;
    Ok((top, slopes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};

    fn atom(z: u8) -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::from_atomic_number(z).unwrap()],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn the_volume_of_one_atom_is_its_sphere() {
        let ar = atom(18);
        let want = alvarez_vdw_radius_bohr(ar.elements[0]).unwrap();
        assert!((volume_radius(&ar) / want - 1.0).abs() < 5e-3);
    }

    #[test]
    fn the_diffusion_limit_of_equal_spheres_is_8_rt_over_3_eta() {
        let t = 298.15;
        let eta = 8.9e-4;
        for r in [2.0, 6.0] {
            let pair = EncounterPair {
                radii: [r, r],
                masses: [18.0, 18.0],
            };
            let want = 8.0 * Units::GAS_CONSTANT_J_MOL_K * t / (3.0 * eta) * 1.0e3;
            let got = diffusion_rate(&pair, t, eta);
            assert!((got / want - 1.0).abs() < 1e-6, "{got} {want}");
        }
        // About 7.4e9 L/mol/s in water at 25 C.
        let pair = EncounterPair {
            radii: [3.0, 3.0],
            masses: [18.0, 18.0],
        };
        let k = diffusion_rate(&pair, t, eta);
        assert!((7.3e9..7.5e9).contains(&k), "{k}");
        // In series with the capture the rate is below both and tends to the slower one.
        let both = association_rate(&pair, t, Some(eta));
        assert!(both < k && both < capture_rate(&pair, t));
        let thick = association_rate(&pair, t, Some(1.0));
        let slow = diffusion_rate(&pair, t, 1.0);
        assert!((thick / slow - 1.0).abs() < 1e-3);
    }

    #[test]
    fn the_capture_of_two_spheres() {
        let pair = EncounterPair {
            radii: [3.0, 4.0],
            masses: [16.0, 32.0],
        };
        let t = 300.0;
        let mu = 16.0 * 32.0 / 48.0 * Units::AMU_TO_KG;
        let r = 7.0 * Units::BOHR_TO_METER;
        let v = (8.0 * Units::BOLTZMANN_J_PER_K * t / (PI * mu)).sqrt();
        let want = PI * r * r * v * Units::AVOGADRO * 1.0e3;
        assert!((capture_rate(&pair, t) / want - 1.0).abs() < 1e-12);
        // Some 1e11 L/mol/s for small molecules.
        assert!((1e10..1e12).contains(&want), "{want}");
    }
}
