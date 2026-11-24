//! `cvt`: canonical variational transition state theory of a step with a reaction path
//! (`kinetics_beyond_tst.md`, stage 3). The dividing surface is put at the maximum of the
//! generalized free energy `G_GT(s, T)` along the path instead of at the saddle point, which
//! multiplies the flux of the step by
//!
//! ```text
//! exp(-(max_s G_GT(s, T) - G_GT(s_saddle, T)) / k_B T) <= 1
//! ```
//!
//! the factor `recrossing` of [`crate::vib::FreeEnergyProfile`] (the maximum from the parabola
//! through the highest point and its neighbours, [`crate::vib::variational_maximum`], as
//! `freqmain -path`). The generalized free energy is the energy plus the vibrational free
//! energy of the transverse modes (the thermochemistry of `freqmain`, with the low-frequency
//! treatment of `--qrrho`), from the frequencies along the path at any temperature, or the
//! profile of `free_energy_profile.csv` of `freqmain -path` at its temperature (within 1 K of
//! the run; the profile is then taken as it is at nearby temperatures). The factor is the same
//! in both directions (one dividing surface), so the step keeps detailed balance. A step
//! without a path keeps conventional TST, and its diagnostic `cvt_path = 0` says so.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::tst::QrrhoSettings;
use super::{Diagnostic, RateCorrection, StepContext};
use crate::domain::{positions, Bohr, Element, MolecularSystem, SpinCharge, Units};
use crate::error::{MoptError, MoptResult};
use crate::kinetics::conditions::Conditions;
use crate::kinetics::network::{Bottleneck, CvtPath, KineticNetwork};
use crate::vib::{thermochemistry, variational_maximum, NormalModes, ThermoSettings};

/// How far (K) the temperature of a given profile may be from that of the run.
pub const PROFILE_TEMPERATURE_TOLERANCE: f64 = 1.0;

/// The generalized free energies (Hartree) of `path` at `c` (energy plus the vibrational free
/// energy of the transverse modes).
pub fn generalized_free_energies(
    path: &CvtPath,
    qrrho: &QrrhoSettings,
    c: &Conditions,
    label: &str,
) -> MoptResult<Vec<f64>> {
    let n = path.arc_length.len();
    if n < 2 || path.energies.len() != n {
        return Err(MoptError::config(format!(
            "{label}: the path of cvt needs at least two points with an arc length and an energy \
             each ({} and {})",
            n,
            path.energies.len()
        )));
    }
    if let Some(freqs) = &path.frequencies {
        if freqs.len() != n {
            return Err(MoptError::config(format!(
                "{label}: the path of cvt has {} sets of frequencies for {n} points",
                freqs.len()
            )));
        }
        // The vibrational part does not depend on the structure; one atom carries it.
        let atom = MolecularSystem::new(
            vec![Element::H],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )?;
        let settings = ThermoSettings {
            temperature: c.temperature,
            pressure: c.pressure,
            quasi_harmonic: qrrho.method,
            cutoff: qrrho.cutoff,
            concentration: c.concentration,
            ..ThermoSettings::default()
        };
        return freqs
            .iter()
            .zip(&path.energies)
            .map(|(w, e)| {
                let modes = NormalModes {
                    wavenumbers: w.iter().copied().filter(|x| *x > 0.0).collect(),
                    curvatures: Vec::new(),
                    displacements: Vec::new(),
                    reduced_masses: Vec::new(),
                    force_constants: Vec::new(),
                    temperatures: Vec::new(),
                };
                Ok(e + thermochemistry(&atom, &modes, 0.0, &settings)?
                    .gibbs
                    .vibrational)
            })
            .collect();
    }
    match (&path.free_energies, path.temperature) {
        (Some(g), Some(t)) if g.len() == n => {
            if (t - c.temperature).abs() > PROFILE_TEMPERATURE_TOLERANCE {
                return Err(MoptError::config(format!(
                    "{label}: the profile of cvt ({}) is at {t} K, the run at {} K; give the \
                     frequencies along the path, or a profile at the temperature of the run",
                    path.source, c.temperature
                )));
            }
            Ok(g.clone())
        }
        (Some(_), None) => Err(MoptError::config(format!(
            "{label}: the profile of cvt ({}) has no temperature (give cvt_temperature)",
            path.source
        ))),
        _ => Err(MoptError::config(format!(
            "{label}: the path of cvt ({}) has neither transverse frequencies nor a free-energy \
             profile of the right length",
            path.source
        ))),
    }
}

/// `ln` of the factor of canonical variational TST of `path` at `c`, with the arc length of
/// the variational dividing surface and its free energy above the saddle point (Hartree).
pub fn cvt_ln_factor(
    path: &CvtPath,
    qrrho: &QrrhoSettings,
    c: &Conditions,
    label: &str,
) -> MoptResult<(f64, f64, f64)> {
    let g = generalized_free_energies(path, qrrho, c, label)?;
    let (saddle, _, at, top) = variational_maximum(&path.arc_length, &path.energies, &g);
    let shift = top - g[saddle];
    Ok((-shift / c.kt(), at, shift))
}

/// Canonical variational TST of the steps with a path.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cvt {
    /// The treatment of low frequencies of the transverse modes.
    pub qrrho: QrrhoSettings,
}

impl RateCorrection for Cvt {
    fn name(&self) -> &str {
        "cvt"
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        let Some(path) = &b.path else {
            // Conventional TST, said per step.
            return Ok(Some((0.0, vec![Diagnostic::new("cvt_path", 0.0)])));
        };
        let (ln_f, at, shift) = cvt_ln_factor(path, &self.qrrho, c, &b.label)?;
        Ok(Some((
            ln_f,
            vec![
                Diagnostic::new("cvt_path", 1.0),
                Diagnostic::new("cvt_arc_length", at),
                Diagnostic::new("cvt_shift_kcal_mol", shift * Units::HARTREE_TO_KCAL_MOL),
            ],
        )))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::vib::{
        free_energy_profile, transverse_profile, Direction, PathPoint, QuasiHarmonic,
    };
    use faer::{Col, Mat};

    /// The model of `vib::vtst`: a straight path along x with the barrier -0.005 x^2 and a
    /// transverse mode that stiffens with x.
    fn model() -> (Vec<PathPoint>, Vec<f64>) {
        let xs: Vec<f64> = (0..21).map(|k| -0.5 + 0.05 * k as f64).collect();
        let points = xs
            .iter()
            .map(|x| PathPoint {
                system: MolecularSystem::new(
                    vec![Element::H],
                    &positions::<Bohr>(&[[*x, 0.0, 0.0]]),
                    SpinCharge::default(),
                    None,
                )
                .unwrap(),
                tangent: Direction::Displacement(Col::from_fn(3, |i| f64::from(u8::from(i == 0)))),
                hessian: Mat::from_fn(3, 3, |i, j| match (i, j) {
                    (0, 0) => -1e-2,
                    (1, 1) => 0.2 + 0.1 * x,
                    (2, 2) => 0.3,
                    _ => 0.0,
                }),
            })
            .collect();
        (points, xs.iter().map(|x| -0.5e-2 * x * x).collect())
    }

    #[test]
    fn cvt_reproduces_the_variational_factor_of_vib_vtst() {
        let (points, energies) = model();
        let profile = transverse_profile(&points, false).unwrap();
        let qrrho = QrrhoSettings {
            method: QuasiHarmonic::None,
            ..QrrhoSettings::default()
        };
        for t in [200.0, 298.15, 600.0] {
            let settings = ThermoSettings {
                temperature: t,
                quasi_harmonic: QuasiHarmonic::None,
                ..ThermoSettings::default()
            };
            let free = free_energy_profile(&points, &energies, &profile, &settings, false).unwrap();
            let c = Conditions {
                temperature: t,
                ..Conditions::default()
            };
            // From the frequencies along the path.
            let path = CvtPath {
                arc_length: profile.arc_length.clone(),
                energies: energies.clone(),
                frequencies: Some(
                    profile
                        .modes
                        .iter()
                        .map(|m| m.wavenumbers.clone())
                        .collect(),
                ),
                ..CvtPath::default()
            };
            let (ln_f, at, _) = cvt_ln_factor(&path, &qrrho, &c, "TS").unwrap();
            assert!(
                (ln_f.exp() / free.recrossing - 1.0).abs() < 1e-9,
                "{t}: {} {}",
                ln_f.exp(),
                free.recrossing
            );
            assert!((at - free.variational_arc_length).abs() < 1e-12);
            assert!(ln_f < 0.0);
            // From the profile of freqmain -path at the same temperature.
            let given = CvtPath {
                arc_length: profile.arc_length.clone(),
                energies: energies.clone(),
                free_energies: Some(free.free_energy.clone()),
                temperature: Some(t),
                source: "free_energy_profile.csv".to_string(),
                ..CvtPath::default()
            };
            let (ln_g, _, _) = cvt_ln_factor(&given, &qrrho, &c, "TS").unwrap();
            assert!((ln_g - ln_f).abs() < 1e-12);
            // A profile at another temperature is refused.
            let far = Conditions {
                temperature: t + 50.0,
                ..c
            };
            assert!(cvt_ln_factor(&given, &qrrho, &far, "TS").is_err());
        }
        // A step without a path keeps TST and says so.
        let b = Bottleneck::saddle(0, "TS0", [0, 1], 0.0);
        let step = StepContext {
            barriers: [0.01, 0.01],
            zero_point: false,
        };
        let (f, d) = Cvt::default()
            .ln_factor(
                &b,
                &step,
                &KineticNetwork::default(),
                &Conditions::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(f, 0.0);
        assert_eq!(d[0].name, "cvt_path");
        assert_eq!(d[0].value, 0.0);
    }
}
