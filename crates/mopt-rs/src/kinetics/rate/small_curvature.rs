//! `small_curvature` and `zero_curvature`: tunnelling through the vibrationally adiabatic
//! ground-state barrier `V_a^G(s)` of a reaction path, with the effective mass of
//! small-curvature tunnelling (the centrifugal-dominant small-curvature semiclassical adiabatic
//! ground-state method of Lu et al., Comput. Phys. Commun. 1992, 71, 235, and Liu, Lynch,
//! Truong, Lu, Truhlar, Garrett, J. Am. Chem. Soc. 1993, 115, 2408) or without it (the mass of
//! the mass-weighted coordinates, zero-curvature tunnelling).
//!
//! The barrier and the masses are those of `freqmain -path` (`path_profile.csv`: the arc
//! length, `adiabatic_energy_hartree`, `sct_mass_ratio`); they do not depend on the
//! temperature, so the factor is computed at the temperature of the run by
//! [`crate::vib::transmission`]:
//!
//! ```text
//! kappa(T) = 1 + (2 / k T) int_0^(V_max - E_0) P(V_max - e) sinh(e / k T) de
//! ```
//!
//! with the threshold at the top of `V_a^G` and the energies counted down to the higher of the
//! two ends of the path. The factor is referred to the top of the barrier and is the same from
//! both sides, so detailed balance is kept. It is the transmission coefficient of CVT/SCT when
//! used with `cvt` (the dividing surface at the maximum of the free-energy profile), and the
//! same factor with conventional TST (where the top of `V_a^G` is close to the saddle point).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::{Diagnostic, RateCorrection, StepContext};
use crate::error::MoptResult;
use crate::kinetics::conditions::Conditions;
use crate::kinetics::network::{Bottleneck, KineticNetwork};
use crate::vib::transmission;

/// Tunnelling along the path of a step.
#[derive(Debug, Clone, Copy)]
pub struct CurvatureTunnelling {
    /// With the effective mass of small-curvature tunnelling (else zero curvature).
    pub small_curvature: bool,
}

impl RateCorrection for CurvatureTunnelling {
    fn name(&self) -> &str {
        if self.small_curvature {
            "small_curvature"
        } else {
            "zero_curvature"
        }
    }

    fn ln_factor(
        &self,
        b: &Bottleneck,
        _step: &StepContext,
        _net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        let Some(path) = &b.tunnelling_path else {
            return Ok(None);
        };
        let straight = vec![1.0; path.arc_length.len()];
        let zct = transmission(&path.arc_length, &path.adiabatic, &straight, c.temperature)
            .map_err(|e| e.context(format!("{} ({})", b.label, path.source)))?;
        let sct = transmission(
            &path.arc_length,
            &path.adiabatic,
            &path.mass_ratio,
            c.temperature,
        )
        .map_err(|e| e.context(format!("{} ({})", b.label, path.source)))?;
        let kappa = if self.small_curvature { sct } else { zct };
        Ok(Some((
            kappa.ln(),
            vec![
                Diagnostic::new("sct_kappa", sct),
                Diagnostic::new("zct_kappa", zct),
            ],
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Units;
    use crate::kinetics::network::TunnellingPath;
    use crate::kinetics::rate::tunneling::bell_ln_kappa;

    #[test]
    fn a_parabolic_adiabatic_barrier_tunnels_as_bell_s_parabola() {
        // V = -k s^2 / 2 with the frequency 1000 cm^-1 (mass 1 amu, mass-weighted s).
        let wavenumber: f64 = 1000.0;
        // hbar omega in Hartree (= omega in atomic units).
        let omega =
            Units::PLANCK_J_S * Units::SPEED_OF_LIGHT_CM_S * wavenumber / Units::HARTREE_TO_JOULE;
        let k = omega * omega * Units::AMU_TO_AU_MASS;
        let reach = (2.0 * 0.05 / k).sqrt();
        let s: Vec<f64> = (0..121)
            .map(|i| reach * (2.0 * i as f64 / 120.0 - 1.0))
            .collect();
        let v: Vec<f64> = s.iter().map(|x| -0.5 * k * x * x).collect();
        let mut b = Bottleneck::saddle(0, "TS", [0, 1], 0.0);
        b.tunnelling_path = Some(TunnellingPath {
            arc_length: s.clone(),
            adiabatic: v,
            mass_ratio: vec![0.5; s.len()],
            source: "test".into(),
        });
        let c = Conditions {
            temperature: 500.0,
            ..Conditions::default()
        };
        let step = StepContext {
            barriers: [0.05, 0.05],
            zero_point: false,
        };
        let net = KineticNetwork::default();
        let (ln_zct, _) = CurvatureTunnelling {
            small_curvature: false,
        }
        .ln_factor(&b, &step, &net, &c)
        .unwrap()
        .unwrap();
        let (ln_sct, d) = CurvatureTunnelling {
            small_curvature: true,
        }
        .ln_factor(&b, &step, &net, &c)
        .unwrap()
        .unwrap();
        // Zero curvature: the parabola (u / 2) / sin(u / 2) of Bell; a lighter mass tunnels
        // more.
        let bell = bell_ln_kappa(omega / (Units::BOLTZMANN_HARTREE_PER_K * 500.0)).unwrap();
        assert!((ln_zct - bell).abs() < 5e-3, "{ln_zct} {bell}");
        assert!(ln_sct > ln_zct);
        assert_eq!(d.len(), 2);
        // Without a path the factor does not apply.
        b.tunnelling_path = None;
        assert!(CurvatureTunnelling {
            small_curvature: true
        }
        .ln_factor(&b, &step, &net, &c)
        .unwrap()
        .is_none());
    }
}
