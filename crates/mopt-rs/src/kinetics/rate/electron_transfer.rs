//! Electron transfer rate constants from EDEEL quantities (Mutsuji, Saita, Maeda, RSC Adv. 2023,
//! 13, 32097; doi:10.1039/D3RA05784D). Equation numbers refer to that paper.
//!
//! Energies and Gibbs energies are per molecule in Hartree. Rate constants are in
//! `(dm^3 mol^-1)^q s^-1`, with `q = 1` for intermolecular and `q = 0` for intramolecular transfer.
//!
//! The same formulas give the electron-transfer steps of a kinetic network
//! ([`transfer_barrier`]): the free energies of the two states fix `dG0`, the step has the
//! reorganization energy `lambda` and the coupling `H_ab`, and either the non-adiabatic
//! Marcus-Levich rate constant `k = (2 |H_ab|^2 / h) (pi^3 / (lambda k_B T))^(1/2)
//! exp(-(lambda + dG0)^2 / (4 lambda k_B T))` (eqn (5)) or the adiabatic one, TST on the lower
//! surface of the two-state model (eqn (9)) with the parabolas `lambda x^2` and
//! `dG0 + lambda (1 - x)^2`, or the Zhu-Nakamura rate of the crossing of the two parabolas
//! along a mode of a given wavenumber (`super::inner_sphere::zhu_nakamura_at_crossing`: the
//! thermal transmission of the crossing, tunnelling below it included, with the coupling).
//! Each is written as the free energy of one bottleneck and a prefactor, so that the two
//! directions balance (`k_f / k_b = exp(-dG0 / k_B T)`).
//! (Formerly `analysis::electron_transfer`, which re-exports this module.)
//!
//! MultiOptPy: `optimization.py:EDEELHandler` (EDEEL quantities); the rate constants are new in
//!  mopt-rs.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::inner_sphere::{zhu_nakamura_at_crossing, InnerSphere};
use crate::domain::{Hartree, Quantity, Units};
use crate::error::{MoptError, MoptResult};

/// Conditions of the rate constant evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferConditions {
    /// Temperature (K); the paper uses 298.15 K.
    pub temperature: f64,
    /// Standard pressure `p0` (Pa); the paper uses 1 atm.
    pub standard_pressure: f64,
    /// Intermolecular transfer (`q = 1` in eqn (1)) or intramolecular (`q = 0`).
    pub intermolecular: bool,
    /// Effective nuclear frequency `nu_n` (1/s); `k_B T / h` when `None`, as in the paper.
    pub nuclear_frequency: Option<f64>,
}

impl Default for TransferConditions {
    fn default() -> Self {
        Self {
            temperature: 298.15,
            standard_pressure: 101_325.0,
            intermolecular: true,
            nuclear_frequency: None,
        }
    }
}

impl TransferConditions {
    fn check(&self) -> MoptResult<()> {
        let ok = |v: f64| v.is_finite() && v > 0.0;
        if !ok(self.temperature)
            || !ok(self.standard_pressure)
            || self.nuclear_frequency.is_some_and(|v| !ok(v))
        {
            return Err(MoptError::config(
                "temperature, pressure, and nuclear frequency must be positive",
            ));
        }
        Ok(())
    }

    /// `k_B T` in Hartree.
    fn kt(&self) -> f64 {
        Units::BOLTZMANN_HARTREE_PER_K * self.temperature
    }

    /// `nu_n` in 1/s.
    pub fn nu(&self) -> f64 {
        self.nuclear_frequency
            .unwrap_or(Units::BOLTZMANN_J_PER_K * self.temperature / Units::PLANCK_J_S)
    }

    /// `(RT / p0)^q` in `(dm^3 / mol)^q`.
    fn volume_factor(&self) -> f64 {
        if self.intermolecular {
            Units::GAS_CONSTANT_J_MOL_K * self.temperature / self.standard_pressure * 1.0e3
        } else {
            1.0
        }
    }
}

fn positive(v: Quantity<Hartree>, what: &str) -> MoptResult<f64> {
    let x = v.value();
    if x.is_finite() && x > 0.0 {
        Ok(x)
    } else {
        Err(MoptError::config(format!(
            "{what} must be positive, got {x}"
        )))
    }
}

/// Diabatic coupling `V12 = V11 - E^C_{n+m+1}` at the crossing `V11 = V22`, where
/// `E^C_{n+m+1}` is the adiabatic energy of the complex with the moving electron; at the crossing
/// this reproduces the two-state relation of eqn (9).
pub fn edeel_coupling(
    v11: Quantity<Hartree>,
    complex_with: Quantity<Hartree>,
) -> Quantity<Hartree> {
    v11 - complex_with
}

/// Lower adiabatic energy of the two-state model, eqn (9):
/// `E = (V11 + V22)/2 - sqrt((V11 - V22)^2 + 4 V12^2) / 2`.
pub fn two_state_lower(
    v11: Quantity<Hartree>,
    v22: Quantity<Hartree>,
    v12: Quantity<Hartree>,
) -> Quantity<Hartree> {
    let (a, b, c) = (v11.value(), v22.value(), v12.value());
    Quantity::new(0.5 * (a + b) - 0.5 * ((a - b).powi(2) + 4.0 * c * c).sqrt())
}

/// Non-adiabatic Gibbs energy of activation in the harmonic (Marcus) picture, eqn (6):
/// `dG_na = (lambda + dG0)^2 / (4 lambda)`.
pub fn nonadiabatic_barrier(
    lambda: Quantity<Hartree>,
    dg0: Quantity<Hartree>,
) -> MoptResult<Quantity<Hartree>> {
    let l = positive(lambda, "the reorganization energy")?;
    Ok(Quantity::new((l + dg0.value()).powi(2) / (4.0 * l)))
}

/// Reorganization energy from the Gibbs energies of the crossing and of the initial and final
/// equilibria, eqn (10):
/// `lambda = 2 G_SX - G_i - G_f + 2 sqrt((G_SX - G_i)(G_SX - G_f))`
/// (`lambda = 4 dG_na` for self-exchange).
pub fn reorganization_energy(
    g_sx: Quantity<Hartree>,
    g_initial: Quantity<Hartree>,
    g_final: Quantity<Hartree>,
) -> MoptResult<Quantity<Hartree>> {
    let (sx, i, f) = (g_sx.value(), g_initial.value(), g_final.value());
    if sx < i || sx < f {
        return Err(MoptError::config(
            "the crossing must lie above both equilibria to define a reorganization energy",
        ));
    }
    Ok(Quantity::new(
        2.0 * sx - i - f + 2.0 * ((sx - i) * (sx - f)).sqrt(),
    ))
}

/// Landau-Zener transition probability, eqn (3) and (4):
/// `P_LZ = 1 - exp(-2 pi gamma)`, `2 pi gamma = pi^(3/2) |V12|^2 / (h nu_n sqrt(lambda k_B T))`.
pub fn landau_zener_probability(
    v12: Quantity<Hartree>,
    lambda: Quantity<Hartree>,
    cond: &TransferConditions,
) -> MoptResult<f64> {
    cond.check()?;
    let l = positive(lambda, "the reorganization energy")?;
    let h_nu = Units::PLANCK_HARTREE_S * cond.nu();
    let two_pi_gamma = PI.powf(1.5) * v12.value().powi(2) / (h_nu * (l * cond.kt()).sqrt());
    Ok(1.0 - (-two_pi_gamma).exp())
}

/// Electronic transmission coefficient, eqn (2): `2 P / (1 + P)` in the normal region
/// (`dG0 >= -lambda`) and `2 P (1 - P)` in the inverted region.
pub fn transmission_coefficient(
    p_lz: f64,
    dg0: Quantity<Hartree>,
    lambda: Quantity<Hartree>,
) -> f64 {
    if dg0.value() >= -lambda.value() {
        2.0 * p_lz / (1.0 + p_lz)
    } else {
        2.0 * p_lz * (1.0 - p_lz)
    }
}

/// Rate constant, eqn (1): `k = (RT/p0)^q kappa_el nu_n exp(-dG / RT)`, with the Gibbs energy
/// of activation `dG` on the adiabatic surface.
pub fn rate_constant(
    dg_activation: Quantity<Hartree>,
    kappa_el: f64,
    cond: &TransferConditions,
) -> MoptResult<f64> {
    cond.check()?;
    Ok(cond.volume_factor() * kappa_el * cond.nu() * (-dg_activation.value() / cond.kt()).exp())
}

/// Marcus rate constant, eqn (5), the small-coupling limit of eqn (1)-(4):
/// `k = (RT/p0)^q (2 |V12|^2 / h) (pi^3 / (lambda k_B T))^(1/2) exp(-dG_na / RT)`.
pub fn marcus_rate_constant(
    v12: Quantity<Hartree>,
    lambda: Quantity<Hartree>,
    dg_na: Quantity<Hartree>,
    cond: &TransferConditions,
) -> MoptResult<f64> {
    cond.check()?;
    let l = positive(lambda, "the reorganization energy")?;
    let kt = cond.kt();
    Ok(
        cond.volume_factor() * 2.0 * v12.value().powi(2) / Units::PLANCK_HARTREE_S
            * (PI.powi(3) / (l * kt)).sqrt()
            * (-dg_na.value() / kt).exp(),
    )
}

/// Electron transfer quantities at one crossing geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferRate {
    /// Reorganization energy (eqn (10)).
    pub lambda: Quantity<Hartree>,
    /// Non-adiabatic Gibbs energy of activation `G_SX - G_initial`.
    pub dg_nonadiabatic: Quantity<Hartree>,
    /// Gibbs energy of activation on the adiabatic surface.
    pub dg_activation: Quantity<Hartree>,
    /// Landau-Zener probability (eqn (3), (4)).
    pub p_lz: f64,
    /// Electronic transmission coefficient (eqn (2)).
    pub kappa_el: f64,
    /// Rate constant (eqn (1)).
    pub rate: f64,
}

/// Rate constant of the EDEEL scheme at a crossing geometry.
///
/// `g_sx`, `g_initial`, and `g_final` are the Gibbs energies of the crossing (on the diabatic
/// surfaces) and of the initial and final equilibria. The adiabatic ground state at the crossing
/// is taken as the transition state, so its Gibbs energy is `G_SX - V12` (eqn (9) at
/// `V11 = V22`).
pub fn edeel_rate(
    g_sx: Quantity<Hartree>,
    g_initial: Quantity<Hartree>,
    g_final: Quantity<Hartree>,
    v12: Quantity<Hartree>,
    cond: &TransferConditions,
) -> MoptResult<TransferRate> {
    let lambda = reorganization_energy(g_sx, g_initial, g_final)?;
    let dg0 = g_final - g_initial;
    let dg_nonadiabatic = g_sx - g_initial;
    let dg_activation = dg_nonadiabatic - v12;
    let p_lz = landau_zener_probability(v12, lambda, cond)?;
    let kappa_el = transmission_coefficient(p_lz, dg0, lambda);
    let rate = rate_constant(dg_activation, kappa_el, cond)?;
    Ok(TransferRate {
        lambda,
        dg_nonadiabatic,
        dg_activation,
        p_lz,
        kappa_el,
        rate,
    })
}

/// The form of an electron-transfer step of a kinetic network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferForm {
    /// Marcus-Levich: the golden-rule rate constant of eqn (5).
    Nonadiabatic,
    /// TST on the lower surface of the two-state model (eqn (9)).
    Adiabatic,
    /// The Zhu-Nakamura rate of the crossing of the two parabolas along one mode (new in
    /// mopt-rs): the thermal transmission of the crossing, from the weak-coupling (Marcus)
    /// to the adiabatic limit, with the tunnelling of the mode.
    ZhuNakamura,
}

impl TransferForm {
    /// `nonadiabatic` (or `marcus`), `adiabatic`, or `zhu_nakamura` (or `zn`).
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "nonadiabatic" | "non_adiabatic" | "marcus" | "marcus_levich" => Ok(Self::Nonadiabatic),
            "adiabatic" => Ok(Self::Adiabatic),
            "zhu_nakamura" | "zn" => Ok(Self::ZhuNakamura),
            other => Err(MoptError::parse(format!(
                "unknown form of an electron transfer '{other}' (nonadiabatic, adiabatic, \
                 zhu_nakamura)"
            ))),
        }
    }

    /// The name of the rate theory of the step.
    pub fn name(self) -> &'static str {
        match self {
            Self::Nonadiabatic => "marcus_nonadiabatic",
            Self::Adiabatic => "marcus_adiabatic",
            Self::ZhuNakamura => "zhu_nakamura_transfer",
        }
    }
}

/// An electron-transfer step of a kinetic network.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferStep {
    /// The reorganization energy `lambda` (Hartree).
    pub reorganization: f64,
    /// The form.
    pub form: TransferForm,
    /// `dG0` as given (Hartree); it must be the difference of the free energies of the two
    /// states, which is used.
    pub delta_g0: Option<f64>,
    /// The wavenumber (1/cm) of the mode along which the parabolas cross, for the
    /// Zhu-Nakamura form (the inner-sphere mode of `etmain`, say).
    pub wavenumber: Option<f64>,
}

/// The free energy of the Zhu-Nakamura bottleneck above the first side, `dE_X - k_B T ln
/// kappa`, of parabolas of the reorganization `lambda` and the reaction free energy `dg0`
/// along a mode of the quantum `quantum` (Hartree; mass-weighted, 1 amu), with the coupling
/// `coupling`, at `kt` (all Hartree).
fn zhu_nakamura_bottleneck(
    lambda: f64,
    dg0: f64,
    coupling: f64,
    quantum: f64,
    kt: f64,
) -> MoptResult<f64> {
    // V1 = k q^2 / 2, V2 = dG0 + k (q - d)^2 / 2 with lambda = k d^2 / 2.
    let k = quantum * quantum * Units::AMU_TO_AU_MASS;
    let d = (2.0 * lambda / k).sqrt();
    let q_crossing = (lambda + dg0) / (k * d);
    let barrier = (lambda + dg0).powi(2) / (4.0 * lambda);
    let inner = InnerSphere {
        quantum,
        reaction_energy: dg0,
        reorganization: lambda,
        q_crossing,
        q_final: d,
        slopes: [k * q_crossing, k * (q_crossing - d)],
    };
    let zn = zhu_nakamura_at_crossing(&inner, coupling, barrier, barrier, kt, 1.0)?;
    // The free energies of the two states hold the mode; the parabolas leave it out.
    Ok(barrier - kt * zn.ln_kappa + kt * zn.ln_partition)
}

/// The bottleneck of an electron-transfer step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferBarrier {
    /// Its free energy (Hartree), in the frame of the free energies of the two sides.
    pub top: f64,
    /// `ln` of the prefactor (1/s).
    pub ln_prefactor: f64,
    /// `d top / d ends` (for the derivatives).
    pub slopes: [f64; 2],
    /// The point of the bottleneck on the coordinate from the first side (0) to the second (1).
    pub at: f64,
}

/// The bottleneck of the electron-transfer step `step` between the free energies `ends` of its
/// two sides (Hartree) with the coupling `coupling` (Hartree) at `temperature` (K).
pub fn transfer_barrier(
    step: &TransferStep,
    coupling: Option<f64>,
    ends: [f64; 2],
    temperature: f64,
    label: &str,
) -> MoptResult<TransferBarrier> {
    let lambda = step.reorganization;
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(MoptError::config(format!(
            "{label}: the reorganization energy must be positive (got {lambda} Hartree)"
        )));
    }
    let h = coupling.ok_or_else(|| {
        MoptError::config(format!(
            "{label}: an electron transfer needs the coupling H_ab (coupling_hartree or \
             coupling_cm1 of the annotations, e.g. from freqmain -diabatize)"
        ))
    })?;
    let dg0 = ends[1] - ends[0];
    if let Some(given) = step.delta_g0 {
        if (given - dg0).abs() > 1e-6 {
            return Err(MoptError::config(format!(
                "{label}: dG0 = {given} Hartree is given, but the free energies of the two \
                 states differ by {dg0} Hartree; detailed balance needs the same number (correct \
                 the energies of the states or leave dG0 out)"
            )));
        }
    }
    let cond = TransferConditions {
        temperature,
        intermolecular: false,
        ..TransferConditions::default()
    };
    match step.form {
        TransferForm::Nonadiabatic => {
            if h.is_nan() || h == 0.0 {
                return Err(MoptError::config(format!(
                    "{label}: the non-adiabatic rate needs a coupling different from 0"
                )));
            }
            let barrier = nonadiabatic_barrier(Quantity::new(lambda), Quantity::new(dg0))?;
            let prefactor = marcus_rate_constant(
                Quantity::new(h),
                Quantity::new(lambda),
                Quantity::new(0.0),
                &cond,
            )?;
            let at = (lambda + dg0) / (2.0 * lambda);
            Ok(TransferBarrier {
                top: ends[0] + barrier.value(),
                ln_prefactor: prefactor.ln(),
                slopes: [1.0 - at, at],
                at,
            })
        }
        TransferForm::Adiabatic => {
            if dg0.abs() > lambda {
                return Err(MoptError::config(format!(
                    "{label}: |dG0| = {} Hartree exceeds the reorganization energy (the inverted \
                     region): one of the states is not a minimum of the lower adiabatic surface; \
                     use the non-adiabatic form",
                    dg0.abs()
                )));
            }
            let lower = |x: f64| {
                let v1 = ends[0] + lambda * x * x;
                let v2 = ends[1] + lambda * (1.0 - x) * (1.0 - x);
                (
                    two_state_lower(Quantity::new(v1), Quantity::new(v2), Quantity::new(h)).value(),
                    v1 - v2,
                )
            };
            // The highest point of the lower surface between the two minima.
            let samples = 256;
            let mut best = 0;
            for k in 1..=samples {
                if lower(k as f64 / samples as f64).0 > lower(best as f64 / samples as f64).0 {
                    best = k;
                }
            }
            let (mut lo, mut hi) = (
                (best as f64 - 1.0).max(0.0) / samples as f64,
                (best as f64 + 1.0).min(samples as f64) / samples as f64,
            );
            let golden = 0.5 * (5f64.sqrt() - 1.0);
            for _ in 0..200 {
                if hi - lo < 1e-13 {
                    break;
                }
                let a = hi - golden * (hi - lo);
                let b = lo + golden * (hi - lo);
                if lower(a).0 < lower(b).0 {
                    lo = a;
                } else {
                    hi = b;
                }
            }
            let at = 0.5 * (lo + hi);
            let (top, d) = lower(at);
            let r = (d * d + 4.0 * h * h).sqrt();
            let slopes = if r > 0.0 {
                [0.5 * (1.0 - d / r), 0.5 * (1.0 + d / r)]
            } else {
                [0.5, 0.5]
            };
            Ok(TransferBarrier {
                top,
                ln_prefactor: (Units::BOLTZMANN_J_PER_K * temperature / Units::PLANCK_J_S).ln(),
                slopes,
                at,
            })
        }
        TransferForm::ZhuNakamura => {
            let wavenumber = step.wavenumber.ok_or_else(|| {
                MoptError::config(format!(
                    "{label}: the Zhu-Nakamura form needs the wavenumber of the mode along \
                     which the states cross (transfer_wavenumber_cm1; the inner-sphere mode \
                     nu_in_cm1 of etmain)"
                ))
            })?;
            if !(wavenumber.is_finite() && wavenumber > 0.0) || h.is_nan() || h == 0.0 {
                return Err(MoptError::config(format!(
                    "{label}: the Zhu-Nakamura form needs a positive wavenumber and a coupling \
                     different from 0"
                )));
            }
            let quantum = wavenumber * Units::PLANCK_J_S * Units::SPEED_OF_LIGHT_CM_S
                / Units::HARTREE_TO_JOULE;
            let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature;
            let bottleneck = |g: f64| zhu_nakamura_bottleneck(lambda, g, h, quantum, kt);
            let above = bottleneck(dg0)
                .map_err(|e| MoptError::config(format!("{label}: {e} (dG0 = {dg0} Hartree)")))?;
            // The slopes for the derivatives: the transmission depends on dG0 too.
            let step_g = 1e-6 * lambda;
            let slope = match (bottleneck(dg0 + step_g), bottleneck(dg0 - step_g)) {
                (Ok(up), Ok(down)) => (up - down) / (2.0 * step_g),
                _ => (lambda + dg0) / (2.0 * lambda),
            };
            Ok(TransferBarrier {
                top: ends[0] + above,
                ln_prefactor: (Units::BOLTZMANN_J_PER_K * temperature / Units::PLANCK_J_S).ln(),
                slopes: [1.0 - slope, slope],
                at: (lambda + dg0) / (2.0 * lambda),
            })
        }
    }
}

/// Activation strain decomposition at the crossing, eqn (11)-(13) (electronic energies).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationStrain {
    /// `E^D(SX)_{n+1} - E^D(initial-EQ)_{n+1}`.
    pub strain_donor: Quantity<Hartree>,
    /// `E^A(SX)_m - E^A(initial-EQ)_m`.
    pub strain_acceptor: Quantity<Hartree>,
    /// `E^C(SX)_{n+m} - E^D(SX)_n - E^A(SX)_m`.
    pub interaction: Quantity<Hartree>,
}

/// Energies entering the activation strain decomposition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrainEnergies {
    /// Donor with the moving electron at the crossing.
    pub donor_with_sx: Quantity<Hartree>,
    /// Donor with the moving electron at its isolated equilibrium.
    pub donor_with_eq: Quantity<Hartree>,
    /// Donor without the moving electron at the crossing.
    pub donor_without_sx: Quantity<Hartree>,
    /// Acceptor without the moving electron at the crossing.
    pub acceptor_without_sx: Quantity<Hartree>,
    /// Acceptor without the moving electron at its isolated equilibrium.
    pub acceptor_without_eq: Quantity<Hartree>,
    /// Complex without the moving electron at the crossing.
    pub complex_without_sx: Quantity<Hartree>,
}

/// Activation strain model analysis of eqn (11)-(13).
pub fn activation_strain(e: &StrainEnergies) -> ActivationStrain {
    ActivationStrain {
        strain_donor: e.donor_with_sx - e.donor_with_eq,
        strain_acceptor: e.acceptor_without_sx - e.acceptor_without_eq,
        interaction: e.complex_without_sx - e.donor_without_sx - e.acceptor_without_sx,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn transfer_steps_have_the_marcus_parabola_and_balance() {
        let t = 298.15;
        let kt = Units::BOLTZMANN_HARTREE_PER_K * t;
        let lambda = 0.03;
        let h = 2e-4;
        let step = TransferStep {
            reorganization: lambda,
            form: TransferForm::Nonadiabatic,
            delta_g0: None,
            wavenumber: None,
        };
        let rate = |dg0: f64| {
            let b = transfer_barrier(&step, Some(h), [0.0, dg0], t, "ET").unwrap();
            (
                b.ln_prefactor - (b.top - 0.0) / kt,
                b.ln_prefactor - (b.top - dg0) / kt,
                b,
            )
        };
        // The prefactor is that of eqn (5) of the EDEEL scheme (Marcus-Levich).
        let cond = TransferConditions {
            temperature: t,
            intermolecular: false,
            ..TransferConditions::default()
        };
        let (f0, b0, _) = rate(0.0);
        let want = marcus_rate_constant(
            Quantity::new(h),
            Quantity::new(lambda),
            Quantity::new(lambda / 4.0),
            &cond,
        )
        .unwrap();
        assert!((f0.exp() / want - 1.0).abs() < 1e-12);
        assert!((f0 - b0).abs() < 1e-12);
        // Activationless at dG0 = -lambda: the rate is the prefactor.
        let (fa, ba, at) = rate(-lambda);
        assert!((fa - at.ln_prefactor).abs() < 1e-12);
        assert!((at.at).abs() < 1e-15);
        // The inverted region: the parabola is symmetric about -lambda, so dG0 = -2 lambda is
        // as slow as dG0 = 0; deeper is slower still.
        let (fi, bi, _) = rate(-2.0 * lambda);
        assert!((fi - f0).abs() < 1e-10, "{fi} {f0}");
        let (fd, _, _) = rate(-3.0 * lambda);
        assert!(fd < fi && fi < fa);
        // Detailed balance: k_f / k_b = exp(-dG0 / k T) everywhere.
        for (dg0, f, b) in [(0.0, f0, b0), (-lambda, fa, ba), (-2.0 * lambda, fi, bi)] {
            assert!(((f - b) - (-dg0 / kt)).abs() < 1e-10);
        }
        // The slopes of the top (for the derivatives) against differences.
        let d = 1e-7;
        let base = transfer_barrier(&step, Some(h), [0.0, -0.01], t, "ET").unwrap();
        let up0 = transfer_barrier(&step, Some(h), [d, -0.01], t, "ET").unwrap();
        let up1 = transfer_barrier(&step, Some(h), [0.0, -0.01 + d], t, "ET").unwrap();
        assert!(((up0.top - base.top) / d - base.slopes[0]).abs() < 1e-6);
        assert!(((up1.top - base.top) / d - base.slopes[1]).abs() < 1e-6);
        // A given dG0 must be that of the states; a coupling is needed.
        let wrong = TransferStep {
            delta_g0: Some(0.005),
            ..step
        };
        assert!(transfer_barrier(&wrong, Some(h), [0.0, 0.0], t, "ET").is_err());
        assert!(transfer_barrier(&step, None, [0.0, 0.0], t, "ET").is_err());
    }

    #[test]
    fn the_adiabatic_transfer_has_the_barrier_of_the_two_state_model() {
        let t = 298.15;
        let lambda = 0.04;
        let step = TransferStep {
            reorganization: lambda,
            form: TransferForm::Adiabatic,
            delta_g0: None,
            wavenumber: None,
        };
        // Symmetric: lambda / 4 - H_ab at x = 1/2 for H_ab < lambda / 2.
        for h in [1e-4, 2e-3, 8e-3] {
            let b = transfer_barrier(&step, Some(h), [0.0, 0.0], t, "ET").unwrap();
            assert!((b.top - (lambda / 4.0 - h)).abs() < 1e-12, "{h}: {}", b.top);
            assert!((b.at - 0.5).abs() < 1e-6);
            assert!(
                // The maximum is found to about 1e-10 in x, which the sharp top at small H_ab
                // turns into 1e-7 in the slopes.
                (b.slopes[0] - 0.5).abs() < 1e-6 && (b.slopes[1] - 0.5).abs() < 1e-6,
                "{h}: {:?} {}",
                b.slopes,
                b.at
            );
        }
        // Small coupling: the crossing of the parabolas less H_ab sqrt(1 - (dG0 / lambda)^2)
        // (the maximum of the lower surface of two crossing lines of slopes lambda +- dG0;
        // corrections of the order of H_ab^2 / lambda).
        let dg0 = -0.01;
        let h = 1e-5;
        let b = transfer_barrier(&step, Some(h), [0.0, dg0], t, "ET").unwrap();
        let crossing = (lambda + dg0).powi(2) / (4.0 * lambda);
        let want = crossing - h * (1.0 - (dg0 / lambda).powi(2)).sqrt();
        assert!((b.top - want).abs() < 1e-8, "{} {want}", b.top);
        // The slopes are the envelope derivatives.
        let d = 1e-7;
        let up0 = transfer_barrier(&step, Some(h), [d, dg0], t, "ET").unwrap();
        assert!(((up0.top - b.top) / d - b.slopes[0]).abs() < 1e-5);
        // The inverted region has no adiabatic barrier between the minima.
        assert!(transfer_barrier(&step, Some(h), [0.0, -2.0 * lambda], t, "ET").is_err());
    }

    #[test]
    fn the_zhu_nakamura_transfer_spans_the_two_limits_and_balances() {
        let t = 298.15;
        let kt = Units::BOLTZMANN_HARTREE_PER_K * t;
        let lambda = 0.03;
        let zn = TransferStep {
            reorganization: lambda,
            form: TransferForm::ZhuNakamura,
            delta_g0: None,
            // A soft mode: little tunnelling, so that the limits are the classical ones.
            wavenumber: Some(60.0),
        };
        let marcus = TransferStep {
            form: TransferForm::Nonadiabatic,
            wavenumber: None,
            ..zn
        };
        let adiabatic = TransferStep {
            form: TransferForm::Adiabatic,
            wavenumber: None,
            ..zn
        };
        let ln_k = |step: &TransferStep, h: f64, dg0: f64| {
            let b = transfer_barrier(step, Some(h), [0.0, dg0], t, "ET").unwrap();
            (
                b.ln_prefactor - b.top / kt,
                b.ln_prefactor - (b.top - dg0) / kt,
                b,
            )
        };
        // Weak coupling in the normal region (a peaked crossing): the Marcus rate.
        for dg0 in [0.0, -0.01] {
            let (z, _, _) = ln_k(&zn, 1e-6, dg0);
            let (m, _, _) = ln_k(&marcus, 1e-6, dg0);
            assert!((z - m).abs() < 0.1, "dG0 {dg0}: ZN {z} vs Marcus {m}");
        }
        // The inverted region (a sloped crossing): the energies below the crossing are not
        // built for that type, and at this weak coupling the formula above the crossing
        // departs from Landau-Zener over many k_B T, so the rate stays below Marcus's (by a
        // factor of about 4.5 here; docs/source/electron_transfer.md).
        let (z, _, _) = ln_k(&zn, 1e-6, -0.05);
        let (m, _, _) = ln_k(&marcus, 1e-6, -0.05);
        assert!(z < m && z > m - 3.0, "inverted: ZN {z} vs Marcus {m}");
        // Strong coupling: transition state theory over the lower surface, with the mode's
        // partition function in the well (the adiabatic form has k_B T / h alone).
        let (z, _, _) = ln_k(&zn, 3e-3, 0.0);
        let (a, _, _) = ln_k(&adiabatic, 3e-3, 0.0);
        let quantum =
            60.0 * Units::PLANCK_J_S * Units::SPEED_OF_LIGHT_CM_S / Units::HARTREE_TO_JOULE;
        let ln_q = super::super::inner_sphere::ln_mode_partition(quantum, kt);
        assert!(
            (z - (a - ln_q)).abs() < 0.3,
            "ZN {z} vs adiabatic {a} - ln Q {ln_q}"
        );
        // Detailed balance and the slopes of the top.
        for (h, dg0) in [(1e-5, -0.012), (2e-3, 0.004), (1e-4, -0.045)] {
            let (f, b, base) = ln_k(&zn, h, dg0);
            assert!(((f - b) + dg0 / kt).abs() < 1e-10);
            let d = 1e-7;
            let up0 = transfer_barrier(&zn, Some(h), [d, dg0], t, "ET").unwrap();
            let up1 = transfer_barrier(&zn, Some(h), [0.0, dg0 + d], t, "ET").unwrap();
            assert!(((up0.top - base.top) / d - base.slopes[0]).abs() < 1e-4);
            assert!(((up1.top - base.top) / d - base.slopes[1]).abs() < 1e-4);
        }
        // The wavenumber is needed.
        let without = TransferStep {
            wavenumber: None,
            ..zn
        };
        assert!(transfer_barrier(&without, Some(1e-4), [0.0, 0.0], t, "ET").is_err());
        assert_eq!(
            TransferForm::parse("zn").unwrap(),
            TransferForm::ZhuNakamura
        );
    }

    fn ev(x: f64) -> Quantity<Hartree> {
        Quantity::new(x * Units::EV_TO_HARTREE)
    }

    #[test]
    fn self_exchange_relations() {
        // lambda = 4 dG_na for dG0 = 0 (paper, below eqn (10)), and eqn (10) inverts eqn (6).
        let l = reorganization_energy(ev(1.0), ev(0.0), ev(0.0)).unwrap();
        assert!((l.value() - ev(4.0).value()).abs() < 1e-15);
        for (lam, dg0) in [(1.0, 0.3), (2.0, -0.5), (0.8, 0.0)] {
            let dg = nonadiabatic_barrier(ev(lam), ev(dg0)).unwrap();
            let back = reorganization_energy(dg, ev(0.0), ev(dg0)).unwrap();
            assert!(
                (back.value() - ev(lam).value()).abs() < 1e-14,
                "{lam} {dg0}"
            );
        }
        // At V11 = V22 the two-state model gives V11 - |V12|.
        let lower = two_state_lower(ev(1.0), ev(1.0), ev(0.2));
        assert!((lower.value() - ev(0.8).value()).abs() < 1e-15);
    }

    #[test]
    fn reproduces_the_published_rate_of_hexaaqua_iron() {
        // Table 2, [Fe(H2O)6]3+/2+: dG = 0.83 eV, kappa_el = 1.00, log10 k = 0.2 (values rounded).
        let k = rate_constant(ev(0.83), 1.0, &TransferConditions::default()).unwrap();
        assert!((k.log10() - 0.2).abs() < 0.1, "log10 k = {}", k.log10());
    }

    #[test]
    fn transmission_coefficient_limits() {
        let cond = TransferConditions::default();
        // Strong coupling: P_LZ -> 1, kappa -> 1; weak coupling: kappa -> 0.
        let strong = landau_zener_probability(ev(0.3), ev(4.0), &cond).unwrap();
        assert!((transmission_coefficient(strong, ev(0.0), ev(4.0)) - 1.0).abs() < 1e-6);
        let weak = landau_zener_probability(ev(1e-4), ev(1.0), &cond).unwrap();
        assert!(transmission_coefficient(weak, ev(0.0), ev(1.0)) < 1e-4);
        // Table 2, [Co(en)3]3+/2+: V12 ~ 0.05 eV, lambda = 4.86 eV, kappa_el = 0.91; with the
        // rounded V12 the coefficient lies between those for 0.05 and 0.055 eV.
        let kappa = |v: f64| {
            let p = landau_zener_probability(ev(v), ev(4.86), &cond).unwrap();
            transmission_coefficient(p, ev(0.0), ev(4.86))
        };
        assert!(kappa(0.05) < 0.91 && 0.91 < kappa(0.055));
        // Marcus limit (eqn (5)) agrees with eqn (1)-(4) for weak coupling.
        let (v12, lam) = (ev(1e-4), ev(1.0));
        let dg_na = nonadiabatic_barrier(lam, ev(0.0)).unwrap();
        let p = landau_zener_probability(v12, lam, &cond).unwrap();
        let full = rate_constant(dg_na, transmission_coefficient(p, ev(0.0), lam), &cond).unwrap();
        let marcus = marcus_rate_constant(v12, lam, dg_na, &cond).unwrap();
        assert!((full / marcus - 1.0).abs() < 1e-3, "{full} vs {marcus}");
    }
}
