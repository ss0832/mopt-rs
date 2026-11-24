//! The inner sphere of an electron transfer at a crossing of two diabatic states: its
//! harmonic parameters along the difference gradient, the nuclear tunnelling factor of Scher
//! and Holstein, the Marcus rate constant with that factor, and the Zhu-Nakamura rate
//! constant at the crossing, after chapter 4 and appendix 4.7 of A. Mutsuji, PhD thesis,
//! Hokkaido University (2025), doi:10.14943/doctoral.k16373 (not published in a peer-reviewed
//! journal; equation numbers refer to the thesis), except the Zhu-Nakamura rate, which is new
//! in mopt-rs.
//!
//! **Harmonic parameters (appendix 4.7, eqn (4.S4)-(4.S9)).** With the mass-weighted gradients
//! `g_i`, `g_f` of the two states at the crossing, the unit vector `u` of the difference
//! gradient `g_i - g_f`, their slopes `g_i u`, `g_f u`, and the inner-sphere barrier `dE_in`
//! (from the initial equilibrium to the crossing at the same solvent coordinate):
//! `V11 = a q^2`, `V22 = a q^2 + b q + c` along `u` with `a = (g_i u)^2 / (4 dE_in)`,
//! `q_SX = 2 dE_in / (g_i u)`, `b = g_f u - g_i u`, `c = dE_in - a q_SX^2 - b q_SX`, the final
//! minimum `q_f = -b / 2a`, `dE_in0 = V22(q_f)`, `lambda_in = a q_f^2`, and the frequency
//! `omega = sqrt(2a)` in the mass-weighted coordinate. Exact for two displaced harmonic states
//! of one frequency along one coordinate.
//!
//! **Scher-Holstein (eqn (4.12)-(4.16)).** With `y = h nu / 2 k_B T` and `r = dE_in0 /
//! lambda_in`:
//!
//! ```text
//! dG(T)   = dE_in0 / 2 + (lambda_in k_B T / h nu) B,
//! B       = coth y - sqrt(r^2 + csch^2 y) + r asinh(r sinh y),
//! dG(inf) = (lambda_in + dE_in0)^2 / (4 lambda_in),
//! Gamma   = exp(-(dG(T) - dG(inf)) / k_B T).
//! ```
//!
//! The thesis prints the exponent of eqn (4.14) without its brackets; it is `-dG(T) / k_B T`.
//! `dG(T)` is the exponent of the golden-rule rate of one quantum mode at the saddle point of
//! its integral over imaginary time, which the tests compute numerically. Its limits: `B ->
//! y (1 + r^2) / 2` and `Gamma -> 1` at high temperature (with `ln Gamma = lambda_in y^2 (1 -
//! r^2)^2 / (48 k_B T)` to lowest order, used below `y = 1e-3`), `Gamma = 1` for an
//! activationless transfer (`r = -1`), and `exp(-S (1 - |r| + |r| ln |r|))` with `S = lambda_in
//! / h nu` at low temperature (the energy-gap law). `sinh y` and `csch y` are taken through
//! `exp(-2y)`, so that large `y` does not overflow.
//!
//! **Marcus with tunnelling (eqn (4.1)):** `k = Gamma 2 |V12|^2 / h (pi^3 / lambda k_B T)^(1/2)
//! exp(-dG / RT)` with the total reorganization `lambda = lambda_in + (lambda_s^i +
//! lambda_s^f) / 2` (eqn (4.17)).
//!
//! **Zhu-Nakamura at the crossing (new in mopt-rs).** The same slopes, the coupling, and the
//! barriers from both sides along `u` make a crossing of the nonadiabatic-tunnelling type
//! (slopes of opposite signs: the Marcus normal region) or of the Landau-Zener type (slopes of
//! the same sign: the inverted region); its thermal transmission `kappa`
//! ([`super::nonadiabatic_tunnelling::PeakedCrossing`], [`super::zhu_nakamura`]; tunnelling
//! below the crossing included for the peaked type, the energies above it for the sloped one)
//! gives `k = (RT/p0)^q kappa (k_B T / h) exp(-dE / k_B T) / Q_n`, the transition state
//! theory of the crossing along the mode, with `Q_n = exp(-h nu / 2 k_B T) / (1 - exp(-h nu /
//! k_B T))` the partition function of the mode in the initial well (from its classical
//! bottom): the electronic barrier `dE` leaves it out, where the Franck-Condon factor of the
//! Marcus rate has it. With a free energy of activation that has it already (the normal modes
//! of the initial equilibrium against those within the seam), the rate is
//! [`ZhuNakamuraRate::with_free_energy`]. At weak coupling it is the Marcus rate with the
//! tunnelling factor of the mode (a test).
//!
//! **Marcus-Levich-Jortner** (J. Jortner, J. Chem. Phys. 1976, 64, 4860; J. Ulstrup,
//! J. Jortner, J. Chem. Phys. 1975, 63, 4358, for the mode in its ground state): the inner
//! sphere as one quantum mode of `h nu` with the Huang-Rhys factor `S = lambda_in / h nu` and
//! its thermal occupation `nbar`, the solvent classical,
//!
//! ```text
//! k = (2 pi / hbar) |V12|^2 (4 pi lambda_s k_B T)^(-1/2)
//!     sum_n W_n exp(-(dG0 + lambda_s + n h nu)^2 / (4 lambda_s k_B T)),
//! W_n = exp(-S (2 nbar + 1)) ((nbar + 1) / nbar)^(n/2) I_n(2 S sqrt(nbar (nbar + 1))),
//! ```
//!
//! `n` the quanta the mode takes up (negative: gives up), `W_n` its thermal Franck-Condon
//! weight (`e^-S S^n / n!` in its ground state). `W_n` is the integral of the generating
//! function over a period, taken on the contour through its saddle point, where the
//! integrand does not oscillate (by the trapezoidal rule, or by its expansion in `1 / a` for a
//! large curvature `a`); the sum runs in logarithms over the quanta that matter. It takes the
//! inputs of the Scher-Holstein factor with the solvent reorganization `lambda_s` (positive: a
//! classical bath) and the reaction free energy `dG0`. Its limits (tests): no inner sphere gives
//! Marcus of the solvent alone, a soft mode Marcus with `lambda_s + lambda_in`, and a stiff one
//! at low temperature the sum over the quanta of its ground state.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::{LN_2, PI};

use super::nonadiabatic_tunnelling::PeakedCrossing;
use super::zhu_nakamura::ZhuNakamuraCrossing;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};

/// The harmonic parameters of the inner sphere along the difference gradient at a crossing
/// (atomic units; mass-weighted coordinates in Bohr amu^(1/2)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InnerSphere {
    /// `h nu` (Hartree): the quantum of the mode.
    pub quantum: f64,
    /// `dE_in0` (Hartree): the reaction energy of the inner sphere at the solvent coordinate
    /// of the crossing.
    pub reaction_energy: f64,
    /// `lambda_in` (Hartree).
    pub reorganization: f64,
    /// The crossing on the coordinate, `q_SX`.
    pub q_crossing: f64,
    /// The minimum of the final state on the coordinate, `q_f`.
    pub q_final: f64,
    /// The slopes `g_i u` and `g_f u` (Hartree / (Bohr amu^(1/2))).
    pub slopes: [f64; 2],
}

impl InnerSphere {
    /// The wavenumber of the mode (1/cm).
    pub fn wavenumber(&self) -> f64 {
        self.quantum * Units::HARTREE_TO_JOULE / (Units::PLANCK_J_S * Units::SPEED_OF_LIGHT_CM_S)
    }

    /// The barrier from the final minimum to the crossing (Hartree).
    pub fn barrier_from_final(&self, barrier_from_initial: f64) -> f64 {
        barrier_from_initial - self.reaction_energy
    }
}

/// The parameters of eqn (4.S4)-(4.S9) from the mass-weighted gradients `g_initial`,
/// `g_final` (Hartree / (Bohr amu^(1/2)), each Cartesian component divided by the square root
/// of the mass of its atom in amu) of the two states at the crossing and the inner-sphere
/// barrier `barrier` (Hartree, positive). Refused at `barrier <= 0` (the crossing at the
/// initial equilibrium, `alpha_SA = 1`, has none) and for equal gradients.
pub fn inner_sphere(g_initial: &[f64], g_final: &[f64], barrier: f64) -> MoptResult<InnerSphere> {
    if g_initial.len() != g_final.len() || g_initial.is_empty() {
        return Err(MoptError::config(
            "inner sphere: the two gradients must have the same nonzero length",
        ));
    }
    if !(barrier.is_finite() && barrier > 0.0) {
        return Err(MoptError::calc(format!(
            "inner sphere: the barrier of the inner sphere must be positive, got {barrier:e} \
             Hartree (the crossing of alpha_SA = 1 is the initial equilibrium)"
        )));
    }
    let diff: Vec<f64> = g_initial.iter().zip(g_final).map(|(a, b)| a - b).collect();
    let norm = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(MoptError::calc(
            "inner sphere: the two gradients are equal (no difference gradient)",
        ));
    }
    let along = |g: &[f64]| g.iter().zip(&diff).map(|(a, d)| a * d).sum::<f64>() / norm;
    let (gi, gf) = (along(g_initial), along(g_final));
    if gi == 0.0 {
        return Err(MoptError::calc(
            "inner sphere: the initial state has no slope along the difference gradient",
        ));
    }
    let a = gi * gi / (4.0 * barrier);
    let q_crossing = 2.0 * barrier / gi;
    let b = gf - gi;
    let c = barrier - a * q_crossing * q_crossing - b * q_crossing;
    let q_final = -b / (2.0 * a);
    let reaction_energy = c - b * b / (4.0 * a);
    let reorganization = a * q_final * q_final;
    // omega^2 = 2a in Hartree / (Bohr^2 amu); in atomic units of mass, h nu = hbar omega.
    let quantum = (2.0 * a / Units::AMU_TO_AU_MASS).sqrt();
    Ok(InnerSphere {
        quantum,
        reaction_energy,
        reorganization,
        q_crossing,
        q_final,
        slopes: [gi, gf],
    })
}

/// The nuclear tunnelling factor of Scher and Holstein with its barriers (Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TunnellingFactor {
    /// `ln Gamma_n`.
    pub ln_gamma: f64,
    /// `dG_na(T)`, eqn (4.13).
    pub barrier_quantum: f64,
    /// `dG_na(T -> inf)`, eqn (4.15).
    pub barrier_classical: f64,
}

impl TunnellingFactor {
    /// `Gamma_n`.
    pub fn gamma(&self) -> f64 {
        self.ln_gamma.exp()
    }

    /// `log10 Gamma_n`.
    pub fn log10(&self) -> f64 {
        self.ln_gamma / std::f64::consts::LN_10
    }
}

/// `asinh(r sinh y)` for `y > 0` without overflow of `sinh y`.
fn asinh_scaled(r: f64, y: f64) -> f64 {
    if r == 0.0 {
        return 0.0;
    }
    // ln sinh y = y + ln(1 - exp(-2y)) - ln 2.
    let ln_sinh = y + (-(-2.0 * y).exp_m1()).ln() - LN_2;
    let ln_z = r.abs().ln() + ln_sinh;
    if ln_z > 20.0 {
        // asinh z = ln(2z) + 1/(4 z^2) + ...: the rest is below 1e-17 of the value.
        r.signum() * (ln_z + LN_2)
    } else {
        (r.signum() * ln_z.exp()).asinh()
    }
}

/// The tunnelling factor of eqn (4.12)-(4.16) for the quantum `quantum` (Hartree), the
/// reaction energy `reaction_energy` and the reorganization energy `reorganization` of the
/// inner sphere (Hartree), at `kt` (Hartree).
pub fn scher_holstein(
    quantum: f64,
    reaction_energy: f64,
    reorganization: f64,
    kt: f64,
) -> MoptResult<TunnellingFactor> {
    let ok = |v: f64| v.is_finite() && v > 0.0;
    if !(ok(quantum) && ok(reorganization) && ok(kt) && reaction_energy.is_finite()) {
        return Err(MoptError::config(format!(
            "Scher-Holstein: the quantum ({quantum:e}), the reorganization energy \
             ({reorganization:e}) and k_B T ({kt:e}) must be positive"
        )));
    }
    let (l, de) = (reorganization, reaction_energy);
    let y = quantum / (2.0 * kt);
    let r = de / l;
    let classical = (l + de).powi(2) / (4.0 * l);
    // B - y (1 + r^2) / 2, the part of the bracket that the high-temperature limit leaves.
    let excess = if y < 1e-3 {
        -y.powi(3) * (1.0 - r * r).powi(2) / 24.0
    } else {
        let e = (-2.0 * y).exp();
        let s = -(-2.0 * y).exp_m1(); // 1 - exp(-2y)
        let coth = (1.0 + e) / s;
        let csch2 = 4.0 * e / (s * s);
        // coth y - sqrt(r^2 + csch^2 y) = (1 - r^2) / (coth y + sqrt(r^2 + csch^2 y)), as
        // coth^2 - csch^2 = 1: both terms are about 1 / y at small y.
        let b = (1.0 - r * r) / (coth + (r * r + csch2).sqrt()) + r * asinh_scaled(r, y);
        b - 0.5 * y * (1.0 + r * r)
    };
    // dG(T) - dG(inf) = (lambda k_B T / h nu) excess = lambda excess / (2 y).
    let difference = l * excess / (2.0 * y);
    Ok(TunnellingFactor {
        ln_gamma: -difference / kt,
        barrier_quantum: classical + difference,
        barrier_classical: classical,
    })
}

/// The Marcus rate constant with the tunnelling factor, eqn (4.1), of a unimolecular
/// transfer (1/s): `coupling`, `reorganization` (the total of eqn (4.17)) and `barrier` in
/// Hartree, at `kt` (Hartree).
pub fn marcus_with_tunnelling(
    ln_gamma: f64,
    coupling: f64,
    reorganization: f64,
    barrier: f64,
    kt: f64,
) -> MoptResult<f64> {
    if !(reorganization.is_finite() && reorganization > 0.0 && kt > 0.0) {
        return Err(MoptError::config(
            "Marcus rate: the reorganization energy and k_B T must be positive",
        ));
    }
    let ln_k = ln_gamma
        + (2.0 * coupling * coupling / Units::PLANCK_HARTREE_S).ln()
        + 0.5 * (PI.powi(3) / (reorganization * kt)).ln()
        - barrier / kt;
    Ok(ln_k.exp())
}

/// The type of a crossing along the difference gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingType {
    /// Slopes of opposite signs (nonadiabatic tunnelling type; the Marcus normal region).
    Peaked,
    /// Slopes of the same sign (Landau-Zener type; the inverted region).
    Sloped,
}

/// A Zhu-Nakamura rate constant at a crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZhuNakamuraRate {
    /// The type of the crossing.
    pub kind: CrossingType,
    /// `ln kappa`, the thermal transmission referred to the crossing energy.
    pub ln_kappa: f64,
    /// The share of `kappa` from the energies below the crossing (tunnelling; 0 for a
    /// sloped crossing, whose energies below the crossing are not included).
    pub below_crossing: f64,
    /// `ln Q_n`, the partition function of the mode in the initial well.
    pub ln_partition: f64,
    /// The rate constant with the electronic barrier (1/s for a unimolecular transfer).
    pub rate: f64,
    /// `(RT/p0)^q k_B T / h`.
    pub frequency: f64,
}

impl ZhuNakamuraRate {
    /// The rate constant with the free energy of activation `barrier` (Hartree) at `kt`,
    /// whose partition functions have the mode already.
    pub fn with_free_energy(&self, barrier: f64, kt: f64) -> f64 {
        self.frequency * (self.ln_kappa - barrier / kt).exp()
    }
}

/// `ln Q` of a harmonic mode of the quantum `quantum` at `kt`, from its classical bottom.
pub fn ln_mode_partition(quantum: f64, kt: f64) -> f64 {
    let x = quantum / kt;
    -0.5 * x - (-(-x).exp_m1()).ln()
}

/// The Zhu-Nakamura rate constant of the crossing of `inner` (its slopes in the mass-weighted
/// coordinate) with the coupling `coupling`, the whole barrier `barrier` from the initial
/// equilibrium (solvent activation included) and the inner-sphere barrier
/// `inner_barrier` from the initial side, at `kt` (all Hartree); `volume_factor` is `(RT /
/// p0)^q` in the units of the rate constant (1 for a unimolecular transfer).
pub fn zhu_nakamura_at_crossing(
    inner: &InnerSphere,
    coupling: f64,
    barrier: f64,
    inner_barrier: f64,
    kt: f64,
    volume_factor: f64,
) -> MoptResult<ZhuNakamuraRate> {
    let [f1, f2] = inner.slopes;
    let product = f1 * f2;
    let difference = (f1 - f2).abs();
    let prefactor = kt / Units::PLANCK_HARTREE_S * volume_factor;
    let ln_partition = ln_mode_partition(inner.quantum, kt);
    if product < 0.0 {
        let crossing = PeakedCrossing::from_diabatic(coupling, difference, product, 1.0)
            .ok_or_else(|| {
                MoptError::calc("Zhu-Nakamura: no peaked crossing from these slopes and coupling")
            })?;
        let lowest = inner_barrier
            .min(inner.barrier_from_final(inner_barrier))
            .max(0.0);
        let avg = crossing.average(kt, lowest);
        Ok(ZhuNakamuraRate {
            kind: CrossingType::Peaked,
            ln_kappa: avg.ln_kappa,
            below_crossing: avg.below_crossing,
            ln_partition,
            rate: prefactor * (avg.ln_kappa - barrier / kt - ln_partition).exp(),
            frequency: prefactor,
        })
    } else {
        let crossing = ZhuNakamuraCrossing::from_diabatic(coupling, difference, product, 1.0)
            .ok_or_else(|| {
                MoptError::calc("Zhu-Nakamura: no sloped crossing from these slopes and coupling")
            })?;
        let ln_kappa = crossing.ln_average(kt);
        Ok(ZhuNakamuraRate {
            kind: CrossingType::Sloped,
            ln_kappa,
            below_crossing: 0.0,
            ln_partition,
            rate: prefactor * (ln_kappa - barrier / kt - ln_partition).exp(),
            frequency: prefactor,
        })
    }
}

/// `ln n!` (Stirling's series beyond 20).
fn ln_factorial(n: u64) -> f64 {
    if n <= 20 {
        return (2..=n).map(|k| (k as f64).ln()).sum();
    }
    let x = n as f64;
    let x2 = x * x;
    x * x.ln() - x + 0.5 * (2.0 * PI * x).ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x2)
        + 1.0 / (1260.0 * x * x2 * x2)
}

/// Above this curvature, the average over the contour comes from its expansion in `1 / a`.
const CONTOUR_EXPANSION_FROM: f64 = 2.0e4;

/// `(1 / 2 pi) int exp(-a (1 - cos p)) cos(n (sin p - p)) dp` over a period: by the
/// trapezoidal rule (its error falls faster than any power with the points, for a periodic
/// entire function whose harmonics end near `a + 2 |n|`), or for a large `a` by
/// `(2 pi a)^(-1/2) (1 + 1 / 8a + 9 / 128 a^2 - 5 n^2 / 24 a^3)`.
fn contour_average(n: f64, a: f64) -> f64 {
    if a > CONTOUR_EXPANSION_FROM {
        return (1.0 + 1.0 / (8.0 * a) + 9.0 / (128.0 * a * a) - 5.0 * n * n / (24.0 * a * a * a))
            / (2.0 * PI * a).sqrt();
    }
    let points = (3.0 * a + 2.0 * n.abs()).ceil() as usize + 64;
    let h = 2.0 * PI / points as f64;
    (0..points)
        .map(|k| {
            let p = -PI + h * k as f64;
            let versine = 2.0 * (0.5 * p).sin().powi(2);
            (-a * versine).exp() * (n * (p.sin() - p)).cos()
        })
        .sum::<f64>()
        / points as f64
}

/// `ln W_n`, the thermal Franck-Condon weight of `n` quanta taken up (negative: given up) by
/// a mode of the Huang-Rhys factor `s` and the thermal occupation `nbar`: its generating
/// function `exp(s ((nbar + 1) (z - 1) + nbar (1 / z - 1)))` over the circle `|z| = x` through
/// its saddle point on the positive axis.
fn ln_mode_weight(n: i64, s: f64, nbar: f64) -> f64 {
    if s == 0.0 {
        return if n == 0 { 0.0 } else { f64::NEG_INFINITY };
    }
    let nf = n as f64;
    if nbar == 0.0 {
        return if n < 0 {
            f64::NEG_INFINITY
        } else {
            -s + nf * s.ln() - ln_factorial(n.unsigned_abs())
        };
    }
    // The saddle point: s ((nbar + 1) x - nbar / x) = n, its root without cancellation.
    let b = nf / s;
    let root = (b * b + 4.0 * nbar * (nbar + 1.0)).sqrt();
    let x = if b >= 0.0 {
        (b + root) / (2.0 * (nbar + 1.0))
    } else {
        2.0 * nbar / (root - b)
    };
    let curvature = s * ((nbar + 1.0) * x + nbar / x);
    let at_saddle = s * ((nbar + 1.0) * (x - 1.0) + nbar * (1.0 / x - 1.0)) - nf * x.ln();
    at_saddle + contour_average(nf, curvature).ln()
}

/// The quanta summed at most in [`marcus_levich_jortner`].
const MOST_QUANTA: f64 = 2.0e7;

/// The Marcus-Levich-Jortner rate constant (1/s) of the coupling `coupling`, the mode of the
/// quantum `quantum` and the inner reorganization `inner_reorganization`, the solvent
/// reorganization `solvent_reorganization` (positive), and the reaction free energy
/// `reaction_energy`, at `kt` (all Hartree), with the thermal occupation of the mode.
pub fn marcus_levich_jortner(
    coupling: f64,
    quantum: f64,
    inner_reorganization: f64,
    solvent_reorganization: f64,
    reaction_energy: f64,
    kt: f64,
) -> MoptResult<f64> {
    let ok = |v: f64| v.is_finite() && v > 0.0;
    if !(ok(quantum) && ok(solvent_reorganization) && ok(kt))
        || !(inner_reorganization.is_finite() && inner_reorganization >= 0.0)
        || !(reaction_energy.is_finite() && coupling.is_finite())
    {
        return Err(MoptError::config(
            "Marcus-Levich-Jortner: the quantum, the solvent reorganization energy, and k_B T \
             must be positive, the inner reorganization energy not negative",
        ));
    }
    let ls = solvent_reorganization;
    let s = inner_reorganization / quantum;
    let x = quantum / kt;
    let nbar = if x > 700.0 {
        0.0
    } else {
        (-x).exp() / -(-x).exp_m1()
    };
    let width = 4.0 * ls * kt;
    // The weights and the Gaussian of the solvent are both log-concave in n, so their product
    // lies between their centres, within a dozen of the larger spread.
    let (centre_w, spread_w) = (s, (s * (2.0 * nbar + 1.0)).sqrt());
    let (centre_g, spread_g) = (
        -(reaction_energy + ls) / quantum,
        (0.5 * width).sqrt() / quantum,
    );
    let reach = 12.0 * spread_w.max(spread_g) + 12.0;
    let mut lo = (centre_w.min(centre_g) - reach).floor();
    let mut hi = (centre_w.max(centre_g) + reach).ceil();
    if nbar == 0.0 {
        lo = lo.max(0.0);
    }
    if s == 0.0 {
        (lo, hi) = (0.0, 0.0);
    }
    if hi - lo > MOST_QUANTA {
        return Err(MoptError::config(format!(
            "Marcus-Levich-Jortner: {:.0} quanta of the mode would be summed; a mode this soft \
             is classical, and Marcus with lambda_s + lambda_in is its rate",
            hi - lo
        )));
    }
    let terms: Vec<f64> = (lo as i64..=hi as i64)
        .map(|n| {
            ln_mode_weight(n, s, nbar) - (reaction_energy + ls + n as f64 * quantum).powi(2) / width
        })
        .collect();
    let best = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = terms.iter().map(|t| (t - best).exp()).sum();
    let ln_prefactor =
        (2.0 * PI * coupling * coupling / Units::HBAR_HARTREE_S).ln() - 0.5 * (PI * width).ln();
    Ok((ln_prefactor + best + sum.ln()).exp())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const EV: f64 = Units::EV_TO_HARTREE;

    #[test]
    fn the_thermal_weights_of_a_mode_are_a_distribution() {
        // Normalized, with the mean S, for warm and cold modes and on both branches of the
        // contour average.
        for (s, nbar) in [
            (3.0_f64, 0.5_f64),
            (50.0, 2.0),
            (0.3, 1e-3),
            (400.0, 30.0),
            (2.0, 1e-14),
        ] {
            let range = (s * (2.0 * nbar + 1.0)).sqrt() * 14.0 + 30.0;
            let (lo, hi) = ((s - range).floor() as i64, (s + range).ceil() as i64);
            let w: Vec<(f64, f64)> = (lo..=hi)
                .map(|n| (n as f64, ln_mode_weight(n, s, nbar).exp()))
                .collect();
            let total: f64 = w.iter().map(|(_, v)| v).sum();
            let mean: f64 = w.iter().map(|(n, v)| n * v).sum();
            assert!((total - 1.0).abs() < 1e-8, "s {s} nbar {nbar}: sum {total}");
            assert!(
                (mean - s).abs() < 1e-8 * s.max(1.0),
                "s {s} nbar {nbar}: mean {mean}"
            );
        }
        // A cold mode: the Poisson weights of its ground state.
        for n in 0..12 {
            let cold = ln_mode_weight(n, 2.5, 1e-15);
            let poisson = -2.5 + n as f64 * 2.5f64.ln() - ln_factorial(n as u64);
            assert!((cold - poisson).abs() < 1e-9, "{n}: {cold} {poisson}");
        }
        // The two ways of the contour average meet.
        for (n, a) in [(0.0, 2.1e4), (150.0, 2.1e4), (-90.0, 2.1e4)] {
            let expansion = contour_average(n, a);
            let points = (3.0 * a + 2.0 * f64::abs(n)).ceil() as usize + 64;
            let h = 2.0 * PI / points as f64;
            let rule: f64 = (0..points)
                .map(|k| {
                    let p = -PI + h * k as f64;
                    (-a * 2.0 * (0.5 * p).sin().powi(2)).exp() * (n * (p.sin() - p)).cos()
                })
                .sum::<f64>()
                / points as f64;
            assert!(
                (expansion / rule - 1.0).abs() < 1e-8,
                "{n} {a}: {expansion} {rule}"
            );
        }
        assert!((ln_factorial(21) - (2..=21).map(|k| (k as f64).ln()).sum::<f64>()).abs() < 1e-12);
    }

    #[test]
    fn marcus_levich_jortner_has_its_limits() {
        let kt = kt(298.15);
        let (v, ls, li) = (1e-4, 0.3 * EV, 0.2 * EV);
        // No inner sphere: Marcus of the solvent alone, whatever the mode.
        let mlj = marcus_levich_jortner(v, 0.2 * EV, 0.0, ls, -0.1 * EV, kt).unwrap();
        let marcus =
            marcus_with_tunnelling(0.0, v, ls, (ls - 0.1 * EV).powi(2) / (4.0 * ls), kt).unwrap();
        assert!((mlj / marcus - 1.0).abs() < 1e-12);
        // A soft mode is classical: Marcus with the whole reorganization.
        let l = ls + li;
        for dg in [-0.4 * EV, 0.0, 0.1 * EV] {
            let mlj = marcus_levich_jortner(v, 2e-4 * EV, li, ls, dg, kt).unwrap();
            let marcus =
                marcus_with_tunnelling(0.0, v, l, (l + dg).powi(2) / (4.0 * l), kt).unwrap();
            assert!(
                (mlj / marcus - 1.0).abs() < 1e-3,
                "dG {dg}: {mlj} vs {marcus}"
            );
        }
        // A stiff mode at low temperature: the sum over the quanta of its ground state.
        let cold = Units::BOLTZMANN_HARTREE_PER_K * 100.0;
        let quantum = 0.18 * EV;
        let dg = -0.9 * EV;
        let mlj = marcus_levich_jortner(v, quantum, li, ls, dg, cold).unwrap();
        let s = li / quantum;
        let ground: f64 = (0..60u64)
            .map(|n| {
                (-s + n as f64 * s.ln()
                    - ln_factorial(n)
                    - (dg + ls + n as f64 * quantum).powi(2) / (4.0 * ls * cold))
                    .exp()
            })
            .sum::<f64>()
            * 2.0
            * PI
            * v
            * v
            / Units::HBAR_HARTREE_S
            / (4.0 * PI * ls * cold).sqrt();
        assert!((mlj / ground - 1.0).abs() < 1e-6, "{mlj} vs {ground}");
        // The stiff mode lifts the deep inverted region far above Marcus (the energy-gap law).
        let dg = -1.5 * EV;
        let mlj = marcus_levich_jortner(v, quantum, li, ls, dg, kt).unwrap();
        let marcus = marcus_with_tunnelling(0.0, v, l, (l + dg).powi(2) / (4.0 * l), kt).unwrap();
        assert!(mlj > 1e3 * marcus, "{mlj} vs {marcus}");
        // Against the tunnelling factor of Scher and Holstein in the normal region, where the
        // two quantum treatments of the mode agree to a few tens of per cent.
        let dg = -0.2 * EV;
        let mlj = marcus_levich_jortner(v, quantum, li, ls, dg, kt).unwrap();
        let sh = scher_holstein(quantum, dg, li, kt).unwrap();
        let sh_rate =
            marcus_with_tunnelling(sh.ln_gamma, v, l, (l + dg).powi(2) / (4.0 * l), kt).unwrap();
        assert!(
            (mlj / sh_rate).ln().abs() < 1.0,
            "MLJ {mlj} vs Scher-Holstein {sh_rate}"
        );
        // A classical bath is needed; a mode too soft to sum is refused.
        assert!(marcus_levich_jortner(v, quantum, li, 0.0, dg, kt).is_err());
        assert!(marcus_levich_jortner(v, 1e-9 * EV, li, ls, dg, kt).is_err());
    }

    fn kt(temperature: f64) -> f64 {
        Units::BOLTZMANN_HARTREE_PER_K * temperature
    }

    /// The minimum over the imaginary time `s` of the exponent of the golden-rule integral
    /// of one quantum mode, `S [(n + 1) (e^(w s) - 1) + n (e^(-w s) - 1)] + dE0 s` (`hbar = 1`,
    /// `w` the quantum, `n` its thermal occupation), by bisection of its derivative.
    fn saddle_exponent(quantum: f64, de0: f64, lambda: f64, kt: f64) -> f64 {
        let s_hr = lambda / quantum;
        let n = 1.0 / ((quantum / kt).exp() - 1.0);
        let phi = |s: f64| {
            s_hr * ((n + 1.0) * ((quantum * s).exp() - 1.0) + n * ((-quantum * s).exp() - 1.0))
                + de0 * s
        };
        let dphi = |s: f64| {
            s_hr * quantum * ((n + 1.0) * (quantum * s).exp() - n * (-quantum * s).exp()) + de0
        };
        let (mut lo, mut hi) = (-200.0 / quantum, 200.0 / quantum);
        for _ in 0..300 {
            let mid = 0.5 * (lo + hi);
            if dphi(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        phi(0.5 * (lo + hi))
    }

    #[test]
    fn eqn_4_13_is_the_saddle_point_of_the_single_mode_rate() {
        for &(nu_cm, de0_ev, lambda_ev, t) in &[
            (1500.0, -0.5, 0.3, 296.0),
            (1500.0, -1.2, 0.4, 296.0),
            (800.0, 0.1, 0.5, 296.0),
            (400.0, -0.3, 0.2, 150.0),
            (1200.0, -2.0, 0.6, 600.0),
        ] {
            let quantum =
                nu_cm * Units::PLANCK_J_S * Units::SPEED_OF_LIGHT_CM_S / Units::HARTREE_TO_JOULE;
            let (de0, lambda, kt) = (de0_ev * EV, lambda_ev * EV, kt(t));
            let f = scher_holstein(quantum, de0, lambda, kt).unwrap();
            let numeric = -saddle_exponent(quantum, de0, lambda, kt) * kt;
            assert!(
                (f.barrier_quantum - numeric).abs() < 1e-10 * numeric.abs().max(1e-3),
                "{nu_cm} {de0_ev} {lambda_ev} {t}: {} vs {numeric}",
                f.barrier_quantum
            );
        }
    }

    #[test]
    fn the_tunnelling_factor_has_its_limits() {
        let lambda = 0.4 * EV;
        let quantum = 0.18 * EV;
        // Activationless: no barrier to tunnel through.
        let f = scher_holstein(quantum, -lambda, lambda, kt(296.0)).unwrap();
        assert!(f.ln_gamma.abs() < 1e-12, "{}", f.ln_gamma);
        // High temperature: Gamma -> 1, the series and the closed form meet.
        for t in [1e4, 1e5, 1e6] {
            let f = scher_holstein(quantum, -0.1 * EV, lambda, kt(t)).unwrap();
            assert!(
                f.ln_gamma >= 0.0 && f.ln_gamma < 0.02,
                "{t}: {}",
                f.ln_gamma
            );
        }
        // The closed form just above y = 1e-3, where the series takes over, is the series
        // (its next term is y^2 smaller).
        let y = 1.05e-3;
        let kt_y = quantum / (2.0 * y);
        let closed = scher_holstein(quantum, -0.1 * EV, lambda, kt_y)
            .unwrap()
            .ln_gamma;
        let r: f64 = -0.1 * EV / lambda;
        let series = lambda * y * y * (1.0 - r * r).powi(2) / (48.0 * kt_y);
        assert!((closed - series).abs() < 1e-4 * series, "{closed} {series}");
        // Low temperature: the energy-gap law of one mode, exp(-S (1 - |r| + |r| ln |r|)).
        let (de0, l) = (-1.5 * EV, 0.5 * EV);
        let f = scher_holstein(quantum, de0, l, kt(5.0)).unwrap();
        let r = (de0 / l).abs();
        let law = -(l / quantum) * (1.0 - r + r * r.ln());
        let exponent = -f.barrier_quantum / kt(5.0);
        assert!(
            (exponent - law).abs() < 2e-2 * law.abs(),
            "{exponent} vs {law}"
        );
        // Tunnelling enhances the inverted region far more than the normal one.
        let inverted = scher_holstein(quantum, -1.2 * EV, lambda, kt(296.0)).unwrap();
        let normal = scher_holstein(quantum, -0.1 * EV, lambda, kt(296.0)).unwrap();
        assert!(inverted.ln_gamma > normal.ln_gamma && normal.ln_gamma > 0.0);
    }

    #[test]
    fn large_quanta_do_not_overflow() {
        let f = scher_holstein(0.5 * EV, -2.0 * EV, 0.6 * EV, kt(1.0)).unwrap();
        assert!(f.ln_gamma.is_finite() && f.barrier_quantum.is_finite());
        // asinh(r sinh y) of both signs, against the direct form where it does not overflow.
        for (r, y) in [
            (-0.3, 2.0),
            (0.7, 0.4),
            (-3.5, 9.0),
            (2.0, 15.0),
            (-1.0, 0.01),
        ] {
            let direct = (r * f64::sinh(y)).asinh();
            assert!((asinh_scaled(r, y) - direct).abs() < 1e-12 * direct.abs().max(1.0));
        }
        assert!((asinh_scaled(-2.0, 400.0) + (400.0 + 2.0f64.ln())).abs() < 1e-12);
    }

    #[test]
    fn the_inner_sphere_of_two_displaced_harmonic_states_is_recovered() {
        // V11 = (k/2) q^2 and V22 = (k/2) (q - d)^2 + dE0 along a unit direction in a
        // mass-weighted space of six coordinates; the gradients at the crossing.
        let omega = 0.0072; // hbar omega in Hartree (about 1580 1/cm)
        let k = omega * omega * Units::AMU_TO_AU_MASS;
        let (d, de0) = (0.9, -0.012);
        let u = [0.3, -0.5, 0.1, 0.7, -0.2, 0.35];
        let norm = u.iter().map(|v: &f64| v * v).sum::<f64>().sqrt();
        let u: Vec<f64> = u.iter().map(|v| v / norm).collect();
        // The crossing: (k/2) q^2 = (k/2)(q - d)^2 + dE0.
        let q = (0.5 * k * d * d + de0) / (k * d);
        let gi: Vec<f64> = u.iter().map(|c| k * q * c).collect();
        let gf: Vec<f64> = u.iter().map(|c| k * (q - d) * c).collect();
        let barrier = 0.5 * k * q * q;
        let p = inner_sphere(&gi, &gf, barrier).unwrap();
        assert!((p.quantum - omega).abs() < 1e-12 * omega, "{}", p.quantum);
        assert!(
            (p.reaction_energy - de0).abs() < 1e-14,
            "{}",
            p.reaction_energy
        );
        assert!((p.reorganization - 0.5 * k * d * d).abs() < 1e-13);
        assert!(inner_sphere(&gi, &gf, 0.0).is_err());
    }

    #[test]
    fn the_rates_of_the_thesis_follow_from_their_parameters() {
        // Thesis Tables 4.3 and 4.S3-4.S7 (296 K): log10 k_calc, V12, lambda, dE (eV), and
        // log10 Gamma_n; eqn (4.1) gives log10 k to the rounding of the printed parameters.
        let rows: [(f64, f64, f64, f64, f64); 14] = [
            // THF, ground final state (Table 4.3).
            (4.5, 1.5e-3, 1.39, 0.40, 0.9),
            (6.0, 1.4e-3, 1.31, 0.31, 0.7),
            (6.9, 2.3e-3, 1.17, 0.27, 0.6),
            (9.3, 1.1e-3, 1.13, 0.08, 0.4),
            (11.3, 3.9e-3, 1.43, 0.00, 0.0),
            (8.7, 3.3e-3, 1.28, 0.16, 0.2),
            (8.6, 1.1e-3, 1.52, 0.35, 4.4),
            (7.7, 9.6e-4, 1.57, 0.42, 4.8),
            (7.0, 9.5e-4, 1.73, 0.58, 6.8),
            // DBE, ground final state (Table 4.S4).
            (8.4, 9.7e-3, 1.08, 0.28, 0.9),
            (10.3, 1.2e-3, 0.72, 0.02, 0.2),
            (4.1, 1.1e-3, 1.57, 1.20, 14.3),
            // n-octane (Table 4.S6).
            (8.9, 1.9e-3, 0.68, 0.17, 1.0),
            (2.2, 1.8e-3, 1.26, 1.62, 19.2),
        ];
        let kt = kt(296.0);
        for (log_k, v, lambda, de, log_gamma) in rows {
            let k = marcus_with_tunnelling(
                log_gamma * std::f64::consts::LN_10,
                v * EV,
                lambda * EV,
                de * EV,
                kt,
            )
            .unwrap();
            // dE has two decimals (0.005 eV is 0.085 in log10 k), V12 two digits.
            assert!(
                (k.log10() - log_k).abs() < 0.2,
                "log10 k {log_k}: {}",
                k.log10()
            );
        }
    }

    #[test]
    fn zhu_nakamura_at_weak_coupling_is_the_marcus_rate() {
        // A small coupling: the thermal Zhu-Nakamura rate of the peaked crossing is Marcus's
        // with the tunnelling of the mode (175 1/cm: Gamma about 1.45 here).
        let t = 296.0;
        let kt = kt(t);
        let omega = 0.0008; // about 175 1/cm
        let k_force = omega * omega * Units::AMU_TO_AU_MASS;
        let (d, de0) = (12.0, -0.004);
        let q = (0.5 * k_force * d * d + de0) / (k_force * d);
        let barrier = 0.5 * k_force * q * q;
        let gi = [k_force * q];
        let gf = [k_force * (q - d)];
        let inner = inner_sphere(&gi, &gf, barrier).unwrap();
        let coupling = 2e-6;
        let zn = zhu_nakamura_at_crossing(&inner, coupling, barrier, barrier, kt, 1.0).unwrap();
        assert_eq!(zn.kind, CrossingType::Peaked);
        let sh = scher_holstein(
            inner.quantum,
            inner.reaction_energy,
            inner.reorganization,
            kt,
        )
        .unwrap();
        assert!(sh.ln_gamma > 0.2, "{}", sh.ln_gamma);
        let marcus =
            marcus_with_tunnelling(sh.ln_gamma, coupling, inner.reorganization, barrier, kt)
                .unwrap();
        let ratio = zn.rate / marcus;
        assert!((ratio - 1.0).abs() < 0.1, "ZN / Marcus = {ratio}");
        // With a free energy of activation that holds the mode, the partition function is
        // not divided again.
        let free = zn.with_free_energy(barrier, kt);
        assert!((free / zn.rate - zn.ln_partition.exp()).abs() < 1e-9 * free / zn.rate);
    }
}
