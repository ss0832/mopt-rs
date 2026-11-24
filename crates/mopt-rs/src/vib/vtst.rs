//! Variational transition state theory along a reaction path, and tunneling through the
//! vibrationally adiabatic barrier.
//!
//! **Free-energy profile.** At every point of the path the generalized transition state is
//! the dividing surface orthogonal to the path. Its free energy is the energy of the point
//! plus the thermal contributions of the transverse modes (and, for a molecule, of the
//! translation and rotation) computed as for a stationary point. Canonical variational
//! theory puts the transition state at the maximum of this profile; the rate constant is
//! that of conventional transition state theory (dividing surface at the saddle point, the
//! point of the highest energy) times `exp(-(G_max - G_saddle) / k T) <= 1`.
//!
//! **Tunneling.** The vibrationally adiabatic ground-state potential `V_a^G(s)` is the
//! energy plus the zero-point energy of the transverse modes. The transmission coefficient
//! is the ratio of the thermally averaged semiclassical transmission probability to the
//! classical one with the threshold at the top of `V_a^G`:
//!
//! ```text
//! theta(E) = (1 / hbar) int sqrt(2 mu_eff(s) (V_a^G(s) - E)) ds     between the turning points
//! P(E)     = 1 / (1 + exp(2 theta(E)))                              E <= V_max
//! P(E)     = 1 - P(2 V_max - E)                                     V_max <= E <= 2 V_max - E_0
//! kappa(T) = 1 + (2 / k T) int_0^(V_max - E_0) P(V_max - e) sinh(e / k T) de
//! ```
//!
//! `E_0` is the higher of the two ends of the path: energies below it are not reached on the
//! path and are not counted, so a path that stops high on the barrier underestimates the
//! tunneling. With `mu_eff = mu` (the mass of the mass-weighted coordinates) this is
//! zero-curvature tunneling (ZCT). Small-curvature tunneling (SCT, the centrifugal-dominant
//! small-curvature semiclassical adiabatic ground-state method of Lu et al., Comput. Phys.
//! Commun. 1992, 71, 235, and Liu et al., J. Am. Chem. Soc. 1993, 115, 2408) lets the
//! tunneling path cut the corner on the concave side of the curved path by a reduced mass
//!
//! ```text
//! mu_eff / mu = min(exp(-2 a - a^2 + (dt/ds)^2), 1),   a = |k| t,
//! t = (hbar / (mu w))^(1/2),   w^2 = sum_m (B_mF / |k|)^2 w_m^2
//! ```
//!
//! where `t` is the turning point of the zero-point vibration along the direction of the
//! curvature vector `k` (`w` is the harmonic frequency of the transverse modes along it).
//! Modes of imaginary frequency do not enter `w`.
//!
//! `V_a^G` is interpolated between the points by cubic Hermite polynomials with the slopes
//! of the parabolas through three neighbouring points, and `sqrt(mu_eff)` linearly.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::path::scalar_derivative;
use super::qff::amu_in_electron_masses;
use super::{thermochemistry, PathPoint, PathProfile, ThermoSettings};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::math::stable::softplus;

/// Points between two points of the path at which the barrier is interpolated.
const SUBDIVISIONS: usize = 16;
/// Intervals of the Simpson rule over the tunneling energies.
const ENERGY_INTERVALS: usize = 400;

/// Free energies of the generalized transition states along a path.
#[derive(Debug, Clone)]
pub struct FreeEnergyProfile {
    /// Temperature (K).
    pub temperature: f64,
    /// Free energy of the generalized transition state at every point (Hartree).
    pub free_energy: Vec<f64>,
    /// Point of the highest energy: the dividing surface of conventional theory.
    pub saddle: usize,
    /// Point of the highest free energy.
    pub variational: usize,
    /// Arc length of the maximum of the free energy, from the parabola through that point
    /// and its neighbours (`sqrt(amu) Bohr`).
    pub variational_arc_length: f64,
    /// Free energy at that maximum (Hartree).
    pub variational_free_energy: f64,
    /// Rate constant of canonical variational theory over that of conventional theory:
    /// `exp(-(G_max - G_saddle) / k T)`.
    pub recrossing: f64,
}

/// Maximum of the parabola through three points, if it has one between the outer points.
fn parabola_maximum(x: [f64; 3], y: [f64; 3]) -> Option<(f64, f64)> {
    let d = [
        (x[0] - x[1]) * (x[0] - x[2]),
        (x[1] - x[0]) * (x[1] - x[2]),
        (x[2] - x[0]) * (x[2] - x[1]),
    ];
    let a = y[0] / d[0] + y[1] / d[1] + y[2] / d[2];
    let b =
        -(y[0] * (x[1] + x[2]) / d[0] + y[1] * (x[0] + x[2]) / d[1] + y[2] * (x[0] + x[1]) / d[2]);
    if a.is_nan() || a >= 0.0 {
        return None;
    }
    let at = -b / (2.0 * a);
    if !(at >= x[0] && at <= x[2]) {
        return None;
    }
    let value = (0..3)
        .map(|k| {
            let others: Vec<usize> = (0..3).filter(|j| *j != k).collect();
            y[k] * (at - x[others[0]]) * (at - x[others[1]]) / d[k]
        })
        .sum();
    Some((at, value))
}

/// Index of the largest value (the first of equal ones).
fn highest(values: &[f64]) -> usize {
    (0..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

/// The point of the highest energy (the dividing surface of conventional theory), the point
/// of the highest free energy, and the arc length and free energy of the maximum of the
/// parabola through that point and its neighbours (not below the point itself), of a path with
/// the arc lengths `s`, the energies `energies`, and the generalized free energies
/// `free_energy` (of the same length, not empty).
pub fn variational_maximum(
    s: &[f64],
    energies: &[f64],
    free_energy: &[f64],
) -> (usize, usize, f64, f64) {
    let count = free_energy.len();
    let saddle = highest(energies);
    let variational = highest(free_energy);
    let (at, value) = if variational > 0 && variational + 1 < count {
        let k = variational;
        parabola_maximum(
            [s[k - 1], s[k], s[k + 1]],
            [free_energy[k - 1], free_energy[k], free_energy[k + 1]],
        )
        .unwrap_or((s[k], free_energy[k]))
    } else {
        (s[variational], free_energy[variational])
    };
    (saddle, variational, at, value.max(free_energy[variational]))
}

/// The free-energy profile of the path `points` with the energies `energies` (Hartree) and
/// the transverse modes of `profile`, at the conditions and with the treatment of low
/// frequencies of `settings`. With `molecular`, translation and rotation of the molecule
/// contribute (not in a periodic cell or on a surface in a fixed frame).
pub fn free_energy_profile(
    points: &[PathPoint],
    energies: &[f64],
    profile: &PathProfile,
    settings: &ThermoSettings,
    molecular: bool,
) -> MoptResult<FreeEnergyProfile> {
    let count = points.len();
    if count == 0 || energies.len() != count || profile.modes.len() != count {
        return Err(MoptError::config(format!(
            "free-energy profile: {count} points, {} energies, {} sets of modes",
            energies.len(),
            profile.modes.len()
        )));
    }
    let mut free_energy = Vec::with_capacity(count);
    for i in 0..count {
        let thermo = thermochemistry(&points[i].system, &profile.modes[i], energies[i], settings)?;
        free_energy.push(if molecular {
            thermo.gibbs.total
        } else {
            energies[i] + thermo.gibbs.vibrational
        });
    }
    let (saddle, variational, variational_arc_length, variational_free_energy) =
        variational_maximum(&profile.arc_length, energies, &free_energy);
    let kt = Units::BOLTZMANN_HARTREE_PER_K * settings.temperature;
    Ok(FreeEnergyProfile {
        temperature: settings.temperature,
        recrossing: (-(variational_free_energy - free_energy[saddle]) / kt).exp(),
        free_energy,
        saddle,
        variational,
        variational_arc_length,
        variational_free_energy,
    })
}

/// `mu_eff / mu` of small-curvature tunneling at every point of `profile`.
pub fn effective_mass_ratio(profile: &PathProfile) -> Vec<f64> {
    let amu = amu_in_electron_masses();
    let count = profile.arc_length.len();
    // Turning point of the zero-point vibration along the curvature vector
    // (sqrt(amu) Bohr); NaN where the path is straight or that direction is not bound.
    let turning: Vec<f64> = (0..count)
        .map(|i| {
            let curvature = profile.curvature[i];
            if curvature.is_nan() || curvature <= 1e-12 {
                return f64::NAN;
            }
            // Square of the frequency along the curvature vector in atomic units.
            let square: f64 = profile.curvature_couplings[i]
                .iter()
                .zip(&profile.modes[i].curvatures)
                .filter(|(_, lambda)| **lambda > 0.0)
                .map(|(b, lambda)| (b / curvature).powi(2) * lambda / amu)
                .sum();
            if square > 0.0 {
                // sqrt(hbar / w) in sqrt(m_e) Bohr.
                square.powf(-0.25) / amu.sqrt()
            } else {
                f64::NAN
            }
        })
        .collect();
    let slope = scalar_derivative(&turning, &profile.arc_length);
    (0..count)
        .map(|i| {
            if !turning[i].is_finite() {
                return 1.0;
            }
            let a = profile.curvature[i] * turning[i];
            let slope = if slope[i].is_finite() { slope[i] } else { 0.0 };
            (-2.0 * a - a * a + slope * slope).exp().min(1.0)
        })
        .collect()
}

/// `int_0^1 (g_l + (g_r - g_l) x) sqrt(f_l + (f_r - f_l) x) dx` for `f_l, f_r >= 0`, in a
/// form without cancellation.
fn piece(f_l: f64, f_r: f64, g_l: f64, g_r: f64) -> f64 {
    let (a, b) = (f_r.max(0.0).sqrt(), f_l.max(0.0).sqrt());
    let sum = a + b;
    if sum.is_nan() || sum <= 0.0 {
        return 0.0;
    }
    let plain = 2.0 / 3.0 * (a * a + a * b + b * b) / sum;
    let weighted = 2.0 / 15.0
        * (3.0 * a.powi(3) + 6.0 * a * a * b + 4.0 * a * b * b + 2.0 * b.powi(3))
        / (sum * sum);
    g_l * plain + (g_r - g_l) * weighted
}

/// Transmission coefficient at `temperature` (K) for the barrier `potential` (Hartree) at
/// the arc lengths `arc_length` (`sqrt(amu) Bohr`) with the effective mass `mass_ratio`
/// (`mu_eff / mu`) at every point. 1 if the potential has no barrier above both ends.
pub fn transmission(
    arc_length: &[f64],
    potential: &[f64],
    mass_ratio: &[f64],
    temperature: f64,
) -> MoptResult<f64> {
    let count = arc_length.len();
    if count < 2 || potential.len() != count || mass_ratio.len() != count {
        return Err(MoptError::config(format!(
            "transmission: {count} arc lengths, {} energies, {} masses (two points at least)",
            potential.len(),
            mass_ratio.len()
        )));
    }
    if !(temperature > 0.0 && temperature.is_finite()) {
        return Err(MoptError::config(
            "transmission: the temperature must be positive",
        ));
    }
    if potential.iter().chain(mass_ratio).any(|v| !v.is_finite())
        || mass_ratio.iter().any(|m| *m < 0.0)
    {
        return Err(MoptError::calc(
            "transmission: the barrier and the masses must be finite",
        ));
    }
    // The barrier on a finer grid.
    let slopes = scalar_derivative(potential, arc_length);
    let (mut s, mut v, mut g) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..count - 1 {
        let h = arc_length[j + 1] - arc_length[j];
        let (g_l, g_r) = (mass_ratio[j].sqrt(), mass_ratio[j + 1].sqrt());
        for k in 0..SUBDIVISIONS {
            let x = k as f64 / SUBDIVISIONS as f64;
            let (x2, x3) = (x * x, x * x * x);
            s.push(arc_length[j] + h * x);
            v.push(
                (2.0 * x3 - 3.0 * x2 + 1.0) * potential[j]
                    + (x3 - 2.0 * x2 + x) * h * slopes[j]
                    + (3.0 * x2 - 2.0 * x3) * potential[j + 1]
                    + (x3 - x2) * h * slopes[j + 1],
            );
            g.push(g_l + (g_r - g_l) * x);
        }
    }
    s.push(arc_length[count - 1]);
    v.push(potential[count - 1]);
    g.push(mass_ratio[count - 1].sqrt());
    let fine = s.len();
    let top = highest(&v);
    let depth = v[top] - v[0].max(v[fine - 1]);
    if depth.is_nan() || depth <= 0.0 {
        return Ok(1.0);
    }
    // Imaginary action at the energy `e` below the top, over the stretch around the top
    // where the barrier is above that energy.
    let scale = (2.0 * amu_in_electron_masses()).sqrt();
    let theta = |e: f64| -> f64 {
        let level = v[top] - e;
        let mut action = 0.0;
        // To the left of the top, then to the right.
        for j in (0..top).rev() {
            let (f_l, f_r) = (v[j] - level, v[j + 1] - level);
            let h = s[j + 1] - s[j];
            if f_l > 0.0 {
                action += h * piece(f_l, f_r, g[j], g[j + 1]);
            } else {
                // The turning point is in this interval, at the fraction `x` from its left.
                if f_r > 0.0 {
                    let x = f_l / (f_l - f_r);
                    let g_at = g[j] + (g[j + 1] - g[j]) * x;
                    action += h * (1.0 - x) * piece(0.0, f_r, g_at, g[j + 1]);
                }
                break;
            }
        }
        for j in top..fine - 1 {
            let (f_l, f_r) = (v[j] - level, v[j + 1] - level);
            let h = s[j + 1] - s[j];
            if f_r > 0.0 {
                action += h * piece(f_l, f_r, g[j], g[j + 1]);
            } else {
                if f_l > 0.0 {
                    let x = f_l / (f_l - f_r);
                    let g_at = g[j] + (g[j + 1] - g[j]) * x;
                    action += h * x * piece(f_l, 0.0, g[j], g_at);
                }
                break;
            }
        }
        scale * action
    };
    // 2 P sinh(e / kT) with P = 1 / (1 + exp(2 theta)), without overflow.
    let beta = 1.0 / (Units::BOLTZMANN_HARTREE_PER_K * temperature);
    let integrand = |e: f64| -> f64 {
        let damping = softplus(2.0 * theta(e));
        (beta * e - damping).exp() - (-beta * e - damping).exp()
    };
    let step = depth / ENERGY_INTERVALS as f64;
    let mut sum = integrand(0.0) + integrand(depth);
    for k in 1..ENERGY_INTERVALS {
        let weight = if k % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * integrand(step * k as f64);
    }
    Ok(1.0 + beta * sum * step / 3.0)
}

/// Tunneling through the vibrationally adiabatic barrier of a path.
#[derive(Debug, Clone)]
pub struct Tunneling {
    /// Temperature (K).
    pub temperature: f64,
    /// Vibrationally adiabatic ground-state potential at every point (Hartree).
    pub adiabatic: Vec<f64>,
    /// `mu_eff / mu` of small-curvature tunneling at every point.
    pub mass_ratio: Vec<f64>,
    /// Point of the highest adiabatic potential.
    pub top: usize,
    /// Lowest energy counted: the higher of the two ends of the path (Hartree).
    pub threshold: f64,
    /// Transmission coefficient of zero-curvature tunneling.
    pub zct: f64,
    /// Transmission coefficient of small-curvature tunneling.
    pub sct: f64,
}

/// Zero- and small-curvature tunneling for the path of `profile` with the energies
/// `energies` (Hartree) at `temperature` (K).
pub fn tunneling(
    energies: &[f64],
    profile: &PathProfile,
    temperature: f64,
) -> MoptResult<Tunneling> {
    let count = profile.arc_length.len();
    if energies.len() != count {
        return Err(MoptError::config(format!(
            "tunneling: {} energies for {count} points",
            energies.len()
        )));
    }
    let adiabatic: Vec<f64> = energies
        .iter()
        .zip(&profile.zero_point_energy)
        .map(|(e, z)| e + z)
        .collect();
    let mass_ratio = effective_mass_ratio(profile);
    let straight = vec![1.0; count];
    Ok(Tunneling {
        temperature,
        zct: transmission(&profile.arc_length, &adiabatic, &straight, temperature)?,
        sct: transmission(&profile.arc_length, &adiabatic, &mass_ratio, temperature)?,
        top: highest(&adiabatic),
        threshold: adiabatic[0].max(adiabatic[count - 1]),
        adiabatic,
        mass_ratio,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::vib::path::tests::circular_valley;
    use crate::vib::transverse_profile;

    /// A parabolic barrier with the imaginary frequency `wavenumber` (cm^-1) and the height
    /// `height` (Hartree) above its ends.
    fn parabola(wavenumber: f64, height: f64, points: usize) -> (Vec<f64>, Vec<f64>, f64) {
        // Frequency in atomic units, and the force constant in Hartree / (amu Bohr^2).
        let omega = wavenumber * Units::WAVENUMBER_TO_HARTREE;
        let k = omega * omega * amu_in_electron_masses();
        let reach = (2.0 * height / k).sqrt();
        let s: Vec<f64> = (0..points)
            .map(|i| reach * (2.0 * i as f64 / (points - 1) as f64 - 1.0))
            .collect();
        let v = s.iter().map(|x| -0.5 * k * x * x).collect();
        (s, v, omega)
    }

    #[test]
    fn a_parabolic_barrier_has_the_transmission_coefficient_of_its_frequency() {
        // u = hbar w / kT = 4.8 at 300 K for 1000 cm^-1: kappa = (u / 2) / sin(u / 2), the
        // semiclassical (and exact) result for a parabola, when the barrier is high enough
        // for the energies below its ends not to matter.
        let (s, v, omega) = parabola(1000.0, 0.05, 81);
        let ones = vec![1.0; s.len()];
        for temperature in [300.0, 500.0, 1000.0] {
            let u = omega / (Units::BOLTZMANN_HARTREE_PER_K * temperature);
            let expected = 0.5 * u / (0.5 * u).sin();
            let found = transmission(&s, &v, &ones, temperature).unwrap();
            assert!(
                (found - expected).abs() < 5e-3 * expected,
                "{temperature} K: {found} against {expected}"
            );
        }
        // A lighter effective mass tunnels more; no barrier, no tunneling.
        let light = vec![0.5; s.len()];
        assert!(
            transmission(&s, &v, &light, 300.0).unwrap()
                > transmission(&s, &v, &ones, 300.0).unwrap()
        );
        let downhill: Vec<f64> = s.iter().map(|x| -x).collect();
        assert_eq!(transmission(&s, &downhill, &ones, 300.0).unwrap(), 1.0);
        assert!(transmission(&s, &v, &ones, 0.0).is_err());
        assert!(transmission(&s[..1], &v[..1], &ones[..1], 300.0).is_err());
    }

    #[test]
    fn the_effective_mass_of_a_circular_valley() {
        // On the circle of radius R the curvature is 1 / (sqrt(m) R) and couples to the
        // radial mode alone (force constant 1, curvature 1 / m): the turning point of its
        // zero-point vibration is constant along the path.
        let radius = 2.0;
        let angles: Vec<f64> = (0..9).map(|k| 0.1 * k as f64).collect();
        let points = circular_valley(radius, &angles);
        let profile = transverse_profile(&points, false).unwrap();
        let mass = points[0].system.masses_amu()[0];
        let amu = amu_in_electron_masses();
        // w = sqrt(1 / (m amu)) in atomic units; t = sqrt(hbar / w) / sqrt(amu).
        let omega = (1.0 / (mass * amu)).sqrt();
        let a = (1.0 / omega).sqrt() / amu.sqrt() / (mass.sqrt() * radius);
        let expected = (-2.0 * a - a * a).exp();
        let ratio = effective_mass_ratio(&profile);
        for (i, r) in ratio.iter().enumerate() {
            assert!(
                (r - expected).abs() < 0.01 * expected,
                "{i}: {r} against {expected}"
            );
        }
        // A flat floor has no barrier: both transmission coefficients are 1, and the free
        // energy is the same at every point.
        let energies = vec![0.0; points.len()];
        let found = tunneling(&energies, &profile, 300.0).unwrap();
        assert!((found.zct - 1.0).abs() < 1e-9 && (found.sct - 1.0).abs() < 1e-9);
        let free = free_energy_profile(
            &points,
            &energies,
            &profile,
            &ThermoSettings::default(),
            false,
        )
        .unwrap();
        let first = free.free_energy[0];
        assert!(free.free_energy.iter().all(|g| (g - first).abs() < 1e-12));
        assert!((free.recrossing - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_variational_transition_state_is_where_the_free_energy_is_highest() {
        // A straight path along x with the barrier -0.005 x^2 and a transverse mode along y
        // that stiffens with x (force constant 0.2 + 0.1 x): the zero-point and thermal
        // free energy of that mode rise with x and move the maximum of the free energy to
        // x > 0, where conventional theory (dividing surface at x = 0) counts too much.
        use crate::domain::{positions, Bohr, Element, MolecularSystem, SpinCharge};
        use crate::vib::Direction;
        use faer::{Col, Mat};
        let xs: Vec<f64> = (0..21).map(|k| -0.5 + 0.05 * k as f64).collect();
        let points: Vec<PathPoint> = xs
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
        let energies: Vec<f64> = xs.iter().map(|x| -0.5e-2 * x * x).collect();
        let profile = transverse_profile(&points, false).unwrap();
        let settings = ThermoSettings {
            quasi_harmonic: crate::vib::QuasiHarmonic::None,
            ..ThermoSettings::default()
        };
        let free = free_energy_profile(&points, &energies, &profile, &settings, false).unwrap();
        assert_eq!(free.saddle, 10);
        assert!(free.variational > free.saddle && free.variational < 20);
        assert!(free.variational_arc_length > profile.arc_length[free.saddle]);
        assert!(
            (free.variational_arc_length - profile.arc_length[free.variational]).abs()
                < profile.arc_length[1]
        );
        assert!(free.variational_free_energy >= free.free_energy[free.variational]);
        assert!(free.recrossing < 1.0 && free.recrossing > 0.0);
        // The free energy is the energy plus the vibrational free energy of the two modes:
        // at 0 K-like stiffness (h nu >> kT) nearly the zero-point energy.
        for (i, energy) in energies.iter().enumerate() {
            let adiabatic = energy + profile.zero_point_energy[i];
            assert!((free.free_energy[i] - adiabatic).abs() < 1e-6);
        }
    }
}
