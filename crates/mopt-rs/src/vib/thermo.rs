//! Thermochemistry of an ideal gas of molecules: translation, rotation (rigid rotor),
//! vibration (harmonic oscillators), and the electronic degeneracy.
//!
//! Per molecule, with `k` the Boltzmann constant in Hartree/K:
//!
//! - translation: `q = (2 pi m k T / h^2)^(3/2) k T / p`, `S = k (5/2 + ln q)`, `E = 3/2 k T`,
//!   `H = 5/2 k T`;
//! - rotation: `q = k T / (sigma h B)` (linear), `q = sqrt(pi) (k T / h)^(3/2) / (sigma
//!   sqrt(A B C))` (otherwise), with the rotational constants `B = h / (8 pi^2 I)` and the
//!   symmetry number `sigma`;
//! - vibration, per mode with `x = theta / T`: `S = k (x / (e^x - 1) - ln(1 - e^-x))`,
//!   `E = k theta / 2 + k T x / (e^x - 1)`, `Cv = k x^2 e^-x / (1 - e^-x)^2`;
//! - electronic: `S = k ln(multiplicity)`.
//!
//! Low frequencies, for which the harmonic oscillator overestimates the entropy:
//!
//! - `truhlar`: wavenumbers below the cutoff are raised to the cutoff in the thermal terms
//!   (Ribeiro, Marenich, Cramer, Truhlar, J. Phys. Chem. B 2011, 115, 14556);
//! - `grimme`: with the weight `w = 1 / (1 + (cutoff / nu)^4)`, the entropy is `w S_HO +
//!   (1 - w) S_FR` with the entropy of a free rotor of the same frequency (Grimme, Chem. Eur.
//!   J. 2012, 18, 9955), and the thermal energy `w E_HO + (1 - w) k T / 2` (Li, Gomes, Sharada,
//!   Bell, Head-Gordon, J. Phys. Chem. C 2015, 119, 1840). The heat capacity is not corrected.
//!
//! By default the entropy and the thermal energy are treated as MultiOptPy pairs them
//! (`grimme` with the thermal energy of Li et al., `truhlar` in both). They can be chosen
//! separately, with separate cutoffs, as GoodVibes (Luchini, Alegre-Requena, Funes-Ardoiz,
//! Paton, F1000Research 2020, 9, 291) does: [`ThermoSettings::enthalpy`] and
//! [`ThermoSettings::enthalpy_cutoff`]. Further choices in the manner of GoodVibes:
//!
//! - a scale factor for the frequencies in the thermal terms and one for the zero-point
//!   energy (which takes the first unless given);
//! - a concentration: the translational entropy for the volume per molecule `1 / (c N_A)`
//!   of a solution instead of `k T / p` of the gas (at 298.15 K, 1 mol/L instead of 1 atm
//!   lowers `T S` by `R T ln 24.47` = 1.89 kcal/mol);
//! - imaginary frequencies up to a limit taken as real ones of the same magnitude (small
//!   imaginary frequencies of a structure that is a minimum for all purposes);
//! - the moment of inertia that bounds that of Grimme's free rotor: his average value of
//!   `1e-44 kg m^2`, or the mean moment of the molecule itself.
//!
//! The zero-point energy is always the harmonic one. Modes below 10 cm^-1 and imaginary
//! frequencies do not enter. The vibrational terms are written in decaying forms with `exp_m1`
//! (`x e^-x / (1 - e^-x)`, `x^2 e^-x / (1 - e^-x)^2`, `ln(1 - e^-x)` through
//! `math::stable::log_diff_exp`), so they stay finite and accurate for any `x = theta / T > 0`:
//! they decay to zero for large `x` (low temperatures, high frequencies) and reach their
//! classical limits without cancellation for small `x`.
//!
//! MultiOptPy: `Thermo/normal_mode_analyzer.py:MolecularVibrations.calculate_thermochemistry`,
//!  `get_rotational_constants`, `_get_rotor_type`. Corrected here (see
//!  `docs/source/migration_vibrations.md`): the rotational constants (MultiOptPy's are `2 pi` too large
//!  and are taken about a point shifted by the centre of mass), and the multiplicity of the
//!  electronic entropy (MultiOptPy's never exceeds 1).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use super::{MassChoice, NormalModes};
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;
use crate::math::projection::{shape, Shape};

/// Grimme's average moment of inertia of the free rotor (kg m^2).
const AVERAGE_MOMENT: f64 = 1e-44;

/// Treatment of low vibrational frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuasiHarmonic {
    /// Harmonic oscillators throughout.
    None,
    /// Frequencies below the cutoff are raised to it.
    Truhlar,
    /// Interpolation between the harmonic oscillator and a free rotor.
    Grimme,
}

impl QuasiHarmonic {
    /// `grimme`, `truhlar`, or `none` (`rrho`), as `-freq_corr`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "grimme" => Ok(Self::Grimme),
            "truhlar" => Ok(Self::Truhlar),
            "none" | "rrho" | "" => Ok(Self::None),
            other => Err(MoptError::config(format!(
                "unknown frequency correction '{other}' (grimme, truhlar, none)"
            ))),
        }
    }
}

/// Treatment of the thermal energy of low vibrational frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnthalpyTreatment {
    /// Harmonic oscillators throughout.
    None,
    /// Frequencies below the cutoff are raised to it.
    Truhlar,
    /// Interpolation between the harmonic oscillator and `k T / 2` (Li, Gomes, Sharada,
    /// Bell, Head-Gordon).
    HeadGordon,
}

impl EnthalpyTreatment {
    /// `head-gordon`, `truhlar`, or `none` (`rrho`).
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "head-gordon" | "headgordon" | "head_gordon" | "hg" => Ok(Self::HeadGordon),
            "truhlar" => Ok(Self::Truhlar),
            "none" | "rrho" => Ok(Self::None),
            other => Err(MoptError::config(format!(
                "unknown correction of the thermal energy '{other}' (head-gordon, truhlar, none)"
            ))),
        }
    }
}

/// The moment of inertia that bounds that of the free rotor in Grimme's entropy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorMoment {
    /// Grimme's average value, `1e-44 kg m^2`.
    Global,
    /// The mean of the moments of inertia of the molecule.
    Molecule,
}

impl RotorMoment {
    /// `global` or `molecule` (`conf`, as GoodVibes calls it).
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "global" => Ok(Self::Global),
            "molecule" | "conf" => Ok(Self::Molecule),
            other => Err(MoptError::config(format!(
                "unknown moment of the free rotor '{other}' (global, molecule)"
            ))),
        }
    }
}

/// Kind of rotor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorType {
    /// A single atom.
    Atom,
    /// A linear molecule.
    Linear,
    /// Any other molecule.
    Regular,
}

/// Conditions and choices of a thermochemistry calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermoSettings {
    /// Temperature (K): `-temp`, 298.15.
    pub temperature: f64,
    /// Pressure (Pa): `-press`, 101325.
    pub pressure: f64,
    /// `-freq_corr`: `grimme`.
    pub quasi_harmonic: QuasiHarmonic,
    /// Cutoff of the correction (cm^-1): `-freq_cutoff`, 100.
    pub cutoff: f64,
    /// Rotational symmetry number.
    pub symmetry_number: u32,
    /// Spin multiplicity `2S + 1`.
    pub multiplicity: u32,
    /// Modes at or below this wavenumber (cm^-1) are left out: 10.
    pub lowest: f64,
    /// Treatment of the thermal energy; `None`: the one MultiOptPy pairs with
    /// `quasi_harmonic` (`grimme`: Head-Gordon, `truhlar`: Truhlar, `none`: none).
    pub enthalpy: Option<EnthalpyTreatment>,
    /// Cutoff of the treatment of the thermal energy (cm^-1); `None`: `cutoff`.
    pub enthalpy_cutoff: Option<f64>,
    /// Concentration (mol/L) for the translational entropy instead of the pressure.
    pub concentration: Option<f64>,
    /// Scale factor of the frequencies in the thermal terms: 1.
    pub frequency_scale: f64,
    /// Scale factor of the frequencies in the zero-point energy; `None`: `frequency_scale`.
    pub zero_point_scale: Option<f64>,
    /// Imaginary frequencies up to this magnitude (cm^-1) are taken as real.
    pub invert_below: Option<f64>,
    /// Bound of the moment of the free rotor in Grimme's entropy.
    pub rotor_moment: RotorMoment,
    /// The masses of the translation and the rotation (and of the normal modes, in a
    /// vibrational analysis): `-freq_masses`, standard.
    pub masses: MassChoice,
}

impl Default for ThermoSettings {
    fn default() -> Self {
        Self {
            temperature: 298.15,
            pressure: 101_325.0,
            quasi_harmonic: QuasiHarmonic::Grimme,
            cutoff: 100.0,
            symmetry_number: 1,
            multiplicity: 1,
            lowest: 10.0,
            enthalpy: None,
            enthalpy_cutoff: None,
            concentration: None,
            frequency_scale: 1.0,
            zero_point_scale: None,
            invert_below: None,
            rotor_moment: RotorMoment::Global,
            masses: MassChoice::Standard,
        }
    }
}

/// A thermodynamic function and its contributions.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Parts {
    /// Sum of the contributions.
    pub total: f64,
    /// Electronic.
    pub electronic: f64,
    /// Translational.
    pub translational: f64,
    /// Rotational.
    pub rotational: f64,
    /// Vibrational.
    pub vibrational: f64,
}

impl Parts {
    fn of(electronic: f64, translational: f64, rotational: f64, vibrational: f64) -> Self {
        Self {
            total: electronic + translational + rotational + vibrational,
            electronic,
            translational,
            rotational,
            vibrational,
        }
    }
}

/// Thermodynamic functions per molecule (energies in Hartree, entropies and heat capacities
/// in Hartree/K).
#[derive(Debug, Clone, PartialEq)]
pub struct Thermochemistry {
    /// Temperature (K).
    pub temperature: f64,
    /// Pressure (Pa).
    pub pressure: f64,
    /// Electronic energy.
    pub electronic_energy: f64,
    /// Harmonic zero-point energy.
    pub zero_point_energy: f64,
    /// Rotational constants `A >= B >= C` (GHz; infinite for an axis without a moment).
    pub rotational_constants_ghz: [f64; 3],
    /// Kind of rotor.
    pub rotor: RotorType,
    /// Symmetry number used.
    pub symmetry_number: u32,
    /// Entropy.
    pub entropy: Parts,
    /// Heat capacity at constant volume.
    pub cv: Parts,
    /// Heat capacity at constant pressure.
    pub cp: Parts,
    /// Internal energy.
    pub internal_energy: Parts,
    /// Enthalpy.
    pub enthalpy: Parts,
    /// Gibbs free energy.
    pub gibbs: Parts,
    /// Electronic energy plus zero-point energy.
    pub energy_0k: f64,
    /// Vibrational entropy, enthalpy and Gibbs free energy without the low-frequency
    /// correction, when one was applied.
    pub uncorrected_vibration: Option<[f64; 3]>,
    /// Number of vibrations that entered.
    pub vibrations: usize,
    /// Number of imaginary frequencies (not included).
    pub imaginary: usize,
    /// Number of imaginary frequencies taken as real.
    pub inverted: usize,
    /// Concentration used for the translational entropy (mol/L), if any.
    pub concentration: Option<f64>,
    /// Scale factors of the frequencies in the thermal terms and in the zero-point energy.
    pub scale: (f64, f64),
    /// The masses used.
    pub masses: MassChoice,
}

/// Rotational constants `h / (8 pi^2 I)` of `sys` about its centre of mass, largest first
/// (GHz), with the standard atomic weights.
pub fn rotational_constants_ghz(sys: &MolecularSystem) -> MoptResult<[f64; 3]> {
    rotational_constants_with_masses(sys, &sys.masses_amu())
}

/// [`rotational_constants_ghz`] with the masses `masses` of the atoms (amu).
pub fn rotational_constants_with_masses(
    sys: &MolecularSystem,
    masses: &[f64],
) -> MoptResult<[f64; 3]> {
    if masses.len() != sys.n_atoms() {
        return Err(MoptError::config(format!(
            "rotational constants: {} masses for {} atoms",
            masses.len(),
            sys.n_atoms()
        )));
    }
    let total: f64 = masses.iter().sum();
    let com: [f64; 3] = std::array::from_fn(|c| {
        sys.atoms()
            .zip(masses)
            .map(|(a, m)| m * sys.position(a).value()[c])
            .sum::<f64>()
            / total
    });
    let mut inertia = Mat::<f64>::zeros(3, 3);
    for (a, atom) in sys.atoms().enumerate() {
        let r = sys.position(atom).value();
        let d = [r[0] - com[0], r[1] - com[1], r[2] - com[2]];
        let square = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for i in 0..3 {
            inertia[(i, i)] += masses[a] * square;
            for j in 0..3 {
                inertia[(i, j)] -= masses[a] * d[i] * d[j];
            }
        }
    }
    let (moments, _) = symmetric_eigen(&inertia)?;
    let unit = Units::AMU_TO_KG * Units::BOHR_TO_METER * Units::BOHR_TO_METER;
    Ok(std::array::from_fn(|k| {
        let moment = moments[k].max(0.0) * unit;
        Units::PLANCK_J_S / (8.0 * PI * PI * moment) * 1e-9
    }))
}

/// The rotor of `sys`: the rigid-body shape of the projection (`math::projection::shape`), so
/// that the rotor, the projected rotations and the number of vibrations always agree.
pub fn rotor_type(sys: &MolecularSystem) -> RotorType {
    match shape(sys) {
        Shape::Point => RotorType::Atom,
        Shape::Linear { .. } => RotorType::Linear,
        Shape::Nonlinear => RotorType::Regular,
    }
}

/// The rotational constants of a rotor of the kind `rotor`: an axis without a moment (the
/// line of a linear rotor, every axis of an atom) has an infinite constant, and the two
/// moments of a linear rotor, which agree up to the noise of the structure off its line, are
/// replaced by their geometric mean.
fn rotor_constants(constants: [f64; 3], rotor: RotorType) -> [f64; 3] {
    match rotor {
        RotorType::Atom => [f64::INFINITY; 3],
        RotorType::Linear => {
            let b = (constants[1] * constants[2]).sqrt();
            [f64::INFINITY, b, b]
        }
        RotorType::Regular => constants,
    }
}

/// Entropy (in units of `k`) and thermal energy (in units of `k T`) of a harmonic
/// oscillator with `x = theta / T > 0`.
///
/// `x / (e^x - 1)` is written `x e^-x / (1 - e^-x)` with `1 - e^-x = -expm1(-x)`: no
/// overflow for large `x` (it decays to zero), and the limit 1 for `x -> 0` without
/// cancellation. `-ln(1 - e^-x)` is `-math::stable::log_diff_exp(0, -x)`, accurate for small
/// `x` (`-ln x`) and large `x` (`e^-x`).
///
/// MultiOptPy: `Thermo/normal_mode_analyzer.py:MolecularVibrations.calculate_thermochemistry`
/// (the vibrational terms, written there with `exp` directly).
pub(crate) fn oscillator(x: f64) -> (f64, f64) {
    let thermal = oscillator_thermal(x);
    let entropy = if x > ENTROPY_TAIL {
        // `S = x e^-x / (1 - e^-x) - ln(1 - e^-x) = (x + 1) e^-x (1 + O(e^-x))`, in
        // logarithms: near the end of the doubles `e^-x` alone is subnormal and has lost
        // its digits.
        ((x + 1.0).ln() - x).exp()
    } else {
        thermal - crate::math::stable::log_diff_exp(0.0, -x)
    };
    (entropy, thermal)
}

/// Beyond this `x` the entropy of an oscillator is `(x + 1) e^-x` to the last digit
/// (the next term is `e^-x` smaller, below `1e-17`).
const ENTROPY_TAIL: f64 = 40.0;

/// `x / (e^x - 1)` for `x > 0` in the decaying form `x e^-x / (-expm1(-x))`, in logarithms
/// for `x > 1` (so that it is not formed from a subnormal `e^-x`).
fn oscillator_thermal(x: f64) -> f64 {
    if x > 1.0 {
        (x.ln() - x - crate::math::stable::log_diff_exp(0.0, -x)).exp()
    } else {
        x * (-x).exp() / -(-x).exp_m1()
    }
}

/// Heat capacity of a harmonic oscillator in units of `k` for `x = theta / T > 0`:
/// `x^2 e^x / (e^x - 1)^2`, written in the decaying form `x^2 e^-x / (-expm1(-x))^2`. The
/// form `(x / expm1 x)^2 e^x` is `0 * inf` (NaN) beyond `x` of about 709 (a mode of
/// 4000 cm^-1 below 8 K); this one decays to zero there and tends to 1 for `x -> 0`.
///
/// MultiOptPy: `Thermo/normal_mode_analyzer.py:MolecularVibrations.calculate_thermochemistry`
/// (the vibrational heat capacity).
pub(crate) fn oscillator_heat(x: f64) -> f64 {
    if x > 1.0 {
        // In logarithms, so that a value near the end of the normal doubles is not formed
        // from the subnormal e^-x (beyond x of about 708) with its lost digits.
        (2.0 * x.ln() - x - 2.0 * crate::math::stable::log_diff_exp(0.0, -x)).exp()
    } else {
        let gap = -(-x).exp_m1();
        (x / gap) * (x / gap) * (-x).exp()
    }
}

/// Thermodynamic functions of `sys` with the normal modes `modes` and the electronic energy
/// `electronic_energy` (Hartree).
pub fn thermochemistry(
    sys: &MolecularSystem,
    modes: &NormalModes,
    electronic_energy: f64,
    settings: &ThermoSettings,
) -> MoptResult<Thermochemistry> {
    let t = settings.temperature;
    if !(t > 0.0 && t.is_finite() && settings.pressure > 0.0 && settings.pressure.is_finite()) {
        return Err(MoptError::config(
            "thermochemistry: temperature and pressure must be positive",
        ));
    }
    if settings.symmetry_number == 0 || settings.multiplicity == 0 {
        return Err(MoptError::config(
            "thermochemistry: symmetry number and multiplicity start at 1",
        ));
    }
    let scale = settings.frequency_scale;
    let zero_point_scale = settings.zero_point_scale.unwrap_or(scale);
    let positive = |v: f64| v > 0.0 && v.is_finite();
    if !(positive(scale) && positive(zero_point_scale)) {
        return Err(MoptError::config(
            "thermochemistry: the scale factors of the frequencies must be positive",
        ));
    }
    if settings.concentration.is_some_and(|c| !positive(c)) {
        return Err(MoptError::config(
            "thermochemistry: the concentration must be positive",
        ));
    }
    let k = Units::BOLTZMANN_HARTREE_PER_K;
    let kt = Units::BOLTZMANN_J_PER_K * t;
    let sigma = f64::from(settings.symmetry_number);

    let s_elec = k * f64::from(settings.multiplicity).ln();

    // Volume per molecule: of the ideal gas, or of a solution of the given concentration.
    let volume = match settings.concentration {
        Some(concentration) => 1.0 / (concentration * 1000.0 * Units::AVOGADRO),
        None => kt / settings.pressure,
    };
    let masses = settings.masses.masses(sys)?;
    let mass = masses.iter().sum::<f64>() * Units::AMU_TO_KG;
    let q_trans =
        (2.0 * PI * mass * kt / (Units::PLANCK_J_S * Units::PLANCK_J_S)).powf(1.5) * volume;
    let s_trans = k * (2.5 + q_trans.ln());

    let rotor = rotor_type(sys);
    let constants = rotor_constants(rotational_constants_with_masses(sys, &masses)?, rotor);
    let (s_rot, c_rot, e_rot) = match rotor {
        RotorType::Atom => (0.0, 0.0, 0.0),
        RotorType::Linear => {
            let q = kt / (sigma * Units::PLANCK_J_S * constants[1] * 1e9);
            (k * (1.0 + q.ln()), k, k * t)
        }
        RotorType::Regular => {
            let product: f64 = constants.iter().map(|c| c * 1e9).product();
            let q = (kt / Units::PLANCK_J_S).powf(1.5) * PI.sqrt() / (sigma * product.sqrt());
            (k * (1.5 + q.ln()), 1.5 * k, 1.5 * k * t)
        }
    };

    // The vibrations: real ones above the lowest wavenumber, and imaginary ones up to the
    // limit, if one is set, as real ones.
    let mut inverted = 0;
    let wavenumbers: Vec<f64> = modes
        .wavenumbers
        .iter()
        .filter_map(|w| {
            if *w > settings.lowest {
                Some(*w)
            } else if *w < -settings.lowest
                && settings.invert_below.is_some_and(|limit| -*w <= limit)
            {
                inverted += 1;
                Some(-*w)
            } else {
                None
            }
        })
        .collect();
    let to_theta = Units::HC_OVER_KB_CM_K;
    let entropy_cutoff = settings.cutoff * to_theta;
    let energy_cutoff = settings.enthalpy_cutoff.unwrap_or(settings.cutoff) * to_theta;
    let energy_treatment = settings.enthalpy.unwrap_or(match settings.quasi_harmonic {
        QuasiHarmonic::None => EnthalpyTreatment::None,
        QuasiHarmonic::Truhlar => EnthalpyTreatment::Truhlar,
        QuasiHarmonic::Grimme => EnthalpyTreatment::HeadGordon,
    });
    let moment_bound = match settings.rotor_moment {
        RotorMoment::Global => AVERAGE_MOMENT,
        RotorMoment::Molecule => {
            // h / (8 pi^2 B) of the axes that have a moment.
            let moments: Vec<f64> = constants
                .iter()
                .filter(|c| c.is_finite() && **c > 0.0)
                .map(|c| Units::PLANCK_J_S / (8.0 * PI * PI * c * 1e9))
                .collect();
            if moments.is_empty() {
                AVERAGE_MOMENT
            } else {
                moments.iter().sum::<f64>() / moments.len() as f64
            }
        }
    };
    let zpe = k * 0.5 * to_theta * zero_point_scale * wavenumbers.iter().sum::<f64>();
    let (mut s_vib, mut e_thermal, mut cv_vib) = (0.0, 0.0, 0.0);
    let (mut s_plain, mut e_plain) = (0.0, 0.0);
    let heat = oscillator_heat;
    for &wavenumber in &wavenumbers {
        let theta = wavenumber * scale * to_theta;
        let x = theta / t;
        let (entropy, thermal) = oscillator(x);
        s_plain += k * entropy;
        e_plain += k * t * thermal;
        match settings.quasi_harmonic {
            QuasiHarmonic::None => s_vib += k * entropy,
            QuasiHarmonic::Truhlar => s_vib += k * oscillator(theta.max(entropy_cutoff) / t).0,
            QuasiHarmonic::Grimme => {
                let weight = 1.0 / (1.0 + (entropy_cutoff / theta).powi(4));
                // Free rotor with the moment of inertia of the frequency, limited by the
                // bound.
                let moment = Units::PLANCK_J_S * Units::PLANCK_J_S
                    / (8.0 * PI * PI * theta * Units::BOLTZMANN_J_PER_K);
                let reduced = moment * moment_bound / (moment + moment_bound);
                let factor = (8.0 * PI.powi(3) * reduced * kt
                    / (Units::PLANCK_J_S * Units::PLANCK_J_S))
                    .max(1e-10);
                let free_rotor = 0.5 + 0.5 * factor.ln();
                s_vib += k * (weight * entropy + (1.0 - weight) * free_rotor);
            }
        }
        match energy_treatment {
            EnthalpyTreatment::None => {
                e_thermal += k * t * thermal;
                cv_vib += k * heat(x);
            }
            EnthalpyTreatment::Truhlar => {
                let raised = theta.max(energy_cutoff) / t;
                e_thermal += k * t * oscillator(raised).1;
                cv_vib += k * heat(raised);
            }
            EnthalpyTreatment::HeadGordon => {
                let weight = 1.0 / (1.0 + (energy_cutoff / theta).powi(4));
                e_thermal += k * t * (weight * thermal + (1.0 - weight) * 0.5);
                cv_vib += k * heat(x);
            }
        }
    }
    let h_vib = zpe + e_thermal;

    let entropy = Parts::of(s_elec, s_trans, s_rot, s_vib);
    let internal_energy = Parts::of(electronic_energy, 1.5 * k * t, e_rot, h_vib);
    let enthalpy = Parts::of(electronic_energy, 2.5 * k * t, e_rot, h_vib);
    let gibbs = Parts::of(
        enthalpy.electronic - t * entropy.electronic,
        enthalpy.translational - t * entropy.translational,
        enthalpy.rotational - t * entropy.rotational,
        enthalpy.vibrational - t * entropy.vibrational,
    );
    let corrected = settings.quasi_harmonic != QuasiHarmonic::None
        || energy_treatment != EnthalpyTreatment::None;
    let uncorrected_vibration = corrected.then(|| {
        let h = zpe + e_plain;
        [s_plain, h, h - t * s_plain]
    });
    Ok(Thermochemistry {
        temperature: t,
        pressure: settings.pressure,
        electronic_energy,
        zero_point_energy: zpe,
        rotational_constants_ghz: constants,
        rotor,
        symmetry_number: settings.symmetry_number,
        entropy,
        cv: Parts::of(0.0, 1.5 * k, c_rot, cv_vib),
        cp: Parts::of(0.0, 2.5 * k, c_rot, cv_vib),
        internal_energy,
        enthalpy,
        gibbs,
        energy_0k: electronic_energy + zpe,
        uncorrected_vibration,
        vibrations: wavenumbers.len(),
        // By the one criterion of the crate, less those taken as real.
        imaginary: super::imaginary_wavenumbers(&modes.wavenumbers).saturating_sub(inverted),
        inverted,
        concentration: settings.concentration,
        scale: (scale, zero_point_scale),
        masses: settings.masses,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};
    use crate::vib::normal_modes;

    /// Hartree/K to J/(mol K).
    const TO_J_MOL_K: f64 = Units::HARTREE_TO_JOULE * Units::AVOGADRO;

    fn molecule(symbols: &[&str], xyz: &[[f64; 3]]) -> MolecularSystem {
        MolecularSystem::new(
            symbols
                .iter()
                .map(|s| Element::parse_token(s).unwrap())
                .collect(),
            &positions::<Angstrom>(xyz),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn an_atom_has_the_sackur_tetrode_entropy() {
        // Argon at 298.15 K and 1 bar: 154.85 J/(mol K).
        let ar = molecule(&["Ar"], &[[0.0; 3]]);
        let modes = normal_modes(&ar, &Mat::zeros(3, 3), true).unwrap();
        let settings = ThermoSettings {
            pressure: 1e5,
            ..ThermoSettings::default()
        };
        let out = thermochemistry(&ar, &modes, -1.0, &settings).unwrap();
        assert_eq!((out.rotor, out.vibrations), (RotorType::Atom, 0));
        assert!((out.entropy.total * TO_J_MOL_K - 154.85).abs() < 0.02);
        let k = Units::BOLTZMANN_HARTREE_PER_K;
        assert!((out.enthalpy.total + 1.0 - 2.5 * k * 298.15).abs() < 1e-15);
        assert!(
            (out.gibbs.total - (out.enthalpy.total - 298.15 * out.entropy.total)).abs() < 1e-15
        );
        // A doublet adds k ln 2.
        let doublet = ThermoSettings {
            multiplicity: 2,
            ..settings
        };
        let open = thermochemistry(&ar, &modes, -1.0, &doublet).unwrap();
        assert!((open.entropy.electronic - k * 2.0_f64.ln()).abs() < 1e-18);
    }

    #[test]
    fn a_diatomic_molecule_rotates_and_vibrates_as_in_the_textbook() {
        // N2: bond 1.0977 Angstrom gives B = 59.9 GHz (1.998 cm^-1); with a wavenumber of
        // 2359 cm^-1 and sigma = 2 the entropy at 298.15 K and 1 bar is 191.6 J/(mol K).
        let n2 = molecule(&["N", "N"], &[[0.0; 3], [0.0, 0.0, 1.0977]]);
        let constants = rotational_constants_ghz(&n2).unwrap();
        assert!(constants[0] > 1e8);
        assert!((constants[1] / (Units::SPEED_OF_LIGHT_M_S * 1e-7) - 1.998).abs() < 0.005);
        assert!((constants[1] - constants[2]).abs() < 1e-6);
        // A stretching force constant that gives 2359 cm^-1.
        let m = n2.masses_amu()[0];
        let hz = 2359.0 * Units::SPEED_OF_LIGHT_M_S * 100.0;
        let curvature = (hz / crate::vib::curvature_to_hz()).powi(2) * (m / 2.0);
        let mut h = Mat::<f64>::zeros(6, 6);
        for (a, b, v) in [(2, 2, 1.0), (5, 5, 1.0), (2, 5, -1.0), (5, 2, -1.0)] {
            h[(a, b)] = v * curvature;
        }
        let modes = normal_modes(&n2, &h, true).unwrap();
        assert!((modes.wavenumbers[5] - 2359.0).abs() < 1e-6);
        let settings = ThermoSettings {
            pressure: 1e5,
            symmetry_number: 2,
            quasi_harmonic: QuasiHarmonic::None,
            ..ThermoSettings::default()
        };
        let out = thermochemistry(&n2, &modes, 0.0, &settings).unwrap();
        assert_eq!((out.rotor, out.vibrations), (RotorType::Linear, 1));
        assert!(
            (out.entropy.total * TO_J_MOL_K - 191.6).abs() < 0.1,
            "{}",
            out.entropy.total * TO_J_MOL_K
        );
        // Zero-point energy h nu / 2.
        let zpe = 0.5 * Units::PLANCK_J_S * hz / Units::HARTREE_TO_JOULE;
        assert!((out.zero_point_energy - zpe).abs() < 1e-12);
        assert!((out.energy_0k - zpe).abs() < 1e-12);
        // Rotation of a linear molecule: E = k T, Cv = k.
        let k = Units::BOLTZMANN_HARTREE_PER_K;
        assert!((out.internal_energy.rotational - k * 298.15).abs() < 1e-15);
        assert!((out.cp.total - out.cv.total - k).abs() < 1e-18);
    }

    #[test]
    fn low_frequencies_are_corrected_and_high_ones_are_not() {
        let n2 = molecule(&["N", "N"], &[[0.0; 3], [0.0, 0.0, 1.0977]]);
        let m = n2.masses_amu()[0];
        let run = |wavenumber: f64, correction: QuasiHarmonic| {
            let hz = wavenumber * Units::SPEED_OF_LIGHT_M_S * 100.0;
            let curvature = (hz / crate::vib::curvature_to_hz()).powi(2) * (m / 2.0);
            let mut h = Mat::<f64>::zeros(6, 6);
            for (a, b, v) in [(2, 2, 1.0), (5, 5, 1.0), (2, 5, -1.0), (5, 2, -1.0)] {
                h[(a, b)] = v * curvature;
            }
            let modes = normal_modes(&n2, &h, true).unwrap();
            let settings = ThermoSettings {
                quasi_harmonic: correction,
                ..ThermoSettings::default()
            };
            thermochemistry(&n2, &modes, 0.0, &settings).unwrap()
        };
        for correction in [QuasiHarmonic::Truhlar, QuasiHarmonic::Grimme] {
            // Far above the cutoff of 100 cm^-1: as the harmonic oscillator.
            let (plain, corrected) = (run(2000.0, QuasiHarmonic::None), run(2000.0, correction));
            // (Grimme's weight differs from 1 by (100 / 2000)^4 there.)
            let k = Units::BOLTZMANN_HARTREE_PER_K;
            assert!((plain.entropy.vibrational - corrected.entropy.vibrational).abs() < 1e-4 * k);
            assert!((plain.enthalpy.vibrational - corrected.enthalpy.vibrational).abs() < 1e-8);
            // At 20 cm^-1 the entropy is reduced, the zero-point energy is kept, and the
            // uncorrected values are reported.
            let (plain, corrected) = (run(20.0, QuasiHarmonic::None), run(20.0, correction));
            assert!(corrected.entropy.vibrational < 0.8 * plain.entropy.vibrational);
            assert_eq!(corrected.zero_point_energy, plain.zero_point_energy);
            let kept = corrected.uncorrected_vibration.unwrap();
            assert!((kept[0] - plain.entropy.vibrational).abs() < 1e-18);
        }
        // Truhlar at 20 cm^-1 is the oscillator of 100 cm^-1 in the thermal terms.
        let (low, at_cutoff) = (
            run(20.0, QuasiHarmonic::Truhlar),
            run(100.0, QuasiHarmonic::None),
        );
        assert!((low.entropy.vibrational - at_cutoff.entropy.vibrational).abs() < 1e-15);
        // Modes below 10 cm^-1 and imaginary ones are left out.
        assert_eq!(run(5.0, QuasiHarmonic::None).vibrations, 0);
        assert!(QuasiHarmonic::parse("Grimme").is_ok() && QuasiHarmonic::parse("x").is_err());
    }

    // The reference values are the 17 digits that identify the doubles.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn oscillator_terms_are_finite_and_exact_from_1_k_to_5000_k() {
        // x = theta / T of 4000 cm^-1 at 1, 8, 298.15, 5000 K, and of 100 and 10.5 cm^-1 at
        // 5000 K, with the heat capacity, x / (e^x - 1) and the entropy (units of k) evaluated
        // with 400 digits (Python decimal; 1 - e^-x needs over 300 at x = 719) for the same
        // doubles.
        let exact = [
            // 1.28e-2492, 2.23e-2496 and 2.23e-2496: zero in doubles.
            (5755.107510015735, 0.0, 0.0, 0.0),
            (
                719.3884387519669,
                1.9386419808980193e-307,
                2.6948472848149713e-310,
                2.6985933101048552e-310,
            ),
            (
                19.30272517194612,
                1.5423048716295548e-6,
                7.9900887128980779e-8,
                8.4040244978811053e-8,
            ),
            (
                1.1510215020031471,
                8.9654184417988672e-1,
                5.3253007347119014e-1,
                9.1278585776848783e-1,
            ),
            (
                0.028775537550078677,
                9.9993100022661591e-1,
                9.8568123290281484e-1,
                4.5482641439999834e+0,
            ),
            (
                0.003021431442758261,
                9.9999923924635031e-1,
                9.9849004503250206e-1,
                6.8020249526006435e+0,
            ),
        ];
        let close = |got: f64, want: f64| {
            got.is_finite() && (got - want).abs() <= 1e-13 * want.abs() + 1e-320
        };
        for (x, heat, thermal, entropy) in exact {
            let (s, e) = oscillator(x);
            assert!(
                close(oscillator_heat(x), heat),
                "{x}: {}",
                oscillator_heat(x)
            );
            assert!(close(e, thermal), "{x}: {e}");
            assert!(close(s, entropy), "{x}: {s}");
        }
        // The former form of the heat capacity, (x / expm1 x)^2 e^x, is 0 * inf there.
        let x: f64 = 719.3884387519669;
        assert!(((x / x.exp_m1()).powi(2) * x.exp()).is_nan());
        // Through the whole thermochemistry: a mode of 4000 cm^-1 at 1 K and 8 K.
        let water = molecule(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
        );
        let modes = modes_of(&[1600.0, 3700.0, 4000.0]);
        for (temperature, correction) in [
            (1.0, QuasiHarmonic::None),
            (8.0, QuasiHarmonic::Grimme),
            (8.0, QuasiHarmonic::Truhlar),
            (5000.0, QuasiHarmonic::Grimme),
        ] {
            let settings = ThermoSettings {
                temperature,
                quasi_harmonic: correction,
                ..ThermoSettings::default()
            };
            let out = thermochemistry(&water, &modes, -76.0, &settings).unwrap();
            for v in [
                out.cv.vibrational,
                out.entropy.vibrational,
                out.enthalpy.vibrational,
                out.gibbs.total,
            ] {
                assert!(v.is_finite(), "{temperature} K: {v}");
            }
        }
    }

    /// Modes with the wavenumbers `wavenumbers` (cm^-1) and nothing else.
    fn modes_of(wavenumbers: &[f64]) -> NormalModes {
        let to_theta = Units::HC_OVER_KB_CM_K;
        NormalModes {
            wavenumbers: wavenumbers.to_vec(),
            curvatures: wavenumbers.iter().map(|w| w.signum()).collect(),
            displacements: Vec::new(),
            reduced_masses: Vec::new(),
            force_constants: Vec::new(),
            temperatures: wavenumbers.iter().map(|w| w.max(0.0) * to_theta).collect(),
        }
    }

    #[test]
    fn choices_in_the_manner_of_goodvibes() {
        let water = molecule(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
        );
        let modes = modes_of(&[-40.0, 30.0, 1600.0, 3700.0]);
        let run = |change: &dyn Fn(&mut ThermoSettings)| {
            let mut settings = ThermoSettings::default();
            change(&mut settings);
            thermochemistry(&water, &modes, -76.0, &settings).unwrap()
        };
        let plain = run(&|_| {});
        let k = Units::BOLTZMANN_HARTREE_PER_K;
        let t = 298.15;
        // 1 mol/L instead of 1 atm: T S falls by k T ln(24.47), 1.89 kcal/mol.
        let solution = run(&|s| s.concentration = Some(1.0));
        let change = t * (plain.entropy.translational - solution.entropy.translational);
        let molar_volume = Units::GAS_CONSTANT_J_MOL_K * t / 101_325.0 * 1000.0;
        assert!((change - k * t * molar_volume.ln()).abs() < 1e-12);
        assert!((change * Units::HARTREE_TO_KCAL_MOL - 1.89).abs() < 0.01);
        assert_eq!(solution.enthalpy.total, plain.enthalpy.total);
        // The concentration of the ideal gas gives the gas back.
        let gas = run(&|s| s.concentration = Some(1.0 / molar_volume));
        assert!((gas.entropy.translational - plain.entropy.translational).abs() < 1e-9 * k);
        // Scale factors: the zero-point energy follows its own, or that of the frequencies.
        let scaled = run(&|s| s.frequency_scale = 0.96);
        assert!((scaled.zero_point_energy - 0.96 * plain.zero_point_energy).abs() < 1e-15);
        assert!(scaled.entropy.vibrational > plain.entropy.vibrational);
        let both = run(&|s| {
            s.frequency_scale = 0.96;
            s.zero_point_scale = Some(1.0);
        });
        assert_eq!(both.zero_point_energy, plain.zero_point_energy);
        assert_eq!(both.entropy.vibrational, scaled.entropy.vibrational);
        assert_eq!(both.scale, (0.96, 1.0));
        // Entropy and thermal energy apart: Grimme's entropy with the plain thermal energy,
        // and the plain entropy with the energy of Li et al.
        let harmonic = run(&|s| s.quasi_harmonic = QuasiHarmonic::None);
        let entropy_only = run(&|s| s.enthalpy = Some(EnthalpyTreatment::None));
        assert_eq!(entropy_only.entropy.vibrational, plain.entropy.vibrational);
        assert_eq!(
            entropy_only.enthalpy.vibrational,
            harmonic.enthalpy.vibrational
        );
        let energy_only = run(&|s| {
            s.quasi_harmonic = QuasiHarmonic::None;
            s.enthalpy = Some(EnthalpyTreatment::HeadGordon);
        });
        assert_eq!(
            energy_only.entropy.vibrational,
            harmonic.entropy.vibrational
        );
        assert_eq!(energy_only.enthalpy.vibrational, plain.enthalpy.vibrational);
        // A cutoff of its own for the thermal energy: with none, the plain thermal energy.
        let no_cutoff = run(&|s| s.enthalpy_cutoff = Some(1e-6));
        assert!((no_cutoff.enthalpy.vibrational - harmonic.enthalpy.vibrational).abs() < 1e-12);
        // The imaginary frequency of 40 cm^-1 as a real one.
        assert_eq!(
            (plain.vibrations, plain.imaginary, plain.inverted),
            (3, 1, 0)
        );
        let inverted = run(&|s| s.invert_below = Some(50.0));
        assert_eq!(
            (inverted.vibrations, inverted.imaginary, inverted.inverted),
            (4, 0, 1)
        );
        assert!(inverted.zero_point_energy > plain.zero_point_energy);
        let kept = run(&|s| s.invert_below = Some(30.0));
        assert_eq!((kept.vibrations, kept.imaginary, kept.inverted), (3, 1, 0));
        // The moment of the molecule (about 1e-47 kg m^2) bounds the free rotor more than
        // Grimme's average: less entropy from the low mode.
        let own = run(&|s| s.rotor_moment = RotorMoment::Molecule);
        assert!(own.entropy.vibrational < plain.entropy.vibrational);
        // What is refused.
        let refused = |change: &dyn Fn(&mut ThermoSettings)| {
            let mut settings = ThermoSettings::default();
            change(&mut settings);
            thermochemistry(&water, &modes, 0.0, &settings).is_err()
        };
        assert!(refused(&|s| s.frequency_scale = 0.0));
        assert!(refused(&|s| s.concentration = Some(-1.0)));
        assert!(EnthalpyTreatment::parse("Head-Gordon").is_ok());
        assert!(EnthalpyTreatment::parse("x").is_err());
        assert!(RotorMoment::parse("conf").is_ok() && RotorMoment::parse("x").is_err());
    }
}
