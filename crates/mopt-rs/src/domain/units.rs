//! Physical constants and unit conversion factors: one CODATA 2018 set for the whole crate.
//!
//! The set is built from the exact SI defining constants (`h`, `c`, `e`, `k_B`, `N_A`), the
//! thermochemical calorie (4.184 J) and the CODATA 2018 values of the Bohr radius, the Hartree
//! energy, the electron mass and the atomic mass constant. Every other factor is derived here,
//! as a `const` expression or, where the derivation needs a square root, as a literal that a
//! unit test checks against its derivation.
//!
//! MultiOptPy's own (older) values live in `parameters::multioptpy_tables` (`MOP_*`) and are
//! used only by `legacy_` code paths and comparisons with MultiOptPy reference data.
//!
//! MultiOptPy: `Parameters/unit_values.py:UnitValueLib` (same role, different values).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

/// Canonical physical constants and unit conversions used across `mopt-rs` (CODATA 2018).
///
/// Internal atomic units throughout `mopt-rs`:
/// - Length: Bohr (`a_0`)
/// - Energy: Hartree (`E_h`)
/// - Force / Gradient: `Hartree / Bohr`
/// - Hessian: `Hartree / Bohr^2`
pub struct Units;

impl Units {
    // Exact SI defining constants (2019 redefinition, adopted by CODATA 2018).

    /// Planck constant `h` in `J * s` (exact).
    pub const PLANCK_J_S: f64 = 6.626_070_15e-34;
    /// Speed of light `c` in `m / s` (exact).
    pub const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;
    /// Speed of light `c` in `cm / s` (exact).
    pub const SPEED_OF_LIGHT_CM_S: f64 = Self::SPEED_OF_LIGHT_M_S * 100.0;
    /// Elementary charge `e` in `C` (exact).
    pub const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;
    /// Boltzmann constant in `J / K` (exact).
    pub const BOLTZMANN_J_PER_K: f64 = 1.380_649e-23;
    /// Avogadro constant (`mol^-1`, exact).
    pub const AVOGADRO: f64 = 6.022_140_76e23;
    /// Thermochemical calorie in `J` (exact by definition).
    pub const CALORIE_J: f64 = 4.184;

    // CODATA 2018 measured constants.

    /// `1 Bohr` in Angstroms (CODATA 2018 Bohr radius `0.529177210903` Angstrom).
    pub const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;
    /// `1 Hartree` in Joules (CODATA 2018 Hartree energy).
    pub const HARTREE_TO_JOULE: f64 = 4.359_744_722_207_1e-18;
    /// Electron mass in `kg` (CODATA 2018).
    pub const ELECTRON_MASS_KG: f64 = 9.109_383_701_5e-31;
    /// Atomic mass constant (`1 amu` / `Dalton`) in `kg` (CODATA 2018).
    pub const AMU_TO_KG: f64 = 1.660_539_066_60e-27;

    // Derived lengths.

    /// `1 Angstrom` in Bohr.
    pub const ANGSTROM_TO_BOHR: f64 = 1.0 / Self::BOHR_TO_ANGSTROM;
    /// `1 Bohr` in meters.
    pub const BOHR_TO_METER: f64 = Self::BOHR_TO_ANGSTROM * 1.0e-10;

    // Derived energies.

    /// `1 Hartree` in `kJ/mol` (`E_h * N_A / 1000`).
    pub const HARTREE_TO_KJ_MOL: f64 = Self::HARTREE_TO_JOULE * Self::AVOGADRO / 1000.0;
    /// `1 kJ/mol` in Hartree.
    pub const KJ_MOL_TO_HARTREE: f64 = 1.0 / Self::HARTREE_TO_KJ_MOL;
    /// `1 Hartree` in `kcal/mol` (`E_h * N_A / 4184 J`).
    pub const HARTREE_TO_KCAL_MOL: f64 = Self::HARTREE_TO_KJ_MOL / Self::CALORIE_J;
    /// `1 kcal/mol` in Hartree.
    pub const KCAL_MOL_TO_HARTREE: f64 = 1.0 / Self::HARTREE_TO_KCAL_MOL;
    /// `1 Hartree` in electron-volts (`E_h / e`).
    pub const HARTREE_TO_EV: f64 = Self::HARTREE_TO_JOULE / Self::ELEMENTARY_CHARGE_C;
    /// `1 eV` in Hartree.
    pub const EV_TO_HARTREE: f64 = 1.0 / Self::HARTREE_TO_EV;
    /// `1 Hartree` in wavenumbers (`cm^-1`, `E_h / (h c)`).
    pub const HARTREE_TO_WAVENUMBER: f64 =
        Self::HARTREE_TO_JOULE / (Self::PLANCK_J_S * Self::SPEED_OF_LIGHT_CM_S);
    /// `1 cm^-1` in Hartree.
    pub const WAVENUMBER_TO_HARTREE: f64 = 1.0 / Self::HARTREE_TO_WAVENUMBER;

    // Derived thermal constants.

    /// Boltzmann constant in `Hartree / K`.
    pub const BOLTZMANN_HARTREE_PER_K: f64 = Self::BOLTZMANN_J_PER_K / Self::HARTREE_TO_JOULE;
    /// Boltzmann constant in `kcal / (mol * K)` (`R / 4184 J`).
    pub const BOLTZMANN_KCAL_MOL_K: f64 =
        Self::BOLTZMANN_J_PER_K * Self::AVOGADRO / (1000.0 * Self::CALORIE_J);
    /// Ideal gas constant `R` in `J / (mol * K)` (`N_A k_B`).
    pub const GAS_CONSTANT_J_MOL_K: f64 = Self::BOLTZMANN_J_PER_K * Self::AVOGADRO;
    /// Second radiation constant `h c / k_B` in `cm * K` (wavenumber to temperature).
    pub const HC_OVER_KB_CM_K: f64 =
        Self::PLANCK_J_S * Self::SPEED_OF_LIGHT_CM_S / Self::BOLTZMANN_J_PER_K;

    // Derived masses and times.

    /// `1 amu` in electron mass atomic units (`m_e`).
    pub const AMU_TO_AU_MASS: f64 = Self::AMU_TO_KG / Self::ELECTRON_MASS_KG;
    /// Planck constant in `Hartree * s` (`h / E_h`).
    pub const PLANCK_HARTREE_S: f64 = Self::PLANCK_J_S / Self::HARTREE_TO_JOULE;
    /// Reduced Planck constant in `Hartree * s` (`hbar / E_h`, equal to the atomic unit of time).
    pub const HBAR_HARTREE_S: f64 = Self::PLANCK_HARTREE_S / (2.0 * PI);
    /// Atomic unit of time in seconds (`hbar / E_h`).
    pub const AU_TIME_TO_SEC: f64 = Self::HBAR_HARTREE_S;
    /// Atomic unit of time in femtoseconds (`fs`).
    pub const AU_TIME_TO_FS: f64 = Self::AU_TIME_TO_SEC * 1.0e15;
    /// `1 fs` in atomic time units.
    pub const FS_TO_AU_TIME: f64 = 1.0 / Self::AU_TIME_TO_FS;

    // Derived forces, pressures and curvatures.

    /// Conversion from piconewton (`pN`) to atomic force (`Hartree / Bohr`).
    pub const PN_TO_AU_FORCE: f64 = 1.0e-12 * Self::BOHR_TO_METER / Self::HARTREE_TO_JOULE;
    /// `1 Hartree / Bohr^3` (atomic unit of pressure) in GPa.
    pub const GPA_PER_AU_PRESSURE: f64 = Self::HARTREE_TO_JOULE
        / (Self::BOHR_TO_METER * Self::BOHR_TO_METER * Self::BOHR_TO_METER)
        / 1.0e9;
    /// `1 eV / Angstrom^3` in GPa (`e * 1e30 / 1e9`).
    pub const GPA_PER_EV_ANGSTROM3: f64 = Self::ELEMENTARY_CHARGE_C * 1.0e21;
    /// `1 Hartree / Bohr^2` in `mdyne / Angstrom` (`E_h / a_0^2` in `N/m`, `1 mdyne/A = 100 N/m`).
    pub const HARTREE_BOHR2_TO_MDYNE_ANGSTROM: f64 =
        Self::HARTREE_TO_JOULE / (Self::BOHR_TO_METER * Self::BOHR_TO_METER) / 100.0;
    /// `1 Hartree / (Bohr^2 * amu)` in `1 / s^2` (mass-weighted curvature to SI).
    pub const MW_CURVATURE_TO_SI: f64 =
        Self::HARTREE_TO_JOULE / (Self::BOHR_TO_METER * Self::BOHR_TO_METER) / Self::AMU_TO_KG;

    /// Conversion factor from mass-weighted Hessian eigenvalue (`Hartree / (Bohr^2 * amu)`)
    /// to harmonic vibrational wavenumber (`cm^-1`): `sqrt(MW_CURVATURE_TO_SI) / (2 pi c)`.
    /// A literal because `sqrt` is not a `const fn`; the unit test checks the derivation.
    pub const EIGVAL_MW_TO_CM1: f64 = 5_140.487_143_715_827;

    /// Degrees to radians.
    pub const DEG_TO_RAD: f64 = PI / 180.0;
    /// Radians to degrees.
    pub const RAD_TO_DEG: f64 = 180.0 / PI;
}

/// Signed wavenumber (cm^-1) of an eigenvalue of the mass-weighted Hessian (Hartree / (Bohr^2
/// amu)): `sqrt(|lambda|)` times [`Units::EIGVAL_MW_TO_CM1`], negative for a negative eigenvalue
/// (an imaginary mode).
///
/// MultiOptPy: the conversion of `ModeKill` in `IRC/modekill.py` (one place here for the IRC and
/// the optimizer's mode kill).
pub fn eigenvalue_to_wavenumber(eigenvalue: f64) -> f64 {
    let size = libm::sqrt(eigenvalue.abs()) * Units::EIGVAL_MW_TO_CM1;
    if eigenvalue < 0.0 {
        -size
    } else {
        size
    }
}

#[cfg(test)]
mod tests {
    use super::Units;

    fn close(value: f64, reference: f64) -> bool {
        ((value - reference) / reference).abs() < 1e-12
    }

    /// Every derived factor against an independent evaluation (CODATA 2018 recommended values
    /// as printed, or a direct formula), to 1e-12 relative.
    #[test]
    fn derived_constants_match_codata_2018() {
        let checks: [(&str, f64, f64); 18] = [
            (
                "Bohr in Angstrom",
                Units::BOHR_TO_ANGSTROM,
                0.529_177_210_903,
            ),
            (
                "Angstrom in Bohr",
                Units::ANGSTROM_TO_BOHR,
                1.889_726_124_625_77,
            ),
            ("Hartree in eV", Units::HARTREE_TO_EV, 27.211_386_245_988),
            (
                "Hartree in kJ/mol",
                Units::HARTREE_TO_KJ_MOL,
                2_625.499_639_479_825,
            ),
            (
                "Hartree in kcal/mol",
                Units::HARTREE_TO_KCAL_MOL,
                627.509_474_063_056,
            ),
            (
                "Hartree in cm^-1",
                Units::HARTREE_TO_WAVENUMBER,
                219_474.631_363_196,
            ),
            (
                "k_B in Hartree/K",
                Units::BOLTZMANN_HARTREE_PER_K,
                3.166_811_563_455_608e-6,
            ),
            (
                "k_B in kcal/(mol K)",
                Units::BOLTZMANN_KCAL_MOL_K,
                1.987_204_258_640_832e-3,
            ),
            (
                "R in J/(mol K)",
                Units::GAS_CONSTANT_J_MOL_K,
                8.314_462_618_153_24,
            ),
            (
                "hc/k_B in cm K",
                Units::HC_OVER_KB_CM_K,
                1.438_776_877_503_934,
            ),
            ("amu in m_e", Units::AMU_TO_AU_MASS, 1_822.888_486_217_313),
            (
                "h/E_h in s",
                Units::PLANCK_HARTREE_S,
                1.519_829_846_057_038e-16,
            ),
            (
                "a.u. time in s",
                Units::AU_TIME_TO_SEC,
                2.418_884_326_585_719e-17,
            ),
            (
                "a.u. time in fs",
                Units::AU_TIME_TO_FS,
                2.418_884_326_585_719e-2,
            ),
            (
                "a.u. pressure in GPa",
                Units::GPA_PER_AU_PRESSURE,
                29_421.015_696_522,
            ),
            ("eV/A^3 in GPa", Units::GPA_PER_EV_ANGSTROM3, 160.217_663_4),
            (
                "Eh/a0^2 in mdyne/A",
                Units::HARTREE_BOHR2_TO_MDYNE_ANGSTROM,
                15.568_931_028_218_95,
            ),
            (
                "eigenvalue to cm^-1",
                Units::EIGVAL_MW_TO_CM1,
                5_140.487_143_715_827,
            ),
        ];
        for (name, value, reference) in checks {
            assert!(close(value, reference), "{name}: {value} vs {reference}");
        }
    }

    /// The square-root literal against its derivation from the base constants.
    #[test]
    fn eigenvalue_factor_matches_its_derivation() {
        let derived = Units::MW_CURVATURE_TO_SI.sqrt()
            / (2.0 * std::f64::consts::PI * Units::SPEED_OF_LIGHT_CM_S);
        assert!(close(Units::EIGVAL_MW_TO_CM1, derived), "{derived}");
    }

    /// Reciprocal pairs are consistent.
    #[test]
    fn reciprocals_are_consistent() {
        let pairs = [
            (Units::BOHR_TO_ANGSTROM, Units::ANGSTROM_TO_BOHR),
            (Units::HARTREE_TO_EV, Units::EV_TO_HARTREE),
            (Units::HARTREE_TO_KCAL_MOL, Units::KCAL_MOL_TO_HARTREE),
            (Units::HARTREE_TO_KJ_MOL, Units::KJ_MOL_TO_HARTREE),
            (Units::HARTREE_TO_WAVENUMBER, Units::WAVENUMBER_TO_HARTREE),
            (Units::AU_TIME_TO_FS, Units::FS_TO_AU_TIME),
        ];
        for (a, b) in pairs {
            assert!((a * b - 1.0).abs() < 1e-15, "{a} * {b}");
        }
    }
}
