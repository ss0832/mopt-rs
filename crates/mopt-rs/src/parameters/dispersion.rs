//! Grimme `DFT-D2`, `DFT-D3(BJ)`, and `DFT-D4` dispersion model parameter sets.
//!
//! References:
//! - DFT-D2: S. Grimme, J. Comput. Chem. 2006, 27, 1787-1799 (DOI: 10.1002/jcc.20495)
//! - DFT-D3(BJ): S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys. 2010, 132, 154104;
//!   S. Grimme, S. Ehrlich, L. Goerigk, J. Comput. Chem. 2011, 32, 1456-1465 (`tad-dftd3` `r4/r2`)
//! - DFT-D4: E. Caldeweyher et al., J. Chem. Phys. 2019, 150, 154122 (`dftd4`)
//!
//! MultiOptPy: `Parameters/d2.py`, `Parameters/d3.py`, `Parameters/d4.py` (published values;
//!  MultiOptPy's in `multioptpy_tables`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, Units};

#[inline]
fn clamp_z86(el: Element) -> usize {
    let z = el.atomic_number() as usize;
    if z <= 86 {
        z
    } else {
        86
    }
}

/// DFT-D2 van der Waals radii `R_0` in Angstroms (`Z = 0..=86`).
const D2_VDW_ANG: [f64; 87] = [
    1.500, // X
    1.001, 1.012, // H, He
    0.825, 1.408, 1.485, 1.452, 1.397, 1.342, 1.287, 1.243, // Li..Ne
    1.144, 1.364, 1.639, 1.716, 1.705, 1.683, 1.639, 1.595, // Na..Ar
    1.485, 1.474, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.650,
    1.727, 1.760, 1.771, 1.749, 1.727, // K..Kr
    1.628, 1.606, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.672,
    1.804, 1.881, 1.892, 1.892, 1.881, // Rb..Xe
    1.802, 1.762, 1.720, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753,
    1.753, 1.753, 1.753, 1.753, 1.788, 1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.758,
    1.989, 1.944, 1.898, 2.005, 1.991, 1.924, // Cs..Rn
];

/// DFT-D2 atomic `C_6` coefficients in `J * nm^6 / mol` (`Z = 0..=86`).
const D2_C6_J_NM6_MOL: [f64; 87] = [
    0.00, // X
    0.14, 0.08, // H, He
    1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63, // Li..Ne
    5.71, 5.71, 10.79, 9.23, 7.84, 5.57, 5.07, 4.61, // Na..Ar
    10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 16.99,
    17.10, 16.37, 12.64, 12.47, 12.01, // K..Kr
    24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 37.32,
    38.71, 38.44, 31.74, 31.50, 29.99, // Rb..Xe
    50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00,
    50.00, 50.00, 50.00, 50.00, 50.00, 50.00, // Cs..Rn
];

/// PBE0/def2-QZVP `<r^4> / <r^2>` multipole expectation ratios (`Z = 0..=86`).
const D3_R4R2: [f64; 87] = [
    10.0000, // X
    8.0589, 3.4698, // H, He
    29.0974, 14.8517, 11.8799, 7.8715, 5.5588, 4.7566, 3.8025, 3.1036, // Li..Ne
    26.1552, 17.2304, 17.7210, 12.7442, 9.5361, 8.1652, 6.7463, 5.6004, // Na..Ar
    29.2012, 22.3934, 19.0598, 16.8590, 15.4023, 12.5589, 13.4788, 12.2309, 11.2809, 10.5569,
    10.1428, 9.4907, 13.4606, 10.8544, 8.9386, 8.1350, 7.1251, 6.1971, // K..Kr
    30.0162, 24.4103, 20.3537, 17.4780, 13.5528, 11.8451, 11.0355, 10.1997, 9.5414, 9.0061, 8.6417,
    8.9975, 14.0834, 11.8333, 10.0179, 9.3844, 8.4110, 7.5152, // Rb..Xe
    32.7622, 27.5708, 23.1671, 21.6003, 20.9615, 20.4562, 20.1010, 19.7475, 19.4828, 15.6013,
    19.2362, 17.4717, 17.8321, 17.4237, 17.1954, 17.1631, 14.5716, 15.8758, 13.8989, 12.4834,
    11.4421, 10.2671, 8.3549, 7.8496, 7.3278, 7.4820, 13.5124, 11.6554, 10.0959, 9.7340, 8.8584,
    8.0125, // Cs..Rn
];

/// DFT-D4 static atomic dipole polarizabilities in `Bohr^3` (`Z = 0..=54`, default `20.0` above Xe).
const D4_REF_ALPHA: [f64; 55] = [
    20.0, // X
    4.50, 1.38, // H, He
    164.20, 38.40, 21.10, 12.00, 7.40, 5.40, 3.80, 2.67, // Li..Ne
    162.70, 71.00, 57.80, 37.00, 25.00, 19.60, 15.00, 11.10, // Na..Ar
    292.80, 160.80, 120.00, 98.00, 84.00, 72.00, 63.00, 56.00, 50.00, 44.00, 42.00, 40.00, 60.00,
    41.00, 29.00, 25.00, 20.00, 16.80, // K..Kr
    320.20, 199.30, 126.70, 119.97, 101.60, 88.42, 80.08, 65.89, 56.10, 23.68, 46.00, 39.72, 70.22,
    55.95, 43.67, 37.65, 35.00, 27.30, // Rb..Xe
];

/// Pauling electronegativities for DFT-D4 EEQ / charge-scaling (`Z = 0..=56`, default `2.0` above Ba).
const D4_PAULING_EN: [f64; 57] = [
    2.00, // X
    2.20, 0.00, // H, He
    0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.00, // Li..Ne
    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0.00, // Na..Ar
    0.82, 1.00, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55,
    2.96, 0.00, // K..Kr
    0.82, 0.95, 1.22, 1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96, 2.05, 2.10,
    2.66, 0.00, // Rb..Xe
    0.79, 0.89, // Cs, Ba
];

/// DFT-D2 van der Waals radius `R_0` in Bohr.
#[inline]
pub fn d2_vdw_radius_bohr(el: Element) -> f64 {
    D2_VDW_ANG[clamp_z86(el)] * Units::ANGSTROM_TO_BOHR
}

/// DFT-D2 `C_6` coefficient in `Hartree * Bohr^6`.
#[inline]
pub fn d2_c6_au(el: Element) -> f64 {
    let b2a = Units::BOHR_TO_ANGSTROM;
    let b2a6 = b2a * b2a * b2a * b2a * b2a * b2a;
    // 1 J * nm^6 / mol = 10^6 J * A^6 / mol
    D2_C6_J_NM6_MOL[clamp_z86(el)] * 1.0e6 / (b2a6 * Units::HARTREE_TO_JOULE * Units::AVOGADRO)
}

/// Global damping parameters for Grimme `DFT-D2`.
#[derive(Debug, Clone, Copy)]
pub struct D2Params {
    /// Global scaling `s6`.
    pub s6: f64,
    /// Damping steepness `d`.
    pub d_damping: f64,
}

impl Default for D2Params {
    fn default() -> Self {
        Self {
            s6: 1.00,
            d_damping: 20.0,
        }
    }
}

/// Global Becke-Johnson damping parameters and `<r^4>/<r^2>` accessor for Grimme `DFT-D3(BJ)`.
#[derive(Debug, Clone, Copy)]
pub struct D3Params {
    /// Two-body `C6` scaling `s6`.
    pub s6: f64,
    /// Two-body `C8` scaling `s8`.
    pub s8: f64,
    /// Becke-Johnson damping `a1`.
    pub a1: f64,
    /// Becke-Johnson damping `a2` (Bohr).
    pub a2: f64,
}

impl Default for D3Params {
    fn default() -> Self {
        // PBE0 defaults matching `multioptpy/Parameters/d3.py`
        Self {
            s6: 1.0,
            s8: 0.7875,
            a1: 0.4289,
            a2: 4.4407,
        }
    }
}

impl D3Params {
    /// Return `<r^4> / <r^2>` multipole expectation value ratio for `el`.
    #[inline]
    pub fn r4r2(&self, el: Element) -> f64 {
        let z = el.atomic_number() as usize;
        if z > 0 && z < D3_R4R2.len() {
            D3_R4R2[z]
        } else {
            10.0
        }
    }

    /// Estimate homoatomic/heteroatomic `C_6(A, B)` in `Hartree * Bohr^6` scaled by coordination numbers.
    #[inline]
    pub fn pair_c6_au(&self, a: Element, b: Element, cn_a: f64, cn_b: f64) -> f64 {
        let c6_a = d2_c6_au(a);
        let c6_b = d2_c6_au(b);
        let base = libm::sqrt((c6_a * c6_b).max(0.0));
        // Smooth CN modulation consistent with D3 coordination-dependent polarizability damping
        let mod_cn = 1.0 / (1.0 + 0.03 * (cn_a + cn_b));
        base * mod_cn
    }

    /// Compute `C_8(A, B) = 3 * C_6(A, B) * sqrt(Q_A * Q_B)` in `Hartree * Bohr^8`.
    #[inline]
    pub fn pair_c8_au(&self, a: Element, b: Element, c6_ab: f64) -> f64 {
        let qa = 0.5 * libm::sqrt(a.atomic_number().max(1) as f64) * self.r4r2(a);
        let qb = 0.5 * libm::sqrt(b.atomic_number().max(1) as f64) * self.r4r2(b);
        3.0 * c6_ab * libm::sqrt(qa * qb)
    }
}

/// Global Becke-Johnson + charge-scaling parameters for Grimme `DFT-D4`.
#[derive(Debug, Clone, Copy)]
pub struct D4Params {
    /// Two-body `C6` scaling `s6`.
    pub s6: f64,
    /// Two-body `C8` scaling `s8`.
    pub s8: f64,
    /// Three-body (ATM) scaling `s9`.
    pub s9: f64,
    /// Becke-Johnson damping `a1`.
    pub a1: f64,
    /// Becke-Johnson damping `a2` (Bohr).
    pub a2: f64,
    /// Charge scaling height `ga`.
    pub ga: f64,
    /// Charge scaling steepness `gc`.
    pub gc: f64,
}

impl Default for D4Params {
    fn default() -> Self {
        // PBE0/def2-QZVP defaults matching `multioptpy/Parameters/d4.py`
        Self {
            s6: 1.0,
            s8: 1.03683,
            s9: 1.0,
            a1: 0.4171,
            a2: 4.5337,
            ga: 3.0,
            gc: 2.0,
        }
    }
}

impl D4Params {
    /// `sqrt(0.5 * <r^4>/<r^2> * sqrt(Z))` expectation-value ratio for `el`.
    #[inline]
    pub fn r4r2(&self, el: Element) -> f64 {
        let z = el.atomic_number() as usize;
        if z > 0 && z <= 56 {
            D3_R4R2[z]
        } else {
            10.0
        }
    }

    /// Static reference polarizability of `el` (atomic units).
    #[inline]
    pub fn polarizability(&self, el: Element) -> f64 {
        let z = el.atomic_number() as usize;
        if z > 0 && z < D4_REF_ALPHA.len() {
            D4_REF_ALPHA[z]
        } else {
            20.0
        }
    }

    /// Pauling electronegativity of `el`.
    #[inline]
    pub fn electronegativity(&self, el: Element) -> f64 {
        let z = el.atomic_number() as usize;
        if z > 0 && z < D4_PAULING_EN.len() {
            D4_PAULING_EN[z]
        } else {
            2.0
        }
    }

    /// Casimir-Polder effective `C_6(A, B)` in `Hartree * Bohr^6` from scaled static polarizabilities.
    #[inline]
    pub fn pair_c6_au(
        &self,
        a: Element,
        b: Element,
        q_a: f64,
        q_b: f64,
        cn_a: f64,
        cn_b: f64,
    ) -> f64 {
        let zeta = |q: f64| {
            libm::exp(self.ga * (1.0 - libm::exp(self.gc * (1.0 - (1.0 - q * 0.1).max(0.1)))))
                .clamp(0.2, 3.0)
        };
        let alpha_a = self.polarizability(a) * zeta(q_a) / (1.0 + 0.025 * cn_a);
        let alpha_b = self.polarizability(b) * zeta(q_b) / (1.0 + 0.025 * cn_b);
        // London-Slater-Kirkwood effective C6 proportional to alpha_A * alpha_B
        let denom = (alpha_a / (a.atomic_number().max(1) as f64).sqrt())
            + (alpha_b / (b.atomic_number().max(1) as f64).sqrt());
        if denom > 1e-12 {
            0.75 * alpha_a * alpha_b / denom
        } else {
            0.0
        }
    }

    /// Pair `C8` coefficient derived from `C6` and the `<r^4>/<r^2>` ratios.
    #[inline]
    pub fn pair_c8_au(&self, a: Element, b: Element, c6_ab: f64) -> f64 {
        let qa = 0.5 * libm::sqrt(a.atomic_number().max(1) as f64) * self.r4r2(a);
        let qb = 0.5 * libm::sqrt(b.atomic_number().max(1) as f64) * self.r4r2(b);
        3.0 * c6_ab * libm::sqrt(qa * qb)
    }
}
