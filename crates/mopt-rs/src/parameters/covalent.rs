//! Single-, double-, and triple-bond molecular covalent radii (Pyykko & Atsumi, H..=Og) in Bohr.
//!
//! References:
//! - P. Pyykko, M. Atsumi, Chem. Eur. J. 2009, 15, 186-197 (DOI: 10.1002/chem.200800987)
//! - P. Pyykko, M. Atsumi, Chem. Eur. J. 2009, 15, 12770-12779 (DOI: 10.1002/chem.200901472)
//! - P. Pyykko, S. Riedel, M. Patzschke, Chem. Eur. J. 2005, 11, 3511-3520 (DOI: 10.1002/chem.200401299)
//!
//! MultiOptPy: `Parameters/covalent_radii.py` (`covalent_radii_lib`, `double_covalent_radii_lib`,
//!  `triple_covalent_radii_lib`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, Units};

/// Single-bond covalent radius in Angstroms for `Z = 0..=118` (`Z = 0` dummy `X` -> `1.00 A`).
/// Elements `Z = 1..=86` match `multioptpy/Parameters/covalent_radii.py` verbatim;
/// `Z = 87..=118` follow Pyykko & Atsumi (2009) Table 2.
const SINGLE_COV_RADII_ANG: [f64; 119] = [
    1.00, // 0: X
    0.32, 0.46, // H, He
    1.33, 1.02, 0.85, 0.75, 0.71, 0.63, 0.64, 0.67, // Li..Ne
    1.55, 1.39, 1.26, 1.16, 1.11, 1.03, 0.99, 0.96, // Na..Ar
    1.96, 1.71, 1.48, 1.36, 1.34, 1.22, 1.19, 1.16, 1.11, 1.10, 1.12, 1.18, 1.24, 1.24, 1.21, 1.16,
    1.14, 1.17, // K..Kr
    2.10, 1.85, 1.63, 1.54, 1.47, 1.38, 1.28, 1.25, 1.25, 1.20, 1.28, 1.36, 1.42, 1.40, 1.40, 1.36,
    1.33, 1.31, // Rb..Xe
    2.32, 1.96, 1.80, 1.63, 1.76, 1.74, 1.73, 1.72, 1.68, 1.69, 1.68, 1.67, 1.66, 1.65, 1.64, 1.70,
    1.62, 1.52, 1.46, 1.37, 1.31, 1.29, 1.22, 1.23, 1.24, 1.33, 1.44, 1.44, 1.51, 1.45, 1.47,
    1.42, // Cs..Rn
    2.23, 2.01, 1.86, 1.75, 1.69, 1.70, 1.71, 1.72, 1.66, 1.66, 1.68, 1.68, 1.65, 1.67, 1.73, 1.76,
    1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65,
    1.57, // Fr..Og
];

/// Double-bond covalent radius in Angstroms for `Z = 0..=118`.
const DOUBLE_COV_RADII_ANG: [f64; 119] = [
    1.00, // X
    0.32, 0.46, // H, He
    1.24, 0.90, 0.78, 0.67, 0.60, 0.57, 0.59, 0.96, // Li..Ne
    1.60, 1.32, 1.13, 1.07, 1.02, 0.94, 0.95, 1.07, // Na..Ar
    1.93, 1.47, 1.16, 1.17, 1.12, 1.11, 1.05, 1.09, 1.03, 1.01, 1.15, 1.20, 1.17, 1.11, 1.14, 1.07,
    1.09, 1.21, // K..Kr
    2.02, 1.57, 1.30, 1.27, 1.25, 1.21, 1.20, 1.14, 1.10, 1.17, 1.39, 1.44, 1.36, 1.30, 1.33, 1.28,
    1.29, 1.35, // Rb..Xe
    2.09, 1.61, 1.39, 1.37, 1.38, 1.37, 1.35, 1.34, 1.34, 1.35, 1.35, 1.33, 1.33, 1.33, 1.31, 1.29,
    1.31, 1.28, 1.26, 1.20, 1.19, 1.16, 1.15, 1.12, 1.21, 1.42, 1.42, 1.35, 1.41, 1.35, 1.38,
    1.45, // Cs..Rn
    2.18, 1.73, 1.53, 1.43, 1.38, 1.34, 1.36, 1.35, 1.35, 1.36, 1.39, 1.40, 1.40, 1.41, 1.39, 1.59,
    1.41, 1.40, 1.36, 1.28, 1.28, 1.25, 1.25, 1.16, 1.16, 1.37, 1.36, 1.43, 1.62, 1.75, 1.65,
    1.57, // Fr..Og
];

/// Triple-bond covalent radius in Angstroms for `Z = 0..=118`.
const TRIPLE_COV_RADII_ANG: [f64; 119] = [
    1.00, // X
    0.32, 0.46, // H, He
    1.24, 0.85, 0.73, 0.60, 0.54, 0.53, 0.53, 0.96, // Li..Ne
    1.60, 1.27, 1.11, 1.02, 0.94, 0.95, 0.93, 0.96, // Na..Ar
    1.93, 1.33, 1.14, 1.08, 1.06, 1.03, 1.03, 1.02, 0.96, 1.01, 1.20, 1.20, 1.21, 1.21, 1.06, 1.07,
    1.10, 1.08, // K..Kr
    2.02, 1.39, 1.24, 1.21, 1.16, 1.13, 1.10, 1.03, 1.06, 1.12, 1.37, 1.44, 1.46, 1.32, 1.27, 1.21,
    1.25, 1.22, // Rb..Xe
    2.09, 1.49, 1.39, 1.31, 1.28, 1.37, 1.35, 1.34, 1.34, 1.32, 1.35, 1.33, 1.33, 1.33, 1.31, 1.29,
    1.31, 1.21, 1.19, 1.15, 1.10, 1.09, 1.07, 1.10, 1.23, 1.42, 1.50, 1.37, 1.35, 1.29, 1.38,
    1.33, // Cs..Rn
    2.18, 1.59, 1.40, 1.36, 1.29, 1.18, 1.16, 1.35, 1.35, 1.36, 1.39, 1.40, 1.40, 1.41, 1.39, 1.59,
    1.41, 1.31, 1.26, 1.21, 1.19, 1.18, 1.13, 1.12, 1.18, 1.30, 1.36, 1.43, 1.62, 1.75, 1.65,
    1.57, // Fr..Og
];

/// Return the Pyykko single-bond covalent radius in Bohr.
#[inline]
pub fn covalent_radius_bohr(el: Element) -> f64 {
    SINGLE_COV_RADII_ANG[el.atomic_number() as usize] * Units::ANGSTROM_TO_BOHR
}

/// Return the Pyykko double-bond covalent radius in Bohr.
#[inline]
pub fn double_covalent_radius_bohr(el: Element) -> f64 {
    DOUBLE_COV_RADII_ANG[el.atomic_number() as usize] * Units::ANGSTROM_TO_BOHR
}

/// Return the Pyykko triple-bond covalent radius in Bohr.
#[inline]
pub fn triple_covalent_radius_bohr(el: Element) -> f64 {
    TRIPLE_COV_RADII_ANG[el.atomic_number() as usize] * Units::ANGSTROM_TO_BOHR
}

/// Sum of single-bond covalent radii `r_cov(a) + r_cov(b)` in Bohr.
#[inline]
pub fn sum_covalent_radii_bohr(a: Element, b: Element) -> f64 {
    covalent_radius_bohr(a) + covalent_radius_bohr(b)
}
