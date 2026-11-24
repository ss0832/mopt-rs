//! Van der Waals radii of the elements.
//!
//! S. Alvarez, "A cartography of the van der Waals territories", Dalton Trans. 2013, 42,
//! 8617-8636 (DOI: 10.1039/C3DT50599E); the values as tabulated in ASE
//! (`ase/data/vdw_alvarez.py`), `Z = 1..=99` with gaps where the publication has none.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy has the UFF distances).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, Units};

/// No radius in the publication.
const NONE: f64 = f64::NAN;

/// Van der Waals radii in Angstrom for `Z = 0..=99`.
const ALVAREZ_VDW_RADII_ANG: [f64; 100] = [
    NONE, // 0: X
    1.20, 1.43, // H, He
    2.12, 1.98, 1.91, 1.77, 1.66, 1.50, 1.46, 1.58, // Li..Ne
    2.50, 2.51, 2.25, 2.19, 1.90, 1.89, 1.82, 1.83, // Na..Ar
    2.73, 2.62, 2.58, 2.46, 2.42, 2.45, 2.45, 2.44, 2.40, 2.40, 2.38, 2.39, 2.32, 2.29, 1.88, 1.82,
    1.86, 2.25, // K..Kr
    3.21, 2.84, 2.75, 2.52, 2.56, 2.45, 2.44, 2.46, 2.44, 2.15, 2.53, 2.49, 2.43, 2.42, 2.47, 1.99,
    2.04, 2.06, // Rb..Xe
    3.48, 3.03, 2.98, 2.88, 2.92, 2.95, NONE, 2.90, 2.87, 2.83, 2.79, 2.87, 2.81, 2.83, 2.79, 2.80,
    2.74, 2.63, 2.53, 2.57, 2.49, 2.48, 2.41, 2.29, 2.32, 2.45, 2.47, 2.60, 2.54, NONE, NONE,
    NONE, // Cs..Rn
    NONE, NONE, 2.80, 2.93, 2.88, 2.71, 2.82, 2.81, 2.83, 3.05, 3.40, 3.05, 2.70, // Fr..Es
];

/// The van der Waals radius of Alvarez in Bohr (`None` where the publication has none).
pub fn alvarez_vdw_radius_bohr(el: Element) -> Option<f64> {
    ALVAREZ_VDW_RADII_ANG
        .get(el.atomic_number() as usize)
        .copied()
        .filter(|radius| radius.is_finite())
        .map(|radius| radius * Units::ANGSTROM_TO_BOHR)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_radii_of_some_elements() {
        let radius = |symbol: &str| {
            alvarez_vdw_radius_bohr(Element::parse_token(symbol).unwrap())
                .map(|r| r * Units::BOHR_TO_ANGSTROM)
        };
        for (symbol, expected) in [
            ("H", 1.20),
            ("C", 1.77),
            ("N", 1.66),
            ("O", 1.50),
            ("S", 1.89),
            ("Br", 1.86),
            ("Kr", 2.25),
            ("Xe", 2.06),
            ("Rn", f64::NAN),
            ("Bi", 2.54),
            ("Es", 2.70),
        ] {
            match radius(symbol) {
                Some(found) => assert!((found - expected).abs() < 1e-12, "{symbol}"),
                None => assert!(expected.is_nan(), "{symbol}"),
            }
        }
        assert!(radius("Pm").is_none() && radius("Og").is_none());
    }
}
