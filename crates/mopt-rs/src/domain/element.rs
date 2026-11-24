//! Chemical element representation (`Z = 0..=118`, where `0` is dummy atom `X`) and atomic masses.
//!
//! MultiOptPy: `Parameters/atomic_number.py` (`number_element`, `element_number`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt;
use std::str::FromStr;

use crate::error::{MoptError, MoptResult};

/// Type-safe chemical element (`Z = 1..=118`, or `Z = 0` for dummy atom `X`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Element(u8);

impl Element {
    /// Dummy atom (`X`, atomic number `0`).
    pub const DUMMY: Self = Self(0);
    /// Hydrogen.
    pub const H: Self = Self(1);
    /// Carbon.
    pub const C: Self = Self(6);
    /// Nitrogen.
    pub const N: Self = Self(7);
    /// Oxygen.
    pub const O: Self = Self(8);

    /// Construct an `Element` from an atomic number `0..=118`.
    pub fn from_atomic_number(z: u8) -> MoptResult<Self> {
        if (z as usize) < ELEMENT_SYMBOLS.len() {
            Ok(Self(z))
        } else {
            Err(MoptError::parse(format!(
                "Atomic number {z} out of supported range 0..=118"
            )))
        }
    }

    /// Parse an element from either an atomic symbol (`"C"`, `"fe"`, `"X"`) or integer string (`"6"`).
    pub fn parse_token(token: &str) -> MoptResult<Self> {
        let trimmed = token.trim();
        if let Ok(z) = trimmed.parse::<u8>() {
            return Self::from_atomic_number(z);
        }
        // Strip trailing atom tags like "C1" or "Fe_a" only if pure alphabetic prefix is unavailable
        if let Some(el) = lookup_symbol(trimmed) {
            return Ok(el);
        }
        let alpha_prefix: String = trimmed
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        if !alpha_prefix.is_empty() {
            if let Some(el) = lookup_symbol(&alpha_prefix) {
                return Ok(el);
            }
        }
        Err(MoptError::parse(format!(
            "Unrecognized chemical element token: '{token}'"
        )))
    }

    /// Atomic number `Z` (`0` for dummy `X`).
    #[inline]
    pub const fn atomic_number(self) -> u8 {
        self.0
    }

    /// Canonical IUPAC symbol (`"H"`, `"He"`, ..., `"Og"`, or `"X"` for `Z = 0`).
    #[inline]
    pub const fn symbol(self) -> &'static str {
        ELEMENT_SYMBOLS[self.0 as usize]
    }

    /// Standard atomic mass in `amu` (Daltons), matching `multioptpy/Parameters/atomic_mass.py`.
    #[inline]
    pub const fn mass_amu(self) -> f64 {
        ATOMIC_MASSES_AMU[self.0 as usize]
    }

    /// Return `true` if this is the dummy atom `X` (`Z == 0`).
    #[inline]
    pub const fn is_dummy(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

impl FromStr for Element {
    type Err = MoptError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_token(s)
    }
}

fn lookup_symbol(sym: &str) -> Option<Element> {
    for (z, &candidate) in ELEMENT_SYMBOLS.iter().enumerate() {
        if sym.eq_ignore_ascii_case(candidate) {
            return Some(Element(z as u8));
        }
    }
    if sym.eq_ignore_ascii_case("Bq") {
        return Some(Element::DUMMY);
    }
    None
}

const ELEMENT_SYMBOLS: [&str; 119] = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

const ATOMIC_MASSES_AMU: [f64; 119] = [
    1.0000,   // 0: X (dummy)
    1.00794,  // 1: H
    4.002602, // 2: He
    6.941,    // 3: Li
    9.012182, // 4: Be
    10.811,   // 5: B
    12.0107,  // 6: C
    14.0067,  // 7: N
    15.9994,  // 8: O
    18.99840, // 9: F
    20.1797,  // 10: Ne
    22.98977, // 11: Na
    24.3050,  // 12: Mg
    26.98154, // 13: Al
    28.0855,  // 14: Si
    30.97376, // 15: P
    32.065,   // 16: S
    35.453,   // 17: Cl
    39.948,   // 18: Ar
    39.0983,  // 19: K
    40.078,   // 20: Ca
    44.95591, // 21: Sc
    47.867,   // 22: Ti
    50.9415,  // 23: V
    51.9961,  // 24: Cr
    54.93805, // 25: Mn
    55.845,   // 26: Fe
    58.93320, // 27: Co
    58.6934,  // 28: Ni
    63.546,   // 29: Cu
    65.38,    // 30: Zn
    69.723,   // 31: Ga
    72.64,    // 32: Ge
    74.92160, // 33: As
    78.96,    // 34: Se
    79.904,   // 35: Br
    83.798,   // 36: Kr
    85.4678,  // 37: Rb
    87.62,    // 38: Sr
    88.90585, // 39: Y
    91.224,   // 40: Zr
    92.90638, // 41: Nb
    95.96,    // 42: Mo
    98.0,     // 43: Tc
    101.07,   // 44: Ru
    102.9055, // 45: Rh
    106.42,   // 46: Pd
    107.8682, // 47: Ag
    112.411,  // 48: Cd
    114.818,  // 49: In
    118.710,  // 50: Sn
    121.760,  // 51: Sb
    127.60,   // 52: Te
    126.9045, // 53: I
    131.293,  // 54: Xe
    132.9055, // 55: Cs
    137.327,  // 56: Ba
    138.9055, // 57: La
    140.116,  // 58: Ce
    140.9077, // 59: Pr
    144.242,  // 60: Nd
    145.0,    // 61: Pm
    150.36,   // 62: Sm
    151.964,  // 63: Eu
    157.25,   // 64: Gd
    158.9254, // 65: Tb
    162.500,  // 66: Dy
    164.9303, // 67: Ho
    167.259,  // 68: Er
    168.9342, // 69: Tm
    173.054,  // 70: Yb
    174.9668, // 71: Lu
    178.49,   // 72: Hf
    180.9479, // 73: Ta
    183.84,   // 74: W
    186.207,  // 75: Re
    190.23,   // 76: Os
    192.217,  // 77: Ir
    195.084,  // 78: Pt
    196.9666, // 79: Au
    200.59,   // 80: Hg
    204.3833, // 81: Tl
    207.2,    // 82: Pb
    208.9804, // 83: Bi
    209.0,    // 84: Po
    210.0,    // 85: At
    222.0,    // 86: Rn
    223.0,    // 87: Fr
    226.0,    // 88: Ra
    227.0,    // 89: Ac
    232.0381, // 90: Th
    231.0359, // 91: Pa
    238.0289, // 92: U
    237.0,    // 93: Np
    244.0,    // 94: Pu
    243.0,    // 95: Am
    247.0,    // 96: Cm
    247.0,    // 97: Bk
    251.0,    // 98: Cf
    252.0,    // 99: Es
    257.0,    // 100: Fm
    258.0,    // 101: Md
    259.0,    // 102: No
    262.0,    // 103: Lr
    267.0,    // 104: Rf
    268.0,    // 105: Db
    271.0,    // 106: Sg
    272.0,    // 107: Bh
    270.0,    // 108: Hs
    276.0,    // 109: Mt
    281.0,    // 110: Ds
    280.0,    // 111: Rg
    285.0,    // 112: Cn
    284.0,    // 113: Nh
    289.0,    // 114: Fl
    288.0,    // 115: Mc
    293.0,    // 116: Lv
    294.0,    // 117: Ts
    294.0,    // 118: Og
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_parsing_roundtrip() {
        assert_eq!(Element::parse_token("C").unwrap(), Element::C);
        assert_eq!(Element::parse_token("6").unwrap(), Element::C);
        assert_eq!(Element::parse_token("fe").unwrap().symbol(), "Fe");
        assert_eq!(Element::parse_token("Fe2").unwrap().symbol(), "Fe");
        assert!((Element::C.mass_amu() - 12.0107).abs() < 1e-6);
    }
}
