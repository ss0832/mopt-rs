//! Parameters of the modified embedded-atom method: reference structures, the elements of
//! the library, and the reference of a pair of elements.
//!
//! The numbers are the first entry of each element in the parameter library distributed
//! with LAMMPS (`potentials/library.meam`, "meam data from vax files fcc,bcc,dia", citing
//! M. I. Baskes, Phys. Rev. B 1992, 46, 2727); see
//! `third_party/DISPERSION_AND_PARAMETERS.md`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::Element;
use crate::error::{MoptError, MoptResult};

/// Reference structure of an element or of a pair of elements: the crystal whose energy as
/// a function of the nearest-neighbour distance is the universal equation of state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lattice {
    /// Face-centred cubic (12 neighbours).
    Fcc,
    /// Body-centred cubic (8 neighbours).
    Bcc,
    /// Hexagonal close-packed with the ideal axial ratio (12 neighbours).
    Hcp,
    /// Diamond (4 neighbours).
    Diamond,
    /// Rock salt, for a pair of elements (6 neighbours of the other element).
    B1,
    /// Caesium chloride, for a pair of elements (8 neighbours of the other element).
    B2,
}

impl Lattice {
    /// From `fcc`, `bcc`, `hcp`, `dia`, `b1`, `b2`.
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "fcc" => Ok(Self::Fcc),
            "bcc" => Ok(Self::Bcc),
            "hcp" => Ok(Self::Hcp),
            "dia" | "diamond" => Ok(Self::Diamond),
            "b1" => Ok(Self::B1),
            "b2" => Ok(Self::B2),
            other => Err(MoptError::config(format!(
                "meam: unknown reference structure '{other}' (known: fcc, bcc, hcp, dia, b1, b2)"
            ))),
        }
    }

    /// Number of nearest neighbours.
    pub fn neighbours(self) -> f64 {
        match self {
            Self::Fcc | Self::Hcp => 12.0,
            Self::Bcc | Self::B2 => 8.0,
            Self::B1 => 6.0,
            Self::Diamond => 4.0,
        }
    }

    /// Shape factor of the third partial density, `(rho^(3))^2 = s (rho^a(3))^2` for one
    /// atom in the structure. Those of the first and second partial densities vanish for
    /// all of these structures.
    pub fn third_shape(self) -> f64 {
        match self {
            Self::Hcp => 1.0 / 3.0,
            Self::Diamond => 32.0 / 9.0,
            Self::Fcc | Self::Bcc | Self::B1 | Self::B2 => 0.0,
        }
    }

    /// Nearest-neighbour distance in units of the lattice constant given in the library
    /// (`a` of the cubic cells, the in-plane `a` of hcp); `None` for the structures of
    /// pairs, whose distance is given directly.
    fn distance_per_lattice_constant(self) -> Option<f64> {
        match self {
            Self::Fcc => Some(0.5_f64.sqrt()),
            Self::Bcc => Some(0.75_f64.sqrt()),
            Self::Hcp => Some(1.0),
            Self::Diamond => Some(0.1875_f64.sqrt()),
            Self::B1 | Self::B2 => None,
        }
    }
}

/// One element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeamElement {
    /// The element.
    pub element: Element,
    /// Reference structure.
    pub lattice: Lattice,
    /// `alpha = sqrt(9 B Omega / E_c)` of the universal equation of state.
    pub alpha: f64,
    /// Decay `beta^(l)` of the partial atomic densities, `l = 0..3`.
    pub beta: [f64; 4],
    /// Nearest-neighbour distance of the reference structure (Angstrom).
    pub re: f64,
    /// Cohesive energy of the reference structure (eV).
    pub ec: f64,
    /// Scale `A` of the embedding energy.
    pub a: f64,
    /// Weights `t^(1)`, `t^(2)`, `t^(3)` of the partial densities as published (`t^(0) = 1`).
    pub t: [f64; 3],
    /// Density scale `rho_0` (matters between different elements only).
    pub rho0: f64,
    /// Form of `G(Gamma)`: 0 or 4 `sqrt(1 + Gamma)`, 1 `exp(Gamma / 2)`,
    /// 3 `2 / (1 + exp(-Gamma))`.
    pub ibar: i32,
}

/// Reference of a pair of different elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeamPair {
    /// Reference structure (each atom surrounded by atoms of the other element).
    pub lattice: Lattice,
    /// Cohesive energy per atom of the reference structure (eV).
    pub ec: f64,
    /// Nearest-neighbour distance of the reference structure (Angstrom).
    pub re: f64,
    /// `alpha` of its equation of state.
    pub alpha: f64,
}

impl MeamPair {
    /// The reference used when none is given for a pair, that of LAMMPS without a parameter
    /// file: fcc, with the means of the cohesive energies, distances, and `alpha` of the two
    /// elements. It is not fitted to any property of the alloy.
    pub fn mean_of(a: &MeamElement, b: &MeamElement) -> Self {
        Self {
            lattice: Lattice::Fcc,
            ec: 0.5 * (a.ec + b.ec),
            re: 0.5 * (a.re + b.re),
            alpha: 0.5 * (a.alpha + b.alpha),
        }
    }
}

/// Library entries: symbol, reference structure, `[alpha, beta0, beta1, beta2, beta3,
/// lattice constant (Angstrom), E_c (eV), A, t1, t2, t3, rho_0]`, `ibar`.
// The cohesive energy of tin is 3.14 eV, not an approximation of pi.
#[allow(clippy::approx_constant)]
pub(super) const TABLE: [(&str, &str, [f64; 12], i32); 24] = [
    (
        "Cu",
        "fcc",
        [
            5.10570729, 3.634, 2.20, 6.0, 2.20, 3.62, 3.54, 1.07, 3.13803254, 2.49438711,
            2.95269237, 1.0,
        ],
        0,
    ),
    (
        "Ag",
        "fcc",
        [
            5.89222008, 4.456, 2.20, 6.0, 2.20, 4.08, 2.85, 1.06, 5.54097609, 2.45015783,
            1.28843988, 1.0,
        ],
        0,
    ),
    (
        "Au",
        "fcc",
        [
            6.34090112, 5.449, 2.20, 6.0, 2.20, 4.07, 3.93, 1.04, 1.58956328, 1.50776392,
            2.60609758, 1.0,
        ],
        0,
    ),
    (
        "Ni",
        "fcc",
        [
            4.99, 2.64, 1.50, 4.50, 1.50, 3.52, 4.45, 1.10, 1.692, 4.987, 3.683, 1.0,
        ],
        1,
    ),
    (
        "Pd",
        "fcc",
        [
            6.43230473, 4.975, 2.20, 6.0, 2.20, 3.89, 3.91, 1.01, 2.33573516, 1.38343023,
            4.47989049, 1.0,
        ],
        0,
    ),
    (
        "Pt",
        "fcc",
        [
            6.44221724, 4.673, 2.20, 6.0, 2.20, 3.92, 5.77, 1.04, 2.73335406, -1.3759593,
            3.29322278, 1.0,
        ],
        0,
    ),
    (
        "Al",
        "fcc",
        [
            4.61, 2.21, 2.20, 6.0, 2.20, 4.05, 3.58, 1.07, -1.78, -2.21, 8.01, 0.6,
        ],
        0,
    ),
    (
        "Pb",
        "fcc",
        [
            6.0564428, 5.306, 2.20, 6.0, 2.20, 4.95, 2.04, 1.01, 2.74022352, 3.06323991, 1.2, 1.0,
        ],
        0,
    ),
    (
        "Rh",
        "fcc",
        [
            6.0045385, 1.131, 1.00, 2.0, 1.00, 3.8, 5.75, 1.05, 2.99, 4.60231784, 4.8, 1.0,
        ],
        0,
    ),
    (
        "Ir",
        "fcc",
        [
            6.52315787, 1.13, 1.00, 2.0, 1.00, 3.84, 6.93, 1.05, 1.5, 8.09942666, 4.8, 1.0,
        ],
        0,
    ),
    (
        "Li",
        "bcc",
        [
            2.97244804, 1.425, 1.00, 1.00169907, 1.00, 3.509, 1.65, 0.87, 0.26395017, 0.44431129,
            -0.2, 1.0,
        ],
        0,
    ),
    (
        "Na",
        "bcc",
        [
            3.64280541, 2.313, 1.00, 1.00173951, 1.00, 4.291, 1.13, 0.9, 3.55398839, 0.68807569,
            -0.2, 1.0,
        ],
        0,
    ),
    (
        "K",
        "bcc",
        [
            3.90128376, 2.687, 1.00, 1.00186667, 1.00, 5.344, 0.94, 0.92, 5.09756981, 0.69413264,
            -0.2, 1.0,
        ],
        0,
    ),
    (
        "V",
        "bcc",
        [
            4.83265262, 4.113, 1.00, 1.00095022, 1.00, 3.04, 5.3, 1.0, 4.20161301, 4.09946561,
            -1.0, 1.0,
        ],
        0,
    ),
    (
        "Nb",
        "bcc",
        [
            4.79306197, 4.374, 1.00, 1.00101441, 1.00, 3.301, 7.47, 1.0, 3.75762849, 3.82514598,
            -1.0, 1.0,
        ],
        0,
    ),
    (
        "Ta",
        "bcc",
        [
            4.89528669, 3.709, 1.00, 1.00099783, 1.00, 3.303, 8.09, 0.99, 6.08617812, 3.35255804,
            -2.9, 1.0,
        ],
        0,
    ),
    (
        "Cr",
        "bcc",
        [
            5.12169218, 3.224, 1.00, 1.00048646, 1.00, 2.885, 4.1, 0.94, -0.207535, 12.2600006,
            -1.9, 1.0,
        ],
        0,
    ),
    (
        "Mo",
        "bcc",
        [
            5.84872871, 4.481, 1.00, 1.00065204, 1.00, 3.15, 6.81, 0.99, 3.47727181, 9.48582009,
            -2.9, 1.0,
        ],
        0,
    ),
    (
        "W",
        "bcc",
        [
            5.62777409, 3.978, 1.00, 1.00065894, 1.00, 3.165, 8.66, 0.98, 3.16353338, 8.24586928,
            -2.7, 1.0,
        ],
        0,
    ),
    (
        "Fe",
        "bcc",
        [
            5.07292627, 2.935, 1.00, 1.00080073, 1.00, 2.866, 4.29, 0.89, 5.13579244, 4.12042448,
            -2.7, 1.0,
        ],
        0,
    ),
    (
        "Mg",
        "hcp",
        [
            5.45, 2.70, 0.0, 0.35, 3.0, 3.20, 1.55, 1.11, 8.00, 4.1, -2.0, 1.0,
        ],
        0,
    ),
    (
        "Si",
        "dia",
        [
            4.87, 4.8, 4.8, 4.8, 4.8, 5.431, 4.63, 1.0, 3.30, 5.105, -0.80, 1.0,
        ],
        1,
    ),
    (
        "C",
        "dia",
        [
            4.38, 4.10, 4.20, 5.00, 3.00, 3.567, 7.37, 1.0, 5.0, 9.34, -1.00, 2.25,
        ],
        1,
    ),
    (
        "Sn",
        "dia",
        [
            5.09, 5.42, 8.0, 5.0, 6.0, 6.483, 3.14, 1.12, 3.0, 5.707, 0.30, 1.0,
        ],
        0,
    ),
];

/// Symbols of the elements of the library.
pub const ELEMENTS: [&str; 24] = [
    "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "Al", "Pb", "Rh", "Ir", "Li", "Na", "K", "V", "Nb", "Ta",
    "Cr", "Mo", "W", "Fe", "Mg", "Si", "C", "Sn",
];

impl MeamElement {
    /// The library entry of `element`.
    pub fn library(element: Element) -> MoptResult<Self> {
        let (_, lattice, p, ibar) = TABLE
            .iter()
            .find(|(symbol, ..)| *symbol == element.symbol())
            .ok_or_else(|| {
                MoptError::config(format!(
                    "no MEAM parameters for {element} (available: {})",
                    ELEMENTS.join(" ")
                ))
            })?;
        let lattice = Lattice::parse(lattice)?;
        let scale = lattice.distance_per_lattice_constant().ok_or_else(|| {
            MoptError::config(format!(
                "meam: {element} has a reference structure of a pair of elements"
            ))
        })?;
        Ok(Self {
            element,
            lattice,
            alpha: p[0],
            beta: [p[1], p[2], p[3], p[4]],
            re: p[5] * scale,
            ec: p[6],
            a: p[7],
            t: [p[8], p[9], p[10]],
            rho0: p[11],
            ibar: *ibar,
        })
    }
}
