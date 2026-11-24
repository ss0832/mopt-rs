//! GFN0-xTB parameters in atomic units, for H to Rn (every element of GFN0-xTB).
//!
//! The fitted values come from [`crate::parameters::gfn0_xtb`] (generated from xtb's
//! `param_gfn0-xtb.txt`); they are converted as xtb's parameter reader and gfn0-rs
//! (`src/params.rs`) convert them: levels, charge shifts, and `KQAT2` from eV, the CN shifts
//! scaled by 0.1 first. The fixed tables (atomic and covalent radii, valence electrons) are xtb's
//! (`src/param/atomicrad.f90`, `covalentradd3.f90`, the reference occupations), taken from the
//! GFN1-xTB element table that is generated from the same files, and converted with xtb's
//! legacy Bohr radius 0.52917726 as xtb does. Pairs of d-block metals have xtb's scaling of
//! their Hamiltonian elements (`setpair` of `gfn_paramset.f90` for GFN0: 1.10, and 0.9
//! between two of Cu, Ag, Au); `param_gfn0-xtb.txt` lists no other pairs.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's GFN0-like model Hessian is
//! `model_hessian::legacy::gfn0xtb`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::Element;
use crate::error::{MoptError, MoptResult};
use crate::parameters::gfn0_xtb as raw;

/// xtb's Hartree in eV (`evtoau = 1 / 27.21138505`).
pub const HARTREE_TO_EV: f64 = 27.211_385_05;
/// eV in Hartree.
pub const EV_TO_HARTREE: f64 = 1.0 / HARTREE_TO_EV;
/// xtb's legacy Bohr radius for its tables (`xtb_mctc_convert::autoaa`).
pub const XTB_TABLE_ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_26;
/// xtb's Boltzmann constant (Hartree / K).
pub const BOLTZMANN_HARTREE_PER_K: f64 = 3.166_808_578_545_117e-6;

/// The elements GFN0-xTB is offered for in mopt-rs: every element of `param_gfn0-xtb.txt`.
pub const ELEMENTS: [&str; 86] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn",
];
/// The elements of [`ELEMENTS`], as error messages name them.
pub const ELEMENT_RANGE: &str = "H to Rn (Z = 1 to 86)";
/// The scaling of the Hamiltonian elements between two d-block metals (`setpair`).
pub const D_BLOCK_PAIR_SCALE: f64 = 1.10;
/// The scaling between two coinage metals (Cu, Ag, Au; `setpair`).
pub const COINAGE_PAIR_SCALE: f64 = 0.9;

/// A shell in atomic units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shell {
    /// Principal quantum number.
    pub n: u8,
    /// Angular momentum (0 s, 1 p, 2 d).
    pub l: usize,
    /// Level (Hartree).
    pub level: f64,
    /// Slater exponent.
    pub exponent: f64,
    /// CN shift (Hartree).
    pub kcn: f64,
    /// Charge shift (Hartree).
    pub kq: f64,
    /// Shell polynomial coefficient.
    pub poly: f64,
}

/// One element in atomic units.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementParams {
    /// Atomic number.
    pub z: u8,
    /// Shells in file order.
    pub shells: Vec<Shell>,
    /// Electronegativity of the Hamiltonian and repulsion.
    pub en: f64,
    /// EEQ hardness.
    pub gam: f64,
    /// Second-order charge shift (Hartree).
    pub kqat2: f64,
    /// Repulsion exponent.
    pub repa: f64,
    /// Repulsion effective charge.
    pub repb: f64,
    /// EEQ electronegativity.
    pub xi: f64,
    /// EEQ CN coefficient.
    pub kappa: f64,
    /// EEQ charge width.
    pub alpg: f64,
    /// SRB `r0`, `cnfak`, approximate-radius `en`, Pauling `en`.
    pub srb: [f64; 4],
    /// EEQ-2019 `chi`, `eta`, `kcnchi`, `rad` (charges of the D4 term).
    pub eeq2019: [f64; 4],
    /// xtb atomic radius (Bohr) of the shell polynomial.
    pub atomic_radius: f64,
    /// D3 covalent radius (Bohr) of the coordination numbers.
    pub covalent_radius: f64,
    /// Valence electrons.
    pub valence_electrons: f64,
    /// Pauling electronegativity of the covalent coordination number.
    pub pauling_en: f64,
}

/// `(atomic radius, D3 covalent radius)` in Angstrom and valence electrons (xtb tables, as the
/// GFN1-xTB element table carries them).
fn fixed(el: Element) -> Option<(f64, f64, f64)> {
    let g = crate::pes::native::gfn1::params::element(el).ok()?;
    Some((g.atomic_rad, g.cov_rad_d3, g.valence_electrons()))
}

/// The parameters of `el`, or an error naming the supported elements.
pub fn element(el: Element) -> MoptResult<ElementParams> {
    let z = el.atomic_number();
    let unsupported = || {
        MoptError::unsupported(format!(
            "GFN0-xTB has parameters for {ELEMENT_RANGE} only, not {} (Z = {z})",
            el.symbol()
        ))
    };
    let r = raw::GFN0_ELEMENTS
        .iter()
        .find(|e| e.z == z)
        .ok_or_else(unsupported)?;
    let (atomic, covalent, valence) = fixed(el).ok_or_else(unsupported)?;
    Ok(ElementParams {
        z,
        shells: r
            .shells
            .iter()
            .map(|s| Shell {
                n: s.n,
                l: s.l as usize,
                level: s.level_ev * EV_TO_HARTREE,
                exponent: s.exponent,
                // xtb's reader scales KCN* by 0.1 before the eV conversion.
                kcn: s.kcn_raw * 0.1 * EV_TO_HARTREE,
                kq: s.kq_ev * EV_TO_HARTREE,
                poly: s.poly,
            })
            .collect(),
        en: r.en,
        gam: r.gam,
        kqat2: r.kqat2_ev * EV_TO_HARTREE,
        repa: r.repa,
        repb: r.repb,
        xi: r.xi,
        kappa: r.kappa,
        alpg: r.alpg,
        srb: r.srb,
        eeq2019: r.eeq2019,
        atomic_radius: atomic * XTB_TABLE_ANGSTROM_TO_BOHR,
        covalent_radius: covalent * XTB_TABLE_ANGSTROM_TO_BOHR,
        valence_electrons: valence,
        pauling_en: r.srb[3],
    })
}

/// The scale of the Hamiltonian elements of an element pair: a `$pairpar` entry, then the rule
/// for two d-block metals, 1 otherwise.
pub fn pair_scale(za: u8, zb: u8) -> f64 {
    let (a, b) = if za <= zb { (za, zb) } else { (zb, za) };
    if let Some((_, _, s)) = raw::GFN0_PAIR_SCALES
        .iter()
        .find(|(x, y, _)| *x == a && *y == b)
    {
        return *s;
    }
    let d_block = |z: u8| matches!(z, 21..=29 | 39..=47 | 57..=79);
    let coinage = |z: u8| matches!(z, 29 | 47 | 79);
    match (d_block(a), d_block(b)) {
        (true, true) if coinage(a) && coinage(b) => COINAGE_PAIR_SCALE,
        (true, true) => D_BLOCK_PAIR_SCALE,
        _ => 1.0,
    }
}

/// Hamiltonian `k_l` and EN scaling `en_l` of angular momentum `l`.
pub fn shell_k_en(l: usize) -> (f64, f64) {
    match l {
        0 => (raw::GFN0_KS, raw::GFN0_ENS),
        1 => (raw::GFN0_KP, raw::GFN0_ENP),
        2 => (raw::GFN0_KD, raw::GFN0_END),
        _ => (raw::GFN0_KF, raw::GFN0_ENF),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn every_element_from_h_to_rn_is_available_and_others_are_refused() {
        assert_eq!(raw::GFN0_ELEMENTS.len(), ELEMENTS.len());
        for (k, s) in ELEMENTS.iter().enumerate() {
            let el = Element::parse_token(s).unwrap();
            assert_eq!(usize::from(el.atomic_number()), k + 1, "{s}");
            let p = element(el).unwrap();
            assert!(!p.shells.is_empty());
            // The radii are xtb's tables, the same as those of GFN1-xTB (generated from xtb).
            let g1 = crate::pes::native::gfn1::params::element(el).unwrap();
            let radius = |a: f64| a * XTB_TABLE_ANGSTROM_TO_BOHR;
            assert!(
                (p.atomic_radius - radius(g1.atomic_rad)).abs() < 1e-12,
                "{s}"
            );
            assert!(
                (p.covalent_radius - radius(g1.cov_rad_d3)).abs() < 1e-12,
                "{s}"
            );
            assert!(
                (p.valence_electrons - g1.valence_electrons()).abs() < 1e-12,
                "{s}"
            );
        }
        for z in [0, 87, 92, 118] {
            let e = element(Element::from_atomic_number(z).unwrap()).unwrap_err();
            assert!(e.to_string().contains(ELEMENT_RANGE), "{e}");
        }
    }

    #[test]
    fn the_shells_are_s_p_d_and_the_pairs_follow_setgfn0pairparam() {
        // No element carries f functions (the lanthanides have 5d 6s 6p, the 4f shell in the
        // core), so the basis needs s, p, d only.
        for r in &raw::GFN0_ELEMENTS {
            assert!(r.shells.iter().all(|s| s.l <= 2), "Z = {}", r.z);
        }
        assert_eq!(pair_scale(79, 47), COINAGE_PAIR_SCALE);
        assert_eq!(pair_scale(57, 78), D_BLOCK_PAIR_SCALE);
        assert_eq!(pair_scale(64, 26), D_BLOCK_PAIR_SCALE);
        assert_eq!(pair_scale(80, 26), 1.0);
        assert_eq!(pair_scale(1, 78), 1.0);
    }
}
