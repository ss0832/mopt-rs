//! GFN-FF element data in the units xtb uses them, for H to Rn (Z = 1 to 86).
//!
//! From [`crate::parameters::gfnff`] (generated from xtb's `gfnff_param.f90`,
//! `covalentradd3.f90`, `sqrtzr4r2.f90`, `gfnff_rab.f`): `rcov` is the D3 covalent radius in Bohr
//! times 4/3 (`covalentRadD3`), `sqrt_zr4r2` is `sqrt(0.5 <r^4>/<r^2> sqrt(Z))`, both as xtb
//! computes them with its legacy Bohr radius 0.52917726. xtb's tables go on to Lr, but its D4
//! reference systems (the C6 coefficients of the dispersion) stop at Rn and xtb 6.7.1 gives no
//! usable GFN-FF result for Fr to Lr, so those are refused.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::Element;
use crate::error::{MoptError, MoptResult};
use crate::parameters::gfnff::{GfnffElementRaw, GFNFF_ELEMENTS, GFNFF_RAB_P};

/// xtb's `autoaa`.
pub const AUTOAA: f64 = 0.529_177_26;
/// xtb's `aatoau`.
pub const AATOAU: f64 = 1.0 / AUTOAA;

/// The elements GFN-FF is offered for in mopt-rs (H to Rn, in the order of Z).
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

/// Element data of one atom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atom {
    /// Raw table entry.
    pub raw: GfnffElementRaw,
    /// `covalentRadD3` (Bohr, times 4/3).
    pub rcov: f64,
    /// `sqrtZr4r2`.
    pub sqrt_zr4r2: f64,
}

impl Atom {
    /// Atomic number.
    pub fn z(&self) -> u8 {
        self.raw.z
    }

    /// xtb's `metal` of the element: 1 a main-group metal, 2 a transition metal, 0 none.
    pub fn metal(&self) -> i32 {
        self.raw.metal
    }
}

/// The table entry of the element `z` (`None` outside [`ELEMENT_RANGE`]).
pub fn raw_of(z: u8) -> Option<&'static GfnffElementRaw> {
    let e = GFNFF_ELEMENTS.get(usize::from(z).checked_sub(1)?)?;
    (e.z == z).then_some(e)
}

/// xtb's `metal` of the element `z` (0 for an element without data).
pub fn metal_of(z: u8) -> i32 {
    raw_of(z).map_or(0, |e| e.metal)
}

/// The data of `el`, or an error naming the supported elements.
pub fn atom(el: Element) -> MoptResult<Atom> {
    let z = el.atomic_number();
    let raw = *raw_of(z).ok_or_else(|| {
        MoptError::unsupported(format!(
            "GFN-FF has parameters for {ELEMENT_RANGE} only, not {} (Z = {z})",
            el.symbol()
        ))
    })?;
    Ok(Atom {
        raw,
        rcov: raw.covalent_radius_ang * AATOAU * 4.0 / 3.0,
        sqrt_zr4r2: (0.5 * (raw.r4_over_r2 * f64::from(z).sqrt())).sqrt(),
    })
}

/// `itabrow6`: period row 1..6.
pub fn row6(z: u8) -> usize {
    match z {
        1..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        _ => 6,
    }
}

/// `gfnffrab`: CN-dependent bond-length estimate of a pair (Bohr), without the per-bond shift.
pub fn rab_estimate(a: &Atom, cn_a: f64, b: &Atom, cn_b: f64) -> f64 {
    rab_estimate_shifted(a, cn_a, b, cn_b, 0.0)
}

/// `gfnffdrab`: `(ra + rb + shift) ff`. As in xtb 6.7.1, without the factor of lanthanide and
/// actinide halides (`scaleF`) that later states of xtb's `gfnff_rab.f` have.
pub fn rab_estimate_shifted(a: &Atom, cn_a: f64, b: &Atom, cn_b: f64, shift: f64) -> f64 {
    let [ena, r0a, cnfa] = a.raw.rab;
    let [enb, r0b, cnfb] = b.raw.rab;
    let (ir, jr) = (row6(a.z()) - 1, row6(b.z()) - 1);
    let ra = r0a + cnfa * cn_a;
    let rb = r0b + cnfb * cn_b;
    let den = (ena - enb).abs();
    let k1 = 0.005 * (GFNFF_RAB_P[ir][0] + GFNFF_RAB_P[jr][0]);
    let k2 = 0.005 * (GFNFF_RAB_P[ir][1] + GFNFF_RAB_P[jr][1]);
    let ff = 1.0 - k1 * den - k2 * den * den;
    if shift == 0.0 {
        (ra + rb) * ff
    } else {
        (ra + rb + shift) * ff
    }
}

/// `ff` of the bond-length estimate and `d R / d CN_a`, `d R / d CN_b`.
pub fn rab_slopes(a: &Atom, b: &Atom) -> (f64, f64, f64) {
    let (ena, enb) = (a.raw.rab[0], b.raw.rab[0]);
    let (ir, jr) = (row6(a.z()) - 1, row6(b.z()) - 1);
    let den = (ena - enb).abs();
    let k1 = 0.005 * (GFNFF_RAB_P[ir][0] + GFNFF_RAB_P[jr][0]);
    let k2 = 0.005 * (GFNFF_RAB_P[ir][1] + GFNFF_RAB_P[jr][1]);
    let ff = 1.0 - k1 * den - k2 * den * den;
    (ff, a.raw.rab[2] * ff, b.raw.rab[2] * ff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_element_of_the_range_has_its_parameters() {
        assert_eq!(GFNFF_ELEMENTS.len(), ELEMENTS.len());
        for (k, symbol) in ELEMENTS.iter().enumerate() {
            let el = Element::parse_token(symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(usize::from(el.atomic_number()), k + 1, "{symbol}");
            let a = atom(el).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(a.z(), el.atomic_number());
            for v in [
                a.rcov,
                a.sqrt_zr4r2,
                a.raw.rab[1],
                a.raw.zeta[0],
                a.raw.zeta[1],
            ] {
                assert!(v.is_finite() && v > 0.0, "{symbol}: {v}");
            }
            assert!(a.raw.alp > 0.0 && a.raw.chi > 0.0, "{symbol}");
        }
    }

    #[test]
    fn elements_beyond_the_range_name_it() {
        for z in [0u8, 87, 92, 103, 118] {
            let el = Element::from_atomic_number(z).unwrap_or_else(|e| panic!("{z}: {e}"));
            let err = match atom(el) {
                Ok(_) => panic!("Z = {z} accepted"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("H to Rn (Z = 1 to 86)") && err.contains(&format!("(Z = {z})")),
                "{err}"
            );
        }
    }

    #[test]
    fn lanthanides_have_the_parameters_of_xtb_6_7_1() {
        // The release's EEQ parameters and bond radius of La and Ce (the later lanthanide
        // extension of the xtb source has others).
        let la = atom(Element::from_atomic_number(57).unwrap()).unwrap();
        assert_eq!((la.raw.chi, la.raw.rab[1]), (1.127797, 3.19903011));
        let ce = atom(Element::from_atomic_number(58).unwrap()).unwrap();
        assert_eq!((ce.raw.chi, ce.raw.rab[1]), (1.127863, 2.8));
    }
}
