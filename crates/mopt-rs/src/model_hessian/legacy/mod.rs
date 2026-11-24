//! MultiOptPy's model Hessians reproduced as they are (`legacy_` names).
//!
//! These follow MultiOptPy 1.21.2's `ModelHessian/*.py` line by line, including its parameters,
//! its approximate dispersion terms, and its numerical quirks, so that runs can be compared with
//! earlier MultiOptPy results; `tests/golden/model_hessians_legacy.json` (from
//! `tools/golden/model_hessians.py`) holds the reference values. The published models are the
//! names without the prefix. Differences between the two are listed in `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `ModelHessian/*.py`, selected by `ModelHessian/approx_hessian.py:ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod common;
mod fischer;
mod fischerd3;
mod fischerd3old;
mod fischerd4;
mod gfn0xtb;
mod gfnff;
mod lindh;
mod lindh2007;
mod lindh2007d3;
mod schlegel;
mod schlegeld3;
mod schlegeld4;
mod shortrange;
mod swart;
mod swartd2;
mod swartd3;
mod swartd4;

use faer::{Col, Mat};

pub use shortrange::{apply_short_range, short_range_correction};

use super::ModelHessian;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// Base names with a MultiOptPy version.
pub const LEGACY_MODELS: &[&str] = &[
    "fischer",
    "fischerd3",
    "fischerd3old",
    "fischerd4",
    "gfn0xtb",
    "gfnff",
    "lindh",
    "lindh2007",
    "lindh2007d3",
    "lindh2007d4",
    "schlegel",
    "schlegeld3",
    "schlegeld4",
    "swart",
    "swartd2",
    "swartd3",
    "swartd4",
];

/// The elements of MultiOptPy's GFN-FF-like model: those of its element table
/// (`element_params` of `Parameters/gfnff_param.py`).
pub const GFNFF_LIKE_RANGE: &str = "H to Rn (Z = 1 to 86)";
/// The elements of MultiOptPy's GFN0-like model: those of its radii and electronegativities
/// (`rad`, `en` of `Parameters/gfn0xtb_param.py`).
pub const GFN0_LIKE_RANGE: &str = "H to Xe (Z = 1 to 54)";

/// Whether MultiOptPy's tables of the model `base` have the element `z` (its other tables, such
/// as the bond and angle parameters or the reference CNs, fall back to MultiOptPy's defaults).
pub fn gfn_like_has_element(base: &str, z: u8) -> bool {
    use crate::parameters::multioptpy_gfn as t;
    let k = usize::from(z);
    if k == 0 {
        return false;
    }
    let has = |table: &[f64]| table.get(k).is_some_and(|v| !v.is_nan());
    if base == "gfnff" {
        has(&t::GFNFF_COV_RADIUS_ANG)
            && has(&t::GFNFF_VDW_RADIUS_ANG)
            && has(&t::GFNFF_ELECTRONEGATIVITY)
            && has(&t::GFNFF_POLARIZABILITY)
    } else {
        has(&t::GFN0_RADIUS_BOHR) && has(&t::GFN0_EN)
    }
}

fn check_gfn_elements(base: &str, sys: &MolecularSystem) -> MoptResult<()> {
    let mut other: Vec<String> = Vec::new();
    for e in &sys.elements {
        let z = e.atomic_number();
        let s = format!("{} (Z = {z})", e.symbol());
        if !gfn_like_has_element(base, z) && !other.contains(&s) {
            other.push(s);
        }
    }
    if other.is_empty() {
        return Ok(());
    }
    let (what, range) = if base == "gfnff" {
        ("GFN-FF", GFNFF_LIKE_RANGE)
    } else {
        ("GFN0-xTB", GFN0_LIKE_RANGE)
    };
    Err(MoptError::unsupported(format!(
        "legacy_{base} (MultiOptPy's {what}-like model Hessian) has element parameters for \
         {range} only, not {}",
        other.join(", ")
    )))
}

/// A MultiOptPy model Hessian.
#[derive(Debug, Clone)]
pub struct LegacyModel {
    name: String,
    base: &'static str,
}

impl LegacyModel {
    /// MultiOptPy's version of `base`.
    pub fn new(base: &'static str) -> MoptResult<Self> {
        if !LEGACY_MODELS.contains(&base) {
            return Err(MoptError::unsupported(format!(
                "the MultiOptPy version of '{base}' (legacy_) is not available yet (available: {})",
                LEGACY_MODELS.join(", ")
            )));
        }
        Ok(Self {
            name: format!("legacy_{base}"),
            base,
        })
    }
}

impl ModelHessian for LegacyModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn hessian(&self, sys: &MolecularSystem, gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        let g: Vec<f64> = gradient.iter().copied().collect();
        Ok(match self.base {
            "fischer" => fischer::fischer(sys),
            "fischerd3" => fischerd3::fischerd3(sys),
            "fischerd3old" => fischerd3old::fischerd3old(sys),
            "fischerd4" => fischerd4::fischerd4(sys),
            "gfnff" => {
                check_gfn_elements("gfnff", sys)?;
                gfnff::gfnff(sys)
            }
            "gfn0xtb" => {
                check_gfn_elements("gfn0xtb", sys)?;
                gfn0xtb::gfn0xtb(sys)
            }
            "schlegel" => schlegel::schlegel(sys),
            "schlegeld3" => schlegeld3::schlegeld3(sys),
            "schlegeld4" => schlegeld4::schlegeld4(sys),
            "lindh" => lindh::lindh(sys, &g)?,
            "lindh2007" => lindh2007::lindh2007(sys, gradient)?,
            "lindh2007d3" => lindh2007d3::lindh2007d3(sys, gradient)?,
            "lindh2007d4" => lindh2007d3::lindh2007d4(sys, gradient)?,
            "swart" => swart::swart(sys),
            "swartd2" => swartd2::swartd2(sys),
            "swartd3" => swartd3::swartd3(sys),
            "swartd4" => swartd4::swartd4(sys),
            other => {
                return Err(MoptError::unsupported(format!(
                    "legacy model '{other}' is not available"
                )))
            }
        })
    }
}
