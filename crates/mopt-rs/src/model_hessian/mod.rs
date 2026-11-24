//! Model (guess) Hessians (`-modelhess`).
//!
//! A [`ModelHessian`] gives an approximate Cartesian Hessian from the structure (and, for some
//! models, the gradient). Two versions exist for the MultiOptPy names:
//!
//! - the models as published (Lindh 1995 and 2007, Schlegel with the Wittbrodt-Schlegel
//!   parameters, Fischer-Almlof, Swart, with genuine D2/D3(BJ)/D4 dispersion), selected by the
//!   MultiOptPy name (`lindh2007d3`, `fischerd4`, ...), and
//! - MultiOptPy's implementations reproduced as they are, selected with the `legacy_` prefix
//!   (`legacy_lindh2007d3`), for comparison with earlier runs.
//!
//! Names are parsed like MultiOptPy's: the first base name contained in the string wins, and the
//! modifiers `ts` (reflect the lowest mode into a first-order saddle Hessian), `sr` (short-range
//! correction), and `clip` (smooth eigenvalue clipping) may appear anywhere, e.g.
//! `lindh2007d3_ts_clip`. xtb's own models are available as `xtb_lindh`, `xtb_lindh_d2`, and
//! `xtb_swart`. The Hessians of the built-in GFN0-xTB (`gfn0xtb`), GFN1-xTB (`gfn1xtb`) and GFN-FF
//! (`gfnff`) at the current structure are models too, whatever the surface of the run. See
//! [`resolve`].
//!
//! MultiOptPy: `ModelHessian/approx_hessian.py:ApproxHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod assemble;
mod fischer;
mod internal;
pub mod legacy;
mod modifiers;
mod morse;
mod schlegel;
mod swart_o1;

use faer::{Col, Mat};

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

pub use assemble::HessianAccumulator;
pub use fischer::Fischer;
pub use internal::{InternalModel, InternalParams, PairModel};
pub use modifiers::{clip_eigenvalues, ts_reflect};
pub use morse::MorseModel;
pub use schlegel::Schlegel;
pub use swart_o1::SwartO1;

/// An approximate Cartesian Hessian.
pub trait ModelHessian: Send + Sync {
    /// Name (as resolved).
    fn name(&self) -> &str;

    /// Cartesian Hessian (Hartree/Bohr^2) at `sys`; `gradient` is the Cartesian gradient there.
    fn hessian(&self, sys: &MolecularSystem, gradient: &Col<f64>) -> MoptResult<Mat<f64>>;
}

/// The GFN0-xTB Hessian at the current structure as the model (`-modelhess gfn0xtb`; MultiOptPy's
/// approximate GFN0-like model is `legacy_gfn0xtb`): the analytic GFN0-xTB Hessian, for H to Rn
/// (isolated systems).
#[derive(Debug, Clone, Copy, Default)]
pub struct Gfn0Hessian {
    calc: crate::pes::native::Gfn0Xtb,
}

impl ModelHessian for Gfn0Hessian {
    fn name(&self) -> &str {
        "gfn0xtb"
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        self.calc.hessian(sys)
    }
}

/// The GFN1-xTB Hessian at the current structure as the model (`-modelhess gfn1xtb`; MultiOptPy
/// has no GFN1-xTB model): the analytic Hessian of the built-in GFN1-xTB
/// ([`crate::pes::native::Gfn1Xtb`]) with the charge and multiplicity of the structure, whatever
/// the calculator of the run. Isolated systems with the elements of GFN1-xTB (H to Rn); other
/// structures are refused with the reason. Each structure starts its SCC from
/// the converged charges of the previous one.
#[derive(Debug, Clone, Default)]
pub struct Gfn1Hessian {
    calc: crate::pes::native::Gfn1Xtb,
}

impl Gfn1Hessian {
    /// Whether GFN1-xTB can give the Hessian of `sys` (an error naming the reason otherwise).
    pub fn check(sys: &MolecularSystem) -> MoptResult<()> {
        if sys.has_pbc() {
            return Err(MoptError::unsupported(
                "-modelhess gfn1xtb: GFN1-xTB in mopt-rs is for isolated systems only; give \
                 periodic cells another model Hessian (gfnff, lindh2007d3, ...)",
            ));
        }
        for e in &sys.elements {
            crate::pes::native::gfn1::params::element(*e)
                .map_err(|err| err.context("-modelhess gfn1xtb"))?;
        }
        Ok(())
    }
}

impl ModelHessian for Gfn1Hessian {
    fn name(&self) -> &str {
        "gfn1xtb"
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        Self::check(sys)?;
        self.calc
            .hessian(sys)
            .map_err(|e| e.context("-modelhess gfn1xtb"))
    }
}

/// The GFN-FF Hessian as the model (`-modelhess gfnff`; MultiOptPy's approximate GFN-FF-like
/// model is `legacy_gfnff`): the analytic Hessian of the built-in GFN-FF, with the topology of
/// the first structure, for the elements it has parameters for (H to Rn), molecules and
/// three-dimensional cells. `legacy_gfnff` covers H to Rn too, `legacy_gfn0xtb` H to Xe (the
/// elements of MultiOptPy's tables).
#[derive(Debug, Clone, Default)]
pub struct GfnffHessian {
    calc: crate::pes::native::Gfnff,
}

impl ModelHessian for GfnffHessian {
    fn name(&self) -> &str {
        "gfnff"
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        self.calc.hessian(sys)
    }
}

/// Dispersion added to a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispersion {
    /// None.
    None,
    /// D2 pair term of xtb's model Hessians.
    D2,
    /// DFT-D3 with Becke-Johnson damping.
    D3,
    /// DFT-D4.
    D4,
}

/// Add the dispersion Hessian of `dispersion` (D3 and D4; the D2 term belongs to the Lindh-type
/// models) to `h`: the Hessian of the genuine PBE-D3(BJ) (exact) or PBE-D4 (two- and three-body,
/// EEQ charges; central differences of the analytic gradient) dispersion energy.
pub(crate) fn add_dispersion(
    h: &mut Mat<f64>,
    sys: &MolecularSystem,
    dispersion: Dispersion,
) -> MoptResult<()> {
    match dispersion {
        Dispersion::None | Dispersion::D2 => Ok(()),
        Dispersion::D3 => {
            let out = crate::pes::dispersion::D3Bj::PBE.evaluate(sys, true)?;
            if let Some(hd) = out.hessian_bohr {
                *h += hd;
            }
            Ok(())
        }
        Dispersion::D4 => {
            let out = crate::pes::dispersion::D4::PBE.evaluate(sys, true)?;
            if let Some(hd) = out.hessian_bohr {
                *h += hd;
            }
            Ok(())
        }
    }
}

/// Base model names (without `legacy_` and modifiers) in MultiOptPy's matching order.
pub fn base_names() -> &'static [&'static str] {
    &BASES
}

/// Base model names in MultiOptPy's matching order.
const BASES: [&str; 23] = [
    "xtb_lindh_d2",
    "xtb_lindh",
    "xtb_swart",
    "gfnff",
    "gfn0xtb",
    "gfn1xtb",
    "fischerd3old",
    "fischerd3",
    "fischerd4",
    "schlegeld3",
    "schlegeld4",
    "schlegel",
    "swartd3",
    "swartd4",
    "swartd2",
    "swart",
    "lindh2007d3",
    "lindh2007d4",
    "lindh2007",
    "lindh",
    "fischer",
    "morse",
    "o1numhess",
];

/// A parsed `-modelhess` name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// Base model.
    pub base: &'static str,
    /// MultiOptPy's implementation instead of the published model.
    pub legacy: bool,
    /// `ts` modifier.
    pub ts: bool,
    /// `sr` modifier.
    pub short_range: bool,
    /// `clip` modifier.
    pub clip: bool,
}

/// The name MultiOptPy uses when `-modelhess` is given without a value.
pub const DEFAULT_MODEL: &str = "fischerd3old";

impl ModelSpec {
    /// Parse `name` (case-insensitive).
    pub fn parse(name: &str) -> MoptResult<Self> {
        let lower = name.to_ascii_lowercase();
        let legacy = lower.contains("legacy");
        let rest = lower.replace("legacy", "");
        let base = BASES
            .iter()
            .copied()
            .find(|b| rest.contains(b))
            .ok_or_else(|| {
                MoptError::config(format!(
                    "unknown model Hessian '{name}' (known: {})",
                    BASES.join(", ")
                ))
            })?;
        let modifiers = rest.replacen(base, "", 1);
        Ok(Self {
            base,
            legacy,
            ts: modifiers.contains("ts"),
            short_range: modifiers.contains("sr"),
            clip: modifiers.contains("clip"),
        })
    }
}

/// A model with its modifiers applied.
struct Modified {
    name: String,
    base: Box<dyn ModelHessian>,
    spec: ModelSpec,
}

impl ModelHessian for Modified {
    fn name(&self) -> &str {
        &self.name
    }

    fn hessian(&self, sys: &MolecularSystem, gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        let mut h = self.base.hessian(sys, gradient)?;
        if self.spec.ts {
            h = ts_reflect(&h)?;
        }
        if self.spec.short_range {
            h = legacy::apply_short_range(&h, sys);
        }
        if self.spec.clip {
            h = clip_eigenvalues(&h, 0.1)?;
        }
        Ok(h)
    }
}

fn base_model(spec: &ModelSpec) -> MoptResult<Box<dyn ModelHessian>> {
    if spec.legacy && spec.base == "gfn1xtb" {
        return Err(MoptError::config(
            "MultiOptPy has no GFN1-xTB model Hessian, so there is no legacy_gfn1xtb; gfn1xtb is \
             the Hessian of the built-in GFN1-xTB",
        ));
    }
    if spec.legacy {
        return Ok(Box::new(legacy::LegacyModel::new(spec.base)?));
    }
    Ok(match spec.base {
        "lindh" => Box::new(InternalModel::lindh_1995()),
        "lindh2007" => Box::new(InternalModel::lindh_2007(Dispersion::D2)),
        "xtb_lindh" => Box::new(InternalModel::xtb_lindh()),
        "xtb_lindh_d2" => Box::new(InternalModel::xtb_lindh_d2()),
        "xtb_swart" | "swartd2" => Box::new(InternalModel::xtb_swart()),
        "swart" => Box::new(SwartO1::default()),
        "schlegel" => Box::new(Schlegel::new(Dispersion::None)),
        "fischer" => Box::new(Fischer::new(Dispersion::None)),
        "morse" => Box::new(MorseModel::default()),
        "lindh2007d3" => Box::new(InternalModel::lindh_2007(Dispersion::D3)),
        "lindh2007d4" => Box::new(InternalModel::lindh_2007(Dispersion::D4)),
        "swartd3" => Box::new(InternalModel::xtb_swart().with_dispersion(Dispersion::D3)),
        "swartd4" => Box::new(InternalModel::xtb_swart().with_dispersion(Dispersion::D4)),
        "schlegeld3" => Box::new(Schlegel::new(Dispersion::D3)),
        "schlegeld4" => Box::new(Schlegel::new(Dispersion::D4)),
        "fischerd3" | "fischerd3old" => Box::new(Fischer::new(Dispersion::D3)),
        "fischerd4" => Box::new(Fischer::new(Dispersion::D4)),
        "gfn0xtb" => Box::new(Gfn0Hessian::default()),
        "gfn1xtb" => Box::new(Gfn1Hessian::default()),
        "gfnff" => Box::new(GfnffHessian::default()),
        "o1numhess" => {
            return Err(MoptError::unsupported(
                "O1NumHess is a Hessian from extra gradient evaluations, not a model: optimizers \
                 ask the calculator for it (wrap the calculator in \
                 pes::o1numhess::O1NumHessCalculator)",
            ))
        }
        other => {
            return Err(MoptError::config(format!(
                "unknown model Hessian '{other}'"
            )))
        }
    })
}

/// The model Hessian called `name` (see the module documentation for the syntax).
pub fn resolve(name: &str) -> MoptResult<Box<dyn ModelHessian>> {
    let spec = ModelSpec::parse(name)?;
    let base = base_model(&spec)?;
    if !(spec.ts || spec.short_range || spec.clip) {
        return Ok(base);
    }
    Ok(Box::new(Modified {
        name: name.to_string(),
        base,
        spec,
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn names_are_parsed_like_multioptpy() {
        let s = ModelSpec::parse("Lindh2007D3_ts").unwrap();
        assert_eq!(
            (s.base, s.legacy, s.ts, s.clip),
            ("lindh2007d3", false, true, false)
        );
        let s = ModelSpec::parse("legacy_fischerd3old_clip").unwrap();
        assert_eq!((s.base, s.legacy, s.clip), ("fischerd3old", true, true));
        assert_eq!(ModelSpec::parse("swartd2").unwrap().base, "swartd2");
        assert_eq!(
            ModelSpec::parse("xtb_lindh_d2").unwrap().base,
            "xtb_lindh_d2"
        );
        assert_eq!(ModelSpec::parse("gfn0xtb").unwrap().base, "gfn0xtb");
        let s = ModelSpec::parse("gfn1xtb_ts").unwrap();
        assert_eq!((s.base, s.legacy, s.ts), ("gfn1xtb", false, true));
        assert!(resolve("legacy_gfn1xtb").is_err());
        assert!(ModelSpec::parse("nonsense").is_err());
    }
}
