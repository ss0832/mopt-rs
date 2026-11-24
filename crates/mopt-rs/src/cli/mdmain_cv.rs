//! Collective variables in `mdmain`: `-cv SPEC ...` writes their values at every step to
//! `<stem>_cv.csv` (and the trajectory analysis gives their statistics and
//! autocorrelations), `-cv_bias "SPEC;center=C;k=K" ...` restrains them harmonically,
//! `k/2 (xi - C)^2` (atomic units of the variable; Hartree per unit squared), and
//! `-umbrella cv=SPEC` samples along one. The forms of `SPEC` are those of
//! [`crate::md::cv_spec`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;
use std::sync::Arc;

use super::mdmain::MdmainConfig;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::md::cv::Umbrella;
use crate::md::cv_spec::parse_cv;
use crate::pes::term::PesTerm;
use crate::workflow::md_cvs::MdCvs;

/// A harmonic restraint of `-cv_bias`.
#[derive(Debug, Clone, PartialEq)]
pub struct CvBias {
    /// The variable.
    pub spec: String,
    /// The centre (atomic units of the variable).
    pub center: f64,
    /// The force constant (Hartree per unit squared).
    pub k: f64,
}

/// The restraints of `-cv_bias "SPEC;center=C;k=K"`.
pub fn parse_cv_biases(tokens: &[String]) -> MoptResult<Vec<CvBias>> {
    tokens
        .iter()
        .map(|t| {
            let mut parts = t.split(';');
            let spec = parts.next().unwrap_or_default().trim().to_string();
            let (mut center, mut k) = (None, None);
            for p in parts {
                let (key, value) = p.split_once('=').ok_or_else(|| {
                    MoptError::config(format!("-cv_bias: '{p}' is not KEY=VALUE"))
                })?;
                let v: f64 = value.trim().parse().map_err(|_| {
                    MoptError::config(format!("-cv_bias {key}={value}: not a number"))
                })?;
                match key.trim() {
                    "center" => center = Some(v),
                    "k" => k = Some(v),
                    other => {
                        return Err(MoptError::config(format!(
                            "-cv_bias: unknown setting '{other}' (center, k)"
                        )))
                    }
                }
            }
            parse_cv(&spec).context(format!("-cv_bias {spec}"))?;
            match (center, k) {
                (Some(center), Some(k)) => Ok(CvBias { spec, center, k }),
                _ => Err(MoptError::config(format!(
                    "-cv_bias {t}: needs center= and k=, as \"distance:1,2;center=3.0;k=0.1\""
                ))),
            }
        })
        .collect()
}

/// Check the specifications of `-cv`.
pub fn check_cvs(specs: &[String]) -> MoptResult<()> {
    for s in specs {
        parse_cv(s).context(format!("-cv {s}"))?;
    }
    Ok(())
}

/// The restraints of `-cv_bias` as terms of the surface.
pub(super) fn cv_terms(config: &MdmainConfig) -> MoptResult<Vec<(String, Arc<dyn PesTerm>)>> {
    config
        .cv_biases
        .iter()
        .map(|b| {
            let term: Arc<dyn PesTerm> =
                Arc::new(Umbrella::new(parse_cv(&b.spec)?, b.center, b.k)?);
            Ok((format!("cv_bias({})", b.spec), term))
        })
        .collect()
}

/// The observer of `-cv` (none without it).
pub(super) fn cv_observer(
    config: &MdmainConfig,
    start: &MolecularSystem,
    dir: &Path,
    stem: &str,
) -> MoptResult<Option<MdCvs>> {
    if config.cvs.is_empty() {
        return Ok(None);
    }
    let cvs = config
        .cvs
        .iter()
        .map(|s| Ok((s.replace(',', " "), parse_cv(s)?)))
        .collect::<MoptResult<Vec<_>>>()?;
    Ok(Some(MdCvs::new(cvs, start, dir, stem)))
}
