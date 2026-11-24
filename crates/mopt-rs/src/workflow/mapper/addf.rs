//! The mapper method `addf`: the anharmonic downward distortion following from a minimum
//! (the method `addf` of the registry of two-point methods, [`crate::workflow::explore::addf`]),
//! its estimates of transition states optimized, and the IRC of each new transition state with
//! its two ends optimized to minima ([`super::follow`]).
//!
//! A minimum has one task, `addf`: the whole following from it, whose paths end at the
//! transition states they pass over, at dissociation channels (the structure falls apart into
//! more fragments than the minimum), or after the largest number of spheres. The settings of
//! the following are the keys of `addf` in the registry, given as `method_KEY=VALUE` (for
//! example `method_paths=8 method_spheres=60`); keys it does not know are refused when the run
//! starts.
//!
//! K. Ohno, S. Maeda, Chem. Phys. Lett. 2004, 384, 277; S. Maeda, K. Ohno, J. Phys. Chem. A
//! 2005, 109, 5742.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's mapper explores by AFIR alone, and
//!  its ADDF is the separate `ieipmain -addf`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use super::follow::{follow_guesses, Guess};
use super::method::{ExplorationMethod, Job, JobContext, JobOutcome, PairSearchRequest, Task};
use super::network::{Network, Node, NodeKind};
use crate::core_infra::json::JsonValue;
use crate::error::{MoptError, MoptResult};
use crate::workflow::pair_search::PairMethodRegistry;

/// The name of the following in the registry of two-point methods.
const REGISTRY_NAME: &str = "addf";

/// The key of the bond scale of the dissociation test (`method_dissociation`).
const DISSOCIATION_KEY: &str = "dissociation";
/// The key of the atoms the dissociation test looks at (`method_dissociation_atoms`).
const DISSOCIATION_ATOMS_KEY: &str = "dissociation_atoms";

/// A path ends as a dissociation channel when its structure has more fragments than the
/// minimum with bonds up to this times the sums of the covalent radii: twice the sum, the
/// definition the documentation of the method gives for the growing spheres. Measured on
/// acetaldehyde with GFN0-xTB (`docs/source/mapper.md`): with 1.5 a stretched C-H bond already
/// counted as a lost hydrogen atom, and only the methyl rotation was found.
pub const ADDF_DISSOCIATION_SCALE: f64 = 2.0;

/// The method `addf`.
#[derive(Debug, Clone)]
pub struct AddfMap {
    settings: Vec<(String, String)>,
    dissociation_scale: f64,
    dissociation_atoms: Option<Vec<usize>>,
}

impl AddfMap {
    /// The registered name.
    pub const NAME: &'static str = "addf";

    /// The method with the `method_KEY=VALUE` settings `extra` (without the prefix): the
    /// settings of the following, and `dissociation`, the bond scale of the test that ends a
    /// path as a dissociation channel.
    pub fn new(extra: &BTreeMap<String, String>) -> MoptResult<Self> {
        let method = PairMethodRegistry::default().get(REGISTRY_NAME)?;
        let known = method.keys();
        let dissociation_scale = match extra.get(DISSOCIATION_KEY) {
            None => ADDF_DISSOCIATION_SCALE,
            Some(v) => v
                .parse::<f64>()
                .ok()
                .filter(|s| s.is_finite() && *s >= 1.0)
                .ok_or_else(|| {
                    MoptError::parse(format!(
                        "method_dissociation: a bond scale of at least 1, got '{v}'"
                    ))
                })?,
        };
        let dissociation_atoms = match extra.get(DISSOCIATION_ATOMS_KEY) {
            None => None,
            Some(v) => Some(
                crate::core_infra::arg_parser::parse_atom_indices_one_based(v)
                    .map_err(|e| e.context("method_dissociation_atoms"))?
                    .into_iter()
                    .map(|i| i.saturating_sub(1))
                    .collect(),
            ),
        };
        let own = [DISSOCIATION_KEY, DISSOCIATION_ATOMS_KEY];
        let unknown: Vec<&str> = extra
            .keys()
            .map(String::as_str)
            .filter(|k| !own.contains(k) && !known.iter().any(|(key, _)| key == k))
            .collect();
        if !unknown.is_empty() {
            return Err(MoptError::parse(format!(
                "the mapper method addf: unknown setting(s) {} (method_KEY with the keys of \
                 addf: {})",
                unknown
                    .iter()
                    .map(|k| format!("method_{k}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                known.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ")
            )));
        }
        Ok(Self {
            settings: extra
                .iter()
                .filter(|(k, _)| !own.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            dissociation_scale,
            dissociation_atoms,
        })
    }
}

impl ExplorationMethod for AddfMap {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        if node.kind != NodeKind::Minimum {
            return Ok(Vec::new());
        }
        Ok(vec![Task {
            key: "addf".to_string(),
            description: "ADDF from the minimum".to_string(),
            parameters: JsonValue::Null,
        }])
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let search = ctx.jobs.pair_search(
            &PairSearchRequest {
                label: "ADDF".to_string(),
                job: job.id,
                method: REGISTRY_NAME.to_string(),
                structures: vec![job.start.clone()],
                settings: self.settings.clone(),
                dissociation_scale: Some(self.dissociation_scale),
                dissociation_atoms: self.dissociation_atoms.clone(),
            },
            ctx.frames,
        )?;
        if search.interrupted {
            return Ok(JobOutcome::stopped());
        }
        let mut outcome = JobOutcome::default();
        if search.estimates.is_empty() {
            ctx.jobs.note(&format!(
                "Job {}: no ADDF path passed a transition state ({}).",
                job.id, search.end
            ));
            return Ok(outcome);
        }
        // The estimates come the lowest first: the transition states most likely to matter.
        let guesses: Vec<Guess> = search
            .estimates
            .into_iter()
            .enumerate()
            .map(|(k, s)| Guess {
                system: s.system,
                origin: format!("estimate {} of the ADDF", k + 1),
            })
            .collect();
        if !follow_guesses(job, ctx, &guesses, &mut outcome)? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}
