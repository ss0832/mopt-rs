//! When the mapper ends (extension point): the rules of
//! [`crate::workflow::explore::stop`] by name, with their parameters after a colon, in a
//! [`StopRuleRegistry`]. A run always ends when no node has a task left, and after
//! `max_iterations` jobs (MultiOptPy's `--max_iter`; 0: no limit); further rules are given as
//! `-map_opt stop=nothing_new:20` (several separated by `;`).
//!
//! Built in: `max_jobs:N` (the same as `--max_iter`), `nothing_new:N` (no new minimum in `N`
//! jobs in a row), `rank_unchanged:R,N` (the energies of the `R` lowest minima unchanged for
//! more than `N` jobs, as `confsearch -nr -nl` counts).
//!
//! MultiOptPy: the `max_iterations` limit and the exhaustion of the queue in
//!  `Wrapper/mapper.py:ReactionNetworkMapper._run_sequential`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::{MaxTrials, NothingNew, RankUnchanged, StopRule};

/// Makes a rule from the text after its name (empty when there is none).
pub type StopRuleFactory = Arc<dyn Fn(&str) -> MoptResult<Box<dyn StopRule>> + Send + Sync>;

/// Stop rules by name.
pub struct StopRuleRegistry {
    entries: Vec<(String, StopRuleFactory)>,
}

fn whole(name: &str, text: &str) -> MoptResult<usize> {
    text.trim().parse().map_err(|_| {
        MoptError::parse(format!(
            "the stop rule {name} takes a whole number (got '{text}')"
        ))
    })
}

impl StopRuleRegistry {
    /// `max_jobs`, `nothing_new`, and `rank_unchanged`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register(
            "max_jobs",
            Arc::new(|p: &str| {
                Ok(Box::new(MaxTrials {
                    max: whole("max_jobs", p)?.max(1),
                }) as Box<dyn StopRule>)
            }),
        );
        r.register(
            "nothing_new",
            Arc::new(|p: &str| {
                Ok(Box::new(NothingNew::new(whole("nothing_new", p)?.max(1))) as Box<dyn StopRule>)
            }),
        );
        r.register(
            "rank_unchanged",
            Arc::new(|p: &str| {
                let (rank, patience) = p.split_once(',').ok_or_else(|| {
                    MoptError::parse(format!(
                        "the stop rule rank_unchanged takes RANK,PATIENCE (got '{p}')"
                    ))
                })?;
                Ok(Box::new(RankUnchanged::new(
                    whole("rank_unchanged", rank)?,
                    whole("rank_unchanged", patience)?,
                )) as Box<dyn StopRule>)
            }),
        );
        r
    }

    /// Register (or replace) a rule.
    pub fn register(&mut self, name: &str, factory: StopRuleFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// The registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// The rule of `spec` (`NAME` or `NAME:PARAMETERS`).
    pub fn create(&self, spec: &str) -> MoptResult<Box<dyn StopRule>> {
        let (name, parameters) = spec.split_once(':').unwrap_or((spec, ""));
        let key = name.trim().to_ascii_lowercase();
        let factory = self
            .entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, f)| f)
            .ok_or_else(|| {
                MoptError::parse(format!(
                    "unknown stop rule '{name}' of the mapper (available: {})",
                    self.names().join(", ")
                ))
            })?;
        factory(parameters)
    }
}

impl Default for StopRuleRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::workflow::explore::TrialSummary;

    #[test]
    fn rules_are_made_from_their_names() {
        let reg = StopRuleRegistry::default();
        let mut rule = reg.create("nothing_new:2").unwrap();
        let s = |trial, found_new| TrialSummary {
            trial,
            energies: &[],
            found_new,
            found_lower: false,
        };
        assert!(rule.after_trial(&s(0, false)).is_none());
        assert!(rule.after_trial(&s(1, false)).is_some());
        assert_eq!(reg.create("max_jobs:3").unwrap().name(), "max_trials");
        assert_eq!(
            reg.create("rank_unchanged:2,5").unwrap().name(),
            "rank_unchanged"
        );
        assert!(reg.create("rank_unchanged:2").is_err());
        assert!(reg.create("nothing_new:x").is_err());
        assert!(reg.create("never").is_err());
    }
}
