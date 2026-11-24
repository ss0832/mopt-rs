//! When a search that collects minima ends (extension point, next to the sampling priorities).
//!
//! After every trial (one sample, or one round of a search in rounds) the search tells its
//! [`StopRules`] what it knows; the first rule that gives a reason ends the search. Built in:
//!
//! - [`MaxTrials`]: a number of trials (MultiOptPy's `-ms`);
//! - [`RankUnchanged`]: the energies of the `rank` lowest minima did not change for more than
//!   `patience` trials in a row, counted once more than `rank` minima are known (MultiOptPy's
//!   `-nr` and `-nl`, as `conformation_search.py` counts them);
//! - [`NothingNew`]: no new minimum in `patience` trials in a row;
//! - [`NoLowerMinimum`]: a round found no minimum below the lowest one known before it (the
//!   end of the iterations of iMTD-GC, which start again from every lower conformer).
//!
//! MultiOptPy: the termination of `Entrypoints/conformation_search.py` (`-ms`, `-nr`, `-nl`,
//!  `no_update_count`); the trait and the other rules are new.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

/// What a search knows after a trial.
#[derive(Debug, Clone, Copy)]
pub struct TrialSummary<'a> {
    /// The trial (from 0).
    pub trial: usize,
    /// Energies of the known minima (Hartree), in the order they were found.
    pub energies: &'a [f64],
    /// Whether the trial found a new minimum.
    pub found_new: bool,
    /// Whether it found one below the lowest known before it.
    pub found_lower: bool,
}

/// A rule that ends a search.
pub trait StopRule: Send {
    /// Canonical name.
    fn name(&self) -> &str;

    /// The reason to stop after the trial `summary`, or `None` to go on.
    fn after_trial(&mut self, summary: &TrialSummary<'_>) -> Option<String>;

    /// A counter for the log (MultiOptPy's `no_update_count`), if the rule keeps one.
    fn counter(&self) -> Option<usize> {
        None
    }
}

/// A number of trials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxTrials {
    /// Trials of the search.
    pub max: usize,
}

impl StopRule for MaxTrials {
    fn name(&self) -> &str {
        "max_trials"
    }

    fn after_trial(&mut self, s: &TrialSummary<'_>) -> Option<String> {
        (s.trial + 1 >= self.max).then(|| "Max samples are reached.".to_string())
    }
}

/// The lowest `rank` energies unchanged for more than `patience` trials.
#[derive(Debug, Clone, PartialEq)]
pub struct RankUnchanged {
    /// Number of lowest minima compared (`-nr`).
    pub rank: usize,
    /// Trials without a change that end the search (`-nl`).
    pub patience: usize,
    previous: Option<Vec<f64>>,
    unchanged: usize,
}

impl RankUnchanged {
    /// MultiOptPy's rule with `-nr rank -nl patience`.
    pub fn new(rank: usize, patience: usize) -> Self {
        Self {
            rank,
            patience,
            previous: None,
            unchanged: 0,
        }
    }
}

impl StopRule for RankUnchanged {
    fn name(&self) -> &str {
        "rank_unchanged"
    }

    fn after_trial(&mut self, s: &TrialSummary<'_>) -> Option<String> {
        if s.energies.len() <= self.rank {
            return None;
        }
        let mut sorted = s.energies.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted.truncate(self.rank);
        if self.previous.as_ref() == Some(&sorted) {
            self.unchanged += 1;
        } else {
            self.unchanged = 0;
        }
        self.previous = Some(sorted);
        (self.unchanged > self.patience)
            .then(|| "The number of lowest energy conformers is not updated.".to_string())
    }

    fn counter(&self) -> Option<usize> {
        Some(self.unchanged)
    }
}

/// No new minimum for `patience` trials in a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NothingNew {
    /// Trials without a new minimum that end the search.
    pub patience: usize,
    since: usize,
}

impl NothingNew {
    /// End after `patience` trials in a row without a new minimum.
    pub fn new(patience: usize) -> Self {
        Self { patience, since: 0 }
    }
}

impl StopRule for NothingNew {
    fn name(&self) -> &str {
        "nothing_new"
    }

    fn after_trial(&mut self, s: &TrialSummary<'_>) -> Option<String> {
        self.since = if s.found_new { 0 } else { self.since + 1 };
        (self.since >= self.patience)
            .then(|| format!("No new minimum in {} trials.", self.patience))
    }

    fn counter(&self) -> Option<usize> {
        Some(self.since)
    }
}

/// A round without a minimum below the lowest one known before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoLowerMinimum;

impl StopRule for NoLowerMinimum {
    fn name(&self) -> &str {
        "no_lower_minimum"
    }

    fn after_trial(&mut self, s: &TrialSummary<'_>) -> Option<String> {
        (!s.found_lower).then(|| "No conformer below the lowest one was found.".to_string())
    }
}

/// Rules that end a search: the first reason wins.
#[derive(Default)]
pub struct StopRules {
    rules: Vec<Box<dyn StopRule>>,
}

impl StopRules {
    /// No rules (the search ends by itself).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `rule`.
    #[must_use]
    pub fn with(mut self, rule: Box<dyn StopRule>) -> Self {
        self.rules.push(rule);
        self
    }

    /// The reason to stop after `summary`; every rule sees every trial.
    pub fn after_trial(&mut self, summary: &TrialSummary<'_>) -> Option<String> {
        let mut reason = None;
        for rule in &mut self.rules {
            let r = rule.after_trial(summary);
            if reason.is_none() {
                reason = r;
            }
        }
        reason
    }

    /// The counter of the rule `name`, if it keeps one.
    pub fn counter(&self, name: &str) -> Option<usize> {
        self.rules
            .iter()
            .find(|r| r.name() == name)
            .and_then(|r| r.counter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(trial: usize, energies: &[f64], found_new: bool) -> TrialSummary<'_> {
        TrialSummary {
            trial,
            energies,
            found_new,
            found_lower: false,
        }
    }

    #[test]
    fn rank_unchanged_counts_as_multioptpy_does() {
        let mut rule = RankUnchanged::new(2, 1);
        // Two minima: no more than the rank, nothing is counted.
        assert!(rule.after_trial(&summary(0, &[-1.0, -2.0], true)).is_none());
        assert!(rule
            .after_trial(&summary(1, &[-1.0, -2.0, -0.5], true))
            .is_none());
        assert_eq!(rule.counter(), Some(0));
        assert!(rule
            .after_trial(&summary(2, &[-1.0, -2.0, -0.5], false))
            .is_none());
        assert_eq!(rule.counter(), Some(1));
        // More than `patience` unchanged trials: stop.
        assert!(rule
            .after_trial(&summary(3, &[-1.0, -2.0, -0.5], false))
            .is_some());
        // A new low minimum resets the count.
        let mut rule = RankUnchanged::new(2, 1);
        let _ = rule.after_trial(&summary(0, &[-1.0, -2.0, -0.5], true));
        let _ = rule.after_trial(&summary(1, &[-1.0, -2.0, -0.5], false));
        assert!(rule
            .after_trial(&summary(2, &[-1.0, -2.0, -0.5, -3.0], true))
            .is_none());
        assert_eq!(rule.counter(), Some(0));
    }

    #[test]
    fn the_first_reason_wins_and_every_rule_counts() {
        let mut rules = StopRules::new()
            .with(Box::new(MaxTrials { max: 3 }))
            .with(Box::new(NothingNew::new(2)));
        assert!(rules.after_trial(&summary(0, &[0.0], true)).is_none());
        assert!(rules.after_trial(&summary(1, &[0.0], false)).is_none());
        assert_eq!(rules.counter("nothing_new"), Some(1));
        let reason = rules
            .after_trial(&summary(2, &[0.0], false))
            .unwrap_or_default();
        assert!(reason.starts_with("Max samples"));
        assert_eq!(rules.counter("nothing_new"), Some(2));
        assert!(NoLowerMinimum
            .after_trial(&summary(0, &[0.0], true))
            .is_some());
    }
}
