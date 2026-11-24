//! The integrators by name. This is the one place where the name of a method is turned
//! into an integrator; further ones are added with [`IntegratorRegistry::register`].
//!
//! MultiOptPy runs LQA for a name it does not know; here an unknown name is an error, so
//! that a misspelt method does not run another one (`docs/source/migration_paths.md`).
//!
//! MultiOptPy: `irc.py:IRC._get_irc_method_class`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use super::dvv::Dvv;
use super::euler::Euler;
use super::gs2::Gs2;
use super::hpc::Hpc;
use super::integrator::{BranchStart, Integrator};
use super::lqa::Lqa;
use super::modekill::ModeKill;
use super::rk4::Rk4;
use crate::error::{MoptError, MoptResult};

/// Builds an integrator for a side.
pub type IntegratorFactory =
    Arc<dyn Fn(BranchStart) -> MoptResult<Box<dyn Integrator>> + Send + Sync>;

/// Names of the integrators of MultiOptPy, in its order, and `gs2`.
pub const INTEGRATORS: &[&str] = &["lqa", "rk4", "modekill", "euler", "dvv", "hpc", "gs2"];

/// Integrators by name.
#[derive(Clone)]
pub struct IntegratorRegistry {
    factories: BTreeMap<String, IntegratorFactory>,
    aliases: BTreeMap<String, String>,
}

impl std::fmt::Debug for IntegratorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntegratorRegistry")
            .field("names", &self.names())
            .finish()
    }
}

impl IntegratorRegistry {
    /// The integrators of mopt-rs: `lqa`, `hpc`, `dvv`, `euler`, `rk4`, `modekill`, and
    /// `gs2` (also `gs`, `gonzalez_schlegel`).
    pub fn builtin() -> Self {
        let mut r = Self {
            factories: BTreeMap::new(),
            aliases: BTreeMap::new(),
        };
        r.register("lqa", Arc::new(|s| Ok(Box::new(Lqa::new(s)))));
        r.register("hpc", Arc::new(|s| Ok(Box::new(Hpc::new(s)))));
        r.register("dvv", Arc::new(|s| Ok(Box::new(Dvv::new(s)?))));
        r.register("euler", Arc::new(|s| Ok(Box::new(Euler::new(s)))));
        r.register("rk4", Arc::new(|s| Ok(Box::new(Rk4::new(s)))));
        r.register("gs2", Arc::new(|s| Ok(Box::new(Gs2::new(s)))));
        r.register("modekill", Arc::new(|s| Ok(Box::new(ModeKill::new(s)?))));
        r.alias("gs", "gs2");
        r.alias("gonzalez_schlegel", "gs2");
        r
    }

    /// Add (or replace) the integrator `name`.
    pub fn register(&mut self, name: &str, factory: IntegratorFactory) {
        self.factories.insert(name.to_ascii_lowercase(), factory);
    }

    /// Let `alias` stand for `target`.
    pub fn alias(&mut self, alias: &str, target: &str) {
        self.aliases
            .insert(alias.to_ascii_lowercase(), target.to_ascii_lowercase());
    }

    /// The names (without aliases), sorted.
    pub fn names(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// The registered name `name` stands for.
    pub fn resolve(&self, name: &str) -> MoptResult<String> {
        let key = name.to_ascii_lowercase();
        let key = self.aliases.get(&key).cloned().unwrap_or(key);
        if self.factories.contains_key(&key) {
            Ok(key)
        } else {
            Err(MoptError::config(format!(
                "unknown IRC method '{name}' (methods: {})",
                self.names().join(", ")
            )))
        }
    }

    /// The integrator `name` for a side from `start`.
    pub fn create(&self, name: &str, start: BranchStart) -> MoptResult<Box<dyn Integrator>> {
        let key = self.resolve(name)?;
        match self.factories.get(&key) {
            Some(factory) => factory(start),
            None => Err(MoptError::config(format!("unknown IRC method '{name}'"))),
        }
    }
}

impl Default for IntegratorRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_integrator_is_registered() {
        let r = IntegratorRegistry::builtin();
        let mut listed: Vec<String> = INTEGRATORS.iter().map(|s| (*s).to_string()).collect();
        listed.sort();
        assert_eq!(r.names(), listed);
        assert_eq!(r.resolve("GS").unwrap(), "gs2");
        assert_eq!(r.resolve("LQA").unwrap(), "lqa");
        assert!(r.resolve("lqaa").is_err());
    }
}
