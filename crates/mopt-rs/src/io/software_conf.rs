//! Lazy parser for `software_path.conf` (`key::value` or `key=value`), never failing on missing files.
//!
//! MultiOptPy: `fileio.py:read_software_path`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Key-value external QM executable registry loaded on demand from `software_path.conf`.
#[derive(Debug, Clone, Default)]
pub struct SoftwarePathConf {
    /// `key -> path` entries in file order of appearance (sorted by key).
    pub entries: BTreeMap<String, String>,
}

impl SoftwarePathConf {
    /// Parse `software_path.conf` text (`orca::/opt/orca/orca` or `gaussian = g16`).
    pub fn parse_str(content: &str) -> Self {
        let mut entries = BTreeMap::new();
        for line in content.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = t.split_once("::").or_else(|| t.split_once('=')) {
                entries.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
        }
        Self { entries }
    }

    /// Load `software_path.conf` if it exists at `path`, otherwise return an empty map without error.
    pub fn load_optional(path: impl AsRef<Path>) -> Self {
        match fs::read_to_string(path) {
            Ok(txt) => Self::parse_str(&txt),
            Err(_) => Self::default(),
        }
    }

    /// Retrieve a configured path by case-insensitive key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .get(&key.trim().to_ascii_lowercase())
            .map(String::as_str)
    }
}
