//! Unified error and result types for `mopt-rs`.
//!
//! Implemented by hand (no `thiserror` / `anyhow`). [`MoptError::context`] prefixes a message
//! with where the failure happened (method, iteration, atom index, ...).
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy raises Python exceptions or prints and
//!  exits).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt;
use std::io;

/// Primary error enumeration across all `mopt-rs` subsystems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoptError {
    /// Filesystem or stream failure.
    Io(String),
    /// Malformed input (XYZ, GJF, MOL, MOL2, JSON, CLI arguments, ...).
    Parse(String),
    /// Invalid user configuration, atom index, or parameter value.
    Config(String),
    /// Linear algebra failure (singular matrix, non-convergent decomposition, NaN).
    Linalg(String),
    /// Potential energy surface or external calculator failure.
    Calculator(String),
    /// Coordinate transformation or constraint solver failure.
    Coordinate(String),
    /// Numerical failure in optimization, path search, or IRC.
    Convergence(String),
    /// Unregistered plugin or unsupported combination of features.
    Unsupported(String),
}

impl MoptError {
    /// Construct a configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Construct a parse error.
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }

    /// Construct a linear algebra error.
    pub fn linalg(msg: impl Into<String>) -> Self {
        Self::Linalg(msg.into())
    }

    /// Construct a calculator error.
    pub fn calc(msg: impl Into<String>) -> Self {
        Self::Calculator(msg.into())
    }

    /// Construct a coordinate/constraint error.
    pub fn coord(msg: impl Into<String>) -> Self {
        Self::Coordinate(msg.into())
    }

    /// Construct a convergence error.
    pub fn conv(msg: impl Into<String>) -> Self {
        Self::Convergence(msg.into())
    }

    /// Construct an unsupported-feature error.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    /// Prefix the message with `ctx` (e.g. `"RS-I-RFO iteration 12"`), keeping the error kind.
    #[must_use]
    pub fn context(self, ctx: impl fmt::Display) -> Self {
        let wrap = |msg: String| format!("{ctx}: {msg}");
        match self {
            Self::Io(m) => Self::Io(wrap(m)),
            Self::Parse(m) => Self::Parse(wrap(m)),
            Self::Config(m) => Self::Config(wrap(m)),
            Self::Linalg(m) => Self::Linalg(wrap(m)),
            Self::Calculator(m) => Self::Calculator(wrap(m)),
            Self::Coordinate(m) => Self::Coordinate(wrap(m)),
            Self::Convergence(m) => Self::Convergence(wrap(m)),
            Self::Unsupported(m) => Self::Unsupported(wrap(m)),
        }
    }
}

impl fmt::Display for MoptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Parse(msg) => write!(f, "Parse error: {msg}"),
            Self::Config(msg) => write!(f, "Configuration error: {msg}"),
            Self::Linalg(msg) => write!(f, "Linear algebra error: {msg}"),
            Self::Calculator(msg) => write!(f, "Calculator error: {msg}"),
            Self::Coordinate(msg) => write!(f, "Coordinate error: {msg}"),
            Self::Convergence(msg) => write!(f, "Convergence error: {msg}"),
            Self::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
        }
    }
}

impl std::error::Error for MoptError {}

impl From<io::Error> for MoptError {
    fn from(err: io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

/// Standard `Result` alias for `mopt-rs`.
pub type MoptResult<T> = Result<T, MoptError>;

/// Extension trait attaching context to a [`MoptResult`].
pub trait ResultExt<T> {
    /// Prefix the error message with `ctx` on failure.
    fn context(self, ctx: impl fmt::Display) -> MoptResult<T>;
}

impl<T> ResultExt<T> for MoptResult<T> {
    fn context(self, ctx: impl fmt::Display) -> MoptResult<T> {
        self.map_err(|e| e.context(ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_is_prefixed_and_kind_is_kept() {
        let err = MoptError::conv("gradient is NaN").context("RS-I-RFO iteration 3");
        assert_eq!(
            err,
            MoptError::Convergence("RS-I-RFO iteration 3: gradient is NaN".into())
        );
        assert_eq!(
            err.to_string(),
            "Convergence error: RS-I-RFO iteration 3: gradient is NaN"
        );
    }
}
