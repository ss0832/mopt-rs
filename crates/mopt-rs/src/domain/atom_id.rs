//! Typed atom indices.
//!
//! [`AtomId`] is a 0-based atom index. Command-line options and input files use 1-based atom
//! numbers; [`AtomId::from_one_based`] is the only conversion from them, so the two conventions
//! cannot be mixed up silently. Cartesian degrees of freedom of an atom are addressed with
//! [`AtomId::dof`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy uses 1-based atom numbers in its options
//!  and 0-based indices in code).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt;

use crate::error::{MoptError, MoptResult};

/// 0-based index of an atom in a structure.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct AtomId(usize);

impl AtomId {
    /// From a 0-based index.
    #[inline]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// From a 1-based atom number (as used on the command line and in input files).
    pub fn from_one_based(number: usize) -> MoptResult<Self> {
        number
            .checked_sub(1)
            .map(Self)
            .ok_or_else(|| MoptError::config("atom numbers are 1-based; got 0"))
    }

    /// Parse a 1-based atom number.
    pub fn parse_one_based(token: &str) -> MoptResult<Self> {
        let n = token
            .trim()
            .parse::<usize>()
            .map_err(|e| MoptError::parse(format!("invalid atom number '{token}': {e}")))?;
        Self::from_one_based(n)
    }

    /// 0-based index.
    #[inline]
    pub const fn index(self) -> usize {
        self.0
    }

    /// 1-based atom number.
    #[inline]
    pub const fn one_based(self) -> usize {
        self.0 + 1
    }

    /// Index of Cartesian component `c` (0 = x, 1 = y, 2 = z) in a flat `3N` vector.
    #[inline]
    pub const fn dof(self, c: usize) -> usize {
        3 * self.0 + c
    }

    /// Check that the atom exists in a structure of `n_atoms` atoms.
    pub fn check(self, n_atoms: usize) -> MoptResult<Self> {
        if self.0 < n_atoms {
            Ok(self)
        } else {
            Err(MoptError::config(format!(
                "atom {} does not exist (the structure has {n_atoms} atoms)",
                self.one_based()
            )))
        }
    }
}

impl fmt::Debug for AtomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomId({})", self.0)
    }
}

impl fmt::Display for AtomId {
    /// Displays the 1-based atom number, as users count atoms.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "atom {}", self.one_based())
    }
}

/// All atom ids of a structure with `n` atoms.
pub fn atom_ids(n: usize) -> impl ExactSizeIterator<Item = AtomId> + Clone {
    (0..n).map(AtomId)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_based_conversion() {
        assert_eq!(AtomId::from_one_based(1).unwrap(), AtomId::new(0));
        assert!(AtomId::from_one_based(0).is_err());
        assert_eq!(AtomId::parse_one_based(" 7 ").unwrap().index(), 6);
        assert_eq!(AtomId::new(2).dof(1), 7);
        assert!(AtomId::new(3).check(3).is_err());
        assert_eq!(AtomId::new(4).to_string(), "atom 5");
    }
}
