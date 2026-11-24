//! Pages of the lists (`tools/list`, `resources/list`, `resources/templates/list`,
//! `prompts/list`, and `tasks/list` of `2025-11-25`): at most a page size of entries, and an
//! opaque `nextCursor` for the rest.
//!
//! A cursor is the offset of the next entry with a check value made from a key of this server
//! process, the list and the offset, so that the server only accepts cursors it gave out: any
//! other cursor (made up, changed, of another list or of another server process) is refused
//! with `-32602`, as the specification asks. Lists are read anew for every page; entries that
//! appear or go between two pages may shift what a page holds.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;

use super::protocol::RpcError;
use crate::core_infra::json::JsonValue;

/// The page size without `--page-size`.
pub const DEFAULT_PAGE_SIZE: usize = 100;
/// The largest page size.
pub const MAX_PAGE_SIZE: usize = 1000;

/// The pages of a server process.
#[derive(Debug, Clone)]
pub struct Pager {
    key: RandomState,
    size: usize,
}

impl Pager {
    /// Pages of `size` entries (1 to [`MAX_PAGE_SIZE`]).
    pub fn new(size: usize) -> Self {
        Self {
            key: RandomState::new(),
            size: size.clamp(1, MAX_PAGE_SIZE),
        }
    }

    /// The page size.
    pub fn size(&self) -> usize {
        self.size
    }

    fn check(&self, list: &str, offset: usize) -> u64 {
        self.key.hash_one((list, offset))
    }

    /// The cursor of the entries of `list` from `offset` on.
    pub fn cursor(&self, list: &str, offset: usize) -> String {
        format!("{offset:x}.{:016x}", self.check(list, offset))
    }

    /// The offset of the cursor `cursor` of `list` (`None` or `null`: the start).
    pub fn offset(&self, list: &str, cursor: Option<&JsonValue>) -> Result<usize, RpcError> {
        let refused = || RpcError::invalid_params(format!("invalid cursor for {list}"));
        let text = match cursor {
            None | Some(JsonValue::Null) => return Ok(0),
            Some(JsonValue::String(s)) => s.as_str(),
            Some(_) => return Err(refused()),
        };
        if text.len() > 64 {
            return Err(refused());
        }
        let (offset, check) = text.split_once('.').ok_or_else(refused)?;
        let offset = usize::from_str_radix(offset, 16).map_err(|_| refused())?;
        let check = u64::from_str_radix(check, 16).map_err(|_| refused())?;
        if offset == 0 || check != self.check(list, offset) || text != self.cursor(list, offset) {
            return Err(refused());
        }
        Ok(offset)
    }

    /// The page of `items` that `cursor` asks for, and the cursor of the rest (if any).
    pub fn page(
        &self,
        list: &str,
        items: Vec<JsonValue>,
        cursor: Option<&JsonValue>,
    ) -> Result<(Vec<JsonValue>, Option<String>), RpcError> {
        let offset = self.offset(list, cursor)?;
        if offset > items.len() {
            return Err(RpcError::invalid_params(format!(
                "invalid cursor for {list} (the list has become shorter)"
            )));
        }
        let end = (offset + self.size).min(items.len());
        let next = (end < items.len()).then(|| self.cursor(list, end));
        Ok((items[offset..end].to_vec(), next))
    }
}

impl Default for Pager {
    fn default() -> Self {
        Self::new(DEFAULT_PAGE_SIZE)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pages_follow_their_cursors_and_others_are_refused() {
        let pager = Pager::new(2);
        let items: Vec<JsonValue> = (0..5).map(|k| JsonValue::Number(f64::from(k))).collect();
        let (first, next) = pager.page("tools/list", items.clone(), None).unwrap();
        assert_eq!(first.len(), 2);
        let next = JsonValue::String(next.unwrap());
        let (second, next) = pager
            .page("tools/list", items.clone(), Some(&next))
            .unwrap();
        assert_eq!(second[0], JsonValue::Number(2.0));
        let next = JsonValue::String(next.unwrap());
        let (third, end) = pager
            .page("tools/list", items.clone(), Some(&next))
            .unwrap();
        assert_eq!((third.len(), end), (1, None));
        // Made up, changed, of another list, of another process, not a string.
        for bad in ["", "2", "2.0", "zz.1", "0.0", "-1.ffff"] {
            let c = JsonValue::String(bad.to_string());
            assert!(
                pager.page("tools/list", items.clone(), Some(&c)).is_err(),
                "{bad}"
            );
        }
        let other = JsonValue::String(pager.cursor("prompts/list", 2));
        assert!(pager
            .page("tools/list", items.clone(), Some(&other))
            .is_err());
        let foreign = JsonValue::String(Pager::new(2).cursor("tools/list", 2));
        assert!(pager
            .page("tools/list", items.clone(), Some(&foreign))
            .is_err());
        assert!(pager
            .page("tools/list", items, Some(&JsonValue::Number(2.0)))
            .is_err());
    }
}
