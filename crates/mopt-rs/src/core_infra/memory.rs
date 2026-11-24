//! The memory a dense matrix may take: what the machine has free, so that a part that would
//! build a matrix larger than that refuses it (or takes a route without it) instead of being
//! stopped by the system.
//!
//! The free memory is `MemAvailable` of `/proc/meminfo` (Linux: free memory plus the caches
//! that can be dropped). Elsewhere it is not known, and the budget is [`UNKNOWN_BUDGET`].
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

/// One gibibyte.
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The share of the free memory a matrix may take: the rest is for the calculator that
/// makes it (an analytic Hessian builds its own intermediates) and for the other parts of
/// the run.
pub const FREE_SHARE: f64 = 0.5;

/// The budget where the free memory is not known (bytes): 2 GiB.
pub const UNKNOWN_BUDGET: f64 = 2.0 * GIB;

/// The free memory of the machine (bytes), where it is known.
pub fn available_bytes() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    available_in(&text)
}

/// `MemAvailable` of the text of `/proc/meminfo` (bytes).
fn available_in(text: &str) -> Option<f64> {
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let mut words = line.split_whitespace().skip(1);
    let value: f64 = words.next()?.parse().ok()?;
    let scale = match words.next().map(str::to_ascii_lowercase).as_deref() {
        Some("kb") | None => 1024.0,
        Some("mb") => 1024.0 * 1024.0,
        Some("gb") => GIB,
        Some("b") => 1.0,
        Some(_) => return None,
    };
    Some(value * scale)
}

/// Where a budget comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetSource {
    /// Given (bytes).
    Given,
    /// [`FREE_SHARE`] of the free memory.
    Free,
    /// The free memory is not known: [`UNKNOWN_BUDGET`].
    Unknown,
}

/// The memory (bytes) a matrix may take: `given` (GiB) where it is given, otherwise
/// [`FREE_SHARE`] of the free memory, or [`UNKNOWN_BUDGET`] where that is not known.
pub fn budget_bytes(given: Option<f64>) -> (f64, BudgetSource) {
    if let Some(g) = given {
        return (g * GIB, BudgetSource::Given);
    }
    match available_bytes() {
        Some(free) => (FREE_SHARE * free, BudgetSource::Free),
        None => (UNKNOWN_BUDGET, BudgetSource::Unknown),
    }
}

/// How a budget is told in a message: `3.2 GiB (half of the free memory)`.
pub fn describe(bytes: f64, source: BudgetSource) -> String {
    let why = match source {
        BudgetSource::Given => "given".to_string(),
        BudgetSource::Free => format!("{:.0}% of the free memory", 100.0 * FREE_SHARE),
        BudgetSource::Unknown => "the free memory of this machine is not known".to_string(),
    };
    format!("{:.2} GiB ({why})", bytes / GIB)
}

/// The bytes of dense matrices of `f64` with `entries` entries in all.
pub fn f64_bytes(entries: f64) -> f64 {
    8.0 * entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_free_memory_is_read_from_meminfo() {
        let text = "MemTotal:       32000000 kB\nMemFree:         1000000 kB\n\
                    MemAvailable:   20000000 kB\nBuffers:          100 kB\n";
        assert_eq!(available_in(text), Some(20_000_000.0 * 1024.0));
        assert_eq!(available_in("MemTotal: 1 kB\n"), None);
        let (bytes, source) = budget_bytes(Some(1.5));
        assert_eq!(source, BudgetSource::Given);
        assert!((bytes - 1.5 * GIB).abs() < 1.0);
        assert!(describe(bytes, source).starts_with("1.50 GiB"));
    }
}
