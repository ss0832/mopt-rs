//! The saddle points of higher order in the network document ([`super::network_json`]): the
//! descent edges (`saddle_edges`) of the `kSP` nodes. A document without them (an earlier
//! mopt-rs, or a run without `higher_saddles`) has none.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::network::{Level, SaddleEdge};
use super::network_json::{as_map, count, num, object, opt, opt_f64, opt_usize, usize_of};
use crate::core_infra::json::JsonValue;
use crate::error::MoptResult;

/// A descent edge as JSON.
pub fn saddle_edge_json(e: &SaddleEdge) -> JsonValue {
    object(vec![
        ("id", count(e.id)),
        ("level", count(e.level.number() as usize)),
        ("saddle", count(e.saddle)),
        ("mode", count(e.mode)),
        ("wavenumber_cm1", opt(e.wavenumber, num)),
        (
            "ends",
            JsonValue::Array(e.ends.iter().map(|x| opt(*x, count)).collect()),
        ),
        ("job", opt(e.job, count)),
    ])
}

/// The descent edge `k` of `v`.
pub fn saddle_edge_of(v: &JsonValue, k: usize) -> MoptResult<SaddleEdge> {
    let what = format!("saddle edge {k}");
    let m = as_map(v, &what)?;
    let ends: Vec<Option<usize>> = m
        .get("ends")
        .and_then(JsonValue::as_array)
        .map(|a| a.iter().map(JsonValue::as_usize).collect())
        .unwrap_or_default();
    Ok(SaddleEdge {
        id: usize_of(m, "id", &what)?,
        level: Level::from_number(usize_of(m, "level", &what)? as u64)?,
        saddle: usize_of(m, "saddle", &what)?,
        mode: usize_of(m, "mode", &what)?,
        wavenumber: opt_f64(m, "wavenumber_cm1"),
        ends: [
            ends.first().copied().flatten(),
            ends.get(1).copied().flatten(),
        ],
        job: opt_usize(m, "job"),
    })
}
