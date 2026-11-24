//! The dissociation channels in the network document ([`super::network_json`]): the
//! fragments (`fragments`), the parts of a channel node (`parts`), and the barrierless
//! dissociations (`dissociation_edges`). A document without them (an earlier mopt-rs) has
//! none.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::channels::{ChannelPart, DissociationEdge, FragmentRecord};
use super::network::Level;
use super::network_json::{
    as_map, count, f64_of, f64s, field, ids, list, num, numbers, object, opt, opt_f64, opt_usize,
    str_of, text, usize_of, usizes,
};
use crate::core_infra::json::JsonValue;
use crate::domain::{Element, MolecularSystem, SpinCharge};
use crate::error::{MoptError, MoptResult};

/// A fragment record as JSON (its key as 16 hexadecimal digits: a number of JSON would lose
/// bits of it).
pub fn fragment_json(f: &FragmentRecord) -> JsonValue {
    object(vec![
        ("id", count(f.id)),
        ("key", text(&format!("{:016x}", f.key))),
        ("formula", text(&f.formula)),
        ("charge", num(f64::from(f.charge))),
        ("multiplicity", num(f64::from(f.multiplicity))),
        ("level", count(f.level.number() as usize)),
        ("energy_hartree", num(f.energy)),
        ("frequencies_cm1", opt(f.frequencies.as_deref(), numbers)),
        (
            "elements",
            JsonValue::Array(f.system.elements.iter().map(|e| text(e.symbol())).collect()),
        ),
        (
            "coords_bohr",
            numbers(
                &f.system
                    .coords_flat_bohr()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
            ),
        ),
    ])
}

/// The fragment record `k` of `v`.
pub fn fragment_of(v: &JsonValue, k: usize) -> MoptResult<FragmentRecord> {
    let what = format!("fragment {k}");
    let m = as_map(v, &what)?;
    let key = u64::from_str_radix(&str_of(m, "key", &what)?, 16)
        .map_err(|_| MoptError::parse(format!("{what}: 'key' is not 16 hexadecimal digits")))?;
    let elements = list(m, "elements", &what)?
        .iter()
        .map(|e| {
            e.as_str()
                .ok_or_else(|| MoptError::parse(format!("{what}: an element is not a text")))
                .and_then(Element::parse_token)
        })
        .collect::<MoptResult<Vec<_>>>()?;
    let coords = f64s(field(m, "coords_bohr", &what)?, &what)?;
    let charge = f64_of(m, "charge", &what)? as i32;
    let multiplicity = usize_of(m, "multiplicity", &what)? as u32;
    let system = MolecularSystem::from_flat_bohr(
        elements,
        &Col::from_fn(coords.len(), |i| coords[i]),
        SpinCharge::checked(i64::from(charge), i64::from(multiplicity))?,
        None,
    )?;
    Ok(FragmentRecord {
        id: usize_of(m, "id", &what)?,
        key,
        formula: str_of(m, "formula", &what)?,
        charge,
        multiplicity,
        level: Level::from_number(usize_of(m, "level", &what)? as u64)?,
        system,
        energy: f64_of(m, "energy_hartree", &what)?,
        frequencies: match m.get("frequencies_cm1") {
            Some(JsonValue::Null) | None => None,
            Some(x) => Some(f64s(x, &what)?),
        },
    })
}

/// The parts of a channel node as JSON.
pub fn parts_json(parts: &[ChannelPart]) -> JsonValue {
    JsonValue::Array(
        parts
            .iter()
            .map(|p| {
                object(vec![
                    ("atoms", ids(&p.atoms)),
                    ("fragment", count(p.fragment)),
                ])
            })
            .collect(),
    )
}

/// The parts of a channel node (`what` names it).
pub fn parts_of(v: &JsonValue, what: &str) -> MoptResult<Vec<ChannelPart>> {
    v.as_array()
        .ok_or_else(|| MoptError::parse(format!("{what}: 'parts' is not a list")))?
        .iter()
        .map(|p| {
            let m = as_map(p, what)?;
            Ok(ChannelPart {
                atoms: usizes(field(m, "atoms", what)?, what)?,
                fragment: usize_of(m, "fragment", what)?,
            })
        })
        .collect()
}

/// A barrierless dissociation as JSON.
pub fn dissociation_edge_json(e: &DissociationEdge) -> JsonValue {
    let mut entries = vec![
        ("id", count(e.id)),
        ("level", count(e.level.number() as usize)),
        ("minimum", count(e.minimum)),
        ("channel", count(e.channel)),
        ("energy_difference_hartree", num(e.energy_difference)),
        ("job", opt(e.job, count)),
    ];
    // Only where the edge was scanned (`confirm_dissociations`).
    if let Some(s) = &e.scan {
        entries.push(("scan", super::extras_json::scan_json(s)));
    }
    object(entries)
}

/// The barrierless dissociation `k` of `v`.
pub fn dissociation_edge_of(v: &JsonValue, k: usize) -> MoptResult<DissociationEdge> {
    let what = format!("dissociation edge {k}");
    let m = as_map(v, &what)?;
    Ok(DissociationEdge {
        id: usize_of(m, "id", &what)?,
        level: Level::from_number(usize_of(m, "level", &what)? as u64)?,
        minimum: usize_of(m, "minimum", &what)?,
        channel: usize_of(m, "channel", &what)?,
        energy_difference: opt_f64(m, "energy_difference_hartree").unwrap_or(f64::NAN),
        job: opt_usize(m, "job"),
        scan: super::extras_json::scan_of(m, &what)?,
    })
}
