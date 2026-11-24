//! What a network document ([`super::network_json`]) has beside its nodes and edges: the path
//! peaks (`path_peaks`: records, no part of the network), the entrance channel (`entrance`),
//! what the user changed while the run went on (`steering`), and the scan of a barrierless
//! dissociation (`scan` of a dissociation edge). Each is written only where there is one, so
//! that the documents of other runs stay as they were, and a document without them (an earlier
//! mopt-rs) has none.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use faer::Col;

use super::entrance::Entrance;
use super::network::{Level, Network};
use super::network_json::{
    as_map, count, f64_of, f64s, field, ids, num, numbers, object, opt, opt_f64, opt_usize, str_of,
    text, usize_of, usizes,
};
use super::peak::{PathPeak, PeakBasis};
use super::scan::DissociationScan;
use super::steer::{AppliedLine, Steering};
use super::Configuration;
use crate::core_infra::json::JsonValue;
use crate::domain::{Element, MolecularSystem, SpinCharge};
use crate::error::{MoptError, MoptResult};

type Map = BTreeMap<String, JsonValue>;

fn peak_json(p: &PathPeak) -> JsonValue {
    object(vec![
        ("id", count(p.id)),
        ("label", text(&p.label())),
        ("level", count(p.level.number() as usize)),
        ("energy_hartree", num(p.energy)),
        (
            "ends",
            JsonValue::Array(p.ends.iter().map(|e| opt(*e, count)).collect()),
        ),
        (
            "barriers_hartree",
            JsonValue::Array(p.barriers.iter().map(|b| opt(*b, num)).collect()),
        ),
        ("method", text(&p.basis.method)),
        ("points", count(p.basis.points)),
        ("converged", JsonValue::Bool(p.basis.converged)),
        ("searches", count(p.basis.searches)),
        ("job", opt(p.job, count)),
        match &p.system {
            Configuration::Structure(s) => (
                "coords_bohr",
                numbers(&s.coords_flat_bohr().iter().copied().collect::<Vec<_>>()),
            ),
            Configuration::Variables(v) => (
                "variables",
                numbers(&v.values.iter().copied().collect::<Vec<_>>()),
            ),
        },
        match &p.system {
            Configuration::Structure(s) => (
                "state",
                object(vec![
                    ("charge", num(f64::from(s.spin_charge.charge))),
                    ("multiplicity", num(f64::from(s.spin_charge.multiplicity))),
                ]),
            ),
            Configuration::Variables(_) => ("state", JsonValue::Null),
        },
    ])
}

fn peak_of(
    v: &JsonValue,
    k: usize,
    elements: &[Element],
    names: Option<&Arc<Vec<String>>>,
    node: &dyn Fn(usize) -> Option<usize>,
) -> MoptResult<PathPeak> {
    let what = format!("path peak {k}");
    let m = as_map(v, &what)?;
    let system = match names {
        Some(names) => {
            let values = f64s(field(m, "variables", &what)?, &what)?;
            Configuration::variables(Col::from_fn(values.len(), |i| values[i]), Arc::clone(names))?
        }
        None => {
            let coords = f64s(field(m, "coords_bohr", &what)?, &what)?;
            let state = as_map(field(m, "state", &what)?, &what)?;
            Configuration::Structure(MolecularSystem::from_flat_bohr(
                elements.to_vec(),
                &Col::from_fn(coords.len(), |i| coords[i]),
                SpinCharge::checked(
                    f64_of(state, "charge", &what)? as i64,
                    f64_of(state, "multiplicity", &what)? as i64,
                )?,
                None,
            )?)
        }
    };
    let pair = |key: &str| -> Vec<JsonValue> {
        m.get(key)
            .and_then(JsonValue::as_array)
            .map(<[JsonValue]>::to_vec)
            .unwrap_or_default()
    };
    let ends = pair("ends");
    let barriers = pair("barriers_hartree");
    let end = |k: usize| ends.get(k).and_then(JsonValue::as_usize).and_then(node);
    let barrier = |k: usize| barriers.get(k).and_then(JsonValue::as_f64);
    Ok(PathPeak {
        id: k,
        level: Level::from_number(usize_of(m, "level", &what)? as u64)?,
        system,
        energy: f64_of(m, "energy_hartree", &what)?,
        ends: [end(0), end(1)],
        barriers: [barrier(0), barrier(1)],
        basis: PeakBasis {
            method: str_of(m, "method", &what)?,
            points: usize_of(m, "points", &what)?,
            converged: m.get("converged").and_then(JsonValue::as_bool) == Some(true),
            searches: opt_usize(m, "searches").unwrap_or(0),
        },
        job: opt_usize(m, "job"),
    })
}

/// The scan of a barrierless dissociation as JSON.
pub fn scan_json(s: &DissociationScan) -> JsonValue {
    object(vec![
        ("distances_bohr", numbers(&s.distances)),
        ("energies_hartree", numbers(&s.energies)),
        ("highest_above_channel_hartree", opt(s.highest, num)),
        ("barrier", JsonValue::Bool(s.barrier)),
        ("separated", JsonValue::Bool(s.separated)),
    ])
}

/// The scan of the dissociation edge `m`, where it has one.
pub fn scan_of(m: &Map, what: &str) -> MoptResult<Option<DissociationScan>> {
    let Some(s) = m.get("scan").and_then(JsonValue::as_object) else {
        return Ok(None);
    };
    Ok(Some(DissociationScan {
        distances: f64s(field(s, "distances_bohr", what)?, what)?,
        energies: f64s(field(s, "energies_hartree", what)?, what)?,
        highest: opt_f64(s, "highest_above_channel_hartree"),
        barrier: s.get("barrier").and_then(JsonValue::as_bool) == Some(true),
        separated: s.get("separated").and_then(JsonValue::as_bool) == Some(true),
    }))
}

/// The entries of the document for what `network` has of them.
pub fn extras_json(network: &Network) -> Vec<(&'static str, JsonValue)> {
    let mut entries = Vec::new();
    if !network.path_peaks.is_empty() {
        entries.push((
            "path_peaks",
            JsonValue::Array(network.path_peaks.iter().map(peak_json).collect()),
        ));
    }
    if let Some(e) = &network.entrance {
        entries.push((
            "entrance",
            object(vec![
                ("channel", count(e.channel)),
                ("structures", count(e.structures)),
                ("inputs", ids(&e.inputs)),
            ]),
        ));
    }
    if !network.steering.is_empty() || !network.steering.first.is_empty() {
        let s = &network.steering;
        entries.push((
            "steering",
            object(vec![
                ("explore_first", ids(&s.first)),
                ("max_jobs", opt(s.max_jobs, count)),
                (
                    "applied",
                    JsonValue::Array(
                        s.applied
                            .iter()
                            .map(|a| {
                                object(vec![
                                    ("after_jobs", count(a.after_jobs)),
                                    ("line", text(&a.line)),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ));
    }
    entries
}

/// Read what the document `m` has of them into `network`; `node` gives the node of the network
/// for a node of the document (`None`: it was left out). With `notes`, what cannot be read is
/// left out with a note instead of an error.
pub fn read_extras(
    m: &Map,
    network: &mut Network,
    names: Option<&Arc<Vec<String>>>,
    node: &dyn Fn(usize) -> Option<usize>,
    mut notes: Option<&mut Vec<String>>,
) -> MoptResult<()> {
    let elements = network.elements.clone();
    if let Some(items) = m.get("path_peaks").and_then(JsonValue::as_array) {
        for (k, v) in items.iter().enumerate() {
            match peak_of(v, network.path_peaks.len(), &elements, names, node) {
                Ok(p) => network.path_peaks.push(p),
                Err(e) => match notes.as_deref_mut() {
                    Some(notes) => notes.push(format!("path peak {k} is left out: {e}")),
                    None => return Err(e),
                },
            }
        }
    }
    if let Some(e) = m.get("entrance").and_then(JsonValue::as_object) {
        let what = "the entrance";
        let read = || -> MoptResult<Option<Entrance>> {
            let Some(channel) = node(usize_of(e, "channel", what)?) else {
                return Ok(None);
            };
            Ok(Some(Entrance {
                channel,
                structures: usize_of(e, "structures", what)?,
                inputs: usizes(field(e, "inputs", what)?, what)?,
            }))
        };
        match read() {
            Ok(entrance) => network.entrance = entrance,
            Err(e) => match notes.as_deref_mut() {
                Some(notes) => notes.push(format!("the entrance is left out: {e}")),
                None => return Err(e),
            },
        }
    }
    if let Some(s) = m.get("steering").and_then(JsonValue::as_object) {
        let what = "the steering";
        let read = || -> MoptResult<Steering> {
            let mut applied = Vec::new();
            if let Some(items) = s.get("applied").and_then(JsonValue::as_array) {
                for a in items {
                    let a = as_map(a, what)?;
                    applied.push(AppliedLine {
                        after_jobs: usize_of(a, "after_jobs", what)?,
                        line: str_of(a, "line", what)?,
                    });
                }
            }
            Ok(Steering {
                first: match s.get("explore_first") {
                    Some(v) => usizes(v, what)?.into_iter().filter_map(node).collect(),
                    None => Vec::new(),
                },
                max_jobs: opt_usize(s, "max_jobs"),
                applied,
            })
        };
        match read() {
            Ok(steering) => network.steering = steering,
            Err(e) => match notes {
                Some(notes) => notes.push(format!("the steering is left out: {e}")),
                None => return Err(MoptError::parse(format!("{what}: {e}"))),
            },
        }
    }
    Ok(())
}
