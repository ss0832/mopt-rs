//! The network as an SQLite database (`store=sqlite`): `reaction_network.sqlite` in the run
//! directory, written whole after every finished job next to `reaction_network.json`, and read
//! back for a report (`run_mapper --report FILE.sqlite`). It opens in any SQLite tool (the
//! `sqlite3` module of Python, the `sqlite3` shell, DB Browser) and is written without a
//! library ([`crate::core_infra::sqlite`]).
//!
//! The tables are those of the JSON document of the network ([`super::network_json`]) in
//! columns that can be queried, and each row keeps the whole entry as JSON (`json`) so that the
//! network is read back exactly; coordinates are a column of their own as little-endian 64-bit
//! floats (`coords_bohr`, x1 y1 z1 x2 ... in Bohr; in Python `numpy.frombuffer(blob, "<f8")`)
//! and the energies along the IRCs are rows of `irc_points`. Layout version 1:
//!
//! | Table | Columns |
//! |---|---|
//! | `format` | `key`, `value`: format, version, mopt_rs_version, units, run (JSON), and, where the network has them, entrance (JSON: the entrance channel), steering (JSON: what `steer.txt` changed), variables (JSON: the names of the variables of a network of plain variables) and whole_elements (JSON: the elements of the whole system of a network of ONIOM cores) |
//! | `elements` | `id` (atom, from 0), `symbol` |
//! | `states` | `id` (from 0), `name` (`PES1`, ...), `charge`, `multiplicity`, `excited_state`, `json`: the electronic states of a run of several (`states`; none for one) |
//! | `inputs` | `id`, `file`, `frame`, `node`, `goal` (1 for the goal of a goal-directed search) |
//! | `nodes` | `id`, `kind`, `number`, `label`, `level`, `energy_hartree`, `free_energy_hartree`, `charge`, `multiplicity`, `excited_state`, `input`, `job`, `from_node`, `excluded`, `coords_bohr`, `json` |
//! | `ts_edges` | `id`, `ts`, `end_1`, `end_2`, `level`, `barrier_1_hartree`, `barrier_2_hartree`, `imaginary_wavenumber_cm1`, `job`, `irc_file`, `json` |
//! | `irc_points` | `id`, `edge`, `side`, `k`, `arc_length`, `energy_hartree` (and `irc_sides`: `edge`, `side`, `end`) |
//! | `irc_transverse` | `id`, `edge`, `side`, `k`, `arc_length`, `wavenumber_cm1`: the lowest curvature across the path at the exact Hessians of a side of an IRC (every `irc_hessian_interval`-th point) |
//! | `path_peaks` | `id`, `label`, `level`, `energy_hartree`, `end_1`, `end_2`, `barrier_1_hartree`, `barrier_2_hartree`, `method`, `points`, `converged`, `searches`, `job`, `json` (with the structure): the path peaks, records beside the network (`path_peaks`) |
//! | `crossing_edges` | `id`, `crossing`, `end_1`, `end_2`, `level`, `barrier_1_hartree`, `barrier_2_hartree`, `gradient_difference`, `job`, `json`, `gap_hartree` (the first state minus the second, in the order of the states), `seam_imaginary_modes` (imaginary modes within the seam; empty where not checked), `seam_check` (`checked`, or why not) |
//! | `open_ts_edges` | `id`, `ts`, `end_1`, `end_2` (empty where the end is unknown), `level`, `barrier_1_hartree`, `barrier_2_hartree`, `imaginary_wavenumber_cm1`, `job`, `json` (with the IRC): the transition states with an end of their IRC that is not known, left out of the kinetics |
//! | `dissociation_edges` | `id`, `minimum` (the complex), `channel` (the DC node), `level`, `energy_difference_hartree`, `job`, `json` (with the scan of `confirm_dissociations`, where there is one) |
//! | `saddle_edges` | `id`, `saddle` (the `kSP` node), `mode`, `end_1`, `end_2` (from `+` and `-`), `level`, `wavenumber_cm1`, `job`, `json` |
//! | `fragments` | `id`, `key`, `formula`, `charge`, `multiplicity`, `level`, `energy_hartree`, `json` (with the structure): the fragments of the channels, each optimized on its own |
//! | `jobs` | `id`, `node`, `method`, `task`, `description`, `status`, `error`, `found_new`, `found_lower`, `trajectory`, `json` |
//! | `warnings` | `id`, `job`, `message` |
//! | `evaluations` | `id`, `energies`, `gradients`, `hessians` |
//!
//! Ids are those of the network (from 0); `ts`, `end_1`, `node`, ... are ids of `nodes`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::Path;

use super::network::Network;
use super::network_json::{from_json_lenient, to_json, NETWORK_FORMAT};
use super::store::{write_whole, NetworkStore};
use crate::core_infra::json::JsonValue;
use crate::core_infra::sqlite::{write_database, Database, Table, Value};
use crate::error::{MoptError, MoptResult, ResultExt};

/// The database file in the run directory.
pub const SQLITE_FILE: &str = "reaction_network.sqlite";
/// The layout version of the database.
pub const SQLITE_LAYOUT_VERSION: i64 = 1;

/// The database of the run (`store=sqlite`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SqliteStore;

impl NetworkStore for SqliteStore {
    fn name(&self) -> &str {
        "sqlite"
    }

    fn write(&self, network: &Network, run: &JsonValue, dir: &Path) -> MoptResult<()> {
        let bytes = write_database(&tables(network, run))?;
        write_whole(&dir.join(SQLITE_FILE), &bytes)
    }
}

fn text(v: Option<&JsonValue>) -> Value {
    match v {
        Some(JsonValue::String(s)) => Value::Text(s.clone()),
        Some(JsonValue::Number(x)) if x.fract() == 0.0 && x.abs() < 9.0e15 => {
            Value::Integer(*x as i64)
        }
        Some(JsonValue::Number(x)) => Value::Real(*x),
        Some(JsonValue::Bool(b)) => Value::Integer(i64::from(*b)),
        Some(JsonValue::Null) | None => Value::Null,
        Some(other) => Value::Text(other.to_string_compact()),
    }
}

fn real(v: Option<&JsonValue>) -> Value {
    v.and_then(JsonValue::as_f64)
        .map_or(Value::Null, Value::Real)
}

fn at<'a>(v: &'a JsonValue, path: &[&str]) -> Option<&'a JsonValue> {
    path.iter().try_fold(v, |v, k| v.get(k))
}

fn item<'a>(v: &'a JsonValue, key: &str, k: usize) -> Option<&'a JsonValue> {
    v.get(key)
        .and_then(JsonValue::as_array)
        .and_then(|a| a.get(k))
}

fn rowid(v: &JsonValue, k: usize) -> i64 {
    v.get("id")
        .and_then(JsonValue::as_usize)
        .map_or(k as i64, |i| i as i64)
}

fn table(name: &str, columns: &str, rows: Vec<(i64, Vec<Value>)>) -> Table {
    Table {
        name: name.to_string(),
        sql: format!("CREATE TABLE {name}(id INTEGER PRIMARY KEY, {columns})"),
        rows,
    }
}

fn list(doc: &JsonValue, key: &str) -> Vec<JsonValue> {
    doc.get(key)
        .and_then(JsonValue::as_array)
        .map(<[JsonValue]>::to_vec)
        .unwrap_or_default()
}

/// The tables of `network` with the state of its run `run`.
pub fn tables(network: &Network, run: &JsonValue) -> Vec<Table> {
    let doc = to_json(network, run.clone());
    let mut format = vec![
        ("format", Value::Text(format!("{NETWORK_FORMAT} (SQLite)"))),
        ("version", Value::Integer(SQLITE_LAYOUT_VERSION)),
        ("mopt_rs_version", Value::Text(crate::VERSION.to_string())),
        ("run", Value::Text(run.to_string_compact())),
    ];
    for key in ["entrance", "steering", "variables", "whole_elements"] {
        if let Some(v) = doc.get(key) {
            format.push((key, Value::Text(v.to_string_compact())));
        }
    }
    if let Some(JsonValue::Object(units)) = doc.get("units") {
        for (k, v) in units {
            let key: &'static str = match k.as_str() {
                "energy" => "unit_energy",
                "length" => "unit_length",
                "wavenumber" => "unit_wavenumber",
                "arc_length" => "unit_arc_length",
                _ => continue,
            };
            format.push((key, text(Some(v))));
        }
    }
    let format_rows = format
        .into_iter()
        .enumerate()
        .map(|(k, (key, value))| {
            (
                k as i64 + 1,
                vec![Value::Null, Value::Text(key.to_string()), value],
            )
        })
        .collect();
    let elements = list(&doc, "elements")
        .iter()
        .enumerate()
        .map(|(k, e)| (k as i64, vec![Value::Null, text(Some(e))]))
        .collect();
    let inputs = list(&doc, "inputs")
        .iter()
        .enumerate()
        .map(|(k, i)| {
            (
                k as i64,
                vec![
                    Value::Null,
                    text(i.get("file")),
                    text(i.get("frame")),
                    text(i.get("node")),
                    Value::Integer(i64::from(
                        i.get("goal").and_then(JsonValue::as_bool) == Some(true),
                    )),
                ],
            )
        })
        .collect();
    let mut nodes = Vec::new();
    for (k, n) in list(&doc, "nodes").into_iter().enumerate() {
        let coords: Vec<u8> = n
            .get("coords_bohr")
            .and_then(JsonValue::as_array)
            .map(|a| {
                a.iter()
                    .flat_map(|x| x.as_f64().unwrap_or(f64::NAN).to_le_bytes())
                    .collect()
            })
            .unwrap_or_default();
        let mut rest = n.clone();
        if let Some(m) = rest.as_object_mut() {
            m.remove("coords_bohr");
        }
        nodes.push((
            rowid(&n, k),
            vec![
                Value::Null,
                text(n.get("kind")),
                text(n.get("number")),
                text(n.get("label")),
                text(n.get("level")),
                real(n.get("energy_hartree")),
                real(n.get("free_energy_hartree")),
                text(at(&n, &["state", "charge"])),
                text(at(&n, &["state", "multiplicity"])),
                text(at(&n, &["state", "excited_state"])),
                text(at(&n, &["origin", "input"])),
                text(at(&n, &["origin", "job"])),
                text(at(&n, &["origin", "from_node"])),
                text(n.get("excluded")),
                Value::Blob(coords),
                Value::Text(rest.to_string_compact()),
            ],
        ));
    }
    let mut edges = Vec::new();
    let mut sides = Vec::new();
    let mut points = Vec::new();
    let mut transverse = Vec::new();
    for (k, e) in list(&doc, "ts_edges").into_iter().enumerate() {
        let id = rowid(&e, k);
        for (s, side) in at(&e, &["irc", "sides"])
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            sides.push((
                sides.len() as i64 + 1,
                vec![
                    Value::Null,
                    Value::Integer(id),
                    Value::Integer(s as i64),
                    text(side.get("name")),
                    text(side.get("end")),
                ],
            ));
            for (p, entry) in side
                .get("transverse_cm1")
                .and_then(JsonValue::as_array)
                .unwrap_or(&[])
                .iter()
                .enumerate()
            {
                let pair = entry.as_array().unwrap_or(&[]);
                transverse.push((
                    transverse.len() as i64 + 1,
                    vec![
                        Value::Null,
                        Value::Integer(id),
                        Value::Integer(s as i64),
                        Value::Integer(p as i64),
                        real(pair.first()),
                        real(pair.get(1)),
                    ],
                ));
            }
            let arcs = side.get("arc_lengths").and_then(JsonValue::as_array);
            let energies = side.get("energies_hartree").and_then(JsonValue::as_array);
            if let (Some(arcs), Some(energies)) = (arcs, energies) {
                for (p, (a, en)) in arcs.iter().zip(energies).enumerate() {
                    points.push((
                        points.len() as i64 + 1,
                        vec![
                            Value::Null,
                            Value::Integer(id),
                            Value::Integer(s as i64),
                            Value::Integer(p as i64),
                            real(Some(a)),
                            real(Some(en)),
                        ],
                    ));
                }
            }
        }
        let mut rest = e.clone();
        if let Some(JsonValue::Object(irc)) = rest.as_object_mut().and_then(|m| m.get_mut("irc")) {
            irc.remove("sides");
        }
        edges.push((
            id,
            vec![
                Value::Null,
                text(e.get("ts")),
                text(item(&e, "ends", 0)),
                text(item(&e, "ends", 1)),
                text(e.get("level")),
                real(item(&e, "barriers_hartree", 0)),
                real(item(&e, "barriers_hartree", 1)),
                real(e.get("imaginary_wavenumber_cm1")),
                text(e.get("job")),
                text(at(&e, &["irc", "file"])),
                Value::Text(rest.to_string_compact()),
            ],
        ));
    }
    let crossings = list(&doc, "crossing_edges")
        .iter()
        .enumerate()
        .map(|(k, e)| {
            (
                rowid(e, k),
                vec![
                    Value::Null,
                    text(e.get("crossing")),
                    text(item(e, "ends", 0)),
                    text(item(e, "ends", 1)),
                    text(e.get("level")),
                    real(item(e, "barriers_hartree", 0)),
                    real(item(e, "barriers_hartree", 1)),
                    real(e.get("gradient_difference")),
                    text(e.get("job")),
                    Value::Text(e.to_string_compact()),
                    // After `json`, so that a database of an earlier mopt-rs is read the same.
                    real(e.get("gap_hartree")),
                    text(e.get("seam_imaginary_modes")),
                    text(e.get("seam_check")),
                ],
            )
        })
        .collect();
    let opens = list(&doc, "open_ts_edges")
        .iter()
        .enumerate()
        .map(|(k, e)| {
            (
                rowid(e, k),
                vec![
                    Value::Null,
                    text(e.get("ts")),
                    text(item(e, "ends", 0)),
                    text(item(e, "ends", 1)),
                    text(e.get("level")),
                    real(item(e, "barriers_hartree", 0)),
                    real(item(e, "barriers_hartree", 1)),
                    real(e.get("imaginary_wavenumber_cm1")),
                    text(e.get("job")),
                    Value::Text(e.to_string_compact()),
                ],
            )
        })
        .collect();
    let fragments = list(&doc, "fragments")
        .iter()
        .enumerate()
        .map(|(k, f)| {
            (
                rowid(f, k),
                vec![
                    Value::Null,
                    text(f.get("key")),
                    text(f.get("formula")),
                    text(f.get("charge")),
                    text(f.get("multiplicity")),
                    text(f.get("level")),
                    real(f.get("energy_hartree")),
                    Value::Text(f.to_string_compact()),
                ],
            )
        })
        .collect();
    let states = list(&doc, "states")
        .iter()
        .enumerate()
        .map(|(k, s)| {
            (
                k as i64,
                vec![
                    Value::Null,
                    Value::Text(format!("PES{}", k + 1)),
                    text(s.get("charge")),
                    text(s.get("multiplicity")),
                    text(s.get("excited_state")),
                    Value::Text(s.to_string_compact()),
                ],
            )
        })
        .collect();
    let saddles = list(&doc, "saddle_edges")
        .iter()
        .enumerate()
        .map(|(k, e)| {
            let end = |i: usize| {
                text(
                    e.get("ends")
                        .and_then(|x| x.as_array())
                        .and_then(|a| a.get(i)),
                )
            };
            (
                rowid(e, k),
                vec![
                    Value::Null,
                    text(e.get("saddle")),
                    text(e.get("mode")),
                    end(0),
                    end(1),
                    text(e.get("level")),
                    real(e.get("wavenumber_cm1")),
                    text(e.get("job")),
                    Value::Text(e.to_string_compact()),
                ],
            )
        })
        .collect();
    let dissociations = list(&doc, "dissociation_edges")
        .iter()
        .enumerate()
        .map(|(k, e)| {
            (
                rowid(e, k),
                vec![
                    Value::Null,
                    text(e.get("minimum")),
                    text(e.get("channel")),
                    text(e.get("level")),
                    real(e.get("energy_difference_hartree")),
                    text(e.get("job")),
                    Value::Text(e.to_string_compact()),
                ],
            )
        })
        .collect();
    let jobs = list(&doc, "jobs")
        .iter()
        .enumerate()
        .map(|(k, j)| {
            (
                rowid(j, k),
                vec![
                    Value::Null,
                    text(j.get("node")),
                    text(j.get("method")),
                    text(j.get("task")),
                    text(j.get("description")),
                    text(j.get("status")),
                    text(j.get("error")),
                    text(j.get("found_new")),
                    text(j.get("found_lower")),
                    text(j.get("trajectory")),
                    Value::Text(j.to_string_compact()),
                ],
            )
        })
        .collect();
    let warnings = list(&doc, "warnings")
        .iter()
        .enumerate()
        .map(|(k, w)| {
            (
                k as i64 + 1,
                vec![Value::Null, text(w.get("job")), text(w.get("message"))],
            )
        })
        .collect();
    let peaks = list(&doc, "path_peaks")
        .iter()
        .enumerate()
        .map(|(k, p)| {
            (
                rowid(p, k),
                vec![
                    Value::Null,
                    text(p.get("label")),
                    text(p.get("level")),
                    real(p.get("energy_hartree")),
                    text(item(p, "ends", 0)),
                    text(item(p, "ends", 1)),
                    real(item(p, "barriers_hartree", 0)),
                    real(item(p, "barriers_hartree", 1)),
                    text(p.get("method")),
                    text(p.get("points")),
                    Value::Integer(i64::from(
                        p.get("converged").and_then(JsonValue::as_bool) == Some(true),
                    )),
                    text(p.get("searches")),
                    text(p.get("job")),
                    Value::Text(p.to_string_compact()),
                ],
            )
        })
        .collect();
    let e = doc.get("evaluations");
    let evaluations = vec![(
        1,
        vec![
            Value::Null,
            text(e.and_then(|e| e.get("energies"))),
            text(e.and_then(|e| e.get("gradients"))),
            text(e.and_then(|e| e.get("hessians"))),
        ],
    )];
    vec![
        table("format", "key TEXT, value", format_rows),
        table("elements", "symbol TEXT", elements),
        table(
            "states",
            "name TEXT, charge INTEGER, multiplicity INTEGER, excited_state INTEGER, json TEXT",
            states,
        ),
        table(
            "inputs",
            "file TEXT, frame INTEGER, node INTEGER, goal INTEGER",
            inputs,
        ),
        table(
            "nodes",
            "kind TEXT, number INTEGER, label TEXT, level INTEGER, energy_hartree REAL, \
             free_energy_hartree REAL, charge INTEGER, multiplicity INTEGER, excited_state \
             INTEGER, input INTEGER, job INTEGER, from_node INTEGER, excluded TEXT, coords_bohr \
             BLOB, json TEXT",
            nodes,
        ),
        table(
            "ts_edges",
            "ts INTEGER, end_1 INTEGER, end_2 INTEGER, level INTEGER, barrier_1_hartree REAL, \
             barrier_2_hartree REAL, imaginary_wavenumber_cm1 REAL, job INTEGER, irc_file TEXT, \
             json TEXT",
            edges,
        ),
        table(
            "irc_sides",
            "edge INTEGER, k INTEGER, side TEXT, end_reason TEXT",
            sides,
        ),
        table(
            "irc_points",
            "edge INTEGER, side INTEGER, k INTEGER, arc_length REAL, energy_hartree REAL",
            points,
        ),
        table(
            "irc_transverse",
            "edge INTEGER, side INTEGER, k INTEGER, arc_length REAL, wavenumber_cm1 REAL",
            transverse,
        ),
        table(
            "path_peaks",
            "label TEXT, level INTEGER, energy_hartree REAL, end_1 INTEGER, end_2 INTEGER, \
             barrier_1_hartree REAL, barrier_2_hartree REAL, method TEXT, points INTEGER, \
             converged INTEGER, searches INTEGER, job INTEGER, json TEXT",
            peaks,
        ),
        table(
            "crossing_edges",
            "crossing INTEGER, end_1 INTEGER, end_2 INTEGER, level INTEGER, barrier_1_hartree \
             REAL, barrier_2_hartree REAL, gradient_difference REAL, job INTEGER, json TEXT, \
             gap_hartree REAL, seam_imaginary_modes INTEGER, seam_check TEXT",
            crossings,
        ),
        table(
            "open_ts_edges",
            "ts INTEGER, end_1 INTEGER, end_2 INTEGER, level INTEGER, barrier_1_hartree REAL, \
             barrier_2_hartree REAL, imaginary_wavenumber_cm1 REAL, job INTEGER, json TEXT",
            opens,
        ),
        table(
            "dissociation_edges",
            "minimum INTEGER, channel INTEGER, level INTEGER, energy_difference_hartree REAL, \
             job INTEGER, json TEXT",
            dissociations,
        ),
        table(
            "saddle_edges",
            "saddle INTEGER, mode INTEGER, end_1 INTEGER, end_2 INTEGER, level INTEGER, \
             wavenumber_cm1 REAL, job INTEGER, json TEXT",
            saddles,
        ),
        table(
            "fragments",
            "key TEXT, formula TEXT, charge INTEGER, multiplicity INTEGER, level INTEGER, \
             energy_hartree REAL, json TEXT",
            fragments,
        ),
        table(
            "jobs",
            "node INTEGER, method TEXT, task TEXT, description TEXT, status TEXT, error TEXT, \
             found_new INTEGER, found_lower INTEGER, trajectory TEXT, json TEXT",
            jobs,
        ),
        table("warnings", "job INTEGER, message TEXT", warnings),
        table(
            "evaluations",
            "energies INTEGER, gradients INTEGER, hessians INTEGER",
            evaluations,
        ),
    ]
}

/// The JSON of a cell (`null` for a missing one).
fn json_of(v: Option<&Value>) -> JsonValue {
    match v {
        Some(Value::Integer(i)) => JsonValue::Number(*i as f64),
        Some(Value::Real(f)) => JsonValue::Number(*f),
        Some(Value::Text(s)) => JsonValue::String(s.clone()),
        _ => JsonValue::Null,
    }
}

/// The network of the database file `path` as far as it can be read, its `run` entry, and
/// notes of what was left out ([`from_json_lenient`] over the document made of the tables).
/// Fails when the file is not an SQLite database or has no `nodes` and `elements`.
pub fn read_sqlite(path: &Path) -> MoptResult<(Network, JsonValue, Vec<String>)> {
    let bytes = std::fs::read(path).map_err(|e| MoptError::from(e).context(path.display()))?;
    let db = Database::open(bytes).context(path.display())?;
    let mut notes = Vec::new();
    let rows = |name: &str, notes: &mut Vec<String>| -> Vec<(i64, Vec<Value>)> {
        match db.rows(name) {
            Ok(Some(r)) => r,
            Ok(None) => {
                notes.push(format!("the database has no table '{name}'"));
                Vec::new()
            }
            Err(e) => {
                notes.push(format!("the table '{name}' could not be read ({e})"));
                Vec::new()
            }
        }
    };
    let mut format: BTreeMap<String, Value> = BTreeMap::new();
    for (_, v) in rows("format", &mut notes) {
        if let (Some(Value::Text(k)), Some(value)) = (v.get(1), v.get(2)) {
            format.insert(k.clone(), value.clone());
        }
    }
    let version = format.get("version").and_then(Value::as_i64);
    if version != Some(SQLITE_LAYOUT_VERSION) {
        notes.push(format!(
            "the database has the layout version {}; this mopt-rs writes version \
             {SQLITE_LAYOUT_VERSION}, and the database is read as far as it goes",
            version.map_or_else(|| "(none)".to_string(), |v| v.to_string())
        ));
    }
    let run = format
        .get("run")
        .and_then(Value::as_str)
        .and_then(|t| JsonValue::parse(t).ok())
        .unwrap_or(JsonValue::Null);
    let elements: Vec<JsonValue> = rows("elements", &mut notes)
        .into_iter()
        .map(|(_, v)| json_of(v.get(1)))
        .collect();
    let inputs: Vec<JsonValue> = rows("inputs", &mut notes)
        .into_iter()
        .map(|(_, v)| {
            let mut m = BTreeMap::new();
            m.insert("file".to_string(), json_of(v.get(1)));
            m.insert("frame".to_string(), json_of(v.get(2)));
            m.insert("node".to_string(), json_of(v.get(3)));
            if matches!(v.get(4), Some(Value::Integer(1))) {
                m.insert("goal".to_string(), JsonValue::Bool(true));
            }
            JsonValue::Object(m)
        })
        .collect();
    // An entry whose JSON cannot be read is left out by the lenient reader with a note: it is
    // given as `null` (which is not an object) at its place.
    let entry = |text: Option<&Value>| {
        text.and_then(Value::as_str)
            .and_then(|t| JsonValue::parse(t).ok())
            .unwrap_or(JsonValue::Null)
    };
    let nodes: Vec<JsonValue> = rows("nodes", &mut notes)
        .into_iter()
        .map(|(_, v)| {
            let mut node = entry(v.get(15));
            let coords: Option<Vec<JsonValue>> = v.get(14).and_then(Value::as_blob).map(|b| {
                b.chunks_exact(8)
                    .map(|c| {
                        let mut a = [0u8; 8];
                        a.copy_from_slice(c);
                        JsonValue::Number(f64::from_le_bytes(a))
                    })
                    .collect()
            });
            if let (Some(m), Some(c)) = (node.as_object_mut(), coords) {
                m.insert("coords_bohr".to_string(), JsonValue::Array(c));
            }
            node
        })
        .collect();
    // The sides of the IRCs, then their points, by edge.
    let mut sides: BTreeMap<(i64, i64), (JsonValue, JsonValue, Vec<JsonValue>, Vec<JsonValue>)> =
        BTreeMap::new();
    for (_, v) in rows("irc_sides", &mut notes) {
        if let (Some(e), Some(k)) = (
            v.get(1).and_then(Value::as_i64),
            v.get(2).and_then(Value::as_i64),
        ) {
            sides.insert(
                (e, k),
                (json_of(v.get(3)), json_of(v.get(4)), Vec::new(), Vec::new()),
            );
        }
    }
    for (_, v) in rows("irc_points", &mut notes) {
        let key = (
            v.get(1).and_then(Value::as_i64).unwrap_or(-1),
            v.get(2).and_then(Value::as_i64).unwrap_or(-1),
        );
        if let Some(side) = sides.get_mut(&key) {
            side.2.push(json_of(v.get(4)));
            side.3.push(json_of(v.get(5)));
        }
    }
    // The curvatures across the paths (a database of an earlier mopt-rs has no such table).
    let mut across: BTreeMap<(i64, i64), Vec<JsonValue>> = BTreeMap::new();
    if let Ok(Some(found)) = db.rows("irc_transverse") {
        for (_, v) in found {
            let key = (
                v.get(1).and_then(Value::as_i64).unwrap_or(-1),
                v.get(2).and_then(Value::as_i64).unwrap_or(-1),
            );
            across
                .entry(key)
                .or_default()
                .push(JsonValue::Array(vec![json_of(v.get(4)), json_of(v.get(5))]));
        }
    }
    let edges: Vec<JsonValue> = rows("ts_edges", &mut notes)
        .into_iter()
        .map(|(id, v)| {
            let mut edge = entry(v.get(10));
            let these: Vec<JsonValue> = sides
                .range((id, i64::MIN)..=(id, i64::MAX))
                .map(|(key, (name, end, arcs, energies))| {
                    let mut m = BTreeMap::new();
                    if let Some(points) = across.get(key) {
                        m.insert(
                            "transverse_cm1".to_string(),
                            JsonValue::Array(points.clone()),
                        );
                    }
                    m.insert("name".to_string(), name.clone());
                    m.insert("end".to_string(), end.clone());
                    m.insert("arc_lengths".to_string(), JsonValue::Array(arcs.clone()));
                    m.insert(
                        "energies_hartree".to_string(),
                        JsonValue::Array(energies.clone()),
                    );
                    JsonValue::Object(m)
                })
                .collect();
            if let Some(JsonValue::Object(irc)) =
                edge.as_object_mut().and_then(|m| m.get_mut("irc"))
            {
                irc.insert("sides".to_string(), JsonValue::Array(these));
            }
            edge
        })
        .collect();
    let crossings: Vec<JsonValue> = rows("crossing_edges", &mut notes)
        .into_iter()
        .map(|(_, v)| entry(v.get(9)))
        .collect();
    // A database of an earlier mopt-rs has no channels.
    let quiet = |name: &str, column: usize| -> Vec<JsonValue> {
        match db.rows(name) {
            Ok(Some(r)) => r.into_iter().map(|(_, v)| entry(v.get(column))).collect(),
            _ => Vec::new(),
        }
    };
    let dissociations = quiet("dissociation_edges", 6);
    let saddles = quiet("saddle_edges", 8);
    let peaks = quiet("path_peaks", 13);
    let opens = quiet("open_ts_edges", 9);
    let states = quiet("states", 5);
    let fragments = quiet("fragments", 7);
    let jobs: Vec<JsonValue> = rows("jobs", &mut notes)
        .into_iter()
        .map(|(_, v)| entry(v.get(10)))
        .collect();
    let warnings: Vec<JsonValue> = rows("warnings", &mut notes)
        .into_iter()
        .map(|(_, v)| {
            let mut m = BTreeMap::new();
            m.insert("job".to_string(), json_of(v.get(1)));
            m.insert("message".to_string(), json_of(v.get(2)));
            JsonValue::Object(m)
        })
        .collect();
    let mut evaluations = BTreeMap::new();
    if let Some((_, v)) = rows("evaluations", &mut notes).first() {
        for (k, key) in ["energies", "gradients", "hessians"].iter().enumerate() {
            evaluations.insert((*key).to_string(), json_of(v.get(k + 1)));
        }
    }
    let mut doc = BTreeMap::new();
    doc.insert(
        "format".to_string(),
        JsonValue::String(NETWORK_FORMAT.to_string()),
    );
    doc.insert(
        "version".to_string(),
        JsonValue::Number(super::network::NETWORK_FORMAT_VERSION as f64),
    );
    doc.insert("elements".to_string(), JsonValue::Array(elements));
    doc.insert("states".to_string(), JsonValue::Array(states));
    doc.insert("inputs".to_string(), JsonValue::Array(inputs));
    doc.insert("nodes".to_string(), JsonValue::Array(nodes));
    doc.insert("ts_edges".to_string(), JsonValue::Array(edges));
    doc.insert("crossing_edges".to_string(), JsonValue::Array(crossings));
    doc.insert(
        "dissociation_edges".to_string(),
        JsonValue::Array(dissociations),
    );
    doc.insert("saddle_edges".to_string(), JsonValue::Array(saddles));
    if !peaks.is_empty() {
        doc.insert("path_peaks".to_string(), JsonValue::Array(peaks));
    }
    if !opens.is_empty() {
        doc.insert("open_ts_edges".to_string(), JsonValue::Array(opens));
    }
    for key in ["entrance", "steering", "variables", "whole_elements"] {
        if let Some(v) = format
            .get(key)
            .and_then(Value::as_str)
            .and_then(|t| JsonValue::parse(t).ok())
        {
            doc.insert(key.to_string(), v);
        }
    }
    doc.insert("fragments".to_string(), JsonValue::Array(fragments));
    doc.insert("jobs".to_string(), JsonValue::Array(jobs));
    doc.insert("warnings".to_string(), JsonValue::Array(warnings));
    doc.insert("evaluations".to_string(), JsonValue::Object(evaluations));
    doc.insert("run".to_string(), run);
    let (network, run, read_notes) =
        from_json_lenient(&JsonValue::Object(doc)).context(path.display())?;
    notes.extend(read_notes);
    Ok((network, run, notes))
}

#[cfg(test)]
#[path = "sqlite_store_tests.rs"]
mod tests;
