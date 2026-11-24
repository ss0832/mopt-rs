//! The network as a JSON document (`reaction_network.json`), with the version of its layout:
//! written after every finished job and read back by a restart, so that every number comes
//! back as it was (shortest round-trip decimal representations).
//!
//! Units: energies in Hartree, coordinates in Bohr, wavenumbers in cm^-1, arc lengths of the
//! IRC in amu^(1/2) Bohr.
//!
//! MultiOptPy: `Wrapper/mapper.py:NetworkGraph.save`, `NetworkGraph.load` (no version, energies
//! and file names only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::sync::Arc;

use faer::Col;

use super::channels_json::{
    dissociation_edge_json, dissociation_edge_of, fragment_json, fragment_of, parts_json, parts_of,
};
use super::configuration::Configuration;
use super::network::{
    CrossingEdge, CrossingKind, ElectronicState, EvaluationTotals, InputRecord, IrcPath, IrcSide,
    JobRecord, JobStatus, Level, Network, NetworkWarning, Node, NodeKind, OpenTsEdge, Origin,
    PossibleBifurcation, SeamCheck, TsEdge, NETWORK_FORMAT_VERSION,
};
use crate::core_infra::json::JsonValue;
use crate::domain::{Bohr, Element, MolecularSystem, PeriodicCell, SpinCharge, Vector3};
use crate::error::{MoptError, MoptResult};

/// The `format` entry of the document.
pub const NETWORK_FORMAT: &str = "mopt-rs reaction network";

type Map = BTreeMap<String, JsonValue>;

pub(super) fn num(x: f64) -> JsonValue {
    JsonValue::Number(x)
}

pub(super) fn count(n: usize) -> JsonValue {
    JsonValue::Number(n as f64)
}

pub(super) fn text(s: &str) -> JsonValue {
    JsonValue::String(s.to_string())
}

pub(super) fn opt<T>(v: Option<T>, f: impl Fn(T) -> JsonValue) -> JsonValue {
    v.map_or(JsonValue::Null, f)
}

pub(super) fn numbers(v: &[f64]) -> JsonValue {
    JsonValue::Array(v.iter().map(|x| num(*x)).collect())
}

pub(super) fn ids(v: &[usize]) -> JsonValue {
    JsonValue::Array(v.iter().map(|x| count(*x)).collect())
}

pub(super) fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

fn state_json(s: &ElectronicState) -> JsonValue {
    object(vec![
        ("charge", num(f64::from(s.charge))),
        ("multiplicity", num(f64::from(s.multiplicity))),
        ("excited_state", count(s.excited_state)),
    ])
}

fn node_json(n: &Node) -> JsonValue {
    let mut entries = vec![
        ("id", count(n.id)),
        ("kind", text(n.kind.tag())),
        ("number", count(n.number)),
        ("label", text(&n.label())),
        ("level", count(n.level.number() as usize)),
        ("energy_hartree", num(n.energy)),
        ("free_energy_hartree", opt(n.free_energy, num)),
        (
            "index_imaginary_cm1",
            opt(n.index_wavenumbers.as_deref(), numbers),
        ),
        ("frequencies_cm1", opt(n.frequencies.as_deref(), numbers)),
        ("state", state_json(&n.state)),
        (
            "origin",
            object(vec![
                ("input", opt(n.origin.input, count)),
                ("job", opt(n.origin.job, count)),
                ("from_node", opt(n.origin.from_node, count)),
            ]),
        ),
        ("excluded", opt(n.excluded.as_deref(), text)),
        // The followed states at the structure: `[number of the state, its place there]`.
        (
            "followed_states",
            JsonValue::Array(
                n.roots
                    .iter()
                    .map(|&(label, root)| JsonValue::Array(vec![count(label), count(root)]))
                    .collect(),
            ),
        ),
        match &n.system {
            Configuration::Structure(s) => (
                "coords_bohr",
                numbers(&s.coords_flat_bohr().iter().copied().collect::<Vec<_>>()),
            ),
            Configuration::Variables(p) => (
                "variables",
                numbers(&p.values.iter().copied().collect::<Vec<_>>()),
            ),
        },
    ];
    if let NodeKind::Crossing { states, .. } = &n.kind {
        entries.push((
            "states",
            JsonValue::Array(states.iter().map(state_json).collect()),
        ));
    }
    if n.kind == NodeKind::Dissociated {
        entries.push(("parts", parts_json(&n.parts)));
    }
    // Only where there are any, so that other files stay as they were.
    if let Some(bare) = n.bare_energy {
        entries.push(("bare_energy_hartree", num(bare)));
    }
    if let Some(cell) = n.system.structure().and_then(|s| s.cell.as_ref()) {
        entries.push(("cell", cell_json(cell)));
    }
    if let Some(whole) = n.whole.as_ref().and_then(Configuration::structure) {
        entries.push((
            "whole_coords_bohr",
            numbers(&whole.coords_flat_bohr().iter().copied().collect::<Vec<_>>()),
        ));
    }
    if !n.environments.is_empty() {
        entries.push((
            "environments",
            JsonValue::Array(
                n.environments
                    .iter()
                    .map(|e| {
                        object(vec![
                            ("energy_hartree", num(e.energy)),
                            ("job", opt(e.job, count)),
                        ])
                    })
                    .collect(),
            ),
        ));
    }
    object(entries)
}

/// A periodic cell: its lattice vectors `a`, `b`, `c` (Bohr) and which of them are periodic.
pub(super) fn cell_json(cell: &PeriodicCell) -> JsonValue {
    let vectors: Vec<f64> = (0..3)
        .flat_map(|k| (0..3).map(move |x| cell.h_bohr[x][k]))
        .collect();
    object(vec![
        ("lattice_bohr", numbers(&vectors)),
        (
            "periodic",
            JsonValue::Array(
                cell.periodic_axes
                    .iter()
                    .map(|&p| JsonValue::Bool(p))
                    .collect(),
            ),
        ),
    ])
}

/// The cell of [`cell_json`] in the entry `cell` of `m`, where there is one.
pub(super) fn cell_of(m: &Map, what: &str) -> MoptResult<Option<PeriodicCell>> {
    let Some(v) = m.get("cell").filter(|v| !matches!(v, JsonValue::Null)) else {
        return Ok(None);
    };
    let c = as_map(v, what)?;
    let l = f64s(field(c, "lattice_bohr", what)?, what)?;
    let [ax, ay, az, bx, by, bz, cx, cy, cz] = l[..] else {
        return Err(MoptError::parse(format!(
            "{what}: a cell has nine numbers (three lattice vectors)"
        )));
    };
    let flags = list(c, "periodic", what)?;
    let [pa, pb, pc] = flags else {
        return Err(MoptError::parse(format!(
            "{what}: a cell has three flags of periodicity"
        )));
    };
    let flag = |v: &JsonValue| {
        v.as_bool().ok_or_else(|| {
            MoptError::parse(format!(
                "{what}: a flag of periodicity is not true or false"
            ))
        })
    };
    PeriodicCell::from_lattice_vectors(
        Vector3::<Bohr>::new([ax, ay, az]),
        Vector3::<Bohr>::new([bx, by, bz]),
        Vector3::<Bohr>::new([cx, cy, cz]),
        [flag(pa)?, flag(pb)?, flag(pc)?],
    )
    .map(Some)
}

/// A side of an IRC.
fn side_json(s: &IrcSide) -> JsonValue {
    let mut entries = vec![
        ("name", text(&s.name)),
        ("end", text(&s.end)),
        ("arc_lengths", numbers(&s.arc_lengths)),
        ("energies_hartree", numbers(&s.energies)),
    ];
    // Only where there are any, so that other files stay as they were: the lowest curvature
    // across the path at the exact Hessians of the side, `[arc length, wavenumber]`.
    if !s.transverse.is_empty() {
        entries.push((
            "transverse_cm1",
            JsonValue::Array(s.transverse.iter().map(|p| numbers(p)).collect()),
        ));
    }
    object(entries)
}

/// The side of an IRC of the entry `s` of `what`.
fn side_of(s: &JsonValue, what: &str) -> MoptResult<IrcSide> {
    let s = as_map(s, what)?;
    Ok(IrcSide {
        name: str_of(s, "name", what)?,
        end: str_of(s, "end", what)?,
        arc_lengths: f64s(field(s, "arc_lengths", what)?, what)?,
        energies: f64s(field(s, "energies_hartree", what)?, what)?,
        transverse: match s.get("transverse_cm1").and_then(JsonValue::as_array) {
            Some(items) => items
                .iter()
                .map(|p| match f64s(p, what)?[..] {
                    [arc, wavenumber] => Ok([arc, wavenumber]),
                    _ => Err(MoptError::parse(format!(
                        "{what}: an entry of 'transverse_cm1' is not an arc length and a \
                         wavenumber"
                    ))),
                })
                .collect::<MoptResult<_>>()?,
            None => Vec::new(),
        },
    })
}

fn ts_edge_json(e: &TsEdge) -> JsonValue {
    object(vec![
        ("id", count(e.id)),
        ("level", count(e.level.number() as usize)),
        ("ts", count(e.ts)),
        ("ends", ids(&e.ends)),
        ("barriers_hartree", numbers(&e.barriers)),
        ("imaginary_wavenumber_cm1", opt(e.imaginary_wavenumber, num)),
        ("irc", irc_json(&e.irc)),
        ("job", opt(e.job, count)),
    ])
}

fn open_ts_edge_json(e: &OpenTsEdge) -> JsonValue {
    object(vec![
        ("id", count(e.id)),
        ("level", count(e.level.number() as usize)),
        ("ts", count(e.ts)),
        (
            "ends",
            JsonValue::Array(e.ends.iter().map(|x| opt(*x, count)).collect()),
        ),
        (
            "barriers_hartree",
            JsonValue::Array(e.barriers.iter().map(|x| opt(*x, num)).collect()),
        ),
        ("imaginary_wavenumber_cm1", opt(e.imaginary_wavenumber, num)),
        ("irc", irc_json(&e.irc)),
        ("job", opt(e.job, count)),
    ])
}

/// The IRC of a transition state: its sides, its file, and its possible bifurcations.
fn irc_json(path: &IrcPath) -> JsonValue {
    let sides = path.sides.iter().map(side_json).collect();
    let mut irc = vec![
        ("sides", JsonValue::Array(sides)),
        ("file", opt(path.file.as_deref(), text)),
    ];
    // Only where there are any, so that other files stay as they were.
    if !path.possible_bifurcations.is_empty() {
        irc.push((
            "possible_bifurcations",
            JsonValue::Array(
                path.possible_bifurcations
                    .iter()
                    .map(|b| {
                        let mut entries = vec![
                            ("side", text(&b.side)),
                            ("after_arc_length", opt(b.after, num)),
                            ("at_arc_length", num(b.at)),
                            ("wavenumber_cm1", num(b.wavenumber)),
                        ];
                        if !b.confirmed_ends.is_empty() {
                            entries.push(("confirmed_ends", ids(&b.confirmed_ends)));
                        }
                        if let Some(s) = b.inflection {
                            entries.push(("inflection_arc_length", num(s)));
                        }
                        object(entries)
                    })
                    .collect(),
            ),
        ));
    }
    object(irc)
}

fn crossing_edge_json(e: &CrossingEdge) -> JsonValue {
    let mut entries = crossing_edge_entries(e);
    // Only where there is a path, so that other files stay as they were.
    if !e.irc.sides.is_empty() || e.irc.file.is_some() {
        entries.push((
            "irc",
            object(vec![
                (
                    "sides",
                    JsonValue::Array(e.irc.sides.iter().map(side_json).collect()),
                ),
                ("file", opt(e.irc.file.as_deref(), text)),
            ]),
        ));
    }
    object(entries)
}

fn crossing_edge_entries(e: &CrossingEdge) -> Vec<(&'static str, JsonValue)> {
    vec![
        ("id", count(e.id)),
        ("level", count(e.level.number() as usize)),
        ("crossing", count(e.crossing)),
        (
            "ends",
            JsonValue::Array(e.ends.iter().map(|x| opt(*x, count)).collect()),
        ),
        (
            "barriers_hartree",
            JsonValue::Array(e.barriers.iter().map(|x| opt(*x, num)).collect()),
        ),
        ("gradient_difference", opt(e.gradient_difference, num)),
        ("gap_hartree", opt(e.gap, num)),
        (
            "seam_imaginary_modes",
            match &e.seam {
                Some(SeamCheck::Checked(n)) => count(*n),
                _ => JsonValue::Null,
            },
        ),
        (
            "seam_check",
            match &e.seam {
                Some(SeamCheck::Checked(_)) => text("checked"),
                Some(SeamCheck::NotChecked(why)) => text(why),
                None => JsonValue::Null,
            },
        ),
        ("job", opt(e.job, count)),
    ]
}

/// The check within the seam of a crossing edge as written by [`crossing_edge_entries`]
/// (`None` for a network written before it was kept).
fn seam_check_of(m: &Map) -> Option<SeamCheck> {
    let said = m.get("seam_check").and_then(JsonValue::as_str)?;
    Some(
        match m.get("seam_imaginary_modes").and_then(JsonValue::as_f64) {
            Some(n) if said == "checked" && n >= 0.0 && n.fract() == 0.0 => {
                SeamCheck::Checked(n as usize)
            }
            _ => SeamCheck::NotChecked(said.to_string()),
        },
    )
}

fn job_json(j: &JobRecord) -> JsonValue {
    let (status, error) = match &j.status {
        JobStatus::Done => ("done", JsonValue::Null),
        JobStatus::Failed(e) => ("failed", text(e)),
    };
    object(vec![
        ("id", count(j.id)),
        ("node", count(j.node)),
        ("method", text(&j.method)),
        ("task", text(&j.task)),
        ("description", text(&j.description)),
        ("status", text(status)),
        ("error", error),
        ("new_nodes", ids(&j.new_nodes)),
        ("new_edges", ids(&j.new_edges)),
        ("found_new", JsonValue::Bool(j.found_new)),
        ("found_lower", JsonValue::Bool(j.found_lower)),
        ("trajectory", opt(j.trajectory.as_deref(), text)),
    ])
}

/// The document of `network`; `extra` goes into it as `run` (settings and the state of the
/// run that the network itself does not hold).
pub fn to_json(network: &Network, extra: JsonValue) -> JsonValue {
    let minima = network.minima().count();
    let transition_states = network.of_kind(&NodeKind::TransitionState).count();
    let mut entries = vec![
        ("format", text(NETWORK_FORMAT)),
        ("version", count(NETWORK_FORMAT_VERSION as usize)),
        ("mopt_rs_version", text(crate::VERSION)),
        (
            "units",
            object(vec![
                ("energy", text("hartree")),
                ("length", text("bohr")),
                ("wavenumber", text("cm^-1")),
                ("arc_length", text("amu^1/2 bohr")),
            ]),
        ),
        (
            "elements",
            JsonValue::Array(network.elements.iter().map(|e| text(e.symbol())).collect()),
        ),
        (
            "whole_elements",
            match &network.whole_elements {
                Some(list) => JsonValue::Array(list.iter().map(|e| text(e.symbol())).collect()),
                None => JsonValue::Null,
            },
        ),
        (
            "variables",
            match &network.variables {
                Some(names) => JsonValue::Array(names.iter().map(|n| text(n)).collect()),
                None => JsonValue::Null,
            },
        ),
        (
            "states",
            JsonValue::Array(network.states.iter().map(state_json).collect()),
        ),
        (
            "inputs",
            JsonValue::Array(
                network
                    .inputs
                    .iter()
                    .map(|i| {
                        let mut entries = vec![
                            ("file", text(&i.file)),
                            ("frame", count(i.frame)),
                            ("node", opt(i.node, count)),
                        ];
                        if i.goal {
                            entries.push(("goal", JsonValue::Bool(true)));
                        }
                        object(entries)
                    })
                    .collect(),
            ),
        ),
        (
            "summary",
            object(vec![
                ("minima", count(minima)),
                ("transition_states", count(transition_states)),
                ("ts_edges", count(network.ts_edges.len())),
                ("crossing_edges", count(network.crossing_edges.len())),
                (
                    "dissociation_channels",
                    count(network.of_kind(&NodeKind::Dissociated).count()),
                ),
                (
                    "dissociation_edges",
                    count(network.dissociation_edges.len()),
                ),
                ("jobs", count(network.jobs.len())),
            ]),
        ),
        (
            "nodes",
            JsonValue::Array(network.nodes.iter().map(node_json).collect()),
        ),
        (
            "ts_edges",
            JsonValue::Array(network.ts_edges.iter().map(ts_edge_json).collect()),
        ),
        (
            "crossing_edges",
            JsonValue::Array(
                network
                    .crossing_edges
                    .iter()
                    .map(crossing_edge_json)
                    .collect(),
            ),
        ),
        (
            "dissociation_edges",
            JsonValue::Array(
                network
                    .dissociation_edges
                    .iter()
                    .map(dissociation_edge_json)
                    .collect(),
            ),
        ),
        (
            "saddle_edges",
            JsonValue::Array(
                network
                    .saddle_edges
                    .iter()
                    .map(super::saddle_json::saddle_edge_json)
                    .collect(),
            ),
        ),
        (
            "fragments",
            JsonValue::Array(network.fragments.iter().map(fragment_json).collect()),
        ),
        (
            "jobs",
            JsonValue::Array(network.jobs.iter().map(job_json).collect()),
        ),
        (
            "warnings",
            JsonValue::Array(
                network
                    .warnings
                    .iter()
                    .map(|w| {
                        object(vec![
                            ("job", opt(w.job, count)),
                            ("message", text(&w.message)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "evaluations",
            object(vec![
                ("energies", count(network.evaluations.energies)),
                ("gradients", count(network.evaluations.gradients)),
                ("hessians", count(network.evaluations.hessians)),
                (
                    "micro_relaxations",
                    count(network.evaluations.micro_relaxations),
                ),
                (
                    "micro_iterations",
                    count(network.evaluations.micro_iterations),
                ),
            ]),
        ),
        ("run", extra),
    ];
    // The path peaks, the entrance and the steering, where the network has any.
    entries.extend(super::extras_json::extras_json(network));
    // The constraints and static bias potentials of the surface, where the run has any.
    if !network.surface.is_empty() {
        entries.push((
            "surface",
            JsonValue::Array(network.surface.iter().map(|s| text(s)).collect()),
        ));
    }
    // The transition states with an end that is not known, where there are any, so that other
    // files stay as they were.
    if !network.open_ts_edges.is_empty() {
        entries.push((
            "open_ts_edges",
            JsonValue::Array(
                network
                    .open_ts_edges
                    .iter()
                    .map(open_ts_edge_json)
                    .collect(),
            ),
        ));
    }
    object(entries)
}

pub(super) fn field<'a>(m: &'a Map, key: &str, what: &str) -> MoptResult<&'a JsonValue> {
    m.get(key)
        .ok_or_else(|| MoptError::parse(format!("{what}: no '{key}'")))
}

pub(super) fn as_map<'a>(v: &'a JsonValue, what: &str) -> MoptResult<&'a Map> {
    v.as_object()
        .ok_or_else(|| MoptError::parse(format!("{what} is not an object")))
}

pub(super) fn list<'a>(m: &'a Map, key: &str, what: &str) -> MoptResult<&'a [JsonValue]> {
    field(m, key, what)?
        .as_array()
        .ok_or_else(|| MoptError::parse(format!("{what}: '{key}' is not a list")))
}

pub(super) fn f64_of(m: &Map, key: &str, what: &str) -> MoptResult<f64> {
    field(m, key, what)?
        .as_f64()
        .ok_or_else(|| MoptError::parse(format!("{what}: '{key}' is not a number")))
}

pub(super) fn usize_of(m: &Map, key: &str, what: &str) -> MoptResult<usize> {
    field(m, key, what)?
        .as_usize()
        .ok_or_else(|| MoptError::parse(format!("{what}: '{key}' is not a count")))
}

pub(super) fn opt_f64(m: &Map, key: &str) -> Option<f64> {
    m.get(key).and_then(JsonValue::as_f64)
}

pub(super) fn opt_usize(m: &Map, key: &str) -> Option<usize> {
    m.get(key).and_then(JsonValue::as_usize)
}

pub(super) fn opt_text(m: &Map, key: &str) -> Option<String> {
    m.get(key).and_then(JsonValue::as_str).map(str::to_string)
}

pub(super) fn str_of(m: &Map, key: &str, what: &str) -> MoptResult<String> {
    opt_text(m, key).ok_or_else(|| MoptError::parse(format!("{what}: '{key}' is not a text")))
}

pub(super) fn f64s(v: &JsonValue, what: &str) -> MoptResult<Vec<f64>> {
    v.as_array()
        .ok_or_else(|| MoptError::parse(format!("{what} is not a list")))?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| MoptError::parse(format!("{what}: not a number")))
        })
        .collect()
}

pub(super) fn usizes(v: &JsonValue, what: &str) -> MoptResult<Vec<usize>> {
    v.as_array()
        .ok_or_else(|| MoptError::parse(format!("{what} is not a list")))?
        .iter()
        .map(|x| {
            x.as_usize()
                .ok_or_else(|| MoptError::parse(format!("{what}: not a count")))
        })
        .collect()
}

fn state_of(v: &JsonValue, what: &str) -> MoptResult<ElectronicState> {
    let m = as_map(v, what)?;
    Ok(ElectronicState {
        charge: f64_of(m, "charge", what)? as i32,
        multiplicity: usize_of(m, "multiplicity", what)? as u32,
        excited_state: usize_of(m, "excited_state", what)?,
    })
}

/// The names of the variables of a network document (`None`: a network of structures).
fn variable_names(m: &Map) -> MoptResult<Option<Arc<Vec<String>>>> {
    match m.get("variables") {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Array(items)) => Ok(Some(Arc::new(
            items
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| MoptError::parse("a name of a variable is not a text"))
                })
                .collect::<MoptResult<Vec<_>>>()?,
        ))),
        Some(_) => Err(MoptError::parse("'variables' is not a list of names")),
    }
}

/// An empty network of the elements, or of the variables, of a document.
/// The elements of the whole system of a network of cores (`whole_elements`), where it has
/// them.
fn whole_elements_of(m: &Map) -> MoptResult<Option<Vec<Element>>> {
    match m.get("whole_elements").and_then(JsonValue::as_array) {
        None => Ok(None),
        Some(items) => items
            .iter()
            .map(|e| {
                e.as_str()
                    .ok_or_else(|| MoptError::parse("an element is not a text"))
                    .and_then(Element::parse_token)
            })
            .collect::<MoptResult<Vec<_>>>()
            .map(Some),
    }
}

fn empty_network(elements: &[Element], names: Option<&Arc<Vec<String>>>) -> Network {
    match names {
        Some(names) => Network::of_variables(names.to_vec()),
        None => Network::new(elements.to_vec()),
    }
}

fn node_of(
    v: &JsonValue,
    elements: &[Element],
    whole_elements: Option<&[Element]>,
    names: Option<&Arc<Vec<String>>>,
    k: usize,
) -> MoptResult<Node> {
    let what = format!("node {k}");
    let m = as_map(v, &what)?;
    let state = state_of(field(m, "state", &what)?, &what)?;
    let kind = match str_of(m, "kind", &what)?.as_str() {
        "EQ" => NodeKind::Minimum,
        "TS" => NodeKind::TransitionState,
        tag @ ("MECP" | "MECI") => {
            let states = list(m, "states", &what)?;
            let [a, b] = states else {
                return Err(MoptError::parse(format!(
                    "{what}: a crossing has two states"
                )));
            };
            NodeKind::Crossing {
                kind: if tag == "MECP" {
                    CrossingKind::Mecp
                } else {
                    CrossingKind::Meci
                },
                states: [state_of(a, &what)?, state_of(b, &what)?],
            }
        }
        "DC" => NodeKind::Dissociated,
        other => match NodeKind::of_saddle_tag(other) {
            Some(kind) => kind,
            None => {
                return Err(MoptError::parse(format!(
                    "{what}: unknown kind '{other}' (EQ, TS, MECP, MECI, DC, 2SP to 9SP)"
                )))
            }
        },
    };
    let system = match names {
        Some(names) => {
            let values = f64s(field(m, "variables", &what)?, &what)?;
            Configuration::variables(Col::from_fn(values.len(), |i| values[i]), Arc::clone(names))?
        }
        None => {
            let coords = f64s(field(m, "coords_bohr", &what)?, &what)?;
            Configuration::Structure(MolecularSystem::from_flat_bohr(
                elements.to_vec(),
                &Col::from_fn(coords.len(), |i| coords[i]),
                SpinCharge::checked(i64::from(state.charge), i64::from(state.multiplicity))?,
                cell_of(m, &what)?,
            )?)
        }
    };
    // The whole structure of a core (ONIOM), in the elements of the whole system.
    let whole = match (m.get("whole_coords_bohr"), whole_elements) {
        (Some(JsonValue::Null) | None, _) => None,
        (Some(v), Some(all)) => {
            let coords = f64s(v, &what)?;
            Some(Configuration::Structure(MolecularSystem::from_flat_bohr(
                all.to_vec(),
                &Col::from_fn(coords.len(), |i| coords[i]),
                SpinCharge::checked(i64::from(state.charge), i64::from(state.multiplicity))?,
                cell_of(m, &what)?,
            )?))
        }
        (Some(_), None) => {
            return Err(MoptError::parse(format!(
                "{what} has a whole structure, and the network no elements of the whole system"
            )))
        }
    };
    let environments = match m.get("environments").and_then(JsonValue::as_array) {
        None => Vec::new(),
        Some(items) => items
            .iter()
            .map(|e| {
                let e = as_map(e, &what)?;
                Ok(super::network::EnvironmentConformer {
                    energy: f64_of(e, "energy_hartree", &what)?,
                    job: opt_usize(e, "job"),
                })
            })
            .collect::<MoptResult<Vec<_>>>()?,
    };
    let origin = as_map(field(m, "origin", &what)?, &what)?;
    let mut node = Node::new(
        kind,
        Level::from_number(usize_of(m, "level", &what)? as u64)?,
        system,
        f64_of(m, "energy_hartree", &what)?,
        state,
        Origin {
            input: opt_usize(origin, "input"),
            job: opt_usize(origin, "job"),
            from_node: opt_usize(origin, "from_node"),
        },
    );
    node.whole = whole;
    node.environments = environments;
    node.id = usize_of(m, "id", &what)?;
    if node.id != k {
        return Err(MoptError::parse(format!(
            "{what} has the number {}",
            node.id
        )));
    }
    node.free_energy = opt_f64(m, "free_energy_hartree");
    node.bare_energy = opt_f64(m, "bare_energy_hartree");
    node.frequencies = match m.get("frequencies_cm1") {
        Some(JsonValue::Null) | None => None,
        Some(v) => Some(f64s(v, &what)?),
    };
    node.index_wavenumbers = match m.get("index_imaginary_cm1") {
        Some(JsonValue::Null) | None => None,
        Some(v) => Some(f64s(v, &what)?),
    };
    node.excluded = opt_text(m, "excluded");
    // None in the file of an earlier mopt-rs.
    if let Some(JsonValue::Array(pairs)) = m.get("followed_states") {
        for pair in pairs {
            match pair.as_array().map(|p| (p.first(), p.get(1))) {
                Some((Some(a), Some(b))) => match (a.as_usize(), b.as_usize()) {
                    (Some(label), Some(root)) => node.roots.push((label, root)),
                    _ => {
                        return Err(MoptError::parse(format!(
                            "{what}: followed_states holds pairs of whole numbers"
                        )))
                    }
                },
                _ => {
                    return Err(MoptError::parse(format!(
                        "{what}: followed_states holds pairs of whole numbers"
                    )))
                }
            }
        }
    }
    if node.kind == NodeKind::Dissociated {
        node.parts = parts_of(field(m, "parts", &what)?, &what)?;
    }
    Ok(node)
}

/// The fragments and the barrierless dissociations of the document `m` (none in a document of
/// an earlier mopt-rs); `node` maps the node numbers of the file to those of `network`, and a
/// dissociation that cannot be read or whose node is missing is left out with a note when
/// `notes` is given (and is an error otherwise).
fn channels_of(
    m: &Map,
    network: &mut Network,
    node: &dyn Fn(usize) -> Option<usize>,
    mut notes: Option<&mut Vec<String>>,
) -> MoptResult<()> {
    let items = |key: &str| match m.get(key) {
        Some(JsonValue::Array(a)) => a.clone(),
        _ => Vec::new(),
    };
    for (k, v) in items("fragments").iter().enumerate() {
        let read = fragment_of(v, k).and_then(|f| {
            if f.id == network.fragments.len() {
                Ok(f)
            } else {
                Err(MoptError::parse(format!(
                    "fragment {k} of the network file has the number {}",
                    f.id
                )))
            }
        });
        match (read, notes.as_deref_mut()) {
            (Ok(f), _) => network.fragments.push(f),
            (Err(e), Some(notes)) => {
                // The channels keep their energies; their fragments are not known.
                notes.push(format!(
                    "the fragments of the network file are left out: {e}"
                ));
                network.fragments.clear();
                break;
            }
            (Err(e), None) => return Err(e),
        }
    }
    for (k, v) in items("dissociation_edges").iter().enumerate() {
        let added =
            dissociation_edge_of(v, k).and_then(|mut e| match (node(e.minimum), node(e.channel)) {
                (Some(a), Some(b)) => {
                    e.minimum = a;
                    e.channel = b;
                    network.add_dissociation_edge(e)
                }
                _ => Err(MoptError::parse("a node of it is left out")),
            });
        match (added, notes.as_deref_mut()) {
            (Ok(_), _) => {}
            (Err(e), Some(notes)) => notes.push(format!(
                "dissociation edge {k} of the network file is left out: {e}"
            )),
            (Err(e), None) => return Err(e),
        }
    }
    Ok(())
}

fn pair(v: &[usize], what: &str) -> MoptResult<[usize; 2]> {
    match v {
        [a, b] => Ok([*a, *b]),
        _ => Err(MoptError::parse(format!("{what}: two ends are needed"))),
    }
}

fn ts_edge_of(v: &JsonValue, k: usize) -> MoptResult<TsEdge> {
    let what = format!("TS edge {k}");
    let m = as_map(v, &what)?;
    let barriers = f64s(field(m, "barriers_hartree", &what)?, &what)?;
    let [b0, b1] = barriers[..] else {
        return Err(MoptError::parse(format!("{what}: two barriers are needed")));
    };
    Ok(TsEdge {
        id: usize_of(m, "id", &what)?,
        level: Level::from_number(usize_of(m, "level", &what)? as u64)?,
        ts: usize_of(m, "ts", &what)?,
        ends: pair(&usizes(field(m, "ends", &what)?, &what)?, &what)?,
        barriers: [b0, b1],
        imaginary_wavenumber: opt_f64(m, "imaginary_wavenumber_cm1"),
        irc: irc_of(m, &what)?,
        job: opt_usize(m, "job"),
    })
}

fn open_ts_edge_of(v: &JsonValue, k: usize) -> MoptResult<OpenTsEdge> {
    let what = format!("TS edge with an unknown end {k}");
    let m = as_map(v, &what)?;
    let two = |key: &str| -> MoptResult<[&JsonValue; 2]> {
        match list(m, key, &what)? {
            [a, b] => Ok([a, b]),
            _ => Err(MoptError::parse(format!("{what}: '{key}' has two entries"))),
        }
    };
    let ends = two("ends")?;
    let barriers = two("barriers_hartree")?;
    Ok(OpenTsEdge {
        id: usize_of(m, "id", &what)?,
        level: Level::from_number(usize_of(m, "level", &what)? as u64)?,
        ts: usize_of(m, "ts", &what)?,
        ends: [ends[0].as_usize(), ends[1].as_usize()],
        barriers: [barriers[0].as_f64(), barriers[1].as_f64()],
        imaginary_wavenumber: opt_f64(m, "imaginary_wavenumber_cm1"),
        irc: irc_of(m, &what)?,
        job: opt_usize(m, "job"),
    })
}

/// The IRC of the edge `m`.
fn irc_of(m: &Map, what: &str) -> MoptResult<IrcPath> {
    let irc = as_map(field(m, "irc", what)?, what)?;
    let sides = list(irc, "sides", what)?
        .iter()
        .map(|s| side_of(s, what))
        .collect::<MoptResult<Vec<_>>>()?;
    Ok(IrcPath {
        sides,
        file: opt_text(irc, "file"),
        possible_bifurcations: match irc.get("possible_bifurcations") {
            None | Some(JsonValue::Null) => Vec::new(),
            Some(_) => list(irc, "possible_bifurcations", what)?
                .iter()
                .map(|b| {
                    let b = as_map(b, what)?;
                    Ok(PossibleBifurcation {
                        side: str_of(b, "side", what)?,
                        after: opt_f64(b, "after_arc_length"),
                        at: f64_of(b, "at_arc_length", what)?,
                        wavenumber: f64_of(b, "wavenumber_cm1", what)?,
                        confirmed_ends: match b.get("confirmed_ends") {
                            None | Some(JsonValue::Null) => Vec::new(),
                            Some(v) => usizes(v, what)?,
                        },
                        inflection: opt_f64(b, "inflection_arc_length"),
                    })
                })
                .collect::<MoptResult<Vec<_>>>()?,
        },
    })
}

fn crossing_edge_of(v: &JsonValue, k: usize) -> MoptResult<CrossingEdge> {
    let what = format!("crossing edge {k}");
    let m = as_map(v, &what)?;
    let two = |key: &str| -> MoptResult<[&JsonValue; 2]> {
        match list(m, key, &what)? {
            [a, b] => Ok([a, b]),
            _ => Err(MoptError::parse(format!("{what}: '{key}' has two entries"))),
        }
    };
    let ends = two("ends")?;
    let barriers = two("barriers_hartree")?;
    // An edge of an earlier mopt-rs, or one without a path, has no meta-IRC.
    let irc = match m.get("irc").and_then(JsonValue::as_object) {
        Some(irc) => IrcPath {
            sides: list(irc, "sides", &what)?
                .iter()
                .map(|s| side_of(s, &what))
                .collect::<MoptResult<Vec<_>>>()?,
            file: opt_text(irc, "file"),
            possible_bifurcations: Vec::new(),
        },
        None => IrcPath::default(),
    };
    Ok(CrossingEdge {
        irc,
        id: usize_of(m, "id", &what)?,
        level: Level::from_number(usize_of(m, "level", &what)? as u64)?,
        crossing: usize_of(m, "crossing", &what)?,
        ends: [ends[0].as_usize(), ends[1].as_usize()],
        barriers: [barriers[0].as_f64(), barriers[1].as_f64()],
        gradient_difference: opt_f64(m, "gradient_difference"),
        gap: opt_f64(m, "gap_hartree"),
        seam: seam_check_of(m),
        job: opt_usize(m, "job"),
    })
}

fn job_of(v: &JsonValue, k: usize) -> MoptResult<JobRecord> {
    let what = format!("job {k}");
    let m = as_map(v, &what)?;
    let status = match str_of(m, "status", &what)?.as_str() {
        "done" => JobStatus::Done,
        "failed" => JobStatus::Failed(opt_text(m, "error").unwrap_or_default()),
        other => {
            return Err(MoptError::parse(format!(
                "{what}: unknown status '{other}'"
            )))
        }
    };
    let flag = |key: &str| m.get(key).and_then(JsonValue::as_bool).unwrap_or(false);
    Ok(JobRecord {
        id: usize_of(m, "id", &what)?,
        node: usize_of(m, "node", &what)?,
        method: str_of(m, "method", &what)?,
        task: str_of(m, "task", &what)?,
        description: opt_text(m, "description").unwrap_or_default(),
        status,
        new_nodes: usizes(field(m, "new_nodes", &what)?, &what)?,
        new_edges: usizes(field(m, "new_edges", &what)?, &what)?,
        found_new: flag("found_new"),
        found_lower: flag("found_lower"),
        trajectory: opt_text(m, "trajectory"),
    })
}

/// The network of the document `doc` and its `run` entry.
pub fn from_json(doc: &JsonValue) -> MoptResult<(Network, JsonValue)> {
    let what = "the network file";
    let m = as_map(doc, what)?;
    if opt_text(m, "format").as_deref() != Some(NETWORK_FORMAT) {
        return Err(MoptError::parse(format!(
            "{what} is not a network of mopt-rs (no format '{NETWORK_FORMAT}')"
        )));
    }
    let version = usize_of(m, "version", what)? as u64;
    if version != NETWORK_FORMAT_VERSION {
        return Err(MoptError::parse(format!(
            "{what} has the layout version {version}; this mopt-rs reads version \
             {NETWORK_FORMAT_VERSION}"
        )));
    }
    let elements = list(m, "elements", what)?
        .iter()
        .map(|e| {
            e.as_str()
                .ok_or_else(|| MoptError::parse("an element is not a text"))
                .and_then(Element::parse_token)
        })
        .collect::<MoptResult<Vec<_>>>()?;
    let names = variable_names(m)?;
    let mut network = empty_network(&elements, names.as_ref());
    network.whole_elements = whole_elements_of(m)?;
    // A network of an earlier mopt-rs, or of a run of one state, has none.
    if let Some(items) = m.get("states").and_then(JsonValue::as_array) {
        for (k, v) in items.iter().enumerate() {
            network
                .states
                .push(state_of(v, &format!("state {}", k + 1))?);
        }
    }
    for (k, v) in list(m, "inputs", what)?.iter().enumerate() {
        let i = as_map(v, &format!("input {k}"))?;
        network.inputs.push(InputRecord {
            file: str_of(i, "file", what)?,
            frame: usize_of(i, "frame", what)?,
            node: opt_usize(i, "node"),
            goal: i.get("goal").and_then(JsonValue::as_bool) == Some(true),
        });
    }
    for (k, v) in list(m, "nodes", what)?.iter().enumerate() {
        // Numbered within its kind; a network of an earlier version has no numbers, and its
        // nodes are numbered in their order.
        let number = v.get("number").and_then(JsonValue::as_usize);
        network.restore_node(
            node_of(
                v,
                &elements,
                network.whole_elements.as_deref(),
                names.as_ref(),
                k,
            )?,
            number,
        );
    }
    for (k, v) in list(m, "ts_edges", what)?.iter().enumerate() {
        let edge = ts_edge_of(v, k)?;
        network.add_ts_edge(edge)?;
    }
    for (k, v) in list(m, "crossing_edges", what)?.iter().enumerate() {
        let edge = crossing_edge_of(v, k)?;
        network.add_crossing_edge(edge)?;
    }
    // A network of an earlier mopt-rs, or without such transition states, has none.
    if let Some(items) = m.get("open_ts_edges").and_then(JsonValue::as_array) {
        for (k, v) in items.iter().enumerate() {
            network.add_open_ts_edge(open_ts_edge_of(v, k)?)?;
        }
    }
    let count = network.nodes.len();
    channels_of(m, &mut network, &|id| (id < count).then_some(id), None)?;
    // A network of an earlier mopt-rs, or of a run without saddle points of higher order,
    // has none.
    if let Some(items) = m.get("saddle_edges").and_then(JsonValue::as_array) {
        for (k, v) in items.iter().enumerate() {
            let edge = super::saddle_json::saddle_edge_of(v, k)?;
            network.add_saddle_edge(edge)?;
        }
    }
    super::extras_json::read_extras(
        m,
        &mut network,
        names.as_ref(),
        &|id| (id < count).then_some(id),
        None,
    )?;
    for (k, v) in list(m, "jobs", what)?.iter().enumerate() {
        network.jobs.push(job_of(v, k)?);
    }
    for v in list(m, "warnings", what)? {
        let w = as_map(v, "a warning")?;
        network.warnings.push(NetworkWarning {
            job: opt_usize(w, "job"),
            message: str_of(w, "message", "a warning")?,
        });
    }
    if let Some(e) = m.get("evaluations").and_then(JsonValue::as_object) {
        network.evaluations = EvaluationTotals {
            energies: opt_usize(e, "energies").unwrap_or(0),
            gradients: opt_usize(e, "gradients").unwrap_or(0),
            hessians: opt_usize(e, "hessians").unwrap_or(0),
            micro_relaxations: opt_usize(e, "micro_relaxations").unwrap_or(0),
            micro_iterations: opt_usize(e, "micro_iterations").unwrap_or(0),
        };
    }
    network.surface = surface_of(m);
    let run = m.get("run").cloned().unwrap_or(JsonValue::Null);
    Ok((network, run))
}

/// The lines of the surface of a network file (its constraints and bias potentials).
fn surface_of(m: &Map) -> Vec<String> {
    m.get("surface")
        .and_then(JsonValue::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The network of `doc` as far as it can be read, for a report of a damaged or partly written
/// file, or one of another version of the layout: a node, edge, job, or input that cannot be
/// read is left out with a note (the others are renumbered around it), and so is an edge one of
/// whose nodes is left out. Only a document that is not an object or has no elements fails.
/// Returns the network, its `run` entry, and the notes.
pub fn from_json_lenient(doc: &JsonValue) -> MoptResult<(Network, JsonValue, Vec<String>)> {
    let what = "the network file";
    let m = as_map(doc, what)?;
    let mut notes = Vec::new();
    if opt_text(m, "format").as_deref() != Some(NETWORK_FORMAT) {
        notes.push(format!(
            "{what} does not say it is a network of mopt-rs (format '{NETWORK_FORMAT}'); it is \
             read as far as it goes"
        ));
    }
    let version = m.get("version").and_then(JsonValue::as_usize);
    if version.map(|v| v as u64) != Some(NETWORK_FORMAT_VERSION) {
        notes.push(format!(
            "{what} has the layout version {}; this mopt-rs writes version \
             {NETWORK_FORMAT_VERSION}, and the file is read as far as it goes",
            version.map_or_else(|| "(none)".to_string(), |v| v.to_string())
        ));
    }
    let elements = list(m, "elements", what)?
        .iter()
        .map(|e| {
            e.as_str()
                .ok_or_else(|| MoptError::parse("an element is not a text"))
                .and_then(Element::parse_token)
        })
        .collect::<MoptResult<Vec<_>>>()?;
    let items = |key: &str, notes: &mut Vec<String>| -> Vec<JsonValue> {
        match m.get(key) {
            Some(JsonValue::Array(a)) => a.clone(),
            Some(_) => {
                notes.push(format!("'{key}' of {what} is not a list; it is left out"));
                Vec::new()
            }
            None => {
                notes.push(format!("{what} has no '{key}'"));
                Vec::new()
            }
        }
    };
    let names = match variable_names(m) {
        Ok(names) => names,
        Err(e) => {
            notes.push(format!(
                "{what}: {e}; it is read as a network of structures"
            ));
            None
        }
    };
    let mut network = empty_network(&elements, names.as_ref());
    network.whole_elements = whole_elements_of(m)?;
    // A network of an earlier mopt-rs, or of a run of one state, has no states.
    let states = m
        .get("states")
        .and_then(JsonValue::as_array)
        .map(<[JsonValue]>::to_vec)
        .unwrap_or_default();
    for (k, v) in states.iter().enumerate() {
        match state_of(v, &format!("state {}", k + 1)) {
            Ok(s) => network.states.push(s),
            Err(e) => notes.push(format!("{what}: {e}; the state is left out")),
        }
    }
    // Nodes, with the map of their places in the file to their numbers here.
    let mut ids: Vec<Option<usize>> = Vec::new();
    for (k, v) in items("nodes", &mut notes).iter().enumerate() {
        match node_of(
            v,
            &elements,
            network.whole_elements.as_deref(),
            names.as_ref(),
            k,
        ) {
            Ok(node) => {
                ids.push(Some(network.nodes.len()));
                let number = v.get("number").and_then(JsonValue::as_usize);
                network.restore_node(node, number);
            }
            Err(e) => {
                notes.push(format!("node {k} of {what} is left out: {e}"));
                ids.push(None);
            }
        }
    }
    let node = |id: usize| ids.get(id).copied().flatten();
    for n in &mut network.nodes {
        n.origin.from_node = n.origin.from_node.and_then(node);
    }
    for (k, v) in items("inputs", &mut notes).iter().enumerate() {
        let read = as_map(v, &format!("input {k}")).and_then(|i| {
            Ok(InputRecord {
                file: str_of(i, "file", what)?,
                frame: usize_of(i, "frame", what)?,
                node: opt_usize(i, "node").and_then(node),
                goal: i.get("goal").and_then(JsonValue::as_bool) == Some(true),
            })
        });
        match read {
            Ok(input) => network.inputs.push(input),
            Err(e) => notes.push(format!("input {k} of {what} is left out: {e}")),
        }
    }
    let mut edges: Vec<Option<usize>> = Vec::new();
    for (k, v) in items("ts_edges", &mut notes).iter().enumerate() {
        let added = ts_edge_of(v, k).and_then(|mut e| {
            match (node(e.ts), node(e.ends[0]), node(e.ends[1])) {
                (Some(ts), Some(a), Some(b)) => {
                    e.ts = ts;
                    e.ends = [a, b];
                    network.add_ts_edge(e)
                }
                _ => Err(MoptError::parse("a node of it is left out")),
            }
        });
        match added {
            Ok(id) => edges.push(Some(id)),
            Err(e) => {
                notes.push(format!(
                    "transition-state edge {k} of {what} is left out: {e}"
                ));
                edges.push(None);
            }
        }
    }
    for (k, v) in items("crossing_edges", &mut notes).iter().enumerate() {
        let added = crossing_edge_of(v, k).and_then(|mut e| {
            let crossing = node(e.crossing)
                .ok_or_else(|| MoptError::parse("its crossing point is left out"))?;
            e.crossing = crossing;
            for end in &mut e.ends {
                *end = end.and_then(node);
            }
            network.add_crossing_edge(e)
        });
        if let Err(e) = added {
            notes.push(format!("crossing edge {k} of {what} is left out: {e}"));
        }
    }
    if let Some(items) = m.get("open_ts_edges").and_then(JsonValue::as_array) {
        for (k, v) in items.iter().enumerate() {
            let added = open_ts_edge_of(v, k).and_then(|mut e| {
                e.ts = node(e.ts)
                    .ok_or_else(|| MoptError::parse("its transition state is left out"))?;
                for end in &mut e.ends {
                    *end = end.and_then(node);
                }
                network.add_open_ts_edge(e)
            });
            if let Err(e) = added {
                notes.push(format!(
                    "transition state with an unknown end {k} of {what} is left out: {e}"
                ));
            }
        }
    }
    channels_of(m, &mut network, &node, Some(&mut notes))?;
    if let Some(items) = m.get("saddle_edges").and_then(JsonValue::as_array) {
        for (k, v) in items.iter().enumerate() {
            let added = super::saddle_json::saddle_edge_of(v, k).and_then(|mut e| {
                e.saddle = node(e.saddle)
                    .ok_or_else(|| MoptError::parse("its saddle point is left out"))?;
                for end in &mut e.ends {
                    *end = end.and_then(node);
                }
                network.add_saddle_edge(e)
            });
            if let Err(e) = added {
                notes.push(format!("saddle edge {k} of {what} is left out: {e}"));
            }
        }
    }
    super::extras_json::read_extras(m, &mut network, names.as_ref(), &node, Some(&mut notes))?;
    for (k, v) in items("jobs", &mut notes).iter().enumerate() {
        match job_of(v, k) {
            Ok(mut job) => {
                job.node = node(job.node).unwrap_or(usize::MAX);
                job.new_nodes = job.new_nodes.iter().filter_map(|&id| node(id)).collect();
                job.new_edges = job
                    .new_edges
                    .iter()
                    .filter_map(|&id| edges.get(id).copied().flatten())
                    .collect();
                network.jobs.push(job);
            }
            Err(e) => notes.push(format!("job {k} of {what} is left out: {e}")),
        }
    }
    for v in items("warnings", &mut notes) {
        let read = as_map(&v, "a warning").and_then(|w| {
            Ok(NetworkWarning {
                job: opt_usize(w, "job"),
                message: str_of(w, "message", "a warning")?,
            })
        });
        match read {
            Ok(w) => network.warnings.push(w),
            Err(e) => notes.push(format!("a warning of {what} is left out: {e}")),
        }
    }
    if let Some(e) = m.get("evaluations").and_then(JsonValue::as_object) {
        network.evaluations = EvaluationTotals {
            energies: opt_usize(e, "energies").unwrap_or(0),
            gradients: opt_usize(e, "gradients").unwrap_or(0),
            hessians: opt_usize(e, "hessians").unwrap_or(0),
            micro_relaxations: opt_usize(e, "micro_relaxations").unwrap_or(0),
            micro_iterations: opt_usize(e, "micro_iterations").unwrap_or(0),
        };
    }
    network.surface = surface_of(m);
    let run = m.get("run").cloned().unwrap_or(JsonValue::Null);
    Ok((network, run, notes))
}

#[cfg(test)]
#[path = "network_json_tests.rs"]
mod tests;
