//! Where the network is kept (extension point): a [`NetworkStore`] writes the whole network
//! into the run directory after every batch of jobs that is kept (every job when one runs at a
//! time). Stores are registered by name in a [`StoreRegistry`].
//!
//! Built in:
//! - `files` (always written; the restart reads it): `reaction_network.json` (the network with
//!   the version of its layout, [`super::network_json`]), `structures/EQ000000.xyz`,
//!   `structures/TS000001.xyz`, ... (every node, with its energy), `nodes.csv`,
//!   `ts_edges.csv`, `crossing_edges.csv`, `dissociation_edges.csv`;
//! - `sqlite`: the network as an SQLite database, `reaction_network.sqlite`
//!   ([`super::sqlite_store`]).
//!
//! Every file is written next to its place and moved there, so that a run stopped or killed
//! while it writes leaves the last whole file; [`forget_beyond`] removes, at a restart, what a
//! killed run wrote beyond the network it left.
//!
//! MultiOptPy: `Wrapper/mapper.py:NetworkGraph.save` (`reaction_network.json`) and
//!  `ReactionNetworkMapper._persist_node_xyz`, `_persist_ts_xyz` (`nodes/EQ000000.xyz`,
//!  `nodes/TS000000.xyz`; mopt-rs numbers transition states as nodes).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use super::network::{Network, Node};
use super::network_json::{from_json, to_json};
use crate::core_infra::json::JsonValue;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult, ResultExt};

/// The network file in the run directory.
pub const NETWORK_FILE: &str = "reaction_network.json";
/// The table of the nodes.
pub const NODES_FILE: &str = "nodes.csv";
/// The table of the path peaks in the run directory (written where the run has any).
pub const PATH_PEAKS_FILE: &str = "path_peaks.csv";
/// The table of the transition-state edges.
pub const TS_EDGES_FILE: &str = "ts_edges.csv";
/// The table of the crossing edges.
pub const CROSSING_EDGES_FILE: &str = "crossing_edges.csv";
/// The table of the barrierless dissociations of complexes into dissociation channels.
pub const DISSOCIATION_EDGES_FILE: &str = "dissociation_edges.csv";
/// The directory of the structures of the nodes.
pub const STRUCTURES_DIR: &str = "structures";

/// Write `content` to `path` through a file next to it.
pub fn write_whole(path: &Path, content: &[u8]) -> MoptResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = std::path::PathBuf::from(part);
    std::fs::write(&part, content).map_err(|e| MoptError::from(e).context(part.display()))?;
    std::fs::rename(&part, path).map_err(|e| MoptError::from(e).context(path.display()))?;
    Ok(())
}

/// The file of the structure of `node`, relative to the run directory.
pub fn structure_file(node: &Node) -> String {
    format!(
        "{STRUCTURES_DIR}/{}{:06}.{}",
        node.kind.tag(),
        node.number,
        super::configuration::frame_extension(node.system.is_variables())
    )
}

/// The file of a node: an xyz file for a structure; for plain variables a variables file
/// (`# label energy=...`, then one `name value` line per variable), which `optmain -os ...`
/// reads as its input.
pub fn node_file_text(node: &Node) -> String {
    let comment = format!("{} energy={:.12}", node.label(), node.energy);
    // The whole structure where the network holds a part of it (the core of an ONIOM
    // surface).
    match node.whole.as_ref().unwrap_or(&node.system) {
        super::configuration::Configuration::Variables(p) => {
            let mut out = format!("# {comment}\n");
            for (name, v) in p.names.iter().zip(p.values.iter()) {
                out.push_str(&format!("{name} {v:.12}\n"));
            }
            out
        }
        other => other.frame(&comment),
    }
}

/// Keeps the network somewhere.
pub trait NetworkStore: Send + Sync {
    /// Its name in the registry.
    fn name(&self) -> &str;

    /// Write `network` with the state of the run `run` into the run directory `dir`.
    fn write(&self, network: &Network, run: &JsonValue, dir: &Path) -> MoptResult<()>;
}

/// The files of the run (`files`).
#[derive(Debug, Clone, Copy, Default)]
pub struct FileStore;

fn opt_text<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or_else(String::new, |x| x.to_string())
}

fn kcal(hartree: f64) -> String {
    format!("{:.4}", hartree * Units::HARTREE_TO_KCAL_MOL)
}

impl NetworkStore for FileStore {
    fn name(&self) -> &str {
        "files"
    }

    fn write(&self, network: &Network, run: &JsonValue, dir: &Path) -> MoptResult<()> {
        for node in &network.nodes {
            let path = dir.join(structure_file(node));
            if !path.is_file() {
                write_whole(&path, node_file_text(node).as_bytes())?;
            }
        }
        // The path peaks: their structures, and the table of the records (no part of the
        // network: `path_peaks`).
        let variables = network.variables.is_some();
        let mut peaks = String::from(
            "id,label,level,energy_hartree,end_1,end_2,height_above_end_1_kcal_mol,\
             height_above_end_2_kcal_mol,method,points,path_converged,ts_searches,job,\
             structure_file,path_file,joined_by\n",
        );
        for p in &network.path_peaks {
            let file = super::peak::peak_structure_file(p.id, variables);
            let path = dir.join(&file);
            if !path.is_file() {
                let comment = format!("{} energy={:.12}", p.label(), p.energy);
                let text = match &p.system {
                    super::configuration::Configuration::Variables(v) => {
                        let mut out = format!("# {comment}\n");
                        for (name, x) in v.names.iter().zip(v.values.iter()) {
                            out.push_str(&format!("{name} {x:.12}\n"));
                        }
                        out
                    }
                    other => other.frame(&comment),
                };
                write_whole(&path, text.as_bytes())?;
            }
            let end = |k: usize| -> MoptResult<String> {
                p.ends[k].map_or(Ok(String::new()), |id| Ok(network.node(id)?.label()))
            };
            // The TS edge that joins its two ends by now, if one does.
            let joined = match p.ends {
                [Some(a), Some(b)] => network
                    .ts_edges
                    .iter()
                    .find(|e| e.ends == [a, b] || e.ends == [b, a])
                    .and_then(|e| network.node(e.ts).ok().map(Node::label)),
                _ => None,
            };
            let _ = writeln!(
                peaks,
                "{},{},{},{:.12},{},{},{},{},\"{}\",{},{},{},{},{},{},{}",
                p.id,
                p.label(),
                p.level.number(),
                p.energy,
                end(0)?,
                end(1)?,
                p.barriers[0].map_or_else(String::new, kcal),
                p.barriers[1].map_or_else(String::new, kcal),
                p.basis.method.replace('"', "'"),
                p.basis.points,
                p.basis.converged,
                p.basis.searches,
                opt_text(p.job),
                file,
                super::peak::peak_path_file(p.id, variables),
                joined.unwrap_or_default()
            );
        }
        if !network.path_peaks.is_empty() || dir.join(PATH_PEAKS_FILE).is_file() {
            write_whole(&dir.join(PATH_PEAKS_FILE), peaks.as_bytes())?;
        }
        let low = network.lowest_minimum().unwrap_or(0.0);
        // The energy of the calculator alone is a column only where the surface has static
        // bias potentials, so that the other files stay as they were.
        let bare = network.nodes.iter().any(|n| n.bare_energy.is_some());
        let mut nodes = String::from(
            "id,label,kind,level,energy_hartree,relative_kcal_mol,free_energy_hartree,charge,\
             multiplicity,excited_state,input,job,from_node,visits,excluded,file,smiles",
        );
        nodes.push_str(if bare { ",bare_energy_hartree\n" } else { "\n" });
        let visits = network.visit_counts();
        for n in &network.nodes {
            let _ = write!(
                nodes,
                "{},{},{},{},{:.12},{},{},{},{},{},{},{},{},{},\"{}\",{},\"{}\"",
                n.id,
                n.label(),
                n.kind.tag(),
                n.level.number(),
                n.energy,
                kcal(n.energy - low),
                n.free_energy
                    .map_or_else(String::new, |g| format!("{g:.12}")),
                n.state.charge,
                n.state.multiplicity,
                n.state.excited_state,
                opt_text(n.origin.input),
                opt_text(n.origin.job),
                opt_text(n.origin.from_node),
                visits.get(n.id).copied().unwrap_or(0),
                n.excluded.as_deref().unwrap_or("").replace('"', "'"),
                structure_file(n),
                // The chemical species of a minimum (its conformers share the string).
                n.smiles(crate::workflow::explore::identity::DEFAULT_BOND_SCALE)
                    .unwrap_or("")
            );
            if bare {
                let _ = write!(
                    nodes,
                    ",{}",
                    n.bare_energy
                        .map_or_else(String::new, |e| format!("{e:.12}"))
                );
            }
            nodes.push('\n');
        }
        write_whole(&dir.join(NODES_FILE), nodes.as_bytes())?;
        let mut edges = String::from(
            "id,ts,end_1,end_2,level,ts_energy_hartree,barrier_from_end_1_kcal_mol,\
             barrier_from_end_2_kcal_mol,imaginary_wavenumber_cm1,job,irc_file\n",
        );
        for e in &network.ts_edges {
            let ts = network.node(e.ts)?;
            let _ = writeln!(
                edges,
                "{},{},{},{},{},{:.12},{},{},{},{},{}",
                e.id,
                ts.label(),
                network.node(e.ends[0])?.label(),
                network.node(e.ends[1])?.label(),
                e.level.number(),
                ts.energy,
                kcal(e.barriers[0]),
                kcal(e.barriers[1]),
                e.imaginary_wavenumber
                    .map_or_else(String::new, |w| format!("{w:.2}")),
                opt_text(e.job),
                e.irc.file.as_deref().unwrap_or("")
            );
        }
        write_whole(&dir.join(TS_EDGES_FILE), edges.as_bytes())?;
        let mut crossings = String::from(
            "id,crossing,end_1,end_2,level,energy_hartree,barrier_from_end_1_kcal_mol,\
             barrier_from_end_2_kcal_mol,gradient_difference,job,gap_hartree,\
             seam_imaginary_modes,seam_check\n",
        );
        for e in &network.crossing_edges {
            let c = network.node(e.crossing)?;
            let end = |k: usize| -> MoptResult<String> {
                e.ends[k].map_or(Ok(String::new()), |id| Ok(network.node(id)?.label()))
            };
            let (modes, said) = match &e.seam {
                Some(s @ super::network::SeamCheck::Checked(n)) => (n.to_string(), s.describe()),
                Some(s) => (String::new(), s.describe()),
                None => (String::new(), String::new()),
            };
            let _ = writeln!(
                crossings,
                "{},{},{},{},{},{:.12},{},{},{},{},{},{},\"{}\"",
                e.id,
                c.label(),
                end(0)?,
                end(1)?,
                e.level.number(),
                c.energy,
                e.barriers[0].map_or_else(String::new, kcal),
                e.barriers[1].map_or_else(String::new, kcal),
                opt_text(e.gradient_difference),
                opt_text(e.job),
                e.gap.map_or_else(String::new, |g| format!("{g:.6e}")),
                modes,
                said.replace('"', "'")
            );
        }
        write_whole(&dir.join(CROSSING_EDGES_FILE), crossings.as_bytes())?;
        let mut dissociations =
            String::from("id,complex,channel,level,energy_difference_kcal_mol,fragments,job\n");
        for e in &network.dissociation_edges {
            let channel = network.node(e.channel)?;
            let fragments: Vec<String> = channel
                .parts
                .iter()
                .map(|p| {
                    network
                        .fragments
                        .get(p.fragment)
                        .map_or_else(|| "?".to_string(), |f| f.formula.clone())
                })
                .collect();
            let _ = writeln!(
                dissociations,
                "{},{},{},{},{},{},{}",
                e.id,
                network.node(e.minimum)?.label(),
                channel.label(),
                e.level.number(),
                kcal(e.energy_difference),
                fragments.join(" + "),
                opt_text(e.job)
            );
        }
        write_whole(&dir.join(DISSOCIATION_EDGES_FILE), dissociations.as_bytes())?;
        write_whole(
            &dir.join(NETWORK_FILE),
            (to_json(network, run.clone()).to_string_pretty() + "\n").as_bytes(),
        )
    }
}

/// The number of the path peak a file of `structures/` or `paths/` is of (`PP000003.xyz`,
/// `PP000003_path.xyz`); `None` for the file of a node.
fn peak_of_file(name: &str) -> Option<usize> {
    let stem = name.split('.').next()?;
    let stem = stem.strip_suffix("_path").unwrap_or(stem);
    stem.strip_prefix("PP")?.parse().ok()
}

/// The number of the node a file of `structures/` or `paths/` is of (`EQ000012.xyz`,
/// `TS000007_irc.xyz`).
fn node_of_file(name: &str) -> Option<usize> {
    let stem = name.strip_suffix(".xyz")?;
    let stem = stem.strip_suffix("_irc").unwrap_or(stem);
    let digits = stem.len() - stem.trim_start_matches(|c: char| !c.is_ascii_digit()).len();
    stem.get(digits..)?.parse().ok()
}

/// Remove what a run that was killed may have left beyond `network` in the run directory
/// `dir`: the structures and IRC paths of nodes the network does not have (written before
/// the network that would have had them), and files that were being written (`*.part`).
/// Returns how many files were removed.
pub fn forget_beyond(dir: &Path, network: &Network) -> MoptResult<usize> {
    let mut removed = 0;
    for sub in [
        dir.to_path_buf(),
        dir.join(STRUCTURES_DIR),
        dir.join(super::observers::PATHS_DIR),
    ] {
        let Ok(entries) = std::fs::read_dir(&sub) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let stale = name.ends_with(".part")
                || (sub != dir
                    && match peak_of_file(&name) {
                        Some(id) => id >= network.path_peaks.len(),
                        None => node_of_file(&name).is_some_and(|id| id >= network.nodes.len()),
                    });
            if stale && entry.path().is_file() {
                std::fs::remove_file(entry.path())
                    .map_err(|e| MoptError::from(e).context(entry.path().display()))?;
                removed += 1;
            }
        }
    }
    Ok(removed)
}

/// The network and the state of the run of the run directory `dir`, if it has one.
pub fn read_network(dir: &Path) -> MoptResult<Option<(Network, JsonValue)>> {
    let path = dir.join(NETWORK_FILE);
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(&path).map_err(|e| MoptError::from(e).context(path.display()))?;
    let doc = JsonValue::parse(&text).context(path.display())?;
    from_json(&doc).context(path.display()).map(Some)
}

/// The network and the state of the run of the run directory `dir` as far as its network file
/// can be read ([`super::network_json::from_json_lenient`]), with notes of what was left out;
/// `None` when there is no network file.
pub fn read_network_lenient(dir: &Path) -> MoptResult<Option<(Network, JsonValue, Vec<String>)>> {
    let path = dir.join(NETWORK_FILE);
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(&path).map_err(|e| MoptError::from(e).context(path.display()))?;
    let doc = JsonValue::parse(&text).context(path.display())?;
    super::network_json::from_json_lenient(&doc)
        .context(path.display())
        .map(Some)
}

/// The first frame of the xyz text: its comment and atoms (Angstrom).
fn first_frame(text: &str) -> Option<(String, Vec<String>, Vec<[f64; 3]>)> {
    let mut lines = text.lines().skip_while(|l| l.trim().is_empty());
    let n: usize = lines.next()?.trim().parse().ok()?;
    let comment = lines.next()?.to_string();
    let mut symbols = Vec::with_capacity(n);
    let mut points = Vec::with_capacity(n);
    for line in lines.take(n) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let at = |k: usize| f.get(k).and_then(|v| v.parse::<f64>().ok());
        symbols.push((*f.first()?).to_string());
        points.push([at(1)?, at(2)?, at(3)?]);
    }
    (points.len() == n && n > 0).then_some((comment, symbols, points))
}

/// The network as far as the files beside the network file give it, for a run directory whose
/// network file is missing or cannot be read: the minima and transition states of
/// `structures/` (`EQ000003.xyz`, `TS000007.xyz`, ...: label from the name, energy from the
/// comment) and the links of `ts_edges.csv` between them (without the energies along their
/// IRCs). Never fails: what cannot be read is left out with a note.
pub fn network_from_files(dir: &Path) -> (Network, Vec<String>) {
    use super::network::{ElectronicState, IrcPath, Level, NodeKind, Origin, TsEdge};
    use crate::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
    let mut notes = Vec::new();
    let mut found: Vec<(u8, usize, std::path::PathBuf)> = Vec::new();
    match std::fs::read_dir(dir.join(STRUCTURES_DIR)) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let Some(stem) = name.strip_suffix(".xyz") else {
                    continue;
                };
                let (kind, digits) = if let Some(d) = stem.strip_prefix("EQ") {
                    (0, d)
                } else if let Some(d) = stem.strip_prefix("TS") {
                    (1, d)
                } else {
                    notes.push(format!(
                        "{STRUCTURES_DIR}/{name} is left out (only minima and transition states \
                         are read back from the structures)"
                    ));
                    continue;
                };
                match digits.parse::<usize>() {
                    Ok(number) => found.push((kind, number, entry.path())),
                    Err(_) => notes.push(format!("{STRUCTURES_DIR}/{name} is left out")),
                }
            }
        }
        Err(e) => notes.push(format!("{STRUCTURES_DIR}/ could not be read: {e}")),
    }
    found.sort();
    let mut network = Network::default();
    let mut by_label = std::collections::HashMap::new();
    for (kind, number, path) in found {
        let shown = path.display().to_string();
        let Some((comment, symbols, points)) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| first_frame(&t))
        else {
            notes.push(format!("{shown} is left out: not an xyz structure"));
            continue;
        };
        let energy = comment.split_whitespace().find_map(|w| {
            w.strip_prefix("energy=")
                .and_then(|v| v.parse::<f64>().ok())
        });
        let Some(energy) = energy else {
            notes.push(format!("{shown} is left out: its comment gives no energy"));
            continue;
        };
        let elements: Option<Vec<Element>> = symbols
            .iter()
            .map(|s| Element::parse_token(s).ok())
            .collect();
        let system = elements.and_then(|el| {
            MolecularSystem::new(
                el,
                &positions::<Angstrom>(&points),
                SpinCharge::default(),
                None,
            )
            .ok()
        });
        let Some(system) = system else {
            notes.push(format!("{shown} is left out: its atoms could not be read"));
            continue;
        };
        if network.nodes.is_empty() {
            network.elements = system.elements.clone();
        } else if system.elements != network.elements {
            notes.push(format!(
                "{shown} is left out: other atoms than the other structures"
            ));
            continue;
        }
        let state = ElectronicState::of(&system, 0);
        let node_kind = if kind == 0 {
            NodeKind::Minimum
        } else {
            NodeKind::TransitionState
        };
        let node = Node::new(
            node_kind,
            Level::Exploration,
            system.into(),
            energy,
            state,
            Origin::default(),
        );
        let id = network.nodes.len();
        network.restore_node(node, Some(number));
        if let Some(n) = network.nodes.get(id) {
            by_label.insert(n.label(), id);
        }
    }
    let table = dir.join(TS_EDGES_FILE);
    match std::fs::read_to_string(&table) {
        Ok(text) => {
            let mut lines = text.lines();
            let header: Vec<&str> = lines.next().unwrap_or_default().split(',').collect();
            let col = |name: &str| header.iter().position(|h| *h == name);
            let (ts, a, b) = (col("ts"), col("end_1"), col("end_2"));
            let (wave, irc, job) = (col("imaginary_wavenumber_cm1"), col("irc_file"), col("job"));
            for (k, line) in lines.enumerate() {
                let f: Vec<&str> = line.split(',').collect();
                let get = |c: Option<usize>| c.and_then(|c| f.get(c)).map(|s| s.trim());
                let id = |c: Option<usize>| get(c).and_then(|l| by_label.get(l).copied());
                let (Some(t), Some(x), Some(y)) = (id(ts), id(a), id(b)) else {
                    notes.push(format!(
                        "row {} of {TS_EDGES_FILE} is left out: a structure of it is missing",
                        k + 1
                    ));
                    continue;
                };
                let energy = |i: usize| network.nodes.get(i).map_or(f64::NAN, |n| n.energy);
                let edge = TsEdge {
                    id: 0,
                    level: Level::Exploration,
                    ts: t,
                    ends: [x, y],
                    barriers: [energy(t) - energy(x), energy(t) - energy(y)],
                    imaginary_wavenumber: get(wave).and_then(|w| w.parse().ok()),
                    irc: IrcPath {
                        sides: Vec::new(),
                        file: get(irc).filter(|s| !s.is_empty()).map(str::to_string),
                        possible_bifurcations: Vec::new(),
                    },
                    job: get(job).and_then(|j| j.parse().ok()),
                };
                if let Err(e) = network.add_ts_edge(edge) {
                    notes.push(format!("row {} of {TS_EDGES_FILE} is left out: {e}", k + 1));
                }
            }
        }
        Err(_) => notes.push(format!(
            "{TS_EDGES_FILE} could not be read: the transition states are shown without their \
             links"
        )),
    }
    (network, notes)
}

/// Makes a store.
pub type StoreFactory = Arc<dyn Fn() -> MoptResult<Box<dyn NetworkStore>> + Send + Sync>;

/// Stores by name.
pub struct StoreRegistry {
    entries: Vec<(String, StoreFactory)>,
}

impl StoreRegistry {
    /// `files` and `sqlite` ([`super::sqlite_store::SqliteStore`]).
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register(
            "files",
            Arc::new(|| Ok(Box::new(FileStore) as Box<dyn NetworkStore>)),
        );
        r.register(
            "sqlite",
            Arc::new(|| Ok(Box::new(super::sqlite_store::SqliteStore) as Box<dyn NetworkStore>)),
        );
        r
    }

    /// Register (or replace) a store.
    pub fn register(&mut self, name: &str, factory: StoreFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// The registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// The store `name`.
    pub fn create(&self, name: &str) -> MoptResult<Box<dyn NetworkStore>> {
        let key = name.to_ascii_lowercase();
        let factory = self
            .entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, f)| f)
            .ok_or_else(|| {
                MoptError::parse(format!(
                    "unknown store '{name}' of the mapper (available: {})",
                    self.names().join(", ")
                ))
            })?;
        factory()
    }
}

impl Default for StoreRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_stores_are_found_by_name_and_files_are_whole() {
        let reg = StoreRegistry::default();
        assert_eq!(reg.create("files").unwrap().name(), "files");
        assert_eq!(reg.create("SQLite").unwrap().name(), "sqlite");
        assert!(reg.create("csv").is_err());
        let dir = std::env::temp_dir().join(format!("mopt_mapper_store_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let net = Network::new(Vec::new());
        FileStore.write(&net, &JsonValue::Null, &dir).unwrap();
        let (back, _) = read_network(&dir).unwrap().unwrap();
        assert!(back.nodes.is_empty());
        assert!(
            dir.join(TS_EDGES_FILE).is_file() && !dir.join("reaction_network.json.part").exists()
        );
        assert!(read_network(&dir.join("none")).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn what_a_killed_run_wrote_beyond_its_network_is_removed() {
        let dir = std::env::temp_dir().join(format!("mopt_mapper_stale_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let net = Network::new(Vec::new());
        FileStore.write(&net, &JsonValue::Null, &dir).unwrap();
        std::fs::create_dir_all(dir.join(STRUCTURES_DIR)).unwrap();
        std::fs::create_dir_all(dir.join("paths")).unwrap();
        for f in [
            "structures/EQ000000.xyz",
            "structures/TS000003.xyz.part",
            "paths/TS000001_irc.xyz",
            "reaction_network.json.part",
            "notes.txt",
        ] {
            std::fs::write(dir.join(f), "x").unwrap();
        }
        assert_eq!(node_of_file("TS000012_irc.xyz"), Some(12));
        assert_eq!(node_of_file("MECP000004.xyz"), Some(4));
        assert_eq!(node_of_file("notes.txt"), None);
        // The network has no node: every structure and path goes, and every part file.
        assert_eq!(forget_beyond(&dir, &net).unwrap(), 4);
        assert!(dir.join("notes.txt").is_file() && dir.join(NETWORK_FILE).is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
