//! `run_mapper -remap RUN_DIR`: the first network as it is read (the run directory of a run
//! that ended or was stopped, or the database of one), its minima as the inputs of the run,
//! its edges and saddle points of higher order as the items of
//! [`crate::workflow::mapper::remap`], and the table `remap.csv` of what each of them became
//! in the network of the run. The first run is only read. The rules are in
//! `docs/source/mapper_design.md`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::domain::Units;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::workflow::mapper::configuration::variable_frames;
use crate::workflow::mapper::driver::LEFT_OUT;
use crate::workflow::mapper::network::{JobStatus, Level, Network, Node, NodeKind};
use crate::workflow::mapper::remap::{
    thinned, RemapItem, RemapKind, RemapSource, CHAIN_FRAMES, REMAP_PREFIX,
};
use crate::workflow::mapper::sqlite_store::{read_sqlite, SQLITE_FILE};
use crate::workflow::mapper::store::{read_network, structure_file, write_whole, NETWORK_FILE};
use crate::workflow::mapper::Configuration;

/// The table of what the items of the first network became, in the run directory.
pub const REMAP_TABLE: &str = "remap.csv";

/// The first network of a remap.
#[derive(Debug, Clone)]
pub struct FirstNetwork {
    /// What was given (`-remap`).
    pub given: PathBuf,
    /// The directory of its files (`structures/`, `paths/`).
    pub dir: PathBuf,
    /// The network.
    pub network: Network,
    /// Its minima, which are the inputs of the run in this order (node ids).
    pub minima: Vec<usize>,
}

impl FirstNetwork {
    /// The network of `path`: a run directory (its `reaction_network.json`, or its database
    /// when it has no such file), or the database of a run.
    pub fn read(path: &Path) -> MoptResult<Self> {
        let (dir, network) = if path.is_file() {
            let (network, _, _) = read_sqlite(path)?;
            let dir = path
                .parent()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
            (dir, network)
        } else if let Some((network, _)) = read_network(path)? {
            (path.to_path_buf(), network)
        } else if path.join(SQLITE_FILE).is_file() {
            let (network, _, _) = read_sqlite(&path.join(SQLITE_FILE))?;
            (path.to_path_buf(), network)
        } else {
            return Err(MoptError::config(format!(
                "run_mapper: -remap {}: no network there ({NETWORK_FILE} or {SQLITE_FILE} of a \
                 run directory, or the database of a run)",
                path.display()
            )));
        };
        let minima: Vec<usize> = network
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Minimum && n.level == Level::Exploration)
            .map(|n| n.id)
            .collect();
        if minima.is_empty() {
            return Err(MoptError::config(format!(
                "run_mapper: -remap {}: the network has no minimum",
                path.display()
            )));
        }
        Ok(Self {
            given: path.to_path_buf(),
            dir,
            network,
            minima,
        })
    }

    /// The file of the structure of its first minimum (the settings of the steps are made
    /// with a structure file).
    pub fn first_file(&self) -> MoptResult<PathBuf> {
        let node = self.network.node(self.minima[0])?;
        let path = self.dir.join(structure_file(node));
        if path.is_file() {
            Ok(path)
        } else {
            Err(MoptError::config(format!(
                "run_mapper: -remap {}: the structure file {} of its first minimum is missing \
                 (the files of the run directory are read beside its network)",
                self.given.display(),
                path.display()
            )))
        }
    }

    /// The inputs of the run: the name, the structure and the state (its index among the
    /// states of the network; `None` for a network of one state) of every minimum.
    pub fn inputs(&self) -> MoptResult<Vec<(String, Configuration, Option<usize>)>> {
        self.minima
            .iter()
            .map(|&id| {
                let node = self.network.node(id)?;
                let state = self.network.states.iter().position(|s| *s == node.state);
                Ok((
                    format!("{}: {}", self.given.display(), node.label()),
                    node.system.clone(),
                    state,
                ))
            })
            .collect()
    }

    /// The inputs of the network as it has them (file, frame, structure, whether it is the
    /// goal): those of its first run, each at the node it became; the minima of the network
    /// when it records none.
    pub fn recorded_inputs(&self) -> MoptResult<Vec<(String, usize, Configuration, bool)>> {
        let mut out = Vec::new();
        for record in &self.network.inputs {
            if let Some(id) = record.node {
                out.push((
                    record.file.clone(),
                    record.frame,
                    self.network.node(id)?.system.clone(),
                    record.goal,
                ));
            }
        }
        if out.len() != self.network.inputs.len() || out.is_empty() {
            return Ok(self
                .inputs()?
                .into_iter()
                .map(|(file, system, _)| (file, 0, system, false))
                .collect());
        }
        Ok(out)
    }

    /// The structures of the path file `file` of the run directory, as those of `like`
    /// (its charge and multiplicity, or its variables); `None` when the file is missing.
    fn path_frames(&self, file: &str, like: &Node) -> MoptResult<Option<Vec<Configuration>>> {
        let path = self.dir.join(file);
        if !path.is_file() {
            return Ok(None);
        }
        let frames: Vec<Configuration> = match &like.system {
            Configuration::Structure(sys) => {
                let read = crate::io::xyz::read_xyz_trajectory_file(&path, sys.spin_charge, None)
                    .context(path.display())?;
                let mut out = Vec::with_capacity(read.frames.len());
                for f in read.frames {
                    if f.system.elements != sys.elements {
                        return Err(MoptError::parse(format!(
                            "{}: a frame has other atoms than the network",
                            path.display()
                        )));
                    }
                    out.push(Configuration::Structure(f.system));
                }
                out
            }
            Configuration::Variables(_) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| MoptError::from(e).context(path.display()))?;
                variable_frames(&text)
                    .context(path.display())?
                    .into_iter()
                    .map(|(_, p)| Configuration::Variables(p))
                    .collect()
            }
        };
        Ok(Some(thinned(frames, CHAIN_FRAMES)))
    }

    /// The items of the run (every transition state, crossing point and saddle point of
    /// higher order of the network once) and what was left out, in words.
    pub fn source(&self) -> MoptResult<(RemapSource, Vec<String>)> {
        let net = &self.network;
        let input_of: HashMap<usize, usize> = self
            .minima
            .iter()
            .enumerate()
            .map(|(k, &id)| (id, k))
            .collect();
        let label = |id: usize| {
            net.node(id)
                .map_or_else(|_| format!("node {id}"), Node::label)
        };
        let mut notes = Vec::new();
        let mut items = Vec::new();
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        for e in net
            .ts_edges
            .iter()
            .filter(|e| e.level == Level::Exploration)
        {
            if !seen.insert(e.ts) {
                continue;
            }
            let ts = net.node(e.ts)?;
            let ends = [
                input_of.get(&e.ends[0]).copied(),
                input_of.get(&e.ends[1]).copied(),
            ];
            let chain = match e.irc.file.as_deref() {
                Some(file) => match self.path_frames(file, ts)? {
                    Some(frames) => frames,
                    None => {
                        notes.push(format!(
                            "The path file {file} of {} is missing: its chain is its ends and \
                             its transition state.",
                            ts.label()
                        ));
                        Vec::new()
                    }
                },
                None => Vec::new(),
            };
            items.push(RemapItem {
                key: format!("ts:{}", e.ts),
                label: format!(
                    "{} between {} and {}",
                    ts.label(),
                    label(e.ends[0]),
                    label(e.ends[1])
                ),
                kind: RemapKind::Transition,
                runner: ends[0].or(ends[1]).unwrap_or(0),
                ends,
                point: ts.system.clone(),
                chain,
            });
        }
        let open = net
            .open_ts_edges
            .iter()
            .filter(|e| e.level == Level::Exploration && !seen.contains(&e.ts))
            .count();
        if open > 0 {
            notes.push(format!(
                "{open} transition state(s) with an end of the IRC that is not known are left \
                 out: the path of a remap needs both minima."
            ));
        }
        for e in net
            .crossing_edges
            .iter()
            .filter(|e| e.level == Level::Exploration)
        {
            if !seen.insert(e.crossing) {
                continue;
            }
            let node = net.node(e.crossing)?;
            let NodeKind::Crossing { states, .. } = &node.kind else {
                continue;
            };
            let index = |s: &crate::workflow::mapper::network::ElectronicState| {
                net.states.iter().position(|t| t == s)
            };
            let (Some(a), Some(b)) = (index(&states[0]), index(&states[1])) else {
                notes.push(format!(
                    "{} is between states that are not among the states of its network: it is \
                     left out.",
                    node.label()
                ));
                continue;
            };
            let ends = [
                e.ends[0].and_then(|id| input_of.get(&id).copied()),
                e.ends[1].and_then(|id| input_of.get(&id).copied()),
            ];
            // The job runs on one of the two states: from the minimum of a side, or from the
            // first minimum of the network on one of them.
            let on = |state: usize| {
                self.minima.iter().position(|&id| {
                    net.node(id)
                        .is_ok_and(|n| net.states.get(state) == Some(&n.state))
                })
            };
            let Some(runner) = ends[0].or(ends[1]).or_else(|| on(a)).or_else(|| on(b)) else {
                notes.push(format!(
                    "{} has no minimum on either of its states in the network: it is left out.",
                    node.label()
                ));
                continue;
            };
            let side = |end: Option<usize>| end.map_or_else(|| "?".to_string(), label);
            items.push(RemapItem {
                key: format!("x:{}", e.crossing),
                label: format!(
                    "{} between {} and {}",
                    node.label(),
                    side(e.ends[0]),
                    side(e.ends[1])
                ),
                kind: RemapKind::Crossing([a, b]),
                runner,
                ends,
                point: node.system.clone(),
                chain: Vec::new(),
            });
        }
        for node in &net.nodes {
            let NodeKind::HigherSaddle { order } = node.kind else {
                continue;
            };
            if node.level != Level::Exploration {
                continue;
            }
            let runner = node
                .origin
                .from_node
                .and_then(|id| input_of.get(&id).copied())
                .unwrap_or(0);
            items.push(RemapItem {
                key: format!("sp:{}", node.id),
                label: node.label(),
                kind: RemapKind::Saddle(usize::from(order)),
                runner,
                ends: [None, None],
                point: node.system.clone(),
                chain: Vec::new(),
            });
        }
        Ok((
            RemapSource {
                minima: self.minima.clone(),
                items,
            },
            notes,
        ))
    }

    /// The highest order of its saddle points of higher order (`None`: it has none).
    pub fn highest_order(&self) -> Option<usize> {
        self.network
            .nodes
            .iter()
            .filter_map(|n| match n.kind {
                NodeKind::HigherSaddle { order } => Some(usize::from(order)),
                _ => None,
            })
            .max()
    }

    /// Write [`REMAP_TABLE`] of the run directory `dir`: every minimum and item of the first
    /// network with its job in the network `second` of the run, what the job added, and the
    /// node of its kind in `second` that is nearest to it. Returns how many items had a job,
    /// and how many of those jobs added a node of the kind of their item.
    pub fn write_table(
        &self,
        dir: &Path,
        source: &RemapSource,
        second: &Network,
    ) -> MoptResult<(usize, usize)> {
        let first = &self.network;
        let kcal = Units::HARTREE_TO_KCAL_MOL;
        let (low1, low2) = (
            first.lowest_minimum().unwrap_or(0.0),
            second.lowest_minimum().unwrap_or(0.0),
        );
        let mut text = String::from(
            "item,kind,first_label,first_energy_hartree,first_relative_kcal_mol,job,status,\
             added,nearest,nearest_distance,nearest_energy_hartree,nearest_relative_kcal_mol\n",
        );
        let labels = |ids: &[usize]| -> String {
            ids.iter()
                .filter_map(|&id| second.node(id).ok())
                .map(Node::label)
                .collect::<Vec<_>>()
                .join(";")
        };
        // The node among `candidates` nearest to `point`, with the distance.
        let nearest = |point: &Configuration, candidates: &mut dyn Iterator<Item = &Node>| {
            let mut best: Option<(&Node, f64)> = None;
            for n in candidates {
                if let Some(d) = distance(point, &n.system) {
                    if best.is_none_or(|(_, b)| d < b) {
                        best = Some((n, d));
                    }
                }
            }
            match best {
                Some((n, d)) => format!(
                    "{},{d:.4},{:.12},{:.4}",
                    n.label(),
                    n.energy,
                    (n.energy - low2) * kcal
                ),
                None => ",,,".to_string(),
            }
        };
        for (k, &id) in self.minima.iter().enumerate() {
            let n = first.node(id)?;
            let node = second
                .inputs
                .get(k)
                .and_then(|i| i.node)
                .and_then(|id| second.node(id).ok());
            let _ = writeln!(
                text,
                "eq:{id},EQ,{},{:.12},{:.4},,{},{},{}",
                n.label(),
                n.energy,
                (n.energy - low1) * kcal,
                if node.is_some() {
                    "minimized"
                } else if second.warnings.iter().any(|w| {
                    w.job.is_none()
                        && w.message.starts_with(&format!("input {} (", k + 1))
                        && w.message.contains(LEFT_OUT)
                }) {
                    "left out"
                } else {
                    "not minimized yet"
                },
                node.map_or_else(String::new, Node::label),
                nearest(&n.system, &mut node.into_iter()),
            );
        }
        let (mut ran, mut added) = (0, 0);
        for item in &source.items {
            let key = format!("{REMAP_PREFIX}{}", item.key);
            let old = item
                .key
                .split_once(':')
                .and_then(|(_, id)| id.parse::<usize>().ok())
                .and_then(|id| first.node(id).ok());
            let of_kind = |n: &Node| match (&item.kind, &n.kind) {
                (RemapKind::Transition, NodeKind::TransitionState)
                | (RemapKind::Crossing(_), NodeKind::Crossing { .. }) => true,
                (RemapKind::Saddle(order), NodeKind::HigherSaddle { order: other }) => {
                    *order == usize::from(*other)
                }
                _ => false,
            };
            let kind = match item.kind {
                RemapKind::Transition => "TS",
                RemapKind::Crossing(_) => "crossing",
                RemapKind::Saddle(_) => "saddle",
            };
            let job = second.jobs.iter().find(|j| j.task == key);
            let (status, ids): (&str, Vec<usize>) = match job {
                None => ("not run", Vec::new()),
                Some(j) => {
                    ran += 1;
                    // What the job added of the kind of the item: the transition states of
                    // its new edges, its new crossing points, its new saddle points.
                    let mut ids: Vec<usize> = j
                        .new_edges
                        .iter()
                        .filter_map(|&e| second.ts_edges.get(e).map(|e| e.ts))
                        .chain(j.new_nodes.iter().copied())
                        .filter(|&id| second.node(id).is_ok_and(of_kind))
                        .collect();
                    ids.sort_unstable();
                    ids.dedup();
                    if !ids.is_empty() {
                        added += 1;
                    }
                    // A job that gave nothing of the kind of its item says so in a warning.
                    let nothing = second
                        .warnings
                        .iter()
                        .any(|w| w.job == Some(j.id) && w.message.contains(LEFT_OUT));
                    let status = match (&j.status, ids.is_empty()) {
                        (JobStatus::Failed(_), _) => "failed",
                        (JobStatus::Done, false) => "added",
                        (JobStatus::Done, true) if nothing => "left out",
                        (JobStatus::Done, true) => "known",
                    };
                    (status, ids)
                }
            };
            let _ = writeln!(
                text,
                "{},{kind},{},{},{},{},{status},{},{}",
                item.key,
                item.label.replace(',', ";"),
                old.map_or_else(String::new, |n| format!("{:.12}", n.energy)),
                old.map_or_else(String::new, |n| format!("{:.4}", (n.energy - low1) * kcal)),
                job.map_or_else(String::new, |j| j.id.to_string()),
                labels(&ids),
                nearest(&item.point, &mut second.nodes.iter().filter(|n| of_kind(n))),
            );
        }
        write_whole(&dir.join(REMAP_TABLE), text.as_bytes())?;
        Ok((ran, added))
    }
}

/// The distance of two points: for structures the root-mean-square distance of the atoms
/// after the best superposition and numbering of identical atoms (Angstrom), for points of
/// plain variables the Euclidean distance; `None` when they cannot be compared.
fn distance(a: &Configuration, b: &Configuration) -> Option<f64> {
    match (a.structure(), b.structure()) {
        (Some(x), Some(y)) => {
            let classes = |s: &crate::domain::MolecularSystem| -> Vec<usize> {
                s.elements
                    .iter()
                    .map(|e| usize::from(e.atomic_number()))
                    .collect()
            };
            crate::coord::invariant_rmsd::invariant_rmsd(
                x,
                y,
                &classes(x),
                &classes(y),
                &crate::coord::invariant_rmsd::RmsdOptions::default(),
            )
            .ok()
            .map(|m| m.rmsd * Units::BOHR_TO_ANGSTROM)
        }
        _ => a.variable_distance(b),
    }
}
