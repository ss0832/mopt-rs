//! What the report of `run_mapper` shows: the options of the page (energy unit, temperature,
//! energy window, how much of the network is drawn and put in the page) and the view of the
//! network they give (the nodes shown, their energies in the unit, the Boltzmann populations of
//! the minima, the places of the map and of the graph, the disconnectivity graph and the
//! lowest-barrier paths, and the data of the page's script).
//!
//! The view is built for networks of tens of thousands of minima and transition states: every
//! lookup goes through maps, the map of more than `map_exact` structures is landmark
//! multidimensional scaling (de Silva and Tenenbaum, "Sparse multidimensional scaling using
//! landmark points", Stanford 2004: classical scaling of `map_landmarks` structures, every other
//! structure placed by Gower's formula from its distances to them, O(n L) distances instead of
//! O(n^2)), the graph is laid out in O(n) steps ([`crate::analysis::graph_layout`]), the
//! disconnectivity graph and the energy diagram draw the lowest `figure_minima` minima, the
//! data of the script are columns of numbers, and the energies along the IRCs are in the page
//! up to a budget of points.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use rayon::prelude::*;

use crate::analysis::disconnectivity::{
    disconnectivity, highest_on_the_way, prune, Connection, Disconnectivity,
};
use crate::analysis::graph_layout::{force_layout, LayoutSettings};
use crate::analysis::path_map::{classical_mds, distance_matrix, mass_weighted_distance};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::workflow::mapper::network::{ElectronicState, Network, Node, NodeKind};
use crate::workflow::mapper::Configuration;

/// The report in the run directory.
pub const REPORT_FILE: &str = "mapper_report.html";

/// The unit of the energies of the page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnergyUnit {
    /// kcal/mol (the default).
    #[default]
    KcalMol,
    /// kJ/mol.
    KjMol,
    /// eV.
    Ev,
    /// Hartree.
    Hartree,
    /// The values of an objective of plain variables as they are (the default of a network of
    /// plain variables, whose values have no physical unit).
    Value,
}

impl EnergyUnit {
    /// `kcal`, `kj`, `ev`, or `hartree` (and their spellings with `/mol`).
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "kcal" | "kcal/mol" | "kcal_mol" | "kcalmol" => Ok(Self::KcalMol),
            "kj" | "kj/mol" | "kj_mol" | "kjmol" => Ok(Self::KjMol),
            "ev" => Ok(Self::Ev),
            "hartree" | "eh" | "au" => Ok(Self::Hartree),
            "value" | "objective" => Ok(Self::Value),
            other => Err(MoptError::parse(format!(
                "-report_opt unit={other}: kcal, kj, ev, hartree, or value"
            ))),
        }
    }

    /// Hartree to this unit.
    pub fn factor(self) -> f64 {
        match self {
            Self::KcalMol => Units::HARTREE_TO_KCAL_MOL,
            Self::KjMol => Units::HARTREE_TO_KJ_MOL,
            Self::Ev => Units::HARTREE_TO_EV,
            Self::Hartree | Self::Value => 1.0,
        }
    }

    /// The name of the unit on the page.
    pub fn label(self) -> &'static str {
        match self {
            Self::KcalMol => "kcal/mol",
            Self::KjMol => "kJ/mol",
            Self::Ev => "eV",
            Self::Hartree => "Hartree",
            Self::Value => "value",
        }
    }

    /// Digits after the point of relative energies.
    pub fn digits(self) -> usize {
        match self {
            Self::Hartree | Self::Value => 6,
            Self::Ev => 3,
            Self::KcalMol | Self::KjMol => 2,
        }
    }
}

/// The options of the page (`-report_opt KEY=VALUE`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReportOptions {
    /// The unit of the energies.
    pub unit: EnergyUnit,
    /// Whether `unit` was given (otherwise a network of plain variables shows its values as
    /// they are).
    pub unit_given: bool,
    /// The temperature of the Boltzmann populations (K).
    pub temperature: f64,
    /// Only nodes at most this far above the lowest minimum (in `unit`); `None`: all.
    pub energy_window: Option<f64>,
    /// The most minima shown (the lowest first), and the links between them.
    pub max_minima: usize,
    /// The minima of the disconnectivity graph and the energy diagram, and of the figures
    /// drawn for pages read without scripts (the lowest first).
    pub figure_minima: usize,
    /// Rows of each table in the page for reading without scripts (the script pages through
    /// all).
    pub table_rows: usize,
    /// The most nodes with their structure in the page (for the viewer of a node, the lowest
    /// first); the others are in the files of the run.
    pub structures: usize,
    /// The most atoms of all structures in the page together.
    pub structure_atoms: usize,
    /// The most atoms of the further structures that the page holds packed (unpacked by the
    /// browser when a node is chosen or its structure saved), after those of `structures`
    /// and `structure_atoms`; every one by default.
    pub structure_packed_atoms: usize,
    /// The most atom lines of the IRC paths in the page (the lowest transition states first,
    /// at most 30 frames each; for the viewer and the downloads of the page).
    pub irc_lines: usize,
    /// The most atom lines of the further IRC paths that the page holds packed (compressed,
    /// unpacked by the browser when one is opened or saved), after those of `irc_lines`;
    /// every one by default.
    pub irc_packed_lines: usize,
    /// The most points of the energies along the IRCs in the page (the lowest transition
    /// states first, at most 40 points each).
    pub profile_points: usize,
    /// Up to this many structures the map is classical scaling of all distances; above, it is
    /// landmark scaling.
    pub map_exact: usize,
    /// The landmarks of landmark scaling.
    pub map_landmarks: usize,
    /// The name of the page in the run directory.
    pub file: String,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            unit: EnergyUnit::default(),
            unit_given: false,
            temperature: 298.15,
            energy_window: None,
            max_minima: usize::MAX,
            figure_minima: 300,
            table_rows: 300,
            structures: 400,
            structure_atoms: 100_000,
            structure_packed_atoms: usize::MAX,
            irc_lines: 60_000,
            irc_packed_lines: usize::MAX,
            profile_points: 100_000,
            map_exact: 1000,
            map_landmarks: 400,
            file: REPORT_FILE.to_string(),
        }
    }
}

/// The keys of `-report_opt`.
const KEYS: &str = "unit, temperature, energy_window, max_minima, figure_minima, table_rows, \
                    structures, structure_atoms, structure_packed_atoms, irc_lines, \
                    irc_packed_lines, \
                    profile_points, map_exact, map_landmarks, file";

impl ReportOptions {
    /// The options of `KEY=VALUE` items (a later item overrides an earlier one; the keys are
    /// the fields of the options, `unit` to `file`).
    pub fn parse(items: &[String]) -> MoptResult<Self> {
        let mut out = Self::default();
        for item in items {
            out.apply(item)?;
        }
        Ok(out)
    }

    /// [`Self::parse`] for a page written from the files of a run: an item that cannot be read
    /// is left out, with a note.
    pub fn parse_lenient(items: &[String]) -> (Self, Vec<String>) {
        let mut out = Self::default();
        let mut notes = Vec::new();
        for item in items {
            if let Err(e) = out.apply(item) {
                notes.push(format!("{e}; the setting is left out"));
            }
        }
        (out, notes)
    }

    /// Take the setting `item` (`KEY=VALUE`).
    fn apply(&mut self, item: &str) -> MoptResult<()> {
        let out = self;
        {
            let (key, value) = item.split_once('=').ok_or_else(|| {
                MoptError::parse(format!("-report_opt takes KEY=VALUE, not '{item}'"))
            })?;
            let (key, value) = (key.trim(), value.trim());
            let all = matches!(value.to_ascii_lowercase().as_str(), "none" | "all" | "off");
            let number = || {
                value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| {
                        MoptError::parse(format!("-report_opt {key}={value}: not a number"))
                    })
            };
            let count = || {
                value.parse::<usize>().map_err(|_| {
                    MoptError::parse(format!("-report_opt {key}={value}: not a whole number"))
                })
            };
            match key {
                "unit" => {
                    out.unit = EnergyUnit::parse(value)?;
                    out.unit_given = true;
                }
                "temperature" => {
                    let t = number()?;
                    if t <= 0.0 {
                        return Err(MoptError::parse(format!(
                            "-report_opt temperature={value}: above 0 K"
                        )));
                    }
                    out.temperature = t;
                }
                "energy_window" if all => out.energy_window = None,
                "energy_window" => out.energy_window = Some(number()?.abs()),
                "max_minima" if all => out.max_minima = usize::MAX,
                "max_minima" => out.max_minima = count()?.max(1),
                "figure_minima" => out.figure_minima = count()?.max(1),
                "table_rows" => out.table_rows = count()?,
                "structures" => out.structures = count()?,
                "structure_atoms" => out.structure_atoms = count()?,
                "structure_packed_atoms" => out.structure_packed_atoms = count()?,
                "irc_lines" => out.irc_lines = count()?,
                "irc_packed_lines" => out.irc_packed_lines = count()?,
                "profile_points" => out.profile_points = count()?,
                "map_exact" => out.map_exact = count()?,
                "map_landmarks" => out.map_landmarks = count()?.max(3),
                "file" => {
                    let bad = value.is_empty()
                        || value.contains(['/', '\\'])
                        || value.starts_with('.')
                        || !value.to_ascii_lowercase().ends_with(".html");
                    if bad {
                        return Err(MoptError::parse(format!(
                            "-report_opt file={value}: a file name in the run directory that \
                             ends in .html"
                        )));
                    }
                    out.file = value.to_string();
                }
                _ => {
                    return Err(MoptError::parse(format!(
                        "-report_opt: unknown key '{key}' ({KEYS})"
                    )))
                }
            }
        }
        Ok(())
    }
}

/// What joins two minima.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    /// A transition state (an edge of [`Network::ts_edges`]).
    TransitionState,
    /// A crossing point of two states (an edge of [`Network::crossing_edges`]).
    Crossing,
}

/// A transition state or crossing point between two minima that are shown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Link {
    /// The transition-state or crossing-point node.
    pub node: usize,
    /// The minima at its two sides (the backward and the forward end of an IRC).
    pub ends: [usize; 2],
    /// Its energy above each end (Hartree).
    pub barriers: [f64; 2],
    /// What it is.
    pub kind: LinkKind,
    /// Its edge (index into the edges of its kind).
    pub edge: usize,
}

/// A chemical species among the minima shown: the minima with one SMILES string, which are
/// its conformers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Species {
    /// The string ([`crate::workflow::explore::smiles`]).
    pub smiles: String,
    /// Its minima (indices into [`View::minima`]), the lowest first.
    pub members: Vec<usize>,
}

/// The most points of the energies along one IRC in the page.
const PROFILE_POINTS_EACH: usize = 40;

/// The most frames of one IRC path in the page.
pub const IRC_FRAMES: usize = 30;

/// At most `most` frames of the xyz trajectory `text` (the first, the last, and frames evenly
/// spread between), the coordinates written with 5 decimals; and the number of atom lines.
/// Frames of plain variables (`name value` lines) are kept the same way. `None` when the text
/// is neither.
fn thin_frames(text: &str, most: usize) -> Option<(String, usize)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut frames: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        if lines[at].trim().is_empty() {
            at += 1;
            continue;
        }
        let n: usize = lines[at].trim().parse().ok()?;
        if n == 0 || at + 2 + n > lines.len() {
            return None;
        }
        frames.push((at, n));
        at += 2 + n;
    }
    if frames.is_empty() {
        return None;
    }
    let count = frames.len();
    let mut keep: Vec<usize> = if count <= most {
        (0..count).collect()
    } else {
        (0..most).map(|k| k * (count - 1) / (most - 1)).collect()
    };
    keep.dedup();
    let mut out = String::new();
    let mut atoms = 0;
    for k in keep {
        let (start, n) = frames[k];
        let comment: String = lines[start + 1].chars().take(160).collect();
        let _ = write!(out, "{n}\n{comment}\n");
        for line in &lines[start + 2..start + 2 + n] {
            let f: Vec<&str> = line.split_whitespace().collect();
            let xyz: Option<Vec<f64>> = f
                .get(1..4)
                .map(|v| v.iter().filter_map(|t| t.parse().ok()).collect());
            // A frame of plain variables has `name value` lines (the frames of
            // `Configuration::frame`).
            let value: Option<f64> = match f.as_slice() {
                [_, v] => v.parse().ok(),
                _ => None,
            };
            match (f.first(), xyz, value) {
                (Some(el), Some(p), _) if p.len() == 3 => {
                    let _ = writeln!(out, "{el} {:.5} {:.5} {:.5}", p[0], p[1], p[2]);
                }
                (Some(name), _, Some(v)) => {
                    let _ = writeln!(out, "{name} {v:.8}");
                }
                _ => return None,
            }
        }
        atoms += n;
    }
    Some((out, atoms))
}

/// Principal coordinates of the map in the page (any two of them can be shown).
pub const MAP_AXES: usize = 4;

/// The map of the nodes shown.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkMap {
    /// The principal coordinates of every node shown (minima, then saddles); those beyond
    /// `shares.len()` are 0.
    pub places: Vec<[f64; MAP_AXES]>,
    /// The share of every coordinate in the spread of the distances (the sum of the positive
    /// eigenvalues; of the landmarks for landmark scaling).
    pub shares: Vec<f64>,
}

impl NetworkMap {
    /// The places on the axes `a` and `b`.
    pub fn plane(&self, a: usize, b: usize) -> Vec<[f64; 2]> {
        self.places.iter().map(|p| [p[a], p[b]]).collect()
    }
}

/// The view of a network for the page.
pub struct View<'a> {
    /// The network.
    pub network: &'a Network,
    /// The options.
    pub options: ReportOptions,
    /// The energy of the lowest minimum (Hartree; 0 without minima).
    pub low: f64,
    /// The minima shown, lowest first (node ids).
    pub minima: Vec<usize>,
    /// The links shown (between minima shown), lowest first.
    pub links: Vec<Link>,
    /// The transition states and crossing points of the links, lowest first (node ids).
    pub saddles: Vec<usize>,
    /// Minima, transition states, and crossing points of the network that are not shown.
    pub hidden: [usize; 3],
    /// The Boltzmann population of every minimum shown (in the order of `minima`).
    pub populations: Vec<f64>,
    /// Whether the populations are of free energies (all minima shown have one).
    pub free_energies: bool,
    /// The connections of the links (indices into `minima`), in the order of `links`.
    pub connections: Vec<Connection>,
    /// The disconnectivity graph of the minima shown.
    pub graph: Disconnectivity,
    /// The part of `graph` that is drawn: the lowest `figure_minima` minima.
    pub drawn: Disconnectivity,
    /// For every minimum shown, the highest point (Hartree) on its lowest-barrier path from
    /// the lowest minimum (`None`: nothing joins them).
    pub reach: Vec<Option<f64>>,
    /// The map of the nodes shown, or why there is no map.
    pub map: Result<NetworkMap, String>,
    /// How the map was made.
    pub map_method: String,
    /// The places of the nodes shown in the graph (a saddle between its two minima).
    pub layout: Vec<[f64; 2]>,
    /// The structures in the page (xyz text by node id).
    pub structures: BTreeMap<usize, String>,
    /// The further structures in the page, packed ([`super::mapper_report_pack::packed`], by
    /// node id).
    pub packed_structures: BTreeMap<usize, String>,
    /// The energies along the IRCs in the page, by link (`(arc length, relative energy)`).
    pub profiles: BTreeMap<usize, Vec<(f64, f64)>>,
    /// The IRC paths in the page (xyz frames by link), read by [`View::load_ircs`].
    pub ircs: BTreeMap<usize, String>,
    /// The further IRC paths in the page, packed (by link: the frames as a raw DEFLATE stream
    /// in base64), read by [`View::load_ircs`].
    pub packed_ircs: BTreeMap<usize, String>,
    /// The surface on a grid behind the map of a network of two plain variables (set by the
    /// report, which has the objective).
    pub surface: Option<SurfaceGrid>,
    /// Shown node id to its index (minima, then saddles).
    index_of: HashMap<usize, usize>,
    /// The crossing points shown.
    crossings: HashSet<usize>,
    /// The first link of every saddle.
    link_of: HashMap<usize, usize>,
    /// Jobs started from every node (by node id).
    visits: Vec<usize>,
}

fn energy_of(network: &Network, id: usize) -> f64 {
    network.node(id).map_or(f64::INFINITY, |n| n.energy)
}

/// The hard constraints of a run in words, `;` between them: the kind, the atoms (numbered
/// from one) and the value held (Angstrom, degrees, or the units of a function), for the log and the
/// reports of `run_mapper` and `run_autots`.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn describe_constraints(constraints: &[crate::pes::constraint::Constraint]) -> String {
    use crate::pes::constraint::ConstraintKind;
    let atoms = |list: &[crate::domain::AtomId]| {
        list.iter()
            .map(|a| (a.index() + 1).to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let length = |t: Option<f64>| {
        t.map_or_else(
            || "at the start".to_string(),
            |v| format!("{:.6} Angstrom", v * Units::BOHR_TO_ANGSTROM),
        )
    };
    let angle = |t: Option<f64>| {
        t.map_or_else(
            || "at the start".to_string(),
            |v| format!("{:.4} degrees", v.to_degrees()),
        )
    };
    constraints
        .iter()
        .map(|c| match &c.kind {
            ConstraintKind::Distance(a, b) if a.len() == 1 && b.len() == 1 => {
                format!("bond {} {}", atoms(&[a[0], b[0]]), length(c.target))
            }
            ConstraintKind::Distance(a, b) => format!(
                "distance of the centres of {} and {} {}",
                atoms(a),
                atoms(b),
                length(c.target)
            ),
            ConstraintKind::Angle(a) => format!("angle {} {}", atoms(a), angle(c.target)),
            ConstraintKind::Dihedral(a) => format!("dihedral {} {}", atoms(a), angle(c.target)),
            ConstraintKind::Coordinate(a, axis) => format!(
                "{} of atom {} {}",
                ["x", "y", "z"].get(*axis).copied().unwrap_or("?"),
                a.index() + 1,
                length(c.target)
            ),
            ConstraintKind::Frozen(a) => format!("fixed atoms {}", atoms(a)),
            ConstraintKind::Orientation(a) => format!("orientation of {}", atoms(a)),
            ConstraintKind::EigenVector(k) => format!("eigenvector {k}"),
            ConstraintKind::AtomPair(a) => format!("motion of {} along their line", atoms(a)),
            ConstraintKind::Function(f) => format!(
                "{} = {}",
                f.term.name(),
                c.target.map_or_else(
                    || "its value at the start".to_string(),
                    |v| format!("{v:.10}")
                )
            ),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The frame of a node, its label and energy in the comment: xyz text (Angstrom) for a
/// structure, the frame of [`Configuration::frame`] for plain variables.
pub fn xyz_of(node: &Node) -> String {
    node.system
        .frame(&format!("{} energy={:.12}", node.label(), node.energy))
}

/// The distance of two points of the map: mass-weighted between structures, Euclidean between
/// points of plain variables.
fn map_distance(a: &Configuration, b: &Configuration) -> MoptResult<f64> {
    match (a.structure(), b.structure()) {
        (Some(x), Some(y)) => mass_weighted_distance(x, y),
        _ => a
            .variable_distance(b)
            .ok_or_else(|| MoptError::config("a structure and a point of variables on one map")),
    }
}

/// The distances between `points` (in parallel for structures).
fn point_distances(points: &[&Configuration]) -> MoptResult<faer::Mat<f64>> {
    if points.iter().all(|p| p.structure().is_some()) {
        let owned: Vec<_> = points
            .iter()
            .filter_map(|p| p.structure().cloned())
            .collect();
        return distance_matrix(&owned);
    }
    let n = points.len();
    let mut d = faer::Mat::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let v = map_distance(points[i], points[j])?;
            d[(i, j)] = v;
            d[(j, i)] = v;
        }
    }
    Ok(d)
}

/// The file of the IRC structures of the transition state `TS<number>` as the mapper writes
/// it.
fn irc_file(number: usize, variables: bool) -> String {
    crate::workflow::mapper::observers::irc_file(number, variables)
}

impl<'a> View<'a> {
    /// The signed modes used to classify this point. Older periodic checkpoints did not
    /// distinguish the fixed-cell spectrum from the coupled index Hessian.
    pub(super) fn imaginary_wavenumbers(&self, node: &Node) -> Option<Vec<f64>> {
        if let Some(w) = &node.index_wavenumbers {
            return Some(w.clone());
        }
        if let Some(w) = self
            .network
            .ts_edges
            .iter()
            .find(|edge| edge.ts == node.id)
            .and_then(|edge| edge.imaginary_wavenumber)
            .or_else(|| {
                self.network
                    .open_ts_edges
                    .iter()
                    .find(|edge| edge.ts == node.id)
                    .and_then(|edge| edge.imaginary_wavenumber)
            })
        {
            return Some(vec![w]);
        }
        if node
            .system
            .structure()
            .is_some_and(|system| system.has_pbc())
        {
            return None;
        }
        let modes: Vec<_> = node
            .frequencies
            .as_ref()?
            .iter()
            .copied()
            .filter(|&w| w < 0.0)
            .collect();
        let order = match node.kind {
            NodeKind::HigherSaddle { order } => usize::from(order),
            NodeKind::TransitionState => 1,
            _ => return Some(modes),
        };
        (modes.len() == order).then_some(modes)
    }

    /// The view of `network` with `options`.
    pub fn new(network: &'a Network, options: ReportOptions) -> Self {
        Self::of(network, options, None)
    }

    /// [`Self::new`] of the minima `only` alone (node ids) and what joins them, with the
    /// energies still relative to the lowest minimum of the whole network: the conformers of
    /// one species, for the figures of that species.
    pub fn of(
        network: &'a Network,
        mut options: ReportOptions,
        only: Option<&HashSet<usize>>,
    ) -> Self {
        if network.variables.is_some() && !options.unit_given {
            options.unit = EnergyUnit::Value;
        }
        let low = network.lowest_minimum().unwrap_or(0.0);
        let factor = options.unit.factor();
        let window = options.energy_window;
        let within = |e: f64| window.is_none_or(|w| (e - low) * factor <= w + 1e-9);
        let mut all_minima: Vec<&Node> = network.minima().collect();
        all_minima.sort_by(|a, b| a.energy.total_cmp(&b.energy).then(a.id.cmp(&b.id)));
        let minima: Vec<usize> = all_minima
            .iter()
            .filter(|n| within(n.energy) && only.is_none_or(|s| s.contains(&n.id)))
            .take(options.max_minima)
            .map(|n| n.id)
            .collect();
        let minimum_index: HashMap<usize, usize> =
            minima.iter().enumerate().map(|(k, &id)| (id, k)).collect();
        let shown = |id: usize| minimum_index.contains_key(&id);
        let mut links = Vec::new();
        for e in &network.ts_edges {
            let top = energy_of(network, e.ts);
            if !within(top) {
                continue;
            }
            // An edge whose IRC runs into another transition state (a TS–TS link) is drawn as
            // a fan: one link from its other end to every minimum beyond.
            let linked = network.joined_by(e);
            for [a, b] in linked {
                if !(shown(a) && shown(b)) {
                    continue;
                }
                let barriers = if [a, b] == e.ends {
                    e.barriers
                } else {
                    [top - energy_of(network, a), top - energy_of(network, b)]
                };
                links.push(Link {
                    node: e.ts,
                    ends: [a, b],
                    barriers,
                    kind: LinkKind::TransitionState,
                    edge: e.id,
                });
            }
        }
        for e in &network.crossing_edges {
            let [Some(a), Some(b)] = e.ends else {
                continue;
            };
            let top = energy_of(network, e.crossing);
            if shown(a) && shown(b) && within(top) {
                links.push(Link {
                    node: e.crossing,
                    ends: [a, b],
                    barriers: [top - energy_of(network, a), top - energy_of(network, b)],
                    kind: LinkKind::Crossing,
                    edge: e.id,
                });
            }
        }
        links.sort_by(|p, q| {
            energy_of(network, p.node)
                .total_cmp(&energy_of(network, q.node))
                .then(p.node.cmp(&q.node))
        });
        let mut saddles: Vec<usize> = Vec::new();
        let mut link_of: HashMap<usize, usize> = HashMap::new();
        let mut crossings = HashSet::new();
        for (k, l) in links.iter().enumerate() {
            if let std::collections::hash_map::Entry::Vacant(slot) = link_of.entry(l.node) {
                slot.insert(k);
                saddles.push(l.node);
            }
            if l.kind == LinkKind::Crossing {
                crossings.insert(l.node);
            }
        }
        // Every transition state and crossing point in the window is shown, also one that
        // joins no two minima shown (an end that is not shown, a crossing point with a side
        // not known, a node without an edge): without a link, beside the minima it touches.
        let mut anchors: HashMap<usize, Vec<usize>> = HashMap::new();
        for e in &network.ts_edges {
            if !link_of.contains_key(&e.ts) {
                let near = e
                    .ends
                    .iter()
                    .filter_map(|id| minimum_index.get(id).copied());
                anchors.entry(e.ts).or_default().extend(near);
            }
        }
        for e in &network.crossing_edges {
            if !link_of.contains_key(&e.crossing) {
                let near = e
                    .ends
                    .iter()
                    .flatten()
                    .filter_map(|id| minimum_index.get(id).copied());
                anchors.entry(e.crossing).or_default().extend(near);
            }
        }
        // A transition state with an end of its IRC that is not known: beside the end that is.
        for e in &network.open_ts_edges {
            if !link_of.contains_key(&e.ts) {
                let near = e
                    .ends
                    .iter()
                    .flatten()
                    .filter_map(|id| minimum_index.get(id).copied());
                anchors.entry(e.ts).or_default().extend(near);
            }
        }
        for n in &network.nodes {
            let saddle = matches!(
                n.kind,
                NodeKind::TransitionState | NodeKind::Crossing { .. }
            );
            if saddle && within(n.energy) && !link_of.contains_key(&n.id) {
                saddles.push(n.id);
                anchors.entry(n.id).or_default();
                if matches!(n.kind, NodeKind::Crossing { .. }) {
                    crossings.insert(n.id);
                }
            }
        }
        saddles.sort_by(|&p, &q| {
            energy_of(network, p)
                .total_cmp(&energy_of(network, q))
                .then(p.cmp(&q))
        });
        let index_of: HashMap<usize, usize> = minima
            .iter()
            .chain(&saddles)
            .enumerate()
            .map(|(k, &id)| (id, k))
            .collect();
        let count =
            |pick: fn(&NodeKind) -> bool| network.nodes.iter().filter(|n| pick(&n.kind)).count();
        let shown_of = |pick: fn(&NodeKind) -> bool| {
            saddles
                .iter()
                .filter(|&&id| network.node(id).is_ok_and(|n| pick(&n.kind)))
                .count()
        };
        let is_ts: fn(&NodeKind) -> bool = |k| *k == NodeKind::TransitionState;
        let is_crossing: fn(&NodeKind) -> bool = |k| matches!(k, NodeKind::Crossing { .. });
        let hidden = [
            all_minima.len() - minima.len(),
            count(is_ts).saturating_sub(shown_of(is_ts)),
            count(is_crossing).saturating_sub(shown_of(is_crossing)),
        ];
        let (populations, free_energies) = boltzmann(network, &minima, options.temperature);
        let connections: Vec<Connection> = links
            .iter()
            .map(|l| Connection {
                a: minimum_index.get(&l.ends[0]).copied().unwrap_or(usize::MAX),
                b: minimum_index.get(&l.ends[1]).copied().unwrap_or(usize::MAX),
                top: energy_of(network, l.node),
            })
            .collect();
        let energies: Vec<f64> = minima.iter().map(|&id| energy_of(network, id)).collect();
        let graph = disconnectivity(&energies, &connections);
        let keep: Vec<bool> = (0..minima.len())
            .map(|i| i < options.figure_minima)
            .collect();
        let drawn = prune(&graph, &keep);
        let reach = if energies.is_empty() {
            Vec::new()
        } else {
            highest_on_the_way(&graph, &connections, &energies, 0)
        };
        let ids: Vec<usize> = minima.iter().chain(&saddles).copied().collect();
        let (map, map_method) = map_of(network, &ids, &options);
        let layout = layout_of(&minima, &saddles, &links, &minimum_index, &anchors);
        // The structures of the saddle points of higher order too: the panel of one shows it.
        let with_higher: Vec<usize> = ids
            .iter()
            .copied()
            .chain(
                network
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.kind, NodeKind::HigherSaddle { .. }))
                    .map(|n| n.id),
            )
            .collect();
        let (structures, packed_structures) = structures_of(network, &with_higher, &options);
        let mut visits = vec![0; network.nodes.len()];
        for j in &network.jobs {
            if let Some(v) = visits.get_mut(j.node) {
                *v += 1;
            }
        }
        let mut view = Self {
            network,
            options,
            low,
            minima,
            links,
            saddles,
            hidden,
            populations,
            free_energies,
            connections,
            graph,
            drawn,
            reach,
            map,
            map_method,
            layout,
            structures,
            packed_structures,
            profiles: BTreeMap::new(),
            ircs: BTreeMap::new(),
            packed_ircs: BTreeMap::new(),
            surface: None,
            index_of,
            crossings,
            link_of,
            visits,
        };
        view.profiles = view.budgeted_profiles();
        view
    }

    /// The box of the map of a network of two plain variables, with a margin of a quarter of
    /// its size on every side (`None`: no such map).
    pub fn surface_box(&self) -> Option<[[f64; 2]; 2]> {
        let names = self.network.variables.as_ref().filter(|n| n.len() == 2)?;
        let map = self.map.as_ref().ok()?;
        let _ = names;
        let mut b = [[f64::INFINITY, f64::NEG_INFINITY]; 2];
        for p in &map.places {
            for a in 0..2 {
                b[a][0] = b[a][0].min(p[a]);
                b[a][1] = b[a][1].max(p[a]);
            }
        }
        if !b.iter().all(|r| r[0].is_finite() && r[1].is_finite()) {
            return None;
        }
        // One scale on both axes: the larger span on each, centred.
        let span = (b[0][1] - b[0][0]).max(b[1][1] - b[1][0]).max(0.5) * 1.5;
        Some(std::array::from_fn(|a| {
            let mid = 0.5 * (b[a][0] + b[a][1]);
            [mid - 0.5 * span, mid + 0.5 * span]
        }))
    }

    /// `e` (Hartree) relative to the lowest minimum, in the unit of the page.
    pub fn rel(&self, e: f64) -> f64 {
        (e - self.low) * self.options.unit.factor()
    }

    /// A difference of energies (Hartree) in the unit of the page.
    pub fn diff(&self, de: f64) -> f64 {
        de * self.options.unit.factor()
    }

    /// A value in the unit of the page, written with its digits.
    pub fn fmt(&self, v: f64) -> String {
        format!("{:.*}", self.options.unit.digits(), v)
    }

    /// The relative energy of node `id` in the unit of the page.
    pub fn rel_of(&self, id: usize) -> f64 {
        self.rel(energy_of(self.network, id))
    }

    /// The label of node `id`.
    pub fn label_of(&self, id: usize) -> String {
        self.network
            .node(id)
            .map_or_else(|_| format!("node {id}"), Node::label)
    }

    /// Whether the page holds the structure of node `id` (as text, or packed).
    pub fn has_structure(&self, id: usize) -> bool {
        self.structures.contains_key(&id) || self.packed_structures.contains_key(&id)
    }

    /// The index of node `id` among the nodes shown (minima, then saddles).
    pub fn index(&self, id: usize) -> Option<usize> {
        self.index_of.get(&id).copied()
    }

    /// The saddle points of order 2 or more with the nodes shown that their descents reach
    /// (indices among the nodes shown): `(node id, label, relative energy, targets)`, those
    /// without a target shown left out. The figures draw them above their targets.
    pub fn higher_saddles(&self) -> Vec<HigherSaddleMark> {
        let mut out = Vec::new();
        for n in self
            .network
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::HigherSaddle { .. }))
        {
            let mut to: Vec<usize> = self
                .network
                .saddle_edges
                .iter()
                .filter(|e| e.saddle == n.id)
                .flat_map(|e| e.ends.into_iter().flatten())
                .filter_map(|id| self.index(id))
                .collect();
            let descended = !to.is_empty();
            if !descended {
                // Kept without descents (without `higher_saddles`): beside the node whose job
                // found it.
                to.extend(
                    n.origin
                        .job
                        .and_then(|j| self.network.jobs.iter().find(|r| r.id == j))
                        .and_then(|r| self.index(r.node)),
                );
            }
            to.sort_unstable();
            to.dedup();
            if !to.is_empty() {
                out.push(HigherSaddleMark {
                    id: n.id,
                    tag: n.label(),
                    energy: self.rel(n.energy),
                    absolute: n.energy,
                    order: match n.kind {
                        NodeKind::HigherSaddle { order } => usize::from(order),
                        _ => 0,
                    },
                    job: n.origin.job,
                    descents: self
                        .network
                        .saddle_edges
                        .iter()
                        .filter(|e| e.saddle == n.id)
                        .map(|e| (e.mode, e.ends))
                        .collect(),
                    to,
                    descended,
                    imaginary: self.imaginary_wavenumbers(n).unwrap_or_default(),
                });
            }
        }
        out
    }

    /// The population of node `id` if it is a minimum shown.
    pub fn population_of(&self, id: usize) -> Option<f64> {
        self.index(id)
            .filter(|&i| i < self.minima.len())
            .map(|i| self.populations[i])
    }

    /// Whether node `id` is a crossing point shown.
    pub fn is_crossing(&self, id: usize) -> bool {
        self.crossings.contains(&id)
    }

    /// Jobs started from node `id`.
    pub fn visits(&self, id: usize) -> usize {
        self.visits.get(id).copied().unwrap_or(0)
    }

    /// The first link of the saddle `id`.
    pub fn link_of(&self, id: usize) -> Option<&Link> {
        self.link_of.get(&id).and_then(|&k| self.links.get(k))
    }

    /// Nodes shown, minima first (the order of the map).
    pub fn shown(&self) -> impl Iterator<Item = usize> + '_ {
        self.minima.iter().chain(self.saddles.iter()).copied()
    }

    /// The highest energy of the nodes shown, relative (0 without nodes).
    pub fn top(&self) -> f64 {
        self.shown().map(|id| self.rel_of(id)).fold(0.0, f64::max)
    }

    /// The lowest `figure_minima` minima, the links between them, and their saddles: what the
    /// figures without scripts draw.
    pub fn figure_part(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let count = self.minima.len().min(self.options.figure_minima);
        let minima: Vec<usize> = self.minima[..count].to_vec();
        let low = |id: usize| self.index(id).is_some_and(|i| i < count);
        let links: Vec<usize> = (0..self.links.len())
            .filter(|&k| low(self.links[k].ends[0]) && low(self.links[k].ends[1]))
            .collect();
        let mut saddles: Vec<usize> = links.iter().map(|&k| self.links[k].node).collect();
        saddles.sort_by_key(|&id| self.index(id));
        saddles.dedup();
        (minima, links, saddles)
    }

    /// The energies along the IRC of `link` as `(arc length, relative energy)`: the backward
    /// side reversed (negative arc lengths, towards its first end), the transition state at
    /// 0, then the forward side. For a crossing point the meta-IRC paths of its two states,
    /// each on its own state: that of the first state at negative arc lengths, the crossing
    /// point at 0, that of the second state at positive ones. Empty for an edge without a path.
    pub fn profile(&self, link: &Link) -> Vec<(f64, f64)> {
        let (sides, first) = match link.kind {
            LinkKind::TransitionState => match self.network.ts_edges.get(link.edge) {
                Some(edge) => (&edge.irc.sides, "backward".to_string()),
                None => return Vec::new(),
            },
            LinkKind::Crossing => match self.network.crossing_edges.get(link.edge) {
                Some(edge) => {
                    // The side of the first state of the crossing point.
                    let first = match self.network.node(edge.crossing).map(|n| &n.kind) {
                        Ok(NodeKind::Crossing { states: [a, _], .. }) => self
                            .network
                            .states
                            .iter()
                            .position(|s| s == a)
                            .map(crate::workflow::mapper::surface_name)
                            .unwrap_or_default(),
                        _ => String::new(),
                    };
                    (&edge.irc.sides, first)
                }
                None => return Vec::new(),
            },
        };
        let mut points: Vec<(f64, f64)> = Vec::new();
        for side in sides {
            let sign = if side.name == first { -1.0 } else { 1.0 };
            for (s, e) in side.arc_lengths.iter().zip(&side.energies) {
                points.push((sign * s, self.rel(*e)));
            }
        }
        if points.is_empty() {
            return points;
        }
        points.push((0.0, self.rel_of(link.node)));
        points.retain(|(s, e)| s.is_finite() && e.is_finite());
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
        points
    }

    /// The chemical species of the minima shown, by their SMILES strings (the bonds of the
    /// default bond scale of the identity test), the species of the lowest minimum first.
    /// Empty for a network of plain variables.
    pub fn species(&self) -> Vec<Species> {
        let scale = crate::workflow::explore::identity::DEFAULT_BOND_SCALE;
        let mut out: Vec<Species> = Vec::new();
        for (i, &id) in self.minima.iter().enumerate() {
            let Some(text) = self.network.node(id).ok().and_then(|n| n.smiles(scale)) else {
                continue;
            };
            match out.iter_mut().find(|s| s.smiles == text) {
                Some(s) => s.members.push(i),
                None => out.push(Species {
                    smiles: text.to_string(),
                    members: vec![i],
                }),
            }
        }
        out
    }

    /// The file of the structures along `link` (the IRC of a transition state, the path of a
    /// crossing point between its two minima), relative to the run directory.
    fn path_file(&self, link: &Link) -> Option<&str> {
        match link.kind {
            LinkKind::TransitionState => self
                .network
                .ts_edges
                .get(link.edge)
                .and_then(|e| e.irc.file.as_deref()),
            LinkKind::Crossing => self
                .network
                .crossing_edges
                .get(link.edge)
                .and_then(|e| e.irc.file.as_deref()),
        }
    }

    /// Put the IRC paths of the lowest transition states in the page, and the paths of the
    /// crossing points between their two minima, read from their files in the run directory
    /// `dir` (one per transition state or crossing point, at most [`IRC_FRAMES`] frames each,
    /// within the budget of lines of the options); a file that is missing, outside the run
    /// directory, or not xyz frames is left out. Returns notes of what was left out.
    pub fn load_ircs(&mut self, dir: &std::path::Path) -> Vec<String> {
        let mut spent = 0;
        let mut packed_spent = 0;
        let mut out = BTreeMap::new();
        let mut packed = BTreeMap::new();
        let mut missing: Vec<String> = Vec::new();
        for (k, l) in self.links.iter().enumerate() {
            if self.link_of.get(&l.node) != Some(&k) {
                continue;
            }
            let Some(file) = self.path_file(l) else {
                continue;
            };
            let inside = std::path::Path::new(file)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)));
            let Some(text) = inside
                .then(|| std::fs::read_to_string(dir.join(file)).ok())
                .flatten()
            else {
                missing.push(file.to_string());
                continue;
            };
            let Some((frames, lines)) = thin_frames(&text, IRC_FRAMES) else {
                missing.push(file.to_string());
                continue;
            };
            if spent + lines <= self.options.irc_lines {
                spent += lines;
                out.insert(k, frames);
            } else if lines <= self.options.irc_packed_lines.saturating_sub(packed_spent) {
                // Beyond what the page holds as text: packed, so that every path can be
                // opened and saved from the page without making it several times as large.
                packed_spent += lines;
                packed.insert(k, super::mapper_report_pack::packed(&frames));
            }
        }
        self.ircs = out;
        self.packed_ircs = packed;
        if missing.is_empty() {
            return Vec::new();
        }
        let shown: Vec<&str> = missing.iter().take(5).map(String::as_str).collect();
        vec![format!(
            "{} IRC path file(s) are missing or cannot be read ({}{}); those transition states \
             are shown without their IRC path",
            missing.len(),
            shown.join(", "),
            if missing.len() > shown.len() {
                ", ..."
            } else {
                ""
            }
        )]
    }

    /// The profiles in the page: the lowest links first, each thinned to at most
    /// [`PROFILE_POINTS_EACH`] points (the ends and the transition state kept), until the
    /// budget of points is spent.
    fn budgeted_profiles(&self) -> BTreeMap<usize, Vec<(f64, f64)>> {
        let mut out = BTreeMap::new();
        let mut spent = 0;
        for (k, l) in self.links.iter().enumerate() {
            let full = self.profile(l);
            if full.len() < 2 {
                continue;
            }
            let thin = thin_out(&full, PROFILE_POINTS_EACH);
            if spent + thin.len() > self.options.profile_points {
                break;
            }
            spent += thin.len();
            out.insert(k, thin);
        }
        out
    }

    /// The data of the page's script: columns of numbers (JSON text).
    pub fn data(&self) -> String {
        let unit_digits = self.options.unit.digits() + 2;
        let ids: Vec<usize> = self.shown().collect();
        let nodes: Vec<Option<&Node>> = ids.iter().map(|&id| self.network.node(id).ok()).collect();
        let mut out = String::with_capacity(64 * ids.len() + 1024);
        let _ = write!(
            out,
            "{{\"unit\":\"{}\",\"digits\":{},\"temperature\":{},\"free\":{},\"kinds\":[\"EQ\",\
             \"TS\",\"MECP\",\"MECI\"],\"minima\":{},\"map_method\":{}",
            self.options.unit.label(),
            self.options.unit.digits(),
            number(self.options.temperature, 2),
            self.free_energies,
            self.minima.len(),
            json_string(&self.map_method)
        );
        let kind = |n: &Node| match n.kind.tag() {
            "EQ" => 0.0,
            "TS" => 1.0,
            "MECP" => 2.0,
            _ => 3.0,
        };
        column(&mut out, "id", ids.iter().map(|&i| i as f64), 0);
        column(
            &mut out,
            "num",
            nodes
                .iter()
                .map(|n| n.map_or(f64::NAN, |n| n.number as f64)),
            0,
        );
        column(
            &mut out,
            "kind",
            nodes.iter().map(|n| n.map_or(1.0, kind)),
            0,
        );
        // The surfaces of a run of several states: the state of every node (of a crossing
        // point its first, `pes2` its second; -1 for none), and their names.
        let states = &self.network.states;
        if states.len() >= 2 {
            let index = |s: &ElectronicState| {
                states
                    .iter()
                    .position(|x| x == s)
                    .map_or(-1.0, |i| i as f64)
            };
            column(
                &mut out,
                "pes",
                nodes.iter().map(|n| {
                    n.map_or(-1.0, |n| match &n.kind {
                        NodeKind::Crossing { states: [a, _], .. } => index(a),
                        _ => index(&n.state),
                    })
                }),
                0,
            );
            column(
                &mut out,
                "pes2",
                nodes.iter().map(|n| {
                    n.map_or(-1.0, |n| match &n.kind {
                        NodeKind::Crossing { states: [_, b], .. } => index(b),
                        _ => -1.0,
                    })
                }),
                0,
            );
            out.push_str(",\"states\":[");
            for (k, s) in states.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                let root = if s.excited_state > 0 {
                    format!(", root {}", s.excited_state)
                } else {
                    String::new()
                };
                out.push_str(&json_string(&format!(
                    "PES{}: charge {}, multiplicity {}{root}",
                    k + 1,
                    s.charge,
                    s.multiplicity
                )));
            }
            out.push(']');
        }
        column(
            &mut out,
            "e",
            ids.iter().map(|&id| self.rel_of(id)),
            unit_digits,
        );
        column(
            &mut out,
            "abs",
            nodes.iter().map(|n| n.map_or(f64::NAN, |n| n.energy)),
            10,
        );
        if nodes
            .iter()
            .any(|n| n.is_some_and(|n| n.free_energy.is_some()))
        {
            let low_g = self
                .minima
                .iter()
                .filter_map(|&id| self.network.node(id).ok().and_then(|n| n.free_energy))
                .fold(f64::INFINITY, f64::min);
            column(
                &mut out,
                "g",
                nodes.iter().map(|n| {
                    n.and_then(|n| n.free_energy)
                        .map_or(f64::NAN, |g| self.diff(g - low_g))
                }),
                unit_digits,
            );
        }
        match &self.map {
            Ok(map) => {
                out.push_str(",\"pc\":[");
                for a in 0..map.shares.len() {
                    if a > 0 {
                        out.push(',');
                    }
                    numbers(&mut out, map.places.iter().map(|p| p[a]), 4);
                }
                out.push(']');
                column(&mut out, "pc_share", map.shares.iter().copied(), 4);
            }
            Err(why) => {
                let _ = write!(out, ",\"map_note\":{}", json_string(why));
            }
        }
        column(&mut out, "gx", self.layout.iter().map(|p| p[0]), 3);
        column(&mut out, "gy", self.layout.iter().map(|p| p[1]), 3);
        column(
            &mut out,
            "job",
            nodes
                .iter()
                .map(|n| n.and_then(|n| n.origin.job).map_or(-1.0, |j| j as f64)),
            0,
        );
        column(&mut out, "pop", self.populations.iter().copied(), 6);
        column(
            &mut out,
            "reach",
            self.reach
                .iter()
                .map(|r| r.map_or(f64::NAN, |r| self.rel(r))),
            unit_digits,
        );
        column(
            &mut out,
            "visits",
            self.minima.iter().map(|&id| self.visits(id) as f64),
            0,
        );
        // Sparse entries by node id.
        out.push_str(",\"freq\":{");
        let mut first = true;
        for n in nodes.iter().flatten() {
            if let Some(f) = &n.frequencies {
                if !first {
                    out.push(',');
                }
                first = false;
                let _ = write!(out, "\"{}\":", n.id);
                numbers(&mut out, f.iter().take(6).copied(), 1);
            }
        }
        // Every imaginary wavenumber of the saddle points (the details of a node).
        out.push_str("},\"imag\":{");
        let mut first = true;
        for n in nodes.iter().flatten() {
            let Some(f) = self.imaginary_wavenumbers(n) else {
                continue;
            };
            if n.kind == NodeKind::Minimum || f.iter().all(|&v| v >= 0.0) {
                continue;
            }
            if !first {
                out.push(',');
            }
            first = false;
            let _ = write!(out, "\"{}\":", n.id);
            numbers(&mut out, f.iter().copied().filter(|&v| v < 0.0), 1);
        }
        out.push_str("},\"excluded\":{");
        let mut first = true;
        for n in nodes.iter().flatten() {
            if let Some(why) = &n.excluded {
                if !first {
                    out.push(',');
                }
                first = false;
                let _ = write!(out, "\"{}\":{}", n.id, json_string(why));
            }
        }
        // The followed electronic states at the structures (sparse, by node id): pairs of
        // the number of the state and its place in the order of the energies there.
        out.push_str("},\"followed\":{");
        let mut first = true;
        for n in nodes.iter().flatten() {
            if n.roots.is_empty() {
                continue;
            }
            if !first {
                out.push(',');
            }
            first = false;
            let _ = write!(out, "\"{}\":[", n.id);
            for (k, (label, root)) in n.roots.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                let _ = write!(out, "[{label},{root}]");
            }
            out.push(']');
        }
        // The chemical species: their strings, and the species of every minimum shown.
        let species = self.species();
        out.push_str("},\"species\":[");
        for (k, s) in species.iter().enumerate() {
            if k > 0 {
                out.push(',');
            }
            out.push_str(&json_string(&s.smiles));
        }
        out.push_str("],\"species_of\":{");
        let mut first = true;
        for (k, s) in species.iter().enumerate() {
            for &i in &s.members {
                if !first {
                    out.push(',');
                }
                first = false;
                let _ = write!(out, "\"{}\":{k}", self.minima[i]);
            }
        }
        out.push_str("},\"model\":{");
        let mut first = true;
        for (id, text) in &self.structures {
            if !first {
                out.push(',');
            }
            first = false;
            let _ = write!(out, "\"{id}\":{}", json_string(text));
        }
        out.push('}');
        if !self.packed_structures.is_empty() {
            out.push_str(",\"model_z\":{");
            for (i, (id, text)) in self.packed_structures.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let _ = write!(out, "\"{id}\":\"{text}\"");
            }
            out.push('}');
        }
        // The links.
        let tree: HashSet<usize> = self.graph.forest.iter().copied().collect();
        out.push_str(",\"edges\":{\"n\":0");
        let l = &self.links;
        column(&mut out, "node", l.iter().map(|l| l.node as f64), 0);
        column(&mut out, "a", l.iter().map(|l| l.ends[0] as f64), 0);
        column(&mut out, "b", l.iter().map(|l| l.ends[1] as f64), 0);
        column(
            &mut out,
            "top",
            l.iter().map(|l| self.rel_of(l.node)),
            unit_digits,
        );
        column(
            &mut out,
            "b0",
            l.iter().map(|l| self.diff(l.barriers[0])),
            unit_digits,
        );
        column(
            &mut out,
            "b1",
            l.iter().map(|l| self.diff(l.barriers[1])),
            unit_digits,
        );
        column(
            &mut out,
            "tree",
            (0..l.len()).map(|k| if tree.contains(&k) { 1.0 } else { 0.0 }),
            0,
        );
        column(
            &mut out,
            "cross",
            l.iter().map(|l| {
                if l.kind == LinkKind::Crossing {
                    1.0
                } else {
                    0.0
                }
            }),
            0,
        );
        let edge = |l: &Link| {
            (l.kind == LinkKind::TransitionState)
                .then(|| self.network.ts_edges.get(l.edge))
                .flatten()
        };
        column(
            &mut out,
            "wave",
            l.iter().map(|l| {
                edge(l)
                    .and_then(|e| e.imaginary_wavenumber)
                    .unwrap_or(f64::NAN)
            }),
            1,
        );
        // IRC files: 1 where the mapper's own name, a name of its own in "irc_files".
        let mut others = Vec::new();
        column(
            &mut out,
            "irc",
            l.iter().enumerate().map(|(k, l)| match self.path_file(l) {
                Some(f)
                    if f == irc_file(
                        self.network.node(l.node).map_or(usize::MAX, |n| n.number),
                        self.network.variables.is_some(),
                    ) =>
                {
                    1.0
                }
                Some(f) => {
                    others.push((k, f.to_string()));
                    2.0
                }
                None => 0.0,
            }),
            0,
        );
        out.push('}');
        out.push_str(",\"irc_files\":{");
        for (i, (k, f)) in others.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "\"{k}\":{}", json_string(f));
        }
        out.push_str("},\"irc_xyz\":{");
        for (i, (k, text)) in self.ircs.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "\"{k}\":{}", json_string(text));
        }
        out.push_str("},\"irc_z\":{");
        for (i, (k, text)) in self.packed_ircs.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "\"{k}\":\"{text}\"");
        }
        out.push_str("},\"variables\":");
        match &self.network.variables {
            Some(names) => {
                out.push('[');
                for (i, n) in names.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&json_string(n));
                }
                out.push(']');
            }
            None => out.push_str("null"),
        }
        let direct =
            self.network.variables.is_some() && self.map_method.starts_with("the variables");
        out.push_str(",\"map_axes\":");
        out.push_str(if direct { "true" } else { "false" });
        let _ = write!(
            out,
            ",\"ext\":\"{}\"",
            crate::workflow::mapper::configuration::frame_extension(
                self.network.variables.is_some()
            )
        );
        out.push_str(",\"surface\":");
        match &self.surface {
            Some(g) => {
                let _ = write!(
                    out,
                    "{{\"x\":[{},{}],\"y\":[{},{}],\"nx\":{},\"ny\":{},\"e\":",
                    number(g.x[0], 6),
                    number(g.x[1], 6),
                    number(g.y[0], 6),
                    number(g.y[1], 6),
                    g.nx,
                    g.ny
                );
                numbers(&mut out, g.values.iter().copied(), unit_digits.max(3));
                out.push('}');
            }
            None => out.push_str("null"),
        }
        out.push_str(",\"profiles\":{");
        for (i, (k, points)) in self.profiles.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "\"{k}\":");
            numbers(
                &mut out,
                points.iter().flat_map(|&(s, e)| [s, e]),
                unit_digits.max(3),
            );
        }
        out.push('}');
        // The saddle points of higher order, drawn above what their descents reach.
        let hs = self.higher_saddles();
        if !hs.is_empty() {
            out.push_str(",\"hs\":{\"n\":0");
            column(&mut out, "id", hs.iter().map(|h| h.id as f64), 0);
            out.push_str(",\"tag\":[");
            for (k, h) in hs.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                out.push_str(&json_string(&h.tag));
            }
            out.push(']');
            column(&mut out, "e", hs.iter().map(|h| h.energy), unit_digits);
            column(
                &mut out,
                "down",
                hs.iter().map(|h| f64::from(u8::from(h.descended))),
                0,
            );
            out.push_str(",\"to\":[");
            for (k, h) in hs.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                numbers(&mut out, h.to.iter().map(|&i| i as f64), 0);
            }
            out.push_str("],\"imag\":[");
            for (k, h) in hs.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                numbers(&mut out, h.imaginary.iter().copied(), 1);
            }
            out.push(']');
            // What the panel of a saddle point of higher order shows: its energy, order and
            // job, and its descents `[mode, end from +, end from -]` (node ids, -1: none) with
            // the labels of the ends.
            column(&mut out, "abs", hs.iter().map(|h| h.absolute), 8);
            column(&mut out, "order", hs.iter().map(|h| h.order as f64), 0);
            column(
                &mut out,
                "job",
                hs.iter().map(|h| h.job.map_or(-1.0, |j| j as f64)),
                0,
            );
            let end = |e: Option<usize>| e.map_or(-1.0, |id| id as f64);
            out.push_str(",\"desc\":[");
            for (k, h) in hs.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                out.push('[');
                for (q, (mode, ends)) in h.descents.iter().enumerate() {
                    if q > 0 {
                        out.push(',');
                    }
                    numbers(
                        &mut out,
                        [*mode as f64, end(ends[0]), end(ends[1])].into_iter(),
                        0,
                    );
                }
                out.push(']');
            }
            out.push_str("],\"names\":{");
            let ends: std::collections::BTreeSet<usize> = hs
                .iter()
                .flat_map(|h| h.descents.iter())
                .flat_map(|(_, ends)| ends.iter().flatten().copied())
                .collect();
            for (k, id) in ends.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                let _ = write!(out, "\"{id}\":{}", json_string(&self.label_of(*id)));
            }
            out.push_str("}}");
        }
        out.push('}');
        out
    }
}

/// A saddle point of higher order as the figures draw it.
#[derive(Debug, Clone, PartialEq)]
pub struct HigherSaddleMark {
    /// Its node.
    pub id: usize,
    /// Its label (`2SP0`).
    pub tag: String,
    /// Its energy relative to the lowest minimum, in the unit of the page.
    pub energy: f64,
    /// Its energy (Hartree).
    pub absolute: f64,
    /// Its order (the number of its unstable modes).
    pub order: usize,
    /// The job that found it (`None`: an input).
    pub job: Option<usize>,
    /// Its descents: the mode (0: the lowest curvature) and the nodes at the ends from `+`
    /// and `-` (`None`: the descent did not settle).
    pub descents: Vec<(usize, [Option<usize>; 2])>,
    /// The indices (among the nodes shown) of what its descents reach, or of the node whose job
    /// found it when it has no descents.
    pub to: Vec<usize>,
    /// Whether `to` are the ends of descents.
    pub descended: bool,
    /// Its imaginary wavenumbers (cm^-1, negative), where known.
    pub imaginary: Vec<f64>,
}

/// An objective of two variables on a grid: the energies relative to the lowest minimum, in
/// the unit of the page, row after row (`y` from `y[0]`), `nx` values per row.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceGrid {
    /// The range of the first variable.
    pub x: [f64; 2],
    /// The range of the second variable.
    pub y: [f64; 2],
    /// Points per row.
    pub nx: usize,
    /// Rows.
    pub ny: usize,
    /// The relative energies (not finite where the objective is not defined).
    pub values: Vec<f64>,
}

/// Points on each side of the grid of the surface behind the map.
pub const SURFACE_POINTS: usize = 72;

/// `v` with at most `decimals` digits after the point and no trailing zeros (`null` for what
/// is not finite).
fn number(v: f64, decimals: usize) -> String {
    if !v.is_finite() {
        return "null".to_string();
    }
    let mut s = format!("{v:.decimals$}");
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    if s == "-0" {
        s = "0".to_string();
    }
    s
}

/// `[a,b,...]` of `values`.
fn numbers(out: &mut String, values: impl Iterator<Item = f64>, decimals: usize) {
    out.push('[');
    for (i, v) in values.enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&number(v, decimals));
    }
    out.push(']');
}

/// `,"name":[...]`.
fn column(out: &mut String, name: &str, values: impl Iterator<Item = f64>, decimals: usize) {
    let _ = write!(out, ",\"{name}\":");
    numbers(out, values, decimals);
}

/// `text` as a JSON string, safe inside a script element of HTML.
fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// At most `most` of `points`: the ends, the highest point, and points evenly spread between.
fn thin_out(points: &[(f64, f64)], most: usize) -> Vec<(f64, f64)> {
    let n = points.len();
    if n <= most || most < 3 {
        return points.to_vec();
    }
    let top = (0..n)
        .max_by(|&a, &b| points[a].1.total_cmp(&points[b].1))
        .unwrap_or(0);
    let mut keep: Vec<usize> = (0..most - 1)
        .map(|k| k * (n - 1) / (most - 2))
        .chain([top])
        .collect();
    keep.sort_unstable();
    keep.dedup();
    keep.into_iter().map(|i| points[i]).collect()
}

/// Boltzmann populations of `minima` at `temperature`, of free energies where every one has
/// one; and whether they are.
fn boltzmann(network: &Network, minima: &[usize], temperature: f64) -> (Vec<f64>, bool) {
    let free = !minima.is_empty()
        && minima
            .iter()
            .all(|&id| network.node(id).is_ok_and(|n| n.free_energy.is_some()));
    let weighting = |id: usize| {
        network.node(id).map_or(f64::INFINITY, |n| {
            if free {
                n.free_energy.unwrap_or(n.energy)
            } else {
                n.energy
            }
        })
    };
    let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature;
    let lowest = minima
        .iter()
        .map(|&id| weighting(id))
        .fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = minima
        .iter()
        .map(|&id| (-(weighting(id) - lowest) / kt).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let populations = weights
        .iter()
        .map(|w| if total > 0.0 { w / total } else { 0.0 })
        .collect();
    (populations, free)
}

/// The places of the nodes `ids` on the map and how it was made: classical multidimensional
/// scaling of the mass-weighted distances of their structures (the ReSPer projection,
/// `analysis::path_map`), of landmarks only above `map_exact` structures.
fn map_of(
    network: &Network,
    ids: &[usize],
    options: &ReportOptions,
) -> (Result<NetworkMap, String>, String) {
    if ids.len() < 3 {
        return (
            Err("fewer than three structures".to_string()),
            String::new(),
        );
    }
    let systems: Vec<_> = ids
        .iter()
        .filter_map(|&id| network.node(id).ok().map(|n| &n.system))
        .collect();
    if systems.len() != ids.len() {
        return (Err("a node is missing".to_string()), String::new());
    }
    // Up to four plain variables: the map is the variables themselves.
    if let Some(names) = network.variables.as_ref().filter(|n| n.len() <= MAP_AXES) {
        let k = names.len();
        let places: Vec<[f64; MAP_AXES]> = systems
            .iter()
            .map(|p| {
                let x = p.flat();
                std::array::from_fn(|a| if a < k { x[a] } else { 0.0 })
            })
            .collect();
        let count = places.len() as f64;
        let spread: Vec<f64> = (0..k)
            .map(|a| {
                let mean = places.iter().map(|q| q[a]).sum::<f64>() / count;
                places.iter().map(|q| (q[a] - mean).powi(2)).sum::<f64>()
            })
            .collect();
        let total: f64 = spread.iter().sum();
        let shares = spread
            .iter()
            .map(|v| if total > 0.0 { v / total } else { 0.0 })
            .collect();
        return (
            Ok(NetworkMap { places, shares }),
            format!("the variables {}", names.join(", ")),
        );
    }
    if ids.len() <= options.map_exact.max(3) {
        let result = point_distances(&systems)
            .and_then(|d| classical_mds(&d, MAP_AXES))
            .map(|e| {
                let c = &e.coordinates;
                let axes = c.ncols().min(MAP_AXES);
                NetworkMap {
                    places: (0..c.nrows())
                        .map(|i| std::array::from_fn(|a| if a < axes { c[(i, a)] } else { 0.0 }))
                        .collect(),
                    shares: e.contributions.iter().take(axes).copied().collect(),
                }
            })
            .map_err(|e| e.to_string());
        let how = format!(
            "classical multidimensional scaling of all {} {}",
            ids.len(),
            if network.variables.is_some() {
                "points (Euclidean distances of the variables)"
            } else {
                "structures"
            }
        );
        return (result, how);
    }
    // Landmarks spread over the energies: every n/L-th structure in the order of energy.
    let mut by_energy: Vec<usize> = (0..ids.len()).collect();
    by_energy.sort_by(|&p, &q| energy_of(network, ids[p]).total_cmp(&energy_of(network, ids[q])));
    let count = options.map_landmarks.clamp(3, ids.len());
    let landmarks: Vec<usize> = (0..count)
        .map(|k| by_energy[k * ids.len() / count])
        .collect();
    let how = format!(
        "landmark multidimensional scaling ({count} landmark structures, the others placed by \
         their distances to them)"
    );
    (landmark_map(&systems, &landmarks), how)
}

/// Landmark scaling: classical scaling of the landmarks, every structure by Gower's formula
/// `y_k = -sum_i X_ik (d_i^2 - r_i) / (2 lambda_k)` with `r_i` the mean squared distance of
/// landmark `i` to the landmarks.
fn landmark_map(systems: &[&Configuration], landmarks: &[usize]) -> Result<NetworkMap, String> {
    let chosen: Vec<&Configuration> = landmarks.iter().map(|&i| systems[i]).collect();
    let d = point_distances(&chosen).map_err(|e| e.to_string())?;
    let embedding = classical_mds(&d, MAP_AXES).map_err(|e| e.to_string())?;
    let x = &embedding.coordinates;
    let axes = x.ncols().min(MAP_AXES);
    let l = landmarks.len();
    let lambda: Vec<f64> = (0..axes)
        .map(|k| embedding.eigenvalues.get(k).copied().unwrap_or(0.0))
        .collect();
    let r: Vec<f64> = (0..l)
        .map(|i| (0..l).map(|j| d[(i, j)] * d[(i, j)]).sum::<f64>() / l as f64)
        .collect();
    let places = systems
        .par_iter()
        .map(|s| {
            let mut y = [0.0; MAP_AXES];
            for (i, &m) in landmarks.iter().enumerate() {
                let di = map_distance(s, systems[m]).map_err(|e| e.to_string())?;
                for (k, yk) in y.iter_mut().enumerate().take(axes) {
                    if lambda[k] > 1e-12 {
                        *yk -= x[(i, k)] * (di * di - r[i]) / (2.0 * lambda[k]);
                    }
                }
            }
            Ok(y)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(NetworkMap {
        places,
        shares: embedding.contributions.iter().take(axes).copied().collect(),
    })
}

/// The places of the graph: the minima by a force-directed layout of the links (the lowest in
/// the middle), each saddle between its two minima (side by side when several join the same
/// pair; beside its minimum for a link of a minimum to itself); a saddle without a link is a
/// node of the layout of its own, joined to the minima it touches (`anchors`, by index among
/// the minima).
fn layout_of(
    minima: &[usize],
    saddles: &[usize],
    links: &[Link],
    minimum_index: &HashMap<usize, usize>,
    anchors: &HashMap<usize, Vec<usize>>,
) -> Vec<[f64; 2]> {
    let pairs: Vec<(usize, usize)> = links
        .iter()
        .filter_map(|l| {
            Some((
                *minimum_index.get(&l.ends[0])?,
                *minimum_index.get(&l.ends[1])?,
            ))
        })
        .collect();
    let n = minima.len();
    let free: Vec<usize> = saddles
        .iter()
        .copied()
        .filter(|id| anchors.contains_key(id))
        .collect();
    let mut joined = pairs.clone();
    for (k, id) in free.iter().enumerate() {
        for &m in anchors.get(id).map(Vec::as_slice).unwrap_or(&[]) {
            joined.push((n + k, m));
        }
    }
    let total = n + free.len();
    let iterations = (300.0 * (2000.0 / total.max(1) as f64).sqrt()).clamp(100.0, 300.0) as usize;
    let laid = force_layout(
        total,
        &joined,
        &LayoutSettings {
            iterations,
            ..LayoutSettings::default()
        },
    );
    let free_place: HashMap<usize, [f64; 2]> = free
        .iter()
        .enumerate()
        .map(|(k, &id)| (id, laid[n + k]))
        .collect();
    let mut places: Vec<[f64; 2]> = laid[..n].to_vec();
    let mut parallel: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for &(a, b) in &pairs {
        parallel.entry((a.min(b), a.max(b))).or_insert((0, 0)).0 += 1;
    }
    let mut first_link: HashMap<usize, (usize, usize)> = HashMap::new();
    for (l, &(a, b)) in links.iter().zip(&pairs) {
        first_link.entry(l.node).or_insert((a, b));
    }
    for id in saddles {
        let Some(&(a, b)) = first_link.get(id) else {
            places.push(free_place.get(id).copied().unwrap_or([0.0, 0.0]));
            continue;
        };
        let (pa, pb) = (places[a], places[b]);
        if a == b {
            places.push([pa[0] + 0.35, pa[1] + 0.35]);
            continue;
        }
        let entry = parallel.entry((a.min(b), a.max(b))).or_insert((1, 0));
        let (total, done) = *entry;
        entry.1 += 1;
        let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
        let d = (dx * dx + dy * dy).sqrt().max(1e-9);
        let shift = 0.18 * (done as f64 - (total as f64 - 1.0) / 2.0);
        places.push([
            0.5 * (pa[0] + pb[0]) - shift * dy / d,
            0.5 * (pa[1] + pb[1]) + shift * dx / d,
        ]);
    }
    places
}

/// The structures in the page: the lowest nodes shown first as text, within the counts of
/// `options` (`structures`, `structure_atoms`), and the others packed
/// (`structure_packed_atoms`; every one by default, so that the structure of any node can be
/// shown and saved).
fn structures_of(
    network: &Network,
    ids: &[usize],
    options: &ReportOptions,
) -> (BTreeMap<usize, String>, BTreeMap<usize, String>) {
    let mut ids: Vec<usize> = ids.to_vec();
    ids.sort_by(|&p, &q| energy_of(network, p).total_cmp(&energy_of(network, q)));
    let (mut text, mut packed) = (BTreeMap::new(), BTreeMap::new());
    let (mut atoms, mut packed_atoms) = (0_usize, 0_usize);
    let mut as_text = true;
    for (k, id) in ids.into_iter().enumerate() {
        let Ok(n) = network.node(id) else {
            continue;
        };
        let count = n.system.structure().map_or(1, |s| s.elements.len());
        as_text = as_text && k < options.structures && atoms + count <= options.structure_atoms;
        if as_text {
            atoms += count;
            text.insert(id, xyz_of(n));
        } else if count <= options.structure_packed_atoms.saturating_sub(packed_atoms) {
            packed_atoms += count;
            packed.insert(id, super::mapper_report_pack::packed(&xyz_of(n)));
        } else {
            break;
        }
    }
    (text, packed)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn options_are_read_and_refused_by_name() {
        let o = ReportOptions::parse(&[
            "unit=kj".to_string(),
            "temperature=500".to_string(),
            "energy_window=20".to_string(),
            "max_minima=50".to_string(),
            "file=view.html".to_string(),
        ])
        .unwrap();
        assert_eq!(o.unit, EnergyUnit::KjMol);
        assert_eq!(
            (o.temperature, o.energy_window, o.max_minima),
            (500.0, Some(20.0), 50)
        );
        assert_eq!(o.file, "view.html");
        let later = ReportOptions::parse(&["unit=ev".to_string(), "unit=hartree".to_string()]);
        assert_eq!(later.unwrap().unit, EnergyUnit::Hartree);
        let all = ReportOptions::parse(&[
            "energy_window=all".to_string(),
            "max_minima=all".to_string(),
        ])
        .unwrap();
        assert_eq!((all.energy_window, all.max_minima), (None, usize::MAX));
        for bad in [
            "unit=furlong",
            "temperature=-3",
            "size=2",
            "max_minima=x",
            "unit",
            "file=../x.html",
            "file=a/b.html",
            "file=page.txt",
        ] {
            assert!(ReportOptions::parse(&[bad.to_string()]).is_err(), "{bad}");
        }
        assert!((EnergyUnit::Ev.factor() - 27.211).abs() < 1e-3);
    }

    #[test]
    fn numbers_are_short_and_strings_are_safe_in_a_script() {
        assert_eq!(number(0.30000000000000004, 6), "0.3");
        assert_eq!(number(-0.0000001, 4), "0");
        assert_eq!(number(12.0, 3), "12");
        assert_eq!(number(f64::NAN, 3), "null");
        assert_eq!(
            json_string("a\"b</script>\n"),
            "\"a\\\"b\\u003c/script\\u003e\\n\""
        );
    }

    #[test]
    fn an_irc_path_is_thinned_to_its_ends_and_frames_between() {
        let frame = |k: usize| format!("2\nframe {k}\nAr 0.0 0.0 {k}.0\nAr 3.8 0.0 0.0\n");
        let text: String = (0..50).map(frame).collect();
        let (thin, atoms) = thin_frames(&text, 30).unwrap();
        assert_eq!(atoms, 60);
        assert!(thin.starts_with("2\nframe 0\nAr 0.00000 0.00000 0.00000\n"));
        assert!(thin.contains("frame 49\n"));
        assert_eq!(thin.matches("frame ").count(), 30);
        let (all, _) = thin_frames(&text[..frame(0).len() * 3], 30).unwrap();
        assert_eq!(all.matches("frame ").count(), 3);
        assert!(thin_frames("not xyz\n", 30).is_none());
        assert!(thin_frames("3\ncomment\nAr 0 0 0\n", 30).is_none());
    }

    #[test]
    fn a_profile_is_thinned_to_its_ends_its_top_and_points_between() {
        let points: Vec<(f64, f64)> = (0..101)
            .map(|i| (i as f64, -((i as f64 - 37.0) / 20.0).powi(2)))
            .collect();
        let thin = thin_out(&points, 10);
        assert!(thin.len() <= 10);
        assert_eq!(thin.first(), points.first());
        assert_eq!(thin.last(), points.last());
        assert!(thin.contains(&points[37]));
        assert_eq!(thin_out(&points[..5], 10).len(), 5);
    }
}
