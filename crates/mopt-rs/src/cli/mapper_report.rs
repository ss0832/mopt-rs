//! The report of `run_mapper`: one HTML page in the run directory (`mapper_report.html`),
//! written after the inputs, after finished jobs, and at the end, so that a run that was
//! stopped leaves a whole page of what it had; `run_mapper --report RUN_DIR` writes it again
//! from the files of a run (finished, stopped, or still running) without computing anything,
//! with other settings of the page (`-report_opt`).
//!
//! The page has the network in four figures (a map of the structures by multidimensional
//! scaling, the network as a graph of minima and links, the disconnectivity graph, the energy
//! diagram), the node chosen in any of them or in the tables (its numbers, links, structure,
//! and the energy along its IRC), the lowest-barrier path between two minima, the tables of the
//! minima (with their Boltzmann populations and the highest point on their way from the lowest
//! minimum), the transition states, and the crossing points, the warnings of the run, its jobs,
//! settings, and evaluations. It takes the style and the structure viewer of the AutoTS report
//! ([`super::autots_report`]): one file that loads nothing from elsewhere, light and dark
//! themes, and figures and tables that stand without scripts.
//!
//! Networks of tens of thousands of nodes: the script draws the map and the graph on a canvas
//! (all nodes, zoom and pan) and pages through the tables; what stands without scripts is drawn
//! for the lowest minima ([`super::mapper_report_data`]). A run writes its page again after a
//! job only when the writing has taken less than a twentieth of the time since the last one, so
//! that it never costs more than about 5 % of the run; `--report` writes it whenever asked.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::autots_report::{LABEL_CONTROLS, STYLE, VIEWER};
use super::autots_static::{download_link, drawing};
use super::mapper_report_data::{xyz_of, LinkKind, SurfaceGrid, View, SURFACE_POINTS};
pub use super::mapper_report_data::{EnergyUnit, ReportOptions, REPORT_FILE};
use super::mapper_report_figures::{
    disconnectivity_graph, energy_color, energy_diagram, escape, network_graph, network_map,
    profile_chart,
};
use super::mapper_report_script::{NETWORK_SCRIPT, NETWORK_STYLE};
use super::mapper_settings::CONFIG_SNAPSHOT;
use super::settings::RUN_SETTINGS_FILE;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::Logger;
use crate::error::{MoptError, MoptResult};
use crate::workflow::mapper::network::{JobStatus, Network, Node, NodeKind};
use crate::workflow::mapper::sqlite_store::read_sqlite;
use crate::workflow::mapper::store::{
    network_from_files, read_network_lenient, write_whole, NETWORK_FILE, TS_EDGES_FILE,
};
use crate::workflow::mapper::{ExplorationEnd, Job, JobOutcome, MapperObserver, Told};

/// Transition states whose IRC profile is drawn for pages read without scripts.
const STATIC_PROFILES: usize = 24;
/// Jobs and warnings listed open; more are folded.
const OPEN_ROWS: usize = 60;
/// Jobs and warnings in the page at most (the newest); the files of the run have all.
const MOST_ROWS: usize = 2000;
/// A page is written again after a job when this many times the time its last writing took
/// has passed since.
const WRITE_SPACING: u32 = 20;

/// The report of a run, as an observer of its exploration.
pub struct MapperReport {
    path: PathBuf,
    title: String,
    calculator: String,
    settings: String,
    options: ReportOptions,
    /// When the page was last written and how long that took.
    last: Option<(Instant, Duration)>,
    /// What the files the page was written from lacked, shown at the top of the page.
    notes: Vec<String>,
    /// Where a page that cannot be written during a run is reported (the run goes on).
    log: Option<Logger>,
    /// The objective of a network of plain variables, for the surface behind its map.
    objective: Option<Box<dyn crate::pes::objective::Objective>>,
}

impl MapperReport {
    /// The report of the run directory `dir` (the file of `options`); `title` names the run,
    /// `settings` is the text of its settings.
    pub fn new(
        dir: &Path,
        title: &str,
        calculator: &str,
        settings: &str,
        options: ReportOptions,
    ) -> Self {
        Self {
            path: dir.join(&options.file),
            title: title.to_string(),
            calculator: calculator.to_string(),
            settings: settings.to_string(),
            options,
            last: None,
            notes: Vec::new(),
            log: None,
            objective: None,
        }
    }

    /// The report that draws the objective `objective` of a network of two plain variables
    /// behind its map (on a grid of [`SURFACE_POINTS`] points on each side).
    pub fn with_objective(mut self, objective: Box<dyn crate::pes::objective::Objective>) -> Self {
        self.objective = Some(objective);
        self
    }

    /// The report that writes a page that cannot be written during a run as a warning of
    /// `log` instead of stopping the run.
    pub fn with_log(mut self, log: Logger) -> Self {
        self.log = Some(log);
        self
    }

    /// The report with `notes` of what the files of the run lacked, shown on the page.
    pub fn with_notes(mut self, notes: Vec<String>) -> Self {
        self.notes = notes;
        self
    }

    /// The file of the report.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write the page of `network`; `state` says whether the run goes on or how it ended
    /// (`ended`: whether that is the end of the run). Returns the notes of the page (what the
    /// files lacked, an IRC path that could not be read).
    pub fn write(
        &mut self,
        network: &Network,
        state: &str,
        ended: bool,
    ) -> MoptResult<Vec<String>> {
        let start = Instant::now();
        let mut view = View::new(network, self.options.clone());
        if let (Some(o), Some(b)) = (&self.objective, view.surface_box()) {
            view.surface = surface_grid(o.as_ref(), &view, b);
        }
        let mut notes = self.notes.clone();
        if let Some(dir) = self.path.parent() {
            notes.extend(view.load_ircs(dir));
        }
        let kinetics = self
            .path
            .parent()
            .and_then(super::mapper_kinetics::read_summary);
        let page = page(self, &view, state, ended, &notes, kinetics.as_ref());
        write_whole(&self.path, page.as_bytes())?;
        self.last = Some((start, start.elapsed()));
        Ok(notes)
    }

    /// [`Self::write`] during a run: a page that cannot be written is a warning of the log of
    /// the run, which goes on.
    fn write_in_run(&mut self, network: &Network, state: &str, ended: bool) -> MoptResult<()> {
        match self.write(network, state, ended) {
            Ok(_) => Ok(()),
            Err(e) => match &self.log {
                Some(log) => {
                    log.warn(&format!(
                        "The report {} could not be written ({e}); the run goes on, and \
                         run_mapper --report RUN_DIR writes it from the files of the run.",
                        self.path.display()
                    ));
                    Ok(())
                }
                None => Err(e),
            },
        }
    }

    /// Whether a page written now keeps the writing below a twentieth of the run.
    fn due(&self) -> bool {
        self.last
            .is_none_or(|(at, took)| at.elapsed() >= took * WRITE_SPACING)
    }
}

/// The objective `o` on a grid over the box `b` (`None` when it cannot be evaluated there).
fn surface_grid(
    o: &dyn crate::pes::objective::Objective,
    view: &View<'_>,
    b: [[f64; 2]; 2],
) -> Option<SurfaceGrid> {
    let n = SURFACE_POINTS;
    let at = |r: [f64; 2], k: usize| r[0] + (r[1] - r[0]) * k as f64 / (n - 1) as f64;
    let mut values = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let x = faer::Col::from_fn(2, |c| if c == 0 { at(b[0], i) } else { at(b[1], j) });
            let v = o.evaluate(&x, false).ok()?.value;
            values.push(if v.is_finite() { view.rel(v) } else { f64::NAN });
        }
    }
    Some(SurfaceGrid {
        x: b[0],
        y: b[1],
        nx: n,
        ny: n,
        values,
    })
}

/// Write the report of the run directory `dir` again from its files (`reaction_network.json`
/// and the settings of the run), with the settings of the page of the run overridden by
/// `items` (`-report_opt KEY=VALUE`); nothing is computed. What the files lack does not stop
/// it: a network file that is damaged is read as far as it goes, a missing one is replaced by
/// the structures and `ts_edges.csv` of the run, settings that cannot be read are left out,
/// and every such thing is a note at the top of the page. Fails only when `dir` is not a
/// directory or the page cannot be written. `source` may also be the SQLite database of a run
/// (`reaction_network.sqlite` of `store=sqlite`, or a copy of it): the page is then written
/// next to it from the database. Returns the file written and the notes.
pub fn rewrite(source: &Path, items: &[String]) -> MoptResult<(PathBuf, Vec<String>)> {
    let (dir, database) = if source.is_dir() {
        (source.to_path_buf(), None)
    } else if source.is_file() {
        let parent = source
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        (parent, Some(source.to_path_buf()))
    } else {
        return Err(MoptError::config(format!(
            "run_mapper --report: {} is neither a run directory nor a database file",
            source.display()
        )));
    };
    let dir = dir.as_path();
    let mut notes = Vec::new();
    let found = match &database {
        Some(db) => read_sqlite(db)
            .map(Some)
            .map_err(|e| MoptError::config(format!("{} ({e})", db.display()))),
        None => read_network_lenient(dir),
    };
    let (network, run) = match found {
        Ok(Some((network, run, read_notes))) => {
            notes.extend(read_notes);
            (network, run)
        }
        found => {
            notes.push(match found {
                Err(e) => format!(
                    "the network could not be read: {e}; the page shows what the structures \
                     and {TS_EDGES_FILE} of the run directory give"
                ),
                _ => format!(
                    "{} has no {NETWORK_FILE}; the page shows what the structures and \
                     {TS_EDGES_FILE} of the run directory give",
                    dir.display()
                ),
            });
            let (network, file_notes) = network_from_files(dir);
            notes.extend(file_notes);
            if network.nodes.is_empty() {
                notes.push(
                    "No structure of a mapper run was found: is this the run directory of \
                     run_mapper?"
                        .to_string(),
                );
            }
            (network, JsonValue::Null)
        }
    };
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
    let settings = read(RUN_SETTINGS_FILE);
    let snapshot = read(CONFIG_SNAPSHOT);
    if settings.is_none() {
        notes.push(format!(
            "{RUN_SETTINGS_FILE} is missing: the settings of the page are the defaults and those \
             of -report_opt"
        ));
    }
    let mut all: Vec<String> = settings
        .as_deref()
        .and_then(|t| JsonValue::parse(t).ok())
        .and_then(|doc| doc.get("report_options").and_then(JsonValue::as_string_vec))
        .unwrap_or_default();
    all.extend(items.iter().cloned());
    let (options, option_notes) = ReportOptions::parse_lenient(&all);
    notes.extend(option_notes);
    let title = network
        .inputs
        .first()
        .and_then(|i| {
            Path::new(&i.file)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "input".to_string());
    let calculator = run
        .get("calculator")
        .and_then(JsonValue::as_str)
        .map(str::to_string)
        .or_else(|| {
            let doc = JsonValue::parse(snapshot.as_deref()?).ok()?;
            let soft = doc.get("step1_settings")?.get("othersoft")?;
            soft.as_str().map(str::to_string)
        })
        .unwrap_or_else(|| {
            notes.push("The calculator of the run is not recorded in its files.".to_string());
            "not recorded".to_string()
        });
    let interrupted = run.get("interrupted").and_then(JsonValue::as_bool) == Some(true);
    let reason = run.get("ended").and_then(JsonValue::as_str);
    let (state, ended) = match (reason, interrupted) {
        (_, true) => (
            "The run was interrupted; run_mapper --restart with its run directory continues it. \
             This page shows what it had."
                .to_string(),
            false,
        ),
        (Some(reason), false) => (format!("The run ended: {reason}"), true),
        (None, false) => (
            "The run had not ended when its files were read for this page (it is running, or \
             it stopped without writing its end)."
                .to_string(),
            false,
        ),
    };
    let shown = |text: &Option<String>| {
        text.as_deref()
            .map_or_else(|| "(none)".to_string(), settings_for_page)
    };
    let text = format!(
        "Options of the run ({RUN_SETTINGS_FILE}):\n{}\n\nSettings file ({CONFIG_SNAPSHOT}):\n{}",
        shown(&settings).trim_end(),
        shown(&snapshot).trim_end()
    );
    if database.is_none() {
        if let Err(error) = super::mapper_job_report::regenerate(dir, &network) {
            notes.push(format!("Job reports: {error}"));
        }
    }
    let mut report = MapperReport::new(dir, &title, &calculator, &text, options).with_notes(notes);
    if network.variables.is_some() {
        // The objective of the run with its -calc_opt settings, for the surface behind the map.
        let mut calculator_options = crate::pes::registry::CalculatorOptions::default();
        let given = snapshot
            .as_deref()
            .and_then(|t| JsonValue::parse(t).ok())
            .and_then(|doc| {
                doc.get("step1_settings")?
                    .get("calc_option")
                    .and_then(JsonValue::as_string_vec)
            })
            .unwrap_or_default();
        for item in given {
            if let Some((k, v)) = item.split_once('=') {
                calculator_options
                    .extra
                    .push((k.trim().to_string(), v.trim().to_string()));
            }
        }
        if let Ok(o) = crate::pes::objective::ObjectiveRegistry::builtin()
            .create(&calculator, &calculator_options)
        {
            report = report.with_objective(o);
        }
    }
    let notes = report.write(&network, &state, ended)?;
    Ok((report.path, notes))
}

fn pct(p: f64) -> String {
    if p >= 0.001 {
        format!("{:.1} %", 100.0 * p)
    } else if p > 0.0 {
        format!("{:.1e} %", 100.0 * p)
    } else {
        "0 %".to_string()
    }
}

fn wavenumbers(node: &Node, count: usize) -> String {
    node.frequencies.as_ref().map_or_else(
        || "—".to_string(),
        |f| {
            f.iter()
                .take(count)
                .map(|&v| {
                    if v < 0.0 {
                        format!("{:.1}i", -v)
                    } else {
                        format!("{v:.1}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        },
    )
}

/// The label of `n` with its other environments where it has any (an ONIOM network): how
/// many, and the spread of their energies with its own.
fn labelled(view: &View<'_>, n: &Node) -> String {
    let name = named(view, n);
    if n.environments.is_empty() {
        return name;
    }
    let (low, high) = n
        .environments
        .iter()
        .map(|e| e.energy)
        .fold((n.energy, n.energy), |(lo, hi), e| (lo.min(e), hi.max(e)));
    format!(
        "{name} <small>(and {} other environment(s) of its core, a spread of {})</small>",
        n.environments.len(),
        view.fmt(view.rel(high) - view.rel(low))
    )
}

/// The label of `n` with its surface where the run has several states (`EQ3 · PES2`,
/// `MECP0 · PES1/PES2`).
fn named(view: &View<'_>, n: &Node) -> String {
    let states = &view.network.states;
    if states.len() < 2 {
        return n.label();
    }
    let name = |s: &crate::workflow::mapper::ElectronicState| {
        states
            .iter()
            .position(|x| x == s)
            .map_or_else(|| "?".to_string(), crate::workflow::mapper::surface_name)
    };
    match &n.kind {
        NodeKind::Crossing { states: [a, b], .. } => {
            format!("{} · {}/{}", n.label(), name(a), name(b))
        }
        _ => format!("{} · {}", n.label(), name(&n.state)),
    }
}

/// The rows of the lowest minima, for reading without scripts.
/// The energy of a node in Hartree for a table: that of the surface, and where the surface
/// has static bias potentials the energy of the calculator alone beside it.
fn hartree_cell(n: &Node) -> String {
    match n.bare_energy {
        Some(bare) => format!("{:.8} (calculator alone {bare:.8})", n.energy),
        None => format!("{:.8}", n.energy),
    }
}

fn minima_rows(view: &View<'_>) -> String {
    let mut rows = String::new();
    for (i, &id) in view.minima.iter().enumerate().take(view.options.table_rows) {
        let Ok(n) = view.network.node(id) else {
            continue;
        };
        let reach = match view.reach.get(i).copied().flatten() {
            _ if i == 0 => "—".to_string(),
            Some(r) => view.fmt(view.rel(r)),
            None => "not joined".to_string(),
        };
        let _ = write!(
            rows,
            "<tr data-node=\"{id}\" tabindex=\"0\"><td>{}</td><td class=\"num\">{}</td><td \
             class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{reach}</td>\
             <td>{}</td><td class=\"num\">{}</td><td>{}</td><td>{}</td><td>{get}</td></tr>",
            labelled(view, n),
            view.fmt(view.rel(n.energy)),
            hartree_cell(n),
            pct(view.populations[i]),
            wavenumbers(n, 3),
            view.visits(id),
            n.origin
                .job
                .map_or_else(|| "input".to_string(), |j| format!("job {j}")),
            n.excluded.as_deref().map_or_else(String::new, escape),
            get = get_button(view.has_structure(id), &format!("node:{id}"), "xyz")
        );
    }
    rows
}

/// The number of imaginary modes of a saddle point and each imaginary wavenumber (cm^-1),
/// `2: 512.3i, 120.4i`, from the vibrations stored on its node; where none are stored (plain
/// variables) the number its kind says, `order` (`None`: not known).
fn imaginary_modes(view: &View<'_>, node: &Node, order: Option<usize>) -> String {
    match view.imaginary_wavenumbers(node) {
        Some(f) => {
            let imaginary: Vec<String> = f
                .iter()
                .filter(|&&v| v < 0.0)
                .map(|&v| format!("{:.1}i", -v))
                .collect();
            if imaginary.is_empty() {
                "0".to_string()
            } else {
                format!("{}: {}", imaginary.len(), imaginary.join(", "))
            }
        }
        None => order.map_or_else(|| "—".to_string(), |k| format!("{k}: —")),
    }
}

/// The section of the saddle points of order 2 or more (`higher_saddles`): each with its
/// order, energy, imaginary modes, and the ends of its descents.
fn higher_saddle_section(view: &View<'_>) -> String {
    let network = view.network;
    let saddles: Vec<&Node> = network
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::HigherSaddle { .. }))
        .collect();
    if saddles.is_empty() {
        return String::new();
    }
    let unit = view.options.unit.label();
    let mut rows = String::new();
    for n in saddles {
        let order = match n.kind {
            NodeKind::HigherSaddle { order } => usize::from(order),
            _ => 0,
        };
        let label = |id: Option<usize>| {
            id.and_then(|i| network.node(i).ok())
                .map_or_else(|| "—".to_string(), Node::label)
        };
        let descents: Vec<String> = network
            .saddle_edges
            .iter()
            .filter(|e| e.saddle == n.id)
            .map(|e| {
                format!(
                    "mode {}: {} / {}",
                    e.mode + 1,
                    label(e.ends[0]),
                    label(e.ends[1])
                )
            })
            .collect();
        let _ = write!(
            rows,
            "<tr data-node=\"{id}\" tabindex=\"0\"><td>{}</td><td class=\"num\">{order}</td>\
             <td class=\"num\">{}</td><td class=\"num\">{:.8}</td><td>{}</td><td>{}</td>\
             <td>{}</td><td>{get}</td></tr>",
            n.label(),
            view.fmt(view.rel(n.energy)),
            n.energy,
            imaginary_modes(view, n, Some(order)),
            escape(&descents.join("; ")),
            n.origin
                .job
                .map_or_else(|| "input".to_string(), |j| format!("job {j}")),
            id = n.id,
            get = get_button(view.has_structure(n.id), &format!("node:{}", n.id), "xyz")
        );
    }
    format!(
        "<section class=\"card glass\" id=\"higher-saddles\"><h2>Saddle points of higher \
         order</h2><div class=\"table-wrap\"><table><thead><tr><th>node</th><th>order</th>\
         <th>relative ({unit})</th><th>energy (Hartree)</th><th>imaginary modes: number and \
         wavenumbers (cm⁻¹)</th><th>descents (+ / −)</th><th>found by</th><th></th></tr>\
         </thead><tbody>{rows}</tbody></table></div><p class=\"note\">In variable cells the \
         imaginary wavenumbers use the effective masses of the coupled atom and cell coordinates. Experimental \
         (higher_saddles): each unstable mode was followed both ways from the saddle point and \
         each start optimized at the order below, the other unstable modes followed; the ends \
         are those saddle points (or minima). A saddle point of higher order is not a state of \
         the kinetics: the flux of the transition state whose IRC reached it is divided among \
         what lies below it. Points of plain variables have the wavenumbers of unit masses.\
         </p></section>"
    )
}

/// What the scans of the barrierless dissociations gave (`confirm_dissociations`), as a
/// sentence for the note of the channels; empty where no edge was scanned.
fn scan_note(network: &Network) -> String {
    let scanned: Vec<_> = network
        .dissociation_edges
        .iter()
        .filter_map(|e| e.scan.as_ref().map(|s| (e, s)))
        .collect();
    if scanned.is_empty() {
        return String::new();
    }
    let label = |id: usize| {
        network
            .nodes
            .get(id)
            .map_or_else(String::new, |n| n.label())
    };
    let barriers: Vec<String> = scanned
        .iter()
        .filter(|(_, s)| s.barrier)
        .map(|(e, s)| {
            format!(
                "{} to {} ({:.2} kcal/mol above the channel)",
                label(e.minimum),
                label(e.channel),
                s.highest.unwrap_or(f64::NAN) * crate::domain::Units::HARTREE_TO_KCAL_MOL
            )
        })
        .collect();
    format!(
        " The dissociations of {} complex(es) were scanned (the distance of the two molecules \
         raised step by step, every point minimized): {}",
        scanned.len(),
        if barriers.is_empty() {
            "none has a maximum above its channel beyond the tolerance.".to_string()
        } else {
            format!(
                "with a barrier, and so not barrierless: {}.",
                barriers.join("; ")
            )
        }
    )
}

/// The sentence about the entrance channel of the network, if it has one.
fn entrance_note(network: &Network) -> String {
    let Some(e) = &network.entrance else {
        return String::new();
    };
    let Some(channel) = network.nodes.get(e.channel) else {
        return String::new();
    };
    format!(
        " {} is the entrance of the network: {} structure(s) of it, its molecules in contact in \
         random orientations, were inputs of the run, and its kinetics starts there.",
        channel.label(),
        e.structures
    )
}

/// The rows of the dissociation channels: their fragments, the complexes that dissociate into
/// them without a barrier, and the transition states whose IRC ends there.
fn channel_rows(view: &View<'_>) -> String {
    let network = view.network;
    let mut rows = String::new();
    for n in network
        .of_kind(&NodeKind::Dissociated)
        .take(view.options.table_rows)
    {
        let fragments: Vec<String> = n
            .parts
            .iter()
            .map(|p| {
                network.fragments.get(p.fragment).map_or_else(
                    || "?".to_string(),
                    |f| {
                        format!(
                            "{} (charge {}, multiplicity {})",
                            escape(&f.formula),
                            f.charge,
                            f.multiplicity
                        )
                    },
                )
            })
            .collect();
        let complexes: Vec<String> = network
            .dissociation_edges
            .iter()
            .filter(|e| e.channel == n.id)
            .filter_map(|e| {
                network.node(e.minimum).ok().map(|m| {
                    format!(
                        "{} (+{})",
                        m.label(),
                        view.fmt(view.rel(n.energy) - view.rel(m.energy))
                    )
                })
            })
            .collect();
        let saddles: Vec<String> = network
            .ts_edges
            .iter()
            .filter(|e| e.ends.contains(&n.id))
            .filter_map(|e| network.node(e.ts).ok().map(|t| t.label()))
            .collect();
        let or_dash = |v: Vec<String>| {
            if v.is_empty() {
                "—".to_string()
            } else {
                v.join(", ")
            }
        };
        let _ = write!(
            rows,
            "<tr data-node=\"{}\" tabindex=\"0\"><td>{}</td><td class=\"num\">{}</td><td \
             class=\"num\">{:.8}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            n.id,
            n.label(),
            view.fmt(view.rel(n.energy)),
            n.energy,
            if fragments.is_empty() {
                // A channel of a periodic cell: the structure itself, apart in its cell.
                escape(
                    n.excluded
                        .as_deref()
                        .unwrap_or("apart in its periodic cell"),
                )
            } else {
                fragments.join(" + ")
            },
            or_dash(complexes),
            or_dash(saddles)
        );
    }
    rows
}

/// A button that saves what the page holds under `key` (`node:ID`, `irc:LINK`: a structure,
/// the frames of a path), named `text`; a dash where the page does not hold it. The page has
/// no links to the files of the run directory, so that it can be moved or sent on its own.
fn get_button(held: bool, key: &str, text: &str) -> String {
    if held {
        format!("<button type=\"button\" class=\"get\" data-get=\"{key}\">{text}</button>")
    } else {
        "<span class=\"note\" title=\"not in this page\">—</span>".to_string()
    }
}

fn link_rows(view: &View<'_>, kind: LinkKind) -> String {
    let tree: HashSet<usize> = view.graph.forest.iter().copied().collect();
    let mut rows = String::new();
    let mut count = 0;
    for &id in &view.saddles {
        if count == view.options.table_rows {
            break;
        }
        let Ok(n) = view.network.node(id) else {
            continue;
        };
        if view.is_crossing(id) != (kind == LinkKind::Crossing) {
            continue;
        }
        count += 1;
        let Some(l) = view.link_of(id) else {
            let _ = write!(
                rows,
                "<tr data-node=\"{id}\" tabindex=\"0\"><td>{}</td><td class=\"num\">{}</td><td \
                 class=\"num\">{}</td><td>— (joins no two minima shown)</td><td \
                 class=\"num\">—</td><td class=\"num\">—</td><td class=\"num\">{}</td><td></td>\
                 <td>{}</td></tr>",
                labelled(view, n),
                view.fmt(view.rel(n.energy)),
                hartree_cell(n),
                imaginary_modes(view, n, (kind == LinkKind::TransitionState).then_some(1)),
                get_button(view.has_structure(id), &format!("node:{id}"), "xyz")
            );
            continue;
        };
        let k = view.links.iter().position(|x| x == l).unwrap_or(usize::MAX);
        let (wave, irc) = match (kind, view.network.ts_edges.get(l.edge)) {
            (LinkKind::TransitionState, Some(_)) => (
                imaginary_modes(view, n, Some(1)),
                if view.ircs.contains_key(&k) || view.packed_ircs.contains_key(&k) {
                    format!(" · {}", get_button(true, &format!("irc:{k}"), "IRC"))
                } else {
                    String::new()
                },
            ),
            (LinkKind::Crossing, _) => {
                let edge = view.network.crossing_edges.get(l.edge);
                (
                    format!(
                        "{}; {}; {}",
                        edge.and_then(|e| e.gradient_difference)
                            .map_or_else(|| "—".to_string(), |g| format!("{g:.4}")),
                        edge.and_then(|e| e.gap)
                            .map_or_else(|| "—".to_string(), |g| format!("{g:.2e}")),
                        crate::workflow::mapper::seam_of(view.network, id).map_or_else(
                            || "—".to_string(),
                            |s| super::autots_report::escape(&s.describe())
                        )
                    ),
                    // The path between the two minima through the crossing point.
                    if view.ircs.contains_key(&k) || view.packed_ircs.contains_key(&k) {
                        format!(" · {}", get_button(true, &format!("irc:{k}"), "path"))
                    } else {
                        String::new()
                    },
                )
            }
            _ => ("—".to_string(), String::new()),
        };
        let _ = write!(
            rows,
            "<tr data-node=\"{}\" tabindex=\"0\"><td>{}</td><td class=\"num\">{}</td><td \
             class=\"num\">{}</td><td>{} → {}</td><td class=\"num\">{}</td><td \
             class=\"num\">{}</td><td class=\"num\">{wave}</td><td>{}</td><td>{}{irc}</td></tr>",
            id,
            labelled(view, n),
            view.fmt(view.rel(n.energy)),
            hartree_cell(n),
            view.label_of(l.ends[0]),
            view.label_of(l.ends[1]),
            view.fmt(view.diff(l.barriers[0])),
            view.fmt(view.diff(l.barriers[1])),
            if tree.contains(&k) { "✓" } else { "" },
            get_button(view.has_structure(id), &format!("node:{id}"), "xyz")
        );
    }
    rows
}

/// The most species drawn in the graph of the species (the lowest first).
const SPECIES_DRAWN: usize = 60;

/// The most species with a disconnectivity graph and an energy diagram of their own in the
/// page (the lowest first); the figures of the others are those of the whole network.
const SPECIES_FIGURES: usize = 12;

/// The graph of the chemical species: a node per species (its area grows with the number of
/// its conformers, its colour is the energy of its lowest conformer) and an edge per two
/// species that transition states join, with the lowest of them above the lowest minimum
/// shown. `between` has, for every two species, that transition state (its node, its energy)
/// and how many there are. A node chooses the lowest conformer of its species, an edge its
/// transition state. Drawn by mopt-rs, so that it is there without scripts.
fn species_graph(
    view: &View<'_>,
    species: &[super::mapper_report_data::Species],
    between: &BTreeMap<(usize, usize), (usize, f64, usize)>,
) -> String {
    let n = species.len().min(SPECIES_DRAWN);
    if n < 2 {
        return String::new();
    }
    let edges: Vec<(usize, usize)> = between
        .keys()
        .copied()
        .filter(|&(a, b)| a < n && b < n)
        .collect();
    let places = crate::analysis::graph_layout::force_layout(
        n,
        &edges,
        &crate::analysis::graph_layout::LayoutSettings::default(),
    );
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in &places {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let (width, height, pad) = (760.0, 440.0, 60.0);
    let scale = ((width - 2.0 * pad) / (hi[0] - lo[0]).max(1e-9))
        .min((height - 2.0 * pad) / (hi[1] - lo[1]).max(1e-9));
    let at = |k: usize| -> (f64, f64) {
        let p = places[k];
        (
            0.5 * width + scale * (p[0] - 0.5 * (lo[0] + hi[0])),
            0.5 * height + scale * (p[1] - 0.5 * (lo[1] + hi[1])),
        )
    };
    let lowest = |k: usize| {
        view.network
            .node(view.minima[species[k].members[0]])
            .map_or(f64::NAN, |node| node.energy)
    };
    let top = (0..n)
        .map(|k| view.rel(lowest(k)))
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" \
         class=\"species-graph\" role=\"img\" aria-label=\"Graph of the chemical species \
         and the transition states between them\">"
    );
    for &(a, b) in &edges {
        let Some(&(ts, energy, count)) = between.get(&(a, b)) else {
            continue;
        };
        let ((xa, ya), (xb, yb)) = (at(a), at(b));
        let _ = write!(
            svg,
            "<g data-node=\"{ts}\" data-a=\"{a}\" data-b=\"{b}\" tabindex=\"0\" \
             class=\"species-edge\"><title>{}: the \
             lowest of {count} transition state(s) between S{} and S{}</title><line \
             x1=\"{xa:.1}\" y1=\"{ya:.1}\" x2=\"{xb:.1}\" y2=\"{yb:.1}\"/><text \
             x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" class=\"edge-label\">{}</text></g>",
            view.label_of(ts),
            a + 1,
            b + 1,
            0.5 * (xa + xb),
            0.5 * (ya + yb) - 4.0,
            view.fmt(view.rel(energy)),
        );
    }
    for (k, s) in species.iter().enumerate().take(n) {
        let (x, y) = at(k);
        let r = 9.0 + 4.0 * (s.members.len() as f64).sqrt();
        let smiles: String = if s.smiles.chars().count() > 28 {
            s.smiles
                .chars()
                .take(27)
                .chain(std::iter::once('…'))
                .collect()
        } else {
            s.smiles.clone()
        };
        let _ = write!(
            svg,
            "<g data-node=\"{}\" data-s=\"{k}\" tabindex=\"0\" class=\"species-node\">\
             <title>S{}: {} ({} \
             conformer(s), the lowest {} at {})</title><circle cx=\"{x:.1}\" cy=\"{y:.1}\" \
             r=\"{r:.1}\" fill=\"{}\"/><text x=\"{x:.1}\" y=\"{:.1}\" \
             text-anchor=\"middle\" class=\"species-name\">S{}</text><text x=\"{x:.1}\" \
             y=\"{:.1}\" text-anchor=\"middle\">{}</text></g>",
            view.minima[s.members[0]],
            k + 1,
            escape(&s.smiles),
            s.members.len(),
            view.label_of(view.minima[s.members[0]]),
            view.fmt(view.rel(lowest(k))),
            energy_color(view.rel(lowest(k)) / top),
            y + 4.0,
            k + 1,
            y + r + 13.0,
            escape(&smiles),
        );
    }
    svg.push_str("</svg>");
    let left_out = if species.len() > n {
        format!(" The {n} lowest of {} species are drawn.", species.len())
    } else {
        String::new()
    };
    format!(
        "<h3>Graph of the species</h3><figure><div class=\"table-wrap species-wrap\"><button \
         type=\"button\" class=\"dl full needs-js-inline\" data-full>Full screen</button>{svg}</div>\
         <figcaption>Every node is a species (the conformers of one SMILES string together): \
         larger with more conformers, coloured by the energy of its lowest conformer. An edge \
         joins two species that transition states join; its number is the lowest of them, in \
         {} above the lowest minimum shown. A node chooses the lowest conformer of its \
         species, an edge its transition state. <span class=\"needs-js-inline\">Drag a node to \
         move it (its edges follow), drag elsewhere to move the view, Ctrl + wheel to zoom, \
         double-click to see all again.</span>{left_out}</figcaption></figure>",
        view.options.unit.label()
    )
}

/// The section of the chemical species: the minima grouped by their SMILES strings (the
/// conformers of one species together), and for every two species joined by transition states
/// the lowest of them, with its height above the lowest conformer of each. Empty for a
/// network of plain variables.
fn species_section(view: &View<'_>) -> String {
    let species = view.species();
    if species.is_empty() {
        return String::new();
    }
    let unit = view.options.unit.label();
    let energy = |i: usize| {
        view.network
            .node(view.minima[i])
            .map_or(f64::NAN, |n| n.energy)
    };
    let mut rows = String::new();
    for (k, s) in species.iter().enumerate() {
        let lowest = s.members[0];
        let id = view.minima[lowest];
        let population: f64 = s.members.iter().map(|&i| view.populations[i]).sum();
        let shown = 8;
        let mut members: Vec<String> = s
            .members
            .iter()
            .take(shown)
            .map(|&i| view.label_of(view.minima[i]))
            .collect();
        if s.members.len() > shown {
            members.push(format!("and {} more", s.members.len() - shown));
        }
        let _ = write!(
            rows,
            "<tr data-node=\"{id}\" tabindex=\"0\"><td>S{} <button type=\"button\" \
             class=\"get needs-js-inline\" data-species=\"{k}\">in the network</button></td>\
             <td><code>{}</code></td><td \
             class=\"num\">{}</td><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>\
             <td>{}</td></tr>",
            k + 1,
            escape(&s.smiles),
            s.members.len(),
            view.label_of(id),
            view.fmt(view.rel(energy(lowest))),
            pct(population),
            members.join(", ")
        );
    }
    // The species of every minimum shown (by node id).
    let of: HashMap<usize, usize> = species
        .iter()
        .enumerate()
        .flat_map(|(k, s)| s.members.iter().map(move |&i| (view.minima[i], k)))
        .collect();
    // The lowest transition state between every two species, and how many there are.
    let mut between: BTreeMap<(usize, usize), (usize, f64, usize)> = BTreeMap::new();
    let mut within = 0;
    for l in &view.links {
        if l.kind != LinkKind::TransitionState {
            continue;
        }
        let (Some(&a), Some(&b)) = (of.get(&l.ends[0]), of.get(&l.ends[1])) else {
            continue;
        };
        if a == b {
            within += 1;
            continue;
        }
        let top = view.network.node(l.node).map_or(f64::NAN, |n| n.energy);
        let entry = between
            .entry((a.min(b), a.max(b)))
            .or_insert((l.node, top, 0));
        entry.2 += 1;
        if top < entry.1 {
            entry.0 = l.node;
            entry.1 = top;
        }
    }
    let mut reactions = String::new();
    for (&(a, b), &(ts, top, count)) in &between {
        let low = |k: usize| energy(species[k].members[0]);
        let _ = write!(
            reactions,
            "<tr data-node=\"{ts}\" tabindex=\"0\"><td>S{} <code>{}</code></td><td>S{} \
             <code>{}</code></td><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>\
             <td class=\"num\">{}</td><td class=\"num\">{count}</td></tr>",
            a + 1,
            escape(&species[a].smiles),
            b + 1,
            escape(&species[b].smiles),
            view.label_of(ts),
            view.fmt(view.rel(top)),
            view.fmt(view.diff(top - low(a))),
            view.fmt(view.diff(top - low(b))),
        );
    }
    let graph = species_graph(view, &species, &between);
    let reactions = if reactions.is_empty() {
        "<p class=\"note\">No transition state shown joins two species.</p>".to_string()
    } else {
        format!(
            "<h3>Reactions between species</h3><div class=\"table-wrap\"><table><thead><tr>\
             <th>species 1</th><th>species 2</th><th>lowest transition state</th>\
             <th>relative ({unit})</th><th>above the lowest conformer of 1</th>\
             <th>above the lowest conformer of 2</th><th>transition states</th></tr></thead>\
             <tbody>{reactions}</tbody></table></div>"
        )
    };
    format!(
        "<section class=\"card glass\" id=\"species\"><h2>Chemical species</h2><p \
         class=\"note\">The minima shown grouped by their SMILES strings: minima with one \
         string are conformers of one species. The strings are written from the structures \
         (bonds from the covalent radii, bond orders from the usual valences, @ and @@ at \
         tetrahedral centres of four different substituents, rings in a Kekule form): R and S \
         are two species, / and \\ about a double bond outside the rings make E and Z two \
         species, and a bond order is a guess where the valence rules do not hold.</p><div \
         class=\"table-wrap\"><table><thead><tr><th>species</th><th>SMILES</th>\
         <th>conformers</th><th>lowest</th><th>relative ({unit})</th><th>population</th>\
         <th>minima</th></tr></thead><tbody>{rows}</tbody></table></div>{graph}{reactions}<p \
         class=\"note\">{} species among the {} minima shown; {within} transition state(s) \
         shown join conformers of one species. The population of a species is the sum of \
         those of its conformers.</p></section>",
        species.len(),
        view.minima.len()
    )
}

/// The IRC profiles of the lowest transition states, for pages read without scripts.
fn static_profiles(view: &View<'_>) -> String {
    let mut out = String::new();
    let mut count = 0;
    for (&k, points) in view.profiles.iter().take(STATIC_PROFILES) {
        let Some(l) = view.links.get(k) else {
            continue;
        };
        count += 1;
        let (a, b) = (view.label_of(l.ends[0]), view.label_of(l.ends[1]));
        let _ = write!(
            out,
            "<figure>{}<figcaption>{} ({} {}): {a} → {b}</figcaption></figure>",
            profile_chart(points, [&a, &b], view.options.unit.label()),
            view.label_of(l.node),
            view.fmt(view.rel_of(l.node)),
            view.options.unit.label()
        );
    }
    if out.is_empty() {
        return String::new();
    }
    format!(
        "<details class=\"no-js-only\"><summary>Energy along the IRC of the {count} lowest \
         transition states</summary><div class=\"profiles\">{out}</div></details>"
    )
}

/// What the map shows, under it.
fn map_note(view: &View<'_>) -> String {
    let Some(names) = &view.network.variables else {
        return format!(
            "Every structure at its place by {} of the mass-weighted distances between the \
             structures (ReSPer; any two of the first four principal coordinates, in Å \
             amu<sup>1/2</sup>, on one scale): minima as circles (the lowest ringed), transition \
             states as diamonds, each link from its two minima through its transition state; \
             colors by relative energy. Distances between permuted atoms are not reduced, so \
             equivalent structures with other atom numbers lie apart.",
            escape(&view.map_method)
        );
    };
    let places = if view.map_method.starts_with("the variables") {
        format!(
            "Every point at its values of the variables {} (any two of them, on one scale)",
            escape(&names.join(", "))
        )
    } else {
        format!(
            "Every point at its place by {} (any two of the first four principal coordinates, \
             on one scale)",
            escape(&view.map_method)
        )
    };
    format!(
        "{places}: minima as circles (the lowest ringed), transition states as diamonds, each \
         link along its IRC where the page holds the path, otherwise from its two minima \
         through its transition state; colors by relative energy.{}",
        if view.surface.is_some() {
            " Behind them the surface itself on a grid, in the same colors (white where it lies \
             above the highest node shown)."
        } else {
            ""
        }
    )
}

/// The values of a point of plain variables as a table.
fn variables_table(point: &crate::workflow::mapper::configuration::VariablePoint) -> String {
    let mut rows = String::new();
    for (name, v) in point.names.iter().zip(point.values.iter()) {
        rows.push_str(&format!(
            "<tr><td>{}</td><td>{v:.8}</td></tr>",
            escape(name)
        ));
    }
    format!(
        "<div class=\"table-wrap\"><table class=\"variables\"><thead><tr><th>variable</th>\
         <th>value</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )
}

/// The details of the lowest minimum as the page has them before its script runs (and where
/// scripts do not run), and the model of its viewer.
fn first_details(view: &View<'_>) -> (String, String) {
    let Some(n) = view
        .minima
        .first()
        .and_then(|&id| view.network.node(id).ok())
    else {
        return (
            "<p class=\"note\">No node yet.</p>".to_string(),
            "{}".to_string(),
        );
    };
    let text = view
        .structures
        .get(&n.id)
        .cloned()
        .unwrap_or_else(|| xyz_of(n));
    if let Some(point) = n.system.point() {
        let file = crate::workflow::mapper::store::node_file_text(n);
        let link = download_link(&format!("{}.txt", n.label()), &file, "Download .txt", "dl");
        let html = format!(
            "<h3>{} <span class=\"pill\">minimum</span></h3><dl class=\"facts\"><dt>Energy</dt>\
             <dd>0 {unit} (the lowest minimum)</dd><dt>Value of the objective</dt><dd>{:.8}</dd>             <dt>Population \
             at {} K</dt><dd>{}</dd></dl>{}<div class=\"controls static-controls\">{link}</div>\
             <p class=\"note no-js-only\">With scripts, a node chosen in the figures or tables \
             is shown here.</p>",
            n.label(),
            n.energy,
            view.options.temperature,
            pct(view.populations.first().copied().unwrap_or(0.0)),
            variables_table(point),
            unit = view.options.unit.label()
        );
        return (html, "{}".to_string());
    }
    let link = download_link(&format!("{}.xyz", n.label()), &text, "Download .xyz", "dl");
    let html = format!(
        "<h3>{} <span class=\"pill\">minimum</span></h3><dl class=\"facts\"><dt>Energy</dt><dd>0 \
         {unit} (the lowest minimum)</dd><dt>Energy (Hartree)</dt><dd>{:.8}</dd><dt>Population \
         at {} K</dt><dd>{}</dd></dl><figure class=\"viewer-box\"><div class=\"viewer\" \
         data-model=\"m0\">{}</div><div class=\"controls static-controls\">{link}</div>\
         </figure><p class=\"note no-js-only\">With scripts, a node chosen in the figures or \
         tables is shown here.</p>",
        n.label(),
        n.energy,
        view.options.temperature,
        pct(view.populations.first().copied().unwrap_or(0.0)),
        drawing(&text),
        unit = view.options.unit.label()
    );
    let mut m = BTreeMap::new();
    m.insert("xyz".to_string(), JsonValue::String(text));
    m.insert("animate".to_string(), JsonValue::Bool(false));
    m.insert("name".to_string(), JsonValue::String(n.label()));
    let mut models = BTreeMap::new();
    models.insert("m0".to_string(), JsonValue::Object(m));
    (html, JsonValue::Object(models).to_string_compact())
}

fn fold(count: usize, what: &str, body: String) -> String {
    if count <= OPEN_ROWS {
        body
    } else {
        format!("<details><summary>The {count} {what}</summary>{body}</details>")
    }
}

fn warnings(network: &Network) -> String {
    if network.warnings.is_empty() {
        return "<p class=\"note\">No warnings.</p>".to_string();
    }
    let total = network.warnings.len();
    let mut list = String::new();
    for w in network
        .warnings
        .iter()
        .skip(total.saturating_sub(MOST_ROWS))
    {
        let _ = write!(
            list,
            "<li>{}{}</li>",
            w.job.map_or_else(String::new, |j| format!("Job {j}: ")),
            escape(&w.message)
        );
    }
    let more = if total > MOST_ROWS {
        format!(
            "<p class=\"note\">The last {MOST_ROWS}; reaction_network.json has all {total}.</p>"
        )
    } else {
        String::new()
    };
    format!(
        "<p class=\"warn\">{total} warning(s): what did not converge, or is not what it was \
         searched for, is not in the network.</p>{}",
        fold(total, "warnings", format!("{more}<ul>{list}</ul>"))
    )
}

fn jobs(network: &Network) -> String {
    let total = network.jobs.len();
    let mut rows = String::new();
    for j in network.jobs.iter().skip(total.saturating_sub(MOST_ROWS)) {
        let from = network
            .node(j.node)
            .map_or_else(|_| "?".to_string(), Node::label);
        let status = match &j.status {
            JobStatus::Done => "<span class=\"pill ok\">done</span>".to_string(),
            JobStatus::Failed(e) => format!("<span class=\"pill bad\">failed</span> {}", escape(e)),
        };
        let new: Vec<String> = j
            .new_nodes
            .iter()
            .filter_map(|&id| network.nodes.get(id).map(Node::label))
            .collect();
        let _ = write!(
            rows,
            "<tr><td>{}</td><td>{from}</td><td>{}</td><td>{status}</td><td>{}</td></tr>",
            j.id,
            escape(&j.description),
            if new.is_empty() {
                "—".to_string()
            } else {
                new.join(", ")
            }
        );
    }
    if rows.is_empty() {
        return "<p class=\"note\">No job has finished yet.</p>".to_string();
    }
    let more = if total > MOST_ROWS {
        format!(
            "<p class=\"note\">The last {MOST_ROWS}; reaction_network.json has all {total}.</p>"
        )
    } else {
        String::new()
    };
    fold(
        total,
        "jobs",
        format!(
            "{more}<div class=\"table-wrap\"><table><thead><tr><th>#</th><th>from</th>\
             <th>task</th><th>status</th><th>new nodes</th></tr></thead><tbody>{rows}</tbody>\
             </table></div>"
        ),
    )
}

fn hidden_note(view: &View<'_>) -> String {
    let [m, t, c] = view.hidden;
    if m + t + c == 0 {
        return String::new();
    }
    let mut why = Vec::new();
    if let Some(w) = view.options.energy_window {
        why.push(format!(
            "the energy window of {} {}",
            view.fmt(w),
            view.options.unit.label()
        ));
    }
    if view.minima.len() == view.options.max_minima {
        why.push(format!("the {} lowest minima", view.options.max_minima));
    }
    format!(
        "<p class=\"warn\">Not shown: {m} minima, {t} transition state(s), {c} crossing \
         point(s){}; populations are among the minima shown. <code>run_mapper --report RUN_DIR \
         -report_opt energy_window=none max_minima=all</code> writes the page again with all.</p>",
        if why.is_empty() {
            " (their links join minima that are not shown)".to_string()
        } else {
            format!(" (beyond {})", why.join(" or "))
        }
    )
}

/// How much of the spread of the distances the axes of the map hold, with a warning where the
/// two axes shown hold less than half of it: two coordinates cannot follow a wide network, and
/// structures far apart can then lie close on the map.
fn map_quality(view: &View<'_>) -> String {
    let Ok(map) = &view.map else {
        return String::new();
    };
    let shares: Vec<String> = map
        .shares
        .iter()
        .enumerate()
        .map(|(a, s)| {
            let name = match &view.network.variables {
                Some(names) if view.map_method.starts_with("the variables") => {
                    names.get(a).cloned().unwrap_or_default()
                }
                _ => format!("PCo{}", a + 1),
            };
            format!("{} {:.0} %", escape(&name), 100.0 * s)
        })
        .collect();
    let of = if view.map_method.starts_with("landmark") {
        " (of the landmarks)"
    } else {
        ""
    };
    let two: f64 = map.shares.iter().take(2).sum();
    let warn = if two < 0.5 {
        format!(
            "<p class=\"warn\" data-map-warning>The two axes shown hold only {:.0} % of the spread \
             of the distances{of}: structures far apart can lie close on this map. Other axes, or \
             the graph (which shows the links whatever the distances), show what this map \
             cannot.</p>",
            100.0 * two
        )
    } else {
        "<p class=\"warn\" data-map-warning hidden></p>".to_string()
    };
    format!(
        "<p class=\"note\" data-map-shares>Share of each axis in the spread of the distances{of}: \
         {}.</p>{warn}",
        shares.join(", ")
    )
}

/// Where the page draws less than all nodes without scripts.
fn part_note(view: &View<'_>) -> String {
    let (minima, links, _) = view.figure_part();
    let rows = view.options.table_rows;
    if minima.len() == view.minima.len() && view.minima.len() <= rows && view.links.len() <= rows {
        return String::new();
    }
    format!(
        "<p class=\"note no-js-only\">Without scripts, the figures draw the {} lowest minima and \
         {} links between them, and the tables list the {rows} lowest of each; with scripts, \
         every node shown is drawn and listed. The files of the run (nodes.csv, ts_edges.csv) \
         have all.</p>",
        minima.len(),
        links.len()
    )
}

/// What the files of the page lacked, at the top of the page.
fn missing_section(notes: &[String]) -> String {
    if notes.is_empty() {
        return String::new();
    }
    let mut list = String::new();
    for n in notes.iter().take(200) {
        let _ = write!(list, "<li>{}</li>", escape(n));
    }
    if notes.len() > 200 {
        let _ = write!(list, "<li>… and {} more</li>", notes.len() - 200);
    }
    format!(
        "<section class=\"card glass\" id=\"missing\"><h2>Missing information</h2><p \
         class=\"warn\">This page was written from incomplete files: what could be read is \
         shown, and what is missing or left out is listed here ({} note(s)).</p><ul>{list}</ul>\
         </section>",
        notes.len()
    )
}

fn page(
    report: &MapperReport,
    view: &View<'_>,
    state: &str,
    ended: bool,
    notes: &[String],
    kinetics: Option<&JsonValue>,
) -> String {
    let network = view.network;
    let goal_note = match network.goal_node() {
        Some(goal) => {
            let file = network
                .inputs
                .iter()
                .find(|i| i.goal)
                // The name of the file alone: the page is read away from where it was made.
                .map_or(String::new(), |i| {
                    let name = Path::new(&i.file)
                        .file_name()
                        .map_or_else(|| i.file.clone(), |f| f.to_string_lossy().into_owned());
                    format!(" ({})", escape(&name))
                });
            let label = network
                .node(goal)
                .map_or_else(|_| "?".to_string(), Node::label);
            let matcher = crate::workflow::explore::identity::StructureMatcher::default();
            if network.goal_joined() == Some(true) {
                format!(
                    "<p class=\"note\">Goal: {label}{file}, joined to a start (the \
                     lowest-barrier path below finds the way).</p>"
                )
            } else if let Some(other) = network.goal_bonds_joined(&matcher) {
                let other = network
                    .node(other)
                    .map_or_else(|_| "?".to_string(), Node::label);
                format!(
                    "<p class=\"note\">Goal: {label}{file}; {other}, with its bond graph (another \
                     conformer), is joined to a start.</p>"
                )
            } else {
                format!("<p class=\"warn\">Goal: {label}{file}, not joined to a start yet.</p>")
            }
        }
        None => String::new(),
    };
    // The constraints and static bias potentials of the surface of the run.
    let goal_note = network.surface.iter().fold(goal_note, |mut out, line| {
        let _ = write!(out, "<p class=\"note\">{}</p>", escape(line));
        out
    });
    let (kinetics_link, kinetics_section) = match kinetics {
        Some(summary) => (
            "<a href=\"#kinetics\">Kinetics</a>",
            super::mapper_report_kinetics::kinetics_section(summary, network),
        ),
        None => ("", String::new()),
    };
    let unit = view.options.unit.label();
    let minima = network.minima().count();
    let ts_count = network
        .nodes
        .iter()
        .filter(|n| n.kind.tag() == "TS")
        .count();
    let crossings = view
        .links
        .iter()
        .filter(|l| l.kind == LinkKind::Crossing)
        .count();
    // The crossing points of the network, named beside the transition states in a run of
    // several states (the nodes, as everywhere: `Network::crossing_points`).
    let crossing_nodes = network.crossing_points();
    let several_states = !network.states.is_empty() || crossing_nodes > 0;
    let found_short = if several_states {
        format!(
            "{minima} minima, {ts_count} transition state(s), {crossing_nodes} crossing point(s)"
        )
    } else {
        format!("{minima} minima, {ts_count} transition state(s)")
    };
    let found_long = if several_states {
        format!(
            "{minima} minima, {ts_count} transition states, and {crossing_nodes} crossing points"
        )
    } else {
        format!("{minima} minima and {ts_count} transition states")
    };
    let species_section = species_section(view);
    let species_link = if species_section.is_empty() {
        ""
    } else {
        "<a href=\"#species\">Species</a>"
    };
    let crossing_link = if crossings == 0 {
        ""
    } else {
        "<a href=\"#crossings\">Crossing points</a>"
    };
    let (first, models) = first_details(view);
    let e = network.evaluations;
    let models = models.replace("</", "<\\/");
    let map_tab = if view.map.is_ok() {
        "<button type=\"button\" data-view-choice=\"map\">Map</button>"
    } else {
        ""
    };
    // The surfaces of a run of several states, one at a time, all, or in layers.
    let surfaces = if network.states.len() < 2 {
        String::new()
    } else {
        let mut buttons = String::new();
        for k in 0..network.states.len() {
            let _ = write!(
                buttons,
                "<button type=\"button\" data-surface=\"{k}\">{}</button>",
                crate::workflow::mapper::surface_name(k)
            );
        }
        format!(
            "<div class=\"toolrow needs-js\" data-surface-bar><span class=\"seg-label\">\
             Surfaces</span><div class=\"seg\" role=\"group\" aria-label=\"Surfaces of the \
             network\">{buttons}<button type=\"button\" data-surface=\"-1\">All</button>\
             <button type=\"button\" data-surface=\"-2\">Layers</button></div></div>"
        )
    };
    // The disconnectivity graph and the energy diagram of every species on its own (those of
    // its conformers and what joins them), shown in the place of those of the whole network
    // when the species is chosen; a species of one minimum has a line instead.
    let species_figures = {
        let species = view.species();
        let (mut dgs, mut diagrams) = (String::new(), String::new());
        if species.len() >= 2 {
            for (k, s) in species.iter().enumerate().take(SPECIES_FIGURES) {
                let caption = format!(
                    "S{} · <code>{}</code>: {} conformer(s); energies above the lowest minimum of \
                     the whole network.",
                    k + 1,
                    escape(&s.smiles),
                    s.members.len()
                );
                if s.members.len() < 2 {
                    let line = format!(
                        "<p class=\"note\" data-species-figure=\"{k}\" hidden>{caption} One \
                         minimum: nothing to draw.</p>"
                    );
                    dgs.push_str(&line);
                    diagrams.push_str(&line);
                    continue;
                }
                let only: HashSet<usize> = s.members.iter().map(|&i| view.minima[i]).collect();
                let part = View::of(view.network, view.options.clone(), Some(&only));
                let _ = write!(
                    dgs,
                    "<figure data-species-figure=\"{k}\" hidden>{}<figcaption>{caption}</figcaption>\
                     </figure>",
                    disconnectivity_graph(&part)
                );
                let _ = write!(
                    diagrams,
                    "<figure data-species-figure=\"{k}\" hidden>{}<figcaption>{caption}</figcaption>\
                     </figure>",
                    energy_diagram(&part)
                );
            }
        }
        (dgs, diagrams)
    };
    let circle_modes = if view.species().len() < 2 {
        ""
    } else {
        "<div class=\"toolrow needs-js\"><span class=\"seg-label\">Circles</span><div \
         class=\"seg\" role=\"group\" aria-label=\"The circles of the figure\"><button \
         type=\"button\" data-circle-mode=\"one\">One circle</button><button type=\"button\" \
         data-circle-mode=\"species\">A circle per species</button></div></div>"
    };
    // The species of the figures: all, or the conformers of one.
    let species_select = {
        let species = view.species();
        if species.len() < 2 {
            String::new()
        } else {
            let mut options = String::from("<option value=\"-1\">All species</option>");
            for (k, s) in species.iter().enumerate() {
                let _ = write!(
                    options,
                    "<option value=\"{k}\">S{} · {} ({})</option>",
                    k + 1,
                    escape(&s.smiles),
                    s.members.len()
                );
            }
            format!(
                "<label class=\"seg-label needs-js-inline\">Species <select data-species-select \
                 aria-label=\"The species shown in the figures\">{options}</select></label>"
            )
        }
    };
    let crossing_section = if crossings == 0 {
        String::new()
    } else {
        format!(
            "<section class=\"card glass\" id=\"crossings\"><h2>Crossing points</h2><div \
             class=\"table-wrap\"><table><thead><tr><th>node</th><th>relative ({unit})</th>\
             <th>energy (Hartree)</th><th>sides</th><th>above side 1</th><th>above side 2</th>\
             <th>|Δg| (Hartree/Bohr); gap (Hartree); within the seam</th><th>on the tree</th>\
             <th></th></tr></thead><tbody>{}\
             </tbody></table></div></section>",
            link_rows(view, LinkKind::Crossing)
        )
    };
    let has_channels = network.of_kind(&NodeKind::Dissociated).next().is_some();
    let channel_link = if has_channels {
        "<a href=\"#channels\">Channels</a>"
    } else {
        ""
    };
    let channel_section = if !has_channels {
        String::new()
    } else {
        format!(
            "<section class=\"card glass\" id=\"channels\"><h2>Dissociation channels</h2><div \
             class=\"table-wrap\"><table><thead><tr><th>node</th><th>relative ({unit})</th>\
             <th>energy (Hartree)</th><th>fragments</th><th>complexes (above them)</th>\
             <th>transition states</th></tr></thead><tbody>{}</tbody></table></div><p \
             class=\"note\">A channel is its fragments apart, each optimized on its own; its \
             energy is the sum of theirs. A complex of two molecules dissociates into its \
             channel without a barrier.{}{}</p></section>",
            channel_rows(view),
            entrance_note(network),
            scan_note(network)
        )
    };
    let mut shown_counts = format!(
        "{} minima and {} transition states or crossing points are shown",
        view.minima.len(),
        view.saddles.len()
    );
    if !network.path_peaks.is_empty() {
        // Path peaks are records beside the network: counted here, drawn nowhere.
        shown_counts.push_str(&format!(
            "; {} path peak(s) (the highest points of paths that gave no verified transition \
             state) are saved in path_peaks.csv and are no part of the network",
            network.path_peaks.len()
        ));
    }
    format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reaction network {title}</title>
<style>{style}{network_style}</style>
</head>
<body>
<header class="topbar glass"><div class="topbar-inner">
<div class="brand">Reaction network · {title}</div>
<nav aria-label="Sections"><a href="#summary">Summary</a><a href="#network">Network</a><a href="#paths">Paths</a><a href="#minima">Minima</a>{species_link}<a href="#ts">Transition states</a>{crossing_link}{channel_link}{higher_link}{kinetics_link}<a href="#warnings">Warnings</a><a href="#jobs">Jobs</a><a href="#settings">Settings</a><a href="#credits">Credits</a></nav>
<div class="seg" role="group" aria-label="Theme"><button type="button" data-theme-choice="auto">Auto</button><button type="button" data-theme-choice="light">Light</button><button type="button" data-theme-choice="dark">Dark</button></div>
</div></header>
<main>
<div class="hero">
<h1>Reaction network · {title}</h1>
<p>mopt-rs {version} · calculator: {calculator} · {found_short}, {jobs_count} job(s)</p>
<p class="{state_class}">{state}</p>
</div>
{missing}
<section class="card glass" id="summary"><h2>Summary</h2>
<p>{found_long} from {inputs} input structure(s) in {jobs_count} job(s); {warn_count} warning(s); {shown_counts}. Energies relative to the lowest minimum, in {unit}; populations by Boltzmann at {temperature} K of the {weighted}.</p>
{hidden}{part}{goal_note}
<p class="note">Evaluations of the run: {gradients} gradient calculation(s), {hessians} exact Hessian calculation(s){energies}; {micro}.</p></section>
<section class="card glass" id="network"><h2>Network</h2>
{surfaces}<div class="toolrow needs-js"><span class="seg-label">Figure</span><div class="seg" role="group" aria-label="Figure of the network">{map_tab}<button type="button" data-view-choice="graph">Graph</button><button type="button" data-view-choice="circle">Circle</button><button type="button" data-view-choice="dg">Disconnectivity graph</button><button type="button" data-view-choice="diagram">Energy diagram</button></div>
{molecules}{labels}{species_select}</div>
<div class="explore"><div>
<div class="view-panel" data-view="map"><h3>Map</h3><div class="axisbar needs-js"><label>Horizontal <select data-axis="0" aria-label="Horizontal axis of the map"></select></label><label>Vertical <select data-axis="1" aria-label="Vertical axis of the map"></select></label></div><div class="canvas-wrap needs-js" data-canvas="map"></div>{map_quality}<div class="no-js-only">{map}</div><p class="note">{map_note} <span class="needs-js-inline">Drag to move, Ctrl + wheel or the buttons to zoom, double-click to fit.</span></p></div>
<div class="view-panel" data-view="graph"><h3>Graph</h3><div class="canvas-wrap needs-js" data-canvas="graph"></div><div class="no-js-only">{graph}</div><p class="note">The minima as nodes, each link (transition state, diamond) as an edge between its two minima; the places are a force-directed layout of the links (Fruchterman and Reingold 1991), the lowest minima in the middle; distances on the page mean nothing more. <span class="needs-js-inline">Drag a minimum to move it (its transition states follow), drag elsewhere to move the view, Ctrl + wheel or the buttons to zoom, double-click to fit.</span></p></div>
<div class="view-panel" data-view="circle"><h3>Circle</h3>{circle_modes}<div class="canvas-wrap needs-js" data-canvas="circle"></div><p class="note no-js-only">This figure needs scripts.</p><p class="note">The minima on a circle and every link as a chord through its transition state. With a lowest-barrier path chosen (below, or with the buttons “Chosen node as start” and “Chosen node as end”), its start is at the left, its end at the right and its minima in their order along the upper half; the other minima are along the lower half, each under the minimum of the path it is joined to in the fewest steps. Without a path the minima follow round the circle in the order of their steps from the lowest minimum. With a species chosen, the circle holds its conformers alone. “A circle per species” draws every species as a circle of its conformers, the circles round a ring, and the links between two species as lines from circle to circle.</p></div>
<div class="view-panel" data-view="dg"><h3>Disconnectivity graph</h3>{species_dg}<figure data-species-figure="-1">{dg}<figcaption>Each minimum is a line from its energy up to where its basin merges with another, at the energy of the lowest transition state between them (diamonds); basins that nothing joins reach the top. The highest point between two minima is where their lines first meet (Becker and Karplus 1997). The {figure_minima} lowest minima are drawn; the merges are those of the whole network.</figcaption></figure></div>
<div class="view-panel" data-view="diagram"><h3>Energy diagram</h3>{species_diagram}<figure data-species-figure="-1">{diagram}<figcaption>Minima (green levels) side by side, lowest first; every transition state above the middle of its two minima. The {figure_minima} lowest minima are drawn.</figcaption></figure></div>
</div>
<aside class="card glass" id="details" aria-live="polite">{first}</aside></div>
</section>
<section class="card glass" id="paths"><h2>Lowest-barrier path</h2>
<p class="note">The path between two minima whose highest point is the lowest of all paths in the network shown (the path of the minimum spanning tree with the energies of the transition states as weights), drawn along the IRCs of its steps.</p>
<div class="pathbar needs-js"><label>From <input id="path-from" list="minima-list" size="9" aria-label="From (a minimum)"></label><label>to <input id="path-to" list="minima-list" size="9" aria-label="To (a minimum)"></label><button type="button" class="dl" id="path-find">Find</button><button type="button" class="dl" id="path-swap">Swap</button><button type="button" class="dl" id="path-use-from">Chosen node as start</button><button type="button" class="dl" id="path-use-to">Chosen node as end</button><button type="button" class="dl" id="path-clear">Clear</button><button type="button" class="dl" id="path-play">Play the path</button><button type="button" class="dl" id="path-save">Download the path (.xyz)</button><datalist id="minima-list"></datalist></div>
<div id="path-result" class="needs-js"></div>
<p class="note no-js-only">Finding paths needs scripts; the table of the minima gives the highest point on the way from the lowest minimum.</p></section>
<section class="card glass" id="minima"><h2>Minima</h2>
<div class="pager needs-js" data-pager="minima"></div>
<div class="table-wrap"><table id="minima-table"><thead><tr><th>node</th><th>relative ({unit})</th><th>energy (Hartree)</th><th>population</th><th>highest on the way from the lowest ({unit})</th><th>lowest wavenumbers (cm⁻¹)</th><th>jobs from it</th><th>found by</th><th>not explored</th><th></th></tr></thead><tbody>{minima_rows}</tbody></table></div></section>
<section class="card glass" id="ts"><h2>Transition states</h2>
<div class="pager needs-js" data-pager="ts"></div>
<div class="table-wrap"><table id="ts-table"><thead><tr><th>node</th><th>relative ({unit})</th><th>energy (Hartree)</th><th>IRC ends</th><th>barrier from end 1</th><th>barrier from end 2</th><th>imaginary modes: number and wavenumbers (cm⁻¹)</th><th>on the tree</th><th></th></tr></thead><tbody>{ts_rows}</tbody></table></div>
<p class="note">“On the tree”: the transition state is on the minimum spanning tree, so it lies on the lowest-barrier paths. The buttons (xyz, IRC) save the structures and paths that this page holds.</p>
{profiles}</section>
{crossing_section}
{species_section}
{channel_section}
{higher_section}
{kinetics_section}
<section class="card glass" id="warnings"><h2>Warnings</h2>{warnings}</section>
<section class="card glass" id="jobs"><h2>Jobs</h2>{progress_links}{jobs}</section>
<section class="card glass" id="settings"><h2>Settings</h2><details><summary>The settings of the run</summary><pre>{settings}</pre></details>
<p class="note">The files of the run are in its directory: reaction_network.json, nodes.csv, ts_edges.csv, structures/, paths/, jobs/, mopt.log. <code>run_mapper --report RUN_DIR</code> writes this page again from them, with the settings of <code>-report_opt</code>.</p></section>
<section class="card glass" id="credits"><h2>Rights and credits</h2><p>This report was generated by mopt-rs {version}, Copyright © 2023–2026 ss0832, free software under the GNU General Public License, version 3 or later (GPL-3.0-or-later), with no warranty. The structure viewer, the figures, the style and the code embedded in this page are part of mopt-rs, under the same license; the energy scale is the viridis map of matplotlib (CC0). The structures, energies and other results on this page are the output of the calculation that was run; mopt-rs claims no rights to them.</p></section>
</main>
<script type="application/json" id="models">{models}</script>
<script type="application/json" id="page-info">{{"layers":null}}</script>
<script type="application/json" id="network-data">{data}</script>
{viewer}
{script}
</body>
</html>
"##,
        molecules = if view.network.variables.is_some() {
            ""
        } else {
            "<span class=\"seg-label\">Molecules</span><div class=\"seg\" role=\"group\" \
             aria-label=\"Display style of every molecule\"><button type=\"button\" \
             data-style-choice=\"ball\">Ball and stick</button><button type=\"button\" \
             data-style-choice=\"stick\">Sticks</button><button type=\"button\" \
             data-style-choice=\"spacefill\">Space-filling</button><button type=\"button\" \
             data-style-choice=\"wire\">Wireframe</button></div>"
        },
        labels = if view.network.variables.is_some() {
            ""
        } else {
            LABEL_CONTROLS
        },
        map_note = map_note(view),
        title = escape(&report.title),
        style = STYLE,
        network_style = NETWORK_STYLE,
        viewer = VIEWER,
        script = NETWORK_SCRIPT,
        version = crate::VERSION,
        calculator = escape(&report.calculator),
        state_class = if ended { "note" } else { "warn" },
        state = escape(state),
        inputs = network.inputs.len(),
        jobs_count = network.jobs.len(),
        warn_count = network.warnings.len(),
        temperature = view.options.temperature,
        weighted = if view.free_energies {
            "free energies of the minima shown"
        } else {
            "energies of the minima shown (not all have a free energy)"
        },
        hidden = hidden_note(view),
        missing = missing_section(notes),
        part = part_note(view),
        gradients = e.gradients,
        hessians = e.hessians,
        energies = if e.energies > 0 {
            format!(", {} of the energy alone", e.energies)
        } else {
            String::new()
        },
        micro = if e.micro_relaxations > 0 {
            format!(
                "{} microiteration(s) in {} relaxation(s) of the ONIOM environment (their \
                 gradients of the lowest level are among the gradients)",
                e.micro_iterations, e.micro_relaxations
            )
        } else {
            "no microiterations (no ONIOM environment was relaxed)".to_string()
        },
        map = network_map(view),
        map_quality = map_quality(view),
        graph = network_graph(view),
        dg = disconnectivity_graph(view),
        diagram = energy_diagram(view),
        species_dg = species_figures.0,
        species_diagram = species_figures.1,
        circle_modes = circle_modes,
        figure_minima = view.minima.len().min(view.options.figure_minima),
        minima_rows = minima_rows(view),
        ts_rows = link_rows(view, LinkKind::TransitionState),
        profiles = static_profiles(view),
        warnings = warnings(network),
        progress_links = {
            let dir = report.path.parent().unwrap_or_else(|| Path::new("."));
            let mut links = Vec::new();
            if dir.join("jobs/index.html").is_file() {
                links.push("<a href=\"jobs/index.html\">Live job progress and search files</a>");
            }
            if dir.join("exploration_history.log").is_file() {
                links.push("<a href=\"exploration_history.log\">Exploration history</a>");
            }
            if dir.join("rcmc_K_contracted.csv").is_file() {
                links.push("<a href=\"rcmc_K_contracted.csv\">Latest RCMC contraction</a>");
            }
            if dir.join("queue_priority.log").is_file() {
                links.push("<a href=\"queue_priority.log\">Latest exploration priorities</a>");
            }
            links.join(" · ")
        },
        jobs = jobs(network),
        settings = escape(&report.settings),
        data = view.data(),
        higher_section = higher_saddle_section(view),
        higher_link = if network
            .nodes
            .iter()
            .any(|n| matches!(n.kind, NodeKind::HigherSaddle { .. }))
        {
            "<a href=\"#higher-saddles\">Higher saddles</a>"
        } else {
            ""
        },
    )
}

impl MapperObserver for MapperReport {
    fn after_inputs(&mut self, network: &Network) -> MoptResult<()> {
        self.write_in_run(
            network,
            "The run had not ended when this page was written (the inputs are optimized). The \
             page is written again after jobs: if the run was stopped, this is what it had, and \
             run_mapper --report RUN_DIR writes it from the files of the run.",
            false,
        )
    }

    fn after_job(
        &mut self,
        network: &Network,
        job: &Job,
        _outcome: &JobOutcome,
        _told: &Told,
    ) -> MoptResult<()> {
        if !self.due() {
            return Ok(());
        }
        self.write_in_run(
            network,
            &format!(
                "The run had not ended when this page was written (after job {}). The page is \
                 written again after jobs: if the run was stopped, this is what it had, and \
                 run_mapper --report RUN_DIR writes it from the files of the run.",
                job.id
            ),
            false,
        )
    }

    fn on_finish(&mut self, network: &Network, end: &ExplorationEnd) -> MoptResult<()> {
        let state = match (&end.reason, end.interrupted) {
            (_, true) => "The run was interrupted; --restart with its run directory continues it. \
                          This page shows what it had."
                .to_string(),
            (Some(reason), false) => format!("The run ended: {reason}"),
            (None, false) => "The run ended.".to_string(),
        };
        self.write_in_run(network, &state, !end.interrupted)
    }
}

/// The settings `text` (JSON) as the page shows them: every value that is an absolute path,
/// or `KEY=PATH` with an absolute path, cut down to its file name, so that the page carries no
/// directories of the place it was written in. Text that is not JSON is shown as it is.
fn settings_for_page(text: &str) -> String {
    match JsonValue::parse(text) {
        Ok(doc) => without_directories(doc).to_string_pretty(),
        Err(_) => text.to_string(),
    }
}

fn without_directories(value: JsonValue) -> JsonValue {
    match value {
        JsonValue::String(s) => JsonValue::String(file_name_of(&s)),
        JsonValue::Array(items) => {
            JsonValue::Array(items.into_iter().map(without_directories).collect())
        }
        JsonValue::Object(m) => JsonValue::Object(
            m.into_iter()
                .map(|(k, v)| (k, without_directories(v)))
                .collect(),
        ),
        other => other,
    }
}

/// `s` with an absolute path (`/...`, `\...`, `C:...`), alone or after `KEY=`, cut down to
/// its file name; anything else as it is.
fn file_name_of(s: &str) -> String {
    let (key, value) = match s.split_once('=') {
        Some((k, v)) if !k.contains(['/', '\\']) => (Some(k), v),
        _ => (None, s),
    };
    let bytes = value.as_bytes();
    let drive = bytes.len() > 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    if !(value.starts_with('/') || value.starts_with('\\') || drive) {
        return s.to_string();
    }
    let name = value.rsplit(['/', '\\']).next().unwrap_or(value);
    match key {
        Some(k) => format!("{k}={name}"),
        None => name.to_string(),
    }
}

#[cfg(test)]
#[path = "mapper_report_tests.rs"]
mod tests;
