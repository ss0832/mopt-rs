//! `orientsearch`: random relative orientations and placements of the parts of a complex,
//! each optimized by `optmain`, and the distinct minima ranked by energy with their Boltzmann
//! weights.
//!
//! The parts are `-part` lists (MultiOptPy; the atoms no list names are one more part) or the
//! molecules of the bond graph. Every sample is written as `<stem>_Sample_<k>.xyz` (MultiOptPy's
//! name) and optimized by `optmain` with all of its options in `<stem>_Sample_<k>_OPT/` of the
//! run directory, so every calculator of the registry, ONIOM, periodic cells, `-mf`, bias
//! potentials, constraints, and `-coord` work as they do there. The search
//! ([`crate::workflow::explore::orient`]) merges the results that are the same minimum (bond
//! graph, atoms and identical molecules exchanged, RMSD, energy) and writes the minima, their
//! table, and the samples ([`super::orientsearch_output`]).
//!
//! A `STOP` file in the run directory ends the search after the sample in progress (a `STOP`
//! file in the directory of that sample ends it at once); `--restart RUN_DIR` repeats the
//! samples with the same seed and answers their evaluations from their journals.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py:run_orientsearch`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use super::constraints::resolve_for_run;
use super::option::reject_unavailable;
use super::optmain::{self, OptmainConfig};
use super::orientsearch_energies::PartEnergies;
use super::orientsearch_output::{self as output, SampleRun};
use super::orientsearch_settings::{parse_orient_options, Criteria, Interaction};
use super::settings::settings_json;
use super::spec::ORIENTSEARCH;
use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{Quantity, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{read_xyz_file, write_xyz_file};
use crate::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use crate::pes::counting::EvaluationCounts;
use crate::pes::objective::ObjectiveRegistry;
use crate::pes::registry::CalculatorRegistry;
use crate::workflow::explore::orient::{
    ComplexParts, OrientAction, OrientSettings, OrientationSearch, SampleOutcome, SampleRecord,
    Scheme,
};
use crate::workflow::explore::PriorityRegistry;
use crate::workflow::observers::STOP_FILE;

/// Typed `orientsearch` settings.
#[derive(Debug, Clone)]
pub struct OrientsearchConfig {
    /// The complex (`INPUT`).
    pub input: PathBuf,
    /// The options as parsed (for the settings of the run and of each sample).
    pub parsed: ParsedArgs,
    /// Run directory to continue (`--restart`).
    pub restart_dir: Option<PathBuf>,
    /// The `optmain` settings every sample is optimized with.
    pub optmain: OptmainConfig,
    /// `-part`.
    pub part_lists: Vec<String>,
    /// `-orient_opt fixed=`: parts (from 1) that stay.
    pub fixed_parts: Vec<usize>,
    /// The search.
    pub search: OrientSettings,
    /// Whether the seed was given (otherwise drawn and written to the settings of the run).
    pub seed_given: bool,
    /// Energies of the parts alone.
    pub interaction: Interaction,
    /// `-orient_opt part_charges=`.
    pub part_charges: Option<Vec<i32>>,
    /// `-orient_opt part_multiplicities=`.
    pub part_multiplicities: Option<Vec<u32>>,
    /// `-temp`: the temperature of the Boltzmann weights (K).
    pub temperature: f64,
    /// `-debug`: the logs of the optimizations on the screen as well.
    pub debug: bool,
    /// Whether `-dist` was given.
    pub distance_given: bool,
}

fn clock_seed() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    ((now as u64) ^ ((now >> 64) as u64)) % 1_000_000_007 + 1
}

impl OrientsearchConfig {
    /// Settings from parsed arguments.
    pub fn from_args(p: &ParsedArgs) -> MoptResult<Self> {
        // The state that the input file names, for the runs started on files written here.
        let with_state = super::charge_spin::with_state_of_input(p);
        let p = &with_state;
        reject_unavailable(p, ORIENTSEARCH)?;
        let input = match p.positionals.as_slice() {
            [one] => PathBuf::from(one),
            [] => {
                return Err(MoptError::config(
                    "orientsearch needs an input structure (xyz)",
                ))
            }
            _ => {
                return Err(MoptError::config(
                    "orientsearch takes one input structure (the complex)",
                ))
            }
        };
        let mut o = parse_orient_options(p.get_list("orient_options"))?;
        // The criteria of the samples: -tcc or -lcc as given, otherwise `criteria=` (tight
        // by default: the default criteria end optimizations of complexes on the flat part of
        // the surface, see running.md).
        let mut parsed = p.clone();
        let given = p.get_bool("tight_convergence_criteria", false)
            || p.get_bool("loose_convergence_criteria", false);
        if !given {
            match o.criteria {
                Criteria::Tight => {
                    parsed
                        .present
                        .insert("tight_convergence_criteria".to_string());
                }
                Criteria::Loose => {
                    parsed
                        .present
                        .insert("loose_convergence_criteria".to_string());
                }
                Criteria::Normal => {}
            }
        }
        let p = &parsed;
        let optmain = OptmainConfig::from_args(p)?;
        if ObjectiveRegistry::builtin().contains(&optmain.calculator) {
            return Err(MoptError::unsupported(
                "orientsearch places the parts of a structure; an objective of plain variables \
                 has no parts",
            ));
        }
        if optmain
            .model_function
            .first()
            .is_some_and(|m| m.eq_ignore_ascii_case("bitss"))
        {
            return Err(MoptError::unsupported(
                "-mf bitss is a search between two given structures, not an optimization of a \
                 sampled one",
            ));
        }
        let samples = p.get_usize("number_of_samples", 5)?;
        if !o.seed_given {
            o.search.seed = clock_seed();
        }
        o.search.samples = samples;
        let distance = p.get_f64("distance", 5.0)?;
        o.search.placement.distance = Quantity::new(distance * Units::ANGSTROM_TO_BOHR);
        Ok(Self {
            input,
            parsed: p.clone(),
            restart_dir: None,
            optmain,
            part_lists: p.get_list("part").to_vec(),
            fixed_parts: o.fixed,
            search: o.search,
            seed_given: o.seed_given,
            interaction: o.interaction,
            part_charges: o.part_charges,
            part_multiplicities: o.part_multiplicities,
            temperature: p.get_f64("temperature", 298.15)?,
            debug: p.get_bool("debug", false),
            distance_given: p.is_present("distance"),
        })
    }

    /// The settings document of the run (with the seed in use).
    pub fn settings_document(&self) -> String {
        let mut p = self.parsed.clone();
        let mut options: Vec<String> = p
            .get_list("orient_options")
            .iter()
            .filter(|t| !t.trim().to_ascii_lowercase().starts_with("seed="))
            .cloned()
            .collect();
        options.push(format!("seed={}", self.search.seed));
        p.values.insert("orient_options".to_string(), options);
        p.present.insert("orient_options".to_string());
        settings_json(&p, ORIENTSEARCH).to_string_pretty()
    }

    /// The `optmain` settings of one sample: the options of the run with `start` as the input,
    /// in `directory`.
    fn sample_config(&self, start: &Path, directory: &Path) -> MoptResult<OptmainConfig> {
        let mut p = self.parsed.clone();
        p.positionals = vec![start.to_string_lossy().into_owned()];
        let mut c = OptmainConfig::from_args(&p)?;
        c.restart_dir = Some(directory.to_path_buf());
        Ok(c)
    }
}

/// Result of a run.
#[derive(Debug, Clone)]
pub struct OrientsearchResult {
    /// Run directory.
    pub directory: PathBuf,
    /// The distinct minima, lowest first.
    pub minima: Vec<output::MinimumReport>,
    /// What became of every sample.
    pub samples: Vec<SampleRecord>,
    /// Whether a `STOP` file ended the run (`--restart` continues it).
    pub interrupted: bool,
    /// The evaluations of the run.
    pub evaluations: EvaluationCounts,
}

/// MultiOptPy-style run directory `<date>/<stem>_ORIENT_<timestamp>/` under `base`.
pub fn run_directory(base: &Path, input: &Path, now: &DateTime) -> PathBuf {
    let stem = input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    base.join(now.date_dir())
        .join(format!("{stem}_ORIENT_{}", now.timestamp()))
}

/// Run `orientsearch` under `base` with the calculators of `registry`.
pub fn run_with(
    config: &OrientsearchConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<OrientsearchResult> {
    let before = registry.counts();
    let spin = config.optmain.state.of_file(&config.input)?;
    let template = read_xyz_file(&config.input, spin, None)?;
    let bond_scale = config.search.matcher.bond_scale;
    let mut parts = if config.part_lists.is_empty() {
        ComplexParts::molecules(&template, &config.fixed_parts, bond_scale)?
    } else {
        ComplexParts::from_lists(
            &template,
            &config.part_lists,
            &config.fixed_parts,
            bond_scale,
        )?
    };
    // The constraints resolved once, at the input: every sample holds these values (a
    // constraint between atoms of two parts holds the value of the input, not that of the
    // random placement), and a part with atoms that the constraints pin in space stays.
    let constraints = resolve_for_run(&config.optmain.settings.constraints, &template)
        .context("constraints at the input")?;
    let held = parts.hold_parts_with(&pinned_atoms(&constraints))?;
    let biases = super::oniom_micro::build_terms(&config.optmain.biases, &template)?;
    super::confsearch_jobs::refuse_unrankable_terms("orientsearch", &biases)?;
    let on_constraints = if constraints.is_empty() || constraints.iter().any(Constraint::is_pending)
    {
        None
    } else {
        Some(ConstraintSet::new(&constraints, &template, None)?)
    };
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => run_directory(base, &config.input, &DateTime::now_utc()),
    };
    std::fs::create_dir_all(&dir)?;
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document()),
    )?;
    let stem = config
        .input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    let mut search = OrientationSearch::new(
        &template,
        &parts,
        config.search.clone(),
        &PriorityRegistry::default(),
    )?;
    output::log_header(config, &template, &parts, &search, &dir, &log);
    if !held.is_empty() {
        log.info(&format!(
            "Part(s) {} hold atoms that the constraints pin in space (-fix, -pc x|y|z|rot): \
             they stay where they are in the input.",
            held.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !constraints.is_empty() {
        log.info(
            "Constraints: resolved at the input; every sample is restored onto them before it \
             is optimized (a constraint between atoms of two parts holds the value of the \
             input).",
        );
    }
    log.info(
        "Ranking: minima by their energy with the bias potentials (the surface they are minima \
         of); the bare energy and the interaction energies without the bias potentials are \
         listed as well.",
    );

    let sample_console = config.debug.then_some(Level::Debug);
    let mut runs: Vec<SampleRun> = Vec::new();
    let mut interrupted = false;
    let mut first_failure: Option<String> = None;
    loop {
        let stop = dir.join(STOP_FILE);
        if stop.exists() {
            let _ = std::fs::remove_file(&stop);
            log.info(&format!(
                "{STOP_FILE} found: the search ends here. Continue with: orientsearch --restart {}",
                dir.display()
            ));
            interrupted = true;
            break;
        }
        let request = match search.ask()? {
            OrientAction::Finished => break,
            OrientAction::Optimize(r) => r,
        };
        let k = request.index;
        let start_file = dir.join(format!("{stem}_Sample_{k}.xyz"));
        let sample_dir = dir.join(format!("{stem}_Sample_{k}_OPT"));
        let placed = match &on_constraints {
            Some(set) => set.restore(&request.start),
            None => Ok(request.start.clone()),
        };
        let mut start = match placed {
            Ok(s) => s,
            Err(e) => {
                // Written as placed, for a look; not optimized.
                let mut s = request.start.clone();
                s.comment = format!("Sample_{k} (not on the constraints)");
                write_xyz_file(&start_file, &s, None)?;
                log.warn(&format!(
                    "Sample {k}: the placement could not be restored onto the constraints ({e}); \
                     the sample is discarded."
                ));
                let run = SampleRun::new(
                    k,
                    request.origin,
                    start_file,
                    sample_dir,
                    EvaluationCounts::default(),
                );
                let record = search.tell_with_bare(None, None)?;
                output::log_sample(&search, &record, &run, &log);
                runs.push(run);
                continue;
            }
        };
        start.comment = format!("Sample_{k}");
        write_xyz_file(&start_file, &start, None)?;
        let counted = registry.counts();
        let result = config
            .sample_config(&start_file, &sample_dir)
            .map(|mut c| {
                c.settings.constraints.clone_from(&constraints);
                c
            })
            .and_then(|c| optmain::run_with(&c, &dir, sample_console, registry));
        let used = registry.counts().since(&counted);
        let mut run = SampleRun::new(k, request.origin, start_file, sample_dir, used);
        let outcome = match result.map(|mut v| v.pop()) {
            Ok(Some(r)) => {
                if r.interrupted {
                    log.info(&format!(
                        "Sample {k} was interrupted; continue with: orientsearch --restart {}",
                        dir.display()
                    ));
                    interrupted = true;
                    runs.push(run);
                    break;
                }
                let summary = r.summary.ok_or_else(|| {
                    MoptError::calc(format!(
                        "sample {k}: the optimization reported no structure"
                    ))
                })?;
                run.finished(&summary, r.converged, r.optimized);
                // Ranked on the surface the sample was minimized on (with the bias
                // potentials), as confsearch ranks; the bare energy goes along.
                Some((
                    SampleOutcome {
                        system: summary.system,
                        energy: summary.biased_energy,
                        converged: r.converged,
                    },
                    summary.energy,
                ))
            }
            Ok(None) => None,
            Err(e) => {
                log.warn(&format!("Sample {k} failed: {e}"));
                first_failure.get_or_insert_with(|| e.to_string());
                None
            }
        };
        let (outcome, bare) = match outcome {
            Some((o, b)) => (Some(o), Some(b)),
            None => (None, None),
        };
        let record = search.tell_with_bare(outcome, bare)?;
        output::log_sample(&search, &record, &run, &log);
        runs.push(run);
    }
    if search.pool().is_empty() {
        if let Some(e) = first_failure {
            return Err(MoptError::calc(format!(
                "orientsearch: no sample gave a minimum; the first failure: {e}"
            )));
        }
    }
    let energies = PartEnergies::compute(config, registry, &template, &parts, &search, &dir, &log)?;
    let minima = output::write_results(
        config, &template, &search, &runs, &energies, &stem, &dir, &log,
    )?;
    let evaluations = registry.counts().since(&before);
    log.info(&evaluations.line("orientsearch"));
    log.info(&format!(
        "Orientation search done: {} samples, {} distinct minima; results in {}",
        search.records().len(),
        search.pool().len(),
        dir.display()
    ));
    log.flush();
    Ok(OrientsearchResult {
        directory: dir,
        minima,
        samples: search.records().to_vec(),
        interrupted,
        evaluations,
    })
}

/// The atoms (from 0) that `constraints` pin in space: frozen atoms, atoms with a Cartesian
/// component held, and oriented groups.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn pinned_atoms(constraints: &[Constraint]) -> Vec<usize> {
    let mut atoms: Vec<usize> = constraints
        .iter()
        .flat_map(|c| match &c.kind {
            ConstraintKind::Frozen(a) | ConstraintKind::Orientation(a) => a.clone(),
            ConstraintKind::Coordinate(a, _) => vec![*a],
            _ => Vec::new(),
        })
        .map(|a| a.index())
        .collect();
    atoms.sort_unstable();
    atoms.dedup();
    atoms
}

/// Whether the scheme in use takes `-dist`.
pub fn uses_distance(scheme: Scheme) -> bool {
    matches!(scheme, Scheme::MultiOptPy | Scheme::Uniform)
}

/// Thin wrapper used by tests and embedding code: settings for `args`.
pub fn config_from_cli(args: &[&str]) -> MoptResult<OrientsearchConfig> {
    let p = super::option::parser("orientsearch", ORIENTSEARCH).parse(args)?;
    OrientsearchConfig::from_args(&p)
}
