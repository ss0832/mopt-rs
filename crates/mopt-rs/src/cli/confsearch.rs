//! `confsearch`: the conformers of a structure (the minima with its bond graph), without
//! duplicates, with rotamers counted and Boltzmann populations; or every minimum
//! (`-cs_opt mode=minima`); or the conformers of a transition state (`-tsconf`).
//!
//! Strategies (`-cs_opt method=`): `afir`, MultiOptPy's sampling with AFIR forces between
//! pairs of atoms (the default; `-bf`, `-ms`, `-nl`, `-nr`, `-tgta`, `-st`, `-nost`, `-tabu`,
//! `-alpha`), and `imtd`, the iMTD-GC workflow (RMSD metadynamics, multi-level optimization,
//! crossing, finishing dynamics). The optimizations take every option of `optmain`; the run
//! has its settings file, journal, restart, `STOP` file, and log as the other commands.
//! The parts live in `workflow::confsearch` and the exploration core `workflow::explore`.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py:run_confsearch`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::confsearch_files::{write_summary, ConfsearchFiles};
use super::confsearch_jobs::{input_spin, CommandJobs, SharedStop};
pub use super::confsearch_settings::{config_from_cli, ConfsearchConfig, Method};
use crate::core_infra::clock::DateTime;
use crate::core_infra::log::{python_float, Level, Logger};
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{format_xyz_system, read_xyz_file};
use crate::pes::objective::ObjectiveRegistry;
use crate::pes::registry::CalculatorRegistry;
use crate::workflow::confsearch::moves::{self, MoveRegistry, TrialSettings};
use crate::workflow::confsearch::ts::{self, TsResult};
use crate::workflow::confsearch::{afir, imtd, Mode, SearchJobs, SearchObserver, SearchResult};
use crate::workflow::explore::{HeatedBoltzmann, PriorityRegistry, SamplingPriority};
use crate::workflow::journal::{RunJournal, JOURNAL_FILE};

/// What a run left behind.
#[derive(Debug, Clone)]
pub struct ConfsearchResult {
    /// The run directory.
    pub directory: PathBuf,
    /// `conformers.xyz` (written when the search ended).
    pub conformers: PathBuf,
    /// Number of conformers (or of transition states with `-tsconf`).
    pub count: usize,
    /// Energies of the conformers, lowest first (Hartree).
    pub energies: Vec<f64>,
    /// Why the search ended.
    pub reason: String,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
}

/// MultiOptPy's run directory `<stem>_<bf>KJ_CS_<date>`, as `<date>/<stem>_<bf>KJ_CS_<time>/`
/// under `base` (the name of the file alone: an input elsewhere does not take the run there).
pub fn run_directory(base: &Path, input: &Path, base_force: f64, now: &DateTime) -> PathBuf {
    base.join(now.date_dir()).join(format!(
        "{}_{}{}_{}",
        stem_of(input),
        base_force as i64,
        super::confsearch_settings::RUN_TAG,
        now.timestamp()
    ))
}

/// The sampling priority of MultiOptPy's strategy.
fn priority(config: &ConfsearchConfig) -> MoptResult<Box<dyn SamplingPriority>> {
    let t = config.sampling_temperature;
    Ok(match config.priority.as_str() {
        "" | "boltzmann_heated" | "tabu_heated" => {
            let alpha = if config.priority == "tabu_heated" {
                Some(config.tabu.unwrap_or(0.5))
            } else {
                config.tabu
            };
            Box::new(HeatedBoltzmann::multioptpy(t, alpha))
        }
        "boltzmann" => Box::new(crate::workflow::explore::Boltzmann { temperature: t }),
        "tabu" => Box::new(crate::workflow::explore::PenalizedBoltzmann {
            temperature: t,
            alpha: config.tabu.unwrap_or(0.5),
        }),
        name => PriorityRegistry::default().create(name)?,
    })
}

fn stem_of(path: &Path) -> String {
    path.file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned())
}

/// Run `confsearch` with the calculators of `registry`, writing the run directory under `base`.
pub fn run_with(
    config: &ConfsearchConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<ConfsearchResult> {
    run_with_moves(config, base, console, registry, &MoveRegistry::builtin())
}

/// The search of `config` from `s` on the jobs `j`: MultiOptPy's AFIR sampling, iMTD-GC, or a
/// move of `moves`. `common` are the settings of every strategy as the caller wants them (those
/// of `config`, without the verification of the minima for a search with the saddle direction
/// held).
pub(crate) fn search_with(
    config: &ConfsearchConfig,
    common: &crate::workflow::confsearch::CommonSettings,
    moves: &MoveRegistry,
    log: &Logger,
    s: &MolecularSystem,
    j: &dyn SearchJobs,
    obs: &mut dyn SearchObserver,
) -> MoptResult<Option<SearchResult>> {
    match &config.method {
        Method::Afir => {
            let mut rule = priority(config)?;
            log.info(&format!("Sampling priority: {}", rule.name()));
            afir::run(s, &config.afir, common, rule.as_mut(), j, obs)
        }
        Method::Imtd => imtd::run(s, &config.imtd, common, j, obs),
        Method::Move(name) => {
            let mut mover = moves.create(name, &config.moves)?;
            let mut rule = priority(config)?;
            log.info(&format!("Sampling priority: {}", rule.name()));
            // The trials and the termination are those of the AFIR sampling.
            let trials = TrialSettings {
                max_samples: config.afir.max_samples,
                rank: config.afir.rank,
                patience: config.afir.patience,
                no_stochastic: config.afir.no_stochastic,
            };
            moves::run(s, mover.as_mut(), &trials, common, rule.as_mut(), j, obs)
        }
    }
}

/// [`run_with`] with the moves of `moves` (`-cs_opt method=NAME`): the built-in ones and
/// those a user registered ([`MoveRegistry::register`]).
pub fn run_with_moves(
    config: &ConfsearchConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    moves: &MoveRegistry,
) -> MoptResult<ConfsearchResult> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(
            config
                .optmain
                .calculator_options
                .pool_threads(config.optmain.threads),
        )
        .build()
        .map_err(|e| MoptError::config(format!("thread pool: {e}")))?;
    pool.install(|| run_inner(config, base, console, registry, moves))
        .context(config.input.display())
}

fn describe(config: &ConfsearchConfig, log: &Logger) {
    let c = &config.common;
    log.info(&format!(
        "Mode: {}; bond graphs compared {}",
        match c.mode {
            Mode::Conformers => "conformers (minima with the bond graph of the optimized input)",
            Mode::Minima => "minima (every distinct minimum)",
        },
        match c.graph_check {
            crate::workflow::confsearch::GraphCheck::Isomorphic => "up to a renumbering",
            crate::workflow::confsearch::GraphCheck::SameNumbering =>
                "in the numbering of the input (-pbc)",
        }
    ));
    log.info(&format!(
        "Duplicates: energy within {:.3} kcal/mol and RMSD below {:.3} Angstrom; rotational \
         constants within {} %; rotamers by renumbering{}; seed {}",
        c.compare.energy * Units::HARTREE_TO_KCAL_MOL,
        c.compare.rmsd * Units::BOHR_TO_ANGSTROM,
        python_float(100.0 * c.compare.rotational),
        if c.compare.mirror {
            " and mirror images"
        } else {
            ""
        },
        c.seed
    ));
    match &config.method {
        Method::Move(name) => {
            let settings: Vec<String> = config
                .moves
                .0
                .iter()
                .map(|(k, v)| format!("move_{k}={v}"))
                .collect();
            log.info(&format!(
                "Strategy: trials by the move {name}{}, at most {} samples, termination -nr {} \
                 -nl {}",
                if settings.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", settings.join(", "))
                },
                config.afir.max_samples,
                config.afir.rank,
                config.afir.patience
            ));
        }
        Method::Afir => log.info(&format!(
            "Strategy: AFIR sampling (MultiOptPy), {} kJ/mol, at most {} samples, termination \
             -nr {} -nl {}",
            python_float(config.afir.base_force),
            config.afir.max_samples,
            config.afir.rank,
            config.afir.patience
        )),
        Method::Imtd => {
            let m = &config.imtd;
            log.info(&format!(
                "Strategy: iMTD-GC (Grimme, J. Chem. Theory Comput. 2019, 15, 2847; Pracht, \
                 Bohle, Grimme, Phys. Chem. Chem. Phys. 2020, 22, 7169): {} metadynamics of {:.1} \
                 ps at {} K, steps of {:.1} fs, a snapshot every {:.0} fs, loose window {:.1} \
                 kcal/mol",
                m.runs.len(),
                m.mtd_time * Units::AU_TIME_TO_FS / 1000.0,
                python_float(m.temperature),
                m.time_step * Units::AU_TIME_TO_FS,
                m.dump_every * Units::AU_TIME_TO_FS,
                config.loose_window_kcal()
            ));
        }
    }
}

fn run_inner(
    config: &ConfsearchConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    moves: &MoveRegistry,
) -> MoptResult<ConfsearchResult> {
    // A move that the registry does not know, or settings it does not have, are refused
    // before anything is computed.
    if let Method::Move(name) = &config.method {
        moves.create(name, &config.moves)?;
    } else if let Some(key) = config.moves.0.keys().next() {
        return Err(MoptError::parse(format!(
            "-cs_opt move_{key}: a setting of a move (method=torsion, kick, lowmode, ...)"
        )));
    }
    if ObjectiveRegistry::builtin().contains(&config.optmain.calculator) {
        return Err(MoptError::unsupported(
            "an objective of plain variables has no atoms, so no bonds and no conformers",
        ));
    }
    let whole = read_xyz_file(&config.input, input_spin(config)?, None)?;
    let evaluations_before = registry.counts();
    let o = &config.optmain;
    // ONIOM microiterations: the conformers are those of the core (the model region with
    // the outer atoms of its links, and the atoms that constraints and bias potentials
    // name), searched on the relaxed surface: every optimization, every step of the
    // dynamics, and every Hessian has the environment relaxed to a minimum. The files are
    // written with the whole system at the end.
    // The constraints resolved once, at the input: every candidate, optimization, dynamics,
    // and IRC holds these values (frozen atoms where they are in the input, automatic -pc
    // values of the input), not those of the structure a move made.
    let input_constraints = super::constraints::resolve_for_run(&o.settings.constraints, &whole)
        .context("constraints at the input")?;
    let biases = super::oniom_micro::build_terms(&o.biases, &whole)?;
    super::confsearch_jobs::refuse_unrankable_terms("confsearch", &biases)?;
    // Every job takes its own copy of a term with state; one that cannot be copied is refused
    // here, before anything is computed.
    for (_, t) in &biases {
        crate::pes::term::fork_term(t)?;
    }
    let wrapped = super::oniom_micro::wrap_constrained(
        &o.calculator_options,
        super::oniom_micro::uncovered_by_optmain(o),
        &input_constraints,
        &biases,
        &super::oniom_micro::other_atoms_of_optmain(o),
        super::oniom_micro::forces_of(&o.settings.criteria),
        whole.clone(),
        &o.calculator,
        registry,
    )?;
    if wrapped.electrons {
        whole.spin_charge.check_electrons(&whole.elements)?;
    }
    let job_atoms = super::confsearch_jobs::JobAtoms {
        constraints: wrapped.constraints.clone(),
        terms: wrapped.terms.clone(),
        fragments: o
            .settings
            .fragments
            .iter()
            .map(|group| group.iter().map(|a| wrapped.on_core(*a)).collect())
            .collect::<MoptResult<_>>()?,
        ccqn_bonds: o
            .ccqn_bonds
            .iter()
            .map(|(a, b)| Ok((wrapped.on_core(*a)?, wrapped.on_core(*b)?)))
            .collect::<MoptResult<_>>()?,
        ccqn_product: match &o.ccqn_product {
            Some(product) => {
                Some(wrapped.structure(read_xyz_file(product, whole.spin_charge, None)?)?)
            }
            None => None,
        },
    };
    let (start, calc, micro, micro_note) =
        (wrapped.start, wrapped.calc, wrapped.micro, wrapped.note);
    // A structure whose bonds run through the cell into a network has no conformers; it is
    // refused before anything is computed.
    crate::coord::graph::whole_molecules(
        &start,
        &crate::coord::graph::MolecularGraph::from_system(&start, config.common.bond_scale),
    )?;
    let calc = crate::pes::o1numhess::wrap_if_requested(
        calc,
        o.settings.model_hessian.as_deref(),
        o.settings.exact_hessian_interval.is_some(),
    );
    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => run_directory(
            base,
            &config.input,
            config.afir.base_force,
            &DateTime::now_utc(),
        ),
    };
    std::fs::create_dir_all(&dir)?;
    let level = console.map(|l| if o.debug { Level::Debug } else { l });
    let log = level.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} confsearch: {}",
        crate::VERSION,
        config.input.display()
    ));
    log.info(&format!("Calculator: {}", calc.description()));
    for paper in calc.citations() {
        log.info(&format!("Reference: {paper}"));
    }
    log.info(&o.calculator_options.software_path_report());
    if let Some(note) = &micro_note {
        log.info(note);
    }
    super::oniom_micro::report_relaxations(micro.as_deref(), &log);
    if let Some(relaxed) = &micro {
        log.info(&format!(
            "ONIOM microiterations: conformers of the {} atoms the optimizer moves; the {} \
             atoms of the environment are relaxed to a minimum for every structure.",
            relaxed.split().core.len(),
            relaxed.split().environment.len() + relaxed.split().fixed.len()
        ));
    }
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document),
    )?;
    describe(config, &log);
    if !o.biases.is_empty() {
        log.info(&format!(
            "Bias potentials: every job has its own copy; conformers are ranked by {}.",
            super::confsearch_files::RANKING
        ));
    }
    if start.has_pbc() {
        log.info(
            "Periodic cell: the molecules are compared made whole, as one cluster; a structure \
             whose bonds run through the cell into a network is refused.",
        );
    }
    if !o.settings.coordinates.eq_ignore_ascii_case("cart") {
        log.info(&format!(
            "Optimizations in the coordinates -coord {}; the dynamics run in Cartesian coordinates.",
            o.settings.coordinates
        ));
    }
    let journal = RunJournal::open_with(&dir, registry.replay())?;
    let journaled = journal.wrap("main", calc.as_ref());
    let stop = SharedStop::new(&dir, log.clone());
    let jobs = CommandJobs::new(
        config,
        job_atoms,
        &journaled,
        &dir,
        log.clone(),
        stop.clone(),
    )?;
    let tabu = config
        .tabu
        .map(|a| HeatedBoltzmann::multioptpy(config.sampling_temperature, Some(a)));
    let mut files = ConfsearchFiles::new(
        &dir,
        &stem_of(&config.input),
        log.clone(),
        tabu,
        config.common.mode == Mode::Conformers,
        config.common.bond_scale,
    )?;
    // The minima of the search of transition-state conformers are constrained minima: their
    // Hessian has the imaginary mode of the reaction, so they are not checked as minima.
    let common = crate::workflow::confsearch::CommonSettings {
        verify: config.common.verify && config.ts.is_none(),
        ..config.common.clone()
    };
    let mut strategy = |s: &MolecularSystem,
                        j: &dyn SearchJobs,
                        obs: &mut dyn SearchObserver|
     -> MoptResult<Option<SearchResult>> {
        search_with(config, &common, moves, &log, s, j, obs)
    };
    let (result, ts_result) = match &config.ts {
        Some(t) => match ts::run(&start, t, &config.common, &jobs, &mut files, &mut strategy)? {
            Some(r) => (Some(r.constrained.clone()), Some(r)),
            None => (None, None),
        },
        None => (strategy(&start, &jobs, &mut files)?, None),
    };
    let interrupted = stop.stopped() || result.as_ref().is_none_or(|r| r.interrupted);
    let mut out = ConfsearchResult {
        conformers: dir.join("conformers.xyz"),
        directory: dir.clone(),
        count: 0,
        energies: Vec::new(),
        reason: String::new(),
        interrupted,
    };
    if let Some(r) = &result {
        out.reason.clone_from(&r.reason);
        if !interrupted {
            log.info(&format!("Reason of Termination: {}", r.reason));
        }
        let imaginary = if config.frequencies && !interrupted {
            frequencies(config, &jobs, r, &dir, &log)?
        } else {
            Vec::new()
        };
        write_summary(&dir, r, config.temperature, &imaginary, &log)?;
        log.info(&format!(
            "Candidates: {} ({} new, {} duplicates, {} rotamers, {} rejected for their bonds, {} \
             outside the window, {} failed, {} off the constraints)",
            r.tally.candidates,
            r.tally.new,
            r.tally.duplicates,
            r.tally.rotamers,
            r.tally.rejected,
            r.tally.outside,
            r.tally.failed,
            r.tally.off_constraints
        ));
        let order = r.ensemble.by_energy();
        out.count = order.len();
        out.energies = order
            .iter()
            .map(|&i| r.ensemble.members()[i].energy)
            .collect();
    }
    if let Some(t) = &ts_result {
        write_transition_states(t, &dir, &log)?;
        let order = t.ensemble.by_energy();
        out.count = order.len();
        out.energies = order
            .iter()
            .map(|&i| t.ensemble.members()[i].energy)
            .collect();
        out.conformers = dir.join("ts_conformers.xyz");
    }
    if interrupted {
        log.info(&format!(
            "Continue with: confsearch --restart {}",
            dir.display()
        ));
    }
    if let Some(relaxed) = &micro {
        // The files of the run with the whole system.
        super::oniom_micro::finish(relaxed, None, &dir, &log)?;
    }
    let (replayed, unused) = journal.replay_counts();
    if replayed > 0 {
        log.info(&format!(
            "Restart: {replayed} evaluations taken from {JOURNAL_FILE}{}.",
            if unused > 0 {
                format!(" ({unused} recorded ones no longer matched and were recomputed)")
            } else {
                String::new()
            }
        ));
    }
    log.info("Conformation search is finished.");
    log.info(
        &registry
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(out)
}

/// `-freq`: the vibrational analysis of every conformer, in `vibrations/EQ<k>/`; returns the
/// number of imaginary frequencies of each.
fn frequencies(
    config: &ConfsearchConfig,
    jobs: &CommandJobs<'_>,
    result: &SearchResult,
    dir: &Path,
    log: &Logger,
) -> MoptResult<Vec<Option<usize>>> {
    let mut out = Vec::new();
    for (k, m) in result.ensemble.members().iter().enumerate() {
        let sub = dir.join("vibrations").join(format!("EQ{k}"));
        std::fs::create_dir_all(&sub)?;
        let quiet = Logger::silent();
        quiet.add_file(sub.join("mopt.log"), Level::Debug)?;
        // As `optmain -freq`: the rigid motions the frame of the surface has, and with
        // constraints the modes on their surface.
        let (problem, project) = jobs.frequency_problem(&m.system)?;
        let mut settings = config.optmain.frequency;
        settings.project_rigid = project;
        if problem.constrained {
            super::freqmain::announce_constraint_surface(&problem.normals, project, &quiet);
        }
        let (_, wavenumbers) = super::freqmain::write_reports(
            &m.system,
            m.energy,
            &problem.hessian,
            &settings,
            &problem.normals,
            &sub,
            &quiet,
        )?;
        if problem.held > 0 {
            super::freqmain::note_constraint_surface(&sub, problem.held)?;
        }
        quiet.flush();
        let n = crate::vib::imaginary_wavenumbers(&wavenumbers);
        log.info(&format!(
            "Vibrational analysis of EQ{k}: {n} imaginary frequencies"
        ));
        out.push(Some(n));
    }
    Ok(out)
}

/// `ts_conformers.xyz` and `ts_conformers.csv` of the transition states.
fn write_transition_states(t: &TsResult, dir: &Path, log: &Logger) -> MoptResult<()> {
    let e = &t.ensemble;
    let order = e.by_energy();
    let Some(&low) = order.first() else {
        return Ok(());
    };
    let e0 = e.members()[low].energy;
    let input = e.members()[0].energy;
    let mut xyz = String::new();
    let mut csv = String::from(
        "rank,ts,energy_hartree,relative_kcal_mol,relative_to_input_kcal_mol,imaginary_cm-1,\
         same_reaction,end_1_hartree,end_2_hartree,origin\n",
    );
    for (rank, &i) in order.iter().enumerate() {
        let m = &e.members()[i];
        let detail = t.candidates.iter().find(|c| c.member == Some(i));
        let mut s = m.system.clone();
        s.comment = format!("TS{i} energy={:.12}", m.energy);
        xyz.push_str(&format_xyz_system(&s, None));
        let _ = writeln!(
            csv,
            "{},{i},{:.10},{:.4},{:.4},{},{},{},{},\"{}\"",
            rank + 1,
            m.energy,
            (m.energy - e0) * Units::HARTREE_TO_KCAL_MOL,
            (m.energy - input) * Units::HARTREE_TO_KCAL_MOL,
            detail
                .and_then(|d| d.wavenumber)
                .map_or(String::new(), |w| format!("{w:.1}")),
            detail
                .and_then(|d| d.same_reaction)
                .map_or(String::new(), |b| b.to_string()),
            detail
                .and_then(|d| d.ends)
                .map_or(String::new(), |x| format!("{:.10}", x[0])),
            detail
                .and_then(|d| d.ends)
                .map_or(String::new(), |x| format!("{:.10}", x[1])),
            m.origin.replace('"', "'")
        );
    }
    std::fs::write(dir.join("ts_conformers.xyz"), xyz)?;
    std::fs::write(dir.join("ts_conformers.csv"), csv)?;
    log.info(&format!(
        "Transition-state conformers: {} (the lowest, TS{low}, {:.4} kcal/mol relative to the \
         input's transition state); {} saddle-point optimizations",
        order.len(),
        (e0 - input) * Units::HARTREE_TO_KCAL_MOL,
        t.candidates.len()
    ));
    log.info(&format!(
        "Transition-state conformer file path: {}",
        dir.join("ts_conformers.xyz").display()
    ));
    Ok(())
}
