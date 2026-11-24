//! `run_mapper`: the reaction network of one or more input structures. Every input is
//! optimized to a minimum; then job after job starts from a known minimum (the choice of the
//! priority, Boltzmann by default) with one of its tasks (an AFIR push between two atoms), finds
//! the transition states on the path of the push, runs their IRCs, and adds what is new to the
//! network, until no minimum has a task left, `--max_iter` jobs have run, or a stop rule ends the
//! search.
//!
//! Everything of a run is in its run directory (`--output_dir`, `mapper_output` by default):
//! `settings.json` and `config_snapshot.json` (the options and the settings file, which
//! `--restart RUN_DIR` reads), `mopt.log`, `evaluations.journal`, the network
//! (`reaction_network.json`, `nodes.csv`, `ts_edges.csv`, `crossing_edges.csv`,
//! `structures/`), the IRC paths (`paths/`), one trajectory and one log per job (`jobs/`), and
//! `mapper_report.html`. The network is written after every batch of jobs that is kept (every
//! job when one runs at a time) and the report after every job; a `STOP` file stops the run
//! within a job, and `--restart RUN_DIR` goes on without running the finished jobs again (the
//! evaluations of the jobs that were stopped come back from the journal). Several jobs run at
//! the same time as the schedule of the settings says (`--max_workers`, `-map_opt workers=N`,
//! [`crate::workflow::mapper::schedule`]).
//!
//! The parts are those of [`crate::workflow::mapper`]; [`run_with_parts`] takes registries
//! with further methods, priorities, stop rules, and stores, and further observers.
//!
//! MultiOptPy: `Entrypoints/mapper.py:run_mapper`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use super::mapper_jobs::{CommandJobs, CommandWorkers, MapperStop, JOURNAL_LABEL};
use super::mapper_report::MapperReport;
pub use super::mapper_settings::{MapperConfig, MapperSettings, CONFIG_SNAPSHOT};
use super::settings::RUN_SETTINGS_FILE;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use std::sync::Arc;

use faer::Col;

use super::variables::read_variables_file;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::read_xyz_trajectory_file;
use crate::pes::objective::{check_dimension, Objective, ObjectiveRegistry};
use crate::pes::registry::CalculatorRegistry;
use crate::workflow::journal::{RunJournal, JOURNAL_FILE};
use crate::workflow::mapper::store::{forget_beyond, read_network, write_whole, NETWORK_FILE};
use crate::workflow::mapper::{
    explore_with, mapper_matcher, Configuration, DriverSettings, ExplorationEnd,
    ExplorationPriorityRegistry, FileStore, InputRecord, MapperDriver, MapperObserver,
    MethodRegistry, MethodSettings, Network, NetworkStore, NodeKind, PathFiles, PrioritySettings,
    StopRuleRegistry, StoreObserver, StoreRegistry, TrajectoryFiles,
};

/// What a run left behind.
#[derive(Debug, Clone)]
pub struct MapperRun {
    /// The run directory.
    pub directory: PathBuf,
    /// The network.
    pub network: Network,
    /// Whether it was stopped (`--restart RUN_DIR` continues it).
    pub interrupted: bool,
    /// Why it ended, when it did.
    pub reason: Option<String>,
}

/// The registries a run takes its parts from.
#[derive(Default)]
pub struct MapperParts {
    /// Ways of exploring a minimum.
    pub methods: MethodRegistry,
    /// Choices of the next minimum.
    pub priorities: ExplorationPriorityRegistry,
    /// Stop rules.
    pub stops: StopRuleRegistry,
    /// Stores of the network.
    pub stores: StoreRegistry,
}

/// An input structure: its file, its frame, and the structure (or the point of plain
/// variables).
struct Input {
    file: String,
    frame: usize,
    system: Configuration,
    /// The goal of a goal-directed search (`goal=FILE`).
    goal: bool,
}

/// The inputs made for the entrance channel of the first of `inputs`
/// (`entrance_structures=N`): N structures of its molecules, turned and placed in contact at
/// random with the seed of the run ([`crate::workflow::mapper::entrance`]). None without the
/// setting.
fn entrance_inputs(config: &MapperConfig, inputs: &[Input]) -> MoptResult<Vec<Input>> {
    let s = &config.settings;
    if s.entrance_structures == 0 {
        return Ok(Vec::new());
    }
    let Some(first) = inputs.iter().find(|i| !i.goal) else {
        return Ok(Vec::new());
    };
    let system = first.system.require_structure("entrance_structures")?;
    let made = crate::workflow::mapper::entrance_structures(
        system,
        &crate::workflow::mapper::EntranceSettings {
            structures: s.entrance_structures,
            contact: s.entrance_contact,
            bond_scale: crate::workflow::explore::identity::DEFAULT_BOND_SCALE,
            seed: s.seed,
        },
    )
    .context(first.file.clone())?;
    Ok(made
        .into_iter()
        .enumerate()
        .map(|(k, system)| Input {
            file: format!("{} (entrance structure {})", first.file, k + 1),
            frame: first.frame,
            system: Configuration::Structure(system),
            goal: false,
        })
        .collect())
}

/// The inputs of a run on the objective `objective`: one variables file each, every one with
/// the same variables.
fn read_variable_inputs(
    config: &MapperConfig,
    objective: &dyn Objective,
) -> MoptResult<Vec<Input>> {
    let mut out: Vec<Input> = Vec::new();
    let mut names: Option<Arc<Vec<String>>> = None;
    let goal = config.settings.goal.as_ref();
    for (p, path) in config.inputs.iter().chain(goal).enumerate() {
        let vars = read_variables_file(path).context(path.display())?;
        let n = vars.values.len();
        check_dimension(objective, n).context(path.display())?;
        let these = vars.names.unwrap_or_else(|| objective.variable_names(n));
        let shared = match &names {
            Some(known) if **known == these => Arc::clone(known),
            Some(known) => {
                return Err(MoptError::parse(format!(
                    "run_mapper: {} has the variables {} and the first input {}; the inputs of \
                     one network need the same variables",
                    path.display(),
                    these.join(", "),
                    known.join(", ")
                )))
            }
            None => {
                let fresh = Arc::new(these);
                names = Some(Arc::clone(&fresh));
                fresh
            }
        };
        out.push(Input {
            file: path.display().to_string(),
            frame: 0,
            system: Configuration::variables(Col::from_fn(n, |i| vars.values[i]), shared)?,
            goal: p == config.inputs.len(),
        });
    }
    if out.is_empty() {
        return Err(MoptError::parse("run_mapper: no input variables file"));
    }
    Ok(out)
}

fn read_inputs(config: &MapperConfig) -> MoptResult<Vec<Input>> {
    let mut out: Vec<Input> = Vec::new();
    let goal = config.settings.goal.as_ref();
    for (p, path) in config.inputs.iter().chain(goal).enumerate() {
        // The goal is the first frame of its file, after the inputs.
        let is_goal = p == config.inputs.len();
        let state = config.steps.biased.state.of_file(path)?;
        let frames = read_xyz_trajectory_file(path, state, None).context(path.display())?;
        let count = if is_goal { 1 } else { frames.frames.len() };
        for (k, frame) in frames.frames.into_iter().enumerate().take(count) {
            let system = frame.system;
            if let Some((first, earlier)) = out
                .first()
                .and_then(|i| i.system.structure().map(|s| (i, s)))
            {
                if system.elements != earlier.elements {
                    return Err(MoptError::parse(format!(
                        "run_mapper: {} (frame {}) has other atoms or another order of the \
                         atoms than {} (frame {}); the inputs of one network need the same atoms \
                         in the same order",
                        path.display(),
                        k + 1,
                        first.file,
                        first.frame + 1
                    )));
                }
                if system.spin_charge != earlier.spin_charge {
                    return Err(MoptError::unsupported(format!(
                        "run_mapper: {} (frame {}) has another charge or multiplicity than {}; \
                         a network of several electronic states needs the search for crossing \
                         points, which is not implemented yet",
                        path.display(),
                        k + 1,
                        first.file
                    )));
                }
            }
            out.push(Input {
                file: path.display().to_string(),
                frame: k,
                system: Configuration::Structure(system),
                goal: is_goal,
            });
        }
    }
    if out.is_empty() {
        return Err(MoptError::parse("run_mapper: the inputs have no structure"));
    }
    Ok(out)
}

/// The key of the entry `run` of the network file with the number of records of the journal
/// that the totals of evaluations of the network count.
pub const JOURNAL_RECORDS: &str = "journal_records";

/// The evaluations recorded in the journal of `dir` after its first `counted` records: those a
/// session made after it last wrote the network, when it was killed before it could write it
/// again. They come back from the journal and are not computed again, so they are counted
/// here, once.
fn uncounted_evaluations(
    dir: &Path,
    counted: usize,
) -> MoptResult<crate::workflow::mapper::network::EvaluationTotals> {
    let (records, _) = crate::workflow::journal::read_journal(&dir.join(JOURNAL_FILE))?;
    let mut totals = crate::workflow::mapper::network::EvaluationTotals::default();
    for r in records.iter().skip(counted) {
        if r.gradient.is_empty() {
            totals.energies += 1;
        } else {
            totals.gradients += 1;
        }
        if r.has_hessian {
            totals.hessians += 1;
        }
    }
    Ok(totals)
}

/// Run the mapper under `base` with the calculators of `registry`.
pub fn run_with(
    config: &MapperConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
) -> MoptResult<MapperRun> {
    run_with_parts(
        config,
        base,
        console,
        registry,
        &MapperParts::default(),
        &mut [],
    )
}

/// [`run_with`] with the parts of `parts` and the further observers `extra`.
pub fn run_with_parts(
    config: &MapperConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    parts: &MapperParts,
    extra: &mut [&mut dyn MapperObserver],
) -> MoptResult<MapperRun> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(
            config
                .steps
                .biased
                .calculator_options
                .pool_threads(config.steps.biased.threads),
        )
        .build()
        .map_err(|e| MoptError::config(format!("thread pool: {e}")))?;
    pool.install(|| run_inner(config, base, console, registry, parts, extra))
}

fn run_inner(
    config: &MapperConfig,
    base: &Path,
    console: Option<Level>,
    registry: &CalculatorRegistry,
    parts: &MapperParts,
    extra: &mut [&mut dyn MapperObserver],
) -> MoptResult<MapperRun> {
    // The networks of several runs as one: nothing is calculated.
    if !config.merge.is_empty() {
        return super::mapper_merge::run(config, base, console, parts);
    }
    // An objective of plain variables in the place of a calculator: its points are the nodes.
    let objectives = ObjectiveRegistry::builtin();
    let objective = if objectives.contains(&config.steps.biased.calculator) {
        Some(objectives.create(
            &config.steps.biased.calculator,
            &config.steps.biased.calculator_options,
        )?)
    } else {
        None
    };
    let inputs = match (&config.single_points, &config.remap, &objective) {
        // The inputs of the first network of a run of single points, as its network has them
        // (the network of the run is that one with other energies).
        (Some(first), _, _) => first
            .recorded_inputs()?
            .into_iter()
            .map(|(file, frame, system, goal)| Input {
                file,
                frame,
                system,
                goal,
            })
            .collect(),
        // The minima of the first network of a remap.
        (None, Some(first), _) => first
            .inputs()?
            .into_iter()
            .map(|(file, system, _)| Input {
                file,
                frame: 0,
                system,
                goal: false,
            })
            .collect(),
        (None, None, Some(o)) => read_variable_inputs(config, o.as_ref())?,
        (None, None, None) => {
            let mut inputs = read_inputs(config)?;
            // The structures of the entrance channel of the first input, after the inputs
            // that were given (and before the goal, which stays the last).
            let made = entrance_inputs(config, &inputs)?;
            let at = inputs.iter().position(|i| i.goal).unwrap_or(inputs.len());
            inputs.splice(at..at, made);
            inputs
        }
    };
    let s = &config.settings;
    // The items of a remap, and the state of every input in the first network.
    let remap = match &config.remap {
        Some(first) => {
            if first.network.states.len() != s.states.len() {
                return Err(MoptError::config(format!(
                    "run_mapper: -remap {}: its network has {} state(s) and this run {} (states): \
                     a network of several states is drawn again on as many states, in the same \
                     order",
                    first.given.display(),
                    first.network.states.len(),
                    s.states.len()
                )));
            }
            if first.network.variables.is_some() != objective.is_some() {
                return Err(MoptError::config(format!(
                    "run_mapper: -remap {}: a network of plain variables is drawn again on an \
                     objective of plain variables, and a network of structures on a calculator of \
                     structures",
                    first.given.display()
                )));
            }
            let (source, notes) = first.source()?;
            let states: Vec<Option<usize>> =
                first.inputs()?.into_iter().map(|(_, _, on)| on).collect();
            Some((first, source, notes, states))
        }
        None => None,
    };
    if objective.is_some() && !s.states.is_empty() {
        return Err(MoptError::unsupported(
            "run_mapper: states: an objective of plain variables has one surface; the states of \
             a run are those of a calculator of structures",
        ));
    }
    // The parts first: what is unknown or not implemented yet is refused before anything runs.
    if let Some(first) = &config.single_points {
        if first.network.states.len() != s.states.len() {
            return Err(MoptError::config(format!(
                "run_mapper: -single_points {}: its network has {} state(s) and this run {} \
                 (states): the energies of a network of several states are those of as many \
                 states, in the same order",
                first.given.display(),
                first.network.states.len(),
                s.states.len()
            )));
        }
        if first.network.variables.is_some() != objective.is_some() {
            return Err(MoptError::config(format!(
                "run_mapper: -single_points {}: a network of plain variables has its energies \
                 on an objective of plain variables, and a network of structures on a \
                 calculator of structures",
                first.given.display()
            )));
        }
    }
    let method = match &remap {
        _ if config.single_points.is_some() => Box::new(super::mapper_points::NoTasks)
            as Box<dyn crate::workflow::mapper::ExplorationMethod>,
        Some((_, source, _, _)) => Box::new(crate::workflow::mapper::Remap::new(
            source.clone(),
            crate::workflow::mapper::RemapSettings::parse(&s.method_extra)?,
            s.states.clone(),
        )?)
            as Box<dyn crate::workflow::mapper::ExplorationMethod>,
        None => {
            let settings = MethodSettings {
                afir: s.afir.clone(),
                top_n: s.top_n,
                extra: s.method_extra.clone(),
                step2: config
                    .document
                    .get("step2_settings")
                    .map(JsonValue::to_string_compact),
            };
            parts.methods.create(&s.method, &settings)?
        }
    };
    // The conformer searches of the transition states before the tasks of the method.
    let method = if s.ts_conformers && remap.is_none() && config.single_points.is_none() {
        Box::new(crate::workflow::mapper::WithTsConformers::new(method))
            as Box<dyn crate::workflow::mapper::ExplorationMethod>
    } else {
        method
    };
    // The upward searches of saddle points of order 2 beside the tasks of the method.
    let method = match s.upward {
        Some(up) if config.single_points.is_none() => {
            Box::new(crate::workflow::mapper::WithUpward::new(method, up))
                as Box<dyn crate::workflow::mapper::ExplorationMethod>
        }
        _ => method,
    };
    // The crossing points between the states of the run (`states`) before the tasks of the
    // method on each surface.
    let method = if s.crossings && remap.is_none() && config.single_points.is_none() {
        // The AFIR pushes of afir_ts towards the crossing points (crossing_afir). (A remap
        // has the crossing points of its first network as its items and runs no search of
        // its own for others.)
        let crossing = crate::workflow::mapper::WithCrossings::new(
            method,
            s.states.clone(),
            s.crossing_pairs,
        )?
        .with_inputs_only(s.crossing_inputs_only)
        .with_direct(s.crossing_direct)
        .with_ts_states(s.ts_states.clone())
        .with_lower_avoiding(!s.crossing_models.amf_upper)
        .with_candidates(s.crossing_models.candidates);
        let crossing = match &s.crossing_pushes {
            Some(pushes) => crossing.with_pushes(pushes.clone(), s.crossing_afir),
            None => crossing.with_afir(
                crate::workflow::mapper::AfirTs::new(s.afir.clone(), s.top_n)?,
                s.crossing_afir,
            ),
        };
        Box::new(crossing) as Box<dyn crate::workflow::mapper::ExplorationMethod>
    } else {
        method
    };
    // The `kinetics_*` settings go to a priority that takes settings (priority=kinetics) and
    // to the kinetics at the end (kinetics=on); settings that neither uses are refused.
    let takes = parts.priorities.takes_options(&s.priority).unwrap_or(false);
    let rrt = s
        .priority
        .eq_ignore_ascii_case(crate::workflow::mapper::RrtPriority::NAME);
    if !takes && !s.kinetics {
        if let Some(key) = s.priority_options.keys().next() {
            return Err(MoptError::parse(format!(
                "run_mapper: kinetics_{key} is a setting of priority=kinetics and of the \
                 kinetics at the end of the run (kinetics=on); neither is chosen"
            )));
        }
    }
    let no_options = std::collections::BTreeMap::new();
    let priority = parts.priorities.create_with(
        &s.priority,
        &PrioritySettings {
            temperature: s.temperature,
        },
        if rrt {
            &s.rrt_options
        } else if takes {
            &s.priority_options
        } else {
            &no_options
        },
    )?;
    // The kinetics at the end, its settings checked before anything runs.
    let mut end_kinetics = if s.kinetics {
        Some(super::mapper_kinetics::EndKinetics::new(
            &s.priority_options,
            s.temperature,
        )?)
    } else {
        None
    };
    let mut rules = Vec::new();
    if s.max_jobs > 0 {
        rules.push(parts.stops.create(&format!("max_jobs:{}", s.max_jobs))?);
    }
    for spec in &s.stop {
        rules.push(parts.stops.create(spec)?);
    }
    let mut stores: Vec<Box<dyn NetworkStore>> = vec![Box::new(FileStore)];
    for name in &s.stores {
        if !name.eq_ignore_ascii_case("files") {
            stores.push(parts.stores.create(name)?);
        }
    }
    // The constraints of the run resolved once at its first input, and every other input put
    // onto the same values (`docs/source/mapper_design.md`).
    let resolved_steps;
    let mut put_back: Vec<String> = Vec::new();
    let (steps, inputs) = match inputs[0].system.structure().cloned() {
        Some(first) if !config.steps.constraints().is_empty() => {
            resolved_steps = config.steps.clone().resolved_at(&first)?;
            let set = crate::pes::constraint::ConstraintSet::new(
                resolved_steps.constraints(),
                &first,
                None,
            )?;
            let inputs = inputs
                .into_iter()
                .enumerate()
                .map(|(k, mut input)| {
                    if let (Some(sys), true) = (input.system.structure(), k > 0) {
                        let on = set.restore(sys).map_err(|e| {
                            MoptError::config(format!(
                                "run_mapper: the input {} (frame {}) cannot be put onto the \
                                 constraints of the run, resolved at the first input: {e}",
                                input.file,
                                input.frame + 1
                            ))
                        })?;
                        let moved = crate::math::linalg::col_norm(
                            &(on.coords_flat_bohr() - sys.coords_flat_bohr()),
                        );
                        if moved > 1e-6 {
                            put_back.push(format!(
                                "The input {} (frame {}) was put onto the constraints at the \
                                 values of the first input (moved {moved:.3e} Bohr).",
                                input.file,
                                input.frame + 1
                            ));
                        }
                        input.system = Configuration::Structure(on);
                    }
                    Ok(input)
                })
                .collect::<MoptResult<Vec<_>>>()?;
            (&resolved_steps, inputs)
        }
        _ => (&config.steps, inputs),
    };
    // The static bias potentials of the run, part of its surface in every stage.
    let bias_terms: Vec<(String, std::sync::Arc<dyn crate::pes::term::PesTerm>)> =
        match inputs[0].system.structure() {
            Some(first) => steps
                .biased
                .biases
                .iter()
                .enumerate()
                .map(|(k, spec)| Ok((spec.label(k), spec.build(first)?)))
                .collect::<MoptResult<_>>()?,
            None => Vec::new(),
        };
    let schedule = s.schedule;
    // ONIOM: the network holds the cores, and every calculation is on the relaxed surface of
    // the core (`cli::mapper_oniom`). The whole structure of the first input is what the
    // calculators are made from.
    let structure = inputs[0].system.structure().cloned();
    let criteria = super::oniom_micro::forces_of(&steps.minimum.settings.criteria);
    let oniom = objective.is_none() && steps.biased.calculator_options.oniom.is_some();
    if s.identity_scope.is_some() && !oniom {
        return Err(MoptError::config(
            "run_mapper: identity_scope is a setting of a network on an ONIOM surface (-oniom in \
             the steps)",
        ));
    }
    if s.afir.cell_pushes && steps.minimum.variable_cell.is_none() {
        return Err(MoptError::config(
            "run_mapper: cell_pushes pushes the cell itself, which moves only on a variable \
             cell (variable_cell in step1_settings)",
        ));
    }
    // -fix of an ONIOM run: atoms of the environment held during the relaxations.
    let held = &steps.held;
    let (added, run_wrapped, whole_system) = match (&structure, oniom) {
        (Some(whole), true) => {
            super::mapper_oniom::refuse_with_oniom(config)?;
            let wrap = |others: &[usize]| {
                super::oniom_micro::wrap_constrained(
                    &steps.biased.calculator_options,
                    super::oniom_micro::uncovered_by_optmain(&steps.biased),
                    held,
                    &[],
                    others,
                    criteria,
                    whole.clone(),
                    &steps.biased.calculator,
                    registry,
                )
            };
            let plain = wrap(&[])?;
            if !plain.constraints.is_empty() {
                return Err(MoptError::unsupported(
                    "run_mapper: -fix holds atoms of the core (the model region and the outer \
                     atoms of its links), which the mapper moves freely; fix atoms of the \
                     environment only",
                ));
            }
            let core: Vec<usize> = match &plain.micro {
                Some(m) => m.split().core.clone(),
                None => {
                    return Err(MoptError::unsupported(
                        "run_mapper: ONIOM with -oniom_opt micro=off: the mapper runs every \
                         calculation on the relaxed surface of the core (microiterations), as \
                         every ONIOM run but the dynamics",
                    ))
                }
            };
            let added = s
                .identity_scope
                .unwrap_or(super::mapper_oniom::IdentityScope::High)
                .added_atoms(whole, &core);
            let wrapped = if added.is_empty() {
                plain
            } else {
                wrap(&added)?
            };
            let micro = wrapped.micro.clone().ok_or_else(|| {
                MoptError::calc("run_mapper: the ONIOM surface has no microiterations")
            })?;
            let mut system = super::mapper_oniom::WholeSystem::of(&micro, whole.clone());
            system.trajectory_atoms = wrapped.trajectory_atoms(&steps.biased.calculator_options)?;
            (added, Some(wrapped), Some(std::sync::Arc::new(system)))
        }
        _ => (Vec::new(), None, None),
    };
    // The inputs as the run moves them: their cores on an ONIOM surface (their environments
    // kept as the starts of the relaxations near them).
    let trajectory_atoms = whole_system
        .as_ref()
        .and_then(|s| s.trajectory_atoms.clone());
    let inputs = match &run_wrapped {
        Some(w) => inputs
            .into_iter()
            .map(|mut input| {
                if let Some(sys) = input.system.structure() {
                    input.system = Configuration::Structure(w.structure(sys.clone())?);
                }
                Ok(input)
            })
            .collect::<MoptResult<Vec<_>>>()?,
        None => inputs,
    };
    let first = &inputs[0].system;
    // The calculator of the surface with `options`; with several jobs at the same time a
    // calculator of the Python side gets a copy per job in worker processes (-parallel),
    // unless -parallel says otherwise.
    let make = |options: &crate::pes::registry::CalculatorOptions| {
        let first = structure.clone().ok_or_else(|| {
            MoptError::config("a calculator of structures for a network of plain variables")
        })?;
        let wrapped = super::oniom_micro::wrap_constrained(
            options,
            super::oniom_micro::uncovered_by_optmain(&steps.biased),
            held,
            &[],
            &added,
            criteria,
            first,
            &steps.biased.calculator,
            registry,
        )?;
        let electrons = wrapped.electrons;
        // The static bias potentials on the calculator, so that every stage of every job
        // evaluates the same surface; the AFIR pushes of the mapper stay terms of the push.
        let calc: Box<dyn crate::pes::calculator::Calculator> = if bias_terms.is_empty() {
            wrapped.calc
        } else {
            Box::new(
                crate::pes::biased::BiasedCalculator::new(
                    std::sync::Arc::from(wrapped.calc),
                    bias_terms.clone(),
                )
                .map_err(|e| e.context("run_mapper"))?,
            )
        };
        let calc = crate::pes::o1numhess::wrap_if_requested(
            calc,
            steps.saddle.settings.model_hessian.as_deref(),
            steps.saddle.settings.exact_hessian_interval.is_some(),
        );
        Ok::<_, MoptError>((calc, electrons))
    };
    let mut shared_options = steps.biased.calculator_options.clone();
    if schedule.workers > 1 && shared_options.parallel == 0 {
        shared_options.parallel = schedule.workers;
    }
    let calc = match (&objective, &structure, run_wrapped) {
        // The relaxed surface that took the environments of the inputs.
        (None, Some(_), Some(wrapped)) => Some(crate::pes::o1numhess::wrap_if_requested(
            wrapped.calc,
            steps.saddle.settings.model_hessian.as_deref(),
            steps.saddle.settings.exact_hessian_interval.is_some(),
        )),
        (None, Some(first), None) => {
            let (calc, electrons) = make(&shared_options)?;
            if electrons {
                first.spin_charge.check_electrons(&first.elements)?;
            }
            Some(calc)
        }
        _ => None,
    };
    let own_calculator = || make(&steps.biased.calculator_options).map(|(c, _)| c);

    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => {
            let dir = base.join(&config.output_dir);
            let used = dir
                .read_dir()
                .map(|mut entries| entries.next().is_some())
                .unwrap_or(false);
            if used {
                return Err(MoptError::config(format!(
                    "run_mapper: the run directory {} exists; continue its run with --restart {} \
                     or give another --output_dir",
                    dir.display(),
                    dir.display()
                )));
            }
            dir
        }
    };
    std::fs::create_dir_all(&dir).map_err(|e| MoptError::from(e).context(dir.display()))?;
    if config.restart_dir.is_none() {
        // The settings first, each file whole or not at all, and settings.json (which
        // --restart reads first) last: a run directory with settings.json can be restarted.
        write_whole(
            &dir.join(CONFIG_SNAPSHOT),
            (config.document.to_string_pretty() + "\n").as_bytes(),
        )?;
        write_whole(
            &dir.join(RUN_SETTINGS_FILE),
            format!("{}\n", config.settings_document).as_bytes(),
        )?;
        // The structure the built-in GFN-FF of the run sets its topology up from, for a
        // command that starts from a result of the run (-calc_opt topology=FILE).
        let given = steps
            .biased
            .calculator_options
            .extra
            .iter()
            .any(|(k, _)| k == "topology");
        if let (Some(first), true) = (
            &structure,
            steps.biased.calculator.eq_ignore_ascii_case("gfnff") && !given,
        ) {
            crate::io::xyz::write_xyz_file(
                dir.join(super::autots::GFNFF_TOPOLOGY_FILE),
                first,
                None,
            )?;
        }
    }
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    // The screen alone (a short line when a job starts and ends) and the log of the run alone
    // (the blocks of the jobs); every line of them is written from this thread.
    let output = super::mapper_jobs::RunOutput {
        both: log.clone(),
        screen: console.map_or_else(Logger::silent, Logger::console),
        file: Some(super::mapper_jobs::BlockLog::new(&dir.join("mopt.log"))),
    };
    log.info(&format!(
        "mopt-rs {} run_mapper: {} input structure(s) from {}",
        crate::VERSION,
        inputs.len(),
        match config.remap.as_ref().or(config.single_points.as_ref()) {
            Some(first) => format!("the minima of the network of {}", first.given.display()),
            None => config
                .inputs
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        }
    ));
    if let Some((first, source, notes, _)) = &remap {
        use crate::workflow::mapper::RemapKind;
        let count =
            |f: &dyn Fn(&RemapKind) -> bool| source.items.iter().filter(|i| f(&i.kind)).count();
        log.info(&format!(
            "Remap: the network of {} is drawn again on the surface of this run: {} minima (the \
             inputs), {} transition state(s), {} crossing point(s), {} saddle point(s) of higher \
             order; one job each, and the first run is only read.",
            first.given.display(),
            first.minima.len(),
            count(&|k| matches!(k, RemapKind::Transition)),
            count(&|k| matches!(k, RemapKind::Crossing(_))),
            count(&|k| matches!(k, RemapKind::Saddle(_))),
        ));
        for note in notes {
            log.warn(&format!("Remap: {note}"));
        }
        if first.highest_order().is_some() && s.higher_saddles.is_none() {
            log.warn(
                "Remap: the saddle points of higher order of the first network are drawn again \
                 with higher_saddles=on only; without it they are left out.",
            );
        }
    }
    log.info(&format!("Run directory: {}", dir.display()));
    match (&calc, &objective) {
        (Some(calc), _) => {
            log.info(&format!("Calculator: {}", calc.description()));
            for paper in calc.citations() {
                log.info(&format!("Reference: {paper}"));
            }
        }
        (None, Some(o)) => {
            if let Some(p) = first.point() {
                log.info(&format!(
                    "Objective: {} ({} plain variables: {})",
                    o.description(),
                    p.names.len(),
                    p.names.join(", ")
                ));
            }
        }
        (None, None) => {}
    }
    for note in &config.notes {
        log.info(note);
    }
    if !steps.constraints().is_empty() {
        log.info(&format!(
            "Constraints: {} held in every stage of every job at the values of the first input \
             ({}).",
            steps.constraints().len(),
            super::mapper_report_data::describe_constraints(steps.constraints())
        ));
    }
    for line in &put_back {
        log.warn(line);
    }
    if !bias_terms.is_empty() {
        log.info(&format!(
            "Bias potentials: {} on the surface of every stage ({}); the energies of the nodes \
             are those with them, and the energy of the calculator alone is kept beside \
             (bare_energy_hartree).",
            bias_terms.len(),
            bias_terms
                .iter()
                .map(|(label, t)| format!("{label}: {}", t.name()))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    log.info(&format!(
        "Method: {}; priority: {} at {} K; duplicates: {} and energies within {} kcal/mol{}; \
         jobs: {}; seed {}",
        method.name(),
        priority.name(),
        s.temperature,
        if objective.is_some() {
            format!(
                "a distance below {} (in the units of the variables)",
                s.variable_tolerance
            )
        } else if whole_system.is_some() {
            format!(
                "the cores of the ONIOM surface, RMSD below {} Angstrom (their energies do not \
                 decide, since the environment moves them; other environments are listed on the \
                 node)",
                s.rmsd_threshold
            )
        } else {
            format!("RMSD below {} Angstrom", s.rmsd_threshold)
        },
        if whole_system.is_some() {
            "any".to_string()
        } else {
            s.energy_tolerance.to_string()
        },
        if s.mirror {
            ", mirror images the same"
        } else {
            ", mirror images distinct"
        },
        if s.max_jobs == 0 {
            "no limit".to_string()
        } else {
            format!("at most {}", s.max_jobs)
        },
        s.seed
    ));
    if let Some(goal) = &s.goal {
        log.info(&format!(
            "Goal: {} (its first frame, optimized like the inputs and never explored); {}{}",
            goal.display(),
            if s.goal_stop {
                "the exploration ends when it is joined to a start"
            } else {
                "the exploration goes on after it is joined to a start (goal_stop=off)"
            },
            match s.accept_temperature {
                Some(t) => format!(
                    "; new minima are explored with the probability min(1, exp(-(E_PT - E_X) / \
                     kT)) at {t} K"
                ),
                None => String::new(),
            }
        ));
    } else if let Some(t) = s.accept_temperature {
        log.info(&format!(
            "New minima are explored with the probability min(1, exp(-(E_PT - E_X) / kT)) at \
             {t} K (accept_temperature)."
        ));
    }
    if method.name().split('+').next() == Some(crate::workflow::mapper::AfirTs::NAME) {
        log.info(&format!(
            "AFIR: {} kJ/mol{} between atoms {} to {} Angstrom apart, at most {} pair(s) per \
         minimum, {} transition-state guess(es) per job",
            s.afir.gamma_kj_mol,
            if s.afir.negative_gamma {
                " (and pulls)"
            } else {
                ""
            },
            s.afir.dist_lower,
            s.afir.dist_upper,
            s.afir.max_pairs,
            s.top_n
        ));
    }
    // The records of the journal the network file had counted when it was written: those
    // after them were made by a session that was killed before it wrote the network again.
    let mut counted_records: Option<usize> = Some(0);
    let mut network = match read_network(&dir)? {
        Some((network, run)) => {
            counted_records = run
                .get(JOURNAL_RECORDS)
                .and_then(JsonValue::as_f64)
                .map(|n| n as usize);
            let same_points = match first {
                Configuration::Structure(sys) => {
                    network.variables.is_none() && network.elements == sys.elements
                }
                Configuration::Variables(p) => network.variables.as_deref() == Some(&p.names[..]),
            };
            if !same_points || network.inputs.len() != inputs.len() {
                return Err(MoptError::config(format!(
                    "run_mapper: {} of {} is not the network of these inputs",
                    NETWORK_FILE,
                    dir.display()
                )));
            }
            if network.states != s.states {
                return Err(MoptError::config(format!(
                    "run_mapper: {} of {} was explored on other states than those of these \
                     settings (states)",
                    NETWORK_FILE,
                    dir.display()
                )));
            }
            // What a session that was killed wrote beyond the network it left (the
            // structures of nodes it had not kept yet) is not the restart's.
            let stale = forget_beyond(&dir, &network)?;
            if stale > 0 {
                log.info(&format!(
                    "Restart: {stale} file(s) of nodes that the network does not have were \
                     removed."
                ));
            }
            log.info(&format!(
                "Restart: {} minima, {} transition states, and {} finished job(s) from {NETWORK_FILE}.",
                network.minima().count(),
                network.of_kind(&NodeKind::TransitionState).count(),
                network.jobs.len()
            ));
            network
        }
        None => {
            let mut network = match first {
                Configuration::Structure(sys) => Network::new(sys.elements.clone()),
                Configuration::Variables(p) => Network::of_variables(p.names.to_vec()),
            };
            network.whole_elements = whole_system.as_ref().map(|w| w.elements());
            network.states.clone_from(&s.states);
            network.inputs = inputs
                .iter()
                .map(|i| InputRecord {
                    file: i.file.clone(),
                    frame: i.frame,
                    node: None,
                    goal: i.goal,
                })
                .collect();
            network
        }
    };
    // The atoms of the constraints keep their places in the identity of the nodes.
    if let Some(first) = inputs[0].system.structure() {
        network.held = steps.held_atoms(first.n_atoms());
    }
    // The surface in words, for the network file and the report.
    network.surface.clear();
    if !steps.constraints().is_empty() {
        network.surface.push(format!(
            "Constraints (held in every stage at the values of the first input): {}",
            super::mapper_report_data::describe_constraints(steps.constraints())
        ));
    }
    if !bias_terms.is_empty() {
        network.surface.push(format!(
            "Static bias potentials (on the surface of every stage; node energies with them, \
             the energy of the calculator alone beside): {}",
            steps
                .biased
                .biases
                .iter()
                .enumerate()
                .map(|(k, spec)| format!("{} ({})", spec.label(k), spec.key))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !bias_terms.is_empty() {
        // Every node keeps the energy of the calculator alone beside that of the surface: the
        // energies of the static terms at its structure taken off (they are what the biased
        // calculator added there).
        let terms = bias_terms.clone();
        network.bare = Some(crate::workflow::mapper::BiasEnergy(std::sync::Arc::new(
            move |point: &Configuration| {
                let sys = point.require_structure("the energy of the bias potentials")?;
                let ctx = crate::pes::term::TermContext::at_iteration(0, sys);
                let mut sum = 0.0;
                for (label, term) in &terms {
                    sum += term
                        .evaluate(sys, &ctx, false)
                        .map_err(|e| e.context(format!("bias '{label}'")))?
                        .energy
                        .value();
                }
                Ok(sum)
            },
        )));
    }
    let journal = RunJournal::open(&dir)?;
    let at_open = journal.replay_counts().1;
    if let Some(counted) = counted_records {
        let uncounted = uncounted_evaluations(&dir, counted)?;
        if uncounted.gradients + uncounted.energies > 0 {
            log.info(&format!(
                "Restart: {} evaluation(s) of a session that ended without writing the network \
                 are counted from {JOURNAL_FILE}.",
                uncounted.gradients + uncounted.energies
            ));
        }
        let e = &mut network.evaluations;
        e.gradients += uncounted.gradients;
        e.hessians += uncounted.hessians;
        e.energies += uncounted.energies;
    }
    let counted_before = network.evaluations;
    let journaled = calc
        .as_ref()
        .map(|c| journal.wrap(JOURNAL_LABEL, c.as_ref()));
    let stop = MapperStop::new(&dir, log.clone());
    let jobs = match (&journaled, &objective) {
        (Some(j), _) => {
            let jobs = CommandJobs::new(steps, j, stop, output, registry, network.evaluations);
            match &whole_system {
                Some(w) => jobs.with_whole(std::sync::Arc::clone(w)),
                None => jobs,
            }
        }
        (None, Some(o)) => CommandJobs::of_objective(
            steps,
            o.as_ref(),
            &objectives,
            stop,
            output,
            registry,
            network.evaluations,
        ),
        (None, None) => {
            return Err(MoptError::config(
                "run_mapper: neither a calculator nor an objective",
            ))
        }
    };
    // (The calculator as it is: every job records its evaluations under its own label.)
    let shared: Option<&dyn crate::pes::calculator::Calculator> = calc.as_deref();
    let workers = CommandWorkers::new(jobs, &dir, &journal, shared, &own_calculator);
    log.info(&format!(
        "{}; {} thread(s) on this machine.",
        schedule.describe(),
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    ));
    if schedule.workers > 1 || !workers.share_calculator() {
        log.info(if workers.share_calculator() {
            "The jobs share the calculator of the run; every job writes its log to jobs/."
        } else {
            "The calculator evaluates in order (it follows states or relaxes an environment \
             from structure to structure): every job has a calculator of its own; every job \
             writes its log to jobs/."
        });
    }
    let mut driver = MapperDriver::new(
        network,
        method,
        Box::new(super::mapper_progress::PriorityLog::new(&dir, priority)),
        rules,
        DriverSettings {
            seed: s.seed,
            matcher: if objective.is_some() {
                // Points of plain variables: the distance in the units of the variables.
                crate::workflow::explore::identity::StructureMatcher {
                    rmsd_tolerance: crate::domain::Quantity::new(s.variable_tolerance),
                    ..mapper_matcher(s.rmsd_threshold, s.energy_tolerance, s.mirror)
                }
            } else if whole_system.is_some() {
                // The cores of an ONIOM network: the environment moves their energies by
                // more than the tolerance of an energy, so the structure of the core alone
                // decides (`docs/source/mapper_design.md`).
                crate::workflow::explore::identity::StructureMatcher {
                    energy_tolerance: crate::domain::Quantity::new(f64::INFINITY),
                    ..mapper_matcher(s.rmsd_threshold, s.energy_tolerance, s.mirror)
                }
            } else {
                mapper_matcher(s.rmsd_threshold, s.energy_tolerance, s.mirror)
            },
            excluded: s.excluded.clone(),
            exclude_bond_rearrangement: s.exclude_bond_rearrangement,
            max_bond_changes: s.max_bond_changes,
            max_species_changes: s.max_species_changes,
            entrance_inputs: inputs
                .iter()
                .enumerate()
                .filter(|(_, i)| i.file.contains("(entrance structure "))
                .map(|(k, _)| k)
                .collect(),
            excited_state: steps.biased.excited_state,
            goal_stop: s.goal_stop,
            goal_bonds: s.goal_bonds,
            accept_temperature: s.accept_temperature,
            input_states: remap
                .as_ref()
                .map(|(_, _, _, states)| states.clone())
                .unwrap_or_default(),
            // Some minima of another network do not come through on another surface.
            skip_failed_inputs: remap.is_some(),
            environments: whole_system.is_some(),
            environment_tolerance: s.energy_tolerance * crate::domain::Units::KCAL_MOL_TO_HARTREE,
            cell_contact_margin: s
                .channels
                .then_some(s.channel_margin * crate::domain::Units::ANGSTROM_TO_BOHR),
            verify_minima: s.verify_minima,
        },
    );
    driver.set_job_limit(s.max_jobs);
    let run_entry = |net: &Network, end: &ExplorationEnd| {
        let mut m = std::collections::BTreeMap::new();
        // The records of the journal that the totals of `net` count: every evaluation that
        // was computed is one record (one evaluation of the calculator of the registry; a
        // Hessian made from gradients counts as those gradients, and then this is an upper
        // bound, which a restart does not go beyond).
        let made =
            |e: &crate::workflow::mapper::network::EvaluationTotals| e.gradients + e.energies;
        let records = at_open + made(&net.evaluations).saturating_sub(made(&counted_before));
        m.insert(
            JOURNAL_RECORDS.to_string(),
            JsonValue::Number(records as f64),
        );
        m.insert("method".to_string(), JsonValue::String(s.method.clone()));
        if let Some(first) = config.remap.as_ref().or(config.single_points.as_ref()) {
            // The network this one comes from.
            let mut source = std::collections::BTreeMap::new();
            let kind = if config.remap.is_some() {
                "remap"
            } else {
                "single_points"
            };
            source.insert("kind".to_string(), JsonValue::String(kind.to_string()));
            source.insert(
                "directory".to_string(),
                JsonValue::String(first.given.display().to_string()),
            );
            source.insert(
                "minima".to_string(),
                JsonValue::Number(first.minima.len() as f64),
            );
            m.insert("source".to_string(), JsonValue::Object(source));
        }
        m.insert(
            "priority".to_string(),
            JsonValue::String(s.priority.clone()),
        );
        m.insert("seed".to_string(), JsonValue::Number(s.seed as f64));
        m.insert(
            "calculator".to_string(),
            JsonValue::String(steps.biased.calculator.clone()),
        );
        // Whether the minima were verified by their Hessians (verify_minima).
        m.insert(
            "minima_verified".to_string(),
            JsonValue::Bool(steps.verify_minima),
        );
        m.insert(
            "ended".to_string(),
            end.reason
                .clone()
                .map_or(JsonValue::Null, JsonValue::String),
        );
        m.insert("interrupted".to_string(), JsonValue::Bool(end.interrupted));
        if let Some(goal) = net.goal_node() {
            let mut g = std::collections::BTreeMap::new();
            g.insert("node".to_string(), JsonValue::Number(goal as f64));
            g.insert(
                "joined".to_string(),
                JsonValue::Bool(net.goal_joined() == Some(true)),
            );
            m.insert("goal".to_string(), JsonValue::Object(g));
        }
        JsonValue::Object(m)
    };
    let mut trajectories =
        TrajectoryFiles::new(&dir, objective.is_some()).with_atoms(trajectory_atoms);
    let mut progress = super::mapper_progress::JobProgress::new(&dir, objective.is_some());
    let mut paths = PathFiles::new(&dir);
    let mut store = StoreObserver::new(&dir, stores, run_entry);
    let title = match config.remap.as_ref().or(config.single_points.as_ref()) {
        Some(first) => {
            let what = if config.remap.is_some() {
                "remap"
            } else {
                "single points"
            };
            first.given.file_name().map_or_else(
                || what.to_string(),
                |s| format!("{} ({what})", s.to_string_lossy()),
            )
        }
        None => config
            .inputs
            .first()
            .and_then(|p| p.file_stem())
            .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned()),
    };
    let mut report = MapperReport::new(
        &dir,
        &title,
        &steps.biased.calculator,
        &format!(
            "{:#?}\n\nSettings file:\n{}",
            config.settings,
            config.document.to_string_pretty()
        ),
        config.report.clone(),
    )
    .with_log(log.clone());
    if objective.is_some() {
        // An objective of its own for the surface behind the map: its evaluations are not
        // those of the run.
        report = report.with_objective(
            ObjectiveRegistry::builtin()
                .create(&steps.biased.calculator, &steps.biased.calculator_options)?,
        );
    }
    let systems: Vec<Configuration> = inputs.iter().map(|i| i.system.clone()).collect();
    let end = {
        let mut observers: Vec<&mut dyn MapperObserver> = Vec::new();
        if s.save_job_steps {
            observers.push(&mut trajectories);
        }
        observers.push(&mut paths);
        observers.push(&mut store);
        // After the store (it reads the network file the store wrote) and before the report
        // (which shows its results).
        if s.save_job_steps {
            observers.push(&mut progress);
        }
        if let Some(k) = end_kinetics.as_mut() {
            k.set_directory(&dir, log.clone());
            observers.push(k);
        }
        if s.html_report {
            observers.push(&mut report);
        }
        for o in extra.iter_mut() {
            observers.push(&mut **o);
        }
        match &config.single_points {
            // The energies of every node of the first network at its structure: the network
            // of this run is that one with them, written by the observers as at the end of
            // any run. A stopped run writes nothing; its evaluations are in the journal.
            Some(first) => {
                log.info(&format!(
                    "Single points: the energies of the {} node(s) and {} fragment(s) of the \
                     network of {} on the surface of this run, at its structures; nothing is \
                     optimized, and the first run is only read.",
                    first.network.nodes.len(),
                    first.network.fragments.len(),
                    first.given.display()
                ));
                match super::mapper_points::evaluate(first, &dir, &workers, schedule.workers)? {
                    Some(points) => {
                        let mut network = points.network;
                        if let Some(e) =
                            crate::workflow::mapper::MapperWorkers::run(&workers).evaluations()
                        {
                            network.evaluations = e;
                        }
                        *driver.network_mut() = network;
                        let end = ExplorationEnd {
                            reason: Some(format!(
                                "the {} single point(s) of the network are computed",
                                points.evaluations
                            )),
                            interrupted: false,
                        };
                        for o in observers.iter_mut() {
                            o.on_finish(driver.network(), &end)?;
                        }
                        end
                    }
                    None => ExplorationEnd {
                        reason: None,
                        interrupted: true,
                    },
                }
            }
            None => explore_with(&mut driver, &workers, &schedule, &systems, &mut observers)?,
        }
    };
    let network = driver.network().clone();
    match (&end.reason, end.interrupted) {
        (_, true) => log.info(&format!(
            "The run was interrupted after {} finished job(s); continue with: run_mapper \
             --restart {}",
            network.jobs.len(),
            dir.display()
        )),
        (Some(reason), false) => log.info(&format!("The exploration ended: {reason}")),
        (None, false) => log.info("The exploration ended."),
    }
    log.info(&format!(
        "Network: {} minima, {} transition states, {} job(s), {} warning(s); {}",
        network.minima().count(),
        network.of_kind(&NodeKind::TransitionState).count(),
        network.jobs.len(),
        network.warnings.len(),
        dir.join(NETWORK_FILE).display()
    ));
    if let Some((first, source, _, _)) = &remap {
        // What every minimum and item of the first network became.
        let (ran, added) = first.write_table(&dir, source, &network)?;
        log.info(&format!(
            "Remap: {ran} of the {} item(s) of the first network had their job, and {added} of \
             those jobs added what they were asked for; {}",
            source.items.len(),
            dir.join(super::mapper_remap::REMAP_TABLE).display()
        ));
    }
    let (replayed, unused) = journal.replay_counts();
    if replayed > 0 {
        log.info(&format!(
            "Restart: {replayed} evaluations taken from {JOURNAL_FILE}{}.",
            if unused > 0 {
                // The records of the jobs kept before the restart are not asked for again,
                // nor the second record of a structure that two jobs evaluated.
                format!(
                    " ({unused} records were not used, those of the jobs kept before the \
                     restart among them)"
                )
            } else {
                String::new()
            }
        ));
    }
    let e = network.evaluations;
    log.info(&format!(
        "Evaluations (the whole run): {} gradient calculation(s), {} exact Hessian \
         calculation(s){}",
        e.gradients,
        e.hessians,
        if e.energies > 0 {
            format!(", {} calculation(s) of the energy alone", e.energies)
        } else {
            String::new()
        }
    ));
    log.flush();
    Ok(MapperRun {
        directory: dir,
        network,
        interrupted: end.interrupted,
        reason: end.reason,
    })
}
