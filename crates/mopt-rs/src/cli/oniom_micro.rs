//! ONIOM microiterations of the commands that move structures (`optmain`, `ircmain`, the
//! chains of `nebmain`, the points of `relaxedscan`, and the searches built on them): the
//! optimizer, the IRC, or the chain moves the core, the environment is relaxed to a minimum
//! before every evaluation ([`crate::pes::oniom::micro`]), and the files of the run hold the
//! whole system.
//!
//! **Every ONIOM run of these commands has them.** A run is never continued without them
//! because one of its options does not fit: what can be taken to the relaxed surface is, and
//! what cannot is refused with the reason. `-oniom_opt micro=off` is the only way to move
//! every atom in the optimizer, and the log says that the environment is then not relaxed.
//!
//! - Fixed atoms of the environment (`-fix`, such as the backbone of a protein model) stay
//!   where they are in the relaxations.
//! - The other constraints and the bias potentials are given in the atom numbers of the
//!   whole system. An atom of the environment that one of them acts on is promoted to the
//!   core ([`CoreSplit::with_core`]): the optimizer moves it, holds the constraint, and feels
//!   the bias, and the rest of the environment is relaxed around it. Constraints are
//!   renumbered to the core ([`constraints_on_core`]), bias potentials and user functions
//!   evaluated at the core in the whole system ([`CoreTerm`]).
//! - A constraint on a Hessian eigenvector is one on an eigenvector of the Hessian of the
//!   relaxed surface (the Hessian the optimizer has).
//! - Excited states (`-es`) and functions of several states (`-mf`) are those of the relaxed
//!   surface; a first Hessian from a file (`-ih`) is taken as the Hessian of the core, or
//!   reduced to it ([`first_hessian`]).
//! - Refused: a bias potential or a user function that does not say which atoms it uses
//!   (it may depend on every atom, and nothing would be left to relax), and a variable cell.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;
use std::sync::Arc;

use super::optmain::OptmainConfig;
use crate::core_infra::log::Logger;
use crate::domain::{AtomId, MolecularSystem};
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;
use crate::pes::constraint::ConstraintFunction;
use crate::pes::constraint::{Constraint, ConstraintAnchor, ConstraintKind};
use crate::pes::oniom::micro::{write_full_structures, CoreSplit, CoreTerm, RelaxedEnvironment};
use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};
use crate::pes::PesTerm;

/// Bias potentials with their labels, as the runs keep them.
pub type Terms = Vec<(String, Arc<dyn PesTerm>)>;

/// The thresholds of the forces of the default criteria of `optmain` (largest component and
/// root mean square, Hartree/Bohr), for the runs that have no criteria of their own.
pub const NORMAL_FORCES: (f64, f64) = (3.0e-4, 2.0e-4);

/// The thresholds of the forces of `criteria`, to which the environment of a run with these
/// criteria is relaxed.
pub fn forces_of(criteria: &crate::opt::convergence::ConvergenceCriteria) -> (f64, f64) {
    (criteria.max_force.value(), criteria.rms_force.value())
}

/// The structure and the calculator a run moves, after [`wrap`].
pub struct Wrapped {
    /// The core with microiterations, else the whole structure.
    pub start: MolecularSystem,
    /// The relaxed surface of the core with microiterations, else the calculator of the
    /// registry.
    pub calc: Box<dyn Calculator>,
    /// Whether the surface treats electrons explicitly (the whole surface; the relaxed
    /// surface of a core does not say, since a core alone has no electron count).
    pub electrons: bool,
    /// The relaxed surface, kept to write the whole system at the end.
    pub micro: Option<Arc<RelaxedEnvironment>>,
    /// What the microiterations set up (or that `micro=off` turned them off), for the log.
    pub note: Option<String>,
    /// The constraints given to [`wrap_constrained`] in the numbering of `start` (renumbered
    /// to the core with microiterations; fixed atoms of the environment are held by the
    /// relaxation instead).
    pub constraints: Vec<Constraint>,
    /// The bias potentials given to [`wrap_constrained`], on the core with microiterations.
    pub terms: Terms,
}

impl Wrapped {
    /// The atom `atom` of the whole system in the numbering of the structures the run moves
    /// (the core with microiterations).
    pub fn on_core(&self, atom: AtomId) -> MoptResult<AtomId> {
        let Some(relaxed) = &self.micro else {
            return Ok(atom);
        };
        relaxed
            .split()
            .core
            .iter()
            .position(|&c| c == atom.index())
            .map(AtomId::new)
            .ok_or_else(|| {
                MoptError::config(format!(
                    "ONIOM microiterations: atom {} is relaxed with the environment; it is \
                     not one the optimizer moves",
                    atom.one_based()
                ))
            })
    }

    /// Selected trajectory atoms in the numbering used by this run.
    pub fn trajectory_atoms(&self, options: &CalculatorOptions) -> MoptResult<Option<Vec<AtomId>>> {
        options
            .oniom
            .as_ref()
            .and_then(|spec| spec.trajectory_atoms())
            .map(|atoms| atoms.into_iter().map(|a| self.on_core(a)).collect())
            .transpose()
    }

    /// The whole structure `whole` as the run moves it: its core with microiterations (its
    /// environment kept as the start of the relaxations near it), itself otherwise.
    pub fn structure(&self, whole: MolecularSystem) -> MoptResult<MolecularSystem> {
        match &self.micro {
            Some(relaxed) => {
                relaxed.remember(&whole)?;
                relaxed.core_of(&whole)
            }
            None => Ok(whole),
        }
    }
}

/// The atoms fixed by `-fix` (0-based, sorted).
pub fn fixed_atoms(constraints: &[Constraint]) -> Vec<usize> {
    let mut fixed: Vec<usize> = constraints
        .iter()
        .filter_map(|c| match &c.kind {
            ConstraintKind::Frozen(atoms) => Some(atoms.iter().map(|a| a.index())),
            _ => None,
        })
        .flatten()
        .collect();
    fixed.sort_unstable();
    fixed.dedup();
    fixed
}

/// What of `config` the microiterations cannot take (the run is refused then).
pub fn uncovered_by_optmain(config: &OptmainConfig) -> Option<&'static str> {
    if config.variable_cell.is_some() {
        Some("a variable cell (-vc; ONIOM has no stress)")
    } else if config
        .model_function
        .first()
        .is_some_and(|name| name.eq_ignore_ascii_case("bitss"))
    {
        Some(
            "-mf bitss (its two images are not set up on the relaxed surface; nebmain and \
             nebmain -dmf search between two structures with microiterations)",
        )
    } else {
        None
    }
}

/// The atoms that settings of `config` other than constraints and bias potentials name
/// (0-based): the fragments of `-gfix` and the reactive bonds of CCQN.
pub fn other_atoms_of_optmain(config: &OptmainConfig) -> Vec<usize> {
    let mut atoms: Vec<usize> = config
        .settings
        .fragments
        .iter()
        .flatten()
        .map(|a| a.index())
        .collect();
    atoms.extend(
        config
            .ccqn_bonds
            .iter()
            .flat_map(|(a, b)| [a.index(), b.index()]),
    );
    atoms
}

/// The atoms of the core of the microiterations of `options` for `start` (the outermost
/// model region and the outer atoms of its links), `None` without ONIOM microiterations.
/// Coordinates of a run that are made from structures (the ReSPer map of a scan) are made
/// from these atoms, so that the environment, which is relaxed, does not define them.
pub fn core_atoms(
    options: &CalculatorOptions,
    start: &MolecularSystem,
) -> MoptResult<Option<Vec<AtomId>>> {
    let Some(spec) = options.oniom.as_ref().filter(|s| s.micro.is_some()) else {
        return Ok(None);
    };
    let regions = spec.resolve(start)?;
    let split = CoreSplit::with_core(&regions, start.n_atoms(), &[], &[])?;
    Ok(Some(split.core.into_iter().map(AtomId::new).collect()))
}

/// The atoms the constraints (other than `-fix`) and the bias potentials act on (0-based,
/// sorted), or why they are not known (a term that does not say its atoms).
pub fn touched_atoms(
    constraints: &[Constraint],
    terms: &[(String, Arc<dyn PesTerm>)],
) -> Result<Vec<usize>, String> {
    let unknown = |what: &str, name: &str| {
        format!(
            "{what} '{name}', which does not say which atoms it uses (it may depend on every \
             atom, and nothing would be left to relax)"
        )
    };
    let mut atoms: Vec<usize> = Vec::new();
    for c in constraints {
        match &c.kind {
            ConstraintKind::Distance(a, b) => atoms.extend(a.iter().chain(b).map(|x| x.index())),
            ConstraintKind::Angle(a) => atoms.extend(a.iter().map(|x| x.index())),
            ConstraintKind::Dihedral(a) => atoms.extend(a.iter().map(|x| x.index())),
            ConstraintKind::Coordinate(a, _) => atoms.push(a.index()),
            ConstraintKind::Orientation(a) => atoms.extend(a.iter().map(|x| x.index())),
            ConstraintKind::AtomPair(a) => atoms.extend(a.iter().map(|x| x.index())),
            // Fixed atoms of the environment are held by the relaxation; an eigenvector is
            // one of the Hessian of the core.
            ConstraintKind::Frozen(_) | ConstraintKind::EigenVector(_) => {}
            ConstraintKind::Function(f) => atoms.extend(
                f.term
                    .atoms()
                    .ok_or_else(|| unknown("the user function", f.term.name()))?
                    .iter()
                    .map(|x| x.index()),
            ),
        }
    }
    for (label, term) in terms {
        atoms.extend(
            term.atoms()
                .ok_or_else(|| unknown("the bias potential", label))?
                .iter()
                .map(|x| x.index()),
        );
    }
    atoms.sort_unstable();
    atoms.dedup();
    Ok(atoms)
}

/// `constraints` (atom numbers of the whole system, structures like `template`) in the
/// numbering of the core of `split`, or why they cannot be (an atom of the environment that
/// was not promoted to the core, a user function that does not say its atoms). Fixed atoms
/// of the environment (`split.fixed`) are left out of `-fix`: the relaxation holds them. A
/// user function is evaluated at the core in the whole system.
pub fn constraints_on_core(
    constraints: &[Constraint],
    split: &CoreSplit,
    template: &MolecularSystem,
) -> Result<Vec<Constraint>, String> {
    let n = split.core.len() + split.environment.len() + split.fixed.len();
    let mut position = vec![None; n];
    for (k, &a) in split.core.iter().enumerate() {
        position[a] = Some(k);
    }
    let on_core = |atoms: &[AtomId], what: &str| -> Result<Vec<AtomId>, String> {
        atoms
            .iter()
            .map(|a| {
                position
                    .get(a.index())
                    .copied()
                    .flatten()
                    .map(AtomId::new)
                    .ok_or_else(|| {
                        format!(
                            "a constraint ({what}) on atom {} of the environment, which the \
                             relaxation of the environment moves",
                            a.index() + 1
                        )
                    })
            })
            .collect()
    };
    let mut out = Vec::with_capacity(constraints.len());
    for c in constraints {
        // A resolved constraint keeps what it is held to (the reference positions of frozen
        // or oriented atoms, the direction of a linear constraint), renumbered with its atoms:
        // without it a later stage would resolve it again at another structure.
        let mut anchor = c.anchor.clone();
        let kind = match &c.kind {
            ConstraintKind::Distance(a, b) => {
                ConstraintKind::Distance(on_core(a, "distance")?, on_core(b, "distance")?)
            }
            ConstraintKind::Angle(a) => ConstraintKind::Angle(array(on_core(a, "angle")?)?),
            ConstraintKind::Dihedral(a) => {
                ConstraintKind::Dihedral(array(on_core(a, "dihedral")?)?)
            }
            ConstraintKind::Coordinate(a, axis) => {
                let [b] = array(on_core(&[*a], "Cartesian coordinate")?)?;
                ConstraintKind::Coordinate(b, *axis)
            }
            ConstraintKind::Frozen(a) => {
                // Fixed atoms of the environment are held by the relaxation.
                let kept: Vec<bool> = a
                    .iter()
                    .map(|x| !split.fixed.contains(&x.index()))
                    .collect();
                let core: Vec<AtomId> = a
                    .iter()
                    .zip(&kept)
                    .filter(|(_, k)| **k)
                    .map(|(x, _)| *x)
                    .collect();
                if core.is_empty() {
                    continue;
                }
                if let Some(ConstraintAnchor::Positions(at)) = &anchor {
                    if at.len() != a.len() {
                        return Err(format!(
                            "a -fix constraint of {} atoms with {} reference positions",
                            a.len(),
                            at.len()
                        ));
                    }
                    anchor = Some(ConstraintAnchor::Positions(
                        at.iter()
                            .zip(&kept)
                            .filter(|(_, k)| **k)
                            .map(|(p, _)| *p)
                            .collect(),
                    ));
                }
                ConstraintKind::Frozen(on_core(&core, "-fix")?)
            }
            ConstraintKind::Orientation(a) => {
                ConstraintKind::Orientation(on_core(a, "orientation")?)
            }
            ConstraintKind::AtomPair(a) => {
                ConstraintKind::AtomPair(array(on_core(a, "atoms_pair")?)?)
            }
            // Of the Hessian the optimizer has: that of the relaxed surface of the core.
            ConstraintKind::EigenVector(k) => ConstraintKind::EigenVector(*k),
            ConstraintKind::Function(f) => {
                let on_core = CoreTerm::new(Arc::clone(&f.term), template, split)?;
                ConstraintKind::Function(ConstraintFunction {
                    term: Arc::new(on_core),
                    period: f.period,
                    fixes_frame: f.fixes_frame,
                    tolerance: f.tolerance,
                    warns_only: f.warns_only,
                })
            }
        };
        // The direction of a linear constraint holds atom numbers of the whole system.
        if let Some(ConstraintAnchor::Direction(direction)) = &anchor {
            let renumbered = direction
                .iter()
                .map(|(atom, v)| {
                    on_core(&[*atom], "the direction of a linear constraint").map(|a| (a[0], *v))
                })
                .collect::<Result<Vec<_>, String>>()?;
            anchor = Some(ConstraintAnchor::Direction(renumbered));
        }
        out.push(Constraint {
            kind,
            target: c.target,
            anchor,
        });
    }
    Ok(out)
}

fn array<const N: usize>(atoms: Vec<AtomId>) -> Result<[AtomId; N], String> {
    atoms
        .try_into()
        .map_err(|_| "a constraint with the wrong number of atoms".to_string())
}

/// The calculator `calculator` of a run that moves structures like `start`: with ONIOM in
/// `options`, the core and the calculator of its relaxed surface; otherwise `start` and the
/// calculator as the registry makes it. `uncovered` is what of the run the microiterations
/// cannot take: the run is refused then.
pub fn wrap(
    options: &CalculatorOptions,
    uncovered: Option<&'static str>,
    start: MolecularSystem,
    calculator: &str,
    registry: &CalculatorRegistry,
) -> MoptResult<Wrapped> {
    wrap_constrained(
        options,
        uncovered,
        &[],
        &[],
        &[],
        NORMAL_FORCES,
        start,
        calculator,
        registry,
    )
}

/// [`wrap`] for a run with `constraints` and the bias potentials `terms` (built on the whole
/// system, atom numbers of the whole system). With ONIOM the run has microiterations: the
/// fixed atoms of the environment are held by the relaxation, the atoms of the environment
/// that a constraint or a bias acts on (and `others`, atoms that other settings of the run
/// name) are promoted to the core, the constraints renumbered
/// to the core ([`constraints_on_core`]), and the terms evaluated at the core in the whole
/// system ([`CoreTerm`]). What cannot be taken to the relaxed surface (`uncovered`, a term
/// that does not say its atoms) is refused; nothing runs without microiterations unless
/// `-oniom_opt micro=off` says so.
pub fn wrap_constrained(
    options: &CalculatorOptions,
    uncovered: Option<&'static str>,
    constraints: &[Constraint],
    terms: &[(String, Arc<dyn PesTerm>)],
    others: &[usize],
    criteria: (f64, f64),
    start: MolecularSystem,
    calculator: &str,
    registry: &CalculatorRegistry,
) -> MoptResult<Wrapped> {
    let unchanged = |start, calc: Box<dyn Calculator>, note| Wrapped {
        start,
        electrons: calc.uses_electrons(),
        calc,
        micro: None,
        note,
        constraints: constraints.to_vec(),
        terms: terms.to_vec(),
    };
    let Some(spec) = options.oniom.as_ref() else {
        let calc = registry.create(calculator, options, &start)?;
        return Ok(unchanged(start, calc, None));
    };
    // The ONIOM surface of whole structures is not a calculator until it is said what it
    // is for: under the microiterations, or with every atom moving because micro=off says so.
    let whole = registry.create_oniom(calculator, options, &start)?;
    let Some(micro) = spec.micro.clone() else {
        let note = "ONIOM without microiterations, as -oniom_opt micro=off says: every atom \
                    moves in this run, and the environment is not relaxed to a minimum at \
                    the structures of the run."
            .to_string();
        let calc = whole.with_every_atom_moving(spec)?;
        return Ok(unchanged(start, calc, Some(note)));
    };
    let refused = |what: &str| {
        MoptError::unsupported(format!(
            "ONIOM: this run cannot have microiterations because of {what}. Every ONIOM run \
             but the molecular dynamics relaxes the environment to a minimum, and a run is \
             not continued without that; -oniom_opt micro=off moves every atom instead"
        ))
    };
    if let Some(what) = uncovered {
        return Err(refused(what));
    }
    let regions = spec.resolve(&start)?;
    let mut touched = touched_atoms(constraints, terms).map_err(|what| refused(&what))?;
    touched.extend(others.iter().copied().filter(|a| *a < start.n_atoms()));
    touched.sort_unstable();
    touched.dedup();
    let plain_core = CoreSplit::with_core(&regions, start.n_atoms(), &[], &[])
        .map(|s| s.core)
        .unwrap_or_default();
    let promoted: Vec<usize> = touched
        .iter()
        .copied()
        .filter(|a| !plain_core.contains(a))
        .collect();
    // A buffer of the environment moved with the core (`micro_core_shell=R`): the free atoms
    // within R of an atom of the model region (minimum images in a cell).
    let fixed = fixed_atoms(constraints);
    let buffer: Vec<usize> = match micro.core_shell {
        Some(radius) => {
            let r = radius * crate::domain::Units::ANGSTROM_TO_BOHR;
            let cell = start.active_cell();
            let x = start.coords_flat_bohr();
            let at = |a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
            let model: Vec<usize> = regions
                .iter()
                .flat_map(|g| g.atoms.iter().map(|a| a.index()))
                .collect();
            (0..start.n_atoms())
                .filter(|a| {
                    !plain_core.contains(a)
                        && !promoted.contains(a)
                        && fixed.binary_search(a).is_err()
                })
                .filter(|&a| {
                    let p = at(a);
                    model.iter().any(|&c| {
                        let q = at(c);
                        let d = match cell {
                            Some(cell) => cell.mic_delta(p, q),
                            None => std::array::from_fn(|i| p[i] - q[i]),
                        };
                        d.iter().map(|v| v * v).sum::<f64>() <= r * r
                    })
                })
                .collect()
        }
        None => Vec::new(),
    };
    let mut promoted_all = promoted.clone();
    promoted_all.extend(buffer.iter().copied());
    promoted_all.sort_unstable();
    let split = CoreSplit::with_core(&regions, start.n_atoms(), &fixed, &promoted_all)?;
    let constraints =
        constraints_on_core(constraints, &split, &start).map_err(|what| refused(&what))?;
    let terms = terms_on_core(terms, &start, &split).map_err(|what| refused(&what))?;
    let electrons = whole.uses_electrons();
    if electrons {
        start.spin_charge.check_electrons(&start.elements)?;
    }
    let lowest = spec
        .lower_levels
        .last()
        .ok_or_else(|| MoptError::config("ONIOM microiterations: no lower level"))?;
    let plain = CalculatorOptions {
        oniom: None,
        ..options.clone()
    };
    // The lowest level with its own settings, as in the ONIOM surface itself.
    let own = plain
        .for_lower_level(spec.options_of_lower_level(spec.lower_levels.len().saturating_sub(1)));
    let low = registry.create(lowest, &own, &start)?;
    // With electronic embedding the model systems see the environment: it is relaxed on the
    // whole surface.
    let mut settings = crate::pes::oniom::micro::MicroOniomSettings {
        embedded: spec.embedding.is_some(),
        ..micro.settings_for(criteria)
    };
    // The methods of the optimizer registry (`micro_optimizer=NAME`, and FIRE where a line
    // search of the limited-memory BFGS fails) come from the optimizers, a layer above the
    // surface that runs the relaxations.
    settings.inner.methods = Some(crate::opt::micro::REGISTRY_METHODS);
    // The molecular dynamics of the start `md` of the replicas of the environment come from
    // the dynamics of the crate, a layer above the surface as well.
    settings.replicas.dynamics = Some(crate::pes::oniom::micro::replicas::EnvironmentDynamics(
        environment_dynamics,
    ));
    let relaxed = Arc::new(
        RelaxedEnvironment::new(whole, low, start, split, settings)?
            .with_counters(registry.counters()),
    );
    let core = relaxed.core_system()?;
    let mut notes = relaxed.setup_notes();
    if !promoted.is_empty() {
        let list: Vec<String> = promoted.iter().map(|a| (a + 1).to_string()).collect();
        notes.push(format!(
            "ONIOM microiterations: {} atom(s) of the environment that a constraint, a bias \
             potential, or another setting of the run names are moved by the optimizer \
             with the core ({}); the rest of the environment is relaxed around them.",
            promoted.len(),
            list.join(", ")
        ));
    }
    if let Some(radius) = micro.core_shell {
        notes.push(format!(
            "ONIOM microiterations: {} free atom(s) of the environment within {radius:.2} \
             Angstrom of the model region are moved by the optimizer with the core \
             (micro_core_shell); the rest of the environment is relaxed around them.",
            buffer.len()
        ));
    }
    Ok(Wrapped {
        start: core,
        electrons,
        calc: Box::new(Arc::clone(&relaxed)),
        micro: Some(relaxed),
        note: (!notes.is_empty()).then(|| notes.join("\n")),
        constraints,
        terms,
    })
}

/// The molecular dynamics that starts a replica of the environment
/// (`-oniom_opt micro_replica_start=md`): the environment of `problem` (the core held where it
/// is) from `start` for the steps of `request` at its temperature, with the velocities of the
/// Maxwell-Boltzmann distribution and Bussi's thermostat of the dynamics of the crate (`md`,
/// as `mdmain` runs them; seeded, so the same for any number of threads), giving the
/// coordinates at the end.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn environment_dynamics(
    problem: &dyn crate::pes::microiter::MicroProblem,
    start: &[f64],
    request: &crate::pes::oniom::micro::replicas::DynamicsRequest,
) -> MoptResult<Vec<f64>> {
    use crate::md::{MdAction, MdSettings, MdSpace, MdValue, MolecularDynamics};
    let n = start.len();
    if request.masses.len() * 3 != n {
        return Err(MoptError::calc(format!(
            "ONIOM replicas: {} masses for {} coordinates of the environment",
            request.masses.len(),
            n
        )));
    }
    let settings = MdSettings {
        steps: request.steps,
        temperature: request.temperature,
        thermostat: "bussi".to_string(),
        seed: request.seed,
        ..MdSettings::default()
    };
    let space = MdSpace::atoms(&request.masses)?;
    let mut md =
        MolecularDynamics::new(settings, space, faer::Col::from_fn(n, |i| start[i]), None)?;
    while !md.finished() {
        match md.ask() {
            MdAction::Evaluate(at) => {
                let y: Vec<f64> = at.x.iter().copied().collect();
                let e = problem.evaluate(&y, false)?;
                md.tell(MdValue::plain(e.energy, e.gradient))?;
            }
            MdAction::Finished(_) => break,
        }
    }
    Ok(md.positions().iter().copied().collect())
}

/// Relaxes the environment of whole structures for the ONIOM runs that do not move the core
/// (vibrational analyses): what they compute is then a minimum of the environment, as in
/// every other ONIOM run. `None` without ONIOM, or with `-oniom_opt micro=off`.
pub struct EnvironmentRelaxer {
    relaxed: Arc<RelaxedEnvironment>,
    excited_state: usize,
}

impl EnvironmentRelaxer {
    /// The relaxer of the ONIOM calculation `name` with `options` for structures like
    /// `template` (the atoms of `fixed`, 0-based, of the environment stay where they are).
    pub fn new(
        name: &str,
        options: &CalculatorOptions,
        template: &MolecularSystem,
        fixed: &[usize],
        excited_state: usize,
        registry: &CalculatorRegistry,
    ) -> MoptResult<Option<Self>> {
        if options.oniom.as_ref().is_none_or(|s| s.micro.is_none()) {
            return Ok(None);
        }
        let held = [Constraint::at_start(ConstraintKind::Frozen(
            fixed.iter().map(|&a| AtomId::new(a)).collect(),
        ))];
        let constraints: &[Constraint] = if fixed.is_empty() { &[] } else { &held };
        let wrapped = wrap_constrained(
            options,
            None,
            constraints,
            &[],
            &[],
            NORMAL_FORCES,
            template.clone(),
            name,
            registry,
        )?;
        Ok(wrapped.micro.map(|relaxed| Self {
            relaxed,
            excited_state,
        }))
    }

    /// `whole` with its environment relaxed, and a line for the log.
    pub fn relax(&self, whole: &MolecularSystem) -> MoptResult<(MolecularSystem, String)> {
        let (relaxed, moved) = self.relaxed.relaxed_system(whole, self.excited_state)?;
        let c = self.relaxed.counts();
        let mut note = format!(
            "ONIOM microiterations: the environment ({} atoms) relaxed to a minimum at the \
             core of the structure before the analysis; its largest displacement is {:.3e} \
             Angstrom ({} inner iteration(s) so far, {} negative-curvature mode(s) removed).",
            self.relaxed.split().environment.len(),
            moved * crate::domain::Units::BOHR_TO_ANGSTROM,
            c.inner_iterations,
            c.killed_modes
        );
        if let Some(t) = self.relaxed.replica_temperature() {
            note.push_str(&format!(
                " With {} replicas of the environment the structure analysed is that of the \
                 replica of the lowest energy: the analysis of a whole structure is that of one \
                 environment, not of the soft minimum at {t} K (whose Hessian of the core is that \
                 of the checks of the commands that move the core).",
                self.relaxed.replica_count()
            ));
        }
        Ok((relaxed, note))
    }
}

/// The calculator `calculator` for energies and derivatives at given whole structures (an
/// analysis of a structure whose environment was relaxed before, a check of a result).
pub fn at_given_structures(
    calculator: &str,
    options: &CalculatorOptions,
    sys: &MolecularSystem,
    registry: &CalculatorRegistry,
) -> MoptResult<Box<dyn Calculator>> {
    if options.oniom.is_some() {
        Ok(registry
            .create_oniom(calculator, options, sys)?
            .at_given_structures())
    } else {
        registry.create(calculator, options, sys)
    }
}

/// The calculator `calculator` of a molecular dynamics (every atom moves by its equation of
/// motion; with ONIOM nothing is relaxed).
pub fn for_dynamics(
    calculator: &str,
    options: &CalculatorOptions,
    sys: &MolecularSystem,
    registry: &CalculatorRegistry,
) -> MoptResult<Box<dyn Calculator>> {
    if options.oniom.is_some() {
        Ok(registry
            .create_oniom(calculator, options, sys)?
            .for_dynamics())
    } else {
        registry.create(calculator, options, sys)
    }
}

/// Inner iterations between two lines of a relaxation in the stream and the log.
pub const REPORT_EVERY: usize = 50;

/// From now on the relaxations of `micro` (if any) write to `log` their gradient every
/// [`REPORT_EVERY`] inner iterations, the end of every descent, and the test of the lowest
/// curvature at its end (not the coordinates of the environment).
pub fn report_relaxations(micro: Option<&RelaxedEnvironment>, log: &Logger) {
    if let Some(relaxed) = micro {
        relaxed.report_to(log.clone(), REPORT_EVERY);
    }
}

/// A saddle-point search on the relaxed surface of microiterations (`micro`, with
/// `settings.saddle_order > 0`) ends at a verified stationary point: at its convergence the
/// optimizer asks for the Hessian of a check of stability (the environment relaxed to the
/// tolerance of the checks, the error of the curvatures estimated; `pes::oniom::micro`),
/// counts its imaginary modes by the one criterion of `opt::mode_kill`, and leaves the
/// structure along any mode beyond the order asked for and goes on with the search (the mode
/// kill of the core), up to the displacements of the mode kill; a structure that still has
/// more ends the run without convergence. A mode kill asked for (`-modekill`) keeps its own
/// settings and judges by such a Hessian too. A note for the log, or `None` where nothing is
/// changed (no microiterations, a minimization).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn verified_saddle_points(
    settings: &mut crate::opt::state_machine::OptimizerSettings,
    micro: Option<&RelaxedEnvironment>,
) -> Option<String> {
    let micro = micro?;
    if settings.saddle_order == 0 {
        return None;
    }
    let mode_kill = settings.mode_kill.get_or_insert_with(Default::default);
    mode_kill.checked_hessian = true;
    Some(format!(
        "ONIOM microiterations: the converged structure of the search for a saddle point of \
         order {} is verified with the Hessian of a check of stability (the environment \
         relaxed to {}, the tolerance of the checks, which the IRC, the vibrational analysis, \
         and the mapper take as well); a mode beyond the order is left and the search goes on \
         (at most {} times).",
        settings.saddle_order,
        describe_check_tolerance(micro),
        mode_kill.max_rounds
    ))
}

/// The tolerance of the environment of the checks of stability of `micro`, for the log.
pub fn describe_check_tolerance(micro: &RelaxedEnvironment) -> String {
    let (largest, rms) = micro.check_tolerance();
    let replicas = micro.replica_temperature().map_or_else(String::new, |t| {
        format!(
            ", every one of the {} replicas of the environment, combined by their soft minimum \
             at {t} K",
            micro.replica_count()
        )
    });
    format!(
        "a largest gradient component of {largest:.1e} and an RMS of {rms:.1e} \
         Hartree/Bohr{replicas}"
    )
}

/// `calc` with the Hessian of the file `file` for its first Hessian (`-ih`). With
/// microiterations (`micro`) the file holds the Hessian of the core, or that of the whole
/// system, which is reduced to the core with the response of the environment taken off
/// ([`RelaxedEnvironment::core_hessian_from`]).
pub fn first_hessian(
    calc: Box<dyn Calculator>,
    file: Option<&Path>,
    start: &MolecularSystem,
    micro: Option<&RelaxedEnvironment>,
) -> MoptResult<Box<dyn Calculator>> {
    let (Some(path), Some(relaxed)) = (file, micro) else {
        return crate::pes::given_hessian::wrap_if_given(calc, file, start);
    };
    let given = crate::io::hessian_file::read_hessian_file(path)?;
    let core = relaxed
        .core_hessian_from(&given)
        .map_err(|e| e.context(format!("{}", path.display())))?;
    Ok(Box::new(crate::pes::given_hessian::GivenHessian::new(
        calc, core,
    )?))
}

/// `terms` (of whole structures like `start`) on the core of `split`, or why one cannot be.
fn terms_on_core(
    terms: &[(String, Arc<dyn PesTerm>)],
    start: &MolecularSystem,
    split: &CoreSplit,
) -> Result<Terms, String> {
    terms
        .iter()
        .map(|(label, term)| {
            let on_core = CoreTerm::new(Arc::clone(term), start, split)?;
            Ok((label.clone(), Arc::new(on_core) as Arc<dyn PesTerm>))
        })
        .collect()
}

/// The bias potentials of `specs` built on `start` (the whole system), labelled as the runs
/// label them.
pub fn build_terms(specs: &[super::bias::BiasSpec], start: &MolecularSystem) -> MoptResult<Terms> {
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    let mut terms = Vec::with_capacity(specs.len());
    for spec in specs {
        let k = counts.entry(spec.key.as_str()).or_insert(0);
        terms.push((spec.label(*k), spec.build(start)?));
        *k += 1;
    }
    Ok(terms)
}

/// After the run: with an adaptive tolerance, the environment of the final core `last`
/// relaxed to the full tolerance; the structure files with the whole system; and what the
/// relaxations did.
pub fn finish(
    relaxed: &RelaxedEnvironment,
    last: Option<&MolecularSystem>,
    dir: &Path,
    log: &Logger,
) -> MoptResult<()> {
    if let Some(core) = last {
        if let Some(f) = relaxed.relax_fully(core)? {
            log.info(&format!(
                "ONIOM microiterations: the environment of the final structure relaxed to the \
                 full tolerance: energy change {:.3e} Hartree, largest gradient component of \
                 the core {:.3e} Hartree/Bohr.",
                f.energy_change, f.max_core_gradient
            ));
        }
    }
    let files = write_full_structures(dir, relaxed)?;
    let c = relaxed.counts();
    log.info(&format!(
        "ONIOM microiterations: {} core atom(s), {} environment atom(s) ({} fixed); {} \
         relaxation(s) of the environment with {} inner iteration(s), {} negative-curvature \
         mode(s) removed, {} relaxation(s) short of the threshold, {} started from the \
         predicted response, {} to a looser tolerance; {files} structure file(s) written with \
         the whole system.",
        relaxed.split().core.len(),
        relaxed.split().environment.len() + relaxed.split().fixed.len(),
        relaxed.split().fixed.len(),
        c.relaxations,
        c.inner_iterations,
        c.killed_modes,
        c.unconverged,
        c.predicted,
        c.loose
    ));
    log.info(&format!(
        "ONIOM microiterations: {} gradient(s) of the lowest level in the relaxations, {} of \
         them in {} check(s) of the curvature ({} Hessian-vector products, {} check(s) at \
         their limit of products); {} relaxation(s) with the preconditioner made again without \
         it; {} trial structure(s) with atoms closer than 0.5 Angstrom refused; {} \
         relaxation(s) took the environment to another local minimum.",
        c.environment_gradients,
        2 * c.curvature_products,
        c.curvature_checks,
        c.curvature_products,
        c.unconverged_checks,
        c.preconditioner_retries,
        c.contacts,
        c.jumps
    ));
    if let Some(summary) = relaxed.replica_summary() {
        log.info(&format!(
            "{summary} The structure files hold the replica of the lowest energy (its number and \
             weight start the comment line), and every replica is in the file beside each, \
             NAME{}.",
            crate::pes::oniom::micro::REPLICAS_SUFFIX
        ));
    }
    if c.fallbacks > 0 {
        log.info(&format!(
            "ONIOM microiterations: {} relaxation(s) went on with FIRE after a line search \
             found no step that lowers the energy (the surface of the relaxation does not \
             follow its gradient there), {} inner iteration(s) with FIRE.",
            c.fallbacks, c.fallback_iterations
        ));
    }
    if c.unconverged > 0 {
        log.warn(&format!(
            "ONIOM microiterations: {} relaxation(s) of the environment stopped short of the \
             threshold (-oniom_opt micro_tolerance), {} of them where the energy no longer \
             decreased (a floor of the lowest level).",
            c.unconverged, c.stalled
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn atoms(list: &[usize]) -> Vec<AtomId> {
        list.iter().map(|&a| AtomId::new(a)).collect()
    }

    #[test]
    fn constraints_on_the_core_are_renumbered_and_others_refused() {
        // Core: atoms 2, 3, 5 of eight (0-based); atom 7 is a fixed atom of the environment.
        let split = CoreSplit {
            core: vec![2, 3, 5],
            environment: vec![0, 1, 4, 6],
            fixed: vec![7],
        };
        let given = vec![
            Constraint {
                kind: ConstraintKind::Distance(atoms(&[2]), atoms(&[5])),
                target: Some(2.5),
                anchor: None,
            },
            Constraint::at_start(ConstraintKind::Dihedral([
                AtomId::new(5),
                AtomId::new(3),
                AtomId::new(2),
                AtomId::new(3),
            ])),
            Constraint::at_start(ConstraintKind::Frozen(atoms(&[3, 7]))),
            // Only fixed atoms of the environment: held by the relaxation, nothing left.
            Constraint::at_start(ConstraintKind::Frozen(atoms(&[7]))),
        ];
        let template = MolecularSystem::from_flat_bohr(
            vec![crate::domain::Element::C; 8],
            &faer::Col::from_fn(24, |k| 1.7 * k as f64 + 0.3 * (k % 3) as f64),
            crate::domain::SpinCharge::default(),
            None,
        )
        .unwrap();
        let core = constraints_on_core(&given, &split, &template).unwrap();
        assert_eq!(core.len(), 3);
        assert_eq!(
            core[0].kind,
            ConstraintKind::Distance(atoms(&[0]), atoms(&[2]))
        );
        assert_eq!(core[0].target, Some(2.5));
        assert_eq!(
            core[1].kind,
            ConstraintKind::Dihedral([
                AtomId::new(2),
                AtomId::new(1),
                AtomId::new(0),
                AtomId::new(1)
            ])
        );
        assert_eq!(core[2].kind, ConstraintKind::Frozen(atoms(&[1])));
        // An atom of the environment, an eigenvector: refused with the reason.
        let outside = [Constraint::at_start(ConstraintKind::Angle([
            AtomId::new(2),
            AtomId::new(3),
            AtomId::new(4),
        ]))];
        let why = constraints_on_core(&outside, &split, &template).unwrap_err();
        assert!(why.contains("atom 5 of the environment"), "{why}");
        // Its atoms are those to promote to the core; an eigenvector passes as it is.
        assert_eq!(touched_atoms(&outside, &[]).unwrap(), vec![2, 3, 4]);
        let eigen = [Constraint::at_start(ConstraintKind::EigenVector(0))];
        let passed = constraints_on_core(&eigen, &split, &template).unwrap();
        assert_eq!(passed[0].kind, ConstraintKind::EigenVector(0));
        assert!(touched_atoms(&eigen, &[]).unwrap().is_empty());
    }

    #[test]
    fn resolved_targets_hold_on_the_core_at_the_rounding_floor() {
        // Core: atoms 2, 3, 5, 6 of eight (0-based); targets resolved on the whole structure
        // are the values of the renumbered constraints on its core.
        let split = CoreSplit {
            core: vec![2, 3, 5, 6],
            environment: vec![0, 1, 4],
            fixed: vec![7],
        };
        let template = MolecularSystem::from_flat_bohr(
            vec![crate::domain::Element::C; 8],
            &faer::Col::from_fn(24, |k| 1.7 * k as f64 + 0.3 * ((k * k) % 5) as f64),
            crate::domain::SpinCharge::default(),
            None,
        )
        .unwrap();
        let kinds = [
            // A bond between the centroids of two fragments (fbond).
            ConstraintKind::Distance(atoms(&[2, 3]), atoms(&[5, 6])),
            ConstraintKind::Angle([AtomId::new(2), AtomId::new(5), AtomId::new(6)]),
            ConstraintKind::Dihedral([
                AtomId::new(3),
                AtomId::new(2),
                AtomId::new(5),
                AtomId::new(6),
            ]),
            ConstraintKind::Frozen(atoms(&[3, 7])),
        ];
        let resolved: Vec<Constraint> = kinds
            .iter()
            .map(|k| {
                Constraint::at_start(k.clone())
                    .resolve(&template, None)
                    .unwrap()
            })
            .collect();
        let on_core = constraints_on_core(&resolved, &split, &template).unwrap();
        let core_ids: Vec<AtomId> = split.core.iter().map(|&a| AtomId::new(a)).collect();
        let core = template.subsystem(&core_ids, template.spin_charge).unwrap();
        for (c, whole) in on_core.iter().zip(&resolved) {
            let again = Constraint::at_start(c.kind.clone())
                .resolve(&core, None)
                .unwrap();
            match (c.target, again.target) {
                (Some(t), Some(v)) => {
                    assert_eq!(c.target, whole.target);
                    assert!((t - v).abs() < 1e-12, "{:?}: {t} {v}", c.kind);
                }
                (None, None) => {}
                other => panic!("{:?}: targets {other:?}", c.kind),
            }
            if let (Some(ConstraintAnchor::Positions(a)), Some(ConstraintAnchor::Positions(b))) =
                (&c.anchor, &again.anchor)
            {
                assert_eq!(a.len(), b.len());
                for (p, q) in a.iter().zip(b) {
                    for k in 0..3 {
                        assert!((p[k] - q[k]).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn resolved_constraints_keep_their_anchors_on_the_core() {
        // Core: atoms 2, 3, 5 of eight (0-based); atom 7 is a fixed atom of the environment.
        let split = CoreSplit {
            core: vec![2, 3, 5],
            environment: vec![0, 1, 4, 6],
            fixed: vec![7],
        };
        let template = MolecularSystem::from_flat_bohr(
            vec![crate::domain::Element::C; 8],
            &faer::Col::from_fn(24, |k| 1.7 * k as f64 + 0.3 * (k % 3) as f64),
            crate::domain::SpinCharge::default(),
            None,
        )
        .unwrap();
        let given = [
            // Frozen atoms 3 and 7 resolved at their positions: atom 7 is held by the
            // relaxation, atom 3 keeps its own reference position.
            Constraint {
                kind: ConstraintKind::Frozen(atoms(&[3, 7])),
                target: None,
                anchor: Some(ConstraintAnchor::Positions(vec![
                    [1.0, 2.0, 3.0],
                    [7.0, 8.0, 9.0],
                ])),
            },
            // A resolved pair direction on atoms 2 and 5.
            Constraint {
                kind: ConstraintKind::AtomPair([AtomId::new(2), AtomId::new(5)]),
                target: Some(0.25),
                anchor: Some(ConstraintAnchor::Direction(vec![
                    (AtomId::new(2), [1.0, 0.0, 0.0]),
                    (AtomId::new(5), [-1.0, 0.0, 0.0]),
                ])),
            },
            Constraint {
                kind: ConstraintKind::Orientation(atoms(&[5, 2])),
                target: None,
                anchor: Some(ConstraintAnchor::Positions(vec![
                    [4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0],
                ])),
            },
        ];
        let core = constraints_on_core(&given, &split, &template).unwrap();
        assert_eq!(core[0].kind, ConstraintKind::Frozen(atoms(&[1])));
        assert_eq!(
            core[0].anchor,
            Some(ConstraintAnchor::Positions(vec![[1.0, 2.0, 3.0]]))
        );
        assert!(core[0].is_resolved());
        assert_eq!(
            core[1].anchor,
            Some(ConstraintAnchor::Direction(vec![
                (AtomId::new(0), [1.0, 0.0, 0.0]),
                (AtomId::new(2), [-1.0, 0.0, 0.0]),
            ]))
        );
        assert_eq!(core[1].target, Some(0.25));
        assert!(core[1].is_resolved());
        assert_eq!(core[2].kind, ConstraintKind::Orientation(atoms(&[2, 0])));
        assert_eq!(core[2].anchor, given[2].anchor);
        // A direction with a component on an atom the relaxation moves is refused.
        let outside = [Constraint {
            kind: ConstraintKind::EigenVector(0),
            target: Some(0.0),
            anchor: Some(ConstraintAnchor::Direction(vec![
                (AtomId::new(2), [1.0, 0.0, 0.0]),
                (AtomId::new(4), [0.0, 1.0, 0.0]),
            ])),
        }];
        let why = constraints_on_core(&outside, &split, &template).unwrap_err();
        assert!(why.contains("atom 5 of the environment"), "{why}");
    }
}
