//! ONIOM: levels of theory on nested regions of a structure, as one calculator.
//!
//! - [`layout`]: the regions, their link atoms, the model systems, and the way back for
//!   gradients and Hessians (link atoms at a share of the bond or at a fixed distance).
//! - [`calculator`]: [`OniomCalculator`], the subtractive N-level energy
//!   (mechanical embedding).
//! - [`OniomSpec`]: what the command line says (`-oniom`, `-oniom_middle`, `-oniom_opt`),
//!   resolved against the starting structure ([`OniomSpec::resolve`]); a
//!   [`crate::pes::registry::CalculatorRegistry`] given a spec in the calculator options
//!   makes the ONIOM calculator, so that every command has it.
//!
//! MultiOptPy: `optimization.py:ONIOMHandler` and `Utils/oniom.py` (`-oniom HIGH LINK LOW`;
//! MultiOptPy reads the option and its handler raises an exception).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod calculator;
pub mod embedding;
pub mod layout;
pub mod micro;
#[cfg(test)]
mod tests;

pub use calculator::{ModelPart, OniomCalculator};

use crate::pes::calculator::Calculator;

/// The ONIOM surface of whole structures as the registry makes it
/// ([`crate::pes::registry::CalculatorRegistry::create_oniom`]): nothing relaxes its
/// environment. It is not a [`Calculator`]. A run turns it into one by saying what it is for,
/// so that no command moves structures on it by oversight: every ONIOM run but the molecular
/// dynamics has microiterations ([`micro::RelaxedEnvironment::new`] takes this type).
pub struct WholeOniom {
    calc: Box<dyn Calculator>,
    /// The terms of its model systems apart (mechanical embedding), for the Hessian of the
    /// core of the microiterations.
    model: Option<calculator::ModelPart>,
}

impl WholeOniom {
    /// The surface `calc` of whole structures.
    pub(crate) fn new(calc: Box<dyn Calculator>) -> Self {
        Self { calc, model: None }
    }

    /// The same with the terms of its model systems apart ([`calculator::ModelPart`]).
    pub(crate) fn with_model_part(self, model: Option<calculator::ModelPart>) -> Self {
        Self { model, ..self }
    }

    /// The surface under the microiterations (the environment is relaxed around it), with
    /// the terms of its model systems apart where it has them.
    pub(crate) fn under_microiterations(
        self,
    ) -> (Box<dyn Calculator>, Option<calculator::ModelPart>) {
        (self.calc, self.model)
    }

    /// For molecular dynamics: every atom moves by its equation of motion.
    pub fn for_dynamics(self) -> Box<dyn Calculator> {
        self.calc
    }

    /// For energies and derivatives at given whole structures: structures whose environment
    /// was relaxed before (analyses), and checks of results.
    pub fn at_given_structures(self) -> Box<dyn Calculator> {
        self.calc
    }

    /// For a run in which every atom moves because `-oniom_opt micro=off` says so; refused
    /// when `spec` has microiterations.
    pub fn with_every_atom_moving(
        self,
        spec: &OniomSpec,
    ) -> crate::error::MoptResult<Box<dyn Calculator>> {
        if spec.micro.is_some() {
            return Err(MoptError::unsupported(
                "ONIOM: a run that moves structures has microiterations; every atom moves \
                 in the optimizer only with -oniom_opt micro=off",
            ));
        }
        Ok(self.calc)
    }

    /// Whether the surface treats electrons explicitly.
    pub fn uses_electrons(&self) -> bool {
        self.calc.uses_electrons()
    }
}
pub use layout::{crossing_bonds, pair_with_nearest, Link, LinkPlacement, Region};

use crate::domain::{AtomId, Element, MolecularSystem, SpinCharge};
use crate::error::{MoptError, MoptResult};

/// MultiOptPy's threshold for a bond across the border: 1.2 times the sum of the covalent
/// radii.
pub const BOND_SCALE: f64 = 1.2;

/// Which bonds of a region get link atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkSpec {
    /// None (`none`): the region is cut without caps (a molecule of its own).
    None,
    /// Every bond to an atom outside (`auto`: closer than [`BOND_SCALE`] times the sum of
    /// the covalent radii).
    Auto,
    /// The given outside atoms (`Q2`), each with its nearest atom of the region (as
    /// MultiOptPy pairs them).
    Given(Vec<AtomId>),
}

/// One region as the command line gives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionSpec {
    /// Its atoms.
    pub atoms: Vec<AtomId>,
    /// Its link atoms.
    pub links: LinkSpec,
    /// Charge and multiplicity of its model system (`None`: those of the real system, with
    /// the multiplicity changed by one where the link atoms change the parity of the
    /// electrons).
    pub spin_charge: Option<SpinCharge>,
}

/// Microiterations of an ONIOM optimization ([`micro`]; on by default, `-oniom_opt
/// micro=off` turns them off).
#[derive(Debug, Clone, PartialEq)]
pub struct MicroOptions {
    /// The mode kill of the environment (`micro_mode_kill=N`; off by default: the low layer
    /// is not mode-killed, the core is by its optimizer).
    pub environment_mode_kill: bool,
    /// With it on: in the first relaxation and then in every this many.
    pub mode_kill_every: usize,
    /// Largest gradient component of the relaxed environment (Hartree/Bohr), when given
    /// (`micro_tolerance`); `None`: the thresholds of the forces of the run itself
    /// ([`MicroOptions::settings_for`]).
    pub tolerance: Option<f64>,
    /// Asked for on the command line (`micro=on` or one of its settings): a run they do not
    /// cover is refused; by default such a run goes without them.
    pub requested: bool,
    /// The Hessian of the core (`micro_hessian=effective|approximate|held`).
    pub core_hessian: micro::CoreHessian,
    /// The minimizer of the environment (`micro_optimizer`).
    pub optimizer: crate::pes::microiter::InnerMethod,
    /// The coordinates of the environment (`micro_coord`).
    pub coordinates: micro::EnvironmentCoordinates,
    /// Start from the predicted response of the environment (`micro_predict`).
    pub predict: bool,
    /// Tolerance as this share of the core gradient (`micro_adaptive`; 0: off).
    pub adaptive: f64,
    /// The check of the curvature from the mode of the last one with the limited-memory
    /// BFGS of the relaxation as preconditioner (`micro_curvature=guided`, the default) or
    /// from a fixed start (`plain`).
    pub guided_curvature: bool,
    /// The model Hessian whose environment block preconditions the relaxations
    /// (`micro_precondition=NAME`; `None`: none).
    pub precondition: Option<String>,
    /// Its shift of the diagonal (`micro_precondition_shift`, Hartree/Bohr^2).
    pub precondition_shift: f64,
    /// The memory the Hessian of the core may take (`micro_hessian_memory`, GiB; `None`: half
    /// of the free memory).
    pub hessian_memory: Option<f64>,
    /// The shell of the environment that responds in the Hessian of the core
    /// (`micro_hessian_shell`, Angstrom; `None`: all of it).
    pub hessian_shell: Option<f64>,
    /// How the response of the environment in the Hessian of the core is solved
    /// (`micro_hessian_solve=auto|dense|cg`).
    pub response_solve: micro::ResponseSolve,
    /// Correct the energy and the core gradient for the gradient a relaxation leaves in the
    /// environment (`micro_correct=on|off`).
    pub correct: bool,
    /// Without `micro_tolerance`, the environment is relaxed to this share of the thresholds
    /// of the forces of the run (`micro_tolerance_factor`).
    pub tolerance_factor: f64,
    /// Atoms of the environment within this distance of the model region are moved by the
    /// optimizer with the core (`micro_core_shell`, Angstrom; `None`: none).
    pub core_shell: Option<f64>,
    /// The largest gradient component of the environment for a Hessian of a check of
    /// stability (`micro_check_tolerance`, Hartree/Bohr; `None`: [`MICRO_CHECK_TOLERANCE`], or
    /// the tolerance of the run where that is tighter).
    pub check_tolerance: Option<f64>,
    /// Replicas of the environment combined by a soft minimum (`micro_replicas`,
    /// `micro_replica_temperature`, `micro_replica_start`, `micro_replica_files`,
    /// `micro_replica_amplitude` in Angstrom, `micro_replica_seed`, `micro_replica_md_steps`;
    /// [`micro::replicas`]). The amplitude is kept in Bohr here.
    pub replicas: micro::ReplicaSettings,
}

/// The largest gradient component of the environment (Hartree/Bohr; the RMS two thirds of
/// it) for every Hessian of a check of stability: the mode kill at the convergence of an
/// optimization, the start of an IRC, the minima and saddle points of the mapper, and a
/// vibrational analysis take their Hessians at the same tolerance, so that they judge the same
/// surface (docs/source/oniom.md, the verification of saddle points: the transition-state
/// search of chorismate mutase relaxed its environment to 3e-4, its IRC to 1e-5, and the two
/// saw different curvatures). It is the floor of the relaxations of a lowest level in double
/// precision ([`MICRO_TOLERANCE_FLOOR`]); on chorismate mutase the soft curvature of the core
/// in one minimum of the environment spread by 7e-7 Hartree/(Bohr^2 amu) over three
/// relaxations to it, against a curvature of 4.4e-5.
pub const MICRO_CHECK_TOLERANCE: f64 = MICRO_TOLERANCE_FLOOR;

/// The share of the thresholds of the forces of a run to which its environment is relaxed
/// without `micro_tolerance` (docs/source/oniom.md, the measurements of the microiterations:
/// on chorismate mutase the core gradient of relaxations from five perturbed environments
/// spread by 5.8e-5 Hartree/Bohr at 1, 5.1e-6 at 0.1, and 2.9e-6 at 0.03; with the inertia of
/// the saddle searches kept, its transition state converged at 1 as at 0.1 with fewer
/// gradients, and the networks of the mapper tests change at 0.1, so 1 stays the default and
/// 0.1 is the setting for searches that do not converge).
pub const MICRO_TOLERANCE_FACTOR: f64 = 1.0;

/// The share of [`MICRO_TOLERANCE_FACTOR`] does not take the tolerance of the largest
/// gradient component below this (Hartree/Bohr; the RMS below two thirds of it) unless the
/// thresholds of the run are below it themselves: relaxations with OpenMM in double precision
/// stopped where their energy no longer fell at 3e-6 to 5e-6, and an IRC (thresholds of 1.5e-5)
/// asked for 1.5e-6 took 5000 inner iterations per relaxation.
pub const MICRO_TOLERANCE_FLOOR: f64 = 1e-5;

impl Default for MicroOptions {
    fn default() -> Self {
        let d = micro::MicroOniomSettings::default();
        Self {
            environment_mode_kill: d.inner.mode_kill.enabled,
            mode_kill_every: d.mode_kill_every,
            tolerance: None,
            requested: false,
            core_hessian: d.core_hessian,
            optimizer: d.inner.method.clone(),
            coordinates: d.coordinates,
            predict: d.predict,
            adaptive: d.adaptive,
            guided_curvature: d.inner.mode_kill.guided,
            precondition: d.precondition.as_ref().map(|p| p.model.clone()),
            precondition_shift: d
                .precondition
                .as_ref()
                .map_or(micro::PRECONDITION_SHIFT, |p| p.shift),
            hessian_memory: d.hessian_memory,
            hessian_shell: None,
            response_solve: d.response_solve,
            correct: d.correct,
            tolerance_factor: MICRO_TOLERANCE_FACTOR,
            core_shell: None,
            check_tolerance: None,
            replicas: d.replicas,
        }
    }
}

impl MicroOptions {
    /// The settings of [`micro::RelaxedEnvironment`] for a run whose own convergence is
    /// tested by the forces `criteria` (largest component and root mean square,
    /// Hartree/Bohr): the environment is relaxed to the same thresholds, unless
    /// `micro_tolerance` gave one for its largest gradient component. The methods of the
    /// optimizer registry (`micro_optimizer=NAME`, FIRE after a failed line search) are not in
    /// them, as this layer is below the optimizers: the commands set `inner.methods` to
    /// `opt::micro::REGISTRY_METHODS`, and code that builds the surface itself does the same.
    pub fn settings_for(&self, criteria: (f64, f64)) -> micro::MicroOniomSettings {
        // The share of the thresholds, not below the floor of a lowest level in double
        // precision (where the thresholds of the run are above it).
        let share = |threshold: f64, floor: f64| {
            (self.tolerance_factor * threshold).max(floor.min(threshold))
        };
        let (gradient_tolerance, rms_tolerance) = match self.tolerance {
            Some(given) => (given, f64::INFINITY),
            None => (
                share(criteria.0, MICRO_TOLERANCE_FLOOR),
                share(criteria.1, MICRO_TOLERANCE_FLOOR * 2.0 / 3.0),
            ),
        };
        // The checks of stability: the given tolerance, else MICRO_CHECK_TOLERANCE, neither
        // looser than that of the run (whose relaxations are then already that tight).
        let check = self
            .check_tolerance
            .unwrap_or(MICRO_CHECK_TOLERANCE)
            .min(gradient_tolerance);
        let check_tolerance = Some((check, (check * 2.0 / 3.0).min(rms_tolerance)));
        let d = micro::MicroOniomSettings::default();
        micro::MicroOniomSettings {
            check_tolerance,
            mode_kill_every: self.mode_kill_every,
            core_hessian: self.core_hessian,
            inner: crate::pes::microiter::MicroSettings {
                gradient_tolerance,
                rms_tolerance,
                method: self.optimizer.clone(),
                mode_kill: crate::pes::microiter::ModeKill {
                    enabled: self.environment_mode_kill,
                    guided: self.guided_curvature,
                    ..d.inner.mode_kill
                },
                ..d.inner.clone()
            },
            predict: self.predict,
            coordinates: self.coordinates,
            adaptive: self.adaptive,
            precondition: self
                .precondition
                .as_ref()
                .map(|model| micro::PreconditionSettings {
                    model: model.clone(),
                    shift: self.precondition_shift,
                }),
            hessian_memory: self.hessian_memory,
            hessian_shell: self
                .hessian_shell
                .map(|r| r * crate::domain::Units::ANGSTROM_TO_BOHR),
            response_solve: self.response_solve,
            correct: self.correct,
            replicas: self.replicas.clone(),
            ..d
        }
    }
}

/// An ONIOM calculation as the command line gives it: the regions (innermost first) and
/// the calculators of the lower levels (the highest level is the calculator of `-os`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OniomSpec {
    /// Limit optimization and mapper XYZ trajectory frames to the real high-layer atoms.
    pub traj_high_only: bool,
    /// The regions, innermost first.
    pub regions: Vec<RegionSpec>,
    /// The calculator names of the levels below the highest, from the next one to the
    /// lowest (one per region).
    pub lower_levels: Vec<String>,
    /// The settings of the calculators of those levels (`-oniom_opt low.KEY=VALUE`,
    /// `middle.KEY=VALUE`), in the order of `lower_levels`; a level without an entry has none.
    pub lower_options: Vec<Vec<(String, String)>>,
    /// Where the link atoms sit.
    pub placement: LinkPlacement,
    /// Microiterations of the optimizations (`None`: every atom moves in the optimizer).
    pub micro: Option<MicroOptions>,
    /// Electronic embedding (`None`: mechanical embedding).
    pub embedding: Option<embedding::ElectronicEmbedding>,
}

impl OniomSpec {
    /// The real high-layer atoms in input order for a reduced XYZ trajectory.
    pub fn trajectory_atoms(&self) -> Option<Vec<AtomId>> {
        self.traj_high_only.then(|| self.high_atoms())
    }

    /// Atoms of the real system in the highest model region, in input order.
    pub fn high_atoms(&self) -> Vec<AtomId> {
        let mut atoms = self
            .regions
            .first()
            .map_or_else(Vec::new, |region| region.atoms.clone());
        atoms.sort_unstable();
        atoms.dedup();
        atoms
    }

    /// The settings of the calculator of lower level `k` (0: the level below the highest).
    pub fn options_of_lower_level(&self, k: usize) -> &[(String, String)] {
        self.lower_options.get(k).map_or(&[], Vec::as_slice)
    }

    /// The regions with their link atoms in `sys`.
    pub fn resolve(&self, sys: &MolecularSystem) -> MoptResult<Vec<Region>> {
        if self.regions.len() != self.lower_levels.len() {
            return Err(MoptError::config(format!(
                "ONIOM: {} region(s) and {} lower level(s)",
                self.regions.len(),
                self.lower_levels.len()
            )));
        }
        self.regions
            .iter()
            .enumerate()
            .map(|(k, spec)| {
                if spec.atoms.is_empty() {
                    return Err(MoptError::config(format!(
                        "ONIOM region {} has no atoms",
                        k + 1
                    )));
                }
                let atoms = spec
                    .atoms
                    .iter()
                    .map(|a| a.check(sys.n_atoms()))
                    .collect::<MoptResult<Vec<_>>>()?;
                let pairs = match &spec.links {
                    LinkSpec::None => Vec::new(),
                    LinkSpec::Auto => crossing_bonds(sys, &atoms, BOND_SCALE),
                    LinkSpec::Given(outside) => {
                        for q2 in outside {
                            q2.check(sys.n_atoms())?;
                            if atoms.contains(q2) {
                                return Err(MoptError::config(format!(
                                    "ONIOM region {}: link atom {} is in the region itself",
                                    k + 1,
                                    q2.one_based()
                                )));
                            }
                        }
                        pair_with_nearest(sys, &atoms, outside)?
                    }
                };
                let links = pairs
                    .into_iter()
                    .map(|(q1, q2)| Link::new(sys, q1, q2, self.placement))
                    .collect::<Vec<_>>();
                let mut elements: Vec<Element> = atoms.iter().map(|a| sys.element(*a)).collect();
                elements.extend(std::iter::repeat_n(Element::H, links.len()));
                let spin_charge = match spec.spin_charge {
                    Some(sc) => sc,
                    None => layout::model_state(&elements, sys.spin_charge),
                };
                Ok(Region {
                    atoms,
                    links,
                    spin_charge,
                    placement: self.placement,
                    follows: spec.spin_charge.is_none().then_some(sys.spin_charge),
                })
            })
            .collect()
    }
}
