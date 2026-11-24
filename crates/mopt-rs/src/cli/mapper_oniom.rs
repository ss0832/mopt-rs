//! `run_mapper` on an ONIOM surface: the network holds the cores (the atoms the optimizer
//! moves: the model region and the outer atoms of its links), every calculation runs on the
//! relaxed surface of the core with ONIOM microiterations (the environment relaxed to a
//! minimum before every evaluation), and the files of the nodes and of their paths hold the
//! whole system with the environment of the job that found them
//! (`docs/source/mapper_design.md`, ONIOM).
//!
//! - **Identity** (`identity_scope`): `high` (the default) tells nodes apart by their cores,
//!   whatever their environments; `active_site:R` adds the atoms of the environment within
//!   `R` Angstrom of the core in the first input, which the optimizer then moves as well.
//!   Structures that differ in the environment alone are one node: its other environments
//!   are listed on it with their energies, and the node is the one of the lowest energy.
//! - **The whole structures**: a job starts from the environment of its node (its whole
//!   structure), and at its end the environment its calculator relaxed at each structure it
//!   found (`Calculator::relaxed_environment`, which the journal records, so that a restart
//!   gives the same) makes the whole structure of that structure.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{Element, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::pes::calculator::Calculator;
use crate::pes::oniom::micro::RelaxedEnvironment;
use crate::workflow::mapper::{Configuration, JobOutcome};

/// The whole system of a network of ONIOM cores.
#[derive(Debug, Clone)]
pub struct WholeSystem {
    /// The structure the cores are part of (the first input; the atoms of the environment
    /// that are held stay where it has them).
    template: MolecularSystem,
    /// The atoms of the core (0-based, ascending): the atoms of the structures of the network
    /// in their order.
    core: Vec<usize>,
    /// Selected trajectory atoms in core numbering.
    pub(super) trajectory_atoms: Option<Vec<crate::domain::AtomId>>,
    /// The atoms of the environment that are relaxed (0-based, ascending).
    environment: Vec<usize>,
}

impl WholeSystem {
    /// The whole system of the relaxed surface `micro`, the cores taken from `template`.
    pub fn of(micro: &RelaxedEnvironment, template: MolecularSystem) -> Self {
        let split = micro.split();
        Self {
            template,
            core: split.core.clone(),
            trajectory_atoms: None,
            environment: split.environment.clone(),
        }
    }

    /// The elements of the whole system.
    pub fn elements(&self) -> Vec<Element> {
        self.template.elements.clone()
    }

    /// The coordinates of the relaxed environment of the whole structure `whole` (flat,
    /// Bohr).
    pub fn environment_of(&self, whole: &MolecularSystem) -> Vec<f64> {
        let x = whole.coords_flat_bohr();
        let mut out = Vec::with_capacity(3 * self.environment.len());
        for &a in &self.environment {
            for k in 0..3 {
                out.push(x[3 * a + k]);
            }
        }
        out
    }

    /// The whole structure of the core `core` with the relaxed environment `environment`
    /// (flat, Bohr; the held atoms where the template has them).
    pub fn assemble(
        &self,
        core: &MolecularSystem,
        environment: &[f64],
    ) -> MoptResult<MolecularSystem> {
        let xc = core.coords_flat_bohr();
        if xc.nrows() != 3 * self.core.len() || environment.len() != 3 * self.environment.len() {
            return Err(MoptError::calc(format!(
                "run_mapper: a core of {} atoms and an environment of {} for a whole system with \
                 a core of {} and an environment of {}",
                xc.nrows() / 3,
                environment.len() / 3,
                self.core.len(),
                self.environment.len()
            )));
        }
        let mut x = self.template.coords_flat_bohr();
        for (k, &a) in self.core.iter().enumerate() {
            for d in 0..3 {
                x[3 * a + d] = xc[3 * k + d];
            }
        }
        for (k, &a) in self.environment.iter().enumerate() {
            for d in 0..3 {
                x[3 * a + d] = environment[3 * k + d];
            }
        }
        let mut whole = MolecularSystem::from_flat_bohr(
            self.template.elements.clone(),
            &x,
            core.spin_charge,
            core.cell.clone(),
        )?;
        whole.comment.clone_from(&core.comment);
        Ok(whole)
    }

    /// The whole structure of `point` as the calculator `calc` relaxed its environment there,
    /// where it did.
    pub fn whole_at(
        &self,
        calc: &dyn Calculator,
        point: &Configuration,
    ) -> MoptResult<Option<Configuration>> {
        let Some(core) = point.structure() else {
            return Ok(None);
        };
        match calc.relaxed_environment(core) {
            Some(env) => Ok(Some(Configuration::Structure(self.assemble(core, &env)?))),
            None => Ok(None),
        }
    }

    /// Give `calc` the environment of the whole structure of `node` as the start of the
    /// relaxations near its core.
    pub fn start_from(&self, calc: &dyn Calculator, node: &crate::workflow::mapper::Node) {
        let (Some(core), Some(whole)) = (
            node.system.structure(),
            node.whole.as_ref().and_then(Configuration::structure),
        ) else {
            return;
        };
        if whole.n_atoms() == self.template.n_atoms() {
            calc.recall_environment(core, &self.environment_of(whole));
        }
    }

    /// The whole structures of the structures of `outcome` (the stationary points, the frames
    /// of the IRCs and of the minimizations of their ends, the paths of the path peaks) as
    /// the calculator `calc` of the job relaxed their environments.
    pub fn complete(&self, calc: &dyn Calculator, outcome: &mut JobOutcome) -> MoptResult<()> {
        let mut points: Vec<Configuration> = Vec::new();
        for r in &outcome.reactions {
            points.push(r.ts.system.clone());
            points.extend(r.ends.iter().flatten().map(|e| e.system.clone()));
            for b in &r.branches {
                points.extend(b.frames.iter().cloned());
                points.push(b.last.clone());
                points.extend(b.settle.iter().map(|p| p.system.clone()));
            }
            for (_, found) in &r.bifurcation_ends {
                points.extend(found.iter().map(|e| e.system.clone()));
            }
        }
        for s in &outcome.saddles {
            points.push(s.point.system.clone());
        }
        for c in &outcome.crossings {
            points.push(c.point.system.clone());
            points.extend(c.ends.iter().flatten().map(|e| e.system.clone()));
        }
        for p in &outcome.peaks {
            points.push(p.point.system.clone());
            points.extend(p.ends.iter().flatten().map(|e| e.system.clone()));
            points.extend(p.path.iter().map(|q| q.system.clone()));
            for side in &p.settles {
                points.extend(side.iter().map(|q| q.system.clone()));
            }
        }
        for point in points {
            if outcome.whole_of(&point).is_some() {
                continue;
            }
            if let Some(whole) = self.whole_at(calc, &point)? {
                outcome.wholes.push((point, whole));
            }
        }
        Ok(())
    }
}

/// Refuse, by name and with the reason, what a network on an ONIOM surface does not take yet:
/// what has not been run on the relaxed surface of a core.
pub fn refuse_with_oniom(config: &super::mapper_settings::MapperConfig) -> MoptResult<()> {
    let refuse = |what: &str, why: &str| {
        Err(MoptError::unsupported(format!(
            "run_mapper on an ONIOM surface: {what} is not available yet ({why})"
        )))
    };
    let s = &config.settings;
    if config.remap.is_some() || config.single_points.is_some() {
        return refuse(
            "-remap and -single_points",
            "a network of cores taken to another surface needs the whole structures on that \
             surface, which is not built",
        );
    }
    if s.method.eq_ignore_ascii_case("nanoreactor") {
        return refuse(
            "method=nanoreactor",
            "its dynamics would move the core on the relaxed surface, and the dynamics of an \
             ONIOM surface has no microiterations",
        );
    }
    if s.entrance_structures > 0 {
        return refuse(
            "entrance_structures",
            "the structures of an entrance channel are made of separated molecules of the \
             whole system",
        );
    }
    if !s.states.is_empty() {
        return refuse(
            "states (several electronic states)",
            "the crossing searches have not been run on the relaxed surface of a core",
        );
    }
    if s.ts_conformers {
        return refuse(
            "ts_conformers",
            "the conformer search of a transition state has not been run on the relaxed \
             surface of a core",
        );
    }
    Ok(())
}

/// Which atoms decide whether two structures of an ONIOM network are the same node
/// (`identity_scope`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdentityScope {
    /// The core (the high layer with the outer atoms of its links).
    High,
    /// The core and the atoms of the environment within this distance of it (Angstrom) in the
    /// first input.
    ActiveSite(f64),
}

impl IdentityScope {
    /// `high`, or `active_site:R` (Angstrom).
    pub fn parse(text: &str) -> MoptResult<Self> {
        let t = text.trim().to_ascii_lowercase();
        if t == "high" {
            return Ok(Self::High);
        }
        if let Some(r) = t.strip_prefix("active_site:") {
            return match r.trim().parse::<f64>() {
                Ok(r) if r.is_finite() && r > 0.0 => Ok(Self::ActiveSite(r)),
                _ => Err(MoptError::parse(format!(
                    "identity_scope={text}: active_site:R with a distance R above 0 (Angstrom)"
                ))),
            };
        }
        if t == "all" {
            return Err(MoptError::unsupported(
                "identity_scope=all: every atom deciding the identity would make every \
                 environment a node of its own, which the microiterations of every calculation \
                 do not keep apart (a relaxed environment is a function of its core); use \
                 high or active_site:R",
            ));
        }
        Err(MoptError::parse(format!(
            "identity_scope={text}: high (the default) or active_site:R (Angstrom)"
        )))
    }

    /// The atoms of the environment of `whole` (outside `core`, 0-based) that this scope
    /// adds to the core.
    pub fn added_atoms(self, whole: &MolecularSystem, core: &[usize]) -> Vec<usize> {
        let Self::ActiveSite(r) = self else {
            return Vec::new();
        };
        let limit = r * Units::ANGSTROM_TO_BOHR;
        (0..whole.n_atoms())
            .filter(|a| !core.contains(a))
            .filter(|&a| {
                core.iter().any(|&c| {
                    whole
                        .distance(crate::domain::AtomId::new(a), crate::domain::AtomId::new(c))
                        .value()
                        <= limit
                })
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_scopes_are_read_and_all_is_refused_with_the_reason() {
        assert_eq!(IdentityScope::parse("high").unwrap(), IdentityScope::High);
        assert_eq!(
            IdentityScope::parse("active_site:4.5").unwrap(),
            IdentityScope::ActiveSite(4.5)
        );
        assert!(IdentityScope::parse("active_site:-1").is_err());
        let e = IdentityScope::parse("all").unwrap_err().to_string();
        assert!(e.contains("microiterations"), "{e}");
        assert!(IdentityScope::parse("core").is_err());
    }
}
