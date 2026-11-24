//! What a run of [`super::MolecularDynamics`] holds fixed at its start: held atoms,
//! constraints, the checks of a barostat, and the conserved motions.
//!
//! MultiOptPy: `moleculardynamics.py:MD.md` (`-fix`, `-cc`, `-pc` at the start).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::MdParts;
use crate::domain::PeriodicCell;
use crate::error::{MoptError, MoptResult};
use crate::md::constraints::Rattle;
use crate::md::settings::MdSettings;
use crate::md::space::{MdSpace, RigidMotion};
use crate::pes::constraint::{Constraint, ConstraintKind};

/// The atoms of `-fix` held in `space`; the other constraints, and those that wait for the
/// first Hessian.
pub(super) fn split_constraints(
    constraints: &[Constraint],
    space: &mut MdSpace,
) -> MoptResult<(Vec<Constraint>, Vec<Constraint>)> {
    let mut others = Vec::new();
    let mut deferred = Vec::new();
    for c in constraints {
        match &c.kind {
            ConstraintKind::Frozen(atoms) => {
                if !space.is_atomic() {
                    return Err(MoptError::config("-fix needs atoms"));
                }
                let dofs: Vec<usize> = atoms
                    .iter()
                    .flat_map(|a| (0..3).map(|k| a.dof(k)))
                    .collect();
                space.freeze(&dofs)?;
            }
            ConstraintKind::EigenVector(_) => deferred.push(c.clone()),
            _ => others.push(c.clone()),
        }
    }
    Ok((others, deferred))
}

/// The barostat needs a cell periodic in three directions, and the virial of constraint
/// forces is not part of the pressure.
pub(super) fn check_barostat(
    parts: &MdParts,
    cell: Option<&PeriodicCell>,
    rattle: &Rattle,
    space: &MdSpace,
    settings: &MdSettings,
) -> MoptResult<()> {
    if parts.barostat.is_none() {
        return Ok(());
    }
    if !cell.is_some_and(|c| c.periodic_axes.iter().all(|p| *p)) {
        return Err(MoptError::config(
            "a barostat needs a cell periodic in all three directions",
        ));
    }
    if !rattle.is_empty() || settings.freeze.is_some() || space.has_held() {
        return Err(MoptError::unsupported(
            "constant pressure with constraints, held atoms, or frozen modes is not available \
             (the virial of the constraint forces is not part of the pressure)",
        ));
    }
    Ok(())
}

/// The velocity exchanges of a reverse non-equilibrium run need a periodic cell, no
/// thermostat or barostat (the exchanged heat is the only flux), and no constraints.
pub(super) fn check_exchange(
    parts: &MdParts,
    cell: Option<&PeriodicCell>,
    rattle: &Rattle,
    settings: &MdSettings,
) -> MoptResult<()> {
    let Some(e) = settings.exchange else {
        return Ok(());
    };
    if e.slabs < 6 || e.slabs % 2 != 0 || e.axis > 2 || e.every == 0 {
        return Err(MoptError::config(
            "velocity exchanges need an even number of slabs (at least six), an axis 0, 1, \
             or 2, and an interval of at least one step",
        ));
    }
    if !cell.is_some_and(|c| c.periodic_axes[e.axis]) {
        return Err(MoptError::config(
            "velocity exchanges need a cell periodic along the axis of the slabs",
        ));
    }
    if parts.thermostat.name() != "none" || parts.barostat.is_some() || !rattle.is_empty() {
        return Err(MoptError::unsupported(
            "velocity exchanges (reverse non-equilibrium) run without a thermostat, a barostat, \
             or constraints: the exchanged heat must be the only flux (-mt nve)",
        ));
    }
    Ok(())
}

/// The conserved motions of the run: those asked for, unless atoms are held, constraints
/// pin the structure, the thermostat acts on single particles, or the cell is periodic (no
/// rotations).
pub(super) fn conserved_motion(
    settings: &MdSettings,
    space: &MdSpace,
    rattle: &Rattle,
    parts: &MdParts,
    cell: Option<&PeriodicCell>,
) -> RigidMotion {
    if !space.is_atomic()
        || space.has_held()
        || rattle.breaks_rigid_body()
        || !parts.thermostat.conserves_momentum()
    {
        return RigidMotion::None;
    }
    match (settings.rigid, cell.is_some()) {
        (RigidMotion::TranslationRotation, true) => RigidMotion::Translation,
        (other, _) => other,
    }
}
