//! A preconditioner of the relaxations of the environment: the environment block of a model
//! Hessian of the free atoms (core and environment), shifted to be positive definite and
//! factorized once, and kept while the free atoms stay near the structure it was built at.
//!
//! In an environment of solvent molecules the soft intermolecular modes lie among stiff bonds
//! and angles. A search for the lowest curvature from Hessian-vector products alone sees the
//! stiff modes first (the spread of the spectrum sets its rate); with the inverse of a model
//! that has the stiff terms the search becomes a preconditioned inverse iteration, whose rate
//! is set by how well the model follows the Hessian (A. V. Knyazev, SIAM J. Sci. Comput. 2001,
//! 23, 517). The same inverse serves as the initial matrix of the limited-memory BFGS of the
//! relaxation (J. Nocedal and S. J. Wright, Numerical Optimization, section 7.2), as model
//! Hessians precondition the optimization of materials (D. Packwood et al., J. Chem. Phys.
//! 2016, 144, 164109).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::linalg::solvers::{Llt, Solve};
use faer::{Col, Mat};

use super::{assemble, CoreSplit};
use crate::domain::{AtomId, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::model_hessian::resolve;

/// Settings of the preconditioner (`-oniom_opt micro_precondition=NAME`,
/// `micro_precondition_shift=S`).
#[derive(Debug, Clone, PartialEq)]
pub struct PreconditionSettings {
    /// The model Hessian (a name of `-modelhess`).
    pub model: String,
    /// Added to the diagonal (Hartree/Bohr^2), so that the factor is positive definite where
    /// the model has no stiffness (a free rotation, an atom without neighbours).
    pub shift: f64,
}

/// The default shift of the diagonal (Hartree/Bohr^2).
pub const PRECONDITION_SHIFT: f64 = 1e-3;

/// The model of `micro_precondition` without a name: of `lindh`, `lindh2007d3`, and the shifts
/// 1e-4 to 1e-2, the one whose relaxations took the fewest gradients on the TS optimization of
/// the Claisen rearrangement in water (docs/source/oniom.md, the microiterations).
pub const PRECONDITION_MODEL: &str = "lindh2007d3";

/// The largest environment (variables) with a preconditioner: its factor is a dense matrix
/// (2916 variables of an enzyme: the model 7.7 s, the factor 1.0 s, 68 MB a matrix).
pub const PRECONDITION_LARGEST: usize = 6000;

/// The factor is built again when a free atom moved this far from where it was built (Bohr).
const REBUILD_BOHR: f64 = 0.3;

/// Tries of the factorization, each with ten times the shift of the one before.
const SHIFT_TRIES: usize = 4;

/// The factorized model of the environment block.
pub(super) struct EnvironmentPreconditioner {
    /// The free atoms (core and environment) where the model was built (flat, Bohr).
    built_at: Col<f64>,
    factor: Llt<f64>,
    /// Variables of the environment.
    dim: usize,
    /// Fixed atoms the model was built with (the neighbours of the free atoms).
    fixed_neighbours: usize,
    /// Seconds of the model and of the factorization.
    seconds: (f64, f64),
}

/// The free atoms of the split, ascending.
fn free_atoms(split: &CoreSplit) -> Vec<usize> {
    let mut free: Vec<usize> = split
        .core
        .iter()
        .chain(&split.environment)
        .copied()
        .collect();
    free.sort_unstable();
    free
}

/// Coordinates of `atoms` of `sys` (flat, Bohr).
fn coordinates_of(sys: &MolecularSystem, atoms: &[usize]) -> Col<f64> {
    let flat = sys.coords_flat_bohr();
    Col::from_fn(3 * atoms.len(), |k| flat[3 * atoms[k / 3] + k % 3])
}

/// Fixed atoms within this distance (Bohr, 4 Angstrom) of an atom of the environment are
/// part of the model of the preconditioner: their bonds, angles, and torsions with the
/// environment hold it as they hold it on the surface (a model of the free atoms alone has
/// none of them, and an atom bonded to the fixed frame looks free).
const FIXED_NEIGHBOURS_BOHR: f64 = 7.559;

/// The fixed atoms of `split` within [`FIXED_NEIGHBOURS_BOHR`] of an atom of its environment
/// in `full` (minimum images in a cell), ascending.
fn fixed_neighbours(full: &MolecularSystem, split: &CoreSplit) -> Vec<usize> {
    if split.fixed.is_empty() {
        return Vec::new();
    }
    let lists =
        crate::pes::native::neighbors::neighbor_list(full, Quantity::new(FIXED_NEIGHBOURS_BOHR));
    let mut fixed = vec![false; full.n_atoms()];
    for &a in &split.fixed {
        if let Some(f) = fixed.get_mut(a) {
            *f = true;
        }
    }
    let mut near = vec![false; full.n_atoms()];
    for &a in &split.environment {
        if let Some(list) = lists.get(a) {
            for n in list {
                let b = n.atom.index();
                if fixed.get(b).copied().unwrap_or(false) {
                    near[b] = true;
                }
            }
        }
    }
    (0..full.n_atoms()).filter(|&a| near[a]).collect()
}

impl EnvironmentPreconditioner {
    /// The model `settings.model` at the core `core` with the environment `env`, built on the
    /// free atoms and the fixed atoms next to the environment, and its environment block
    /// factorized.
    pub(super) fn build(
        settings: &PreconditionSettings,
        template: &MolecularSystem,
        split: &CoreSplit,
        core: &Col<f64>,
        env: &[f64],
    ) -> MoptResult<Self> {
        let context = |e: MoptError| {
            e.context(format!(
                "ONIOM microiterations: the preconditioner of the environment ({})",
                settings.model
            ))
        };
        let started = std::time::Instant::now();
        let model = resolve(&settings.model).map_err(context)?;
        let full = assemble(template, split, core, env)?;
        let free = free_atoms(split);
        let neighbours = fixed_neighbours(&full, split);
        let mut atoms: Vec<usize> = free.iter().chain(&neighbours).copied().collect();
        atoms.sort_unstable();
        let ids: Vec<AtomId> = atoms.iter().map(|&a| AtomId::new(a)).collect();
        let sub = full.subsystem(&ids, full.spin_charge)?;
        let h = model
            .hessian(&sub, &Col::zeros(sub.dof()))
            .map_err(context)?;
        let model_seconds = started.elapsed().as_secs_f64();
        // The rows of the environment in the atoms of the model.
        let rows: Vec<usize> = split
            .environment
            .iter()
            .filter_map(|a| atoms.binary_search(a).ok())
            .collect();
        if rows.len() != split.environment.len() {
            return Err(MoptError::calc(
                "ONIOM microiterations: an atom of the environment is not among the free atoms",
            ));
        }
        let m = 3 * rows.len();
        let block = |shift: f64| {
            Mat::from_fn(m, m, |i, j| {
                let (a, b) = (3 * rows[i / 3] + i % 3, 3 * rows[j / 3] + j % 3);
                0.5 * (h[(a, b)] + h[(b, a)]) + if i == j { shift } else { 0.0 }
            })
        };
        let factor_started = std::time::Instant::now();
        let mut shift = settings.shift.max(f64::MIN_POSITIVE);
        for _ in 0..SHIFT_TRIES {
            if let Ok(factor) = block(shift).llt(faer::Side::Lower) {
                return Ok(Self {
                    built_at: coordinates_of(&full, &free),
                    factor,
                    dim: m,
                    fixed_neighbours: neighbours.len(),
                    seconds: (model_seconds, factor_started.elapsed().as_secs_f64()),
                });
            }
            shift *= 10.0;
        }
        Err(context(MoptError::linalg(format!(
            "the model block is not positive definite even with {shift:.1e} on its diagonal"
        ))))
    }

    /// What the log says of the build, after the number of variables: the fixed neighbours
    /// in the model and the time of the model and of the factorization.
    pub(super) fn describe(&self) -> String {
        format!(
            ", with {} fixed neighbour atom(s); model {:.2} s, factor {:.2} s",
            self.fixed_neighbours, self.seconds.0, self.seconds.1
        )
    }

    /// Whether the free atoms of `core` and `env` are near those the model was built at.
    pub(super) fn is_near(
        &self,
        template: &MolecularSystem,
        split: &CoreSplit,
        core: &Col<f64>,
        env: &[f64],
    ) -> bool {
        let Ok(full) = assemble(template, split, core, env) else {
            return false;
        };
        let now = coordinates_of(&full, &free_atoms(split));
        now.nrows() == self.built_at.nrows()
            && now
                .iter()
                .zip(self.built_at.iter())
                .all(|(a, b)| (a - b).abs() < REBUILD_BOHR)
    }

    /// `P^-1 v`, or `None` for a vector of another length.
    pub(super) fn apply(&self, v: &Col<f64>) -> Option<Col<f64>> {
        if v.nrows() != self.dim {
            return None;
        }
        let rhs = Mat::from_fn(self.dim, 1, |i, _| v[i]);
        let x = self.factor.solve(rhs.as_ref());
        let out = Col::from_fn(self.dim, |i| x[(i, 0)]);
        let finite = out.iter().all(|value| value.is_finite());
        finite.then_some(out)
    }
}
