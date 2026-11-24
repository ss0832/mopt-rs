//! The saddle direction of a transition state in internal coordinates: the reaction
//! coordinate that the conformers of a transition state keep, and its reactive bonds.
//!
//! The imaginary mode (Cartesian displacements) is written in the primitive internal
//! coordinates of the transition state, `w = B dx`; the coordinates of the atoms the mode moves
//! with at least a share of its largest component form `c(x) = sum_k w_k (q_k(x) - q_k(TS))`
//! (normalized). The bonds with at least half the largest bond component are the reactive
//! bonds, held at their lengths on their own.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::coord::internal::{Primitive, PrimitiveOptions, PrimitiveSet};
use crate::domain::{AtomId, EvalOutput, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::pes::constraint::{Constraint, ConstraintKind};
use crate::pes::term::{PesTerm, TermContext};

/// The reaction coordinate `c(x) = sum_k w_k (q_k(x) - q_k(TS))`.
pub(super) struct ReactionCoordinate {
    primitives: Vec<Primitive>,
    weights: Vec<f64>,
    reference: Vec<f64>,
}

impl ReactionCoordinate {
    fn value_and_gradient(&self, sys: &MolecularSystem) -> (f64, Col<f64>) {
        let mut value = 0.0;
        let mut grad = Col::zeros(sys.dof());
        for ((p, w), q0) in self
            .primitives
            .iter()
            .zip(&self.weights)
            .zip(&self.reference)
        {
            let (q, g) = p.value_and_gradient(sys);
            value += w * p.difference(q, *q0);
            for (gi, x) in grad.iter_mut().zip(g.iter()) {
                *gi += w * x;
            }
        }
        (value, grad)
    }
}

impl PesTerm for ReactionCoordinate {
    fn name(&self) -> &str {
        "reaction_coordinate"
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let (value, gradient) = self.value_and_gradient(sys);
        // Central differences of the analytic gradient: the primitives give no second
        // derivatives of their combination here.
        let hessian = if need_hessian {
            Some(crate::pes::fd_hessian::structure_hessian(
                sys,
                1e-4,
                false,
                |moved| Ok(self.value_and_gradient(moved).1),
            )?)
        } else {
            None
        };
        Ok(EvalOutput {
            energy: Quantity::new(value),
            gradient_bohr: gradient,
            hessian_bohr: hessian,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }

    /// A combination of primitive internal coordinates; its Hessian comes from differences.
    fn frame(&self) -> crate::pes::term::Frame {
        crate::pes::term::Frame::Invariant
    }

    fn history(&self) -> crate::pes::term::History {
        crate::pes::term::History::Static
    }
}

/// Atoms whose displacement in the Cartesian mode `mode` is at least `share` of the largest.
fn reacting_atoms(mode: &Col<f64>, share: f64) -> Vec<AtomId> {
    let n = mode.nrows() / 3;
    let size: Vec<f64> = (0..n)
        .map(|a| (0..3).map(|k| mode[3 * a + k].powi(2)).sum::<f64>().sqrt())
        .collect();
    let top = size.iter().copied().fold(0.0_f64, f64::max);
    (0..n)
        .filter(|&a| top > 0.0 && size[a] >= share * top)
        .map(AtomId::new)
        .collect()
}

/// The atoms of a primitive internal coordinate.
fn primitive_atoms(p: &Primitive) -> Vec<AtomId> {
    match p {
        Primitive::Bond(a) => a.to_vec(),
        Primitive::Angle(a) => a.to_vec(),
        Primitive::LinearBend { atoms, .. } => atoms.to_vec(),
        Primitive::Dihedral(a) => a.to_vec(),
    }
}

/// The reaction coordinate of the transition state `ts` along the Cartesian mode `mode`, with
/// the labels and weights of its internal coordinates: the primitive internal coordinates
/// (bonds of covalent radii times `bond_scale`) of the atoms the mode moves, with at least
/// `share` of its largest component, except the bonds in `held`.
pub(super) fn reaction_coordinate(
    ts: &MolecularSystem,
    mode: &Col<f64>,
    bond_scale: f64,
    share: f64,
    held: &[[AtomId; 2]],
) -> MoptResult<Option<(ReactionCoordinate, Vec<(String, f64)>)>> {
    let set = PrimitiveSet::generate(
        ts,
        &PrimitiveOptions {
            bond_scale,
            connect_fragments: true,
        },
    )?;
    let b = set.wilson_b(ts);
    let w: Vec<f64> = (0..b.nrows())
        .map(|i| (0..b.ncols()).map(|j| b[(i, j)] * mode[j]).sum())
        .collect();
    let top = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    if top <= 0.0 {
        return Err(MoptError::coord(
            "the imaginary mode moves no internal coordinate of the transition state",
        ));
    }
    // Bonds held on their own are left out, so that no constraint repeats another.
    let is_held = |p: &Primitive| match p {
        Primitive::Bond([a, c]) => held
            .iter()
            .any(|h| (h[0] == *a && h[1] == *c) || (h[0] == *c && h[1] == *a)),
        _ => false,
    };
    // Coordinates with an atom that the mode hardly moves are left out: the torsion of a
    // group next to the reaction (an ethyl group at a reacting atom) changes along the mode
    // only because its frame turns, and holding it would hold the conformation of the group.
    let moving = reacting_atoms(mode, share);
    let reacting = |p: &Primitive| primitive_atoms(p).iter().all(|a| moving.contains(a));
    let kept: Vec<usize> = (0..w.len())
        .filter(|&i| {
            let p = &set.primitives[i];
            w[i].abs() >= share * top && !is_held(p) && reacting(p)
        })
        .collect();
    let norm = kept.iter().map(|&i| w[i] * w[i]).sum::<f64>().sqrt();
    if kept.is_empty() || norm < 1e-3 * top {
        return Ok(None);
    }
    let values = set.values(ts);
    let coordinate = ReactionCoordinate {
        primitives: kept.iter().map(|&i| set.primitives[i].clone()).collect(),
        weights: kept.iter().map(|&i| w[i] / norm).collect(),
        reference: kept.iter().map(|&i| values[i]).collect(),
    };
    let labels = kept
        .iter()
        .map(|&i| (set.primitives[i].label(), w[i] / norm))
        .collect();
    Ok(Some((coordinate, labels)))
}

/// The reactive bonds of the mode (bonds with at least half the largest component among the
/// bonds), held at their lengths in the transition state, with their labels.
pub(super) fn reactive_bonds(
    ts: &MolecularSystem,
    mode: &Col<f64>,
    bond_scale: f64,
) -> MoptResult<Vec<(Constraint, String)>> {
    let set = PrimitiveSet::generate(
        ts,
        &PrimitiveOptions {
            bond_scale,
            connect_fragments: true,
        },
    )?;
    let b = set.wilson_b(ts);
    let values = set.values(ts);
    let bonds: Vec<(usize, [AtomId; 2], f64)> = set
        .primitives
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match p {
            Primitive::Bond(ab) => {
                let w: f64 = (0..b.ncols()).map(|j| b[(i, j)] * mode[j]).sum();
                Some((i, *ab, w.abs()))
            }
            _ => None,
        })
        .collect();
    let top = bonds.iter().fold(0.0_f64, |m, x| m.max(x.2));
    Ok(bonds
        .into_iter()
        .filter(|x| top > 0.0 && x.2 >= 0.5 * top)
        .map(|(i, [a, c], _)| {
            (
                Constraint {
                    kind: ConstraintKind::Distance(vec![a], vec![c]),
                    target: Some(values[i]),
                    anchor: None,
                },
                set.primitives[i].label(),
            )
        })
        .collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_atoms_a_mode_hardly_moves_are_left_out() {
        // Atom 0 moves 1, atom 1 moves 0.5, atom 2 moves 0.05.
        let mode = Col::from_fn(9, |k| match k {
            0 => 0.6,
            1 => 0.8,
            3 => 0.5,
            8 => 0.05,
            _ => 0.0,
        });
        assert_eq!(
            reacting_atoms(&mode, 0.1),
            vec![AtomId::new(0), AtomId::new(1)]
        );
        assert_eq!(reacting_atoms(&mode, 0.01).len(), 3);
        assert!(reacting_atoms(&Col::zeros(9), 0.1).is_empty());
    }
}
