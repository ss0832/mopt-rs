//! New candidate structures by crossing conformers in internal coordinates.
//!
//! The genetic crossing of the iMTD-GC conformer search (Pracht, Bohle, Grimme, Phys. Chem.
//! Chem. Phys. 2020, 22, 7169), implemented from its description: with the Z-matrix of a
//! reference conformer (the lowest one), every ordered pair `(i, j)` of other conformers gives
//! the structure `Z_ref + Z_i - Z_j`, the difference between conformers `i` and `j` (mostly
//! of their dihedral angles) carried over to the reference. Structural motifs that occur in
//! several conformers are thus combined. A candidate is kept when its bond graph is the one
//! of the reference (no atoms that clash into new bonds, no ring opened) and it is not within
//! `rmsd` of the reference, of a parent, or of a candidate kept before; at most `max` pairs
//! are tried, in an order shuffled by the seeded generator of the caller. The candidates are
//! optimized by the caller.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::coord::graph::{whole_molecules, MolecularGraph};
use crate::coord::invariant_rmsd::direct_rmsd;
use crate::coord::zmatrix::{transfer, ZMatrix};
use crate::core_infra::rng::Rng;
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;

/// Settings of the crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossoverSettings {
    /// Largest number of pairs tried.
    pub max: usize,
    /// Candidates closer than this (Bohr) to the reference, a parent, or one another are
    /// left out.
    pub rmsd: f64,
    /// Scale of the covalent radii of the bonds of the graph check.
    pub bond_scale: f64,
}

impl Default for CrossoverSettings {
    /// 300 pairs and 0.25 Angstrom, the values of the example output of CREST's
    /// documentation; MultiOptPy's bond scale 1.1.
    fn default() -> Self {
        Self {
            max: 300,
            rmsd: 0.25 * Units::ANGSTROM_TO_BOHR,
            bond_scale: crate::coord::graph::MULTIOPTPY_BOND_SCALE,
        }
    }
}

/// Candidates from `reference` and `parents` (structures of the same atoms in the same
/// numbering, with the bond graph of the reference).
pub fn crossover_candidates(
    reference: &MolecularSystem,
    parents: &[MolecularSystem],
    settings: &CrossoverSettings,
    rng: &mut Rng,
) -> MoptResult<Vec<MolecularSystem>> {
    let graph = MolecularGraph::from_system(reference, settings.bond_scale);
    let whole_ref = whole_molecules(reference, &graph)?;
    let zmat = ZMatrix::from_graph(&graph, &whole_ref)?;
    let base = zmat.values(&whole_ref);
    let mut wholes = Vec::with_capacity(parents.len());
    for p in parents {
        wholes.push(whole_molecules(p, &graph)?);
    }
    let values: Vec<_> = wholes.iter().map(|p| zmat.values(p)).collect();
    let mut pairs: Vec<(usize, usize)> = (0..values.len())
        .flat_map(|i| {
            (0..values.len())
                .filter(move |&j| j != i)
                .map(move |j| (i, j))
        })
        .collect();
    rng.shuffle(&mut pairs);
    pairs.truncate(settings.max);
    let mut kept: Vec<MolecularSystem> = Vec::new();
    for (i, j) in pairs {
        let crossed = transfer(&base, &values[j], &values[i]);
        let Ok(candidate) = zmat.to_system(&crossed, &whole_ref) else {
            continue;
        };
        if MolecularGraph::from_system(&candidate, settings.bond_scale) != graph {
            continue;
        }
        let near =
            |s: &MolecularSystem| direct_rmsd(&candidate, s).is_ok_and(|d| d < settings.rmsd);
        if near(&whole_ref) || wholes.iter().any(near) || kept.iter().any(near) {
            continue;
        }
        kept.push(candidate);
    }
    Ok(kept)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coord::graph::MULTIOPTPY_BOND_SCALE;
    use crate::coord::primitives::dihedral_angle_rad;
    use crate::domain::{AtomId, SpinCharge};
    use crate::io::xyz::parse_xyz_str;

    /// Pentane (anti, anti) with the hydrogens of the two inner methylene groups.
    const PENTANE: &str = "17\n\nC 0.000 0.000 0.000\nC 1.270 0.840 0.000\nC 2.540 0.000 0.000\n\
C 3.810 0.840 0.000\nC 5.080 0.000 0.000\nH 0.000 -0.640 0.890\nH 0.000 -0.640 -0.890\n\
H -0.890 0.640 0.000\nH 1.270 1.480 0.890\nH 1.270 1.480 -0.890\nH 2.540 -0.640 0.890\n\
H 2.540 -0.640 -0.890\nH 3.810 1.480 0.890\nH 3.810 1.480 -0.890\nH 5.080 -0.640 0.890\n\
H 5.080 -0.640 -0.890\nH 5.970 0.640 0.000\n";

    fn dihedral(s: &MolecularSystem, a: [usize; 4]) -> f64 {
        let p = |i: usize| s.position(AtomId::new(i)).value();
        dihedral_angle_rad(p(a[0]), p(a[1]), p(a[2]), p(a[3]), None).to_degrees()
    }

    /// `sys` with the dihedral about the bond `b - c` turned by `angle` (rad): the atoms on the
    /// side of `c` rotate.
    fn turned(sys: &MolecularSystem, b: usize, c: usize, angle: f64) -> MolecularSystem {
        let g = MolecularGraph::from_system(sys, MULTIOPTPY_BOND_SCALE);
        let mut side = vec![c];
        let mut k = 0;
        while k < side.len() {
            for &n in g.neighbors(side[k]) {
                if n != b && !side.contains(&n) {
                    side.push(n);
                }
            }
            k += 1;
        }
        let (pb, pc) = (
            sys.position(AtomId::new(b)).value(),
            sys.position(AtomId::new(c)).value(),
        );
        let axis: Vec<f64> = (0..3).map(|i| pc[i] - pb[i]).collect();
        let len = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
        let u = [axis[0] / len, axis[1] / len, axis[2] / len];
        let (co, si) = (angle.cos(), angle.sin());
        let mut out = sys.clone();
        for &a in side.iter().skip(1) {
            let p = sys.position(AtomId::new(a)).value();
            let v = [p[0] - pc[0], p[1] - pc[1], p[2] - pc[2]];
            let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            let cr = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let r: [f64; 3] =
                std::array::from_fn(|i| pc[i] + v[i] * co + cr[i] * si + u[i] * dot * (1.0 - co));
            out.set_position(AtomId::new(a), crate::domain::Position::new(r));
        }
        out
    }

    #[test]
    fn two_gauche_conformers_cross_into_the_doubly_gauche_one() {
        let tt = parse_xyz_str(PENTANE, SpinCharge::default(), None).unwrap();
        let third = 2.0 * std::f64::consts::PI / 3.0;
        // Gauche about the first inner bond one way, about the second the other way.
        let gt = turned(&tt, 1, 2, third);
        let tg = turned(&tt, 2, 3, -third);
        let found = crossover_candidates(
            &tt,
            &[gt, tg],
            &CrossoverSettings::default(),
            &mut Rng::from_seed(1),
        )
        .unwrap();
        // Z_tt + Z_gt - Z_tg and Z_tt + Z_tg - Z_gt: both dihedrals turned the same way, the
        // two mirror images of the doubly gauche conformer.
        assert_eq!(found.len(), 2);
        let wrap = |x: f64| (x + 540.0) % 360.0 - 180.0;
        for c in &found {
            let a = wrap(dihedral(c, [0, 1, 2, 3]) - dihedral(&tt, [0, 1, 2, 3]));
            let b = wrap(dihedral(c, [1, 2, 3, 4]) - dihedral(&tt, [1, 2, 3, 4]));
            assert!((a.abs() - 120.0).abs() < 1.0, "{a}");
            assert!((a - b).abs() < 1.0, "{a} {b}");
        }
    }
}
