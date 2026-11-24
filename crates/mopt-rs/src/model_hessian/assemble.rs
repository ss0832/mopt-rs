//! Cartesian assembly of model Hessians from internal-coordinate force constants,
//! `H += k b b^T` with the Wilson B row `b` of a primitive, and of pair blocks.
//!
//! B rows come from automatic differentiation of the primitive coordinates
//! ([`crate::coord::primitives`]), which use minimum images in periodic cells.
//!
//! MultiOptPy: `ModelHessian/calc_params.py` (B-vector assembly of the model Hessians).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::coord::primitives::{
    bond_angle_rad, bond_distance, dihedral_angle_rad, out_of_plane_angle_rad,
};
use crate::domain::{AtomId, MolecularSystem, PeriodicCell};
use crate::math::ad::{local_value_grad, mic_delta_ad, Scalar};

/// Accumulates a `3N x 3N` model Hessian.
pub struct HessianAccumulator<'a> {
    sys: &'a MolecularSystem,
    cell: Option<&'a PeriodicCell>,
    h: Mat<f64>,
}

fn unit<T: Scalar>(v: [T; 3]) -> [T; 3] {
    let n = T::norm3(v);
    v.map(|x| x / n)
}

impl<'a> HessianAccumulator<'a> {
    /// Empty Hessian for `sys`.
    pub fn new(sys: &'a MolecularSystem) -> Self {
        Self {
            sys,
            cell: sys.active_cell(),
            h: Mat::zeros(sys.dof(), sys.dof()),
        }
    }

    /// Minimum-image vector `r_a - r_b`.
    pub fn delta(&self, a: AtomId, b: AtomId) -> [f64; 3] {
        let (pa, pb) = (self.sys.position(a).value(), self.sys.position(b).value());
        match self.cell {
            Some(c) => c.mic_delta(pa, pb),
            None => [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]],
        }
    }

    /// Squared minimum-image distance.
    pub fn r2(&self, a: AtomId, b: AtomId) -> f64 {
        let d = self.delta(a, b);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }

    pub(crate) fn add_rows<const K: usize>(
        &mut self,
        atoms: [AtomId; K],
        b: &[[f64; 3]; K],
        k: f64,
    ) {
        if k == 0.0 || !b.iter().flatten().all(|x| x.is_finite()) {
            return;
        }
        for (&a, ba) in atoms.iter().zip(b) {
            for (ca, &xa) in ba.iter().enumerate() {
                let wa = k * xa;
                if wa == 0.0 {
                    continue;
                }
                for (&bb, bv) in atoms.iter().zip(b) {
                    for (cb, &xb) in bv.iter().enumerate() {
                        self.h[(a.dof(ca), bb.dof(cb))] += wa * xb;
                    }
                }
            }
        }
    }

    fn positions<const K: usize>(&self, atoms: [AtomId; K]) -> [[f64; 3]; K] {
        atoms.map(|a| self.sys.position(a).value())
    }

    /// Stretch `a-b` with force constant `k`.
    pub fn stretch(&mut self, a: AtomId, b: AtomId, k: f64) {
        let cell = self.cell;
        let d = local_value_grad(&self.positions([a, b]), |x| bond_distance(x[0], x[1], cell));
        self.add_rows([a, b], &d.gradient, k);
    }

    /// Bend `a-m-b` at `m`.
    pub fn bend(&mut self, a: AtomId, m: AtomId, b: AtomId, k: f64) {
        let cell = self.cell;
        let d = local_value_grad(&self.positions([a, m, b]), |x| {
            bond_angle_rad(x[0], x[1], x[2], cell)
        });
        self.add_rows([a, m, b], &d.gradient, k);
    }

    /// Decius linear bend `a-m-b` measured along the unit direction `e` perpendicular to the
    /// axis: `q = e . (n_ma + n_mb)`.
    pub fn linear_bend(&mut self, a: AtomId, m: AtomId, b: AtomId, e: [f64; 3], k: f64) {
        let cell = self.cell;
        let d = local_value_grad(&self.positions([a, m, b]), |x| {
            let n1 = unit(mic_delta_ad(x[0], x[1], cell));
            let n2 = unit(mic_delta_ad(x[2], x[1], cell));
            let e = e.map(Scalar::from_f64);
            Scalar::dot3(e, [n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2]])
        });
        self.add_rows([a, m, b], &d.gradient, k);
    }

    /// Torsion `a-b-c-d`.
    pub fn torsion(&mut self, atoms: [AtomId; 4], k: f64) {
        let cell = self.cell;
        let d = local_value_grad(&self.positions(atoms), |x| {
            dihedral_angle_rad(x[0], x[1], x[2], x[3], cell)
        });
        self.add_rows(atoms, &d.gradient, k);
    }

    /// Wilson out-of-plane angle of the bond `center-a` from the plane of `center`, `b`, `c`.
    pub fn out_of_plane(&mut self, center: AtomId, a: AtomId, b: AtomId, c: AtomId, k: f64) {
        let cell = self.cell;
        let d = local_value_grad(&self.positions([center, a, b, c]), |x| {
            out_of_plane_angle_rad(x[0], x[1], x[2], x[3], cell)
        });
        self.add_rows([center, a, b, c], &d.gradient, k);
    }

    /// Pair term with `d2E/dd2 = block` for `d = r_a - r_b`: `+block` on the diagonal atom
    /// blocks, `-block` on the off-diagonal ones.
    pub fn pair(&mut self, a: AtomId, b: AtomId, block: &[[f64; 3]; 3]) {
        for (p, row) in block.iter().enumerate() {
            for (q, &v) in row.iter().enumerate() {
                self.h[(a.dof(p), a.dof(q))] += v;
                self.h[(b.dof(p), b.dof(q))] += v;
                self.h[(a.dof(p), b.dof(q))] -= v;
                self.h[(b.dof(p), a.dof(q))] -= v;
            }
        }
    }

    /// The system.
    pub fn system(&self) -> &MolecularSystem {
        self.sys
    }

    /// The assembled Hessian (symmetrized).
    pub fn finish(self) -> Mat<f64> {
        let mut h = self.h;
        crate::math::linalg::symmetrize(&mut h);
        h
    }
}
