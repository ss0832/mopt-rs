//! Building blocks of MultiOptPy's model Hessians, reproduced as they are
//! (`ModelHessian/calc_params.py`, `Utils/bond_connectivity.py`, and the rigid-body projection
//! of `Utils/calc_tools.py`).
//!
//! Wilson B-vectors (`stretch2`, `bend2`, `torsion2`, `outofplane2`) follow MultiOptPy line by
//! line, including its guards for degenerate geometries; connectivity uses MultiOptPy's 1.1 x
//! covalent-radius criterion and its angle and dihedral enumeration (which, for example, skips
//! angles closing a three-membered ring).
//!
//! MultiOptPy: `ModelHessian/calc_params.py`, `Utils/bond_connectivity.py`, and the rigid-body
//!  projection of `Utils/calc_tools.py:Calculationtools`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use crate::domain::{Element, MolecularSystem, Units};
use crate::parameters::multioptpy_tables::{MOP_BOHR2ANGSTROMS, MOP_HARTREE2KCALMOL};
use crate::parameters::{covalent_radius_bohr, uff_vdw_distance_bohr, uff_vdw_well_depth_hartree};

/// MultiOptPy's Bohr per mopt-rs Bohr: the shared parameter tables are converted from Angstrom
/// with the CODATA 2018 Bohr radius, MultiOptPy converts them with its own (0.52917721067), and
/// the legacy models reproduce MultiOptPy (`Parameters/unit_values.py:UnitValueLib`).
const MOP_BOHR_PER_BOHR: f64 = Units::BOHR_TO_ANGSTROM / MOP_BOHR2ANGSTROMS;

/// MultiOptPy's Hartree per mopt-rs Hartree (tables in kcal/mol; MultiOptPy's 627.509).
const MOP_HARTREE_PER_HARTREE: f64 = Units::HARTREE_TO_KCAL_MOL / MOP_HARTREE2KCALMOL;

/// UFF van der Waals distance in MultiOptPy's Bohr (`UFF_VDW_distance_lib`).
pub(crate) fn mop_vdw_distance_bohr(el: Element) -> f64 {
    uff_vdw_distance_bohr(el) * MOP_BOHR_PER_BOHR
}

/// UFF van der Waals well depth in MultiOptPy's Hartree (`UFF_VDW_well_depth_lib`).
pub(crate) fn mop_vdw_well_depth_hartree(el: Element) -> f64 {
    uff_vdw_well_depth_hartree(el) * MOP_HARTREE_PER_HARTREE
}

pub(crate) type V3 = [f64; 3];

pub(crate) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Positions (Bohr) as rows.
pub(crate) fn coords(sys: &MolecularSystem) -> Vec<V3> {
    let x = sys.coords_flat_bohr();
    (0..sys.n_atoms())
        .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
        .collect()
}

/// MultiOptPy's single-bond covalent radius (Bohr; dummy atoms 1 Angstrom).
pub(crate) fn cov_radius(el: Element) -> f64 {
    if el.is_dummy() {
        1.0 / MOP_BOHR2ANGSTROMS
    } else {
        covalent_radius_bohr(el) * MOP_BOHR_PER_BOHR
    }
}

/// `stretch2`: distance and B-vectors (`-u`, `u` with `u = (a - b) / |a - b|`).
pub(crate) fn stretch2(a: V3, b: V3) -> (f64, [V3; 2]) {
    let d = sub(a, b);
    let r = norm(d);
    let u = [d[0] / r, d[1] / r, d[2] / r];
    (r, [[-u[0], -u[1], -u[2]], u])
}

/// `bend2`: angle at `b` and B-vectors.
pub(crate) fn bend2(a: V3, b: V3, c: V3) -> (f64, [V3; 3]) {
    let (r1, b1) = stretch2(a, b);
    let (r2, b2) = stretch2(b, c);
    let mut c_o = 0.0;
    let mut crap = 0.0;
    for i in 0..3 {
        c_o += b1[0][i] * b2[1][i];
        crap += b1[0][i] * b1[0][i];
        crap += b2[1][i] * b2[1][i];
    }
    // Rounding can leave the cosine of a linear arrangement just outside [-1, 1], where
    // MultiOptPy's `arccos` and `sqrt` return NaN; inside the interval nothing changes.
    let c_o = c_o.clamp(-1.0, 1.0);
    let (mut fir, si) = if crap.sqrt() < 1e-12 {
        (PI - crap.sqrt().asin(), crap.sqrt())
    } else {
        (c_o.acos(), (1.0 - c_o * c_o).sqrt())
    };
    if (fir - PI).abs() < 1e-12 {
        fir = PI;
    }
    let mut bf = [[0.0; 3]; 3];
    let (d1, d2) = (r1 * si, r2 * si);
    for i in 0..3 {
        bf[0][i] = if d1 < 1e-12 {
            0.0
        } else {
            (c_o * b1[0][i] - b2[1][i]) / d1
        };
        bf[2][i] = if d2 < 1e-12 {
            0.0
        } else {
            (c_o * b2[1][i] - b1[0][i]) / d2
        };
        bf[1][i] = -(bf[0][i] + bf[2][i]);
    }
    (fir, bf)
}

/// `torsion2`: torsion angle (MultiOptPy's value, whose cosine has misplaced parentheses; only
/// the B-vectors enter the Hessians) and B-vectors.
pub(crate) fn torsion2(p: [V3; 4]) -> (f64, [V3; 4]) {
    let (r1, bij) = stretch2(p[0], p[1]);
    let (r2, bjk) = stretch2(p[1], p[2]);
    let (r3, bkl) = stretch2(p[2], p[3]);
    let (fi2, _) = bend2(p[0], p[1], p[2]);
    let (fi3, _) = bend2(p[1], p[2], p[3]);
    let (s2, s3, c2, c3) = (fi2.sin(), fi3.sin(), fi2.cos(), fi3.cos());
    let costau = (bij[0][1] * bjk[1][2] - bjk[1][1] * bij[0][2])
        * (bjk[0][1] * bkl[1][2] - bjk[0][2] * bkl[1][1])
        + (bij[0][2] * bjk[1][0] - bij[0][0] * bjk[1][2])
            * (bjk[0][2] * bkl[1][0] - bjk[0][0] * bkl[1][2])
        + (bij[0][0] * bjk[1][1] - bij[0][1] * bjk[1][0])
            * (bjk[0][0] * bkl[1][1] - bjk[0][1] * bkl[1][0])
            / (s2 * s3);
    let sintau = (bij[1][0] * (bjk[0][1] * bkl[1][2] - bjk[0][2] * bkl[1][1])
        + bij[1][1] * (bjk[0][2] * bkl[1][0] - bjk[0][0] * bkl[1][2])
        + bij[1][2] * (bjk[0][0] * bkl[1][1] - bjk[0][1] * bkl[1][0]))
        / (s2 * s3);
    let mut tau = sintau.atan2(costau);
    if tau.abs() == PI {
        tau = PI;
    }
    let mut bt = [[0.0; 3]; 4];
    for ix in 1..=3usize {
        let iy = if ix + 1 > 3 { ix + 1 - 3 } else { ix + 1 };
        let iz = if iy + 1 > 3 { iy + 1 - 3 } else { iy + 1 };
        let (x, y, z) = (ix - 1, iy - 1, iz - 1);
        bt[0][x] = (bij[1][y] * bjk[1][z] - bij[1][z] * bjk[1][y]) / (r1 * s2 * s2);
        bt[3][x] = (bkl[0][y] * bjk[0][z] - bkl[0][z] * bjk[0][y]) / (r3 * s3 * s3);
        bt[1][x] = -((r2 - r1 * c2) * bt[0][x] + r3 * c3 * bt[3][x]) / r2;
        bt[2][x] = -(bt[0][x] + bt[1][x] + bt[3][x]);
    }
    (tau, bt)
}

/// `outofplane2`: angle of atom 0 above the plane of 1 and 2 around the center 3, and
/// B-vectors (zero for degenerate geometries).
pub(crate) fn outofplane2(p: [V3; 4]) -> (f64, [V3; 4]) {
    let unit = |v: V3| {
        let n = norm(v);
        [v[0] / n, v[1] / n, v[2] / n]
    };
    let e42 = unit(sub(p[1], p[3]));
    let e43 = unit(sub(p[2], p[3]));
    // Clamped as in `bend2` (MultiOptPy also evaluates two angles it does not use).
    let fi1 = dot(e43, e42).clamp(-1.0, 1.0).acos();
    if (fi1 - PI).abs() < 1e-12 {
        return (0.0, [[0.0; 3]; 4]);
    }
    let n = cross(sub(p[1], p[3]), sub(p[2], p[3]));
    if dot(n, n) < 1e-12 {
        return (0.0, [[0.0; 3]; 4]);
    }
    let tip = [n[0] + p[3][0], n[1] + p[3][1], n[2] + p[3][2]];
    let (teta, br) = bend2(p[0], p[3], tip);
    let teta = teta - 0.5 * PI;
    let mut bt = [[0.0; 3]; 4];
    for ix in 1..=3usize {
        let iy = (ix + 1) % 4 + (ix + 1) / 4;
        let iz = (iy + 1) % 4 + (iy + 1) / 4;
        bt[0][ix - 1] = -br[2][ix - 1];
        bt[1][ix - 1] = -br[2][iy - 1];
        bt[2][ix - 1] = -br[2][iz - 1];
        bt[3][ix - 1] = -(bt[0][ix - 1] + bt[1][ix - 1] + bt[2][ix - 1]);
    }
    for row in &mut bt {
        for v in row.iter_mut() {
            *v = -*v;
        }
    }
    (teta, bt)
}

/// `H[3a+n][3b+m] += k b_a[n] b_b[m]` for every pair of the atoms of one internal coordinate.
pub(crate) fn add_outer(h: &mut Mat<f64>, atoms: &[usize], b: &[V3], k: f64) {
    for (ia, &a) in atoms.iter().enumerate() {
        for (ib, &bb) in atoms.iter().enumerate() {
            for n in 0..3 {
                for m in 0..3 {
                    h[(3 * a + n, 3 * bb + m)] += k * b[ia][n] * b[ib][m];
                }
            }
        }
    }
}

/// Copy the upper triangle into the lower one (MultiOptPy's "symmetrization").
pub(crate) fn mirror_upper(h: &mut Mat<f64>) {
    let n = h.nrows();
    for i in 0..n {
        for j in 0..i {
            h[(i, j)] = h[(j, i)];
        }
    }
}

/// MultiOptPy's `project_out_hess_tr_and_rot_for_coord`: `P^T H P` with the projector onto the
/// complement of rigid translations and rotations (no mass weighting, Gram-Schmidt in the order
/// x, y, z translations then rotations), then `(H + H^T) / 2`.
pub(crate) fn project_tr_rot(h: &Mat<f64>, x: &[V3]) -> Mat<f64> {
    let n = x.len();
    let c = x.iter().fold([0.0; 3], |acc, p| {
        [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
    });
    let c = [c[0] / n as f64, c[1] / n as f64, c[2] / n as f64];
    let mut vecs: Vec<Vec<f64>> = Vec::with_capacity(6);
    for k in 0..3 {
        let mut v = vec![0.0; 3 * n];
        for a in 0..n {
            v[3 * a + k] = 1.0;
        }
        vecs.push(v);
    }
    for k in 0..3 {
        let mut v = vec![0.0; 3 * n];
        for (a, p) in x.iter().enumerate() {
            let (px, py, pz) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
            let r = match k {
                0 => [0.0, -pz, py],
                1 => [pz, 0.0, -px],
                _ => [-py, px, 0.0],
            };
            v[3 * a..3 * a + 3].copy_from_slice(&r);
        }
        vecs.push(v);
    }
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for v in &vecs {
        let mut w = v.clone();
        for b in &basis {
            let d: f64 = v.iter().zip(b).map(|(p, q)| p * q).sum();
            w.iter_mut().zip(b).for_each(|(wi, bi)| *wi -= d * bi);
        }
        let nw = w.iter().map(|t| t * t).sum::<f64>().sqrt();
        if nw > 1e-10 {
            basis.push(w.into_iter().map(|t| t / nw).collect());
        }
    }
    let dim = 3 * n;
    let mut p = Mat::<f64>::identity(dim, dim);
    for b in &basis {
        for i in 0..dim {
            for j in 0..dim {
                p[(i, j)] -= b[i] * b[j];
            }
        }
    }
    let ph = p.transpose() * h * &p;
    let mut out = Mat::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            out[(i, j)] = 0.5 * (ph[(i, j)] + ph[(j, i)]);
        }
    }
    out
}

/// MultiOptPy's `BondConnectivity` (1.1 x sum of covalent radii).
#[derive(Debug, Clone)]
pub(crate) struct Connectivity {
    pub bonded: Vec<Vec<bool>>,
}

impl Connectivity {
    pub(crate) fn new(elements: &[Element], x: &[V3]) -> Self {
        Self::with_factor(elements, x, 1.1)
    }

    pub(crate) fn with_factor(elements: &[Element], x: &[V3], factor: f64) -> Self {
        let n = x.len();
        let mut bonded = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let lim = (cov_radius(elements[i]) + cov_radius(elements[j])) * factor;
                    bonded[i][j] = norm(sub(x[i], x[j])) <= lim;
                }
            }
        }
        Self { bonded }
    }

    pub(crate) fn n(&self) -> usize {
        self.bonded.len()
    }

    /// `bond_connect_table`: pairs `i <= j`.
    pub(crate) fn bonds(&self) -> Vec<[usize; 2]> {
        let n = self.n();
        let mut out = Vec::new();
        for i in 0..n {
            for j in i..n {
                if self.bonded[i][j] {
                    out.push([i, j]);
                }
            }
        }
        out
    }

    /// `angle_connect_table`: `[j, i, n]` with center `i`, skipping three-membered rings.
    pub(crate) fn angles(&self) -> Vec<[usize; 3]> {
        let n = self.n();
        let mut out = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if self.bonded[i][j] {
                    for k in j + 1..n {
                        if self.bonded[i][k] && !self.bonded[j][k] {
                            out.push([j, i, k]);
                        }
                    }
                }
            }
        }
        out
    }

    /// `dihedral_angle_connect_table`, with MultiOptPy's case analysis and order.
    pub(crate) fn dihedrals(&self) -> Vec<[usize; 4]> {
        let a = self.angles();
        let b = &self.bonded;
        let mut out = Vec::new();
        for i in 0..a.len() {
            for j in i + 1..a.len() {
                let (ai, aj) = (a[i], a[j]);
                if (ai[1] == aj[1] && ai[2] == aj[2]) || (ai[1] == aj[2] && ai[2] == aj[1]) {
                    let c = [ai[0], ai[1], ai[2], aj[0]];
                    if b[c[2]][c[3]] {
                        out.push(c);
                        continue;
                    }
                    let c = [aj[0], ai[0], ai[1], ai[2]];
                    if b[c[1]][c[0]] {
                        out.push(c);
                        continue;
                    }
                }
                if (ai[1] == aj[1] && ai[0] == aj[0]) || (ai[1] == aj[0] && ai[0] == aj[1]) {
                    let c = [aj[2], ai[0], ai[1], ai[2]];
                    if b[c[1]][c[0]] {
                        out.push(c);
                        continue;
                    }
                    let c = [ai[0], ai[1], ai[2], aj[2]];
                    if b[c[2]][c[3]] {
                        out.push(c);
                        continue;
                    }
                }
                if (ai[1] == aj[0] && ai[2] == aj[1]) || (ai[1] == aj[1] && ai[2] == aj[0]) {
                    let c = [ai[0], ai[1], ai[2], aj[2]];
                    if b[c[2]][c[3]] {
                        out.push(c);
                        continue;
                    }
                    let c = [aj[2], ai[0], ai[1], ai[2]];
                    if b[c[1]][c[0]] {
                        out.push(c);
                        continue;
                    }
                }
                if (ai[0] == aj[1] && ai[1] == aj[2]) || (ai[0] == aj[2] && ai[1] == aj[1]) {
                    let c = [aj[0], ai[0], ai[1], ai[2]];
                    if b[c[1]][c[0]] {
                        out.push(c);
                        continue;
                    }
                    let c = [ai[0], ai[1], ai[2], aj[0]];
                    if b[c[2]][c[3]] {
                        out.push(c);
                        continue;
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exactly linear arrangements, where rounding puts the cosine outside [-1, 1] for some
    /// directions: the angle is 0 or pi and the B-vectors vanish instead of becoming NaN.
    #[test]
    fn linear_arrangements_stay_finite() {
        let mut beyond = 0;
        for k in 0..200 {
            let t = 0.37 + 0.731 * k as f64;
            let d = [
                t.sin() * (1.3 * t).cos(),
                t.sin() * (1.3 * t).sin(),
                t.cos(),
            ];
            let at = |s: f64| [s * d[0], s * d[1], s * d[2]];
            let (u1, u2) = (stretch2(at(1.7), at(0.0)).1, stretch2(at(0.0), at(-2.9)).1);
            let c: f64 = (0..3).map(|i| u1[0][i] * u2[1][i]).sum();
            beyond += usize::from(c.abs() > 1.0);
            // a - b - c in a line, and a folded back onto c.
            for (a, b, c, expect) in [
                (at(1.7), at(0.0), at(-2.9), PI),
                (at(1.7), at(0.0), at(2.9), 0.0),
            ] {
                let (angle, bf) = bend2(a, b, c);
                assert!((angle - expect).abs() < 1e-7, "{angle}");
                assert!(bf.iter().flatten().all(|v| v.is_finite()));
            }
            let (angle, bt) = outofplane2([at(1.0), at(2.0), at(-1.5), at(0.0)]);
            assert!(angle.is_finite() && bt.iter().flatten().all(|v| v.is_finite()));
        }
        // The case the clamp is for does occur.
        assert!(beyond > 0);
    }
}
