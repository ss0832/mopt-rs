//! The classical terms of GFN1-xTB: repulsion and the halogen-bond correction.
//!
//! - Repulsion: `E = sum_(A<B) Z_A Z_B exp(-sqrt(alpha_A alpha_B) R^1.5) / R` (xtb
//!   `src/xtb/repulsion.F90`).
//! - Halogen bond: for every halogen X (Br, I; the strength of Cl is zero) and acceptor A
//!   (N, O, S) within 20 Bohr, with B the atom nearest to X,
//!   `E = k_X (1/2 - cos(B-X-A)/4)^6 ((R0/R_XA)^12 - 0.44 (R0/R_XA)^6) / (1 + (R0/R_XA)^12)`,
//!   `R0 = 1.3 (R_X + R_A)` (atomic radii) (xtb `src/xtb/halogen.f90`, `xbpot`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::hamiltonian::{bohr, diff, dot};
use super::params::{
    Gfn1Element, REP_KEXP, XB_ACCEPTORS, XB_CUTOFF2, XB_DAMP, XB_HALOGENS, XB_RAD,
};

fn add(grad: &mut [f64], atom: usize, g: [f64; 3], sign: f64) {
    for k in 0..3 {
        grad[3 * atom + k] += sign * g[k];
    }
}

/// Repulsion energy; its gradient is added to `grad`.
pub fn repulsion(el: &[&Gfn1Element], x: &[[f64; 3]], grad: &mut [f64]) -> f64 {
    let mut energy = 0.0;
    for i in 0..x.len() {
        for j in 0..i {
            let alpha = (el[i].rep_alpha * el[j].rep_alpha).sqrt();
            let zz = el[i].rep_zeff * el[j].rep_zeff;
            let d = diff(x, i, j);
            let r2 = dot(d, d);
            let r = r2.sqrt();
            let rk = r.powf(REP_KEXP);
            let e = zz * (-alpha * rk).exp() / r;
            energy += e;
            let f = -(alpha * rk * REP_KEXP + 1.0) * e / r2;
            let g = d.map(|v| f * v);
            add(grad, i, g, 1.0);
            add(grad, j, g, -1.0);
        }
    }
    energy
}

/// Halogen-bond energy; its gradient is added to `grad`.
pub fn halogen_bond(el: &[&Gfn1Element], x: &[[f64; 3]], grad: &mut [f64]) -> f64 {
    const ALP: f64 = 6.0;
    const LJ: i32 = 6;
    let n = x.len();
    let mut energy = 0.0;
    for xa in 0..n {
        if !XB_HALOGENS.contains(&el[xa].z) || el[xa].xb_strength == 0.0 {
            continue;
        }
        // The partner: the atom nearest to the halogen.
        let Some(b) = (0..n).filter(|&m| m != xa).min_by(|&p, &q| {
            let dp = diff(x, p, xa);
            let dq = diff(x, q, xa);
            dot(dp, dp).total_cmp(&dot(dq, dq))
        }) else {
            continue;
        };
        for aa in 0..n {
            if !XB_ACCEPTORS.contains(&el[aa].z) {
                continue;
            }
            let dxa = diff(x, aa, xa);
            let d2ax = dot(dxa, dxa);
            if d2ax >= XB_CUTOFF2 {
                continue;
            }
            let cc = el[xa].xb_strength;
            let r0 = XB_RAD * bohr(el[xa].atomic_rad + el[aa].atomic_rad);
            let dxb = diff(x, b, xa);
            let dba = diff(x, aa, b);
            let (d2bx, d2ab) = (dot(dxb, dxb), dot(dba, dba));
            let rax = d2ax.sqrt();
            let rbx = d2bx.sqrt();
            let xy = (d2bx * d2ax).sqrt();
            let term = (d2bx + d2ax - d2ab) / xy;
            let base = 0.5 - 0.25 * term;
            let aterm = base.powf(ALP);
            let t14 = (r0 / rax).powi(LJ);
            let num = t14 * t14 - XB_DAMP * t14;
            let den = 1.0 + t14 * t14;
            let termlj = num / den;
            energy += aterm * cc * termlj;
            let lj = f64::from(LJ);
            let mut dtermlj = 2.0 * lj * num * t14 * t14 / (rax * den * den);
            dtermlj += lj * t14 * (XB_DAMP - 2.0 * t14) / (rax * den);
            dtermlj *= aterm * cc / rax;
            add(grad, aa, dxa.map(|v| dtermlj * v), 1.0);
            add(grad, xa, dxa.map(|v| dtermlj * v), -1.0);
            let pref = -0.25 * ALP * base.powf(ALP - 1.0) * cc * termlj;
            let dc = (2.0 / rbx - term / rax) * pref / rax;
            add(grad, aa, dxa.map(|v| dc * v), 1.0);
            add(grad, xa, dxa.map(|v| dc * v), -1.0);
            let dc = (2.0 / rax - term / rbx) * pref / rbx;
            add(grad, b, dxb.map(|v| dc * v), 1.0);
            add(grad, xa, dxb.map(|v| dc * v), -1.0);
            let t13 = 2.0 * pref / xy;
            add(grad, aa, dba.map(|v| t13 * v), -1.0);
            add(grad, b, dba.map(|v| t13 * v), 1.0);
        }
    }
    energy
}
