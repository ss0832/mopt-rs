//! GFN-FF bending parameters, ported from xtb's `gfnff_ini` ("bend" section of
//! `src/gfnff/gfnff_ini.f90`; LGPL-3.0-or-later), with the branches of the main-group and the
//! transition metals (eta coordination, linear ligands, NN on a metal).
//!
//! In a periodic cell the outer atoms are neighbour images: xtb loops over the cell `c1` of
//! the first, its position `j` in that cell's list, the cell `c2` of the second, and the
//! positions `k <= j` in that list (skipping the same image), and drops a pair seen the other
//! way round; for a molecule this is every pair `k < j` once.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::constants as k;
use super::lattice::{Cell, Nb};
use super::neighbors::bangl_points;
use super::rings::{ringsatom, ringsbend};
use super::topology::{amide, folded, Setup};

/// One bending term: centre `i`, outer `j`, `k` (images in cells `cj`, `ck`), equilibrium
/// angle (radians), force constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub cj: Cell,
    pub ck: Cell,
    pub theta0: f64,
    pub fc: f64,
}

/// The neighbours of an atom grouped by cell, in the order of the cells.
pub fn by_cell(nb: &[Nb]) -> Vec<Vec<Nb>> {
    let mut out: Vec<Vec<Nb>> = Vec::new();
    for &(j, c) in nb {
        match out.last_mut() {
            Some(group) if group[0].1 == c => group.push((j, c)),
            _ => out.push(vec![(j, c)]),
        }
    }
    out
}

/// xtb's pairs of neighbour images of one atom for the bends (see the module comment).
pub fn image_pairs(nb: &[Nb]) -> Vec<(Nb, Nb)> {
    let groups = by_cell(nb);
    let mut seen: Vec<(Nb, Nb)> = Vec::new();
    let mut out = Vec::new();
    for g1 in &groups {
        for (jx, &a) in g1.iter().enumerate() {
            for g2 in &groups {
                for (kx, &b) in g2.iter().enumerate().take(jx + 1) {
                    if g1[0].1 == g2[0].1 && kx == jx {
                        continue;
                    }
                    if seen.contains(&(b, a)) {
                        continue;
                    }
                    seen.push((a, b));
                    out.push((a, b));
                }
            }
        }
    }
    out
}

/// The bending terms; `pi` are the pi atoms after Hueckel and `pbo(a, b)` the pi bond order.
pub fn angles(
    s: &Setup,
    x: &[[f64; 3]],
    pi: &[bool],
    pbo: &dyn Fn(usize, usize) -> f64,
) -> Vec<Angle> {
    let n = x.len();
    let f = s.frame(x);
    let hyb = &s.nb.hyb;
    let pi_flags: Vec<usize> = pi.iter().map(|&b| usize::from(b)).collect();
    let nb_atoms = folded(&s.nb.nb);
    let mut out = Vec::new();
    for i in 0..n {
        let nn = s.nb.nb[i].len();
        if nn <= 1 || nn > 6 {
            continue;
        }
        let ati = s.el[i].z();
        let group = s.el[i].raw.group;
        for ((jj, cj), (kk, ck)) in image_pairs(&s.nb.nb[i]) {
            {
                let (atj, atk) = (s.el[jj].z(), s.el[kk].z());
                let fijk = s.el[i].raw.angl * s.el[jj].raw.angl2 * s.el[kk].raw.angl2;
                if fijk < k::FCTHR {
                    continue;
                }
                let phi = bangl_points(f.at(jj, cj), x[i], f.at(kk, ck));
                let deg = phi * 180.0 / PI;
                // Eta cases of a metal, even with a CN below 6 (e.g. CaCp+).
                if s.el[i].metal() > 0 && deg < 60.0 {
                    continue;
                }
                // Eta coordination to a transition metal.
                let mut feta = 1.0;
                if s.imetal[i] == 2 && s.nb.itag[jj] == -1 && pi[jj] {
                    feta = 0.3;
                }
                if s.imetal[i] == 2 && s.nb.itag[kk] == -1 && pi[kk] {
                    feta *= 0.3;
                }
                let cnt = |z: u8| usize::from(atj == z) + usize::from(atk == z);
                let nh = cnt(1) as f64;
                let nnn = cnt(7);
                let _ = nnn;
                let no = cnt(8);
                let nsi = cnt(14) as f64;
                let nc = cnt(6) as f64;
                let nmet = (usize::from(s.el[jj].metal() != 0) + usize::from(s.el[kk].metal() != 0))
                    as f64;
                let npi = usize::from(pi[jj]) + usize::from(pi[kk]);
                let rings = ringsbend(&s.rings, i, jj, kk);
                let mut triple = (hyb[i] == 1 || hyb[jj] == 1) || (hyb[i] == 1 || hyb[kk] == 1);
                let no_metal = s.imetal[i] == 0 && s.imetal[jj] == 0 && s.imetal[kk] == 0;
                let fqq = 1.0
                    - (s.qa[i] * s.qa[jj] + s.qa[i] * s.qa[kk])
                        * k::QFACBEN
                        * if no_metal { 1.0 } else { 2.5 };
                let mut f2 = 1.0;
                let mut r0 = 100.0;
                if hyb[i] == 1 {
                    r0 = 180.0;
                }
                if hyb[i] == 2 {
                    r0 = 120.0;
                }
                if hyb[i] == 3 {
                    r0 = 109.5;
                }
                if hyb[i] == 3 && ati > 10 {
                    if nn <= 3 {
                        r0 = k::AHEAVY3;
                    }
                    if nn >= 4 {
                        r0 = k::AHEAVY4;
                    }
                    if nn == 4 && group == 5 {
                        r0 = 109.5;
                    }
                    if nn == 4 && group == 4 && ati > 49 {
                        r0 = 109.5;
                    }
                    if group == 4 || group == 5 || group == 6 {
                        r0 -= nh * 5.0;
                    }
                }
                if hyb[i] == 5 {
                    r0 = 90.0;
                    f2 = 0.11;
                    if deg > k::LINTHR {
                        r0 = 180.0;
                    }
                }
                if ati == 5 && (hyb[i] == 3 || hyb[i] == 2) {
                    r0 = 115.0;
                }
                if ati == 6 {
                    if hyb[i] == 3 && nh == 2.0 {
                        r0 = 108.6;
                    }
                    if hyb[i] == 3 && no == 1 {
                        r0 = 108.5;
                    }
                    if hyb[i] == 2 && no == 2 {
                        r0 = 122.0;
                    }
                    if hyb[i] == 2 && no == 1 {
                        f2 = 0.7;
                    }
                    if hyb[i] == 1 && no == 2 {
                        triple = false;
                        f2 = 2.0;
                    }
                    if hyb[i] == 3 && nn > 4 && deg > k::LINTHR {
                        r0 = 180.0;
                    }
                }
                if ati == 8 && nn == 2 {
                    r0 = 104.5;
                    if nh == 2.0 {
                        r0 = 100.0;
                        f2 = 1.20;
                    }
                    r0 += 7.0 * nsi;
                    r0 += 14.0 * nmet;
                    if npi == 2 {
                        r0 = 109.0;
                    }
                    if nmet > 0.0 && deg > k::LINTHR {
                        // Metal coordination can be linear.
                        r0 = 180.0;
                        f2 = 0.3;
                    }
                }
                if ati == 7 && nn == 2 {
                    f2 = 1.4;
                    r0 = 115.0;
                    if rings != 0 {
                        r0 = 105.0;
                    }
                    if atk == 8 || atj == 8 {
                        r0 = 103.0;
                    }
                    if atk == 9 || atj == 9 {
                        r0 = 102.0;
                    }
                    if hyb[i] == 1 {
                        r0 = 180.0;
                    }
                    // NN on a metal.
                    if s.imetal[jj] == 2 && hyb[i] == 1 && atk == 7 {
                        r0 = 135.0;
                    }
                    if s.imetal[kk] == 2 && hyb[i] == 1 && atj == 7 {
                        r0 = 135.0;
                    }
                }
                if ati == 7 && hyb[i] == 3 {
                    if npi > 0 {
                        if amide(&s.el, hyb, &nb_atoms, &pi_flags, i) {
                            r0 = 115.0;
                            f2 = 1.2;
                        } else {
                            let sumppi = pbo(i, jj) + pbo(i, kk);
                            r0 = 113.0;
                            f2 = 1.0 - sumppi * 0.7;
                        }
                    } else {
                        r0 = 104.0;
                        f2 = 0.40;
                        f2 += nh * 0.19;
                        f2 += no as f64 * 0.25;
                        f2 += nc * 0.01;
                    }
                }
                if rings == 3 {
                    r0 = 82.0;
                }
                if rings == 4 {
                    r0 = 96.0;
                }
                if rings == 5 && ati == 6 {
                    r0 = 109.0;
                }
                if rings == 0 && ringsatom(&s.rings, i) == 3 {
                    let rj = ringsatom(&s.rings, jj);
                    let rk = ringsatom(&s.rings, kk);
                    if rj + rk == 102 {
                        r0 += 4.0;
                    }
                }
                if triple {
                    f2 = 0.60;
                    if atj == 7 || atk == 7 {
                        f2 = 1.00;
                    }
                    // Linear ligands on a transition metal.
                    if (s.imetal[jj] == 2 || s.imetal[kk] == 2) && deg > k::LINTHR {
                        if ati == 6 && (atj == 6 || atk == 6) {
                            f2 = 3.0;
                        }
                        if ati == 6 && (atj == 7 || atk == 7) {
                            f2 = 3.0;
                        }
                        if ati == 6 && (s.el[jj].raw.group == 6 || s.el[kk].raw.group == 6) {
                            f2 = 14.0;
                        }
                        if ati == 7 && atj == 7 {
                            f2 = 10.0;
                        }
                        if ati == 7 && (atj == 6 || atk == 6) {
                            f2 = 10.0;
                        }
                        if ati == 7 && (atj == 8 || atk == 8) {
                            r0 = 180.0;
                            f2 = 12.0;
                        }
                    }
                }
                if group == 4 && nn == 2 && s.nb.itag[i] == 1 {
                    if ati == 6 {
                        r0 = 145.0;
                    }
                    if ati > 6 {
                        r0 = 90.0;
                    }
                }
                if group == 6 && nn == 4 && no >= 1 {
                    r0 = 115.0;
                }
                if group == 7 && hyb[i] == 1 {
                    if matches!(ati, 9 | 17 | 35 | 53) {
                        r0 = 90.0;
                    }
                    if ati > 9 && deg > k::LINTHR {
                        r0 = 180.0;
                    }
                    f2 = 0.6 / f64::from(ati).powf(0.15);
                }
                if hyb[i] == 3 && group == 4 && ati > 32 && s.qa[i] > 0.4 {
                    if deg > 140.0 {
                        r0 = 180.0;
                    }
                    if deg < 100.0 {
                        r0 = 90.0;
                    }
                    f2 = 1.0;
                }
                if s.imetal[i] > 0 {
                    if hyb[i] == 0 {
                        r0 = 90.0;
                        f2 = 1.35;
                    }
                    if hyb[i] == 1 {
                        r0 = 180.0;
                    }
                    if hyb[i] == 2 {
                        r0 = 120.0;
                    }
                    if hyb[i] == 3 {
                        r0 = 109.5;
                    }
                    if deg > k::LINTHR {
                        r0 = 180.0;
                    }
                }
                let fn_ = 1.0 - 2.36 / (nn as f64).powi(2);
                let theta0 = r0 * PI / 180.0;
                let fbsmall = 1.0 - k::FBS1 * (-0.64 * (theta0 - PI).powi(2)).exp();
                out.push(Angle {
                    i,
                    j: jj,
                    k: kk,
                    cj,
                    ck,
                    theta0,
                    fc: fijk * fqq * f2 * fn_ * fbsmall * feta,
                });
            }
        }
    }
    out
}
