//! GFN-FF torsion and out-of-plane parameters, and the geometry-dependent hybridization fix of
//! three-coordinated C without pi bonds, ported from xtb's `gfnff_ini` ("torsion",
//! "out-of-plane", and "modify hyb due to pi assignment" of `src/gfnff/gfnff_ini.f90`;
//! `alphaCO`, `chktors`, `ssort` of `gfnff_ini2.F90`; `omegaPBC` of `constr.f90`;
//! LGPL-3.0-or-later), non-metal branches.
//!
//! In a periodic cell, as xtb does: a torsion of the bond `(j, i, n_j)` takes a neighbour `k`
//! of `i` and a neighbour `l` of `j` whose cell relative to `j` added to `n_j` (`fTrSum` of the
//! setup's translations) is one of the 27 around the home cell, and is dropped otherwise; the
//! angles of `chktors` and the inversion angles are those of the images; the three neighbours
//! of an out-of-plane term are taken cell by cell (`nbLoc`) and sorted with their cells by the
//! distances of their images.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use super::angles::by_cell;
use super::bonds::BondParams;
use super::constants as k;
use super::lattice::{is_central, Cell, Frame, Nb, HOME};
use super::neighbors::{bangl_points, dist_norm2, V3};
use super::rings::{ringsbond, ringstorl, ringstors};
use super::topology::{amide, folded, Setup};

/// One torsion (or out-of-plane) term: atoms as in xtb's `tlist(1:4)` and the cells of their
/// images, `nrot` (0 out-of-plane pi, -1 out-of-plane saturated N with a double minimum),
/// `phi0` (radians), force constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torsion {
    pub atoms: [usize; 4],
    pub cells: [Cell; 4],
    pub nrot: i32,
    pub phi0: f64,
    pub fc: f64,
}

/// `omegaPBC`: the inversion angle of `pl` over the plane `pi`-`pj`-`pk`.
pub fn omega_points(pi: V3, pj: V3, pk: V3, pl: V3) -> f64 {
    let re = [0, 1, 2].map(|c| pi[c] - pj[c]);
    let rd = [0, 1, 2].map(|c| pk[c] - pj[c]);
    let rv = [0, 1, 2].map(|c| pl[c] - pi[c]);
    let mut rn = [
        re[1] * rd[2] - re[2] * rd[1],
        re[2] * rd[0] - re[0] * rd[2],
        re[0] * rd[1] - re[1] * rd[0],
    ];
    let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let (nn, nv) = (norm(&rn), norm(&rv));
    rn = rn.map(|v| v / nn);
    let rvu = rv.map(|v| v / nv);
    (rn[0] * rvu[0] + rn[1] * rvu[1] + rn[2] * rvu[2]).asin()
}

/// `omegaPBC` for a molecule.
pub fn omega(x: &[V3], i: usize, j: usize, kk: usize, l: usize) -> f64 {
    omega_points(x[i], x[j], x[kk], x[l])
}

/// The three neighbours of a three-coordinated atom as xtb takes them from `nbLoc`: all from
/// one cell in order; from two cells the first of each, then the second of the first cell (or
/// of the second); from three cells the first of each.
fn three_neighbors(nb: &[Nb]) -> Option<[Nb; 3]> {
    let g = by_cell(nb);
    match g.len() {
        1 if g[0].len() >= 3 => Some([g[0][0], g[0][1], g[0][2]]),
        2 => {
            let third = g[0].get(1).or_else(|| g[1].get(1)).copied()?;
            Some([g[0][0], g[1][0], third])
        }
        3 => Some([g[0][0], g[1][0], g[2][0]]),
        _ => None,
    }
}

/// xtb's `ssort` (selection sort, ascending; equal values keep the last minimum) of `ind` by
/// `keys`.
fn ssort(keys: &mut [f64], ind: &mut [usize]) {
    let n = keys.len();
    for ii in 1..n {
        let i = ii - 1;
        let mut kk = i;
        let mut pp = keys[i];
        for j in ii..n {
            if keys[j] > pp {
                continue;
            }
            kk = j;
            pp = keys[j];
        }
        if kk == i {
            continue;
        }
        keys[kk] = keys[i];
        keys[i] = pp;
        ind.swap(i, kk);
    }
}

/// The hybridization fix after the pi assignment: three-coordinated group-4 atoms with sp2 but
/// no pi bond become sp3 when strongly pyramidal (more than 40 degrees).
pub fn fix_hybridization(s: &mut Setup, x: &[V3], pi: &[bool]) {
    let lattice = s.lattice;
    let f = Frame {
        x,
        lattice: lattice.as_ref(),
    };
    for i in 0..x.len() {
        if s.nb.hyb[i] == 2 && !pi[i] && s.nb.nb[i].len() == 3 && s.el[i].raw.group == 4 {
            let Some([(jj, cj), (kk, ck), (ll, cl)]) = three_neighbors(&s.nb.nb[i]) else {
                continue;
            };
            let phi = omega_points(x[i], f.at(jj, cj), f.at(kk, ck), f.at(ll, cl));
            if phi.abs() * 180.0 / PI > 40.0 {
                s.nb.hyb[i] = 3;
            }
        }
    }
}

/// `alphaCO`: one of the bonded atoms is a pi C carrying a terminal pi O, the other an sp3 C.
fn alpha_co(s: &Setup, pi: &[bool], a: usize, b: usize) -> bool {
    let hyb = &s.nb.hyb;
    let z = |i: usize| s.el[i].z();
    let has_co = |c: usize| {
        s.nb.nb[c]
            .iter()
            .filter(|&&(j, _)| z(j) == 8 && pi[j] && s.nb.nb[j].len() == 1)
            .count()
            == 1
    };
    if pi[a] && hyb[b] == 3 && z(a) == 6 && z(b) == 6 && has_co(a) {
        return true;
    }
    pi[b] && hyb[a] == 3 && z(b) == 6 && z(a) == 6 && has_co(b)
}

/// `chktors`: true when an angle of the torsion is near 180 degrees: at `i` between the
/// images of `j` and `k`, at the image of `j` between `i` and the image of `l`.
fn chktors(f: &Frame<'_>, i: usize, j: Nb, kk: Nb, l: Nb) -> bool {
    // xtb's truncated pi of this routine.
    #[allow(clippy::approx_constant)]
    const PI_X: f64 = 3.141_592_6;
    let (pj, pk, pl) = (f.at(j.0, j.1), f.at(kk.0, kk.1), f.at(l.0, l.1));
    if bangl_points(pj, f.x[i], pk) * 180.0 / PI_X > 170.0 {
        return true;
    }
    bangl_points(f.x[i], pj, pl) * 180.0 / PI_X > 170.0
}

/// Torsions of every bond and the out-of-plane terms.
pub fn torsions(
    s: &Setup,
    x: &[V3],
    pi: &[bool],
    pibo: &[f64],
    bonds: &[BondParams],
    pbo: &dyn Fn(usize, usize) -> f64,
) -> Vec<Torsion> {
    let hyb = &s.nb.hyb;
    let nbn = |a: usize| s.nb.nb[a].len();
    let z = |a: usize| s.el[a].z();
    let tors = |a: usize| s.el[a].raw.tors;
    let tors2 = |a: usize| s.el[a].raw.tors2;
    let group = |a: usize| s.el[a].raw.group;
    let piflags: Vec<usize> = pi.iter().map(|&b| usize::from(b)).collect();
    let f = s.frame(x);
    let nb_atoms = folded(&s.nb.nb);
    let mut out = Vec::new();
    for (m, &(jj, ii, cj)) in s.bonds.iter().enumerate() {
        let btyp = bonds[m].btyp;
        if btyp == 3 || btyp == 6 {
            continue;
        }
        let mut fij = tors(ii) * tors(jj);
        if fij < k::FCTHR {
            continue;
        }
        if tors(ii) < 0.0 || tors(jj) < 0.0 {
            continue;
        }
        // No torsions about hypercoordinated transition metals.
        if (s.el[ii].metal() > 1 && nbn(ii) > 4) || (s.el[jj].metal() > 1 && nbn(jj) > 4) {
            continue;
        }
        let fqq = 1.0 + (s.qa[ii] * s.qa[jj]).abs() * k::QFACTOR;
        let rings = ringsbond(&s.rings, ii, jj);
        let lring = rings > 0;
        let sp3ij = hyb[ii] == 3 && hyb[jj] == 3;
        let nhi = 1 + s.nb.nb[ii].iter().filter(|&&(a, _)| z(a) == 1).count();
        let nhj = 1 + s.nb.nb[jj].iter().filter(|&&(a, _)| z(a) == 1).count();
        fij *= (nhi as f64 * nhj as f64).powf(0.07);
        if alpha_co(s, pi, ii, jj) {
            fij *= 1.3;
        }
        if amide(&s.el, hyb, &nb_atoms, &piflags, ii) && hyb[jj] == 3 && z(jj) == 6 {
            fij *= 1.3;
        }
        if amide(&s.el, hyb, &nb_atoms, &piflags, jj) && hyb[ii] == 3 && z(ii) == 6 {
            fij *= 1.3;
        }
        if btyp == 4 {
            fij *= 0.2;
        }
        for &(kk, ck) in &s.nb.nb[ii] {
            if kk == jj && ck == cj {
                continue;
            }
            for &(ll, cl_relative) in &s.nb.nb[jj] {
                // The image of l follows the image of j.
                let Some(cl) = s.translations.sum(cl_relative, cj) else {
                    continue;
                };
                if !is_central(cl) {
                    continue;
                }
                if (ll == ii && cl == HOME) || (ll == kk && cl == ck) {
                    continue;
                }
                if chktors(&f, ii, (jj, cj), (kk, ck), (ll, cl)) {
                    continue;
                }
                let mut fkl = tors2(kk) * tors2(ll);
                if z(kk) == 7 && !pi[kk] {
                    fkl = tors2(kk) * tors2(ll) * 0.5;
                }
                if z(ll) == 7 && !pi[ll] {
                    fkl = tors2(kk) * tors2(ll) * 0.5;
                }
                if fkl < k::FCTHR {
                    continue;
                }
                if tors(kk) < 0.0 || tors(ll) < 0.0 {
                    continue;
                }
                let mut f1 = k::TORSF[0];
                let mut f2 = 0.0;
                fkl *= (nbn(kk) as f64 * nbn(ll) as f64).powf(-0.14);
                let mut nrot;
                let mut phi;
                if lring {
                    let rings4 = if rings > 3 {
                        ringstors(&s.rings, ii, jj, kk, ll)
                    } else {
                        3
                    };
                    nrot = 1;
                    if btyp == 2 {
                        nrot = 2;
                    }
                    phi = 0.0;
                    if btyp == 1 && rings4 > 0 {
                        let ringl = ringstorl(&s.rings, ii, jj, kk, ll);
                        let notpicon = !pi[kk] && !pi[ll];
                        if rings4 == 3 && notpicon {
                            nrot = 1;
                            phi = 0.0;
                            f1 = k::FR3;
                        }
                        if rings4 == 4 && ringl == rings4 && notpicon {
                            nrot = 6;
                            phi = 30.0;
                            f1 = k::FR4;
                        }
                        if rings4 == 5 && ringl == rings4 && notpicon {
                            nrot = 6;
                            phi = 30.0;
                            f1 = k::FR5;
                        }
                        if rings4 == 6 && ringl == rings4 && notpicon {
                            nrot = 3;
                            phi = 60.0;
                            f1 = k::FR6;
                        }
                    }
                    if rings4 == 0 && btyp == 1 && nbn(kk) == 1 && nbn(ll) == 1 {
                        nrot = 6;
                        phi = 30.0;
                        f1 = 0.30;
                    }
                    if btyp == 2
                        && rings == 5
                        && u32::from(z(ii)) * u32::from(z(jj)) == 42
                        && (amide(&s.el, hyb, &nb_atoms, &piflags, ii)
                            || amide(&s.el, hyb, &nb_atoms, &piflags, jj))
                    {
                        f1 = 5.0;
                    }
                } else {
                    phi = 180.0;
                    nrot = 1;
                    if hyb[ii] == 3 && hyb[jj] == 3 {
                        nrot = 3;
                    }
                    if btyp == 2 {
                        nrot = 2;
                    }
                    if pi[ii] && !pi[jj] && hyb[jj] == 3 {
                        f1 = 0.5;
                        if z(ii) == 7 {
                            f1 = 0.2;
                        }
                        phi = 180.0;
                        nrot = 3;
                    }
                    if pi[jj] && !pi[ii] && hyb[ii] == 3 {
                        f1 = 0.5;
                        if z(jj) == 7 {
                            f1 = 0.2;
                        }
                        phi = 180.0;
                        nrot = 3;
                    }
                }
                if hyb[ii] == 3 && hyb[jj] == 3 {
                    let (gi, gj) = (group(ii), group(jj));
                    if gi == 5 && gj == 5 {
                        nrot = 3;
                        phi = 60.0;
                        f1 = 3.0;
                    }
                    if (gi == 5 && gj == 6) || (gi == 6 && gj == 5) {
                        nrot = 2;
                        phi = 90.0;
                        f1 = 1.0;
                        if z(ii) >= 15 && z(jj) >= 15 {
                            f1 = 20.0;
                        }
                    }
                    if gi == 6 && gj == 6 {
                        nrot = 2;
                        phi = 90.0;
                        f1 = 5.0;
                        if z(ii) >= 16 && z(jj) >= 16 {
                            f1 = 25.0;
                        }
                    }
                }
                if pibo[m] > 0.0 {
                    f2 = pibo[m] * (-2.5 * (1.24 - pibo[m]).powi(14)).exp();
                    if !pi[kk] && z(kk) > 10 {
                        f2 *= 1.3;
                    }
                    if !pi[ll] && z(ll) > 10 {
                        f2 *= 1.3;
                    }
                    f1 *= 0.55;
                }
                if hyb[kk] == 5 || hyb[ll] == 5 {
                    fkl *= 1.5;
                }
                let fctot = (f1 + 10.0 * k::TORSF[1] * f2) * fqq * fij * fkl;
                if fctot > k::FCTHR {
                    out.push(Torsion {
                        atoms: [ll, ii, jj, kk],
                        cells: [cl, HOME, cj, ck],
                        nrot,
                        phi0: phi * PI / 180.0,
                        fc: fctot,
                    });
                }
                let sp3kl = hyb[kk] == 3 && hyb[ll] == 3;
                if sp3kl && sp3ij && !lring && btyp < 5 {
                    let mut ff = k::TORSF[5];
                    if z(ii) == 7 || z(jj) == 7 {
                        ff = k::TORSF[6];
                    }
                    if z(ii) == 8 || z(jj) == 8 {
                        ff = k::TORSF[7];
                    }
                    out.push(Torsion {
                        atoms: [ll, ii, jj, kk],
                        cells: [cl, HOME, cj, ck],
                        nrot: 1,
                        phi0: PI,
                        fc: ff * fij * fkl * fqq,
                    });
                }
            }
        }
    }
    // Out-of-plane terms of three-coordinated pi atoms and N.
    for i in 0..x.len() {
        if nbn(i) != 3 {
            continue;
        }
        if !pi[i] && z(i) != 7 {
            continue;
        }
        let Some(three) = three_neighbors(&s.nb.nb[i]) else {
            continue;
        };
        let nb3: Vec<usize> = s.nb.nb[i].iter().map(|&(a, _)| a).collect();
        // xtb sorts the atoms by the distances of their images, then the cells by the same
        // distances (recomputed from the unsorted pairs), so that each atom keeps its cell.
        let distance = |a: usize, c: Cell| dist_norm2(x[i], f.at(a, c));
        let mut keys = three.map(|(a, c)| distance(a, c));
        let mut ind = three.map(|(a, _)| a);
        ssort(&mut keys, &mut ind);
        let mut keys2 = three.map(|(a, c)| distance(a, c));
        let mut cells_idx = [0usize, 1, 2];
        ssort(&mut keys2, &mut cells_idx);
        let cells = cells_idx.map(|q| three[q].1);
        let [jj, kk, ll] = ind;
        let mut t = Torsion {
            atoms: [i, jj, kk, ll],
            cells: [HOME, cells[0], cells[1], cells[2]],
            nrot: 0,
            phi0: 0.0,
            fc: 0.0,
        };
        if !pi[i] && z(i) == 7 {
            t.nrot = -1;
            t.phi0 = 80.0 * PI / 180.0;
            for &idum in &nb3 {
                t.fc += 0.60 * s.el[idum].raw.repz.sqrt();
            }
        } else {
            let ncarbo = nb3.iter().filter(|&&a| z(a) == 8 || z(a) == 16).count();
            let nf = nb3.iter().filter(|&&a| group(a) == 7).count();
            let fqq = 1.0 + s.qa[i] * 5.0;
            let sumppi = pbo(i, jj) + pbo(i, kk) + pbo(i, ll);
            let f2 = 1.0 - sumppi * k::TORSF[4];
            t.fc = k::TORSF[2] * f2 * fqq;
            if z(i) == 5 && ncarbo > 0 {
                t.fc *= 38.0;
            }
            if z(i) == 6 && ncarbo > 0 {
                t.fc *= 38.0;
            }
            if z(i) == 6 && nf > 0 && ncarbo == 0 {
                t.fc *= 10.0;
            }
            if z(i) == 7 && ncarbo > 0 {
                t.fc = t.fc * 10.0 / f2;
            }
        }
        out.push(t);
    }
    out
}
