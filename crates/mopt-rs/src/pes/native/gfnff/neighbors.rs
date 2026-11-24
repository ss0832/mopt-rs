//! Bond detection and hybridization of GFN-FF, ported from xtb: `gfnff_neigh`
//! (`gfnff_ini2.F90`), `get_nb` / `fillnb` / `jth_nb` (`neighbor.f90`), `gfnff_dlogcoord`
//! (`gfnff_eg.f90`), `nn_nearest_noM`, and `bangl` / `banglPBC` (`constr.f90`)
//! (LGPL-3.0-or-later), with the branches for the main-group metals, boron, the noble gases,
//! and the transition metals (their hybridization from their neighbours without hydrogens).
//!
//! A neighbour is an atom and the cell of its image ([`Nb`]); in a periodic cell bonds are
//! looked for in the 27 cells around the home cell, and the neighbours of an atom are ordered
//! by cell (xtb's order of the cells) and then by atom, which is what "the first neighbour"
//! means in xtb's rules. A molecule has the home cell only.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::constants as k;
use super::elements::{rab_estimate, Atom};
use super::lattice::{Frame, Nb};

pub type V3 = [f64; 3];

pub fn dist(a: V3, b: V3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Distance as gfortran's `NORM2` computes it (scaled sum of squares), where xtb orders atoms by
/// distance and equal distances are decided by the last bits.
pub fn dist_norm2(a: V3, b: V3) -> f64 {
    let mut scale = 1.0_f64;
    let mut result = 0.0_f64;
    for c in 0..3 {
        let v = a[c] - b[c];
        if v != 0.0 {
            let abs = v.abs();
            if scale < abs {
                let r = scale / abs;
                result = 1.0 + result * r * r;
                scale = abs;
            } else {
                let r = abs / scale;
                result += r * r;
            }
        }
    }
    scale * result.sqrt()
}

/// mctc's `indexHeapSort` (Numerical Recipes heap sort, values within machine epsilon count as
/// equal): the 0-based indices of `a` in ascending order, ties in xtb's order.
pub fn index_heap_sort(a: &[f64]) -> Vec<usize> {
    let n = a.len();
    // 1-based as in the Fortran.
    let mut indx: Vec<usize> = (0..=n).collect();
    if n <= 1 {
        return (0..n).collect();
    }
    let tol = f64::EPSILON;
    let val = |k: usize| a[k - 1];
    let mut il = n / 2 + 1;
    let mut ir = n;
    loop {
        let tmp;
        if il > 1 {
            il -= 1;
            tmp = indx[il];
        } else {
            tmp = indx[ir];
            indx[ir] = indx[1];
            ir -= 1;
            if ir < 1 {
                indx[1] = tmp;
                break;
            }
        }
        let tv = val(tmp);
        let mut ii = il;
        let mut ij = 2 * il;
        while ij <= ir {
            if ij < ir && val(indx[ij]) < val(indx[ij + 1]) - tol {
                ij += 1;
            }
            if tv < val(indx[ij]) - tol {
                indx[ii] = indx[ij];
                ii = ij;
                ij *= 2;
            } else {
                ij = ir + 1;
            }
        }
        indx[ii] = tmp;
    }
    indx[1..].iter().map(|&k| k - 1).collect()
}

/// xtb's `jth_nb`: the `jth` (1-based) closest of the neighbours `nb` of `i` (distances to
/// the images as `NORM2`, order of `indexHeapSort`).
pub fn jth_nb(nb: &[Nb], f: &Frame<'_>, i: usize, jth: usize) -> Option<Nb> {
    if jth == 0 || jth > nb.len() {
        return None;
    }
    let d: Vec<f64> = nb
        .iter()
        .map(|&(j, n)| dist_norm2(f.x[i], f.at(j, n)))
        .collect();
    Some(nb[index_heap_sort(&d)[jth - 1]])
}

/// `bangl` / `banglPBC`: the angle at `b` of the points `a`, `b`, `c` (radians).
pub fn bangl_points(a: V3, b: V3, c: V3) -> f64 {
    let d2 = |p: V3, q: V3| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
    let (d2ij, d2jk, d2ik) = (d2(a, b), d2(b, c), d2(a, c));
    let xy = (d2ij * d2jk).sqrt();
    let temp = (0.5 * (d2ij + d2jk - d2ik) / xy).clamp(-1.0, 1.0);
    temp.acos()
}

/// `bangl(xyz, i, j, k)`: angle at `j` (radians).
pub fn bangl(x: &[V3], i: usize, j: usize, kk: usize) -> f64 {
    bangl_points(x[i], x[j], x[kk])
}

/// `gfnff_dlogcoord`: the log-capped erf CN and its derivatives `d[a][b] = d logCN_b / d x_a`
/// (xtb's `dlogCN(:, a, b)`, with xtb's truncated `sqrt(pi)`).
pub fn log_cn_derivatives(el: &[Atom], x: &[V3]) -> (Vec<f64>, Vec<Vec<V3>>) {
    const KN: f64 = -7.5;
    const SQRTPI: f64 = 1.772_453_850_91;
    let n = x.len();
    let thr = k::CNTHR.sqrt();
    let mut cn = vec![0.0; n];
    for i in 1..n {
        for j in 0..i {
            let r = dist(x[i], x[j]);
            if r > thr {
                continue;
            }
            let r0 = el[i].rcov + el[j].rcov;
            let dr = (r - r0) / r0;
            let e = 0.5 * (1.0 + libm::erf(KN * dr));
            cn[i] += e;
            cn[j] += e;
        }
    }
    let cnmax = k::PARAM_CNMAX;
    let logcn: Vec<f64> = cn
        .iter()
        .map(|&c| (1.0 + cnmax.exp()).ln() - (1.0 + (cnmax - c).exp()).ln())
        .collect();
    let dlog: Vec<f64> = cn
        .iter()
        .map(|&c| cnmax.exp() / (cnmax.exp() + c.exp()))
        .collect();
    let mut d = vec![vec![[0.0; 3]; n]; n];
    for i in 0..n {
        for j in 0..i {
            let r = dist(x[i], x[j]);
            if r > thr {
                continue;
            }
            let r0 = el[i].rcov + el[j].rcov;
            let dr = (r - r0) / r0;
            let der = KN / SQRTPI * (-(KN * KN) * dr * dr).exp() / r0;
            let rij = [0, 1, 2].map(|c| der * (x[j][c] - x[i][c]) / r);
            for c in 0..3 {
                d[j][j][c] += dlog[j] * rij[c];
                d[i][j][c] = -dlog[j] * rij[c];
                d[j][i][c] = dlog[i] * rij[c];
                d[i][i][c] += -dlog[i] * rij[c];
            }
        }
    }
    (logcn, d)
}

/// The log CN and `sum_j |d CN_i / d x_j|` (the metallic-character estimate of `gfnff_ini`).
pub fn log_cn(el: &[Atom], x: &[V3]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let (logcn, d) = log_cn_derivatives(el, x);
    let mchar = (0..n)
        .map(|i| {
            let s: f64 = (0..n)
                .map(|j| (d[j][i][0].powi(2) + d[j][i][1].powi(2) + d[j][i][2].powi(2)).sqrt())
                .sum();
            (-0.005 * el[i].raw.en.powi(8)).exp() * s / (logcn[i] + 1.0)
        })
        .collect();
    (logcn, mchar)
}

/// xtb's element factors of the bond-detection radius (`fat`).
fn fat(z: u8) -> f64 {
    match z {
        1 => 1.02,
        4 => 1.03,
        5 => 1.02,
        8 => 1.02,
        9 => 1.05,
        10 => 1.10,
        11 => 1.01,
        12 => 1.02,
        15 => 0.97,
        18 => 1.10,
        19 => 1.02,
        20 => 1.02,
        34 => 0.99,
        38 => 1.02,
        50 => 1.01,
        51 => 0.99,
        52 => 0.95,
        53 => 0.98,
        56 => 1.02,
        76 => 1.02,
        82 => 1.06,
        83 => 0.95,
        _ => 1.0,
    }
}

/// Neighbour lists and hybridization (`gfnff_neigh` output).
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbors {
    /// Full neighbour list `nbf` (final `nb` of xtb).
    pub nb: Vec<Vec<Nb>>,
    /// `nb` of the second detection (no hypercoordinated atoms), used during setup.
    pub nb_reduced: Vec<Vec<Nb>>,
    /// `nbm` (no metals and unusual coordination).
    pub nbm: Vec<Vec<Nb>>,
    /// Atoms excluded from the ring search (`nb(numnb-1, i) == 1`).
    pub cluster: Vec<bool>,
    /// Hybridization (0 unknown, 1 sp, 2 sp2, 3 sp3, 5 hypervalent).
    pub hyb: Vec<i32>,
    /// Special tags (1 carbene / NO2 nitrogen).
    pub itag: Vec<i32>,
}

/// `get_nb` / `fillnb` for `icase` 1 (full), 2 (no hypercoordinated), 3 (no metals and
/// unusual coordination), over the cells of `f` (an atom may be bonded to its own image).
fn fill(
    el: &[Atom],
    f: &Frame<'_>,
    rtmp: &[Vec<f64>],
    mchar: &[f64],
    nbf: Option<&[Vec<Nb>]>,
    icase: u8,
) -> Vec<Vec<Nb>> {
    let n = f.len();
    let mut out = vec![Vec::new(); n];
    for cell in f.bond_cells() {
        for i in 0..n {
            for j in 0..n {
                let r = dist(f.x[i], f.at(j, cell));
                if r <= 0.0 {
                    continue;
                }
                let mut fm = 1.0;
                if nbf.is_none() {
                    // The full case: the radius of metal atoms is changed.
                    for a in [&el[i], &el[j]] {
                        if a.metal() == 2 {
                            fm *= k::RTHR2;
                        }
                        if a.metal() == 1 {
                            fm *= k::RTHR2 + 0.025;
                        }
                    }
                }
                if let Some(nbf) = nbf {
                    let (nnfi, nnfj) = (nbf[i].len(), nbf[j].len());
                    if icase == 2 {
                        let crit = |a: &Atom| if a.raw.group <= 2 { 4 } else { 6 };
                        if nnfi > crit(&el[i]) || nnfj > crit(&el[j]) {
                            continue;
                        }
                    }
                    if icase == 3 {
                        if mchar[i] > 0.25 || el[i].metal() > 0 {
                            continue;
                        }
                        if mchar[j] > 0.25 || el[j].metal() > 0 {
                            continue;
                        }
                        if nnfi > el[i].raw.normcn as usize && el[i].z() > 10 {
                            continue;
                        }
                        if nnfj > el[j].raw.normcn as usize && el[j].z() > 10 {
                            continue;
                        }
                    }
                }
                if r < fm * k::RTHR * rtmp[i][j] {
                    out[i].push((j, cell));
                }
            }
        }
    }
    out
}

/// The bond radius `rtmp` of `gfnff_neigh` of atoms `i` and `j` (symmetric; the same
/// operations in the same order for both orders of the pair).
pub fn bond_radius(el: &[Atom], qa: &[f64], i: usize, j: usize) -> f64 {
    let (i, j) = (i.max(j), i.min(j));
    let cn = |a: &Atom| f64::from(a.raw.normcn);
    // The charge changes the radius of a metal atom twice as much.
    let f1 = if el[i].metal() > 0 { 2.0 } else { 1.0 } * k::RQSHRINK;
    let f2 = if el[j].metal() > 0 { 2.0 } else { 1.0 } * k::RQSHRINK;
    let mut r = rab_estimate(&el[j], cn(&el[j]), &el[i], cn(&el[i]));
    r = r - qa[i] * f1 - qa[j] * f2;
    r * fat(el[i].z()) * fat(el[j].z())
}

/// `gfnff_neigh` with topology charges `qa`.
pub fn neighbors(el: &[Atom], f: &Frame<'_>, qa: &[f64], mchar: &[f64]) -> Neighbors {
    let n = f.len();
    let mut rtmp = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let r = bond_radius(el, qa, i, j);
            rtmp[i][j] = r;
            rtmp[j][i] = r;
        }
    }
    let nbf = fill(el, f, &rtmp, mchar, None, 1);
    let nb2 = fill(el, f, &rtmp, mchar, Some(&nbf), 2);
    let nbm = fill(el, f, &rtmp, mchar, Some(&nbf), 3);

    let mut cluster = vec![false; n];
    for i in 0..n {
        if el[i].z() < 11
            && nbf[i].len() > 2
            && nbf[i]
                .iter()
                .any(|&(kk, _)| el[kk].metal() != 0 || nb2[kk].len() > 4)
        {
            cluster[i] = true;
        }
    }

    // An atom of the first rows pi-bonded to one metal (eta coordination, as in Cp or an
    // alkyne on a metal) takes its hybridization from the list without metals.
    let mut hyb = vec![0i32; n];
    let mut itag = vec![0i32; n];
    let mut nbdum = nbf.clone();
    for i in 0..n {
        let ati = el[i].z();
        if ati > 10 {
            continue;
        }
        let (numnbf, numnbm) = (nbf[i].len(), nbm[i].len());
        let mut eta = ati == 6 && ((numnbf >= 4 && numnbm == 3) || (numnbf == 3 && numnbm == 2));
        let metals: Vec<usize> = nbf[i]
            .iter()
            .filter(|&&(kk, _)| el[kk].metal() != 0)
            .map(|&(kk, _)| kk)
            .collect();
        if metals.is_empty() {
            eta = false;
        } else if metals.len() == 1 {
            // M-CR2-R is not an eta coordination: another neighbour must be bonded to M too.
            let im = metals[metals.len() - 1];
            let ncm = nbf[i]
                .iter()
                .filter(|&&(kk, _)| kk != im)
                .map(|&(kk, _)| nbf[kk].iter().filter(|&&(l, _)| l == im).count())
                .sum::<usize>();
            if ncm == 0 {
                eta = false;
            }
        }
        if eta {
            itag[i] = -1;
            nbdum[i] = nbm[i].clone();
        }
    }
    let nbdum = nbdum;
    let deg = |rad: f64| rad * 180.0 / std::f64::consts::PI;
    for i in 0..n {
        let ati = el[i].z();
        let group = el[i].raw.group;
        let numnbm = nbm[i].len();
        let numnbf = nbf[i].len();
        let numnb = nb2[i].len();
        let nbdiff = numnbf as i64 - numnb as i64;
        let nbmdiff = numnbf as i64 - numnbm as i64;
        let nb20i = nbdum[i].len();
        let nh = nbdum[i].iter().filter(|&&(j, _)| el[j].z() == 1).count();
        match group {
            1 => {
                if nb20i == 2 {
                    hyb[i] = 1;
                }
                if nb20i > 2 {
                    hyb[i] = 3;
                }
                if nb20i > 4 {
                    hyb[i] = 0;
                }
            }
            2 => {
                if nb20i == 2 {
                    hyb[i] = 1;
                }
                if nb20i > 2 {
                    hyb[i] = 3;
                }
                if nb20i > 4 {
                    hyb[i] = 0;
                }
            }
            3 => {
                if nb20i > 4 {
                    hyb[i] = 3;
                }
                if nb20i > 4 && ati > 10 && nbdiff == 0 {
                    hyb[i] = 5;
                }
                if nb20i == 4 {
                    hyb[i] = 3;
                }
                if nb20i == 3 {
                    hyb[i] = 2;
                }
                if nb20i == 2 {
                    hyb[i] = 1;
                }
            }
            4 => {
                if nb20i >= 4 {
                    hyb[i] = 3;
                }
                if nb20i > 4 && ati > 10 && nbdiff == 0 {
                    hyb[i] = 5;
                }
                if nb20i == 3 {
                    hyb[i] = 2;
                }
                if nb20i == 2 {
                    let (a, b) = (nbdum[i][0], nbdum[i][1]);
                    let phi = bangl_points(f.at(a.0, a.1), f.x[i], f.at(b.0, b.1));
                    if deg(phi) < 150.0 {
                        hyb[i] = 2;
                        itag[i] = 1;
                    } else {
                        hyb[i] = 1;
                    }
                    if qa[i] < -0.4 {
                        hyb[i] = 2;
                        itag[i] = 0;
                    }
                }
                if nb20i == 1 {
                    hyb[i] = 1;
                }
            }
            5 => {
                if nb20i >= 4 {
                    hyb[i] = 3;
                }
                if nb20i > 4 && ati > 10 && nbdiff == 0 {
                    hyb[i] = 5;
                }
                if nb20i == 3 {
                    hyb[i] = 3;
                }
                if nb20i == 3 && ati == 7 {
                    let (mut kk, mut ll, mut nn) = (0, 0, 0);
                    for &(jj, _) in nbdum[i].iter().take(3) {
                        let z = el[jj].z();
                        if z == 8 && nb2[jj].len() == 1 {
                            kk += 1;
                        }
                        if z == 5 && nb2[jj].len() == 4 {
                            ll += 1;
                        }
                        if z == 16 && nb2[jj].len() == 4 {
                            nn += 1;
                        }
                    }
                    if nn == 1 && ll == 0 && kk == 0 {
                        hyb[i] = 3;
                    }
                    if ll == 1 && nn == 0 {
                        hyb[i] = 2;
                    }
                    if kk >= 1 {
                        hyb[i] = 2;
                        itag[i] = 1;
                    }
                    if nbmdiff > 0 && nn == 0 {
                        hyb[i] = 2;
                    }
                }
                if nb20i == 2 {
                    hyb[i] = 2;
                    let ((jj, nj), (kk, nk)) = (nbdum[i][0], nbdum[i][1]);
                    let phi = bangl_points(f.at(jj, nj), f.x[i], f.at(kk, nk));
                    if nbdum[jj].len() == 1 && el[jj].z() == 6 {
                        hyb[i] = 1;
                    }
                    if nbdum[kk].len() == 1 && el[kk].z() == 6 {
                        hyb[i] = 1;
                    }
                    if nbdum[jj].len() == 1 && el[jj].z() == 7 {
                        hyb[i] = 1;
                    }
                    if nbdum[kk].len() == 1 && el[kk].z() == 7 {
                        hyb[i] = 1;
                    }
                    // M-NC-R, as in nitriles on a metal.
                    if el[jj].metal() > 0 || el[kk].metal() > 0 {
                        hyb[i] = 1;
                    }
                    if el[jj].z() == 7
                        && el[kk].z() == 7
                        && nbdum[jj].len() <= 2
                        && nbdum[kk].len() <= 2
                    {
                        hyb[i] = 1;
                    }
                    if deg(phi) > k::LINTHR {
                        hyb[i] = 1;
                    }
                }
                if nb20i == 1 {
                    hyb[i] = 1;
                }
            }
            6 => {
                if nb20i >= 3 {
                    hyb[i] = 3;
                }
                if nb20i > 3 && ati > 10 && nbdiff == 0 {
                    hyb[i] = 5;
                }
                if nb20i == 2 {
                    hyb[i] = 3;
                }
                if nb20i == 2 && nbmdiff > 0 {
                    // nn_nearest_noM: the neighbour count of the closest neighbour that is not a
                    // metal.
                    let mut best: Option<(f64, usize)> = None;
                    for &(jj, nj) in &nb2[i] {
                        if el[jj].metal() != 0 {
                            continue;
                        }
                        let d = dist(f.x[i], f.at(jj, nj));
                        if best.is_none_or(|(b, _)| d < b) {
                            best = Some((d, jj));
                        }
                    }
                    let nnn = best.map_or(0, |(_, jj)| nb2[jj].len());
                    if nnn == 3 {
                        hyb[i] = 2;
                    }
                    if nnn == 4 {
                        hyb[i] = 3;
                    }
                }
                if nb20i == 1 {
                    hyb[i] = 2;
                }
                if nb20i == 1 && nbdiff == 0 {
                    let first = nb2[i][0].0;
                    if nb2[first].len() == 1 {
                        hyb[i] = 1;
                    }
                }
            }
            7 => {
                if nb20i == 2 {
                    hyb[i] = 1;
                }
                if nb20i > 2 && ati > 10 {
                    hyb[i] = 5;
                }
            }
            8 => {
                hyb[i] = 0;
                if nb20i > 0 && ati > 2 {
                    hyb[i] = 5;
                }
            }
            g if g <= 0 => {
                // The transition metals; the hydrogens are not counted (unless every
                // neighbour is one).
                let mut nni = nb20i;
                if nh != 0 && nh != nni {
                    nni -= nh;
                }
                if nni <= 2 {
                    hyb[i] = 1;
                }
                if nni <= 2 && g <= -6 {
                    hyb[i] = 2;
                }
                if nni == 3 {
                    hyb[i] = 2;
                }
                // Early metals tetrahedral, late ones square planar: both 3.
                if nni == 4 {
                    hyb[i] = 3;
                }
                // Sc to La are tetrahedral with five neighbours.
                if nni == 5 && g == -3 {
                    hyb[i] = 3;
                }
            }
            _ => {}
        }
    }
    // Two bonded carbene C are an arine.
    for i in 0..n {
        for idx in 0..nbdum[i].len() {
            let kk = nbdum[i][idx].0;
            if el[kk].z() == 6 && el[i].z() == 6 && itag[i] == 1 && itag[kk] == 1 {
                itag[i] = 0;
                itag[kk] = 0;
            }
        }
    }
    Neighbors {
        // The final list: the full one, without metals for the eta-coordinated atoms.
        nb: nbdum,
        nb_reduced: nb2,
        nbm,
        cluster,
        hyb,
        itag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_sort_orders_ascending() {
        let a = [3.0, 1.0, 2.0, 5.0, 4.0, 0.5];
        let idx = index_heap_sort(&a);
        let sorted: Vec<f64> = idx.iter().map(|&k| a[k]).collect();
        assert_eq!(sorted, vec![0.5, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(index_heap_sort(&[2.0]), vec![0]);
        assert!(index_heap_sort(&[]).is_empty());
    }

    #[test]
    fn norm2_distance_equals_euclidean() {
        let (a, b) = ([0.3, -1.2, 2.5], [-0.7, 0.4, 1.0]);
        assert!((dist_norm2(a, b) - dist(a, b)).abs() < 1.0e-15);
        assert_eq!(dist_norm2(a, a), 0.0);
    }
}
